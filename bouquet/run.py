"""The :class:`Bouquet` orchestrator.

Owns the TokaMaker solver (``mygs``), the HDF5 output path, the resolved
baseline and the uncertainty envelope, so the per-draw generation step is
auto-wired instead of threaded by hand.

Composable methods for interactive work (inspect the baseline before spending
GS-solve compute), plus a thin :meth:`run` convenience for scripts/CI::

    import bouquet as bq

    bouquet = bq.Bouquet(config)
    bouquet.setup_solver()
    bouquet.prepare_baseline()      # reconstruction OR imas, transparently
    bouquet.plot_baseline()         # gate: is the baseline good?
    bouquet.generate()
    bouquet.filter()
    bouquet.export()

    # or, once the baseline is trusted:
    bouquet = bq.Bouquet(config).run()
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BouquetConfig
    from .baseline import Baseline


# `_shape_from_boundary` now lives in `bouquet.utils` -- it is pure geometry
# with no orchestration in it, and TokaMaker_interface needs it too.  Importing
# it there rather than here breaks the run <-> TokaMaker_interface cycle that a
# runtime `from .run import _shape_from_boundary` would otherwise create (run
# already imports from TokaMaker_interface).  Re-exported here because the
# original home is the documented one and callers import it from this module.
from .utils import _shape_from_boundary  # noqa: F401  (compatibility re-export)


class Bouquet:
    """Stateful driver: solver -> baseline -> generate -> filter -> export."""

    def __init__(self, config: "BouquetConfig"):
        self.config = config
        self.mygs = None                          # set by setup_solver()
        self.baseline: Optional["Baseline"] = None
        self._resolved_uncertainty = None         # resolved sigma profiles + length scales
        self.diagnostics: Optional[list] = None   # generate() per-draw output
        self.generation_log: Optional[str] = None # captured generate() solver chatter
        self._selection = None                    # filter() result

    # ── constructors ----------------------------------------------------
    @classmethod
    def from_geqdsk(cls, geqdsk_path, *, profiles, mesh,
                    n_draws=20, header="bouquet", cocos=1, time=None,
                    impurity_Z=6.0, **solver_kwargs) -> "Bouquet":
        """Minimal constructor for the reconstruction path (g-file + profiles).

        ``profiles`` is an IDA ``.cdf`` or a p-file (auto-detected).
        ``impurity_Z`` is the machine impurity charge (carbon 6.0 by default;
        set it for your device -- it controls the Z_eff<->ni mapping and is the
        IDA path's main-ion derivation, see :class:`ReconstructionSource`).
        Extra keyword args go to :class:`SolverConfig` (e.g. ``order``,
        ``nthreads``). Reach into ``bq.uncertainty`` / ``bq.generation``
        afterwards for the advanced knobs.
        """
        from .config import (BouquetConfig, SolverConfig, ReconstructionSource,
                             GenerationConfig)
        cfg = BouquetConfig(
            source=ReconstructionSource(geqdsk_path=geqdsk_path,
                                        profiles_path=profiles,
                                        cocos=cocos, time=time,
                                        impurity_Z=impurity_Z),
            solver=SolverConfig(mesh_path=mesh, **solver_kwargs),
            generation=GenerationConfig(n_equils=n_draws),
            output_header=header,
        )
        # geqdsk validated default workflow: the standard flagship l_i loop
        # (Fix C / perturb_jind_in_anchor drops draws on stiff geqdsks).
        cfg.generation.perturb_jind_in_anchor = False
        # Unified forward decomposition (matches the IMAS path): j_inductive is
        # pure ohmic and j_BS carries the full recon-anchored bootstrap. Closes
        # exactly, is non-negative, and yields better than the isolated-edge-spike
        # split. Flip to True only for dedicated edge-spike studies.
        cfg.generation.isolate_edge_jBS = False
        return cls(cfg)

    @classmethod
    def from_imas(cls, ids_path, *, mesh, time=None,
                  n_draws=20, header="bouquet",
                  ida_path=None, LCFS_geqdsk=None, impurity_Z=6.0,
                  kinetic_source=None, anchor_pressure_to_equilibrium=False,
                  **solver_kwargs) -> "Bouquet":
        """Minimal constructor for the IMAS/OMAS path (no reconstruction).

        Extra keyword args go to :class:`SolverConfig`. Reach into
        ``bq.uncertainty`` / ``bq.generation`` afterwards for advanced knobs.

        IDA-hybrid kinetics: pass ``ida_path`` (an IDA ``.cdf``) to take the
        baseline ne/Te/Ti/omega_tor (and sigma envelopes) from IDA fits while
        keeping FUSE Z_eff/currents/equilibrium. ``kinetic_source`` defaults to
        ``"ida_hybrid"`` when an ``ida_path`` is given, else ``"fuse"``.

        ``LCFS_geqdsk`` is OPTIONAL: a g-file whose LCFS replaces the source
        boundary outline as the isoflux target, for when you have a better
        separatrix for the slice than the dd carries (typically a magnetics-only
        reconstruction). Omit it to use the source's own boundary.
        """
        from .config import (BouquetConfig, SolverConfig, ImasSource,
                             GenerationConfig)
        if kinetic_source is None:
            kinetic_source = "ida_hybrid" if ida_path else "fuse"
        cfg = BouquetConfig(
            source=ImasSource(ids_path=ids_path, time=time, ida_path=ida_path,
                              impurity_Z=impurity_Z, LCFS_geqdsk=LCFS_geqdsk),
            solver=SolverConfig(mesh_path=mesh, **solver_kwargs),
            generation=GenerationConfig(n_equils=n_draws,
                                        kinetic_source=kinetic_source,
                                        anchor_pressure_to_equilibrium=anchor_pressure_to_equilibrium),
            output_header=header,
        )
        # IDA-hybrid: source the kinetic sigma envelopes from the same IDA .cdf
        # (resolve_uncertainty fires its IDA branch whenever unc.ida_path is set).
        if ida_path:
            cfg.uncertainty.ida_path = ida_path
        # IMAS validated default workflow: diff+C (anchor bootstrap to the source
        # via the fixed FUSE_jBS-SWB diff, and perturb j_ind in the recon-anchor
        # to avoid the find_optimal_scale/corrector homogenization).
        cfg.generation.jBS_baseline_mode = "diff"
        cfg.generation.perturb_jind_in_anchor = True
        # FUSE/IMAS sources carry a FULL Sauter bootstrap (core hump + edge), not
        # an isolated edge spike, so the edge-spike isolation + shelf-blend
        # decomposition (a DIII-D g-file construct) does not apply: it mislabels a
        # redundant j_BS,edge and mangles the per-draw j_inductive. Use the full
        # profile; the draws then store the clean residual j_phi - j_BS - j_NBI.
        cfg.generation.isolate_edge_jBS = False
        return cls(cfg)

    # ── ergonomic config accessors (so `bq.uncertainty.ne_scalar_sigma = ...`,
    # `bq.generation.n_equils = ...` read like a control panel) ──
    @property
    def uncertainty(self):
        """The :class:`UncertaintyConfig` (sigma profiles, GPR lengths, switchboard)."""
        return self.config.uncertainty

    @property
    def generation(self):
        """The :class:`GenerationConfig` (n_equils, tolerances, homotopy)."""
        return self.config.generation

    @property
    def solver(self):
        """The :class:`SolverConfig` (mesh, order, threads, F0)."""
        return self.config.solver

    @property
    def source(self):
        """The baseline source (:class:`ReconstructionSource` or :class:`ImasSource`)."""
        return self.config.source

    @property
    def filtering(self):
        """The :class:`FilterConfig` (boundary RMS + coil-spec thresholds)."""
        return self.config.filtering

    @property
    def fixed_components(self):
        """The :class:`FixedComponentsConfig` (fast pressure, NBI/RF current)."""
        return self.config.fixed_components

    @property
    def archive(self):
        """A :class:`~bouquet.BouquetArchive` over this run's ``{header}.h5``."""
        from .archive import BouquetArchive
        return BouquetArchive(self.config.output_header)

    def describe(self, stream=None) -> str:
        """Print (and return) the config, grouped by section, non-defaults only.

        Source paths + output header always show; every other knob appears only
        when it differs from its dataclass default -- so the true deltas of a run
        stand out instead of being buried in a ~60-line knob dump (F2/F3).
        """
        import dataclasses as _dc
        import numpy as np

        def _default(f):
            if f.default is not _dc.MISSING:
                return f.default
            if f.default_factory is not _dc.MISSING:      # type: ignore[attr-defined]
                return f.default_factory()
            return _dc.MISSING

        def _differs(v, d):
            if d is _dc.MISSING:                # required field (e.g. a path)
                return True
            if isinstance(v, np.ndarray) or isinstance(d, np.ndarray):
                return v is not None           # arrays: show when set
            try:
                return bool(v != d)
            except Exception:
                return True

        def _fmt(v):
            if isinstance(v, np.ndarray):
                return f"<ndarray {v.shape}>"
            if isinstance(v, dict) and v:
                return "{" + ", ".join(f"{k}: <..>" if isinstance(x, np.ndarray)
                                       else f"{k}: {x!r}" for k, x in v.items()) + "}"
            return repr(v)

        cfg = self.config
        lines = [f"Bouquet '{cfg.output_header}'  (source: {type(cfg.source).__name__})"]
        # every section (source included): required fields (paths) + non-defaults
        for name in ("source", "solver", "uncertainty", "generation", "filtering",
                     "fixed_components"):
            obj = getattr(cfg, name)
            for f in _dc.fields(obj):
                v = getattr(obj, f.name)
                if v is None or (isinstance(v, dict) and not v):
                    continue
                if _differs(v, _default(f)):
                    lines.append(f"    {name}.{f.name} = {_fmt(v)}")
        if cfg.verbose:
            lines.append("    verbose = True")
        text = "\n".join(lines)
        print(text, file=stream)
        return text

    @property
    def output_header(self):
        """The output archive header -- draws are written to ``{header}.h5``."""
        return self.config.output_header

    @output_header.setter
    def output_header(self, value):
        # A real setter (not a bare instance attribute): without it ``run.output_
        # header = ...`` would silently shadow the config and generate() would
        # still write to the old header.
        self.config.output_header = value

    def set_slice(self, *, time=None, header=None) -> "Bouquet":
        """Re-point to a new time slice, reusing the existing solver.

        The multi-slice mechanism for the **IMAS path**, where one IDS holds
        many time slices and the ``OFT_env`` singleton forbids standing up a
        second solver: keep one :meth:`setup_solver`, then for each slice call
        ``set_slice(time=t, header=...)`` and :meth:`run` (or generate). Clearing
        the cached baseline/uncertainty forces a re-solve; the next
        :meth:`prepare_baseline` then re-reads this slice's own LCFS boundary,
        re-points the solver isoflux, and resets the coil reg/bounds + pristine
        equilibrium (:meth:`_repoint_imas_geometry`), so each slice is a fully
        independent bouquet.

        Reconstruction sources are single-equilibrium (one g-file = one slice
        with its own boundary), so there is no time axis to sweep -- passing
        ``time`` raises. To run several reconstructions, build a fresh
        :class:`Bouquet` per g-file. ``header`` may still be set on either path
        to redirect the output archive.
        """
        if time is not None:
            if not hasattr(self.config.source, "time"):
                raise TypeError(
                    f"{type(self.config.source).__name__} has no time axis to "
                    "sweep; build a separate Bouquet per source")
            self.config.source.time = time
        if header is not None:
            self.config.output_header = header
        self.baseline = None
        self._resolved_uncertainty = None
        self.diagnostics = None
        self._selection = None
        return self

    # ── stage 1: solver -------------------------------------------------
    def setup_solver(self) -> "Bouquet":
        """Read mesh, build regions, stand up ``mygs``, set isoflux + VSC + reg.

        Common to every baseline source -- perturbed draws are always solved
        with TokaMaker. Returns self for chaining. Idempotent: a no-op if
        ``mygs`` already exists, so it is safe to call once and then reuse the
        solver across multiple baselines/time-slices (OFT_env is a per-process
        singleton, so re-creating it would raise).
        """
        import numpy as np

        if self.mygs is not None:
            return self
        from OpenFUSIONToolkit import OFT_env
        from OpenFUSIONToolkit.TokaMaker import TokaMaker
        from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh

        from .config import ReconstructionSource, ImasSource
        from .io.geqdsk import read_geqdsk

        sc = self.config.solver
        src = self.config.source

        myOFT = OFT_env(nthreads=sc.nthreads)
        mygs = TokaMaker(myOFT)

        mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(sc.mesh_path)
        mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
        mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)

        # F0 and reference LCFS boundary come from the g-file (reconstruction)
        # or the IDS vacuum_toroidal_field + boundary outline (IMAS).
        F0 = sc.F0
        eqdsk_ref = None
        boundary_RZ = None
        if isinstance(src, ReconstructionSource):
            eqdsk_ref = read_geqdsk(src.geqdsk_path, cocos=src.cocos)
            if F0 is None:
                F0 = abs(eqdsk_ref.R_center * eqdsk_ref.B_center)
            boundary_RZ = np.column_stack(
                [eqdsk_ref.boundary_R, eqdsk_ref.boundary_Z]
            )
        elif isinstance(src, ImasSource):
            from .io.imas import read_imas_geometry
            _imas_F0, boundary_RZ = read_imas_geometry(src)
            if F0 is None:
                F0 = _imas_F0
        if F0 is None:
            raise ValueError("F0 could not be determined; set SolverConfig.F0")

        mygs.setup(order=sc.order, F0=F0)
        mygs.settings.maxits = 800
        mygs.settings.pm = False
        mygs.update_settings()
        mygs.set_coil_vsc(sc.coil_vsc)

        # Isoflux: explicit config wins; otherwise the source's LCFS boundary
        iso_pts, iso_w = sc.isoflux_pts, sc.isoflux_weights
        if iso_pts is None and boundary_RZ is not None:
            iso_pts = boundary_RZ
            iso_w = np.ones(len(iso_pts)) * 500.0
        if iso_pts is not None:
            mygs.set_isoflux(iso_pts, weights=iso_w)

        # Optional X-point pin: drive B_pol -> 0 at the configured saddle
        # point(s). Opt-in via SolverConfig.saddle_targets (default None).
        if sc.saddle_targets is not None:
            _sad = np.asarray(sc.saddle_targets, dtype=np.float64).reshape(-1, 2)
            _sw = (np.asarray(sc.saddle_weights, dtype=np.float64)
                   if sc.saddle_weights is not None else None)
            mygs.set_saddle_constraints(_sad, weights=_sw)

        # Coil regularisation: SolverConfig.coil_reg targets when given, else
        # the historical pull toward zero + small VSC freedom.  Also publishes
        # the WEAK exploratory reg the draw path swaps in for the SWB phase.
        self._apply_coil_reg(mygs)

        self.mygs = mygs
        self._myOFT = myOFT          # keep the env alive
        self._eqdsk_ref = eqdsk_ref
        self._boundary_RZ = boundary_RZ   # LCFS shape for IMAS forward-solve init
        self._F0 = F0                     # vacuum R*Bt applied at setup (fixed)
        # Snapshot the pristine post-setup equilibrium (zero coils, no plasma)
        # for a full per-slice reset in a multi-slice sweep -- see
        # _reset_solver_state. copy_eq/replace_eq need OFT PR #248+.
        self._clean_eq = mygs.copy_eq() if hasattr(mygs, "copy_eq") else None
        return self

    def _apply_coil_reg(self, mygs):
        """Install the coil regularisation: configured targets, else toward zero.

        ``SolverConfig.coil_reg`` is a list of
        ``{"coils": {name: coeff}, "target": float, "weight": float}``. When it
        is empty the historical behaviour is used unchanged -- every coil pulled
        toward ZERO at unit weight, plus a weak VSC term.

        Also publishes ``mygs._weak_coil_reg``: the SAME terms at the historical
        WEAK magnitude (weight 1.0 on every coil term, the VSC channel at 1e-2),
        which the draw path swaps in for the exploratory SWB phase in place of
        the strong reg. The point is the TARGETS: once coil_reg carries
        measured-current targets, a weak reg aimed at zero is not "the recon
        setup held loosely", it is a pull along the very coil null space the
        targets exist to remove, and the exploratory solve follows it (measured:
        the recon-anchor moved tens of kA-turn, of order 100 sigma of the coil
        measurement precision, on the coil carrying that null space). The weight
        is deliberately NOT raised: at weight 1.0 the converged coil currents
        move by at most ~0.2 sigma and boundary RMS by ~0.03 mm, with yield and
        failure rate unchanged, whereas raising it tripled the endpoint shift
        and cost up to half a millimetre of boundary RMS for no measured gain.
        With ``coil_reg`` empty nothing is published and the draw path builds
        its historical toward-zero weak reg, so that case is bit-identical.

        Called from BOTH :meth:`setup_solver` and :meth:`_reset_solver_state`.
        That matters: ``_repoint_imas_geometry`` resets the solver immediately
        before the IMAS baseline solve, so targets installed only at setup were
        silently discarded and the solve ran on the zero-target default. A
        1e8-weight target moved its coil by 0.2 % for exactly that reason.
        """
        spec = list(getattr(self.config.solver, "coil_reg", None) or [])
        if spec:
            # Drop terms naming coils this MESH does not model. A measurement
            # source is not mesh-specific: DIII-D pf_active carries all 24
            # circuits while the shipped D3D mesh models 20 coil sets (no
            # E567UP/E567DN/E89UP/E89DN), and coil_reg_term raises KeyError on
            # an unknown name -- which would kill setup_solver outright.
            known = set(mygs.coil_sets) | {"#VSC"}
            dropped = [t for t in spec if not set(t["coils"]) <= known]
            spec = [t for t in spec if set(t["coils"]) <= known]
            if dropped:
                # Report the whole TERM, not just the off-mesh coil. A term is
                # dropped WHOLE, so a difference constraint like
                # {"F1A": 1.0, "E567UP": -1.0} also releases F1A, which then falls
                # through to the target=0, weight=1 default -- the opposite of what
                # was asked. Naming only E567UP would leave the operator unaware
                # that F1A moved too.
                import warnings
                warnings.warn(
                    "coil_reg: dropping %d term(s) that name coil(s) absent from this "
                    "mesh. EVERY coil in a dropped term loses its target and reverts "
                    "to the target=0, weight=1 default: %s"
                    % (len(dropped),
                       "; ".join("{%s} (absent: %s)"
                                 % (", ".join(sorted(t["coils"])),
                                    ", ".join(sorted(set(t["coils"]) - known)))
                                 for t in dropped)))
            spec = self._check_coil_reg_turns(mygs, spec)
            named = {c for t in spec for c in t["coils"]}

            def _build(exploratory):
                """The term list: configured weights, or the weak exploratory one.

                Same terms and same targets either way -- ``exploratory`` only
                drops every coil term to the historical weak weight of 1.0 and
                leaves the VSC channel its own weak term (a #VSC term in the
                config is a constraint for the CONSTRAINED phase; carrying it at
                weight 1.0 into the exploration would clamp the vertical-stability
                channel the exploration needs).  Coils no term names keep the
                zero target they have always had -- there is no measured target
                for them to be held at.
                """
                out = [mygs.coil_reg_term(
                           dict(t["coils"]), target=float(t.get("target", 0.0)),
                           weight=1.0 if exploratory else float(t.get("weight", 1.0)))
                       for t in spec
                       if not (exploratory and set(t["coils"]) == {"#VSC"})]
                out += [mygs.coil_reg_term({n: 1.0}, target=0.0, weight=1.0)
                        for n in mygs.coil_sets if n not in named]
                if exploratory or "#VSC" not in named:
                    out.append(
                        mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1e-2))
                return out

            reg_terms = _build(False)
            mygs._weak_coil_reg = _build(True)
        else:
            reg_terms = [mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0)
                         for name in mygs.coil_sets]
            reg_terms.append(
                mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1e-2))
            # no measured targets -> nothing to publish, and an earlier slice's
            # stash must not survive into a run that has none
            if hasattr(mygs, "_weak_coil_reg"):
                del mygs._weak_coil_reg
        mygs.set_coil_reg(reg_terms=reg_terms)
        return reg_terms

    @staticmethod
    def _mesh_net_turns(mygs, name):
        """Turns this MESH carries for coil set *name*, or None if unknowable.

        TokaMaker sums ``nturns`` over a coil set's sub-coils into
        ``coil_sets[name]["net_turns"]``. A stub/mesh-less solver object (or an
        older build) may expose only the names, in which case the convention
        cannot be confirmed and the caller must treat it as unconfirmed.
        """
        sets = getattr(mygs, "coil_sets", None)
        info = sets.get(name) if isinstance(sets, dict) else None
        try:
            return float(info["net_turns"])
        except (TypeError, KeyError, ValueError):
            return None

    def _check_coil_reg_turns(self, mygs, spec):
        """Drop terms whose circuit-amps -> ampere-turns factor the mesh contradicts.

        ``coil_targets`` converts a measured circuit current with the DEVICE's
        turn table, but which side carries the turns is a property of the MESH:
        the shipped D3D mesh gives its F-coil sets ``net_turns = 1``, so the
        x58/x55 factor supplies them, and gives ECOILA/ECOILB ``net_turns = 61``,
        so those convert at x1.0. Keyed by device, that x1.0 is an ASSUMPTION
        about one mesh -- a second registered mesh of the same machine (e.g. one
        that splits the E-coil into E567UP/E567DN/E89UP/E89DN) would take the
        same implicit 1.0 and, if it does not carry the turns either, be pinned
        to a target wrong by its whole turn count at W0 = 100, i.e. a strong,
        confidently wrong constraint.

        Exactly one side must carry the turns, which is checkable without
        knowing the physical count: a factor != 1 needs ``net_turns == 1``, and
        a factor of 1 needs ``net_turns != 1``. Both-1 (turns nowhere) and
        neither-1 (turns twice) are refused, as is a mesh that cannot report
        ``net_turns`` at all. A refused term is dropped with the same warning
        the off-mesh drop gives -- its coils revert to target=0, weight=1, which
        is the historical behaviour rather than a wrong strong target.

        Only terms carrying a ``"turns"`` key are checked: that key is the
        conversion CLAIM stamped by :func:`coil_targets.coil_reg_from_measured`.
        A hand-built term makes no claim and is left alone.
        """
        bad = []
        for t in spec:
            f = t.get("turns")
            if f is None:
                continue
            for c in t["coils"]:
                if c == "#VSC":
                    continue
                nt = self._mesh_net_turns(mygs, c)
                if nt is None:
                    bad.append((t, c, f, "this mesh does not report net_turns"))
                elif (float(f) == 1.0) == (nt == 1.0):
                    bad.append((t, c, f, "mesh net_turns = %g" % nt))
        if not bad:
            return spec
        import warnings
        drop = {id(t) for t, _c, _f, _w in bad}
        warnings.warn(
            "coil_reg: dropping %d term(s) whose turns convention this mesh does not "
            "confirm. The circuit-amps -> ampere-turns factor is a property of the "
            "MESH, not of the device, so an unconfirmed factor would pin the coil to a "
            "target wrong by its whole turn count -- at the configured weight. EVERY "
            "coil in a dropped term reverts to the target=0, weight=1 default: %s. "
            "Pass `turns` explicitly to coil_reg_from_measured for this mesh, or build "
            "the term by hand (a term with no 'turns' key claims nothing and is not "
            "checked)."
            % (len(drop),
               "; ".join("{%s} (%s: factor %g, %s)"
                         % (", ".join(sorted(t["coils"])), c, float(f), why)
                         for t, c, f, why in bad)))
        return [t for t in spec if id(t) not in drop]

    def _seed_coil_init(self, mygs):
        """Seed the inverse iterate from ``SolverConfig.coil_init`` ({name: A-t}).

        Distinct from ``coil_reg``: this sets a STARTING POINT on the degenerate
        coil manifold without adding a term that fights the boundary.  Coils the
        mesh does not model are dropped (a measurement covers more circuits than
        a mesh models); coils the setting does not name keep whatever ``init_psi``
        left them at.

        Must run AFTER ``init_psi``, which reinitialises coil currents from the
        regularisation and would overwrite an earlier set.

        KNOWN NO-OP for the shipped inverse baseline solve: the inverse solver
        re-solves every coil current at each Picard step, so the seed is
        discarded before it can influence the converged answer.  It is kept
        because it is the only hook for choosing a basin if a forward-mode or
        warm-started baseline path is ever added, and because a silent
        `set_coil_currents` buried in the hot baseline solve was itself the trap
        -- it now lives in one named place.  Do not reach for it expecting the
        baseline to move; use ``coil_reg`` (see :mod:`bouquet.coil_targets`).

        Returns the dict that was installed, or None when the setting is unset.
        """
        ci = getattr(self.config.solver, "coil_init", None)
        if not ci:
            return None
        if not hasattr(ci, "items"):
            raise TypeError(
                "solver.coil_init must be a {coil_name: current_A_turns} mapping, "
                f"got {type(ci).__name__}")
        known = set(mygs.coil_sets)
        use = {k: float(v) for k, v in ci.items() if k in known}
        cur, _ = mygs.get_coil_currents()
        cur = dict(cur)
        cur.update(use)
        mygs.set_coil_currents(cur)
        return cur

    def _reset_solver_state(self):
        """Restore the clean post-:meth:`setup_solver` coil state.

        ``generate_bouquet`` installs a STRONG coil regularization (and, when
        requested, hard drift bounds) that pull the coils toward *this run's*
        baseline coils, leaves them active on ``mygs`` when it returns, and
        leaves the coil currents at the last draw's drifted values. A
        subsequent slice in a :meth:`set_slice` sweep must inherit none of that.
        Restore the pristine post-setup equilibrium (zero coils) captured in
        :meth:`setup_solver`, then re-apply the setup-time reg (the configured
        targets, or the toward-zero default when there are none -- which also
        refreshes the weak exploratory stash for THIS slice) and clear any
        stashed drift bounds.
        """
        mygs = self.mygs
        # full reset of the equilibrium + coil currents to the post-setup state
        if getattr(self, "_clean_eq", None) is not None:
            mygs.replace_eq(source_eq=self._clean_eq)
        self._apply_coil_reg(mygs)
        if hasattr(mygs, "_coil_drift_bounds"):
            mygs.set_coil_bounds(None)        # widen: prior slice had bounds set
            delattr(mygs, "_coil_drift_bounds")
        if hasattr(mygs, "_strong_coil_reg"):
            delattr(mygs, "_strong_coil_reg")

    def _repoint_imas_geometry(self):
        """Re-read THIS slice's LCFS boundary and re-point the solver isoflux.

        Each IMAS time slice is an *independent* equilibrium: its own boundary
        outline drives the isoflux constraints and the forward-solve psi init,
        so a multi-slice sweep (via :meth:`set_slice`) must not inherit the
        first slice's shape. Also resets the coil reg/bounds
        (:meth:`_reset_solver_state`) so the slice does not inherit the prior
        slice's coil constraints. F0 = R*B_t is set by the slow TF coils and is
        held fixed at :meth:`setup_solver` (changing it needs a fresh G-S
        setup); a slice whose F0 differs materially is flagged -- a true B_t
        ramp is out of scope for one solver. An explicit
        ``SolverConfig.isoflux_pts`` still overrides the per-slice boundary.
        """
        import numpy as np
        import warnings
        from .io.imas import read_imas_geometry

        sc = self.config.solver
        self._reset_solver_state()
        F0_slice, boundary_RZ = read_imas_geometry(self.config.source)
        self._boundary_RZ = boundary_RZ
        iso_pts, iso_w = sc.isoflux_pts, sc.isoflux_weights
        if iso_pts is None:
            iso_pts = boundary_RZ
            iso_w = np.ones(len(iso_pts)) * 500.0
        self.mygs.set_isoflux(iso_pts, weights=iso_w)
        if sc.F0 is None and getattr(self, "_F0", None) and \
                abs(F0_slice - self._F0) > 1e-3 * abs(self._F0):
            warnings.warn(
                f"IMAS slice F0={F0_slice:.4f} differs from the solver's "
                f"F0={self._F0:.4f} (set at setup). B_t is held fixed across "
                f"slices; a genuine B_t ramp needs a separate solver/process."
            )

    # ── stage 2: baseline (reconstruction OR imas) ----------------------
    def prepare_baseline(self) -> "Baseline":
        """Resolve the baseline from ``config.source`` and cache it.

        Delegates to :func:`bouquet.baseline.resolve_baseline`, which dispatches
        on source type. Generation depends only on the returned
        :class:`~bouquet.baseline.Baseline`, never on reconstruction directly.
        """
        from .baseline import resolve_baseline
        from .config import ImasSource

        # single_profile_jphi: drop the per-draw Sauter recompute BEFORE the
        # baseline work, so the IMAS forward solve does not spend a bootstrap
        # call either. The total j_phi is anchored to the source either way, so
        # the baseline equilibrium is unchanged -- only the (about to be
        # collapsed) split differs.
        if self.config.generation.single_profile_jphi:
            self.config.generation.recalculate_j_BS = False

        self.baseline = resolve_baseline(self.config, self.mygs)

        # IMAS path: read_imas_baseline does no GS solve, so establish a converged
        # baseline equilibrium on mygs here (the reconstruction path gets this for
        # free from reconstruct_equilibrium). This also sets l_i_target to the
        # TokaMaker-solved li_1 and records IDS-vs-TokaMaker li for sanity.
        if isinstance(self.config.source, ImasSource) and self.mygs is not None:
            # re-point the solver to THIS slice's boundary first, so a
            # multi-slice sweep treats each time as its own equilibrium
            self._repoint_imas_geometry()
            self._forward_solve_imas_baseline()

        # single_profile_jphi: collapse the decomposition so the archive matches
        # what the draws actually perturb (the total). Done AFTER the baseline
        # solve so the equilibrium itself is unchanged -- only the bookkeeping
        # split is folded back into j_inductive.
        if self.config.generation.single_profile_jphi:
            self._collapse_jphi_split()

        # Reconstruction path: surface a glanceable quality summary (the verbose
        # solver chatter was captured to baseline.reconstruction_log).
        if self.baseline.reconstruction_metrics is not None:
            self._print_reconstruction_summary()
        return self.baseline

    def _collapse_jphi_split(self) -> None:
        """Fold every j_phi component back into j_inductive (single-profile mode).

        Leaves ``baseline.j_phi`` untouched -- it is already the source total --
        and sets ``j_inductive = j_phi`` with the bootstrap and driven components
        zeroed, so nothing downstream can reintroduce a split. Also forces
        ``recalculate_j_BS=False``: with no baseline bootstrap there is nothing
        for the per-draw Sauter call to anchor to, and the draw path then perturbs
        the total directly (see TokaMaker_interface ~line 2136).
        """
        import numpy as np

        bl = self.baseline
        gc = self.config.generation
        j_phi = np.asarray(bl.j_phi, dtype=float)
        dropped = {
            name: float(np.max(np.abs(np.asarray(getattr(bl, name), dtype=float))))
            for name in ("j_BS", "j_NBI", "j_RF")
            if getattr(bl, name, None) is not None
        }
        bl.j_inductive = j_phi.copy()
        bl.j_BS = np.zeros_like(j_phi)
        for name in ("j_NBI", "j_RF"):
            if getattr(bl, name, None) is not None:
                setattr(bl, name, np.zeros_like(j_phi))
        gc.recalculate_j_BS = False          # already forced in prepare_baseline
        summary = ", ".join(f"{k} peak {v/1e6:.4f}" for k, v in dropped.items()) or "none"
        print(f"[single-profile] j_phi kept as one profile (peak "
              f"{np.max(np.abs(j_phi))/1e6:.4f} MA/m^2); folded in: {summary} "
              f"[MA/m^2]; recalculate_j_BS forced False (no Sauter per draw)")

    def reconstruct(self) -> "Baseline":
        """Reconstruct the baseline equilibrium and print a quality summary.

        Intent-revealing entry point for the reconstruction path: ensures the
        solver is up, runs the GS reconstruction, and prints the curated metrics
        so you can confirm at a glance that it succeeded before spending compute
        on the draws. Equivalent to ``setup_solver(); prepare_baseline()``.
        """
        from .config import ReconstructionSource

        if not isinstance(self.config.source, ReconstructionSource):
            raise TypeError(
                "reconstruct() is for a ReconstructionSource; the IMAS path has "
                "no reconstruction step -- call prepare_baseline() (or run())."
            )
        self.setup_solver()
        return self.prepare_baseline()

    def prepare(self) -> "Baseline":
        """Stand up the solver and resolve the baseline -- **either path** (F3).

        Symmetric with :meth:`reconstruct` (which is the reconstruction-path
        alias): ``prepare()`` = ``setup_solver(); prepare_baseline()`` and works
        for both the g-file and IMAS sources, each printing its own baseline
        quality summary. Use it when a notebook should read the same on both
        paths; ``reconstruct()`` remains for the g-file-specific intent.
        """
        self.setup_solver()
        return self.prepare_baseline()

    def _print_reconstruction_summary(self):
        """Print the reconstruction-fidelity block (TokaMaker vs input, % error).

        Global scalars are shown as ``value (input ref, +/-% err)`` against the
        input g-file's own values; geometric residuals that should be ~0
        (boundary, axis offset, j_phi RMS) are shown absolute. See
        :func:`bouquet.baseline._reconstruction_metrics`.
        """
        import numpy as np

        m = self.baseline.reconstruction_metrics
        tag = self.config.source.geqdsk_path.split("/")[-1]
        mark = "PASS ✅" if m.get("verdict") == "PASS" else "CHECK ⚠"
        # blank lines so the summary stands out after any solver output above
        print(f"\n\n=== Reconstruction — {tag} {'=' * max(3, 40 - len(tag))} {mark}")

        def line(label, val, ref, err, unit="", fmt=".3f"):
            u = f" {unit}" if unit else ""
            lhs = f"{format(val, fmt)}{u}"
            print(f"  {label:<12} {lhs:<13} (input {format(ref, fmt)}{u}, {err:+.2f}%)")

        print(f"  {'converged':<12} {'yes' if m.get('converged') else 'NO ⚠'}")
        line("Ip", m['Ip_MA'], m['Ip_efit_MA'], m['Ip_err_pct'], "MA")
        # l_i on the targeted estimator (matched pair -- ~0 by construction),
        # then the free cross-estimator pair which is NOT driven by anything
        # and so is the honest estimator-drift monitor (issue #20).
        line("l_i(3)", m['li'], m['li_efit'], m['li_err_pct'])
        if np.isfinite(m.get('li1_cross_err_pct', float('nan'))):
            line("  l_i(1)x", m['li1_cross'], m['li1_cross_efit'],
                 m['li1_cross_err_pct'])
        line("q0", m['q0'], m['q0_efit'], m['q0_err_pct'], fmt=".2f")
        line("q95", m['q95'], m['q95_efit'], m['q95_err_pct'], fmt=".2f")
        line("beta_N", m['beta_n'], m['beta_n_efit'], m['beta_n_err_pct'], fmt=".2f")
        line("beta_p", m['beta_p'], m['beta_p_efit'], m['beta_p_err_pct'], fmt=".2f")
        line("kappa", m['kappa'], m['kappa_efit'], m['kappa_err_pct'])
        line("delta", m['delta'], m['delta_efit'], m['delta_err_pct'])
        line("j_sep(.99)", m['j_sep_MA'], m['j_sep_efit_MA'], m['j_sep_err_pct'], "MA/m²")
        line("W_MHD", m['W_MHD_MJ'], m['W_MHD_efit_MJ'], m['W_MHD_err_pct'], "MJ")
        print(f"  {'boundary':<12} RMS {m['boundary_rms_mm']:.2f} mm   "
              f"max {m['boundary_max_mm']:.2f} mm   axis off {m['axis_offset_mm']:.2f} mm")
        print(f"  {'jphi resid':<12} core RMS {m['jphi_core_rms_MA']:.3f}   "
              f"edge RMS {m['jphi_edge_rms_MA']:.3f} MA/m²")
        # Step-6 matched (== l_i_target) vs step-7 realized, issue #25.  Printed
        # ALWAYS, not only when out of band -- a drift that surfaces only when
        # it breaches is a drift nobody watches shrink or grow.
        _lr = m.get('li_realized_post_corrective', float('nan'))
        if np.isfinite(_lr):
            _dp = m.get('li_corrective_drift_pct', float('nan'))
            _bp = m.get('li_corrective_band_pct', float('nan'))
            _flag = ("  ⚠ OUTSIDE the l_i band"
                     if m.get('li_corrective_out_of_band') else "")
            print(f"  {'l_i post-7':<12} {_lr:.5f}       "
                  f"(target {m['li']:.5f} = step-6 matched, {_dp:+.3f}%, "
                  f"band ±{_bp:.2f}%){_flag}")

    @staticmethod
    def _close_ip_q0_predictor(gc, bl, eq_snap, geom, probe, psi_N,
                               j_ind, j_BS_swb, j_fixed, FUSE_tot,
                               sgn, Ip_t, c_signed, ip_ind, ip_bs, ip_fix,
                               close_ip):
        """Solve-free (s_ohm, s_bs) for ``closure_channel="sawtooth_bootstrap"``.

        ``c_signed`` is the P'-term constant ALREADY paired with the data's
        current-direction convention (``utils.closure_sign_convention``), the
        same pair the plain channels close on: the 2x2 here solves the same
        affine Ip identity, so it must not see the anchor's unsigned ``c``.

        Returns ``(ohm_scale, bs_scale, extra, state)`` -- ``extra`` is merged
        into ``Baseline.ip_closure``; ``state`` is what the post-solve corrector
        needs (``None`` when the gate rejected the slice and the plain bootstrap
        channel took over, since there is then nothing to correct).

        **The reference is FUSE's own total at its OWN current -- the
        REQUESTED profile, not the renormalised anchor.**  The anchor snapshot
        handed in here is, by construction, the converged forward solve of the
        source's own total (``_forward_solve_imas_baseline`` does
        ``solve_jphi(bl.j_phi)`` and only then takes ``copy_eq()``, before
        ``solve_with_bootstrap`` moves the equilibrium), so ``q0_anchor`` --
        TokaMaker's own q for that solve, never the dd's q estimator (issue
        #20) -- costs no dedicated solve.  But ``solve_jphi`` hands TokaMaker a
        jphi-linterp SHAPE and TokaMaker renormalises it to Ip_target, and
        FUSE's ``core_profiles`` total does not carry Ip (-3.89 % on the
        reference validation slice), so the anchor actually ran on ~1.039x the
        requested profile.  ``q0_anchor`` therefore belongs to a rescaled
        current that FUSE never claimed.

        Pinning to it would propagate that known DATA artefact into the current
        split -- every other channel absorbs the Ip deficit into ONE scale and
        leaves the shape alone.  So the target is un-renormalised back to
        FUSE's own current, to the same first order the whole predictor uses
        (``q0 ~ 1/j_phi(0)`` at frozen geometry):

        .. code-block:: text

            q0_target = q0_anchor * (j_achieved(0) / j_requested(0))

        and the axis row is matched against ``j_requested(0)`` = the source
        total at the clipped axis.  Both axis currents and their ratio are
        recorded so the un-renormalisation is auditable.

        **What the gate actually tests** (``utils.q0_gate_admits``): the slice
        is admitted when the source's sawtooth model is ACTIVE there, or, when
        it is not, when the source's OWN axis safety factor ``|q0_dd|`` is at
        or below ``q0_gate``.  ``|q0_target|`` -- the estimator mapping
        un-renormalised above -- is the fallback basis, used ONLY when the
        source carries no axis q at all; it reads systematically lower than
        ``q0_dd``, and gating on it admitted idle-sawtooth ramp slices whose
        own ``q0_dd`` sat above the threshold.  The basis actually used is
        recorded as ``q0_gate_basis``.  Magnitudes throughout: q carries a
        COCOS sign and a negative value would pass ``<= q0_gate`` trivially.

        Consequence, and the point of the channel: where the recomputed
        bootstrap has negligible core content this drives ``s_ohm -> ~1`` and
        the mode reduces to ``"bootstrap"``, as the plan predicts (measured
        0.9977 on the reference validation slice).  It diverges only where the
        bootstrap carries real core current, which is exactly the regime the
        q0 pin exists for.

        **Measured accuracy of the first-order model.**  Matching the axis
        current exactly (it IS exact by construction, to 1e-15) still left
        q0 = 0.9871 against a target of 0.9794 on the reference slice -- an
        0.8 % model error, comparable to the ~0.9 % scale adjustment being
        made.  The Ip-closed hybrid reproduces its requested current as an
        INTEGRAL but not pointwise on axis: the single-pass jphi-linterp solve
        lands the achieved j_phi a fraction of a percent off the request, and
        the re-converged geometry is not quite the frozen anchor either.  So
        ``q0_tol`` is not a numerical nicety -- it is the band inside which
        this linearisation is trustworthy, and a slice whose residual exceeds
        it needs the corrector's MEASURED ``dq0/ds_ohm``, not a wider band.
        """
        import numpy as np

        from .utils import close_ip_q0, unrenormalise_q0

        psi_q = np.ascontiguousarray(np.asarray(geom["psi_q"], dtype=float))
        psi_geom = np.asarray(geom["psi_N"], dtype=float)
        # Every axis value at the SAME sample: psi_q[0], the psi_pad-clipped
        # axis. Never psi_N = 0 exactly -- get_q silently collapses the surface
        # tracer onto the magnetic axis there (fsa_current_geometry docstring).
        _ax = lambda j: float(np.interp(psi_q[0], psi_geom,
                                        np.asarray(j, dtype=float)))
        q0_anchor = float(np.asarray(eq_snap.get_q(psi=psi_q.copy())[1],
                                     dtype=float)[0])
        # ACHIEVED: the anchor's GS-reconstructed own profile (round-trips to
        # its achieved Ip).  REQUESTED: the source total that was handed in.
        # Their ratio IS TokaMaker's Ip renormalisation of the anchor.
        j_achieved0 = float(np.asarray(probe, dtype=float)[0])
        j_requested0 = _ax(FUSE_tot)
        # The algebra lives in utils.unrenormalise_q0 so the tests exercise
        # the SHIPPED formula, not a re-derivation (same rule as close_ip).
        q0_target = unrenormalise_q0(q0_anchor, j_achieved0, j_requested0)
        j_renorm_ratio = j_achieved0 / j_requested0
        j_ref0 = j_requested0
        j_ind0, j_bs0, j_fix0 = _ax(j_ind), _ax(j_BS_swb), _ax(j_fixed)

        saw = dict(getattr(bl, "sawtooth", None) or {})
        q0_dd = saw.get("q0_dd")
        saw_active = bool(saw.get("active"))
        q0_gate = float(getattr(gc, "q0_gate", 1.1))
        # The fallback basis is q0_TARGET -- the q0 actually being claimed
        # for FUSE's own current -- not the renormalised anchor value, which
        # on a slice with a large Ip deficit can sit on the other side of the
        # threshold.  abs(): q carries a COCOS sign (this dd's own q[0] reads -0.99), and a
        # negative target would make "q0 <= q0_gate" trivially true and bypass
        # the gate silently.  Only the COMPARISON is on the magnitude -- the
        # raw signed values are what get recorded, and the residual/Newton
        # algebra is sign-agnostic because target and solved q share the
        # estimator.
        # Gate on the SOURCE's own |q0_dd| (physically clamped on a sawtoothing
        # discharge) OR sawtooth activity; q0_target is the estimator mapping
        # and reads lower -- gating on it admitted idle-sawtooth ramp slices.
        from .utils import q0_gate_admits
        gated, gate_basis = q0_gate_admits(saw_active, q0_dd, q0_target,
                                           q0_gate)
        print(f"[imas SWB-split:ohmic q0] q0_target={q0_target:.4f} "
              f"(= q0_anchor {q0_anchor:.4f} x j_achieved/j_requested "
              f"{j_renorm_ratio:.4f}, un-renormalised onto FUSE's own current; "
              f"psi_N={psi_q[0]:.1e}) | "
              f"q0_dd={'n/a' if q0_dd is None else format(q0_dd, '.4f')} | "
              f"sawtooth source {'ACTIVE' if saw_active else ('idle' if saw.get('present') else 'absent')}"
              f" (index {saw.get('source_index')}, max|j_par|="
              f"{saw.get('j_par_max_abs', 0.0):.3e} A/m^2) | q0_gate={q0_gate:g} "
              f"on {gate_basis}", flush=True)

        extra = dict(
            q0_target=q0_target,
            q0_anchor=q0_anchor,
            q0_target_source=(
                "q0_anchor * (j_achieved0/j_requested0): the anchor copy_eq "
                "snapshot is the converged forward solve of the source total, "
                "but solve_jphi renormalises that SHAPE to Ip_target, so its "
                "q0 belongs to a current FUSE never claimed; the ratio undoes "
                "that to first order (q0 ~ 1/j_phi(0) at frozen geometry). "
                "get_q at the psi_pad-clipped axis; never the dd's own q "
                "estimator (issue #20)"),
            q0_target_psi_N=float(psi_q[0]),
            q0_dd=q0_dd,
            q0_gate=q0_gate,
            q0_gate_basis=gate_basis,
            sawtooth_active=saw_active,
            sawtooth_present=bool(saw.get("present", False)),
            sawtooth_j_par_max_abs=float(saw.get("j_par_max_abs", 0.0)),
            j_ref0_achieved=j_achieved0,
            j_ref0_requested=j_requested0,
            j_renorm_ratio=j_renorm_ratio,
            j_ref0_used=j_ref0,
            j_ind0=j_ind0, j_bs0=j_bs0, j_fix0=j_fix0,
            n_extra_solves=0,
        )
        if not gated:
            _gq = q0_dd if gate_basis.startswith("|q0_dd|") else q0_target
            print("[imas SWB-split:ohmic q0] GATE REJECTED: no active sawtooth "
                  f"source and {gate_basis} {abs(float(_gq)):.4f} > q0_gate {q0_gate:g} -- the "
                  "q0 pin is not physically justified here (reversed shear / "
                  "early ramp: the source's own q0 is model-dependent). "
                  "Falling back to closure_channel='bootstrap'.", flush=True)
            ohm_scale, bs_scale = close_ip(
                "bootstrap", sgn * Ip_t, c_signed, ip_ind, ip_bs, ip_fix)
            extra["sawtooth_verdict"] = "gate rejected -> bootstrap fallback"
            return ohm_scale, bs_scale, extra, None

        ohm_scale, bs_scale = close_ip_q0(
            sgn * Ip_t, c_signed, ip_ind, ip_bs, ip_fix,
            j_ind0, j_bs0, j_fix0, j_ref0)
        j0_pred = ohm_scale * j_ind0 + bs_scale * j_bs0 + j_fix0
        # First-order predicted q0 of the closed hybrid: q0 ~ 1/j0 at frozen
        # geometry, so q0_pred = q0_target * j_ref0/j0_pred -- identically
        # q0_target when the 2x2 solved exactly.  Kept as an explicit record
        # because a bounds-clipped or degenerate solve shows up here first.
        extra.update(
            sawtooth_verdict="gate admitted -> predictor",
            q0_predictor_ohm_scale=float(ohm_scale),
            q0_predictor_bs_scale=float(bs_scale),
            q0_axis_current_target=j_ref0,
            q0_axis_current_predicted=float(j0_pred),
            q0_predicted=float(q0_target * j_ref0 / j0_pred) if j0_pred else None,
        )
        print(f"[imas SWB-split:ohmic q0] predictor 2x2 (0 extra solves): "
              f"s_ohm={ohm_scale:.4f} s_bs={bs_scale:.4f}; axis j "
              f"{j0_pred/1e6:.4f} -> target {j_ref0/1e6:.4f} MA/m^2 "
              f"(= FUSE's own requested axis j; the anchor's ACHIEVED was "
              f"{j_achieved0/1e6:.4f}, ratio {j_renorm_ratio:.4f})", flush=True)
        state = dict(
            q0_target=q0_target, psi_q=psi_q,
            j_ind=np.asarray(j_ind, dtype=float),
            j_BS_swb=np.asarray(j_BS_swb, dtype=float),
            j_fixed=np.asarray(j_fixed, dtype=float),
            j_ind0=j_ind0, j_bs0=j_bs0, j_fix0=j_fix0,
            ip_ind=float(ip_ind), ip_bs=float(ip_bs),
            ohm_scale=float(ohm_scale), bs_scale=float(bs_scale),
            q0_tol=float(getattr(gc, "q0_tol", 0.01)),
            Ip_t=float(Ip_t), sgn=float(sgn), c_signed=float(c_signed),
            ip_fix=float(ip_fix),
        )
        return ohm_scale, bs_scale, extra, state

    @staticmethod
    def _close_ip_structured_predictor(gc, bl, eq_snap, geom, probe, psi_N,
                                       j_ind, j_BS_swb, j_fixed, FUSE_tot,
                                       sgn, Ip_t, w_lin, c_signed,
                                       ip_ind, ip_bs, ip_fix,
                                       psi_pad=1e-3, pprime_sign=1.0):
        """Solve-free multiplier PROFILES for ``closure_channel="structured"``.

        Returns ``(s_ind, s_bs, ohm_eff, bs_eff, extra, state)``.  ``s_ind`` and
        ``s_bs`` are arrays on the kinetic grid; ``ohm_eff``/``bs_eff`` are the
        Ip-weighted means that go into ``Baseline.ohm_scale``/``bs_scale`` and
        into :func:`~bouquet.utils.closure_health` (they satisfy the scalar
        closure equation exactly -- see ``close_ip_structured``).  ``extra`` is
        merged into ``Baseline.ip_closure``; ``state`` is what the post-solve
        corrector needs, and is ``None`` when the sawtooth gate rejected the
        slice (no axis row was imposed, so there is no q0 to correct).

        ``c_signed`` is the P'-term constant ALREADY paired with the data's
        current-direction convention (``utils.closure_sign_convention``): this
        channel closes the SAME affine Ip identity as the scalar ones, so it
        must not see the anchor's unsigned ``c``.

        Everything about the reference and the gate is shared VERBATIM with the
        ``sawtooth_bootstrap`` predictor -- same ``unrenormalise_q0``, same
        ``q0_gate_admits``, same psi_pad-clipped axis sample -- so a
        structured-vs-scalar comparison is a comparison of the CLOSURE and not
        of two different q0 references.  The only difference is what absorbs
        the deficit: two numbers there, two smooth profiles here.

        Where the gate rejects, the Ip row alone is imposed and the answer is
        the minimal-norm radial closure of Ip -- NOT a fallback to
        ``"bootstrap"``.  That is deliberate and is the whole point of the
        channel: the scalar channels fall back because a scalar has nothing
        else to do, while here the trust weights still say where the deficit
        belongs.  The rejection is printed and recorded either way.
        """
        import numpy as np

        from .utils import (close_ip_structured, close_ip_structured_soft,
                            li_closure_geometry, sigma_from_weights,
                            structured_basis_eval, unrenormalise_q0,
                            q0_gate_admits)

        psi_q = np.ascontiguousarray(np.asarray(geom["psi_q"], dtype=float))
        psi_geom = np.asarray(geom["psi_N"], dtype=float)
        _ax = lambda j: float(np.interp(psi_q[0], psi_geom,
                                        np.asarray(j, dtype=float)))
        q0_anchor = float(np.asarray(eq_snap.get_q(psi=psi_q.copy())[1],
                                     dtype=float)[0])
        j_achieved0 = float(np.asarray(probe, dtype=float)[0])
        j_requested0 = _ax(FUSE_tot)
        q0_target = unrenormalise_q0(q0_anchor, j_achieved0, j_requested0)
        j_ref0 = j_requested0
        j_ind0, j_bs0, j_fix0 = _ax(j_ind), _ax(j_BS_swb), _ax(j_fixed)

        saw = dict(getattr(bl, "sawtooth", None) or {})
        q0_dd = saw.get("q0_dd")
        saw_active = bool(saw.get("active"))
        q0_gate = float(getattr(gc, "q0_gate", 1.1))
        gated, gate_basis = q0_gate_admits(saw_active, q0_dd, q0_target,
                                           q0_gate)
        basis_spec = getattr(gc, "structured_basis", None)
        wspec = getattr(gc, "structured_weights", None)
        axis = (dict(psi=float(psi_q[0]), j_ind0=j_ind0, j_bs0=j_bs0,
                     j_fix0=j_fix0, j_ref0=j_ref0) if gated else None)
        print(f"[imas SWB-split:ohmic structured] q0_target={q0_target:.4f} "
              f"(q0_anchor {q0_anchor:.4f} x j_achieved/j_requested "
              f"{j_achieved0 / j_requested0:.4f}) | q0_dd="
              f"{'n/a' if q0_dd is None else format(q0_dd, '.4f')} | "
              f"sawtooth {'ACTIVE' if saw_active else 'idle/absent'} | "
              f"gate {'ADMITS' if gated else 'REJECTS'} on {gate_basis} -> "
              f"{'Ip + axis-current rows' if gated else 'Ip row only'}",
              flush=True)

        # ---- the second global measurement: l_i ----------------------------
        # A PLAIN INPUT (GenerationConfig.structured_li_target): bouquet never
        # fetches it, never guesses which code produced it and applies no
        # definition offset -- whatever cross-code offset the caller's l_i
        # carries must already be in the number handed in.
        li_target = getattr(gc, "structured_li_target", None)
        li_sigma = getattr(gc, "structured_li_sigma", None)
        li_kind = str(getattr(gc, "structured_li_kind", "li_1"))
        ip_sigma = getattr(gc, "structured_ip_sigma", None)
        ip_sigma_frac = getattr(gc, "structured_ip_sigma_frac", None)
        if ip_sigma is not None and ip_sigma_frac is not None:
            raise RuntimeError(
                "closure_channel='structured': structured_ip_sigma and "
                "structured_ip_sigma_frac are mutually exclusive (got "
                f"{ip_sigma!r} A and {ip_sigma_frac!r} x Ip); set one, so the "
                "recorded sigma_Ip is unambiguous.")
        if ip_sigma_frac is not None:
            ip_sigma = float(ip_sigma_frac) * float(Ip_t)
        soft = bool(getattr(gc, "structured_soft", False))
        # The UP side of the one-sided inductive prior (None = symmetric, i.e.
        # byte-identical to every run made before the field existed).  Both
        # solvers take it; the sign iteration lives inside them.
        sig_ind_up = getattr(gc, "structured_sigma_ind_up", None)
        if sig_ind_up is not None:
            sig_ind_up = [float(v) for v in sig_ind_up]
        li_geom = None
        if li_target is not None:
            li_target = float(li_target)
            li_geom = li_closure_geometry(eq_snap, geom,
                                          convention="jphi-linterp",
                                          pprime_sign=float(pprime_sign),
                                          psi_pad=float(psi_pad))
        if soft and li_target is not None and li_sigma is None:
            raise RuntimeError(
                "closure_channel='structured' with structured_soft=True and "
                "structured_li_target set needs structured_li_sigma: the soft "
                "channel weights l_i by its own error bar and bouquet will "
                "not invent one.  Set the sigma, or use the hard channel "
                "(structured_soft=False), which imposes the target exactly.")
        if soft:
            _K = structured_basis_eval(basis_spec, psi_geom).shape[0]
            _sig = sigma_from_weights(wspec, _K)
            out = close_ip_structured_soft(
                psi_geom, w_lin, c_signed, sgn * Ip_t,
                (None if ip_sigma is None else float(ip_sigma)),
                j_ind, j_BS_swb, j_fixed,
                basis=basis_spec, sigma_ind=_sig["ind"], sigma_bs=_sig["bs"],
                sigma_ind_up=sig_ind_up,
                li_target=li_target,
                li_sigma=(None if li_sigma is None else float(li_sigma)),
                li_kind=li_kind, li_geom=li_geom,
                axis=axis, axis_sigma=None)   # the q0 pin stays HARD
        else:
            out = close_ip_structured(
                psi_geom, w_lin, c_signed, sgn * Ip_t,
                j_ind, j_BS_swb, j_fixed,
                basis=basis_spec, weights=wspec, axis=axis,
                li_target=li_target, li_kind=li_kind, li_geom=li_geom,
                sigma_ind_up=sig_ind_up)

        s_ind = np.asarray(out["s_ind"], dtype=float)
        s_bs = np.asarray(out["s_bs"], dtype=float)
        # s at a few named radii, so the record is readable without the arrays
        _radii = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)
        _at = lambda s: {f"{r:.2f}": float(np.interp(r, psi_geom, s))
                         for r in _radii}
        extra = dict(
            structured_basis=dict(out["basis"]),
            structured_weights_name=out["weights_name"],
            structured_weights_ind=[float(v) for v in out["weights_ind"]],
            structured_weights_bs=[float(v) for v in out["weights_bs"]],
            structured_sigma_ind_up=(
                None if out.get("sigma_ind_up") is None
                else [float(v) for v in out["sigma_ind_up"]]),
            structured_one_sided_ind=bool(out.get("one_sided_ind", False)),
            structured_sign_pattern=(
                None if out.get("sign_pattern") is None
                else "".join("+" if v else "-" for v in out["sign_pattern"])),
            structured_n_sign_iter=int(out.get("n_sign_iter", 0) or 0),
            structured_coeffs_a=[float(v) for v in out["a"]],
            structured_coeffs_b=[float(v) for v in out["b"]],
            structured_s_ind_at=_at(s_ind),
            structured_s_bs_at=_at(s_bs),
            structured_s_ind_profile=[float(v) for v in s_ind],
            structured_s_bs_profile=[float(v) for v in s_bs],
            structured_s_ind_min=float(s_ind.min()),
            structured_s_ind_max=float(s_ind.max()),
            structured_s_bs_min=float(s_bs.min()),
            structured_s_bs_max=float(s_bs.max()),
            structured_structure_ind=float(out["structure_ind"]),
            structured_structure_bs=float(out["structure_bs"]),
            structured_constraints=list(out["constraints"]),
            structured_ip_residual=float(out["ip_residual"]),
            structured_ip_residual_pct=float(out["ip_residual_pct"]),
            # The closure's OWN Ip, as a magnitude [A].  On the HARD channel Ip
            # is imposed exactly and this is Ip_target to machine precision; on
            # the SOFT channel it is the POSTERIOR Ip, which differs from the
            # measurement BY DESIGN (the measurement carries a sigma).  It is
            # the reference the closed hybrid's round-trip gate compares
            # against on the soft channel -- see that gate's comment.
            structured_ip_posterior=abs(float(out["Ip_hybrid"])),
            structured_axis_residual=(None if out["axis_residual"] is None
                                      else float(out["axis_residual"])),
            structured_kkt_cond=(None if out["kkt_cond"] is None
                                 else float(out["kkt_cond"])),
            # The prior-INDEPENDENT conditioning of the constraint rows --
            # the number the refusal is actually taken on (structured_kkt_cond
            # carries the prior's dynamic range and is only a diagnostic), so
            # a post-hoc audit of why a slice was or was not refused has it.
            structured_constraint_cond=(None if out.get("constraint_cond")
                                        is None
                                        else float(out["constraint_cond"])),
            structured_deficit=float(out["deficit"]),
            structured_solver=out["solver"],
            structured_soft=bool(soft),
            structured_ohm_scale_eff_basis=out["ohm_scale_eff_basis"],
            structured_bs_scale_eff_basis=out["bs_scale_eff_basis"],
            ohm_scale_is_effective_mean=True,
            q0_target=q0_target, q0_anchor=q0_anchor,
            q0_target_source=(
                "same reference as sawtooth_bootstrap: q0_anchor * "
                "(j_achieved0/j_requested0), un-renormalised onto the source's "
                "own current; get_q at the psi_pad-clipped axis"),
            q0_target_psi_N=float(psi_q[0]),
            q0_dd=q0_dd, q0_gate=q0_gate, q0_gate_basis=gate_basis,
            sawtooth_active=saw_active,
            sawtooth_present=bool(saw.get("present", False)),
            sawtooth_j_par_max_abs=float(saw.get("j_par_max_abs", 0.0)),
            j_ref0_achieved=j_achieved0, j_ref0_requested=j_requested0,
            j_renorm_ratio=j_achieved0 / j_requested0,
            j_ref0_used=j_ref0,
            j_ind0=j_ind0, j_bs0=j_bs0, j_fix0=j_fix0,
            n_extra_solves=0,
            sawtooth_verdict=("gate admitted -> Ip + axis rows"
                              if gated else
                              "gate rejected -> Ip row only (minimal-norm "
                              "radial closure; NOT a bootstrap fallback)"),
        )
        # ---- the l_i block of the record -----------------------------------
        extra.update(
            structured_li_kind=out["li_kind"],
            structured_li_target=out["li_target"],
            structured_li_anchor=out["li_anchor"],
            structured_li_anchor_at_target_Ip=out["li_anchor_at_target_Ip"],
            structured_li_predicted=out["li_predicted"],
            structured_li_predictor_residual=out["li_predictor_residual"],
            structured_li_sigma=(None if li_sigma is None else float(li_sigma)),
            structured_ip_sigma=(None if ip_sigma is None else float(ip_sigma)),
            structured_ip_sigma_frac=(None if ip_sigma_frac is None
                                      else float(ip_sigma_frac)),
            structured_li_tol=float(getattr(gc, "structured_li_tol", 0.005)),
            structured_li_target_source=(
                None if li_target is None else
                "GenerationConfig.structured_li_target -- a plain caller input; "
                "bouquet applies no cross-code definition offset to it"),
            structured_li_geometry=(
                None if li_geom is None else
                dict(vol=float(li_geom["vol"]),
                     perimeter=float(li_geom["perimeter"]),
                     perimeter_source=li_geom["perimeter_source"],
                     perimeter_get_stats_dl=float(
                         li_geom["perimeter_get_stats_dl"]),
                     perimeter_ratio_dl_over_L=float(
                         li_geom["perimeter_ratio_dl_over_L"]),
                     R_axis=float(li_geom["R_axis"]),
                     dpsi_dpsiN=float(li_geom["dpsi_dpsiN"]),
                     psi_pad=float(li_geom["psi_pad"]))),
            # MODEL-space, predictor stage: the posterior mode's own fit to its
            # own Ip row.  ``structured_residual_sigma_Ip`` (below, and
            # refreshed by the corrector) is the ACHIEVED distance of the
            # delivered closure's Ip from the MEASUREMENT, in sigma units --
            # the same predictor/achieved split the l_i rows use.
            structured_residual_sigma_Ip_model=out.get("residual_sigma_Ip"),
            structured_residual_sigma_Ip=None,
            # MODEL-space, predictor stage: the posterior mode's own fit to its
            # own l_i row.  It is not what the solved equilibrium delivered --
            # the posterior always fits its own row well, so it essentially
            # never flags.  ``structured_residual_sigma_li`` is the ACHIEVED
            # (post-corrector) residual in sigma units and is written by
            # :meth:`_close_ip_structured_corrector`; see the l_i-bookkeeping
            # note in that method's docstring.
            structured_residual_sigma_li_model=out.get("residual_sigma_li"),
            structured_residual_sigma_li=None,
            structured_residual_sigma_axis=out.get("residual_sigma_axis"),
            structured_objective=out.get("objective"),
            structured_prior_chi2=out.get("prior_chi2"),
            structured_gn_iterations=out.get("n_iter"),
        )
        if li_target is not None:
            print("[imas SWB-split:ohmic structured] l_i row "
                  f"({out['li_kind']}, {'SOFT sigma=' + format(float(li_sigma), '.4f') if soft else 'HARD'}"
                  f"): anchor(s==1) {out['li_anchor']:.4f} -> predicted "
                  f"{out['li_predicted']:.4f} vs target {out['li_target']:.4f} "
                  f"(residual {out['li_predictor_residual']:+.2e}); geometry "
                  f"vol={li_geom['vol']:.3f} m^3, LCFS perimeter="
                  f"{li_geom['perimeter']:.3f} m, R_axis="
                  f"{li_geom['R_axis']:.4f} m", flush=True)
        if soft:
            print("[imas SWB-split:ohmic structured] SOFT posterior mode: "
                  f"sigma_Ip={'hard' if ip_sigma is None else format(float(ip_sigma), '.4g')} "
                  f"| objective {out['objective']:.4e} (prior chi2 "
                  f"{out['prior_chi2']:.4e}) in {out['n_iter']} "
                  f"Gauss-Newton iteration(s); z_Ip="
                  f"{'n/a' if out['residual_sigma_Ip'] is None else format(out['residual_sigma_Ip'], '+.3f')}"
                  f" z_li="
                  f"{'n/a' if out['residual_sigma_li'] is None else format(out['residual_sigma_li'], '+.3f')}",
                  flush=True)
        print(f"[imas SWB-split:ohmic structured] s_ind {s_ind.min():.3f}"
              f"-{s_ind.max():.3f} (eff {out['ohm_scale_eff']:.4f}, structure "
              f"{out['structure_ind']:.4f}) | s_bs {s_bs.min():.3f}"
              f"-{s_bs.max():.3f} (eff {out['bs_scale_eff']:.4f}, structure "
              f"{out['structure_bs']:.4f}) | weights={out['weights_name']} | "
              f"Ip residual {out['ip_residual_pct']:+.2e}% | "
              + ("KKT cond n/a (soft)" if out["kkt_cond"] is None
                 else f"KKT cond {out['kkt_cond']:.2e}"), flush=True)

        # The corrector state is built whenever there is ANYTHING to correct:
        # the q0 row (gate admitted) or the l_i row.  Both corrections share
        # the SAME single extra solve.
        state = None
        if gated or li_target is not None:
            state = dict(
                q0_target=q0_target, psi_q=psi_q, psi_geom=psi_geom,
                j_ind=np.asarray(j_ind, dtype=float),
                j_BS_swb=np.asarray(j_BS_swb, dtype=float),
                j_fixed=np.asarray(j_fixed, dtype=float),
                axis=(None if axis is None else dict(axis)),
                w_lin=np.asarray(w_lin, dtype=float),
                c_signed=float(c_signed), Ip_signed=float(sgn * Ip_t),
                ip_ind=float(ip_ind), ip_bs=float(ip_bs),
                ip_fix=float(ip_fix),
                basis=basis_spec, weights=wspec,
                q0_tol=float(getattr(gc, "q0_tol", 0.01)),
                gated=bool(gated),
                soft=bool(soft), ip_sigma=ip_sigma,
                sigma_ind=(None if not soft else _sig["ind"]),
                sigma_bs=(None if not soft else _sig["bs"]),
                sigma_ind_up=sig_ind_up,
                li_target=li_target, li_sigma=li_sigma, li_kind=li_kind,
                li_geom=li_geom, psi_pad=float(psi_pad),
                li_tol=float(getattr(gc, "structured_li_tol", 0.005)),
                li_max_corrector_steps=int(
                    getattr(gc, "structured_li_max_corrector_steps", 1) or 1),
            )
        return (s_ind, s_bs, float(out["ohm_scale_eff"]),
                float(out["bs_scale_eff"]), extra, state)

    @staticmethod
    def _structured_roundtrip_gate(Ip_measured):
        """The post-corrector round-trip gate, with the measurement bound.

        :meth:`_close_ip_structured_corrector` calls its ``roundtrip_gate``
        with ONE positional (the assembled Ip) plus the soft channel's
        ``posterior``/``sigma_Ip``, so the measurement has to be closed over
        HERE -- :func:`bouquet.utils.ip_roundtrip_gate` takes it as a REQUIRED
        positional, and handing that function over raw is a ``TypeError`` at
        the first corrector that runs.  Named rather than written inline at
        the call site so that a test can pin the object production actually
        passes against the contract the corrector actually uses.

        Nothing about the acceptance moves: ``posterior``/``sigma_Ip`` pass
        straight through, so the soft channel is still compared against its
        own posterior and the hard channel against *Ip_measured*, on the one
        unchanged ``IP_ROUNDTRIP_TOL_PCT`` budget.
        """
        from .utils import ip_roundtrip_gate

        def _gate(ip_closed, posterior=None, sigma_Ip=None):
            return ip_roundtrip_gate(ip_closed, Ip_measured,
                                     posterior=posterior, sigma_Ip=sigma_Ip)
        return _gate

    @staticmethod
    def _close_ip_structured_corrector(state, bl, mygs, solve_jphi,
                                       ip_of=None, roundtrip_gate=None):
        """Correct q0 and/or l_i after the structured solve, at most 2 solves.

        Same contract and (by default) the same cost ceiling as
        :meth:`_close_ip_q0_corrector`: read the solved equilibrium off a
        ``copy_eq()`` snapshot, accept the predictor if every constrained
        quantity is inside its tolerance, otherwise take ONE step and accept
        whatever that gives.  When BOTH the q0 row and the l_i row miss, they
        are corrected TOGETHER in that single re-solve -- the cost ceiling is
        one extra GS solve per slice, not one per constraint.

        What moves is the constraint rows' right-hand sides, not the
        coefficients.  The Ip-closed set is a (2K-1)-dimensional affine
        manifold with no privileged direction on it, but each constraint has a
        privileged row, and a model calibrated on the measured pair inverts
        exactly:

        .. code-block:: text

            j_ref0' = j_ref0 * (q0_solved / q0_target)        (q0 ~ 1/j_phi(0))
            li_row' = li_row * (li_target / li_solved)**(1/p) (achieved ~ row**p)

        The first is the ``q0 ~ 1/j0`` frozen-geometry model (the scalar
        corrector's Newton step is its first-order expansion).

        The second is a LOG-GAIN model for l_i, with ``p =
        utils.LI_GAIN_EXPONENT = 2``, applied identically in hard and soft
        mode.  The predictor's row is EXACT in the coefficient algebra -- with
        Ip pinned, l_i is linear in the coefficients and the row is satisfied
        to machine precision -- so everything that survives into the solved
        equilibrium is the frozen-anchor-geometry error.  That error is NOT a
        level offset the proportional model could absorb: the frozen quantity
        that actually moves is ``dpsi_dpsiN = |psi_axis - psi_bnd|``, and
        ``li_achieved / li_model = Delta_psi_solved / Delta_psi_anchor`` to
        ~1 %.  Since ``Delta_psi ~ sqrt(l_i)``, l_i enters the achieved
        equilibrium twice -- through the shape integral the closure prescribes
        AND through the flux range that responds to it -- so the achieved value
        follows the row as ``row**2``, not linearly.  Taking the square root of
        the ratio is therefore the parameter-free inverse; see
        :data:`~bouquet.utils.LI_GAIN_EXPONENT` and, for the campaign
        measurement behind it (gain 2.14 measured against the 0.96 the old
        proportional update assumed, sign-flipping the residual on 133/133
        hard slices), the out-of-tree closure-cloud corrector diagnosis.
        Multiplicative rather than additive because l_i is a positive quantity
        and the correction must stay scale-free.

        **Optional second step.**  With
        ``GenerationConfig.structured_li_max_corrector_steps = 2`` a second
        corrector solve is taken, CONDITIONALLY, on slices whose corrected l_i
        still misses ``structured_li_tol``.  After step 1 there are two
        measured ``(row, achieved)`` pairs on this slice, so step 2 reads the
        slice's own log-gain off them
        (:func:`~bouquet.utils.li_gain_exponent_secant`) instead of assuming
        one -- a secant iteration, no fitted constant, and it changes nothing
        about what "converged" means.  It corrects l_i only; the axis row keeps
        whatever step 1 gave it.  The default is 1, so the shipped behaviour
        differs from the previous release only by the gain law.

        **l_i bookkeeping.**  Three stages are recorded under unambiguous
        names:

        =================================== ==================================
        ``structured_li_achieved_predictor``  achieved l_i after the PREDICTOR
                                              solve
        ``structured_li_residual_predictor``  that minus the target
        ``structured_li_achieved_corrected``  achieved l_i after the LAST
                                              corrector solve (== the
                                              predictor's when no corrector
                                              solve was taken -- that IS what
                                              was delivered)
        ``structured_li_residual_corrected``  that minus the target: the number
                                              the run actually delivers
        =================================== ==================================

        The older names are kept as ALIASES of the CORRECTED values:
        ``structured_li_solved`` and ``structured_li_residual`` always were,
        and ``structured_li_solved_residual`` -- which used to be written once
        at the predictor stage and never refreshed, so that the field whose
        name read like the delivered residual was the one that was not -- now
        is too.  ``structured_li_solved_predictor`` keeps its (already
        unambiguous) predictor meaning.  ``structured_residual_sigma_li`` is
        the ACHIEVED corrected residual in sigma units; the posterior mode's
        own model-space fit to its own row, which is what that name used to
        hold, is under ``structured_residual_sigma_li_model``.

        **Ip bookkeeping.**  A corrector re-solve runs the SAME closure, so on
        the soft channel the Ip row is still a measurement with ``sigma_Ip``
        and the re-solve lands on a NEW posterior Ip.  It is recorded as such
        -- ``structured_ip_posterior``, ``structured_ip_measured_residual_pct``
        and the achieved ``structured_residual_sigma_Ip`` are all refreshed
        from the delivered solve -- and the soft-Ip closure-health flag is
        replaced rather than stacked.  Nothing here snaps Ip back to the
        measurement; the posterior IS the answer the channel was asked for.

        A corrected l_i still outside ``structured_li_tol`` on the hard channel
        adds a reason to ``closure_limited_reasons`` -- a FLAG, never a retry:
        the cost ceiling is the point, and the honest record of a missed hard
        row is a flagged slice, not an unbounded loop.

        A refusal from a re-solve (multipliers out of bounds, singular KKT, a
        non-converged posterior mode) is caught: the last accepted equilibrium
        is KEPT and the residuals recorded, exactly as the scalar corrector
        keeps the predictor when its step would leave the scale bounds.
        Returns the new ``nl_its`` when a corrector solve was taken, else
        ``None``.
        """
        import numpy as np

        from .utils import (LI_GAIN_EXPONENT, close_ip_structured,
                            close_ip_structured_soft, li_achieved,
                            li_corrector_row, li_gain_exponent_secant)

        gated = bool(state.get("gated", state.get("axis") is not None))
        soft = bool(state.get("soft"))
        li_target = state.get("li_target")
        li_kind = str(state.get("li_kind", "li_1"))
        li_tol = float(state.get("li_tol", 0.005))
        li_sigma = state.get("li_sigma")
        psi_pad = float(state.get("psi_pad", 1e-3))
        max_li_steps = max(1, int(state.get("li_max_corrector_steps", 1) or 1))
        q0_target = state["q0_target"]
        snap = mygs.copy_eq()

        gain_law = (f"row' = row * (target/achieved)**(1/p), p = "
                    f"{LI_GAIN_EXPONENT:g} (parameter-free: Delta_psi ~ "
                    "sqrt(l_i), so achieved ~ row**2); step 2 (when enabled) "
                    "uses this slice's own log-secant p")

        rec = {}
        reasons = []
        # Soft-channel achieved z_Ip, refreshed by every corrector re-solve;
        # None means "nothing new to say" and the predictor's flag stands.
        ip_z_final = None
        q0_tok = res = None
        if gated:
            q0_tok = float(np.asarray(snap.get_q(psi=state["psi_q"].copy())[1],
                                      dtype=float)[0])
            res = q0_tok - q0_target
            rec.update(q0_solved_predictor=q0_tok, q0_predictor_residual=res,
                       q0_tol=state["q0_tol"], q0_solved=q0_tok,
                       q0_residual=res)
        li_tok = li_res = None
        li_perimeter = (None if state.get("li_geom") is None
                        else float(state["li_geom"]["perimeter"]))
        if li_target is not None:
            # The solve's OWN l_i from TokaMaker's exact volume integrals
            # (get_globals), normalised the SAME way the target is -- with the
            # LCFS contour's perimeter, NOT get_stats' dl, which reads ~1 %
            # high and would charge the closure ~2 % of l_i(1) it did not do.
            # Reusing the predictor's own perimeter keeps target, predictor and
            # readback on one convention; see utils.lcfs_perimeter.
            li_tok, _li_info = li_achieved(snap, li_kind=li_kind,
                                           psi_pad=psi_pad,
                                           perimeter=li_perimeter)
            li_res = li_tok - float(li_target)
            rec.update(structured_li_solved_predictor=li_tok,
                       structured_li_achieved_predictor=li_tok,
                       structured_li_residual_predictor=li_res,
                       # "corrected" starts as the predictor and is refreshed
                       # after every corrector solve: with 0 extra solves the
                       # predictor IS the delivered equilibrium.
                       structured_li_achieved_corrected=li_tok,
                       structured_li_residual_corrected=li_res,
                       # aliases of the CORRECTED values (see the docstring)
                       structured_li_solved=li_tok,
                       structured_li_residual=li_res,
                       structured_li_solved_residual=li_res,
                       structured_residual_sigma_li=(
                           None if li_sigma in (None, 0)
                           else float(li_res) / float(li_sigma)),
                       structured_li_tol=li_tol,
                       structured_li_max_corrector_steps=max_li_steps,
                       structured_li_gain_law=gain_law,
                       structured_li_solved_get_stats=float(snap.get_stats(
                           lcfs_pad=psi_pad,
                           li_normalization=("std" if li_kind == "li_1"
                                             else "iter"))["l_i"]),
                       structured_li_estimator=(
                           "utils.li_achieved: TokaMaker get_globals "
                           "(Bp_vol, vol, Ip) in the target's normalisation, "
                           f"perimeter={li_perimeter!r} m from the anchor's "
                           "traced LCFS contour (NOT get_stats' dl)"))

        q0_usable = (gated and np.isfinite(q0_tok) and np.isfinite(q0_target)
                     and q0_target != 0.0)
        li_usable = (li_target is not None and np.isfinite(li_tok)
                     and li_tok > 0.0 and float(li_target) > 0.0)
        # BEFORE the tolerance tests: abs(nan) > tol is False, so a non-finite
        # solved q0 or l_i used to make BOTH want_* False and the slice was
        # reported as "predictor accepted, no extra solve" with
        # closure_limited untouched -- the readback failing is not the same
        # thing as the readback landing inside tolerance.  The q0_usable /
        # li_usable guards below could not catch it either: they were only
        # consulted once want_* had already been decided.
        q0_unreadable = bool(gated and not (np.isfinite(q0_tok)
                                            and np.isfinite(res)))
        li_unreadable = bool(li_target is not None
                             and not (np.isfinite(li_tok)
                                      and np.isfinite(li_res)))
        if q0_unreadable:
            reasons.append("q0 is not finite: no residual and no corrector "
                           "step")
        if li_unreadable:
            reasons.append("l_i is not finite: no residual and no corrector "
                           "step")
        want_q0 = bool(gated and not q0_unreadable
                       and abs(res) > state["q0_tol"])
        want_li = bool(li_target is not None and not li_unreadable
                       and abs(li_res) > li_tol)
        if want_q0 and not q0_usable:
            reasons.append("q0 reference unusable for a step")
            want_q0 = False
        if want_li and not li_usable:
            reasons.append("l_i reference unusable for a step")
            want_li = False

        def _resolve(axis_now, li_row_now):
            """One corrected minimal-norm / posterior-mode solve."""
            if soft:
                return close_ip_structured_soft(
                    state["psi_geom"], state["w_lin"], state["c_signed"],
                    state["Ip_signed"],
                    (None if state.get("ip_sigma") is None
                     else float(state["ip_sigma"])),
                    state["j_ind"], state["j_BS_swb"], state["j_fixed"],
                    basis=state["basis"], sigma_ind=state["sigma_ind"],
                    sigma_bs=state["sigma_bs"],
                    sigma_ind_up=state.get("sigma_ind_up"),
                    li_target=li_row_now,
                    li_sigma=(None if li_sigma is None else float(li_sigma)),
                    li_kind=li_kind, li_geom=state.get("li_geom"),
                    axis=axis_now, axis_sigma=None)
            return close_ip_structured(
                state["psi_geom"], state["w_lin"], state["c_signed"],
                state["Ip_signed"], state["j_ind"], state["j_BS_swb"],
                state["j_fixed"], basis=state["basis"],
                weights=state["weights"], axis=axis_now,
                li_target=li_row_now, li_kind=li_kind,
                li_geom=state.get("li_geom"),
                sigma_ind_up=state.get("sigma_ind_up"))

        nl_out = None
        if not (want_q0 or want_li):
            verdict = "structured predictor (0 extra solves)"
            if reasons:
                verdict = "structured predictor kept (" + "; ".join(reasons) + ")"
            rec.update(n_extra_solves=0, sawtooth_verdict=verdict)
            _bits = []
            if gated:
                _bits.append(f"q0={q0_tok:.4f} vs {q0_target:.4f} "
                             f"(residual {res:+.4f}, tol {state['q0_tol']:g})")
            if li_target is not None:
                _bits.append(f"l_i={li_tok:.4f} vs {float(li_target):.4f} "
                             f"(residual {li_res:+.4f}, tol {li_tol:g})")
            print("[imas SWB-split:ohmic structured] solved "
                  + "; ".join(_bits or ["(no constrained quantity)"])
                  + " -- predictor accepted, no extra solve", flush=True)
        else:
            axis = None if state.get("axis") is None else dict(state["axis"])
            j_ref_new = ratio_q0 = ratio_li = None
            # The predictor's own measured (row, achieved) pair: on the hard
            # channel the predictor row IS the target, imposed exactly.
            li_row = None if li_target is None else float(li_target)
            li_pairs = ([] if li_target is None
                        else [(float(li_row), float(li_tok))])
            li_exponents = []
            li_cur = li_tok
            q0_cur = q0_tok
            n_solves = 0
            step = 0
            while True:
                step += 1
                if want_q0 and step == 1:
                    ratio_q0 = float(q0_cur / q0_target)
                    j_ref_new = float(axis["j_ref0"] * ratio_q0)
                    axis["j_ref0"] = j_ref_new
                if want_li:
                    if step == 1:
                        p_use = float(LI_GAIN_EXPONENT)
                        ratio_li = float(li_target) / float(li_cur)
                    else:
                        # Two measured pairs on THIS slice -> read the log-gain
                        # off them rather than assuming it.  Falls back to the
                        # parameter-free exponent if the secant is degenerate.
                        p_sec = li_gain_exponent_secant(
                            li_pairs[-2][0], li_pairs[-2][1],
                            li_pairs[-1][0], li_pairs[-1][1])
                        p_use = (float(LI_GAIN_EXPONENT) if p_sec is None
                                 else float(p_sec))
                    li_exponents.append(p_use)
                try:
                    # INSIDE the try: li_corrector_row raises ValueError (not
                    # RuntimeError) on a non-positive or non-finite l_i, and
                    # the docstring above promises that a refusal here keeps
                    # the last accepted equilibrium rather than killing the
                    # run.
                    if want_li:
                        li_row = li_corrector_row(li_target, li_cur,
                                                  row=li_row, exponent=p_use)
                    out = _resolve(axis, li_row)
                except (RuntimeError, ValueError) as e:
                    _kept = ("predictor" if n_solves == 0
                             else f"corrector step {n_solves}")
                    rec.update(structured_corrector_refusal=str(e)[:300])
                    if n_solves == 0:
                        rec.update(n_extra_solves=0,
                                   sawtooth_verdict="structured predictor kept "
                                                    "(corrector step refused)")
                    print("[imas SWB-split:ohmic structured] corrector step "
                          f"{step} REFUSED ({str(e)[:160]}) -- keeping the "
                          f"{_kept} and recording the residuals", flush=True)
                    break

                s_ind = np.asarray(out["s_ind"], dtype=float)
                s_bs = np.asarray(out["s_bs"], dtype=float)
                # Overwrite the predictor's s-at-named-radii record too, not
                # just the min/max: every structured field in ip_closure must
                # describe the SAME (final) multiplier profiles, or a reader
                # comparing s_ind_at against s_ind_min is comparing two
                # different closures.
                _pg = np.asarray(state["psi_geom"], dtype=float)
                _at = lambda s: {f"{r:.2f}": float(np.interp(r, _pg, s))
                                 for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)}
                bl.ohm_scale = float(out["ohm_scale_eff"])
                bl.bs_scale = float(out["bs_scale_eff"])
                bl.j_inductive = s_ind * state["j_ind"]
                bl.j_BS = s_bs * state["j_BS_swb"]
                bl.j_phi = bl.j_inductive + bl.j_BS + state["j_fixed"]
                nl_out = solve_jphi(np.asarray(bl.j_phi, dtype=float))
                snap2 = mygs.copy_eq()
                n_solves += 1
                rec.update(
                    n_extra_solves=n_solves,
                    structured_coeffs_a=[float(v) for v in out["a"]],
                    structured_coeffs_b=[float(v) for v in out["b"]],
                    structured_s_ind_min=float(s_ind.min()),
                    structured_s_ind_max=float(s_ind.max()),
                    structured_s_bs_min=float(s_bs.min()),
                    structured_s_bs_max=float(s_bs.max()),
                    structured_s_ind_profile=[float(v) for v in s_ind],
                    structured_s_bs_profile=[float(v) for v in s_bs],
                    structured_s_ind_at=_at(s_ind),
                    structured_s_bs_at=_at(s_bs),
                    structured_sign_pattern=(
                        None if out.get("sign_pattern") is None else
                        "".join("+" if v else "-"
                                for v in out["sign_pattern"])),
                    structured_n_sign_iter=int(out.get("n_sign_iter", 0) or 0),
                    structured_structure_ind=float(out["structure_ind"]),
                    structured_structure_bs=float(out["structure_bs"]),
                    structured_ip_residual_pct=float(out["ip_residual_pct"]),
                    structured_corrector_targets=[
                        t for t, w in (("q0", want_q0), ("l_i", want_li)) if w])
                # The corrector re-solves the SAME closure -- on the soft
                # channel the Ip row is still the MEASUREMENT with its sigma,
                # so the re-solve lands on a new POSTERIOR Ip and must not be
                # read (or recorded) as snapping back to Ip_meas.  Refresh the
                # Ip bookkeeping from the delivered solve, in the same
                # model/achieved split the l_i rows use.
                _ip_hyb = out.get("Ip_hybrid")
                if _ip_hyb is not None and np.isfinite(_ip_hyb):
                    _ip_meas = abs(float(state["Ip_signed"]))
                    _ip_post = abs(float(_ip_hyb))
                    _sig_ip = state.get("ip_sigma")
                    _z = (None if not (soft and _sig_ip)
                          else (_ip_post - _ip_meas) / float(_sig_ip))
                    rec.update(
                        structured_ip_posterior=_ip_post,
                        structured_ip_residual=float(out.get(
                            "ip_residual", _ip_hyb - float(state["Ip_signed"]))),
                        structured_residual_sigma_Ip_model=out.get(
                            "residual_sigma_Ip"))
                    if soft:
                        rec.update(
                            structured_ip_measured_residual_pct=(
                                100.0 * (_ip_post - _ip_meas) / _ip_meas),
                            structured_residual_sigma_Ip=_z)
                        ip_z_final = _z
                if gated:
                    q0_cur = float(np.asarray(
                        snap2.get_q(psi=state["psi_q"].copy())[1],
                        dtype=float)[0])
                    rec.update(q0_solved=q0_cur,
                               q0_residual=q0_cur - q0_target)
                    if want_q0:
                        rec.update(structured_corrector_j_ref0=j_ref_new,
                                   structured_corrector_j_ref0_ratio=ratio_q0)
                if li_target is not None:
                    li_cur, _ = li_achieved(snap2, li_kind=li_kind,
                                            psi_pad=psi_pad,
                                            perimeter=li_perimeter)
                    li_cur = float(li_cur)
                    li_pairs.append((float(li_row), li_cur))
                    _lres = li_cur - float(li_target)
                    rec.update(
                        structured_li_achieved_corrected=li_cur,
                        structured_li_residual_corrected=_lres,
                        structured_li_solved=li_cur,
                        structured_li_residual=_lres,
                        structured_li_solved_residual=_lres,
                        structured_residual_sigma_li=(
                            None if li_sigma in (None, 0)
                            else _lres / float(li_sigma)),
                        structured_li_corrector_row=li_row,
                        structured_li_corrector_ratio=ratio_li,
                        structured_li_corrector_row_factor=(
                            None if li_row is None
                            else float(li_row) / float(li_target)),
                        structured_li_gain_exponents=[float(p)
                                                      for p in li_exponents],
                        structured_li_corrector_rows=[float(r)
                                                      for r, _a in li_pairs],
                        structured_li_corrector_achieved=[float(a)
                                                          for _r, a in li_pairs],
                        structured_li_predicted=out["li_predicted"])
                if not (want_li and step < max_li_steps
                        and abs(li_cur - float(li_target)) > li_tol):
                    break

            if n_solves:
                _tgt = ", ".join([t for t, w in (("q0", want_q0),
                                                 ("l_i", want_li)) if w])
                rec["sawtooth_verdict"] = (
                    f"structured predictor + {n_solves} corrector solve"
                    + ("" if n_solves == 1 else "s") + f" ({_tgt})")
                _msg = [f"[imas SWB-split:ohmic structured] {n_solves} "
                        f"corrector solve{'' if n_solves == 1 else 's'}:"]
                if want_q0:
                    _msg.append(f" axis target x{ratio_q0:.4f}; q0 "
                                f"{q0_tok:.4f}->{q0_cur:.4f} vs target "
                                f"{q0_target:.4f} (residual "
                                f"{q0_cur - q0_target:+.4f});")
                if li_target is not None:
                    _pstr = ", ".join(f"{p:.3f}" for p in li_exponents) or "n/a"
                    _msg.append(f" l_i row x{li_row / float(li_target):.4f} "
                                f"(gain p={_pstr}); l_i "
                                f"{li_tok:.4f}->{li_cur:.4f} vs target "
                                f"{float(li_target):.4f} (residual "
                                f"{li_cur - float(li_target):+.4f});")
                _msg.append(f" s_ind {s_ind.min():.3f}-{s_ind.max():.3f}, "
                            f"s_bs {s_bs.min():.3f}-{s_bs.max():.3f}")
                print("".join(_msg), flush=True)

        rec["ohm_scale"] = float(getattr(bl, "ohm_scale", 1.0))
        rec["bs_scale"] = float(getattr(bl, "bs_scale", 1.0))

        # ---- the health record, re-derived from what was DELIVERED ---------
        # The predictor's block was computed from the PREDICTOR's effective
        # scalars.  A corrector solve replaces both, so f_BS_closed and the
        # bs_scale reason describe a closure this run did not deliver -- and
        # `closure_limited` could not acquire a new reason at all, because the
        # soft-Ip branch rebuilt it from the predictor's reason tuple.  One
        # refresh, from the delivered scales, same function and thresholds;
        # everything below then appends to THAT.
        from .utils import closure_health, SOFT_IP_FLAG_PREFIX
        _prev = getattr(bl, "ip_closure", None) or {}
        _z_for_health = ip_z_final if (soft and ip_z_final is not None
                                       and np.isfinite(ip_z_final)) else None
        _health = closure_health(rec["ohm_scale"], rec["bs_scale"],
                                 state["Ip_signed"], state["c_signed"],
                                 state["ip_ind"], state["ip_bs"],
                                 state["ip_fix"],
                                 soft_ip_residual_sigma=_z_for_health)
        _reasons = list(_health["closure_limited_reasons"])
        if soft and _z_for_health is None:
            # No corrector re-solve, so nothing new to say about the
            # posterior: carry the predictor's soft-Ip flag rather than
            # dropping it.
            _reasons += [r for r in (_prev.get("closure_limited_reasons", ())
                                     or ())
                         if str(r).startswith(SOFT_IP_FLAG_PREFIX)]

        def _flag(why):
            if why not in _reasons:
                _reasons.append(why)
            print("[imas SWB-split:ohmic structured] WARNING closure-limited: "
                  + why, flush=True)

        # A readback that could not be taken at all.
        for _why in reasons:
            if "not finite" in _why:
                _flag(_why)

        # Hard-channel l_i acceptance: FLAG, never retry.  The corrected
        # residual is the delivered one, so this is the first place in the
        # record where a missed hard l_i row can be seen at all.
        _lres_final = rec.get("structured_li_residual_corrected")
        if li_target is not None and not soft:
            if _lres_final is None or not np.isfinite(_lres_final):
                _flag("l_i was not readable after the corrector "
                      f"({_lres_final!r})")
            elif abs(float(_lres_final)) > li_tol:
                _flag(f"l_i misses its hard row by {float(_lres_final):+.4f} "
                      f"(> tol {li_tol:g}) after "
                      f"{rec.get('n_extra_solves', 0)} corrector solve(s)")

        # q0 acceptance: the SAME flag `_close_ip_q0_corrector` raises.  The
        # residual was recorded here and read by nothing, so a slice whose
        # axis row the corrector could not land was indistinguishable from one
        # it did -- the asymmetry between the two channels' `closure_limited`
        # contracts that the q0 channel closed.  Flag only: q0_tol is
        # untouched and nothing retries.
        if gated:
            _q0res = rec.get("q0_residual")
            _q0tol = float(state["q0_tol"])
            if _q0res is None or not np.isfinite(float(_q0res)):
                _flag("q0 residual is not finite after the corrector")
            elif abs(float(_q0res)) > _q0tol:
                _flag(f"q0 misses its row by {float(_q0res):+.4f} "
                      f"(> q0_tol {_q0tol:g}) after "
                      f"{rec.get('n_extra_solves', 0)} corrector solve(s)")

        # A REFUSED corrector step is a flag whatever it was correcting.  With
        # a hard l_i target the branch above already says so (the delivered
        # residual is the predictor's and misses); the q0-only and soft cases
        # recorded `structured_corrector_refusal` and then reported a clean
        # closure on the predictor equilibrium.
        if rec.get("structured_corrector_refusal") and (li_target is None
                                                        or soft):
            _flag("the corrector step was refused, keeping the predictor "
                  f"({str(rec['structured_corrector_refusal'])[:120]})")

        # Soft-channel Ip acceptance: the same FLAG, on the DELIVERED
        # posterior.  closure_health already carries it when the corrector
        # re-solved (soft_ip_residual_sigma above); nothing stacks, because
        # the predictor's copy of that flag was dropped with the rest of the
        # predictor's block.
        rec["closure_limited_reasons"] = tuple(_reasons)
        rec["closure_limited"] = bool(_reasons)
        if gated:
            # the q0 bar this channel was just judged against, recorded
            # alongside the health bars it was judged with
            _health["closure_limited_thresholds"] = dict(
                _health["closure_limited_thresholds"],
                q0_tol=float(state["q0_tol"]))
        for _k, _v in _health.items():
            if _k not in ("closure_limited", "closure_limited_reasons"):
                rec[_k] = _v

        # ---- the assembly gates, re-taken on the delivered profile ---------
        # The corrector reassembles bl.j_phi and nothing re-checked it: the
        # round-trip gate exists to catch an assembly, and this IS one.  Same
        # gate, same tolerance, same reference rule (the soft channel's
        # posterior moves with the re-solve, so the refreshed posterior is
        # what it is compared against -- see utils.ip_roundtrip_gate).
        if ip_of is not None:
            rec["Ip_hybrid"] = float(ip_of(bl.j_phi))
            if roundtrip_gate is not None:
                _g2 = roundtrip_gate(
                    rec["Ip_hybrid"],
                    posterior=(rec.get("structured_ip_posterior") if soft
                               else None),
                    sigma_Ip=(state.get("ip_sigma") if soft else None))
                rec["structured_roundtrip_post_corrector_err_pct"] = float(
                    _g2["err_pct"])
                rec["structured_roundtrip_post_corrector_reference"] = \
                    _g2["reference_name"]
        if getattr(bl, "ip_closure", None) is not None:
            bl.ip_closure.update(rec)
        return nl_out

    @staticmethod
    def _close_ip_q0_corrector(state, bl, mygs, solve_jphi, ip_of=None,
                               roundtrip_gate=None):
        """At most ONE Newton step on q0 after the closed-hybrid solve.

        Returns the new ``nl_its`` when a corrector solve was taken, else
        ``None``.  Never loops -- see the call site.  Ip stays exact by
        construction (the step moves along the Ip-closed manifold
        ``s_bs(s_ohm) = (sgn*Ip - c_signed - s_ohm*lin(ohm) - lin(fix))/lin(bs)``),
        on the same sign-paired constant the predictor closed on, so
        the achieved-Ip error is recorded rather than defended.

        **Every exit re-derives the closure-health record from the scales this
        method actually delivered.**  The predictor's block is computed before
        any of this runs; leaving it in place let a corrector that halved
        ``bs_scale`` be recorded as unflagged, with the predictor's
        ``f_BS_closed``.  ``Ip_hybrid`` and the assembly round-trip are
        likewise re-taken on the delivered profile when *ip_of* /
        *roundtrip_gate* are supplied (the corrector IS an assembly, and the
        gate exists to check assemblies).

        **A missed q0 is flagged, never retried.**  ``q0_tol`` is unchanged and
        no branch here loops: a residual outside it on an admitted slice adds a
        ``closure_limited`` reason so the Delta' consumer sees the same kind of
        flag a missed l_i or an over-stretched ``bs_scale`` raises.

        **Non-finite and degenerate inputs get their own named exits.**  A
        non-finite solved q0 used to fall through ``abs(nan) <= tol`` into the
        Newton branch and be reported as "q0 insensitive to s_ohm"; a bootstrap
        with ~no current in the Ip measure used to raise ZeroDivisionError from
        ``-ip_ind/ip_bs``.  Both now keep the predictor, say so by name, and are
        flagged -- the same shape as the other two refusal branches, and the
        same relative floor (``1e-6 * Ip_t``) :func:`~bouquet.utils.close_ip`
        uses for the component it divides by.
        """
        import numpy as np

        from .utils import closure_health

        q0_target = state["q0_target"]
        # Read q off a copy_eq() SNAPSHOT, never the live solver: the geqdsk
        # save path already carries a suspected live-state mutation by the q
        # tracer, and a diagnostic read must not be able to move the
        # equilibrium the baseline is about to be archived from.
        q0_tok = float(np.asarray(mygs.copy_eq().get_q(psi=state["psi_q"].copy())[1],
                                  dtype=float)[0])
        res = q0_tok - q0_target
        rec = dict(q0_solved_predictor=q0_tok,
                   q0_predictor_residual=res,
                   q0_tol=state["q0_tol"])
        nl_out = None
        if not (np.isfinite(q0_tok) and np.isfinite(res)):
            # BEFORE the tolerance test: abs(nan) <= tol is False, so a
            # non-finite q0 otherwise reaches the Newton branch and is
            # mis-reported as an insensitivity.  An unreadable q0 is not a
            # small residual -- it is no residual at all.
            rec.update(n_extra_solves=0, q0_solved=q0_tok, q0_residual=res,
                       sawtooth_verdict="predictor kept (q0 unreadable: "
                                        "non-finite solved q0)")
            print(f"[imas SWB-split:ohmic q0] solved q0={q0_tok} is NOT FINITE "
                  "-- no residual and no Newton direction; keeping the "
                  "predictor and flagging the slice closure-limited",
                  flush=True)
        elif abs(res) <= state["q0_tol"]:
            rec.update(n_extra_solves=0,
                       sawtooth_verdict="predictor (0 extra solves)",
                       q0_solved=q0_tok, q0_residual=res)
            print(f"[imas SWB-split:ohmic q0] solved q0={q0_tok:.4f} vs "
                  f"q0_target={q0_target:.4f} (residual {res:+.4f}, tol "
                  f"{state['q0_tol']:g}) -- predictor accepted, no extra solve",
                  flush=True)
        elif abs(state["ip_bs"]) < 1e-6 * state["Ip_t"]:
            # The Newton step moves along the Ip-closed manifold
            # s_bs(s_ohm) = (sgn*Ip - c - s_ohm*lin(ohm) - lin(fix))/lin(bs),
            # which does not exist when the bootstrap carries ~no current in
            # the Ip measure.  close_ip's own relative floor, by name, instead
            # of a ZeroDivisionError two lines down.  (close_ip_q0's
            # determinant can clear its floor with ip_bs == 0 whenever j_bs0
            # has core content, so the predictor really can hand this over.)
            rec.update(n_extra_solves=0, q0_solved=q0_tok, q0_residual=res,
                       sawtooth_verdict="predictor kept (j_BS integrates to "
                                        "~0; no Ip-closed manifold to step "
                                        "along)")
            print("[imas SWB-split:ohmic q0] j_BS integrates to "
                  f"{state['ip_bs']:.3e} A (< 1e-6 x Ip_target) -- the "
                  "Ip-closed manifold is degenerate in s_bs; keeping the "
                  f"predictor and recording the residual {res:+.4f}",
                  flush=True)
        else:
            # dq0/ds_ohm along the Ip-closed manifold, from q0 ~ 1/j0:
            #   ds_bs/ds_ohm = -lin(ohm)/lin(bs)
            #   dj0/ds_ohm   = j_ind0 + j_bs0 * ds_bs/ds_ohm
            #   dq0/ds_ohm   = -q0 * (dj0/ds_ohm) / j0
            dsbs = -state["ip_ind"] / state["ip_bs"]
            dj0 = state["j_ind0"] + state["j_bs0"] * dsbs
            j0 = (state["ohm_scale"] * state["j_ind0"]
                  + state["bs_scale"] * state["j_bs0"] + state["j_fix0"])
            dq0ds = -q0_tok * dj0 / j0 if j0 else 0.0
            if not np.isfinite(dq0ds) or abs(dq0ds) < 1e-9:
                rec.update(n_extra_solves=0, q0_solved=q0_tok, q0_residual=res,
                           sawtooth_verdict="predictor kept (q0 insensitive to "
                                            "s_ohm along the Ip-closed manifold)")
                print("[imas SWB-split:ohmic q0] dq0/ds_ohm ~ 0 -- no usable "
                      "Newton direction; keeping the predictor and recording "
                      f"the residual {res:+.4f}", flush=True)
            else:
                s_new = state["ohm_scale"] + (q0_target - q0_tok) / dq0ds
                sbs_new = (state["sgn"] * state["Ip_t"] - state["c_signed"]
                           - s_new * state["ip_ind"]
                           - state["ip_fix"]) / state["ip_bs"]
                if not (0.2 < s_new < 5.0 and 0.2 < sbs_new < 5.0):
                    rec.update(n_extra_solves=0, q0_solved=q0_tok,
                               q0_residual=res,
                               q0_corrector_ohm_scale=float(s_new),
                               q0_corrector_bs_scale=float(sbs_new),
                               sawtooth_verdict="predictor kept (corrector step "
                                                "leaves the (0.2, 5) scale bounds)")
                    print(f"[imas SWB-split:ohmic q0] Newton step would give "
                          f"s_ohm={s_new:.3f} s_bs={sbs_new:.3f}, outside "
                          "(0.2, 5) -- keeping the predictor and recording the "
                          f"residual {res:+.4f}", flush=True)
                else:
                    bl.ohm_scale = float(s_new)
                    bl.bs_scale = float(sbs_new)
                    bl.j_inductive = s_new * state["j_ind"]
                    bl.j_BS = sbs_new * state["j_BS_swb"]
                    bl.j_phi = bl.j_inductive + bl.j_BS + state["j_fixed"]
                    nl_out = solve_jphi(np.asarray(bl.j_phi, dtype=float))
                    q0_new = float(np.asarray(
                        mygs.copy_eq().get_q(psi=state["psi_q"].copy())[1],
                        dtype=float)[0])
                    rec.update(n_extra_solves=1,
                               q0_corrector_ohm_scale=float(s_new),
                               q0_corrector_bs_scale=float(sbs_new),
                               q0_dq0_ds_ohm=float(dq0ds),
                               q0_solved=q0_new, q0_residual=q0_new - q0_target,
                               sawtooth_verdict="predictor + 1 corrector solve")
                    print(f"[imas SWB-split:ohmic q0] 1 corrector solve: "
                          f"s_ohm {state['ohm_scale']:.4f}->{s_new:.4f}, "
                          f"s_bs {state['bs_scale']:.4f}->{sbs_new:.4f}; "
                          f"q0 {q0_tok:.4f}->{q0_new:.4f} vs target {q0_target:.4f} "
                          f"(residual {q0_new - q0_target:+.4f})", flush=True)
        rec["ohm_scale"] = float(getattr(bl, "ohm_scale", 1.0))
        rec["bs_scale"] = float(getattr(bl, "bs_scale", 1.0))
        # ---- closure health, re-derived from the DELIVERED scales ----------
        # The predictor's block was computed from the predictor's scales; a
        # corrector step replaces both, so f_BS_closed, closure_limited and
        # its reasons all have to be re-taken or they describe an equilibrium
        # this run did not deliver.  Same function, same thresholds.
        _health = closure_health(rec["ohm_scale"], rec["bs_scale"],
                                 state["sgn"] * state["Ip_t"],
                                 state["c_signed"], state["ip_ind"],
                                 state["ip_bs"], state["ip_fix"])
        _reasons = list(_health["closure_limited_reasons"])
        _q0_res = rec.get("q0_residual")
        _q0_tol = float(state["q0_tol"])
        if _q0_res is None or not np.isfinite(float(_q0_res)):
            _reasons.append("q0 residual is not finite after the corrector")
        elif abs(float(_q0_res)) > _q0_tol:
            # Flag only -- never a retry, and q0_tol itself is untouched.  The
            # channel deliberately spends at most one extra solve; what it owes
            # the consumer is that a slice it could not land on q0 is not
            # indistinguishable from one it did.
            _reasons.append(f"q0 missed by {float(_q0_res):+.4f} "
                            f"(> q0_tol {_q0_tol:g}) after the corrector")
        _health["closure_limited_reasons"] = tuple(_reasons)
        _health["closure_limited"] = bool(_reasons)
        _health["closure_limited_thresholds"] = dict(
            _health["closure_limited_thresholds"], q0_tol=_q0_tol)
        rec.update(_health)
        if _health["closure_limited"]:
            print("[imas SWB-split:ohmic q0] WARNING closure-limited after the "
                  "corrector: " + "; ".join(_reasons)
                  + " -- treat this slice's current split (and any Delta' "
                    "built on it) as unvalidated", flush=True)
        # ---- the assembly gates, re-taken on the delivered profile ---------
        if ip_of is not None:
            rec["Ip_hybrid"] = float(ip_of(bl.j_phi))
            if roundtrip_gate is not None:
                rec["fsa_roundtrip_post_corrector_err_pct"] = float(
                    roundtrip_gate(rec["Ip_hybrid"]))
        if getattr(bl, "ip_closure", None) is not None:
            bl.ip_closure.update(rec)
        return nl_out

    def _forward_solve_imas_baseline(self):
        """Forward GS solve of the IMAS baseline (j_phi + pressure) on mygs.

        Initialises psi from the IDS LCFS shape, then solves with the IMAS total
        toroidal current (jphi-linterp) and the thermal+fast pressure. Sets
        ``l_i_target`` to the TokaMaker-solved li_3 (``li_normalization='iter'``
        -- the single scale the whole chain uses after issue #20) and stores
        the IDS/TokaMaker li_1/li_3 comparison in ``baseline.li_metrics``.

        Deterministic at ``nthreads=1``: a fresh process re-running this from the
        same baseline reproduces the equilibrium bit-for-bit, which is what makes
        the parallel-draws path land every worker on an identical baseline without
        shipping any flux state. (A warm-start from a snapshotted psi was tried and
        rejected: ``set_psi`` leaves the LCFS limiting points un-retraced, which
        the ``std`` li_1 normalization divides by -- it drifted li_1 by ~2.4e-3
        while the cold re-solve matched to 0.)
        """
        import numpy as np

        from .utils import pchip_derivative

        bl = self.baseline
        mygs = self.mygs
        psi_N = np.asarray(bl.psi_N, dtype=float)
        psi_pad = 1e-3
        EC = 1.602176634e-19

        # init psi from the LCFS shape parameters
        R0, Z0, a, kappa, delta = _shape_from_boundary(self._boundary_RZ)
        mygs.init_psi(R0, Z0, a, kappa, delta)
        self._seed_coil_init(mygs)

        # kinetic profiles + total pressure on the equilibrium grid (IMAS shares
        # psi_N between the kinetic and current grids).
        def k2e(arr):
            # PCHIP regrid (shared helper) -- must match the draw path
            from .utils import pchip_interp
            return pchip_interp(bl.psi_N_kinetic, arr, psi_N)

        ne, te, ni, ti = k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti)
        Zeff = np.clip(k2e(bl.Zeff), 1.0, None)
        p_total = EC * (ne * te + ni * ti)
        if bl.p_fast is not None:
            p_total = p_total + k2e(bl.p_fast)
        # Impurity (carbon) thermal pressure + diff anchor so the baseline forward
        # solve uses the full dd equilibrium.pressure (mirrors generate_bouquet /
        # perturb_kinetic_equilibrium; single-ion e*(ne*Te+ni*Ti) omits carbon).
        if getattr(bl, "Z_imp", None):
            from .physics import impurity_pressure
            _zf = getattr(bl, "z_fast", None)
            _ne_th = ne if _zf is None else np.maximum(ne - k2e(_zf), 0.0)
            p_total = p_total + impurity_pressure(_ne_th, ni, ti, bl.Z_imp)
        if getattr(bl, "p_diff", None) is not None:
            p_total = p_total + k2e(bl.p_diff)

        def solve_jphi(j_phi):
            ffp = {"type": "jphi-linterp", "y": np.asarray(j_phi, dtype=float), "x": psi_N}
            nl_its = -1
            for _pass in range(2):   # 2nd pass refines the jphi-linterp flux scaling
                psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
                pp_y = pchip_derivative(psi_N, p_total) / psi_range
                pp_y[-1] = 0.0
                mygs.set_targets(Ip=bl.Ip_target, pax=float(p_total[0]))
                mygs.set_profiles(
                    pp_prof={"type": "linterp", "y": pp_y, "x": psi_N}, ffp_prof=ffp,
                )
                try:
                    _, nl_its = mygs.solve(return_its=True)
                except ValueError as exc:
                    raise RuntimeError(
                        f"IMAS forward solve failed to converge (pass {_pass + 1}/2) "
                        f"for {self.config.source.ids_path}: {exc}. The baseline "
                        "l_i target cannot be established from this equilibrium."
                    ) from exc
            return int(nl_its)

        # First solve on the source total current to land a converged
        # equilibrium -- needed as the geometry for the SWB call below.
        nl_its = solve_jphi(np.asarray(bl.j_phi, dtype=float))

        # ---- SWB-consistent bootstrap on FUSE's own ohmic current ------------
        # The draw path (perturb_kinetic_equilibrium, recalculate_j_BS branch)
        # reconstructs   new_jphi = j_inductive + scale_jBS * SWB_spike(kinetics)
        # and ASSUMES bl.j_BS == the SWB spike at sigma=0. The IMAS reader takes
        # j_BS from the source's OWN bootstrap model (FUSE/Sauter), which differs
        # from OFT's SWB -- so a sigma=0 draw lands at j_inductive + SWB_spike !=
        # bl.j_phi and every draw inherits a fixed (SWB - source_jBS) offset.
        #
        # Fix: keep the inductive component as the reader's j_inductive.
        # NOTE that is a RESIDUAL, j_tor - j_BS - j_NBI - j_RF (imas.py), NOT
        # to_toroidal(j_ohmic): on postdictive FUSE files the two agree to
        # ~0.4% of Ip at flattop, but any unmodelled non-inductive term the
        # dd carries (verified: NOT the sawteeth source, which nets exactly
        # zero current and is already folded into FUSE's diffused j_ohmic;
        # the observed gap is a near-axis j_non_inductive artifact) lands in
        # this component and is what ohm_scale rescales.  Recompute the
        # bootstrap via
        # SWB, and rebuild the total as ohmic + SWB + fixed. We do NOT make the
        # inductive a residual against SWB (an earlier version did, which forced
        # j_ind to absorb the SWB-vs-FUSE bootstrap shape difference), and we do
        # NOT re-fit the total to a proxy l_i (which lowered li_1 and floored the
        # draws). The per-draw GPR then perturbs FUSE's real ohmic current, and a
        # sigma=0 draw reproduces ohmic + SWB exactly. The total l_i is whatever
        # this self-consistent (FUSE ohmic + OFT bootstrap) combination gives.
        #
        # isolate_edge_jBS: FUSE work uses the FULL bootstrap profile (False), so
        # the baseline and the draws (which read results["isolated_j_BS"] under
        # the same flag) build j_phi from the same SWB current.
        # Two reconciliation modes (GenerationConfig.jBS_baseline_mode), both
        # keeping FUSE's actual ohmic current as the perturbable inductive:
        #   "diff"    -> keep the FUSE total; store a fixed correction
        #                jBS_diff = FUSE_jBS - SWB that is added to the baseline
        #                AND every draw (anchors to FUSE; SWB delta tracks
        #                kinetics; risks edge misalignment if the pedestal moves).
        #   "rescale" -> rescale SWB by one factor so the proxy l_i matches the
        #                FUSE source; fully self-consistent (no fixed profile).
        # The SWB bootstrap is floored at 0 first (drops the inner negative lobe).
        # closure_channel="sawtooth_bootstrap" / "structured" carry state from
        # their (solve-free) predictor to the at-most-one-solve corrector that
        # runs AFTER the common tail's forward solve; None everywhere else.
        _q0_state = None
        _structured_state = None
        if self.config.generation.recalculate_j_BS:
            from .TokaMaker_interface import (_swb_jbs_to_toroidal,
                                              smooth_jbs_transition)
            from .sampling import calc_cylindrical_li_proxy
            from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
            from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
            from scipy.optimize import brentq

            gc = self.config.generation
            iso = bool(gc.isolate_edge_jBS); mode = str(gc.jBS_baseline_mode)
            j_ind = np.asarray(bl.j_inductive, dtype=float)   # FUSE ohmic (kept)
            j_BS_src = np.asarray(bl.j_BS, dtype=float)        # source bootstrap (FUSE)
            FUSE_tot = np.asarray(bl.j_phi, dtype=float)       # source total (j_tor)
            j_fixed = FUSE_tot - j_ind - j_BS_src              # = j_NBI + j_RF
            # 'ohmic' mode: freeze the ANCHOR geometry now. solve_with_bootstrap
            # iterates its own GS solves (generic inductive seed + its bootstrap)
            # and leaves mygs on a different equilibrium; integrating FUSE's
            # profile on that landed geometry read +31% of Ip on an ohmic-ramp
            # slice (vs +0.8% on the anchor) and collapsed the closure. Every Ip
            # integral in the ohmic branch is taken on this snapshot.
            _anchor = None
            if str(gc.jBS_baseline_mode) == "ohmic":
                # Validate the channel BEFORE solve_with_bootstrap: the
                # run-time dispatch would otherwise burn the full SWB
                # iteration sequence and only then refuse a typo.
                from .utils import warn_deprecated_channel
                warn_deprecated_channel(getattr(gc, "closure_channel",
                                                "bootstrap"))
                if str(getattr(gc, "closure_channel", "bootstrap")) \
                        not in ("bootstrap", "ohmic", "sawtooth_bootstrap",
                                "structured"):
                    raise ValueError(
                        f"unknown closure_channel "
                        f"{gc.closure_channel!r} "
                        "(expected 'ohmic', 'bootstrap', "
                        "'sawtooth_bootstrap' or 'structured')")
                from .utils import fsa_current_geometry as _fcg
                from .physics import capture_equilibrium_fsa as _cef
                _anchor = {"eq": mygs.copy_eq()}
                _anchor["geom"] = _fcg(_anchor["eq"], np.asarray(psi_N, dtype=float), psi_pad=psi_pad)
                _anchor["inv_r2_src"] = "get_q ravgs dict"
                if _anchor["geom"]["inv_R2"] is None:
                    # BQ_FSA_NPSI overrides the contour-quadrature resolution.
                    # A bare int() turned a typo into a ValueError from deep
                    # inside the quadrature, and accepted "0"/negatives --
                    # which bypass the max(257, ...) floor and hand npsi=0 to
                    # capture_equilibrium_fsa.  Refuse both, by name.
                    _npsi_env = __import__("os").environ.get("BQ_FSA_NPSI")
                    _npsi = max(257, psi_N.size)
                    if _npsi_env is not None and _npsi_env.strip():
                        try:
                            _npsi = int(_npsi_env)
                        except ValueError:
                            raise ValueError(
                                f"BQ_FSA_NPSI={_npsi_env!r} is not an "
                                "integer; unset it or set a positive "
                                "flux-grid size") from None
                        if _npsi < 2:
                            raise ValueError(
                                f"BQ_FSA_NPSI={_npsi} is not a usable "
                                "flux-grid size (need >= 2; the default is "
                                f"{max(257, psi_N.size)})")
                    _cap = _cef(mygs, npsi=_npsi,
                                psi_pad=psi_pad, exact_inv_R2=True)
                    if _cap.get("avg_inv_R2") is None:
                        raise RuntimeError("ohmic mode: <1/R^2> unavailable on the anchor geometry")
                    _anchor["geom"]["inv_R2"] = np.interp(
                        np.asarray(_anchor["geom"]["psi_q"], float),
                        np.asarray(_cap["psi_N"], float), np.asarray(_cap["avg_inv_R2"], float))
                    _anchor["inv_r2_src"] = "capture_equilibrium_fsa contour quadrature (anchor, pre-SWB)"
                _anchor["Ip_anchor"] = abs(float(mygs.get_stats(lcfs_pad=psi_pad)["Ip"]))
            swb_seed = create_power_flux_fun(psi_N.size, 1.5, 1.5)["y"]
            swb = solve_with_bootstrap(
                mygs, ne, te, ni, ti, Zeff, bl.Ip_target, swb_seed,
                scale_jBS=1.0, isolate_edge_jBS=iso,
                diagnostic_plots=False, verbose=False,
            )
            # Same axis-transition smoothing every per-draw spike receives, so
            # the sigma=0 draw reproduces this baseline split exactly.
            j_BS_swb = smooth_jbs_transition(
                _swb_jbs_to_toroidal(mygs, swb["isolated_j_BS"], psi_pad))
            if gc.floor_j_BS:
                j_BS_swb = np.clip(j_BS_swb, 0.0, None)
            ratio = j_BS_swb.max() / max(j_BS_src.max(), 1.0)

            if mode == "diff":
                bl.jBS_diff = j_BS_src - j_BS_swb        # added to baseline + draws
                bl.j_BS = j_BS_swb
                bl.j_phi = FUSE_tot                      # total anchored to FUSE
                bl.bs_scale = 1.0
                print(f"[imas SWB-split:diff] FUSE total preserved; "
                      f"diff min/max={bl.jBS_diff.min():.2e}/{bl.jBS_diff.max():.2e}; "
                      f"SWB/FUSE jBS peak={ratio:.3f}")
            elif mode == "rescale":
                tgt = calc_cylindrical_li_proxy(mygs, FUSE_tot, psi_pad)
                _f = lambda s: calc_cylindrical_li_proxy(
                    mygs, j_ind + s * j_BS_swb + j_fixed, psi_pad) - tgt
                try:
                    scale = float(brentq(_f, 0.2, 4.0, xtol=1e-4))
                except Exception:
                    scale = 1.0
                bl.jBS_diff = None
                bl.bs_scale = scale
                bl.j_BS = scale * j_BS_swb
                bl.j_phi = j_ind + bl.j_BS + j_fixed
                print(f"[imas SWB-split:rescale] scale={scale:.3f}; FUSE ohmic kept; "
                      f"SWB/FUSE jBS peak={ratio:.3f}")
            elif mode == "ohmic":
                # Hybrid: FUSE ohmic + SWB bootstrap on the (IDA) kinetics +
                # FUSE fixed (NBI/RF), with Ip closed by rescaling j_ohmic ONLY.
                # Rationale: 'diff' pins the total to FUSE (erasing the pedestal
                # current change that better kinetics imply) and 'rescale' scales
                # the bootstrap to close l_i. Here the bootstrap and NBCD are
                # taken as-is and the inductive absorbs the Ip mismatch, because
                # it is the component with the weakest independent constraint.
                # NOTE solve_jphi() passes the profile as a "jphi-linterp" SHAPE
                # and TokaMaker rescales it uniformly to Ip_target, which would
                # scale j_BS and j_fixed too -- so the closure MUST happen here,
                # before the solve. We integrate with the same dA the l_i proxy
                # uses, on the converged first-pass geometry, and record the
                # proxy's own error on the FUSE total so a biased proxy cannot
                # masquerade as a physical ohmic rescale.
                # Ip integral: bouquet's LCFS-truncated FSA current integral,
                #   I_p = int dpsi (V'/2pi) <j_phi/R>
                # (utils.Ip_fsa_integral, convention "jphi-linterp" -- the variable
                # bouquet's arrays are handed to set_profiles as), evaluated on a
                # copy_eq() SNAPSHOT of the converged first-pass geometry so no
                # later solve can move it. NOT TokaMaker.compute_flux_integral:
                # that integrates the whole reg==1 limiter region with the flux
                # function held at its LCFS value outside the plasma, charging the
                # scrape-off area at f(psi_N=1) (+11.9% of Ip on the D3D-like
                # anchor; see the utils.py module note). And NOT the cylindrical
                # l_i proxy (1/<R> for <1/R>; +7.75% on the FUSE validation
                # case). Both are still
                # evaluated and RECORDED below so the biases stay visible.
                from .utils import (Ip_fsa_integral, Ip_fsa_weights,
                                    eq_jphi_profile, close_ip,
                                    closure_sign_convention)
                from .sampling import get_li_proxy_geometry
                from scipy import integrate as _integ
                _psi_ip = np.asarray(psi_N, dtype=float)
                _eq_snap = _anchor["eq"]          # frozen BEFORE solve_with_bootstrap
                _geom = _anchor["geom"]
                _inv_r2_src = _anchor["inv_r2_src"]
                _probe = eq_jphi_profile(_geom, "jphi-linterp", eq=_eq_snap)
                # P' sign: identically +1 BY CONSTRUCTION.  The affine c term
                # is built from the ANCHOR's own get_profiles P' on the
                # anchor's own geometry -- the same solved equilibrium -- so
                # the GS identity fixes its orientation; the external (FUSE)
                # profile carries no P' and its current-direction convention
                # enters only through sgn/abs handling of the linear parts
                # below.  Two failed detectors are retired here: the old
                # sign(dot(eq j_phi, cp j_tor)) heuristic tested a
                # current-direction convention (it flipped on ohmic-ramp
                # slices, costing +1.31% of Ip), and the "self-consistency
                # probe" that replaced it was a tautology -- the probe is
                # built with pprime_sign=+1, so +1 reproduced the anchor's Ip
                # to machine precision and could never lose (adversarial
                # review, 2026-09-01).  The roundtrip gate below validates
                # the measure either way.
                _pps = 1.0
                _ip = lambda j: float(Ip_fsa_integral(
                    _eq_snap, _psi_ip, np.asarray(j, dtype=float),
                    convention="jphi-linterp", pprime_sign=_pps, geom=_geom))
                # The jphi-linterp measure is AFFINE: I_p[J] = int w J + c, with c
                # the P'-term of the GS source (~3% of Ip). Summing per-component
                # _ip() calls counts c once PER COMPONENT -- the first version of
                # this closure did exactly that and landed the hybrid +4.5% over
                # Ip (which TokaMaker would then have renormalised out of the whole
                # shape, j_BS included). Close on the LINEAR part and carry c once.
                _w_lin, _c_affine = Ip_fsa_weights(_geom, convention="jphi-linterp",
                                                   pprime_sign=_pps)
                _lin = lambda j: float(_integ.trapezoid(
                    _w_lin * np.asarray(j, dtype=float), np.asarray(_geom["psi_N"], float)))
                # the measure's own self-consistency: the equilibrium's OWN
                # profile must integrate to its true Ip (validated +0.0055%)
                _ip_roundtrip = _ip(_probe)
                # NaN is FALSE in every threshold comparison below, so a
                # non-finite measure would sail through both gates and die
                # much later in the scale-range guard with a message blaming
                # the data.  Refuse it here, by name.
                if not (np.all(np.isfinite(_w_lin)) and np.isfinite(_c_affine)
                        and np.isfinite(_ip_roundtrip)):
                    raise RuntimeError(
                        "ohmic mode: the FSA measure is non-finite (NaN/inf "
                        "in the weights, the affine P' term, or the "
                        "roundtrip) -- the anchor geometry or its profiles "
                        "carry bad values; refusing to close Ip on it")
                # BQ_CLOSURE_DUMP=<path.npz>: dump the measure's ingredients so the
                # roundtrip error can be localised in psi_N (diagnostic only).
                _dumpf = __import__("os").environ.get("BQ_CLOSURE_DUMP")
                if _dumpf:
                    _gp = np.asarray(_geom["psi_N"], float)
                    _cum = _integ.cumulative_trapezoid(_w_lin * np.asarray(_probe, float), _gp, initial=0.0)
                    np.savez(_dumpf,
                             psi_N_geom=_gp, w_lin=np.asarray(_w_lin, float),
                             c_affine=float(_c_affine), probe=np.asarray(_probe, float),
                             cum_lin=_cum, ip_roundtrip=float(_ip_roundtrip),
                             Ip_target=float(abs(bl.Ip_target)), pprime_sign=float(_pps),
                             inv_R2=np.asarray(_geom.get("inv_R2", []), float),
                             psi_q=np.asarray(_geom.get("psi_q", []), float),
                             fuse_tot=np.asarray(FUSE_tot, float),
                             psi_N_kin=np.asarray(psi_N, float))
                    print(f"[closure-dump] wrote {_dumpf}", flush=True)
                # recorded-only alternatives
                _ip_fsaconv = lambda j: float(Ip_fsa_integral(
                    _eq_snap, _psi_ip, np.asarray(j, dtype=float),
                    convention="fsa", geom=_geom))   # documented ~+0.9% bias
                _ip_oft = lambda j: float(_eq_snap.compute_flux_integral(
                    _psi_ip, np.asarray(j, dtype=float)))
                _geo_cyl = get_li_proxy_geometry(_eq_snap, psi_N.size, psi_pad)
                _dA_cyl = np.asarray(_geo_cyl["dA"], dtype=float)
                _ip_cyl = lambda j: float(_integ.trapezoid(np.asarray(j, float) * _dA_cyl))
                Ip_t = abs(float(bl.Ip_target))
                ip_fuse_tot = _ip(FUSE_tot)
                fuse_tot_err_pct = 100.0 * (abs(ip_fuse_tot) - Ip_t) / Ip_t
                # Diagnostics BEFORE any gate, so a refusal still leaves the
                # evidence: is the MEASURE wrong (roundtrip), or does FUSE's
                # total genuinely not carry Ip (which TokaMaker otherwise hides
                # by renormalising the shape)?
                _ip_solved = _anchor["Ip_anchor"]
                _e = lambda v: 100.0 * (abs(v) - Ip_t) / Ip_t
                print("[imas SWB-split:ohmic DIAG] Ip_target=%.1f A | FSA roundtrip(eq own profile) %+.3f%% | "
                      "FSA(FUSE_tot) %+.3f%% | fsa-convention(FUSE_tot) %+.3f%% | OFT compute_flux_integral(FUSE_tot) %+.3f%% | "
                      "cyl proxy(FUSE_tot) %+.3f%% | anchor eq Ip %+.3f%% | <1/R^2> from %s"
                      % (Ip_t, _e(_ip_roundtrip), fuse_tot_err_pct, _e(_ip_fsaconv(FUSE_tot)),
                         _e(_ip_oft(FUSE_tot)), _e(_ip_cyl(FUSE_tot)), _e(_ip_solved), _inv_r2_src), flush=True)
                # GATE: validity of the MEASURE. The equilibrium's own GS current
                # profile must round-trip to its Ip (bouquet validated +0.0055%,
                # measured -0.002% on the FUSE validation case). If it does
                # not, no closure built
                # on this geometry is meaningful -- stop. (Re-targeted 2026-08-21,
                # user-approved: the previous gate tested whether FUSE's total
                # carries Ip, which is a property of the FUSE DATA, not of the
                # measure -- and absorbing that deficit into s, visibly, is the
                # whole point of this mode. In every other mode TokaMaker
                # absorbs it silently by renormalising the shape.)
                # Reference: the anchor's ACHIEVED Ip, not Ip_target -- "its
                # Ip" means the equilibrium the profile came from.  A first
                # pass that converged 0.6% off target would otherwise fail a
                # perfect measure (and a 0.4% solver offset would eat most of
                # the gate's budget while masking a real measure error).
                _rt_err = 100.0 * (abs(_ip_roundtrip) - _ip_solved) / _ip_solved
                if abs(_rt_err) > 0.5:
                    raise RuntimeError(
                        f"ohmic mode: the FSA current measure does not round-trip "
                        f"the equilibrium's own profile to its achieved Ip "
                        f"({_rt_err:+.3f}% vs the anchor's {_ip_solved:.1f} A); "
                        "the geometry/measure is wrong, refusing to close Ip on it")
                # Property of the DATA: reported, recorded, absorbed by s.
                print(f"[imas SWB-split:ohmic] FUSE core_profiles total carries "
                      f"{fuse_tot_err_pct:+.2f}% of Ip_target (exact FSA measure) "
                      f"-> absorbed into the closure channel's scale",
                      flush=True)
                ip_ind, ip_bs, ip_fix = _lin(j_ind), _lin(j_BS_swb), _lin(j_fixed)
                # The data's current-direction convention: sign of the LINEAR
                # total.  (Taking it from _ip(FUSE_tot) folded the anchor's c
                # into the vote, which could flip it on a low-Ip ramp slice
                # where |lin| < |c|.)  The affine term MUST be signed with it:
                # _c_affine is built on the anchor, which is always solved to
                # abs(Ip), so it carries the positive orientation whatever the
                # data does, while ip_* carry the data's sign.  Pairing them
                # (closure_sign_convention) is the contract close_ip's own
                # test_negative_current_convention_closes_too states; the
                # unpaired version was off by 2c (~6% of Ip) on sgn=-1 data,
                # and the self-check below could not see it because it reused
                # the same unpaired c.
                sgn, _Ip_signed, _c_signed = closure_sign_convention(
                    ip_ind, ip_bs, ip_fix, _c_affine, Ip_t)
                # Which channel absorbs the Ip closure -- "bootstrap"
                # (default): keep j_ohmic exactly as FUSE diffused it,
                #   lin(ohm) + s_BS * lin(bs) + lin(fix) + c = Ip_target;
                # "ohmic": rescale j_inductive instead,
                #   s * lin(ohm) + lin(bs) + lin(fix) + c = Ip_target.
                # The two are the extreme attributions of the same deficit;
                # run both to bracket the closure uncertainty.  The algebra
                # and its refusals live in utils.close_ip so the tests
                # exercise the SHIPPED formulas, not a re-derivation.
                # "sawtooth_bootstrap" adds a SECOND target (q0) and so
                # determines BOTH scales -- see the block below.
                # "structured" replaces the two scalars with two smooth radial
                # multiplier PROFILES and returns the minimal-norm one that
                # closes the same constraints -- the MSE arbiter showed the
                # true correction is radially structured and shot-dependent,
                # which no scalar can be.  Its "ohm_scale"/"bs_scale" are the
                # Ip-weighted MEANS of those profiles (recorded as such): they
                # satisfy the scalar closure equation exactly, so every
                # downstream consumer -- closure_health included -- keeps
                # working, while j_inductive/j_BS carry the real structure.
                _chan = str(getattr(gc, "closure_channel", "bootstrap"))
                _q0_extra = {}
                _s_ind = _s_bs = None
                if _chan == "sawtooth_bootstrap":
                    # The q0 channel closes the SAME affine identity, so it
                    # gets the same sign-paired (target, constant) pair.
                    ohm_scale, bs_scale, _q0_extra, _q0_state = \
                        self._close_ip_q0_predictor(
                            gc, bl, _eq_snap, _geom, _probe, psi_N,
                            j_ind, j_BS_swb, j_fixed, FUSE_tot,
                            sgn, Ip_t, _c_signed, ip_ind, ip_bs, ip_fix,
                            close_ip)
                elif _chan == "structured":
                    _s_ind, _s_bs, ohm_scale, bs_scale, _q0_extra, \
                        _structured_state = self._close_ip_structured_predictor(
                            gc, bl, _eq_snap, _geom, _probe, psi_N,
                            j_ind, j_BS_swb, j_fixed, FUSE_tot,
                            sgn, Ip_t, _w_lin, _c_signed,
                            ip_ind, ip_bs, ip_fix,
                            psi_pad=psi_pad, pprime_sign=_pps)
                else:
                    ohm_scale, bs_scale = close_ip(
                        _chan, _Ip_signed, _c_signed, ip_ind, ip_bs, ip_fix)
                bl.jBS_diff = None
                bl.bs_scale = float(bs_scale)
                bl.ohm_scale = float(ohm_scale)
                bl.j_BS = (bs_scale if _s_bs is None else _s_bs) * j_BS_swb
                bl.j_inductive = (ohm_scale if _s_ind is None
                                  else _s_ind) * j_ind
                bl.j_phi = bl.j_inductive + bl.j_BS + j_fixed
                # GATE: the ALGEBRA of the assembly above.  It asks one
                # question -- does the hybrid this method just built integrate
                # to the current the closure said it would? -- so its reference
                # is the CLOSURE'S OWN Ip, and the tolerance (0.05 %) is a
                # machine-precision budget for the assembly, not a physics
                # acceptance.
                #
                # On every HARD channel the closure's own Ip IS Ip_target (Ip is
                # imposed exactly), so the reference is Ip_target and nothing
                # about this gate changes.  On the SOFT structured channel it is
                # not: Ip there is a Gaussian MEASUREMENT with sigma_Ip, so the
                # posterior mode lands a little off it by design (0.02-0.1 % at
                # sigma_Ip = 0.5 % of Ip -- well inside the 0.5 % the user set
                # as reasonable for this channel, 2026-09-16).  Comparing the
                # posterior against Ip_meas with the assembly's tolerance tested
                # the DATA statement rather than the algebra and refused ~40 of
                # 335 cloud slices for being exactly what the channel is for.
                # The tolerance is UNCHANGED; only the reference moves, and the
                # distance from the measurement is recorded (and flagged beyond
                # 1 sigma_Ip) instead of being refused.
                #
                # It is fed the SAME affine measure the closure solved: linear
                # part + the SIGNED constant.  _ip() adds the unsigned
                # _c_affine, so on sgn=-1 data it would disagree with the
                # closure by 2c and fail a correct result (identical to
                # _ip(bl.j_phi) whenever sgn == +1).  Both correctors are
                # handed this same measure as ip_of, never _ip.
                from .utils import ip_roundtrip_gate
                _ip_signed = lambda _j: float(_lin(_j) + _c_signed)
                _ip_closed = float(_ip_signed(bl.j_phi))
                _soft_ip = bool(_chan == "structured"
                                and _q0_extra.get("structured_soft"))
                _gate = ip_roundtrip_gate(
                    _ip_closed, Ip_t,
                    posterior=(_q0_extra.get("structured_ip_posterior")
                               if _soft_ip else None),
                    sigma_Ip=(_q0_extra.get("structured_ip_sigma")
                              if _soft_ip else None))
                _closed_err = _gate["err_pct"]
                _ip_meas_resid_pct = (_gate["measured_residual_pct"]
                                      if _soft_ip else None)
                _ip_z = _gate["residual_sigma_Ip"] if _soft_ip else None
                if _soft_ip:
                    print("[imas SWB-split:ohmic structured] SOFT Ip: posterior "
                          f"{_gate['reference'] / 1e6:.4f} MA vs measured "
                          f"{Ip_t / 1e6:.4f} MA ({_ip_meas_resid_pct:+.3f}%"
                          + ("" if _ip_z is None else f", z_Ip={_ip_z:+.2f}")
                          + "); round-trip of the assembled hybrid against the "
                          f"posterior {_closed_err:+.3e}%", flush=True)
                _jd = getattr(bl, "jphi_diff", None)
                ip_jd = _lin(k2e(_jd)) if _jd is not None else 0.0
                # Closure health, every channel: how much reconciliation one
                # scale was asked to do.  closure-limited slices are flagged
                # for downstream (Delta') consumers -- not refused, but not
                # to be read as validated either.
                from .utils import closure_health
                _health = closure_health(ohm_scale, bs_scale, _Ip_signed,
                                         _c_signed, ip_ind, ip_bs, ip_fix,
                                         soft_ip_residual_sigma=_ip_z)
                if _health["closure_limited"]:
                    print("[imas SWB-split:ohmic] WARNING closure-limited: "
                          + "; ".join(_health["closure_limited_reasons"])
                          + " -- treat this slice's current split (and any "
                          "Delta' built on it) as unvalidated", flush=True)
                _oft_tot, _oft_bs = _ip_oft(FUSE_tot), _ip_oft(j_BS_swb)
                _oft_fix, _oft_ind = _ip_oft(j_fixed), _ip_oft(j_ind)
                _cyl_tot, _cyl_bs = _ip_cyl(FUSE_tot), _ip_cyl(j_BS_swb)
                _cyl_fix, _cyl_ind = _ip_cyl(j_fixed), _ip_cyl(j_ind)
                _would_be = lambda tot, bs_, fix_, ind_: (
                    float(((np.sign(tot) or 1.0) * Ip_t - bs_ - fix_) / ind_)
                    if abs(ind_) > 1e-6 * Ip_t else float("nan"))
                bl.ip_closure = dict(
                    integrator="bouquet Ip_fsa_integral (LCFS-truncated FSA, jphi-linterp) on copy_eq snapshot",
                    pprime_sign=_pps,
                    inv_R2_source=_inv_r2_src,
                    affine_pprime_term_c=float(_c_affine),
                    affine_pprime_term_pct_of_Ip=100.0 * float(_c_affine) / Ip_t,
                    # The data's current-direction convention, and the affine
                    # constant as it was actually paired with the target.
                    current_direction_sign=float(sgn),
                    affine_pprime_term_c_signed=float(_c_signed),
                    fsa_roundtrip_Ip=_ip_roundtrip,
                    fsa_roundtrip_err_pct=_rt_err,
                    Ip_anchor=float(_ip_solved),
                    Ip_target=Ip_t,
                    Ip_fuse_total=ip_fuse_tot,
                    fuse_total_err_pct=fuse_tot_err_pct,
                    Ip_ohmic_unscaled=ip_ind, Ip_jBS_swb=ip_bs, Ip_fixed=ip_fix,
                    closure_channel=_chan,
                    **_health,
                    ohm_scale=float(ohm_scale), bs_scale=float(getattr(bl, 'bs_scale', 1.0)),
                    # On the same signed affine measure the closure solved
                    # (identical to _ip(bl.j_phi) for the positive convention).
                    Ip_hybrid=float(_lin(bl.j_phi) + _c_signed),
                    jphi_diff_dropped_Ip=ip_jd,
                    jphi_diff_dropped_pct_of_Ip=100.0 * ip_jd / Ip_t,
                    swb_over_fuse_jBS_peak=float(ratio),
                    # recorded-only alternatives (NOT used for closure).
                    # Each integrator runs ONCE per profile (they were
                    # re-evaluated up to 4x for print+dict), and the would-be
                    # scales guard their divisor: a ~0 alternative-integrator
                    # j_ind must record NaN, not raise ZeroDivisionError from
                    # a diagnostic.  The proxy carries its own sign
                    # convention (negative on the FUSE validation case where
                    # OFT's integral is positive), so each would-be uses ITS
                    # OWN total's sign -- otherwise the diagnostic reads as a
                    # nonsensical negative rescale.
                    fsaconv_fuse_total_err_pct=100.0 * (abs(_ip_fsaconv(FUSE_tot)) - Ip_t) / Ip_t,
                    oft_flux_integral_fuse_total=_oft_tot,
                    oft_flux_integral_err_pct=100.0 * (abs(_oft_tot) - Ip_t) / Ip_t,
                    oft_ohm_scale_would_be=_would_be(_oft_tot, _oft_bs,
                                                     _oft_fix, _oft_ind),
                    proxy_Ip_fuse_total=_cyl_tot,
                    proxy_fuse_total_err_pct=100.0 * (abs(_cyl_tot) - Ip_t) / Ip_t,
                    proxy_ohm_scale_would_be=_would_be(_cyl_tot, _cyl_bs,
                                                       _cyl_fix, _cyl_ind))
                if _q0_extra:
                    bl.ip_closure.update(_q0_extra)
                if _soft_ip:
                    # ACHIEVED (delivered-hybrid) Ip bookkeeping for the soft
                    # channel, written AFTER _q0_extra so it is not overwritten
                    # by the predictor's model-space value.  Refreshed by
                    # :meth:`_close_ip_structured_corrector` when a corrector
                    # solve is taken.
                    bl.ip_closure.update(
                        structured_ip_measured_residual_pct=_ip_meas_resid_pct,
                        structured_residual_sigma_Ip=_ip_z,
                        structured_ip_gate_reference=_gate["reference_name"],
                        structured_ip_gate_err_pct=float(_closed_err))
                print(f"[imas SWB-split:ohmic] channel={_chan} ohm_scale={ohm_scale:.4f} bs_scale={float(getattr(bl,'bs_scale',1.0)):.4f}  "
                      f"linear Ip parts: ohm={ip_ind/1e6:.3f} jBS={ip_bs/1e6:.3f} "
                      f"fixed={ip_fix/1e6:.3f} + P'-term c={_c_affine/1e6:+.4f} MA "
                      f"-> hybrid={_ip(bl.j_phi)/1e6:.4f} "
                      f"(target {Ip_t/1e6:.4f}); FSA-integral err on FUSE total "
                      f"{fuse_tot_err_pct:+.2f}% (roundtrip {bl.ip_closure['fsa_roundtrip_err_pct']:+.3f}%) "
                      f"[OFT compute_flux_integral would be {bl.ip_closure['oft_flux_integral_err_pct']:+.2f}%, "
                      f"s={bl.ip_closure['oft_ohm_scale_would_be']:.4f}; cyl proxy "
                      f"{bl.ip_closure['proxy_fuse_total_err_pct']:+.2f}%, s={bl.ip_closure['proxy_ohm_scale_would_be']:.4f}]; "
                      f"jphi_diff anchor NOT applied ({100*ip_jd/Ip_t:+.2f}% of Ip); "
                      f"SWB/FUSE jBS peak={ratio:.3f}")
                # The dropped anchor is recorded above; CLEAR it so it cannot
                # leak past the baseline: generate() passes bl.jphi_diff
                # through unconditionally, so a workflow='custom' ohmic
                # generate() would otherwise fold it into every draw and the
                # archived baseline while the forward solve below omits it.
                bl.jphi_diff = None
            else:
                raise ValueError(f"unknown jBS_baseline_mode {mode!r} "
                                 "(expected 'diff', 'rescale' or 'ohmic')")
            # Solve the resulting total so coils + li_1 reflect this equilibrium.
            # Anchor to equilibrium.j_tor (add the fixed jphi_diff) so the baseline
            # l_i/coils reflect the same total the draws use (== equilibrium.j_tor),
            # not the core_profiles total. The diff-mode component split above stays
            # on core_profiles.j_tor; jphi_diff is the fixed equilibrium offset.
            _jphi_solve = np.asarray(bl.j_phi, dtype=float)
            # jphi_diff re-anchors the total to FUSE's equilibrium.j_tor; in
            # 'ohmic' mode the whole point is NOT to anchor to FUSE, so skip it
            # (its magnitude is recorded in bl.ip_closure for the reader).
            if getattr(bl, "jphi_diff", None) is not None and mode != "ohmic":
                _jphi_solve = _jphi_solve + k2e(bl.jphi_diff)
            nl_its = solve_jphi(_jphi_solve)

            # Corrective iteration (opt-in): the single jphi-linterp solve
            # imposes the request with pre-solve geometry, so the ACHIEVED FSA
            # j_phi lands a few % off the anchor once psi converges (-> l_i /
            # q(psi_N) biased vs the equilibrium IDS). Reuse the recon path's
            # Newton corrector to drive the output onto the anchor target.
            if getattr(self.config.generation, "imas_corrective_jphi", False):
                from .TokaMaker_interface import _corrective_jphi_iteration
                _pr = mygs.psi_bounds[1] - mygs.psi_bounds[0]
                _pp_y = pchip_derivative(psi_N, p_total) / _pr
                _pp_y[-1] = 0.0
                _pp_prof = {"type": "linterp", "y": _pp_y, "x": psi_N}
                _, _n_corr, _corr_hist = _corrective_jphi_iteration(
                    mygs, psi_N, _jphi_solve, _pp_prof,
                    abs(bl.Ip_target), float(p_total[0]), 1e-3,
                    min_iters=2, max_iters=8, rtol=0.02, verbose=True,
                    damping=0.5, protect_state=True)
                print(f"[imas corrective-jphi] converged in {_n_corr} iteration(s)")

            # ---- q0 corrector (closure_channel="sawtooth_bootstrap") --------
            # The predictor above is FIRST ORDER (q0 ~ 1/j_phi(0) at the frozen
            # anchor geometry).  The closed-hybrid solve that just ran is the
            # first time the real q0 is knowable, and it costs nothing extra to
            # read it.  If the predictor already landed inside q0_tol we are
            # done at ZERO extra solves; otherwise ONE analytic Newton step
            # along the Ip-closed manifold and whatever that gives is accepted
            # and recorded.  Deliberately no loop: this channel exists to cost
            # about what "bootstrap" costs, and a residual that is reported is
            # worth more than a residual that is iterated away invisibly.
            if _q0_state is not None:
                # The same NAMED gate the predictor stage used, re-run on
                # what the corrector delivers.  This channel is always HARD in
                # Ip, so there is no posterior and the reference is Ip_target.
                _nl_corr = self._close_ip_q0_corrector(
                    _q0_state, bl, mygs, solve_jphi, ip_of=_ip_signed,
                    roundtrip_gate=lambda _ipc: ip_roundtrip_gate(
                        _ipc, Ip_t)["err_pct"])
                if _nl_corr is not None:
                    nl_its = _nl_corr    # the state l_i/coils are read from
            # Same contract for closure_channel="structured" (present whenever
            # there is something to correct: the sawtooth gate admitted an axis
            # row, or an l_i target was given, or both -- and BOTH corrections
            # then share the one extra solve).  This channel can be SOFT in Ip,
            # so the gate keeps its posterior/sigma_Ip pass-through and only
            # the measurement is bound here.
            if _structured_state is not None:
                _nl_corr = self._close_ip_structured_corrector(
                    _structured_state, bl, mygs, solve_jphi, ip_of=_ip,
                    roundtrip_gate=self._structured_roundtrip_gate(Ip_t))
                if _nl_corr is not None:
                    nl_its = _nl_corr

        # Convergence sanity: the solve completed (it raises otherwise), so
        # verify it landed on the requested current before trusting its l_i.
        Ip_achieved = float(mygs.get_globals()[0])
        ip_err_pct = 100.0 * (abs(Ip_achieved) - abs(bl.Ip_target)) / abs(bl.Ip_target)
        if abs(ip_err_pct) > 1.0:
            import warnings
            warnings.warn(
                f"IMAS forward solve converged but Ip is {ip_err_pct:+.2f}% off "
                f"target ({Ip_achieved/1e6:.3f} vs {bl.Ip_target/1e6:.3f} MA); "
                "the derived l_i target may be unreliable."
            )

        tok_li1 = float(mygs.get_stats(lcfs_pad=psi_pad, li_normalization="std")["l_i"])
        tok_li3 = float(mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")["l_i"])

        metrics = dict(bl.li_metrics or {})
        metrics.update(tokamaker_li_1=tok_li1, tokamaker_li_3=tok_li3,
                       forward_solve_nl_its=nl_its,
                       forward_solve_ip_err_pct=ip_err_pct,
                       jBS_baseline_mode=str(self.config.generation.jBS_baseline_mode),
                       bs_scale=float(getattr(bl, "bs_scale", 1.0)),
                       ohm_scale=float(getattr(bl, "ohm_scale", 1.0)))
        if getattr(bl, "ip_closure", None):
            metrics["ip_closure"] = dict(bl.ip_closure)
            metrics["closure_limited"] = bool(
                bl.ip_closure.get("closure_limited", False))
        # Sawtooth gate inputs travel with EVERY IMAS baseline, not just the
        # runs that used closure_channel="sawtooth_bootstrap": a fan-out
        # needs to see which slices the gate would admit or reject without
        # re-reading a 100s-of-MB dd per slice, and the q0-channel run is
        # exactly the run you do not have yet when you are choosing slices.
        # q0_target lands here too when the channel computed one.
        if getattr(bl, "sawtooth", None):
            _sw = dict(bl.sawtooth)
            _icl = getattr(bl, "ip_closure", None) or {}
            if "q0_target" in _icl:
                _sw["q0_target"] = _icl["q0_target"]
                _sw["q0_anchor"] = _icl.get("q0_anchor")
            metrics["sawtooth"] = _sw
        bl.li_metrics = metrics
        # Target TokaMaker li_3 ('iter').  The IMAS path is not itself affected
        # by the geqdsk estimator mismatch (both sides come from TokaMaker),
        # but the DOWNSTREAM machinery is shared: perturb_kinetic_equilibrium
        # measures every draw's l_i with li_normalization='iter' after issue
        # #20, so l_i_target must be on that scale or the per-draw acceptance
        # band compares two different functionals (~25% apart).
        bl.l_i_target = tok_li3
        bl.l_i_scale = "iter(li3)"
        print(
            f"[imas forward-solve] converged ({nl_its} its, "
            f"Ip {ip_err_pct:+.2f}%) recalc_jBS="
            f"{self.config.generation.recalculate_j_BS} "
            f"TokaMaker li_1={tok_li1:.4f} li_3={tok_li3:.4f} | "
            f"IDS li_1={metrics.get('ids_li_1')} li_3={metrics.get('ids_li_3')}"
        )

    def plot_baseline(self):
        """Diagnostic figure for the resolved baseline -- the gate before
        spending GS-solve compute on the draws.

        Three panels from the in-memory :class:`~bouquet.baseline.Baseline`
        (no HDF5 needed): kinetic profiles (n_e/n_i, T_e/T_i), total pressure
        (thermal + fast), and the separated toroidal currents
        (j_phi = j_inductive + j_BS [+ j_NBI + j_RF]). Returns ``(fig, axes)``.
        Call after :meth:`prepare_baseline` / :meth:`reconstruct`.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.baseline is None:
            raise ValueError("call prepare_baseline() before plot_baseline()")
        bl = self.baseline
        EC = 1.602176634e-19
        pk = np.asarray(bl.psi_N_kinetic, dtype=float)
        pe = np.asarray(bl.psi_N, dtype=float)

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        # kinetic profiles (densities left axis, temperatures right axis)
        a = ax[0]
        a.plot(pk, np.asarray(bl.ne) / 1e19, "-", color="tab:blue", label=r"$n_e$")
        a.plot(pk, np.asarray(bl.ni) / 1e19, "--", color="tab:blue", label=r"$n_i$")
        a.set_ylabel(r"$n$ [$10^{19}$ m$^{-3}$]"); a.set_xlabel(r"$\psi_N$")
        at = a.twinx()
        at.plot(pk, np.asarray(bl.te) / 1e3, "-", color="tab:red", label=r"$T_e$")
        at.plot(pk, np.asarray(bl.ti) / 1e3, "--", color="tab:red", label=r"$T_i$")
        at.set_ylabel(r"$T$ [keV]", color="tab:red")
        a.set_title(f"kinetic profiles ({bl.provenance})")
        a.legend(loc="upper right", fontsize=8); a.grid(alpha=0.3)

        # total pressure (thermal + fast)
        p_th = EC * (np.asarray(bl.ne) * np.asarray(bl.te)
                     + np.asarray(bl.ni) * np.asarray(bl.ti))
        ax[1].plot(pk, p_th / 1e3, "-", color="k", label="thermal")
        if bl.p_fast is not None:
            ax[1].plot(pk, np.asarray(bl.p_fast) / 1e3, ":", color="tab:purple",
                       label="fast")
        ax[1].set_ylabel("p [kPa]"); ax[1].set_xlabel(r"$\psi_N$")
        ax[1].set_title("pressure"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

        # separated toroidal currents
        ax[2].plot(pe, np.asarray(bl.j_phi) / 1e6, "-", color="k", label=r"$j_\phi$ total")
        ax[2].plot(pe, np.asarray(bl.j_inductive) / 1e6, "-", color="tab:orange",
                   label=r"$j_{ind}$")
        ax[2].plot(pe, np.asarray(bl.j_BS) / 1e6, "-", color="tab:green", label=r"$j_{BS}$")
        for nm, arr in (("j_NBI", bl.j_NBI), ("j_RF", bl.j_RF)):
            if arr is not None and np.any(np.asarray(arr)):
                ax[2].plot(pe, np.asarray(arr) / 1e6, "--", lw=1, label=nm)
        ax[2].set_ylabel(r"$j$ [MA/m$^2$]"); ax[2].set_xlabel(r"$\psi_N$")
        ax[2].set_title("separated currents"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

        ttl = (f"Baseline  Ip={bl.Ip_target/1e6:.3f} MA  "
               f"l_i(target)={bl.l_i_target:.3f}")
        fig.suptitle(ttl, fontsize=11); fig.tight_layout()
        return fig, ax

    def verify_sigma0_consistency(self, tol_frac=0.02, swb_iterations=3):
        """Regression guard: the draw pipeline must reproduce the baseline
        j_BS split when the kinetics are UNPERTURBED (sigma=0).

        Replays the per-draw pre-SWB sequence -- state-anchor solve at
        the baseline j_phi/pressure, ``solve_with_bootstrap`` on the baseline
        kinetics, toroidal conversion, axis-transition smoothing -- and
        compares the resulting bootstrap spike to ``baseline.j_BS``.  Any
        systematic deviation found here is inherited by EVERY draw as a
        j_phi target bias: the 2026-07 hollow-core/q0-offset bug was exactly
        such a sigma=0 inconsistency (recon-only axis smoothing), invisible
        to l_i but a wholesale +12% shift of the q0 distribution.

        The anchor here is NOT started from the reconstruction's converged
        state: psi is re-initialised (cold-started) from the LCFS shape
        first, because inheriting that state can leave the under-relaxed
        Picard iteration already on this forward solve's own fixed point,
        with no gradient to descend -- it then parks just above ``nl_tol``
        and exhausts ``maxits`` (see the comment at the call site below).
        That makes this one step a deliberate departure from the draw path,
        but the sequence was never a mirror of it in the first place: a
        per-draw anchor warm-starts from the PREVIOUS draw's landed state,
        which a single standalone check has no equivalent of.  Cold and warm
        starts converge to the same equilibrium here; only the starting
        point of the iteration differs.

        Costs one SWB call (~1 min). Call after ``reconstruct()`` /
        ``prepare_baseline()`` and before ``generate()``; leaves ``mygs``
        re-anchored on the baseline equilibrium.

        Note: this check exercises the shared-smoothing spike treatment. With
        ``GenerationConfig.jbs_delta_mode`` the sigma=0 draw reproduces the
        baseline split exactly by construction (the same SWB call this method
        makes becomes the cached reference), so the check remains a pure
        SWB-context-reproducibility probe there.

        Parameters
        ----------
        tol_frac : float
            Pass threshold on ``max|spike0 - j_BS|`` as a fraction of
            ``max(j_BS)`` (default 2%).
        swb_iterations : int
            Iterations for the SWB call (match GenerationConfig).

        Returns
        -------
        dict with ``spike0`` (the sigma=0 draw-context j_BS), ``max_dev``,
        ``rms_dev`` [A/m^2], ``max_dev_frac`` (of peak j_BS), ``psi_worst``,
        and ``passed``.
        """
        import numpy as np
        from scipy.interpolate import interp1d
        from .TokaMaker_interface import (_swb_jbs_to_toroidal,
                                          smooth_jbs_transition)
        from .utils import pchip_derivative
        from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun

        if self.baseline is None or self.mygs is None:
            raise ValueError("call setup_solver() + prepare_baseline() / "
                             "reconstruct() before verify_sigma0_consistency()")
        if self.config.generation.single_profile_jphi:
            # No inductive/bootstrap split exists, so there is nothing for the
            # sigma=0 draw to reproduce. Report a pass rather than spending a
            # bootstrap solve comparing zeros.
            print("[sigma0-check] SKIPPED: single_profile_jphi=True -- j_phi is "
                  "one profile, so there is no j_BS split to verify")
            return {"spike0": None, "max_dev": 0.0, "rms_dev": 0.0,
                    "max_dev_frac": 0.0, "psi_worst": float("nan"),
                    "passed": True, "skipped": "single_profile_jphi"}
        bl = self.baseline
        mygs = self.mygs
        gc = self.config.generation
        # psi_pad is a ReconstructionSource field; ImasSource has none, so
        # fall back to the pipeline default (matches the forward-solve sites).
        psi_pad = getattr(self.config.source, "psi_pad", 1e-3)
        psi_N = np.asarray(bl.psi_N, dtype=float)
        EC = 1.602176634e-19

        # kinetics + pressure on the equilibrium grid, mirroring
        # generate_bouquet's baseline assembly (incl. impurity/fast/diff terms)
        from .utils import pchip_interp
        pk = np.asarray(bl.psi_N_kinetic, dtype=float)
        k2e = lambda a: pchip_interp(pk, a, psi_N)
        ne_eq, te_eq = k2e(bl.ne), k2e(bl.te)
        ni_eq, ti_eq = k2e(bl.ni), k2e(bl.ti)
        Zeff_eq = np.clip(k2e(bl.Zeff), 1.0, None)
        pressure = EC * (ne_eq * te_eq + ni_eq * ti_eq)
        if getattr(bl, "Z_imp", None):
            from .physics import impurity_pressure
            _zf = getattr(bl, "z_fast", None)
            _ne_th_eq = (ne_eq if _zf is None
                         else np.maximum(ne_eq - k2e(_zf), 0.0))
            pressure = pressure + impurity_pressure(_ne_th_eq, ni_eq, ti_eq,
                                                    bl.Z_imp)
        if bl.p_fast is not None:
            pressure = pressure + k2e(bl.p_fast)
        if getattr(bl, "p_diff", None) is not None:
            pressure = pressure + np.asarray(bl.p_diff, dtype=float)

        # state-anchor solve at the baseline j_phi (mirrors the per-draw flow)
        pp = {"type": "linterp",
              "y": pchip_derivative(psi_N, pressure) /
                   (mygs.psi_bounds[1] - mygs.psi_bounds[0]),
              "x": psi_N}
        pp["y"][-1] = 0.0
        ffp = {"type": "jphi-linterp",
               "y": np.asarray(bl.j_phi, dtype=float).copy(), "x": psi_N}
        # ---- psi re-initialisation before the state-anchor solve ------------
        # The reconstruction leaves mygs on its own converged inverse-mode
        # state.  When that state already sits (to ~1e-4 in the nonlinear
        # residual) ON this forward jphi-linterp solve's fixed point, the
        # under-relaxed Picard iteration has no gradient to descend and parks
        # in a small limit cycle just ABOVE nl_tol=1e-6 instead of crossing it
        # -- the solve then burns all 800 iterations and raises
        # 'Exceeded "maxits"'.  Unlike the per-draw anchor
        # (TokaMaker_interface.py) this call is not wrapped in try/except, so
        # the failure propagates and kills the run.
        # Re-initialising psi from the LCFS shape starts the iteration far
        # enough from the fixed point that it converges normally (same
        # convention as the IMAS forward-solve init at run.py:612).
        #
        # STATE GUARD.  init_psi DISCARDS the reconstruction's converged state
        # and installs a cold analytic psi.  Before this re-init a failed solve
        # here left mygs on that converged state -- benign, which is why the
        # failure was allowed to propagate untouched.  Now a failure would
        # leave the caller holding a cold, non-converged psi instead.  The
        # exception must stay fatal (this is a verification routine; a silent
        # fallback would defeat it), but the state it leaves behind should be
        # the one it was handed.  Snapshot, restore on the way out, re-raise.
        _snap = None
        _can_snap = (hasattr(mygs, "copy_eq") and hasattr(mygs, "replace_eq"))
        if getattr(self, "_boundary_RZ", None) is not None:
            if _can_snap:
                _snap = mygs.copy_eq()
            _R0, _Z0, _a, _kappa, _delta = _shape_from_boundary(
                self._boundary_RZ)
            mygs.init_psi(_R0, _Z0, _a, _kappa, _delta)
        mygs.set_targets(Ip=float(bl.Ip_target), pax=float(pressure[0]))
        mygs.set_profiles(pp_prof=pp, ffp_prof=ffp)
        try:
            mygs.solve()
        except Exception:
            if _snap is not None:
                # do not hand back the cold psi this method installed
                mygs.replace_eq(source_eq=_snap)
            raise

        seed = create_power_flux_fun(len(psi_N), 1.5, 1.5)["y"]
        res = solve_with_bootstrap(
            mygs, ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
            float(bl.Ip_target), seed,
            scale_jBS=float(getattr(bl, "bs_scale", 1.0)),
            isolate_edge_jBS=bool(gc.isolate_edge_jBS),
            diagnostic_plots=False, iterations=swb_iterations)
        spike0 = smooth_jbs_transition(
            _swb_jbs_to_toroidal(mygs, res["isolated_j_BS"], psi_pad))
        if gc.floor_j_BS:
            spike0 = np.clip(spike0, 0.0, None)

        ref = np.asarray(bl.j_BS, dtype=float)
        dev = spike0 - ref
        peak = float(np.max(np.abs(ref)))
        # Floored-zone carve-out: where floor_inductive_split clamped the
        # baseline j_inductive to 0, the deficit was absorbed into bl.j_BS --
        # so bl.j_BS deliberately differs from the raw SWB spike there (on a
        # strong-pedestal case, ~4% of peak at psi_N~0.98).  The
        # per-draw sampler has its own floor-zone handling, so these points
        # are excluded from the pass criterion and reported separately.
        floored = np.asarray(bl.j_inductive, dtype=float) <= 0.0
        dev_eval = np.where(floored, 0.0, dev)
        iworst = int(np.argmax(np.abs(dev_eval)))
        out = dict(spike0=spike0,
                   max_dev=float(np.max(np.abs(dev_eval))),
                   rms_dev=float(np.sqrt(np.mean(dev_eval ** 2))),
                   max_dev_frac=float(np.max(np.abs(dev_eval)) / peak),
                   psi_worst=float(psi_N[iworst]),
                   n_floored=int(floored.sum()),
                   max_dev_floored=(float(np.max(np.abs(dev[floored])))
                                    if floored.any() else 0.0),
                   passed=bool(np.max(np.abs(dev_eval)) <= tol_frac * peak))
        # leave mygs re-anchored on the baseline equilibrium, not SWB's state
        mygs.set_targets(Ip=float(bl.Ip_target), pax=float(pressure[0]))
        mygs.set_profiles(pp_prof=pp, ffp_prof=ffp)
        try:
            mygs.solve()
        except (ValueError, RuntimeError):
            pass
        status = "PASS" if out["passed"] else "FAIL"
        _fl = (f"; {out['n_floored']} floored pts excluded "
               f"(dev there {out['max_dev_floored']/1e6:.4f} MA/m²)"
               if out["n_floored"] else "")
        print(f"[sigma0-check] {status}: max|spike0 - j_BS| = "
              f"{out['max_dev']/1e6:.4f} MA/m² ({100*out['max_dev_frac']:.2f}% "
              f"of peak, worst at psi_N={out['psi_worst']:.3f}; "
              f"tol {100*tol_frac:.1f}%{_fl})")
        return out

    # ── stage 3: perturbed bouquet --------------------------------------
    def _validate_workflow(self) -> None:
        """Hard guard enforcing the validated per-path workflow at generate().

        ``from_imas`` auto-applies diff+C; ``from_geqdsk`` auto-applies the
        standard flagship l_i loop. This raises on a known-bad override so a
        user can't accidentally select a workflow that won't work. Set
        ``config.generation.allow_unsafe_workflow=True`` to bypass (warns
        instead) for deliberate backend tests / experiments.

        Relaxed: ``perturb_jind_in_anchor=True`` (route R2) on the
        reconstruction/geqdsk path used to be a hard error. It was barred
        because R2 hands :math:`I_p` to the inductive amplitude alone via
        ``Ip_flux_integral_vs_target``, and that root was evaluated on
        ``solve_with_bootstrap``'s landed equilibrium rather than the anchor
        -- measured on the synthetic D3D-like example at :math:`\\sigma=0`,
        where the archived split must be reproduced exactly, it returned
        ``0.8373`` instead of ``1.000`` and pulled :math:`l_i` 2.0 % low. With
        the root pinned to the anchor geometry and calibrated against the
        archived total (see
        :class:`bouquet.TokaMaker_interface._AnchorIpRenorm`) the same case
        returns ``0.99915`` with :math:`l_i` +0.10 %, so the guard no longer
        applies to the fixed implementation and is downgraded to a one-line
        note. ``generate()``'s production defaults are unchanged: R2 stays
        opt-in, for the deterministic/anchor context.
        """
        from .config import ReconstructionSource, ImasSource
        gc = self.config.generation
        uc = self.config.uncertainty
        problems = []
        # Named-preset facade (F4): "custom" (or the deprecated
        # allow_unsafe_workflow=True) downgrades the guard to a warning; the two
        # named presets assert the source matches.
        is_recon = isinstance(self.config.source, ReconstructionSource)
        wf = getattr(gc, "workflow", "auto")
        if wf == "geqdsk-standard" and not is_recon:
            problems.append("workflow='geqdsk-standard' but the source is IMAS")
        if wf == "imas-diff-c" and is_recon:
            problems.append("workflow='imas-diff-c' but the source is a g-file")
        custom = (wf == "custom") or bool(gc.allow_unsafe_workflow)
        # Rule: j_inductive must be perturbed (both workflows do it via
        # jphi_scalar_sigma; zero sigma freezes it).
        if float(getattr(uc, "jphi_scalar_sigma", 0.0)) <= 0.0:
            problems.append("jphi_scalar_sigma<=0 freezes j_inductive "
                            "perturbation (violates the all-profiles rule)")
        if isinstance(self.config.source, ReconstructionSource):
            # perturb_jind_in_anchor (route R2) is no longer a hard error on
            # the geqdsk path -- see the method docstring.  It is still not
            # the default, so say so once when it is selected.
            if gc.perturb_jind_in_anchor:
                print("NOTE: geqdsk path + perturb_jind_in_anchor=True "
                      "(route R2). The Ip renormalisation is now evaluated on "
                      "the anchor geometry (sigma=0 -> scale 1.000), but R2 "
                      "accepts the band-conditioned anchor draw and can still "
                      "exhaust its resamples on a stiff g-file; the standard "
                      "l_i loop (perturb_jind_in_anchor=False) remains the "
                      "default for ensembles.")
        elif isinstance(self.config.source, ImasSource):
            if not gc.perturb_jind_in_anchor:
                problems.append("IMAS path without Fix C "
                                "(perturb_jind_in_anchor=False): the "
                                "find_optimal_scale/corrector matching loop "
                                "homogenizes draws; use diff+C "
                                "(perturb_jind_in_anchor=True)")
            if gc.jBS_baseline_mode not in ("diff", "rescale", "ohmic"):
                problems.append(f"IMAS path jBS_baseline_mode="
                                f"{gc.jBS_baseline_mode!r} not in "
                                f"('diff','rescale','ohmic')")
            elif gc.jBS_baseline_mode == "ohmic":
                if str(getattr(gc, "closure_channel", "bootstrap")) \
                        not in ("bootstrap", "ohmic", "sawtooth_bootstrap",
                                "structured"):
                    # catch the typo HERE: the run-time dispatch only reaches
                    # its unknown-channel refusal after the full SWB solve
                    problems.append(
                        f"closure_channel="
                        f"{gc.closure_channel!r} not in "
                        f"('bootstrap','ohmic','sawtooth_bootstrap',"
                        f"'structured')")
                elif str(getattr(gc, "closure_channel", "")) == "structured":
                    # Same rule as the typo above: a structured-channel
                    # misconfiguration used to raise deep inside the
                    # predictor, i.e. AFTER the full SWB iteration sequence.
                    # The run-time refusals stay (defence in depth); these
                    # catch the same mistakes before anything is solved.
                    if (getattr(gc, "structured_ip_sigma", None) is not None
                            and getattr(gc, "structured_ip_sigma_frac", None)
                            is not None):
                        problems.append(
                            "closure_channel='structured': "
                            "structured_ip_sigma and structured_ip_sigma_frac "
                            "are mutually exclusive; set one, so the recorded "
                            "sigma_Ip is unambiguous")
                    if (getattr(gc, "structured_soft", False)
                            and getattr(gc, "structured_li_target", None)
                            is not None
                            and getattr(gc, "structured_li_sigma", None)
                            is None):
                        problems.append(
                            "closure_channel='structured' with "
                            "structured_soft=True and structured_li_target "
                            "set needs structured_li_sigma: the soft channel "
                            "weights l_i by its own error bar and bouquet "
                            "will not invent one")
                    _steps = getattr(gc, "structured_li_max_corrector_steps",
                                     1)
                    if _steps is not None and int(_steps) < 1:
                        # was silently coerced to 1 by max(1, int(x) or 1)
                        problems.append(
                            "structured_li_max_corrector_steps="
                            f"{_steps!r} must be >= 1 (it is a ceiling on "
                            "extra solves, and 0 does not mean 'no corrector' "
                            "-- the predictor readback always happens)")
                    _ltol = getattr(gc, "structured_li_tol", 0.005)
                    if _ltol is not None and float(_ltol) <= 0.0:
                        problems.append(
                            f"structured_li_tol={_ltol!r} must be > 0 "
                            "(a non-positive acceptance band flags every "
                            "slice closure-limited)")
                # Baseline-only for now: the draw path's sigma=0 reproduction
                # of an ohmic-closed baseline has not been verified, so the
                # UQ ensemble refuses the mode -- UNLESS workflow='custom'
                # (or allow_unsafe_workflow), which downgrades every problem
                # below to a printed WARN, this one included.  So the
                # unvalidated draw path IS reachable, deliberately, behind
                # that opt-out; test_custom_workflow_downgrades_to_warning
                # pins the downgrade.  Nothing here makes it unreachable.
                problems.append(
                    "jBS_baseline_mode='ohmic' is baseline-only for now "
                    "(draw-path sigma=0 reproduction unverified); run the "
                    "baseline study without generate()")
        if not problems:
            return
        msg = ("bouquet workflow guard: " + "; ".join(problems)
               + ". This is the validated per-path workflow lock; set "
               "config.generation.workflow='custom' to override "
               "(backend tests / experiments only).")
        if custom:
            print("WARN: " + msg)
        else:
            raise ValueError(msg)

    def generate(self, n: Optional[int] = None, progress_callback=None) -> list:
        """Generate the perturbed bouquet and archive to ``{header}.h5``.

        Auto-feeds the baseline (j_phi, j_inductive, l_i_target, Ip_target) and
        the uncertainty envelope (kinetic sigmas, j_phi sigma, GPR lengths).
        ``n`` overrides ``config.generation.n_equils`` for a quick smaller run.

        With ``config.generation.n_inspec_target`` set, the run keeps drawing
        until that many draws pass the coil + boundary filters -- the SAME
        filters, with the same settings, that :meth:`filter` then applies:
        ``filtering.coil_filter`` (chi2 by default, with its per-coil sigma,
        DAQ era and acceptance thresholds) and ``filtering.rms_max_mm``, so
        the count matches what :meth:`filter` marks ``selected`` --
        capped by ``max_total_draws``; ``n_equils`` is then
        the initial allocation rather than the total, and ``n`` overrides that
        allocation, not the target. Out-of-spec draws are still archived.

        Requires :meth:`prepare_baseline` first; raises if ``self.baseline`` is
        None.
        """
        import numpy as np
        from .baseline import resolve_uncertainty
        from .TokaMaker_interface import generate_bouquet
        from .utils import initialize_equilibrium_database

        if self.baseline is None:
            raise ValueError("call prepare_baseline() before generate()")
        if self.mygs is None:
            raise ValueError("call setup_solver() before generate()")

        self._validate_workflow()

        bl = self.baseline
        gc = self.config.generation
        fc = self.config.filtering
        n_equils = int(n if n is not None else gc.n_equils)

        # BouquetConfig validates in __post_init__, but the documented notebook
        # idiom mutates fields afterwards (`bq.generation.n_equils = ...`), so
        # re-check the until-N pair here -- the point where they take effect.
        from .config import require_integer_count as _int_count
        _int_count(gc.n_inspec_target, "generation.n_inspec_target")
        _int_count(gc.max_total_draws, "generation.max_total_draws")
        if gc.n_inspec_target is not None and int(gc.n_inspec_target) < 1:
            raise ValueError("generation.n_inspec_target must be >= 1 or None")
        if (gc.n_inspec_target is not None
                and gc.max_total_draws is not None
                and int(gc.max_total_draws) < int(gc.n_inspec_target)):
            # the constructor validates this pair, but the documented
            # notebook idiom mutates the fields afterwards -- catch it here
            # rather than after minutes of baseline work inside generate
            raise ValueError(
                f"generation.max_total_draws ({int(gc.max_total_draws)}) is "
                f"below n_inspec_target ({int(gc.n_inspec_target)}): the cap "
                "would stop the run before the target could ever be met")
        if gc.n_inspec_target is None and gc.max_total_draws is not None:
            import warnings as _w
            _w.warn(
                "max_total_draws has no effect without n_inspec_target: "
                f"this run draws exactly {n_equils} (max_total_draws="
                f"{gc.max_total_draws} ignored).", UserWarning, stacklevel=2)

        env = resolve_uncertainty(self.config, bl)
        self._resolved_uncertainty = env

        header = self.config.output_header
        initialize_equilibrium_database(header)

        # Restore the reconstruction isoflux targets before sampling (the recon
        # solve leaves them in place; an explicit restore is harmless otherwise).
        if bl.recon is not None and "isoflux_pts" in bl.recon:
            self.mygs.set_isoflux(bl.recon["isoflux_pts"], weights=bl.recon["weights"])

        psi_pad = float(getattr(self.config.source, "psi_pad", 1e-3))

        # Zeff is consumed on the EQUILIBRIUM grid (psi_N) by solve_with_bootstrap,
        # whereas the perturbed kinetic profiles (ne/te/ni/ti, sigmas) live on the
        # kinetic grid (psi_N_kinetic). PCHIP-regrid Zeff down to psi_N (shared
        # helper, matching every other kin->eq site).
        from .utils import pchip_interp
        Zeff_eq = np.clip(
            pchip_interp(bl.psi_N_kinetic, bl.Zeff, np.asarray(bl.psi_N, dtype=float)),
            1.0, None,
        )

        # The per-draw solver emits a large volume of diagnostic text (homotopy
        # passes, bootstrap/boundary diagnostics, DLSODE chatter) -- enough to
        # bloat a notebook by tens of MB. Capture it unless verbose, so output
        # stays readable; the full text is kept on generation_log for debugging.
        # Set BouquetConfig.verbose=True to stream it (and the tqdm progress bar).
        #
        # Center the per-draw bootstrap scale on the calibrated bs_scale so the
        # SWB amplitude correction established in prepare_baseline applies to
        # EVERY draw; the configured jBS_scale_range spread is retained as
        # bootstrap-model uncertainty around that center. bs_scale == 1.0 (no
        # SWB rebuild, e.g. reconstruction path) leaves the range unchanged.
        _bs = float(getattr(bl, "bs_scale", 1.0))
        _jbs_range = (None if gc.jBS_scale_range is None
                      else (gc.jBS_scale_range[0] * _bs, gc.jBS_scale_range[1] * _bs))

        from .utils import capture_native_output
        verbose = bool(getattr(self.config, "verbose", False))
        with capture_native_output(enabled=not verbose) as _cap:
            self.diagnostics = generate_bouquet(
                self.mygs, np.asarray(bl.psi_N, dtype=float), n_equils, header,
                np.asarray(bl.j_phi, dtype=float),
                bl.ne, bl.te, bl.ni, bl.ti,
                env["sigma_ne"], env["sigma_te"], env["sigma_ni"], env["sigma_ti"],
                env["sigma_jphi"],
                env["n_ls"], env["t_ls"], env["j_ls"],
                bl.Ip_target, bl.l_i_target, Zeff_eq,
                input_jinductive=np.asarray(bl.j_inductive, dtype=float),
                baseline_j_BS=(np.asarray(bl.j_BS, dtype=float)
                               + (0.0 if bl.jBS_diff is None
                                  else np.asarray(bl.jBS_diff, dtype=float))),
                l_i_tolerance=gc.l_i_tolerance,
                psi_pad=psi_pad,
                constrain_sawteeth=gc.constrain_sawteeth,
                recalculate_j_BS=gc.recalculate_j_BS,
                isolate_edge_jBS=gc.isolate_edge_jBS,
                floor_j_BS=gc.floor_j_BS,
                jBS_diff=(None if bl.jBS_diff is None
                          else np.asarray(bl.jBS_diff, dtype=float)),
                accept_anchor_inband=gc.accept_anchor_inband,
                perturb_jind_in_anchor=gc.perturb_jind_in_anchor,
                jBS_scale_range=_jbs_range,
                jbs_delta_mode=gc.jbs_delta_mode,
                swb_iterations=gc.swb_iterations,
                diagnostic_plots=gc.diagnostic_plots,
                capture_live_eq=gc.capture_live_eq,
                capture_npsi=gc.capture_npsi,
                capture_exact_inv_R2=gc.capture_exact_inv_R2,
                scan_key=gc.scan_key,
                pfile_bytes=bl.pfile_bytes,
                baseline_eqdsk_bytes=bl.eqdsk_bytes,
                baseline_pfile_bytes=bl.pfile_bytes,
                psi_N_kinetic=np.asarray(bl.psi_N_kinetic, dtype=float),
                coil_drift=gc.coil_drift,
                coil_drift_hard_factor=gc.coil_drift_hard_factor,
                homotopy_passes=gc.homotopy_passes,
                inspec_F_max=fc.inspec_F_max,
                inspec_VSC_max=fc.inspec_VSC_max,
                # until-N: the stopping rule reads its thresholds from the SAME
                # FilterConfig that .filter() will later cut on, so the loop
                # counts exactly what the postprocess marks 'selected'. An
                # explicit n= override is an allocation, not a target, so it
                # does not disable the target.
                n_inspec_target=gc.n_inspec_target,
                max_total_draws=gc.max_total_draws,
                inspec_rms_max_mm=fc.rms_max_mm,
                # ...including the COIL criterion: same filter, same sigma,
                # same acceptance numbers and -- via _coil_daq_era() -- the
                # same era floor .filter() will resolve. A loop still counting
                # the legacy +/-2% band while .filter() cuts on chi2 would
                # stop on one set of draws and select a different one.
                coil_filter=fc.coil_filter,
                coil_sigma=fc.coil_sigma,
                coil_device=self.config.device,
                coil_daq_era=self._coil_daq_era(),
                coil_chi2_max=fc.chi2_max,
                coil_z_max=fc.z_max,
                seed=gc.seed,
                # Fixed additive components, summed into every draw, never perturbed.
                p_fast=bl.p_fast,
                # Pressure anchor: impurity (carbon) thermal pressure via the
                # single effective Z_imp, + the diff offset that anchors the solve
                # pressure to the dd equilibrium.pressure (mirrors jBS_diff).
                Z_imp=getattr(bl, "Z_imp", None),
                z_fast=getattr(bl, "z_fast", None),
                p_diff=getattr(bl, "p_diff", None),
                # Total-current anchor to equilibrium.j_tor (fixed offset; rides
                # under the SWB bootstrap + perturbed j_ind in every draw).
                jphi_diff=getattr(bl, "jphi_diff", None),
                j_NBI=bl.j_NBI,
                j_RF=bl.j_RF,
                # Switchboard: auxiliary perturbed profiles -- rotation /
                # transport channels (passive) + Zeff (active).
                aux_sigmas=env.get("aux_sigmas"),
                aux_baselines=env.get("aux_baselines"),
                aux_length_scales=env.get("aux_length_scales"),
                progress_callback=progress_callback,
                # Provenance marker stored on the baseline for robust path
                # detection in plotting (independent of the aux switchboard).
                source_kind=("imas"
                             if type(self.config.source).__name__ == "ImasSource"
                             else "geqdsk"),
                # BOTH paths: archive the ACHIEVED FSA j_phi of each converged
                # solve (baseline + draws) so the stored 1-D current always
                # matches the stored eqdsk in the same group. On the IMAS path
                # the single-pass forward solve lands a few % off its anchor
                # target; on the geqdsk path the per-draw corrective iteration
                # bounds the target-vs-achieved gap to its tolerance (~2-3%
                # core RMS) -- storing the achieved output removes even that.
                store_achieved_jphi=True,
            )
        self.generation_log = _cap["text"] or None

        # until-N outcome, OUTSIDE the capture: on the default quiet path the
        # in-loop prints and generate_bouquet's cap-missed RuntimeWarning were
        # swallowed into generation_log (capture_native_output redirects
        # stderr too), so a run that failed to deliver the requested ensemble
        # returned with zero visible signal unless .filter() happened to run.
        # A failure signal may not live only in a log attribute.
        if gc.n_inspec_target is not None:
            from .filtering import until_n_delivered
            _tgt = int(gc.n_inspec_target)
            _got = until_n_delivered(self.diagnostics)
            _tries = len(self.diagnostics or [])
            if _got < _tgt:
                import warnings as _w
                _msg = (f"until-N did not reach its target: {_got}/{_tgt} "
                        f"in-spec draws after {_tries} attempts (cap "
                        f"{gc.max_total_draws or 'default'}). The archive "
                        "holds every attempt; raise max_total_draws, loosen "
                        "the filter thresholds deliberately, or treat the "
                        "low yield as a finding about this equilibrium.")
                print(f"[until-N] WARNING: {_msg}")
                _w.warn(_msg, RuntimeWarning, stacklevel=2)
            else:
                print(f"[until-N] target met: {_got}/{_tgt} in-spec draws "
                      f"in {_tries} attempts.")

        # Stamp provenance (schema/version/timestamp + full config JSON) onto the
        # archive so the run is self-describing and load_config() can round-trip it.
        from .utils import write_provenance
        write_provenance(header, config=self.config, scan_key=gc.scan_key)

        return self.diagnostics

    def plot_bouquet(self, mode: str = "all", selection: str = "all",
                     layout: str = "stack", pub_style: bool = False):
        """Overlay plots of the generated bouquet (kinetic / pressure / j_phi /
        boundary). Thin wrapper over :func:`bouquet.plotting.plot_bouquet` on
        this run's HDF5 (``{output_header}.h5``). ``pub_style=False`` (default)
        is one combined dashboard figure; ``pub_style=True`` (with
        ``layout="row"``) gives the separate side-by-side publication figures."""
        from .plotting import plot_bouquet as _plot_bouquet
        return _plot_bouquet(f"{self.config.output_header}.h5",
                             scan_key=self.config.generation.scan_key,
                             mode=mode, selection=selection,
                             layout=layout, pub_style=pub_style)

    # ── thin post-run wrappers (auto-wire header + scan_key -- kill the manual
    # (HEADER, scan_key) re-threading; F1) ─────────────────────────────────
    def plot_traces(self, **kwargs):
        """Per-draw l_i / LCFS-deviation / anchor-displacement traces for this run."""
        from .plotting import plot_traces as _f
        kwargs.setdefault("li_band", self.config.generation.l_i_tolerance)
        kwargs.setdefault("rms_max_mm", self.config.filtering.rms_max_mm)
        return _f(f"{self.config.output_header}.h5",
                  scan_key=self.config.generation.scan_key, **kwargs)

    def plot_coil_currents(self, **kwargs):
        """Per-coil drift heatmap for this run."""
        from .plotting import plot_coil_currents as _f
        return _f(f"{self.config.output_header}.h5",
                  scan_key=self.config.generation.scan_key, **kwargs)

    def plot_spec_summary(self, **kwargs):
        """In-spec fraction summary (coil + boundary) for this run."""
        from .plotting import plot_spec_summary as _f
        kwargs.setdefault("rms_max_mm", self.config.filtering.rms_max_mm)
        return _f(self.config.output_header,
                  scan_key=self.config.generation.scan_key, **kwargs)

    def selected_indices(self, selection: str = "selected") -> list:
        """Stored draw indices for this run's scan (``selected``/``all``/``excluded``)."""
        from .filtering import select_indices
        return select_indices(self.config.output_header,
                               scan_key=self.config.generation.scan_key,
                               selection=selection)

    def output_spread(self, selection: str = "all", print_table: bool = True) -> dict:
        """Across-draw spread of the global output scalars: l_i(1)/l_i(3), the
        volume-averaged pressure ``<P>``, and ``beta_N``.

        Convenience over ``run.archive.scan().spread(...)`` -- surfaces the
        pressure-quantity uncertainty alongside the l_i variance for this run's
        scan in one call. See :meth:`bouquet.archive.ScanView.spread`.
        """
        return self.archive.scan(self.config.generation.scan_key).spread(
            selection=selection, print_table=print_table)

    # ── stage 4: filter + export ----------------------------------------
    def filter(self, rms_max_mm: Optional[float] = None, plot: bool = False) -> dict:
        """Mark the machine-realizable subset (coil + boundary filters).

        Non-destructive: writes pass flags into the HDF5. Returns a summary dict.
        With ``plot=True`` the coil-drift and boundary distribution figures are
        produced and returned under ``summary["figures"] = (coil_fig, bnd_fig)``
        (F7 -- the notebooks re-called the module filters just to get these).
        """
        from .filtering import filter_coil_currents, filter_boundaries, filter_coil_chi2

        header = self.config.output_header
        fc = self.config.filtering
        rms = fc.rms_max_mm if rms_max_mm is None else rms_max_mm

        sk = self.config.generation.scan_key
        coil_filter_used = fc.coil_filter
        if fc.coil_filter == "chi2":
            from .coil_spec import CoilSigmaUnavailable
            # the era sets the sigma floor, i.e. an acceptance criterion -- say
            # which one was used and where it came from, once per filter call
            era = self._coil_daq_era()
            print(self._coil_era_note(era))
            try:
                coil_summary = filter_coil_chi2(
                    header, None, scan_key=sk, chi2_max=fc.chi2_max,
                    apply=True, sigma=fc.coil_sigma, device=self.config.device,
                    era=era, z_max=fc.z_max,
                )
                coil_fig = None
                if plot:  # drift-distribution figure only; writes no flags
                    _, coil_fig = filter_coil_currents(
                        header, scan_key=sk, F_max_pct=fc.inspec_F_max * 100.0,
                        VSC_max_pct=fc.inspec_VSC_max * 100.0, apply=False, plot=True,
                    )
            except CoilSigmaUnavailable as e:
                import warnings
                # one wording, shared with the until-N loop's own fallback, so
                # "the loop falls back exactly where .filter() does" is a fact
                # about one function rather than two copies of a message
                from .filtering import _coil_fallback_message
                warnings.warn(
                    _coil_fallback_message(e, fc.inspec_F_max, fc.inspec_VSC_max),
                    stacklevel=2)
                coil_filter_used = "legacy(fallback)"
                coil_summary, coil_fig = filter_coil_currents(
                    header, scan_key=sk,
                    F_max_pct=fc.inspec_F_max * 100.0,
                    VSC_max_pct=fc.inspec_VSC_max * 100.0,
                    apply=True, plot=plot,
                )
        else:
            coil_summary, coil_fig = filter_coil_currents(
                header, scan_key=sk,
                F_max_pct=fc.inspec_F_max * 100.0,
                VSC_max_pct=fc.inspec_VSC_max * 100.0,
                apply=True, plot=plot,
            )
        bnd_summary, bnd_fig = filter_boundaries(
            header, scan_key=sk, rms_max_mm=rms, apply=True, plot=plot,
        )
        # one scan key -> each summary is a single {counts, draws} dict
        self._selection = {"coil": coil_summary, "boundary": bnd_summary,
                           "coil_filter_used": coil_filter_used}
        if plot:
            self._selection["figures"] = (coil_fig, bnd_fig)
        self._print_generation_summary(coil_summary, bnd_summary)
        return self._selection

    def _coil_daq_era(self):
        """Acquisition era label for the era-dependent coil tolerance floor, or None.

        The era selects the sigma FLOOR, i.e. an acceptance criterion, so it is
        only ever taken from something the user actually stated:

          1. ``filtering.coil_daq_era`` -- the explicit override;
          2. an explicit ``source.pulse`` / ``source.shot`` field, mapped through
             the device's own era bands (:func:`devices.era_for_pulse`).

        Route 2 uses the pulse number as a DATE PROXY for the coil-current
        acquisition upgrade -- the upgrade has a date, and the pulse index is the
        only monotone clock the archive carries -- so the band boundary is
        approximate and a pulse near it may belong to the other era.  On DIII-D
        the two floors it chooses between are 825 A-t (``"pre2014"``) and
        325 A-t (``"modern"``), refined per coil; set ``filtering.coil_daq_era``
        to override the proxy with a stated era.  Either way :meth:`filter`
        prints the decision (:meth:`_coil_era_note`) once per call.

        Nothing is scraped out of a run header, mesh name or file path.  That
        fallback used to exist and was a silent tolerance relaxation: any six
        consecutive digits anywhere in the header matched, so naming a run after
        a mesh resolution or a date handed every coil a 2.5x looser floor.

        ``None`` (era unknown) leaves the device default -- the tightest floor --
        in force, and the filter warns that it is doing so.
        """
        fc = self.config.filtering
        if getattr(fc, "coil_daq_era", None):
            return fc.coil_daq_era
        from .devices import era_for_pulse, resolve_device
        pulse = self._coil_source_pulse()
        if pulse is None or self.config.device is None:
            # without a named device there is no era table to map a pulse onto;
            # detection from the mesh happens later, inside resolve_coil_sigma.
            return None
        spec = resolve_device(self.config.device)
        return era_for_pulse(spec, pulse) if spec is not None else None

    def _coil_source_pulse(self):
        """Explicit ``source.pulse``/``source.shot`` as an int, else None."""
        src = self.config.source
        for attr in ("pulse", "shot"):
            v = getattr(src, attr, None)
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
        return None

    def _coil_era_note(self, era) -> str:
        """One line naming the acquisition era the coil chi2 filter is about to
        use, how it was arrived at, and the sigma floor it buys.

        The era moves an acceptance threshold, so the decision belongs in the run
        log and not only in a config field -- especially when it came from the
        pulse number, which is a date proxy for the acquisition upgrade and so
        puts an approximate boundary between two floors.  :meth:`filter` prints
        this once per call (not per draw); when ``filtering.coil_sigma`` is set
        the sigma bypasses the device model altogether and the line says so.
        """
        from .devices import resolve_device, tolerance_for
        fc = self.config.filtering
        lead = "  coil chi2 filter:"
        if fc.coil_sigma is not None:
            named = f" ({fc.coil_sigma!r})" if isinstance(fc.coil_sigma, str) else ""
            return (f"{lead} per-coil sigma from filtering.coil_sigma{named}; "
                    "the acquisition era does not apply")
        spec = resolve_device(self.config.device) if self.config.device else None

        def _floor(label):
            """' (sigma floor F A-t)' for *label*, or '' if no device is named."""
            if spec is None:
                return ""
            return f" (sigma floor {tolerance_for(spec, era=label)[0]:g} A-t)"

        if era is not None:
            if getattr(fc, "coil_daq_era", None):
                return f"{lead} era {era!r}{_floor(era)}  [explicit; filtering.coil_daq_era]"
            pulse, band = self._coil_source_pulse(), ""
            for lo, hi, _fl, lab in (spec.sigma_floor_by_era if spec else ()):
                if lab == era:
                    if lo <= 0:
                        band = f"pulse {pulse} < {hi:g} -> "
                    elif hi == float("inf"):
                        band = f"pulse {pulse} >= {lo:g} -> "
                    else:
                        band = f"pulse {pulse} in [{lo:g}, {hi:g}) -> "
                    break
            return (f"{lead} {band}era {era!r}{_floor(era)}  [automatic: the pulse "
                    "number is a date proxy for the acquisition upgrade, so the "
                    "boundary is approximate; set filtering.coil_daq_era to override]")
        # era undetermined -> tolerance_for falls back to the device's last band
        if spec is None:
            why = ("BouquetConfig.device is not set, so no era table applies yet"
                   if self.config.device is None else "no era table for this device")
            return (f"{lead} era undetermined ({why}) -> the device detected from the "
                    "mesh uses its default band, which carries the TIGHTEST floor  "
                    "[set filtering.coil_daq_era to state the era]")
        if not spec.sigma_floor_by_era:
            return (f"{lead} device {spec.name!r} has no acquisition eras; sigma floor "
                    f"{spec.sigma_floor:g} A-t")
        pulse = self._coil_source_pulse()
        why = ("no pulse on source.pulse / source.shot" if pulse is None
               else f"pulse {pulse} falls in no era band of device {spec.name!r}")
        default = spec.sigma_floor_by_era[-1][3]
        return (f"{lead} era undetermined ({why}) -> default band {default!r}"
                f"{_floor(None)}, the TIGHTEST floor  "
                "[set filtering.coil_daq_era to state the era]")

    def _print_generation_summary(self, coil_summary, bnd_summary):
        """Concise post-generation summary (draws / coil spec / boundary / in-spec),
        in the style of the reconstruction summary."""
        from .filtering import select_indices
        header = self.config.output_header
        sk = self.config.generation.scan_key
        n_all = len(select_indices(header, scan_key=sk, selection="all"))
        n_sel = len(select_indices(header, scan_key=sk, selection="selected"))

        # summaries are single per-scan dicts (one scan key); fall back to the
        # first entry if a multi-scan dict is ever passed in.
        def _one(summary):
            if not summary:
                return {}
            if "n_total" in summary or "rms_stats" in summary:
                return summary
            return next(iter(summary.values()), {}) or {}

        cs = _one(coil_summary)
        rs = (_one(bnd_summary).get("rms_stats") or {})
        fc = self.config.filtering
        tag = header.split("/")[-1]
        frac = 100.0 * n_sel / max(n_all, 1)
        print(f"\n=== Bouquet — {tag} {'=' * max(3, 34 - len(tag))}  "
              f"{n_sel}/{n_all} in-spec ({frac:.0f}%)")
        # getattr, not attribute access: .filter() is exercised with stub
        # config objects (the coil-filter wrapper tests), and a summary line
        # is not a thing a run may die on.
        _gtarget = getattr(self.config.generation, "n_inspec_target", None)
        if _gtarget is None:
            print(f"  draws         {n_all} generated")
        else:
            # The delivered-vs-requested line is the point of until-N: state
            # both, and whether the target was actually met, rather than
            # letting a short bouquet read as a completed run.
            _tgt = int(_gtarget)
            _verdict = ("target met" if n_sel >= _tgt
                        else f"SHORT of target by {_tgt - n_sel}")
            print(f"  draws         {n_all} generated to deliver "
                  f"{n_sel}/{_tgt} requested in-spec ({_verdict})")
        # name the criterion that was actually applied -- the chi2 filter reports
        # its own thresholds in the summary, the legacy rule the +/- band.
        if "chi2_max" in cs:
            zm = cs.get("z_max")
            crit = (f"chi2/nu <= {cs['chi2_max']:g}"
                    + (f", |z| <= {zm:g}" if zm is not None else "")
                    + f" [{cs.get('sigma_model', {}).get('acceptance', {}).get('source', '?')}"
                    + f", {cs.get('n_coils', '?')} coils]")
        else:
            crit = f"within ±{fc.inspec_F_max * 100:.0f}%"
        print(f"  coil spec     {cs.get('n_pass', '?')}/{cs.get('n_total', '?')} "
              f"{crit}   ({cs.get('n_fail', '?')} out-of-spec)")
        if rs:
            print(f"  boundary      RMS median {rs.get('median', float('nan')):.2f} mm"
                  f"   max {rs.get('max', float('nan')):.2f} mm")

    def export(self, out_path: Optional[str] = None, selection: str = "selected"):
        """Write a pruned HDF5 with only the selected draws.

        Defaults to ``{header}_selected.h5``.
        """
        from .filtering import export_filtered

        header = self.config.output_header
        out = out_path if out_path is not None else f"{header}_selected.h5"
        export_filtered(header, out, selection=selection, overwrite=True)
        return out

    def export_bundle(self, out_dir: str, formats=("geqdsk", "profiles"),
                      selection: str = "selected") -> dict:
        """Extract a per-draw file bundle (geqdsk / pfile / profiles JSON) for
        the ``selection`` draws to ``out_dir``. Returns ``{draw: {fmt: path}}``.

        Thin wrapper over :meth:`BouquetArchive.scan(...).extract`; the
        source-agnostic hand-off to codes that don't consume the HDF5 archive
        or IMAS IDS. For IMAS/OMAS IDS output use :func:`export_imas_drawset`.
        """
        sc = self.archive.scan(self.config.generation.scan_key)
        return sc.extract(out_dir, formats=formats, selection=selection)

    def export_ids(self, out_dir: str, selection: str = "selected",
                   time=None, fidelity: str = "auto") -> list:
        """Write one perturbed IMAS/OMAS IDS per ``selection`` draw to
        ``out_dir`` (IMAS source only). Thin wrapper over
        :func:`export_imas_drawset`; ``fidelity`` picks the exact
        (captured-geometry) or baseline-ratio current split."""
        from .config import ImasSource
        from .io.imas import export_imas_drawset
        if not isinstance(self.config.source, ImasSource):
            raise TypeError("export_ids is for an ImasSource; the reconstruction "
                            "path has no template IDS. Use export_bundle().")
        return export_imas_drawset(
            self.config.output_header, self.config.source.ids_path, out_dir,
            scan_key=self.config.generation.scan_key,
            time=self.config.source.time if time is None else time,
            selection=selection, fidelity=fidelity)

    # ── convenience -----------------------------------------------------
    def run(self) -> "Bouquet":
        """setup_solver -> prepare_baseline -> generate -> filter -> export.

        For scripts / CI where the baseline is already trusted. Returns self,
        fully populated (baseline, diagnostics, HDF5 path all reachable).

        Idempotent on the early stages: if the solver is already up (e.g. you
        called :meth:`reconstruct` first, or are reusing it across slices) it is
        not rebuilt, and an already-prepared baseline is not re-solved.
        """
        self.setup_solver()                       # idempotent
        if self.baseline is None:
            self.prepare_baseline()
        self.generate()
        self.filter()
        self.export()
        return self

    def run_slices(self, times, scan_keys=None, header=None, export=False) -> dict:
        """Sweep an IMAS time series into ONE archive, one ``scan_key`` per slice.

        Wraps the ``set_slice -> prepare_baseline -> generate -> filter`` loop
        (F5): keeps a single solver, writes every slice's draws under
        ``scan/<key>/`` in ``{header}.h5``, and returns a per-slice summary
        ``{scan_key: {time, n_all, n_sel, l_i, Ip}}``. ``scan_keys`` defaults to
        the time in ms (``round(t*1000)``). Set ``export=True`` to also write the
        selected-only copy after the last slice.

        Reconstruction sources have no time axis (:meth:`set_slice` raises on
        ``time``); build one :class:`Bouquet` per g-file instead.
        """
        times = [float(t) for t in times]
        if scan_keys is None:
            scan_keys = [int(round(t * 1000)) for t in times]     # ms labels
        if len(scan_keys) != len(times):
            raise ValueError("scan_keys must match times in length")
        if header is not None:
            self.config.output_header = header
        self.setup_solver()                                       # once
        results = {}
        for t, sk in zip(times, scan_keys):
            self.set_slice(time=t)
            self.config.generation.scan_key = sk
            self.prepare_baseline()
            self.generate()
            self.filter()
            bl = self.baseline
            results[sk] = dict(
                time=t,
                n_all=len(self.selected_indices("all")),
                n_sel=len(self.selected_indices("selected")),
                l_i=float(getattr(bl, "l_i_target", float("nan"))),
                Ip=float(getattr(bl, "Ip_target", float("nan"))),
            )
        if export:
            self.export()
        return results
