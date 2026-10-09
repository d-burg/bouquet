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
from .edge_pressure import (NegativeSeparatrixPressure,
                            resolve_edge_pressure, solver_pax,
                            solver_pp_profile, solver_pprime)
from . import coords
from .swb import SwbBaseline


def _baseline_negative_psep_named(fn):
    """``prepare_baseline``: a :class:`~bouquet.edge_pressure.
    NegativeSeparatrixPressure` raised while the BASELINE is built (the
    g-file reconstruction, the IMAS forward solve or the unified engine) is
    re-raised naming the object refused and why (disclosed 2026-10-06: the
    refusal of c709aae reaches the baseline on the default legacy path, not
    only the draws)."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except NegativeSeparatrixPressure as exc:
            src = type(getattr(self.config, "source", None)).__name__
            eng = getattr(self.config.generation, "reconstruction_engine",
                          "legacy")
            raise NegativeSeparatrixPressure(
                f"prepare_baseline REFUSED THE BASELINE (source "
                f"{src}, reconstruction_engine={eng!r}): the input's own "
                "pressure at the separatrix (psi_N = 1, as the solver is "
                "handed it: thermal + impurity + fast) is negative, which "
                "is unphysical input; under separatrix_pressure='offset' "
                "(the default) it would raise the axis-pressure target and "
                "write a negative boundary PRES.  Correct the input's edge "
                "profiles, or set generation.separatrix_pressure='legacy' "
                "(which never reads the edge value) to build it as before "
                f"2026-10-04.  [{exc}]") from exc
    return wrapper


def _zero_perturbation_env(env):
    """The uncertainty envelope of ``generate()`` with every sigma zero (the
    shapes, length scales and baselines kept): the zero-perturbation draw
    of ``verify_sigma0_consistency``'s engine route."""
    import numpy as np
    out = dict(env)
    for k in ("sigma_ne", "sigma_te", "sigma_ni", "sigma_ti", "sigma_jphi"):
        if out.get(k) is not None:
            out[k] = np.zeros_like(np.asarray(out[k], dtype=float))
    if out.get("aux_sigmas"):
        out["aux_sigmas"] = {
            c: np.zeros_like(np.asarray(v, dtype=float))
            for c, v in out["aux_sigmas"].items()}
    return out


def _terms(mygs, triples):
    """The solver's coil-regularisation terms of ``(coils, target, weight)``
    triples (:meth:`Bouquet._apply_coil_reg`)."""
    return [mygs.coil_reg_term(dict(c), target=float(tg), weight=float(w))
            for c, tg, w in triples]


class Bouquet(SwbBaseline):
    """Stateful driver: solver -> baseline -> generate -> filter -> export."""

    def __init__(self, config: "BouquetConfig"):
        self.config = config
        self.mygs = None                          # set by setup_solver()
        self.baseline: Optional["Baseline"] = None
        self._resolved_uncertainty = None         # resolved sigma profiles + length scales
        self.diagnostics: Optional[list] = None   # generate() per-draw output
        self.generation_log: Optional[str] = None # captured generate() solver chatter
        # generate(): one record per REJECTED draw attempt (reason code, stage,
        # error) -- never archived, never counted toward until-N
        self.draw_rejections: list = []
        # generate(): every draw solve that raised (DrawSolveGuard records)
        self.solve_failures: Optional[list] = None
        # prepare_baseline(): a half-built baseline from a FAILED build, for
        # debugging only -- never used by generate()
        self._failed_baseline = None
        # prepare_baseline(): how the engine-dependent settings were resolved
        # (bouquet.engine.resolve_engine_defaults)
        self._engine_resolved_defaults: Optional[dict] = None
        self._selection = None                    # filter() result

    # ── constructors ----------------------------------------------------
    @classmethod
    def from_geqdsk(cls, geqdsk_path, *, profiles, mesh,
                    n_draws=20, header="bouquet", cocos=1, time=None,
                    impurity_Z=6.0, reconstruction_engine=None,
                    **solver_kwargs) -> "Bouquet":
        """Minimal constructor for the reconstruction path (g-file + profiles).

        ``profiles`` is an IDA ``.cdf`` or a p-file (auto-detected).
        ``impurity_Z`` is the machine impurity charge (carbon 6.0 by default;
        set it for your device -- it controls the Z_eff<->ni mapping and is the
        IDA path's main-ion derivation, see :class:`ReconstructionSource`).
        Extra keyword args go to :class:`SolverConfig` (e.g. ``order``,
        ``nthreads``). Reach into ``bq.uncertainty`` / ``bq.generation``
        afterwards for the advanced knobs.

        ``reconstruction_engine`` (``None``: the :class:`GenerationConfig`
        default, ``"unified"`` since 2026-10-06) selects the reconstruction
        engine; ``"legacy"`` runs the legacy reconstruction and draws.  It
        may equally be set afterwards (``bq.generation.
        reconstruction_engine = "legacy"``): the engine-dependent workflow
        settings (``isolate_edge_jBS``, ``perturb_jind_in_anchor``;
        :data:`bouquet.engine.ENGINE_DEPENDENT_DEFAULTS`) are left unset
        here and resolved to the validated values of the engine configured
        when :meth:`prepare_baseline` runs (legacy g-file: the full-profile
        decomposition and the standard l_i loop).
        """
        from .config import (BouquetConfig, SolverConfig, ReconstructionSource,
                             GenerationConfig)
        gkw = ({} if reconstruction_engine is None
               else dict(reconstruction_engine=reconstruction_engine))
        cfg = BouquetConfig(
            source=ReconstructionSource(geqdsk_path=geqdsk_path,
                                        profiles_path=profiles,
                                        cocos=cocos, time=time,
                                        impurity_Z=impurity_Z),
            solver=SolverConfig(mesh_path=mesh, **solver_kwargs),
            generation=GenerationConfig(n_equils=n_draws, **gkw),
            output_header=header,
        )
        # the engine-dependent workflow settings are resolved at
        # prepare_baseline(), for the engine configured THEN (legacy g-file:
        # the standard flagship l_i loop, perturb_jind_in_anchor=False, and
        # the full-profile decomposition, isolate_edge_jBS=False)
        return cls(cfg)

    @classmethod
    def from_imas(cls, ids_path, *, mesh, time=None, ida_time=None,
                  n_draws=20, header="bouquet",
                  ida_path=None, LCFS_geqdsk=None, impurity_Z=6.0,
                  ni_source="all", zeff_from_fuse=False,
                  kinetic_source=None, anchor_pressure_to_equilibrium=False,
                  reconstruction_engine=None, solve_method=None,
                  **solver_kwargs) -> "Bouquet":
        """Minimal constructor for the IMAS/OMAS path (no reconstruction).

        Extra keyword args go to :class:`SolverConfig`. Reach into
        ``bq.uncertainty`` / ``bq.generation`` afterwards for advanced knobs.

        IDA-hybrid kinetics: pass ``ida_path`` (an IDA ``.cdf``) to take the
        baseline ne/Te/Ti/Zeff/omega_tor (and the ne/Te/ni/Ti/Z_eff sigma
        envelopes) from IDA fits while keeping FUSE currents/equilibrium.
        ``ni_source`` picks the IDA ni route ("Zeff"/"CER"/"all") for both the
        baseline ni and its propagated sigma; ``zeff_from_fuse=True`` keeps the
        FUSE Z_eff instead of IDA's. ``kinetic_source`` defaults to
        ``"ida_hybrid"`` when an ``ida_path`` is given, else ``"fuse"``.

        ``ida_time`` picks the IDA slice (default ``time``); ``time`` then picks
        only the dd slices.  With FUSE_JBS_ORDER=replay_first, a row
        ``(time, ida_time)`` of ``ida_provenance.json["replay_pairing"]`` means
        the dd j_bootstrap(time) was computed on IDA(ida_time);
        ``aux['pairing_consistent']`` records that check when the table sits
        beside ``ids_path``.

        ``LCFS_geqdsk`` is OPTIONAL: a g-file whose LCFS replaces the source
        boundary outline as the isoflux target, for when you have a better
        separatrix for the slice than the dd carries (typically a magnetics-only
        reconstruction). Omit it to use the source's own boundary.

        ``reconstruction_engine`` (``None``: the :class:`GenerationConfig`
        default, ``"unified"`` since 2026-10-06) selects the reconstruction
        engine; ``"legacy"`` runs the legacy IMAS baseline and draws.  It
        may equally be set afterwards: the engine-dependent workflow
        settings (``isolate_edge_jBS``, ``perturb_jind_in_anchor``;
        :data:`bouquet.engine.ENGINE_DEPENDENT_DEFAULTS`) are left unset
        here and resolved to the validated values of the engine configured
        when :meth:`prepare_baseline` runs (legacy IDS: diff+C and the
        full-profile decomposition).  ``anchor_pressure_to_equilibrium=True``
        is a legacy-path setting (refused under the engine).
        ``solve_method`` (``GenerationConfig.solve_method``) is the same
        choice by its current name: ``"engine"`` is ``"unified"``.
        """
        from .config import (BouquetConfig, SolverConfig, ImasSource,
                             GenerationConfig)
        if kinetic_source is None:
            kinetic_source = "ida_hybrid" if ida_path else "fuse"
        gkw = ({} if reconstruction_engine is None
               else dict(reconstruction_engine=reconstruction_engine))
        if solve_method is not None:
            gkw["solve_method"] = solve_method
        cfg = BouquetConfig(
            source=ImasSource(ids_path=ids_path, time=time, ida_time=ida_time, ida_path=ida_path,
                              impurity_Z=impurity_Z, ni_source=ni_source,
                              zeff_from_fuse=zeff_from_fuse,
                              LCFS_geqdsk=LCFS_geqdsk),
            solver=SolverConfig(mesh_path=mesh, **solver_kwargs),
            generation=GenerationConfig(n_equils=n_draws,
                                        kinetic_source=kinetic_source,
                                        anchor_pressure_to_equilibrium=anchor_pressure_to_equilibrium,
                                        **gkw),
            output_header=header,
        )
        # IDA-hybrid: source the kinetic sigma envelopes from the same IDA .cdf
        # (resolve_uncertainty fires its IDA branch whenever unc.ida_path is set).
        if ida_path:
            cfg.uncertainty.ida_path = ida_path
        # The engine-dependent workflow settings are resolved at
        # prepare_baseline(), for the engine configured THEN.  Legacy IDS:
        # diff+C (jBS_baseline_mode="diff", the dataclass default: anchor
        # the bootstrap to the source via the fixed FUSE_jBS-SWB diff;
        # perturb_jind_in_anchor=True: perturb j_ind in the recon-anchor to
        # avoid the find_optimal_scale/corrector homogenization) and the
        # full-profile decomposition (isolate_edge_jBS=False: FUSE/IMAS
        # sources carry a FULL Sauter bootstrap, so the edge-spike isolation
        # would mislabel a redundant j_BS,edge and mangle the per-draw
        # j_inductive).
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

    def set_slice(self, *, time=None, ida_time=None, header=None) -> "Bouquet":
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
            # a stale IDA slice must not ride along to a new dd slice
            if hasattr(self.config.source, "ida_time"):
                self.config.source.ida_time = ida_time
        if header is not None:
            self.config.output_header = header
        self.baseline = None
        self._resolved_uncertainty = None
        self.diagnostics = None
        self.solve_failures = None
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

        # The coil solve: OpenFUSIONToolkit's bounded (BVLS) mode, entered
        # ONCE, here, before the reconstruction's first solve, on every path
        # -- so the reconstruction, the sigma=0 check and every draw use one
        # coil solver whatever order they run in (the mode is one-way and
        # every generate() enters it; see bouquet.solver_state).  Installs
        # +/-1e98, which never binds.  The swb method never installs coil
        # bounds (no homotopy), so it keeps the unbounded solve throughout,
        # as before the bounded mode existed (DrawMethod.bounded_coil_mode).
        from .draw_methods import method_hooks
        from .solver_state import enter_bounded_coil_mode, keep_unbounded_coil_mode
        if method_hooks(self.config.generation).bounded_coil_mode:
            enter_bounded_coil_mode(mygs)
        else:
            keep_unbounded_coil_mode(mygs)

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
        (That swap is the LEGACY draw path's.  The engine draws install the
        reconstruction's own list instead: ``mygs._recon_coil_reg`` -- the
        terms installed here -- with its JSON record
        ``mygs._recon_coil_reg_record``, both published on every call.)

        Called from BOTH :meth:`setup_solver` and :meth:`_reset_solver_state`.
        That matters: ``_repoint_imas_geometry`` resets the solver immediately
        before the IMAS baseline solve, so targets installed only at setup were
        silently discarded and the solve ran on the zero-target default. A
        1e8-weight target moved its coil by 0.2 % for exactly that reason.
        """
        spec = list(getattr(self.config.solver, "coil_reg", None) or [])
        if spec:
            self._check_coil_reg_orientation(spec)
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
                for them to be held at.  Returns ``(coils, target, weight)``
                triples; :func:`_terms` builds the solver's terms from them.
                """
                out = [(dict(t["coils"]), float(t.get("target", 0.0)),
                        1.0 if exploratory else float(t.get("weight", 1.0)))
                       for t in spec
                       if not (exploratory and set(t["coils"]) == {"#VSC"})]
                out += [({n: 1.0}, 0.0, 1.0)
                        for n in mygs.coil_sets if n not in named]
                if exploratory or "#VSC" not in named:
                    out.append(({"#VSC": 1.0}, 0.0, 1e-2))
                return out

            triples = _build(False)
            mygs._weak_coil_reg = _terms(mygs, _build(True))
            source = "configured"
        else:
            triples = [({name: 1.0}, 0.0, 1.0) for name in mygs.coil_sets]
            triples.append(({"#VSC": 1.0}, 0.0, 1e-2))
            source = "default"
            # no measured targets -> nothing to publish, and an earlier slice's
            # stash must not survive into a run that has none
            if hasattr(mygs, "_weak_coil_reg"):
                del mygs._weak_coil_reg
        reg_terms = _terms(mygs, triples)
        # THE term list the reconstruction solves under (the engine
        # reconstruction records it on Bouquet._engine_run and every engine
        # draw installs exactly it, the sigma=0 draw included): the solver
        # terms, and the same terms as plain data for the records
        mygs._recon_coil_reg = reg_terms
        mygs._recon_coil_reg_record = dict(
            source=source,
            terms=[dict(coils={str(k): float(v) for k, v in c.items()},
                        target=float(tg), weight=float(w))
                   for c, tg, w in triples])
        mygs.set_coil_reg(reg_terms=reg_terms)
        return reg_terms

    def _check_coil_reg_orientation(self, spec):
        """Refuse coil targets oriented differently from the IMAS baseline.

        :func:`coil_targets.measured_from_pf_active` brings measured coil
        currents into the positive-Ip solve frame by the same factor the IMAS
        reader applies to the plasma currents, and
        :func:`coil_targets.coil_reg_from_measured` records that factor on each
        term as ``"source_current_sign"``.  Nothing here re-signs a target --
        the factor is applied exactly once, at the read.  But when the IMAS
        baseline was read with a DIFFERENT factor (an explicit
        ``ImasSource.current_orientation`` that the coil read did not share, or
        targets taken from another file), the coils would be pinned toward the
        mirror-image field at the configured weight.  That is refused.

        Checked only once an IMAS baseline exists (``prepare_baseline`` resets
        the solver, and re-applies the reg, immediately before the baseline
        solve); a term with no recorded factor claims nothing and is not
        checked.  The reconstruction path solves in the same positive frame but
        records no source factor (``source_current_sign`` is +1.0 there by
        definition), so it is not checked either.
        """
        bl = getattr(self, "baseline", None)
        if bl is None or getattr(bl, "provenance", None) != "imas":
            return
        want = float(getattr(bl, "source_current_sign", 1.0))
        bad = sorted({c for t in spec if t.get("source_current_sign") is not None
                      and float(t["source_current_sign"]) != want
                      for c in t["coils"]})
        if bad:
            got = sorted({float(t["source_current_sign"]) for t in spec
                          if t.get("source_current_sign") is not None
                          and float(t["source_current_sign"]) != want})
            raise ValueError(
                "coil_reg: coil target(s) %s were brought into the solve frame "
                "with orientation factor %s, but the IMAS baseline's currents "
                "were read with source_current_sign = %+.0f (%s). Pinning the "
                "coils with the other factor drives them toward the mirror-"
                "image vertical and shaping field. Rebuild the targets with "
                "measured_from_pf_active(..., current_orientation=%+.0f) so "
                "both use the same factor."
                % (", ".join(bad), ", ".join("%+.0f" % g for g in got), want,
                   getattr(bl, "source_current_sign_origin", None) or "?",
                   want))

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
        self._iso = (iso_pts, iso_w)      # re-applied by _swb_solve's reset
        if sc.F0 is None and getattr(self, "_F0", None) and \
                abs(F0_slice - self._F0) > 1e-3 * abs(self._F0):
            warnings.warn(
                f"IMAS slice F0={F0_slice:.4f} differs from the solver's "
                f"F0={self._F0:.4f} (set at setup). B_t is held fixed across "
                f"slices; a genuine B_t ramp needs a separate solver/process."
            )

    # ── stage 2: baseline (reconstruction OR imas) ----------------------
    @_baseline_negative_psep_named
    def prepare_baseline(self) -> "Baseline":
        """Resolve the baseline from ``config.source`` and cache it.

        Delegates to :func:`bouquet.baseline.resolve_baseline`, which dispatches
        on source type. Generation depends only on the returned
        :class:`~bouquet.baseline.Baseline`, never on reconstruction directly.
        """
        from .baseline import resolve_baseline
        from .config import ImasSource, resolve_solve_method
        resolve_solve_method(self.config.generation)

        # The engine-dependent settings (isolate_edge_jBS,
        # perturb_jind_in_anchor) are resolved HERE, once, for the engine
        # configured now -- whatever it was when the config was built -- and
        # the resolution goes on the baseline and into the archive.
        self._resolve_engine_defaults()

        # GenerationConfig.reconstruction_engine="unified": the ONE
        # reconstruction engine (bouquet.engine) builds the baseline for
        # either input type.  "legacy" never enters it, and everything below
        # is the legacy path, unchanged.
        if getattr(self.config.generation, "reconstruction_engine",
                   "legacy") == "unified":
            from .engine import prepare_engine_baseline
            _bl = prepare_engine_baseline(self)
            self._record_engine_resolved_defaults(_bl)
            self._record_coil_solve_mode(_bl)
            self._report_sigma_exceeds_profile(_bl)
            self._remember_baseline_state()
            return _bl

        # the self-consistent bootstrap loop runs in the baseline too, so its
        # workflow refusals fire here (before single_profile_jphi rewrites
        # recalculate_j_BS below), not only at generate()
        self._check_jbs_loop_workflow(self.config.generation)

        # the two edge-pressure settings do not reach the solver's own
        # bootstrap helper, which the non-loop routes call (it keeps the
        # pre-change P' and axis target): say so, once, whenever the
        # settings are not the pre-change ones -- the default included
        _edge = resolve_edge_pressure(self.config.generation)
        if not _edge.is_pre_change and not bool(getattr(
                self.config.generation, "jbs_self_consistent", False)):
            print("[edge-pressure] NOTE: edge_pprime_pin="
                  f"{_edge.edge_pprime_pin}, separatrix_pressure="
                  f"{_edge.separatrix_pressure!r} act on every solve bouquet "
                  "sets up; the solver's solve_with_bootstrap helper (used "
                  "by the legacy non-loop routes for their intermediate "
                  "bootstrap evaluation) builds its own P' and axis target "
                  "and is NOT affected", flush=True)

        # single_profile_jphi: drop the per-draw Sauter recompute BEFORE the
        # baseline work, so the IMAS forward solve does not spend a bootstrap
        # call either. The total j_phi is anchored to the source either way, so
        # the baseline equilibrium is unchanged -- only the (about to be
        # collapsed) split differs.
        if self.config.generation.single_profile_jphi:
            self.config.generation.recalculate_j_BS = False
        coords.check_run(self.config)
        self._check_structured_mse_reachable(self.config)
        if self.config.generation.imas_baseline == "swb":
            from .config import swb_config_problems
            _p = swb_config_problems(self.config)
            if _p:
                raise ValueError('imas_baseline="swb" refuses: ' + "; ".join(_p))
        elif self.config.generation.swb_saw_q is not None:
            raise ValueError('swb_saw_q needs imas_baseline="swb"')

        # A baseline is usable only once EVERY stage below has completed.  A
        # failure part-way (a JBSNotConverged or GS failure on pass k of the
        # IMAS loop, a closure refusal, a failed reconstruction) used to leave
        # self.baseline half-built -- bl.j_* and ip_closure from pass k,
        # l_i_target still the provisional value -- and generate() only
        # checks for None.  So: no baseline while building, none after a
        # failure (the half-built object is kept on _failed_baseline for
        # debugging only), and a previous slice's baseline never survives a
        # failed rebuild.
        self.baseline = None
        self._failed_baseline = None
        bl_new = resolve_baseline(self.config, self.mygs)
        self.baseline = bl_new
        try:
            # IMAS path: read_imas_baseline does no GS solve, so establish a
            # converged baseline equilibrium on mygs here (the reconstruction
            # path gets this for free from reconstruct_equilibrium). This also
            # sets l_i_target to the TokaMaker-solved li_1 and records
            # IDS-vs-TokaMaker li for sanity.
            if (isinstance(self.config.source, ImasSource)
                    and self.mygs is not None):
                # re-point the solver to THIS slice's boundary first, so a
                # multi-slice sweep treats each time as its own equilibrium
                self._repoint_imas_geometry()
                if self.config.generation.imas_baseline == "swb":
                    self._swb_imas_baseline()
                else:
                    self._forward_solve_imas_baseline()

            # single_profile_jphi: collapse the decomposition so the archive
            # matches what the draws actually perturb (the total). Done AFTER
            # the baseline solve so the equilibrium itself is unchanged --
            # only the bookkeeping split is folded back into j_inductive.
            if self.config.generation.single_profile_jphi:
                self._collapse_jphi_split()

            # Reconstruction path, jbs_loop_on_fail="flag": a baseline whose
            # loop did not converge is delivered, so it must carry the same
            # health flag the IMAS path sets (closure_limited + the reason).
            self._flag_nonconverged_recon_loop()
        except BaseException as _bl_exc:
            self._failed_baseline = bl_new
            self.baseline = None
            print(f"[baseline] FAILED ({type(_bl_exc).__name__}): no usable "
                  "baseline -- generate() / verify_sigma0_consistency() will "
                  "refuse until prepare_baseline() succeeds (the half-built "
                  "object is on Bouquet._failed_baseline for debugging only)",
                  flush=True)
            raise

        # Reconstruction path: surface a glanceable quality summary (the verbose
        # solver chatter was captured to baseline.reconstruction_log).
        if self.baseline.reconstruction_metrics is not None:
            self._print_reconstruction_summary()
        self._record_engine_resolved_defaults(self.baseline)
        self._record_coil_solve_mode(self.baseline)
        self._report_sigma_exceeds_profile(self.baseline)
        self._remember_baseline_state()
        return self.baseline

    def _resolve_engine_defaults(self) -> dict:
        """Resolve the engine-dependent settings of the configuration
        (:func:`bouquet.engine.resolve_engine_defaults`) in place, keep the
        record for the baseline (``Bouquet._engine_resolved_defaults``) and
        return it."""
        from .engine import resolve_engine_defaults
        rec = resolve_engine_defaults(self.config, stacklevel=4)
        self._engine_resolved_defaults = rec
        return rec

    def _ensure_engine_defaults_resolved(self) -> None:
        """Defensive: an engine-dependent setting still unset (``None``)
        here -- a baseline not built by :meth:`prepare_baseline`, or a field
        reset afterwards -- is resolved now, as :meth:`prepare_baseline`
        would have, never read as ``None``."""
        from .engine import ENGINE_DEPENDENT_DEFAULTS
        gc = self.config.generation
        if any(getattr(gc, n, None) is None for n in ENGINE_DEPENDENT_DEFAULTS):
            self._resolve_engine_defaults()
            self._record_engine_resolved_defaults(self.baseline)

    def _record_engine_resolved_defaults(self, bl) -> None:
        """Put the resolution record of the engine-dependent settings
        (:meth:`_resolve_engine_defaults`) on *bl*
        (``Baseline.engine_resolved_defaults``, archived as the
        ``_baseline`` attr ``engine_resolved_defaults_json``) and, under the
        unified engine, in its record."""
        rec = getattr(self, "_engine_resolved_defaults", None)
        if rec is None or not hasattr(bl, "engine_resolved_defaults"):
            return                    # no record, or not a Baseline
        bl.engine_resolved_defaults = {k: dict(v) for k, v in rec.items()}
        if isinstance(getattr(bl, "engine", None), dict):
            bl.engine["engine_resolved_defaults"] = {
                k: dict(v) for k, v in rec.items()}

    def _record_coil_solve_mode(self, bl) -> None:
        """Record the coil-solve mode the solver ran the reconstruction in --
        ``"bounded"`` (entered at :meth:`setup_solver`,
        :func:`bouquet.solver_state.enter_bounded_coil_mode`), else
        ``"unknown"`` -- on *bl* (``Baseline.coil_solve_mode``, both paths)
        and, under the unified engine, in its record (``Baseline.engine[
        "coil_solve_mode"]``, archived with it)."""
        from .solver_state import coil_solve_mode
        if bl is None or self.mygs is None:
            return
        bl.coil_solve_mode = coil_solve_mode(self.mygs)
        if isinstance(getattr(bl, "engine", None), dict):
            bl.engine["coil_solve_mode"] = bl.coil_solve_mode

    def _remember_baseline_state(self) -> None:
        """Fingerprint the solver state :meth:`prepare_baseline` leaves (a
        copy of ``mygs.get_psi(False)``), so :meth:`save_baseline_eqdsk` can
        refuse once a later solve has moved it.  ``None`` when it cannot be
        read."""
        import numpy as np
        self._baseline_psi = None
        try:
            if self.mygs is not None:
                self._baseline_psi = np.array(self.mygs.get_psi(False),
                                              dtype=float, copy=True)
        except Exception:
            self._baseline_psi = None

    def save_baseline_eqdsk(self, filename, *, nr=257, nz=257,
                            truncate_eq=True, lcfs_pad=None):
        """Write the RECONSTRUCTION's own equilibrium -- the live solver
        state :meth:`prepare_baseline` left -- as a g-file carrying the FULL
        pressure, as every g-file bouquet writes does.

        ``PRES`` is the solver's pressure (zero at ``psi_N = 1``) plus this
        baseline's own separatrix pressure, ``Baseline.edge_pressure[
        "p_sep_applied"]`` (:func:`bouquet.edge_pressure.delivered_p_sep`:
        ``p_sep`` under ``separatrix_pressure="offset"``, 0 under
        ``"legacy"``, where the call is exactly a bare ``save_eqdsk``);
        ``PPRIME`` is unchanged.  The grid, padding (``lcfs_pad`` defaults
        to ``source.psi_pad``) and truncation are those of the archive's
        ``_baseline`` g-file written by :meth:`generate`, so the two carry
        the same pressure frame.  A bare ``mygs.save_eqdsk`` writes the
        solver frame and does NOT.

        Refuses without a baseline, and once the solver no longer holds the
        state :meth:`prepare_baseline` left (:meth:`generate` or any later
        solve moves it; the archive's ``_baseline`` g-file is then the
        delivered one).  Returns *filename*."""
        import numpy as np
        from .edge_pressure import delivered_p_sep, save_full_pressure_eqdsk
        if self.baseline is None or self.mygs is None:
            raise RuntimeError("save_baseline_eqdsk: no baseline on a live "
                               "solver; call prepare_baseline() first")
        p_sep = delivered_p_sep(getattr(self.baseline, "edge_pressure", None))
        snap = getattr(self, "_baseline_psi", None)
        try:
            now = np.asarray(self.mygs.get_psi(False), dtype=float)
        except Exception:
            now = None
        if (snap is None or now is None or now.shape != snap.shape
                or not np.array_equal(now, snap)):
            raise RuntimeError(
                "save_baseline_eqdsk: the solver no longer holds the state "
                "prepare_baseline() left (a later solve -- generate(), a "
                "sigma0 check -- moved it); call prepare_baseline() again, "
                "or use the archive's _baseline g-file")
        if lcfs_pad is None:
            lcfs_pad = float(getattr(self.config.source, "psi_pad", 1e-3))
        save_full_pressure_eqdsk(self.mygs, filename, p_sep, nr=nr, nz=nz,
                                 truncate_eq=truncate_eq, lcfs_pad=lcfs_pad)
        return filename

    def _report_sigma_exceeds_profile(self, bl) -> None:
        """One line at baseline time when an input kinetic sigma exceeds the
        profile it perturbs over a stated fraction of the radius
        (:func:`bouquet.baseline.sigma_exceeds_profile`).  REPORT ONLY: the
        envelope is resolved as :meth:`generate` will resolve it (quietly;
        its own log lines are printed there), nothing is stored, clipped or
        changed, and a resolution that cannot be made yet is not an error
        here."""
        import contextlib
        import io
        import warnings
        self.sigma_exceeds_profile = []
        try:
            from .baseline import (resolve_uncertainty,
                                   sigma_exceeds_profile_line)
            with contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = resolve_uncertainty(self.config, bl)
            recs = list(env.get("sigma_exceeds_profile") or [])
        except Exception:
            return
        self.sigma_exceeds_profile = recs
        if recs:
            print("[baseline] " + sigma_exceeds_profile_line(recs),
                  flush=True)

    def _flag_nonconverged_recon_loop(self) -> None:
        """Mark a geqdsk baseline whose self-consistent j_BS loop did not
        converge (only reachable with ``jbs_loop_on_fail="flag"``; "raise"
        raised inside the reconstruction) as ``closure_limited``.

        The IMAS path records the same thing on ``ip_closure`` /
        ``li_metrics`` (``run.py`` ``_finish``); on the reconstruction path
        the record was only inside ``reconstruction_metrics["jbs_loop"]``, so
        a driver excluding closure-limited slices could not see it.  Sets
        ``reconstruction_metrics["jbs_converged"] = False``,
        ``["closure_limited"] = True`` and appends the loop's flag reason to
        ``["closure_limited_reasons"]``; prints and warns.  A flag, never a
        retry: nothing is re-solved and no bar moves.
        """
        import warnings
        from .jbs_loop import flag_reason
        bl = self.baseline
        m = getattr(bl, "reconstruction_metrics", None)
        if not m:
            return
        rec = m.get("jbs_loop")
        if rec is None or bool(rec.get("converged", False)):
            return
        m = dict(m)
        m["jbs_converged"] = False
        reasons = list(m.get("closure_limited_reasons", ()) or ())
        why = flag_reason(rec)
        if why not in reasons:
            reasons.append(why)
        m["closure_limited_reasons"] = tuple(reasons)
        m["closure_limited"] = True
        msg = ("reconstruction baseline: the self-consistent j_BS loop "
               "did NOT converge (jbs_loop_on_fail='flag') -- the slice "
               "is delivered flagged closure_limited: " + why)
        print("[recon jbs-loop] WARNING " + msg, flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        bl.reconstruction_metrics = m

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
            for name in ("j_BS", "j_NBI", "j_RF", "j_other")
            if getattr(bl, name, None) is not None
        }
        bl.j_inductive = j_phi.copy()
        bl.j_BS = np.zeros_like(j_phi)
        for name in ("j_NBI", "j_RF", "j_other", "j_sawteeth"):
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
        if m.get("closure_limited"):
            print(f"  {'closure':<12} LIMITED ⚠ -- "
                  + "; ".join(str(r) for r in
                              m.get("closure_limited_reasons", ())))
        line("Ip", m['Ip_MA'], m['Ip_efit_MA'], m['Ip_err_pct'], "MA")
        # l_i on the targeted estimator (matched pair -- ~0 by construction),
        # then the free cross-estimator pair which is NOT driven by anything
        # and so is the honest estimator-drift monitor (issue #20).
        line("l_i(3)", m['li'], m['li_efit'], m['li_err_pct'])
        if np.isfinite(m.get('li1_cross_err_pct', float('nan'))):
            line("  l_i(1)x", m['li1_cross'], m['li1_cross_efit'],
                 m['li1_cross_err_pct'])
        # q0 at LIKE radii (both at psi_N = q0_psi_N, the solver's first
        # traced surface); the g-file's axis value is printed beside it
        line(f"q0@{m.get('q0_psi_N', 0.02):g}", m['q0'], m['q0_efit'],
             m['q0_err_pct'], fmt=".2f")
        if np.isfinite(m.get('q0_efit_axis', float('nan'))):
            print(f"  {'':<12} {'':<13} (g-file axis q(0) "
                  f"{m['q0_efit_axis']:.2f}, {m['q0_err_pct_vs_axis']:+.2f}% "
                  "-- a different radius)")
        line("q95", m['q95'], m['q95_efit'], m['q95_err_pct'], fmt=".2f")
        line("beta_N", m['beta_n'], m['beta_n_efit'], m['beta_n_err_pct'], fmt=".2f")
        line("beta_p", m['beta_p'], m['beta_p_efit'], m['beta_p_err_pct'], fmt=".2f")
        line("kappa", m['kappa'], m['kappa_efit'], m['kappa_err_pct'])
        line("delta", m['delta'], m['delta_efit'], m['delta_err_pct'])
        line("j_sep(.99)", m['j_sep_MA'], m['j_sep_efit_MA'], m['j_sep_err_pct'], "MA/m²")
        line("W_MHD", m['W_MHD_MJ'], m['W_MHD_efit_MJ'], m['W_MHD_err_pct'], "MJ")
        # the separatrix pressure: both frames, each against the input's
        # same-definition quantity (printed only when one is non-zero)
        _ep = m.get("edge_pressure") or {}
        _lk = m.get("pressure_like_for_like") or {}
        if _lk and (_ep.get("p_sep") or _lk.get("p_edge_input")):
            print(f"  {'p_sep':<12} {_ep.get('p_sep', float('nan')):.4g} Pa "
                  f"(input edge {_lk.get('p_edge_input', float('nan')):.4g} "
                  f"Pa; separatrix_pressure="
                  f"{_ep.get('separatrix_pressure')!r}, added back "
                  f"{_ep.get('p_sep_applied', 0.0):.4g} Pa; edge_pprime_pin="
                  f"{_ep.get('edge_pprime_pin')})")
            for _k, _nm in (("solver", "p - p_sep"), ("full", "full p")):
                _f = _lk["frames"][_k]
                print(f"  {'  ' + _nm:<12} beta_N {_f['beta_n']['tokamaker']:.3f} "
                      f"(input {_f['beta_n']['input']:.3f}, "
                      f"{_f['beta_n']['err_pct']:+.2f}%)   W_MHD "
                      f"{_f['W_MHD_MJ']['tokamaker']:.4f} MJ (input "
                      f"{_f['W_MHD_MJ']['input']:.4f}, "
                      f"{_f['W_MHD_MJ']['err_pct']:+.2f}%)")
        print(f"  {'boundary':<12} RMS {m['boundary_rms_mm']:.2f} mm   "
              f"max {m['boundary_max_mm']:.2f} mm   axis off {m['axis_offset_mm']:.2f} mm")
        print(f"  {'jphi resid':<12} core RMS {m['jphi_core_rms_MA']:.3f}   "
              f"edge RMS {m['jphi_edge_rms_MA']:.3f} MA/m²")
        if m.get("ind_scale_fallback"):
            _r = (m.get("ind_scale_fallback_records") or [{}])[-1]
            print(f"  {'ind. amp.':<12} FALLBACK 1.0 ⚠ in "
                  f"{m.get('ind_scale_fallback_n')} fit(s): "
                  f"{_r.get('reason', '?')}; bracket {_r.get('bracket')} "
                  f"residuals {_r.get('residual_at_bracket')}")
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
                               close_ip, q0_ref=None):
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

        **Self-consistent bootstrap loop** (``jbs_self_consistent=True``): the
        predictor is re-run every pass on that pass's geometry.  ``q0_ref``
        (a dict with ``q0_target``, ``q0_anchor``, ``j_achieved0``,
        ``j_requested0``) then carries the reference computed ONCE on the
        original anchor -- a data-derived target, held fixed, and so is the
        axis-current row derived from it.  ``None`` (the default) is the
        historical behaviour, line for line.  With
        ``jbs_loop_q0_corrector=True`` the dict also carries ``axis_row``: the
        row the loop's q0 pin (:class:`bouquet.jbs_loop.AxisRowPin`) moved from
        the q0 measured on the previous pass; the 2x2 then matches that row
        instead of ``j_requested0`` (the target itself is unchanged).
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
        if q0_ref is None:
            q0_anchor = float(np.asarray(eq_snap.get_q(psi=psi_q.copy())[1],
                                         dtype=float)[0])
            # ACHIEVED: the anchor's GS-reconstructed own profile (round-trips
            # to its achieved Ip).  REQUESTED: the source total that was
            # handed in.  Their ratio IS TokaMaker's Ip renormalisation of the
            # anchor.
            j_achieved0 = float(np.asarray(probe, dtype=float)[0])
            j_requested0 = _ax(FUSE_tot)
            # The algebra lives in utils.unrenormalise_q0 so the tests
            # exercise the SHIPPED formula, not a re-derivation (same rule as
            # close_ip).
            q0_target = unrenormalise_q0(q0_anchor, j_achieved0, j_requested0)
        else:
            # held reference from the ORIGINAL anchor (j_BS loop pass)
            q0_anchor = float(q0_ref["q0_anchor"])
            j_achieved0 = float(q0_ref["j_achieved0"])
            j_requested0 = float(q0_ref["j_requested0"])
            q0_target = float(q0_ref["q0_target"])
        j_renorm_ratio = j_achieved0 / j_requested0
        j_ref0 = j_requested0
        # jbs_loop_q0_corrector=True (loop only): the axis row the loop's
        # q0 pin moved from the q0 measured on the previous pass
        # (jbs_loop.AxisRowPin); absent by default -> the held row above
        _pin_row = (None if q0_ref is None else q0_ref.get("axis_row"))
        if _pin_row is not None:
            j_ref0 = float(_pin_row)
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
            **({} if _pin_row is None else dict(
                q0_axis_row_source=(
                    "jbs_loop_q0_corrector: the axis row moved by the loop's "
                    "q0 pin from the q0 measured on the previous pass (the "
                    "anchor's requested axis current is j_ref0_requested)"))),
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
                                       psi_pad=1e-3, pprime_sign=1.0,
                                       q0_ref=None, x_retry=None):
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

        ``q0_ref``: the self-consistent bootstrap loop's held anchor
        reference, exactly as in :meth:`_close_ip_q0_predictor` (including
        the optional ``axis_row`` of ``jbs_loop_q0_corrector=True``); ``None``
        is the historical behaviour.

        ``x_retry`` (loop passes only): the previous pass's coefficients.  If
        the SOFT closure refuses with the Levenberg no-descent error, it is
        retried ONCE from them (:func:`bouquet.utils.soft_closure_with_retry`,
        recorded as ``structured_closure_retry``); a second refusal is a real
        refusal.  ``None`` (every non-loop caller) never retries.
        """
        import numpy as np

        from .config import resolve_structured_preset
        from .utils import (close_ip_structured, close_ip_structured_soft,
                            li_closure_geometry, sigma_from_weights,
                            soft_closure_with_retry,
                            structured_basis_eval, unrenormalise_q0,
                            q0_gate_admits)

        # Resolve the preset HERE as well as at construction: a config whose
        # closure_channel was set after the GenerationConfig was built has not
        # been through __post_init__ with the channel visible, and would
        # otherwise run on the raw shipped fields -- the configuration this
        # channel's own study superseded.  Idempotent: for the ordinary case
        # (channel set at construction) it fills nothing and warns not at all.
        preset_rec = resolve_structured_preset(gc)
        if preset_rec["source"] == "default":
            print("[imas SWB-split:ohmic structured] prior: preset "
                  f"{preset_rec['name']!r} applied BY DEFAULT (no "
                  "structured_preset given)"
                  + (" -- filled " + ", ".join(preset_rec["fields"])
                     if preset_rec["fields"] else "")
                  + " | its sigmas are relative-unit PRIORS from a study on "
                    "one device with one integrated-modelling source for the "
                    "inductive current, not device constants: read the "
                    "closure-health flags below before trusting them, and "
                    "pass structured_preset='none' to decline the default",
                  flush=True)
        elif preset_rec["source"] == "default-declined-custom-basis":
            print("[imas SWB-split:ohmic structured] prior: the default "
                  "preset was DECLINED (an explicit structured_basis is set "
                  "and the preset's ladders are widths at the shipped basis's "
                  "radii); every structured field keeps its own default",
                  flush=True)

        psi_q = np.ascontiguousarray(np.asarray(geom["psi_q"], dtype=float))
        psi_geom = np.asarray(geom["psi_N"], dtype=float)
        _ax = lambda j: float(np.interp(psi_q[0], psi_geom,
                                        np.asarray(j, dtype=float)))
        if q0_ref is None:
            q0_anchor = float(np.asarray(eq_snap.get_q(psi=psi_q.copy())[1],
                                         dtype=float)[0])
            j_achieved0 = float(np.asarray(probe, dtype=float)[0])
            j_requested0 = _ax(FUSE_tot)
            q0_target = unrenormalise_q0(q0_anchor, j_achieved0, j_requested0)
        else:
            # held reference from the ORIGINAL anchor (j_BS loop pass)
            q0_anchor = float(q0_ref["q0_anchor"])
            j_achieved0 = float(q0_ref["j_achieved0"])
            j_requested0 = float(q0_ref["j_requested0"])
            q0_target = float(q0_ref["q0_target"])
        j_ref0 = j_requested0
        # jbs_loop_q0_corrector=True (loop only): the axis row moved by the
        # loop's q0 pin (see _close_ip_q0_predictor); absent by default
        _pin_row = (None if q0_ref is None else q0_ref.get("axis_row"))
        if _pin_row is not None:
            j_ref0 = float(_pin_row)
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
        # ---- the third measurement: MSE pitch angles (optional) -----------
        # Validated HERE, before anything is solved, so a required block that
        # is unusable costs nothing.  None -> the channel exactly as before.
        mse_ch, mse_status = Bouquet._structured_mse_block(gc)
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
            out = soft_closure_with_retry(
                lambda _x0: close_ip_structured_soft(
                    psi_geom, w_lin, c_signed, sgn * Ip_t,
                    (None if ip_sigma is None else float(ip_sigma)),
                    j_ind, j_BS_swb, j_fixed,
                    basis=basis_spec, sigma_ind=_sig["ind"],
                    sigma_bs=_sig["bs"],
                    sigma_ind_up=sig_ind_up,
                    li_target=li_target,
                    li_sigma=(None if li_sigma is None else float(li_sigma)),
                    li_kind=li_kind, li_geom=li_geom,
                    axis=axis, axis_sigma=None,   # the q0 pin stays HARD
                    x0=_x0,
                    # the noise-floor acceptance belongs to the self-consistent
                    # bootstrap loop only; the legacy path keeps the strict
                    # stop test (it raises where it always raised)
                    accept_noise_floor=bool(
                        getattr(gc, "jbs_self_consistent", False))),
                x_prev=x_retry, who="imas SWB-split:ohmic structured")
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
            # WHICH prior configuration was in force, and WHO chose it: a
            # preset applied by default and one named by the caller are the
            # same numbers and not the same statement, and only the record can
            # tell them apart afterwards.
            structured_preset=(preset_rec["name"] or "none"),
            structured_preset_source=preset_rec["source"],
            structured_preset_filled=(", ".join(preset_rec["fields"])
                                      or "none"),
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
            **({} if _pin_row is None else dict(
                q0_axis_row_source=(
                    "jbs_loop_q0_corrector: the axis row moved by the loop's "
                    "q0 pin from the q0 measured on the previous pass (the "
                    "anchor's requested axis current is j_ref0_requested)"))),
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
            # how the soft solve stopped (noise_floor = accepted at the
            # objective's rounding-noise floor, see close_ip_structured_soft)
            # and whether the loop's single logged retry was needed
            structured_gn_stop_reason=out.get("gn_stop_reason"),
            structured_gn_stop=out.get("gn_stop"),
            structured_n_noise_floor_accepts=out.get("n_noise_floor_accepts"),
            structured_closure_retry=out.get("closure_retry"),
            structured_closure_retry_first_error=out.get(
                "closure_retry_first_error"),
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

        if mse_status is not None:
            extra.update(Bouquet._structured_mse_predictor_record(
                gc, mse_ch, mse_status, bl=bl))

        # The corrector state is built whenever there is ANYTHING to correct:
        # the q0 row (gate admitted) or the l_i row.  Both corrections share
        # the SAME single extra solve.  A usable MSE block needs the state too
        # (the MSE stage runs between the predictor solve and the corrector).
        state = None
        if gated or li_target is not None or mse_ch is not None:
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
            if mse_ch is not None:
                from .utils import structured_objective_no_mse

                def _sig(W):
                    W = np.asarray(W, dtype=float)
                    with np.errstate(divide="ignore"):
                        return [float(v) for v in 1.0 / np.sqrt(W)]
                # With MSE on the trust weights are an ABSOLUTE sigma^-2 that
                # trades against chi2_MSE (their scale is no longer free), so
                # the ladder actually in force is recorded as widths.
                extra.update(
                    structured_mse_prior_sigma_ind=_sig(out["weights_ind"]),
                    structured_mse_prior_sigma_bs=_sig(out["weights_bs"]),
                    structured_mse_prior_sigma_ind_up=(
                        None if out.get("weights_ind_up") is None
                        else _sig(out["weights_ind_up"])),
                    structured_mse_prior_weights_name=str(
                        out.get("weights_name", "")),
                    structured_mse_prior_scale=(
                        "ABSOLUTE: with MSE the objective is x'Wx + chi2_MSE, "
                        "so W = sigma^-2 in peak-normalised coefficient units "
                        "trades against the chords; scaling W moves the "
                        "answer, and a uniform ladder is a sigma = 1 prior, "
                        "not 'no prior'"))
                state.update(
                    mse=mse_ch,
                    mse_required=bool(getattr(gc, "structured_mse_required",
                                              False)),
                    mse_fd_step=float(getattr(gc, "structured_mse_fd_step",
                                              0.02)),
                    mse_steps=int(getattr(gc, "structured_mse_steps", 1)),
                    x_pred=np.concatenate([np.asarray(out["a"], dtype=float),
                                           np.asarray(out["b"], dtype=float)]),
                    F_pred=float(structured_objective_no_mse(out)),
                    free=np.isfinite(np.concatenate([
                        np.asarray(out["weights_ind"], dtype=float),
                        np.asarray(out["weights_bs"], dtype=float)])),
                    mse_lin=None,
                )
        return (s_ind, s_bs, float(out["ohm_scale_eff"]),
                float(out["bs_scale_eff"]), extra, state)

    # ── closure_channel="structured": MSE pitch angles ─────────────────────
    @staticmethod
    def _check_structured_mse_reachable(cfg):
        """Refuse MSE settings on a path that never reads them.

        The MSE term exists only in the structured closure, which runs only on
        the IMAS path with ``recalculate_j_BS``, ``jBS_baseline_mode="ohmic"``
        and ``closure_channel="structured"``.  Anywhere else ``mse_data`` (or
        any non-default ``structured_mse_*`` knob) would be accepted and never
        used -- the class of silent no-op the ``closure_channel`` guard
        refuses, and refused here the same way:

        * ``structured_mse_required=True`` -- ALWAYS raises: a required
          constraint that cannot be applied is not something a workflow
          opt-out can waive.
        * otherwise -- raises ``ValueError``; ``workflow='custom'`` (or the
          deprecated ``allow_unsafe_workflow=True``) downgrades it to a printed
          WARN, exactly as for every other workflow-guard problem, and the MSE
          term is then NOT applied.
        """
        from dataclasses import MISSING

        from .config import GenerationConfig, ImasSource
        gc = cfg.generation
        _fields = GenerationConfig.__dataclass_fields__
        set_knobs = []
        for name in ("mse_data", "structured_mse_required",
                     "structured_mse_fd_step", "structured_mse_steps",
                     "structured_mse_sigma_sys", "structured_mse_min_chords"):
            dflt = _fields[name].default
            if dflt is MISSING or not hasattr(gc, name):
                continue
            val = getattr(gc, name)
            if (val is not None if dflt is None else val != dflt):
                set_knobs.append(name)
        if not set_knobs:
            return
        why = []
        if not isinstance(cfg.source, ImasSource):
            why.append("the source is not an IMAS source")
        if not bool(getattr(gc, "recalculate_j_BS", False)):
            why.append("recalculate_j_BS is off"
                       + (" (forced off by single_profile_jphi=True)"
                          if bool(getattr(gc, "single_profile_jphi", False))
                          else ""))
        if str(getattr(gc, "jBS_baseline_mode", "")) != "ohmic":
            why.append(f"jBS_baseline_mode={gc.jBS_baseline_mode!r} "
                       "(needs 'ohmic')")
        if str(getattr(gc, "closure_channel", "")) != "structured":
            why.append(f"closure_channel={gc.closure_channel!r} "
                       "(needs 'structured')")
        if not why:
            return
        if bool(getattr(gc, "structured_mse_required", False)):
            raise ValueError(
                "structured_mse_required=True, but the structured closure "
                "that consumes mse_data will not run: " + "; ".join(why)
                + ".  Refusing rather than ignoring a required constraint.")
        msg = (", ".join(set_knobs) + " set, but the structured closure that "
               "reads mse_data will not run: " + "; ".join(why)
               + " -- it would otherwise be silently ignored.  Use "
               "closure_channel='structured' with jBS_baseline_mode='ohmic' "
               "on an IMAS source, or leave mse_data / structured_mse_* at "
               "their defaults.")
        if (str(getattr(gc, "workflow", "")) == "custom"
                or bool(getattr(gc, "allow_unsafe_workflow", False))):
            print("WARN: " + msg + " (workflow='custom': continuing; the MSE "
                  "term is NOT applied)", flush=True)
            return
        raise ValueError(msg)

    @staticmethod
    def _structured_mse_block(gc):
        """``(chords, status)`` for the structured channel's MSE term.

        ``(None, None)`` when no ``mse_data`` was given and none is required --
        the channel exactly as it was.  ``(chords, "usable")`` for a block
        :func:`bouquet.mse.mse_chords` accepts.  An absent or unusable block
        with ``structured_mse_required`` RAISES
        :class:`~bouquet.mse.MSEDataUnusable`; an unusable block without it is
        WARNED and returned as ``(None, "unusable ...")`` so the record says
        the term was not applied and why.
        """
        import warnings
        from .mse import MSE_ER_BIAS_NOTE as _mse_er_bias_note
        from .mse import MSE_MIN_CHORDS, MSEDataUnusable, mse_chords

        md = getattr(gc, "mse_data", None)
        required = bool(getattr(gc, "structured_mse_required", False))
        if md is None and not required:
            return None, None
        # configuration errors (a bad knob, an unknown key) are REFUSED here
        # as plain ValueErrors -- never turned into a per-slice "not applied"
        from .config import validate_structured_mse_settings
        validate_structured_mse_settings(gc)
        try:
            ch = mse_chords(
                md, min_chords=int(getattr(gc, "structured_mse_min_chords",
                                           MSE_MIN_CHORDS)),
                sigma_sys=float(getattr(gc, "structured_mse_sigma_sys", 0.0)))
        except MSEDataUnusable as e:
            if required:
                raise MSEDataUnusable(
                    "closure_channel='structured' with "
                    f"structured_mse_required=True: {e}.  Refusing to run the "
                    "closure without the MSE constraint it was asked for.") \
                    from e
            msg = (f"closure_channel='structured': mse_data is UNUSABLE ({e}); "
                   "the MSE term is NOT applied on this slice "
                   "(structured_mse_required=False)")
            warnings.warn(msg, stacklevel=3)
            print("[imas SWB-split:ohmic structured] WARNING " + msg,
                  flush=True)
            return None, f"unusable, not applied: {e}"
        if not ch["er_applied"] and not ch["er_corrected"]:
            print("[imas SWB-split:ohmic structured] WARNING MSE: E_r is "
                  "neither supplied (Er) nor declared corrected "
                  "(er_corrected=True), so the forward model takes E_R = 0 -- "
                  "BIASED in a rotating plasma (" + _mse_er_bias_note + ")",
                  flush=True)
        _er_nan = [i for i, r in ch["excluded"] if r.startswith("E_r is not")]
        if _er_nan:
            print("[imas SWB-split:ohmic structured] WARNING MSE: "
                  f"{len(_er_nan)} chord(s) at input index {_er_nan} have a "
                  "non-finite E_r inside the supplied E_r profile and are "
                  "EXCLUDED (recorded with that reason), not fitted without "
                  "their E_r term", flush=True)
        return ch, "usable"

    @staticmethod
    def _structured_mse_predictor_record(gc, ch, status, bl=None):
        """The predictor-stage MSE block of ``ip_closure`` (inputs only)."""
        from .mse import mse_er_terms
        rec = dict(
            structured_mse=bool(ch is not None),
            structured_mse_required=bool(getattr(gc, "structured_mse_required",
                                                 False)),
            structured_mse_status=str(status),
        )
        if ch is None:
            return rec
        rec.update(Bouquet._structured_mse_chord_record(ch, bl))
        rec.update(
            structured_mse_sigma_sys=float(ch["sigma_sys"]),
            structured_mse_er_applied=bool(ch["er_applied"]),
            structured_mse_er_corrected=bool(ch["er_corrected"]),
            structured_mse_er_terms=mse_er_terms(ch),
            structured_mse_er_neglected=bool(not ch["er_applied"]
                                             and not ch["er_corrected"]),
            structured_mse_fd_step=float(getattr(gc, "structured_mse_fd_step",
                                                 0.02)),
            structured_mse_steps=int(getattr(gc, "structured_mse_steps", 1)),
            structured_mse_forward_model=(
                "tan(gamma) = (A1 Bz + A5 Er) / (A2 Bphi + A3 BR + A4 Bz) on "
                "the SOLVED equilibrium (bouquet.mse): the standard form "
                "(A1 Bz + A5 Er) / (A2 Bphi + A3 BR + A4 Bz + A6 Ez + A7 Er) "
                "with Ez = 0 and A7 = 0 (a block applying Er with non-zero "
                "A7 is refused); linearised in the structured coefficients "
                "by forward differences"),
        )
        return rec

    #: Where the per-chord MSE arrays live (named in ``ip_closure``).
    _MSE_PER_CHORD_WHERE = (
        "Baseline.mse_record (per-chord arrays and the Jacobian; archived as "
        "datasets under _baseline/structured_mse, not in the ip_closure "
        "attribute)")

    @staticmethod
    def _mse_record_put(bl, **arrays):
        """Store per-chord MSE arrays on ``bl.mse_record`` (O(n_chords) data).

        ``ip_closure`` is archived as ONE JSON attribute, whose size an HDF5
        file caps at 64 kB; the MSE term's per-chord blocks and its
        ``n_chords x 2K`` Jacobian grow with the chord count, so they are kept
        OUT of it -- here, and archived as datasets (see
        :func:`bouquet.utils.store_baseline_profiles`) -- while ``ip_closure``
        carries only chord-count-independent scalars and summaries.
        """
        import numpy as np
        if bl is None:
            return
        rec = getattr(bl, "mse_record", None)
        if rec is None:
            rec = {}
            bl.mse_record = rec
        for k, v in arrays.items():
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
                rec[k] = [str(x) for x in v]
            elif isinstance(v, (list, tuple)) and not v:
                rec[k] = np.zeros(0)
            else:
                rec[k] = np.asarray(v)

    @staticmethod
    def _structured_mse_chord_record(ch, bl=None):
        """Which chords the MSE term uses, and every excluded one with why.

        Returns the ``ip_closure`` summary (counts; exclusions tallied by
        reason) and puts the per-chord arrays on ``bl.mse_record``.
        """
        import numpy as np
        ex = list(ch.get("excluded", ()))
        by_reason = {}
        for _i, r in ex:
            by_reason[str(r)] = by_reason.get(str(r), 0) + 1
        Bouquet._mse_record_put(
            bl,
            chord_index=np.asarray(ch["index"], dtype=np.int64),
            chord_R=np.asarray(ch["R"], dtype=float),
            chord_Z=np.asarray(ch["Z"], dtype=float),
            tgamma_meas=np.asarray(ch["tgamma"], dtype=float),
            sigma_eff=np.asarray(ch["sigma_eff"], dtype=float),
            excluded_index=np.asarray([int(i) for i, _r in ex],
                                      dtype=np.int64),
            excluded_reason=[str(r) for _i, r in ex])
        return dict(
            structured_mse_n_chords=int(ch["n_active"]),
            structured_mse_n_chords_total=int(ch["n_total"]),
            structured_mse_n_excluded=len(ex),
            structured_mse_excluded_by_reason=by_reason,
            structured_mse_per_chord=Bouquet._MSE_PER_CHORD_WHERE,
        )

    @staticmethod
    def _structured_predictor_readback(state, mygs):
        """The predictor's solved q0 / l_i, under the corrector's own names.

        Exactly the corrector's readbacks (q0 from ``get_q`` at ``psi_q[0]``;
        l_i from :func:`bouquet.utils.li_achieved` with the anchor's
        perimeter), taken on a ``copy_eq()`` snapshot; only the quantities the
        slice constrains are read.  No GS solve.
        """
        import numpy as np

        from .utils import li_achieved

        gated = bool(state.get("gated", state.get("axis") is not None))
        li_target = state.get("li_target")
        if not gated and li_target is None:
            return {}
        snap = mygs.copy_eq()
        rec = {}
        if gated:
            q0 = float(np.asarray(snap.get_q(psi=state["psi_q"].copy())[1],
                                  dtype=float)[0])
            rec.update(q0_solved_predictor=q0,
                       q0_predictor_residual=q0 - state["q0_target"])
        if li_target is not None:
            _per = (None if state.get("li_geom") is None
                    else float(state["li_geom"]["perimeter"]))
            li, _ = li_achieved(snap, li_kind=str(state.get("li_kind",
                                                              "li_1")),
                                psi_pad=float(state.get("psi_pad", 1e-3)),
                                perimeter=_per)
            rec.update(structured_li_solved_predictor=float(li),
                       structured_li_achieved_predictor=float(li),
                       structured_li_residual_predictor=float(li)
                       - float(li_target))
        return rec

    @staticmethod
    def _close_ip_structured_mse_stage(state, bl, mygs, solve_jphi,
                                       field_at=None):
        """Add ``chi2_MSE`` to the structured closure; ``free.sum() + steps`` solves.

        Runs after the common tail has SOLVED the predictor's hybrid (so
        ``mygs`` holds the predictor equilibrium) and before
        :meth:`_close_ip_structured_corrector`.

        1. Read the field at the chords off the predictor equilibrium, map it
           onto the discharge's STATED orientation
           (:func:`bouquet.mse.mse_orientation`: the block's
           ``ip_sign``/``bt_sign`` against the equilibrium's own directions,
           read off its field about ``mygs.o_point``; frozen for the rest of
           the slice, never chosen by fit) and form ``tan_gamma(x_pred)``.
           All four orientations are still evaluated
           (:func:`bouquet.mse.mse_orientation_check`); when another fits the
           chords better than the stated one by more than
           ``bouquet.mse.MSE_ORIENTATION_DCHI2`` the slice is FLAGGED and the
           stated orientation is kept.
        2. :func:`bouquet.utils.structured_mse_outer`: forward-difference
           Jacobian (one solve per free coefficient), then
           ``structured_mse_steps`` re-solve(s) of the SAME closure (same
           prior, Ip/axis/l_i rows) with the linearised chi^2 added, each
           followed by a solve of the new hybrid.  The last solve is the
           delivered equilibrium.
        3. Refresh ``bl`` and ``ip_closure`` from the delivered closure, and
           store the linear model re-centred on it in ``state["mse_lin"]`` so
           every corrector re-solve keeps the MSE term.

        A refusal anywhere (a closure out of its scale bounds, a failed solve,
        an unusable field) RAISES when ``structured_mse_required``; otherwise
        the predictor's hybrid is re-solved (the FD probes moved ``mygs``),
        kept, and the slice is FLAGGED closure-limited.  ``field_at(R, Z)``
        (tests) replaces the live field read and returns ``(B, found)`` like
        :func:`bouquet.mse.mse_field_at`.  Returns the last solve's ``nl_its``.

        **Chords off the solver mesh.**  The first field read (on the
        predictor equilibrium) reports, per chord, whether the interpolator
        could place it on the mesh at all.  A chord it cannot is EXCLUDED --
        recorded with its reason under ``structured_mse_excluded_chords`` and
        announced -- never evaluated from a stale buffer; if that leaves fewer
        than ``structured_mse_min_chords`` the stage refuses (loudly, by the
        path above).  The mesh does not move between solves, so every later
        read must find every remaining chord: one that does not is a refusal,
        not a reused value.
        """
        import numpy as np

        from .mse import (MSE_ORIENTATION_DCHI2, MSE_REASON_OFF_MESH,
                          MSEDataUnusable, mse_equilibrium_orientation,
                          mse_er_terms, mse_exclude, mse_field_at,
                          mse_orientation, mse_orientation_check,
                          mse_tan_gamma)
        from .utils import (MSE_FLAG_PREFIX, close_ip_structured,
                            close_ip_structured_soft, closure_health,
                            structured_basis_eval, structured_mse_linear_model,
                            structured_mse_outer)

        ch = state["mse"]
        required = bool(state.get("mse_required", False))
        soft = bool(state.get("soft"))
        if field_at is None:
            field_at = lambda R, Z: mse_field_at(mygs, R, Z)
        psi_g = np.asarray(state["psi_geom"], dtype=float)
        Phi = structured_basis_eval(state["basis"], psi_g)
        K = Phi.shape[0]
        j_ind = np.asarray(state["j_ind"], dtype=float)
        j_bs = np.asarray(state["j_BS_swb"], dtype=float)
        j_fix = np.asarray(state["j_fixed"], dtype=float)
        last = {"nl": None, "n": 0}
        # the one-sided prior's up ladder, exactly as the predictor used it
        up_ladder = state.get("sigma_ind_up")

        def _hybrid(x):
            x = np.asarray(x, dtype=float)
            return ((1.0 + x[:K] @ Phi) * j_ind + (1.0 + x[K:] @ Phi) * j_bs
                    + j_fix)

        class _SolverFailure(RuntimeError):
            """A GS solve or field read failed inside the MSE stage."""

        def _is_solver_failure(e):
            # TokaMaker raises BARE ``Exception`` for solver/field failures;
            # bouquet's own solve wrappers raise RuntimeError / ValueError.
            # Anything else (TypeError, KeyError, AttributeError, ...) is a
            # bug, and is re-raised untouched rather than turned into a
            # "refused" slice.
            return type(e) is Exception or isinstance(
                e, (RuntimeError, ValueError, FloatingPointError))

        def _solve(j):
            last["n"] += 1
            try:
                last["nl"] = solve_jphi(np.asarray(j, dtype=float))
            except Exception as e:
                if _is_solver_failure(e):
                    raise _SolverFailure(
                        f"GS solve failed ({type(e).__name__}: {e})") from e
                raise

        _field_raw = field_at

        def field_at(R, Z):
            try:
                return _field_raw(R, Z)
            except Exception as e:
                if _is_solver_failure(e):
                    raise _SolverFailure(
                        f"field read failed ({type(e).__name__}: {e})") from e
                raise

        def _resolve(lin):
            if soft:
                return close_ip_structured_soft(
                    psi_g, state["w_lin"], state["c_signed"],
                    state["Ip_signed"],
                    (None if state.get("ip_sigma") is None
                     else float(state["ip_sigma"])),
                    j_ind, j_bs, j_fix, basis=state["basis"],
                    sigma_ind=state["sigma_ind"], sigma_bs=state["sigma_bs"],
                    sigma_ind_up=up_ladder,
                    li_target=state.get("li_target"),
                    li_sigma=(None if state.get("li_sigma") is None
                              else float(state["li_sigma"])),
                    li_kind=str(state.get("li_kind", "li_1")),
                    li_geom=state.get("li_geom"),
                    axis=(None if state.get("axis") is None
                          else dict(state["axis"])),
                    axis_sigma=None, mse_lin=lin)
            return close_ip_structured(
                psi_g, state["w_lin"], state["c_signed"], state["Ip_signed"],
                j_ind, j_bs, j_fix, basis=state["basis"],
                weights=state["weights"],
                axis=(None if state.get("axis") is None
                      else dict(state["axis"])),
                li_target=state.get("li_target"),
                li_kind=str(state.get("li_kind", "li_1")),
                li_geom=state.get("li_geom"),
                sigma_ind_up=up_ladder, mse_lin=lin)

        rec = {}
        stage_flags = []
        prev = getattr(bl, "ip_closure", None) or {}
        # The corrector reads its "*_predictor" fields off whatever mygs holds
        # when it runs -- after this stage that is the MSE-stage equilibrium,
        # not the predictor.  Read the predictor's q0 / l_i HERE, before the
        # first probe moves anything, so those names keep their meaning (the
        # corrector then records its entry state under "*_mse_stage").
        rec.update(Bouquet._structured_predictor_readback(state, mygs))
        try:
            B0, found = field_at(ch["R"], ch["Z"])
            B0 = np.asarray(B0, dtype=float).reshape(-1, 3)
            found = np.asarray(found, dtype=bool).ravel()
            if not found.all():
                _off = [int(i) for i in np.asarray(ch["index"])[~found]]
                ch = mse_exclude(ch, ~found, MSE_REASON_OFF_MESH)
                state["mse"] = ch
                B0 = B0[found]
                rec.update(Bouquet._structured_mse_chord_record(ch, bl))
                print("[imas SWB-split:ohmic structured] WARNING MSE: "
                      f"{len(_off)} chord(s) at input index {_off} are OFF "
                      "the solver mesh and are EXCLUDED (no field can be "
                      f"read there); {int(ch['n_active'])} chord(s) remain",
                      flush=True)
                if int(ch["n_active"]) < int(ch.get("min_chords", 1)):
                    raise MSEDataUnusable(
                        f"only {int(ch['n_active'])} MSE chord(s) remain on "
                        f"the solver mesh (chord(s) {_off} are off it); at "
                        f"least {int(ch['min_chords'])} are required")

            def _field_strict():
                """The field at the SAME chords; any not-found is a refusal."""
                B, fnd = field_at(ch["R"], ch["Z"])
                fnd = np.asarray(fnd, dtype=bool).ravel()
                if not fnd.all():
                    raise RuntimeError(
                        "MSE chord(s) at input index "
                        f"{[int(i) for i in np.asarray(ch['index'])[~fnd]]} "
                        "were not found on the mesh on a later field read -- "
                        "refusing to reuse a stale field value")
                return np.asarray(B, dtype=float).reshape(-1, 3)

            _axis = getattr(mygs, "o_point", None)
            eq_or = mse_equilibrium_orientation(B0, ch["R"], ch["Z"], _axis)
            sp, st = mse_orientation(ch, eq_or)
            chk = mse_orientation_check(B0, ch, sp, st)
            rec.update(
                structured_mse_orientation=dict(
                    pol=sp, tor=st,
                    ip_sign_data=float(ch["ip_sign"]),
                    bt_sign_data=float(ch["bt_sign"]),
                    ip_sign_equilibrium=float(eq_or["ip"]),
                    bt_sign_equilibrium=float(eq_or["bt"]),
                    n_chords_agreeing_ip=int(eq_or["n_ip_agree"]),
                    rule=("STATED, not fitted: sign_pol = ip_sign(data) * "
                          "ip_sign(equilibrium), sign_tor = bt_sign(data) * "
                          "bt_sign(equilibrium); the equilibrium's signs are "
                          "read off its field (B_phi; poloidal circulation "
                          "about the magnetic axis)")),
                structured_mse_orientation_chi2_table=dict(chk["table"]),
                structured_mse_orientation_delta_chi2=float(
                    chk["delta_chi2"]),
                structured_mse_orientation_disagrees=bool(chk["disagrees"]),
                structured_mse_orientation_note=chk["note"])
            if chk["disagrees"]:
                stage_flags.append(
                    MSE_FLAG_PREFIX + "the data disagree with the stated field "
                    f"orientation (sign_pol {sp:+.0f}, sign_tor {st:+.0f}): "
                    f"orientation {chk['best_other']} fits the chords better "
                    f"by delta chi2 = {chk['delta_chi2']:.4g} (> "
                    f"{MSE_ORIENTATION_DCHI2:g}); the stated orientation is "
                    "KEPT -- check ip_sign/bt_sign and the sign of E_r")
                print("[imas SWB-split:ohmic structured] WARNING closure-"
                      "limited: " + stage_flags[-1], flush=True)
            tg_pred = mse_tan_gamma(B0, ch, sp, st)

            # Can the MSE term move the closure at all?  A zero-Jacobian
            # dry run of the SAME closure (no GS solve) reports the dimension
            # of the free-coefficient space left once the hard rows are
            # imposed; zero means the constraints use every free coefficient
            # and chi2_MSE could only be recorded, never fitted.
            _dry = _resolve(structured_mse_linear_model(
                state["x_pred"], tg_pred,
                np.zeros((int(ch["n_active"]), 2 * K)), ch))
            if int(_dry.get("mse_free_dim", 1)) == 0:
                _msg = ("no free coefficient is left once the hard "
                        "constraints are imposed (null space of dimension 0) "
                        "-- the MSE term cannot move this closure")
                if required:
                    raise RuntimeError(
                        "closure_channel='structured': " + _msg + ", and "
                        "structured_mse_required=True -- refusing to report "
                        "a closure the MSE constraint did not shape")
                state["mse_n_solves"] = 0
                state["mse_stage_word"] = "MSE stage not applied"
                rec.update(structured_mse_status="not applied: " + _msg,
                           structured_mse_n_solves=0)
                if stage_flags:
                    _r = list(prev.get("closure_limited_reasons", ()) or ())
                    _r += [w for w in stage_flags if w not in _r]
                    rec.update(closure_limited=True,
                               closure_limited_reasons=tuple(_r))
                if getattr(bl, "ip_closure", None) is not None:
                    bl.ip_closure.update(rec)
                print("[imas SWB-split:ohmic structured] WARNING MSE: "
                      + _msg + "; NOT applied (no solve spent)", flush=True)
                return None

            def _tan_gamma_of(x):
                _solve(_hybrid(x))
                return mse_tan_gamma(_field_strict(), ch, sp, st)

            res = structured_mse_outer(
                state["x_pred"], state["F_pred"], tg_pred, _tan_gamma_of,
                _resolve, ch, state["free"],
                fd_step=float(state.get("mse_fd_step", 0.02)),
                n_steps=int(state.get("mse_steps", 1)))
        except (RuntimeError, ValueError, FloatingPointError) as e:
            if required:
                raise RuntimeError(
                    "closure_channel='structured': the MSE-constrained "
                    f"closure could not be delivered ({e}) and "
                    "structured_mse_required=True -- refusing to fall back "
                    "to the closure without MSE") from e
            # the finite-difference probes moved mygs: put the predictor's
            # equilibrium back before anything reads it
            _solve(bl.j_phi)
            why = (MSE_FLAG_PREFIX + "stage refused, predictor kept "
                   f"({str(e)[:160]})")
            reasons = list(prev.get("closure_limited_reasons", ()) or ())
            for _w in stage_flags + [why]:
                if _w not in reasons:
                    reasons.append(_w)
            state["mse_n_solves"] = int(last["n"])
            rec.update(structured_mse_status="refused, not applied: "
                                             + str(e)[:300],
                       structured_mse_n_solves=int(last["n"]),
                       closure_limited=True,
                       closure_limited_reasons=tuple(reasons))
            if getattr(bl, "ip_closure", None) is not None:
                bl.ip_closure.update(rec)
            print("[imas SWB-split:ohmic structured] WARNING closure-limited: "
                  + why, flush=True)
            return last["nl"]

        out, R = res["out"], res["record"]
        s_ind = np.asarray(out["s_ind"], dtype=float)
        s_bs = np.asarray(out["s_bs"], dtype=float)
        bl.ohm_scale = float(out["ohm_scale_eff"])
        bl.bs_scale = float(out["bs_scale_eff"])
        # the multiplier the draws apply (PR #70 review B5: stale here)
        bl.bs_scale_profile = s_bs.copy()
        bl.j_inductive = s_ind * j_ind
        bl.j_BS = s_bs * j_bs
        bl.j_phi = bl.j_inductive + bl.j_BS + j_fix
        # re-centred on the DELIVERED equilibrium, for the corrector's re-solves
        state["mse_lin"] = structured_mse_linear_model(res["x"], res["tg"],
                                                       R["jacobian"], ch)
        state["mse_sign"] = (sp, st)
        state["mse_applied"] = True
        Bouquet._mse_record_put(
            bl,
            residual_sigma_before=np.asarray(R["residual_sigma_before"],
                                             dtype=float),
            residual_sigma_after=np.asarray(R["residual_sigma_after"],
                                            dtype=float),
            tgamma_pred_before=np.asarray(R["tgamma_pred_before"], dtype=float),
            tgamma_pred_after=np.asarray(R["tgamma_pred_after"], dtype=float),
            jacobian=np.asarray(R["jacobian"], dtype=float))
        state["mse_n_solves"] = int(last["n"])
        # what the delivered-equilibrium check compares against, and the
        # closure the delivered profiles come from until a corrector re-solve
        # replaces it (see _structured_mse_delivered)
        state["mse_chi2_before"] = float(R["chi2_before"])
        state["mse_objective_before"] = float(R["objective_before"])
        state["mse_delivered_out"] = out
        state["mse_corrector_resolved"] = False

        _at = lambda s: {f"{r:.2f}": float(np.interp(r, psi_g, s))
                         for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)}
        _fl = lambda v: [float(x) for x in np.ravel(v)]
        rec.update(
            structured_mse_status="applied",
            structured_mse_n_solves=int(last["n"]),
            structured_mse_n_fd_solves=int(R["n_fd_solves"]),
            structured_mse_chi2_before=float(R["chi2_before"]),
            structured_mse_chi2_after=float(R["chi2_after"]),
            structured_mse_chi2_model_after=float(R["chi2_model_after"]),
            structured_mse_chi2_red_before=float(R["chi2_before"])
            / int(ch["n_active"]),
            structured_mse_chi2_red_after=float(R["chi2_after"])
            / int(ch["n_active"]),
            structured_mse_residual_sigma_max_abs_before=float(
                np.max(np.abs(R["residual_sigma_before"]))),
            structured_mse_residual_sigma_max_abs_after=float(
                np.max(np.abs(R["residual_sigma_after"]))),
            structured_mse_objective_before=float(R["objective_before"]),
            structured_mse_objective_after_model=float(
                R["objective_after_model"]),
            structured_mse_objective_after=float(R["objective_after"]),
            structured_mse_linearisation_residual_max_sigma=float(
                R["steps"][-1]["linearisation_residual_max_sigma"]),
            structured_mse_linearisation_residual_rms_sigma=float(
                R["steps"][-1]["linearisation_residual_rms_sigma"]),
            structured_mse_step_log=[dict(s) for s in R["steps"]],
            structured_mse_jacobian_shape=[int(v) for v in
                                           np.shape(R["jacobian"])],
            structured_mse_er_terms=mse_er_terms(ch),
            # the same closure fields the corrector refreshes, so every
            # structured_* entry describes the DELIVERED multiplier profiles
            ohm_scale=float(bl.ohm_scale), bs_scale=float(bl.bs_scale),
            structured_coeffs_a=_fl(out["a"]),
            structured_coeffs_b=_fl(out["b"]),
            structured_s_ind_min=float(s_ind.min()),
            structured_s_ind_max=float(s_ind.max()),
            structured_s_bs_min=float(s_bs.min()),
            structured_s_bs_max=float(s_bs.max()),
            structured_s_ind_profile=_fl(s_ind),
            structured_s_bs_profile=_fl(s_bs),
            structured_s_ind_at=_at(s_ind),
            structured_s_bs_at=_at(s_bs),
            structured_sign_pattern=(
                None if out.get("sign_pattern") is None else
                "".join("+" if v else "-" for v in out["sign_pattern"])),
            structured_n_sign_iter=int(out.get("n_sign_iter", 0) or 0),
            structured_structure_ind=float(out["structure_ind"]),
            structured_structure_bs=float(out["structure_bs"]),
            structured_ip_residual=float(out["ip_residual"]),
            structured_ip_residual_pct=float(out["ip_residual_pct"]),
            structured_ip_posterior=abs(float(out["Ip_hybrid"])),
            structured_axis_residual=(None if out.get("axis_residual") is None
                                      else float(out["axis_residual"])),
            structured_constraints=list(out["constraints"]),
            structured_li_predicted=out.get("li_predicted"),
            structured_li_predictor_residual=out.get("li_predictor_residual"),
            structured_residual_sigma_Ip_model=out.get("residual_sigma_Ip"),
            structured_residual_sigma_li_model=out.get("residual_sigma_li"),
            structured_objective=out.get("objective"),
            structured_prior_chi2=out.get("prior_chi2"),
            structured_gn_iterations=out.get("n_iter"),
            # how the soft solve stopped (noise_floor = accepted at the
            # objective's rounding-noise floor, see close_ip_structured_soft)
            # and whether the loop's single logged retry was needed
            structured_gn_stop_reason=out.get("gn_stop_reason"),
            structured_gn_stop=out.get("gn_stop"),
            structured_n_noise_floor_accepts=out.get("n_noise_floor_accepts"),
            structured_closure_retry=out.get("closure_retry"),
            structured_closure_retry_first_error=out.get(
                "closure_retry_first_error"),
        )
        # Ip bookkeeping (soft: a new posterior) and the health record, from
        # the DELIVERED scales -- same function, same thresholds.
        z_ip = None
        if soft and state.get("ip_sigma"):
            _ip_meas = abs(float(state["Ip_signed"]))
            _ip_post = abs(float(out["Ip_hybrid"]))
            z_ip = (_ip_post - _ip_meas) / float(state["ip_sigma"])
            rec.update(structured_ip_measured_residual_pct=(
                100.0 * (_ip_post - _ip_meas) / _ip_meas),
                structured_residual_sigma_Ip=z_ip)
        health = closure_health(bl.ohm_scale, bl.bs_scale, state["Ip_signed"],
                                state["c_signed"], state["ip_ind"],
                                state["ip_bs"], state["ip_fix"],
                                soft_ip_residual_sigma=z_ip)
        reasons = list(health["closure_limited_reasons"])
        for why in stage_flags:
            if why not in reasons:
                reasons.append(why)
        for why in res["flags"]:
            if why not in reasons:
                reasons.append(why)
            print("[imas SWB-split:ohmic structured] WARNING closure-limited: "
                  + why, flush=True)
        for k, v in health.items():
            rec[k] = v
        rec["closure_limited_reasons"] = tuple(reasons)
        rec["closure_limited"] = bool(reasons)
        if getattr(bl, "ip_closure", None) is not None:
            bl.ip_closure.update(rec)
        print("[imas SWB-split:ohmic structured] MSE: "
              f"{int(ch['n_active'])} chords, orientation (pol {sp:+.0f}, tor "
              f"{st:+.0f}), {R['n_solves']} solves "
              f"({R['n_fd_solves']} finite-difference + {R['n_steps']} "
              f"step); chi2 {R['chi2_before']:.2f} -> {R['chi2_after']:.2f} "
              f"(linear model {R['chi2_model_after']:.2f}; linearisation "
              "residual max "
              f"{R['steps'][-1]['linearisation_residual_max_sigma']:.3f} "
              f"sigma); objective {R['objective_before']:.4g} -> "
              f"{R['objective_after']:.4g}; {mse_er_terms(ch)}", flush=True)
        return last["nl"]

    @staticmethod
    def _structured_mse_jbs_stage(state, bl, mygs, solve_jphi, jbs0, refresh,
                                  evaluate, measure, weights, settings, Ip,
                                  gate_q0=False, field_at=None, pre_mse=None,
                                  q0_pin=None):
        """The MSE term under the self-consistent bootstrap loop.

        Runs after the j_BS loop has converged WITHOUT the MSE term (``state``
        is the structured closure state of that converged pass and ``mygs``
        holds its equilibrium, solved with bootstrap ``jbs0``):

        1. the forward-difference Jacobian of tan(gamma) ONCE at that state,
           with j_BS held fixed during the differences (an approximation of
           the loop map's true Jacobian, recorded as such);
        2. chord steps -- closure with the linearised MSE term on the CURRENT
           geometry and bootstrap -> GS solve -> ``evaluate`` (+relaxation) ->
           refresh the linearisation offset and the closure state -- until
           the j_BS residuals hold on two consecutive steps AND the synthetic
           tan(gamma) moved by less than
           ``jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA`` sigma on every chord
           (at most ``jbs_max_passes`` steps: every chord step is also a
           pass of the bootstrap loop, so the loop's own ceiling applies);
        3. the Jacobian recomputed ONCE at the converged state and one final
           step taken, because a stale Jacobian biases the stationary point of
           a chord iteration, not only its rate; the Jacobian change and the
           objective change are recorded.

        ``refresh(jbs, k)`` re-runs the structured predictor on the current
        mygs geometry with bootstrap ``jbs`` and returns the new state;
        ``evaluate(eq)`` is the Redl bootstrap on ``eq``; ``measure(eq)``
        returns ``{li, q0}``; ``weights(eq)`` returns ``(w, x)`` for the
        residual norm.  Returns ``(nl_its, final_state, record)``.  A refusal
        (closure out of bounds, failed solve, unusable field) raises when
        ``structured_mse_required``.  Otherwise the PRE-MSE loop result is
        delivered, and verified, not assumed:

        * the state the chord steps overwrite (``bl.j_phi / j_BS /
          j_inductive / jBS_diff / jphi_diff / ohm_scale / bs_scale`` and
          ``bl.ip_closure``) is snapshotted HERE, before the first chord step,
          and restored exactly (``pre_mse["restore"]`` restores the caller's
          own closure state the same way);
        * mygs is put back on the pre-MSE equilibrium (``replace_eq`` of the
          snapshot) and the current the loop's last pass actually SOLVED
          (``pre_mse["j_solved"]``; the closure's ``bl.j_phi`` when not
          given) is re-solved;
        * Redl on that re-solve is checked against the pre-MSE bootstrap
          ``jbs0`` at the loop's own ``rtol_j``/``rtol_Ip``, and its l_i (q0
          where gated) against the pre-MSE equilibrium's at ``tol_li``
          (``tol_q0``) -- the record's ``converged`` is that check, with the
          residuals and the reason, and the slice is flagged either way.

        ``q0_pin`` (``jbs_loop_q0_corrector=True`` with an active axis row;
        :class:`bouquet.jbs_loop.AxisRowPin`): the chord steps are passes of
        the loop, so the pin keeps acting -- ``refresh`` moves the axis row
        from each step's measured q0 -- and every step's criterion, the final
        step's and the refusal re-solve's ADD ``|q0 - q0_target| <= q0_tol``
        (recorded per step as ``q0_residual``).  ``None``: exactly as before.
        """
        import numpy as np

        from .mse import (MSE_ORIENTATION_DCHI2, MSE_REASON_OFF_MESH,
                          MSEDataUnusable, mse_chi2,
                          mse_equilibrium_orientation, mse_er_terms,
                          mse_exclude, mse_field_at, mse_orientation,
                          mse_orientation_check, mse_tan_gamma)
        from .utils import (MSE_FLAG_PREFIX, close_ip_structured,
                            close_ip_structured_soft, closure_health,
                            soft_closure_with_retry,
                            structured_basis_eval, structured_mse_jacobian,
                            structured_mse_linear_model,
                            structured_objective_no_mse)
        from .jbs_loop import (MSE_CHORD_OFFSET_TOL_SIGMA, JBSNotConverged,
                               profile_residuals, JBS_RELAX_FLOOR,
                               JBS_REQUIRED_CONSECUTIVE,
                               JBS_GROWTH_ABORT_PASSES)

        ch = state["mse"]
        required = bool(state.get("mse_required", False))
        soft = bool(state.get("soft"))
        # field_at(R, Z) -> (B, found), exactly as bouquet.mse.mse_field_at
        # and the legacy stage's field_at: a chord the interpolator cannot
        # place on the mesh is REPORTED, never read from a stale buffer
        if field_at is None:
            field_at = lambda R, Z: mse_field_at(mygs, R, Z)
        psi_g = np.asarray(state["psi_geom"], dtype=float)
        Phi = structured_basis_eval(state["basis"], psi_g)
        K = Phi.shape[0]
        sig = np.asarray(ch["sigma_eff"], dtype=float)
        fd_step = float(state.get("mse_fd_step", 0.02))
        free = state["free"]
        n_free = int(np.count_nonzero(free))
        # "n": EVERY GS solve this stage spends (FD probes, chord steps, the
        # final step, a refusal's restore solve); "n_fd": the FD probes alone
        last = {"nl": None, "n": 0, "n_fd": 0}

        def _hybrid(st, jbs, x):
            x = np.asarray(x, dtype=float)
            return ((1.0 + x[:K] @ Phi) * np.asarray(st["j_ind"], float)
                    + (1.0 + x[K:] @ Phi) * np.asarray(jbs, float)
                    + np.asarray(st["j_fixed"], float))

        class _SolverFailure(RuntimeError):
            """A GS solve or field read failed inside the MSE stage."""

        def _is_solver_failure(e):
            # as the legacy stage: TokaMaker raises BARE ``Exception`` for
            # solver/field failures, bouquet's wrappers RuntimeError /
            # ValueError; anything else (TypeError, KeyError, ...) is a bug
            # and is re-raised untouched, never turned into a refused slice
            return type(e) is Exception or isinstance(
                e, (RuntimeError, ValueError, FloatingPointError))

        def _solve(j):
            last["n"] += 1
            try:
                last["nl"] = solve_jphi(np.asarray(j, dtype=float))
            except Exception as e:
                if _is_solver_failure(e):
                    raise _SolverFailure(
                        f"GS solve failed ({type(e).__name__}: {e})") from e
                raise

        _field_raw = field_at

        def field_at(R, Z):
            try:
                return _field_raw(R, Z)
            except Exception as e:
                if _is_solver_failure(e):
                    raise _SolverFailure(
                        f"field read failed ({type(e).__name__}: {e})") from e
                raise

        def _field_strict():
            """The field at the SAME chords; a chord not found is a refusal
            (the mesh does not move between solves), never a reused value."""
            B, fnd = field_at(ch["R"], ch["Z"])
            fnd = np.asarray(fnd, dtype=bool).ravel()
            if not fnd.all():
                raise RuntimeError(
                    "MSE chord(s) at input index "
                    f"{[int(i) for i in np.asarray(ch['index'])[~fnd]]} "
                    "were not found on the mesh on a later field read -- "
                    "refusing to reuse a stale field value")
            return np.asarray(B, dtype=float).reshape(-1, 3)

        def _resolve(st, lin, x_retry=None, log=True):
            if soft:
                # every chord step is a pass of the bootstrap loop: ONE logged
                # retry from the previous step's coefficients after a
                # no-descent refusal (soft_closure_with_retry)
                _o = soft_closure_with_retry(
                    lambda _x0: close_ip_structured_soft(
                        st["psi_geom"], st["w_lin"], st["c_signed"],
                        st["Ip_signed"],
                        (None if st.get("ip_sigma") is None
                         else float(st["ip_sigma"])),
                        st["j_ind"], st["j_BS_swb"], st["j_fixed"],
                        basis=st["basis"], sigma_ind=st["sigma_ind"],
                        sigma_bs=st["sigma_bs"],
                        sigma_ind_up=st.get("sigma_ind_up"),
                        li_target=st.get("li_target"),
                        li_sigma=(None if st.get("li_sigma") is None
                                  else float(st["li_sigma"])),
                        li_kind=str(st.get("li_kind", "li_1")),
                        li_geom=st.get("li_geom"),
                        axis=(None if st.get("axis") is None
                              else dict(st["axis"])),
                        axis_sigma=None, mse_lin=lin, x0=_x0,
                        # a pass of the self-consistent bootstrap loop
                        accept_noise_floor=True),
                    x_prev=x_retry, who="jbs-loop MSE chord")
                if log:
                    srec["closure_retry"].append(
                        int(_o.get("closure_retry", 0)))
                    srec["closure_stop_reason"].append(
                        _o.get("gn_stop_reason"))
                return _o
            return close_ip_structured(
                st["psi_geom"], st["w_lin"], st["c_signed"], st["Ip_signed"],
                st["j_ind"], st["j_BS_swb"], st["j_fixed"],
                basis=st["basis"], weights=st["weights"],
                axis=(None if st.get("axis") is None else dict(st["axis"])),
                li_target=st.get("li_target"),
                li_kind=str(st.get("li_kind", "li_1")),
                li_geom=st.get("li_geom"),
                sigma_ind_up=st.get("sigma_ind_up"), mse_lin=lin)

        def _x_of(out):
            return np.concatenate([np.asarray(out["a"], dtype=float),
                                   np.asarray(out["b"], dtype=float)])

        srec = dict(stage=("structured MSE chord iteration with j_BS "
                           "re-evaluated after every solve"),
                    chord_max_steps=int(settings["max_passes"]),
                    chord_offset_tol_sigma=float(MSE_CHORD_OFFSET_TOL_SIGMA),
                    jacobian_note=("forward differences with j_BS held "
                                   "fixed: an approximation of the loop "
                                   "map's Jacobian"),
                    r_j=[], r_I=[], dl_i=[], dq0=[], I_BS=[], omega=[],
                    offset_change_max_sigma=[], pass_ok=[], n_passes=0,
                    converged=False, stop_reason=None,
                    relax_halve_on=int(settings.get("relax_halve_on", 1)),
                    relax_current=None,
                    current_relaxation=(
                        "not applied in the chord steps: each step solves "
                        "the closure's own current, on which the MSE "
                        "linearisation is centred"),
                    closure_retry=[], closure_stop_reason=[])
        rec = {}
        stage_flags = []
        # ---- the pre-MSE state, captured BEFORE any chord step: every
        # refresh() re-closes on the latest geometry and OVERWRITES these
        # (bl.ip_closure is replaced wholesale), so a refusal must restore
        # this snapshot, not whatever the last refresh left ---------------
        import copy as _copy
        _pre_bl = {}
        for _a in ("j_phi", "j_BS", "j_inductive", "jBS_diff", "jphi_diff",
                   "ohm_scale", "bs_scale"):
            if hasattr(bl, _a):
                _v = getattr(bl, _a)
                _pre_bl[_a] = (np.array(_v, dtype=float, copy=True)
                               if isinstance(_v, np.ndarray) else _v)
        _pre_icl = _copy.deepcopy(getattr(bl, "ip_closure", None))
        _pre_eq = mygs.copy_eq() if hasattr(mygs, "copy_eq") else None
        _pre_meas = {}
        prev = _pre_icl or {}
        try:
            meas_prev = dict(measure(mygs.copy_eq()))
            _pre_meas.update(meas_prev)
            # ---- chords off the solver mesh: EXCLUDED at the first read
            # (on the converged pre-MSE equilibrium), recorded and announced;
            # fewer than min_chords left is a refusal (as the legacy stage)
            B0, found = field_at(ch["R"], ch["Z"])
            B0 = np.asarray(B0, dtype=float).reshape(-1, 3)
            found = np.asarray(found, dtype=bool).ravel()
            if not found.all():
                _off = [int(i) for i in np.asarray(ch["index"])[~found]]
                ch = mse_exclude(ch, ~found, MSE_REASON_OFF_MESH)
                state["mse"] = ch
                sig = np.asarray(ch["sigma_eff"], dtype=float)
                B0 = B0[found]
                rec.update(Bouquet._structured_mse_chord_record(ch, bl))
                print("[imas SWB-split:ohmic structured] WARNING MSE: "
                      f"{len(_off)} chord(s) at input index {_off} are OFF "
                      "the solver mesh and are EXCLUDED (no field can be "
                      f"read there); {int(ch['n_active'])} chord(s) remain",
                      flush=True)
                if int(ch["n_active"]) < int(ch.get("min_chords", 1)):
                    raise MSEDataUnusable(
                        f"only {int(ch['n_active'])} MSE chord(s) remain on "
                        f"the solver mesh (chord(s) {_off} are off it); at "
                        f"least {int(ch['min_chords'])} are required")
            # ---- orientation: STATED (ip_sign/bt_sign of the block against
            # the equilibrium's own directions), never fitted; audited ----
            _axis = getattr(mygs, "o_point", None)
            eq_or = mse_equilibrium_orientation(B0, ch["R"], ch["Z"], _axis)
            sp, st_sign = mse_orientation(ch, eq_or)
            chk = mse_orientation_check(B0, ch, sp, st_sign)
            rec.update(
                structured_mse_orientation=dict(
                    pol=sp, tor=st_sign,
                    ip_sign_data=float(ch["ip_sign"]),
                    bt_sign_data=float(ch["bt_sign"]),
                    ip_sign_equilibrium=float(eq_or["ip"]),
                    bt_sign_equilibrium=float(eq_or["bt"]),
                    n_chords_agreeing_ip=int(eq_or["n_ip_agree"]),
                    rule=("STATED, not fitted: sign_pol = ip_sign(data) * "
                          "ip_sign(equilibrium), sign_tor = bt_sign(data) * "
                          "bt_sign(equilibrium); the equilibrium's signs are "
                          "read off its field (B_phi; poloidal circulation "
                          "about the magnetic axis)")),
                structured_mse_orientation_chi2_table=dict(chk["table"]),
                structured_mse_orientation_delta_chi2=float(
                    chk["delta_chi2"]),
                structured_mse_orientation_disagrees=bool(chk["disagrees"]),
                structured_mse_orientation_note=chk["note"])
            if chk["disagrees"]:
                stage_flags.append(
                    MSE_FLAG_PREFIX + "the data disagree with the stated field "
                    f"orientation (sign_pol {sp:+.0f}, sign_tor "
                    f"{st_sign:+.0f}): orientation {chk['best_other']} fits "
                    f"the chords better by delta chi2 = "
                    f"{chk['delta_chi2']:.4g} (> {MSE_ORIENTATION_DCHI2:g}); "
                    "the stated orientation is KEPT -- check ip_sign/bt_sign "
                    "and the sign of E_r")
                print("[imas SWB-split:ohmic structured] WARNING closure-"
                      "limited: " + stage_flags[-1], flush=True)
            tg_pred = mse_tan_gamma(B0, ch, sp, st_sign)
            x_pred = np.asarray(state["x_pred"], dtype=float)
            F_pred = float(state["F_pred"])
            chi2_0, z0 = mse_chi2(tg_pred, ch)
            F_before = F_pred + chi2_0
            jbs = np.asarray(jbs0, dtype=float)
            cur = state

            # ---- can the MSE term move the closure at all?  A zero-Jacobian
            # dry run of the SAME closure (no GS solve, not logged as a chord
            # step's closure) reports the free dimension left once the hard
            # rows are imposed; zero -> chi2_MSE could only be recorded ------
            _dry = _resolve(cur, structured_mse_linear_model(
                x_pred, tg_pred, np.zeros((int(ch["n_active"]), 2 * K)), ch),
                log=False)
            _fdim = (_dry or {}).get("mse_free_dim")
            if _fdim is not None and int(_fdim) == 0:
                _msg = ("no free coefficient is left once the hard "
                        "constraints are imposed (null space of dimension 0) "
                        "-- the MSE term cannot move this closure")
                if required:
                    raise RuntimeError(
                        "closure_channel='structured': " + _msg + ", and "
                        "structured_mse_required=True -- refusing to report "
                        "a closure the MSE constraint did not shape")
                state["mse_n_solves"] = 0
                state["mse_stage_word"] = "MSE stage not applied"
                rec.update(structured_mse_status="not applied: " + _msg,
                           structured_mse_n_solves=0)
                if stage_flags:
                    _r = list(prev.get("closure_limited_reasons", ()) or ())
                    _r += [w for w in stage_flags if w not in _r]
                    rec.update(closure_limited=True,
                               closure_limited_reasons=tuple(_r))
                if getattr(bl, "ip_closure", None) is not None:
                    bl.ip_closure.update(rec)
                print("[imas SWB-split:ohmic structured] WARNING MSE: "
                      + _msg + "; NOT applied (no solve spent)", flush=True)
                # nothing was solved: the converged pre-MSE loop state stands
                srec.update(converged=True, applied=False, n_solves=0,
                            stop_reason="MSE stage not applied (no solve "
                                        "spent): " + _msg)
                return last["nl"], state, srec

            def _tg_at(st, jb):
                def _f(x):
                    last["n_fd"] += 1
                    _solve(_hybrid(st, jb, x))
                    return mse_tan_gamma(_field_strict(), ch, sp, st_sign)
                return _f

            J1 = structured_mse_jacobian(_tg_at(cur, jbs), x_pred, tg_pred,
                                         free, step=fd_step)
            x_lin, tg_lin = x_pred, tg_pred
            omega = float(settings["relax"])
            halve_on = int(settings.get("relax_halve_on", 1))
            grow_streak = 0
            omega_cur = None
            streak = 0
            grow = 0
            rj_prev = None
            steps = []
            out = None
            converged = False
            n_step = 0
            n_chord = int(settings["max_passes"])
            for s in range(n_chord):
                n_step = s + 1
                lin = structured_mse_linear_model(x_lin, tg_lin, J1, ch)
                out = _resolve(cur, lin, x_retry=x_lin)
                x_new = _x_of(out)
                _solve(_hybrid(cur, jbs, x_new))
                snap = mygs.copy_eq()
                tg_new = np.asarray(mse_tan_gamma(_field_strict(), ch, sp,
                                                  st_sign), dtype=float)
                if not np.all(np.isfinite(tg_new)):
                    raise RuntimeError("the solve of the MSE-constrained "
                                       "closure returned an unusable "
                                       "tan(gamma)")
                J = np.asarray(evaluate(snap), dtype=float)
                w, xg = weights(snap)
                r = profile_residuals(J, jbs, w, xg, Ip)
                m = dict(measure(snap))
                dl_i = (None if (m.get("li") is None
                                 or meas_prev.get("li") is None)
                        else abs(float(m["li"]) - float(meas_prev["li"])))
                dq0 = (None if (m.get("q0") is None
                                or meas_prev.get("q0") is None)
                       else abs(float(m["q0"]) - float(meas_prev["q0"])))
                off = float(np.max(np.abs(tg_new - tg_lin) / sig))
                ok = (r["r_j"] <= settings["rtol_j"]
                      and r["r_I"] <= settings["rtol_Ip"]
                      and dl_i is not None and dl_i <= settings["tol_li"]
                      and (not gate_q0 or (dq0 is not None
                                           and dq0 <= settings["tol_q0"])))
                if q0_pin is not None:
                    # the pin's added criterion (never a replacement)
                    srec.setdefault("q0_residual", []).append(
                        q0_pin.residual(m.get("q0")))
                    ok = ok and q0_pin.within_tol(m.get("q0"))
                ok = bool(ok)
                chi2_new, _z = mse_chi2(tg_new, ch)
                lres = (tg_new - (tg_lin + J1 @ (x_new - x_lin))) / sig
                steps.append(dict(
                    chi2_model=float(out["mse_chi2_model"]),
                    chi2_achieved=float(chi2_new),
                    objective_model=float(out["mse_objective_model"]),
                    objective_achieved=float(
                        structured_objective_no_mse(out) + chi2_new),
                    linearisation_residual_max_sigma=float(
                        np.max(np.abs(lres))),
                    linearisation_residual_rms_sigma=float(
                        np.sqrt(np.mean(lres ** 2))),
                    coeff_step_max=float(np.max(np.abs(x_new - x_lin))),
                    offset_change_max_sigma=off,
                    r_j=r["r_j"], r_I=r["r_I"], dl_i=dl_i, dq0=dq0,
                    jbs_ok=ok))
                srec["r_j"].append(r["r_j"])
                srec["r_I"].append(r["r_I"])
                srec["dl_i"].append(dl_i)
                srec["dq0"].append(dq0)
                srec["I_BS"].append(r["I_BS"])
                srec["omega"].append(omega_cur)
                srec["offset_change_max_sigma"].append(off)
                srec["pass_ok"].append(ok)
                srec["n_passes"] = n_step
                print(f"  [jbs-loop MSE chord {n_step}] r_j={r['r_j']:.3e} "
                      f"r_I={r['r_I']:.3e}"
                      + ("" if dl_i is None else f" dl_i={dl_i:.2e}")
                      + ("" if dq0 is None else f" dq0={dq0:.2e}")
                      + f" tan(gamma) moved {off:.3f} sigma; chi2 "
                      f"{chi2_new:.3f}", flush=True)
                streak = streak + 1 if ok else 0
                if streak >= JBS_REQUIRED_CONSECUTIVE \
                        and off <= MSE_CHORD_OFFSET_TOL_SIGMA:
                    converged = True
                    break
                if rj_prev is not None and r["r_j"] > rj_prev:
                    if omega_cur is not None and \
                            omega_cur <= JBS_RELAX_FLOOR + 1e-15:
                        grow += 1
                    else:
                        grow = 0
                    # halve only on SUSTAINED growth, exactly as the kernel
                    # (jbs_relax_halve_on consecutive growing steps)
                    grow_streak += 1
                    if grow_streak >= halve_on:
                        omega = max(0.5 * omega, JBS_RELAX_FLOOR)
                        grow_streak = 0
                else:
                    grow = 0
                    grow_streak = 0
                rj_prev = r["r_j"]
                if grow >= JBS_GROWTH_ABORT_PASSES:
                    srec["stop_reason"] = ("r_j grew at the relaxation "
                                           "floor")
                    break
                if s == n_chord - 1:
                    break
                jbs = (1.0 - omega) * jbs + omega * J
                omega_cur = omega
                cur = refresh(jbs, s)
                x_lin, tg_lin = x_new, tg_new
                meas_prev = m

            if not converged:
                srec["stop_reason"] = srec["stop_reason"] or (
                    f"no convergence within {n_chord} chord "
                    "steps (j_BS residuals on two consecutive steps and "
                    "tan(gamma) offset change)")
            else:
                # ---- one Jacobian refresh at the converged state + a final
                # step (the stale-J stationary-point bias) -----------------
                jbs_next = (1.0 - omega) * jbs + omega * J
                cur_next = refresh(jbs_next, n_step)   # geometry of E_new
                J2 = structured_mse_jacobian(_tg_at(cur, jbs), x_new,
                                             tg_new, free, step=fd_step)
                dJ = float(np.linalg.norm(J2 - J1)
                           / max(np.linalg.norm(J1), 1e-300))
                obj_conv = steps[-1]["objective_achieved"]
                lin2 = structured_mse_linear_model(x_new, tg_new, J2, ch)
                out = _resolve(cur_next, lin2, x_retry=x_new)
                x_f = _x_of(out)
                _solve(_hybrid(cur_next, jbs_next, x_f))
                snap = mygs.copy_eq()
                tg_f = np.asarray(mse_tan_gamma(_field_strict(), ch, sp, st_sign),
                                  dtype=float)
                if not np.all(np.isfinite(tg_f)):
                    raise RuntimeError("the final MSE step returned an "
                                       "unusable tan(gamma)")
                J = np.asarray(evaluate(snap), dtype=float)
                w, xg = weights(snap)
                r = profile_residuals(J, jbs_next, w, xg, Ip)
                mf = dict(measure(snap))
                dl_i = (None if (mf.get("li") is None or m.get("li") is None)
                        else abs(float(mf["li"]) - float(m["li"])))
                dq0 = (None if (mf.get("q0") is None or m.get("q0") is None)
                       else abs(float(mf["q0"]) - float(m["q0"])))
                okf = bool(r["r_j"] <= settings["rtol_j"]
                           and r["r_I"] <= settings["rtol_Ip"]
                           and dl_i is not None
                           and dl_i <= settings["tol_li"]
                           and (not gate_q0 or (dq0 is not None and
                                                dq0 <= settings["tol_q0"])))
                if q0_pin is not None:
                    # the delivered step: logged in the pin's per-pass record
                    # (no advance -- nothing is solved after it)
                    q0_pin.observe(
                        mf.get("q0"),
                        ((cur_next.get("axis") or {}).get("j_ref0",
                                                          q0_pin.row)),
                        stage="MSE final step")
                    srec.setdefault("q0_residual", []).append(
                        q0_pin.residual(mf.get("q0")))
                    okf = bool(okf and q0_pin.within_tol(mf.get("q0")))
                chi2_f, _zf = mse_chi2(tg_f, ch)
                lres = (tg_f - (tg_new + J2 @ (x_f - x_new))) / sig
                steps.append(dict(
                    final_jacobian_refresh=True,
                    chi2_model=float(out["mse_chi2_model"]),
                    chi2_achieved=float(chi2_f),
                    objective_model=float(out["mse_objective_model"]),
                    objective_achieved=float(
                        structured_objective_no_mse(out) + chi2_f),
                    linearisation_residual_max_sigma=float(
                        np.max(np.abs(lres))),
                    linearisation_residual_rms_sigma=float(
                        np.sqrt(np.mean(lres ** 2))),
                    coeff_step_max=float(np.max(np.abs(x_f - x_new))),
                    offset_change_max_sigma=float(
                        np.max(np.abs(tg_f - tg_new) / sig)),
                    r_j=r["r_j"], r_I=r["r_I"], dl_i=dl_i, dq0=dq0,
                    jbs_ok=okf))
                for _k, _v in (("r_j", r["r_j"]), ("r_I", r["r_I"]),
                               ("dl_i", dl_i), ("dq0", dq0),
                               ("I_BS", r["I_BS"]), ("omega", omega),
                               ("offset_change_max_sigma",
                                steps[-1]["offset_change_max_sigma"]),
                               ("pass_ok", okf)):
                    srec[_k].append(_v)
                srec["n_passes"] = n_step + 1
                srec.update(jacobian_refresh_rel_change=dJ,
                            objective_at_convergence=float(obj_conv),
                            objective_after_final_step=float(
                                steps[-1]["objective_achieved"]),
                            objective_change_final_step=float(
                                steps[-1]["objective_achieved"] - obj_conv))
                print(f"  [jbs-loop MSE final] Jacobian refreshed "
                      f"(|dJ|/|J| = {dJ:.3e}); r_j={r['r_j']:.3e} "
                      f"r_I={r['r_I']:.3e}; objective {obj_conv:.4g} -> "
                      f"{steps[-1]['objective_achieved']:.4g}", flush=True)
                if not okf:
                    converged = False
                    srec["stop_reason"] = ("the final step after the "
                                           "Jacobian refresh left the j_BS "
                                           "residuals outside tolerance"
                                           + ("" if q0_pin is None else
                                              " (or q0 outside q0_tol: the "
                                              "q0 pin's criterion)"))
                else:
                    srec["stop_reason"] = (
                        "j_BS residuals on two consecutive chord steps, "
                        "tan(gamma) offset converged, final Jacobian "
                        "refresh step inside tolerance")
                cur, jbs, J1, x_new, tg_new = cur_next, jbs_next, J2, x_f, tg_f
            srec["converged"] = bool(converged)
            srec["final"] = dict(r_j=srec["r_j"][-1] if srec["r_j"] else None,
                                 r_I=srec["r_I"][-1] if srec["r_I"] else None,
                                 dl_i=srec["dl_i"][-1] if srec["dl_i"]
                                 else None,
                                 dq0=srec["dq0"][-1] if srec["dq0"] else None)
            # counted, not inferred: the Jacobian refresh runs whenever the
            # chord steps converged, even if the final step then fails
            srec["n_fd_solves"] = int(last["n_fd"])
        except (RuntimeError, ValueError, FloatingPointError) as e:
            if required:
                raise RuntimeError(
                    "closure_channel='structured': the MSE-constrained "
                    f"closure could not be delivered ({e}) and "
                    "structured_mse_required=True -- refusing to fall back "
                    "to the closure without MSE") from e
            # ---- restore EXACTLY the pre-MSE state -----------------------
            for _a, _v in _pre_bl.items():
                setattr(bl, _a, (np.array(_v, dtype=float, copy=True)
                                 if isinstance(_v, np.ndarray) else _v))
            bl.ip_closure = _copy.deepcopy(_pre_icl)
            pre = dict(pre_mse or {})
            if pre.get("restore") is not None:
                pre["restore"]()
            if _pre_eq is not None and hasattr(mygs, "replace_eq"):
                mygs.replace_eq(source_eq=_pre_eq)
            _j_pre = pre.get("j_solved")
            _j_pre = (np.asarray(bl.j_phi, dtype=float) if _j_pre is None
                      else np.asarray(_j_pre, dtype=float))
            _solve(_j_pre)
            # ---- verify the restored solve against the pre-MSE state ------
            if not _pre_meas and _pre_eq is not None:
                _pre_meas.update(dict(measure(_pre_eq)))
            _snap_r = mygs.copy_eq()
            _Jr = np.asarray(evaluate(_snap_r), dtype=float)
            _wr, _xr = weights(_snap_r)
            _rr = profile_residuals(_Jr, np.asarray(jbs0, dtype=float), _wr,
                                    _xr, Ip)
            _mr = dict(measure(_snap_r))
            _dli = (None if (_mr.get("li") is None
                             or _pre_meas.get("li") is None)
                    else abs(float(_mr["li"]) - float(_pre_meas["li"])))
            _dq0 = (None if (_mr.get("q0") is None
                             or _pre_meas.get("q0") is None)
                    else abs(float(_mr["q0"]) - float(_pre_meas["q0"])))
            _ok = bool(np.isfinite(_rr["r_j"])
                       and _rr["r_j"] <= settings["rtol_j"]
                       and np.isfinite(_rr["r_I"])
                       and _rr["r_I"] <= settings["rtol_Ip"]
                       and _dli is not None and _dli <= settings["tol_li"]
                       and (not gate_q0 or (_dq0 is not None
                                            and _dq0 <= settings["tol_q0"])))
            if q0_pin is not None:
                # the restored pre-MSE state must still meet the pin
                _ok = bool(_ok and q0_pin.within_tol(_mr.get("q0")))
            why = (MSE_FLAG_PREFIX + "stage refused, predictor kept "
                   f"({str(e)[:160]})")
            # reasons of the RESTORED (pre-MSE) closure record, not of the
            # dict the last refresh left behind
            reasons = list(prev.get("closure_limited_reasons", ()) or ())
            for _w in stage_flags + [why]:
                if _w not in reasons:
                    reasons.append(_w)
            # every solve the stage spent, the restore re-solve included
            state["mse_n_solves"] = int(last["n"])
            srec["n_solves"] = int(last["n"])
            srec["n_fd_solves"] = int(last["n_fd"])
            rec.update(structured_mse_status="refused, not applied: "
                                             + str(e)[:300],
                       structured_mse_n_solves=int(last["n"]),
                       closure_limited=True,
                       closure_limited_reasons=tuple(reasons))
            if getattr(bl, "ip_closure", None) is not None:
                bl.ip_closure.update(rec)
            print("[imas SWB-split:ohmic structured] WARNING closure-limited: "
                  + why, flush=True)
            _chk = dict(r_j=_rr["r_j"], r_I=_rr["r_I"], dl_i=_dli, dq0=_dq0,
                        ok=_ok, reference=(
                            "Redl on the re-solve vs the pre-MSE bootstrap "
                            "jbs0; l_i/q0 vs the pre-MSE equilibrium"),
                        pre_mse_final=pre.get("final"),
                        **({} if q0_pin is None else dict(
                            q0_residual=q0_pin.residual(_mr.get("q0")),
                            q0_tol=q0_pin.q0_tol)),
                        solved=("the pre-MSE loop's last solved current"
                                if pre.get("j_solved") is not None else
                                "bl.j_phi of the restored closure"))
            if _ok:
                _stop = ("MSE stage refused (not required): the pre-MSE loop "
                         "state was restored and re-solved, and the re-solve "
                         "holds the loop tolerances (r_j "
                         f"{_rr['r_j']:.2e}, r_I {_rr['r_I']:.2e}, |dl_i| "
                         f"{_dli:.2e})")
            else:
                _stop = ("MSE stage refused (not required): the restored "
                         "pre-MSE state does NOT reproduce within the loop "
                         f"tolerances on re-solve (r_j {_rr['r_j']:.2e} tol "
                         f"{settings['rtol_j']:.0e}, r_I {_rr['r_I']:.2e} tol "
                         f"{settings['rtol_Ip']:.0e}, |dl_i| "
                         + ("n/a" if _dli is None else f"{_dli:.2e}")
                         + f" tol {settings['tol_li']:.0e}"
                         + ("" if not gate_q0 else
                            ", |dq0| " + ("n/a" if _dq0 is None
                                          else f"{_dq0:.2e}")
                            + f" tol {settings['tol_q0']:.0e}")
                         + ("" if q0_pin is None else
                            ", q0-q0_target " + (
                                "n/a" if q0_pin.residual(_mr.get("q0"))
                                is None else
                                f"{q0_pin.residual(_mr.get('q0')):+.2e}")
                            + f" q0_tol {q0_pin.q0_tol:g}")
                         + ")")
            print(("  [jbs-loop MSE refused] " if _ok else
                   "  [jbs-loop MSE refused] WARNING ") + _stop, flush=True)
            srec.update(converged=_ok, stop_reason=_stop,
                        refused=str(e)[:300], restored_pre_mse=True,
                        restore_check=_chk,
                        final=dict(r_j=_rr["r_j"], r_I=_rr["r_I"],
                                   dl_i=_dli, dq0=_dq0))
            if not _ok:
                # the same failure policy as a chord stage that does not
                # converge (below): "raise" raises, "flag" delivers flagged
                msg = ("self-consistent j_BS loop [MSE chord stage] did not "
                       f"converge: {_stop}")
                srec["fail_message"] = msg
                if settings.get("on_fail", "raise") == "raise":
                    raise JBSNotConverged(msg, srec)
            return last["nl"], state, srec

        # ---- deliver the last solve ----------------------------------------
        s_ind = np.asarray(out["s_ind"], dtype=float)
        s_bs = np.asarray(out["s_bs"], dtype=float)
        bl.ohm_scale = float(out["ohm_scale_eff"])
        bl.bs_scale = float(out["bs_scale_eff"])
        # the multiplier the draws apply (PR #70 review B5: stale here)
        bl.bs_scale_profile = s_bs.copy()
        bl.j_inductive = s_ind * np.asarray(cur["j_ind"], dtype=float)
        # the bootstrap the delivered equilibrium was SOLVED with
        _jbs_del = np.asarray(cur["j_BS_swb"], dtype=float)
        bl.j_BS = s_bs * _jbs_del
        bl.j_phi = bl.j_inductive + bl.j_BS + np.asarray(cur["j_fixed"],
                                                         dtype=float)
        cur["mse_lin"] = structured_mse_linear_model(x_new, tg_new, J1, ch)
        cur["mse_sign"] = (sp, st_sign)
        cur["mse_applied"] = True
        # the chords actually used (a refreshed state carries the block as it
        # was read, before any off-mesh exclusion)
        cur["mse"] = ch
        # what the delivered-equilibrium check (_structured_mse_delivered)
        # compares against, and the closure the delivered profiles come from;
        # the loop's corrector is record-only, so nothing re-solves after it
        cur["mse_chi2_before"] = float(chi2_0)
        cur["mse_objective_before"] = float(F_before)
        cur["mse_delivered_out"] = out
        cur["mse_corrector_resolved"] = False
        cur["mse_n_solves"] = int(last["n"])
        chi2_f, z_f = mse_chi2(tg_new, ch)
        # per-chord arrays and the Jacobian: datasets on bl.mse_record, never
        # the size-capped ip_closure attribute (as the legacy stage)
        Bouquet._mse_record_put(
            bl,
            residual_sigma_before=np.asarray(z0, dtype=float),
            residual_sigma_after=np.asarray(z_f, dtype=float),
            tgamma_pred_before=np.asarray(tg_pred, dtype=float),
            tgamma_pred_after=np.asarray(tg_new, dtype=float),
            jacobian=np.asarray(J1, dtype=float))
        _fl = lambda v: [float(x) for x in np.ravel(v)]
        _at = lambda s: {f"{r:.2f}": float(np.interp(r, psi_g, s))
                         for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)}
        F_after = steps[-1]["objective_achieved"]
        flags = []
        if not (np.isfinite(F_after) and F_after <= F_before):
            flags.append(MSE_FLAG_PREFIX
                         + f"achieved objective rose {F_before:.6g} -> "
                           f"{F_after:.6g} (linearisation residual max "
                           f"{steps[-1]['linearisation_residual_max_sigma']:.3g}"
                           " sigma): the linear tan(gamma) model failed on "
                           "this slice")
        # counted, not inferred (FD probes + chord steps + the final step)
        n_solves = int(last["n"])
        srec["n_solves"] = n_solves
        rec.update(
            structured_mse_status="applied",
            structured_mse_jbs_loop=True,
            structured_mse_n_solves=n_solves,
            structured_mse_n_fd_solves=int(srec["n_fd_solves"]),
            structured_mse_chi2_before=float(chi2_0),
            structured_mse_chi2_after=float(chi2_f),
            structured_mse_chi2_model_after=float(steps[-1]["chi2_model"]),
            structured_mse_chi2_red_before=float(chi2_0) / int(ch["n_active"]),
            structured_mse_chi2_red_after=float(chi2_f) / int(ch["n_active"]),
            structured_mse_residual_sigma_max_abs_before=float(
                np.max(np.abs(z0))),
            structured_mse_residual_sigma_max_abs_after=float(
                np.max(np.abs(z_f))),
            structured_mse_objective_before=float(F_before),
            structured_mse_objective_after_model=float(
                steps[-1]["objective_model"]),
            structured_mse_objective_after=float(F_after),
            structured_mse_linearisation_residual_max_sigma=float(
                steps[-1]["linearisation_residual_max_sigma"]),
            structured_mse_linearisation_residual_rms_sigma=float(
                steps[-1]["linearisation_residual_rms_sigma"]),
            structured_mse_step_log=[dict(s) for s in steps],
            structured_mse_jacobian_shape=[int(v) for v in np.shape(J1)],
            structured_mse_er_terms=mse_er_terms(ch),
            ohm_scale=float(bl.ohm_scale), bs_scale=float(bl.bs_scale),
            structured_coeffs_a=_fl(out["a"]),
            structured_coeffs_b=_fl(out["b"]),
            structured_s_ind_min=float(s_ind.min()),
            structured_s_ind_max=float(s_ind.max()),
            structured_s_bs_min=float(s_bs.min()),
            structured_s_bs_max=float(s_bs.max()),
            structured_s_ind_profile=_fl(s_ind),
            structured_s_bs_profile=_fl(s_bs),
            structured_s_ind_at=_at(s_ind),
            structured_s_bs_at=_at(s_bs),
            structured_sign_pattern=(
                None if out.get("sign_pattern") is None else
                "".join("+" if v else "-" for v in out["sign_pattern"])),
            structured_n_sign_iter=int(out.get("n_sign_iter", 0) or 0),
            structured_structure_ind=float(out["structure_ind"]),
            structured_structure_bs=float(out["structure_bs"]),
            structured_ip_residual=float(out["ip_residual"]),
            structured_ip_residual_pct=float(out["ip_residual_pct"]),
            structured_ip_posterior=abs(float(out["Ip_hybrid"])),
            structured_axis_residual=(None if out.get("axis_residual") is None
                                      else float(out["axis_residual"])),
            structured_constraints=list(out["constraints"]),
            structured_li_predicted=out.get("li_predicted"),
            structured_li_predictor_residual=out.get("li_predictor_residual"),
            structured_residual_sigma_Ip_model=out.get("residual_sigma_Ip"),
            structured_residual_sigma_li_model=out.get("residual_sigma_li"),
            structured_objective=out.get("objective"),
            structured_prior_chi2=out.get("prior_chi2"),
            structured_gn_iterations=out.get("n_iter"),
            # how the soft solve stopped (noise_floor = accepted at the
            # objective's rounding-noise floor, see close_ip_structured_soft)
            # and whether the loop's single logged retry was needed
            structured_gn_stop_reason=out.get("gn_stop_reason"),
            structured_gn_stop=out.get("gn_stop"),
            structured_n_noise_floor_accepts=out.get("n_noise_floor_accepts"),
            structured_closure_retry=out.get("closure_retry"),
            structured_closure_retry_first_error=out.get(
                "closure_retry_first_error"),
        )
        z_ip = None
        if soft and cur.get("ip_sigma"):
            _ip_meas = abs(float(cur["Ip_signed"]))
            _ip_post = abs(float(out["Ip_hybrid"]))
            z_ip = (_ip_post - _ip_meas) / float(cur["ip_sigma"])
            rec.update(structured_ip_measured_residual_pct=(
                100.0 * (_ip_post - _ip_meas) / _ip_meas),
                structured_residual_sigma_Ip=z_ip)
        health = closure_health(bl.ohm_scale, bl.bs_scale, cur["Ip_signed"],
                                cur["c_signed"], cur["ip_ind"],
                                cur["ip_bs"], cur["ip_fix"],
                                soft_ip_residual_sigma=z_ip)
        _prev_now = getattr(bl, "ip_closure", None) or {}
        reasons = list(health["closure_limited_reasons"])
        for why in stage_flags:
            if why not in reasons:
                reasons.append(why)
        for why in flags:
            if why not in reasons:
                reasons.append(why)
            print("[imas SWB-split:ohmic structured] WARNING closure-limited: "
                  + why, flush=True)
        for k, v in health.items():
            rec[k] = v
        rec["closure_limited_reasons"] = tuple(reasons)
        rec["closure_limited"] = bool(reasons)
        if getattr(bl, "ip_closure", None) is not None:
            bl.ip_closure.update(rec)
        print("[imas SWB-split:ohmic structured] MSE (j_BS loop): "
              f"{int(ch['n_active'])} chords, stated orientation (pol "
              f"{sp:+.0f}, tor {st_sign:+.0f}), {n_solves} solves "
              f"({srec['n_fd_solves']} finite-difference + "
              f"{srec['n_passes']} chord/final steps); chi2 "
              f"{chi2_0:.2f} -> {chi2_f:.2f}; objective {F_before:.4g} -> "
              f"{F_after:.4g}; {mse_er_terms(ch)}", flush=True)
        if not srec["converged"]:
            msg = ("self-consistent j_BS loop [MSE chord stage] did not "
                   f"converge: {srec['stop_reason']}"
                   + ("" if q0_pin is None else
                      "; q0-q0_target per step "
                      + "[" + ", ".join(
                          "n/a" if v is None else f"{float(v):+.2e}"
                          for v in srec.get("q0_residual", [])) + "]"
                      + f" (q0_tol {q0_pin.q0_tol:g})"))
            srec["fail_message"] = msg
            print("  [jbs-loop] " + msg, flush=True)
            if settings.get("on_fail", "raise") == "raise":
                raise JBSNotConverged(msg, srec)
        return last["nl"], cur, srec

    @staticmethod
    def _structured_mse_delivered(state, bl, mygs, field_at=None):
        """Judge chi2_MSE on the equilibrium the slice finally DELIVERS.

        The q0/l_i corrector may re-solve after the MSE stage; this re-reads
        the field (no solve) with the frozen orientation, so the record's last
        word -- and the closure-health flags -- are about the delivered
        equilibrium, not an intermediate one.  REPORTING ONLY: nothing here
        solves, retries, or changes an iteration count or a tolerance.

        Recorded: the delivered chi2 (total and reduced), the per-chord
        residuals in sigma (``(tan_gamma_pred - tan_gamma_meas) / sigma_eff``,
        on ``bl.mse_record`` with the other per-chord arrays) and their
        largest magnitude (in ``ip_closure``), and the delivered objective
        ``F_noMSE(delivered closure) + chi2_delivered``.  That objective is
        COMPARABLE with the pre-MSE one (``structured_mse_objective_before``)
        when no corrector re-solve replaced the MSE stage's closure, or on the
        hard channel (whose ``F_noMSE`` is the trust prior alone, a function of
        the coefficients only); on the soft channel a corrector re-solve moved
        the l_i / axis rows the posterior objective is measured against, so it
        is recorded as not comparable and only the chi2 rule applies.

        The slice is FLAGGED closure-limited (``MSE_FLAG_PREFIX``), and
        ``structured_mse_delivered_worse`` set, when the delivered chi2 is
        above the pre-MSE chi2, or the delivered objective (where comparable)
        is above the pre-MSE objective.  A chord the delivered read cannot
        find is flagged the same way, never filled from a stale value.
        """
        import numpy as np

        from .mse import mse_chi2, mse_field_at, mse_tan_gamma
        from .utils import MSE_FLAG_PREFIX, structured_objective_no_mse

        ch = state["mse"]
        sp, st = state["mse_sign"]
        icl = getattr(bl, "ip_closure", None)
        flags = []
        rec = {}
        B, found = (mse_field_at(mygs, ch["R"], ch["Z"]) if field_at is None
                    else field_at(ch["R"], ch["Z"]))
        found = np.asarray(found, dtype=bool).ravel()
        c2 = None
        if not found.all():
            flags.append(
                MSE_FLAG_PREFIX + "chord(s) at input index "
                f"{[int(i) for i in np.asarray(ch['index'])[~found]]} were not "
                "found on the mesh on the delivered equilibrium; the delivered "
                "chi2 is not reported (never from a stale field value)")
            rec.update(structured_mse_chi2_delivered=None,
                       structured_mse_delivered_worse=True)
        else:
            tg = mse_tan_gamma(B, ch, sp, st)
            c2, z = mse_chi2(tg, ch)
            Bouquet._mse_record_put(
                bl, residual_sigma_delivered=np.asarray(z, dtype=float),
                tgamma_pred_delivered=np.asarray(tg, dtype=float))
            c2_before = float(state.get("mse_chi2_before", float("nan")))
            F_before = float(state.get("mse_objective_before", float("nan")))
            out_d = state.get("mse_delivered_out")
            resolved = bool(state.get("mse_corrector_resolved"))
            soft = bool(state.get("soft"))
            comparable = out_d is not None and (not resolved or not soft)
            F_del = (float(structured_objective_no_mse(out_d)) + float(c2)
                     if comparable else None)
            rec.update(
                structured_mse_chi2_delivered=float(c2),
                structured_mse_chi2_red_delivered=float(c2)
                / int(ch["n_active"]),
                structured_mse_residual_sigma_delivered_max_abs=float(
                    np.max(np.abs(z))),
                structured_mse_objective_delivered=F_del,
                structured_mse_objective_delivered_comparable=bool(comparable),
                structured_mse_objective_delivered_note=(
                    "F_noMSE(delivered closure) + chi2_delivered, on the "
                    + ("MSE stage's closure (no corrector re-solve)"
                       if not resolved else
                       "corrector's re-solved closure (hard channel: F_noMSE "
                       "is the trust prior, a function of the coefficients "
                       "only)") if comparable else
                    "not comparable: a soft-channel corrector re-solve moved "
                    "the l_i / axis rows the posterior objective is measured "
                    "against; only the chi2 rule applies"))
            if not (np.isfinite(c2) and c2 <= c2_before):
                flags.append(
                    MSE_FLAG_PREFIX + f"delivered chi2 {float(c2):.6g} is "
                    f"above the pre-MSE chi2 {c2_before:.6g}: the delivered "
                    "equilibrium fits the chords worse than the closure "
                    "without MSE")
            if comparable and not (np.isfinite(F_del) and F_del <= F_before):
                flags.append(
                    MSE_FLAG_PREFIX + f"delivered objective {F_del:.6g} is "
                    f"above the pre-MSE objective {F_before:.6g}")
            rec["structured_mse_delivered_worse"] = bool(flags)
        if icl is not None:
            reasons = list(icl.get("closure_limited_reasons", ()) or ())
            for why in flags:
                if why not in reasons:
                    reasons.append(why)
                print("[imas SWB-split:ohmic structured] WARNING closure-"
                      "limited: " + why, flush=True)
            rec["closure_limited_reasons"] = tuple(reasons)
            rec["closure_limited"] = bool(reasons)
            icl.update(rec)
        return None if c2 is None else float(c2)

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
                                       ip_of=None, roundtrip_gate=None,
                                       record_only=False):
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

        ``record_only=True`` (the self-consistent bootstrap loop): take NO
        corrector step; read back and record every residual and run every
        acceptance flag exactly as above, on the delivered equilibrium.
        ``structured_li_tol`` / ``q0_tol`` are the same bars.  The loop
        re-solves the predictor on refreshed geometry every pass, but with
        the SAME held targets -- the axis row stays at the anchor's requested
        axis current and the l_i row at its target; nothing moves them -- so
        a residual the corrector step would have reduced is, under the loop,
        only measured and flagged, not corrected -- by default.  With
        ``jbs_loop_q0_corrector=True`` the axis row is moved once per pass
        from the q0 measured on that pass's equilibrium (the ``j_ref0'`` rule
        above, applied per pass by :class:`bouquet.jbs_loop.AxisRowPin`) and
        the loop converges only with ``|q0 - q0_target| <= q0_tol``; the l_i
        row stays held either way (its log-gain update is not part of the
        flag).  A NOTICE says which mode is in force whenever such a channel
        runs with the loop on.
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
        # After an APPLIED MSE stage the equilibrium this corrector starts from
        # is the MSE stage's, not the predictor's: its entry readbacks are
        # recorded as "*_mse_stage", and the "*_predictor" names keep the
        # values the MSE stage read off the predictor before it moved
        # anything.  Without an MSE stage every name is exactly as before.
        _entry_mse = bool(state.get("mse_applied"))
        if _entry_mse:
            rec.update(structured_corrector_entry=(
                "MSE-stage equilibrium: the *_mse_stage fields are this "
                "corrector's entry readbacks; the *_predictor fields were read "
                "off the predictor by the MSE stage before it solved anything"))
        if gated:
            q0_tok = float(np.asarray(snap.get_q(psi=state["psi_q"].copy())[1],
                                      dtype=float)[0])
            res = q0_tok - q0_target
            if _entry_mse:
                rec.update(q0_solved_mse_stage=q0_tok,
                           q0_mse_stage_residual=res)
            else:
                rec.update(q0_solved_predictor=q0_tok,
                           q0_predictor_residual=res)
            rec.update(q0_tol=state["q0_tol"], q0_solved=q0_tok,
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
            if _entry_mse:
                rec.update(structured_li_achieved_mse_stage=li_tok,
                           structured_li_residual_mse_stage=li_res)
            else:
                rec.update(structured_li_solved_predictor=li_tok,
                           structured_li_achieved_predictor=li_tok,
                           structured_li_residual_predictor=li_res)
            rec.update(
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
        want_q0 = bool(gated and not q0_unreadable and not record_only
                       and abs(res) > state["q0_tol"])
        want_li = bool(li_target is not None and not li_unreadable
                       and not record_only and abs(li_res) > li_tol)
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
                    axis=axis_now, axis_sigma=None,
                    mse_lin=state.get("mse_lin"))
            return close_ip_structured(
                state["psi_geom"], state["w_lin"], state["c_signed"],
                state["Ip_signed"], state["j_ind"], state["j_BS_swb"],
                state["j_fixed"], basis=state["basis"],
                weights=state["weights"], axis=axis_now,
                li_target=li_row_now, li_kind=li_kind,
                li_geom=state.get("li_geom"),
                sigma_ind_up=state.get("sigma_ind_up"),
                mse_lin=state.get("mse_lin"))

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
                bl.bs_scale_profile = s_bs.copy()
                bl.j_inductive = s_ind * state["j_ind"]
                bl.j_BS = s_bs * state["j_BS_swb"]
                bl.j_phi = bl.j_inductive + bl.j_BS + state["j_fixed"]
                nl_out = solve_jphi(np.asarray(bl.j_phi, dtype=float))
                snap2 = mygs.copy_eq()
                n_solves += 1
                if state.get("mse_applied"):
                    # the delivered profiles now come from THIS closure
                    state["mse_delivered_out"] = out
                    state["mse_corrector_resolved"] = True
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
        if state.get("mse") is not None:
            # the MSE stage's own flags are not closure_health's to re-derive
            from .utils import MSE_FLAG_PREFIX
            _reasons += [r for r in (_prev.get("closure_limited_reasons", ())
                                     or ())
                         if str(r).startswith(MSE_FLAG_PREFIX)
                         and r not in _reasons]

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
                # A REFUSED step delivers the last accepted solve, and ``rec``
                # only carries a posterior once a corrector step has been
                # accepted.  When none was, the delivered profile is the
                # predictor's, so its posterior (already on bl.ip_closure) is
                # the reference -- not Ip_target, which would judge a soft
                # posterior on the hard channel's budget.
                _post = rec.get("structured_ip_posterior")
                if _post is None:
                    _post = (getattr(bl, "ip_closure", None) or {}).get(
                        "structured_ip_posterior")
                _g2 = roundtrip_gate(
                    rec["Ip_hybrid"],
                    posterior=(_post if soft else None),
                    sigma_Ip=(state.get("ip_sigma") if soft else None))
                rec["structured_roundtrip_post_corrector_err_pct"] = float(
                    _g2["err_pct"])
                rec["structured_roundtrip_post_corrector_reference"] = \
                    _g2["reference_name"]
        if state.get("mse") is not None:
            # The MSE stage's solves are extra solves too: n_extra_solves
            # counts EVERY GS solve after the predictor's, and the verdict
            # says where they went.  (No MSE block: nothing here runs.)
            _n_mse = int(state.get("mse_n_solves", 0) or 0)
            _n_cor = int(rec.get("n_extra_solves", 0) or 0)
            _what = state.get("mse_stage_word") or (
                "MSE stage" if state.get("mse_applied")
                else "MSE stage refused")
            rec["structured_corrector_n_solves"] = _n_cor
            rec["n_extra_solves"] = _n_cor + _n_mse
            _v = str(rec.get("sawtooth_verdict", ""))
            _v = _v.replace("(0 extra solves)", "(0 corrector solves)")
            rec["sawtooth_verdict"] = _v.replace(
                "structured predictor",
                f"structured predictor + {_what} ({_n_mse} solve"
                f"{'' if _n_mse == 1 else 's'})", 1)
        if getattr(bl, "ip_closure", None) is not None:
            bl.ip_closure.update(rec)
        return nl_out

    @staticmethod
    def _close_ip_q0_corrector(state, bl, mygs, solve_jphi, ip_of=None,
                               roundtrip_gate=None, record_only=False):
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

        ``record_only=True`` (the self-consistent bootstrap loop): NO Newton
        step is taken.  The loop re-solves the predictor on refreshed geometry
        every pass, but the axis row is NOT moved: ``q0_target`` and the
        axis-current row (``j_requested0``) are computed once on the original
        anchor and held for every pass (no ``on_pass`` update), so at the
        fixed point the loop enforces the requested axis current, not
        ``q0 = q0_target``.  The q0 residual is read back on the delivered
        equilibrium and flagged against the unchanged ``q0_tol`` -- record
        only; the residual the Newton step exists to remove is not corrected.
        That is the DEFAULT (``jbs_loop_q0_corrector=False``).  With
        ``jbs_loop_q0_corrector=True`` the pin acts inside the loop instead:
        the axis row is moved once per pass from the q0 measured on that
        pass's equilibrium (:class:`bouquet.jbs_loop.AxisRowPin`, the
        structured corrector's ``j_ref0' = j_ref0 * q0_solved/q0_target``
        applied per pass; this Newton step is its first-order expansion) and
        the loop converges only when ``|q0 - q0_target| <= q0_tol`` as well;
        this method still runs record-only on the delivered equilibrium, which
        then meets the target.  A NOTICE says which mode is in force whenever
        this channel runs with the loop on.
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
        elif record_only:
            rec.update(n_extra_solves=0, q0_solved=q0_tok, q0_residual=res,
                       sawtooth_verdict="j_BS loop delivered (record-only: "
                                        "no Newton step under the loop; "
                                        "axis row held)")
            print(f"[imas SWB-split:ohmic q0] solved q0={q0_tok:.4f} vs "
                  f"q0_target={q0_target:.4f} (residual {res:+.4f}, tol "
                  f"{state['q0_tol']:g}) -- record-only under the j_BS loop "
                  "(axis row held, no Newton step); recorded and flagged "
                  "against q0_tol, not corrected", flush=True)
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
                    bl.bs_scale_profile = None
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
        coord = getattr(bl, "coord", coords.PSI)
        psi_pad = 1e-3
        from .physics import ELEMENTARY_CHARGE as EC

        # init psi from the LCFS shape parameters
        R0, Z0, a, kappa, delta = _shape_from_boundary(self._boundary_RZ)
        mygs.init_psi(R0, Z0, a, kappa, delta)
        self._seed_coil_init(mygs)
        # the pressure handed to the solver (bouquet.edge_pressure)
        _edge = resolve_edge_pressure(self.config.generation)

        # kinetic profiles + total pressure on the equilibrium grid (IMAS shares
        # psi_N between the kinetic and current grids).
        def k2e(arr):
            # PCHIP regrid (shared helper) -- must match the draw path
            from .utils import pchip_interp
            return pchip_interp(bl.psi_N_kinetic, arr, psi_N)

        ne, te, ni, ti = k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti)
        Zeff = np.clip(k2e(bl.Zeff), 1.0, None)
        p_total = EC * (ne * te + ni * ti)
        # Per-component copies for the report-only core-pressure hollowness
        # record; the composition arithmetic is untouched (so bit-identical).
        _pc = {"electron_thermal": EC * ne * te, "ion_thermal": EC * ni * ti}
        if bl.p_fast is not None:
            _pc["fast"] = k2e(bl.p_fast)
            p_total = p_total + k2e(bl.p_fast)
        # Impurity (carbon) thermal pressure + diff anchor so the baseline forward
        # solve uses the full dd equilibrium.pressure (mirrors generate_bouquet /
        # perturb_kinetic_equilibrium; single-ion e*(ne*Te+ni*Ti) omits carbon).
        if getattr(bl, "Z_imp", None):
            from .physics import impurity_pressure
            _zf = getattr(bl, "z_fast", None)
            _ne_th = ne if _zf is None else np.maximum(ne - k2e(_zf), 0.0)
            _pc["impurity"] = impurity_pressure(_ne_th, ni, ti, bl.Z_imp)
            p_total = p_total + impurity_pressure(_ne_th, ni, ti, bl.Z_imp)
        if getattr(bl, "p_diff", None) is not None:
            _pc["anchor_diff"] = k2e(bl.p_diff)
            p_total = p_total + k2e(bl.p_diff)

        def solve_jphi(j_phi):
            ffp = coords.oft_prof("jphi-linterp", psi_N, np.asarray(j_phi, dtype=float), coord)
            nl_its = -1
            for _pass in range(2):   # 2nd pass refines the jphi-linterp flux scaling
                psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
                pp_y = solver_pprime(psi_N, p_total, psi_range, _edge)
                mygs.set_targets(Ip=bl.Ip_target, pax=solver_pax(p_total, _edge))
                mygs.set_profiles(
                    pp_prof=coords.oft_prof("linterp", psi_N, pp_y, coord), ffp_prof=ffp,
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
        # NOTE that is a RESIDUAL, j_tor - j_BS - j_NBI - j_RF - j_other
        # (imas.py), NOT to_toroidal(j_ohmic): every driven core_sources
        # current is held fixed in its own channel, but the dd's unattributed
        # current, j_total - (ohmic + bootstrap + sources), lands in this
        # component and is what ohm_scale rescales.  Recompute the
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
        # self-consistent loop only: the request the delivered equilibrium was
        # solved from (recorded by the loop's pass solve), for the ONE-state
        # storage at the end of this method.  None on the legacy path.
        _delivery = None
        # Same refusal as _validate_workflow, here for baseline-only callers
        # (prepare_baseline() without generate()): a non-default
        # closure_channel outside the ohmic hybrid split is never read.
        # Same downgrade too: workflow='custom' / allow_unsafe_workflow turn
        # it into a printed WARN, so validation and the baseline solve agree.
        _gc0 = self.config.generation
        _chan0 = str(getattr(_gc0, "closure_channel", "bootstrap"))
        if _chan0 != "bootstrap" and (
                str(_gc0.jBS_baseline_mode) != "ohmic"
                or not bool(_gc0.recalculate_j_BS)):
            _msg0 = (
                f"closure_channel={_chan0!r} is only read when "
                "jBS_baseline_mode='ohmic' and recalculate_j_BS=True "
                f"(have jBS_baseline_mode={str(_gc0.jBS_baseline_mode)!r}, "
                f"recalculate_j_BS={bool(_gc0.recalculate_j_BS)}); "
                "it would otherwise be silently ignored. Set "
                "jBS_baseline_mode='ohmic' or leave closure_channel at "
                "'bootstrap'.")
            if (str(getattr(_gc0, "workflow", "")) == "custom"
                    or bool(getattr(_gc0, "allow_unsafe_workflow", False))):
                print("WARN: " + _msg0 + " (workflow='custom': continuing; "
                      "the channel is NOT applied)")
            else:
                raise ValueError(_msg0)
        # swb_seed="source": SWB inputs from the source split as read (before
        # any closure touches bl), shared by the baseline split, the draws and
        # the sigma=0 check.
        bl.swb_seed_profile = bl.swb_jphi_fixed = bl.swb_jphi_saw = None
        # resolved here, capability-checked at the SWB call (PR #70 review
        # B4: the "source" default raised on a toolkit without jphi_fixed
        # even when no SWB call ever runs)
        _seed_mode, self._swb_seed_record = self._resolve_swb_seed()
        if _seed_mode == "source":
            self._swb_source_split(psi_N)
        if self.config.generation.recalculate_j_BS:
            from .TokaMaker_interface import smooth_jbs_transition
            from .sampling import calc_cylindrical_li_proxy
            from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
            from scipy.optimize import brentq

            gc = self.config.generation
            iso = bool(gc.isolate_edge_jBS); mode = str(gc.jBS_baseline_mode)
            j_ind = np.asarray(bl.j_inductive, dtype=float)   # FUSE ohmic (kept)
            j_BS_src = np.asarray(bl.j_BS, dtype=float)        # source bootstrap (FUSE)
            FUSE_tot = np.asarray(bl.j_phi, dtype=float)       # source total (j_tor)
            j_fixed = FUSE_tot - j_ind - j_BS_src              # = j_NBI + j_RF + j_other
            # 'ohmic' mode: freeze the ANCHOR geometry now. solve_with_bootstrap
            # iterates its own GS solves (generic inductive seed + its bootstrap)
            # and leaves mygs on a different equilibrium; integrating FUSE's
            # profile on that landed geometry read +31% of Ip on an ohmic-ramp
            # slice (vs +0.8% on the anchor) and collapsed the closure. Every Ip
            # integral in the ohmic branch is taken on this snapshot.
            # Self-consistent bootstrap loop settings (validated).  ON (the
            # default, GenerationConfig.jbs_self_consistent=True) runs
            # _imas_jbs_loop; OFF runs the historical frozen-SWB block below
            # unchanged.
            from .jbs_loop import jbs_settings as _jbs_settings
            _jbs = _jbs_settings(gc)
            _loop_on = bool(_jbs["enabled"])
            _delivery = ({"mode": str(mode), "request": None} if _loop_on
                         else None)

            def _closure_geometry(_src_label):
                """FSA closure geometry of the equilibrium mygs holds NOW.

                ``{eq: copy_eq() snapshot, geom, inv_r2_src, Ip_anchor}`` --
                the anchor (called once before any bootstrap work, as it
                always was) and, with ``jbs_self_consistent``, every loop
                pass's current iterate.  <1/R^2> falls back to the traced
                contour quadrature on OFT builds whose get_q lacks it.
                """
                from .utils import fsa_current_geometry as _fcg
                from .physics import capture_equilibrium_fsa as _cef
                _anchor = {"eq": mygs.copy_eq()}
                _anchor["geom"] = _fcg(
                    _anchor["eq"],
                    np.asarray(coords.psi_at(_anchor["eq"], psi_N, coord), dtype=float),
                    psi_pad=psi_pad)
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
                    _anchor["inv_r2_src"] = "capture_equilibrium_fsa contour quadrature (" + _src_label + ")"
                _anchor["Ip_anchor"] = abs(float(mygs.get_stats(lcfs_pad=psi_pad)["Ip"]))
                return _anchor

            def _ohmic_close(_anchor, j_BS_swb, ratio, _pass=None):
                """The 'ohmic'-mode Ip closure on ONE geometry + bootstrap.

                ``_anchor`` is a :func:`_closure_geometry` context (the
                frozen pre-SWB anchor on the legacy path; the current
                iterate on every pass of the self-consistent loop),
                ``j_BS_swb`` the base bootstrap profile the channel scales,
                ``ratio`` its peak over the source bootstrap's (recorded).
                ``_pass`` (loop only) carries the held q0 reference (and,
                with ``jbs_loop_q0_corrector=True``, the pinned ``axis_row``
                of this pass).  Sets ``bl.j_*`` / ``bl.ip_closure`` exactly
                as the inline block it was lifted from did and returns
                ``(q0_state, structured_state, closure_ctx)``.
                """
                _q0_state = None
                _structured_state = None
                _pq0 = None if _pass is None else _pass.get("q0_ref")
                _pxr = None if _pass is None else _pass.get("x_retry")
                if (_pass is not None and _pq0 is not None
                        and _pass.get("axis_row") is not None):
                    # jbs_loop_q0_corrector: this pass's pinned axis row
                    _pq0 = dict(_pq0, axis_row=float(_pass["axis_row"]))
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
                # it reads its input as an area density (see the utils.py module
                # note). And NOT the cylindrical
                # l_i proxy (1/<R> for <1/R>; +7.75% on the FUSE validation
                # case). Both are still
                # evaluated and RECORDED below so the biases stay visible.
                from .utils import (Ip_fsa_integral, Ip_fsa_weights,
                                    eq_jphi_profile, close_ip,
                                    closure_sign_convention)
                from .sampling import get_li_proxy_geometry
                from scipy import integrate as _integ
                _eq_snap = _anchor["eq"]          # frozen BEFORE solve_with_bootstrap
                _psi_ip = np.asarray(coords.psi_at(_eq_snap, psi_N, coord), dtype=float)
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
                _geo_cyl = get_li_proxy_geometry(_eq_snap, psi_N.size, psi_pad, psi_N, coord)
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
                # The IMAS reader already brought every source current into
                # the anchor's positive frame (Baseline.source_current_sign),
                # so sgn is +1 on any dd whose currents agree with its own ip.
                # -1 here means the components disagree with the frame they
                # were normalised to -- report it rather than let the pairing
                # above quietly re-sign a mixed-frame closure.
                if sgn < 0.0:
                    print("[imas SWB-split:ohmic] WARNING: the linear component "
                          "total is NEGATIVE after the reader's current-"
                          "orientation normalisation (source_current_sign="
                          f"{float(getattr(bl, 'source_current_sign', 1.0)):+.0f}): "
                          f"ohm={ip_ind / 1e6:+.4f} jBS={ip_bs / 1e6:+.4f} "
                          f"fixed={ip_fix / 1e6:+.4f} MA -- the source's "
                          "current profiles disagree with its plasma current",
                          flush=True)
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
                            close_ip, q0_ref=_pq0)
                elif _chan == "structured":
                    _s_ind, _s_bs, ohm_scale, bs_scale, _q0_extra, \
                        _structured_state = self._close_ip_structured_predictor(
                            gc, bl, _eq_snap, _geom, _probe, psi_N,
                            j_ind, j_BS_swb, j_fixed, FUSE_tot,
                            sgn, Ip_t, _w_lin, _c_signed,
                            ip_ind, ip_bs, ip_fix,
                            psi_pad=psi_pad, pprime_sign=_pps,
                            q0_ref=_pq0, x_retry=_pxr)
                else:
                    ohm_scale, bs_scale = close_ip(
                        _chan, _Ip_signed, _c_signed, ip_ind, ip_bs, ip_fix)
                bl.jBS_diff = None
                bl.bs_scale = float(bs_scale)
                bl.ohm_scale = float(ohm_scale)
                bl.bs_scale_profile = (None if _s_bs is None
                                       else np.asarray(_s_bs, dtype=float).copy())
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
                    # What the reader multiplied the source currents by to
                    # bring them into this (positive) frame; -1 = reversed-Ip
                    # source.  current_direction_sign above is read AFTER it.
                    source_current_sign=float(getattr(bl, "source_current_sign", 1.0)),
                    source_current_sign_origin=getattr(
                        bl, "source_current_sign_origin", None),
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
                return _q0_state, _structured_state, dict(
                    ip_signed=_ip_signed, Ip_t=Ip_t,
                    ip_roundtrip_gate=ip_roundtrip_gate)

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
                _anchor = _closure_geometry("anchor, pre-SWB")
            def _imas_jbs_loop():
                """The IMAS baseline with ``jbs_self_consistent=True``.

                Every ``jBS_baseline_mode`` and closure channel; returns the
                ``nl_its`` of the delivered solve.  See
                :mod:`bouquet.jbs_loop` for the kernel and
                docs/physics-notes.md for the method.
                """
                from .physics import evaluate_jBS
                from .jbs_loop import (flag_reason, jsonable,
                                       residual_weights, run_jbs_loop,
                                       oft_build_info)
                from .utils import li_achieved, ip_roundtrip_gate as _gate_fn
                from .physics import EVALUATE_JBS_VERSION

                Ip_abs = abs(float(bl.Ip_target))
                _init = str(_jbs["init"])
                _corr_on = bool(getattr(gc, "imas_corrective_jphi", False))

                def _redl(eq):
                    j, d = evaluate_jBS(eq, psi_N, ne, te, ni, ti, Zeff,
                                        psi_pad=psi_pad, isolate_edge=iso,
                                        smooth_axis=True, coord=coord)
                    if gc.floor_j_BS:
                        j = np.clip(j, 0.0, None)
                    return j, d

                def _pass_solve(j_phi):
                    """One loop solve: the SAME 2-pass solve_jphi the tail
                    uses, plus the opt-in corrective iteration when it is on,
                    so the delivered equilibrium is always the last pass."""
                    j_phi = np.asarray(j_phi, dtype=float)
                    nl = solve_jphi(j_phi)
                    _delivery["request"] = j_phi.copy()
                    if _corr_on:
                        from .TokaMaker_interface import \
                            _corrective_jphi_iteration
                        _pr = mygs.psi_bounds[1] - mygs.psi_bounds[0]
                        _pp_y = solver_pprime(psi_N, p_total, _pr, _edge)
                        _cr = _corrective_jphi_iteration(
                            mygs, psi_N, j_phi,
                            coords.oft_prof("linterp", psi_N, _pp_y, coord),
                            abs(bl.Ip_target), solver_pax(p_total, _edge),
                            1e-3,
                            min_iters=2, max_iters=8, rtol=0.02,
                            verbose=True, damping=0.5, protect_state=True,
                            return_request=True, coord=coord)
                        if _cr[3] is not None:
                            # the corrective iteration's landed state: the
                            # request one solve reproduces it from
                            _delivery["request"] = np.asarray(_cr[3], float)
                    return nl

                def _li3(eq):
                    return float(li_achieved(eq, li_kind="li_3",
                                             psi_pad=psi_pad)[0])

                def _swb_legacy():
                    """jbs_init="swb": the legacy SWB result as the initial
                    guess (A/B only), with mygs put back on the anchor."""
                    _snap_a = mygs.copy_eq()
                    _kw = dict(scale_jBS=1.0, isolate_edge_jBS=iso,
                               diagnostic_plots=False, verbose=False,
                               **gc.bootstrap_kwargs)
                    _seed = coords.swb_seed(psi_N, coords.psi_at(
                        mygs, psi_N, coord))
                    _grid = coords.swb_grid_kwargs(psi_N, coord)
                    _why = (None if _grid else "this OFT build's "
                            "solve_with_bootstrap has no grid argument")
                    try:
                        _swb = solve_with_bootstrap(
                            mygs, ne, te, ni, ti, Zeff, bl.Ip_target, _seed,
                            **_grid, **_kw)
                        _passed = bool(_grid)
                    except ValueError as _e:
                        if not _grid or coord != coords.PSI:
                            raise
                        _why = f"SWB refused its grid argument: {_e}"
                        mygs.replace_eq(source_eq=_snap_a)
                        _passed = False
                        _swb = solve_with_bootstrap(
                            mygs, ne, te, ni, ti, Zeff, bl.Ip_target,
                            coords.swb_seed(np.linspace(0.0, 1.0,
                                                        psi_N.size)),
                            **_kw)
                    # SWB's j_BS is TokaMaker jphi already, on its grid
                    _j = smooth_jbs_transition(np.asarray(
                        _swb["isolated_j_BS"], dtype=float))
                    if gc.floor_j_BS:
                        _j = np.clip(_j, 0.0, None)
                    mygs.replace_eq(source_eq=_snap_a)
                    print("[imas jbs-loop] init: legacy solve_with_bootstrap "
                          + ("with psi_N=" if _passed else
                             f"WITHOUT psi_N= ({_why})")
                          + "; the fixed point does not depend on the init",
                          flush=True)
                    return _j, dict(swb_psi_N_passed=bool(_passed),
                                    swb_psi_N_reason=_why)

                def _finish(rec, nl):
                    rec = jsonable(rec)
                    metrics = dict(bl.li_metrics or {})
                    metrics["jbs_loop"] = rec
                    bl.li_metrics = metrics
                    if getattr(bl, "ip_closure", None) is not None:
                        bl.ip_closure["jbs_loop"] = rec
                        bl.ip_closure["jbs_converged"] = bool(
                            rec.get("converged", False))
                        if not rec.get("converged", False):
                            _r = list(bl.ip_closure.get(
                                "closure_limited_reasons", ()) or ())
                            _why = flag_reason(rec)
                            if _why not in _r:
                                _r.append(_why)
                            bl.ip_closure["closure_limited_reasons"] = \
                                tuple(_r)
                            bl.ip_closure["closure_limited"] = True
                    return nl

                # ================= diff: baseline pinned, draws iterate =====
                if mode == "diff":
                    _jphi_solve = np.asarray(FUSE_tot, dtype=float)
                    if getattr(bl, "jphi_diff", None) is not None:
                        _jphi_solve = _jphi_solve + k2e(bl.jphi_diff)
                    nl = _pass_solve(_jphi_solve)
                    # The offset is taken on the DELIVERED baseline geometry
                    # (the equilibrium every sigma=0 draw converges back to),
                    # so a sigma=0 draw reproduces the source j_BS exactly.
                    _snap = mygs.copy_eq()
                    j_red, d_red = _redl(_snap)
                    bl.jBS_diff = j_BS_src - j_red
                    bl.j_BS = j_red
                    bl.j_phi = FUSE_tot
                    bl.bs_scale = 1.0
                    _ratio = j_red.max() / max(j_BS_src.max(), 1.0)
                    print(f"[imas jbs-loop:diff] source total preserved; "
                          f"jBS_diff = source j_BS - evaluate_jBS(delivered "
                          f"baseline) min/max={bl.jBS_diff.min():.2e}/"
                          f"{bl.jBS_diff.max():.2e}; Redl/source jBS peak="
                          f"{_ratio:.3f}; the draws iterate", flush=True)
                    rec = dict(
                        enabled=True, label="imas baseline diff",
                        init=_init, grid="psi_N native (source grid)",
                        n_passes=0, converged=True, jbs_converged=True,
                        stop_reason=("diff mode: the baseline total is "
                                     "pinned to the source, so the baseline "
                                     "needs no loop; every draw iterates"),
                        jBS_diff_definition=(
                            "source j_BS - evaluate_jBS(delivered baseline "
                            "equilibrium, source kinetics): a pure model "
                            "offset on the baseline geometry, frozen in "
                            "psi_N labels"),
                        I_BS=[float(d_red["I_BS"])],
                        redl_over_source_jBS_peak=float(_ratio),
                        evaluate_jBS_version=EVALUATE_JBS_VERSION,
                        oft_build=oft_build_info(),
                        tolerances=None, wall_s=0.0)
                    return _finish(rec, nl)

                # ================= E_0 and the initial guess ================
                _snap0 = mygs.copy_eq()
                _swb_info = None
                if _init == "swb":
                    j0, _swb_info = _swb_legacy()
                else:
                    j0, _d0 = _redl(_snap0)

                # ================= rescale ==================================
                if mode == "rescale":
                    st = {}

                    def _step(jbs, k, relax=None):
                        _lpx = ({} if coord == coords.PSI
                                else dict(x=psi_N, coord=coord))
                        tgt = calc_cylindrical_li_proxy(mygs, FUSE_tot,
                                                        psi_pad, **_lpx)
                        _f = lambda s: calc_cylindrical_li_proxy(
                            mygs, j_ind + s * jbs + j_fixed, psi_pad,
                            **_lpx) - tgt
                        try:
                            scale = float(brentq(_f, 0.2, 4.0, xtol=1e-4))
                            _sc_ok = True
                        except Exception as _sc_exc:
                            # the historical fallback, unchanged (scale 1.0,
                            # same bracket) -- but never silently: printed,
                            # recorded per pass, and warned after the loop
                            scale = 1.0
                            _sc_ok = False
                            st.setdefault("fallbacks", []).append(dict(
                                k=int(k), error=(f"{type(_sc_exc).__name__}: "
                                                 f"{str(_sc_exc)[:200]}")))
                            print(f"[imas jbs-loop:rescale] WARNING pass "
                                  f"{k + 1}: the l_i-proxy root find failed "
                                  f"on the bracket [0.2, 4.0] "
                                  f"({type(_sc_exc).__name__}: "
                                  f"{str(_sc_exc)[:160]}); FALLING BACK to "
                                  "bootstrap scale 1.0 for this pass",
                                  flush=True)
                        bl.jBS_diff = None
                        bl.bs_scale = scale
                        bl.j_BS = scale * np.asarray(jbs, dtype=float)
                        bl.j_phi = j_ind + bl.j_BS + j_fixed
                        _js = np.asarray(bl.j_phi, dtype=float)
                        if getattr(bl, "jphi_diff", None) is not None:
                            _js = _js + k2e(bl.jphi_diff)
                        st["nl"] = _pass_solve(_js if relax is None
                                               else relax(_js))
                        snap = mygs.copy_eq()
                        w, x, _wk = residual_weights(snap, psi_N, psi_pad,
                                                     coord=coord)
                        st.setdefault("scales", []).append(float(scale))
                        st.setdefault("scale_bracketed", []).append(_sc_ok)
                        print(f"[imas jbs-loop:rescale] pass {k + 1}: "
                              f"scale={scale:.4f}", flush=True)
                        return dict(w=w, x=x, li=_li3(snap), snap=snap)

                    res = run_jbs_loop(
                        j0, _step, lambda m: _redl(m["snap"])[0], _jbs,
                        Ip=Ip_abs, meas0=dict(li=_li3(_snap0)),
                        gate_li=True, gate_q0=False,
                        label="imas baseline rescale", init=_init)
                    rec = dict(res["record"])
                    rec.update(rescale_scales=st.get("scales"),
                               rescale_bracketed=st.get("scale_bracketed"),
                               J_final_minus_used_max=float(np.max(np.abs(
                                   res["J_final"] - res["jbs_used"]))))
                    _fb = list(st.get("fallbacks") or [])
                    rec["rescale_fallback_passes"] = [f["k"] + 1 for f in _fb]
                    rec["rescale_fallback_errors"] = [f["error"] for f in _fb]
                    _br = list(st.get("scale_bracketed") or [])
                    rec["rescale_delivered_on_fallback"] = bool(
                        _br and not _br[-1])
                    if _fb:
                        import warnings as _w
                        _msg = (
                            "IMAS rescale baseline (j_BS loop): the l_i-proxy "
                            "root find failed on pass(es) "
                            f"{rec['rescale_fallback_passes']} and the "
                            "bootstrap scale FELL BACK to 1.0 there"
                            + (" -- INCLUDING the delivered pass, so the "
                               "delivered bs_scale=1.0 is a fallback, not a "
                               "fit" if rec["rescale_delivered_on_fallback"]
                               else "") + " (recorded as "
                            "rescale_fallback_passes / rescale_fallback_errors "
                            "in the loop record)")
                        print("[imas jbs-loop:rescale] WARNING " + _msg,
                              flush=True)
                        _w.warn(_msg, RuntimeWarning, stacklevel=2)
                    if _swb_info:
                        rec.update(_swb_info)
                    return _finish(rec, st["nl"])

                # ================= ohmic (every closure channel) ============
                if mode != "ohmic":
                    raise ValueError(f"unknown jBS_baseline_mode {mode!r} "
                                     "(expected 'diff', 'rescale' or "
                                     "'ohmic')")
                from .utils import (eq_jphi_profile, unrenormalise_q0,
                                    q0_gate_admits)
                _chan = str(getattr(gc, "closure_channel", "bootstrap"))
                # ---- the q0 reference: ONCE, on the ORIGINAL anchor --------
                # (a data-derived target, exactly what the predictor computes
                # on its first call; handed to every pass so it never moves)
                _geom_a = _anchor["geom"]
                _psi_q = np.ascontiguousarray(
                    np.asarray(_geom_a["psi_q"], dtype=float))
                _psi_g = np.asarray(_geom_a["psi_N"], dtype=float)
                _probe_a = eq_jphi_profile(_geom_a, "jphi-linterp",
                                           eq=_anchor["eq"])
                _q0_anchor = float(np.asarray(_anchor["eq"].get_q(
                    psi=_psi_q.copy())[1], dtype=float)[0])
                _j_ach0 = float(np.asarray(_probe_a, dtype=float)[0])
                _j_req0 = float(np.interp(_psi_q[0], _psi_g,
                                          np.asarray(FUSE_tot, dtype=float)))
                _q0_target = unrenormalise_q0(_q0_anchor, _j_ach0, _j_req0)
                q0_ref = dict(q0_target=_q0_target, q0_anchor=_q0_anchor,
                              j_achieved0=_j_ach0, j_requested0=_j_req0)
                _saw = dict(getattr(bl, "sawtooth", None) or {})
                _gated, _ = q0_gate_admits(
                    bool(_saw.get("active")), _saw.get("q0_dd"), _q0_target,
                    float(getattr(gc, "q0_gate", 1.1)))
                axis_active = bool(_gated and _chan in ("sawtooth_bootstrap",
                                                        "structured"))
                # ---- the q0 pin under the loop (jbs_loop_q0_corrector) ------
                # OFF (default): record-only, the axis row held.  ON: the row
                # is moved once per pass from the measured q0 and the loop
                # additionally requires |q0 - q0_target| <= q0_tol.
                _pin_flag = bool(getattr(gc, "jbs_loop_q0_corrector", False))
                _pin = None
                if axis_active and _pin_flag:
                    from .jbs_loop import AxisRowPin
                    _pin = AxisRowPin(_q0_target,
                                      float(getattr(gc, "q0_tol", 0.01)),
                                      _j_req0, label=f"imas baseline {_chan}")
                if axis_active:
                    if _pin is None:
                        print(f"[imas jbs-loop] NOTICE: closure_channel="
                              f"'{_chan}' pins q0 via the axis row, and "
                              "under the self-consistent j_BS loop its q0 "
                              "corrector is RECORD-ONLY -- the axis row is "
                              "held at the anchor's requested axis current, "
                              "no Newton step is taken, and the q0 residual "
                              "is only flagged against q0_tol "
                              "(jbs_loop_q0_corrector=False, the default; "
                              "set it True for the acting pin)", flush=True)
                    else:
                        print(f"[imas jbs-loop] NOTICE: closure_channel="
                              f"'{_chan}' pins q0 via the axis row, and the "
                              "q0 pin ACTS under the self-consistent j_BS "
                              "loop (jbs_loop_q0_corrector=True): the axis "
                              "row is moved once per pass from the q0 "
                              "measured on that pass's equilibrium, and "
                              "convergence additionally requires |q0 - "
                              f"q0_target| <= q0_tol ({_pin.q0_tol:g}; "
                              f"q0_target {_q0_target:.4f})", flush=True)
                elif _pin_flag:
                    print("[imas jbs-loop] NOTICE: jbs_loop_q0_corrector=True "
                          "has no effect on this slice: "
                          + (f"closure_channel='{_chan}' has no axis row"
                             if _chan not in ("sawtooth_bootstrap",
                                              "structured")
                             else "the sawtooth gate rejected the axis row")
                          + " (nothing pins q0)", flush=True)
                _li_target = getattr(gc, "structured_li_target", None)
                _li_kind = (str(getattr(gc, "structured_li_kind", "li_1"))
                            if (_chan == "structured"
                                and _li_target is not None) else "li_3")

                def _q0_of(eq):
                    return float(np.asarray(eq.get_q(psi=_psi_q.copy())[1],
                                            dtype=float)[0])

                def _li_of(eq):
                    return float(li_achieved(eq, li_kind=_li_kind,
                                             psi_pad=psi_pad)[0])

                st = dict(ctx=_anchor, pass_log=[],
                          q0s=None, ss=None, oc=None, nl=None, pass0=None)

                def _pin_gate(rec):
                    """``jbs_loop_q0_corrector=True``: the q0 target checked on
                    the DELIVERED equilibrium (after the MSE stage, when there
                    is one) and recorded in the closure-health block.  Without
                    an MSE stage the loop's own criterion already held on this
                    very equilibrium; the check is kept so no path can deliver
                    a q0 outside ``q0_tol`` as converged.  A miss fails exactly
                    as the loop fails (raise, or flag under "flag")."""
                    from .jbs_loop import JBSNotConverged
                    _icl = bl.ip_closure if bl.ip_closure is not None else {}
                    _r = _icl.get("q0_residual")
                    try:
                        _r = float(_r)
                    except (TypeError, ValueError):
                        _r = float("nan")
                    if not np.isfinite(_r):
                        # no corrector readback: measure it here, the same way
                        _r = _q0_of(mygs.copy_eq()) - _q0_target
                    _ok = bool(np.isfinite(_r) and abs(_r) <= _pin.q0_tol)
                    _over = (float(abs(_r) / _pin.q0_tol) if np.isfinite(_r)
                             else None)
                    _pr = _pin.record()
                    _pr.update(delivered_q0_residual=(float(_r)
                                                      if np.isfinite(_r)
                                                      else None),
                               delivered_q0_residual_over_tol=_over,
                               delivered_within_q0_tol=_ok)
                    rec["q0_pin"] = _pr
                    if bl.ip_closure is not None:
                        bl.ip_closure.update(
                            q0_pin_mode=("acting under the j_BS loop "
                                         "(jbs_loop_q0_corrector=True): axis "
                                         "row moved once per pass from the "
                                         "measured q0"),
                            q0_pin_acted=bool(_pin.n_updates > 0),
                            q0_pin_n_row_updates=int(_pin.n_updates),
                            q0_pin_axis_row_initial=float(_pin.row0),
                            q0_pin_axis_row_final=float(_pin.row),
                            q0_residual_over_tol=_over,
                            q0_pin_delivered_within_tol=_ok)
                    _ovs = "n/a" if _over is None else f"{_over:.3f}"
                    print(f"[imas jbs-loop] q0 pin: delivered q0 - q0_target "
                          f"= {_r:+.4e} ({_ovs} x q0_tol {_pin.q0_tol:g}) "
                          "after "
                          f"{_pin.n_updates} axis-row update(s); axis row "
                          f"{_pin.row0 / 1e6:.5f} -> {_pin.row / 1e6:.5f} "
                          "MA/m^2", flush=True)
                    if _ok or not rec.get("converged", False):
                        return
                    _why = (f"q0 pin (jbs_loop_q0_corrector=True): the "
                            f"delivered equilibrium misses q0_target by "
                            f"{_r:+.4e} (> q0_tol {_pin.q0_tol:g}) although "
                            "the loop's other criteria held")
                    rec["converged"] = False
                    rec["jbs_converged"] = False
                    rec["stop_reason"] = _why
                    rec["fail_message"] = ("self-consistent j_BS loop [imas "
                                           f"baseline ohmic/{_chan}] did not "
                                           f"converge: {_why}")
                    print("  [jbs-loop] " + rec["fail_message"], flush=True)
                    if _jbs.get("on_fail", "raise") == "raise":
                        from .jbs_loop import jsonable as _js
                        raise JBSNotConverged(rec["fail_message"], _js(rec))

                def _step(jbs, k, relax=None):
                    jbs = np.asarray(jbs, dtype=float)
                    _ratio = jbs.max() / max(j_BS_src.max(), 1.0)
                    q0s, ss, oc = _ohmic_close(
                        st["ctx"], jbs, _ratio,
                        _pass=dict(q0_ref=q0_ref, k=k,
                                   x_retry=st.get("x_prev"),
                                   axis_row=(None if _pin is None
                                             else _pin.row)))
                    _icl0 = bl.ip_closure or {}
                    if _icl0.get("structured_coeffs_a") is not None:
                        st["x_prev"] = np.concatenate([
                            np.asarray(_icl0["structured_coeffs_a"], float),
                            np.asarray(_icl0["structured_coeffs_b"], float)])
                    # the pass solves the closure's current relaxed against
                    # the previous pass's solved current (jbs_relax_current);
                    # bl.j_phi stays the closure's own assembly
                    _j_solved = np.array(bl.j_phi if relax is None
                                         else relax(bl.j_phi), dtype=float,
                                         copy=True)
                    st["j_solved"] = _j_solved
                    st["nl"] = _pass_solve(_j_solved)
                    snap = mygs.copy_eq()
                    ctx_new = _closure_geometry(
                        f"j_BS loop pass {k + 1}")
                    g = ctx_new["geom"]
                    w = (g["dV_dpsi"] / (2.0 * np.pi) * g["dpsi_dpsiN"]
                         * g["inv_R2"] / g["inv_R"])
                    meas = dict(w=w, x=np.asarray(g["psi_N"], dtype=float),
                                li=_li_of(snap), snap=snap,
                                q0=(_q0_of(snap) if axis_active else None))
                    if _pin is not None:
                        # the axis value of the current this pass SOLVED
                        # (the relaxed blend, not the row): what the q0 pin
                        # moves the next row from
                        meas["axis_current_solved"] = float(np.interp(
                            _psi_q[0], _psi_g, _j_solved))
                    _icl = bl.ip_closure or {}
                    st["pass_log"].append(dict(
                        k=k, ohm_scale=float(bl.ohm_scale),
                        bs_scale=float(bl.bs_scale),
                        j_ref0_used=_icl.get("j_ref0_used"),
                        fsa_roundtrip_err_pct=_icl.get(
                            "fsa_roundtrip_err_pct"),
                        closure_limited=bool(_icl.get("closure_limited",
                                                      False)),
                        closure_stop_reason=_icl.get(
                            "structured_gn_stop_reason"),
                        closure_noise_floor_accepts=_icl.get(
                            "structured_n_noise_floor_accepts"),
                        closure_gn_stop=_icl.get("structured_gn_stop"),
                        closure_retry=_icl.get("structured_closure_retry"),
                        closure_retry_first_error=_icl.get(
                            "structured_closure_retry_first_error")))
                    if k == 0:
                        # the "predictor" readbacks the corrector bookkeeping
                        # reports: the first pass, measured the corrector's
                        # own way
                        p0 = dict(q0=(_q0_of(snap) if (ss or q0s) else None))
                        if ss is not None and ss.get("li_target") is not None:
                            p0["li"] = float(li_achieved(
                                snap, li_kind=ss["li_kind"], psi_pad=psi_pad,
                                perimeter=float(ss["li_geom"]["perimeter"]))[0])
                        st["pass0"] = p0
                    st.update(ctx=ctx_new, q0s=q0s, ss=ss, oc=oc)
                    return meas

                _meas0 = dict(li=_li_of(_snap0),
                              q0=(_q0_of(_snap0) if axis_active else None))
                res = run_jbs_loop(
                    j0, _step, lambda m: _redl(m["snap"])[0], _jbs,
                    Ip=Ip_abs, meas0=_meas0, gate_li=True,
                    gate_q0=axis_active,
                    label=f"imas baseline ohmic/{_chan}", init=_init,
                    q0_pin=_pin)
                rec = dict(res["record"])
                rec.update(closure_channel=_chan, axis_row_active=axis_active,
                           axis_row=("held at the source's requested axis "
                                     "current (a data-derived target, section "
                                     "2.4); q0 checked against q0_tol on the "
                                     "delivered equilibrium") if axis_active
                           else None,
                           li_kind_measured=_li_kind,
                           q0_reference=dict(q0_ref),
                           pass_closure_log=st["pass_log"],
                           J_final_minus_used_max=float(np.max(np.abs(
                               res["J_final"] - res["jbs_used"]))),
                           correctors=("record-only under the loop: no "
                                       "corrector step; every pass re-solves "
                                       "the predictor on refreshed geometry "
                                       "with the rows held (axis row at the "
                                       "anchor's requested axis current, no "
                                       "row rescaling); readback + "
                                       "acceptance flags (structured_li_tol,"
                                       " q0_tol) on the delivered "
                                       "equilibrium"))
                if _pin is not None:
                    rec.update(
                        axis_row=("moved once per pass by the q0 pin "
                                  "(jbs_loop_q0_corrector=True) from the q0 "
                                  "measured on that pass's equilibrium, "
                                  "starting at the source's requested axis "
                                  "current; convergence requires |q0 - "
                                  "q0_target| <= q0_tol on the delivered "
                                  "equilibrium"),
                        correctors=("the q0 pin acts under the loop (axis "
                                    "row moved per pass, record['q0_pin']); "
                                    "no separate corrector step: the "
                                    "correctors read back and flag on the "
                                    "delivered equilibrium (q0_tol, "
                                    "structured_li_tol); the l_i row is held "
                                    "at its target"))
                if _swb_info:
                    rec.update(_swb_info)
                q0s, ss, oc = st["q0s"], st["ss"], st["oc"]
                nl = st["nl"]
                _ip_signed_f = oc["ip_signed"]
                _Ip_t_f = oc["Ip_t"]
                def _refresh_structured(jbs, k):
                    """The structured predictor re-run on the CURRENT mygs
                    geometry with bootstrap *jbs* (the MSE chord stage's
                    per-step refresh); returns the new state."""
                    jbs = np.asarray(jbs, dtype=float)
                    if _pin is not None:
                        # a chord step is a pass of the loop: move the axis
                        # row from the q0 of the step just solved.  Its
                        # closure imposed the row exactly (hard axis row, no
                        # current relaxation in the chord steps), so the
                        # solved axis current IS that closure's row.
                        _ax_prev = ((st.get("ss") or {}).get("axis") or {})
                        _pin.observe(_q0_of(mygs.copy_eq()),
                                     _ax_prev.get("j_ref0", _pin.row),
                                     stage=f"MSE chord step {k + 1}")
                        _pin.advance()
                    ctx_new = _closure_geometry(f"MSE chord step {k + 1}")
                    q0s_, ss_, oc_ = _ohmic_close(
                        ctx_new, jbs, jbs.max() / max(j_BS_src.max(), 1.0),
                        _pass=dict(q0_ref=q0_ref, k=k,
                                   x_retry=st.get("x_prev"),
                                   axis_row=(None if _pin is None
                                             else _pin.row)))
                    _icl1 = bl.ip_closure or {}
                    st.setdefault("refresh_log", []).append(dict(
                        k=k, closure_stop_reason=_icl1.get(
                            "structured_gn_stop_reason"),
                        closure_noise_floor_accepts=_icl1.get(
                            "structured_n_noise_floor_accepts"),
                        closure_retry=_icl1.get("structured_closure_retry")))
                    if _icl1.get("structured_coeffs_a") is not None:
                        st["x_prev"] = np.concatenate([
                            np.asarray(_icl1["structured_coeffs_a"], float),
                            np.asarray(_icl1["structured_coeffs_b"], float)])
                    st.update(ctx=ctx_new, q0s=q0s_, ss=ss_, oc=oc_)
                    return ss_

                # ---- MSE pitch angles: converge j_BS without MSE (done),
                # Jacobian once, chord steps with j_BS re-evaluated, one final
                # Jacobian refresh (docs/physics-notes.md) ------------------
                if ss is not None and ss.get("mse") is not None:
                    # what a refusal of the stage must put back: the closure
                    # state of the converged pass and the current it SOLVED
                    # (the stage snapshots bl and the equilibrium itself)
                    _st_pre = {k_: st.get(k_) for k_ in
                               ("ctx", "q0s", "ss", "oc", "x_prev")}
                    _pin_pre = None if _pin is None else _pin.snapshot()

                    def _restore_pre_mse():
                        st.update(_st_pre)
                        if _pin is not None:
                            _pin.restore(_pin_pre)

                    _pre_mse = dict(j_solved=st.get("j_solved"),
                                    restore=_restore_pre_mse,
                                    final=dict(res["record"].get("final")
                                               or {}))
                    _nl_m, ss, _mse_rec = self._structured_mse_jbs_stage(
                        ss, bl, mygs, solve_jphi=_pass_solve,
                        jbs0=res["jbs_used"],
                        refresh=lambda jbs, k: _refresh_structured(jbs, k),
                        evaluate=lambda eq: _redl(eq)[0],
                        measure=lambda eq: dict(
                            li=_li_of(eq),
                            q0=(_q0_of(eq) if axis_active else None)),
                        weights=lambda eq: residual_weights(
                            eq, psi_N, psi_pad, coord=coord)[:2],
                        settings=_jbs, Ip=Ip_abs, gate_q0=axis_active,
                        pre_mse=_pre_mse, q0_pin=_pin)
                    if _nl_m is not None:
                        nl = _nl_m
                    _mse_rec["refresh_closure_log"] = st.get("refresh_log", [])
                    rec["mse_stage"] = _mse_rec
                    if not _mse_rec.get("converged", False):
                        rec["converged"] = False
                        rec["jbs_converged"] = False
                        rec["stop_reason"] = ("MSE chord stage: "
                                              + str(_mse_rec.get(
                                                  "stop_reason")))
                        rec["final"] = _mse_rec.get("final", rec.get("final"))
                    _ip_signed_f = st["oc"]["ip_signed"]
                    _Ip_t_f = st["oc"]["Ip_t"]
                # ---- the correctors: readback + acceptance, no step --------
                if q0s is not None:
                    self._close_ip_q0_corrector(
                        q0s, bl, mygs, solve_jphi, ip_of=_ip_signed_f,
                        roundtrip_gate=lambda _ipc: _gate_fn(
                            _ipc, _Ip_t_f)["err_pct"],
                        record_only=True)
                if ss is not None:
                    self._close_ip_structured_corrector(
                        ss, bl, mygs, solve_jphi, ip_of=_ip_signed_f,
                        roundtrip_gate=self._structured_roundtrip_gate(
                            _Ip_t_f),
                        record_only=True)
                    if ss.get("mse_applied"):
                        self._structured_mse_delivered(ss, bl, mygs)
                if (q0s is not None or ss is not None) \
                        and bl.ip_closure is not None:
                    _p0 = st.get("pass0") or {}
                    # n_extra_solves counts EVERY GS solve after the first
                    # pass's: the loop's passes and, with an MSE block, the
                    # MSE chord stage's solves (FD probes, chord steps, final
                    # step, a refusal's restore solve) -- as the legacy stage
                    _n_loop = int(rec.get("n_passes", 1)) - 1
                    _n_mse = (0 if ss is None or ss.get("mse") is None
                              else int(ss.get("mse_n_solves", 0) or 0))
                    _upd = dict(n_extra_solves=_n_loop + _n_mse,
                                sawtooth_verdict=(
                                    f"j_BS loop: predictor re-solved on "
                                    f"refreshed geometry for "
                                    f"{rec.get('n_passes')} pass(es) "
                                    "(record-only: no corrector step, rows "
                                    "held)"))
                    _mse_suffix = ""
                    if ss is not None and ss.get("mse") is not None:
                        _what = ss.get("mse_stage_word") or (
                            "MSE chord stage" if ss.get("mse_applied")
                            else "MSE chord stage refused")
                        _mse_suffix = (f" + {_what} ({_n_mse} solve"
                                       + ("" if _n_mse == 1 else "s") + ")")
                        _upd.update(structured_loop_n_extra_solves=_n_loop,
                                    structured_mse_n_extra_solves=_n_mse)
                    if _p0.get("q0") is not None and (
                            ss is not None and ss.get("gated")
                            or q0s is not None):
                        _upd.update(q0_solved_predictor=_p0["q0"],
                                    q0_predictor_residual=(
                                        _p0["q0"] - _q0_target))
                    if _p0.get("li") is not None and ss is not None:
                        _lt = float(ss["li_target"])
                        _upd.update(
                            structured_li_solved_predictor=_p0["li"],
                            structured_li_achieved_predictor=_p0["li"],
                            structured_li_residual_predictor=_p0["li"] - _lt)
                    if _pin is not None:
                        _upd["sawtooth_verdict"] = (
                            f"j_BS loop: predictor re-solved on refreshed "
                            f"geometry for {rec.get('n_passes')} pass(es) "
                            "with the q0 pin acting (axis row moved per "
                            "pass; no separate corrector step)")
                    _upd["sawtooth_verdict"] += _mse_suffix
                    bl.ip_closure.update(_upd)
                if _pin is not None:
                    _pin_gate(rec)
                return _finish(rec, nl)


            if _loop_on:
                nl_its = _imas_jbs_loop()
            else:
                swb_seed, swb_fix = self._swb_inputs(mygs, psi_N, coord)
                swb = solve_with_bootstrap(
                    mygs, ne, te, ni, ti, Zeff, bl.Ip_target, swb_seed,
                    scale_jBS=1.0, isolate_edge_jBS=iso,
                    diagnostic_plots=False, verbose=False,
                    **swb_fix,
                    **coords.swb_grid_kwargs(psi_N, coord),
                    **gc.bootstrap_kwargs,
                )
                # Same axis-transition smoothing every per-draw spike receives, so
                # the sigma=0 draw reproduces this baseline split exactly.
                j_BS_swb = smooth_jbs_transition(
                    np.asarray(swb["isolated_j_BS"], dtype=float))
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
                    tgt = calc_cylindrical_li_proxy(mygs, FUSE_tot, psi_pad,
                                                    psi_N, coord)
                    _f = lambda s: calc_cylindrical_li_proxy(
                        mygs, j_ind + s * j_BS_swb + j_fixed, psi_pad, psi_N,
                        coord) - tgt
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
                    _q0_state, _structured_state, _oc = _ohmic_close(
                        _anchor, j_BS_swb, ratio)
                    _ip_signed = _oc["ip_signed"]
                    Ip_t = _oc["Ip_t"]
                    ip_roundtrip_gate = _oc["ip_roundtrip_gate"]
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
                    _pp_y = solver_pprime(psi_N, p_total, _pr, _edge)
                    _pp_prof = coords.oft_prof("linterp", psi_N, _pp_y, coord)
                    _, _n_corr, _corr_hist = _corrective_jphi_iteration(
                        mygs, psi_N, _jphi_solve, _pp_prof,
                        abs(bl.Ip_target), solver_pax(p_total, _edge), 1e-3,
                        min_iters=2, max_iters=8, rtol=0.02, verbose=True,
                        damping=0.5, protect_state=True,
                        coord=coord)
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
                    # MSE pitch angles (only when a usable mse_data block was
                    # given): linearise tan(gamma) on solved equilibria and
                    # re-solve the closure with chi2_MSE in its objective, BEFORE
                    # the q0/l_i corrector, which then keeps the MSE term in every
                    # re-solve it takes.
                    if _structured_state.get("mse") is not None:
                        _nl_mse = self._close_ip_structured_mse_stage(
                            _structured_state, bl, mygs, solve_jphi)
                        if _nl_mse is not None:
                            nl_its = _nl_mse
                    _nl_corr = self._close_ip_structured_corrector(
                        _structured_state, bl, mygs, solve_jphi,
                        ip_of=_ip_signed,
                        roundtrip_gate=self._structured_roundtrip_gate(Ip_t))
                    if _nl_corr is not None:
                        nl_its = _nl_corr
                    if _structured_state.get("mse_applied"):
                        self._structured_mse_delivered(_structured_state, bl, mygs)

        self._finish_imas_baseline(nl_its, psi_pad, ctx=dict(
            edge=_edge, p_total=p_total, psi_N=psi_N, p_components=_pc,
            delivery=_delivery, ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
            k2e=k2e))

    def _finish_imas_baseline(self, nl_its, psi_pad=1e-3, *, ctx):
        """Common tail of the IMAS baselines: check Ip on ``mygs``'s converged
        state, record TokaMaker li_1/li_3 and the closure metrics in
        ``li_metrics``, and set ``l_i_target`` (li_3, 'iter').  ``ctx``: the
        solve's own pressure (``p_total``, ``p_components``, ``psi_N``), the
        edge-pressure settings (``edge``; None: no record) and the
        self-consistent loop's delivery (``delivery`` + the kinetics)."""
        import numpy as np

        bl = self.baseline
        mygs = self.mygs
        _edge, p_total = ctx.get("edge"), ctx.get("p_total")
        psi_N = np.asarray(ctx.get("psi_N", bl.psi_N), dtype=float)
        _pc, _delivery = ctx.get("p_components"), ctx.get("delivery")
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
        tok_li3 = float((_st3 := mygs.get_stats(
            lcfs_pad=psi_pad, li_normalization="iter"))["l_i"])
        # the pressure handed to the solver and both frames of beta / W_MHD
        from .edge_pressure import archive_record as _edge_record
        from .edge_pressure import solver_p_scale as _p_scale
        bl.edge_pressure = _edge_record(_edge, p_total, stats=_st3,
                                        p_scale=_p_scale(mygs))

        # ---- core-pressure hollowness health record (report-only) ----------
        # Describes the core shape of the INPUT pressure the solve above was
        # handed (p_total, and its thermal species only) and of the ACHIEVED
        # pressure the converged equilibrium carries, read back on the same
        # grid (get_profiles only samples the converged state).  Nothing reads
        # it back; any failure is recorded, never raised.
        from .physics import core_pressure_hollow_record
        try:
            _p_ach, _ach_reason = None, None
            try:
                _p_ach = mygs.get_profiles(psi=psi_N)[3]
                # the solver's pressure is zero at psi_N = 1; under
                # separatrix_pressure="offset" p_sep is added back
                # (bouquet.edge_pressure) so the achieved pressure is in the
                # same full frame as the input p_total
                from .edge_pressure import applied_offset as _applied_offset
                _p_ach = (np.asarray(_p_ach, dtype=float)
                          + _applied_offset(p_total, _edge))
            except Exception as exc:    # pragma: no cover - live-solver only
                _ach_reason = f"get_profiles failed: {exc}"
            _cph = core_pressure_hollow_record(
                psi_N, p_total, input_components=_pc,
                achieved_total=_p_ach, achieved_reason=_ach_reason)
        except Exception as exc:        # pragma: no cover - defensive
            _cph = {"unavailable": f"health record failed: {exc}"}
        bl.core_pressure_hollow = _cph

        metrics = dict(bl.li_metrics or {})
        # Archived with the baseline next to the ip_closure / sawtooth blocks.
        metrics["core_pressure_hollow"] = _cph
        metrics.update(tokamaker_li_1=tok_li1, tokamaker_li_3=tok_li3,
                       forward_solve_nl_its=nl_its,
                       forward_solve_ip_err_pct=ip_err_pct,
                       jBS_baseline_mode=str(self.config.generation.jBS_baseline_mode),
                       bs_scale=float(getattr(bl, "bs_scale", 1.0)),
                       ohm_scale=float(getattr(bl, "ohm_scale", 1.0)),
                       source_current_sign=float(getattr(bl, "source_current_sign", 1.0)),
                       source_current_sign_origin=getattr(
                           bl, "source_current_sign_origin", None),
                       source_b0_sign=getattr(bl, "source_b0_sign", None))
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
        # how the core_sources slice and its beam / sawteeth entries were
        # matched in time (owner decision 2026-10-06): archived with the
        # baseline's metrics so every slice of a sweep carries its dt
        if getattr(bl, "source_time_match", None):
            metrics["source_time_match"] = bl.source_time_match
        if getattr(self, "_swb_seed_record", None):
            metrics["swb_seed"] = dict(self._swb_seed_record)
        bl.li_metrics = metrics
        # Target TokaMaker li_3 ('iter').  The IMAS path is not itself affected
        # by the geqdsk estimator mismatch (both sides come from TokaMaker),
        # but the DOWNSTREAM machinery is shared: perturb_kinetic_equilibrium
        # measures every draw's l_i with li_normalization='iter' after issue
        # #20, so l_i_target must be on that scale or the per-draw acceptance
        # band compares two different functionals (~25% apart).
        bl.l_i_target = tok_li3
        bl.l_i_scale = "iter(li3)"
        if (_delivery is not None and _delivery.get("request") is not None
                and _delivery["mode"] in ("diff", "rescale")):
            self._deliver_imas_state(_delivery, ctx["ne"], ctx["te"],
                                     ctx["ni"], ctx["ti"], ctx["Zeff"],
                                     psi_pad, ctx["k2e"])
        print(
            f"[imas forward-solve] converged ({nl_its} its, "
            f"Ip {ip_err_pct:+.2f}%) recalc_jBS="
            f"{self.config.generation.recalculate_j_BS} "
            f"TokaMaker li_1={tok_li1:.4f} li_3={tok_li3:.4f} | "
            f"IDS li_1={metrics.get('ids_li_1')} li_3={metrics.get('ids_li_3')}"
        )

    def _deliver_imas_state(self, delivery, ne, te, ni, ti, Zeff, psi_pad,
                            k2e):
        """The modelling-source path's ONE state, stored in the draws' form
        (``jbs_self_consistent=True``; ``jBS_baseline_mode`` "diff" or
        "rescale").

        ``mygs`` holds the delivered equilibrium F (``l_i_target`` was just
        read off it) and ``delivery["request"]`` is the jphi-linterp input
        that produced it -- the source total (+ ``jphi_diff``) in diff mode,
        the last pass's solved current in rescale mode.  jphi-linterp scales
        that input uniformly to Ip, and the source total reads 3.5 % short of
        Ip in the exact FSA measure on the synthetic example, so the stored
        split was NOT what F carries: a draw's Ip bookkeeping charged that
        shortfall to the inductive alone (route R2 scale 1.046-1.049 at zero
        perturbation).  Here the request is normalised to ``Ip_target`` in
        that same 'exact' measure on F (F is unchanged: the request's SHAPE
        is), the bootstrap stays the draws' own sigma=0 composition on F (in
        diff mode ``j_BS + jBS_diff`` IS the source bootstrap, exactly as
        before), the fixed parts and ``jphi_diff`` stay as read, and the
        inductive is the residual -- so it carries the whole normalisation.
        ``jBS_baseline_mode="ohmic"`` (baseline-only; the draws refuse it) is
        left as it was.
        """
        import numpy as np
        from .TokaMaker_interface import (DELIVERED_SPLIT_CONVENTION,
                                          _achieved_jphi_fsa,
                                          _deliver_request_split,
                                          _draw_jbs_composer,
                                          _request_offset)
        bl = self.baseline
        mygs = self.mygs
        gc = self.config.generation
        psi_N = np.asarray(bl.psi_N, dtype=float)
        coord = getattr(bl, "coord", coords.PSI)
        jdiff = (None if getattr(bl, "jBS_diff", None) is None
                 else np.asarray(bl.jBS_diff, dtype=float))
        if delivery["mode"] == "diff" and jdiff is not None:
            # Redl on F (+ the model offset): the source bootstrap, unchanged
            j_bs0 = np.asarray(bl.j_BS, dtype=float) + jdiff
        else:
            comp = _draw_jbs_composer(
                psi_N, ne, te, ni, ti, Zeff, psi_pad,
                bool(gc.isolate_edge_jBS), float(getattr(bl, "bs_scale", 1.0)),
                bool(gc.floor_j_BS), jdiff, None, None, coord=coord)
            j_bs0 = np.asarray(comp(mygs.copy_eq())[0], dtype=float)
            bl.j_BS = j_bs0 - (0.0 if jdiff is None else jdiff)
        # every channel the draws hold fixed, once each (the draws' _jfix =
        # j_NBI + j_RF + j_other): j_other left out landed in the inductive
        # residual and every draw added it again (PR #70 review B1)
        fixed = self._draw_fixed_current(psi_N)
        jd = (np.zeros_like(psi_N) if getattr(bl, "jphi_diff", None) is None
              else np.asarray(k2e(bl.jphi_diff), dtype=float))
        dv = _deliver_request_split(mygs, psi_N, psi_pad, bl.Ip_target,
                                    delivery["request"], j_bs0, fixed + jd,
                                    label="imas delivered state", coord=coord)
        bl.j_inductive = np.asarray(dv["j_inductive"], dtype=float)
        # bl.j_phi excludes jphi_diff (generate adds it): j_phi + jphi_diff
        # is the normalised request
        bl.j_phi = np.asarray(dv["request"], dtype=float) - jd
        bl.jphi_request_offset, _n_fl_t = _request_offset(
            bl.j_inductive, dv["achieved"], j_bs0, fixed + jd)
        _st = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
        from .physics import SOLVER_Q0_PSI_N
        bl.delivered_state = dict(
            convention=DELIVERED_SPLIT_CONVENTION, path="imas",
            jBS_baseline_mode=str(delivery["mode"]),
            l_i=float(bl.l_i_target), l_i_scale="iter(li3)",
            q0=float(_st.get("q_0", float("nan"))),
            # get_stats' q0 is q at psi_N 0.02, not on axis (and not at the
            # closure's psi_q[0], ip_closure["q0_target_psi_N"])
            q0_psi_N=float(SOLVER_Q0_PSI_N),
            q95=float(_st.get("q_95", float("nan"))),
            Ip_target=float(bl.Ip_target),
            edge_pressure=getattr(bl, "edge_pressure", None),
            request_normalisation=float(dv["kappa"]),
            achieved_normalisation=float(dv["kappa_achieved"]),
            n_negative_inductive=int(dv["n_negative_inductive"]),
            n_floored_target_inductive=int(_n_fl_t),
            j_phi_achieved=_achieved_jphi_fsa(
                mygs, psi_N, psi_pad,
                sign_ref=np.asarray(dv["request"], dtype=float), coord=coord),
            how=("the forward solve's delivered equilibrium (diff: the "
                 "source total; rescale: the loop's last pass); one "
                 "jphi-linterp solve of j_phi + jphi_diff reproduces it"))
        print(f"[imas delivered state] stored split = the Ip-normalised "
              f"request of the delivered equilibrium (x{dv['kappa']:.6f} in "
              f"the exact measure; the inductive carries it); l_i(3)="
              f"{bl.l_i_target:.6f} q0={bl.delivered_state['q0']:.4f} "
              f"q95={bl.delivered_state['q95']:.4f}", flush=True)

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
        from .physics import ELEMENTARY_CHARGE as EC
        pk = np.asarray(bl.psi_N_kinetic, dtype=float)
        pe = np.asarray(bl.psi_N, dtype=float)
        xl = r"$\Phi_N$" if getattr(bl, "coord", "psi_n") == "phi_n" else r"$\psi_N$"

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        # kinetic profiles (densities left axis, temperatures right axis)
        a = ax[0]
        a.plot(pk, np.asarray(bl.ne) / 1e19, "-", color="tab:blue", label=r"$n_e$")
        a.plot(pk, np.asarray(bl.ni) / 1e19, "--", color="tab:blue", label=r"$n_i$")
        a.set_ylabel(r"$n$ [$10^{19}$ m$^{-3}$]"); a.set_xlabel(xl)
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
        ax[1].set_ylabel("p [kPa]"); ax[1].set_xlabel(xl)
        ax[1].set_title("pressure"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

        # separated toroidal currents
        ax[2].plot(pe, np.asarray(bl.j_phi) / 1e6, "-", color="k", label=r"$j_\phi$ total")
        ax[2].plot(pe, np.asarray(bl.j_inductive) / 1e6, "-", color="tab:orange",
                   label=r"$j_{ind}$")
        ax[2].plot(pe, np.asarray(bl.j_BS) / 1e6, "-", color="tab:green", label=r"$j_{BS}$")
        _fixed = (("j_NBI", bl.j_NBI), ("j_RF", bl.j_RF),
                  ("j_other", getattr(bl, "j_other", None)))
        if getattr(bl, "j_saw", None) is not None:
            # swb_saw_q: j_saw replaces j_other's sawteeth share in j_phi
            _jst = getattr(bl, "j_sawteeth", None)
            _jo = np.asarray(bl.j_other if bl.j_other is not None else 0.0, dtype=float)
            _fixed = _fixed[:2] + (
                ("j_other - j_sawteeth", _jo - (0.0 if _jst is None else np.asarray(_jst))),
                ("j_saw", bl.j_saw))
        for nm, arr in _fixed:
            if arr is not None and np.any(np.asarray(arr)):
                ax[2].plot(pe, np.asarray(arr) / 1e6, "--", lw=1, label=nm)
        ax[2].set_ylabel(r"$j$ [MA/m$^2$]"); ax[2].set_xlabel(xl)
        ax[2].set_title("separated currents"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

        ttl = (f"Baseline  Ip={bl.Ip_target/1e6:.3f} MA  "
               f"l_i(target)={bl.l_i_target:.3f}")
        fig.suptitle(ttl, fontsize=11); fig.tight_layout()
        return fig, ax

    def _bootstrap_multiplier(self):
        """The baseline's bootstrap multiplier on psi_N, or None when it is 1.

        ``bl.j_BS = m * SWB(scale 1)`` with ``m`` the structured closure's
        ``s_bs(psi)`` or the scalar ``bs_scale``.
        """
        import numpy as np
        bl = self.baseline
        prof = getattr(bl, "bs_scale_profile", None)
        if prof is not None:
            return np.asarray(prof, dtype=float)
        bs = float(getattr(bl, "bs_scale", 1.0))
        return None if bs == 1.0 else bs * np.ones_like(
            np.asarray(bl.psi_N, dtype=float))

    #: The fixed (held, never perturbed) current channels every draw adds:
    #: ``perturb_kinetic_equilibrium``'s ``_jfix``.  The delivered state and
    #: both sigma=0 guards hold exactly these.
    DRAW_FIXED_CHANNELS = ("j_NBI", "j_RF", "j_other")

    def _draw_fixed_current(self, psi_N):
        """``j_NBI + j_RF + j_other`` of the baseline on ``psi_N`` (None ->
        zero): the fixed current a draw holds (``_jfix``)."""
        import numpy as np
        bl = self.baseline
        fixed = np.zeros_like(np.asarray(psi_N, dtype=float))
        for _nm in self.DRAW_FIXED_CHANNELS:
            if getattr(bl, _nm, None) is not None:
                fixed = fixed + np.asarray(getattr(bl, _nm), dtype=float)
        return fixed

    def _draws_run_jbs_loop(self, method=None):
        """Whether ``generate()``'s draws compose their bootstrap with the
        self-consistent loop (``_DrawJBSComposer``) rather than SWB: the
        loop settings ``generate()`` hands the draws are enabled (after the
        draw method's own say), ``recalculate_j_BS`` is on, and neither
        PIN_JPHI nor DIFF_BS (which take precedence over the loop in
        ``perturb_kinetic_equilibrium``) is set."""
        import os
        from .jbs_loop import jbs_settings as _jbs_settings
        gc = self.config.generation
        if not bool(gc.recalculate_j_BS):
            return False
        st = _jbs_settings(gc, draw=True)
        if method is not None:
            st = method.loop_settings_for(st)
            st = method.draw_jbs_loop(st if st["enabled"] else None)
        if not (st and st.get("enabled")):
            return False
        return not (bool(getattr(gc, "pin_jphi", False))
                    or os.environ.get("PIN_JPHI", "0") == "1"
                    or os.environ.get("DIFF_BS", "0") == "1")

    def _draw_bootstrap_scaling(self, method=None):
        """``(jBS_scale_range, jBS_scale_profile, record)``: how the draws
        carry the baseline's bootstrap multiplier ``m`` (``bs_scale``, or
        the structured ``s_bs(psi)``; :meth:`_bootstrap_multiplier`).  The
        ONE rule ``generate()`` and both sigma=0 guards use, so a guard
        replays exactly the draws' scale and never compensates for them.

        * SWB draws: ``m`` after SWB (``jBS_scale_profile``), the configured
          ``jBS_scale_range`` the per-draw jitter inside SWB -- the baseline
          built ``j_BS = m * SWB(scale 1)``, and OFT applies a scale inside
          SWB's own iteration, so ``SWB(m) != m * SWB(1)``.
        * Loop draws: the composer is linear in its scale (``spike =
          scale * Redl``) and takes no profile, so ``m`` rides in the scale:
          the range re-centred on ``m`` (``(m, m)`` when no range is set),
          no profile.  The baseline composed ``j_BS`` at that scale, so a
          sigma=0 draw reproduces it (PR #70 review B2: the loop draws had
          lost ``m``, off by ``1/m`` in rescale mode).  A non-uniform
          ``s_bs(psi)`` cannot reach the loop composer and is REFUSED.
        """
        import numpy as np
        gc = self.config.generation
        mult = self._bootstrap_multiplier()
        rng = (None if gc.jBS_scale_range is None
               else tuple(gc.jBS_scale_range))
        loop = self._draws_run_jbs_loop(method)
        rec = dict(draws=("self-consistent loop" if loop else "SWB"),
                   multiplier=(None if mult is None else dict(
                       min=float(np.min(mult)), max=float(np.max(mult)),
                       source=("Baseline.bs_scale_profile" if getattr(
                           self.baseline, "bs_scale_profile", None)
                           is not None else "Baseline.bs_scale"))))
        if loop and mult is not None:
            u = np.unique(np.asarray(mult, dtype=float))
            if u.size != 1:
                raise ValueError(
                    "the baseline carries a non-uniform bootstrap multiplier "
                    "s_bs(psi) (Baseline.bs_scale_profile, from the structured "
                    "closure) but the draws run the self-consistent j_BS loop, "
                    "whose composer takes a scalar scale only: the draws "
                    "cannot reproduce the baseline bootstrap.  Run the draws "
                    "with jbs_self_consistent=False (SWB draws apply s_bs(psi) "
                    "after SWB), or a closure that delivers a scalar bs_scale")
            m = float(u[0])
            rng = (m, m) if rng is None else (rng[0] * m, rng[1] * m)
            mult = None
            rec["applied_as"] = ("scale_jBS: jBS_scale_range re-centred on "
                                 f"bs_scale {m:.9g}")
        else:
            rec["applied_as"] = ("jBS_scale_profile after SWB"
                                 if mult is not None else "none (1.0)")
        if method is not None:
            rng, mult = method.scale_settings(rng, mult)
        rec["jBS_scale_range"] = None if rng is None else [float(v) for v in rng]
        return rng, mult, rec

    def _resolve_swb_seed(self):
        """``(mode, record)`` of ``GenerationConfig.swb_seed`` for this run.

        ``None`` (the default) resolves to ``"source"`` when the installed
        OpenFUSIONToolkit's ``solve_with_bootstrap`` takes ``jphi_fixed``,
        else to ``"generic"`` -- never an error, whether or not SWB runs.  An
        explicit ``"source"`` on a toolkit without ``jphi_fixed`` is
        ``"source_unavailable"``: nothing is raised here (no SWB may run);
        the first SWB call that would use it raises
        (:meth:`_swb_inputs`, ``generate()``).  Stamped in
        ``li_metrics["swb_seed"]``."""
        gc = self.config.generation
        req = getattr(gc, "swb_seed", None)
        cap = "jphi_fixed" in coords._swb_params()
        if req is None:
            mode = "source" if cap else "generic"
            how = ("auto: this OpenFUSIONToolkit's solve_with_bootstrap "
                   + ("takes jphi_fixed" if cap else
                      "has no jphi_fixed, so the generic seed"))
        elif req == "source" and not cap:
            mode = "source_unavailable"
            how = ("set 'source', but this OpenFUSIONToolkit's "
                   "solve_with_bootstrap has no jphi_fixed: refused at the "
                   "first SWB call")
        else:
            mode, how = str(req), "set"
        return mode, dict(requested=req, resolved=mode,
                          oft_jphi_fixed=bool(cap), how=how)

    def _swb_source_unavailable(self):
        """The refusal of an explicit ``swb_seed="source"`` at an SWB call
        on a toolkit without ``jphi_fixed``."""
        return RuntimeError(
            "swb_seed='source' needs an OpenFUSIONToolkit whose "
            "solve_with_bootstrap takes jphi_fixed, and an SWB call is about "
            "to run; set swb_seed='generic' (or None: resolved to the "
            "toolkit's capability)")

    def _swb_inputs(self, mygs, psi_N, coord):
        """``(inductive seed, extra SWB kwargs)``: the baseline's source seed
        and ``jphi_fixed`` when set (``swb_seed`` resolved to "source"), else
        the generic seed and no kwargs.  An explicit ``swb_seed="source"``
        the toolkit cannot honour is refused here, at the SWB call.
        """
        bl = self.baseline
        if getattr(bl, "swb_seed_profile", None) is not None:
            return bl.swb_seed_profile, {"jphi_fixed": bl.swb_jphi_fixed}
        if (getattr(self, "_swb_seed_record", None) or {}).get(
                "resolved") == "source_unavailable":
            raise self._swb_source_unavailable()
        return coords.swb_seed(psi_N, coords.psi_at(mygs, psi_N, coord)), {}

    def verify_sigma0_consistency(self, tol_frac=0.02, draw_route=True,
                                  draw_routes=None):
        """Regression guard: the draw pipeline must reproduce the baseline
        j_BS split when the kinetics are UNPERTURBED (sigma=0).

        Replays the per-draw pre-SWB sequence -- state-anchor solve at
        the baseline j_phi/pressure, ``solve_with_bootstrap`` on the baseline
        kinetics, axis-transition smoothing -- and
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

        **With ``jbs_self_consistent=True``** the check is the zero-perturbation
        identity of the draws.  The reconstruction is ONE equilibrium F
        (``Baseline.delivered_state``; ``l_i_target`` is its l_i), stored as
        the Ip-normalised jphi-linterp request of F; the state anchor above is
        one solve of that request (``jphi_diff`` included), i.e. F.
        ``passed`` REQUIRES the draw's own route -- the ``draw_route`` block:
        :func:`perturb_kinetic_equilibrium` with the arguments ``generate()``
        hands a draw, every sigma zero, from the state this method was called
        on, once per route in ``draw_routes`` (default: every route the
        configuration can use, :meth:`_sigma0_draw_routes`) -- to reproduce F
        at the loop's own, unchanged tolerances: the draw's loop converged,
        its bootstrap within ``jbs_rtol_j`` (profile, current-weighted) and
        ``jbs_rtol_Ip`` (current) of ``baseline.j_BS`` (+ ``jBS_diff``), and
        ``|l_i - l_i_target| <= jbs_tol_li``; q0, q95 and the total-current
        profile against F are reported beside them.  ``draw_route=False``
        leaves ``passed`` False (the identity is then unverified).  The
        generate()-level coil regularisation, hard bounds and homotopy are
        not part of the replay.  The loop "solved the baseline's way" -- the
        baseline inductive held, one jphi-linterp solve per pass, as F itself
        is solved -- is kept beside it as ``passed_baseline_way`` (with its
        ``r_j_vs_baseline`` / ``r_I_vs_baseline`` / ``dl_i_vs_baseline``) and
        no longer decides ``passed``.  ``tol_frac`` is then unused (no SWB
        is called).  Route R2's inductive Ip
        renormalisation keeps its own σ=0 budget as well
        (``tests/test_seeded_reproducibility.py``).

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

        Returns
        -------
        dict with ``spike0`` (the sigma=0 draw-context j_BS), ``max_dev``,
        ``rms_dev`` [A/m^2], ``max_dev_frac`` (of peak j_BS), ``psi_worst``
        (in the run coordinate ``coord``), and ``passed``.
        """
        # the method's own check (engine and swb: their zero-perturbation
        # draw; swb's equal to solve B to the bit); None: the legacy check below
        from .draw_methods import method_hooks
        _hooks = method_hooks(self.config.generation)
        _r = _hooks.verify_sigma0(self)
        if _r is not None:
            return _r
        import numpy as np
        from scipy.interpolate import interp1d
        from .TokaMaker_interface import smooth_jbs_transition
        from .utils import pchip_derivative
        # OpenFUSIONToolkit only on the legacy route, the one that calls
        # solve_with_bootstrap (the engine's check runs on its own solver)
        from OpenFUSIONToolkit.TokaMaker.bootstrap import \
            solve_with_bootstrap

        if self.baseline is None or self.mygs is None:
            raise ValueError("call setup_solver() + prepare_baseline() / "
                             "reconstruct() before verify_sigma0_consistency()")
        self._ensure_engine_defaults_resolved()
        self._refuse_unified_engine_draws("verify_sigma0_consistency()")
        _edge = resolve_edge_pressure(self.config.generation)
        if self.config.generation.single_profile_jphi:
            # No inductive/bootstrap split exists, so there is nothing for the
            # sigma=0 draw to reproduce. Report a pass rather than spending a
            # bootstrap solve comparing zeros.
            print("[sigma0-check] SKIPPED: single_profile_jphi=True -- j_phi is "
                  "one profile, so there is no j_BS split to verify")
            return {"spike0": None, "max_dev": 0.0, "rms_dev": 0.0,
                    "max_dev_frac": 0.0, "psi_worst": float("nan"),
                    "coord": getattr(self.baseline, "coord", coords.PSI),
                    "passed": True, "skipped": "single_profile_jphi"}
        bl = self.baseline
        mygs = self.mygs
        gc = self.config.generation
        # The state this check was handed (what generate() would start its
        # draws from): the draw-route replay under the loop starts from it.
        _entry_snap = (mygs.copy_eq()
                       if (draw_route
                           and bool(getattr(gc, "jbs_self_consistent", False))
                           and hasattr(mygs, "copy_eq")
                           and hasattr(mygs, "replace_eq")) else None)
        # psi_pad is a ReconstructionSource field; ImasSource has none, so
        # fall back to the pipeline default (matches the forward-solve sites).
        psi_pad = getattr(self.config.source, "psi_pad", 1e-3)
        psi_N = np.asarray(bl.psi_N, dtype=float)
        coord = getattr(bl, "coord", coords.PSI)
        from .physics import ELEMENTARY_CHARGE as EC

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
        pp = solver_pp_profile(
            psi_N, pressure, (mygs.psi_bounds[1] - mygs.psi_bounds[0]), _edge, coord=coord)
        ffp = coords.oft_prof("jphi-linterp", psi_N,
                              np.asarray(bl.j_phi, dtype=float).copy(), coord)
        if (bool(getattr(gc, "jbs_self_consistent", False))
                and getattr(bl, "jphi_diff", None) is not None):
            # self-consistent loop: the anchor is the reconstruction's own
            # solve -- its stored request INCLUDES the fixed total-current
            # anchor jphi_diff (as generate()'s baseline solve and every draw
            # carry it)
            ffp["y"] = ffp["y"] + pchip_interp(
                pk, np.asarray(bl.jphi_diff, dtype=float), psi_N)
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
        mygs.set_targets(Ip=float(bl.Ip_target), pax=solver_pax(pressure, _edge))
        mygs.set_profiles(pp_prof=pp, ffp_prof=ffp)
        try:
            mygs.solve()
        except Exception:
            if _snap is not None:
                # do not hand back the cold psi this method installed
                mygs.replace_eq(source_eq=_snap)
            raise

        from .jbs_loop import jbs_settings as _jbs_settings
        _jbs = _jbs_settings(gc, draw=True)
        if _jbs["enabled"]:
            return self._verify_sigma0_jbs_loop(
                _jbs, pp, ffp, pressure, ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
                psi_N, psi_pad, entry_snap=_entry_snap,
                draw_route=draw_route, draw_routes=draw_routes)

        seed, swb_fix = self._swb_inputs(mygs, psi_N, coord)
        # As the draw does (the same helper generate() uses): SWB at the
        # jitter's centre, then the baseline's multiplier (bs_scale or
        # s_bs(psi)) after SWB.
        from .TokaMaker_interface import sigma0_reference_scale
        _rng0, _mult, _ = self._draw_bootstrap_scaling()
        res = solve_with_bootstrap(
            mygs, ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq,
            float(bl.Ip_target), seed,
            scale_jBS=float(sigma0_reference_scale(_rng0)),
            isolate_edge_jBS=bool(gc.isolate_edge_jBS),
            **coords.swb_grid_kwargs(psi_N, coord),
            diagnostic_plots=False, **swb_fix, **gc.bootstrap_kwargs)
        spike0 = (1.0 if _mult is None else _mult) * smooth_jbs_transition(
            np.asarray(res["isolated_j_BS"], dtype=float))
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
                   psi_worst=float(psi_N[iworst]), coord=coord,
                   n_floored=int(floored.sum()),
                   max_dev_floored=(float(np.max(np.abs(dev[floored])))
                                    if floored.any() else 0.0),
                   passed=bool(np.max(np.abs(dev_eval)) <= tol_frac * peak))
        # leave mygs re-anchored on the baseline equilibrium, not SWB's state
        mygs.set_targets(Ip=float(bl.Ip_target), pax=solver_pax(pressure, _edge))
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
              f"of peak, worst at {'Phi_N' if coord == coords.PHI else 'psi_N'}"
              f"={out['psi_worst']:.3f}; "
              f"tol {100*tol_frac:.1f}%{_fl})")
        return out

    def _verify_sigma0_engine(self):
        """``verify_sigma0_consistency`` under ``reconstruction_engine=
        "unified"``: ONE draw through the very route ``generate()`` runs --
        :meth:`generate` itself, ``n=1``, every perturbation zero and the
        bootstrap scale 1.0, archived into a temporary file (the
        configuration's own archive is never touched): generate_bouquet's
        baseline re-solve and warm start, its strong coil regularisation and
        the RECONSTRUCTION's own one installed for the loop (the term list
        it solved under, ``_engine_run["coil_reg"]``; recorded as
        ``coil_reg``), the isoflux re-pointed to the draw's own boundary,
        the homotopy and the post-homotopy stage, under
        ``engine_draw_solve_maxits``.  Judged on that route at the unchanged
        tolerances, stage by stage, every gated quantity recorded with its
        value and bound (``gates``):

        * ``stages["loop"]`` -- the draw's delivered loop state (before the
          homotopy): the first request BIT-IDENTICAL to the stored request,
          the loop converged, its bootstrap within ``jbs_rtol_j`` /
          ``jbs_rtol_Ip`` of ``lambda_BS*``, ``|dl_i| <= jbs_tol_li``,
          ``|dq0| <= jbs_tol_q0`` at the q-row radius;
        * ``stages["archived"]`` -- the state the draw ARCHIVES (after the
          homotopy and the post-homotopy passes): the bootstrap it carries
          within ``jbs_rtol_j`` / ``jbs_rtol_Ip`` of ``lambda_BS*`` on that
          geometry, ``|dl_i| <= jbs_tol_li``, ``|dq0| <= jbs_tol_q0``; the
          homotopy stage and the flux-range change reported;
        * ``stages["archived_geometry"]`` -- the archived coil currents and
          LCFS against the reconstruction's, read from the temporary
          archive (:meth:`_sigma0_archived_geometry`): the max F-coil and
          VSC-coil drift within the hard coil bound the route rejects at
          and the LCFS rms deviation within the in-spec boundary cut
          (:meth:`_sigma0_geometry_gates`);
        * a draw that is REJECTED fails (``rejection``).

        ``passed`` needs every stage.  The top-level ``r_j`` / ``r_I`` /
        ``dl_i`` / ``dq0`` / ``dq95`` are the ARCHIVED state's (what a draw
        delivers); the identity and loop fields are the loop stage's.

        The solver state is put back afterwards -- ALL of it
        (:class:`bouquet.solver_state.SolverState`: the equilibrium object
        with psi, coil currents, coil regularisation, targets, profiles and
        the isoflux / saddle constraints; the settings; the VSC gains; the
        coil bounds on record; the coil-regularisation stashes generate()
        leaves on the solver object) -- and so is every attribute
        :meth:`generate` sets on this object.  The one-way coil-bound mode
        every generate() enters was entered once at :meth:`setup_solver`
        (:func:`bouquet.solver_state.enter_bounded_coil_mode`), before the
        reconstruction, so the check, a second check and the generate()
        after it start from bit for bit the same state and run the same
        coil solve the reconstruction did."""
        import os
        import tempfile
        from .jbs_loop import jsonable
        from .solver_state import SolverState
        gc = self.config.generation
        mygs = self.mygs
        _guard = SolverState.capture(mygs)
        keep_attrs = ("diagnostics", "generation_log", "draw_rejections",
                      "solve_failures", "engine_draw_cap_events",
                      "_resolved_uncertainty", "_sigma0_route")
        saved = {k: self.__dict__[k] for k in keep_attrs
                 if k in self.__dict__}
        saved_cfg = dict(header=self.config.output_header,
                         target=gc.n_inspec_target, cap=gc.max_total_draws)
        probe = {}
        out = dict(invariant="engine-draw", route="generate()",
                   criterion=(
                       "the generate() draw route with every perturbation "
                       "zero, its loop under the reconstruction's own coil "
                       "regularisation: loop stage -- pass-1 request "
                       "bit-identical, loop converged, r_j <= rtol_j, r_I "
                       "<= rtol_Ip, |dl_i| <= tol_li, |dq0| <= tol_q0; "
                       "archived state (after the homotopy and the "
                       "post-homotopy stage) -- r_j <= rtol_j, r_I <= "
                       "rtol_Ip, |dl_i| <= tol_li, |dq0| <= tol_q0; its "
                       "coil drift within the hard coil bound and its LCFS "
                       "rms within the in-spec boundary cut; not rejected"))
        try:
            with tempfile.TemporaryDirectory(prefix="bq_sigma0_") as td:
                self.config.output_header = os.path.join(td, "sigma0_route")
                gc.n_inspec_target = None
                gc.max_total_draws = None
                self._sigma0_route = probe
                try:
                    diags = self.generate(n=1)
                finally:
                    self._sigma0_route = None
                rej = list(getattr(self, "draw_rejections", []) or [])
                # the archived draw's coils and LCFS against the
                # reconstruction's, read from the temporary archive by the
                # same contours and currents the filters use
                geo = (None if (rej or not diags) else
                       self._sigma0_archived_geometry(
                           self.config.output_header, gc.scan_key))
        finally:
            self.config.output_header = saved_cfg["header"]
            gc.n_inspec_target = saved_cfg["target"]
            gc.max_total_draws = saved_cfg["cap"]
            for k in keep_attrs:
                if k in saved:
                    self.__dict__[k] = saved[k]
                else:
                    self.__dict__.pop(k, None)
            # every piece of solver state the route touched (the isoflux
            # targets included: they are in the equilibrium object)
            _guard.restore()
        G = probe.get("draws")
        s = G.ctx.loop if G is not None else {}
        out["tolerances"] = dict(rtol_j=s.get("rtol_j"),
                                 rtol_Ip=s.get("rtol_Ip"),
                                 tol_li=s.get("tol_li"),
                                 tol_q0=s.get("tol_q0"))
        loop, arch = probe.get("loop"), probe.get("archived")
        d0 = diags[0] if diags else None
        out["stages"] = dict(loop=loop, archived=arch)
        if rej:
            out["rejection"] = jsonable(rej[0])
        if d0 is not None:
            out["stages"]["archived_coils"] = dict(
                homotopy_pass=d0.get("homotopy_pass"),
                max_F_drift_pct=d0.get("max_F_drift_pct"),
                max_VSC_drift_pct=d0.get("max_VSC_drift_pct"))
        out["record"] = (d0 or {}).get("engine")
        # the coil regularisation the draw's loop solved under (the
        # reconstruction's own term list)
        out["coil_reg"] = (out["record"] or {}).get("coil_reg")
        if loop is not None:
            for k in ("request_bit_identical", "request_max_abs_diff",
                      "loop_converged", "n_passes", "dq0_stats",
                      "dq0_stats_psi_N", "amplitude", "solves"):
                out[k] = loop.get(k)
        if arch is not None:
            for k in ("r_j", "r_I", "dl_i", "dq0", "dq0_psi_N", "dq95",
                      "flux_range", "flux_range_rel"):
                out[k] = arch.get(k)
        # the archived state's coils and boundary against the
        # reconstruction's: GATED (each with its value and bound)
        geo_gates = (None if geo is None else
                     self._sigma0_geometry_gates(geo))
        out["stages"]["archived_geometry"] = (
            None if geo is None else dict(geo, gates=geo_gates))
        # every gated quantity of every stage, with its bound
        out["gates"] = dict(
            loop=(None if loop is None else loop.get("gates")),
            archived=(None if arch is None else arch.get("gates")),
            archived_geometry=geo_gates)
        from .engine_draws import _gates_pass
        ok = bool(not rej and d0 is not None and loop is not None
                  and arch is not None and loop["passed"]
                  and arch["passed"] and geo_gates is not None
                  and _gates_pass(geo_gates))
        out["passed"] = ok
        out["passed_reason"] = (
            "the generate() draw route at zero perturbation reproduces the "
            "reconstruction at both stages (loop tolerances)" if ok else
            "the generate() draw route at zero perturbation misses the "
            "reconstruction (see stages / rejection)")

        def _f(v, fmt):
            return "n/a" if v is None else format(v, fmt)
        print(f"[sigma0-check engine route] {'PASS' if ok else 'FAIL'}: "
              + (f"REJECTED ({rej[0].get('reason')}); " if rej else "")
              + ("" if loop is None else
                 f"loop: request {'bit-identical' if loop['request_bit_identical'] else 'DIFFERS'}, "
                 f"{'converged' if loop['loop_converged'] else 'NOT converged'}, "
                 f"r_j={loop['r_j']:.2e} r_I={loop['r_I']:.2e} "
                 f"dl_i={loop['dl_i']:+.2e}; ")
              + ("" if arch is None else
                 f"archived: r_j={arch['r_j']:.2e} r_I={arch['r_I']:.2e} "
                 f"dl_i={arch['dl_i']:+.2e} dq0={arch['dq0']:+.2e} "
                 f"dq95={_f(arch['dq95'], '+.2e')} "
                 f"dflux_rel={_f(arch['flux_range_rel'], '+.2e')}")
              + ("" if geo_gates is None else
                 "; coils/LCFS: " + ", ".join(
                     f"{k}={_f(g['value'], '.3g')} (bound "
                     f"{_f(g['bound'], 'g')}"
                     f"{'' if g['passed'] is not False else ' MISSED'})"
                     for k, g in geo_gates.items()))
              + f" (tol r_j {s.get('rtol_j')}, r_I {s.get('rtol_Ip')}, "
                f"l_i {s.get('tol_li')}, q0 {s.get('tol_q0')})", flush=True)
        return out

    def _sigma0_archived_geometry(self, header, scan_key):
        """The zero-perturbation draw's archived coil currents and LCFS
        against the reconstruction's, from the archive *header*: per-coil
        drift [%] (:func:`bouquet.TokaMaker_interface._coil_drift_pct`, the
        homotopy's own measure) of draw 0's ``coil_currents`` against
        ``_baseline/coil_currents`` (the reconstruction's state at
        ``generate()`` entry), the max non-VSC F-coil and max VSC-coil drift
        (:func:`~bouquet.TokaMaker_interface._coil_max_drifts`), and the
        LCFS deviation [mm] of draw 0's ``perturbed_lcfs_ref`` from the
        reconstruction's ``recon_lcfs_ref`` (:func:`bouquet.filtering.
        _boundary_devs`, the boundary filter's own metric).  Unreadable
        pieces are ``None`` (the gate then fails)."""
        import h5py
        import numpy as np
        from .filtering import _baseline_boundary, _boundary_devs
        from .TokaMaker_interface import _coil_drift_pct, _coil_max_drifts
        from .utils import _group_path, _read_coil_names, _resolve_h5
        out = dict(coil_drift_pct=None, max_F_drift_pct=None,
                   max_VSC_drift_pct=None, boundary_rms_mm=None,
                   boundary_max_mm=None,
                   vsc_coils=list(getattr(self.config.solver, "coil_vsc",
                                          None) or ()),
                   reference=("coils: the archived _baseline coil_currents "
                              "(the reconstruction's state at generate() "
                              "entry); LCFS: _baseline/recon_lcfs_ref"))
        with h5py.File(_resolve_h5(header), "r") as hf:
            gp = _group_path(scan_key, 0)
            if gp not in hf:
                return out
            grp = hf[gp]
            rms, mx = _boundary_devs(_baseline_boundary(hf, scan_key), grp)
            out["boundary_rms_mm"] = (float(rms) if np.isfinite(rms)
                                      else None)
            out["boundary_max_mm"] = (float(mx) if np.isfinite(mx)
                                      else None)
            bpath = gp.rsplit("/", 1)[0] + "/_baseline" if "/" in gp \
                else "_baseline"
            bl = hf.get(bpath)
            if ("coil_currents" in grp and bl is not None
                    and "coil_currents" in bl):
                cur = dict(zip(_read_coil_names(grp),
                               np.asarray(grp["coil_currents"], float)))
                base = dict(zip(_read_coil_names(bl),
                                np.asarray(bl["coil_currents"], float)))
                base = {k: float(v) for k, v in base.items() if k in cur}
                if base:
                    d = _coil_drift_pct(cur, base)
                    vsc = tuple(c for c in out["vsc_coils"] if c in base)
                    mf, mv = _coil_max_drifts(d, vsc)
                    out.update(coil_drift_pct={k: float(v)
                                               for k, v in d.items()},
                               max_F_drift_pct=float(mf),
                               max_VSC_drift_pct=float(mv))
        return out

    def _sigma0_geometry_gates(self, geo):
        """The zero-perturbation gates on the archived coils and LCFS:
        the max F-coil and VSC-coil drift against the HARD coil bound the
        draw route rejects at -- ``coil_drift_hard_factor x coil_drift``
        when hard bounds are configured, else the tightest homotopy stage
        (``homotopy_passes[-1]``, the hard bounds its saturation test
        works against), else (``engine_draw_homotopy=False``)
        ``coil_drift`` -- and the LCFS rms deviation against the boundary
        cut the in-spec filter applies (:meth:`_boundary_cut`; not gated
        when the cut is disabled, recorded so)."""
        from .engine_draws import sigma0_gate
        gc = self.config.generation
        hf = getattr(gc, "coil_drift_hard_factor", None)
        if hf is not None:
            bF = bV = 100.0 * float(hf) * float(gc.coil_drift)
            how = "coil_drift_hard_factor x coil_drift"
        elif bool(getattr(gc, "engine_draw_homotopy", True)) and \
                gc.homotopy_passes:
            bF, bV = (100.0 * float(v) for v in gc.homotopy_passes[-1])
            how = "homotopy_passes[-1] (the tightest homotopy stage)"
        else:
            bF = bV = 100.0 * float(gc.coil_drift)
            how = "coil_drift"
        cut, cut_src = self._boundary_cut(quiet=True)
        return dict(
            coil_F_drift_pct=sigma0_gate(geo["max_F_drift_pct"], bF,
                                         setting=how),
            coil_VSC_drift_pct=sigma0_gate(geo["max_VSC_drift_pct"], bV,
                                           setting=how),
            boundary_rms_mm=sigma0_gate(
                geo["boundary_rms_mm"], cut,
                setting=f"filtering.rms_max_mm ({cut_src})",
                absolute=False))

    def _verify_sigma0_jbs_loop(self, settings, pp, ffp, pressure, ne_eq,
                                te_eq, ni_eq, ti_eq, Zeff_eq, psi_N, psi_pad,
                                entry_snap=None, draw_route=True,
                                draw_routes=None):
        """``verify_sigma0_consistency`` under the self-consistent loop.

        ``mygs`` holds the state anchor: one jphi-linterp solve of the stored
        request -- the reconstruction's ONE state F (``Baseline.
        delivered_state``).  Two results:

        * ``passed`` -- the draw's OWN route(s) at zero perturbation
          (:meth:`_sigma0_draw_route`, every route the configuration can use)
          reproduce F at the loop's unchanged tolerances: bootstrap profile
          (``r_j <= jbs_rtol_j``), bootstrap current (``r_I <= jbs_rtol_Ip``)
          and ``|l_i - l_i_target| <= jbs_tol_li`` with the draw's loop
          converged; q0, q95 and the total-current profile difference are
          reported beside them.  Not run (``draw_route=False``) -> ``passed``
          is False: the identity is then unverified.
        * ``passed_baseline_way`` -- the loop reproduces F when solved the way
          F itself is solved (the reconstruction's inductive held, one
          jphi-linterp solve per pass); recorded, it no longer decides
          ``passed``.
        """
        import numpy as np
        from .jbs_loop import (profile_residuals, residual_weights,
                               run_jbs_loop, jsonable)
        from .TokaMaker_interface import _draw_jbs_composer

        bl = self.baseline
        mygs = self.mygs
        gc = self.config.generation
        _edge = resolve_edge_pressure(gc)
        coord = getattr(bl, "coord", coords.PSI)
        Ip = float(bl.Ip_target)
        jdiff = (None if getattr(bl, "jBS_diff", None) is None
                 else np.asarray(bl.jBS_diff, dtype=float))
        compose = _draw_jbs_composer(
            psi_N, ne_eq, te_eq, ni_eq, ti_eq, Zeff_eq, psi_pad,
            bool(gc.isolate_edge_jBS), float(getattr(bl, "bs_scale", 1.0)),
            bool(gc.floor_j_BS), jdiff, None, None, coord=coord)
        j_ind = np.asarray(bl.j_inductive, dtype=float)
        # the fixed current the draws hold (j_NBI + j_RF + j_other), not the
        # stored split's residual: a residual is self-consistent with a
        # wrong split and hid j_other counted twice (PR #70 review B1).  How
        # far the stored split is from closing on it is recorded.
        j_fix = self._draw_fixed_current(psi_N)
        _resid = (np.asarray(bl.j_phi, dtype=float) - j_ind
                  - np.asarray(bl.j_BS, dtype=float)
                  - (0.0 if jdiff is None else jdiff))
        _peak_phi = float(np.max(np.abs(np.asarray(bl.j_phi, dtype=float))))
        split_closure = dict(
            channels=list(self.DRAW_FIXED_CHANNELS),
            max_abs_frac=float(np.max(np.abs(_resid - j_fix))
                               / (_peak_phi or 1.0)),
            definition=("max |j_phi - j_inductive - j_BS (- jBS_diff) - "
                        "(j_NBI + j_RF + j_other)| / max |j_phi| of the "
                        "stored split: 0 to rounding when the stored "
                        "inductive excludes exactly what the draws hold"))
        if getattr(bl, "jphi_diff", None) is not None:
            # same kinetic -> equilibrium regrid the baseline solve applies
            from .utils import pchip_interp
            j_fix = j_fix + pchip_interp(
                np.asarray(bl.psi_N_kinetic, dtype=float),
                np.asarray(bl.jphi_diff, dtype=float), psi_N)
        ref = np.asarray(bl.j_BS, dtype=float) + (0.0 if jdiff is None
                                                  else jdiff)

        # The reconstruction's final state F is, on both paths, ONE
        # jphi-linterp solve of the stored request (the g-file
        # reconstruction ends on its l_i re-match, not on the corrective
        # iteration), so "the baseline's way" is a single solve per pass.
        def _solve(j):
            from .utils import pchip_derivative
            _pr = mygs.psi_bounds[1] - mygs.psi_bounds[0]
            _pp = solver_pp_profile(psi_N, pressure, _pr, _edge, coord=coord)
            mygs.set_targets(Ip=Ip, pax=solver_pax(pressure, _edge))
            mygs.set_profiles(pp_prof=_pp, ffp_prof=coords.oft_prof(
                "jphi-linterp", psi_N, np.asarray(j, float), coord))
            mygs.solve()

        def _step(spk, k, relax=None):
            # the baseline's own inductive, held
            _j = j_ind + spk + j_fix
            _solve(_j if relax is None else relax(_j))
            _snap = mygs.copy_eq()
            _w, _x, _k = residual_weights(_snap, psi_N, psi_pad, coord=coord)
            return dict(w=_w, x=_x, snap=_snap,
                        li=float(mygs.get_stats(li_normalization="iter",
                                                lcfs_pad=psi_pad)["l_i"]))

        # the reference: the ONE reconstruction state (l_i_target IS its l_i)
        _ds = getattr(bl, "delivered_state", None) or {}
        _st0 = mygs.get_stats(li_normalization="iter", lcfs_pad=psi_pad)
        from .physics import SOLVER_Q0_PSI_N
        reference = dict(
            l_i=float(bl.l_i_target),
            q0=float(_ds.get("q0", float("nan"))),
            # every q0 in this check is get_stats' (psi_N 0.02): like for like
            q0_psi_N=float(SOLVER_Q0_PSI_N),
            q95=float(_ds.get("q95", float("nan"))),
            recorded=bool(_ds),
            anchor_resolve=dict(l_i=float(_st0["l_i"]),
                                q0=float(_st0.get("q_0", float("nan"))),
                                q95=float(_st0.get("q_95", float("nan")))),
            definition=("the reconstruction's delivered equilibrium "
                        "(Baseline.delivered_state; l_i == l_i_target); "
                        "anchor_resolve is this check's own re-solve of the "
                        "stored request"))

        spike0, _full0, _d0 = compose(mygs.copy_eq())
        li0 = float(_st0["l_i"])
        res = run_jbs_loop(spike0, _step,
                           lambda m: compose(m["snap"])[0], settings,
                           Ip=abs(Ip), meas0=dict(li=li0), gate_li=True,
                           label="sigma=0 check",
                           init_source=("evaluate_jBS on the state anchor "
                                        "with the baseline (sigma=0) "
                                        "kinetics"),
                           raise_on_fail=False)
        w, x, _k = residual_weights(mygs.copy_eq(), psi_N, psi_pad, coord=coord)
        cmp_ = profile_residuals(res["jbs_used"], ref, w, x, abs(Ip))
        li_s0 = float(mygs.get_stats(li_normalization="iter",
                                     lcfs_pad=psi_pad)["l_i"])
        li_ref = float(bl.l_i_target)
        li_ref_name = "l_i_target (the delivered reconstruction state)"
        dli = abs(li_s0 - li_ref)
        dev = np.asarray(res["jbs_used"], dtype=float) - ref
        peak = float(np.max(np.abs(ref))) or 1.0
        # the loop reproduces F solved F's way -- recorded, not `passed`
        passed_baseline_way = bool(res["converged"]
                      and cmp_["r_j"] <= settings["rtol_j"]
                      and cmp_["r_I"] <= settings["rtol_Ip"]
                      and dli <= settings["tol_li"])
        out = dict(spike0=np.asarray(res["jbs_used"], dtype=float),
                   max_dev=float(np.max(np.abs(dev))),
                   rms_dev=float(np.sqrt(np.mean(dev ** 2))),
                   max_dev_frac=float(np.max(np.abs(dev)) / peak),
                   psi_worst=float(psi_N[int(np.argmax(np.abs(dev)))]),
                   passed=False, passed_baseline_way=passed_baseline_way,
                   invariant="jbs-loop",
                   loop_converged=bool(res["converged"]),
                   r_j_vs_baseline=float(cmp_["r_j"]),
                   r_I_vs_baseline=float(cmp_["r_I"]),
                   li_sigma0=li_s0, li_baseline=li_ref,
                   li_baseline_reference=li_ref_name,
                   dl_i_vs_baseline=float(dli),
                   reference=reference,
                   split_closure=split_closure,
                   record=jsonable(res["record"]))
        # leave mygs re-anchored on the baseline equilibrium
        mygs.set_targets(Ip=Ip, pax=solver_pax(pressure, _edge))
        mygs.set_profiles(pp_prof=pp, ffp_prof=ffp)
        try:
            mygs.solve()
        except (ValueError, RuntimeError):
            pass
        print(f"[sigma0-check jbs-loop] baseline's way (recorded, not the "
              f"verdict): {'PASS' if passed_baseline_way else 'FAIL'}: "
              f"loop {'converged' if res['converged'] else 'NOT converged'} "
              f"in {res['record']['n_passes']} pass(es); vs baseline "
              f"r_j={cmp_['r_j']:.3e} (tol {settings['rtol_j']:.0e}), "
              f"r_I={cmp_['r_I']:.3e} (tol {settings['rtol_Ip']:.0e}), "
              f"|dl_i|={dli:.2e} (tol {settings['tol_li']:.0e})")
        if not draw_route:
            out["passed_reason"] = ("draw route not run (draw_route=False): "
                                    "the zero-perturbation identity of the "
                                    "draws is unverified")
            print("[sigma0-check jbs-loop] FAIL (unverified): "
                  + out["passed_reason"])
            return out
        # the draw's own route(s) at sigma=0 -- THIS decides `passed`; mygs
        # is handed back exactly as left above
        _post = (mygs.copy_eq() if (hasattr(mygs, "copy_eq")
                                    and hasattr(mygs, "replace_eq"))
                 else None)
        try:
            out["draw_route"] = self._sigma0_draw_route(
                settings, entry_snap, pressure, ne_eq, te_eq, ni_eq,
                ti_eq, Zeff_eq, psi_N, psi_pad, ref, li_ref, li_ref_name,
                routes=draw_routes, reference=reference)
        finally:
            if _post is not None:
                mygs.replace_eq(source_eq=_post)
        out["passed"] = bool(out["draw_route"].get("passed_draw_route"))
        out["passed_reason"] = (
            "every route the configuration can use reproduces the "
            "reconstruction state at the loop tolerances" if out["passed"]
            else "a zero-perturbation draw route misses the reconstruction "
                 "state at the loop tolerances (see draw_route)")
        print(f"[sigma0-check jbs-loop] {'PASS' if out['passed'] else 'FAIL'}"
              f": {out['passed_reason']}")
        return out

    def _sigma0_draw_routes(self):
        """The draw routes the configuration can use: the configured one
        first, plus the other when the workflow admits it (the g-file path
        runs either route; the modelling-source path refuses the standard
        route unless ``workflow="custom"`` / ``allow_unsafe_workflow``)."""
        from .config import ReconstructionSource
        gc = self.config.generation
        mine = "ip_renorm" if gc.perturb_jind_in_anchor else "standard"
        other = "standard" if mine == "ip_renorm" else "ip_renorm"
        src = getattr(self.config, "source", None)
        custom = (str(getattr(gc, "workflow", "auto")) == "custom"
                  or bool(getattr(gc, "allow_unsafe_workflow", False)))
        if isinstance(src, ReconstructionSource) or (src is not None
                                                     and custom):
            return (mine, other)
        return (mine,)

    def _sigma0_draw_route(self, settings, entry_snap, pressure, ne_eq,
                           te_eq, ni_eq, ti_eq, Zeff_eq, psi_N, psi_pad, ref,
                           li_delivered, li_delivered_name, routes=None,
                           reference=None):
        """The ``draw_route`` block of :meth:`verify_sigma0_consistency`.

        Runs :func:`perturb_kinetic_equilibrium` -- the function every draw of
        ``generate()`` runs -- with the arguments ``generate()`` /
        ``generate_bouquet`` hand it, every sigma set to zero (kinetic,
        j_phi and the auxiliary channels), the bootstrap scale at the centre
        the draws are sampled around (``sigma0_reference_scale`` of the
        bs_scale-centred ``jBS_scale_range``; the reconstruction's own value
        for any range symmetric about 1) and, in ``jbs_delta_mode``, the
        sigma=0 reference evaluated the way ``generate_bouquet`` caches it.
        Each route starts from ``entry_snap`` (the state the check was
        handed).  Per route it measures, against the ONE reconstruction state
        (``ref`` = its bootstrap, ``li_delivered`` = its l_i, ``reference`` =
        its q0/q95 and ``Baseline.delivered_state["j_phi_achieved"]``): the
        bootstrap profile / current residuals, l_i, q0, q95 and the
        total-current profile.  ``passed_draw_route`` (every route: loop
        converged, ``r_j <= rtol_j``, ``r_I <= rtol_Ip``, ``|dl_i| <=
        tol_li``) is what :meth:`verify_sigma0_consistency`'s ``passed``
        requires.  Default routes: :meth:`_sigma0_draw_routes`.
        """
        import numpy as np
        from .baseline import resolve_uncertainty
        from .jbs_loop import profile_residuals, residual_weights
        from .sampling import make_rng
        from .TokaMaker_interface import (_achieved_jphi_fsa,
                                          perturb_kinetic_equilibrium,
                                          sigma0_reference_scale)

        bl = self.baseline
        mygs = self.mygs
        _edge = resolve_edge_pressure(self.config.generation)
        gc = self.config.generation
        coord = getattr(bl, "coord", coords.PSI)
        from .physics import ELEMENTARY_CHARGE as EC
        Ip = float(bl.Ip_target)
        if routes is None:
            routes = self._sigma0_draw_routes()
        reference = dict(reference or {})
        _ds = getattr(bl, "delivered_state", None) or {}
        _ja_ref = _ds.get("j_phi_achieved")
        _bs = float(getattr(bl, "bs_scale", 1.0))
        # exactly the draws' bootstrap scaling (generate() uses the same
        # helper): the range's centre and the after-SWB profile
        try:
            _rng_range, _prof, _scaling = self._draw_bootstrap_scaling()
        except ValueError as _se:
            return dict(error=f"bootstrap scaling: {str(_se)[:400]}",
                        passed_draw_route=False, routes={})
        scale0 = float(sigma0_reference_scale(_rng_range))
        # generate_bouquet hands every draw the THERMAL pressure on the
        # equilibrium grid; the draw adds impurity/fast/diff itself
        p_th = EC * (ne_eq * te_eq + ni_eq * ti_eq)
        psi_kin = np.asarray(bl.psi_N_kinetic, dtype=float)
        zk, zj = np.zeros_like(psi_kin), np.zeros_like(psi_N)
        jdiff = (None if getattr(bl, "jBS_diff", None) is None
                 else np.asarray(bl.jBS_diff, dtype=float))
        _off = getattr(bl, "jphi_request_offset", None)
        blk = dict(
            what=("perturb_kinetic_equilibrium at zero perturbation with "
                  "generate()'s draw arguments (the draw's own route); "
                  "generate()'s coil regularisation, hard bounds and "
                  "homotopy are not part of the replay"),
            scale_jBS=scale0, scale_jBS_reconstruction=_bs,
            bootstrap_scaling=_scaling,
            fixed_channels=list(self.DRAW_FIXED_CHANNELS),
            l_i_target=float(bl.l_i_target),
            li_delivered=float(li_delivered),
            li_delivered_reference=str(li_delivered_name),
            reference=reference,
            tolerances=dict(rtol_j=settings["rtol_j"],
                            rtol_Ip=settings["rtol_Ip"],
                            tol_li=settings["tol_li"]),
            criterion=("loop converged and r_j <= rtol_j and r_I <= rtol_Ip "
                       "and |l_i(draw) - l_i_target| <= tol_li, on EVERY "
                       "route; q0, q95 and the total-current profile are "
                       "reported beside it"),
            gates=("verify_sigma0_consistency's `passed` (self-consistent "
                   "loop): the zero-perturbation identity of the draws"),
            routes={})
        try:
            env = resolve_uncertainty(self.config, bl)
        except Exception as e:
            blk.update(error=f"resolve_uncertainty: {type(e).__name__}: "
                             f"{str(e)[:300]}", passed_draw_route=False)
            print("[sigma0-check draw-route] NOT RUN: " + blk["error"])
            return blk
        _aux = env.get("aux_sigmas")
        aux_zero = (None if not _aux else
                    {k: np.zeros_like(np.asarray(v, dtype=float))
                     for k, v in _aux.items()})

        def _restore():
            if entry_snap is not None:
                mygs.replace_eq(source_eq=entry_snap)

        dref = dbase = None
        if bool(getattr(gc, "jbs_delta_mode", False)):
            # generate_bouquet's delta cache under the loop: evaluate_jBS
            # (RAW, at the centre scale) on a state-anchor solve of the
            # baseline total (+ jphi_diff) at the full pressure
            from .physics import evaluate_jBS
            from .utils import pchip_derivative
            _restore()
            _pr = mygs.psi_bounds[1] - mygs.psi_bounds[0]
            _pp = solver_pp_profile(psi_N, pressure, _pr, _edge, coord=coord)
            _yc = np.asarray(bl.j_phi, dtype=float).copy()
            if getattr(bl, "jphi_diff", None) is not None:
                _yc = _yc + np.asarray(bl.jphi_diff, dtype=float)
            mygs.set_targets(Ip=Ip, pax=solver_pax(pressure, _edge))
            mygs.set_profiles(pp_prof=_pp, ffp_prof=coords.oft_prof(
                "jphi-linterp", psi_N, _yc, coord))
            try:
                mygs.solve()
            except (ValueError, RuntimeError):
                pass                      # generate_bouquet tolerates it too
            _sel, _d = evaluate_jBS(mygs, psi_N, ne_eq, te_eq, ni_eq, ti_eq,
                                    Zeff_eq, psi_pad=psi_pad,
                                    isolate_edge=bool(gc.isolate_edge_jBS),
                                    smooth_axis=False, coord=coord)
            dref = scale0 * np.asarray(_sel, dtype=float)
            dbase = np.asarray(ref, dtype=float)
            blk["delta_mode"] = True

        passed_all = True
        for route in routes:
            rr = dict(route=str(route),
                      perturb_jind_in_anchor=(route == "ip_renorm"))
            try:
                _restore()
                d = perturb_kinetic_equilibrium(
                    mygs, psi_N, p_th, bl.ne, bl.te, bl.ni, bl.ti,
                    np.asarray(bl.j_phi, dtype=float), zk, zk, zk, zk, zj,
                    env["n_ls"], env["t_ls"], env["j_ls"], Ip,
                    float(bl.l_i_target), Zeff_eq, len(psi_N),
                    input_jinductive=np.asarray(bl.j_inductive, dtype=float),
                    l_i_tolerance=gc.l_i_tolerance, psi_pad=psi_pad,
                    constrain_sawteeth=gc.constrain_sawteeth,
                    recalculate_j_BS=gc.recalculate_j_BS,
                    isolate_edge_jBS=gc.isolate_edge_jBS,
                    floor_j_BS=gc.floor_j_BS, jBS_diff=jdiff,
                    accept_anchor_inband=gc.accept_anchor_inband,
                    perturb_jind_in_anchor=(route == "ip_renorm"),
                    scale_jBS=scale0, jBS_scale_profile=_prof,
                    **gc.bootstrap_kwargs,
                    edge_pressure=_edge,
                    diagnostic_plots=False, psi_N_kinetic=psi_kin,
                    p_fast=bl.p_fast, z_fast=getattr(bl, "z_fast", None),
                    z2_fast=getattr(bl, "z2_fast", None),
                    zeff_includes_fast=bool(getattr(
                        bl, "zeff_includes_fast", False)),
                    j_NBI=bl.j_NBI, j_RF=bl.j_RF,
                    j_other=getattr(bl, "j_other", None),
                    aux_sigmas=aux_zero,
                    aux_baselines=env.get("aux_baselines"),
                    aux_length_scales=env.get("aux_length_scales"),
                    ni_from_zeff=env.get("ni_from_zeff", True),
                    zeff_dne=env.get("zeff_dne"),
                    # generate_bouquet's own defaults for these two
                    max_proxy_draws=500, p_thresh=0.05,
                    rng=make_rng(gc.seed),
                    spike_delta_ref=dref, spike_delta_baseline=dbase,
                    Z_imp=getattr(bl, "Z_imp", None),
                    p_diff=getattr(bl, "p_diff", None),
                    jphi_diff=getattr(bl, "jphi_diff", None),
                    jbs_loop=settings, jphi_request_offset=_off,
                    coord=coord)[6]
                snap = mygs.copy_eq()
                w, x, _k = residual_weights(snap, psi_N, psi_pad, coord=coord)
                ctx = d.get("_jbs_ctx") or {}
                if ctx.get("spike_used") is not None:
                    spk = np.asarray(ctx["spike_used"], dtype=float)
                    rr["bootstrap_compared"] = ("the draw's loop bootstrap "
                                                "(composed, incl. jBS_diff)")
                else:
                    spk = np.asarray(d["j_BS_edge"] if d.get("j_BS_edge")
                                     is not None else d["j_BS"], dtype=float)
                    rr["bootstrap_compared"] = "the draw's archived j_BS split"
                cmp_ = profile_residuals(spk, ref, w, x, abs(Ip))
                _st = mygs.get_stats(li_normalization="iter",
                                     lcfs_pad=psi_pad)
                li_d = float(_st["l_i"])
                q0_d = float(_st.get("q_0", float("nan")))
                q95_d = float(_st.get("q_95", float("nan")))
                jl = d.get("jbs_loop") or {}
                conv = bool(jl.get("converged", False))
                dli_t = abs(li_d - float(bl.l_i_target))
                dli_d = abs(li_d - float(li_delivered))
                # the total-current profile: the draw's achieved current vs
                # the reconstruction's (current-weighted, the loop's norm)
                jphi_cmp = None
                if _ja_ref is not None:
                    try:
                        _jd = _achieved_jphi_fsa(mygs, psi_N, psi_pad,
                                                 sign_ref=_ja_ref, coord=coord)
                        _c = profile_residuals(_jd, np.asarray(
                            _ja_ref, dtype=float), w, x, abs(Ip))
                        jphi_cmp = dict(r_j=float(_c["r_j"]),
                                        r_I=float(_c["r_I"]))
                    except Exception as _je:
                        jphi_cmp = dict(error=f"{type(_je).__name__}: "
                                              f"{str(_je)[:200]}")
                ok = bool(conv and cmp_["r_j"] <= settings["rtol_j"]
                          and cmp_["r_I"] <= settings["rtol_Ip"]
                          and dli_t <= settings["tol_li"])
                rr.update(loop_converged=conv,
                          n_loops=jl.get("n_loops"),
                          passes_used=jl.get("n_passes_total"),
                          r_j=float(cmp_["r_j"]), r_I=float(cmp_["r_I"]),
                          li_draw=li_d, dl_i_vs_l_i_target=float(dli_t),
                          dl_i_vs_delivered=float(dli_d),
                          q0_draw=q0_d, q95_draw=q95_d,
                          dq0_vs_reference=float(
                              q0_d - reference.get("q0", float("nan"))),
                          dq95_vs_reference=float(
                              q95_d - reference.get("q95", float("nan"))),
                          j_phi_vs_reference=jphi_cmp,
                          r2_ip_scale=d.get("r2_ip_scale"),
                          r2_f_ind=d.get("r2_f_ind"),
                          j0_scales=[float(v) for v in
                                     (d.get("j0_scales") or [])],
                          passed_draw_route=ok, error=None)
            except Exception as e:
                ok = False
                rr.update(passed_draw_route=False,
                          error=f"{type(e).__name__}: {str(e)[:300]}")
            passed_all = passed_all and ok
            blk["routes"][str(route)] = rr
            if rr.get("error"):
                print(f"[sigma0-check draw-route {route}] FAIL: the draw "
                      f"route raised {rr['error']}")
            else:
                _sc = (f"R2 scale={rr['r2_ip_scale']:.6f}"
                       if rr.get("r2_ip_scale") is not None else
                       "j0 scales=" + ",".join(f"{v:.4f}"
                                               for v in rr["j0_scales"]))
                _jc = rr.get("j_phi_vs_reference") or {}
                print(f"[sigma0-check draw-route {route}] "
                      f"{'PASS' if ok else 'FAIL'}: loop "
                      f"{'converged' if rr['loop_converged'] else 'NOT converged'}"
                      f" in {rr['passes_used']} pass(es); vs the "
                      f"reconstruction state r_j={rr['r_j']:.3e} (tol "
                      f"{settings['rtol_j']:.0e}), r_I={rr['r_I']:.3e} (tol "
                      f"{settings['rtol_Ip']:.0e}), |dl_i|="
                      f"{rr['dl_i_vs_l_i_target']:.2e} (tol "
                      f"{settings['tol_li']:.0e}); dq0="
                      f"{rr['dq0_vs_reference']:+.2e} dq95="
                      f"{rr['dq95_vs_reference']:+.2e}"
                      + (f" j_phi r_j={_jc['r_j']:.2e}" if "r_j" in _jc
                         else "") + f"; {_sc}")
        blk["passed_draw_route"] = bool(passed_all)
        return blk

    @staticmethod
    def _check_jbs_loop_workflow(gc) -> None:
        """Refuse a self-consistent bootstrap loop with nothing to iterate.

        Raised outright -- not a workflow-lock problem the ``"custom"`` escape
        hatch could downgrade -- because the combination has no meaning:
        ``single_profile_jphi`` has no bootstrap component, and
        ``recalculate_j_BS=False`` turns off exactly the re-evaluation the
        loop is.  Called from :meth:`prepare_baseline` (the baseline loops
        without ``generate()``) and from :meth:`_validate_workflow`.
        """
        from .jbs_loop import validate_jbs_settings
        validate_jbs_settings(gc)
        if not bool(getattr(gc, "jbs_self_consistent", False)):
            return
        if bool(getattr(gc, "single_profile_jphi", False)):
            raise ValueError(
                "jbs_self_consistent=True (the default) with "
                "single_profile_jphi=True: there is no bootstrap component "
                "to iterate (j_phi is one profile).  Set "
                "generation.jbs_self_consistent=False for a single-profile "
                "run.")
        if not bool(getattr(gc, "recalculate_j_BS", True)):
            raise ValueError(
                "jbs_self_consistent=True (the default) with "
                "recalculate_j_BS=False: the loop re-evaluates the bootstrap "
                "on every equilibrium, which is exactly what "
                "recalculate_j_BS=False turns off.  Set "
                "generation.jbs_self_consistent=False to keep the baseline "
                "bootstrap frozen in the draws.")

    # ── stage 3: perturbed bouquet --------------------------------------
    def _check_edge_pressure_unchanged(self, gc) -> None:
        """Refuse draws whose edge-pressure settings (``edge_pprime_pin``,
        ``separatrix_pressure``) differ from the ones the baseline was
        reconstructed with: the legacy draws read the CONFIG's at generate()
        (their pressures would then be in another frame than the baseline
        they perturb), the engine draws the reconstruction's (the changed
        setting would be silently ignored).  A baseline that predates the
        record is not checked."""
        from .edge_pressure import resolve_edge_pressure
        rec = getattr(self.baseline, "edge_pressure", None) or {}
        if "separatrix_pressure" not in rec:
            return
        now = resolve_edge_pressure(gc).record()
        diff = [k for k in ("edge_pprime_pin", "separatrix_pressure")
                if k in rec and rec[k] != now[k]]
        if diff:
            raise ValueError(
                "generation." + ", ".join(f"{k}={now[k]!r}" for k in diff)
                + " differs from the baseline's ("
                + ", ".join(f"{k}={rec[k]!r}" for k in diff)
                + ", the settings prepare_baseline() reconstructed it with): "
                "the draws would mix pressure frames (legacy) or ignore the "
                "change (engine). Re-run prepare_baseline() with the new "
                "settings, or restore them.")

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
        self._check_jbs_loop_workflow(gc)
        # a method with its own route rules (engine: one draw route for both
        # inputs; swb: one SWB solve per draw) replaces the legacy ones below
        from .draw_methods import method_hooks
        _hooks = method_hooks(gc)
        _own = _hooks.workflow_problems(self)
        if _own is not None:
            problems += _own
            if not problems:
                return
            msg = ("bouquet workflow guard: " + "; ".join(problems)
                   + ". Set config.generation.workflow='custom' to override "
                   "(backend tests / experiments only).")
            if custom:
                print("WARN: " + msg)
                return
            raise ValueError(msg)
        # closure_channel is consumed ONLY by the IMAS hybrid baseline
        # (jBS_baseline_mode="ohmic" with recalculate_j_BS=True).  Anywhere
        # else it used to be resolved (the structured preset even printed
        # "applied by default"), then never read -- so the caller believed a
        # closure ran that did not.  Refuse instead of ignoring.
        _chan = str(getattr(gc, "closure_channel", "bootstrap"))
        if _chan != "bootstrap":
            _is_imas = isinstance(self.config.source, ImasSource)
            if not _is_imas:
                problems.append(
                    f"closure_channel={_chan!r} is only read on the IMAS "
                    "path with jBS_baseline_mode='ohmic' (the hybrid "
                    "baseline); a g-file source has no FUSE j_inductive to "
                    "close on, so the channel would be silently ignored")
            elif str(gc.jBS_baseline_mode) != "ohmic":
                problems.append(
                    f"closure_channel={_chan!r} is only read when "
                    f"jBS_baseline_mode='ohmic' (it is "
                    f"{str(gc.jBS_baseline_mode)!r}), so it would be "
                    "silently ignored; set jBS_baseline_mode='ohmic' or "
                    "leave closure_channel at 'bootstrap'")
            elif not bool(gc.recalculate_j_BS):
                problems.append(
                    f"closure_channel={_chan!r} needs recalculate_j_BS=True "
                    "(the closure runs inside the SWB bootstrap split); with "
                    "recalculate_j_BS=False it would be silently ignored")
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

    def generate(self, n: Optional[int] = None, progress_callback=None,
                 on_inspec=None, stop_check=None) -> list:
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
        from .TokaMaker_interface import DrawSolveGuard, generate_bouquet
        from .utils import initialize_equilibrium_database

        if self.baseline is None:
            raise ValueError("call prepare_baseline() before generate()")
        if self.mygs is None:
            raise ValueError("call setup_solver() before generate()")
        self._ensure_engine_defaults_resolved()
        self._refuse_unified_engine_draws("generate()")

        self._validate_workflow()

        bl = self.baseline
        # A baseline set directly skips prepare's guard: re-check its coordinate.
        coords.check_run(self.config, getattr(bl, "coord", coords.PSI))
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
        # the draw method of the solve method (bouquet.draw_methods): the
        # legacy draws, the unified engine or swb
        _m = self._draw_method(env)
        env = _m.draw_env(env)

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
        # The baseline built j_BS as (multiplier) x SWB(scale 1); the draws
        # apply the same multiplier after SWB (jBS_scale_profile) and keep
        # jBS_scale_range as the per-draw jitter inside it.  Passing the
        # multiplier INTO SWB as scale_jBS instead is not the same thing:
        # OFT applies it inside SWB's self-consistent iteration.
        # Under the self-consistent loop the composer takes the multiplier
        # as its (linear) scale instead: the range re-centred on bs_scale
        # (_draw_bootstrap_scaling, shared with both sigma=0 guards).
        # The method's scale range and profile (swb: neither, SWB re-solves
        # j_BS at scale 1; engine: the range on top of s_bs(x*)).
        _jbs_range, _bs_mult, _bs_scaling = self._draw_bootstrap_scaling(_m)
        self._draw_bootstrap_scaling_record = _bs_scaling
        if ((getattr(self, "_swb_seed_record", None) or {}).get("resolved")
                == "source_unavailable" and bool(gc.recalculate_j_BS)
                and not self._draws_run_jbs_loop(_m)):
            # the draws would call SWB with the generic seed in place of the
            # source seed that was asked for
            raise self._swb_source_unavailable()

        # The LCFS boundary cut, resolved ONCE and OUTSIDE the output capture:
        # the announcement must reach the user (inside the capture it went to
        # generation_log only), and this one local is both what the until-N
        # verdict applies and what is stamped on the archive below.
        _cut_mm, _cut_source = self._boundary_cut()

        from .utils import capture_native_output
        from .jbs_loop import jbs_settings as _jbs_settings
        _jbs_draw = _m.loop_settings_for(_jbs_settings(gc, draw=True))
        verbose = bool(getattr(self.config, "verbose", False))
        _rejections = []
        # the draw-loop GS iteration cap, its recovery and the failed-solve
        # record (DrawSolveGuard)
        with capture_native_output(enabled=not verbose) as _cap, \
                DrawSolveGuard(self.mygs,
                               _m.solve_maxits(gc.draw_solve_maxits),
                               retry_urf=_m.solve_retry_urf(gc.draw_solve_retry_urf),
                               loose_tol=_m.solve_loose_tol(gc.draw_solve_loose_tol)) as _solve_guard:
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
                jBS_scale_profile=_bs_mult,
                edge_pressure=resolve_edge_pressure(gc),
                jbs_delta_mode=gc.jbs_delta_mode,
                diagnostic_plots=gc.diagnostic_plots,
                capture_live_eq=gc.capture_live_eq,
                capture_npsi=gc.capture_npsi,
                capture_exact_inv_R2=gc.capture_exact_inv_R2,
                write_ifile=gc.write_ifile,
                ifile_npsi=gc.ifile_npsi,
                ifile_ntheta=gc.ifile_ntheta,
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
                inspec_rms_max_mm=_cut_mm,
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
                z2_fast=getattr(bl, "z2_fast", None),
                zeff_includes_fast=bool(getattr(bl, "zeff_includes_fast",
                                                False)),
                p_diff=getattr(bl, "p_diff", None),
                # Total-current anchor to equilibrium.j_tor (fixed offset; rides
                # under the SWB bootstrap + perturbed j_ind in every draw).
                jphi_diff=getattr(bl, "jphi_diff", None),
                j_NBI=bl.j_NBI,
                j_RF=bl.j_RF,
                j_other=getattr(bl, "j_other", None),
                # Switchboard: auxiliary perturbed profiles -- rotation /
                # transport channels (passive) + Zeff (active).
                aux_sigmas=env.get("aux_sigmas"),
                aux_baselines=env.get("aux_baselines"),
                aux_length_scales=env.get("aux_length_scales"),
                # Who draws ni when zeff is active (see UncertaintyConfig).
                ni_from_zeff=env.get("ni_from_zeff", True),
                zeff_dne=env.get("zeff_dne"),
                progress_callback=progress_callback,
                # shared until-N hooks (bouquet.parallel); None on the serial path
                on_inspec=on_inspec,
                stop_check=stop_check,
                # Provenance marker stored on the baseline for robust path
                # detection in plotting (independent of the aux switchboard).
                source_kind=("imas"
                             if type(self.config.source).__name__ == "ImasSource"
                             else "geqdsk"),
                # Baseline provenance for readers: the li_metrics dict (l_i
                # comparison, forward-solve residuals, jBS_baseline_mode,
                # scales) and -- on a closed hybrid baseline -- the full
                # ip_closure health record with its closure_limited verdict.
                # Downstream bands need that flag per slice; before this it
                # lived only on the in-memory Baseline object.
                baseline_meta=getattr(bl, "li_metrics", None),
                # BOTH paths: archive the ACHIEVED FSA j_phi of each converged
                # solve (baseline + draws) so the stored 1-D current always
                # matches the stored eqdsk in the same group. On the IMAS path
                # the single-pass forward solve lands a few % off its anchor
                # target; on the geqdsk path the per-draw corrective iteration
                # bounds the target-vs-achieved gap to its tolerance (~2-3%
                # core RMS) -- storing the achieved output removes even that.
                store_achieved_jphi=_m.achieved_jphi(True),
                # Self-consistent bootstrap: the per-draw loop settings (None
                # -> the legacy frozen-SWB draws, bit for bit).
                jbs_loop=_m.draw_jbs_loop(_jbs_draw if _jbs_draw["enabled"]
                                          else None),
                rejection_log=_rejections,
                # the ONE reconstruction state (loop only; None = legacy)
                jphi_request_offset=getattr(bl, "jphi_request_offset", None),
                delivered_state=getattr(bl, "delivered_state", None),
                # closure_channel="structured" + mse_data only (else None):
                # the per-chord MSE arrays, archived as datasets
                baseline_mse_record=getattr(bl, "mse_record", None),
                solve_guard=_solve_guard,
                draw_method=_m,
                coord=getattr(bl, "coord", coords.PSI),
                source_seed_profile=getattr(bl, "swb_seed_profile", None),
                source_jphi_fixed=getattr(bl, "swb_jphi_fixed", None),
                **gc.bootstrap_kwargs,
            )
        self.generation_log = _cap["text"] or None
        self.draw_rejections = list(_rejections)
        # Outside the capture: failed solves are caught by the draw path, so
        # this line is their only trace in a quiet run's log.
        self.solve_failures = list(_solve_guard.records)
        print(_solve_guard.summary())

        # Rejected draw attempts, OUTSIDE the capture: a draw the j_BS loop
        # (or the coil-saturation guard, or the homotopy) rejected is never
        # archived and never counted, so the only trace of it is here.
        from .TokaMaker_interface import _rejection_summary
        _n_arch = len(self.diagnostics or [])
        _summ = _rejection_summary(self.draw_rejections,
                                   len(self.draw_rejections) + _n_arch,
                                   _n_arch)
        print(f"[generate] {_summ}")
        _loop_codes = _m.loop_codes((
            "jbs_not_converged", "coil_saturation_jbs_loop",
            "jbs_post_homotopy", "coil_saturation_post_homotopy",
            "jbs_post_homotopy_error", "anchor_solve_failed"))
        _n_loop = sum(1 for r in self.draw_rejections
                      if r.get("reason") in _loop_codes)
        if _n_loop:
            print(f"[generate] {_n_loop} of them rejected by the "
                  "self-consistent j_BS loop stage or its coil-saturation "
                  "guard; per-attempt records: Bouquet.draw_rejections")
        # the method's own summary, outside the capture (engine: its capped
        # solves, Bouquet.engine_draw_cap_events)
        _m.summarize(self)

        # The cut the until-N loop counted against, next to the counts it
        # produced (generation provenance). Only an until-N loop applies a cut
        # while drawing; a fixed-N run records none (and clears a stale one).
        from .utils import stamp_generation_provenance
        _until = gc.n_inspec_target is not None
        try:
            stamp_generation_provenance(
                header, scan_key=gc.scan_key,
                inspec_rms_max_mm=(_cut_mm if _until else None),
                inspec_cut_source=(_cut_source if _until else None))
        except OSError as _exc:        # the draws are stored; say what is missing
            import warnings as _w
            _w.warn(f"the until-N boundary cut ({_cut_mm} mm, {_cut_source}) "
                    f"was applied but could not be stamped on the archive: {_exc}",
                    RuntimeWarning, stacklevel=2)

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
            # attempts, not stored draws: failed draws leave no group, so the
            # count comes from the generation-provenance stamp
            from .utils import read_generation_provenance
            _tries = read_generation_provenance(header, scan_key=gc.scan_key).get(
                "n_attempted") or len(self.diagnostics or [])
            _shared_done = False
            if stop_check is not None:
                try:
                    _shared_done = bool(stop_check())
                except Exception:
                    _shared_done = False
            if _shared_done:
                # This worker's LOCAL target was not the run's target: the
                # pooled count crossed the shared target, which is the only
                # verdict that matters here.
                print(f"[until-N] shared target reached: this worker "
                      f"delivered {_got} in-spec draws in {_tries} attempts.")
            elif _got < _tgt:
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

        # The baseline's self-consistent bootstrap record (schema v3 jbs_loop
        # block on _baseline; a frozen baseline carries none).
        from .utils import store_baseline_jbs_loop
        store_baseline_jbs_loop(header, self._baseline_jbs_record(),
                                scan_key=gc.scan_key)
        # the method's own record on _baseline (engine: the reconstruction's
        # engine record + the draws' settings; swb: solve A/B, the saw reset)
        _m.store_baseline(header, gc.scan_key, bl)

        # Stamp provenance (schema/version/timestamp + full config JSON) onto the
        # archive so the run is self-describing and load_config() can round-trip it.
        from .utils import stamp_coil_solve_mode, write_provenance
        write_provenance(header, config=self.config, scan_key=gc.scan_key)
        # the coil-solve mode the run's solver was in (both paths; read
        # back by load_config, which warns on a replay of a run made in
        # another mode or before the record)
        stamp_coil_solve_mode(header, scan_key=gc.scan_key,
                              mode=getattr(bl, "coil_solve_mode", None))
        # how the engine-dependent settings were resolved (both paths)
        from .utils import stamp_engine_resolved_defaults
        stamp_engine_resolved_defaults(
            header, scan_key=gc.scan_key,
            record=getattr(bl, "engine_resolved_defaults", None))
        # IMAS path: the source's current orientation (what the reader
        # multiplied every dd current by to reach bouquet's positive frame).
        from .config import ImasSource
        if isinstance(self.config.source, ImasSource):
            from .utils import stamp_source_orientation
            stamp_source_orientation(
                header, scan_key=gc.scan_key,
                current_sign=float(getattr(bl, "source_current_sign", 1.0)),
                b0_sign=getattr(bl, "source_b0_sign", None),
                current_sign_origin=getattr(bl, "source_current_sign_origin",
                                            None))

        return self.diagnostics

    def _draw_method(self, env):
        """The :class:`~bouquet.draw_methods.DrawMethod` of this run's solve
        method (docs/draw-methods.md): the legacy draws, the unified engine
        (from the live reconstruction of this session) or swb (solve B with
        resampled inputs)."""
        from .config import resolve_solve_method
        from .draw_methods import DrawMethod
        sm = resolve_solve_method(self.config.generation)
        if sm == "engine":
            from .engine_draws import build_generate_context
            _s0 = getattr(self, "_sigma0_route", None)
            if _s0 is not None:
                # verify_sigma0_consistency's route: this very draw route
                # with every perturbation zero (and bootstrap scale 1.0)
                env = _zero_perturbation_env(env)
            m = build_generate_context(self, env)
            gc = self.config.generation
            # the scale multiplies the Redl bootstrap ON TOP of s_bs(x*)
            # (which already carries the reconstruction's bootstrap
            # scaling), so the configured range is used as is -- 1.0 is the
            # reconstruction's value
            m.scale_range = (None if gc.jBS_scale_range is None
                             else (float(gc.jBS_scale_range[0]),
                                   float(gc.jBS_scale_range[1])))
            if _s0 is not None:
                m.scale_range = (1.0, 1.0)
                m.sigma0_probe = _s0
                _s0["draws"] = m
            return m
        if sm == "swb":
            from .swb import build_swb_context
            return build_swb_context(self, env)
        return DrawMethod()

    def _refuse_unified_engine_draws(self, what):
        """Refuse draws that would not reproduce their baseline.

        The draws run on the unified engine (Stage 3, bouquet.engine_draws)
        when the baseline was built by it in THIS session.  Refused -- the
        mismatched cases, where either route would be inconsistent at zero
        perturbation (the legacy routes compose the bootstrap with the
        legacy toroidal conversion and keep the pressure-driven term frozen
        in the inductive):

        * ``reconstruction_engine="unified"`` with a baseline the engine did
          not build (or whose live state is not in this session);
        * ``reconstruction_engine="legacy"`` with an engine-built baseline.
        """
        unified = getattr(self.config.generation, "reconstruction_engine",
                          "legacy") == "unified"
        bl = getattr(self, "baseline", None)
        eng_bl = getattr(bl, "engine", None) is not None
        run = getattr(self, "_engine_run", None)
        live = bool(run and run.get("baseline") is bl and bl is not None)
        if unified and not (eng_bl and live):
            raise NotImplementedError(
                f"{what}: reconstruction_engine='unified' draws (Stage 3) "
                "run from the unified engine's reconstruction built by "
                "prepare_baseline() in this session; this baseline was not "
                "(or its live state is gone) -- call prepare_baseline() with "
                "the unified engine first")
        if not unified and eng_bl:
            raise NotImplementedError(
                f"{what}: the baseline was built by the unified "
                "reconstruction engine but reconstruction_engine is now "
                "'legacy'; the legacy draw routes would not reproduce this "
                "reconstruction at zero perturbation (different bootstrap "
                "conversion and pressure-term bookkeeping; Stage 3 draws run "
                "on the engine).  Set reconstruction_engine='unified' or "
                "rebuild the baseline with 'legacy'.")
        # the edge-pressure settings the baseline was reconstructed with
        self._check_edge_pressure_unchanged(self.config.generation)

    def _baseline_jbs_record(self):
        """The baseline's self-consistent bootstrap loop record, or ``None``
        (a frozen baseline): ``li_metrics["jbs_loop"]`` on the IMAS path,
        ``reconstruction_metrics["jbs_loop"]`` on the geqdsk path."""
        bl = getattr(self, "baseline", None)
        if bl is None:
            return None
        for attr in ("li_metrics", "reconstruction_metrics"):
            rec = (getattr(bl, attr, None) or {}).get("jbs_loop")
            if rec is not None:
                return rec
        return None

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
        kwargs.setdefault("rms_max_mm", self._boundary_cut(quiet=True)[0])
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
        _cut = self._boundary_cut(quiet=True)[0]
        if _cut is not None:            # disabled: keep the plot's own scale
            kwargs.setdefault("rms_max_mm", _cut)
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
    def filter(self, rms_max_mm=None, plot: bool = False) -> dict:
        """Mark the machine-realizable subset (coil + boundary filters).

        ``rms_max_mm``: None (default) applies ``filtering.rms_max_mm``; a
        number, ``"auto"`` or ``"off"`` overrides it for this call (see
        :meth:`boundary_cut`). The cut applied is printed and stamped on the
        archive with its source.

        Non-destructive: writes pass flags into the HDF5. Returns a summary dict.
        With ``plot=True`` the coil-drift and boundary distribution figures are
        produced and returned under ``summary["figures"] = (coil_fig, bnd_fig)``
        (F7 -- the notebooks re-called the module filters just to get these).
        """
        from .filtering import filter_coil_currents, filter_boundaries, filter_coil_chi2

        header = self.config.output_header
        fc = self.config.filtering

        sk = self.config.generation.scan_key
        coil_filter_used = fc.coil_filter
        # the boundary cut: the argument when given (a number, "auto" or
        # "off"), else filtering.rms_max_mm -- the SAME resolution the until-N
        # loop used (identity). Announced either way.
        if rms_max_mm is None:
            rms, rms_source = self._boundary_cut()
        else:
            rms, rms_source = self._boundary_cut(setting=rms_max_mm,
                                                 origin="argument")
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
        self._check_against_inloop_cut(rms, rms_source)
        bnd_summary, bnd_fig = filter_boundaries(
            header, scan_key=sk, rms_max_mm=rms, apply=True, plot=plot,
            cut_source=rms_source,
        )
        # one scan key -> each summary is a single {counts, draws} dict
        self._selection = {"coil": coil_summary, "boundary": bnd_summary,
                           "coil_filter_used": coil_filter_used}
        if plot:
            self._selection["figures"] = (coil_fig, bnd_fig)
        self._print_generation_summary(coil_summary, bnd_summary)
        return self._selection

    def boundary_cut(self):
        """``(rms_max_mm, source)`` of the LCFS boundary cut this run applies.

        Resolved from ``filtering.rms_max_mm`` exactly as :meth:`generate`'s
        until-N verdict and :meth:`filter` resolve it; ``rms_max_mm`` is None
        when the cut is disabled (``source="disabled"``). Quiet: prints
        nothing. Use it to draw the threshold a plot should show, e.g.
        ``bq.plot_traces(h5, rms_max_mm=run.boundary_cut()[0])``.
        """
        return self._boundary_cut(quiet=True)

    def _boundary_cut(self, quiet=False, setting=None, origin="config"):
        """``(rms_max_mm, source)`` -- the LCFS boundary cut this run applies.

        *setting* is ``filtering.rms_max_mm`` (``origin="config"``) or an
        explicit :meth:`filter` argument (``origin="argument"``):

        * a number -> that cut (``"explicit"``);
        * ``"off"`` or ``None`` -> no cut (``(None, "disabled")``; ``None``
          keeps its historical meaning, "no boundary cut");
        * ``"auto"`` -> the device's calibrated value (``"device:<name>"``,
          e.g. 8.5 mm on DIII-D from its boundary-UQ study) with the device
          taken from ``config.device`` or detected from the mesh's coil
          names, else the generic 5.0 mm (``"generic"``).

        Used by :meth:`generate`'s until-N verdict and by :meth:`filter`, so
        the two agree by construction. Printed once per resolution unless
        *quiet*.
        """
        from .config import check_boundary_cut_setting
        from .devices import boundary_cut_for
        if origin == "config":
            setting = self.config.filtering.rms_max_mm
        check_boundary_cut_setting(
            setting, "filtering.rms_max_mm" if origin == "config"
            else "filter(rms_max_mm=...)")
        if setting is None or setting == "off":
            return self._announce_boundary_cut(None, "disabled", None, quiet,
                                               origin, setting)
        if not isinstance(setting, str):
            return self._announce_boundary_cut(float(setting), "explicit", None,
                                               quiet, origin, setting)
        spec = self._boundary_cut_device()
        val, src = boundary_cut_for(spec)
        return self._announce_boundary_cut(val, src, spec, quiet, origin, setting)

    def _check_against_inloop_cut(self, rms, rms_source):
        """Warn when filter() cuts at a different boundary bound than the one
        the until-N loop counted its target against (read from the archive):
        the delivered count then does not describe this selection."""
        import warnings
        from .utils import read_generation_provenance
        try:
            gp = read_generation_provenance(self.config.output_header,
                                            scan_key=self.config.generation.scan_key)
        except OSError:
            return
        if gp.get("inspec_cut_source") is None:     # not an until-N run
            return
        loop = gp.get("inspec_rms_max_mm")
        loop = None if loop is None else float(loop)
        if loop != (None if rms is None else float(rms)):
            fmt = lambda v, s: ("no boundary cut" if v is None else f"{v:g} mm") + f" ({s})"
            warnings.warn(
                "boundary cut differs from the until-N loop's: the loop counted "
                f"its in-spec target against {fmt(loop, gp['inspec_cut_source'])} "
                f"but filter() now cuts at {fmt(rms, rms_source)}, so the "
                "delivered count does not describe this selection.",
                UserWarning, stacklevel=3)

    def _boundary_cut_device(self):
        """The device whose calibrated cut ``"auto"`` resolves to, or None:
        ``config.device``, else the live solver's coil names, else the
        archived baseline's coil names (a filter-only session)."""
        from .devices import resolve_device
        names = None
        try:
            if self.mygs is not None and getattr(self.mygs, "coil_sets", None):
                names = list(self.mygs.coil_sets)
        except Exception:
            names = None
        if names is None:
            # no live solver (e.g. a filter-only session): the archive's
            # baseline carries the coil names the chi2 filter reads too
            try:
                import h5py
                from .utils import _read_coil_names, _scan_key
                bkey = _scan_key(self.config.generation.scan_key)
                bl = f"scan/{bkey}/_baseline" if bkey is not None else "_baseline"
                with h5py.File(f"{self.config.output_header}.h5", "r") as hf:
                    if bl in hf:
                        names = _read_coil_names(hf[bl]) or None
            except OSError:
                names = None
        spec = resolve_device(self.config.device, names)
        if spec is None and self.config.device is None and names is None:
            # no coil names anywhere (e.g. the baseline coil read failed): the
            # device the until-N loop resolved, as the archive recorded it
            spec = self._archived_loop_device()
        return spec

    def _archived_loop_device(self):
        from .devices import DEVICES
        from .utils import read_generation_provenance
        try:
            src = read_generation_provenance(
                self.config.output_header,
                scan_key=self.config.generation.scan_key).get("inspec_cut_source")
        except OSError:
            return None
        if isinstance(src, str) and src.startswith("device:"):
            return DEVICES.get(src.split(":", 1)[1])
        return None

    def _announce_boundary_cut(self, val, src, spec, quiet, origin="config",
                               setting=None):
        """One announcement per resolved cut, every source alike."""
        key = (val, src, origin)
        if not quiet and getattr(self, "_boundary_cut_announced", None) != key:
            self._boundary_cut_announced = key
            where = ("filtering.rms_max_mm" if origin == "config"
                     else "filter(rms_max_mm=...) argument")
            if src == "disabled":
                spelled = "None" if setting is None else repr(setting)
                print(f"[boundary cut] DISABLED: no LCFS boundary cut ({where}="
                      f"{spelled}"
                      + ("; None keeps its historical meaning, 'off' is the "
                         "explicit spelling, 'auto' the device/generic cut"
                         if setting is None else "") + ")")
            elif src == "explicit":
                print(f"[boundary cut] LCFS rms <= {val:g} mm (explicit {where})")
            elif src == "generic":
                print(f"[boundary cut] LCFS rms <= {val:g} mm (generic: no device "
                      "calibration; set config.device or filtering.rms_max_mm)")
            else:
                from .devices import GENERIC_BOUNDARY_RMS_MM
                print(f"[boundary cut] LCFS rms <= {val:g} mm ({src} calibration: "
                      f"{spec.boundary_provenance}; replaces the generic "
                      f"{GENERIC_BOUNDARY_RMS_MM:g} mm; filtering.rms_max_mm "
                      "overrides)")
        return val, src

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
        (captured-geometry) or template-geometry current conversion."""
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
            try:
                self.prepare_baseline()
            except Exception as exc:
                self._record_refusal(exc)         # then re-raise, unchanged
                raise
        self.generate()
        self.filter()
        self.export()
        return self

    def _record_refusal(self, exc, time=None):
        """Write the slice as REFUSED (``write_refused_scan``) after
        ``prepare_baseline`` raised, so a series reader reports it as
        ``status="refused"`` rather than a gap; *time* (the slice time [s])
        is stamped with it.  Returns the recorded reason, or None when
        nothing could be recorded (flat layout, or the scan already holds
        draws -- then the earlier draws stand)."""
        import warnings
        from .utils import write_refused_scan
        sk = self.config.generation.scan_key
        reason = f"prepare_baseline raised {type(exc).__name__}: {exc}"[:500]
        if sk is None:
            return None
        try:
            write_refused_scan(self.config.output_header, sk, reason,
                               time=time)
        except ValueError as e:               # already holds draws
            warnings.warn(f"slice {sk!r} not marked refused: {e}", UserWarning,
                          stacklevel=3)
            return None
        print(f"[refused] scan {sk!r}: {reason}")
        return reason

    def run_slices(self, times, scan_keys=None, header=None, export=False,
                   on_refusal="record") -> dict:
        """Sweep an IMAS time series into ONE archive, one ``scan_key`` per slice.

        Wraps the ``set_slice -> prepare_baseline -> generate -> filter`` loop
        (F5): keeps a single solver, writes every slice's draws under
        ``scan/<key>/`` in ``{header}.h5``, and returns a per-slice summary
        ``{scan_key: {time, n_all, n_sel, l_i, Ip}}``. ``scan_keys`` defaults to
        the time in ms (``round(t*1000)``). Set ``export=True`` to also write the
        selected-only copy after the last slice.

        Reconstruction sources have no time axis (:meth:`set_slice` raises on
        ``time``); build one :class:`Bouquet` per g-file instead.

        A slice whose ``prepare_baseline`` raises (a closure refusal, a
        failed gate, a time-matching refusal, ...) is written to the archive
        as REFUSED (:func:`~bouquet.utils.write_refused_scan`, with the
        exception as the reason and the slice time as ``refused_time``), so
        a later :func:`~bouquet.draw_bands` reports it as
        ``status="refused"`` instead of a gap.  ``on_refusal="record"``
        (the default since 2026-10-06; owner decision -- one refused
        baseline ended a whole series) records it in the summary
        (``refused=<reason>``, no draws) and carries on with the next
        slice; after the last slice the count and the reasons are printed
        and warned once.  ``"raise"`` re-raises at the first refusal, the
        behaviour before.
        """
        if on_refusal not in ("raise", "record"):
            raise ValueError("on_refusal must be 'raise' or 'record', got "
                             f"{on_refusal!r}")
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
            try:
                self.prepare_baseline()
            except Exception as exc:
                reason = self._record_refusal(exc, time=t)
                if on_refusal == "raise":
                    raise
                self.baseline = None
                results[sk] = dict(time=t, n_all=0, n_sel=0, l_i=float("nan"),
                                   Ip=float("nan"),
                                   refused=reason or f"{type(exc).__name__}: {exc}")
                continue
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
        refused = {k: r for k, r in results.items() if "refused" in r}
        if refused:
            import warnings
            lines = [f"  scan {k!r} (t = {r['time']:.9g} s): {r['refused']}"
                     for k, r in refused.items()]
            msg = (f"run_slices: {len(refused)} of {len(results)} slices "
                   "REFUSED (recorded in the archive as refused_reason / "
                   "refused_time; no draws):\n" + "\n".join(lines))
            print(f"[run_slices] {msg}", flush=True)
            warnings.warn(msg, UserWarning, stacklevel=2)
        if export:
            self.export()
        return results
