"""The unified reconstruction engine (``GenerationConfig.reconstruction_engine
= "unified"``, the default since 2026-10-06; ``"legacy"``, opt-in, never
enters this module).

ONE loop reconstructs the baseline for both input types (docs/engine.md):

* a source adapter (:mod:`bouquet.adapters`) turns the input into a contract:
  kinetics, the fixed pressure, PARALLEL current components ``<j.B>``
  (inductive, fixed driven), the boundary, the measurement rows and the
  sign frame;
* every pass composes the solver's variable on the LATEST solved geometry
  ``G_k`` -- the only place a toroidal current is formed::

      J_k = F<1/R> [ s_ind lambda_ind + s_bs lambda_BS,k + lambda_fix ] / <B^2>
            + p' (<R> - F^2 <1/R>/<B^2>)                      (identity I2)

  (the components are stored as ``<j.B>``, so the field-aligned conversion is
  ``<j.B> F<1/R>/<B^2>``; the pressure-driven term is recomputed every pass
  from that pass's own ``p'`` and geometry);
* the structured closure (zero solves) picks the scale-function coefficients
  ``x`` so the composed current meets the rows, each row carrying the
  discrepancy measured on the previous SOLVED equilibrium;
* the solved current is relaxed (``CurrentRelaxer``), ONE Grad-Shafranov
  solve is taken, and everything is measured on the new equilibrium: Redl,
  l_i, Ip and the uniform factor, q0, tan(gamma), request minus achieved;
* the bootstrap iterate, the discrepancies (and, optionally, the MSE
  linearisation -- its offset every pass, its Jacobian Broyden-updated under
  ``engine_mse_jacobian="fd_broyden"``, re-taken by finite differences at
  convergence -- and the delivery correction) are updated.

The iteration is today's kernel, :func:`bouquet.jbs_loop.run_jbs_loop`, with
``step`` = compose + closure + relax + solve + measure; the engine adds its
rows through the kernel's ``extra`` hook and the existing
:class:`~bouquet.jbs_loop.AxisRowPin`.  Convergence uses ONLY existing
constants (see :func:`convergence_table`); failure is exactly the loop's.

A final unrelaxed composition on the last geometry is solved (two passes) and
checked (:func:`bouquet.jbs_loop.check_delivered` plus every row on the
delivered equilibrium).  That solve, its request, its geometry snapshot, the
coefficients ``x*``, ``lambda_BS*`` and the discrepancies ARE the
reconstruction (:class:`EngineState`), and the :class:`~bouquet.baseline.
Baseline` the rest of the package consumes is built from it.

The solver is behind a small backend interface (:class:`TokaMakerBackend`;
the fast tests drive the same engine with a toy Grad-Shafranov stand-in).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import coords
from .edge_pressure import (pressure_frames, resolve_edge_pressure,
                            solver_pax, solver_pp_profile)

#: Version stamp of the engine (recorded with every run).
ENGINE_VERSION = "unified-engine/2 (Stage 3: reconstruction and draws)"

#: ``GenerationConfig.reconstruction_engine`` values.
ENGINE_CHOICES = ("legacy", "unified")
#: ``GenerationConfig.engine_preset`` values (docs/engine.md, "Presets").
ENGINE_PRESETS = ("structured", "structured_uniform", "bootstrap_scalar",
                  "sawtooth_two_scalar", "two_scalar_li")
#: The measurement rows the engine knows.
ENGINE_ROWS = ("Ip", "l_i", "q0", "mse")
#: ``GenerationConfig.engine_mse_jacobian`` values.
ENGINE_MSE_JACOBIANS = ("fd_broyden", "fd_chord")
#: How many times the MSE stage re-takes its chord Jacobian at a converged
#: MSE loop (:meth:`UnifiedEngine._refresh_jacobian`) before it declares the
#: fit NOT converged because the fresh-Jacobian Gauss-Newton step still
#: exceeds the stage's criterion.  A COST ceiling (each refresh is 1 + n_free
#: solves, each continuation a loop under the same pass ceiling), not a
#: tolerance: it can only turn a converged stage into a non-converged one,
#: never the reverse.  Introduced 2026-10-07 with the refresh; 3 is
#: owner-approved 2026-10-07 (non-converged only: it can only mark a stage
#: not converged); recorded in :func:`convergence_table`.
MSE_JACOBIAN_MAX_REFRESHES = 3
#: The engine fields of :class:`~bouquet.config.GenerationConfig` and their
#: defaults (refused when changed with ``reconstruction_engine="legacy"``).
ENGINE_FIELD_DEFAULTS = {
    "engine_preset": "structured",
    "engine_rows": ("Ip", "l_i"),
    "engine_delivery_correction": False,
    "engine_mse_jacobian": "fd_chord",
    # under-relaxation of the l_i row's discrepancy update (1.0: the update
    # before the setting existed; see UnifiedEngine.on_pass)
    "engine_li_row_relaxation": 1.0,
    # the IDS adapter's inductive choice (adapters.IDS_INDUCTIVE_CHOICES;
    # "residual" by definition, owner decision 2026-10-02)
    "engine_ids_inductive": "residual",
    # the IDS l_i row target's normalisation radius
    # (adapters.IMAS_LI3_RADIUS_CHOICES; see GenerationConfig)
    "imas_li3_radius": "auto",
    # the draws on the engine (bouquet.engine_draws, docs/engine.md "Draws")
    "engine_draw_q0_row": False,
    "engine_draw_homotopy": True,
    "engine_draw_bootstrap_refresh": False,
    # the GS iteration cap on every solve inside an engine draw (the owner's
    # value, 2026-09-30: 100; see docs/engine.md "The solve cap")
    "engine_draw_solve_maxits": 100,
    # the delivered MSE fit's chord chi^2 / N above which it is FLAGGED
    # (mse_chi2_per_chord_high; never acceptance).  10.0: owner-approved
    # 2026-10-07, flag only (see GenerationConfig.mse_chi2n_flag)
    "mse_chi2n_flag": 10.0,
}


def mse_scheme_text(scheme) -> str:
    """What the MSE stage does with its Jacobian, in words, for the scheme
    actually in use (``engine_mse_jacobian``)."""
    return ("finite differences once at the no-MSE convergence, then "
            "Broyden updates every pass" if scheme == "fd_broyden" else
            "finite differences once at the no-MSE convergence, held fixed "
            "(chord method; the offset is refreshed from every solve)") + (
        "; re-taken by the same finite differences at each MSE loop's "
        "convergence, the loop continuing with it while its Gauss-Newton "
        "step exceeds the stage's criterion")


#: The engine's SWB edge taper (OFT's ``taper_edge_*`` of
#: ``solve_with_bootstrap``), off by default as in OFT; read from
#: ``GenerationConfig.bootstrap_kwargs`` and overridden there.
ENGINE_EDGE_TAPER_DEFAULT = {"taper_edge_jBS": False, "taper_edge_psi0": 0.999,
                             "taper_edge_shape": 2}
#: The ``bootstrap_kwargs`` keys the engine honours (the edge taper; and
#: ``use_sauter_eps`` at True, accepted as a no-op for configs written for
#: the toolkit's SWB: the engine's Redl epsilon is
#: ``GenerationConfig.eps_definition`` (default ``r_over_R_geo``, owner
#: decision E7); ``False`` -- SWB's ``<a>/<R>`` -- is refused, pointing at
#: ``eps_definition="a_over_R"``, see :func:`validate_engine_settings`).  Any
#: other key configures solve_with_bootstrap, which the engine never runs:
#: refused.
ENGINE_BOOTSTRAP_KWARGS = frozenset(ENGINE_EDGE_TAPER_DEFAULT) | {
    "use_sauter_eps"}


def engine_edge_taper(gc) -> dict:
    """``dict(on, psi0, shape)``: the engine's edge taper from
    ``bootstrap_kwargs`` (:data:`ENGINE_EDGE_TAPER_DEFAULT` where unset)."""
    from .physics import EDGE_TAPER_SHAPES
    bk = dict(ENGINE_EDGE_TAPER_DEFAULT)
    bk.update({k: v for k, v in (getattr(gc, "bootstrap_kwargs", None)
                                 or {}).items() if k in bk})
    on, psi0, shape = (bk["taper_edge_jBS"], bk["taper_edge_psi0"],
                       bk["taper_edge_shape"])
    if not isinstance(on, (bool, np.bool_)):
        raise ValueError(f"bootstrap_kwargs['taper_edge_jBS'] must be a "
                         f"bool, got {on!r}")
    if not (np.isfinite(float(psi0)) and 0.0 < float(psi0) < 1.0):
        raise ValueError(f"bootstrap_kwargs['taper_edge_psi0'] must be in "
                         f"(0, 1), got {psi0!r}")
    if int(shape) not in EDGE_TAPER_SHAPES or int(shape) != shape:
        raise ValueError(f"bootstrap_kwargs['taper_edge_shape'] must be one "
                         f"of {sorted(EDGE_TAPER_SHAPES)}, got {shape!r}")
    return dict(on=bool(on), psi0=float(psi0), shape=int(shape))


#: Rows each preset admits (``Ip`` is mandatory for every preset).
PRESET_ROWS = {
    "structured": frozenset(("Ip", "l_i", "q0", "mse")),
    "structured_uniform": frozenset(("Ip", "l_i", "q0", "mse")),
    "bootstrap_scalar": frozenset(("Ip",)),
    "sawtooth_two_scalar": frozenset(("Ip", "q0")),
    "two_scalar_li": frozenset(("Ip", "l_i")),
}
#: Presets on the shipped four-Gaussian basis (the soft solver serves their
#: soft rows); the others are the constant-basis scalar closures.
STRUCTURED_BASIS_PRESETS = ("structured", "structured_uniform")
#: ``_baseline`` attribute carrying the engine record (JSON; an ADDED
#: attribute, v3 readers are unaffected).
ENGINE_ATTR = "engine_json"

_MU0 = 4.0e-7 * np.pi


class EngineClosureRefused(RuntimeError):
    """The closure refused a pass (scale bounds, a degenerate row, a failed
    Ip round trip): raised with its reason, never flagged away."""


class EngineSolveError(RuntimeError):
    """A Grad-Shafranov solve of the engine failed (or returned a state the
    engine cannot measure)."""


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def gfile_li_row_tol() -> float:
    """The g-file hard l_i row's absolute tolerance (decision 7).

    NOT a new number: it is the default ``li_tol`` of
    :func:`bouquet.TokaMaker_interface._rematch_li_request` -- the same 1e-3
    absolute tolerance the legacy step-5 secant and its re-match use --
    read from that function's signature so the two cannot drift apart."""
    import inspect
    from .TokaMaker_interface import _rematch_li_request
    return float(inspect.signature(_rematch_li_request)
                 .parameters["li_tol"].default)


def _closure_scale_bounds():
    """The closure's ``scale_bounds`` default (0.2 < s < 5), read from
    :func:`bouquet.utils.close_ip_structured` (recorded, never passed)."""
    import inspect
    from .utils import close_ip_structured
    return tuple(inspect.signature(close_ip_structured)
                 .parameters["scale_bounds"].default)


def _is_default(name, v):
    d = ENGINE_FIELD_DEFAULTS[name]
    if name == "engine_rows":
        try:
            return tuple(v) == tuple(d)
        except TypeError:
            return False
    if name in ("engine_li_row_relaxation", "mse_chi2n_flag"):
        # a number equal to 1 (1 or 1.0) is the default; a bool is not
        import numbers
        return (isinstance(v, numbers.Real)
                and not isinstance(v, (bool, np.bool_)) and v == d)
    return (type(v) is type(d)) and v == d


#: The way back to the legacy paths, appended to every refusal of a setting
#: the unified engine cannot honour (the engine is the default since
#: 2026-10-06, so a config written for the legacy paths reaches these
#: refusals unless it says which engine it wants).
LEGACY_ENGINE_HINT = (
    "  To run the LEGACY reconstruction and draws instead (the default "
    "until 2026-10-06), set generation.reconstruction_engine=\"legacy\" "
    "(or build with Bouquet.from_geqdsk / from_imas(..., "
    "reconstruction_engine=\"legacy\")).")


def validate_engine_settings(gc) -> None:
    """Refuse malformed or ineffective engine settings, by name.

    * ``reconstruction_engine`` must be ``"legacy"`` or ``"unified"``;
    * with ``"legacy"`` every ``engine_*`` field must hold its default -- a
      set value would silently do nothing;
    * with ``"unified"``: a known preset, a row list the preset admits
      (always with ``"Ip"``), a bool delivery correction, a known MSE
      Jacobian scheme; the self-consistent loop on (the engine IS the loop),
      a bootstrap to recompute, the ``anchor`` initial guess, no
      single-profile mode; ``"mse"`` needs ``mse_data`` with E_r-corrected
      pitch angles (checked when the rows are read).
    """
    eng = getattr(gc, "reconstruction_engine", "legacy")
    if not isinstance(eng, str) or eng not in ENGINE_CHOICES:
        raise ValueError(f"generation.reconstruction_engine must be one of "
                         f"{ENGINE_CHOICES}, got {eng!r}")
    vals = {k: getattr(gc, k, d) for k, d in ENGINE_FIELD_DEFAULTS.items()}
    if eng == "legacy":
        changed = [k for k, v in vals.items() if not _is_default(k, v)]
        if changed:
            raise ValueError(
                f"generation.{', '.join(changed)} set with "
                "reconstruction_engine='legacy': these configure the unified "
                "engine only and would have no effect; set "
                "reconstruction_engine='unified' or leave them at their "
                "defaults")
        return
    preset = vals["engine_preset"]
    if preset not in ENGINE_PRESETS:
        raise ValueError(f"generation.engine_preset must be one of "
                         f"{ENGINE_PRESETS}, got {preset!r}")
    rows = vals["engine_rows"]
    if isinstance(rows, str) or not isinstance(rows, (list, tuple)):
        raise ValueError("generation.engine_rows must be a list of row "
                         f"names from {ENGINE_ROWS}, got {rows!r}")
    rows = tuple(rows)
    bad = [r for r in rows if r not in ENGINE_ROWS]
    if bad or len(set(rows)) != len(rows):
        raise ValueError(f"generation.engine_rows: unknown or repeated row(s) "
                         f"{bad or list(rows)} (known: {ENGINE_ROWS})")
    if "Ip" not in rows:
        raise ValueError("generation.engine_rows must include 'Ip'")
    extra = set(rows) - PRESET_ROWS[preset]
    if extra:
        raise ValueError(f"generation.engine_preset={preset!r} admits rows "
                         f"{sorted(PRESET_ROWS[preset])}; got also "
                         f"{sorted(extra)}")
    if preset == "sawtooth_two_scalar" and "q0" not in rows:
        raise ValueError("generation.engine_preset='sawtooth_two_scalar' "
                         "needs the 'q0' row (two scalars, two rows)")
    if preset == "two_scalar_li" and "l_i" not in rows:
        raise ValueError("generation.engine_preset='two_scalar_li' needs "
                         "the 'l_i' row (two scalars, two rows)")
    mx = vals["engine_draw_solve_maxits"]
    import numbers
    if mx is not None and (isinstance(mx, (bool, np.bool_))
                           or not isinstance(mx, numbers.Integral)
                           or int(mx) < 1):
        raise ValueError(f"generation.engine_draw_solve_maxits={mx!r} must "
                         "be an integer >= 1, or None for the solver's own "
                         "cap")
    # draw_solve_maxits and the draw-solve rescue (the LEGACY draws' cap):
    # refused with the standard rule, ENGINE_UNREAD_LEGACY_FIELDS
    dc = vals["engine_delivery_correction"]
    if not isinstance(dc, (bool, np.bool_)):
        raise ValueError(f"generation.engine_delivery_correction must be a "
                         f"bool, got {dc!r}")
    for _b in ("engine_draw_q0_row", "engine_draw_homotopy",
               "engine_draw_bootstrap_refresh"):
        if not isinstance(vals[_b], (bool, np.bool_)):
            raise ValueError(f"generation.{_b} must be a bool, got "
                             f"{vals[_b]!r}")
    if vals["engine_draw_q0_row"] and "q0" not in rows:
        raise ValueError("generation.engine_draw_q0_row=True keeps the "
                         "reconstruction's q0 row in the draws, but "
                         "engine_rows has no 'q0' (there is no row, target "
                         "or radius to keep)")
    rr = vals["engine_li_row_relaxation"]
    if (isinstance(rr, (bool, np.bool_)) or not isinstance(rr, numbers.Real)
            or not np.isfinite(float(rr)) or not 0.0 < float(rr) <= 1.0):
        raise ValueError(f"generation.engine_li_row_relaxation={rr!r} must "
                         "be a number with 0 < r <= 1 (an under-relaxation "
                         "of the l_i row's update; 1.0 is the default)")
    from .adapters import IDS_INDUCTIVE_CHOICES
    ii = vals["engine_ids_inductive"]
    if not isinstance(ii, str) or ii not in IDS_INDUCTIVE_CHOICES:
        raise ValueError(f"generation.engine_ids_inductive must be one of "
                         f"{IDS_INDUCTIVE_CHOICES}, got {ii!r}")
    from .adapters import IMAS_LI3_RADIUS_CHOICES
    lr = vals["imas_li3_radius"]
    if not isinstance(lr, str) or lr not in IMAS_LI3_RADIUS_CHOICES:
        raise ValueError(f"generation.imas_li3_radius must be one of "
                         f"{IMAS_LI3_RADIUS_CHOICES}, got {lr!r}")
    cf = vals["mse_chi2n_flag"]
    if (isinstance(cf, (bool, np.bool_)) or not isinstance(cf, numbers.Real)
            or not np.isfinite(float(cf)) or not float(cf) > 0.0):
        raise ValueError(f"generation.mse_chi2n_flag={cf!r} must be a "
                         "finite number > 0 (the chi^2 / N above which a "
                         "delivered MSE fit is FLAGGED; a flag, never "
                         "acceptance)")
    mj = vals["engine_mse_jacobian"]
    if mj not in ENGINE_MSE_JACOBIANS:
        raise ValueError(f"generation.engine_mse_jacobian must be one of "
                         f"{ENGINE_MSE_JACOBIANS}, got {mj!r}")
    if "mse" in rows and getattr(gc, "mse_data", None) is None:
        raise ValueError("generation.engine_rows has 'mse' but "
                         "generation.mse_data is None")
    _mse_knobs_unread(gc, rows)
    _legacy_knobs_unread(gc, vals)
    bk = dict(getattr(gc, "bootstrap_kwargs", None) or {})
    swb_only = sorted(set(bk) - ENGINE_BOOTSTRAP_KWARGS)
    if swb_only:
        raise ValueError(
            f"generation.bootstrap_kwargs keys {swb_only} configure "
            "solve_with_bootstrap, which reconstruction_engine='unified' "
            "never runs (they would be silently ignored); the engine reads "
            f"only {sorted(ENGINE_BOOTSTRAP_KWARGS)}")
    if not bool(bk.get("use_sauter_eps", True)):
        raise ValueError(
            "generation.bootstrap_kwargs['use_sauter_eps']=False: the "
            "engine's Redl epsilon is set by generation.eps_definition "
            "(default 'r_over_R_geo', owner decision E7), not by this "
            "toolkit key; for the <a>/<R> of use_sauter_eps=False set "
            "generation.eps_definition='a_over_R'")
    if engine_edge_taper(gc)["on"] and not resolve_edge_pressure(
            gc).edge_pprime_pin:
        raise ValueError(
            "the engine's edge taper (bootstrap_kwargs['taper_edge_jBS']=True) "
            "refuses edge_pprime_pin=False: the taper takes the "
            "request to zero at the LCFS while the unpinned P' stays finite "
            "there, and the solve does not converge.  Either turn the taper "
            "off (bootstrap_kwargs={'taper_edge_jBS': False}) or zero the "
            "edge pressure gradient (edge_pprime_pin=True)")
    for name, want in (("jbs_self_consistent", True),
                       ("recalculate_j_BS", True),
                       ("single_profile_jphi", False)):
        if bool(getattr(gc, name, want)) != want:
            raise ValueError(
                f"reconstruction_engine='unified' needs generation.{name}="
                f"{want}: the engine is the self-consistent bootstrap loop."
                + LEGACY_ENGINE_HINT)
    if str(getattr(gc, "jbs_init", "anchor")) != "anchor":
        raise ValueError("reconstruction_engine='unified' starts from the "
                         "anchor (generation.jbs_init='anchor'); the legacy "
                         "'swb' initial guess is not available."
                         + LEGACY_ENGINE_HINT)


#: MSE knobs the unified engine reads, with the defaults that mean "unset"
#: (``structured_mse_steps`` is NOT read: the engine iterates the chords to
#: convergence, so a set value would be silently ignored).
_ENGINE_MSE_KNOBS = ("mse_data", "structured_mse_required",
                     "structured_mse_fd_step", "structured_mse_steps",
                     "structured_mse_sigma_sys", "structured_mse_min_chords")


def _mse_knobs_unread(gc, rows):
    """Refuse MSE settings the unified engine will never read -- the rule
    ``Bouquet._check_structured_mse_reachable`` applies on the legacy path
    (whose check the engine's dispatch never reaches), applied to the
    engine: without the ``"mse"`` row nothing MSE is read, and with it
    ``structured_mse_steps`` still is not.  ``structured_mse_required=True``
    that cannot be honoured ALWAYS raises; otherwise ``workflow='custom'``
    (or ``allow_unsafe_workflow``) downgrades the refusal to a printed WARN,
    exactly as the legacy guard."""
    from dataclasses import MISSING
    from .config import GenerationConfig
    f = GenerationConfig.__dataclass_fields__
    set_ = []
    for name in _ENGINE_MSE_KNOBS:
        if name not in f or not hasattr(gc, name):
            continue
        d = f[name].default
        if d is MISSING:
            continue
        v = getattr(gc, name)
        if (v is not None) if d is None else (v != d):
            set_.append(name)
    unread = (set_ if "mse" not in rows else
              [n for n in set_ if n == "structured_mse_steps"])
    if not unread:
        return
    why = ("engine_rows has no 'mse'" if "mse" not in rows else
           "the engine iterates the chords to convergence and never reads "
           "structured_mse_steps")
    if bool(getattr(gc, "structured_mse_required", False)) \
            and "mse" not in rows:
        raise ValueError(
            "structured_mse_required=True, but the unified engine will not "
            f"apply the MSE term: {why}.  Refusing rather than ignoring a "
            "required constraint." + LEGACY_ENGINE_HINT)
    msg = (", ".join(unread) + " set, but the unified engine never reads "
           f"it: {why} -- it would otherwise be silently ignored."
           + LEGACY_ENGINE_HINT)
    if (str(getattr(gc, "workflow", "")) == "custom"
            or bool(getattr(gc, "allow_unsafe_workflow", False))):
        print("WARN: " + msg + " (workflow='custom': continuing)", flush=True)
        return
    raise ValueError(msg)


#: Legacy-path settings the unified engine NEVER reads, each with what
#: replaces it under the engine (or why nothing does).  A value other than
#: the field's default is REFUSED under ``reconstruction_engine="unified"``
#: (:func:`_legacy_knobs_unread`) -- it would otherwise be silently ignored.
ENGINE_UNREAD_LEGACY_FIELDS = {
    "closure_channel": "the engine's closure is engine_preset / engine_rows",
    "jBS_baseline_mode": "the engine composes the inductive and the Redl "
                         "bootstrap itself (engine_preset / engine_rows)",
    "structured_preset": "engine_preset",
    "structured_basis": "engine_preset (its basis)",
    "structured_weights": "engine_preset (its weights)",
    "structured_sigma_ind_up": "engine_preset (its inductive prior)",
    "structured_li_target": "nothing: the engine's l_i row targets the "
                            "SOURCE's own l_i (engine_rows 'l_i')",
    "structured_li_sigma": "nothing: the IDS soft l_i row uses the preset's "
                           "sigma (bouquet.adapters)",
    "structured_li_kind": "nothing: the engine's l_i row is li_3 always",
    "structured_ip_sigma": "nothing: the IDS soft Ip row uses the preset's "
                           "sigma (bouquet.adapters)",
    "structured_ip_sigma_frac": "nothing: the IDS soft Ip row uses the "
                                "preset's sigma (bouquet.adapters)",
    "structured_soft": "nothing: the rows are hard for a g-file and soft for "
                       "an IDS source, by source",
    "structured_li_max_corrector_steps": "engine_li_row_relaxation (the "
                                         "engine iterates the l_i row inside "
                                         "the loop)",
    "anchor_pressure_to_equilibrium": "nothing: the engine's pressure is "
                                      "kinetic + impurity + fast, with no "
                                      "p_diff",
    "imas_corrective_jphi": "engine_delivery_correction (the engine's "
                            "remedy for the jphi-linterp delivery defect)",
    "jbs_loop_q0_corrector": "engine_rows with 'q0' (engine_draw_q0_row "
                             "keeps it in the draws)",
    "floor_j_BS": "nothing: the engine never floors the bootstrap",
    "accept_anchor_inband": "nothing: the engine draws have no legacy "
                            "anchor in-band shortcut",
    "diagnostic_plots": "nothing: the engine draws make no per-draw SWB "
                        "diagnostic plots",
    # owner-approved 2026-10-05: refused like the rest.  Both are
    # ENGINE_DEPENDENT_DEFAULTS (default None, resolved per engine at
    # prepare_baseline()): None or the engine's own value is accepted
    "isolate_edge_jBS": "nothing: the engine never isolates the edge "
                        "bootstrap (its bootstrap is Redl on the whole "
                        "profile); leave it unset (None: resolved per "
                        "engine at prepare_baseline(), False under "
                        "reconstruction_engine='legacy')",
    "perturb_jind_in_anchor": "nothing: one engine draw route for both "
                              "input types replaces Fix C and the standard "
                              "l_i loop; leave it unset (None: resolved per "
                              "engine at prepare_baseline(), True for an "
                              "IDS source under "
                              "reconstruction_engine='legacy')",
    # #75 review (owner decision D5): the legacy draws' cap and its rescue
    "draw_solve_maxits": "engine_draw_solve_maxits (default "
                         f"{ENGINE_FIELD_DEFAULTS['engine_draw_solve_maxits']}"
                         "; the engine draws' cap); leave it unset ('auto': "
                         "resolved per engine at prepare_baseline(), "
                         "100 under reconstruction_engine='legacy')",
    "draw_solve_retry_urf": "nothing: the engine draws are capped by "
                            "engine_draw_solve_maxits and never rescued",
    "draw_solve_loose_tol": "nothing: the engine draws are capped by "
                            "engine_draw_solve_maxits and never rescued",
    # review PR60 B8: the opt-in for solve_with_bootstrap's convergence keys
    # (djBS_tol, saw_relax), which the engine refuses with every other SWB
    # key -- the flag itself unlocks nothing under the engine
    "bootstrap_convergence_override": "nothing: the engine never runs "
                                      "solve_with_bootstrap (its bootstrap "
                                      "is physics.evaluate_jBS), and the "
                                      "convergence keys the flag admits are "
                                      "refused under the engine",
}


#: Legacy-path fields whose VALIDATED value depends on the reconstruction
#: engine (and, on the legacy engine, on the input type): field ->
#: {engine: value, or {source kind: value}}.  ``GenerationConfig`` holds
#: ``None`` for them by default ("resolve per engine"); the factories set
#: none of them, and :func:`resolve_engine_defaults` fills them ONCE, at
#: ``Bouquet.prepare_baseline()`` -- so the engine named when the run starts
#: decides, whatever it was when the configuration was built (until
#: 2026-10-07 the factories applied the values at construction, and a
#: configuration switched to ``"legacy"`` afterwards ran the engine's).
#: The legacy values are the ones the legacy workflow was validated with
#: (``from_geqdsk``: the full-profile decomposition and the standard l_i
#: loop; ``from_imas``: diff+C); the unified values are the fields' former
#: dataclass defaults, which the engine never reads.
ENGINE_DEPENDENT_DEFAULTS = {
    "isolate_edge_jBS": {"unified": True,
                         "legacy": {"reconstruction": False, "imas": False}},
    "perturb_jind_in_anchor": {"unified": False,
                               "legacy": {"reconstruction": False,
                                          "imas": True}},
    # #75 review: the legacy draws' GS cap (TokaMaker_interface.
    # DRAW_SOLVE_MAXITS, draws only); the engine never reads it
    "draw_solve_maxits": {"unified": None, "legacy": 100},
}

#: The UNSET marker of an :data:`ENGINE_DEPENDENT_DEFAULTS` field whose
#: ``None`` is a meaningful value (default: ``None`` is the marker).
#: ``draw_solve_maxits=None`` has meant "the solver's own setup cap" since
#: the field existed (stored configs carry it so), so "resolve per engine"
#: is ``"auto"``.
ENGINE_DEPENDENT_UNSET = {"draw_solve_maxits": "auto"}


def engine_dependent_unset(name, value) -> bool:
    """Whether *value* is the unset marker of engine-dependent field
    *name* (:data:`ENGINE_DEPENDENT_UNSET`; ``None`` otherwise)."""
    marker = ENGINE_DEPENDENT_UNSET.get(name)
    if marker is None:
        return value is None
    return isinstance(value, str) and value == marker


def _source_kind(source) -> str:
    """``"imas"`` for an :class:`~bouquet.config.ImasSource`, else
    ``"reconstruction"``."""
    from .config import ImasSource
    return "imas" if isinstance(source, ImasSource) else "reconstruction"


def engine_validated_value(name, engine, source_kind):
    """The value of :data:`ENGINE_DEPENDENT_DEFAULTS` field *name* the
    *engine* (``"legacy"`` / ``"unified"``) was validated with, for an input
    of *source_kind* (``"reconstruction"`` / ``"imas"``)."""
    v = ENGINE_DEPENDENT_DEFAULTS[name][engine]
    return v[source_kind] if isinstance(v, dict) else v


def resolve_engine_defaults(config, *, stacklevel=2) -> dict:
    """Fill every :data:`ENGINE_DEPENDENT_DEFAULTS` field of
    ``config.generation`` that is unset (``None``) with the value the
    configured engine was validated with, IN PLACE, and return the record
    ``{field: {"value": v, "origin": ...}}``.

    ``origin`` is ``"resolved from engine=<engine>"`` for a filled field and
    ``"explicit"`` for one the caller set.  An explicit value that
    contradicts the engine's validated value is KEPT -- never overridden --
    with a warning naming the field (the record then also carries
    ``"engine_validated"``); under ``"unified"`` such a value is refused
    besides (:func:`validate_engine_settings`, since the engine never reads
    it).  A field this function filled earlier for another engine or input
    type (the configuration was switched after a previous
    ``prepare_baseline()``) is re-resolved, not taken for an explicit
    value.  Idempotent otherwise.  Called by ``Bouquet.prepare_baseline()``
    (where the record goes on the baseline and into the archive), and
    defensively by ``generate()`` / ``verify_sigma0_consistency()``."""
    import warnings
    gc = config.generation
    eng = str(getattr(gc, "reconstruction_engine", "legacy"))
    if eng not in ENGINE_CHOICES:
        validate_engine_settings(gc)          # refuses, by name
    kind = _source_kind(config.source)
    mine = dict(getattr(gc, "_engine_resolved", None) or {})
    rec = {}
    for name in ENGINE_DEPENDENT_DEFAULTS:
        want = engine_validated_value(name, eng, kind)
        unset = ENGINE_DEPENDENT_UNSET.get(name)
        v = getattr(gc, name, unset)
        prev = mine.get(name)
        if prev is not None and not engine_dependent_unset(name, v) \
                and _same_value(v, prev[2]) \
                and (prev[0], prev[1]) != (eng, kind):
            v = unset                         # ours, for another engine
        if engine_dependent_unset(name, v):
            setattr(gc, name, want)
            mine[name] = (eng, kind, want)
            rec[name] = {"value": want,
                         "origin": f"resolved from engine={eng}"}
            continue
        if prev is not None and _same_value(v, prev[2]) \
                and (prev[0], prev[1]) == (eng, kind):
            rec[name] = {"value": v, "origin": f"resolved from engine={eng}"}
            continue
        mine.pop(name, None)
        rec[name] = {"value": v, "origin": "explicit"}
        if not _same_value(v, want):
            rec[name]["engine_validated"] = want
            warnings.warn(
                f"generation.{name}={v!r} was set explicitly, but "
                f"reconstruction_engine={eng!r} was validated with "
                f"{name}={want!r} (for a {kind} source): the explicit value "
                "is KEPT and this run uses a configuration that engine was "
                f"not validated with.  Leave {name} unset (None) to have it "
                "resolved per engine.", UserWarning, stacklevel=stacklevel)
            print(f"WARN: generation.{name}={v!r} (explicit) contradicts "
                  f"reconstruction_engine={eng!r}'s validated "
                  f"{name}={want!r}; kept", flush=True)
    gc._engine_resolved = mine
    return rec


def _legacy_knobs_unread(gc, vals):
    """Refuse a legacy-path setting the unified engine never reads
    (:data:`ENGINE_UNREAD_LEGACY_FIELDS`) when it holds anything but its
    default -- the rule :func:`_mse_knobs_unread` applies to the MSE knobs,
    with the same ``workflow='custom'`` / ``allow_unsafe_workflow``
    downgrade to a printed WARN.  Also ``homotopy_passes`` changed while
    ``engine_draw_homotopy=False`` (no homotopy runs)."""
    from dataclasses import MISSING
    from .config import GenerationConfig
    f = GenerationConfig.__dataclass_fields__
    bad = []
    for name, instead in ENGINE_UNREAD_LEGACY_FIELDS.items():
        if name not in f or not hasattr(gc, name):
            continue
        fl = f[name]
        d = (fl.default if fl.default is not MISSING else
             fl.default_factory() if fl.default_factory is not MISSING
             else MISSING)
        if d is MISSING:
            continue
        v = getattr(gc, name)
        if name in ENGINE_DEPENDENT_DEFAULTS:
            # unset, or the value the engine resolves it to
            d = engine_validated_value(name, "unified", "reconstruction")
            same = (v is None or engine_dependent_unset(name, v)
                    or _same_value(v, d))
        else:
            same = (v is None) if d is None else _same_value(v, d)
        if not same:
            bad.append(f"{name}={v!r} (default {d!r}; under the engine: "
                       f"{instead})")
    if not vals.get("engine_draw_homotopy", True) and "homotopy_passes" in f:
        d = f["homotopy_passes"].default_factory()
        v = getattr(gc, "homotopy_passes", d)
        if not _same_value(v, d):
            bad.append(f"homotopy_passes={v!r} with engine_draw_homotopy="
                       "False (no homotopy runs in an engine draw)")
    if not bad:
        return
    msg = ("set with reconstruction_engine='unified' (the default since "
           "2026-10-06), but the unified engine never reads them -- they "
           "would otherwise be silently ignored: " + "; ".join(bad) + "."
           + LEGACY_ENGINE_HINT)
    if (str(getattr(gc, "workflow", "")) == "custom"
            or bool(getattr(gc, "allow_unsafe_workflow", False))):
        print("WARN: " + msg + " (workflow='custom': continuing)", flush=True)
        return
    raise ValueError(msg)


def _same_value(v, d):
    """``v == d`` for the scalar / sequence values a config holds (a
    sequence compares element-wise as numbers; a bool never equals a
    number)."""
    if isinstance(v, (bool, np.bool_)) != isinstance(d, (bool, np.bool_)):
        return False
    try:
        if isinstance(d, (list, tuple)) or isinstance(v, (list, tuple,
                                                          np.ndarray)):
            return bool(np.array_equal(np.asarray(v, dtype=float),
                                       np.asarray(d, dtype=float)))
        return bool(v == d)
    except (TypeError, ValueError):
        return False


def engine_draw_maxits(gc):
    """The GS iteration cap of every solve inside an engine draw
    (``GenerationConfig.engine_draw_solve_maxits``; ``None``: the solver's
    own cap)."""
    v = getattr(gc, "engine_draw_solve_maxits",
                ENGINE_FIELD_DEFAULTS["engine_draw_solve_maxits"])
    return None if v is None else int(v)


def _full_frame(meas) -> dict:
    """The REPORTED pressure-integral numbers of a final measurement: its
    full-frame block (:func:`bouquet.edge_pressure.pressure_frames`; exactly
    the solver's stats when no separatrix pressure is added back), or the
    stats themselves for a backend that reports no frames."""
    pf = meas.get("pressure_frames")
    if pf is not None:
        return pf["full"]
    return meas.get("stats") or {}


def engine_settings(gc) -> dict:
    """The validated engine settings of a :class:`GenerationConfig`.

    ``loop`` is :func:`bouquet.jbs_loop.jbs_settings` with the closure-half
    current gate ON (decision 9: a standing criterion -- it adds one, it
    loosens none); the other tolerances are read from where they live."""
    from .jbs_loop import MSE_CHORD_OFFSET_TOL_SIGMA, jbs_settings
    validate_engine_settings(gc)
    loop = dict(jbs_settings(gc))
    loop["gate_current_residual"] = True
    return dict(
        engine=str(gc.reconstruction_engine),
        preset=str(gc.engine_preset),
        rows=tuple(gc.engine_rows),
        delivery_correction=bool(gc.engine_delivery_correction),
        mse_jacobian=str(gc.engine_mse_jacobian),
        li_row_relaxation=float(getattr(
            gc, "engine_li_row_relaxation",
            ENGINE_FIELD_DEFAULTS["engine_li_row_relaxation"])),
        ids_inductive=str(getattr(
            gc, "engine_ids_inductive",
            ENGINE_FIELD_DEFAULTS["engine_ids_inductive"])),
        li3_radius=str(getattr(gc, "imas_li3_radius",
                               ENGINE_FIELD_DEFAULTS["imas_li3_radius"])),
        draw_q0_row=bool(getattr(gc, "engine_draw_q0_row", False)),
        draw_homotopy=bool(getattr(gc, "engine_draw_homotopy", True)),
        draw_bootstrap_refresh=bool(getattr(
            gc, "engine_draw_bootstrap_refresh", False)),
        draw_solve_maxits=engine_draw_maxits(gc),
        edge_pressure=resolve_edge_pressure(gc).record(),
        edge_taper=engine_edge_taper(gc),
        # the Redl eps and the R of nu* (GenerationConfig.eps_definition)
        eps_definition=_eps_definition_of(gc),
        loop=loop,
        q0_tol=float(gc.q0_tol),
        structured_li_tol=float(gc.structured_li_tol),
        mse_fd_step=float(gc.structured_mse_fd_step),
        mse_tol_sigma=float(MSE_CHORD_OFFSET_TOL_SIGMA),
        mse_chi2n_flag=float(getattr(gc, "mse_chi2n_flag",
                                     ENGINE_FIELD_DEFAULTS["mse_chi2n_flag"])),
    )


def _eps_definition_of(gc) -> str:
    """``GenerationConfig.eps_definition`` (the default for an object
    without the field), validated."""
    from .physics import EPS_DEFINITION_DEFAULT, check_eps_definition
    return check_eps_definition(getattr(gc, "eps_definition",
                                        EPS_DEFINITION_DEFAULT))


def eps_record(eps_definition) -> dict:
    """The record of the Redl epsilon a run evaluates with: name, formula,
    the R in ``nu*`` and the ``evaluate_jBS`` version tag."""
    from .physics import (EPS_DEFINITION_DEFAULT, EPS_DEFINITIONS, NU_STAR_R,
                          check_eps_definition, evaluate_jbs_version)
    d = check_eps_definition(EPS_DEFINITION_DEFAULT if eps_definition is None
                             else eps_definition)
    return dict(eps_definition=d, eps_formula=EPS_DEFINITIONS[d],
                nu_star_R=NU_STAR_R[d],
                evaluate_jBS_version=evaluate_jbs_version(d))


def convergence_table(settings: dict, contract=None) -> list:
    """Every convergence constant the engine uses, with where it lives."""
    from .utils import IP_ROUNDTRIP_TOL_PCT
    lp = settings["loop"]
    rows = [
        ("r_j", lp["rtol_j"], "GenerationConfig.jbs_rtol_j"),
        ("r_I", lp["rtol_Ip"], "GenerationConfig.jbs_rtol_Ip"),
        ("|dl_i|", lp["tol_li"], "GenerationConfig.jbs_tol_li"),
        ("|dq0| (q0 row active)", lp["tol_q0"], "GenerationConfig.jbs_tol_q0"),
        ("|q0 - q0_target| (q0 row active)", settings["q0_tol"],
         "GenerationConfig.q0_tol (via jbs_loop.AxisRowPin)"),
        ("l_i, g-file hard row", gfile_li_row_tol(),
         "TokaMaker_interface._rematch_li_request li_tol default"),
        ("l_i, IDS hard row / soft-row discrepancy",
         settings["structured_li_tol"], "GenerationConfig.structured_li_tol"),
        ("MSE tan(gamma) change [sigma_eff]", settings["mse_tol_sigma"],
         "jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA"),
        ("MSE fresh-Jacobian step, tan(gamma) move [sigma_eff]",
         settings["mse_tol_sigma"],
         "jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA (the same criterion)"),
        ("MSE Jacobian refreshes (cost ceiling)", MSE_JACOBIAN_MAX_REFRESHES,
         "engine.MSE_JACOBIAN_MAX_REFRESHES"),
        ("closure-half current residual", lp["rtol_j"],
         "GenerationConfig.jbs_rtol_j (jbs_loop current gate, standing)"),
        ("consecutive passes", lp["required_consecutive"],
         "jbs_loop.JBS_REQUIRED_CONSECUTIVE"),
        ("pass ceiling", lp["max_passes"], "GenerationConfig.jbs_max_passes"),
        ("omega floor", lp["relax_floor"], "jbs_loop.JBS_RELAX_FLOOR"),
        ("growth abort passes", lp["growth_abort_passes"],
         "jbs_loop.JBS_GROWTH_ABORT_PASSES"),
        ("closure scale bounds", list(_closure_scale_bounds()),
         "utils.close_ip_structured scale_bounds default"),
        ("Ip round trip [%]", IP_ROUNDTRIP_TOL_PCT,
         "utils.IP_ROUNDTRIP_TOL_PCT"),
    ]
    return [dict(criterion=a, value=b, origin=c) for a, b, c in rows]


# ---------------------------------------------------------------------------
#  composition (identity I2)
# ---------------------------------------------------------------------------
def conversion_factor(geom) -> np.ndarray:
    """``F<1/R>/<B^2>``: the field-aligned ``<j.B>`` -> ``<j_phi>`` factor
    (conversion (c) of the verification report) on an engine geometry
    (keys ``F``, ``inv_R``, ``B2``) -- the package's ONE conversion,
    :func:`bouquet.physics.field_aligned_conversion`."""
    from .physics import field_aligned_conversion
    return field_aligned_conversion(geom["F"], geom["inv_R"], geom["B2"])


def composed_factor(geom) -> np.ndarray:
    """:func:`conversion_factor` times the edge taper the geometry carries:
    the factor :func:`compose` turns a parallel component into its share of
    the request with."""
    kap = conversion_factor(geom)
    w = geom.get("edge_taper")
    return kap if w is None else kap * np.asarray(w, dtype=float)


def pressure_term(geom) -> np.ndarray:
    """``p'(<R> - F^2<1/R>/<B^2>)``: the pressure-driven (diamagnetic +
    Pfirsch-Schlueter) part of ``<j_phi>``, whose ``<j.B>`` is zero."""
    F = np.asarray(geom["F"], dtype=float)
    return np.asarray(geom["pprime"], dtype=float) * (
        np.asarray(geom["R_avg"], dtype=float)
        - F ** 2 * np.asarray(geom["inv_R"], dtype=float)
        / np.asarray(geom["B2"], dtype=float))


def compose(geom, jB_ind, jB_bs, jB_fix, s_ind=1.0, s_bs=1.0):
    """``(J, parts)``: the ``jphi-linterp`` current of the components on
    *geom* (identity I2).  ``parts``: ``ind``, ``bs``, ``fix`` (toroidal,
    scaled), ``pressure`` and the factor ``kappa``."""
    kap = conversion_factor(geom)
    P = pressure_term(geom)
    ind = np.asarray(s_ind) * kap * np.asarray(jB_ind, dtype=float)
    bs = np.asarray(s_bs) * kap * np.asarray(jB_bs, dtype=float)
    fix = kap * np.asarray(jB_fix, dtype=float)
    # the SWB edge taper (geom["edge_taper"], set by the backend that
    # measured the geometry): every component goes to zero at the LCFS
    w = geom.get("edge_taper")
    if w is not None:
        w = np.asarray(w, dtype=float)
        ind, bs, fix, P = ind * w, bs * w, fix * w, P * w
    return ind + bs + fix + P, dict(ind=ind, bs=bs, fix=fix, pressure=P,
                                    kappa=kap)


def li_of_current(j, geom, li_kind):
    """The closure's l_i model of an arbitrary ``jphi-linterp`` current on
    *geom* (its Ip weights, affine P' term and l_i geometry)."""
    from scipy.integrate import cumulative_trapezoid
    from .utils import li_value
    lg = geom["li_geom"]
    psi = np.asarray(lg["psi_N"], dtype=float)
    I = cumulative_trapezoid(np.asarray(geom["w_lin"], float)
                             * np.asarray(j, float), psi, initial=0.0) \
        + np.asarray(lg["affine_cum"], dtype=float)
    return float(li_value(psi, I, lg, li_kind))


def complete_geometry(geom):
    """Add the Ip weights ``w_lin``/``c_affine`` (:func:`bouquet.utils.
    Ip_fsa_weights`, ``jphi-linterp``) to a measured geometry dict."""
    from .utils import Ip_fsa_weights
    g = dict(geom)
    w, c = Ip_fsa_weights(g, convention="jphi-linterp")
    g["w_lin"], g["c_affine"] = np.asarray(w, dtype=float), float(c)
    return g


def _lin(geom, j):
    from scipy.integrate import trapezoid
    return float(trapezoid(np.asarray(geom["w_lin"], float)
                           * np.asarray(j, float),
                           np.asarray(geom["psi_N"], float)))


def _delivery_stats(req, A, geom):
    """``c`` (achieved / request, the uniform Ip factor, in the linear Ip
    measure) and request - achieved/c in % of the peak, core (psi_N < 0.8)
    and edge (psi_N >= 0.8)."""
    psi = np.asarray(geom["psi_N"], dtype=float)
    li_r = _lin(geom, req)
    c = _lin(geom, A) / li_r if li_r != 0.0 else float("nan")
    d = np.asarray(req, float) - np.asarray(A, float) / c
    pk = float(np.max(np.abs(np.asarray(A, float) / c))) or 1.0
    core, edge = psi < 0.8, psi >= 0.8

    def _st(m):
        v = 100.0 * d[m] / pk
        return (float(np.max(np.abs(v))) if v.size else None,
                float(np.sqrt(np.mean(v ** 2))) if v.size else None)
    (cm, cr), (em, er) = _st(core), _st(edge)
    return dict(c=float(c), core_max_pct=cm, core_rms_pct=cr,
                edge_max_pct=em, edge_rms_pct=er,
                edge_argmax_psiN=(float(psi[edge][int(np.argmax(np.abs(
                    d[edge])))]) if np.any(edge) else None)), d


# ---------------------------------------------------------------------------
#  the state (what a draw inherits)
# ---------------------------------------------------------------------------
@dataclass
class EngineState:
    """The reconstruction's state -- designed so a draw can hold it.

    A draw's first pass composes on :attr:`geom` with the unperturbed
    components, :attr:`x` and :attr:`lambda_bs`, so its request is
    bit-identical to :attr:`request` (docs/engine.md, "Draws")."""

    x: Optional[np.ndarray] = None            # closure coefficients (2K)
    lambda_bs: Optional[np.ndarray] = None    # Redl <j.B> iterate in use
    li_discrepancy: float = 0.0               # l_i row d_r
    q0_row: Optional[float] = None            # axis-current row (AxisRowPin)
    q0_target: Optional[float] = None
    mse_J: Optional[np.ndarray] = None        # d tan(gamma) / d x
    mse_x0: Optional[np.ndarray] = None
    mse_tg0: Optional[np.ndarray] = None
    mse_sign: Optional[tuple] = None
    delivery_correction: Optional[np.ndarray] = None   # toroidal Delta
    geom: Optional[dict] = None               # geometry the next pass uses
    request: Optional[np.ndarray] = None      # the last solved request
    extras: dict = field(default_factory=dict)

    def record(self) -> dict:
        from .jbs_loop import jsonable
        g = self.geom or {}
        keep = ("psi_N", "psi_q", "F", "R_avg", "inv_R", "inv_R2", "B2",
                "pprime", "dV_dpsi", "dpsi_dpsiN", "w_lin", "c_affine",
                "edge_taper")
        return jsonable(dict(
            x=self.x, lambda_bs=self.lambda_bs,
            li_discrepancy=self.li_discrepancy, q0_row=self.q0_row,
            q0_target=self.q0_target, mse_J=self.mse_J, mse_x0=self.mse_x0,
            mse_tg0=self.mse_tg0,
            mse_sign=(None if self.mse_sign is None else list(self.mse_sign)),
            delivery_correction=self.delivery_correction,
            geometry_snapshot={k: g.get(k) for k in keep if k in g},
            li_geom_snapshot=({k: v for k, v in (g.get("li_geom") or {}).items()
                               if k in ("dpsi_dpsiN", "vol", "perimeter",
                                        "R_axis", "affine_cum", "psi_pad")}
                              or None),
            request=self.request, **self.extras))


# ---------------------------------------------------------------------------
#  the rows the kernel does not know (its ``extra`` hook)
# ---------------------------------------------------------------------------
class EngineRows:
    """The engine's added criteria: the l_i row and the MSE chords.

    l_i: the delivered l_i against what the closure predicted it would be,
    ``e = l_i(E_k+1) - (l_i_model(x_k; G_k) + d_k-1)``.  On a HARD row the
    closure imposes ``l_i_model + d = target``, so ``e`` IS the delivered
    residual ``l_i - target``; on a SOFT row ``e`` is the gap between the
    residual the fit weighed and the delivered one (the LiRowPin soft
    semantics).  ``|e| <= tol``.
    MSE (after the Jacobian): ``max_i |tg_i(E_k+1) - tg_i(E_k)| / sigma_eff_i
    <= MSE_CHORD_OFFSET_TOL_SIGMA``.
    """

    def __init__(self, engine, *, li_tol=None, mse_tol=None):
        self.eng = engine
        self.li_tol = li_tol
        self.mse_tol = mse_tol
        names = []
        if li_tol is not None:
            names.append("li_row")
        if mse_tol is not None:
            names.append("mse_chords")
        self.names = tuple(names)
        self.log = dict(li_row_error=[], li_row_ok=[], mse_dtg_max_sigma=[],
                        mse_ok=[])

    def observe(self, k, meas):
        p = self.eng._pending
        ok, never, txt = True, None, []
        if self.li_tol is not None:
            li = meas.get("li")
            pred = p["li_predicted_plus_d"]
            e = (None if (li is None or pred is None
                          or not np.isfinite(li)) else float(li - pred))
            o = bool(e is not None and abs(e) <= self.li_tol)
            self.log["li_row_error"].append(e)
            self.log["li_row_ok"].append(o)
            ok = ok and o
            if e is None:
                never = ("the l_i row cannot be evaluated on this pass (no "
                         "finite delivered or predicted l_i)")
            txt.append("li_row=n/a" if e is None else
                       f"li_row={e:+.2e} (tol {self.li_tol:g})")
        if self.mse_tol is not None:
            dt = p.get("mse_dtg_max_sigma")
            o = bool(dt is not None and np.isfinite(dt) and dt <= self.mse_tol)
            self.log["mse_dtg_max_sigma"].append(dt)
            self.log["mse_ok"].append(o)
            ok = ok and o
            txt.append("mse_dtg=n/a" if dt is None else
                       f"mse_dtg={dt:.2e}sig (tol {self.mse_tol:g})")
        return ok, never, " ".join(txt)

    def record(self):
        from .jbs_loop import jsonable
        return jsonable(dict(li_tol=self.li_tol, mse_tol_sigma=self.mse_tol,
                             **self.log))

    def history_text(self):
        out = []
        if self.li_tol is not None:
            out.append("li_row_error=[" + ", ".join(
                "n/a" if v is None else f"{v:+.2e}"
                for v in self.log["li_row_error"])
                + f"] (tol {self.li_tol:g})")
        if self.mse_tol is not None:
            out.append("mse_dtg_max_sigma=[" + ", ".join(
                "n/a" if v is None else f"{v:.2e}"
                for v in self.log["mse_dtg_max_sigma"])
                + f"] (tol {self.mse_tol:g})")
        return " ".join(out)


# ---------------------------------------------------------------------------
#  the engine
# ---------------------------------------------------------------------------
class UnifiedEngine:
    """One reconstruction: the loop, the optional MSE stage, the delivery.

    *backend* provides ``solve(request, n_passes=1)``, ``snapshot()``,
    ``restore(eq)``, ``measure(want_chords=...)`` and ``n_solves`` (see
    :class:`TokaMakerBackend`).  *anchor* is the backend measurement of the
    anchor equilibrium E_0 (already solved)."""

    def __init__(self, contract, backend, settings, *, anchor, label=""):
        self.c = contract
        self.b = backend
        self.s = settings
        self.label = str(label or f"engine {contract.kind}")
        self.psi = np.asarray(contract.psi_N, dtype=float)
        self.state = EngineState()
        self.passes = []
        self._pending = None
        self.solves = dict(anchor=int(backend.n_solves))
        self.notices = []
        #: closure-limited flags the engine raised (never retried): the MSE
        #: orientation audit, an MSE term not applied
        self.flags = []
        self._resolve_rows()
        self._prior()
        g0 = complete_geometry(anchor["geom"])
        self.state.geom = g0
        self.state.lambda_bs = np.asarray(anchor["redl"], dtype=float).copy()
        self.anchor = anchor
        if self.s["delivery_correction"]:
            self.state.delivery_correction = np.zeros_like(self.psi)
        self._mse_phase = None
        self._phase_name = "loop"
        #: chi^2 of the reconstruction WITHOUT MSE at the stage's chords
        #: (the FD base state), once the stage has read them
        self._mse_pre = None
        self._mse_stage_fd = None    # the MSE stage's Jacobian record
        #: the MSE outcome recorded as ``engine_record()["mse"]``
        self.mse_summary = dict(requested=("mse" in settings["rows"]),
                                applied=False, mse_converged=None,
                                stage_failed=False, failure=None, flags=[])
        self._d_first = True
        self.pin = None
        if "q0" in self.rows:
            # the axis-current row, started with one AxisRowPin step from the
            # anchor: j_ref0 = j0 * q0(anchor) / q0_target, with j0 the
            # anchor's ACHIEVED axis current (its request need not carry Ip;
            # every closure's current does, so the achieved one is what a
            # closed pass solves)
            from .jbs_loop import AxisRowPin
            q = self.rows["q0"]
            j0a = float(np.interp(float(g0["psi_q"][0]), g0["psi_N"],
                                  anchor["achieved"]))
            row0 = j0a * float(anchor["q_row"]) / float(q["target"])
            self.pin = AxisRowPin(q["target"], self.s["q0_tol"], row0,
                                  label=self.label)
            self.state.q0_target = float(q["target"])

    # ---- rows ------------------------------------------------------------
    def _resolve_rows(self):
        c, s = self.c, self.s
        want = set(s["rows"])
        rows = dict(Ip=c.rows["Ip"])
        if "l_i" in want:
            if c.rows.get("l_i") is None:
                from .adapters import EngineInputRefused
                raise EngineInputRefused(
                    f"{self.label}: engine_rows has 'l_i' but the source "
                    "carries no l_i")
            rows["l_i"] = dict(c.rows["l_i"])
            li = rows["l_i"]
            if li["hard"] and li.get("tol") is None:
                li["tol"] = s["structured_li_tol"]
            li["criterion_tol"] = (float(li["tol"]) if li["hard"]
                                   else s["structured_li_tol"])
        self.preset = s["preset"]
        if "q0" in want:
            q = c.rows.get("q0")
            if q is None or not q.get("admitted", False):
                why = ("the source carries no q" if q is None else
                       f"the sawtooth gate rejected it ({q['gate_basis']})")
                msg = (f"[{self.label}] NOTICE: the q0 row was requested but "
                       f"is not active: {why}")
                if self.preset == "sawtooth_two_scalar":
                    msg += ("; preset 'sawtooth_two_scalar' falls back to "
                            "'bootstrap_scalar' (rows: Ip), as the legacy "
                            "sawtooth channel falls back to 'bootstrap'")
                    self.preset = "bootstrap_scalar"
                print(msg, flush=True)
                self.notices.append(msg)
            else:
                rows["q0"] = dict(q)
        if "mse" in want:
            if c.rows.get("mse") is None:
                from .adapters import EngineInputRefused
                raise EngineInputRefused(f"{self.label}: engine_rows has "
                                         "'mse' but no MSE rows were read")
            rows["mse"] = c.rows["mse"]
        self.rows = rows
        # soft rows go to the soft solver on the structured-basis presets
        # and on two_scalar_li (whose l_i row is the one soft row an IDS
        # carries); the other scalar presets impose their rows exactly
        self.soft = bool(self.preset in STRUCTURED_BASIS_PRESETS
                         + ("two_scalar_li",)
                         and (not rows["Ip"]["hard"]
                              or ("l_i" in rows and not rows["l_i"]["hard"])))
        if self.soft and "l_i" in rows and rows["l_i"]["hard"]:
            from .adapters import EngineInputRefused
            raise EngineInputRefused(
                f"{self.label}: a soft Ip row with a hard l_i row has no "
                "closure (the soft solver takes l_i as a measurement and the "
                "hard one imposes Ip exactly)")

    def _prior(self):
        from .utils import (STRUCTURED_BASIS_DEFAULT, STRUCTURED_PRESETS,
                            _weights_from_sigma)
        if self.preset == "structured":
            sp = STRUCTURED_PRESETS["li_soft_onesided"]
            self.basis = dict(STRUCTURED_BASIS_DEFAULT)
            self.sigma_ind = np.asarray(sp["sigma_ind"], dtype=float)
            self.sigma_bs = np.asarray(sp["sigma_bs"], dtype=float)
            self.sigma_up = np.asarray(sp["sigma_ind_up"], dtype=float)
            self.prior_name = "li_soft_onesided (utils.STRUCTURED_PRESETS)"
        elif self.preset == "structured_uniform":
            # the documented no-prior sensitivity: every coefficient
            # penalised equally (utils.STRUCTURED_WEIGHTS_UNIFORM, W = 1 i.e.
            # sigma = 1), no one-sided up-ladder.  On the hard closure only
            # the weights' RATIOS matter (no preference); with soft rows or
            # MSE chords it is an absolute sigma = 1 prior (see there).
            from .utils import STRUCTURED_WEIGHTS_UNIFORM as _U
            self.basis = dict(STRUCTURED_BASIS_DEFAULT)
            self.sigma_ind = 1.0 / np.sqrt(np.asarray(_U["ind"], float))
            self.sigma_bs = 1.0 / np.sqrt(np.asarray(_U["bs"], float))
            self.sigma_up = None
            self.prior_name = "uniform (utils.STRUCTURED_WEIGHTS_UNIFORM)"
        elif self.preset == "two_scalar_li":
            # the legacy secant's l_i family as a named closure: ONE scalar
            # on the inductive and ONE on the bootstrap (constant basis),
            # rows Ip + l_i.  Hard rows (g-file): a 2 x 2 system, no prior
            # enters.  Soft rows (IDS): the soft solver, where the constant
            # basis's sigma = 1 is the documented uniform prior.
            self.basis = dict(kind="constant")
            self.sigma_ind = np.array([1.0])
            self.sigma_bs = np.array([1.0])
            self.sigma_up = None
            self.prior_name = "constant basis, two scalars (Ip + l_i)"
        elif self.preset == "bootstrap_scalar":
            self.basis = dict(kind="constant")
            self.sigma_ind = np.array([0.0])      # pinned: s_ind = 1
            self.sigma_bs = np.array([1.0])
            self.sigma_up = None
            self.prior_name = "constant basis, s_ind pinned (bootstrap)"
        else:
            self.basis = dict(kind="constant")
            self.sigma_ind = np.array([1.0])
            self.sigma_bs = np.array([1.0])
            self.sigma_up = None
            self.prior_name = "constant basis, two scalars (sawtooth)"
        self.weights = dict(name=self.prior_name,
                            ind=tuple(_weights_from_sigma(self.sigma_ind)),
                            bs=tuple(_weights_from_sigma(self.sigma_bs)))

    # ---- closure ---------------------------------------------------------
    def close(self, geom, lam_bs, *, x=None, mse_lin=None, x_prev=None):
        """The closure on *geom* with bootstrap *lam_bs* (zero solves).

        Returns ``dict(out, jc, parts, x, li_predicted, axis, ip_gate)``;
        ``jc`` is the INTENDED current (the delivery correction is added by
        the caller).  With *x* given, the closure is not run: the current of
        those coefficients is composed (the FD Jacobian's perturbed
        requests)."""
        from .utils import (close_ip_structured, close_ip_structured_soft,
                            closure_sign_convention, ip_roundtrip_gate,
                            soft_closure_with_retry, structured_basis_eval)
        c = self.c
        # integrals and interpolations over the geometry's psi_N; the basis
        # on the run grid (identical in a psi_N run)
        psi = np.asarray(geom["psi_N"], dtype=float)
        _bx = (None if getattr(self.b, "coord", "psi_n") == "psi_n"
               else self.psi)
        _, parts = compose(geom, c.jB_ind, lam_bs, c.jB_fix)
        j_ind, j_bs = parts["ind"], parts["bs"]
        j_fix = parts["fix"] + parts["pressure"]
        Ip_abs = float(self.rows["Ip"]["target"])
        ip_ind, ip_bs, ip_fix = (_lin(geom, j_ind), _lin(geom, j_bs),
                                 _lin(geom, j_fix))
        sgn, Ip_signed, c_signed = closure_sign_convention(
            ip_ind, ip_bs, ip_fix, geom["c_affine"], Ip_abs)
        axis = None
        if "q0" in self.rows:
            psi0 = float(geom["psi_q"][0])
            axis = dict(psi=psi0, j_ind0=float(np.interp(psi0, psi, j_ind)),
                        j_bs0=float(np.interp(psi0, psi, j_bs)),
                        j_fix0=float(np.interp(psi0, psi, j_fix)),
                        j_ref0=float(self.pin.row))
        li_t = None
        li = self.rows.get("l_i")
        if li is not None:
            li_t = float(li["target"]) - float(self.state.li_discrepancy)
        Phi = structured_basis_eval(self.basis, self.psi)
        K = Phi.shape[0]
        if x is not None:
            x = np.asarray(x, dtype=float)
            s_ind, s_bs = 1.0 + x[:K] @ Phi, 1.0 + x[K:] @ Phi
            jc = s_ind * j_ind + s_bs * j_bs + j_fix
            return dict(out=None, jc=jc, parts=parts, x=x, axis=axis,
                        li_predicted=None, ip_gate=None)
        try:
            if self.soft:
                sig_ip = self.rows["Ip"].get("sigma")

                def _solve(x0):
                    return close_ip_structured_soft(
                        psi, geom["w_lin"], c_signed, Ip_signed,
                        (None if self.rows["Ip"]["hard"] else sig_ip),
                        j_ind, j_bs, j_fix, basis=self.basis,
                        sigma_ind=self.sigma_ind, sigma_bs=self.sigma_bs,
                        li_target=li_t,
                        li_sigma=(None if li is None else
                                  (None if li["hard"] else li["sigma"])),
                        li_kind=(li["kind"] if li else "li_1"),
                        li_geom=(geom["li_geom"] if li else None),
                        axis=axis, axis_sigma=None,
                        sigma_ind_up=self.sigma_up, mse_lin=mse_lin, x0=x0,
                        accept_noise_floor=True, basis_x=_bx)
                out = soft_closure_with_retry(_solve, x_prev=x_prev,
                                              who=self.label + " closure")
            else:
                out = close_ip_structured(
                    psi, geom["w_lin"], c_signed, Ip_signed, j_ind, j_bs,
                    j_fix, basis=self.basis, weights=self.weights, axis=axis,
                    li_target=li_t, li_kind=(li["kind"] if li else "li_1"),
                    li_geom=(geom["li_geom"] if li else None),
                    sigma_ind_up=self.sigma_up, mse_lin=mse_lin, basis_x=_bx)
            xs = np.concatenate([np.asarray(out["a"], float),
                                 np.asarray(out["b"], float)])
            jc = out["s_ind"] * j_ind + out["s_bs"] * j_bs + j_fix
            ip_closed = _lin(geom, jc) + c_signed
            gate = ip_roundtrip_gate(
                ip_closed, Ip_abs,
                posterior=(out["Ip_hybrid"] if self.soft else None),
                sigma_Ip=self.rows["Ip"].get("sigma"))
        except (RuntimeError, ValueError) as e:
            raise EngineClosureRefused(
                f"{self.label}: the closure refused ({e})") from e
        lp = out.get("li_predicted")
        return dict(out=out, jc=jc, parts=parts, x=xs, axis=axis,
                    li_predicted=(None if lp is None else float(lp)),
                    ip_gate=gate, sgn=float(sgn), Ip_signed=float(Ip_signed),
                    c_signed=float(c_signed))

    def _mse_lin(self):
        if self._mse_phase is None:
            return None
        from .utils import structured_mse_linear_model
        st = self.state
        return structured_mse_linear_model(st.mse_x0, st.mse_tg0, st.mse_J,
                                           self.rows["mse"]["chords"])

    def _tg(self, m):
        """tan(gamma) at the chords in use.  A chord that was found at the
        first read and is missing now is a refusal, never a stale value (the
        mesh does not move, so it should not happen)."""
        from .mse import mse_tan_gamma
        found = m.get("chords_found")
        if found is not None and not bool(np.all(found)):
            miss = [int(i) for i in np.asarray(
                self.rows["mse"]["chords"]["index"])[~np.asarray(found)]]
            raise EngineSolveError(
                f"{self.label}: MSE chord(s) at input index {miss} were on "
                "the solver mesh at the first read and are not now -- "
                "refusing to use a field that cannot be read")
        sp, stt = self.state.mse_sign
        return mse_tan_gamma(m["B_chords"], self.rows["mse"]["chords"], sp,
                             stt)

    # ---- one pass (the kernel's step) -----------------------------------
    def step(self, jbs, k, relax=None):
        st = self.state
        geom = st.geom
        cl = self.close(geom, jbs, mse_lin=self._mse_lin(),
                        x_prev=st.x)
        jint = np.asarray(relax(cl["jc"]) if relax is not None else cl["jc"],
                          dtype=float)
        dlt = st.delivery_correction
        req = jint + (0.0 if dlt is None else dlt)
        self.b.solve(req, n_passes=1)
        m = self.b.measure(want_chords=(self._mse_phase is not None))
        g1 = complete_geometry(m["geom"])
        dstat, dvec = _delivery_stats(req, m["achieved"], g1)
        if dlt is not None:
            dstat["intended_minus_achieved"] = _delivery_stats(
                jint, m["achieved"], g1)[0]
        # the coefficients whose current was SOLVED: the relaxer's blend of
        # the closure's coefficients (the composition is affine in x), so the
        # MSE linearisation is centred on what the equilibrium carries
        beta = (1.0 if relax is None else float(relax.beta))
        xs_prev = st.extras.get("x_solved")
        x_solved = (np.asarray(cl["x"], float) if (k == 0 or beta >= 1.0
                                                   or xs_prev is None)
                    else (1.0 - beta) * np.asarray(xs_prev, float)
                    + beta * np.asarray(cl["x"], float))
        st.extras["x_solved"] = x_solved
        lik = self.rows.get("l_i", {}).get("kind", "li_3")
        li_pred_d = (None if cl["li_predicted"] is None else
                     float(cl["li_predicted"]) + float(st.li_discrepancy))
        p = dict(k=k, cl=cl, cl_geom=geom, jint=jint, req=req, m=m, geom=g1,
                 x_solved=x_solved,
                 li_predicted_plus_d=li_pred_d, dvec=dvec, dstat=dstat,
                 d_used=float(st.li_discrepancy))
        if "l_i" in self.rows:
            p["li_model_solved_new_geom"] = li_of_current(jint, g1, lik)
        if self._mse_phase is not None:
            tg = self._tg(m)
            prev = self._mse_phase["tg_prev"]
            p["tg"] = tg
            p["mse_dtg_max_sigma"] = float(np.max(np.abs(tg - prev) / self.rows[
                "mse"]["chords"]["sigma_eff"]))
        self._pending = p
        meas = dict(w=g1["w_lin"] * conversion_factor(g1), x=g1["psi_N"],
                    li=m["li"], redl=m["redl"],
                    q0=(m["q_row"] if "q0" in self.rows else None))
        if "q0" in self.rows:
            meas["axis_current_solved"] = float(np.interp(
                float(g1["psi_q"][0]), g1["psi_N"], jint))
        # the next pass composes on this pass's solved geometry
        st.geom = g1
        st.x = cl["x"]
        st.request = req
        self.passes.append(self._pass_record(p))
        return meas

    def _pass_record(self, p):
        cl, m, out = p["cl"], p["m"], p["cl"]["out"] or {}
        rec = dict(
            k=int(p["k"]), phase=self._phase_name, x=cl["x"].tolist(),
            s_ind_range=[float(np.min(out["s_ind"])), float(np.max(out[
                "s_ind"]))] if "s_ind" in out else None,
            s_bs_range=[float(np.min(out["s_bs"])), float(np.max(out[
                "s_bs"]))] if "s_bs" in out else None,
            ip_gate_err_pct=(None if cl["ip_gate"] is None
                             else cl["ip_gate"]["err_pct"]),
            residual_sigma_Ip=out.get("residual_sigma_Ip"),
            closure_retry=out.get("closure_retry"),
            closure_stop=out.get("gn_stop_reason"),
            li=m["li"], li_predicted=cl["li_predicted"],
            li_discrepancy_used=p["d_used"],
            q_row=m.get("q_row"), Ip=m.get("Ip"),
            delivery=p["dstat"],
            n_solves=int(self.b.n_solves))
        if "tg" in p:
            rec["mse_dtg_max_sigma"] = p["mse_dtg_max_sigma"]
        return rec

    def on_pass(self, k, meas, J, entry):
        """Row updates between passes (never after the last one)."""
        p = self._pending
        om = entry.get("omega_next")
        if om is None:
            return
        st = self.state
        if "l_i" in self.rows:
            raw = float(p["m"]["li"]) - float(p["li_model_solved_new_geom"])
            r = float(self.s.get("li_row_relaxation", 1.0))
            if r == 1.0:
                # the update before engine_li_row_relaxation existed
                st.li_discrepancy = (raw if self._d_first else
                                     (1.0 - om) * st.li_discrepancy
                                     + om * raw)
            else:
                # under-relaxed: the step toward the measured discrepancy is
                # scaled by r (the first update moves d from 0 with w = 1);
                # same fixed point, per-pass gain times r
                w = r * (1.0 if self._d_first else float(om))
                st.li_discrepancy = ((1.0 - w) * st.li_discrepancy + w * raw)
            self._d_first = False
            self.passes[-1]["li_discrepancy_next"] = st.li_discrepancy
        if st.delivery_correction is not None:
            st.delivery_correction = np.asarray(p["dvec"], dtype=float).copy()
            self.passes[-1]["delivery_correction_max"] = float(np.max(np.abs(
                st.delivery_correction)))
        if self._mse_phase is not None:
            # Broyden (good) update of d tan(gamma)/d x from the pass just
            # solved, then refresh the linearisation point from the solve
            x_new = np.asarray(p["x_solved"], dtype=float)
            dx = x_new - st.mse_x0
            dtg = p["tg"] - st.mse_tg0
            nn = float(dx @ dx)
            if nn > 0.0 and self.s["mse_jacobian"] == "fd_broyden":
                st.mse_J = st.mse_J + np.outer(dtg - st.mse_J @ dx, dx) / nn
                self._mse_phase["n_broyden"] += 1
            st.mse_x0, st.mse_tg0 = x_new, np.asarray(p["tg"], float)
            self._mse_phase["tg_prev"] = np.asarray(p["tg"], float)

    # ---- the whole reconstruction ----------------------------------------
    def run(self):
        from .jbs_loop import run_jbs_loop
        t0 = time.perf_counter()
        s, st = self.s, self.state
        li = self.rows.get("l_i")
        rows_x = EngineRows(self, li_tol=(None if li is None
                                          else li["criterion_tol"]))
        meas0 = dict(li=self.anchor["li"],
                     q0=(self.anchor["q_row"] if self.pin else None))
        res = run_jbs_loop(
            st.lambda_bs, self.step, lambda m: m["redl"], s["loop"],
            Ip=float(self.rows["Ip"]["target"]), meas0=meas0, gate_li=True,
            gate_q0=bool(self.pin), label=self.label,
            init_source=("evaluate_jBS <j.B> on the anchor (the source's own "
                         "total current at the full pressure)"),
            on_pass=self.on_pass, q0_pin=self.pin, extra=rows_x)
        phases = [dict(name="loop", record=res["record"])]
        self.solves["loop"] = int(self.b.n_solves) - self.solves["anchor"]
        converged = bool(res["converged"])
        last = res
        if "mse" in self.rows and not converged:
            # jbs_loop_on_fail="flag": the loop did not converge, so the MSE
            # stage (which starts from a converged loop) does not run -- said
            # loudly, never a silent omission; a REQUIRED MSE term refuses
            from .utils import MSE_FLAG_PREFIX
            why = ("the self-consistent loop did not converge "
                   "(jbs_loop_on_fail='flag'), so the MSE stage, which starts "
                   "from a converged loop, did not run")
            if self.rows["mse"].get("required"):
                from .adapters import EngineInputRefused
                raise EngineInputRefused(
                    f"{self.label}: structured_mse_required=True but {why}")
            self.flags.append(MSE_FLAG_PREFIX + why + " -- the MSE term was "
                              "NOT applied")
            print(f"[{self.label}] WARNING closure-limited: "
                  + self.flags[-1], flush=True)
            phases.append(dict(name="mse", record=None, jacobian=dict(
                applied=False, reason=why, n_solves=0,
                n_free=int(np.count_nonzero(self._mse_free())))))
            self.mse_summary.update(applied=False, mse_converged=False,
                                    not_applied_reason=why)
        mse_applied = False
        saved = None
        if "mse" in self.rows and converged:
            # the converged reconstruction WITHOUT MSE, kept so that a
            # failure inside the MSE stage (structured_mse_required=False)
            # delivers it instead of nothing (see _mse_stage_failed)
            saved = self._save_no_mse_state()
            try:
                res_m, fd = self._mse_stage(res)
            except Exception as exc:
                if self.rows["mse"].get("required"):
                    raise
                self._mse_stage_failed(saved, exc, phases, where="stage")
                res_m, fd = None, None
            if fd is None:
                pass                       # failed: restored above
            elif res_m is None:            # too few chords on the mesh
                phases.append(dict(name="mse", record=None, jacobian=fd))
                self._mse_phase = None
                self.solves["mse_fd"] = fd["n_solves"]
            else:
                phases.append(dict(name="mse", record=res_m["record"],
                                   jacobian=fd))
                converged = bool(res_m["converged"])
                last = res_m
                mse_applied = True
        self._finish_state(last)
        n_before = int(self.b.n_solves)
        try:
            delivered = self._deliver(last, converged)
        except Exception as exc:
            # the delivery of the MSE fit is part of the MSE stage: a
            # failure there (structured_mse_required=False) also falls back
            # to the reconstruction without MSE, re-delivered and re-checked
            if not mse_applied or self.rows["mse"].get("required"):
                raise
            self._mse_stage_failed(saved, exc, phases, where="delivery")
            last, converged, mse_applied = res, bool(res["converged"]), False
            self._finish_state(last)
            n_before = int(self.b.n_solves)
            delivered = self._deliver(last, converged)
        if mse_applied:
            self._mse_fit_record(delivered, converged)
        self.solves["delivery"] = int(self.b.n_solves) - n_before
        self.solves["total"] = int(self.b.n_solves)
        return dict(converged=bool(converged and delivered["ok"]),
                    loop_converged=converged, phases=phases,
                    delivered=delivered, state=st,
                    wall_s=float(time.perf_counter() - t0))

    def _finish_state(self, last):
        """The state the delivery composes: the bootstrap iterate of the
        stage delivered (*last*), the fixed pressure and Ip, the q0 row."""
        st = self.state
        st.lambda_bs = np.asarray(last["jbs_used"], dtype=float)
        # the fixed pressure every pass solved with (a draw perturbs it)
        st.extras["pressure"] = np.asarray(self.c.pressure, dtype=float)
        st.extras["Ip"] = float(self.c.Ip)
        if self.pin is not None:
            st.q0_row = float(self.pin.row)

    # ---- the MSE fallback: the converged reconstruction without MSE --------
    def _capture_backend(self):
        """The backend's full state: :class:`bouquet.solver_state.
        SolverState` (equilibrium, settings, coil bounds, stashes) for a
        TokaMaker backend, else the backend's own ``snapshot()``."""
        mygs = getattr(self.b, "mygs", None)
        if mygs is not None and hasattr(mygs, "copy_eq"):
            from .solver_state import SolverState
            return ("solver_state", SolverState.capture(mygs))
        return ("snapshot", self.b.snapshot())

    def _restore_backend(self, cap):
        kind, obj = cap
        if kind == "solver_state":
            obj.restore()
        else:
            self.b.restore(obj)

    def _save_no_mse_state(self):
        """Everything the no-MSE delivery reads, as the converged loop left
        it: the backend (its equilibrium is the loop's last solved one), the
        engine state, the last pass, the q0 pin, the chord set and the
        solve count."""
        import copy
        return dict(backend=self._capture_backend(),
                    state=copy.deepcopy(self.state),
                    pending=copy.deepcopy(self._pending),
                    pin=copy.deepcopy(self.pin),
                    d_first=self._d_first,
                    rows_mse=self.rows["mse"],
                    chords=getattr(self.b, "chords", None),
                    n_solves=int(self.b.n_solves))

    def _mse_stage_failed(self, saved, exc, phases, *, where):
        """``structured_mse_required=False`` and the MSE stage (or the
        delivery of its fit) raised: restore the converged reconstruction
        without MSE (:meth:`_save_no_mse_state`), flag ``mse_stage_failed``
        with the exception text, warn loudly.  The caller then delivers the
        restored state through the ordinary delivery, whose solve and checks
        re-verify it.  Never silent and never a retry of the MSE fit."""
        import copy
        import warnings
        from .utils import MSE_FLAG_PREFIX
        n_spent = int(self.b.n_solves) - int(saved["n_solves"])
        self._restore_backend(saved["backend"])
        # in place: the caller (run) and the record hold this object
        from dataclasses import fields as _fields
        _st = copy.deepcopy(saved["state"])
        for _f in _fields(_st):
            setattr(self.state, _f.name, getattr(_st, _f.name))
        self._pending = copy.deepcopy(saved["pending"])
        self.pin = copy.deepcopy(saved["pin"])
        self._d_first = saved["d_first"]
        self.rows["mse"] = saved["rows_mse"]
        if hasattr(self.b, "chords"):
            self.b.chords = saved["chords"]
        self._mse_phase = None
        self._phase_name = "loop"
        txt = f"{type(exc).__name__}: {exc}"
        self.mse_summary.update(
            applied=False, mse_converged=False, stage_failed=True,
            failure=dict(where=str(where), type=type(exc).__name__,
                         message=str(exc), n_solves_spent=n_spent))
        if "mse_stage_failed" not in self.mse_summary["flags"]:
            self.mse_summary["flags"].append("mse_stage_failed")
        self.flags.append(
            MSE_FLAG_PREFIX + "mse_stage_failed: the MSE "
            + ("stage" if where == "stage" else "fit's delivery")
            + f" raised ({txt}); structured_mse_required=False, so the "
            "converged reconstruction WITHOUT MSE was restored, re-solved, "
            "re-checked and delivered -- the MSE term was NOT applied")
        # every solve since the snapshot (the stage's, the failed
        # delivery's) is counted once, under "mse_failed"
        for _k in ("mse_fd", "mse_passes", "mse_refresh"):
            self.solves.pop(_k, None)
        self.solves["mse_failed"] = (int(self.solves.get("mse_failed", 0))
                                     + n_spent)
        msg = f"[{self.label}] WARNING closure-limited: " + self.flags[-1]
        print(msg, flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        rec = getattr(exc, "record", None)
        for ph in phases:
            if ph.get("name") == "mse":
                ph["superseded_by_fallback"] = True
        # the failed phase's Jacobian record: every key a delivered MSE
        # phase's carries, as far as the stage got (n_free, the FD's
        # n_solves, the scheme, orientation, J_initial, the passes' Broyden
        # updates and refresh rounds) when the Jacobian was taken; n_free
        # and n_solves (= the solves spent) when it was not.  n_solves_spent
        # is always every solve since the snapshot (solves["mse_failed"]).
        taken = self._mse_stage_fd
        jac = dict(taken) if taken is not None else dict(
            n_solves=n_spent,
            n_free=int(np.count_nonzero(self._mse_free())))
        jac.update(applied=False, failed=True, where=str(where), reason=txt,
                   jacobian_taken=taken is not None,
                   n_solves_spent=n_spent)
        phases.append(dict(name="mse", record=(rec if isinstance(rec, dict)
                                               else None), jacobian=jac))

    def _mse_fit_record(self, delivered, converged):
        """The chord chi^2 of the DELIVERED MSE fit against the raw
        (E_r-corrected) chords with the stage's own weights, beside the
        reconstruction's without MSE (the FD base state), and the two FLAGS
        (flag only, never acceptance):

        * ``mse_worse_than_without`` -- delivered chi^2 above the pre-MSE
          chi^2;
        * ``mse_chi2_per_chord_high`` -- chi^2 / N above
          ``GenerationConfig.mse_chi2n_flag`` (default 10.0: owner-approved
          2026-10-07, flag only; never an acceptance criterion).

        Under ``jbs_loop_on_fail="flag"`` a non-converged MSE iterate is
        delivered carrying ``mse_converged=False`` and the same records."""
        from .utils import MSE_FLAG_PREFIX
        chk = delivered["checks"].get("mse")
        sm = self.mse_summary
        sm["applied"] = True
        sm["mse_converged"] = bool(converged and chk is not None
                                   and chk.get("ok", False))
        if chk is None:
            return
        n = int(self.rows["mse"]["chords"]["n_active"])
        c2 = float(chk["chi2"])
        thr = float(self.s["mse_chi2n_flag"])
        pre = self._mse_pre or {}
        c2p = pre.get("chi2")
        worse = bool(c2p is not None and not (np.isfinite(c2)
                                              and c2 <= float(c2p)))
        high = bool(not np.isfinite(c2) or c2 / n > thr)
        chk.update(n_chords=n, chi2_per_chord=c2 / n,
                   chi2_pre_mse=c2p,
                   chi2_per_chord_pre_mse=(None if c2p is None
                                           else float(c2p) / n),
                   worse_than_without=worse,
                   chi2n_flag=thr, chi2_per_chord_high=high,
                   chi2_basis=("raw E_r-corrected chords of the stage, "
                               "weights folded into sigma_eff "
                               "(bouquet.mse.mse_chi2)"),
                   mse_converged=sm["mse_converged"])
        sm.update(chi2=c2, n_chords=n, chi2_per_chord=c2 / n,
                  chi2_pre_mse=c2p, chi2_per_chord_pre_mse=chk[
                      "chi2_per_chord_pre_mse"],
                  worse_than_without=worse, chi2n_flag=thr,
                  chi2_per_chord_high=high)
        if worse:
            sm["flags"].append("mse_worse_than_without")
            self.flags.append(
                MSE_FLAG_PREFIX + f"mse_worse_than_without: delivered chi2 "
                f"{c2:.6g} ({n} chords) is above the chi2 {float(c2p):.6g} "
                "of the reconstruction without MSE: the delivered "
                "equilibrium fits the chords worse than the one without "
                "them")
            print(f"[{self.label}] WARNING closure-limited: "
                  + self.flags[-1], flush=True)
        if high:
            sm["flags"].append("mse_chi2_per_chord_high")
            self.flags.append(
                MSE_FLAG_PREFIX + f"mse_chi2_per_chord_high: delivered "
                f"chi2/N = {c2 / n:.4g} ({n} chords) is above "
                f"mse_chi2n_flag = {thr:g} (a FLAG, never acceptance; "
                "threshold owner-approved 2026-10-07) -- check the time "
                "slice, the calibration, the sigmas and the E_r correction")
            print(f"[{self.label}] WARNING closure-limited: "
                  + self.flags[-1], flush=True)
        if not sm["mse_converged"]:
            print(f"[{self.label}] WARNING: the delivered MSE fit is NOT "
                  "converged (mse_converged=False; jbs_loop_on_fail="
                  "'flag')", flush=True)

    def _mse_stage(self, res):
        """FD Jacobian at convergence (1 base + one solve per free
        coefficient), then the loop again with the chords as rows: the
        Jacobian held fixed (``engine_mse_jacobian="fd_chord"``, the
        default) or Broyden-updated every pass (``"fd_broyden"``), the
        linearisation offset refreshed from every solve either way.

        At the loop's convergence the Jacobian is RE-TAKEN once by the same
        finite differences (:meth:`_refresh_jacobian`) and the Gauss-Newton
        step the closure takes with it is formed: a chord iteration with a
        stale Jacobian stops at ``J0^T W r + grad(prior) = 0``, not at the
        chi^2 minimum.  If that step would move tan(gamma) by more than the
        stage's own criterion (``MSE_CHORD_OFFSET_TOL_SIGMA`` sigma_eff on
        some chord), the loop CONTINUES with the refreshed Jacobian (same
        pass ceiling) and the Jacobian is re-taken again at its convergence,
        at most :data:`MSE_JACOBIAN_MAX_REFRESHES` times; the stage is
        converged only when the fresh-Jacobian step is within the criterion
        (a stricter definition of converged, never a looser one)."""
        from .jbs_loop import JBSNotConverged, run_jbs_loop
        from .mse import (MSE_ORIENTATION_DCHI2, MSE_REASON_OFF_MESH,
                          mse_chi2, mse_equilibrium_orientation, mse_exclude,
                          mse_orientation, mse_orientation_check)
        from .utils import MSE_FLAG_PREFIX
        st, s = self.state, self.s
        # the stage's Jacobian record as far as it got: a stage that fails
        # after the Jacobian was taken keeps it on its failed phase
        # (_mse_stage_failed), with the same keys as a delivered one
        self._mse_stage_fd = None
        n0 = int(self.b.n_solves)
        lam = np.asarray(res["jbs_used"], dtype=float)
        g_last = self._geom_of_last_closure
        x0 = np.asarray(st.x, dtype=float)
        m0, snap = self._mse_base_solve(g_last, lam, x0)
        ch = self.rows["mse"]["chords"]
        # ---- chords OFF the solver mesh: excluded at this first read, with
        # their reason (the loop's chord stage does the same); fewer than
        # min_chords left: not applied (flagged) or, if required, refused
        B0 = np.asarray(m0["B_chords"], dtype=float).reshape(-1, 3)
        found = np.asarray(m0.get("chords_found", np.isfinite(B0).all(1)),
                           dtype=bool).ravel()
        excl = []
        if not found.all():
            excl = [int(i) for i in np.asarray(ch["index"])[~found]]
            ch = mse_exclude(ch, ~found, MSE_REASON_OFF_MESH)
            B0 = B0[found]
            self.rows["mse"] = dict(self.rows["mse"], chords=ch)
            self.b.chords = ch
            msg = (f"[{self.label}] WARNING MSE: {len(excl)} chord(s) at "
                   f"input index {excl} are OFF the solver mesh and are "
                   f"EXCLUDED; {int(ch['n_active'])} remain")
            print(msg, flush=True)
            self.notices.append(msg)
            if int(ch["n_active"]) < int(ch.get("min_chords", 1)):
                why = (f"only {int(ch['n_active'])} MSE chord(s) remain on "
                       f"the solver mesh (chord(s) {excl} are off it); at "
                       f"least {int(ch['min_chords'])} are required")
                if self.rows["mse"].get("required"):
                    from .adapters import EngineInputRefused
                    raise EngineInputRefused(f"{self.label}: {why}")
                self.flags.append(MSE_FLAG_PREFIX + why + " -- the MSE term "
                                  "was NOT applied")
                self.mse_summary.update(applied=False, mse_converged=False,
                                        not_applied_reason=why)
                self.b.restore(snap)
                return None, dict(applied=False, reason=why,
                                  n_solves=int(self.b.n_solves) - n0,
                                  n_free=int(np.count_nonzero(
                                      self._mse_free())),
                                  excluded_off_mesh=excl)
        # ---- orientation: STATED (the block's ip_sign/bt_sign against the
        # equilibrium's own directions, read off its field), never fitted;
        # audited, and flagged when another fits better by delta chi2 > 1
        eq_or = mse_equilibrium_orientation(B0, ch["R"], ch["Z"], m0["axis"])
        sp, stt = mse_orientation(ch, eq_or)
        chk = mse_orientation_check(B0, ch, sp, stt)
        st.mse_sign = (float(sp), float(stt))
        self.mse_summary["orientation_stated"] = [float(sp), float(stt)]
        if chk["disagrees"]:
            self.flags.append(
                MSE_FLAG_PREFIX + "the data disagree with the stated field "
                f"orientation (sign_pol {sp:+.0f}, sign_tor {stt:+.0f}): "
                f"orientation {chk['best_other']} fits the chords better by "
                f"delta chi2 = {chk['delta_chi2']:.4g} (> "
                f"{MSE_ORIENTATION_DCHI2:g}); the stated orientation is KEPT "
                "-- check ip_sign/bt_sign")
            print(f"[{self.label}] WARNING closure-limited: "
                  + self.flags[-1], flush=True)
        table = chk["table"]
        tg0 = self._tg(dict(m0, B_chords=B0, chords_found=None))
        # the chord chi^2 of the reconstruction WITHOUT MSE (this base solve
        # IS its last closure, solved unrelaxed), against the same chords
        c2_pre, _z = mse_chi2(tg0, ch)
        self._mse_pre = dict(chi2=float(c2_pre), n_chords=int(ch["n_active"]),
                             chi2_per_chord=float(c2_pre)
                             / int(ch["n_active"]))
        J = self._mse_fd(g_last, lam, x0, tg0, snap)
        self.b.restore(snap)
        st.mse_J, st.mse_x0, st.mse_tg0 = J, x0, tg0
        st.geom = complete_geometry(m0["geom"])
        fd = dict(n_solves=int(self.b.n_solves) - n0,
                  n_free=int(np.count_nonzero(self._mse_free())),
                  fd_step=float(s["mse_fd_step"]),
                  applied=True, excluded_off_mesh=excl,
                  orientation=dict(
                      pol=float(sp), tor=float(stt),
                      ip_sign_data=float(ch["ip_sign"]),
                      bt_sign_data=float(ch["bt_sign"]),
                      ip_sign_equilibrium=float(eq_or["ip"]),
                      bt_sign_equilibrium=float(eq_or["bt"]),
                      rule=("STATED, not fitted: sign_pol = ip_sign(data) * "
                            "ip_sign(equilibrium), sign_tor = bt_sign(data) "
                            "* bt_sign(equilibrium)"),
                      delta_chi2=float(chk["delta_chi2"]),
                      disagrees=bool(chk["disagrees"]), note=chk["note"]),
                  sign_table={k: float(v) for k, v in table.items()},
                  scheme=mse_scheme_text(s["mse_jacobian"]),
                  J_initial=J.tolist(), tg_base=tg0.tolist(),
                  chi2_pre_mse=float(c2_pre))
        self._mse_stage_fd = fd
        self._mse_phase = dict(tg_prev=tg0, n_broyden=0)
        self._phase_name = "mse"
        meas0 = dict(li=m0["li"], q0=(m0["q_row"] if self.pin else None))
        n1 = int(self.b.n_solves)
        rounds, cnt = [], dict(n_ref=0)
        lp = s["loop"]
        try:
            out = self._mse_passes(lam, meas0, rounds, lp, cnt)
        finally:
            # filled on success AND on a raise (a pass ceiling, a failed
            # solve, the refresh cap): the failed phase records the passes'
            # Broyden updates and refresh rounds as far as they got
            fd["n_broyden_updates"] = int(self._mse_phase["n_broyden"])
            n_ref = int(cnt["n_ref"])
            fd["n_pass_solves"] = int(self.b.n_solves) - n1 - n_ref
            fd["refresh"] = dict(
                rule=("at each MSE loop's convergence the Jacobian is "
                      "re-taken by the same finite differences around the "
                      "last closure's coefficients; the closure's "
                      "Gauss-Newton step with it must move tan(gamma) by <= "
                      "mse_tol_sigma sigma_eff on every chord (else the loop "
                      "continues with it)"),
                max_refreshes=int(MSE_JACOBIAN_MAX_REFRESHES),
                rounds=rounds, n_solves=n_ref,
                fresh_J_stationary=bool(rounds and rounds[-1]["ok"]))
            if rounds:
                fd["jacobian_refresh_rel_change"] = rounds[-1][
                    "jacobian_refresh_rel_change"]
                fd["refresh_step_norm"] = rounds[-1]["refresh_step_norm"]
                fd["refresh_step_dtg_max_sigma"] = rounds[-1][
                    "refresh_step_dtg_max_sigma"]
        self.solves["mse_fd"] = fd["n_solves"]
        self.solves["mse_passes"] = fd["n_pass_solves"]
        self.solves["mse_refresh"] = n_ref
        return out, fd

    def _mse_passes(self, lam, meas0, rounds, lp, cnt):
        """The MSE loop, re-run with the refreshed Jacobian until the
        fresh-Jacobian step is within the criterion (see :meth:`_mse_stage`);
        appends each refresh's record to *rounds* and its solves to
        ``cnt["n_ref"]`` (as it goes: a raise leaves both as far as they
        got)."""
        from .jbs_loop import JBSNotConverged
        s = self.s
        while True:
            out = self._mse_loop(lam, meas0, label=self.label + " +MSE"
                                 + ("" if not rounds else
                                    f" (refreshed J {len(rounds)})"),
                                 init_source=(
                                     "the converged loop's bootstrap iterate"
                                     if not rounds else
                                     "the previous MSE loop's bootstrap "
                                     "iterate (refreshed Jacobian)"))
            if not out["converged"]:
                break
            nr = int(self.b.n_solves)
            rf = self._refresh_jacobian(out)
            cnt["n_ref"] += int(self.b.n_solves) - nr
            rounds.append(rf["record"])
            if rf["ok"]:
                break
            if len(rounds) >= MSE_JACOBIAN_MAX_REFRESHES:
                why = (f"the Gauss-Newton step with the refreshed Jacobian "
                       f"still moves tan(gamma) by "
                       f"{rf['record']['refresh_step_dtg_max_sigma']:.3g} "
                       f"sigma_eff (> {s['mse_tol_sigma']:g}) after "
                       f"{len(rounds)} refresh(es) "
                       f"(MSE_JACOBIAN_MAX_REFRESHES = "
                       f"{MSE_JACOBIAN_MAX_REFRESHES}): the fit is not at "
                       "the fresh-Jacobian stationary point")
                rec = dict(out["record"])
                rec.update(converged=False, stop_reason=why,
                           fail_message=f"{self.label} +MSE: {why}")
                out = dict(out, converged=False, record=rec)
                print(f"  [engine] {self.label} +MSE: NOT converged: {why}",
                      flush=True)
                if lp.get("on_fail", "raise") == "raise":
                    raise JBSNotConverged(rec["fail_message"], rec)
                break
            # continue with the refreshed Jacobian, from its base solve
            # (state installed by _refresh_jacobian)
            lam = np.asarray(out["jbs_used"], dtype=float)
            meas0 = rf["meas0"]
            rounds[-1]["previous_loop_record"] = out["record"]
        return out

    def _mse_free(self):
        return np.concatenate([self.sigma_ind > 0, self.sigma_bs > 0])

    def _mse_base_solve(self, g, lam, x):
        """Solve the composition of coefficients *x* on geometry *g* with
        bootstrap *lam* (plus the delivery correction), unrelaxed; return
        its chord measurement and a snapshot of it (one solve)."""
        dlt = self.state.delivery_correction
        base = self.close(g, lam, x=x)["jc"] + (0.0 if dlt is None else dlt)
        self.b.solve(base, n_passes=1)
        m = self.b.measure(want_chords=True)
        return m, self.b.snapshot()

    def _mse_fd(self, g, lam, x, tg, snap):
        """``d tan(gamma) / d x`` by forward differences around the base
        state *snap* (coefficients *x*, tan(gamma) *tg*): one solve per free
        coefficient, each from the base state
        (:func:`bouquet.utils.structured_mse_jacobian`)."""
        from .utils import structured_mse_jacobian
        dlt = self.state.delivery_correction

        def _tg_of(xp):
            self.b.restore(snap)
            r = self.close(g, lam, x=xp)["jc"] + (0.0 if dlt is None
                                                  else dlt)
            self.b.solve(r, n_passes=1)
            return self._tg(self.b.measure(want_chords=True))

        return structured_mse_jacobian(_tg_of, x, tg, self._mse_free(),
                                       step=self.s["mse_fd_step"])

    def _mse_loop(self, lam, meas0, *, label, init_source):
        from .jbs_loop import run_jbs_loop
        li = self.rows.get("l_i")
        rows_x = EngineRows(self, li_tol=(None if li is None
                                          else li["criterion_tol"]),
                            mse_tol=self.s["mse_tol_sigma"])
        return run_jbs_loop(
            lam, self.step, lambda m: m["redl"], dict(self.s["loop"]),
            Ip=float(self.rows["Ip"]["target"]), meas0=meas0, gate_li=True,
            gate_q0=bool(self.pin), label=label, init_source=init_source,
            on_pass=self.on_pass, q0_pin=self.pin, extra=rows_x)

    def _refresh_jacobian(self, out):
        """Re-take the chord Jacobian at a converged MSE loop and judge the
        Gauss-Newton step it implies.

        Base: the last pass's closure coefficients ``x_b`` composed on the
        geometry that pass composed on, with its bootstrap iterate, solved
        unrelaxed (one solve; tan(gamma) ``tg_b``); then the same forward
        differences as the stage's first Jacobian (one solve per free
        coefficient).  With the linear model re-centred on ``(x_b, tg_b)``
        the closure (zero solves, every other row as the last pass had it)
        is run once with the Jacobian in use and once with the refreshed
        one; the refreshed step ``x_new - x_b`` is converted to the
        tan(gamma) move it predicts, ``max |J_new (x_new - x_b)| /
        sigma_eff``, and compared with the stage's own criterion
        ``mse_tol_sigma`` (the coefficients have no tolerance of their
        own).  Recorded: ``jacobian_refresh_rel_change`` = ``|J_new -
        J_old|_F / |J_old|_F``, ``refresh_step_norm`` = ``max |x_new -
        x_b|``, the old-Jacobian step for comparison and the relative change
        of the step.

        ``ok``: the backend is put back at the converged pass's state and
        the stage's state is untouched (the delivery composes exactly what
        it did before).  Not ``ok``: the refreshed Jacobian and the base
        state become the linearisation the loop continues from."""
        from .utils import structured_mse_linear_model
        st, s = self.state, self.s
        ch = self.rows["mse"]["chords"]
        sig = np.asarray(ch["sigma_eff"], dtype=float)
        p = self._pending
        g = p["cl_geom"]
        lam = np.asarray(out["jbs_used"], dtype=float)
        x_b = np.asarray(st.x, dtype=float)
        J_old = np.asarray(st.mse_J, dtype=float).copy()
        n0 = int(self.b.n_solves)
        conv = self.b.snapshot()
        m_b, snap_b = self._mse_base_solve(g, lam, x_b)
        tg_b = self._tg(m_b)
        J_new = self._mse_fd(g, lam, x_b, tg_b, snap_b)
        lin_old = structured_mse_linear_model(x_b, tg_b, J_old, ch,
                                              who=self.label + " MSE refresh")
        lin_new = structured_mse_linear_model(x_b, tg_b, J_new, ch,
                                              who=self.label + " MSE refresh")
        x_old = np.asarray(self.close(g, lam, mse_lin=lin_old,
                                      x_prev=x_b)["x"], dtype=float)
        x_new = np.asarray(self.close(g, lam, mse_lin=lin_new,
                                      x_prev=x_b)["x"], dtype=float)
        d_new, d_old = x_new - x_b, x_old - x_b
        dtg_new = float(np.max(np.abs(J_new @ d_new) / sig))
        dtg_old = float(np.max(np.abs(J_old @ d_old) / sig))
        nJ = float(np.linalg.norm(J_old))
        nd = max(float(np.linalg.norm(d_new)), float(np.linalg.norm(d_old)))
        ok = bool(np.isfinite(dtg_new) and dtg_new <= s["mse_tol_sigma"])
        rec = dict(
            n_solves=int(self.b.n_solves) - n0,
            jacobian_refresh_rel_change=float(
                np.linalg.norm(J_new - J_old) / max(nJ, 1e-300)),
            refresh_step_norm=float(np.max(np.abs(d_new))),
            refresh_step_norm_old_J=float(np.max(np.abs(d_old))),
            refresh_step_rel_change=(float(np.linalg.norm(d_new - d_old)
                                           / nd) if nd > 0.0 else 0.0),
            refresh_step_dtg_max_sigma=dtg_new,
            old_J_step_dtg_max_sigma=dtg_old,
            tol_sigma=float(s["mse_tol_sigma"]), ok=ok,
            base_dtg_vs_last_pass_max_sigma=float(np.max(np.abs(
                tg_b - np.asarray(p["tg"], dtype=float)) / sig)),
            continued=not ok)
        print(f"  [engine] {self.label} +MSE: Jacobian refreshed at "
              f"convergence: |dJ|/|J| = "
              f"{rec['jacobian_refresh_rel_change']:.3e}, fresh-J step "
              f"moves tan(gamma) by {dtg_new:.3e} sigma (tol "
              f"{s['mse_tol_sigma']:g})"
              + (" -- within the criterion" if ok else
                 " -- CONTINUING with the refreshed Jacobian"), flush=True)
        out_d = dict(ok=ok, record=rec)
        if ok:
            self.b.restore(conv)
            return out_d
        self.b.restore(snap_b)
        st.mse_J, st.mse_x0, st.mse_tg0 = J_new, x_b, tg_b
        st.geom = complete_geometry(m_b["geom"])
        self._mse_phase["tg_prev"] = tg_b
        out_d["meas0"] = dict(li=m_b["li"],
                              q0=(m_b["q_row"] if self.pin else None))
        return out_d

    @property
    def _geom_of_last_closure(self):
        # the geometry the LAST pass composed on is the one before the last
        # solve; the step stores the solved one in state.geom, so keep both
        return self._pending["cl_geom"] if "cl_geom" in (self._pending or {}) \
            else self.state.geom

    def _deliver(self, res, converged):
        """The delivery solve and its checks (docs/engine.md, "Delivery")."""
        from .jbs_loop import JBSNotConverged, check_delivered, jsonable
        st, s = self.state, self.s
        lp = s["loop"]
        geom = st.geom
        lam = st.lambda_bs
        prev = self._pending
        cl = self.close(geom, lam, mse_lin=self._mse_lin(), x_prev=st.x)
        dlt = st.delivery_correction
        R = cl["jc"] + (0.0 if dlt is None else dlt)
        self.b.solve(R, n_passes=2)
        m = self.b.measure(want_chords=(self._mse_phase is not None),
                           final=True)
        g1 = complete_geometry(m["geom"])
        w = g1["w_lin"] * conversion_factor(g1)
        chk = check_delivered(m["redl"], lam, w, g1["psi_N"],
                              float(self.rows["Ip"]["target"]), lp)
        checks = dict(loop=dict(r_j=chk["r_j"], r_I=chk["r_I"],
                                ok=bool(chk["ok"])))
        misses = [] if chk["ok"] else [
            f"r_j={chk['r_j']:.2e} (tol {lp['rtol_j']:g}) / r_I="
            f"{chk['r_I']:.2e} (tol {lp['rtol_Ip']:g})"]
        dli = abs(float(m["li"]) - float(prev["m"]["li"]))
        checks["dl_i"] = dict(value=dli, tol=lp["tol_li"],
                              ok=bool(dli <= lp["tol_li"]))
        if not checks["dl_i"]["ok"]:
            misses.append(f"|dl_i| vs the last pass {dli:.2e} "
                          f"(tol {lp['tol_li']:g})")
        from .jbs_loop import _relative_difference
        cur = _relative_difference(cl["jc"], prev["jint"], w, g1["psi_N"])
        checks["current_residual"] = dict(value=cur, tol=lp["rtol_j"],
                                          ok=bool(cur is not None
                                                  and cur <= lp["rtol_j"]))
        if not checks["current_residual"]["ok"]:
            misses.append(f"closure-half current residual {cur} "
                          f"(tol {lp['rtol_j']:g})")
        li = self.rows.get("l_i")
        if li is not None:
            e = float(m["li"]) - (float(cl["li_predicted"])
                                  + float(st.li_discrepancy))
            checks["l_i"] = dict(delivered=float(m["li"]),
                                 target=float(li["target"]),
                                 predicted_plus_d=float(cl["li_predicted"])
                                 + float(st.li_discrepancy),
                                 error=e, tol=li["criterion_tol"],
                                 hard=bool(li["hard"]),
                                 residual_sigma=(None if li["hard"] else
                                                 (float(m["li"]) - li["target"])
                                                 / li["sigma"]),
                                 ok=bool(abs(e) <= li["criterion_tol"]))
            if not checks["l_i"]["ok"]:
                misses.append(f"l_i row error {e:+.2e} (tol "
                              f"{li['criterion_tol']:g})")
        if self.pin is not None:
            r = float(m["q_row"]) - self.pin.q0_target
            dq = abs(float(m["q_row"]) - float(prev["m"]["q_row"]))
            checks["q0"] = dict(delivered=float(m["q_row"]),
                                target=self.pin.q0_target, residual=r,
                                tol=self.pin.q0_tol,
                                ok=bool(abs(r) <= self.pin.q0_tol),
                                dq0=dq, dq0_tol=lp["tol_q0"],
                                dq0_ok=bool(dq <= lp["tol_q0"]))
            if not checks["q0"]["ok"]:
                misses.append(f"q0 - q0_target {r:+.2e} (q0_tol "
                              f"{self.pin.q0_tol:g})")
            if not checks["q0"]["dq0_ok"]:
                misses.append(f"|dq0| vs the last pass {dq:.2e} (tol "
                              f"{lp['tol_q0']:g})")
        if self._mse_phase is not None:
            from .mse import mse_chi2
            tg = self._tg(m)
            ch = self.rows["mse"]["chords"]
            dt = float(np.max(np.abs(tg - prev["tg"]) / ch["sigma_eff"]))
            c2, z = mse_chi2(tg, ch)
            checks["mse"] = dict(dtg_max_sigma=dt, tol=s["mse_tol_sigma"],
                                 ok=bool(dt <= s["mse_tol_sigma"]),
                                 chi2=float(c2), residual_sigma=z.tolist(),
                                 tgamma=tg.tolist())
            if not checks["mse"]["ok"]:
                misses.append(f"MSE tan(gamma) change {dt:.2e} sigma (tol "
                              f"{s['mse_tol_sigma']:g})")
        dstat, dvec = _delivery_stats(R, m["achieved"], g1)
        st.x = cl["x"]
        st.request = R
        ok = not misses
        rec = dict(ok=bool(ok), checks=checks, misses=misses,
                   delivery=dstat, n_passes_solve=2,
                   closure=_closure_record(cl))
        self.delivered_meas = m
        self.delivered_closure = cl
        self.delivered_geom_solved = g1
        self.delivered_dvec = dvec
        if not ok:
            msg = (f"{self.label}: the delivered equilibrium fails the "
                   "loop / row criteria: " + "; ".join(misses))
            rec["fail_message"] = msg
            print("  [engine] " + msg, flush=True)
            if converged and lp.get("on_fail", "raise") == "raise":
                raise JBSNotConverged(msg, jsonable(rec))
        return rec


def _closure_record(cl):
    """JSON-safe summary of one closure result."""
    from .jbs_loop import jsonable
    out = cl["out"] or {}
    keep = ("solver", "weights_name", "constraints", "ohm_scale_eff",
            "bs_scale_eff", "structure_ind", "structure_bs", "Ip_hybrid",
            "ip_residual_pct", "residual_sigma_Ip", "residual_sigma_li",
            "li_target", "li_predicted", "li_anchor", "axis_residual",
            "prior_chi2", "objective", "gn_stop_reason", "closure_retry",
            "mse_chi2_model", "n_sign_iter", "sign_pattern", "a", "b")
    rec = {k: out.get(k) for k in keep if k in out}
    rec["s_ind"] = out.get("s_ind")
    rec["s_bs"] = out.get("s_bs")
    rec["ip_gate"] = cl.get("ip_gate")
    rec["axis"] = cl.get("axis")
    return jsonable(rec)


# ---------------------------------------------------------------------------
#  the TokaMaker backend (solver tests only; the fast suite uses a toy)
# ---------------------------------------------------------------------------
class TokaMakerBackend:
    """The engine's view of a live TokaMaker solver.

    ``solve`` is the IMAS baseline's ``solve_jphi`` (P' from the fixed
    pressure over the CURRENT flux range, ``jphi-linterp`` request,
    ``set_targets(Ip, pax)``), one or two passes; ``measure`` reads the
    solved equilibrium: :func:`bouquet.utils.fsa_current_geometry`,
    :func:`bouquet.physics.evaluate_jBS` (its Redl ``<j.B>`` with the shared
    innermost-surface repair :func:`~bouquet.TokaMaker_interface.
    smooth_jbs_transition`), :func:`bouquet.utils.li_closure_geometry`,
    :func:`bouquet.utils.li_achieved`, q at the row radius, the achieved
    ``<j_phi>`` (:func:`bouquet.utils.eq_jphi_profile`) and the field at the
    MSE chords (:func:`bouquet.mse.mse_field_at`).

    ``maxits``: the GS iteration cap set on the solver for EVERY solve of
    this backend and restored after it; ``None`` leaves the solver's own cap
    untouched.  The reconstruction's backend is built with ``None``; an
    engine draw's (and the zero-perturbation draw's) with
    ``GenerationConfig.engine_draw_solve_maxits`` (default 100).  A solve that hits the
    cap fails exactly as any failed solve (:class:`EngineSolveError`); it is
    never re-solved at another tolerance.

    :meth:`set_inputs` replaces the pressure and the kinetics a DRAW solves
    and measures with (``None`` keeps the contract's).

    ``edge_pressure``: the two settings of :mod:`bouquet.edge_pressure`
    (``edge_pprime_pin``, ``separatrix_pressure``); every ``P'`` and axis
    target of this backend is built by that module's helper, and a final
    measurement carries both pressure frames (``pressure_frames``)."""

    def __init__(self, mygs, contract, *, psi_pad=1e-3, li_kind="li_3",
                 q_psi=None, chords=None, maxits=None, edge_pressure=None,
                 edge_taper=None, coord="psi_n", eps_definition=None):
        self.mygs = mygs
        self.c = contract
        #: the Redl eps / nu* R of every evaluation of this backend
        #: (GenerationConfig.eps_definition; None: evaluate_jBS's default)
        self.eps_definition = eps_definition
        #: the run grid (psi_N, or Phi_N with coord="phi_n"); every solve
        #: tags its profiles with coord and every measurement samples the
        #: geometry at the nodes' psi_N on that solve's own map
        self.psi = np.asarray(contract.psi_N, dtype=float)
        self.coord = coords.check_coord(coord)
        #: the SWB edge taper's factor on psi (engine_edge_taper(); None:
        #: off), carried on every measured geometry so compose() applies it;
        #: in a Phi_N run it is re-evaluated on each geometry's psi_N
        self.edge_taper = None
        self._taper = None
        if edge_taper and edge_taper.get("on"):
            from .physics import edge_taper_weight
            self._taper = (float(edge_taper["psi0"]), int(edge_taper["shape"]))
            self.edge_taper = edge_taper_weight(self.psi, *self._taper)
        self.psi_pad = float(psi_pad)
        self.li_kind = str(li_kind)
        self.q_psi = (float(np.clip(self.psi[0], psi_pad, 1 - psi_pad))
                      if q_psi is None else float(q_psi))
        self.chords = chords
        self.n_solves = 0
        self.p = np.asarray(contract.pressure, dtype=float)
        #: the two edge-pressure settings (bouquet.edge_pressure); None:
        #: the defaults (separatrix_pressure="offset"); the pre-change
        #: settings (EdgePressure.pre_change()) are the old arrays bit for bit
        self.edge = resolve_edge_pressure(edge_pressure)
        self.kin = None
        if maxits is not None and (isinstance(maxits, bool) or int(maxits)
                                   != maxits or int(maxits) < 1):
            raise ValueError(f"engine backend: maxits={maxits!r} must be an "
                             "integer >= 1 or None")
        self.maxits = None if maxits is None else int(maxits)
        #: wall time [s] of the last GS solve (a failed one included)
        self.last_solve_s = None

    def set_inputs(self, pressure=None, kinetics=None):
        """The pressure [Pa] and the kinetics (``ne, te, ni, ti, zeff`` on
        ``psi_N``) the following solves and measurements use -- a draw's
        own; ``None`` restores the contract's."""
        self.p = np.asarray(self.c.pressure if pressure is None else pressure,
                            dtype=float)
        self.kin = None if kinetics is None else dict(kinetics)

    def _kinetics(self):
        return self.c.kinetics if self.kin is None else self.kin

    def flux_integral(self, psi_N, profile):
        """The solver's flux-surface integral of *profile* on the current
        equilibrium (the draw sampler's pressure match)."""
        return self.mygs.flux_integral(
            np.asarray(coords.psi_at(self.mygs, np.asarray(psi_N, dtype=float),
                                     self.coord), dtype=float),
            np.asarray(profile, dtype=float))

    def redl(self, kinetics=None):
        """Redl ``<j.B>`` on the CURRENT equilibrium with *kinetics*
        (default: the backend's), with the shared innermost-surface repair --
        exactly the ``redl`` :meth:`measure` returns, without the rest."""
        from .physics import evaluate_jBS
        from .TokaMaker_interface import smooth_jbs_transition
        kin = self._kinetics() if kinetics is None else kinetics
        _j, d = evaluate_jBS(self.mygs.copy_eq(), self.psi, kin["ne"],
                             kin["te"], kin["ni"], kin["ti"], kin["zeff"],
                             psi_pad=self.psi_pad, isolate_edge=False,
                             smooth_axis=False, coord=self.coord,
                             eps_definition=self.eps_definition)
        return smooth_jbs_transition(np.asarray(d["j_dot_B"], dtype=float))

    def p_sep(self) -> float:
        """The separatrix pressure added back at reporting / delivery for
        the pressure CURRENTLY set (a draw's own): ``p[-1]`` under
        ``separatrix_pressure="offset"``, exactly 0 under ``"legacy"``."""
        return self.edge.p_offset(self.p)

    def solve(self, request, n_passes=1):
        mygs = self.mygs
        req = np.asarray(request, dtype=float)
        if not np.all(np.isfinite(req)):
            raise EngineSolveError("engine: refusing to hand a non-finite "
                                   "request to the GS solver")
        ffp = coords.oft_prof("jphi-linterp", self.psi, req, self.coord)
        saved = None
        if self.maxits is not None:
            saved = int(mygs.settings.maxits)
            if saved != self.maxits:
                mygs.settings.maxits = self.maxits
                mygs.update_settings()
            else:
                saved = None
        try:
            for k in range(int(n_passes)):
                psi_range = mygs.psi_bounds[1] - mygs.psi_bounds[0]
                mygs.set_targets(Ip=float(self.c.Ip),
                                 pax=solver_pax(self.p, self.edge))
                mygs.set_profiles(pp_prof=solver_pp_profile(
                    self.psi, self.p, psi_range, self.edge,
                    coord=self.coord), ffp_prof=ffp)
                _t0 = time.perf_counter()
                try:
                    mygs.solve()
                except ValueError as e:
                    self.last_solve_s = time.perf_counter() - _t0
                    raise EngineSolveError(
                        f"engine GS solve failed (pass {k + 1}/{n_passes}"
                        + ("" if self.maxits is None else
                           f", maxits {self.maxits}") + f"): {e}") from e
                finally:
                    self.n_solves += 1
                self.last_solve_s = time.perf_counter() - _t0
        finally:
            if saved is not None:
                mygs.settings.maxits = saved
                mygs.update_settings()

    def snapshot(self):
        return self.mygs.copy_eq()

    def restore(self, eq):
        self.mygs.replace_eq(source_eq=eq)

    def measure(self, want_chords=False, final=False):
        from .physics import evaluate_jBS
        from .TokaMaker_interface import smooth_jbs_transition
        from .utils import (eq_jphi_profile, fsa_current_geometry,
                            li_achieved, li_closure_geometry)
        mygs, kin, pad = self.mygs, self._kinetics(), self.psi_pad
        eq = mygs.copy_eq()
        # the nodes' psi_N on THIS solve's map (the nodes themselves in a
        # psi_N run): every integral and interpolation of this measurement
        # uses geom["psi_N"]
        geom = fsa_current_geometry(
            eq, np.asarray(coords.psi_at(eq, self.psi, self.coord), float),
            psi_pad=pad, want_pprime=True)
        if geom["inv_R2"] is None:
            raise EngineSolveError("engine: this OFT build's get_q returns no "
                                   "<1/R^2>; the jphi-linterp Ip measure "
                                   "cannot be formed")
        _j, d = evaluate_jBS(eq, self.psi, kin["ne"], kin["te"], kin["ni"],
                             kin["ti"], kin["zeff"], psi_pad=pad,
                             isolate_edge=False, smooth_axis=False,
                             coord=self.coord,
                             eps_definition=self.eps_definition)
        redl = smooth_jbs_transition(np.asarray(d["j_dot_B"], dtype=float))
        geom["F"] = np.asarray(d["F"], dtype=float)
        geom["B2"] = np.asarray(d["avg_B2"], dtype=float)
        geom["li_geom"] = li_closure_geometry(eq, geom, psi_pad=pad)
        geom["edge_taper"] = self.edge_taper
        if self._taper is not None and self.coord != coords.PSI:
            from .physics import edge_taper_weight
            geom["edge_taper"] = edge_taper_weight(geom["psi_N"], *self._taper)
        li = float(li_achieved(eq, li_kind=self.li_kind, psi_pad=pad)[0])
        # q on the geometry's own surfaces (the legacy _q0_of reads index 0
        # of the same call), at the row radius
        _pq = np.ascontiguousarray(np.asarray(geom["psi_q"], dtype=float))
        _q = np.asarray(eq.get_q(psi=_pq.copy())[1], dtype=float)
        q_row = float(np.interp(self.q_psi, _pq, _q))
        Ip = abs(float(eq.get_globals()[0]))
        achieved = np.asarray(eq_jphi_profile(geom, "jphi-linterp", eq=eq),
                              dtype=float)
        out = dict(geom=geom, redl=redl, redl_I_BS=float(d["I_BS"]), li=li,
                   q_row=q_row, Ip=Ip, achieved=achieved)
        if want_chords and self.chords is not None:
            from .mse import mse_field_at
            # (B, found): an off-mesh chord is REPORTED (NaN, found=False),
            # never read from the interpolator's stale buffer
            B, found = mse_field_at(mygs, self.chords["R"], self.chords["Z"])
            out["B_chords"] = np.asarray(B, dtype=float).reshape(-1, 3)
            out["chords_found"] = np.asarray(found, dtype=bool).ravel()
            out["axis"] = tuple(float(v) for v in np.asarray(
                mygs.o_point, dtype=float).ravel()[:2])
        if final:
            stats = mygs.get_stats(lcfs_pad=pad, li_normalization="iter")
            out["stats"] = {k: float(v) for k, v in stats.items()
                            if np.isscalar(v) and np.isfinite(float(v))}
            # both frames of beta / W_MHD: the solver's own (its pressure
            # is zero at the boundary) and with p_sep added back
            out["pressure_frames"] = pressure_frames(stats, self.p_sep())
            # the solver's uniform P' rescale of this state (recorded)
            from .edge_pressure import solver_p_scale
            out["p_scale"] = solver_p_scale(mygs)
            out["li_1"] = float(li_achieved(eq, li_kind="li_1",
                                            psi_pad=pad)[0])
        return out


# ---------------------------------------------------------------------------
#  archive (an ADDED _baseline attribute; v3 readers unaffected)
# ---------------------------------------------------------------------------
#: JSON longer than this is stored as a string DATASET named
#: :data:`ENGINE_ATTR` (HDF5 caps an object header, i.e. all attributes of a
#: group together, at 64 KiB); the attribute then carries only a pointer.
ENGINE_ATTR_MAX_BYTES = 60000
#: The pointer the attribute carries when the record is a dataset.
ENGINE_DATASET_POINTER = '{"stored_as": "dataset"}'


def write_engine_json(grp, record) -> None:
    """Write an engine record onto an h5 group as JSON: the attribute
    :data:`ENGINE_ATTR` when it fits (:data:`ENGINE_ATTR_MAX_BYTES`), else a
    string dataset of the same name with :data:`ENGINE_DATASET_POINTER` in
    the attribute.  ``None`` writes nothing."""
    if record is None:
        return
    import json
    import h5py
    from .jbs_loop import jsonable
    txt = json.dumps(jsonable(record), allow_nan=True)
    if ENGINE_ATTR in grp and isinstance(grp[ENGINE_ATTR], h5py.Dataset):
        del grp[ENGINE_ATTR]
    if len(txt.encode()) <= ENGINE_ATTR_MAX_BYTES:
        grp.attrs[ENGINE_ATTR] = txt
    else:
        grp.create_dataset(ENGINE_ATTR, data=txt,
                           dtype=h5py.string_dtype())
        grp.attrs[ENGINE_ATTR] = ENGINE_DATASET_POINTER


def read_engine_json(grp):
    """Inverse of :func:`write_engine_json`: the record, or ``None``."""
    import json
    raw = grp.attrs.get(ENGINE_ATTR)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    if str(raw) == ENGINE_DATASET_POINTER:
        v = grp[ENGINE_ATTR][()]
        raw = v.decode() if isinstance(v, bytes) else str(v)
    return json.loads(str(raw))


def store_baseline_engine(header, record, scan_key=None):
    """Write the engine record onto the archive's ``_baseline`` group
    (:func:`write_engine_json`: the JSON attribute :data:`ENGINE_ATTR`, or a
    dataset of that name when the record is too large for an attribute).
    ``None`` writes nothing; no-op without a ``_baseline`` group (the same
    contract as :func:`bouquet.utils.store_baseline_state`)."""
    if record is None:
        return
    import h5py
    from .utils import _baseline_group_path, _resolve_h5
    db = _resolve_h5(header)
    with h5py.File(db, "a") as hf:
        gp = _baseline_group_path(scan_key)
        if gp in hf:
            write_engine_json(hf[gp], record)


def load_baseline_engine(header, scan_key=None):
    """The engine record stored by :func:`store_baseline_engine`, or
    ``None`` (a legacy archive)."""
    import h5py
    from .utils import _baseline_group_path, _resolve_h5
    db = _resolve_h5(header)
    with h5py.File(db, "r") as hf:
        gp = _baseline_group_path(scan_key)
        if gp not in hf:
            return None
        return read_engine_json(hf[gp])


# ---------------------------------------------------------------------------
#  the reconstruction, end to end (backend-agnostic)
# ---------------------------------------------------------------------------
def reconstruct(adapter, backend, settings, *, label=""):
    """Anchor -> finalize the contract -> engine -> result + record.

    The adapter's ``read()`` must have run (``adapter._c``); the backend
    solves the anchor request (two passes, the legacy anchor), measures it,
    the adapter finalizes the contract on it, and the engine runs.  Returns
    ``(engine, result, record)``."""
    t0 = time.perf_counter()
    c0 = adapter._c
    backend.solve(c0.anchor_request, n_passes=2)
    anchor = backend.measure(want_chords=False)
    ag = complete_geometry(anchor["geom"])
    contract = adapter.finalize(anchor["redl"], ag)
    eng = UnifiedEngine(contract, backend, settings, anchor=anchor,
                        label=label)
    res = eng.run()
    rec = engine_record(eng, res, wall_s=float(time.perf_counter() - t0))
    return eng, res, rec


def edge_pressure_record(eng) -> dict:
    """The engine record's ``edge_pressure`` block: the two settings, the
    contract pressure's ``p_sep`` / axis value / offset applied / axis
    target (:func:`bouquet.edge_pressure.describe`), both pressure frames
    of the delivered equilibrium (``None`` for a backend that reports none)
    and the solver's uniform ``P'`` rescale of it, ``p_scale``
    (:data:`bouquet.edge_pressure.P_SCALE_DEFINITION`; ``None`` for a
    backend that reports none)."""
    from .edge_pressure import describe
    out = describe(eng.s.get("edge_pressure"), eng.c.pressure)
    m = getattr(eng, "delivered_meas", None) or {}
    out["frames"] = m.get("pressure_frames")
    out["p_scale"] = m.get("p_scale")
    return out


def engine_record(eng, res, wall_s=None) -> dict:
    """The full JSON-safe engine record."""
    from .jbs_loop import jsonable, oft_build_info
    return jsonable(dict(
        version=ENGINE_VERSION,
        edge_pressure=edge_pressure_record(eng),
        contract=eng.c.record(),
        settings=dict(preset_requested=eng.s["preset"],
                      preset_in_force=eng.preset, rows_requested=list(
                          eng.s["rows"]), rows_active=sorted(eng.rows),
                      solver=("soft (close_ip_structured_soft)" if eng.soft
                              else "hard (close_ip_structured)"),
                      prior=eng.prior_name,
                      delivery_correction=eng.s["delivery_correction"],
                      mse_jacobian=eng.s["mse_jacobian"],
                      li_row_relaxation=eng.s.get("li_row_relaxation", 1.0),
                      ids_inductive=eng.s.get("ids_inductive", "auto"),
                      edge_pressure=eng.s.get("edge_pressure"),
                      edge_taper=eng.s.get("edge_taper"),
                      bootstrap_eps=eps_record(eng.s.get("eps_definition")),
                      loop=eng.s["loop"]),
        convergence=convergence_table(eng.s),
        composition=("J = F<1/R>/<B^2> [s_ind <j.B>_ind + s_bs <j.B>_BS + "
                     "<j.B>_fix] + p'(<R> - F^2<1/R>/<B^2>), on the latest "
                     "solved geometry (identity I2)"),
        row_update=("l_i: d_k = (1-omega) d_k-1 + omega [l_i(E_k+1) - "
                    "l_i_model(solved current; G_k+1)], closure target "
                    "T - d; q0: jbs_loop.AxisRowPin; MSE: offset refreshed "
                    "from each solve, Jacobian: "
                    + mse_scheme_text(eng.s["mse_jacobian"])
                    + ("" if eng.s.get("li_row_relaxation", 1.0) == 1.0 else
                       f"; the l_i step is under-relaxed: omega -> "
                       f"{eng.s['li_row_relaxation']:g} omega "
                       "(engine_li_row_relaxation; the first update from "
                       f"d = 0 takes {eng.s['li_row_relaxation']:g})")),
        notices=list(eng.notices),
        flags=list(eng.flags),
        mse=dict(eng.mse_summary),
        converged=bool(res["converged"]),
        loop_converged=bool(res["loop_converged"]),
        phases=res["phases"], passes=eng.passes,
        delivered=res["delivered"], state=res["state"].record(),
        solves=dict(eng.solves), wall_s=wall_s,
        oft_build=oft_build_info()))


# ---------------------------------------------------------------------------
#  wiring: Bouquet.prepare_baseline() with reconstruction_engine="unified"
# ---------------------------------------------------------------------------
#: What the stored current split IS on an engine baseline
#: (``Baseline.delivered_state["convention"]``).
ENGINE_SPLIT_CONVENTION = (
    "unified engine: jphi-linterp REQUEST of the delivery solve (one "
    "jphi-linterp solve of j_phi reproduces the delivered equilibrium); "
    "j_BS = s_bs F<1/R>/<B^2> <j.B>_BS*, j_pressure = p'(<R> - "
    "F^2<1/R>/<B^2>) (the pressure-driven current, its own bucket: owner "
    "decision D2), j_NBI/j_RF = F<1/R>/<B^2> <j.B>_fix (j_RF: the rf part "
    "plus any other driven source entry) on the delivery composition's "
    "geometry, all times its edge taper; j_inductive the residual (it "
    "carries s_ind F<1/R>/<B^2> <j.B>_ind and any delivery correction)")


def split_pressure_term(geom):
    """The pressure-driven current of the archived split, its own bucket
    ``j_pressure`` (owner decision D2; never part of ``j_BS``): p'G times
    the geometry's edge taper."""
    P = pressure_term(geom)
    w = geom.get("edge_taper")
    return P if w is None else P * np.asarray(w, dtype=float)


def _lcfs_deviation_mm(mygs, pts):
    """``(rms, max)`` [mm] nearest-neighbour distance from *pts* to the
    solved LCFS -- the legacy reconstruction's own measure (its step 9:
    tricontour of psi at the LCFS level; of its curves, the innermost one
    that goes around the magnetic axis, :func:`~bouquet.utils.
    select_closed_lcfs`).  A measurement only: no solve, no filter and no
    acceptance decision reads the contour selected here."""
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from .utils import magnetic_axis_of, select_closed_lcfs
    psi_arr = mygs.get_psi(False)
    lev = float(mygs.psi_bounds[0])
    fig, ax = plt.subplots(1, 1)
    try:
        cs = ax.tricontour(mygs.r[:, 0], mygs.r[:, 1], mygs.lc, psi_arr,
                           levels=[lev])
        segs = [v for seg in cs.allsegs for v in seg if len(v) > 4]
    finally:
        plt.close(fig)
    pts_l = select_closed_lcfs(segs, context="engine reconstruction metrics",
                               axis=magnetic_axis_of(mygs))
    if pts_l is None:
        return float("nan"), float("nan")
    d, _ = cKDTree(pts_l).query(np.asarray(pts, dtype=float))
    return float(np.sqrt(np.mean(d ** 2)) * 1e3), float(np.max(d) * 1e3)


def _split(eng, res):
    """The Baseline's toroidal split of the delivered request:
    ``(R, j_inductive, j_BS, j_NBI, j_RF, j_pressure)`` with ``R = j_inductive
    + j_BS + j_NBI + j_RF + j_pressure`` exactly."""
    st, c = res["state"], eng.c
    g = st.geom
    kap = composed_factor(g)
    out = eng.delivered_closure["out"]
    R = np.asarray(st.request, dtype=float)
    # the pressure-driven p'G (docs/current-conventions.md, A7) is its own
    # bucket, j_pressure (owner decision D2): never part of j_BS, so the
    # bootstrap is the field-aligned s_bs kappa lambda alone (as
    # evaluate_jBS/4's toroidal output and the IMAS reader's j_BS)
    j_BS = np.asarray(out["s_bs"], float) * kap * np.asarray(st.lambda_bs)
    P = np.asarray(split_pressure_term(g), dtype=float)
    j_NBI = kap * np.asarray(c.jB_fix_parts["nbi"], float)
    # j_RF carries the RF part AND any other driven core_sources entry (the
    # IDS adapter's "other" part; absent on the g-file path), so the split
    # sums exactly to the request
    j_RF = kap * (np.asarray(c.jB_fix_parts["rf"], float)
                  + np.asarray(c.jB_fix_parts.get("other", 0.0), float))
    return R, R - j_BS - j_NBI - j_RF - P, j_BS, j_NBI, j_RF, P


def _delivered_state(eng, res, rec, path):
    from .physics import SOLVER_Q0_PSI_N
    m = eng.delivered_meas
    stats = m.get("stats") or {}
    A = np.asarray(m["achieved"], dtype=float)
    cfac = float(rec["delivered"]["delivery"]["c"])
    R = np.asarray(res["state"].request, dtype=float)
    return dict(
        convention=ENGINE_SPLIT_CONVENTION, path=path,
        l_i=float(m["li"]), l_i_scale="iter(li3)",
        q0=float(stats.get("q_0", float("nan"))),
        # get_stats' q0 is q at psi_N = SOLVER_Q0_PSI_N, not on axis; the
        # q0 ROW is measured and targeted at its own radius (psi_q[0])
        q0_psi_N=float(SOLVER_Q0_PSI_N),
        q0_row_psi_N=(None if "q0" not in eng.rows
                      else float(eng.rows["q0"]["psi"])),
        q95=float(stats.get("q_95", float("nan"))),
        Ip_target=float(eng.c.Ip), request_normalisation=cfac,
        edge_pressure=edge_pressure_record(eng),
        achieved_normalisation=None, n_floored_inductive=0,
        j_phi_achieved=A,
        how=("unified engine delivery solve: the unrelaxed composition on "
             "the last solved geometry with x*, lambda_BS* and the row "
             "discrepancies, solved twice; one jphi-linterp solve of j_phi "
             "reproduces it")), R - A / cfac


def delivered_loop_record(rec) -> dict:
    """The loop record of the stage whose iterate was DELIVERED: the last
    phase that has one, skipping an MSE phase that was not applied (too few
    chords on the mesh: no record) or that FAILED and fell back to the
    reconstruction without MSE (``jacobian["failed"]``)."""
    for ph in reversed(rec["phases"]):
        jac = ph.get("jacobian") or {}
        if ph.get("record") is None or jac.get("failed") \
                or ph.get("superseded_by_fallback"):
            continue
        return ph["record"]
    raise ValueError("engine record: no delivered loop record")


def _flag_reason(res, rec):
    from .jbs_loop import flag_reason
    if not res["loop_converged"]:
        return flag_reason(delivered_loop_record(rec))
    return ("unified engine: the delivered equilibrium fails "
            + "; ".join(rec["delivered"]["misses"]))


def engine_closure_health(eng, where):
    """:func:`bouquet.utils.closure_health` of the engine's DELIVERED
    closure, on either input type: the raw-component Ip mismatch, the
    bootstrap scale against its +/-50 % prior (``|s_bs - 1| > 0.5`` ->
    :data:`~bouquet.utils.BOOTSTRAP_PRIOR_FLAG`, printed and warned; a
    closure failure, never clamped) and, on a soft Ip row, the Ip residual.
    The scale judged is the closure's EFFECTIVE bootstrap scale
    (``bs_scale_eff``: the Ip-weighted mean of ``s_bs(psi)`` on a
    structured preset, the scalar itself otherwise) -- recorded as
    ``bs_scale_basis`` with the profile's range ``s_bs_range``."""
    from .utils import BS_SCALE_PRIOR_HALFWIDTH, closure_health
    cl = eng.delivered_closure
    out = cl["out"]
    ch = closure_health(
        out["ohm_scale_eff"], out["bs_scale_eff"], cl["Ip_signed"],
        cl["c_signed"], out["Ip_lin_ind"], out["Ip_lin_bs"],
        out["Ip_lin_fix"],
        soft_ip_residual_sigma=out.get("residual_sigma_Ip"), where=where)
    sb = out.get("s_bs")
    ch["bs_scale"] = float(out["bs_scale_eff"])
    ch["bs_scale_basis"] = (
        "bs_scale_eff: the closure's effective bootstrap scale (the "
        "Ip-weighted mean of s_bs(psi) on a structured preset; the scalar "
        "itself otherwise)")
    ch["s_bs_range"] = (None if sb is None else
                        [float(np.min(sb)), float(np.max(sb))])
    ch["closure_limited_thresholds"] = dict(
        ch["closure_limited_thresholds"],
        bs_prior_halfwidth=float(BS_SCALE_PRIOR_HALFWIDTH),
        bs_scale_max=1.0 + float(BS_SCALE_PRIOR_HALFWIDTH))
    return ch


def _gfile_baseline(bq, eng, res, rec, ad, iso_pts, iso_w):
    from .baseline import Baseline, _reconstruction_metrics
    mygs, src, c = bq.mygs, bq.config.source, eng.c
    eqdsk = ad.eqdsk
    R, j_ind, j_BS, j_NBI, j_RF, P = _split(eng, res)
    ds, offset = _delivered_state(eng, res, rec, "reconstruction")
    m = eng.delivered_meas
    A = np.asarray(m["achieved"], dtype=float)
    jt_in = np.abs(np.asarray(eqdsk.j_tor_averaged_direct, dtype=float))
    psi = np.asarray(eqdsk.psi_N, dtype=float)
    bnd_rms, bnd_max = _lcfs_deviation_mm(mygs, iso_pts)
    core, edge = psi < 0.8, psi > 0.9
    quality = dict(
        jphi_core_rms=float(np.sqrt(np.mean((A[core] - jt_in[core]) ** 2))),
        jphi_edge_rms=float(np.sqrt(np.mean((A[edge] - jt_in[edge]) ** 2))),
        li_scale="iter(li3)", boundary_rms_mm=bnd_rms,
        boundary_max_dev_mm=bnd_max,
        Ip_error_pct=float(100.0 * abs(m["Ip"] - c.Ip) / c.Ip))
    _, F_prof, Fp_prof, _, _ = mygs.get_profiles(psi=psi)
    recon = dict(
        ne=c.kinetics["ne"], te=c.kinetics["te"], ni=c.kinetics["ni"],
        ti=c.kinetics["ti"], Zeff=c.kinetics["zeff"],
        isoflux_pts=np.asarray(iso_pts, float).copy(),
        weights=np.asarray(iso_w, float).copy(),
        psi_lcfs_val=float(mygs.psi_bounds[0]),
        j_inductive_fit=j_ind.copy(), j_phi_fit=A.copy(),
        j_BS_used=j_BS.copy(), psi=mygs.get_psi(False),
        ffprime=np.asarray(F_prof * Fp_prof, dtype=float),
        Ip_tokamaker=float(m["Ip"]), eqdsk_jtor=jt_in,
        eqdsk_psi_N=psi.copy(), eqdsk_pres=np.asarray(eqdsk.pres).copy(),
        eqdsk_boundary_R=np.asarray(eqdsk.boundary_R).copy(),
        eqdsk_boundary_Z=np.asarray(eqdsk.boundary_Z).copy(),
        eqdsk_ffprim=np.asarray(eqdsk.ffprim).copy(),
        eqdsk_li=dict(eqdsk.li), eqdsk_Ip=eqdsk.Ip,
        pres_tokamaker=np.asarray(c.pressure).copy(), psi_N_grid=psi.copy(),
        li_final=float(m["li"]), li_realized_post_corrective=float(m["li"]),
        quality=quality, request_jphi=R.copy(), engine=True)
    metrics = _reconstruction_metrics(mygs, eqdsk, recon, src,
                                      float(m["li"]),
                                      l_i_realized_post_corrective=float(
                                          m["li"]),
                                      edge_pressure=eng.s.get(
                                          "edge_pressure"))
    loop_rec = dict(delivered_loop_record(rec))
    loop_rec["converged"] = bool(res["converged"])
    if not res["converged"]:
        loop_rec["stop_reason"] = _flag_reason(res, rec)
    metrics["jbs_loop"] = loop_rec
    metrics["engine"] = dict(version=ENGINE_VERSION,
                             converged=bool(res["converged"]),
                             solves=dict(rec["solves"]),
                             flags=list(rec.get("flags", ())))
    if rec.get("flags"):
        metrics["closure_limited"] = True
        metrics["closure_limited_reasons"] = tuple(
            list(metrics.get("closure_limited_reasons", ()) or ())
            + list(rec["flags"]))
    # the closure's health on the delivered closure, recorded as the IDS
    # path records it on ip_closure.  Only the +/-50 % bootstrap prior is
    # folded into the baseline's closure_limited here: the raw-component
    # Ip mismatch and the soft-Ip residual are recorded (with
    # closure_health's own verdict) but were never a g-file flag, and
    # making them one is not this change
    from .utils import BOOTSTRAP_PRIOR_FLAG, merge_closure_flags
    ch = engine_closure_health(eng, "engine g-file reconstruction")
    ch["folded_into_baseline_flags"] = [
        r for r in ch["closure_limited_reasons"]
        if str(r).startswith(BOOTSTRAP_PRIOR_FLAG)]
    metrics["closure_health"] = ch
    merge_closure_flags(metrics, dict(
        closure_limited_reasons=ch["folded_into_baseline_flags"]))
    with open(src.geqdsk_path, "rb") as fh:
        eqdsk_bytes = fh.read()
    kn = c.kinetics_native
    bl = Baseline(
        # the run grid (the g-file's psi_N, or its Phi_N in a toroidal-flux
        # run); recon keeps the g-file's own psi_N
        psi_N=np.asarray(c.psi_N, dtype=float), coord=ad.coord,
        j_phi=R, j_inductive=j_ind, j_BS=j_BS,
        psi_N_kinetic=np.asarray(kn["psi_N"], float), ne=kn["ne"],
        te=kn["te"], ni=kn["ni"], ti=kn["ti"], Zeff=kn["Zeff"],
        Ip_target=float(c.Ip), l_i_target=float(m["li"]),
        provenance="reconstruction", j_NBI=j_NBI, j_RF=j_RF,
        p_fast=kn.get("p_fast"), Z_imp=c.pressure_parts.get("Z_imp"),
        eqdsk_bytes=eqdsk_bytes, pfile_bytes=kn.get("raw_bytes"),
        aux={"zeff": np.asarray(kn["Zeff"], dtype=float)}, recon=recon,
        reconstruction_metrics=metrics, jphi_request_offset=offset,
        delivered_state=ds, engine=rec,
        edge_pressure=rec.get("edge_pressure"),
        # PR #56 (owner item E2 stamp): how the IDA Z_eff / n_i were
        # resolved; archived as _baseline li_metrics_json (as on the legacy
        # reconstruction route)
        li_metrics=({"zeff_provenance": dict(kn["zeff_provenance"])}
                    if kn.get("zeff_provenance") else None))
    # the third bucket (owner decision D2), archived beside the split
    from .schema import SPLIT_PRESSURE_SEPARATE
    bl.j_pressure = P
    bl.current_split_convention = SPLIT_PRESSURE_SEPARATE
    return bl


def _ids_baseline(bq, eng, res, rec, bl_src):
    import copy
    c = eng.c
    bl = copy.copy(bl_src)
    R, j_ind, j_BS, j_NBI, j_RF, P = _split(eng, res)
    ds, offset = _delivered_state(eng, res, rec, "imas")
    out = eng.delivered_closure["out"]
    cl = eng.delivered_closure
    m = eng.delivered_meas
    ch = engine_closure_health(eng, "engine IDS reconstruction")
    icl = dict(ch)
    icl.update(engine=True, closure=_closure_record(cl),
               jbs_loop=delivered_loop_record(rec),
               jbs_converged=bool(res["converged"]))
    extra = list(rec.get("flags", ()))
    if not res["converged"]:
        extra.append(_flag_reason(res, rec))
    if extra:
        icl["closure_limited"] = True
        icl["closure_limited_reasons"] = tuple(
            list(icl.get("closure_limited_reasons", ())) + extra)
    lim = dict(bl_src.li_metrics or {})
    lim.update(tokamaker_li_3=float(m["li"]),
               tokamaker_li_1=m.get("li_1"), engine=True,
               li3_radius=c.provenance.get("li3_radius"),
               bootstrap_prior=dict(
                   bs_scale=ch["bs_scale"],
                   halfwidth=ch["closure_limited_thresholds"][
                       "bs_prior_halfwidth"],
                   out_of_prior=any(
                       str(r).startswith("bootstrap_scale_out_of_prior")
                       for r in ch["closure_limited_reasons"])))
    bl.j_phi, bl.j_inductive, bl.j_BS = R, j_ind, j_BS
    bl.j_NBI, bl.j_RF = j_NBI, j_RF
    # the third bucket (owner decision D2) on the engine's own delivery
    # geometry, replacing the reader's (whose inductive carried it)
    from .schema import SPLIT_PRESSURE_SEPARATE
    bl.j_pressure = P
    bl.current_split_convention = SPLIT_PRESSURE_SEPARATE
    # the engine's j_RF carries every other driven entry (sawteeth included):
    # the reader's j_other / j_sawteeth are not separate channels here
    bl.j_other, bl.j_sawteeth = None, None
    bl.jBS_diff, bl.jphi_diff, bl.p_diff = None, None, None
    bl.bs_scale = float(out["bs_scale_eff"])
    bl.ohm_scale = float(out["ohm_scale_eff"])
    bl.l_i_target = float(m["li"])
    bl.ip_closure = icl
    bl.li_metrics = lim
    bl.delivered_state = ds
    bl.jphi_request_offset = offset
    bl.engine = rec
    bl.edge_pressure = rec.get("edge_pressure")
    return bl


#: Per-chord MSE arrays of the engine record, moved OUT of the JSON
#: (``Baseline.engine``) into ``Baseline.mse_record`` -- archived as datasets
#: under ``_baseline/structured_mse`` -- with the key they went to left in
#: their place: ``(path in the record, mse_record key)``.
ENGINE_MSE_ARRAYS = (
    (("delivered", "checks", "mse", "tgamma"), "engine_mse_tgamma"),
    (("delivered", "checks", "mse", "residual_sigma"),
     "engine_mse_residual_sigma"),
    (("state", "mse_J"), "engine_mse_J"),
    (("state", "mse_x0"), "engine_mse_x0"),
    (("state", "mse_tg0"), "engine_mse_tg0"),
)


def mse_out(bl) -> dict:
    """Move the engine record's per-chord MSE arrays and Jacobians
    (:data:`ENGINE_MSE_ARRAYS`, plus each MSE phase's ``J_initial`` /
    ``tg_base``) from ``bl.engine`` to ``bl.mse_record`` (addendum item 2 of
    the Stage 2 report: O(n_chords) data are datasets, not JSON).  Each
    moved entry is replaced by the string ``"mse_record[<key>]"``.  Returns
    the moved arrays (empty without MSE)."""
    rec = getattr(bl, "engine", None)
    if not rec:
        return {}
    moved = {}

    def _take(node, path, key):
        for p in path[:-1]:
            node = node.get(p) if isinstance(node, dict) else None
            if node is None:
                return
        if not isinstance(node, dict) or node.get(path[-1]) is None:
            return
        moved[key] = np.asarray(node[path[-1]], dtype=float)
        node[path[-1]] = f"mse_record[{key}]"

    for path, key in ENGINE_MSE_ARRAYS:
        _take(rec, path, key)
    for i, ph in enumerate(rec.get("phases") or ()):
        jac = ph.get("jacobian") if isinstance(ph, dict) else None
        if isinstance(jac, dict):
            _take(ph, ("jacobian", "J_initial"),
                  f"engine_mse_phase{i}_J_initial")
            _take(ph, ("jacobian", "tg_base"),
                  f"engine_mse_phase{i}_tg_base")
    if moved:
        mr = dict(getattr(bl, "mse_record", None) or {})
        mr.update(moved)
        bl.mse_record = mr
    return moved


def reconstruction_coil_reg(mygs) -> dict:
    """The coil regularisation installed on *mygs* now -- the one the
    engine reconstruction about to run solves under: ``terms`` (the solver's
    term objects, ``None`` when not known) and ``record`` (JSON-safe:
    ``source`` and, when known, the ``terms`` as ``coils`` / ``target`` /
    ``weight``).

    * ``"strong (left installed by generate())"`` -- a previous
      ``generate()`` left its strong regularisation installed
      (``_strong_coil_reg``; ``Bouquet._reset_solver_state`` clears it);
    * ``"configured"`` / ``"default"`` -- what :meth:`bouquet.run.Bouquet.
      _apply_coil_reg` installed (``SolverConfig.coil_reg``'s terms at
      their configured weights, or the toward-zero default);
    * ``"unknown"`` -- a solver object ``setup_solver`` did not prepare
      (a test stand-in): nothing is on record.

    A draw is the reconstruction's closure perturbed in its inputs, not
    re-regularised: :meth:`bouquet.engine_draws.GenerateEngineDraws.draw`
    installs exactly ``terms`` for every draw, the zero-perturbation draw
    included."""
    strong = getattr(mygs, "_strong_coil_reg", None)
    if strong is not None:
        return dict(terms=list(strong), record=dict(
            source="strong (left installed by generate())", terms=None))
    terms = getattr(mygs, "_recon_coil_reg", None)
    rec = getattr(mygs, "_recon_coil_reg_record", None)
    if terms is None or rec is None:
        return dict(terms=None, record=dict(source="unknown", terms=None))
    return dict(terms=list(terms), record=dict(
        source=str(rec["source"]),
        terms=[dict(t) for t in rec["terms"]]))


def prepare_engine_baseline(bq):
    """``Bouquet.prepare_baseline()`` under ``reconstruction_engine=
    "unified"``, for both input types.

    Sets up the solver exactly as the legacy path of the same input does
    (g-file: isoflux on the g-file boundary at weight 200 and ``init_psi``
    from its LCFS shape; IDS: the slice's boundary re-pointed and
    ``init_psi`` from its shape), runs :func:`reconstruct` and returns the
    :class:`~bouquet.baseline.Baseline` the rest of the package consumes,
    with ``Baseline.engine`` carrying the full engine record.  A failure
    leaves no baseline (the half-built one is on
    ``Bouquet._failed_baseline``), as the legacy path."""
    from .adapters import GFileAdapter, IdsAdapter
    from .config import ImasSource, ReconstructionSource
    from .utils import _shape_from_boundary, capture_native_output
    cfg, gc, src, mygs = bq.config, bq.config.generation, bq.config.source, \
        bq.mygs
    if mygs is None:
        raise ValueError("the unified engine needs a live TokaMaker solver; "
                         "call setup_solver() before prepare_baseline()")
    s = engine_settings(gc)
    if (isinstance(src, ReconstructionSource)
            and s["ids_inductive"] != ENGINE_FIELD_DEFAULTS[
                "engine_ids_inductive"]):
        raise ValueError(
            f"generation.engine_ids_inductive={s['ids_inductive']!r} set "
            "with a g-file source: it configures the IDS adapter only and "
            "would have no effect; leave it at its default "
            f"{ENGINE_FIELD_DEFAULTS['engine_ids_inductive']!r}")
    if (isinstance(src, ReconstructionSource)
            and s["li3_radius"] != ENGINE_FIELD_DEFAULTS["imas_li3_radius"]):
        raise ValueError(
            f"generation.imas_li3_radius={s['li3_radius']!r} set with a "
            "g-file source: it configures the IDS l_i row only and would "
            "have no effect; leave it at its default "
            f"{ENGINE_FIELD_DEFAULTS['imas_li3_radius']!r}")
    if (isinstance(src, ImasSource)
            and not bool(getattr(src, "hold_sawteeth", True))):
        raise ValueError(
            "source.hold_sawteeth=False is a legacy-reader setting: the "
            "unified engine's IDS adapter always holds the sawteeth "
            "core_sources entry fixed as a driven current (identifier 701 "
            "-> 'other'), so it would be silently ignored here; leave it "
            "True, or run the legacy reader (solve_method='legacy')")
    bq.baseline = None
    bq._failed_baseline = None
    bq._engine_run = None
    verbose = bool(getattr(cfg, "verbose", False))
    bl = None
    try:
        with capture_native_output(enabled=not verbose) as cap:
            if isinstance(src, ReconstructionSource):
                ad = GFileAdapter(src, cfg)
                c0 = ad.read()
                psi_pad = float(src.psi_pad)
                iso_pts = np.asarray(c0.boundary, dtype=float)
                iso_w = np.ones(len(iso_pts)) * 200.0
                mygs.set_isoflux(iso_pts, weights=iso_w)
                geo = ad.eqdsk.geometry
                mygs.init_psi(geo["R"][-1], geo["Z"][-1], geo["a"][-1],
                              geo["kappa"][-1], geo["delta"][-1])
                bl_src = None
            elif isinstance(src, ImasSource):
                from .baseline import resolve_baseline
                bl_src = resolve_baseline(cfg, mygs)
                ad = IdsAdapter(src, cfg, bl_src,
                                inductive=s["ids_inductive"])
                c0 = ad.read()
                psi_pad = ad.psi_pad
                bq._repoint_imas_geometry()
                R0, Z0, a, kappa, delta = _shape_from_boundary(
                    bq._boundary_RZ)
                mygs.init_psi(R0, Z0, a, kappa, delta)
                bq._seed_coil_init(mygs)
            else:
                raise TypeError(f"unknown baseline source type "
                                f"{type(src).__name__}")
            mse = (c0.rows.get("mse") if "mse" in s["rows"] else None)
            backend = TokaMakerBackend(
                mygs, c0, psi_pad=psi_pad, li_kind="li_3",
                chords=(None if mse is None else mse["chords"]),
                edge_pressure=s["edge_pressure"],
                edge_taper=s["edge_taper"],
                coord=getattr(ad, "coord", "psi_n"),
                eps_definition=s.get("eps_definition"),
                # the reconstruction runs under the solver's own cap
                # (engine_draw_solve_maxits caps the DRAWS only)
                maxits=None)
            # the coil regularisation this reconstruction solves under
            # (recorded; every draw installs exactly it)
            coil_reg = reconstruction_coil_reg(mygs)
            eng, res, rec = reconstruct(ad, backend, s,
                                        label=f"engine {c0.kind}")
            rec["coil_reg"] = dict(coil_reg["record"])
            if c0.kind == "gfile":
                bl = _gfile_baseline(bq, eng, res, rec, ad, iso_pts, iso_w)
            else:
                bl = _ids_baseline(bq, eng, res, rec, bl_src)
            mse_out(bl)
        bl.reconstruction_log = (cap["text"] or None)
    except BaseException as exc:
        bq._failed_baseline = bl
        bq.baseline = None
        print(f"[baseline] FAILED ({type(exc).__name__}) in the unified "
              "engine: no usable baseline", flush=True)
        raise
    bq.baseline = bl
    # the LIVE reconstruction the draws inherit (bouquet.engine_draws): its
    # contract, state, basis and delivered measurement, in this session
    bq._engine_run = dict(engine=eng, result=res, psi_pad=float(psi_pad),
                          q_psi=getattr(backend, "q_psi", None),
                          baseline=bl, coil_reg=coil_reg)
    if bl.reconstruction_metrics is not None:
        bq._flag_nonconverged_recon_loop()
        bq._print_reconstruction_summary()
    elif not res["converged"]:
        # the IDS baseline records it on ip_closure / li_metrics (closure
        # health); warned here as loudly as _flag_nonconverged_recon_loop
        # does for a g-file baseline
        import warnings
        msg = ("engine IDS baseline: NOT converged (jbs_loop_on_fail="
               "'flag') -- delivered flagged closure_limited: "
               + _flag_reason(res, rec))
        print("[engine] WARNING " + msg, flush=True)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    print(f"[engine] {ENGINE_VERSION}: {'converged' if res['converged'] else 'NOT converged (flagged)'}; "
          f"l_i(3)={bl.l_i_target:.6f}; solves {rec['solves']}", flush=True)
    return bl
