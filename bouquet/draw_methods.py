"""The draw-method strategy: how one draw of :func:`generate_bouquet` is made.

``generate_bouquet`` calls ONE :class:`DrawMethod` wherever the solve methods
differ: :class:`DrawMethod` itself (legacy),
:class:`bouquet.engine_draws.GenerateEngineDraws` (engine) and
:class:`bouquet.swb_draws.SwbDraws` (swb).  See docs/draw-methods.md.
"""
from __future__ import annotations


class DrawMethod:
    """The legacy draws (``solve_method="legacy"``), and the base every other
    method overrides."""

    name = "legacy"
    #: Bouquet.setup_solver enters OFT's bounded (BVLS) coil solve for the
    #: method (bouquet.solver_state); class-level, read before any draw
    bounded_coil_mode = True
    strong_coil_reg = True
    pin_bounds_on_skip = True
    iso_update = True
    homotopy = True

    # ---- generate_bouquet -------------------------------------------------
    def validate(self, **settings):
        """Refuse settings this method cannot run (before any solve)."""

    def jphi_baseline(self, jphi_baseline):
        """Whether the baseline is re-solved from its jphi_total first."""
        return jphi_baseline

    #: generate_bouquet's q0 gate as applied (:meth:`note_q0_gate`): None
    #: until a generate_bouquet call has decided it
    q0_gate = None

    def note_q0_gate(self, record):
        """generate_bouquet's decision on the q0 >= 1 gate
        (``configured``, ``applied``, ``q0``, ``reason``;
        :data:`bouquet.filtering.Q0_GATE_RECORD_KEY`): kept so the scan's
        in-spec stamp records the rule AS APPLIED."""
        self.q0_gate = dict(record)

    def baseline_resolve_start(self, mygs):
        """Where that baseline re-solve starts.  ``None``: re-initialise psi
        from the landed LCFS first (the converged-on-entry guard of issue
        #24, for a reconstruction solved in another mode).  A string: the
        method has put *mygs* on its own start state and says which (printed);
        no psi re-init."""
        return None

    def homotopy_isoflux(self, mygs):
        """The isoflux the draw's homotopy holds the boundary with (the
        iso-update before the coil-bound passes).  ``None`` (legacy): re-point
        it to the draw's own traced LCFS, every traced point at weight 200.
        A dict: ``pts`` / ``w`` to install (``pts=None``: leave the isoflux
        in place) and ``stamp`` (n, weight, origin, sha256), which the method
        archives on the draw."""
        return None

    def solve_pressure(self, psi_N, pressure_solve):
        """The pressure every draw solve and the baseline archive use."""
        return pressure_solve

    def edge(self, edge):
        """The edge-pressure settings (bouquet.edge_pressure)."""
        return edge

    def draw(self, mygs, rng, scale, count, *, coil_guard=None, bnd_diag=None,
             solve_guard=None, inputs=None, legacy=None):
        """One draw: ``(ne, te, ni, ti, w_ExB, jphi_total, diagnostics)``."""
        return legacy()

    def rejection_reason(self, exc, stage):
        """The rejection code of a draw that raised *exc* at *stage*."""
        from .TokaMaker_interface import _draw_rejection_reason
        return _draw_rejection_reason(exc, stage)

    def annotate_rejection(self, record, exc):
        return record

    def drift_without_homotopy(self, mygs, baseline_coils, coil_drift, *,
                               ip_aligned=True, skip_hard=False):
        """``(drifts, pass_idx, F_lim, VSC_lim)`` when the draw's coil drift
        is measured instead of bounded by the homotopy; None: the homotopy."""
        return None

    def cap_solver(self, mygs):
        pass

    def uncap_solver(self, mygs):
        pass

    def hit_cap(self, exc):
        return False

    def announce_cap(self, *args, **kwargs):
        pass

    def last_homotopy_solve_seconds(self):
        return None

    def announce_rollback_failed(self, exc, after, legacy=None):
        return legacy()

    def post_homotopy_jbs(self, *args, legacy=None, **kwargs):
        """The post-homotopy bootstrap check: ``(record, spike, full,
        jphi_total)``."""
        return legacy()

    def post_homotopy_split(self, jphi_total, spike, full, legacy=None):
        """``(jphi_ind, jphi_bs, jphi_bs_edge)`` of a re-solved draw."""
        return legacy()

    def post_hoc(self, mygs, diagnostics, in_spec, **kwargs):
        """The archived draw's ``in_spec`` after the method's own filters."""
        return in_spec

    def lcfs_pressure(self, p_lcfs):
        """The separatrix pressure the draw's written g-file carries."""
        return p_lcfs

    def stored_pressures(self, thermal, total):
        return thermal, total

    def archived_split(self, diagnostics, jphi_total, default=None):
        """``(jphi_bs, jphi_ind)`` archived for the draw."""
        return default

    def store_draw(self, header, count, scan_key, diagnostics):
        """Method-specific attrs/blocks on the draw's archive group."""

    def until_n(self, ok, reasons, diagnostics, **kwargs):
        return ok, reasons

    def announce_coil_reg(self):
        """Said when the method installs no strong coil reg."""

    # ---- Bouquet.generate -------------------------------------------------
    def draw_env(self, env):
        """The uncertainty envelope the draws use (engine: the one it was
        built with, zeroed on the sigma=0 route)."""
        return env

    def scale_settings(self, jbs_range, bs_mult):
        """``(jbs_scale_range, jbs_scale_profile)`` handed to the draws."""
        return jbs_range, bs_mult

    def loop_settings_for(self, settings):
        return settings

    def solve_maxits(self, maxits):
        """The draw-loop GS iteration cap (DrawSolveGuard)."""
        return maxits

    def solve_retry_urf(self, retry_urf):
        """DrawSolveGuard's retry under-relaxation factors."""
        return retry_urf

    def solve_loose_tol(self, loose_tol):
        """DrawSolveGuard's loose-tolerance rescue (None: off)."""
        return loose_tol

    def achieved_jphi(self, store):
        """Whether each draw archives its achieved FSA jphi_total."""
        return store

    def draw_jbs_loop(self, settings):
        """The per-draw self-consistent loop settings (None: off)."""
        return settings

    def loop_codes(self, codes):
        """The rejection codes reported as loop-stage rejections."""
        return codes

    def store_baseline(self, header, scan_key, baseline):
        """Method-specific attrs/blocks on the archive's ``_baseline``."""

    def summarize(self, bq):
        """After generate_bouquet, outside the output capture."""

    # ---- class-level hooks (no run state): method_hooks(gc) -------------
    @classmethod
    def verify_sigma0(cls, bq):
        """The method's own sigma=0 check; None: the legacy one."""
        return None

    @classmethod
    def workflow_problems(cls, bq):
        """The method's own workflow check (a list of problems); None: the
        legacy one."""
        return None


def method_hooks(gc):
    """The draw-method CLASS of ``gc``'s solve method -- for the class-level
    hooks (:meth:`DrawMethod.verify_sigma0`, :meth:`DrawMethod.
    workflow_problems`) that need no run state."""
    from .config import resolve_solve_method
    sm = resolve_solve_method(gc)
    if sm == "engine":
        from .engine_draws import GenerateEngineDraws
        return GenerateEngineDraws
    if sm == "swb":
        from .swb_draws import SwbDraws
        return SwbDraws
    return DrawMethod
