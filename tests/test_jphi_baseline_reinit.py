"""Contract tests for the ``jphi_baseline`` psi re-init and its counted
fallback -- issue #24's THIRD converged-on-entry / masked-failure site.

Split out of the step-7 corrective-iteration package (#25, PR #27) because
this is the one piece of it that moves a repo golden: re-initialising psi
lands the baseline solve on a different member of the free-boundary coil
degeneracy, which shifts the mode-1 pinned max coil drift.  The golden was
regenerated for it under an explicit approval; see the PR and the commit that
carries the regenerated fixture.

Two things are pinned here, both solve-free:

* the fallback is COUNTED (``ANCHOR_MASKED_FAILURES["jphi_baseline"]``) and
  says what it reverts to, instead of printing one line and moving on; and
* the psi re-init happens, uses the landed LCFS, and precedes the solve it is
  meant to protect.

They cannot catch a numerical regression inside a GS solve -- the seeded runs
and the regenerated golden do that.  What they catch is that the guard is
still wired in at all.
"""
from __future__ import annotations

import ast

import pytest
import inspect
import re


def test_jphi_baseline_is_a_counted_masked_failure_site():
    from bouquet.TokaMaker_interface import (ANCHOR_MASKED_FAILURES,
                                             _count_masked_anchor_failure)

    assert "jphi_baseline" in ANCHOR_MASKED_FAILURES, (
        "the jphi-linterp baseline solve is #24's third masked site and must "
        "have its own counter key")
    before = ANCHOR_MASKED_FAILURES["jphi_baseline"]
    try:
        _count_masked_anchor_failure("jphi_baseline",
                                     RuntimeError('Exceeded "maxits"'))
        assert ANCHOR_MASKED_FAILURES["jphi_baseline"] == before + 1
    finally:
        ANCHOR_MASKED_FAILURES["jphi_baseline"] = before


def test_the_jphi_baseline_fallback_counts_and_names_what_reverts():
    """The old handler printed one line and moved on, so an archive could not
    say whether its per-draw diagnostics were referenced to the jphi-linterp
    baseline or to recon's inverse-mode LCFS -- references ~1.6 mm / ~0.9 % in
    l_i(3) apart.  Now it is counted, and the revert is spelled out."""
    from bouquet.TokaMaker_interface import generate_bouquet

    src = inspect.getsource(generate_bouquet)
    # Capture the ACTUAL handler block -- from its `except` clause to the
    # section that follows it -- rather than slicing a fixed-size window back
    # from the print.  A byte window silently changes what it covers whenever
    # the handler or its comments are edited, so it could fail on a still-valid
    # contract (or, worse, pass by picking up a neighbouring block).
    blk = re.search(
        r"except \(ValueError, RuntimeError\) as _bl_exc:"
        r"(?P<body>.*?)"
        r"# ---- Boundary-shift diagnostic",
        src, re.S)
    assert blk, (
        "could not locate the jphi-baseline `except ... as _bl_exc:` handler "
        "ahead of the Boundary-shift diagnostic section; the handler was "
        "renamed, removed, or moved out of the block this test anchors on")
    handler = blk.group("body")
    assert "[jphi-baseline] solve failed" in handler, \
        "the jphi-baseline failure handler is gone"
    assert '_count_masked_anchor_failure("jphi_baseline"' in handler, (
        "the jphi-baseline fallback is still an uncounted mask (issue #24)")
    assert "REVERTING" in handler, \
        "the fallback no longer says what it reverts to"


def test_the_jphi_baseline_solve_reinitialises_psi():
    """#24's converged-on-entry guard, same treatment as the sigma=0 anchor got
    in #22: re-init psi from the landed LCFS before the forward solve, and do
    not make a failed trace fatal."""
    from bouquet.TokaMaker_interface import generate_bouquet

    src = inspect.getsource(generate_bouquet)
    blk = re.search(r"if jphi_baseline:(.*?)# ---- Boundary-shift diagnostic",
                    src, re.S)
    assert blk, "could not locate the jphi_baseline block"
    body = blk.group(1)
    assert "init_psi(" in body, (
        "the jphi-linterp baseline solve no longer re-initialises psi; it is "
        "the third converged-on-entry site (issue #24)")
    assert "_shape_from_boundary" in body and "safe_trace_surf" in body
    # the re-init must precede the solve it protects
    assert body.index("init_psi(") < body.index("mygs.solve()"), \
        "psi is re-initialised AFTER the solve it is supposed to protect"


def test_the_failed_trace_warning_carries_the_exception_detail():
    """The non-fatal fallback must say WHY the trace was unusable.

    This WARN line is the only record a run keeps of a failed LCFS trace (the
    fallback is deliberately non-fatal), so dropping the exception detail makes
    a raised ``safe_trace_surf`` indistinguishable from a merely-too-short
    contour after the fact.
    """
    from bouquet.TokaMaker_interface import generate_bouquet

    src = inspect.getsource(generate_bouquet)
    blk = re.search(r"if jphi_baseline:(.*?)# ---- Boundary-shift diagnostic",
                    src, re.S)
    assert blk, "could not locate the jphi_baseline block"
    body = blk.group(1)

    # the exception must be bound, not swallowed anonymously
    assert re.search(r"except Exception as \w+:", body), (
        "the safe_trace_surf guard swallows its exception anonymously; bind it "
        "so the WARN line can report it")
    warn = re.search(r"could not trace an LCFS(.*?)\)\n", body, re.S)
    assert warn, "the failed-trace WARN line is gone"
    # the warning must interpolate something, not be a bare constant string
    assert "{" in warn.group(1), (
        "the failed-trace WARN line is a fixed string again; it must include "
        "the caught exception / the reason the contour was rejected")


def test_shape_from_boundary_is_shared_not_imported_from_the_orchestrator():
    """``TokaMaker_interface`` must not import the helper from ``bouquet.run``.

    ``run`` already imports from ``TokaMaker_interface``, so a runtime
    ``from .run import _shape_from_boundary`` closes a dependency cycle and
    couples a core library module to the orchestrator/CLI layer.  The helper
    lives in ``bouquet.utils``; ``run`` re-exports it for its original callers.
    """
    import bouquet.run as run
    import bouquet.utils as utils
    import bouquet.TokaMaker_interface as tmi

    src = inspect.getsource(tmi)
    assert "from .run import" not in src and "from bouquet.run import" not in src, (
        "TokaMaker_interface imports from bouquet.run -- circular-import risk; "
        "move the shared helper into bouquet.utils instead")

    assert hasattr(utils, "_shape_from_boundary"), \
        "the shared helper is not in bouquet.utils"
    # one function, three names -- the re-export must not be a second copy
    assert (utils._shape_from_boundary
            is run._shape_from_boundary
            is tmi._shape_from_boundary), (
        "_shape_from_boundary has been duplicated rather than re-exported; the "
        "copies will drift")


# ---------------------------------------------------------------------------
#  the psi re-init must not lose the state it discards
# ---------------------------------------------------------------------------
def _extract_jphi_baseline_block():
    """The REAL ``if jphi_baseline:`` body, dedented and ready to exec.

    Same technique as the kin->eq contract test: run the shipped source rather
    than a paraphrase of it, so the assertions below cannot drift away from the
    code they guard.
    """
    import textwrap
    from bouquet.TokaMaker_interface import generate_bouquet

    src = inspect.getsource(generate_bouquet)
    m = re.search(r"\n[ ]*if jphi_baseline:\n(?P<body>.*?)"
                  r"\n[ ]*# ---- Boundary-shift diagnostic", src, re.S)
    assert m, "could not locate the jphi_baseline block"
    return textwrap.dedent(m.group("body"))


class _JphiBaselineStub:
    """A solver that traces a usable LCFS and then fails its forward solve."""

    def __init__(self, fail=True, snapshots=True):
        self._fail = fail
        self.state = "recon-converged"
        self.init_psi_called = False
        self.replaced_with = None
        self._snap_id = 0
        if not snapshots:                       # emulate a minimal solver object
            del self.__class__.copy_eq          # (see _NoSnap below)

    def copy_eq(self):
        self._snap_id += 1
        return ("snap", self._snap_id, self.state)

    def replace_eq(self, source_eq=None):
        self.replaced_with = source_eq
        self.state = source_eq[2] if source_eq else self.state

    def init_psi(self, R0, Z0, a, kappa, delta):
        self.init_psi_called = True
        self.state = "cold-analytic"            # the recon state is now GONE

    def set_targets(self, Ip=None, pax=None):
        self.pax = pax

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        self.pp = None if pp_prof is None else pp_prof["y"]

    def solve(self):
        if self._fail:
            raise RuntimeError('Exceeded "maxits"')
        self.state = "baseline-converged"

    def get_globals(self):
        return (1.0e6,)

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"l_i": 0.65}

    def get_coil_currents(self):
        return ({"F1A": 1.0}, None)

    @property
    def psi_bounds(self):
        return (0.0, 1.0)


#: the edge-pressure settings the block is run at: the pre-change ones (the
#: inline expressions bit for bit) and the defaults since 2026-10-02
#: ("offset": the axis target is p_axis - p_sep)
EDGES = ("pre_change", "default")
_P_EDGE = 500.0                      # p_sep of the block's solve pressure


def _run_block(stub, capsys=None, edge="pre_change"):
    """Exec the shipped jphi_baseline block against ``stub``."""
    import numpy as np
    from bouquet.TokaMaker_interface import (_count_masked_anchor_failure,
                                             ANCHOR_MASKED_FAILURES)
    from bouquet.utils import _shape_from_boundary, pchip_derivative
    from bouquet import coords
    from bouquet.edge_pressure import (PRE_CHANGE_EDGE_PRESSURE,
                                       resolve_edge_pressure, solver_pax,
                                       solver_pp_profile)

    psi_N = np.linspace(0.0, 1.0, 65)
    th = np.linspace(0.0, 2.0 * np.pi, 128)
    lcfs = np.c_[1.7 + 0.6 * np.cos(th), 1.1 * 0.6 * np.sin(th)]

    ns = dict(
        mygs=stub, np=np, psi_N=psi_N, psi_pad=1e-3,
        safe_trace_surf=lambda g, v: lcfs,
        _shape_from_boundary=_shape_from_boundary,
        pchip_derivative=pchip_derivative,
        coords=coords, coord="psi_n",
        # the block builds its P' and axis target through the one helper
        # (bouquet.edge_pressure); at the pre-change settings it is the
        # inline expressions it replaced, bit for bit
        # (tests/test_edge_pressure.py)
        solver_pp_profile=solver_pp_profile, solver_pax=solver_pax,
        _edge=resolve_edge_pressure(PRE_CHANGE_EDGE_PRESSURE
                                    if edge == "pre_change" else None),
        initial_Ip_target=1.0e6,
        pressure_solve=1.0e4 * (1.0 - psi_N ** 2) + _P_EDGE,
        input_j_phi=1.0e6 * (1.0 - psi_N ** 2),
        jphi_diff=None,
        recon_lcfs_ref=None,
        _baseline_li3=0.65,
        _count_masked_anchor_failure=_count_masked_anchor_failure,
    )
    before = dict(ANCHOR_MASKED_FAILURES)
    try:
        exec(compile(_extract_jphi_baseline_block(),
                     "<jphi_baseline block>", "exec"), ns)
    finally:
        ANCHOR_MASKED_FAILURES.update(before)   # do not leak counts across tests
    return ns


@pytest.mark.parametrize("edge", EDGES)
def test_a_failed_baseline_solve_restores_the_state_the_reinit_discarded(
        edge):
    """FAILURE INJECTION: solve raises after init_psi -> state must be restored.

    ``init_psi`` throws away the reconstruction's converged equilibrium for a
    cold analytic psi.  The handler advertises "REVERTING to the recon
    (inverse) reference", and before the re-init existed that was literally
    true -- a failed solve left mygs untouched on recon's converged state,
    which is why issue #24 classed this fallback as benign.  Unguarded, the
    message would be printed while mygs sits on a COLD, non-converged psi,
    which then poisons the warm start of every subsequent draw.
    """
    stub = _JphiBaselineStub(fail=True)
    _run_block(stub, edge=edge)

    assert stub.init_psi_called, "the re-init did not run; test premise is void"
    assert stub.replaced_with is not None, (
        "the failed baseline solve did NOT restore the snapshot -- mygs is left "
        "on the cold psi init_psi installed, and every subsequent draw "
        "warm-starts from it")
    assert stub.state == "recon-converged", (
        f"state is {stub.state!r} after the fallback; the handler promises a "
        "revert to the recon reference and must actually deliver it")


def test_the_snapshot_is_taken_before_init_psi_discards_the_state():
    """A snapshot taken after init_psi would preserve the cold psi, not the
    recon state -- restoring it would be a no-op dressed as a guard."""
    src = _extract_jphi_baseline_block()
    tree = ast.parse(src)
    calls = [(n.func.attr, n.lineno) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    copy_lines = [ln for name, ln in calls if name == "copy_eq"]
    init_lines = [ln for name, ln in calls if name == "init_psi"]
    assert copy_lines, "no copy_eq snapshot before the psi re-init"
    assert init_lines, "the psi re-init is gone"
    assert min(copy_lines) < min(init_lines), (
        "the snapshot is taken AFTER init_psi has already discarded the "
        "reconstruction's converged state")


@pytest.mark.parametrize("edge", EDGES)
def test_a_successful_baseline_solve_does_not_restore_anything(edge):
    """Guard the guard: the restore must be on the FAILURE path only, or the
    baseline solve's own result would be thrown away."""
    stub = _JphiBaselineStub(fail=False)
    _run_block(stub, edge=edge)
    # the axis target the block hands the solver, at each setting
    p0 = 1.0e4 + _P_EDGE
    assert stub.pax == (p0 if edge == "pre_change" else p0 - _P_EDGE)
    assert stub.pp[-1] == 0.0                   # the pin, on at both

    assert stub.init_psi_called
    assert stub.replaced_with is None, (
        "a SUCCESSFUL baseline solve restored the pre-solve snapshot, "
        "discarding the very equilibrium it just computed")
    assert stub.state == "baseline-converged"
