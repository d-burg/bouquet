"""Contract tests for the step-7 corrective-iteration package (issue #25).

Three coordinated changes, all cheap to pin without a GS solve:

1. ``protect_state=True`` on the geqdsk-path ``_corrective_jphi_iteration``
   (keep-best on the FULL-domain j_phi RMS) -- the semantics the IMAS path has
   used since the corrector was reused there.  The loop's own stopping rule is
   on the EDGE RMS, so an iterate that improves the edge while degrading the
   core satisfies it and used to be kept.
2. ``l_i_target`` = the step-6 MATCHED value (``result['li_final']``) instead of
   a fresh ``get_stats`` read of whatever state step 7 happened to end on, with
   the post-corrective read retained as a separate archived diagnostic.
3. A report-style condition -- never a raise -- when step 7 moves l_i(3) off
   the matched value by more than the per-draw band.

A fourth change originally packaged here -- the ``jphi_baseline`` psi re-init
and its counted fallback (#24's third masked site) -- was SPLIT OUT into its
own PR, because it shifts a repo golden (the mode-1 pinned coil drift) and so
carries an approved golden regeneration with it.  Its contract tests moved with
it; nothing in this file depends on it.

These are deliberately solve-free.  They cannot catch a numerical regression
inside a GS solve -- the seeded before/after runs in the PR do that.  What they
catch is the plumbing, the state-protection semantics and the definitions,
which is what actually drifted.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pytest


# ---------------------------------------------------------------------------
#  1. keep-best on the full domain
# ---------------------------------------------------------------------------
class _KeepBestStub:
    """A solver whose successive iterates get BETTER then WORSE on the full
    domain while the EDGE keeps improving.

    That is the exact shape the corrective iteration mishandles: its stopping
    rule reads the edge, so it accepts the last iterate -- which is the worst
    one everywhere else.  Constructed to be non-monotone on purpose; a stub
    that improved monotonically could not tell keep-best from last-iterate.
    """

    def __init__(self, psi_N, target):
        self.psi_N = np.asarray(psi_N, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.eq_id = 0
        self.replaced_with = None
        self._outputs = []
        # full-domain error shrinks then grows; edge error shrinks throughout
        core = self.psi_N <= 0.9
        edge = ~core
        for core_amp, edge_amp in ((3.0e4, 3.0e4), (1.0e4, 2.0e4),
                                   (8.0e4, 1.0e4)):
            out = self.target.copy()
            out[core] += core_amp
            out[edge] += edge_amp
            self._outputs.append(out)
        self._it = 0

    # -- the corrective iteration's solver contract -------------------------
    def set_targets(self, Ip=None, pax=None):
        pass

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        pass

    def copy_eq(self):
        self.eq_id += 1
        return ("eq", self.eq_id, self._it)

    def replace_eq(self, source_eq=None):
        self.replaced_with = source_eq

    def solve(self):
        pass

    def get_profiles(self, npsi=None, psi_pad=None, psi=None):
        out = self._outputs[min(self._it, len(self._outputs) - 1)]
        self._it += 1
        # get_jphi_from_GS(f*fp, pp, <R>, <1/R>) must return `out`; the monkey-
        # patched helper below just hands the stashed profile straight back.
        self._pending = out
        return (None, np.zeros_like(out), np.zeros_like(out), None,
                np.zeros_like(out))

    def get_q(self, npsi=None, psi_pad=None, psi=None):
        n = self.psi_N.size
        ravgs = {"<R>": np.ones(n), "<1/R>": np.ones(n),
                 "<1/R^2>": np.ones(n), "dV/dPsi": np.ones(n)}
        return (None, None, ravgs, None, None, None)


@pytest.fixture()
def _patched_jphi_from_GS(monkeypatch):
    """Route ``get_jphi_from_GS`` back to the stub's pending output.

    ``_corrective_jphi_iteration`` imports it from
    ``OpenFUSIONToolkit.TokaMaker.util`` at call time.  Stub the module in when
    OFT is not installed so this stays a real behavioural test of the keep-best
    logic on every run of the suite, not one that quietly skips.
    """
    import sys
    import types

    holder = {}

    def _fake(ffp, pp, R_avg, inv_R):
        return holder["stub"]._pending

    try:
        import OpenFUSIONToolkit.TokaMaker.util as _u
        monkeypatch.setattr(_u, "get_jphi_from_GS", _fake, raising=False)
    except ImportError:
        pkg = types.ModuleType("OpenFUSIONToolkit")
        sub = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
        util = types.ModuleType("OpenFUSIONToolkit.TokaMaker.util")
        util.get_jphi_from_GS = _fake
        sub.util = util
        pkg.TokaMaker = sub
        for name, mod in (("OpenFUSIONToolkit", pkg),
                          ("OpenFUSIONToolkit.TokaMaker", sub),
                          ("OpenFUSIONToolkit.TokaMaker.util", util)):
            monkeypatch.setitem(sys.modules, name, mod)
    return holder


def test_protect_state_lands_on_the_best_full_domain_state(_patched_jphi_from_GS):
    """keep-best restores the best FULL-domain iterate, not the last one."""
    from bouquet.TokaMaker_interface import _corrective_jphi_iteration

    psi_N = np.linspace(0.0, 1.0, 101)
    target = 1.0e6 * (1.0 - psi_N ** 2)
    stub = _KeepBestStub(psi_N, target)
    _patched_jphi_from_GS["stub"] = stub

    out, n_iters, edge_hist = _corrective_jphi_iteration(
        stub, psi_N, target, {"type": "linterp", "y": psi_N, "x": psi_N},
        1.0e6, 1.0e4, 1e-3, min_iters=2, max_iters=3, rtol=0.0,
        verbose=False, protect_state=True)

    rms = lambda a: float(np.sqrt(np.mean((a - target) ** 2)))  # noqa: E731
    assert rms(out) == pytest.approx(rms(stub._outputs[1]), rel=1e-12), (
        "protect_state returned the last iterate, not the best full-domain one")
    assert stub.replaced_with is not None, \
        "protect_state did not restore an equilibrium at all"
    # the edge history is still the raw per-iterate sequence (unchanged contract)
    assert len(edge_hist) == n_iters


def test_without_protect_state_the_last_iterate_is_kept(_patched_jphi_from_GS):
    """Guard the guard: the stub must actually discriminate.

    If this passed too, the test above would be vacuous -- it would be
    asserting a property the default path already had.
    """
    from bouquet.TokaMaker_interface import _corrective_jphi_iteration

    psi_N = np.linspace(0.0, 1.0, 101)
    target = 1.0e6 * (1.0 - psi_N ** 2)
    stub = _KeepBestStub(psi_N, target)
    _patched_jphi_from_GS["stub"] = stub

    out, _n, _h = _corrective_jphi_iteration(
        stub, psi_N, target, {"type": "linterp", "y": psi_N, "x": psi_N},
        1.0e6, 1.0e4, 1e-3, min_iters=2, max_iters=3, rtol=0.0,
        verbose=False, protect_state=False)

    rms = lambda a: float(np.sqrt(np.mean((a - target) ** 2)))  # noqa: E731
    assert rms(out) == pytest.approx(rms(stub._outputs[-1]), rel=1e-12)
    assert stub.replaced_with is None


def _stub_type_without(method):
    """A ``_KeepBestStub`` clone that genuinely lacks ``method``.

    Built by re-typing the namespace rather than subclassing, because an
    inherited attribute cannot be deleted -- and ``hasattr`` must really
    return False for the gate under test to be exercised.
    """
    ns = {k: v for k, v in vars(_KeepBestStub).items()
          if k not in ("__dict__", "__weakref__", method)}
    return type(f"_KeepBestStubNo_{method}", (), ns)


@pytest.mark.parametrize("missing", ["copy_eq", "replace_eq"])
def test_keep_best_is_disabled_unless_BOTH_snapshot_methods_exist(
        _patched_jphi_from_GS, missing):
    """``protect_state`` must gate on copy_eq AND replace_eq, not just one.

    Every snapshot the keep-best path takes is eventually consumed by
    ``replace_eq`` (per-iteration solve-failure restore, and the final
    keep-best restore).  A copy_eq-only gate would let a solver object that
    lacks ``replace_eq`` through, and the restore would then raise
    AttributeError -- converting a recoverable state into a crash.  With
    either half missing the iteration must simply fall back to the
    last-iterate behaviour, exactly as protect_state=False does.
    """
    from bouquet.TokaMaker_interface import _corrective_jphi_iteration

    psi_N = np.linspace(0.0, 1.0, 101)
    target = 1.0e6 * (1.0 - psi_N ** 2)
    stub = _stub_type_without(missing)(psi_N, target)
    assert not hasattr(stub, missing), "the stub still exposes " + missing
    _patched_jphi_from_GS["stub"] = stub

    out, _n, _h = _corrective_jphi_iteration(
        stub, psi_N, target, {"type": "linterp", "y": psi_N, "x": psi_N},
        1.0e6, 1.0e4, 1e-3, min_iters=2, max_iters=3, rtol=0.0,
        verbose=False, protect_state=True)

    rms = lambda a: float(np.sqrt(np.mean((a - target) ** 2)))  # noqa: E731
    assert rms(out) == pytest.approx(rms(stub._outputs[-1]), rel=1e-12), (
        f"with {missing} absent, keep-best must be OFF and the last iterate "
        "returned; a partially-gated keep-best ran instead")
    if missing != "replace_eq":
        assert stub.replaced_with is None, (
            "keep-best attempted a restore despite the gate being unmet")


def test_the_geqdsk_call_site_asks_for_protect_state_and_keeps_its_knobs():
    """The geqdsk path was the one corrective call still trusting its last
    iterate.  Pin BOTH halves: state protection on, knob values untouched --
    the package explicitly does not retune the corrector.
    """
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    src = inspect.getsource(reconstruct_equilibrium)
    call = re.search(r"_corrective_jphi_iteration\((.*?)\n    \)", src,
                     re.S)
    assert call, "could not find the corrective-iteration call in step 7"
    body = call.group(1)
    assert "protect_state=True" in body, (
        "the geqdsk corrective iteration lost protect_state=True; it would go "
        "back to landing on whatever the last Newton step produced")
    for knob in ("min_iters=2", "max_iters=8", "rtol=0.05"):
        assert knob in body, (
            f"the corrective iteration's {knob} changed -- this package "
            f"deliberately changes NO knob values, only state protection")


# ---------------------------------------------------------------------------
#  2. + 3. the l_i target definition and the report
# ---------------------------------------------------------------------------
def test_l_i_target_is_the_step6_matched_value_not_a_post_step7_read():
    """``l_i_target`` must come from ``result['li_final']``.

    The old code re-read ``get_stats`` AFTER the reconstruction returned, i.e.
    after step 7 -- so the ensemble was banded around a value step 6 had not
    matched.  The post-corrective number is kept, under its own name, as a
    diagnostic; it is now taken from the value ``reconstruct_equilibrium``
    measured at step 7b rather than re-read here (same state, same result --
    verified bit-identical -- but pinned to the step it is named after).
    """
    import bouquet.baseline as bl

    src = inspect.getsource(bl._reconstruct_baseline
                            if hasattr(bl, "_reconstruct_baseline")
                            else bl)
    m = re.search(r"^\s*l_i_target = (.+)$", src, re.M)
    assert m, "could not find the l_i_target assignment in baseline.py"
    assert 'result["li_final"]' in m.group(1) or \
           "result['li_final']" in m.group(1), (
        f"l_i_target is assigned from {m.group(1).strip()!r}, not from the "
        f"step-6 matched value result['li_final'] (issue #25)")
    m2 = re.search(r"^\s*l_i_realized_post_corrective = (.+(?:\n\s+.+)?)$",
                   src, re.M)
    assert m2, (
        "the post-corrective diagnostic was dropped instead of being retained "
        "under its own name -- provenance must grow, not shrink")
    assign = m2.group(1)
    assert "li_realized_post_corrective" in assign and "result" in assign, (
        f"l_i_realized_post_corrective is assigned from {assign.strip()!r}; it "
        "must consume result['li_realized_post_corrective'] (the step-7b "
        "measurement) rather than re-reading solver state here, which is only "
        "'post step 7' for as long as nothing in between mutates it")
    assert "get_stats" not in assign, (
        "the redundant get_stats re-trace is back; it costs a second "
        "q-profile trace and can diverge in provenance from step 7b")


def test_reconstruct_equilibrium_returns_both_l_i_numbers():
    """``li_final`` (matched) and ``li_realized_post_corrective`` are distinct
    keys, so a consumer cannot silently pick up the wrong one."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    src = inspect.getsource(reconstruct_equilibrium)
    for key in ("'li_final'", "'li_realized_post_corrective'",
                "'li_corrective_drift_pct'"):
        assert key in src, f"the result dict lost {key}"


def test_the_post_corrective_l_i_condition_is_reported_not_raised():
    """It is a loud, archived condition.  Making it fatal would be a NEW
    acceptance criterion, which is not approved -- so no raise may appear in
    the block, and the out-of-band flag must be archived."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    src = inspect.getsource(reconstruct_equilibrium)
    block = re.search(r"# ---- 7b\..*?# ---- 8\. Final profiles", src, re.S)
    assert block, "the step-7b post-corrective l_i report is gone"
    body = block.group(0)
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))
    assert "raise" not in code and "assert " not in code, (
        "the post-corrective l_i check became a hard failure; that is a new "
        "acceptance criterion, not an approved part of this package")
    assert "WARNING" in body, "the out-of-band case is no longer loud"
    assert "li3_corrective_out_of_band" in src, \
        "the out-of-band condition is not archived on the quality dict"


def test_reconstruct_equilibrium_takes_the_l_i_band_from_the_caller():
    """The band is ``config.generation.l_i_tolerance``, not a hardcoded number:
    the report compares against the SAME band the draws are held to."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium
    import bouquet.baseline as bl

    params = inspect.signature(reconstruct_equilibrium).parameters
    assert "l_i_tolerance" in params
    assert params["l_i_tolerance"].default == 0.01
    assert "l_i_tolerance=float(config.generation.l_i_tolerance)" in \
        inspect.getsource(bl), \
        "baseline.py does not pass the configured l_i band to the recon"


def test_reconstruction_metrics_carries_the_post_corrective_fields():
    """The condition must survive into the archive, not just into stdout."""
    import bouquet.baseline as bl

    class _Stats(dict):
        pass

    class _GS:
        o_point = (1.7, 0.03)

        def get_stats(self, lcfs_pad=None, li_normalization=None):
            return {"q_0": 1.0, "q_95": 4.0, "beta_n": 2.0, "beta_pol": 80.0,
                    "kappa": 1.8, "delta": 0.5, "W_MHD": 1.0e6, "l_i": 0.68}

    class _Eqdsk:
        psi_N = np.linspace(0.0, 1.0, 51)
        Ip = 1.4e6
        li = {}

    class _Src:
        psi_pad = 1e-3

    result = {
        "Ip_tokamaker": 1.4e6,
        "j_phi_fit": np.linspace(1e6, 0.0, 51),
        "eqdsk_jtor": np.linspace(1e6, 0.0, 51),
        "quality": {"li3_corrective_drift_pct": 0.75,
                    "li3_corrective_band_pct": 1.0,
                    "li3_corrective_out_of_band": False},
    }
    with pytest.warns(UserWarning):          # no real EFIT reference on the stub
        m = bl._reconstruction_metrics(_GS(), _Eqdsk(), result, _Src(), 0.6842,
                                       l_i_realized_post_corrective=0.6893)
    assert m["li_realized_post_corrective"] == pytest.approx(0.6893)
    assert m["li_corrective_drift_pct"] == pytest.approx(0.75)
    assert m["li_corrective_band_pct"] == pytest.approx(1.0)
    assert m["li_corrective_out_of_band"] is False
    # the target itself is still the matched value that was handed in
    assert m["li"] == pytest.approx(0.6842)


@pytest.mark.parametrize("protect_state", [True, False])
def test_a_first_solve_failure_returns_the_input_not_a_NameError(
        _patched_jphi_from_GS, capsys, protect_state):
    """If the FIRST solve raises, the function must still return sanely.

    The failure handler ``break``s out of the loop, but ``j_phi_output`` is
    only assigned AFTER a successful solve -- so on an iteration-0 failure the
    return statement raised ``NameError: name 'j_phi_output' is not defined``,
    masking a solve failure behind an unrelated-looking crash.  The truthful
    degradation is "no correction was applied": return the uncorrected input.
    """
    from bouquet.TokaMaker_interface import _corrective_jphi_iteration

    psi_N = np.linspace(0.0, 1.0, 101)
    target = 1.0e6 * (1.0 - psi_N ** 2)

    class _FailFirst(_KeepBestStub):
        def solve(self):
            raise RuntimeError('Exceeded "maxits"')

    stub = _FailFirst(psi_N, target)
    _patched_jphi_from_GS["stub"] = stub

    out, n_iters, edge_hist = _corrective_jphi_iteration(
        stub, psi_N, target, {"type": "linterp", "y": psi_N, "x": psi_N},
        1.0e6, 1.0e4, 1e-3, min_iters=2, max_iters=3, rtol=0.0,
        verbose=False, protect_state=protect_state)

    np.testing.assert_array_equal(out, target), \
        "a first-solve failure must hand back the uncorrected input j_phi"
    assert edge_hist == [], "no iterate succeeded, so there is no RMS history"
    assert n_iters == 1
    # and it must SAY so, unconditionally -- not only under verbose
    assert "no corrective iteration was applied" in capsys.readouterr().out, (
        "a total corrective-iteration failure is silent; the caller cannot "
        "distinguish it from a converged result without inspecting the history")
    if protect_state:
        assert stub.replaced_with is not None, (
            "the failed iterate's state was not restored from the snapshot")
