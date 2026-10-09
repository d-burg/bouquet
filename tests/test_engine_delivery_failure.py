"""The unified engine's delivery check FAILS loudly -- negative controls.

Every other engine test delivers a good equilibrium, so a regression that
made the delivery check always pass, never raise, or that let the backend
return a failed solve as converged would leave the fast suite green (the
adversarial review's surviving mutants E3/E4/E7/E8/E11/E12).  Here:

* a toy backend whose DELIVERY solve (the 2-pass solve after the loop, not
  the 2-pass anchor) bends the request: under ``jbs_loop_on_fail="raise"``
  the reconstruction raises ``JBSNotConverged`` with the delivery record;
  under ``"flag"`` it returns ``converged=False`` (the loop itself did
  converge) with the misses listed;
* each delivered row check -- l_i, q0, MSE -- reports its own miss when the
  delivered state misses it;
* ``TokaMakerBackend.solve`` turns a failed GS solve into
  ``EngineSolveError`` (chained, counted, the iteration cap put back) and
  never returns it as a solved state.

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

import _engine_toy as T
from bouquet.engine import EngineSolveError, TokaMakerBackend, reconstruct
from bouquet.jbs_loop import JBSNotConverged


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class _BendingDelivery(T.ToyGS):
    """The toy, except that the DELIVERY solve (the second 2-pass solve; the
    first is the anchor) delivers a bent current: the core of the request
    raised by *amp* -- a different equilibrium from the one the loop
    converged on (l_i, q0 and the pitch angles all move)."""

    def __init__(self, *a, amp=0.08, **k):
        super().__init__(*a, **k)
        self.amp = float(amp)
        self.n_two_pass = 0

    def solve(self, request, n_passes=1):
        if int(n_passes) == 2:
            self.n_two_pass += 1
            if self.n_two_pass == 2:
                request = np.asarray(request, dtype=float) * (
                    1.0 + self.amp * np.exp(-0.5 * (self.psi / 0.25) ** 2))
        return super().solve(request, n_passes=n_passes)


def _anchor_li_q():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"], b.q_at(b.state, T.PAD)


LI0, Q0 = _anchor_li_q()


def _run(adapter, backend, **gc):
    adapter.read()
    return _quiet(reconstruct, adapter, backend, T.settings(**gc),
                  label="toy")


def test_the_unbent_control_delivers():
    """Control: the same configuration without the bend delivers ok (so the
    failures below are the bend's, not the configuration's)."""
    eng, res, rec = _run(T.ToyAdapter(li_target=LI0 * 1.01),
                         _BendingDelivery(amp=0.0))
    assert res["converged"] and rec["delivered"]["ok"]


def test_a_failed_delivery_raises_under_the_raise_policy():
    with pytest.raises(JBSNotConverged) as ei:
        _run(T.ToyAdapter(li_target=LI0 * 1.01), _BendingDelivery())
    msg = str(ei.value)
    assert "delivered equilibrium fails" in msg
    assert "l_i row error" in msg


def test_a_failed_delivery_is_not_converged_under_the_flag_policy():
    eng, res, rec = _run(T.ToyAdapter(li_target=LI0 * 1.01),
                         _BendingDelivery(), jbs_loop_on_fail="flag")
    d = rec["delivered"]
    assert res["loop_converged"] is True        # the loop itself converged
    assert res["converged"] is False            # ... the delivery did not
    assert rec["converged"] is False
    assert d["ok"] is False and d["misses"]
    assert d["fail_message"].startswith("toy: the delivered equilibrium")
    assert d["checks"]["l_i"]["ok"] is False
    assert any(m.startswith("l_i row error") for m in d["misses"])
    # the record states what the delivered state is
    assert d["checks"]["l_i"]["delivered"] == eng.b.state["li"]


def test_a_delivered_q0_miss_is_its_own_miss():
    ad = T.ToyAdapter(li_target=LI0 * 1.01, q0_target=Q0 * 0.97)
    eng, res, rec = _run(ad, _BendingDelivery(),
                         engine_rows=["Ip", "l_i", "q0"],
                         jbs_loop_on_fail="flag")
    q = rec["delivered"]["checks"]["q0"]
    assert q["ok"] is False
    assert abs(q["residual"]) > q["tol"]
    assert any(m.startswith("q0 - q0_target") for m in
               rec["delivered"]["misses"])
    assert res["converged"] is False


def test_a_delivered_mse_miss_is_its_own_miss():
    import test_engine as TE
    md, ch = TE._mse_data()
    ad = T.ToyAdapter(li_target=LI0 * 1.01,
                      mse=dict(chords=ch, er_terms="toy"))
    eng, res, rec = _run(ad, _BendingDelivery(chords=ch),
                         engine_rows=["Ip", "l_i", "mse"], mse_data=md,
                         jbs_loop_on_fail="flag")
    m = rec["delivered"]["checks"]["mse"]
    assert m["ok"] is False and m["dtg_max_sigma"] > m["tol"]
    assert any(x.startswith("MSE tan(gamma) change") for x in
               rec["delivered"]["misses"])
    assert res["converged"] is False


# ---------------------------------------------------------------------------
#  the live backend: a failed GS solve is never a solved state
# ---------------------------------------------------------------------------
class _FailingSolver:
    """The surface ``TokaMakerBackend.solve`` touches; ``solve`` raises the
    solver's ValueError on the calls listed in *fail* (1-based)."""

    def __init__(self, fail, maxits=80):
        self.settings = SimpleNamespace(maxits=maxits)
        self.fail = set(fail)
        self.n = 0
        self.updates = []
        self.solved = []

    psi_bounds = (-0.3, 0.0)

    def update_settings(self):
        self.updates.append(self.settings.maxits)

    def set_targets(self, **kw):
        pass

    def set_profiles(self, **kw):
        pass

    def solve(self):
        self.n += 1
        if self.n in self.fail:
            raise ValueError('Error in solve: Exceeded "maxits"')
        self.solved.append(self.n)


def _contract():
    psi = T.PSI
    return SimpleNamespace(psi_N=psi, pressure=T.pressure(psi), Ip=1.2e6)


@pytest.mark.parametrize("maxits", [None, 40])
@pytest.mark.parametrize("fail_at", [1, 2])
def test_the_backend_raises_on_a_failed_solve(maxits, fail_at):
    gs = _FailingSolver({fail_at})
    b = TokaMakerBackend(gs, _contract(), maxits=maxits)
    with pytest.raises(EngineSolveError) as ei:
        b.solve(np.ones(T.PSI.size), n_passes=2)
    e = ei.value
    assert isinstance(e.__cause__, ValueError)
    assert f"pass {fail_at}/2" in str(e) and "maxits" in str(e)
    assert b.n_solves == fail_at                # the failed solve counted
    assert gs.solved == list(range(1, fail_at)) # nothing after it
    assert b.last_solve_s is not None
    assert gs.settings.maxits == 80             # the cap put back
    if maxits is not None:
        assert gs.updates == [40, 80]


def test_the_backend_refuses_a_non_finite_request():
    gs = _FailingSolver(set())
    b = TokaMakerBackend(gs, _contract())
    r = np.ones(T.PSI.size)
    r[3] = np.nan
    with pytest.raises(EngineSolveError, match="non-finite"):
        b.solve(r)
    assert gs.n == 0


@pytest.mark.parametrize("amp, li_ok", [(0.003, True), (0.005, False)])
def test_the_li_row_check_binds_at_its_own_tolerance(amp, li_ok):
    """The delivered l_i row check is decided by ITS tolerance (the g-file
    row tol, 1e-3): a bend whose l_i error is 0.83x the tolerance passes the
    l_i check, one at 1.4x fails it (the other checks have their own
    verdicts)."""
    eng, res, rec = _run(T.ToyAdapter(li_target=LI0 * 1.01),
                         _BendingDelivery(amp=amp), jbs_loop_on_fail="flag")
    c = rec["delivered"]["checks"]["l_i"]
    assert 0.5 * c["tol"] < abs(c["error"]) < 2.0 * c["tol"]
    assert c["ok"] is li_ok
    assert (abs(c["error"]) <= c["tol"]) is li_ok
