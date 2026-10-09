"""The soft structured closure's stop test at the objective's rounding floor.

:func:`bouquet.utils.close_ip_structured_soft` refuses ("Levenberg damping
could not find a descent step") when no damped step can be verified downhill
and the scaled gradient is above ``rtol * max|J| * max(sqrt F, 1)``.  With an
Ip row computed as an MA-scale difference over a kA-scale sigma, the
objective's rounding noise can exceed the ``F (1 + 1e-14)`` acceptance slack
while the iterate is within ~1e-10 of the minimiser: every Levenberg trial then
rounds "uphill" and the solver used to refuse a converged state -- a false
alarm.  The noise-aware test accepts such an iterate (``stop_reason =
"noise_floor"``) when the gradient is at the floor the noise implies AND every
trial's predicted decrease is below the noise estimate; a genuinely
inconsistent model still refuses.

Covered here (synthetic fixtures only, no device data):

* **natural false alarms**: parameter sets of the ``test_li_closure``
  fixture on which the pre-change solver refused on the development machine.
  Whether a given set refuses is platform-dependent (rounding), so the
  assertion is platform-independent: every set now returns, and returns the
  minimiser (agreeing with a restart from a different point);
* **a deterministic false alarm**: an iterate 3e-10 from the minimiser whose
  every trial reads uphill (an injected offset standing in for rounding) is
  accepted at the noise floor and recorded;
* **genuine refusals stand**: the same injection far from the minimiser, and a
  Jacobian inconsistent with the residual, still raise;
* **the logged single retry** (:func:`bouquet.utils.soft_closure_with_retry`)
  and the ``x0`` start it uses;
* **the acceptance is opt-in** (``accept_noise_floor=True``, passed by the
  self-consistent bootstrap loop's callers only).  The default is the
  historical strict solver: it refuses exactly where, and with exactly the
  message, it always did, and returns bit-identical results everywhere else;
  a non-finite or non-positive noise estimate accepts nothing; every
  acceptance is printed as well as recorded.

The mechanism tests below call the solver with ``accept_noise_floor=True``
(:func:`_call`'s default) because that is the mode they test.
"""
import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

import bouquet.utils as BU
from bouquet.utils import (close_ip_structured_soft, sigma_from_weights,
                           soft_closure_with_retry, structured_li_of,
                           STRUCTURED_WEIGHTS_PHYSICS)
from test_li_closure import _model

#: (li_target factor, Ip target factor, sigma_Ip / Ip): sets on which the
#: pre-change solver raised the Levenberg refusal on the development machine
#: (a scan of 3000 random sets refused 25; 25 shown).  One-sided prior on.
_FALSE_ALARM_SETS = [
    (1.0069297099145569, 0.9512146526113613, 0.007703949047503366),
    (1.003393892806839, 0.9726500395682364, 0.008220352991053642),
    (0.9621788817374078, 0.9741902971147135, 0.007856640666959075),
    (1.007753941490527, 0.9622756878387196, 0.007135252517246443),
    (1.0080013943223602, 0.9595994533943384, 0.003132818702584988),
    (1.0097902914782784, 0.9701343796451593, 0.009759427507689234),
    (0.9791063433935695, 0.9746483649087511, 0.008200915387347943),
    (0.9579164766979977, 0.9731030166834227, 0.0078530498816373),
    (1.0000026552078254, 0.9749273087566446, 0.004609849143367182),
    (0.9885947146658917, 0.977304145926648, 0.007549546669624399),
    (1.0208973297123578, 0.9738369617828587, 0.006892608310792594),
    (0.9520231668836994, 0.994917156090763, 0.004468532140982961),
    (0.9936923469157107, 0.9832114838783428, 0.003770763259932519),
    (1.0080715481421496, 0.9789220684688512, 0.004297837847851584),
    (0.9901729117595474, 0.9671593057325181, 0.003834799902094013),
    (0.9751317231507399, 0.9704176366284205, 0.008547669890477325),
    (0.9919955000139028, 0.980141390847388, 0.006499091105999772),
    (0.969487343011888, 0.9751812229888399, 0.009913864815656526),
    (0.9953654405114493, 0.9809866059941278, 0.002672304082859812),
    (0.9993802028096295, 0.9763156862302276, 0.0049630189036630485),
    (0.9932389045191663, 0.9641799637243065, 0.005155780068996978),
    (0.9666581127836632, 0.9740904593443447, 0.00970655501024776),
    (0.9758308790215956, 0.9859204552130187, 0.005060354369337593),
    (0.9774953826481618, 0.9889577736101652, 0.0039008260425204685),
    (1.0332443144088492, 0.9715457652719514, 0.00694268914755585),
]
_UP = [0.02] * 4


def _call(f_li, f_ip, s_ip, **kw):
    m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
    sig = sigma_from_weights(STRUCTURED_WEIGHTS_PHYSICS, 4)
    li0 = structured_li_of(m)[0]
    kw.setdefault("accept_noise_floor", True)
    if kw["accept_noise_floor"] is None:        # the solver's own default
        del kw["accept_noise_floor"]
    return close_ip_structured_soft(
        psi, w, c, Ip_s * f_ip, s_ip * abs(Ip_s), ji, jb, jf,
        sigma_ind=sig["ind"], sigma_bs=sig["bs"],
        sigma_ind_up=kw.pop("sigma_ind_up", _UP), li_target=li0 * f_li,
        li_sigma=0.04, li_geom=lg, **kw)


def _x(out):
    return np.concatenate([out["a"], out["b"]])


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("params", _FALSE_ALARM_SETS)
def test_former_false_alarms_return_the_minimiser(params):
    out = _call(*params)
    assert out["gn_stop_reason"] in ("objective_change", "step",
                                     "gradient_floor", "noise_floor")
    # the minimiser: a restart from a point 1e-3 away lands on the same
    # coefficients and objective
    x = _x(out)
    ref = _call(*params, x0=x + 1e-3 * np.linspace(-1.0, 1.0, x.size))
    assert np.max(np.abs(_x(ref) - x)) < 1e-7
    assert out["objective"] == pytest.approx(ref["objective"], rel=1e-9,
                                             abs=1e-15)
    if out["gn_stop_reason"] == "noise_floor":
        g = out["gn_stop"]
        assert g["predicted_decrease"] <= g["noise_F"]
        assert g["gradient_l2"] <= g["gradient_floor_noise"]
        assert g["noise_F"] < 1e-10 * g["objective"]


def test_an_ordinary_solve_never_stops_at_the_noise_floor():
    out = _call(1.02, 0.99, 0.005)
    assert out["gn_stop_reason"] in ("objective_change", "step")
    assert out["n_noise_floor_accepts"] == 0
    assert out["started_from_x0"] is False


# ---------------------------------------------------------------------------
def _inject_uphill(monkeypatch, bump_of_F):
    """Every evaluation of the l_i row EXCEPT at the first point Gauss-Newton
    evaluates (its start; the model build's own ``x=None`` call is not a GN
    point) reads higher by ``bump_of_F`` x F in the objective -- a
    deterministic stand-in for rounding that makes every trial look uphill.
    The Jacobian is untouched (it is the smooth model's)."""
    orig = BU.structured_li_of
    st = {}

    def li_of(model, x=None):
        li, ip = orig(model, x)
        if x is None or "sign" not in st:
            return li, ip
        xx = np.asarray(x, dtype=float)
        if "x_start" not in st:
            st["x_start"] = xx.copy()
            return li, ip
        if np.array_equal(xx, st["x_start"]):
            return li, ip
        return li + st["sign"] * st["bump"], ip

    monkeypatch.setattr(BU, "structured_li_of", li_of)

    def arm(r_li, F, li_sigma):
        # dF = 2 |r_li| bump / li_sigma  -> bump for dF = bump_of_F * F
        st["sign"] = float(np.sign(r_li)) or 1.0
        st["bump"] = bump_of_F * F * li_sigma / (2.0 * max(abs(r_li), 1e-12))
    return st, arm


def _minimiser(params):
    out = _call(*params, sigma_ind_up=None)
    return out, _x(out)


def test_a_deterministic_false_alarm_is_accepted_at_the_noise_floor(
        monkeypatch):
    params = (1.03, 0.97, 0.005)
    out0, xs = _minimiser(params)
    r_li = out0["residual_sigma_li"]
    v = np.linspace(-1.0, 1.0, xs.size)
    x0 = xs + 3e-10 * v / np.max(np.abs(v))
    st, arm = _inject_uphill(monkeypatch, 1e-12)
    arm(r_li, out0["objective"], 0.04)
    out = _call(*params, sigma_ind_up=None, x0=x0)
    assert out["gn_stop_reason"] == "noise_floor"
    g = out["gn_stop"]
    assert g["gradient"] > g["gradient_floor"]        # the old test refused
    assert g["predicted_decrease"] <= g["noise_F"]
    assert g["gradient_l2"] <= g["gradient_floor_noise"]
    assert g["n_trials"] == 16
    # accepted AT the start: within 3e-10 of the minimiser
    assert np.max(np.abs(_x(out) - xs)) <= 3.0001e-10
    assert out["n_noise_floor_accepts"] == 1 and out["started_from_x0"]


def test_the_same_injection_far_from_the_minimiser_still_refuses(monkeypatch):
    """A model that predicts a real decrease the objective does not deliver
    is inconsistent: the refusal stands."""
    params = (1.03, 0.97, 0.005)
    out0, xs = _minimiser(params)
    v = np.linspace(-1.0, 1.0, xs.size)
    x0 = xs + 1e-3 * v
    st, arm = _inject_uphill(monkeypatch, 1.0)        # far above any decrease
    arm(out0["residual_sigma_li"], out0["objective"], 0.04)
    with pytest.raises(RuntimeError, match="Levenberg damping could not find"
                                           ".*predicted decrease"):
        _call(*params, sigma_ind_up=None, x0=x0)


def test_an_inconsistent_jacobian_still_refuses(monkeypatch):
    """The l_i row's Jacobian pointing the wrong way (x -50): no damped step
    descends although the (wrong) model predicts a large decrease."""
    orig = BU.structured_li_gradient
    monkeypatch.setattr(BU, "structured_li_gradient",
                        lambda model, x=None: -50.0 * orig(model, x))
    with pytest.raises(RuntimeError, match="Levenberg damping could not find"):
        _call(1.10, 1.0, 0.005, sigma_ind_up=None)


# ---------------------------------------------------------------------------
def test_x0_changes_the_start_not_the_answer():
    for up in (None, _UP):
        a = _call(1.03, 0.97, 0.005, sigma_ind_up=up)
        xa = _x(a)
        b = _call(1.03, 0.97, 0.005, sigma_ind_up=up,
                  x0=xa + 0.05 * np.linspace(-1.0, 1.0, xa.size))
        assert np.max(np.abs(_x(b) - xa)) < 1e-7
        assert b["started_from_x0"]
    with pytest.raises(ValueError, match="x0"):
        _call(1.03, 0.97, 0.005, x0=np.zeros(3))


_REFUSAL = RuntimeError("close_ip_structured_soft: Levenberg damping could "
                        "not find a descent step at objective 1e-2")


def test_retry_once_from_the_previous_coefficients():
    calls = []

    def solve(x0):
        calls.append(None if x0 is None else np.array(x0))
        if x0 is None:
            raise _REFUSAL
        return dict(ok=True)

    out = soft_closure_with_retry(solve, x_prev=np.arange(8.0))
    assert out["closure_retry"] == 1
    assert "Levenberg" in out["closure_retry_first_error"]
    assert calls[0] is None and np.array_equal(calls[1], np.arange(8.0))


def test_a_second_refusal_is_a_real_refusal():
    def solve(x0):
        raise _REFUSAL
    with pytest.raises(RuntimeError, match="after one retry"):
        soft_closure_with_retry(solve, x_prev=np.zeros(8))


def test_no_retry_without_previous_coefficients_or_for_other_errors():
    n = {"c": 0}

    def refuse(x0):
        n["c"] += 1
        raise _REFUSAL
    with pytest.raises(RuntimeError, match="Levenberg"):
        soft_closure_with_retry(refuse, x_prev=None)
    assert n["c"] == 1

    def other(x0):
        n["c"] += 1
        raise RuntimeError("close_ip_structured_soft: s_ind(psi) reaches 0.1")
    n["c"] = 0
    with pytest.raises(RuntimeError, match="s_ind"):
        soft_closure_with_retry(other, x_prev=np.zeros(8))
    assert n["c"] == 1
    ok = soft_closure_with_retry(lambda x0: dict(), x_prev=np.zeros(8))
    assert ok["closure_retry"] == 0 and ok["closure_retry_first_error"] is None


# ---------------------------------------------------------------------------
#  opt-in: the default is the historical strict solver
# ---------------------------------------------------------------------------
#: The historical refusal message, verbatim (base solver): no noise estimate.
_HISTORICAL_REFUSAL = (
    r"^close_ip_structured_soft: Levenberg damping could not find a descent "
    r"step at objective \S+ \(scaled gradient \S+ > floor \S+\) -- the "
    r"measurement rows and the prior are inconsistent on this basis$")


def _deterministic_false_alarm(monkeypatch):
    """The injected false alarm of the test above: ``(params, x0)``."""
    params = (1.03, 0.97, 0.005)
    out0, xs = _minimiser(params)
    v = np.linspace(-1.0, 1.0, xs.size)
    x0 = xs + 3e-10 * v / np.max(np.abs(v))
    st, arm = _inject_uphill(monkeypatch, 1e-12)
    arm(out0["residual_sigma_li"], out0["objective"], 0.04)
    return params, x0


@pytest.mark.parametrize("flag", [None, False])
def test_the_default_refuses_the_false_alarm_with_the_historical_message(
        monkeypatch, flag):
    """``accept_noise_floor`` omitted (None here) or False: the strict
    solver.  The iterate the opt-in mode accepts is REFUSED, with the
    historical message verbatim (no noise estimate is even formed)."""
    params, x0 = _deterministic_false_alarm(monkeypatch)
    with pytest.raises(RuntimeError, match=_HISTORICAL_REFUSAL):
        _call(*params, sigma_ind_up=None, x0=x0, accept_noise_floor=flag)


def test_opt_in_accepts_and_prints_the_same_false_alarm(monkeypatch, capsys):
    params, x0 = _deterministic_false_alarm(monkeypatch)
    out = _call(*params, sigma_ind_up=None, x0=x0, accept_noise_floor=True)
    assert out["gn_stop_reason"] == "noise_floor"
    assert out["n_noise_floor_accepts"] == 1
    printed = capsys.readouterr().out
    assert "NOISE-FLOOR ACCEPTANCE" in printed
    assert "historical gradient test refused" in printed


@pytest.mark.parametrize("factor", [float("inf"), float("nan"), 0.0, -1.0])
def test_a_non_finite_or_non_positive_noise_estimate_accepts_nothing(
        monkeypatch, factor):
    """The finiteness guard: an estimate that is inf (which would otherwise
    accept ANY stalled iterate), NaN, zero or negative is not an estimate.
    The refusal stands even with the opt-in; the guard only ever makes
    acceptance stricter."""
    params, x0 = _deterministic_false_alarm(monkeypatch)
    monkeypatch.setattr(BU, "NOISE_FLOOR_FACTOR", factor)
    with pytest.raises(RuntimeError, match="Levenberg damping could not find"
                                           ".*rounding noise of the "
                                           "objective n/a"):
        _call(*params, sigma_ind_up=None, x0=x0, accept_noise_floor=True)


@pytest.mark.parametrize("params", _FALSE_ALARM_SETS
                         + [(1.02, 0.99, 0.005), (1.03, 0.97, 0.005)])
@pytest.mark.parametrize("up", [None, _UP])
def test_the_strict_default_is_the_opt_in_minus_the_noise_floor(params, up):
    """Whatever this platform's rounding does on a parameter set, the two
    modes differ ONLY where the opt-in accepted at the noise floor: there the
    default raises the historical refusal; everywhere else the returned
    closures are bit-identical."""
    try:
        opt = _call(*params, sigma_ind_up=up, accept_noise_floor=True)
    except RuntimeError as e:
        with pytest.raises(RuntimeError, match="Levenberg|converge|reaches"):
            _call(*params, sigma_ind_up=up, accept_noise_floor=None)
        assert "Levenberg" not in str(e) or "predicted decrease" in str(e)
        return
    if opt["n_noise_floor_accepts"] > 0:
        with pytest.raises(RuntimeError, match=_HISTORICAL_REFUSAL):
            _call(*params, sigma_ind_up=up, accept_noise_floor=None)
        return
    strict = _call(*params, sigma_ind_up=up, accept_noise_floor=None)
    for k in ("a", "b", "s_ind", "s_bs"):
        np.testing.assert_array_equal(strict[k], opt[k])
    assert strict["objective"] == opt["objective"]
    assert strict["gn_stop_reason"] == opt["gn_stop_reason"]
    assert strict["n_noise_floor_accepts"] == 0
