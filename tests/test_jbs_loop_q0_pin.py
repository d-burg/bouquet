"""The q0 pin acting under the self-consistent loop -- fast half (no solver).

``GenerationConfig.jbs_loop_q0_corrector`` (default ``False``).  With the loop
ON and a channel whose axis row pins q0, the flag moves the axis row once per
pass from the q0 measured on that pass's solved equilibrium
(:class:`bouquet.jbs_loop.AxisRowPin`) and ADDS ``|q0 - q0_target| <= q0_tol``
to the loop's convergence criteria.

Tested here without a GS solver, on toy fixed-point maps in the style of
``test_jbs_loop.py``:

* flag OFF: the kernel is the kernel without the pin -- same record keys,
  same criteria, and the pre-change numbers of two reference problems;
* flag ON: a toy where q0 responds to the solved axis current AND to the
  bootstrap (through a geometry factor) converges to the JOINT fixed point
  -- bootstrap, l_i and q0 together -- within the default pass ceiling and
  relaxations, with ``|q0 - q0_target| <= q0_tol`` on the delivered pass,
  while the same toy with the row held misses the target;
* flag ON with an unreachable target fails loudly at the (unchanged) ceiling,
  with the q0 residual in the message, the record and the flag reason;
* the legacy path (``jbs_self_consistent=False``) never reads the flag, and
  the predictors match the pinned row only when the loop hands one over.

The live checks (``sawtooth_bootstrap`` and ``structured`` on the synthetic
example, flag ON vs OFF) are in ``test_jbs_loop_q0_pin_solver.py``.

Synthetic inputs only; no device data.
"""
import inspect
import types

import numpy as np
import pytest

from bouquet.jbs_loop import (AXIS_ROW_UPDATE_RULE, AxisRowPin,
                              JBSNotConverged, flag_reason, jbs_settings,
                              profile_residuals, run_jbs_loop,
                              validate_jbs_settings)
from test_jbs_loop import (_GC, _IP, _W, _X, _affine_problem, _shape,
                           _two_state_problem)

Q0_TOL = 0.01          # GenerationConfig.q0_tol default (unchanged)


# ---------------------------------------------------------------------------
#  the config field
# ---------------------------------------------------------------------------
def test_the_flag_defaults_off_and_is_validated_as_a_bool():
    from bouquet.config import GenerationConfig
    assert GenerationConfig().jbs_loop_q0_corrector is False
    for good in (True, False, np.bool_(True)):
        validate_jbs_settings(_GC(jbs_loop_q0_corrector=good))
    for bad in (1, 0, "yes", None, 1.0):
        with pytest.raises(ValueError, match="jbs_loop_q0_corrector"):
            validate_jbs_settings(_GC(jbs_loop_q0_corrector=bad))


# ---------------------------------------------------------------------------
#  flag OFF: the kernel without the pin, bit for bit
# ---------------------------------------------------------------------------
_PRE_KEYS = sorted([
    "I_BS", "I_BS_used", "converged", "criteria", "current_blended",
    "current_gap", "current_relaxation", "current_residual_unrelaxed",
    "current_residual_unrelaxed_definition", "dl_i", "dq0", "enabled",
    "evaluate_jBS_version", "final", "grid", "init", "init_source",
    "jBS_peak", "jBS_peak_psiN", "jbs_converged", "label", "li", "n_passes",
    "oft_build", "omega", "omega_halved_at_pass", "pass_ok", "q0", "r_I",
    "r_j", "relax_current", "relax_halve_on", "stop_reason", "tolerances",
    "wall_s"])
_PRE_FINAL = sorted(["I_BS", "current_gap", "current_residual_unrelaxed",
                     "dl_i", "dq0", "r_I", "r_j"])
#: what a loop that relaxes the current at beta < 1 adds since 2026-10-06
#: (r_j / r_I measured against the bootstrap the pass SOLVED; the iterate
#: residual kept beside it) -- the two-state reference runs at beta = 0.7
_SOLVED_KEYS = ["I_BS_used_iterate", "bootstrap_blended",
                "r_I_iterate", "r_j_iterate", "residual_definition"]
_SOLVED_FINAL = ["r_I_iterate", "r_j_iterate"]


def _ref_affine(**kw):
    Jstar, step, ev = _affine_problem(-0.5)
    return run_jbs_loop(_shape() * 0.5, step, ev, jbs_settings(_GC()),
                        Ip=_IP, gate_q0=True, meas0=dict(li=1.0, q0=1.0),
                        **kw)


def _ref_two_state(**kw):
    step, ev, st, L, J = _two_state_problem()
    s = jbs_settings(_GC(jbs_relax_current=0.7, jbs_max_passes=40))
    return run_jbs_loop(J * 0.6, step, ev, s, Ip=_IP,
                        meas0=dict(li=st["L"]), gate_li=True, **kw)


def _strip(rec):
    r = dict(rec)
    r.pop("wall_s")
    return r


@pytest.mark.parametrize("run", [_ref_affine, _ref_two_state])
def test_flag_off_is_the_kernel_without_the_pin(run):
    a = run()
    b = run(q0_pin=None)
    assert _strip(a["record"]) == _strip(b["record"])
    assert np.array_equal(a["jbs_used"], b["jbs_used"])
    assert np.array_equal(a["J_final"], b["J_final"])
    r = a["record"]
    # no key of the pin leaks into a pin-less record
    relaxed = run is _ref_two_state
    assert sorted(r.keys()) == sorted(_PRE_KEYS
                                      + (_SOLVED_KEYS if relaxed else []))
    assert sorted(r["final"].keys()) == sorted(
        _PRE_FINAL + (_SOLVED_FINAL if relaxed else []))
    assert sorted(r["criteria"]) == ["dl_i", "dq0", "r_I", "r_j"]
    assert "q0-q0_target" not in flag_reason(r)


def test_flag_off_reproduces_the_pre_change_kernel_numbers():
    """Frozen from the kernel BEFORE the pin existed (same toy problems):
    pass count, stop reason and every per-pass residual."""
    r = _ref_affine()["record"]
    assert (r["n_passes"], r["converged"]) == (5, True)
    assert r["stop_reason"] == ("all active criteria met on 2 consecutive "
                                "passes")
    np.testing.assert_allclose(r["r_j"], [
        0.6000000000000001, 0.03797468354430373, 0.001873828856964407,
        9.375292977904443e-05, 4.687492675831254e-06], rtol=1e-12, atol=0)
    np.testing.assert_allclose(r["dq0"], [
        5.0000000000105516e-05, 5.249999999978883e-05,
        2.6249999998118057e-06, 1.3124999997948805e-07,
        6.562500098894475e-09], rtol=1e-9, atol=0)
    o = _ref_two_state()
    r = o["record"]
    # beta = 0.7: since 2026-10-06 r_j is measured against the bootstrap
    # the pass SOLVED.  The PATH is unchanged -- the residual against the
    # iterate (the pre-pin kernel's r_j, frozen below) is recorded as
    # r_j_iterate and is the same on the passes both take -- but the
    # solved state needs one more pass to meet the tolerances (it met them
    # on the iterate at pass 8 only).
    assert (r["n_passes"], r["converged"]) == (9, True)
    np.testing.assert_allclose(r["r_j_iterate"][:8], [
        0.36894014841400047, 0.15704589064415098, 0.043629647034653746,
        0.013611487326422986, 0.0041066677460088455, 0.0012506097201954445,
        0.0003798751853432873, 0.00011547028381639267], rtol=1e-12, atol=0)
    np.testing.assert_allclose(r["li"][7], 0.6209078828084782, rtol=1e-12)
    assert r["r_j"][-1] <= 1e-3 and r["r_j"][-2] <= 1e-3
    assert r["r_j"][7] > r["r_j_iterate"][7]       # the solved state lags


# ---------------------------------------------------------------------------
#  the toy: q0 responds to the solved axis current AND to the bootstrap
# ---------------------------------------------------------------------------
_AX = np.exp(-(_X / 0.3) ** 2)       # axis-peaked inductive shape, _AX[0]=1
_J0_REQ = 1.0e6                      # the anchor's requested axis current
_C = 1.0e6                           # q0 = (C / j0)**gain * geo(bootstrap)


def _toy(pin=None, gain=1.0, c=0.3, e=0.2, q0_floor=None, li_follows=True):
    """closure -> (relaxed) solve -> Redl, with an axis row.

    * the closure imposes the axis row exactly: ``jc(0) = row`` (the pin's
      row when a pin is handed in, else the held requested current);
    * the "equilibrium" has ``q0 = (C / js(0))**gain * geo``, where ``js`` is
      the SOLVED (beta-relaxed) current and ``geo = 1 + 0.01 (I_BS/1e5 - 1)``
      carries the bootstrap's effect on q0 through the geometry;
    * Redl: ``J = J* (1 + e (js(0)/j0_req - 1)) + c (jbs - J*)`` -- the
      bootstrap responds to the axis current too, so bootstrap and q0 are
      genuinely coupled.
    """
    st = {}

    def step(jbs, k, relax=None):
        jbs = np.asarray(jbs, dtype=float)
        row = _J0_REQ if pin is None else pin.row
        jc = (row - jbs[0]) * _AX + jbs          # axis value == row
        js = jc if relax is None else relax(jc)
        st["js"], st["jbs"] = np.asarray(js, float).copy(), jbs.copy()
        geo = _geo(jbs)
        q0 = (_C / js[0]) ** gain * geo
        if q0_floor is not None:
            q0 = max(q0, q0_floor)
        li = (1.0 + 0.3 * js[0] / _J0_REQ) if li_follows else 1.3
        return dict(w=_W, x=_X, li=li, q0=q0,
                    axis_current_solved=float(js[0]))

    def evaluate(meas):
        return (_shape() * (1.0 + e * (st["js"][0] / _J0_REQ - 1.0))
                + c * (st["jbs"] - _shape()))

    return step, evaluate, st


def _geo(jbs):
    return 1.0 + 0.01 * (float(np.trapezoid(_W * jbs, _X)) / 1e5 - 1.0)


def _jbs_star(j0, c=0.3, e=0.2):
    """The toy's bootstrap fixed point at a given solved axis current."""
    return _shape() * (1.0 + e * (j0 / _J0_REQ - 1.0) / (1.0 - c))


def _joint_fixed_point(q0_target, gain=1.0):
    """The axis current at which the toy's self-consistent q0 is the target
    (bootstrap at its own fixed point): a 1-D root, solved to rounding."""
    from scipy.optimize import brentq
    f = lambda j0: (_C / j0) ** gain * _geo(_jbs_star(j0)) - q0_target
    return brentq(f, 0.5 * _J0_REQ, 2.0 * _J0_REQ, xtol=1e-9, rtol=1e-15)


def _run(pin, beta=0.7, **toy):
    step, ev, st = _toy(pin=pin, **toy)
    s = jbs_settings(_GC(jbs_relax_current=beta))
    out = run_jbs_loop(_shape().copy(), step, ev, s, Ip=_IP,
                       meas0=dict(li=1.3, q0=1.0), gate_li=True,
                       gate_q0=True, q0_pin=pin, label="toy")
    return out, st, s


# ---------------------------------------------------------------------------
#  flag ON: the joint fixed point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gain", [0.7, 1.0, 1.4])
@pytest.mark.parametrize("beta", [1.0, 0.7])
def test_flag_on_converges_to_the_joint_fixed_point(gain, beta):
    q0_target = 1.02
    pin = AxisRowPin(q0_target, Q0_TOL, _J0_REQ, label="toy")
    out, st, s = _run(pin, beta=beta, gain=gain)
    r = out["record"]
    # converged within the DEFAULT pass ceiling and relaxations
    assert out["converged"], r["stop_reason"]
    assert r["n_passes"] <= s["max_passes"] == 8
    assert r["criteria"]["q0_residual"] is True and r["criteria"]["dq0"]
    p = r["q0_pin"]
    # the last two passes meet EVERY criterion, the q0 target included
    for i in (-1, -2):
        assert r["r_j"][i] <= s["rtol_j"] and r["r_I"][i] <= s["rtol_Ip"]
        assert r["dl_i"][i] <= s["tol_li"] and r["dq0"][i] <= s["tol_q0"]
        assert abs(p["q0_residual"][i]) <= Q0_TOL
    # the delivered equilibrium is the last pass: its q0 meets the target
    # and its bootstrap is the one last evaluated
    assert p["final_q0_residual"] == p["q0_residual"][-1]
    assert r["final"]["q0_residual_over_tol"] <= 1.0
    assert abs(r["q0"][-1] - q0_target) <= Q0_TOL
    assert profile_residuals(out["J_final"], out["jbs_used"], _W, _X,
                             _IP)["r_j"] <= s["rtol_j"]
    # ... and it is the JOINT fixed point of the toy: the solved axis current
    # sits where the self-consistent q0 is the target, and the bootstrap is
    # the self-consistent one at that axis current
    j0_star = _joint_fixed_point(q0_target, gain)
    j0_del = p["axis_current_solved"][-1]
    # |q0 - target| <= q0_tol bounds |dj0/j0| by ~ q0_tol / (gain q0)
    assert abs(j0_del - j0_star) / j0_star <= Q0_TOL / (gain * q0_target)
    assert profile_residuals(_jbs_star(j0_del), out["jbs_used"], _W, _X,
                             _IP)["r_j"] <= 2 * s["rtol_j"]
    # the pin acted: the row moved from the requested current, and it was
    # moved after every pass but the last (never after the delivered one)
    assert p["n_row_updates"] == r["n_passes"] - 1 >= 1
    assert p["axis_row"][0] == _J0_REQ and p["axis_row_next"][-1] is None
    assert p["axis_row_final"] == p["axis_row"][-1]


def test_the_same_toy_with_the_row_held_misses_the_target():
    """Flag OFF on the toy that flag ON lands: the loop converges (its step
    criteria hold) with the q0 residual outside q0_tol -- the residual the
    record-only corrector can only flag."""
    out, st, s = _run(None)
    r = out["record"]
    assert out["converged"]
    assert "q0_pin" not in r and "q0_residual" not in r["criteria"]
    assert abs(r["q0"][-1] - 1.02) > Q0_TOL


def test_the_row_update_is_the_legacy_structured_corrector_rule():
    """beta = 1 (no relaxation of the solved current): the solved axis
    current IS the row, and the update is exactly
    j_ref0' = j_ref0 * q0_solved / q0_target."""
    pin = AxisRowPin(1.02, Q0_TOL, _J0_REQ)
    pin.observe(0.99, _J0_REQ)
    assert pin.advance() == pytest.approx(_J0_REQ * 0.99 / 1.02, rel=1e-15)
    # relaxed solve: the step is taken from the current actually SOLVED
    pin.observe(1.01, 0.97e6)
    assert pin.advance() == pytest.approx(0.97e6 * 1.01 / 1.02, rel=1e-15)
    assert pin.n_updates == 2
    assert "q0_solved/q0_target" in AXIS_ROW_UPDATE_RULE


def test_a_row_the_observation_cannot_define_is_kept_not_invented():
    pin = AxisRowPin(1.0, Q0_TOL, 1e6)
    pin.observe(float("nan"), 1e6)
    assert pin.advance() is None and pin.row == 1e6
    pin.observe(-1.0, 1e6)                   # would flip the row's sign
    assert pin.advance() is None and pin.row == 1e6
    assert pin.n_updates == 0
    assert not pin.within_tol(float("nan"))
    for bad in (dict(q0_target=0.0), dict(q0_tol=0.0), dict(row0=np.inf)):
        kw = dict(q0_target=1.0, q0_tol=Q0_TOL, row0=1e6)
        kw.update(bad)
        with pytest.raises(ValueError):
            AxisRowPin(kw["q0_target"], kw["q0_tol"], kw["row0"])


def test_a_pass_without_q0_stops_the_pinned_loop_at_once():
    pin = AxisRowPin(1.02, Q0_TOL, _J0_REQ)
    step, ev, st = _toy(pin=pin)

    def step_no_q0(jbs, k, relax=None):
        m = step(jbs, k, relax=relax)
        m["q0"] = None
        return m

    s = jbs_settings(_GC(jbs_loop_on_fail="flag"))
    out = run_jbs_loop(_shape().copy(), step_no_q0, ev, s, Ip=_IP,
                       meas0=dict(li=1.3), gate_li=True, q0_pin=pin)
    assert not out["converged"] and out["record"]["n_passes"] == 1
    assert "jbs_loop_q0_corrector" in out["record"]["stop_reason"]


# ---------------------------------------------------------------------------
#  flag ON, unreachable target: loud failure at the unchanged ceiling
# ---------------------------------------------------------------------------
def _unreachable(policy):
    # q0 cannot go below 0.9 whatever the axis current; the target is 0.5.
    # The bootstrap and l_i do not depend on the axis current here, so every
    # bootstrap criterion holds and ONLY the q0 target is out of reach.
    pin = AxisRowPin(0.5, Q0_TOL, _J0_REQ)
    step, ev, st = _toy(pin=pin, e=0.0, q0_floor=0.9, li_follows=False)
    s = jbs_settings(_GC(jbs_loop_on_fail=policy))
    return pin, s, lambda: run_jbs_loop(
        _shape().copy(), step, ev, s, Ip=_IP, meas0=dict(li=1.3, q0=0.9),
        gate_li=True, gate_q0=True, q0_pin=pin, label="toy")


def test_flag_on_unreachable_target_raises_at_the_ceiling():
    pin, s, run = _unreachable("raise")
    with pytest.raises(JBSNotConverged) as ei:
        run()
    rec = ei.value.record
    assert rec["n_passes"] == s["max_passes"] == 8    # the ceiling, unchanged
    assert rec["stop_reason"].startswith("pass ceiling 8 reached")
    # the bootstrap converged; the q0 target alone blocked convergence
    assert rec["r_j"][-1] <= s["rtol_j"] and rec["r_I"][-1] <= s["rtol_Ip"]
    assert all(abs(v) > Q0_TOL for v in rec["q0_pin"]["q0_residual"])
    assert rec["final"]["q0_residual"] == pytest.approx(0.4)
    assert rec["final"]["q0_residual_over_tol"] == pytest.approx(40.0)
    assert "q0-q0_target=" in str(ei.value) and "q0_tol 0.01" in str(ei.value)


def test_flag_on_unreachable_target_is_flagged_never_delivered_as_converged():
    pin, s, run = _unreachable("flag")
    out = run()
    assert out["converged"] is False
    rec = out["record"]
    assert rec["jbs_converged"] is False and rec["n_passes"] == 8
    why = flag_reason(rec)
    assert why.startswith("j_BS loop: did not converge")
    assert "q0-q0_target=4.00e-01" in why and "q0_tol 1.00e-02" in why


# ---------------------------------------------------------------------------
#  the run.py wiring: legacy path untouched, predictors match the pinned row
# ---------------------------------------------------------------------------
def test_the_legacy_path_never_reads_the_flag():
    """``jbs_self_consistent=False``: the flag is read only inside the loop's
    ohmic branch; the legacy ``_ohmic_close`` call hands no pass state (so no
    pinned row can reach a predictor) and the correctors never read it."""
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet._forward_solve_imas_baseline)
    loop_src, legacy_src = src.split("            if _loop_on:\n", 1)
    assert "jbs_loop_q0_corrector" not in legacy_src
    ohmic = loop_src.split("# ================= ohmic", 1)[1]
    assert "jbs_loop_q0_corrector" in ohmic
    assert "_ohmic_close(\n                        _anchor, j_BS_swb, ratio)" \
        in legacy_src
    for fn in (Bouquet._close_ip_q0_corrector,
               Bouquet._close_ip_structured_corrector,
               Bouquet._close_ip_q0_predictor,
               Bouquet._close_ip_structured_predictor):
        assert 'getattr(gc, "jbs_loop_q0_corrector"' not in \
            inspect.getsource(fn), fn.__name__
    # the pin is built only when the loop's axis row is active AND the flag
    # is set; the OFF notice still says RECORD-ONLY and names the flag
    assert "if axis_active and _pin_flag:" in ohmic
    assert "RECORD-ONLY" in ohmic and "jbs_loop_q0_corrector=False" in ohmic
    assert "pin ACTS under the self-consistent" in ohmic


def _scalar_predictor(q0_ref):
    from bouquet.run import Bouquet
    from bouquet.utils import close_ip
    psi = np.linspace(0.0, 1.0, 51)
    psi_q = psi.copy()
    psi_q[0] = 1e-3
    j_ind = 1.2e6 * (1.0 - psi) ** 2 + 1e4
    j_bs = 2e5 * np.exp(-0.5 * ((psi - 0.9) / 0.05) ** 2) + 5e4 * (1 - psi)
    j_fix = 1e5 * (1.0 - psi) ** 3
    w = 1.0 + psi
    lin = lambda j: float(np.trapezoid(w * j, psi))     # noqa: E731
    Ip = lin(j_ind + j_bs + j_fix)
    gc = types.SimpleNamespace(q0_gate=1.1, q0_tol=Q0_TOL)
    bl = types.SimpleNamespace(sawtooth=dict(active=True, present=True,
                                             q0_dd=0.95))
    tot = j_ind + j_bs + j_fix
    return Bouquet._close_ip_q0_predictor(
        gc, bl, None, dict(psi_N=psi, psi_q=psi_q), tot, psi,
        j_ind, j_bs, j_fix, tot, 1.0, Ip, 0.0, lin(j_ind), lin(j_bs),
        lin(j_fix), close_ip, q0_ref=q0_ref)


def test_the_predictor_matches_the_pinned_row_only_when_handed_one():
    ref = dict(q0_target=0.98, q0_anchor=0.97, j_achieved0=1.30e6,
               j_requested0=1.28e6)
    so, sb, extra, state = _scalar_predictor(dict(ref))
    # held (flag OFF): the row is the requested axis current, no pin key
    assert extra["j_ref0_used"] == ref["j_requested0"]
    assert extra["q0_axis_current_predicted"] == pytest.approx(
        ref["j_requested0"], rel=1e-12)
    assert "q0_axis_row_source" not in extra
    # pinned (flag ON): the 2x2 matches the moved row; the target is unchanged
    so2, sb2, extra2, state2 = _scalar_predictor(dict(ref, axis_row=1.25e6))
    assert extra2["j_ref0_used"] == 1.25e6
    assert extra2["q0_axis_current_predicted"] == pytest.approx(1.25e6,
                                                                rel=1e-12)
    assert extra2["q0_target"] == ref["q0_target"] == state2["q0_target"]
    assert "jbs_loop_q0_corrector" in extra2["q0_axis_row_source"]
    assert (so2, sb2) != (so, sb)


# ---------------------------------------------------------------------------
#  the MSE chord stage keeps the pin acting (mocked solver + MSE algebra)
# ---------------------------------------------------------------------------
def _mse_stage(monkeypatch, q0_target, q0_floor=None, on_fail="raise"):
    """A converging chord stage (tan(gamma) mocked flat, Redl returns the
    iterate) whose equilibrium has q0 = 2e6 / j_solved(0).  ``refresh`` does
    what the loop's ``_refresh_structured`` does with the pin: observe the
    step just solved, advance the row, close the next step on it."""
    import bouquet.mse as M
    import bouquet.utils as U
    from types import SimpleNamespace
    from bouquet.run import Bouquet
    from test_mse_refusal_restore import (_IP as IP, _NCH, _W as W, _X as X,
                                          _Eq, _chords, _field_at, _jbs0,
                                          _patch_mse)
    _patch_mse(monkeypatch)
    monkeypatch.setattr(M, "mse_er_terms", lambda ch: "none (mocked)")

    def _close(*a, **k):
        return dict(a=[0.0], b=[0.0], mse_chi2_model=0.0,
                    mse_objective_model=0.0, s_ind=np.ones_like(X),
                    s_bs=np.ones_like(X), ohm_scale_eff=1.0,
                    bs_scale_eff=1.0, Ip_hybrid=IP, sign_pattern=None,
                    structure_ind=0.0, structure_bs=0.0, ip_residual=0.0,
                    ip_residual_pct=0.0, axis_residual=None, constraints=[])

    monkeypatch.setattr(U, "close_ip_structured", _close)
    jbs0 = _jbs0()
    j_ind = 2.0e6 * (1.0 - X) ** 2
    row0 = float(j_ind[0] + jbs0[0])
    pin = AxisRowPin(q0_target, Q0_TOL, row0, label="mse toy")

    def mkstate(row):
        # the closure's current has axis value == row (hard axis row)
        return dict(mse=_chords(),
                    mse_required=False, soft=False, psi_geom=X, basis=None,
                    free=np.array([True, True]), x_pred=[0.0, 0.0],
                    F_pred=0.0, j_ind=j_ind * (row - jbs0[0]) / j_ind[0],
                    j_BS_swb=jbs0, j_fixed=np.zeros_like(X), w_lin=W,
                    c_signed=0.0, Ip_signed=IP, weights=None,
                    ip_ind=0.7 * IP, ip_bs=0.3 * IP, ip_fix=0.0,
                    axis=dict(j_ref0=row))

    cur = {"st": mkstate(row0)}
    eq = _Eq(cur["st"]["j_ind"] + jbs0)
    bl = SimpleNamespace(j_phi=eq.j.copy(), j_BS=jbs0.copy(),
                         j_inductive=j_ind.copy(), jBS_diff=None,
                         jphi_diff=None, ohm_scale=1.0, bs_scale=1.0,
                         ip_closure=dict(closure_limited=False,
                                         closure_limited_reasons=()))

    def q0_of(j):
        q = 2.0e6 / float(j[0])
        return q if q0_floor is None else max(q, q0_floor)

    def refresh(jbs, k):
        pin.observe(q0_of(eq.j), cur["st"]["axis"]["j_ref0"],
                    stage=f"MSE chord step {k + 1}")
        pin.advance()
        cur["st"] = mkstate(pin.row)
        return cur["st"]

    s = jbs_settings(_GC(jbs_loop_on_fail=on_fail))
    run = lambda: Bouquet._structured_mse_jbs_stage(     # noqa: E731
        cur["st"], bl, eq, eq.solve_jphi, jbs0, refresh,
        lambda snap: jbs0, lambda snap: dict(li=0.8, q0=q0_of(snap.j)),
        lambda snap: (W, X), s, IP, gate_q0=True,
        field_at=_field_at, q0_pin=pin)
    return pin, eq, run


def test_the_mse_chord_stage_keeps_the_pin_acting(monkeypatch):
    pin, eq, run = _mse_stage(monkeypatch, q0_target=1.0)
    nl, cur, srec = run()
    assert srec["converged"], srec["stop_reason"]
    # the first chord step missed the target, the moved row landed it, and
    # every step's (and the final step's) residual is recorded
    res = srec["q0_residual"]
    assert abs(res[0]) > Q0_TOL and abs(res[-1]) <= Q0_TOL
    assert len(res) == srec["n_passes"]
    assert pin.n_updates >= 1 and pin.log["stage"][-1] == "MSE final step"
    # the delivered (final-step) equilibrium meets the target
    assert abs(2.0e6 / eq.j[0] - 1.0) <= Q0_TOL


def test_the_mse_chord_stage_fails_loudly_when_the_pin_cannot_land(
        monkeypatch):
    pin, eq, run = _mse_stage(monkeypatch, q0_target=1.0, q0_floor=1.2,
                              on_fail="flag")
    nl, cur, srec = run()
    assert srec["converged"] is False
    assert all(abs(v) > Q0_TOL for v in srec["q0_residual"])
    assert "q0-q0_target per step" in srec["fail_message"]
    pin, eq, run = _mse_stage(monkeypatch, q0_target=1.0, q0_floor=1.2,
                              on_fail="raise")
    with pytest.raises(JBSNotConverged, match="q0_tol 0.01"):
        run()
