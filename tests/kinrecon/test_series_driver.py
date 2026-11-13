"""The series driver :func:`bouquet.kinrecon.series.run_series` with the equilibrium solve replaced
(no OFT): analytic separatrix-like geometry (as ``test_coordinates_kinetics_series.py``), a mocked
forward model whose "equilibrium" is one of a few known geometries, and a mocked fit.

* Filter mode carries the state of the ACCEPTED center: the mocked filter fit advances the state on
  several perturbed geometries (Jacobian columns, a trial) after reaching its center, so the last
  state the fit computed is not the center's.  The carried state must equal the state advanced from
  the previous slice on the center's geometry plus the corrections' steady response, computed here
  directly with the interval solver (exact to round-off: same operations, same inputs).
* Filter mode absorbs only a converged fit: for every other status the slice keeps its evolved,
  unfitted equilibrium (the solver is restored to it) and state, and the record says
  ``absorbed: False``.
* Independent reference: the chain of intervals built by ``run_series`` (state carried between
  slices, Ip linear between the slices' values, lumped outer layer) against a separately coded
  vertex-centered finite-volume solution of the same diffusion problem, integrated EXACTLY in time
  (matrix exponential of the semi-discrete system with the Ip ramp as augmented states).  Shared
  inputs: the geometry profiles ``V', <|grad rho|^2/R^2>, <1/R^2>, F`` and the layer conductance
  ``A_layer`` (scalar).  Everything else -- discretization, time integration, the boundary coupling
  ``k psi'(1) + A_layer dpsi_b/dt = Ip(t)/F_lcfs``, the Ip ramp and the chaining -- is independent.
  Static geometry, uniform conductivity, no non-inductive current.  Bound (fixed before the first
  run): the gauge-free profile ``psi - psi_b`` within 1e-3 of its span and the boundary flux change
  within 1e-3 (relative) at every slice (100 cells, the default step); the test discriminates: the
  same series without the Ip ramp differs from the reference by more than 10x the bound.  (The
  boundary flux converges slowly: the layer shares the LAST CELL's dpsi/dt, a first-order coupling
  stated in the diffusion module; measured 7e-4 at 100 cells, 2e-4 at 200, 3e-5 at 400.)
* The sawtooth operator inside a series: with ``sawtooth="periodic"`` and a state with q >= 1 the
  crashes are recorded and change nothing (states identical to ``sawtooth="off"``); with q < 1 on axis
  a crash acts (the carried, instantaneous state jumps), and the reported slice state is continuous
  in the crash time: a crash moved from just before a slice time to just after it changes the slice
  state by about its share of the interval, not by the whole jump.
* Time-varying ``F0``: each slice's ``F0`` is set on the forward model before its solve.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_coordinates_kinetics_series as GEO  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import MU0, CurrentState  # noqa: E402

IP0 = GEO.IP
SIGMA = 2.0e6          # uniform conductivity [S/m]: skin depth over 20 ms ~ 0.09 m (resolved by the grids used)


# ============================================================== mocks
class FakeFM:
    """Forward-model stand-in: its "equilibrium" is a key into ``lags``; ``gs`` is itself."""

    def __init__(self, lags, F0=GEO.F0):
        self.lags = lags
        self.key = "pred"
        self.gs = self
        self.n_solves = 0
        self.probe_correction = np.zeros(0)
        self.case = SimpleNamespace(case_id="fake")
        self.case_id = "fake"
        self.F0 = F0
        self.F0_at_solve = []
        self.rows_record = {"coils": {"per_coil": {}}}

    def snapshot(self):
        return dict(key=self.key)

    def restore(self, snap):
        self.key = snap["key"]

    def set_F0(self, F0):
        self.F0 = float(F0)

    def reconfigure(self, measurements, reference_boundary, coil_roles=None, case_id="case", ip=None,
                    generation_coil_reg=None):
        self.ip_measured = ip

    def boundary_distance(self):
        return {}


def _assembly(lag, state):
    rho = np.asarray(lag["phys"]["rho"])
    return dict(state=state, fractions={}, currents_A={}, v_loop=0.0, sawtooth=dict(active=False),
                q_model=dict(rho=rho, q=np.asarray(lag["phys"]["q"])), profiles_rho=dict(rho=rho))


def _result(lag, state, ok=True):
    return CM.ConsistentResult(ok, "consistent" if ok else "solve_failed", 2, _assembly(lag, state), lag,
                               SimpleNamespace(stats={}), dict(passes=[], q_check=None), 0.0)


def _install(monkeypatch, fm, state0, lag_of_slice):
    """Mocked consistent solve: slice 0 returns ``state0``; later slices evaluate the supplied state on
    the geometry ``lag_of_slice(fm)`` twice (two passes) and leave the solver at that equilibrium."""

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        fm_.n_solves += 1
        fm_.F0_at_solve.append(fm_.F0)
        lag = lag_of_slice(fm_)
        if state_fn is None:
            return _result(lag, state0)
        state_fn(lag)
        st = state_fn(lag)
        return _result(lag, st)

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)


def _src():
    return SimpleNamespace(jdotb_at=lambda ge, **kw: np.zeros_like(np.asarray(ge, float)))


def _slices(times, ips, F0s=None):
    bnd = np.zeros((8, 2))
    return [SE.SliceData(time=t, measurements=None, reference_boundary=bnd, kinetics=None, sources=_src(),
                         label=f"s{k}", meta=dict(ip=ip, generation_coil_reg=1.0),
                         F0=None if F0s is None else F0s[k])
            for k, (t, ip) in enumerate(zip(times, ips))]


def _lag(sigma_scale=1.0, q0=1.1, sigma_const=None):
    lag = GEO._lag(q0=q0, sigma_scale=sigma_scale)
    lag["jbs"] = np.zeros_like(lag["phys"]["psi_n"])
    if sigma_const is not None:
        lag["sigma"] = np.full_like(lag["phys"]["psi_n"], float(sigma_const))
    return lag


def _steady(lag, ip, n=100):
    rf = CurrentState.uniform_faces(n)
    st = CM.build_solver(lag, lag["jbs"], ip).steady_state(rf)
    return CurrentState(rf, st.psi, st.psi_boundary, 0.0)


# ============================================================== filter mode
CENTER_A = (0.01, -0.02, 0.015)


def _fake_reconstruct(status, lags, center_key="center", after=("col0", "col1", "trial")):
    """A filter fit that reaches its center on ``lags[center_key]`` and then advances the state on the
    geometries ``after`` (as the fit's Jacobian columns / trials do), leaving the solver at the last one."""
    calls = []

    def reconstruct(fm, kin, src, cfg, settings, params0=None, start=None, label="", state_fn=None, **kw):
        names = [f"a_corr_{k}" for k in range(3)]
        lag_c = lags[center_key]
        st_c = state_fn(lag_c)
        shapes = CM.correction_shapes(lag_c, cfg, fm.ip_measured)
        dpsi, dpb = CM.correction_response(lag_c, shapes, CENTER_A, st_c.rho_faces)
        asm_state = CurrentState(st_c.rho_faces, st_c.psi + dpsi, st_c.psi_boundary + dpb, st_c.time)
        center = dict(res=_result(lag_c, asm_state), snap=dict(key=center_key), probe_correction=np.zeros(0))
        for k in after:
            state_fn(lags[k])
            fm.key = k
        calls.append(label)
        return SimpleNamespace(status=status, theta=np.array(CENTER_A), sigma=np.full(3, 0.01), free=names,
                               params=dict(a_corr=CENTER_A), chi2={},
                               record=dict(center=None if status in ("failed_initial_consistency",) else center,
                                           specs={n: dict(prior=(0.0, 0.02)) for n in names}, n_solves=7,
                                           wall_s=0.0, final_stopping_tests=None,
                                           stopping_summary=dict(status=status)))

    reconstruct.calls = calls
    return reconstruct


def _filter_setup(monkeypatch, status):
    lags = dict(pred=_lag(), center=_lag(sigma_scale=1.3), col0=_lag(sigma_scale=0.7), col1=_lag(sigma_scale=1.6),
                trial=_lag(sigma_scale=0.5))
    fm = FakeFM(lags)
    st0 = _steady(lags["pred"], IP0)
    _install(monkeypatch, fm, st0, lambda f: f.lags[f.key])
    rec = _fake_reconstruct(status, lags)
    monkeypatch.setattr(SE._fit, "reconstruct", rec)
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    seen = []
    out = SE.run_series(fm, _slices([0.0, 0.02, 0.04], [IP0, 1.01 * IP0, 1.02 * IP0]), cfg,
                        SE.SeriesSettings(initial="relaxed", sawtooth="off", filter=True),
                        callback=lambda r: seen.append(fm.key))
    return out, lags, cfg, seen


def _advance(lag_prev, lag, carried, t0, t1, ip0, ip1):
    sol, _ = SE.interval_solver(lag_prev, lag, lag_prev["jbs"], lag["jbs"], t0, t1, ip0, ip1)
    return sol.advance(carried, t1, dt=SE.SeriesSettings().dt_max, scheme="trbdf2")


def _state(d):
    return CurrentState(np.asarray(d["rho_faces"]), np.asarray(d["psi"]), d["psi_boundary"], d["time"])


def test_filter_carries_the_accepted_centers_state(monkeypatch):
    out, lags, cfg, seen = _filter_setup(monkeypatch, "converged")
    assert out["status"] == "completed"
    s1 = out["slices"][1]
    f = s1["filter"]
    assert f["status"] == "converged" and f["absorbed"] is True
    # expected: carried state of slice 0 advanced over [0, 0.02] between slice 0's geometry (pred) and the
    # center's geometry, plus the corrections' steady response on the center's geometry
    c0 = _state(out["slices"][0]["carried_state"])
    e = _advance(lags["pred"], lags["center"], c0, 0.0, 0.02, IP0, 1.01 * IP0)
    shapes = CM.correction_shapes(lags["center"], cfg, 1.01 * IP0)
    dpsi, dpb = CM.correction_response(lags["center"], shapes, CENTER_A, e.rho_faces)
    got = _state(s1["carried_state"])
    span = np.ptp(e.psi + dpsi)
    assert np.max(np.abs(got.psi - (e.psi + dpsi))) <= 1e-12 * span
    assert abs(got.psi_boundary - (e.psi_boundary + dpb)) <= 1e-12 * span
    # the state of the fit's last solve (the "trial" geometry) is measurably different: the check above
    # would fail if that state were carried
    last = _advance(lags["pred"], lags["trial"], c0, 0.0, 0.02, IP0, 1.01 * IP0)
    assert np.max(np.abs((last.psi - last.psi[0]) - (e.psi - e.psi[0]))) > 1e-6 * span
    assert f["last_solve_state_vs_center"]["max_dpsi_rel"] > 1e-6
    assert f["center_state_check"]["max_dpsi_rel"] <= 1e-12
    # the solver holds the center's equilibrium when the slice is recorded, and the next interval starts from it
    assert seen[1] == "center"
    s2 = out["slices"][2]
    assert s2["filter"]["absorbed"] is True
    e2 = _advance(lags["center"], lags["center"], got, 0.02, 0.04, 1.01 * IP0, 1.02 * IP0)
    sh2 = CM.correction_shapes(lags["center"], cfg, 1.02 * IP0)
    d2, b2 = CM.correction_response(lags["center"], sh2, CENTER_A, e2.rho_faces)
    got2 = _state(s2["carried_state"])
    assert np.max(np.abs(got2.psi - (e2.psi + d2))) <= 1e-12 * np.ptp(e2.psi + d2)


def test_filter_times_fit_only_at_the_anchor_slices(monkeypatch):
    lags = dict(pred=_lag(), center=_lag(sigma_scale=1.3), col0=_lag(sigma_scale=0.7), col1=_lag(sigma_scale=1.6),
                trial=_lag(sigma_scale=0.5))
    fm = FakeFM(lags)
    _install(monkeypatch, fm, _steady(lags["pred"], IP0), lambda f: f.lags[f.key])
    rec = _fake_reconstruct("converged", lags)
    monkeypatch.setattr(SE._fit, "reconstruct", rec)
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    times = [0.0, 0.02, 0.04, 0.06]
    st = SE.SeriesSettings(initial="relaxed", sawtooth="off", filter=True, filter_times=(0.04,))
    out = SE.run_series(fm, _slices(times, [IP0, 1.01 * IP0, 1.02 * IP0, 1.03 * IP0]), cfg, st)
    assert [s["filter"] is not None for s in out["slices"][1:]] == [False, True, False]
    assert out["slices"][2]["filter"]["absorbed"] is True and len(rec.calls) == 1
    assert st.filter_at(0.04) and not st.filter_at(0.02) and SE.SeriesSettings(filter=True).filter_at(0.02)
    assert out["settings"]["filter_times"] == (0.04,)
    with pytest.raises(ValueError, match="need filter=True"):
        SE.SeriesSettings(filter_times=(0.04,))
    with pytest.raises(ValueError, match="empty"):
        SE.SeriesSettings(filter=True, filter_times=())


def test_supplied_initial_profile_can_be_scaled_to_a_domain_current_fraction():
    lag = _lag()
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    rho = np.linspace(0.0, 1.0, 51)
    prof = (rho, 1.0e6 * (1.0 - 0.8 * rho ** 2))
    raw = SE._given_state(lag, SE.SeriesSettings(initial="from_profile", initial_profile=prof), cfg, IP0)
    st = SE._given_state(lag, SE.SeriesSettings(initial="from_profile", initial_profile=prof,
                                                initial_profile_domain_fraction=0.99), cfg, IP0)
    assert "profile_scale" not in raw.meta
    assert st.meta["enclosed_current_domain_edge"] == pytest.approx(0.99 * IP0, rel=1e-12)
    assert st.meta["profile_scale"] == pytest.approx(0.99 * IP0 / raw.meta["enclosed_current_domain_edge"], rel=1e-12)
    np.testing.assert_allclose(st.psi, st.meta["profile_scale"] * raw.psi, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match="initial_profile_domain_fraction"):
        SE.SeriesSettings(initial_profile_domain_fraction=1.5)
    # an enclosed fraction slightly above 1 only with an explicit tolerance; the excess is recorded
    with pytest.raises(ValueError, match="initial_profile_domain_fraction"):
        SE.SeriesSettings(initial_profile_domain_fraction=1.006)
    over = SE._given_state(lag, SE.SeriesSettings(initial="from_profile", initial_profile=prof,
                                                  initial_profile_domain_fraction=1.006,
                                                  initial_profile_domain_fraction_tolerance=0.01), cfg, IP0)
    assert over.meta["enclosed_current_domain_edge"] == pytest.approx(1.006 * IP0, rel=1e-12)
    assert over.meta["domain_fraction_excess"] == pytest.approx(0.006, rel=1e-12)
    assert st.meta["domain_fraction_excess"] == 0.0


@pytest.mark.parametrize("status", ["stalled_no_significant_descent", "max_iter", "failed_trials", "jacobian_failed",
                                    "converged_no_final_jacobian"])
def test_filter_absorbs_only_converged_fits(monkeypatch, status):
    out, lags, cfg, seen = _filter_setup(monkeypatch, status)
    c0 = _state(out["slices"][0]["carried_state"])
    e = _advance(lags["pred"], lags["pred"], c0, 0.0, 0.02, IP0, 1.01 * IP0)      # evolved, unfitted
    for k, s in enumerate(out["slices"][1:], start=1):
        f = s["filter"]
        assert f["status"] == status and f["status"] not in SE.ACCEPTED_FIT_STATUSES
        assert f["absorbed"] is False and "not absorbed" in f["note"]
        assert seen[k] == "pred"                        # the solver was restored to the evolved equilibrium
    got = _state(out["slices"][1]["carried_state"])
    assert np.max(np.abs(got.psi - e.psi)) <= 1e-12 * np.ptp(e.psi)
    assert got.psi_boundary == pytest.approx(e.psi_boundary, abs=1e-12 * np.ptp(e.psi))


def test_filter_absorbs_a_within_noise_fit_with_its_label(monkeypatch):
    """A fit stopped at the optimum to within the measured noise is absorbed like a converged one and keeps its
    label in the slice record."""
    out, lags, cfg, seen = _filter_setup(monkeypatch, "converged_within_noise")
    ref, _, _, _ = _filter_setup(monkeypatch, "converged")
    for a, b in zip(out["slices"][1:], ref["slices"][1:]):
        assert a["filter"]["status"] == "converged_within_noise" and a["filter"]["absorbed"] is True
        assert np.array_equal(np.asarray(a["carried_state"]["psi"]), np.asarray(b["carried_state"]["psi"]))
    assert seen[1] == "center"


def test_filter_with_a_fitted_ip_carries_the_state_at_the_fitted_current(monkeypatch):
    """With the Ip target fitted (``ip_rel`` in the fit's parameters) the absorbed state is the evolved state scaled
    to the fitted current plus the corrections, and the next interval starts from that current."""
    lags = dict(pred=_lag(), center=_lag(sigma_scale=1.3), col0=_lag(sigma_scale=0.7), col1=_lag(sigma_scale=1.6),
                trial=_lag(sigma_scale=0.5))
    fm = FakeFM(lags)
    _install(monkeypatch, fm, _steady(lags["pred"], IP0), lambda f: f.lags[f.key])
    k_ip = 1.004
    base = _fake_reconstruct("converged", lags)

    def rec(*a, **kw):
        out = base(*a, **kw)
        out.free = list(out.free) + ["ip_rel"]
        out.params = dict(out.params, ip_rel=k_ip - 1.0)
        return out

    monkeypatch.setattr(SE._fit, "reconstruct", rec)
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    out = SE.run_series(fm, _slices([0.0, 0.02, 0.04], [IP0, 1.01 * IP0, 1.02 * IP0]), cfg,
                        SE.SeriesSettings(initial="relaxed", sawtooth="off", filter=True))
    s1 = out["slices"][1]
    c0 = _state(out["slices"][0]["carried_state"])
    e = CM.scale_state(_advance(lags["pred"], lags["center"], c0, 0.0, 0.02, IP0, 1.01 * IP0), k_ip)
    shapes = CM.correction_shapes(lags["center"], cfg, 1.01 * IP0 * k_ip)
    dpsi, dpb = CM.correction_response(lags["center"], shapes, CENTER_A, e.rho_faces)
    got = _state(s1["carried_state"])
    assert np.max(np.abs(got.psi - (e.psi + dpsi))) <= 1e-12 * np.ptp(e.psi + dpsi)
    assert s1["ip_carried_A"] == pytest.approx(1.01 * IP0 * k_ip, rel=1e-14)
    assert s1["filter"]["absorbed"] is True


def _ip_offset_series(monkeypatch, ip_rel, ip_meas, n, filt=True, sigma_scale=1.0):
    """Stand-in filter fit (after a review's toy): it returns ``ip_rel`` and no corrections, and evaluates the state the
    way the fit does (``fit._scaled_state_fn``: to the target from the state's own recorded current)."""
    lag = _lag(sigma_scale=sigma_scale)
    fm = FakeFM(dict(pred=lag))

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        fm_.n_solves += 1
        fm_.F0_at_solve.append(fm_.F0)
        if state_fn is None:
            return _result(lag, _steady(lag, ip_meas))
        fn = SE._fit._scaled_state_fn(state_fn, params, fm_.ip_measured)
        fn(lag)
        return _result(lag, fn(lag))

    def reconstruct(fm_, kin, src, cfg, settings, params0=None, start=None, label="", state_fn=None, **kw):
        p = dict(a_corr=(0.0, 0.0, 0.0), ip_rel=ip_rel)
        st = SE._fit._scaled_state_fn(state_fn, p, fm_.ip_measured)(lag)
        center = dict(res=_result(lag, st), snap=dict(key="pred"), probe_correction=np.zeros(0))
        names = [f"a_corr_{k}" for k in range(3)] + ["ip_rel"]
        return SimpleNamespace(status="converged", theta=np.zeros(4), sigma=np.full(4, 0.01), free=names, params=p,
                               chi2={}, record=dict(center=center, specs={nm: dict(prior=(0.0, 0.02)) for nm in names},
                                                    n_solves=1, wall_s=0.0, final_stopping_tests=None,
                                                    stopping_summary=dict(status="converged")))

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    monkeypatch.setattr(SE._fit, "reconstruct", reconstruct)
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    out = SE.run_series(fm, _slices(np.arange(n) * 0.02, np.full(n, ip_meas)), cfg,
                        SE.SeriesSettings(initial="relaxed", sawtooth="off", filter=filt))
    assert out["status"] == "completed"
    return out, lag


@pytest.mark.parametrize("sigma_scale", [1.0, 10.0])
def test_a_constant_ip_offset_recovers_the_consistent_evolution(monkeypatch, sigma_scale):
    """Filter fits that find a constant fitted Ip 1 % below the measured one at every slice give the evolution of a
    series driven with that current all along (to round-off): the carried state is rescaled once, not at every
    slice, and the interval's Ip boundary follows the fitted current."""
    n, eps = 30, -0.01
    b, lag = _ip_offset_series(monkeypatch, eps, IP0, n, sigma_scale=sigma_scale)
    c, _ = _ip_offset_series(monkeypatch, 0.0, (1 + eps) * IP0, n, sigma_scale=sigma_scale)
    for sb, sc in zip(b["slices"][1:], c["slices"][1:]):
        pb, pc = _state(sb["carried_state"]), _state(sc["carried_state"])
        span = np.ptp(pc.psi)
        assert np.max(np.abs((pb.psi - pb.psi_boundary) - (pc.psi - pc.psi_boundary))) <= 1e-10 * span
        assert sb["ip_carried_A"] == pytest.approx((1 + eps) * IP0, rel=1e-12)
        assert sb["ip_offset_carried"] == pytest.approx(eps, rel=1e-12)
        assert sb["ip_convention"].startswith("carried fitted offset")
    s2 = b["slices"][2]["interval"]
    assert s2["ip_start_A"] == pytest.approx(s2["ip_end_A"], rel=1e-12)      # no ramp back to the measured Ip
    assert b["ip_convention"] == SE.IP_CONVENTION


def test_prediction_after_a_fitted_first_slice_has_no_ip_step(monkeypatch):
    """After a first slice whose fitted Ip is 3 % below the measured one, the next interval is driven at the carried
    offset (no ramp back to the measured Ip, which would put a loop-voltage step at the second slice)."""
    lag = _lag()
    fm = FakeFM(dict(pred=lag))
    st0 = _steady(lag, 0.97 * IP0)
    _install(monkeypatch, fm, st0, lambda f: f.lags["pred"])

    def reconstruct(fm_, kin, src, cfg, settings, params0=None, initial_profiles=None, label="", **kw):
        res = _result(lag, st0)
        res.assembly["ip"] = 0.97 * IP0
        center = dict(res=res, snap=dict(key="pred"), probe_correction=np.zeros(0))
        names = [f"a_corr_{k}" for k in range(3)] + ["ip_rel"]
        return SimpleNamespace(status="converged", theta=np.zeros(4), sigma=np.full(4, 0.1), free=names,
                               params=dict(a_corr=(0.0, 0.0, 0.0), ip_rel=-0.03), chi2={},
                               record=dict(center=center, specs={nm: dict(prior=(0.0, 1.0)) for nm in names},
                                           n_solves=5, wall_s=0.0, iterations=[], final_stopping_tests=None,
                                           stopping_summary=dict(status="converged")))

    monkeypatch.setattr(SE._fit, "reconstruct", reconstruct)
    out = SE.run_series(fm, _slices(np.arange(4) * 0.02, np.full(4, IP0)), CM.CurrentModelSettings(r0=GEO.R0),
                        SE.SeriesSettings(initial="mse_fit", sawtooth="off"))
    assert out["status"] == "completed"
    assert out["slices"][0]["ip_carried_A"] == pytest.approx(0.97 * IP0, rel=1e-12)
    for s_ in out["slices"][1:]:
        assert s_["interval"]["ip_start_A"] == pytest.approx(0.97 * IP0, rel=1e-12)
        assert s_["interval"]["ip_end_A"] == pytest.approx(0.97 * IP0, rel=1e-12)


def test_scale_state_scales_the_current_and_keeps_the_boundary_value():
    lag = _lag()
    st = _steady(lag, IP0)
    sc = CM.scale_state(st, 1.02)
    assert sc.psi_boundary == st.psi_boundary
    np.testing.assert_allclose(sc.psi - sc.psi_boundary, 1.02 * (st.psi - st.psi_boundary), rtol=1e-14)
    sol = CM.build_solver(lag, lag["jbs"], 1.02 * IP0)
    assert sol.profiles(sc)["ip_domain_edge"] == pytest.approx(
        1.02 * CM.build_solver(lag, lag["jbs"], IP0).profiles(st)["ip_domain_edge"], rel=1e-9)


# ============================================================== independent reference
def _reference_matrices(lag, n_nodes):
    """Vertex-centered finite volumes for a dpsi/dt = (k psi')' on rho in [0, 1] with
    k psi'(1) + A_layer dpsi_b/dt = Ip/F_lcfs (lumped layer at the last node).  Returns (M, K, e_last)."""
    geom = lag["geom"]
    x = np.linspace(0.0, 1.0, n_nodes)
    h = x[1] - x[0]

    def coef(r):
        g = geom.evaluate(np.atleast_1d(r), ["dvol_drho", "avg_grad_rho2_r2", "fpol", "avg_inv_r2"])
        vp = np.where(np.atleast_1d(r) == 0.0, 0.0, g["dvol_drho"])
        a = vp * g["avg_inv_r2"] * SIGMA / (2 * np.pi * g["fpol"])
        k = vp * g["avg_grad_rho2_r2"] / (2 * np.pi * MU0 * g["fpol"])
        return a, k

    mid = 0.5 * (x[1:] + x[:-1])
    _, kf = coef(mid)
    m = np.zeros(n_nodes)
    lo = np.concatenate(([0.0], mid))
    hi = np.concatenate((mid, [1.0]))
    for i in range(n_nodes):           # control-volume integral of a (Gauss-Legendre, 4 points)
        gp, gw = np.polynomial.legendre.leggauss(4)
        r = 0.5 * (hi[i] - lo[i]) * gp + 0.5 * (hi[i] + lo[i])
        m[i] = 0.5 * (hi[i] - lo[i]) * np.sum(gw * coef(r)[0])
    A_layer = SE._layer_integrals(lag, lag["sigma"], lag["jbs"])[0]
    m[-1] += A_layer
    K = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes - 1):
        c = kf[i] / h
        K[i, i] -= c
        K[i, i + 1] += c
        K[i + 1, i + 1] -= c
        K[i + 1, i] += c
    e = np.zeros(n_nodes)
    e[-1] = 1.0
    return x, m, K, e, A_layer


def _reference(lag, psi0_fn, times, ips, n_nodes=1201):
    """Exact-in-time solution of the semi-discrete reference with Ip linear between slices."""
    x, m, K, e, _ = _reference_matrices(lag, n_nodes)
    F_lcfs = float(lag["layer"].fpol_lcfs)
    y = psi0_fn(x)
    out = [y.copy()]
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        slope = (ips[k] - ips[k - 1]) / dt
        # augmented z = [psi, Ip, dIp/dt]:  psi' = M^-1 (K psi + e Ip / F),  Ip' = dIp/dt,  (dIp/dt)' = 0
        n = x.size
        G = np.zeros((n + 2, n + 2))
        G[:n, :n] = K / m[:, None]
        G[:n, n] = e / (m * F_lcfs)
        G[n, n + 1] = 1.0
        z = np.concatenate((y, [ips[k - 1], slope]))
        y = (expm(G * dt) @ z)[:n]
        out.append(y.copy())
    return x, out


def _psi0_fn(lag):
    """Analytic non-relaxed start: enclosed current I/F = (Ip0/F_lcfs) u(rho) with a peaked u, so that
    psi' = (I/F)/k; psi(0) = 0."""
    geom = lag["geom"]
    F_lcfs = float(lag["layer"].fpol_lcfs)

    def dpsi(r):
        g = geom.evaluate(np.atleast_1d(r), ["dvol_drho", "avg_grad_rho2_r2", "fpol"])
        k = g["dvol_drho"] * g["avg_grad_rho2_r2"] / (2 * np.pi * MU0 * g["fpol"])
        u = 1.0 - (1.0 - np.atleast_1d(r) ** 2) ** 3          # all of Ip inside the domain edge, no edge current
        return (IP0 / F_lcfs) * u / k

    gp, gw = np.polynomial.legendre.leggauss(8)

    def seg(a, b):                           # composite 8-point Gauss-Legendre on 1e-3-wide pieces
        n = max(1, int(np.ceil((b - a) / 1e-3)))
        e = np.linspace(a, b, n + 1)
        r = (0.5 * (e[1:] - e[:-1])[:, None] * gp + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
        return float(np.sum(np.tile(gw, n) * np.repeat(0.5 * np.diff(e), gp.size) * dpsi(r)))

    def psi(xs):
        xs = np.atleast_1d(np.asarray(xs, float))
        cum, out, last = 0.0, np.zeros_like(xs), 0.0
        for i, xv in enumerate(xs):          # xs increasing
            if xv > last:
                cum += seg(last, xv)
                last = xv
            out[i] = cum
        return out

    return psi


def _series_states(monkeypatch, lag, n_cells, times, ips, psi0_fn, dt_max=None):
    rf = CurrentState.uniform_faces(n_cells)
    rc = 0.5 * (rf[1:] + rf[:-1])
    vals = psi0_fn(np.concatenate((rc, [1.0])))
    st0 = CurrentState(rf, vals[:-1], float(vals[-1]), 0.0)
    fm = FakeFM(dict(pred=lag))
    _install(monkeypatch, fm, st0, lambda f: f.lags["pred"])
    st = SE.SeriesSettings(initial="relaxed", sawtooth="off")
    if dt_max is not None:
        st.dt_max = float(dt_max)
    out = SE.run_series(fm, _slices(times, ips), CM.CurrentModelSettings(r0=GEO.R0, n_cells=n_cells), st)
    assert out["status"] == "completed", out["status"]
    return rc, [_state(s["carried_state"]) for s in out["slices"]]


def _errors(rc, states, x, ref):
    prof, bnd = [], []
    for st, y in zip(states[1:], ref[1:]):
        yr = CubicSpline(x, y)(rc) - y[-1]
        yp = st.psi - st.psi_boundary
        prof.append(float(np.max(np.abs(yp - yr)) / np.ptp(yr)))
        db_ref = y[-1] - ref[0][-1]
        db = st.psi_boundary - states[0].psi_boundary
        bnd.append(float(abs(db - db_ref) / abs(db_ref)))
    return max(prof), max(bnd)


def test_series_chain_matches_an_independent_diffusion_reference(monkeypatch):
    lag = _lag(sigma_const=SIGMA)
    times = np.arange(6) * 0.02
    ips = IP0 * (1.0 + 0.05 * np.arange(6))
    psi0 = _psi0_fn(lag)
    x, ref = _reference(lag, psi0, times, ips)
    errs = {}
    for n in (100,):
        rc, states = _series_states(monkeypatch, lag, n, times, ips, psi0)
        errs[n] = _errors(rc, states, x, ref)
    assert errs[100][0] <= 1e-3 and errs[100][1] <= 1e-3, errs
    # discrimination: the same chain without the Ip ramp is far outside the bound
    rc, flat = _series_states(monkeypatch, lag, 100, times, np.full(6, IP0), psi0)
    e_flat = _errors(rc, flat, x, ref)
    assert max(e_flat) > 10 * 1e-3, e_flat


def test_series_chain_converges_in_space_against_the_reference(monkeypatch):
    """Grid convergence of the same chain against the same reference: at a quarter of the default time
    step (so the time error does not mask the spatial one), on 200, 400 and 800 cells, both errors (profile
    and boundary-flux change) fall by at least a factor 2 per doubling of the cells -- the first-order rate
    that the layer's coupling to the last cell guarantees (diffusion module)."""
    lag = _lag(sigma_const=SIGMA)
    times = np.arange(6) * 0.02
    ips = IP0 * (1.0 + 0.05 * np.arange(6))
    psi0 = _psi0_fn(lag)
    x, ref = _reference(lag, psi0, times, ips)
    dt = 0.25 * SE.SeriesSettings().dt_max
    errs = {}
    for n in (200, 400, 800):
        rc, states = _series_states(monkeypatch, lag, n, times, ips, psi0, dt_max=dt)
        errs[n] = _errors(rc, states, x, ref)
    for a, b in ((200, 400), (400, 800)):
        for i in (0, 1):
            assert errs[a][i] >= 2.0 * errs[b][i], (a, b, i, errs)


# ============================================================== sawteeth inside a series
def _sawtooth_series(monkeypatch, ip, mode, crash_times=None):
    lag = _lag()
    fm = FakeFM(dict(pred=lag))
    st0 = _steady(lag, ip)
    _install(monkeypatch, fm, st0, lambda f: f.lags["pred"])
    if crash_times is not None:
        # a measured crash record that covers the first slice's history (the first-slice model state is
        # mocked here; the record before t0 only satisfies the coverage requirement)
        crash_times = [round(-1.6 + 0.05 * k, 6) for k in range(32)] + list(crash_times)
    s = SE.SeriesSettings(initial="relaxed", sawtooth=mode, sawtooth_log_strength=np.log(400.0),
                          crash_times=crash_times)
    times = np.arange(4) * 0.02
    out = SE.run_series(fm, _slices(times, np.full(4, ip)), CM.CurrentModelSettings(r0=GEO.R0), s)
    assert out["status"] == "completed"
    return out, lag, st0


def _q(lag, state, ip):
    return CM.build_solver(lag, lag["jbs"], ip).profiles(state)["q"]


def _gauge_free(d):
    return np.asarray(d["psi"]) - d["psi_boundary"]


def test_sawteeth_in_a_series_identity_where_q_at_least_one(monkeypatch):
    on, lag, st0 = _sawtooth_series(monkeypatch, IP0, "periodic")
    assert np.min(_q(lag, st0, IP0)) >= 1.0
    off, _, _ = _sawtooth_series(monkeypatch, IP0, "off")
    crashes = [c for s in on["slices"][1:] for c in s["crashes"]]
    assert len(crashes) >= 2                                 # crashes were scheduled inside the series ...
    assert not any(c["active"] for c in crashes)
    assert all(c["dpsi_max"] == 0.0 for c in crashes)        # ... and the operator changed nothing (exact)
    # the states equal the crash-free series' to round-off (the step is split at each crash time, so the
    # arithmetic differs; bound 1e-12 of the span)
    for a, b in zip(on["slices"], off["slices"]):
        for key in ("carried_state", "state_psi"):
            if key in a:
                pa, pb = np.asarray(a[key]["psi"]), np.asarray(b[key]["psi"])
                span = np.ptp(pb)
                assert np.max(np.abs(pa - pb)) <= 1e-12 * span, key
                assert abs(a[key]["psi_boundary"] - b[key]["psi_boundary"]) <= 1e-12 * span, key


def test_sawteeth_in_a_series_act_where_q_below_one_and_slice_state_is_continuous(monkeypatch):
    """One measured crash just BEFORE the first evolved slice time (t1 - eps) or just AFTER it (t1 + eps):
    the carried (instantaneous) state jumps by the whole crash, the reported slice state only by the
    crash's share of the interval (about eps/interval = 0.5 %; bound 5 %, set before the first run)."""
    eps = 1.0e-4
    t1 = 0.02
    ip = 1.8 * IP0                                          # q < 1 on axis in the steady state
    before, lag, st0 = _sawtooth_series(monkeypatch, ip, "crash_times", crash_times=[t1 - eps])
    assert np.min(_q(lag, st0, ip)) < 1.0
    after, _, _ = _sawtooth_series(monkeypatch, ip, "crash_times", crash_times=[t1 + eps])
    off, _, _ = _sawtooth_series(monkeypatch, ip, "off")
    b1, a1, o1 = before["slices"][1], after["slices"][1], off["slices"][1]
    assert [c["active"] for c in b1["crashes"]] == [True] and a1["crashes"] == []
    assert [c["active"] for c in after["slices"][2]["crashes"]] == [True]
    span = np.ptp(_gauge_free(o1["carried_state"]))
    jump = np.max(np.abs(_gauge_free(b1["carried_state"]) - _gauge_free(a1["carried_state"])))
    assert jump > 1e-3 * span                                # the operator acted on the state
    # the crash-free slice is the instantaneous state (no crash in its interval)
    assert np.array_equal(np.asarray(a1["state_psi"]["psi"]), np.asarray(o1["state_psi"]["psi"]))
    d_slice = np.max(np.abs(_gauge_free(b1["state_psi"]) - _gauge_free(a1["state_psi"])))
    assert d_slice <= 0.05 * jump, (d_slice, jump)


# ============================================================== time-varying F0
def test_each_slice_is_solved_with_its_own_F0(monkeypatch):
    lag = _lag()
    fm = FakeFM(dict(pred=lag))
    _install(monkeypatch, fm, _steady(lag, IP0), lambda f: f.lags["pred"])
    F0s = [GEO.F0, GEO.F0 * 1.004, None, GEO.F0 * 0.997]
    out = SE.run_series(fm, _slices(np.arange(4) * 0.02, np.full(4, IP0), F0s), CM.CurrentModelSettings(r0=GEO.R0),
                        SE.SeriesSettings(initial="relaxed", sawtooth="off"))
    assert fm.F0_at_solve == [GEO.F0, GEO.F0 * 1.004, GEO.F0 * 1.004, GEO.F0 * 0.997]
    assert [s["vacuum_F0"]["F0"] for s in out["slices"]] == fm.F0_at_solve
    assert out["slices"][2]["vacuum_F0"]["source"].startswith("not given")


# ============================================================== first slice: fit status, strength, measured crashes
def _initial_fit_series(monkeypatch, status, policy="refuse", log_strength=np.log(300.0), sawtooth="periodic"):
    """Series with ``initial="mse_fit"`` whose first-slice fit is mocked: it ends with ``status`` at the
    steady state of the slice's geometry and returns ``log_strength`` as its fitted strength."""
    lag = _lag()
    fm = FakeFM(dict(pred=lag))
    st0 = _steady(lag, IP0)
    _install(monkeypatch, fm, st0, lambda f: f.lags["pred"])
    seen = {}

    def reconstruct(fm_, kin, src, cfg, settings, params0=None, initial_profiles=None, label="", **kw):
        seen.update(free=tuple(settings.free), cfg=cfg, params0=dict(params0))
        names = (["log_strength"] if "log_strength" in settings.free else []) + [f"a_corr_{k}" for k in range(3)]
        center = dict(res=_result(lag, st0), snap=dict(key="pred"), probe_correction=np.zeros(0))
        params = dict(log_strength=log_strength, f=params0.get("f"), a_corr=(0.0, 0.0, 0.0))
        return SimpleNamespace(status=status, theta=np.zeros(len(names)), sigma=np.full(len(names), 0.1), free=names,
                               params=params, chi2={},
                               record=dict(center=center, specs={n: dict(prior=(0.0, 1.0)) for n in names},
                                           n_solves=5, wall_s=0.0, iterations=[], final_stopping_tests=None,
                                           stopping_summary=dict(status=status)))

    monkeypatch.setattr(SE._fit, "reconstruct", reconstruct)
    s = SE.SeriesSettings(initial="mse_fit", sawtooth=sawtooth, initial_fit_nonconverged=policy,
                          crash_times=([round(-1.6 + 0.05 * k, 6) for k in range(33)] if sawtooth == "crash_times"
                                       else None))
    out = SE.run_series(fm, _slices(np.arange(3) * 0.02, np.full(3, IP0)), CM.CurrentModelSettings(r0=GEO.R0), s)
    return out, seen


@pytest.mark.parametrize("status", ["stalled_no_significant_descent", "max_iter", "jacobian_failed", "failed_trials"])
def test_series_refuses_to_start_from_a_nonconverged_first_slice_fit(monkeypatch, status):
    out, _ = _initial_fit_series(monkeypatch, status)
    assert out["status"] == "refused_initial_fit_not_converged" and status in out["reason"]
    assert len(out["slices"]) == 1 and out["slices"][0]["status"] == "initial_fit_not_converged"
    lab, _ = _initial_fit_series(monkeypatch, status, policy="proceed_labeled")
    assert lab["status"] == "completed" and len(lab["slices"]) == 3
    assert lab["initial_fit_nonconverged_start"]["status"] == status
    assert status in lab["slices"][0]["fit"]["start_label"]
    ok, _ = _initial_fit_series(monkeypatch, "converged")
    assert ok["status"] == "completed" and ok["slices"][0]["fit"]["start_label"] is None
    assert "initial_fit_nonconverged_start" not in ok
    with pytest.raises(ValueError):
        SE.SeriesSettings(initial_fit_nonconverged="maybe")


def test_series_starts_from_a_within_noise_first_slice_fit_with_its_label(monkeypatch):
    out, _ = _initial_fit_series(monkeypatch, "converged_within_noise")
    assert out["status"] == "completed" and len(out["slices"]) == 3
    assert out["slices"][0]["fit"]["status"] == "converged_within_noise"
    assert "converged_within_noise" in out["slices"][0]["fit"]["start_label"]
    assert "initial_fit_nonconverged_start" not in out


def test_first_slice_fit_sets_the_periodic_crash_train(monkeypatch):
    """With periodic sawteeth the first-slice fit has ``log_strength`` free by default, and its fitted value
    (not the prior) sets the period of the series' crash train."""
    assert SE.SeriesSettings(sawtooth="periodic").initial_fit.free == ("log_strength", "a_corr")
    assert SE.SeriesSettings().initial_fit.free == ("a_corr",)          # default: no prescribed train
    assert SE.SeriesSettings(sawtooth="off").initial_fit.free == ("a_corr",)
    with pytest.raises(ValueError):
        SE.SeriesSettings(sawtooth="off", initial_fit=SE._fit.FitSettings(free=("log_strength",)))
    ls = np.log(250.0)
    out, seen = _initial_fit_series(monkeypatch, "converged", log_strength=ls)
    assert "log_strength" in seen["free"]
    ct = out["crash_times_used"]
    assert ct["log_strength"] == ls and ct["log_strength_source"].startswith("fitted at the first slice")
    s = SE.SeriesSettings(sawtooth="periodic")
    assert ct["period_s"] == pytest.approx(SE._period(s, ls), rel=1e-15)
    assert ct["period_s"] != pytest.approx(SE._period(s), rel=1e-3)


def test_measured_crash_times_start_the_series(monkeypatch):
    """With measured crash times the first slice is solved with the measured-crash model state over the
    first interval's length before t0 (window), the strength of the measured mean interval, and a fit
    with only the corrections free; a record that does not cover the history is refused with the reason."""
    out, seen = _initial_fit_series(monkeypatch, "converged", sawtooth="crash_times")
    assert out["status"] == "completed"
    assert seen["free"] == ("a_corr",)
    cfg0 = seen["cfg"]
    assert tuple(cfg0.window) == pytest.approx((-0.02, 0.0)) and cfg0.crash_times is not None
    rec = out["first_slice_crash_start"]
    assert rec["mean_interval_s"] == pytest.approx(0.05, rel=1e-9)
    from bouquet.kinrecon.current import sawtooth as SAW
    assert seen["params0"]["log_strength"] == pytest.approx(np.log(SAW.crash_strength(0.37, 1 / 0.05)), rel=1e-12)
    lag = _lag()
    fm = FakeFM(dict(pred=lag))
    _install(monkeypatch, fm, _steady(lag, IP0), lambda f: f.lags["pred"])
    short = SE.SeriesSettings(initial="relaxed", sawtooth="crash_times", crash_times=[-0.5, -0.4, 0.01])
    with pytest.raises(ValueError, match="cannot start the series"):
        SE.run_series(fm, _slices(np.arange(3) * 0.02, np.full(3, IP0)), CM.CurrentModelSettings(r0=GEO.R0), short)


def test_measured_crash_start_runs_the_model(monkeypatch):
    """The first-slice settings of a measured-crash series drive the current model's measured-crash slice
    run (synthetic geometry, no equilibrium solve): the state is the window average over the window."""
    lag = _lag()
    s = SE.SeriesSettings(initial="relaxed", sawtooth="crash_times",
                          crash_times=[round(-1.6 + 0.05 * k, 6) for k in range(33)])
    cfg0, p0, rec = SE._measured_crash_start(s, CM.CurrentModelSettings(r0=GEO.R0), np.array([0.0, 0.02]))
    from bouquet.kinrecon.current import sawtooth as SAW
    sol = CM.build_solver(lag, lag["jbs"], 1.8 * IP0)
    rf = CurrentState.uniform_faces(cfg0.n_cells)
    lam = float(np.exp(p0["log_strength"]))
    period = 1.0 / float(SAW.rate_from_strength(lam, p0["f"]))
    assert period == pytest.approx(rec["mean_interval_s"], rel=1e-12)
    sched = SAW.schedule_for_slice(cfg0.crash_times, cfg0.window, history=cfg0.history)
    sim = SAW.simulate_slice(sol, rf, sched, p0["f"], dt=period / cfg0.sawtooth_nsteps,
                             nsteps_orbit=cfg0.sawtooth_nsteps)
    assert sim["orbit"].converged and sched.n_crashes_history >= 30
    assert any(e.record.active for e in sim["events"])


# ============================================================== outer layer |F| in time
def test_interval_layer_fpol_is_linear_in_time_and_matches_both_slices():
    lag0 = _lag()
    lag1 = _lag()
    f0 = float(lag0["layer"].fpol_lcfs)
    lag1["layer"] = SimpleNamespace(**{k: getattr(lag0["layer"], k) for k in dir(lag0["layer"]) if not k.startswith("_")})
    lag1["layer"].fpol_lcfs = 1.006 * f0
    lag1["layer"].integral = lag0["layer"].integral
    lag1["layer"].conductance_density = lag0["layer"].conductance_density
    lag1["layer"].source_density = lag0["layer"].source_density
    jni = lag0["jbs"]
    rf = CurrentState.uniform_faces(100)
    sol, info = SE.interval_solver(lag0, lag1, jni, jni, 0.0, 0.02, IP0, 1.01 * IP0)
    c0, c1 = sol.coefficients(0.0, rf), sol.coefficients(0.02, rf)
    assert c0.F_lcfs == f0 and c1.F_lcfs == 1.006 * f0          # each end has its slice's own |F|
    assert c0.bc_value / c0.F_lcfs == IP0 / f0 and c1.bc_value / c1.F_lcfs == pytest.approx(1.01 * IP0 / (1.006 * f0),
                                                                                             rel=1e-15)
    assert info["layer_fpol"] == "linear" and info["fpol_lcfs_change"] == pytest.approx(0.006 * f0, rel=1e-12)
    solm, infom = SE.interval_solver(lag0, lag1, jni, jni, 0.0, 0.02, IP0, 1.01 * IP0, "interval_mean")
    assert solm.coefficients(0.0, rf).F_lcfs == pytest.approx(1.003 * f0, rel=1e-15)
    with pytest.raises(ValueError):
        SE.SeriesSettings(layer_fpol="cubic")


# ============================================================== relaxed-start guard
def test_relaxed_start_guard_warns_outside_its_range(monkeypatch):
    lag = _lag(q0=1.1)
    fm = FakeFM(dict(pred=lag))
    _install(monkeypatch, fm, _steady(lag, IP0), lambda f: f.lags["pred"])
    sl = _slices(np.arange(2) * 0.02, np.full(2, IP0))
    out = SE.run_series(fm, sl, CM.CurrentModelSettings(r0=GEO.R0), SE.SeriesSettings(initial="relaxed", sawtooth="off"))
    g = out["relaxed_start_guard"]
    assert g["warned"] is False and g["q_min"] == pytest.approx(np.min(np.asarray(lag["phys"]["q"])[
        np.asarray(lag["phys"]["rho"]) <= 0.95]))
    sl = _slices(np.arange(2) * 0.02, np.full(2, IP0))
    sl[0].meta["reference_q_min"] = 1.5
    with pytest.warns(RuntimeWarning, match="q_min = 1.5 > 1.2"):
        out = SE.run_series(fm, sl, CM.CurrentModelSettings(r0=GEO.R0),
                            SE.SeriesSettings(initial="relaxed", sawtooth="off"))
    assert out["relaxed_start_guard"]["warned"] and out["status"] == "completed"


def test_first_crash_phase_moves_the_periodic_train():
    s1 = SE.SeriesSettings(sawtooth_log_strength=np.log(400.0))
    s4 = SE.SeriesSettings(sawtooth_log_strength=np.log(400.0), sawtooth_first_crash_phase=0.25)
    tau = SE._period(s1)
    a, b = SE._crash_times(s1, 0.0, 0.1), SE._crash_times(s4, 0.0, 0.1)
    assert np.array_equal(a, tau * np.arange(1, a.size + 1))              # the default: one period after t0
    assert np.allclose(b, a - 0.75 * tau, rtol=0, atol=1e-15)
    with pytest.raises(ValueError):
        SE.SeriesSettings(sawtooth_first_crash_phase=0.0)
