"""rho_tor,norm coordinates, solver knots, the q check, kinetic inputs and the time-dependent path
(no OFT; synthetic, separatrix-like geometry as in ``test_current_outer_layer.py``).

Bounds are set from the formulas (exact identities to round-off) or stated beside each assertion.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon import coordinates as CO  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import kinetics as KI  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import MU0, CurrentState, FluxSurfaceGeometry, OuterLayerGeometry  # noqa: E402
from bouquet.kinrecon.current import sawtooth as SAW  # noqa: E402

R0, B0, F0, DPSI, IP, C = 1.7, 2.0, 3.4, 0.18, 1.2e6, 0.995


def F(x):
    return F0 * (1.0 - 0.02 * x)


def g1(x):
    return (1.0 + 0.08 * x) / R0 ** 2


def q_ref(x, q0=1.1):
    return q0 + 2.2 * x ** 2 - 0.5 * np.log(np.maximum(1.0 - x, 1e-300))


def _lag(q0=1.1, sigma_scale=1.0):
    """A lagged-input dict like ``current_model.evaluate_geometry`` returns, on analytic surfaces."""
    xs = np.linspace(np.sqrt(1e-3), np.sqrt(C), 200) ** 2
    xs[-1] = C
    q = q_ref(xs, q0)
    vps = 4 * np.pi ** 2 * q / (F(xs) * g1(xs))
    iref = IP * (1.6 * xs - 0.6 * xs ** 2)
    bp2 = 2 * np.pi * MU0 * iref / vps
    geom = FluxSurfaceGeometry.from_psi_coordinates(xs, 0.0, DPSI, vps, bp2, g1(xs), F(xs), B0=B0, R0=R0, nrho=201,
                                                    psi_n_edge=C)
    xl = 1.0 - (1.0 - C) * np.geomspace(1.0, 1e-2, 40)
    xl[0] = C
    ql = q_ref(xl, q0)
    vl = 4 * np.pi ** 2 * ql / (F(xl) * g1(xl))
    layer = OuterLayerGeometry.from_psi_coordinates(xl, 0.0, DPSI, vl, g1(xl), F(xl),
                                                    avg_grad_psi2_r2=2 * np.pi * MU0 * IP * (1.6 * xl - 0.6 * xl ** 2) / vl,
                                                    fpol_lcfs=F(1.0))
    ge = np.unique(np.concatenate((np.linspace(1e-3, C, 300), xl)))
    sig = sigma_scale * 1.0e8 * (0.02 + 0.98 * np.clip(1.0 - ge ** 2, 0.0, None) ** 1.5)
    to_rho, to_psin = CM._psin_to_rho(np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"]))
    rmap = CO.rho_map_from_geometry(geom, layer)
    phys = dict(psi_n=ge, rho=rmap.rho_of_psin(ge), q=q_ref(ge, q0), F=F(ge), inv_r2=g1(ge), b2=F(ge) ** 2 * g1(ge))
    return dict(geom=geom, layer=layer, psi_n_edge=C, to_rho=to_rho, to_psin=to_psin, rho_map=rmap, phys=phys,
                sigma=sig, jbs=B0 * 4.0e5 * np.exp(-((ge - 0.975) / 0.02) ** 2))


# ============================================================== rho map and knots
def test_rho_map_identities_and_round_trip():
    lag = _lag()
    m = lag["rho_map"]
    geom = lag["geom"]
    # inside the domain: rho = rho_dom sqrt(Phi(c)/Phi_LCFS) exactly (table construction)
    xs, rs = np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"])
    np.testing.assert_allclose(m.rho_of_psin(xs), rs * m.rho_edge, rtol=1e-12)
    assert 0.0 < m.rho_edge < 1.0 and m.phi_lcfs > m.phi_edge
    # Phi_LCFS = Phi(c) + int F <1/R^2>/(2 pi) dV over the layer (independent quadrature of the same tabulation)
    L = lag["layer"]
    dens = L.fpol * L.avg_inv_r2 / (2 * np.pi)
    assert m.phi_lcfs == pytest.approx(geom.phi_b + L.integral(dens), rel=1e-12)
    r = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(m.rho_of_psin(m.psin_of_rho(r)), r, atol=1e-10)
    with pytest.raises(ValueError):
        m.rho_of_psin([1.2])


@pytest.mark.parametrize("rule", CO.KNOT_RULES)
def test_knots_monotone_cover_axis_edge_and_layer(rule):
    m = _lag()["rho_map"]
    s = CO.KnotSettings(rule=rule)
    x, rec = CO.solver_knots(m, s)
    assert x[0] == 0.0 and x[-1] == 1.0 and np.all(np.diff(x) > 0)
    assert np.any(np.abs(x - C) < 1e-15)                         # the domain edge is a knot
    assert np.count_nonzero((x > C) & (x < 1.0)) == s.n_layer_knots
    assert rec["n_layer"] == s.n_layer_knots + 1
    if rule == "rho_tor_uniform":
        rk = m.rho_of_psin(x[x < C])
        # uniform in rho: interior spacing constant to round-off of the map inversion
        np.testing.assert_allclose(np.diff(rk), 1.0 / (s.n_knots - 1), rtol=1e-6)
        assert x[1] < 1e-3                                        # packed at the axis (psi_N ~ rho^2)


def test_knots_change_with_the_equilibrium_only_for_the_rho_rule():
    a, b = _lag(q0=1.1)["rho_map"], _lag(q0=0.8)["rho_map"]
    xa, _ = CO.solver_knots(a, CO.KnotSettings())
    xb, _ = CO.solver_knots(b, CO.KnotSettings())
    assert np.max(np.abs(xa - xb)) > 1e-3
    for rule in ("sqrt_psin_uniform", "psin_uniform"):
        np.testing.assert_array_equal(CO.solver_knots(a, CO.KnotSettings(rule=rule))[0],
                                      CO.solver_knots(b, CO.KnotSettings(rule=rule))[0])


# ============================================================== q check
def test_q_check_passes_on_identical_profiles_and_fails_on_a_core_mismatch():
    r = np.linspace(0.0, 1.0, 101)
    q = 1.0 + 2.5 * r ** 2
    ok = CO.q_consistency(r, q, r, q)
    assert ok["passed"] and ok["max_rel"] < 1e-14
    # the kind of mismatch the q check must catch: an equilibrium with q0 ~ 0.965 against a model
    # whose q has a reversed core with q(0) ~ 1.7 and a minimum ~ 0.97 near rho 0.11
    qm = 0.97 + 0.73 * np.exp(-(r / 0.06) ** 2) + 2.5 * np.clip(r - 0.11, 0, None) ** 2
    qe = 0.965 + 2.5 * np.clip(r - 0.11, 0, None) ** 2
    bad = CO.q_consistency(r, qe, r, qm)
    assert not bad["passed"] and bad["max_rel"] > 0.01
    # 1 % everywhere is exactly at the criterion's edge; 1.2 % fails
    assert CO.q_consistency(r, 1.012 * q, r, q)["passed"] is False
    assert CO.q_consistency(r, 1.009 * q, r, q)["passed"] is True
    # tables that do not cover the checked range are a failure, never a pass
    assert CO.q_consistency(r[r < 0.5], q[r < 0.5], r, q)["passed"] is False


def test_axis_q_is_the_regular_extrapolation_and_ignores_near_axis_artifacts():
    """The axis value is the regular-singular extrapolation q0 + q2 rho^2 + q4 rho^4 fitted on the axis
    window: exact for such a profile, and unaffected by scatter on the innermost surfaces (outside the
    window), where the raw innermost value is off by percent."""
    r = np.linspace(0.0244, 0.95, 186)
    q = 0.87 + 0.9 * r ** 2 - 0.3 * r ** 4
    a = CO.axis_q(r, q)
    assert a["q0"] == pytest.approx(0.87, rel=1e-12) and a["rms_rel"] < 1e-13
    qa = q.copy()
    inner = r < 0.045
    qa[inner] *= 1.0 + 0.02 * np.sin(97.0 * r[inner])          # trace scatter near the axis (outside the window)
    assert abs(qa[0] / 0.87 - 1) > 0.01
    assert CO.axis_q(r, qa)["q0"] == pytest.approx(0.87, rel=1e-12)
    with pytest.raises(ValueError):
        CO.axis_q(r[r < 0.06], q[r < 0.06])                    # too few points in the window


def test_q_check_tests_the_extrapolated_axis_at_the_same_criterion():
    """A model q that agrees with the equilibrium's to < 1 % on the interior range but bends away inside
    it (axis values differing by > 1 %) fails through the axis test when it is on; by default
    the axis values are recorded, not tested (the interior criterion is unchanged)."""
    re = np.linspace(0.0244, 0.95, 186)
    qe = 1.0 + 0.8 * re ** 2
    rm = np.linspace(0.0, 1.0, 101)
    qm = 1.0 + 0.8 * rm ** 2 - 0.025 * np.exp(-(rm / 0.05) ** 2)
    on = CO.QCheckSettings(test_axis=True)
    out = CO.q_consistency(re, qe, rm, qm, on)
    assert out["interior_passed"] is True and out["max_rel"] < 0.01
    assert out["axis"]["rel"] > 0.01 and out["axis"]["passed"] is False and out["passed"] is False
    off = CO.q_consistency(re, qe, rm, qm, CO.QCheckSettings(test_axis=False))
    assert off["passed"] is True and off["axis"]["tested"] is False
    same = CO.q_consistency(re, qe, rm, 1.0 + 0.8 * rm ** 2, on)
    assert same["passed"] and same["axis"]["rel"] < 1e-12
    dflt = CO.q_consistency(re, qe, rm, qm)                   # default: axis values recorded, not tested
    assert dflt["passed"] is True and dflt["axis"]["tested"] is False and dflt["axis"]["rel"] > 0.01


# ============================================================== kinetic inputs
def _kin_arrays():
    x = np.linspace(0.0, 1.0, 51)
    ne = 5e19 * (1.1 - 0.6 * x ** 2)
    te = 3e3 * (1.05 - x ** 2) + 50
    ti = 2.5e3 * (1.05 - x ** 2) + 60
    nfast = 4e18 * np.exp(-(x / 0.4) ** 2)
    return x, ne, te, ti, nfast


def test_quasineutrality_with_fast_ions_and_zeff_identity():
    x, ne, te, ti, nf = _kin_arrays()
    zvb = 1.8 + 0.4 * x
    k = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"vb": ("zeff", zvb)}, zeff_source="vb", z_imp=6.0,
                                   n_fast=nf, n_fast_source="provider")
    np.testing.assert_allclose(k.ni + 6.0 * k.nimp + k.n_fast, ne, rtol=1e-14)                       # (K1)
    np.testing.assert_allclose((k.ni + 36.0 * k.nimp + k.n_fast) / ne, zvb, rtol=1e-13)               # (K2)
    # the fast ions are removed from the thermal ions, not from the impurity
    k0 = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"vb": ("zeff", zvb)}, zeff_source="vb", z_imp=6.0)
    np.testing.assert_allclose(k0.ni - k.ni, nf, rtol=1e-12)
    np.testing.assert_allclose(k0.nimp, k.nimp, rtol=1e-14)
    p = k.thermal_pressure_at(rho=x)
    np.testing.assert_allclose(p, KI.EC * (ne * te + (k.ni + k.nimp) * ti), rtol=1e-12)
    assert k.coordinate == "rho_tor_norm" and "quasineutrality" in k.provenance["ni"]


def test_zeff_source_is_required_and_disagreement_recorded():
    x, ne, te, ti, nf = _kin_arrays()
    zvb = np.full_like(x, 2.4)
    nc = ne * 0.6 / 30.0                                   # CER-like carbon density -> Z_eff = 1.6
    cands = {"vb": ("zeff", zvb), "cer": ("impurity_density", nc)}
    with pytest.raises(ValueError, match="no\\s+default"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=cands, zeff_source=None, z_imp=6.0)
    k = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=cands, zeff_source="cer", z_imp=6.0)
    np.testing.assert_allclose(k.zeff, 1.6, rtol=1e-13)
    d = k.record["pairwise_disagreement"]["cer vs vb"]
    assert d["median_rel"] == pytest.approx(1.6 / 2.4 - 1.0, rel=1e-12)


def test_kinetic_refusals_without_clipping():
    x, ne, te, ti, nf = _kin_arrays()
    with pytest.raises(ValueError, match="< 1"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"z": ("zeff", np.full_like(x, 0.9))},
                                   zeff_source="z", z_imp=6.0)
    with pytest.raises(ValueError, match="<= 0"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates={"z": ("zeff", np.full_like(x, 1.5))},
                                   zeff_source="z", z_imp=6.0, n_fast=0.99 * ne)
    with pytest.raises(ValueError, match="cover"):
        KI.assemble_kinetic_inputs(x[:40], ne[:40], te[:40], ti[:40],
                                   zeff_candidates={"z": ("zeff", np.full(40, 1.5))}, zeff_source="z", z_imp=6.0)


def test_psi_n_inputs_mapped_through_the_producers_table_or_kept():
    x, ne, te, ti, nf = _kin_arrays()                      # read as psi_N of a foreign equilibrium
    tab = (np.linspace(0, 1, 101), np.sqrt(np.linspace(0, 1, 101)) ** 1.0)       # rho = sqrt(psi_N)
    z = {"z": ("zeff", np.full_like(x, 1.5))}
    k = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=z, zeff_source="z", z_imp=6.0, coordinate="psi_n",
                                   psi_n_to_rho=tab, mapping_source="the profile fit's equilibrium")
    assert k.coordinate == "rho_tor_norm" and "own table" in k.record["mapping"]
    np.testing.assert_allclose(k.profiles_at(rho=np.sqrt(0.36))["ne"], np.interp(0.36, x, ne), rtol=2e-3)
    k2 = KI.assemble_kinetic_inputs(x, ne, te, ti, zeff_candidates=z, zeff_source="z", z_imp=6.0, coordinate="psi_n")
    assert k2.coordinate == "psi_n" and "ours" in k2.record["mapping"]


# ============================================================== optimizer stopping logic
def test_one_damped_step_is_not_convergence(monkeypatch):
    """A heavily damped first step decreases chi^2 by < 1e-3 relative (the previous rule's stop); the
    fit must continue to the MAP point, which it reaches to round-off on this linear problem."""
    import test_fit_unit as TU
    fm = TU.FakeFM()
    TU._install(monkeypatch, fm)
    kin, src = TU._kin()
    st = FIT.FitSettings(free=("log_strength",), lm_lambda0=1e4, max_iter=40)
    out = FIT.reconstruct(fm, kin, src, CM.CurrentModelSettings(r0=1.0), st, params0=dict(log_strength=0.5))
    th, cov = TU._expected(FIT.PARAMETERS["log_strength"]["prior"][1])
    it0 = out.record["iterations"][0]
    acc = [t for t in it0["trials"] if t["status"] == "consistent"][-1]
    assert acc["realized_dchi2"] < 1e-3 * it0["chi2"]               # what the previous rule called converged
    assert out.status == "converged"
    assert sum(1 for e in out.record["iterations"] if e.get("accepted")) > 1
    assert abs(out.theta[0] - th[0]) < 1e-6


def test_stopping_tests_need_a_small_undamped_step():
    A = np.array([[4.0]])
    g = np.array([-4.0])                      # undamped GN step = +1 = 2 posterior sigma
    st = FIT.FitSettings()
    r = FIT.stopping_tests(A, g, st, 0.0, [1e-6, 1e-6])
    assert not r["converged"]                 # small relative decreases alone are not enough
    r = FIT.stopping_tests(A, np.array([-0.04]), st, 0.0, [])
    assert r["converged"] and "Gauss-Newton" in r["criterion"]


# ============================================================== time-dependent path
def _interval(lag, jni, t0, t1, ip0, ip1):
    return SE.interval_solver(lag, lag, jni, jni, t0, t1, ip0, ip1)[0]


def test_sawtooth_inactive_when_q_at_least_one():
    lag = _lag(q0=1.3)
    jni = lag["jbs"]
    sol = _interval(lag, jni, 0.0, 0.05, IP, IP)
    rf = CurrentState.uniform_faces(100)
    st = CM.build_solver(lag, jni, IP).steady_state(rf)
    st = CurrentState(rf, st.psi, st.psi_boundary, 0.0)
    q = CM.build_solver(lag, jni, IP).profiles(st)["q"]
    assert np.min(q) >= 1.0
    s = SE.SeriesSettings(initial="relaxed", sawtooth="periodic", sawtooth_log_strength=np.log(50.0))
    crashes = SE._crash_times(s, 0.0, 0.05)
    assert crashes.size >= 2 and crashes[0] < 0.05
    e1, ev, _ = SE.advance_interval(sol, st, 0.05, s, crashes, 1e-3)
    assert len(ev) >= 1 and not any(e.record.active for e in ev)
    # the operator changed nothing at any crash time (exact identity, not round-off)
    assert all(e.record.dpsi_max == 0.0 for e in ev)


def test_evolution_from_true_state_reproduces_and_relaxed_does_not():
    """Manufactured: a non-relaxed start (the steady state of a 3x more resistive core) evolved by ONE
    fine-step advance (dt 0.1 ms, Ip ramp) is the truth; the slice-by-slice chain of the series
    (20 ms intervals, dt 1 ms, the state carried between intervals) from the true state reproduces it
    (relative error of psi' below 1e-4: TR-BDF2 at 10x the step), the relaxed start does not (error
    above 1e-2 at the first slices)."""
    lag = _lag()
    jni = lag["jbs"]
    rf = CurrentState.uniform_faces(100)
    hot = CM.build_solver(lag, jni, IP, sigma=lag["sigma"] * np.where(lag["phys"]["psi_n"] < 0.3, 1 / 3.0, 1.0))
    s_true0 = hot.steady_state(rf)
    s_true0 = CurrentState(rf, s_true0.psi, s_true0.psi_boundary, 0.0)
    times = np.arange(0.0, 0.2001, 0.02)
    ips = IP * (1.0 + 0.1 * times)
    truth = [s_true0]
    s = s_true0
    sol_all = _interval(lag, jni, 0.0, times[-1], ips[0], ips[-1])
    for t in times[1:]:
        s = sol_all.advance(s, t, dt=1e-4)
        truth.append(s)
    st = SE.SeriesSettings(initial="relaxed", sawtooth="off")

    def chain(start):
        out = [start]
        s = start
        for k in range(1, len(times)):
            sol = _interval(lag, jni, times[k - 1], times[k], ips[k - 1], ips[k])
            s, _, _ = SE.advance_interval(sol, s, times[k], st, np.zeros(0), 1e-3)
            out.append(s)
        return out

    def err(a, b):
        da, db = np.diff(a.psi), np.diff(b.psi)
        return float(np.max(np.abs(da - db)) / np.max(np.abs(db)))

    from_true = chain(s_true0)
    relaxed0 = CM.build_solver(lag, jni, IP).steady_state(rf)
    from_relaxed = chain(CurrentState(rf, relaxed0.psi, relaxed0.psi_boundary, 0.0))
    for k in range(1, len(times)):
        assert err(from_true[k], truth[k]) < 1e-4, (k, err(from_true[k], truth[k]))
    assert err(from_relaxed[1], truth[1]) > 1e-2
    # the initial error decays (the relaxed start approaches the truth over the window)
    assert err(from_relaxed[-1], truth[-1]) < err(from_relaxed[1], truth[1])
