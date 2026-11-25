"""The time-averaged q floor, the default core treatment of single-slice fits and series: the RESISTIVE floor
(``current.sawtooth.resistive_floor_steady`` and ``ResistiveFloor``, the stationary-sawtooth rule of FUSE's
current-diffusion actor) by default, and the opt-in sharp clamp ``current.sawtooth.floor_q``.

Synthetic geometry as in ``test_coordinates_kinetics_series.py`` and the mocked series of ``test_series_driver.py``
(no OFT).  The model's steady state at the fixture's Ip keeps q > 1.6 everywhere; at 2.2 times that Ip it has q < 1
inside rho ~0.56 with q ~2.7 at the domain edge, so the floor q = 1 acts there.

Checked, sharp clamp (opt-in): Ip conserved, q >= floor after, the identity where q >= floor, refusal when q at the
domain edge is below the floor, continuity of q at the floor radius, idempotence (and no face counted as floored when
nothing changed).  Resistive floor (default form; the tests that pin the radius to a face select FUSE's grid rule,
``rule="grid"``, kept for comparison -- the default continuous rule is tested in ``test_q_floor_continuous.py``): Ip
conserved, the resistivity inside the flattening radius equal to its value there and unchanged outside, q raised,
FUSE's steady-state rule (the first face from the axis whose steady state has q >= floor everywhere, else the
outermost face below the floor, recorded), no current jump at the flattening radius in the grid limit (the sharp
clamp's jump does not shrink), the identity where q >= floor, refusal at the domain edge, the inner and outer
crossings of a reversed-shear profile; its time-dependent form takes the flattening radius from the state at the
start of each step.  The default treatment of a single slice (the floor
on the MODEL state, the corrections added after it, bit-identical to the plain steady state where the floor does not
act); a series under the default applies the same floor on every step and records it per slice; the opt-in sawtooth
treatments stay reachable and are refused where they are not selected.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coordinates_kinetics_series as GEO  # noqa: E402
import test_correction_local_return as LR  # noqa: E402
import test_series_driver as SD  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import CurrentState  # noqa: E402
from bouquet.kinrecon.current import sawtooth as SAW  # noqa: E402

IP_LOW_Q = 2.2 * GEO.IP           # q < 1 in the core of the model's steady state on the fixture geometry
ROUND_OFF = 1e-12                 # relative: q rebuilt from psi differences on the cells (round trip q -> psi -> q)
ULP2 = 4.5e-16                    # relative: two ulps, the diffusion's 1/(1/sigma) round trip through its resistivity hook


def _steady(ip, n=100):
    lag = GEO._lag()
    sol = CM.build_solver(lag, lag["jbs"], ip)
    return sol, sol.steady_state(CurrentState.uniform_faces(n))


def test_floor_q_conserves_ip_and_holds_q_at_the_floor():
    sol, st = _steady(IP_LOW_Q)
    p0 = sol.profiles(st)
    assert p0["q"][1:-1].min() < 1.0 < p0["q"][-1]
    st1, rec = SAW.floor_q(st, sol, 1.0)
    p1 = sol.profiles(st1)
    assert rec["acted"] and rec["q_min_before"] < 1.0 and 0.0 < rec["rho_floor"] < 1.0
    # Ip and the boundary gauge unchanged; the enclosed current outside the floored region unchanged to the bit
    assert p1["ip"] == p0["ip"] and st1.psi_boundary == st.psi_boundary and st1.psi[-1] == st.psi[-1]
    outside = p0["q"] >= 1.0
    outside[0] = False
    np.testing.assert_array_equal(p1["I_enclosed"][outside], p0["I_enclosed"][outside])
    # q >= floor on every face after (to the round trip), and q = max(q_before, floor) on the interior faces
    assert p1["q"].min() >= 1.0 - ROUND_OFF
    np.testing.assert_allclose(p1["q"][1:-1], np.maximum(p0["q"][1:-1], 1.0), rtol=ROUND_OFF)
    # the current inside the floored region is flattened (less on the axis), its total returns at the outer face
    assert p1["jdotb"][0] < p0["jdotb"][0] and rec["d_enclosed_current_max_over_ip"] > 0.0
    assert st1.meta["q_floor"]["rho_floor"] == rec["rho_floor"]


def test_floor_q_is_continuous_at_the_floor_radius_and_idempotent():
    sol, st = _steady(IP_LOW_Q)
    q0 = sol.profiles(st)["q"]
    st1, rec = SAW.floor_q(st, sol, 1.0)
    q1 = sol.profiles(st1)["q"]
    rf = st.rho_faces
    k = int(np.searchsorted(rf, rec["rho_floor"])) - 1        # last face inside the floor radius
    assert rf[k] <= rec["rho_floor"] < rf[k + 1] and q0[k] < 1.0 <= q0[k + 1]
    # continuous: the step of q across the floor radius is no larger than the state's own step there
    assert abs(q1[k + 1] - q1[k]) <= abs(q0[k + 1] - q0[k]) * (1 + ROUND_OFF)
    assert np.max(np.abs(np.diff(q1[1:-1]))) <= np.max(np.abs(np.diff(q0[1:-1]))) * (1 + ROUND_OFF)
    # the slope breaks there: flat inside, rising outside
    assert np.allclose(q1[1:k + 1], 1.0, rtol=ROUND_OFF) and q1[k + 2] > q1[k + 1]
    # a floored state is left as it is (the identity, the same object); its faces at the floor to round-off are
    # counted as below the floor, not as floored
    st2, rec2 = SAW.floor_q(st1, sol, 1.0)
    assert st2 is st1 and rec2["acted"] is False and rec2["n_faces_floored"] == 0
    assert isinstance(rec2["n_faces_below_floor"], int)          # recorded apart (round-off faces, if any)


def _stub_floor_case(n=64, m=20):
    """A stub solver whose q is fixed (0.8 on faces 0..m, 1.5 outside: faces 1..m always floored at q_floor = 1) and
    a state whose psi rebuild is exact outside the floored region (psi dyadic, uniform faces k/n with n a power of
    two), so the rebuilt psi of any state that agrees with it outside depends only on that outside part."""
    rf = np.arange(n + 1) / n
    q = np.where(np.arange(n + 1) <= m, 0.8, 1.5)
    sol = SimpleNamespace(profiles=lambda state: dict(q=q, I_enclosed=np.linspace(0.0, 1.0, n + 1), ip=1.0),
                          geometry_at=lambda t: SimpleNamespace(phi_b=1.0))
    rc = 0.5 * (rf[1:] + rf[:-1])
    st0 = CurrentState(rf, -(1.0 - rc ** 2), 0.0)
    st_a, rec_a = SAW.floor_q(st0, sol, 1.0)
    assert rec_a["acted"] and rec_a["n_faces_floored"] == m
    # the rebuild is exact outside the floored region (the premise of the two tests below)
    np.testing.assert_array_equal(st_a.psi[m:], st0.psi[m:])
    return sol, st_a, n, m


def test_floor_q_returns_its_input_when_the_rebuilt_psi_is_off_by_one_ulp():
    """"Nothing changed" is equality of the rebuilt psi within n ulp per cell (n cells), not bitwise: the input state
    itself is returned when the rebuild differs from it by 1 ulp in some cells."""
    sol, st_a, n, m = _stub_floor_case()
    psi = st_a.psi.copy()
    psi[1:m:3] = np.nextafter(psi[1:m:3], np.inf)            # 1 ulp in some cells of the floored region
    st_x = CurrentState(st_a.rho_faces, psi, st_a.psi_boundary, st_a.time)
    assert not np.array_equal(st_x.psi, st_a.psi)            # its rebuilt psi (st_a's, below) is not bitwise equal
    st2, rec2 = SAW.floor_q(st_x, sol, 1.0)
    assert st2 is st_x and rec2["acted"] is False and rec2["n_faces_floored"] == 0 and rec2["rho_floor"] is None
    assert rec2["n_faces_below_floor"] == m


def test_floor_q_acts_when_the_rebuilt_psi_differs_by_clearly_more_than_n_ulp():
    sol, st_a, n, m = _stub_floor_case()
    psi = st_a.psi.copy()
    psi[1:m:3] += 4 * n * np.spacing(np.abs(psi[1:m:3]))     # 4 n ulp in the same cells
    st_x = CurrentState(st_a.rho_faces, psi, st_a.psi_boundary, st_a.time)
    st2, rec2 = SAW.floor_q(st_x, sol, 1.0)
    assert st2 is not st_x and rec2["acted"] is True and rec2["n_faces_floored"] == m
    # the rebuild does not read the changed cells: it returns the floored psi of the first call, bit for bit
    np.testing.assert_array_equal(st2.psi, st_a.psi)


def test_floor_q_is_the_identity_where_q_is_above_the_floor():
    sol, st = _steady(GEO.IP)
    assert sol.profiles(st)["q"].min() > 1.0
    st1, rec = SAW.floor_q(st, sol, 1.0)
    assert st1 is st and rec["acted"] is False and rec["n_faces_floored"] == 0 and rec["rho_floor"] is None


def test_floor_q_refuses_a_state_below_the_floor_at_the_domain_edge():
    sol, st = _steady(IP_LOW_Q)
    q = sol.profiles(st)["q"]
    with pytest.raises(SAW.QFloorRefused, match="domain edge"):
        SAW.floor_q(st, sol, 1.01 * q[-1])
    with pytest.raises(SAW.QFloorRefused, match="last interior face"):
        SAW.floor_q(st, sol, 0.5 * (q[-2] + q[-1]) if q[-2] > q[-1] else 1.0001 * q[-2])
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="q_floor"):
            SAW.floor_q(st, sol, bad)
    with pytest.raises(ValueError, match="q_floor"):
        CM.CurrentModelSettings(r0=GEO.R0, q_floor=0.0)


def _face_q(sol, st):
    return SAW.face_q(st, sol.geometry_at(st.time).phi_b)


def test_resistive_floor_conserves_ip_flattens_the_resistivity_and_raises_q():
    # FUSE's grid rule (rule="grid", kept for comparison; the default continuous rule: test_q_floor_continuous.py)
    sol, st0 = _steady(IP_LOW_Q)
    rf = st0.rho_faces
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    p0, p1 = sol.profiles(st0), s2.profiles(st)
    assert rec["acted"] and rec["q_floor_form"] == "resistive" and rec["q_min_before"] < 1.0
    k = rec["n_cells_flattened"]
    assert k >= 1 and rec["rho_floor"] == rf[k]
    # Ip is the steady state's own Ip condition; the boundary gauge is kept
    assert p1["ip"] == p0["ip"] == pytest.approx(IP_LOW_Q, rel=1e-12) and st.psi_boundary == st0.psi_boundary
    # the resistivity on every cell inside the flattening radius is the plain profile's value AT that radius
    # (eq. F1); outside it is unchanged (to the hook's 1/(1/sigma) round trip)
    rc = st.rho_cells
    sig_inv = float(sol.sigma_fn(0.0, np.array([rf[k]]))[0])
    np.testing.assert_allclose(p1["conductivity"][rc < rf[k]], sig_inv, rtol=ULP2, atol=0)
    np.testing.assert_allclose(p1["conductivity"][rc > rf[k]], p0["conductivity"][rc > rf[k]], rtol=ULP2, atol=0)
    # the core current is flatter (less on the axis) and q is raised; the face q of the operator is the profiles' q
    assert p1["jdotb"][0] < p0["jdotb"][0] and rec["q_min_after"] > rec["q_min_before"]
    np.testing.assert_allclose(_face_q(s2, st), p1["q"], rtol=1e-12)
    assert st.meta["q_floor"]["rho_floor"] == rec["rho_floor"] and st.meta["q_floor"]["form"] == "resistive"


@pytest.mark.parametrize("fac", [1.74, 1.86, 2.2])
def test_resistive_floor_takes_the_flattening_radius_by_the_fuse_steady_state_rule(fac):
    """FUSE's grid rule (rule="grid", kept for comparison): the first face from the axis whose steady state has
    q >= floor on every face; when none up to the outermost face where the plain state's q is below the floor does,
    that outermost face, recorded as not meeting the floor."""
    sol, st0 = _steady(fac * GEO.IP)
    rf = st0.rho_faces
    q0 = _face_q(sol, st0)
    k_max = int(np.flatnonzero(q0 < 1.0)[-1])
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    k = rec["n_cells_flattened"]

    def meets(j):
        f = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[j]).with_solver()
        return bool(np.all(_face_q(f, f.steady_state(rf)) >= 1.0))

    assert 1 <= k <= k_max and rec["n_steady_states"] == k + 1
    assert not any(meets(j) for j in range(1, k))
    if rec["q_floor_met"]:
        assert meets(k) and rec["q_min_after"] >= 1.0
    else:
        assert k == k_max and not meets(k) and rec["q_min_after"] < 1.0
    assert rec["q_floor_met"] is (fac < 2.0)           # the fixture: met below about 2.0 Ip, not met at 2.2 Ip


def test_resistive_floor_has_no_current_jump_at_the_flattening_radius_in_the_grid_limit():
    """The step of <J.B> between the two cells around the flattening radius shrinks with the cell size (only the
    half-cell discretization of the resistivity at the radius is left); the sharp clamp's step does not."""
    lag = GEO._lag()
    sol = CM.build_solver(lag, lag["jbs"], IP_LOW_Q)
    step_res, step_sharp = {}, {}
    for n in (100, 400):
        rf = CurrentState.uniform_faces(n)
        st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
        j = s2.profiles(st)["jdotb"]
        k = rec["n_cells_flattened"]
        step_res[n] = abs(j[k] - j[k - 1]) / abs(j[k - 1])
        st_s, rec_s = SAW.floor_q(sol.steady_state(rf), sol, 1.0)
        js = sol.profiles(st_s)["jdotb"]
        ks = int(np.searchsorted(st_s.rho_cells, rec_s["rho_floor"]))
        step_sharp[n] = abs(js[ks] - js[ks - 1]) / abs(js[ks - 1])
    assert step_res[400] <= 0.3 * step_res[100]                  # first order in the cell size (ratio 0.25)
    assert min(step_sharp.values()) > 0.2 and step_sharp[400] > 0.5 * step_sharp[100]


def test_resistive_floor_is_the_identity_where_q_is_above_the_floor_and_refuses_the_edge():
    sol, st0 = _steady(GEO.IP)
    st, rec, s2 = SAW.resistive_floor_steady(sol, st0.rho_faces, 1.0)
    assert s2 is sol and rec["acted"] is False and rec["n_cells_flattened"] == 0 and rec["rho_floor"] is None
    np.testing.assert_array_equal(st.psi, st0.psi)
    q = _face_q(sol, st0)
    with pytest.raises(SAW.QFloorRefused, match="domain edge"):
        SAW.resistive_floor_steady(sol, st0.rho_faces, 1.01 * q[-1])
    with pytest.raises(SAW.QFloorRefused, match="domain edge"):
        SAW.ResistiveFloor(sol, 1.01 * q[-1]).region(st0)
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="q_floor"):
            SAW.resistive_floor_steady(sol, st0.rho_faces, bad)
    with pytest.raises(ValueError, match="q_floor_form"):
        CM.CurrentModelSettings(r0=GEO.R0, q_floor_form="smooth")


def test_reversed_shear_records_the_inner_and_the_outer_crossing():
    """An off-axis region below the floor (q0 > floor): both crossings are recorded, the inner one falling through
    the floor ('down') and the outer one rising ('up'); FUSE's grid rule (rule="grid", kept for comparison) flattens
    out to the outermost face below the floor; the sharp clamp leaves the axis untouched.  (The default continuous
    rule does not act on this state: test_q_floor_continuous.py.)"""
    lag = GEO._lag()
    rho = lag["phys"]["rho"]
    jni = lag["jbs"] + 8.0 * np.max(np.abs(lag["jbs"])) * np.exp(-0.5 * ((rho - 0.5) / 0.08) ** 2)
    sol = CM.build_solver(lag, jni, 2.3 * GEO.IP)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    q = _face_q(sol, st0)
    assert q[0] > 1.0 and q.min() < 1.0 < q[-1]
    cr = SAW.floor_crossings(rf, q, 1.0)
    assert [c["direction"] for c in cr] == ["down", "up"] and cr[0]["rho"] < rf[np.argmin(q)] < cr[1]["rho"]
    k, r_inv = SAW.ResistiveFloor(sol, 1.0, rule="grid").region(st0)
    assert r_inv == rf[k] and q[k] < 1.0 <= q[k + 1] and cr[1]["rho"] > r_inv
    st_s, rec_s = SAW.floor_q(st0, sol, 1.0)
    assert [c["direction"] for c in rec_s["rho_floor_crossings"]] == ["down", "up"]
    assert rec_s["rho_floor"] == pytest.approx(cr[1]["rho"], rel=1e-12)
    st, rec, _ = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    assert [c["direction"] for c in rec["rho_floor_crossings"]] == ["down", "up"] and rec["acted"]


def test_time_dependent_floor_takes_the_radius_from_the_state_at_the_start_of_the_step():
    # FUSE's grid rule (rule="grid", kept for comparison; the default continuous rule: test_q_floor_continuous.py)
    sol, st0 = _steady(IP_LOW_Q)
    mod = SAW.ResistiveFloor(sol, 1.0, rule="grid")
    s2 = mod.with_solver()
    assert sol.resistivity_modifier is None and s2.resistivity_modifier is mod
    k, r_inv = mod.region(st0)
    q = _face_q(sol, st0)
    assert k == int(np.flatnonzero(q < 1.0)[-1]) and r_inv == st0.rho_faces[k]
    c = s2.coefficients(0.0, st0.rho_faces, st0)
    c_plain = sol.coefficients(0.0, st0.rho_faces, st0)
    rc = st0.rho_cells
    np.testing.assert_allclose(c.sigma_c[rc < r_inv], float(sol.sigma_fn(0.0, np.array([r_inv]))[0]), rtol=ULP2, atol=0)
    np.testing.assert_allclose(c.sigma_c[rc > r_inv], c_plain.sigma_c[rc > r_inv], rtol=ULP2, atol=0)
    # one step: Ip conserved; with q >= floor everywhere the modifier changes nothing
    st1 = s2.step(st0, 1e-3)
    assert s2.profiles(st1)["ip"] == pytest.approx(IP_LOW_Q, rel=1e-12)
    sol_hi, st_hi = _steady(GEO.IP)
    m_hi = SAW.ResistiveFloor(sol_hi, 1.0)
    assert m_hi.region(st_hi) == (0, None)
    ref = sol_hi.step(st_hi, 1e-3).psi
    np.testing.assert_allclose(m_hi.with_solver().step(st_hi, 1e-3).psi, ref, rtol=0, atol=1e-12 * np.ptp(ref))
    with pytest.raises(ValueError, match="already has"):
        SAW.ResistiveFloor(s2, 1.0).with_solver()


def test_single_slice_default_floors_the_model_state_then_adds_the_corrections():
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    assert cfg.core_relaxation == "q_floor" and cfg.q_floor == 1.0 and cfg.q_floor_form == "resistive"
    assert FIT.FitSettings().free == ("a_corr", "s_press")
    rf = CurrentState.uniform_faces(cfg.n_cells)
    # (1) the floor acts: the uncorrected state is the resistive floor of the model's steady state
    a0 = CM.assemble(lag, kin, src, {}, cfg, IP_LOW_Q)
    sol = CM.build_solver(lag, lag["jbs"], IP_LOW_Q)
    want, frec, _ = SAW.resistive_floor_steady(sol, rf, 1.0)
    np.testing.assert_array_equal(a0["state"].psi, want.psi)
    core = a0["core"]
    assert core["core_relaxation"] == "q_floor" and core["q_floor"] == 1.0 and core["acted"]
    assert core["q_floor_form"] == "resistive" and core["n_cells_flattened"] == frec["n_cells_flattened"]
    assert core["rho_floor"] == pytest.approx(frec["rho_floor"] * lag["rho_map"].rho_edge, rel=1e-15)
    assert core["rho_floor_domain"] == frec["rho_floor"]
    assert core["q_min_model"] == pytest.approx(frec["q_min_after"], rel=1e-12)
    assert core["q_axis_model_minus_floor"] == pytest.approx(core["q_axis_model"] - 1.0, rel=1e-15)
    # (2) the corrections are added AFTER the floor: their steady response is not floored away
    amps = (-0.03, 0.0, 0.0)             # a core correction that lowers q further
    a1 = CM.assemble(lag, kin, src, dict(a_corr=amps), cfg, IP_LOW_Q)
    shapes = CM.correction_shapes(lag, cfg, IP_LOW_Q)
    sol_c = CM.build_solver(lag, lag["jbs"] + sum(a * g for a, g in zip(amps, shapes)), IP_LOW_Q)
    resp = sol_c.steady_state(rf).psi - sol.steady_state(rf).psi
    np.testing.assert_allclose(a1["state"].psi - a0["state"].psi, resp, rtol=1e-12, atol=1e-12 * np.ptp(resp))
    assert a1["core"]["q_min_model"] != pytest.approx(a0["core"]["q_min_model"], rel=1e-6)
    # (3) where the floor does not act the default is the plain steady state with the corrections, to the bit
    amps = (0.03, 0.02, 0.01)
    none = CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="none")
    for a in ((0.0, 0.0, 0.0), amps):
        d = CM.assemble(lag, kin, src, dict(a_corr=a), cfg, GEO.IP)
        n = CM.assemble(lag, kin, src, dict(a_corr=a), none, GEO.IP)
        assert not d["core"]["acted"]
        np.testing.assert_array_equal(d["state"].psi, n["state"].psi)
        np.testing.assert_array_equal(d["jphi"], n["jphi"])
    # (4) the opt-in sharp clamp is the earlier operator on the model state
    sharp = CM.CurrentModelSettings(r0=GEO.R0, q_floor_form="sharp")
    a_s = CM.assemble(lag, kin, src, {}, sharp, IP_LOW_Q)
    np.testing.assert_array_equal(a_s["state"].psi, SAW.floor_q(sol.steady_state(rf), sol, 1.0)[0].psi)
    assert a_s["core"]["q_floor_form"] == "sharp"
    # the sawtooth parameters are not read on the default path: recorded as not used
    assert a0["params"]["f"] is None and a0["params"]["log_strength"] is None
    assert a0["params_not_used"] == ["log_strength", "f"]


def test_the_q_floor_setting_moves_the_floor():
    lag, kin, src = LR._fixture_inputs()
    for form in ("resistive", "sharp"):
        out = {}
        for qf in (0.95, 1.0, 1.05):
            a = CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0, q_floor=qf, q_floor_form=form),
                            IP_LOW_Q)
            out[qf] = a["core"]
            assert a["core"]["q_floor"] == qf and a["core"]["q_floor_form"] == form
            if form == "sharp":
                assert a["core"]["q_min_model"] == pytest.approx(qf, rel=ROUND_OFF)
        assert out[0.95]["rho_floor"] < out[1.0]["rho_floor"] < out[1.05]["rho_floor"]
        assert out[0.95]["q_min_model"] < out[1.0]["q_min_model"] < out[1.05]["q_min_model"]


def test_opt_in_sawtooth_treatments_are_reachable_and_refused_elsewhere():
    lag, kin, src = LR._fixture_inputs()
    ls = float(np.log(SAW.crash_strength(0.37, 1.0 / 0.05)))
    orb = CM.assemble(lag, kin, src, dict(log_strength=ls, f=0.37),
                      CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="periodic_orbit"), IP_LOW_Q)
    assert orb["sawtooth"]["mode"].startswith("periodic orbit") and orb["sawtooth"]["active"]
    assert orb["core"]["core_relaxation"] == "periodic_orbit" and orb["core"]["q_floor"] is None
    crash = CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="crash_times", crash_times=(-0.2, -0.15, -0.1, -0.05),
                                    window=(-0.04, 0.0), crash_record_start=-2.0, history=0.3)
    m = CM.assemble(lag, kin, src, dict(log_strength=ls, f=0.37), crash, IP_LOW_Q)
    assert m["sawtooth"]["mode"] == "measured crash times" and m["core"]["core_relaxation"] == "crash_times"
    # the sawtooth parameters and crash times are refused where their treatment is not selected
    with pytest.raises(ValueError, match="opt-in sawtooth"):
        CM.assemble(lag, kin, src, dict(log_strength=ls), CM.CurrentModelSettings(r0=GEO.R0), IP_LOW_Q)
    with pytest.raises(ValueError, match="needs the sawtooth parameter"):
        CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="periodic_orbit"), IP_LOW_Q)
    with pytest.raises(ValueError, match="explicit|opt-in"):
        CM.CurrentModelSettings(r0=GEO.R0, crash_times=(0.1,))
    with pytest.raises(ValueError, match="core_relaxation must be one of"):
        CM.CurrentModelSettings(r0=GEO.R0, core_relaxation="sawtooth")
    with pytest.raises(ValueError, match="opt-in sawtooth core treatments"):
        FIT.reconstruct(SimpleNamespace(ms=None), kin, src, CM.CurrentModelSettings(r0=GEO.R0),
                        FIT.FitSettings(free=("log_strength", "a_corr")))


def test_series_default_floors_every_step_with_the_single_slice_treatment(monkeypatch):
    """Both modes use the same treatment by default: the first slice is solved with core_relaxation 'q_floor' and the
    series' q_floor, and every interval applies the resistive floor on every step, the region decided at the start of
    the step (the single-slice region of the step's sources, test_q_floor_continuous.py)."""
    lag = SD._lag()
    ip = IP_LOW_Q
    fm = SD.FakeFM(dict(pred=lag))
    st0 = SD._steady(lag, ip)
    seen = []

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        seen.append((cfg.core_relaxation, cfg.q_floor, cfg.q_floor_form, state_fn is None))
        fm_.n_solves += 1
        if state_fn is None:
            return SD._result(lag, st0)
        st = state_fn(lag)
        res = SD._result(lag, st)
        res.assembly["core"] = dict(st.meta["core"])      # what current_model.assemble records for a supplied state
        return res

    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0)
    out = SE.run_series(fm, SD._slices(np.arange(3) * 0.02, np.full(3, ip)), cfg, SE.SeriesSettings(initial="relaxed"))
    assert out["status"] == "completed" and out["first_slice_core_relaxation"] == "q_floor"
    assert seen[0] == ("q_floor", 1.0, "resistive", True) and all(s[1] == 1.0 for s in seen)
    assert out["settings"]["sawtooth_f"] is None and "sawtooth_f" in out["settings"]["settings_not_used"]
    # the starting state has q < 1; the floor acts on the steps of the first interval and raises q against the same
    # interval advanced without any core treatment; the slices record it
    sol = CM.build_solver(lag, lag["jbs"], ip)
    assert sol.profiles(st0)["q"][1:-1].min() < 1.0
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.02, ip, ip)
    st_in = CurrentState(st0.rho_faces, st0.psi, st0.psi_boundary, 0.0)
    plain = isol.advance(st_in, 0.02, dt=SE.SeriesSettings().dt_max)
    rec1 = out["slices"][1]
    st1 = CurrentState(rec1["carried_state"]["rho_faces"], rec1["carried_state"]["psi"],
                       rec1["carried_state"]["psi_boundary"])
    assert sol.profiles(st1)["q"].min() > sol.profiles(plain)["q"].min()
    for rec in out["slices"][1:]:
        c = rec["core"]
        assert c["core_relaxation"] == "q_floor" and c["q_floor"] == 1.0 and c["q_floor_form"] == "resistive"
        assert c["n_steps"] == 20 and "rho_floor" in c and "rho_floor_crossings" in c and c["dt"] == 1e-3
        assert "core_vs_floor" in rec
    assert rec1["core"]["acted_in_interval"] and rec1["core"]["n_steps_floor_acted"] >= 1
    # one step of the series' advance is one step of the solver with the resistive floor of the starting state
    s = SE.SeriesSettings()
    end, core = SE.advance_q_floor(isol, st_in, 1e-3, s, 1e-3, 1.0)
    ref = SAW.ResistiveFloor(isol, 1.0).with_solver().step(st_in, 1e-3)
    np.testing.assert_array_equal(end.psi, ref.psi)
    k0, r0 = SAW.ResistiveFloor(isol, 1.0).region(st_in)
    assert core["n_steps"] == 1 and core["n_steps_floor_acted"] == 1 and core["rho_floor_last_step"] == r0 and k0 > 0
    # the opt-in sharp form: one plain step followed by the clamp (the earlier operator)
    end_s, core_s = SE.advance_q_floor(isol, st_in, 1e-3, s, 1e-3, 1.0, form="sharp")
    np.testing.assert_array_equal(end_s.psi, SAW.floor_q(isol.step(st_in, 1e-3), isol, 1.0)[0].psi)
    assert core_s["q_floor_form"] == "sharp" and core_s["n_steps"] == 1


def test_series_floor_does_not_act_on_the_in_window_corrections(monkeypatch):
    """Where the floor does not act, the q-floor series is the series without any core treatment (to round-off), with
    and without an in-window correction source; the corrections' response is added after the floor."""
    lag = SD._lag()
    sol_pred = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.02, GEO.IP, GEO.IP)[0]
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, GEO.IP)))
    assert sol_pred.profiles(st_in)["q"].min() > 1.0
    shapes = CM.correction_shapes(lag, CM.CurrentModelSettings(r0=GEO.R0), GEO.IP, form="in_window")
    c = sum(a * g for a, g in zip((0.02, -0.01, 0.01), shapes))
    sol_all = SE.interval_solver(lag, lag, lag["jbs"] + c, lag["jbs"] + c, 0.0, 0.02, GEO.IP, GEO.IP)[0]
    s = SE.SeriesSettings()
    plain = sol_all.advance(st_in, 0.02, dt=1e-3)
    for form in ("resistive", "sharp"):
        end, core = SE.advance_q_floor(sol_pred, st_in, 0.02, s, 1e-3, 1.0, sol_all=sol_all, form=form)
        np.testing.assert_allclose(end.psi, plain.psi, rtol=0, atol=1e-12 * np.ptp(plain.psi))
        assert core["n_steps_floor_acted"] == 0 and core["corrections"].startswith("in-window")
