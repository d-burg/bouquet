"""The NON-DEFAULT options of the resistive q floor (``current.sawtooth``, "Non-default options"), each off by default:

* ``total_current="flat_at_level"``: inside the floor's region the total current is that of q flat at the level (any
  level, not only 1); Ip is the solver's Ip condition; the region and the branch are the default rule's; nothing in
  branch (b); the same in a time-dependent step;
* ``transition="blend_margin"``: the number-free blend of the level across the plain axis-q margin; identical to the
  default outside the band, continuous at the boundary inside it;
* ``total_current="flat_at_level_own_region"``: the same flat-at-level total current inside the option's own region
  (the default rule's criterion on the flat-at-level states): q at or above the level outside it, the smallest such
  edge, and the same in a time-dependent step;
* ``q_floor_branch_freeze="first_assembly"`` (diagnostic): the first assembly's branch for the whole fit, carried by
  the freeze, not read from a record;
* with all three off the default path does not enter any of the new code (the default results are unchanged).

Synthetic geometry as in ``test_q_floor.py`` (no OFT).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coordinates_kinetics_series as GEO  # noqa: E402
import test_correction_local_return as LR  # noqa: E402
import test_fit_unit as FU  # noqa: E402
import test_q_floor_continuous as QC  # noqa: E402
import test_series_driver as SD  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import fit as FIT  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import CurrentState  # noqa: E402
from bouquet.kinrecon.current import sawtooth as SAW  # noqa: E402

FLAT_ATOL = 1e-11      # q inside the region against the level: the closed form to round-off (measured ~4e-13)
IP_RTOL = 1e-12        # Ip against the solver's Ip condition (the same bound as the default rule's tests)
RF = CurrentState.uniform_faces(100)


def _raising(*a, **k):
    raise RuntimeError("new code entered on the default path")


# ====================================================================== the default path is untouched
def test_default_path_never_enters_the_new_code(monkeypatch):
    """With every option off (the defaults) the single-slice floor, an assembly, a default fit and a series interval
    never call the new code: making it raise changes nothing, and the outputs are bitwise those of the run without the
    patch."""
    assert CM.CurrentModelSettings().q_floor_total_current == "off"
    assert CM.CurrentModelSettings().q_floor_transition == "step"
    assert CM.CurrentModelSettings().q_floor_branch_freeze == "off"
    sol = QC._solver(QC.IP_MET)
    assert getattr(sol, "source_modifier", "absent") is None
    st_ref, rec_ref, s_ref = SAW.resistive_floor_steady(sol, RF, 1.0)
    assert s_ref.source_modifier is None and "total_current" not in rec_ref and "branch_frozen" not in rec_ref
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    a_ref = CM.assemble(lag, kin, src, {}, cfg, QC.IP_MET)
    fit_ref = QC._default_fit(monkeypatch)
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 0.01, QC.IP_MET, 1.02 * QC.IP_MET)
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, QC.IP_MET)))
    e_ref, c_ref = SE.advance_q_floor(isol, st_in, 0.01, SE.SeriesSettings(), 1e-3, 1.0)
    with monkeypatch.context() as mp:
        for mod, attr in ((SAW, "level_flattening_source_change"), (SAW, "_check_options"),
                          (CM, "_floor_with_options"), (CM, "open_branch_freeze")):
            mp.setattr(mod, attr, _raising)
        st, rec, _ = SAW.resistive_floor_steady(sol, RF, 1.0)
        a = CM.assemble(lag, kin, src, {}, cfg, QC.IP_MET)
        e, c = SE.advance_q_floor(isol, st_in, 0.01, SE.SeriesSettings(), 1e-3, 1.0)
        o = QC._default_fit(mp)
    np.testing.assert_array_equal(st.psi, st_ref.psi)
    assert st.psi_boundary == st_ref.psi_boundary and rec["rho_floor"] == rec_ref["rho_floor"]
    np.testing.assert_array_equal(a["jphi"], a_ref["jphi"])
    np.testing.assert_array_equal(a["state"].psi, a_ref["state"].psi)
    np.testing.assert_array_equal(e.psi, e_ref.psi)
    assert e.psi_boundary == e_ref.psi_boundary and c["rho_floor"] == c_ref["rho_floor"]
    QC._assert_same_fit(o, fit_ref)


def test_a_zero_source_change_leaves_the_solver_bitwise_unchanged():
    """The diffusion's source hook adds exactly what it is given: a modifier returning zeros gives the same
    coefficients and the same steady state as no modifier, to the bit."""
    import copy
    sol = QC._solver(QC.IP_MET)
    s2 = copy.copy(sol)
    s2.source_modifier = lambda t, rf, c, ip_mode, state: np.zeros(rf.size - 1)
    c0, c1 = sol.coefficients(0.0, RF), s2.coefficients(0.0, RF)
    np.testing.assert_array_equal(c0.s_c, c1.s_c)
    np.testing.assert_array_equal(c0.jni_c, c1.jni_c)
    np.testing.assert_array_equal(sol.steady_state(RF).psi, s2.steady_state(RF).psi)


def test_option_values_and_combinations_are_checked():
    sol = QC._solver(QC.IP_MET)
    for kw, msg in ((dict(total_current="flat"), "total_current"), (dict(transition="smooth"), "transition"),
                    (dict(branch="both"), "branch"),
                    (dict(total_current="flat_at_level", rule="grid"), "continuous"),
                    (dict(total_current="flat_at_level", transition="blend_margin"), "not defined"),
                    (dict(branch="axis", transition="blend_margin"), "not defined")):
        with pytest.raises(ValueError, match=msg):
            SAW.resistive_floor_steady(sol, RF, 1.0, **kw)
    for kw in (dict(q_floor_total_current="x"), dict(q_floor_transition="x"), dict(q_floor_branch_freeze="x"),
               dict(q_floor_total_current="flat_at_level", q_floor_form="sharp"),
               dict(q_floor_branch_freeze="first_assembly", q_floor_transition="blend_margin")):
        with pytest.raises(ValueError):
            CM.CurrentModelSettings(**kw)
    import copy
    s2 = copy.copy(sol)
    s2.source_modifier = lambda *a: 0.0
    with pytest.raises(ValueError, match="source modifier"):
        SAW.ResistiveFloor(s2, 1.0, rho_inv=0.3, total_current="flat_at_level").with_solver()


# ====================================================================== total-current flattening at the level
@pytest.mark.parametrize("ip, level", [(QC.IP_MET, 1.0), (QC.IP_MET, 0.95), (QC.IP_LOW_Q, 1.0), (QC.IP_LOW_Q, 0.9)])
def test_total_current_option_holds_q_flat_at_the_level_inside_the_floors_region(ip, level):
    """Inside the region the default rule decides (same branch, same radius) q equals the level on every face; Ip is
    the solver's Ip condition; the record states the option, the resistivity-only state's q and the non-inductive
    current the option adds."""
    sol = QC._solver(ip)
    st0 = sol.steady_state(RF)
    _, rd, _ = SAW.resistive_floor_steady(sol, RF, level)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, level, total_current="flat_at_level")
    assert rec["floor_branch"] == rd["floor_branch"] == "axis" and rec["acted"]
    assert rec["rho_floor"] == rd["rho_floor"]
    q = QC._fq(s2, st)
    inside = RF <= rec["rho_floor"]
    assert np.max(np.abs(q[inside] - level)) <= FLAT_ATOL
    assert rec["q_flat_max_abs_dev"] <= FLAT_ATOL and abs(rec["q_axis_after"] - level) <= FLAT_ATOL
    assert s2.profiles(st)["ip"] == pytest.approx(sol.profiles(st0)["ip"], rel=IP_RTOL)
    assert st.psi_boundary == st0.psi_boundary
    assert rec["total_current"] == "flat_at_level" and np.isfinite(rec["total_current_ni_added_over_ip"])
    assert rec["q_min_resistive_only"] == pytest.approx(rd["q_min_after"], rel=1e-12)
    assert rec["q_floor_met"] == bool(np.all(q[~inside] >= level))
    assert st.meta["q_floor"]["total_current"] == "flat_at_level"
    # the source change is that of the closed form (the hook's ds), zero outside the region
    c_tc, c_r = s2.coefficients(0.0, RF), SAW.ResistiveFloor(sol, level, rho_inv=rec["rho_floor"],
                                                               rho_faces=RF).with_solver().coefficients(0.0, RF)
    ds = c_tc.s_c - c_r.s_c
    rc = 0.5 * (RF[1:] + RF[:-1])
    assert np.all(ds[RF[:-1] >= rec["rho_floor"]] == 0.0) and np.any(ds[rc < rec["rho_floor"]] != 0.0)


def test_total_current_option_does_nothing_in_branch_b_and_c():
    """Branch (b) (below the level off axis only) and (c) (nowhere): no region, so the option changes nothing."""
    sol = QC._reversed_shear()
    st_d, rd, _ = SAW.resistive_floor_steady(sol, RF, 1.0)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, total_current="flat_at_level")
    assert rd["floor_branch"] == rec["floor_branch"] == "off_axis" and not rec["acted"] and s2 is sol
    np.testing.assert_array_equal(st.psi, st_d.psi)
    sol_c = QC._solver(QC.IP_MET)
    _, rc_, s3 = SAW.resistive_floor_steady(sol_c, RF, 0.5, total_current="flat_at_level")
    assert rc_["floor_branch"] == "none" and not rc_["acted"] and s3 is sol_c


def test_total_current_option_converges_with_the_grid_and_q0_is_the_level_on_every_grid():
    sol = QC._solver(QC.IP_MET)
    out = {n: SAW.resistive_floor_steady(sol, CurrentState.uniform_faces(n), 1.0, total_current="flat_at_level")
           for n in (100, 200, 400)}
    r = {n: out[n][1]["rho_floor"] for n in out}
    assert abs(r[100] - r[200]) >= abs(r[200] - r[400])
    for n, (st, rec, s2) in out.items():
        assert abs(rec["q_axis_after"] - 1.0) <= FLAT_ATOL, n


def test_total_current_option_in_a_time_dependent_step_conserves_ip_and_relaxes_to_the_single_slice_state():
    """In a series step the option changes the sources inside the step's region; Ip after a step is the interval's
    Ip condition, and a long interval at fixed sources ends on the single-slice state with the option."""
    lag = SD._lag()
    ip = QC.IP_MET
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 20.0, ip, ip)
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, ip)))
    e1, c1 = SE.advance_q_floor(isol, st_in, 1e-3, SE.SeriesSettings(), 1e-3, 1.0, total_current="flat_at_level")
    mod = SAW.ResistiveFloor(isol, 1.0, total_current="flat_at_level")
    assert mod.with_solver().profiles(e1)["ip"] == pytest.approx(ip, rel=IP_RTOL)
    assert c1["total_current"] == "flat_at_level" and c1["acted"]
    # 20 s at fixed sources (many resistive times of the fixture; the default rule relaxes to ~1e-11 there as well)
    e, c = SE.advance_q_floor(isol, st_in, 20.0, SE.SeriesSettings(), 0.25, 1.0, total_current="flat_at_level")
    st_ss, rec_ss, s_ss = SAW.resistive_floor_steady(isol, e.rho_faces, 1.0, t=20.0, total_current="flat_at_level")
    q_e = SAW.face_q(e, isol.geometry_at(20.0).phi_b)
    q_s = QC._fq(s_ss, st_ss)
    assert np.max(np.abs(q_e - q_s)) <= 1e-9 * np.max(q_s)


def test_total_current_option_makes_q0_continuous_at_the_branch_boundary():
    """Where the plain axis q crosses the level (Ip scanned), the option puts q0 at the level on the (a) side, and
    the plain q0 on the (b)/(c) side is at the level too: q0 changes by the plain state's own slope only."""
    lag = SD._lag_ni()

    def q0_plain(ip_):
        s_ = CM.build_solver(lag, lag["jbs"], ip_)
        return float(QC._fq(s_, s_.steady_state(RF))[0])
    lo, hi = GEO.IP, 3.0 * GEO.IP
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q0_plain(mid) > 1.0 else (lo, mid)
    q0s = {}
    for e in (-1e-6, 1e-6):
        s_ = CM.build_solver(lag, lag["jbs"], 0.5 * (lo + hi) * (1.0 + e))
        st, rec, s2 = SAW.resistive_floor_steady(s_, RF, 1.0, total_current="flat_at_level")
        q0s[e] = (float(QC._fq(s2, st)[0]), rec["floor_branch"])
    assert q0s[1e-6][1] == "axis" and q0s[-1e-6][1] != "axis"
    assert abs(q0s[1e-6][0] - q0s[-1e-6][0]) <= 1e-4


# ====================================================================== blended transition (number-free)
def test_blend_is_the_default_rule_outside_its_band():
    """Branch (a) and branch (c): the blend is the default rule to the bit; branch (b) beyond the margin (lambda 0):
    no action, as the default."""
    for sol in (QC._solver(QC.IP_MET), QC._solver(QC.IP_LOW_Q)):
        st_d, rd, _ = SAW.resistive_floor_steady(sol, RF, 1.0)
        st, rec, _ = SAW.resistive_floor_steady(sol, RF, 1.0, transition="blend_margin")
        np.testing.assert_array_equal(st.psi, st_d.psi)
        assert rec["rho_floor"] == rd["rho_floor"] and rec["blend_lambda"] == 1.0 and rec["level_used"] == 1.0
    sol = QC._reversed_shear()
    q = QC._fq(sol, sol.steady_state(RF))
    level = float(np.min(q)) + 0.4 * (float(q[0]) - float(np.min(q)))     # q0 - level > level - q_min: lambda 0
    st_d, rd, _ = SAW.resistive_floor_steady(sol, RF, level)
    st, rec, _ = SAW.resistive_floor_steady(sol, RF, level, transition="blend_margin")
    assert rd["floor_branch"] == "off_axis" and rec["blend_lambda"] == 0.0 and not rec["acted"]
    np.testing.assert_array_equal(st.psi, st_d.psi)


def test_blend_lands_q_min_on_the_blended_level_inside_the_band_and_is_continuous_at_the_boundary():
    """Inside the band the region is searched for the blended level (q_min on it where met); at the boundary
    (plain q0 = level) the blended state meets the default branch (a) state."""
    sol = QC._reversed_shear()
    q = QC._fq(sol, sol.steady_state(RF))
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, transition="blend_margin")
    lam, lev = rec["blend_lambda"], rec["level_used"]
    qmn = float(np.min(q))
    assert 0.0 < lam < 1.0 and lev == pytest.approx(qmn + lam * (1.0 - qmn), rel=1e-14)
    assert rec["acted"] and rec["floor_branch"] == "off_axis" and rec["q_floor_met"] is False
    assert 0.0 <= rec["q_min_after"] - lev <= QC.Q_LAND
    # at the boundary (levels just above and just below the plain axis q): the blended state's q0 changes by an amount
    # that shrinks with the distance from the boundary (continuous); the default rule keeps its jump
    q0 = float(q[0])

    def gap(eps, **kw):
        a = SAW.resistive_floor_steady(sol, RF, q0 * (1.0 + eps), **kw)
        b = SAW.resistive_floor_steady(sol, RF, q0 * (1.0 - eps), **kw)
        assert a[1]["floor_branch"] == "axis" and b[1]["floor_branch"] == "off_axis"
        return abs(float(QC._fq(a[2], a[0])[0]) - float(QC._fq(b[2], b[0])[0]))
    g1, g2 = gap(1e-5, transition="blend_margin"), gap(1e-6, transition="blend_margin")
    p1, p2 = gap(1e-5), gap(1e-6)
    assert g2 <= 0.2 * g1 and g2 < 1e-2 * p2 and p2 >= 0.5 * p1


# ====================================================================== frozen branch (diagnostic)
def _fit_with(monkeypatch, level, free=("s_bs", "a_corr"), **cfg_kw):
    """:func:`test_q_floor_continuous._default_fit` with extra model settings."""
    real = CM.CurrentModelSettings

    def settings(*a, **k):
        return real(*a, **dict(k, **cfg_kw))
    with monkeypatch.context() as mp:
        mp.setattr(QC.CM, "CurrentModelSettings", settings)
        return QC._default_fit(mp, q_floor=level, free=free)


def test_frozen_branch_keeps_the_first_assemblys_branch_for_the_whole_fit(monkeypatch):
    """At the branch boundary the default fit's record shows a flip; with the frozen branch every assembly of the fit
    is on the first assembly's branch (no change, no mixed Jacobian), recorded as frozen."""
    level, dq = QC._boundary_level()
    ref = QC._default_fit(monkeypatch, q_floor=level, free=("s_bs", "a_corr"))
    assert ref.record["floor_branch_record"]["branch_changed"] is True
    fz = _fit_with(monkeypatch, level, q_floor_branch_freeze="first_assembly")
    r = fz.record["floor_branch_record"]
    assert r["branch_changed"] is False and r["n_jacobians_mixed_branches"] == 0 and r["n_assemblies"] > 0
    c = fz.record["core"]
    assert c["branch_frozen"] is True and c["branch_freeze"].startswith("branch frozen")
    assert fz.record["model_settings"]["q_floor_branch_freeze"] == "first_assembly"
    assert CM._BRANCH_FREEZE.get() is None


def test_frozen_branch_outside_a_fit_is_the_plain_branch_and_forced_branches_are_recorded():
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, q_floor_branch_freeze="first_assembly")
    a = CM.assemble(lag, kin, src, {}, cfg, QC.IP_MET)
    assert a["core"]["branch_freeze"].startswith("no open freeze")
    a_d = CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0), QC.IP_MET)
    np.testing.assert_array_equal(a["state"].psi, a_d["state"].psi)
    sol = QC._solver(QC.IP_MET)           # plain branch (a)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, branch="off_axis")
    assert rec["floor_branch"] == "off_axis" and rec["floor_branch_plain"] == "axis" and not rec["acted"]
    sol_b = QC._reversed_shear()          # plain branch (b): forced (a) flattens from the axis
    st, rec, s2 = SAW.resistive_floor_steady(sol_b, RF, 1.0, branch="axis")
    assert rec["floor_branch"] == "axis" and rec["floor_branch_plain"] == "off_axis" and rec["acted"]
    _, rec, _ = SAW.resistive_floor_steady(sol, RF, 0.5, branch="axis")
    assert rec["floor_branch"] == "axis" and not rec["acted"] and "nothing to flatten" in rec["branch_frozen_note"]


# ====================================================================== pieces of the options tested directly
def test_total_current_cut_cell_is_the_overlap_weighted_blend_continuous_in_the_edge():
    """The cut cell takes the overlap-weighted source change, so the flat-at-level state is continuous in the edge:
    across a face the state changes in proportion to the distance from it (not by a whole cell's change)."""
    sol = QC._solver(QC.IP_MET)
    f_k = float(RF[30])

    def gap(eps):
        a, b = (SAW.ResistiveFloor(sol, 1.0, rho_inv=f_k * (1.0 + s * eps), rho_faces=RF,
                                   total_current="flat_at_level").with_solver() for s in (1.0, -1.0))
        sa, sb = a.steady_state(RF), b.steady_state(RF)
        return float(np.max(np.abs(QC._fq(a, sa) - QC._fq(b, sb))))
    g1, g2 = gap(1e-6), gap(1e-8)
    assert 0.0 < g2 <= 0.2 * g1
    # the change on a cut cell lies strictly between none and the whole cell's change
    c = SAW.ResistiveFloor(sol, 1.0, rho_inv=RF[30], rho_faces=RF).with_solver().coefficients(0.0, RF)
    r_mid = 0.5 * (RF[30] + RF[31])
    ds_mid = SAW.level_flattening_source_change(RF, c, True, r_mid, 1.0)
    ds_full = SAW.level_flattening_source_change(RF, c, True, RF[31], 1.0)
    assert 0.0 < ds_mid[30] / ds_full[30] < 1.0


@pytest.mark.parametrize("tc", ["flat_at_level", "flat_at_level_own_region"])
def test_the_source_hook_changes_the_non_inductive_current_with_the_sources(tc):
    """Where the option changes the cell sources, <J_NI.B> on the cells is changed with them: the coefficients keep
    the relation s_c = V' <J_NI.B> h / (2 pi F^2) of the plain solver, so the outputs stay consistent."""
    sol = QC._solver(QC.IP_LOW_Q)
    _, rec, s2 = SAW.resistive_floor_steady(sol, RF, 1.0, total_current=tc)
    h = np.diff(RF)
    for s_ in (sol, s2):
        c = s_.coefficients(0.0, RF)
        vc, fc = c.geo_c["dvol_drho"], c.geo_c["fpol"]
        np.testing.assert_allclose(c.s_c, vc * c.jni_c / (2.0 * np.pi * fc ** 2) * h, rtol=1e-12, atol=0.0)
    c0, c1 = sol.coefficients(0.0, RF), s2.coefficients(0.0, RF)
    assert np.any(c1.jni_c != c0.jni_c)


def test_blend_lambda_is_the_plain_axis_q_margin_fraction():
    """``lambda = 1 - (q0 - L) / (L - q_min)`` from the plain state: a level set so that ``q0 - L`` is half (a
    quarter) of ``L - q_min`` gives lambda 1/2 (3/4), and the blended level is ``q_min + lambda (L - q_min)``."""
    sol = QC._reversed_shear()
    q = SAW.face_q(sol.steady_state(RF), sol.geometry_at(0.0).phi_b)
    q0, qmn = float(q[0]), float(np.min(q[np.isfinite(q)]))
    for frac, lam in ((0.5, 0.5), (0.25, 0.75)):
        level = (q0 + frac * qmn) / (1.0 + frac)          # (q0 - level) / (level - q_min) = frac
        _, rec, _ = SAW.resistive_floor_steady(sol, RF, level, transition="blend_margin")
        assert rec["blend_lambda"] == pytest.approx(lam, rel=1e-12)
        assert rec["level_used"] == pytest.approx(qmn + lam * (level - qmn), rel=1e-12)


def _series_with(monkeypatch, **cfg_kw):
    """A three-slice series on the mocked solve of ``test_q_floor_continuous`` (the slice's state is the advanced
    model state), with extra model settings."""
    lag = SD._lag()
    ip = QC.IP_LOW_Q
    st0 = SD._steady(lag, ip)

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        fm_.n_solves += 1
        if state_fn is None:
            return SD._result(lag, st0)
        st = state_fn(lag)
        res = SD._result(lag, st)
        res.assembly["core"] = dict(st.meta["core"])
        return res
    with monkeypatch.context() as mp:
        mp.setattr(SE._cm, "consistent_solve", consistent)
        cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0, **cfg_kw)
        return SE.run_series(SD.FakeFM(dict(pred=lag)), SD._slices(np.arange(3) * 0.02, np.full(3, ip)), cfg,
                             SE.SeriesSettings(initial="relaxed"))


@pytest.mark.parametrize("tc", ["flat_at_level", "flat_at_level_own_region"])
def test_run_series_passes_the_total_current_option_to_every_interval(monkeypatch, tc):
    """The series applies the model's total-current option in its intervals: the interval records carry it and the
    carried states differ from those of the default series."""
    ref = _series_with(monkeypatch)
    out = _series_with(monkeypatch, q_floor_total_current=tc)
    assert out["status"] == ref["status"] == "completed"
    for a, b in zip(out["slices"][1:], ref["slices"][1:]):
        assert a["core"]["total_current"] == tc and "total_current" not in b["core"]
        assert a["core"]["acted_in_interval"]
        assert np.max(np.abs(np.asarray(a["carried_state"]["psi"]) - np.asarray(b["carried_state"]["psi"]))) > 0.0


# ====================================================================== records do not steer; record keys
def test_the_frozen_branch_is_carried_by_the_freeze_not_read_from_the_record(monkeypatch):
    """The branch the freeze holds is the one the floor used (filled through ``branch_holder``), whatever the record
    says: a record with a garbage branch does not change it."""
    lag, kin, src = LR._fixture_inputs()
    cfg = CM.CurrentModelSettings(r0=GEO.R0, q_floor_branch_freeze="first_assembly")
    real = SAW.resistive_floor_steady

    def garbage_record(*a, **k):
        st, rec, s_ = real(*a, **k)
        return st, dict(rec, floor_branch="garbage"), s_
    sol_m = CM.build_solver(lag, lag["jbs"], QC.IP_MET)
    hold, tok = CM.open_branch_freeze()
    try:
        with monkeypatch.context() as mp:
            mp.setattr(CM._saw, "resistive_floor_steady", garbage_record)
            CM._floor_with_options(sol_m, RF, cfg)
    finally:
        CM.close_branch_freeze(tok)
    assert hold["branch"] == "axis"
    used = {}
    real(QC._reversed_shear(), RF, 1.0, branch="axis", branch_holder=used)
    assert used == dict(branch="axis")
    real(QC._reversed_shear(), RF, 1.0, branch_holder=used)
    assert used == dict(branch="off_axis")


@pytest.mark.parametrize("tc", ["flat_at_level", "flat_at_level_own_region"])
def test_total_current_options_record_met_outside_the_region_under_its_own_key(tc):
    sol = QC._solver(QC.IP_LOW_Q)
    st, rec, s2 = SAW.resistive_floor_steady(sol, RF, 0.9, total_current=tc)
    q = QC._fq(s2, st)
    assert rec["q_floor_met_outside_region"] is bool(np.all(q[RF > rec["rho_floor"]] >= 0.9))
    assert rec["q_floor_met_outside_region_definition"] == SAW.Q_FLOOR_MET_OUTSIDE_REGION_DEFINITION
    assert rec["total_current_region"] == SAW.TOTAL_CURRENT_REGION_DEFINITIONS[tc]
    _, rd, _ = SAW.resistive_floor_steady(sol, RF, 0.9)
    assert "q_floor_met_outside_region" not in rd and "total_current_region" not in rd


def test_series_slice_records_count_the_filter_fits_jacobians(monkeypatch):
    """A series slice's branch record takes the Jacobian counts of its filter fit (the slice's own solves compute
    none); a slice without a fit records that it computed no Jacobian."""
    lag = SD._lag()
    ip = QC.IP_LOW_Q
    st0 = SD._steady(lag, ip)
    fit_rec = dict(n_jacobians=3, n_jacobians_mixed_branches=1, jacobians_mixed_branches=[dict(label="c1", mixed=True)],
                   jacobians_mixed_branches_total=1, jacobians_mixed_branches_truncated=False)

    def fake_filter(fm, sl, cfg, s, p_pred, state_fn, holder, res, ip1):
        return dict(absorbed=False, free=[], floor_branch_record=fit_rec), res

    def consistent(fm_, kin, src, params, cfg, ip=None, start=None, z0=None, label="", initial_profiles=None,
                   state_fn=None):
        fm_.n_solves += 1
        if state_fn is None:
            return SD._result(lag, st0)
        st = state_fn(lag)
        res = SD._result(lag, st)
        res.assembly["core"] = dict(st.meta["core"])
        return res
    monkeypatch.setattr(SE._cm, "consistent_solve", consistent)
    monkeypatch.setattr(SE, "_filter_slice", fake_filter)
    cfg = CM.CurrentModelSettings(r0=SD.GEO.R0, q_floor=1.0)
    out = SE.run_series(SD.FakeFM(dict(pred=lag)), SD._slices(np.arange(2) * 0.02, np.full(2, ip)), cfg,
                        SD.no_follow_series(initial="relaxed", filter=True))
    r = out["slices"][1]["floor_branch_record"]
    assert r["n_jacobians"] == 3 and r["n_jacobians_mixed_branches"] == 1 and r["jacobians_from"].startswith("the slice")
    assert r["jacobians_mixed_branches"] == fit_rec["jacobians_mixed_branches"]
    out0 = SE.run_series(SD.FakeFM(dict(pred=lag)), SD._slices(np.arange(2) * 0.02, np.full(2, ip)), cfg,
                         SE.SeriesSettings(initial="relaxed", filter=False))
    r0 = out0["slices"][1]["floor_branch_record"]
    assert r0["n_jacobians"] == 0 and r0["jacobians_from"].startswith("no fit")


# ====================================================================== total current with its own region
_OWN_CASES = [(QC.IP_MET, 1.0), (QC.IP_LOW_Q, 1.0), (QC.IP_LOW_Q, 0.9)]


def _own_criterion(sol, rf, level, r):
    """min q - level on the faces outside ``r`` of the flat-at-level state with the edge at ``r`` (the option's
    criterion), from the public pieces."""
    s_ = SAW.ResistiveFloor(sol, level, rho_inv=r, rho_faces=rf, total_current="flat_at_level").with_solver()
    q = QC._fq(s_, s_.steady_state(rf))
    return float(np.min(q[rf > r])) - level


@pytest.mark.parametrize("ip, level", _OWN_CASES)
@pytest.mark.parametrize("n", [100, 200, 400])
def test_own_region_option_meets_the_level_outside_its_region_and_its_region_is_the_smallest(ip, level, n):
    """Its own region: q is the level inside (to round-off), at or above it on every face outside; Ip is the solver's
    Ip condition; the region lies inside the plain outermost crossing, and the edge one face further in fails the
    criterion (the scan's bracket)."""
    sol = QC._solver(ip)
    rf = CurrentState.uniform_faces(n)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, level, total_current="flat_at_level_own_region")
    q = QC._fq(s2, st)
    r = rec["rho_floor"]
    assert rec["floor_branch"] == "axis" and rec["acted"]
    assert np.max(np.abs(q[rf <= r] - level)) <= FLAT_ATOL and rec["q_flat_max_abs_dev"] <= FLAT_ATOL
    assert np.all(q[rf > r] >= level) and rec["q_floor_met_outside_region"] is True
    assert rec["q_min_outside_region"] >= level
    assert s2.profiles(st)["ip"] == pytest.approx(sol.profiles(st0)["ip"], rel=IP_RTOL)
    plain_out = SAW.floor_region(rf, QC._fq(sol, st0), level)["rho_outer"]
    assert 0.0 < r <= plain_out
    k = int(np.searchsorted(rf, r))          # the first face at or beyond the edge; the face before it fails
    assert rf[k - 1] < r and (k - 1 == 0 or _own_criterion(sol, rf, level, float(rf[k - 1])) < 0.0)
    assert rec["total_current"] == "flat_at_level_own_region" and np.isfinite(rec["total_current_ni_added_over_ip"])


@pytest.mark.parametrize("ip, level", _OWN_CASES)
@pytest.mark.parametrize("n", [100, 200, 400])
def test_flat_at_level_state_with_its_edge_at_the_plain_outermost_crossing_meets_the_level_outside(ip, level, n):
    """The argument that the own-region search always finds an edge: with the edge at the plain state's outermost
    upward crossing the enclosed current of q = level equals the plain one there, so the state outside is the plain
    state (exact in the continuum; with the cut cell blended, approximately), which is at or above the level."""
    sol = QC._solver(ip)
    rf = CurrentState.uniform_faces(n)
    r = SAW.floor_region(rf, QC._fq(sol, sol.steady_state(rf)), level)["rho_outer"]
    assert _own_criterion(sol, rf, level, r) >= 0.0


def test_own_region_option_in_a_time_dependent_step_relaxes_to_its_single_slice_state():
    """In a series the step's region is the option's own region at the step's time: a long interval at fixed
    sources ends on the single-slice state of the option, and Ip after a step is the interval's Ip condition."""
    lag = SD._lag()
    ip = QC.IP_MET
    isol, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, 20.0, ip, ip)
    st_in = CurrentState(*(lambda s_: (s_.rho_faces, s_.psi, s_.psi_boundary, 0.0))(SD._steady(lag, ip)))
    tc = "flat_at_level_own_region"
    e1, c1 = SE.advance_q_floor(isol, st_in, 1e-3, SE.SeriesSettings(), 1e-3, 1.0, total_current=tc)
    assert SAW.ResistiveFloor(isol, 1.0, total_current=tc).with_solver().profiles(e1)["ip"] == pytest.approx(
        ip, rel=IP_RTOL)
    assert c1["total_current"] == tc and c1["acted"]
    e, c = SE.advance_q_floor(isol, st_in, 20.0, SE.SeriesSettings(), 0.25, 1.0, total_current=tc)
    st_ss, rec_ss, s_ss = SAW.resistive_floor_steady(isol, e.rho_faces, 1.0, t=20.0, total_current=tc)
    assert c["rho_floor"] == rec_ss["rho_floor"]
    q_e = SAW.face_q(e, isol.geometry_at(20.0).phi_b)
    assert np.max(np.abs(q_e - QC._fq(s_ss, st_ss))) <= 1e-9 * np.max(QC._fq(s_ss, st_ss))
