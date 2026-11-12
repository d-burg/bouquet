"""Fast tests of ``bouquet.kinrecon.diagnostics`` / ``score`` (no OFT needed).

An analytic stand-in for a TokaMaker equilibrium (circular flux surfaces on a
two-triangle mesh) exercises the bookkeeping: off-mesh and outside-LCFS
channels, the MSE formula against an independent hand calculation, E_r
double-count refusal, input immutability, and a noise-closure check.  The
live-solver tests are in ``test_score_solver.py``.  Synthetic numbers only.
"""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon import diagnostics as diag  # noqa: E402
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet  # noqa: E402
from bouquet.kinrecon.score import chi2_summary, chi2_under_sigma_convention, score  # noqa: E402

R0, Z0, A = 1.7, 0.0, 0.5      # analytic axis and LCFS minor radius [m]
PSI_AX, PSI_B = -1.0, -0.2     # psi on axis and at the LCFS [Wb/rad]
F0 = -3.4                      # R*B_phi [T m]


class _Interp:
    def __init__(self, kind):
        self.kind = kind

    def eval(self, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        R, Z = pts[:, 0], pts[:, 1]
        rho2 = ((R - R0) ** 2 + (Z - Z0) ** 2) / A ** 2
        psi = PSI_AX + (PSI_B - PSI_AX) * rho2
        if self.kind == "PSI":
            return psi[:, None]
        dpsi = (PSI_B - PSI_AX) / A ** 2
        dpdR, dpdZ = 2 * dpsi * (R - R0), 2 * dpsi * (Z - Z0)
        return np.column_stack((-dpdZ / R, F0 / R, dpdR / R))


class FakeEq:
    """Analytic equilibrium with the TokaMaker read-only API used by kinrecon."""

    def __init__(self, xpoint=None):
        self.r = np.array([[1.0, -1.0, 0.0], [2.4, -1.0, 0.0], [2.4, 1.0, 0.0], [1.0, 1.0, 0.0]])
        self.lc = np.array([[0, 1, 2], [0, 2, 3]])
        self.lim_contour = np.array([[1.01, -0.99], [2.39, -0.99], [2.39, 0.99], [1.01, 0.99]])
        self.o_point = np.array([R0, Z0])
        self.psi_bounds = np.array([PSI_B, PSI_AX])
        self._x = xpoint
        self.coils = {"C1": 1.5e4, "C2": -2.0e4}
        self.calls = 0

    def get_field_eval(self, kind):
        self.calls += 1
        return _Interp(kind.upper())

    def abspsi_to_normalized(self, psi):
        return (psi - PSI_AX) / (PSI_B - PSI_AX)

    def get_xpoints(self):
        return (None, False) if self._x is None else (np.array([self._x]), True)

    def get_globals(self):
        return 1.0e6, np.array([R0, Z0]), 10.0, 1.0e5, -0.01, -3.0, 1.0

    def get_coil_currents(self):
        return dict(self.coils), np.zeros(2)


def _group(name, measured, sigma, **geometry):
    n = len(measured)
    return ConstraintGroup(name=name, labels=[f"{name}{i}" for i in range(n)],
                           measured=measured, sigma=sigma, valid=np.ones(n, bool),
                           geometry=geometry, meta={"sigma_model": "test"})


# --- (e) MSE formula against an independent hand calculation -------------------
def test_mse_formula_hand_calculation():
    # One chord, numbers chosen by hand; evaluated term by term.
    B = np.array([[0.05, -1.9, -0.32]])          # B_R, B_phi, B_Z [T]
    c = np.array([[-0.8, 0.1, 0.02, 2.0e-6, 0.03, -0.6, 0.7, 1.0e-7, 5.0e-7]])
    er, ez = 2.0e4, -1.0e3                        # [V/m]
    num = (-0.8) * (-0.32) + 0.1 * 0.05 + 0.02 * (-1.9) + 2.0e-6 * 2.0e4
    den = 0.03 * (-0.32) + (-0.6) * 0.05 + 0.7 * (-1.9) + 1.0e-7 * 2.0e4 + 5.0e-7 * (-1.0e3)
    # num = 0.256 + 0.005 - 0.038 + 0.04 = 0.263
    # den = -0.0096 - 0.03 - 1.33 + 0.002 - 0.0005 = -1.3681
    assert num == pytest.approx(0.263, rel=1e-12)
    assert den == pytest.approx(-1.3681, rel=1e-12)
    tg = diag.mse_from_field(B, c, er, ez, quantity="tan_polarisation_angle")
    assert tg[0] == pytest.approx(0.263 / -1.3681, rel=1e-14)
    ang = diag.mse_from_field(B, c, er, ez)
    assert ang[0] == pytest.approx(np.arctan(0.263 / -1.3681), rel=1e-14)


def test_mse_efit_coefficient_mapping():
    # EFIT form (A1 Bz + A5 Er)/(A2 Bphi + A3 BR + A4 Bz + A6 Ez + A7 Er) with
    # c0=A1, c3=A5, c4=A4, c5=A3, c6=A2, c7=A7, c8=A6, c1=c2=0.
    A1, A2, A3, A4, A5, A6, A7 = 0.9, -0.7, 0.4, 0.05, 3e-6, 2e-7, 1e-7
    BR, Bphi, Bz, Er, Ez = 0.02, 2.1, 0.25, -1.5e4, 300.0
    efit = (A1 * Bz + A5 * Er) / (A2 * Bphi + A3 * BR + A4 * Bz + A6 * Ez + A7 * Er)
    c = np.array([[A1, 0, 0, A5, A4, A3, A2, A7, A6]])
    got = diag.mse_from_field([[BR, Bphi, Bz]], c, Er, Ez, quantity="tan_polarisation_angle")
    assert got[0] == pytest.approx(efit, rel=1e-14)


def test_mse_orientation_signs_and_bad_quantity():
    B = np.array([[0.05, -1.9, -0.32]])
    c = np.array([[-0.8, 0, 0, 0, 0.03, -0.6, 0.7, 0, 0]])
    flipped = diag.mse_from_field(-B * np.array([1, 0, 1]) + B * np.array([0, 1, 0]), c,
                                  quantity="tan_polarisation_angle")
    stated = diag.mse_from_field(B, c, sign_pol=-1.0, quantity="tan_polarisation_angle")
    assert stated[0] == flipped[0]
    with pytest.raises(ValueError):
        diag.mse_from_field(B, c, quantity="gamma_deg")


# --- geometry handling -------------------------------------------------------
def test_off_mesh_channel_is_nan_with_reason():
    eq = FakeEq()
    syn = diag.flux_loop(eq, {"r": [1.5, 3.0, -1.0], "z": [0.0, 0.0, 0.0]})
    assert np.isfinite(syn.value[0]) and np.isnan(syn.value[1:]).all()
    assert syn.reason[1] == diag.REASON_OFF_MESH
    assert syn.reason[2] == diag.REASON_BAD_GEOMETRY
    # IMAS COCOS 11 flux: -2*pi times TokaMaker's per-radian psi
    psi = PSI_AX + (PSI_B - PSI_AX) * (0.2 / A) ** 2
    assert syn.value[0] == pytest.approx(-2 * np.pi * psi, rel=1e-14)


def test_flux_loop_relative_to_reference_loop():
    eq = FakeEq()
    geo = {"r": np.array([1.5, 2.0, 2.2]), "z": np.array([0.0, 0.3, -0.2])}
    absolute = diag.flux_loop(eq, geo).value
    rel = diag.flux_loop(eq, geo, reference=(1.5, 0.0)).value
    assert rel[0] == 0.0
    assert np.allclose(rel, absolute - absolute[0], rtol=0, atol=1e-15)
    # through score(), with the reference named in meta
    g = _group("flux_loop", absolute - absolute[0], np.full(3, 1e-3), **geo)
    g.meta["flux_reference"] = "flux_loop0"
    tab = score(eq, MeasurementSet("synthetic", 1.0, 0.01, {"flux_loop": g}))
    assert np.max(np.abs(tab.residual("flux_loop"))) == 0.0
    g.meta["flux_reference"] = "nope"
    with pytest.raises(KeyError):
        score(eq, MeasurementSet("synthetic", 1.0, 0.01, {"flux_loop": g}))
    # an off-mesh reference poisons every channel, with a reason
    syn = diag.flux_loop(eq, geo, reference=(5.0, 0.0))
    assert np.isnan(syn.value).all() and all(r.startswith("reference loop") for r in syn.reason)


def test_probe_angle_convention_is_imas_clockwise():
    # IMAS DD: poloidal_angle clockwise from +R, so a = -pi/2 is a probe along +Z
    # and a = +pi/2 along -Z.  Pin the conversion helper and the diagnostic.
    from bouquet.kinrecon.measurements import probe_direction
    eq = FakeEq()
    B = _Interp("B").eval([[2.0, 0.1]])[0]
    for a, expect in ((-np.pi / 2, B[2]), (np.pi / 2, -B[2]), (0.0, B[0]),
                      (np.pi / 4, (B[0] - B[2]) / np.sqrt(2))):
        nr, nz = probe_direction(a)
        syn = diag.bpol_probe(eq, {"r": [2.0], "z": [0.1], "direction_r": [nr], "direction_z": [nz]})
        assert syn.value[0] == pytest.approx(expect, rel=1e-12, abs=1e-15)


def test_probe_direction_and_units():
    eq = FakeEq()
    s = 1 / np.sqrt(2)
    syn = diag.bpol_probe(eq, {"r": [2.0], "z": [0.1], "direction_r": [s], "direction_z": [s]})
    B = _Interp("B").eval([[2.0, 0.1]])[0]
    assert syn.value[0] == pytest.approx(s * B[0] + s * B[2], rel=1e-14)


def test_thomson_inside_outside_and_private_flux():
    eq = FakeEq(xpoint=(R0, -0.45))       # X-point inside the circle: cuts it at Z < -0.45
    geo = {"r": [1.7, 1.9, 2.3, 1.7], "z": [0.0, 0.3, 0.0, -0.48]}
    prof = lambda x: 5e19 * (1 - x ** 2) + 1e19  # noqa: E731
    syn = diag.thomson_ne(eq, geo, prof)
    psin = ((np.array(geo["r"]) - R0) ** 2 + np.array(geo["z"]) ** 2) / A ** 2
    assert syn.value[0] == pytest.approx(prof(0.0)) and syn.value[1] == pytest.approx(prof(psin[1]))
    assert np.isnan(syn.value[2]) and syn.reason[2] == diag.REASON_OUTSIDE_LCFS   # psi_N > 1
    assert np.isnan(syn.value[3]) and syn.reason[3] == diag.REASON_OUTSIDE_LCFS   # beyond X-point
    ext = diag.thomson_ne(eq, geo, prof, outside="extrapolate")
    assert ext.value[2] == pytest.approx(prof(psin[2]))


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        diag.q(FakeEq(), {})
    with pytest.raises(NotImplementedError):
        diag.vloop(FakeEq(), {})


# --- score bookkeeping -------------------------------------------------------
def _measurement_set(eq, rng=None, noise=1.0):
    geo_fl = {"r": np.linspace(1.1, 2.3, 30), "z": np.linspace(-0.9, 0.9, 30)[::-1]}
    geo_pr = {"r": np.linspace(1.1, 2.3, 30), "z": np.linspace(-0.8, 0.8, 30),
              "direction_r": np.ones(30), "direction_z": np.linspace(-1, 1, 30)}
    geo_ts = {"r": np.full(20, 1.9), "z": np.linspace(-0.6, 0.6, 20)}
    geo_mse = {"r": np.linspace(1.75, 2.15, 12), "z": np.zeros(12),
               "coefficients": np.tile([-0.8, 0, 0, 0, 0.03, -0.6, 0.7, 0, 0], (12, 1))}
    prof = lambda x: 4e19 * (1 - x) + 5e18  # noqa: E731
    groups = {}
    for name, syn, geo, sig in (
            ("flux_loop", diag.flux_loop(eq, geo_fl), geo_fl, 1e-3),
            ("bpol_probe", diag.bpol_probe(eq, geo_pr), geo_pr, 2e-3),
            ("ip", diag.ip(eq), {}, 5e3),
            ("pf_current", diag.pf_current(eq, {"coil_names": ["C1", "C2"]}),
             {"coil_names": ["C1", "C2"]}, 100.0),
            ("mse", diag.mse(eq, geo_mse), geo_mse, 2e-3),
            ("thomson_ne", diag.thomson_ne(eq, geo_ts, prof), geo_ts, 1e18)):
        v = np.nan_to_num(syn.value, nan=1e19)
        s = np.full(v.shape, sig)
        if rng is not None:
            v = v + noise * s * rng.standard_normal(v.shape)
        groups[name] = _group(name, v, s, **geo)
    return MeasurementSet("synthetic", 1.0, 0.01, groups), prof


def test_score_zero_noise_and_bookkeeping():
    eq = FakeEq()
    ms, prof = _measurement_set(eq)
    snapshot = {k: (g.valid.copy(), g.measured.copy(), dict(g.meta)) for k, g in ms.groups.items()}
    tab = score(eq, ms, kinetics={"n_e": prof})
    for name in tab.model:
        assert np.max(np.abs(tab.residual(name))) == 0.0
    # Thomson channels outside the circle were excluded with reasons through the
    # table's model-valid mask; the input set is the table's set, not a copy, untouched
    excl = tab.excluded["thomson_ne"]
    assert len(excl) > 0 and all(r == diag.REASON_OUTSIDE_LCFS for _, r in excl)
    mv = tab.model_valid["thomson_ne"]
    assert np.count_nonzero(~mv) == len(excl)
    assert tab.n_valid("thomson_ne") == np.count_nonzero(ms.groups["thomson_ne"].valid & mv)
    assert tab.measurements is ms
    for k, g in ms.groups.items():
        assert np.array_equal(g.valid, snapshot[k][0]) and np.array_equal(g.measured, snapshot[k][1])
        assert g.meta == snapshot[k][2]
    assert "score_notes" not in ms.meta
    assert any("E_R = 0" in n for n in tab.notes)


def test_score_skips_and_explicit_requests():
    eq = FakeEq()
    ms, prof = _measurement_set(eq)
    tab = score(eq, ms)            # no kinetics: thomson skipped with a note
    assert "thomson_ne" not in tab.model
    assert any("thomson_ne: skipped" in n for n in tab.notes)
    with pytest.raises(ValueError):
        score(eq, ms, groups=["thomson_ne"])
    ms.groups["q"] = _group("q", [1.0], [0.1])
    with pytest.raises(NotImplementedError):
        score(eq, ms, kinetics={"n_e": prof}, groups=["q"])
    ms.groups["mse"].meta["er_corrected"] = True
    with pytest.raises(ValueError):
        score(eq, ms, kinetics={"e_r": np.ones(12)}, groups=["mse"])


def test_score_noise_closure_fake():
    eq = FakeEq()
    rng = np.random.default_rng(20261110)
    ms, prof = _measurement_set(eq, rng=rng)
    summ = chi2_under_sigma_convention(score(eq, ms, kinetics={"n_e": prof}))
    c, n, _ = summ["total"]
    assert abs(c / n - 1.0) < 4.0 * np.sqrt(2.0 / n)


def test_chi2_summary_names_its_sigma_conventions():
    eq = FakeEq()
    ms, prof = _measurement_set(eq)
    for g in ms.groups.values():
        g.meta.pop("sigma_model_id", None)
    first = next(iter(ms.groups))
    ms.groups[first].meta["sigma_model_id"] = "model_a:reading"
    tab = score(eq, ms, kinetics={"n_e": prof})
    summ = chi2_under_sigma_convention(tab)
    assert chi2_summary is chi2_under_sigma_convention                 # deprecated name, same function
    assert set(summ.sigma_conventions) == set(summ)                     # every row named, total included
    assert summ.sigma_conventions[first] == "model_a:reading"
    others = {k: v for k, v in summ.sigma_conventions.items() if k not in (first, "total")}
    assert all(v == (ms.groups[k].meta.get("sigma_model") or "unstated") for k, v in others.items())
    assert summ.sigma_conventions["total"] == sorted(set(summ.sigma_conventions[k] for k in tab.model))
    assert summ.mixed == (len(summ.sigma_conventions["total"]) > 1)
    c, n, r = summ[first]                                               # tuple rows unchanged
    assert n == tab.n_valid(first)


def test_residual_table_model_valid_mask():
    from bouquet.kinrecon.interfaces import ResidualTable
    g = ConstraintGroup(name="x", labels=["a", "b", "c", "d"], measured=[1.0, 2.0, 3.0, 4.0],
                        sigma=[1.0, 1.0, 2.0, 1.0], valid=[True, True, True, False])
    ms = MeasurementSet("synthetic", 1.0, 0.01, {"x": g})
    model = np.array([2.0, np.nan, 5.0, 100.0])
    # without a mask every measurement-valid channel is used (NaN propagates)
    assert np.isnan(ResidualTable(ms, {"x": model}).chi2())
    tab = ResidualTable(ms, {"x": model}, model_valid={"x": np.isfinite(model)},
                        excluded={"x": [("b", "test reason")]}, notes=["n1"])
    np.testing.assert_array_equal(tab.used("x"), [True, False, True, False])
    np.testing.assert_array_equal(tab.residual("x"), [1.0, 1.0])
    assert tab.chi2() == 2.0 and tab.n_valid() == 2
    assert g.valid.tolist() == [True, True, True, False]        # set untouched
    assert tab.notes == ["n1"] and tab.excluded["x"][0][1] == "test reason"


def test_one_mse_formula():
    from bouquet.kinrecon import measurements as M
    assert M.mse_from_field is diag.mse_from_field
    c = np.tile([-0.8, 0.1, 0.2, 1e-5, 0.03, -0.6, 0.7, 2e-6, 1e-6], (3, 1))
    B = np.array([[0.1, -2.0, -0.3], [0.05, -1.9, 0.2], [-0.1, -2.1, 0.1]])
    np.testing.assert_array_equal(
        M.tan_gamma_imas(c, B[:, 0], B[:, 2], B[:, 1], 1e4, -2e3),
        diag.mse_from_field(B, c, 1e4, -2e3, quantity="tan_polarisation_angle"))


def test_package_reexports_score_without_oft():
    import subprocess
    code = ("import sys; sys.modules['OpenFUSIONToolkit'] = None\n"
            "import bouquet.kinrecon as k\n"
            "from bouquet.kinrecon.score import score as s\n"
            "assert k.score is s and callable(k.chi2_summary)\n"
            "assert 'OpenFUSIONToolkit' not in [m for m in sys.modules if sys.modules[m]]\n")
    env = dict(os.environ, PYTHONPATH=_REPO)
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr
