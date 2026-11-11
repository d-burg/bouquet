"""Orientation (signs) of the scored quantities: reversed Ip and reversed B_t for every group (no OFT).

score() compares every signed quantity in the discharge's frame through a STATED orientation map
(``orientation={"sign_ip", "sign_bt"}``) and refuses when the equilibrium's signs, after the map, disagree
with the discharge's (measured Ip sign, declared ``discharge_orientation``).  These tests pin that on an
analytic stand-in equilibrium whose every output is mirrored exactly by the parity symmetry
(psi -> -psi, coil currents -> -, B_phi -> -), so a correct map gives chi^2 = 0 exactly; plus the refusals of
:func:`bouquet.kinrecon.forward.orientation_record` for the forward-solve frame.
"""
import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as diag
from bouquet.kinrecon.forward import orientation_record
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
from bouquet.kinrecon.score import chi2_report, score, score_common

R0, A = 1.7, 0.5
GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current", "mse")


class Eq:
    """psi = s_ip * c ((R-R0)^2 + Z^2) (TokaMaker sign: s_ip = +1 is Ip > 0), B_phi = F0/R; coil currents and the
    diamagnetic flux mirror with the plasma current and with F0 respectively."""

    def __init__(self, s_ip=1.0, F0=-3.4):
        self.s, self.F0 = s_ip, F0
        self.r = np.array([[1.0, -1.0, 0], [2.4, -1.0, 0], [2.4, 1.0, 0], [1.0, 1.0, 0]], float)
        self.lc = np.array([[0, 1, 2], [0, 2, 3]])
        self.lim_contour = self.r[:, :2]

    def get_field_eval(self, kind):
        eq = self

        class E:
            def eval(self, pts):
                pts = np.asarray(pts, float).reshape(-1, 2)
                Rr, Zz = pts[:, 0], pts[:, 1]
                c = -eq.s * 3.2
                if kind.upper() == "PSI":
                    return (c * ((Rr - R0) ** 2 + Zz ** 2))[:, None]
                return np.column_stack((-2 * c * Zz / Rr, eq.F0 / Rr, 2 * c * (Rr - R0) / Rr))
        return E()

    def get_globals(self):
        return (self.s * 1.0e6, None, None, None, np.sign(self.F0) * 0.01, None, None)

    def get_coil_currents(self):
        return {"F1": self.s * 2.0e4, "F2": -self.s * 5.0e3}, None


def _ms(eq, mse_orientation=None, declare=None):
    geo = dict(flux_loop={"r": [1.2, 2.2], "z": [0.5, -0.4]},
               bpol_probe={"r": [1.2, 2.2], "z": [0.3, 0.1], "direction_r": [0.0, 0.6], "direction_z": [1.0, 0.8]},
               pf_current={"coil_names": ["F1", "F2"]},
               mse={"r": [1.9, 2.1], "z": [0.0, 0.0],
                    "coefficients": np.array([[0.8, 0, 0, 0, 0.05, 0.3, -0.7, 0, 0]] * 2)})
    vals = dict(flux_loop=diag.flux_loop(eq, geo["flux_loop"]).value, bpol_probe=diag.bpol_probe(eq, geo["bpol_probe"]).value,
                ip=diag.ip(eq).value, diamagnetic_flux=diag.diamagnetic_flux(eq).value,
                pf_current=diag.pf_current(eq, geo["pf_current"]).value, mse=diag.mse(eq, geo["mse"]).value)
    gs = {}
    for k, v in vals.items():
        meta = {"sigma_model": "t"}
        if k == "mse" and mse_orientation is not None:
            meta["field_orientation"] = mse_orientation
        gs[k] = ConstraintGroup(k, [f"{k}{i}" for i in range(len(v))], v, np.abs(v) * 0.01 + 1e-6,
                                np.ones(len(v), bool), geometry=geo.get(k, {}), meta=meta)
    meta = {"discharge_orientation": declare} if declare else {}
    return MeasurementSet("t", 0, 0, gs, meta=meta)


@pytest.mark.parametrize("F0", [-3.4, 3.4])
def test_reversed_ip_scored_through_the_stated_map(F0):
    data = _ms(Eq(s_ip=-1.0, F0=F0))                  # a reversed-Ip discharge
    eq = Eq(s_ip=+1.0, F0=F0)                         # solved in the positive-Ip frame
    with pytest.raises(ValueError, match="orientation mismatch"):
        score(eq, data)                               # identity map: refused, not a huge chi2
    tab = score(eq, data, orientation={"sign_ip": -1, "sign_bt": 1})
    for g in GROUPS:
        assert tab.chi2(g) == pytest.approx(0.0, abs=1e-20), g
    assert tab.frame["sign_ip"] == -1.0 and tab.frame["source"] == "caller"
    # the same map from the set's meta
    data.meta["orientation_map"] = {"sign_ip": -1, "sign_bt": 1}
    assert score(eq, data).chi2() == pytest.approx(0.0, abs=1e-20)
    with pytest.raises(ValueError, match="Ip <= 0"):
        orientation_record(data, F0=F0)               # the forward solve itself stays in the positive frame


def test_reversed_bt_every_group():
    data = _ms(Eq(F0=-3.4), declare={"bt_sign": -1})   # B_t < 0 declared for the discharge
    eq_pos = Eq(F0=3.4)                                # an equilibrium solved with F0 > 0
    with pytest.raises(ValueError, match="B_t"):
        score(eq_pos, data)
    tab = score(eq_pos, data, orientation={"sign_ip": 1, "sign_bt": -1})
    for g in ("flux_loop", "bpol_probe", "ip", "pf_current", "diamagnetic_flux", "mse"):
        assert tab.chi2(g) == pytest.approx(0.0, abs=1e-20), g
    # solved with the discharge's own signed F0: identity map, everything matches
    for F0 in (-3.4, 3.4):
        d = _ms(Eq(F0=F0), declare={"bt_sign": np.sign(F0)})
        assert score(Eq(F0=F0), d).chi2() == 0.0
        rec = orientation_record(d, F0)
        assert rec["F0_sign"] == np.sign(F0) and rec["ip_sign"] == 1.0
    # without a declared B_t sign the B_t error cannot be detected, and shows as chi2 in MSE and the
    # diamagnetic flux only (documented limitation: declare discharge_orientation)
    tab = score(Eq(F0=3.4), _ms(Eq(F0=-3.4)))
    assert tab.chi2("mse") > 1.0 and tab.chi2("diamagnetic_flux") > 1.0
    assert tab.chi2("flux_loop") == 0.0 and tab.chi2("bpol_probe") == 0.0 and tab.chi2("pf_current") == 0.0


def test_both_reversed_and_mse_map_consistency():
    data = _ms(Eq(s_ip=-1.0, F0=3.4), declare={"ip_sign": -1, "bt_sign": 1})
    tab = score(Eq(s_ip=1.0, F0=-3.4), data, orientation={"sign_ip": -1, "sign_bt": -1})
    assert tab.chi2() == pytest.approx(0.0, abs=1e-20)
    # an MSE field_orientation that contradicts the stated map is refused
    bad = _ms(Eq(s_ip=-1.0, F0=3.4), mse_orientation=(1.0, 1.0))
    with pytest.raises(ValueError, match="field_orientation"):
        score(Eq(s_ip=1.0, F0=-3.4), bad, orientation={"sign_ip": -1, "sign_bt": -1})
    with pytest.raises(ValueError, match="field_orientation"):
        orientation_record(_ms(Eq(), mse_orientation=(1.0, -1.0)), -3.4)
    # a declared Ip sign that contradicts the measured Ip is refused
    with pytest.raises(ValueError, match="disagrees"):
        score(Eq(), _ms(Eq(), declare={"ip_sign": -1}))


def test_score_common_uses_the_intersection_and_report_states_dof():
    data = _ms(Eq())
    data.groups["flux_loop"].geometry["r"] = np.array([1.2, 2.2])

    class EqOff(Eq):
        """Same equilibrium, but its mesh does not reach the second flux loop (model returns NaN there)."""
        def get_field_eval(self, kind):
            base = Eq.get_field_eval(self, kind)

            class E:
                def eval(self, pts):
                    out = base.eval(pts)
                    pts = np.asarray(pts, float).reshape(-1, 2)
                    out[pts[:, 0] > 2.1] = np.nan
                    return out
            return E()

    t_alone = score(Eq(), data)
    assert t_alone.n_valid("flux_loop") == 2
    tabs, rec = score_common([Eq(), EqOff()], data, groups=["flux_loop"])
    assert [t.n_valid("flux_loop") for t in tabs] == [1, 1]
    assert rec["flux_loop"]["n_common"] == 1 and rec["flux_loop"]["removed_for_comparability"] == ["flux_loop1"]
    assert any(r == "not modelable in every compared equilibrium" for _, r in tabs[0].excluded["flux_loop"])
    rep = chi2_report(tabs[0], n_params=0)
    assert rep["total"]["n_channels"] == 1 and rep["total"]["dof"] == 1 and "not a pass/fail" in rep["interpretation"]
    assert chi2_report(t_alone, n_params=1)["total"]["dof"] == t_alone.n_valid() - 1


def test_zero_sigma_is_refused_at_scoring():
    data = _ms(Eq())
    data.groups["ip"].sigma[0] = 0.0
    tab = score(Eq(), data)
    with pytest.raises(ValueError, match="sigma <= 0"):
        tab.chi2("ip")
    data.groups["ip"].weight[0] = 0.0                  # weight 0: not used, not counted
    tab = score(Eq(), data)
    assert tab.n_valid("ip") == 0 and tab.chi2("ip") == 0.0
