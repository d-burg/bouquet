"""Orientation (signs) of the scored quantities: reversed Ip and reversed B_t (no OFT).

score() compares Ip, loops, probes and the diamagnetic flux signed; only MSE has a stated
orientation map (independent review, report 32, F4).  These tests pin that behavior on an analytic
stand-in equilibrium and the refusals of :func:`bouquet.kinrecon.forward.orientation_record`.
"""
import numpy as np
import pytest

from bouquet.kinrecon import diagnostics as diag
from bouquet.kinrecon.forward import orientation_record
from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
from bouquet.kinrecon.score import score

R0, A = 1.7, 0.5


class Eq:
    """psi = s_ip * c ((R-R0)^2 + Z^2) (TokaMaker sign: s_ip = +1 is Ip > 0), B_phi = F0/R."""

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


def _ms(eq, mse_orientation=None):
    geo = dict(flux_loop={"r": [1.2, 2.2], "z": [0.5, -0.4]},
               bpol_probe={"r": [1.2, 2.2], "z": [0.3, 0.1], "direction_r": [0.0, 0.6], "direction_z": [1.0, 0.8]},
               mse={"r": [1.9, 2.1], "z": [0.0, 0.0],
                    "coefficients": np.array([[0.8, 0, 0, 0, 0.05, 0.3, -0.7, 0, 0]] * 2)})
    vals = dict(flux_loop=diag.flux_loop(eq, geo["flux_loop"]).value, bpol_probe=diag.bpol_probe(eq, geo["bpol_probe"]).value,
                ip=diag.ip(eq).value, diamagnetic_flux=diag.diamagnetic_flux(eq).value,
                mse=diag.mse(eq, geo["mse"]).value)
    gs = {}
    for k, v in vals.items():
        meta = {"sigma_model": "t"}
        if k == "mse" and mse_orientation is not None:
            meta["field_orientation"] = mse_orientation
        gs[k] = ConstraintGroup(k, [f"{k}{i}" for i in range(len(v))], v, np.abs(v) * 0.01 + 1e-6,
                                np.ones(len(v), bool), geometry=geo.get(k, {}), meta=meta)
    return MeasurementSet("t", 0, 0, gs)


def test_reversed_ip_flips_every_signed_group_and_is_refused():
    data = _ms(Eq(s_ip=-1.0))                    # a reversed-Ip discharge
    tab = score(Eq(s_ip=+1.0), data)             # solved in the positive frame
    for g in ("flux_loop", "bpol_probe", "ip"):
        assert tab.chi2(g) > 1e3, g              # every signed quantity flips sign
    assert tab.chi2("mse") > 1.0                 # B_pol flips, B_phi does not
    with pytest.raises(ValueError, match="Ip <= 0"):
        orientation_record(data, F0=-3.4)


def test_reversed_bt_is_consistent_when_F0_is_signed():
    for F0 in (-3.4, 3.4):
        data = _ms(Eq(F0=F0))
        tab = score(Eq(F0=F0), data)             # same sign of F0 in the solve: everything matches
        assert tab.chi2() == 0.0
        rec = orientation_record(data, F0)
        assert rec["F0_sign"] == np.sign(F0) and rec["ip_sign"] == 1.0
    # the same B_t < 0 data scored on an equilibrium solved with F0 > 0 fails MSE and the diamagnetic flux
    tab = score(Eq(F0=3.4), _ms(Eq(F0=-3.4)))
    assert tab.chi2("mse") > 1.0 and tab.chi2("diamagnetic_flux") > 1.0
    assert tab.chi2("flux_loop") == 0.0 and tab.chi2("bpol_probe") == 0.0
    # a stated MSE map other than (1, 1) is refused for a solve in the discharge's own frame
    with pytest.raises(ValueError, match="field_orientation"):
        orientation_record(_ms(Eq(), mse_orientation=(1.0, -1.0)), -3.4)
