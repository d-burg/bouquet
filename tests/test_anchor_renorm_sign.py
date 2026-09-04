"""_AnchorIpRenorm P'-sign handling: convention flip vs measure validation.

The sign(dot(probe, reference)) selection expresses the affine c term in
the reference profile's current-direction convention -- legitimate here
(unlike the ohmic closure, which fixes +1 by construction).  But the
self_check used to be evaluated WITH the flipped sign, so a legitimate -1
flip read as self_check = -200 % and drowned the geometry validation it
exists to perform.  These tests pin the decoupling with a synthetic
equilibrium stub -- no solver.
"""
import numpy as np
import pytest

from bouquet.TokaMaker_interface import _AnchorIpRenorm

_N = 81


class _StubEq:
    """Just enough of TokaMaker_equilibrium for fsa_current_geometry."""

    psi_bounds = (0.0, 0.9)

    def __init__(self):
        self._psi = None

    def copy_eq(self):
        return self

    def get_q(self, psi=None):
        psi = np.asarray(psi, dtype=float)
        R0, a = 1.7, 0.6
        r = a * np.sqrt(np.clip(psi, 1e-6, None))
        ravgs = {
            "<R>": R0 + 0.1 * r ** 2 / a,
            "<1/R>": (1.0 / R0) * (1.0 + 0.05 * (r / a) ** 2),
            "<1/R^2>": ((1.0 / R0) * (1.0 + 0.05 * (r / a) ** 2)) ** 2
                        * (1.0 + 0.02 * (r / a) ** 2),
            "dV/dPsi": 4.0 * np.pi ** 2 * R0 * r
                        * (a / (2.0 * np.sqrt(np.clip(psi, 1e-6, None)))),
        }
        return None, None, ravgs

    def get_profiles(self, psi=None):
        psi = np.asarray(psi, dtype=float)
        pprime = -8.0e3 * (1.0 - psi)
        F = 3.0 * np.ones_like(psi)
        Fp = (-0.4 * (1.0 - psi) ** 2) / F
        return [None, F, Fp, None, pprime]

    def get_stats(self, lcfs_pad=None):
        return {"Ip": self._ip_true}


def _make_stub_and_probe():
    """Stub whose reported Ip equals the measure of its own GS profile, so
    the self-check is ~0 by construction and any deviation is machinery."""
    from bouquet.utils import (fsa_current_geometry, Ip_fsa_weights,
                               eq_jphi_profile)
    from scipy.integrate import trapezoid
    eq = _StubEq()
    psi = np.linspace(0.01, 0.99, _N)
    geom = fsa_current_geometry(eq, psi)
    probe = eq_jphi_profile(geom, "jphi-linterp", eq=eq)
    w, c = Ip_fsa_weights(geom, convention="jphi-linterp")
    eq._ip_true = float(trapezoid(w * probe, psi)) + c
    return eq, psi, probe


def test_same_convention_reference_keeps_plus_one_and_clean_self_check():
    eq, psi, probe = _make_stub_and_probe()
    r = _AnchorIpRenorm(eq, psi, probe.copy(), eq._ip_true, 1e-3,
                        mode="exact")
    assert r._pprime_sign == 1.0
    assert abs(r.self_check) < 1e-9


def test_flipped_convention_flips_c_but_not_the_self_check(capsys):
    """The load-bearing decoupling: a reference in the OPPOSITE convention
    selects -1 (c expressed in the reference's convention, with a printed
    note) while self_check still validates the measure at ~0 -- the old
    evaluation returned -200 % here."""
    eq, psi, probe = _make_stub_and_probe()
    r = _AnchorIpRenorm(eq, psi, -probe, -eq._ip_true, 1e-3, mode="exact")
    assert r._pprime_sign == -1.0
    assert "OPPOSITE current-direction convention" in capsys.readouterr().out
    assert abs(r.self_check) < 1e-9, (
        f"self_check {r.self_check:+.3f} -- the convention flip leaked back "
        "into the measure validation")
