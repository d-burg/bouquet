"""Fast-ion charge must not be charged to the impurity (beam-heavy IMAS).

The corrected inversion lives in physics.impurity_charge_with_fast_ions
(the helper imas.py wires in): subtract the fast-ion charge from ne AND
renormalize the full-ne Zeff onto the thermal electrons.  Doing only the
subtraction recovers half the bias -- 3.19 instead of the true 6.0 for C6
at 25 % fast fraction -- and the first round of these tests blessed that
half-fix because they only asserted "closer than raw", not "correct".
"""
import numpy as np
import pytest

from bouquet.physics import (effective_impurity_charge,
                             impurity_charge_with_fast_ions)


def _case(fast_frac):
    """Synthetic C6 plasma with a known impurity charge and a fast-D population."""
    ne = np.full(64, 5.0e19)
    nC = 0.02 * ne                       # carbon, Z = 6
    n_fast = fast_frac * ne              # fast deuterium, Z = 1
    ni = ne - 6.0 * nC - n_fast          # thermal D closes quasineutrality
    zeff = (ni + 36.0 * nC) / ne         # thermal numerator / FULL ne (IMAS)
    return ne, ni, n_fast, zeff


def test_corrected_inversion_recovers_the_true_charge_exactly():
    """The load-bearing assertion: Z_imp == 6, not merely closer than raw."""
    for frac in (0.05, 0.10, 0.25):
        ne, ni, n_fast, zeff = _case(frac)
        z, ne_th = impurity_charge_with_fast_ions(ne, ni, zeff, n_fast)
        assert z == pytest.approx(6.0, abs=1e-9), frac
        np.testing.assert_allclose(ne_th, ne - n_fast)


def test_subtraction_alone_is_only_half_the_fix():
    """Regression pin for the half-applied form: thermal ne with the
    full-ne Zeff lands between raw and true (3.19 for this case) -- if
    this ever starts returning ~6 the helper's renormalization has been
    silently duplicated somewhere."""
    ne, ni, n_fast, zeff = _case(0.25)
    z_raw = effective_impurity_charge(ne, ni, zeff)
    z_half = effective_impurity_charge(ne - n_fast, ni, zeff)
    assert z_raw == pytest.approx(1.946, abs=0.01)
    assert z_half == pytest.approx(3.1875, abs=0.01)
    assert abs(z_half - 6.0) > 1.0          # decisively not the answer


def test_correction_is_inert_without_fast_ions():
    ne, ni, n_fast, zeff = _case(0.0)
    z, ne_th = impurity_charge_with_fast_ions(ne, ni, zeff, n_fast)
    assert z == pytest.approx(effective_impurity_charge(ne, ni, zeff),
                              abs=1e-12)
    np.testing.assert_array_equal(ne_th, ne)


def test_raw_bias_grows_with_fast_fraction():
    """Documents why the correction exists: the uncorrected call drifts
    monotonically off the true charge as the beam fraction rises."""
    errs = [abs(effective_impurity_charge(*_case(f)[:2], _case(f)[3]) - 6.0)
            for f in (0.0, 0.1, 0.25)]
    assert errs[0] < errs[1] < errs[2]


def test_overwhelming_fast_charge_degrades_loudly_not_negatively():
    """z_fast >= ne on some surfaces: those get a non-finite renormalized
    zeff and are excluded by the validity mask -- never a negative or
    wild Z_imp."""
    ne, ni, n_fast, zeff = _case(0.25)
    z_fast = ne.copy()                   # pathological: all charge is fast
    z, ne_th = impurity_charge_with_fast_ions(ne, ni, zeff, z_fast)
    assert z is None                     # no surface with usable dilution
    np.testing.assert_array_equal(ne_th, np.zeros_like(ne))


def test_imas_wiring_uses_the_helper():
    """imas.py must call the shipped helper -- the first-round tests never
    touched the changed file and stayed green through a revert."""
    import inspect
    import bouquet.io.imas as imas
    src = inspect.getsource(imas)
    assert "impurity_charge_with_fast_ions(ne, ni, Zeff, z_fast)" in src
    assert "effective_impurity_charge(ne_th" not in src


class TestThermalDrawPathConsistency:
    """(b) follow-up: every impurity consumer must run on ne - z_fast.

    The baseline reader derived Z_imp/p_imp on thermal ne while run.py's
    forward solves and the per-draw assembly still used the full ne with
    the thermal-derived Z_imp -- a sigma=0 pressure skew of
    e*z_fast*ti/Z_imp between reader and solver.
    """

    def _plasma(self, fast_frac=0.25):
        ne = np.full(48, 5.0e19)
        nC = 0.02 * ne
        z_fast = fast_frac * ne
        ni = ne - 6.0 * nC - z_fast
        zeff = (ni + 36.0 * nC) / ne
        return ne, ni, z_fast, zeff, nC

    def test_thermal_ni_derivation_recovers_the_exact_carbon(self):
        """main_ion_density_from_zeff(z_fast=...) inverts the same set the
        baseline Z_imp came from: ni and nz land exactly on the truth."""
        from bouquet.physics import main_ion_density_from_zeff
        ne, ni, z_fast, zeff, nC = self._plasma()
        ni_back = main_ion_density_from_zeff(ne, zeff, 6.0, z_fast=z_fast)
        np.testing.assert_allclose(ni_back, ni, rtol=1e-12)
        nz = (np.maximum(ne - z_fast, 0.0) - ni_back) / 6.0
        np.testing.assert_allclose(nz, nC, rtol=1e-12)

    def test_z_fast_none_reduces_to_the_plain_form(self):
        from bouquet.physics import main_ion_density_from_zeff
        ne, ni, z_fast, zeff, nC = self._plasma(fast_frac=0.0)
        np.testing.assert_allclose(
            main_ion_density_from_zeff(ne, zeff, 6.0, z_fast=None),
            main_ion_density_from_zeff(ne, zeff, 6.0,
                                       z_fast=np.zeros_like(ne)),
            rtol=1e-12)

    def test_baseline_carries_z_fast_field(self):
        from bouquet.baseline import Baseline
        assert Baseline.__dataclass_fields__["z_fast"].default is None

    def test_consumers_are_wired_thermal(self):
        """Source-level wiring pins: the draw assembly, the baseline
        reference assembly, and run.py's two forward-solve assemblies all
        subtract z_fast before impurity_pressure."""
        import inspect
        import bouquet.TokaMaker_interface as tmi
        import bouquet.run as brun
        tsrc = inspect.getsource(tmi)
        rsrc = inspect.getsource(brun)
        assert tsrc.count("z_fast=z_fast") >= 2       # sig pass-throughs
        assert "impurity_pressure(_ne_th_eq, ni_eq, ti_eq, Z_imp)" in tsrc
        assert "impurity_pressure(_kin2eq(_ne_bl)" in tsrc
        assert "impurity_pressure(_ne_th, ni, ti, bl.Z_imp)" in rsrc
        assert "impurity_pressure(_ne_th_eq, ni_eq, ti_eq,\n" in rsrc
