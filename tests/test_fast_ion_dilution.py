"""Fast-ion charge must not be charged to the impurity (issue: beam-heavy IMAS)."""
import numpy as np

from bouquet.physics import effective_impurity_charge


def _case(fast_frac):
    """Synthetic C6 plasma with a known impurity charge and a fast-D population."""
    ne = np.full(64, 5.0e19)
    nC = 0.02 * ne                       # carbon, Z = 6
    n_fast = fast_frac * ne              # fast deuterium, Z = 1
    ni = ne - 6.0 * nC - n_fast          # thermal D closes quasineutrality
    zeff = (ni + 36.0 * nC) / ne
    return ne, ni, n_fast, zeff


def test_fast_ion_charge_biases_z_imp_low():
    """Raw ne charges the beam to the impurity, pulling Z_imp off C6."""
    ne, ni, n_fast, zeff = _case(0.25)
    z_raw = effective_impurity_charge(ne, ni, zeff)
    z_corr = effective_impurity_charge(ne - n_fast, ni, zeff)
    assert z_raw < z_corr
    assert abs(z_corr - 6.0) < abs(z_raw - 6.0)


def test_correction_is_inert_without_fast_ions():
    ne, ni, n_fast, zeff = _case(0.0)
    assert np.allclose(n_fast, 0.0)
    assert effective_impurity_charge(ne - n_fast, ni, zeff) == \
           effective_impurity_charge(ne, ni, zeff)


def test_bias_grows_with_fast_fraction():
    errs = [abs(effective_impurity_charge(*_case(f)[:2], _case(f)[3]) - 6.0)
            for f in (0.0, 0.1, 0.25)]
    assert errs[0] < errs[1] < errs[2]
