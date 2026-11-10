"""EFIT AA1GAM..AA7GAM -> IMAS 9-coefficient mapping (synthetic values)."""
import numpy as np
import pytest

from bouquet.kinrecon.measurements import (efit_to_imas_coefficients,
                                           imas_to_efit_coefficients, tan_gamma_imas)


def _efit_tan(A, b_r, b_z, b_phi, e_r, e_z):
    """The EFIT form written out independently of the mapping."""
    num = A[1] * b_z + A[8] * b_r + A[5] * e_r
    den = A[2] * b_phi + A[3] * b_r + A[4] * b_z + A[6] * e_z + A[7] * e_r
    return num / den


def test_order_matches_dd_documentation():
    c = efit_to_imas_coefficients(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    # numerator BZ, BR, Bphi, ER ; denominator BZ, BR, Bphi, ER, EZ
    np.testing.assert_array_equal(c, [[1.0, 8.0, 0.0, 5.0, 4.0, 3.0, 2.0, 7.0, 6.0]])


def test_forward_model_agrees_with_efit_form():
    rng = np.random.default_rng(3)
    n = 50
    A = {i: rng.normal(size=n) for i in range(1, 9)}
    A[2] = A[2] + 3.0  # keep the denominator away from zero
    b_r, b_z, b_phi = rng.normal(size=(3, n)) * [[0.2], [0.3], [2.0]]
    e_r, e_z = rng.normal(size=(2, n)) * 1e4
    c = efit_to_imas_coefficients(A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8])
    np.testing.assert_allclose(tan_gamma_imas(c, b_r, b_z, b_phi, e_r, e_z),
                               _efit_tan(A, b_r, b_z, b_phi, e_r, e_z), rtol=1e-12)


def test_missing_coefficients_are_zero_and_inverse_round_trips():
    a1, a2, a3, a4 = (np.array([0.1, 0.2]) * k for k in (1, 2, 3, 4))
    c = efit_to_imas_coefficients(a1, a2, a3, a4)
    assert c.shape == (2, 9)
    np.testing.assert_array_equal(c[:, [1, 2, 3, 7, 8]], 0.0)
    back = imas_to_efit_coefficients(c)
    for k, v in (("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4)):
        np.testing.assert_array_equal(back[k], v)


def test_bphi_numerator_has_no_efit_slot():
    c = np.zeros((1, 9))
    c[0, 2] = 1.0
    with pytest.raises(ValueError, match="B_phi"):
        imas_to_efit_coefficients(c)
