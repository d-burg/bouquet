"""The p-file ion species roles (issue #19).

``PFile.compute_zeff`` used to read the main ion as ``Z[-2]``: on a natural
two-entry ``[impurity, main]`` block that lands on the IMPURITY, and a carbon
plasma came out at Z_eff ~ 30 instead of ~ 1.3, silently (34 on the fixture
below).  The roles are now
identified by charge and mass and by whether the file carries a beam density
(:meth:`PFile.ion_roles`), and an ambiguous block raises.  The shipped p-files
are three-entry ``[C, D, D]`` blocks: their numbers must not move by a bit.
"""
import os

import numpy as np
import pytest

from bouquet.io.pfile import PFile, read_pfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHIPPED = (
    os.path.join(_HERE, "data", "p_synthetic.peqdsk"),
    os.path.join(_HERE, "..", "examples", "D3D-like",
                 "D3Dlike_Hmode_baseline.peqdsk"),
)


def _pf(Z, A, nb=None):
    pf = PFile.new()
    x = np.linspace(0.0, 1.0, 64)
    pf.set_profile("ne", x, 3.0 * (1.0 - 0.9 * x ** 2))
    pf.set_profile("ni", x, 2.8 * (1.0 - 0.9 * x ** 2))
    if nb is not None:
        pf.set_profile("nb", x, nb * (1.0 - 0.9 * x ** 2))
    pf.set_ion_species(N=list(Z), Z=list(Z), A=list(A))
    return pf


def _zeff(pf):
    pf.compute_quasineutrality()
    return pf.compute_zeff()[1]


#: on axis: ne 3, ni 2.8, nz1 = 0.2/6 -> (2.8 + 0.2/6 * 36)/3
_PHYSICAL = (2.8 + 0.2 / 6.0 * 36.0) / 3.0


def test_a_two_entry_block_gives_the_physical_zeff():
    z = _zeff(_pf([6, 1], [12, 2]))
    assert z[0] == pytest.approx(_PHYSICAL, rel=1e-12)
    assert z[0] < 2.0          # the aliased read gave 34 here


def test_two_and_three_entry_blocks_agree_without_a_beam():
    two = _zeff(_pf([6, 1], [12, 2]))
    three = _zeff(_pf([6, 1, 1], [12, 2, 2]))
    np.testing.assert_array_equal(two, three)


def test_the_roles_do_not_depend_on_the_order():
    np.testing.assert_array_equal(_zeff(_pf([1, 6], [2, 12])),
                                  _zeff(_pf([6, 1], [12, 2])))


def test_three_entry_roles_with_a_beam():
    pf = _pf([6, 1, 1], [12, 2, 2], nb=0.05)
    r = pf.ion_roles()
    assert (r["impurity"], r["main"], r["beam"]) == (0, 1, 2)
    assert (r["Z_imp"], r["Z_main"], r["Z_beam"]) == (6.0, 1.0, 1.0)


def test_the_beam_keeps_its_listed_charge():
    """The last species is the beam whatever its charge (the per-draw
    writer stores the fast ions' effective charge there)."""
    r = _pf([6, 1, 2], [12, 2, 2], nb=0.05).ion_roles()
    assert (r["main"], r["beam"], r["Z_beam"]) == (1, 2, 2.0)


def test_a_zero_beam_density_is_no_beam():
    r = _pf([6, 1], [12, 2], nb=0.0).ion_roles()
    assert r["beam"] is None and r["Z_main"] == 1.0


def test_a_two_entry_block_with_a_beam_density_is_ambiguous_for_zeff():
    """[C, D] with a non-zero nb: D could be the main ion or the beam."""
    pf = _pf([6, 1], [12, 2], nb=0.05)
    pf.set_profile("nz1", np.linspace(0, 1, 64), np.full(64, 0.01))
    with pytest.raises(ValueError, match="ambiguous"):
        pf.compute_zeff()


def test_quasineutrality_needs_only_the_impurity_and_the_beam():
    """ni is given, so quasineutrality never needs the main ion: [C, D] with
    a non-zero nb reads D as the beam (Z_beam 1, the 1.4.0 value for a
    two-entry block), and an impurity-only block works without a beam."""
    pf = _pf([6, 1], [12, 2], nb=0.05)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pf.compute_quasineutrality()
    nb = pf._get_data("nb")
    np.testing.assert_array_equal(pf._get_data("nz1"),
                                  (pf.ne - pf.ni - 1.0 * nb) / 6.0)
    pf = _pf([6], [12])
    pf.compute_quasineutrality()
    np.testing.assert_array_equal(pf._get_data("nz1"), (pf.ne - pf.ni) / 6.0)
    with pytest.raises(ValueError, match="no hydrogenic main ion"):
        pf.compute_zeff()


@pytest.mark.parametrize("Z,A,nb,match", [
    ([6, 2], [12, 4], None, "non-hydrogenic main ion is not supported"),
    ([2], [4], None, "no hydrogenic main ion"),          # a helium plasma
    ([6, 1], [12, 12], None, "non-hydrogenic main ion is not supported"),
])
def test_unidentifiable_roles_raise(Z, A, nb, match):
    with pytest.raises(ValueError, match=match):
        _pf(Z, A, nb=nb).ion_roles()


def test_an_impurity_read_as_the_beam_is_ambiguous():
    """[C] with a non-zero nb: the only species is read as the beam, so the
    impurity quasineutrality needs is ambiguous."""
    with pytest.raises(ValueError, match="ambiguous"):
        _pf([6], [12], nb=0.05).compute_quasineutrality()


def test_no_species_block_raises():
    pf = _pf([6, 1], [12, 2])
    del pf._raw["N Z A"]
    with pytest.raises(ValueError, match="N Z A"):
        pf.compute_zeff()


@pytest.mark.parametrize("path", _SHIPPED)
def test_the_shipped_three_entry_pfiles_are_bit_unchanged(path):
    """The position read the fix replaced, evaluated inline: the shipped
    [C, D, D] p-files give the same nz1 and Z_eff to the bit."""
    pf = read_pfile(path)
    Z = np.asarray(pf.ion_species["Z"], dtype=float)
    assert len(Z) == 3
    ne, ni = pf.ne, pf.ni
    nb = pf._get_data("nb")
    nb = np.zeros_like(ne) if nb is None else nb
    nz1_old = (ne - ni - float(Z[-1]) * nb) / Z[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        zeff_old = (ni * Z[-2] ** 2 + nz1_old * Z[0] ** 2
                    + nb * Z[-1] ** 2) / ne
    np.nan_to_num(zeff_old, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pf.compute_quasineutrality()
    np.testing.assert_array_equal(pf._get_data("nz1"), nz1_old)
    np.testing.assert_array_equal(pf.compute_zeff()[1], zeff_old)


def _zeff_140(pf):
    """1.4.0's position read (Z_imp = Z[0], Z_main = Z[-2], Z_beam = Z[-1],
    and Z_beam 1 in quasineutrality for a two-entry block), inline."""
    Z = np.asarray(pf.ion_species["Z"], dtype=float)
    ne, ni = pf.ne, pf.ni
    nb = pf._get_data("nb")
    nb = np.zeros_like(ne) if nb is None else nb
    zb_q = float(Z[-1]) if len(Z) > 2 else 1.0
    nz1 = (ne - ni - zb_q * nb) / Z[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        zeff = (ni * Z[-2] ** 2 + nz1 * Z[0] ** 2 + nb * Z[-1] ** 2) / ne
    np.nan_to_num(zeff, copy=False, nan=1.0, posinf=1.0, neginf=1.0)
    return nz1, zeff


@pytest.mark.parametrize("Z,A,nb", [
    ([1, 1], [2, 2], None),        # [D, D], no beam density
    ([1, 1], [2, 2], 0.05),        # [D, D], main + beam
    ([1, 1], [2, 3], None),        # [D, T]
    ([1, 1], [2, 3], 0.05),
])
def test_a_clean_plasma_block_loads_as_in_1_4_0(Z, A, nb):
    """Review of PR #100, F2: an impurity-free block (no non-hydrogenic
    species) carries the quasineutrality residual at charge 1, so Z_eff = 1
    and nz1 are 1.4.0's, bit for bit."""
    pf = _pf(Z, A, nb=nb)
    nz1_old, zeff_old = _zeff_140(pf)
    r = pf.ion_roles()
    assert r["impurity"] is None and r["impurity_hydrogenic_residual"]
    assert r["Z_imp"] == 1.0 and r["Z_main"] == 1.0
    pf.compute_quasineutrality()
    np.testing.assert_array_equal(pf._get_data("nz1"), nz1_old)
    z = pf.compute_zeff()[1]
    np.testing.assert_array_equal(z, zeff_old)
    np.testing.assert_allclose(z, 1.0, rtol=1e-12)


@pytest.mark.parametrize("nb", [None, 0.05])
def test_a_helium_main_ion_is_refused_as_unsupported(nb):
    """Review of PR #100, F3: [C, He, D] is a He plasma with a D beam (with
    nb) or He ash beside a D main ion (without): a non-hydrogenic main ion is
    not supported, and D is never silently promoted past the He."""
    pf = _pf([6, 2, 1], [12, 4, 2], nb=nb)
    for call in (pf.ion_roles, pf.compute_zeff, pf.compute_quasineutrality):
        with pytest.raises(ValueError, match="non-hydrogenic main ion is "
                                             "not supported"):
            call()


def test_a_heavier_second_impurity_is_still_accepted():
    """[C, Ne, D, D]: the first impurity is nz1's, a heavier second one is
    not a main-ion candidate (nothing lighter than the impurity)."""
    r = _pf([6, 10, 1, 1], [12, 20, 2, 2], nb=0.05).ion_roles()
    assert (r["impurity"], r["main"], r["beam"]) == (0, 2, 3)


def test_the_diamagnetic_rotation_reads_the_impurity_by_role():
    """Review of PR #100, F11: the impurity diamagnetic term uses the role,
    so a main-ion-first block agrees with the OMFIT order."""
    x = np.linspace(0.0, 1.0, 64)
    psi = -1.0 + x
    out = []
    for Z, A in (([6, 1], [12, 2]), ([1, 6], [2, 12])):
        pf = _pf(Z, A)
        pf.set_profile("te", x, 2.0 * (1.0 - 0.9 * x ** 2))
        pf.set_profile("ti", x, 2.0 * (1.0 - 0.9 * x ** 2))
        pf.compute_quasineutrality()
        pf.compute_diamagnetic_rotations(psi)
        out.append(np.asarray(pf.omgpp, dtype=float))
    np.testing.assert_array_equal(out[0], out[1])
