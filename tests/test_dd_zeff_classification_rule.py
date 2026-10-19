"""The EXPERIMENTAL Z_eff classification (``ImasSource.zeff_fast_ions``,
``io.imas._dd_zeff``) on ``_dd_total_zeff``'s post-review rule (1.4.1).

* the window: every node with n_e > 0 and finite inputs, the pedestal and
  the edge included (it was psi_N <= 0.8);
* the charge weights: each ion's charge state ``ion[].z_ion`` when the dd
  carries it, else ``z_n`` -- in the check, in the no-dataset value and in
  the run's Z_eff; the species derivations keep ``z_n``.

Without ``z_ion`` (every shipped dd) and with a stored zeff that fits on
every node, nothing moves by a bit.  Fast: the plain dd reader on the
synthetic beam dd of ``test_ni_fast_subtraction``; no solve.
"""
import warnings

import numpy as np
import pytest

import test_ni_fast_subtraction as NF
import test_zeff_dataset_check_post_review as PR
from bouquet.config import ImasSource
from bouquet.io.imas import ZEFF_CONVENTION_RTOL, read_imas_baseline


def _read(ddp):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = read_imas_baseline(
            ImasSource(ids_path=ddp, time=1.0, impurity_Z=NF.Z_IMP,
                       zeff_fast_ions=True),
            kinetic_source="fuse")
    return bl, [str(x.message) for x in w]


def test_a_pedestal_misfit_is_caught_and_located(tmp_path):
    """A stored total right on the core but 10 % off beyond psi_N 0.85: the
    core-only window classified it thermal+fast without a word."""
    ddp, *_ = NF._build(tmp_path, dd_mutate=PR._pedestal_off(1.10))
    bl, msgs = _read(ddp)
    rec = bl.li_metrics["zeff_dd_provenance"]
    assert rec["window"] == "every node with n_e > 0 and finite inputs"
    assert rec["d_all"] == pytest.approx(0.1 / 1.1, rel=1e-9)
    assert rec["d_all_psi_N"] > 0.85
    assert rec["matches_neither"] is True
    neither = [m for m in msgs if "matches neither" in m]
    assert len(neither) == 1
    assert f"psi_N = {rec['d_all_psi_N']:.3f}" in neither[0]


def test_a_fit_on_every_node_is_unchanged(tmp_path):
    ddp, *_ = NF._build(tmp_path, dd_mutate=PR._pedestal_off(1.0))
    bl, msgs = _read(ddp)
    rec = bl.li_metrics["zeff_dd_provenance"]
    assert rec["convention"] == "thermal+fast"
    assert rec["d_all"] <= ZEFF_CONVENTION_RTOL
    assert rec["matches_neither"] is False
    assert not [m for m in msgs if "matches neither" in m]


def test_a_partially_ionised_impurity_is_weighted_by_its_charge_state(
        tmp_path):
    """A correct IMAS total with argon at z_ion = 16: the z_n-weighted
    numerator misses it by more than the 1 % convention tolerance; the
    charge-state one matches it, and the run's Z_eff is that total."""
    ddp, *_ = NF._build(tmp_path, dd_mutate=PR._with_argon("imas"))
    cp = PR._cp(ddp)
    stored = np.asarray(cp["zeff"], dtype=float)
    z_n_total = PR._total(cp, PR._z_n)
    assert np.max(np.abs(stored - z_n_total) / stored) > ZEFF_CONVENTION_RTOL
    bl, msgs = _read(ddp)
    rec = bl.li_metrics["zeff_dd_provenance"]
    assert rec["convention"] == "thermal+fast"
    assert rec["d_all"] < 1e-12 and rec["matches_neither"] is False
    assert not [m for m in msgs if "matches neither" in m]
    np.testing.assert_allclose(bl.Zeff, stored, rtol=1e-12)
    np.testing.assert_array_equal(bl.aux["zeff"], stored)


def test_without_a_dataset_the_value_is_charge_state_weighted(tmp_path):
    ddp, *_ = NF._build(tmp_path, dd_mutate=PR._with_argon())
    bl, _ = _read(ddp)
    q_total = PR._total(PR._cp(ddp), PR._z_ion)
    np.testing.assert_allclose(bl.aux["zeff"], q_total, rtol=1e-12)
    np.testing.assert_allclose(bl.Zeff, q_total, rtol=1e-12)
    assert bl.zeff_includes_fast is True


def test_the_species_derivations_keep_z_n(tmp_path):
    a, *_ = NF._build(tmp_path, dd_mutate=PR._with_argon())
    (tmp_path / "b").mkdir()

    def _no_z_ion(dd):
        PR._with_argon()(dd)
        del dd["core_profiles"]["profiles_1d"][0]["ion"][-1]["z_ion"]
    b, *_ = NF._build(tmp_path / "b", dd_mutate=_no_z_ion)
    bla, _ = _read(a)
    blb, _ = _read(b)
    np.testing.assert_array_equal(bla.ni, blb.ni)
    assert bla.Z_imp == blb.Z_imp


@pytest.mark.parametrize("kind", [None, "thermal", "all"])
def test_the_carbon_fixtures_are_bit_identical(tmp_path, kind):
    """No z_ion, a stored zeff (if any) that fits on every node: the run's
    Z_eff, its draw centre and the convention are the pre-1.4.1 ones."""
    ddp, *_ = NF._build(tmp_path, dd_mutate=NF._store_zeff(kind))
    bl, _ = _read(ddp)
    cp = PR._cp(ddp)
    ne = np.asarray(cp["electrons"]["density_thermal"], dtype=float)
    num = np.zeros_like(ne)
    z2 = np.zeros_like(ne)
    for ion in cp["ion"]:                       # the z_n accumulation
        Zs = PR._z_n(ion)
        num += np.asarray(ion["density_thermal"], dtype=float) * Zs * Zs
        if "density_fast" in ion:
            z2 += Zs * Zs * np.asarray(ion["density_fast"], dtype=float)
    th = num / ne
    al = th + z2 / np.clip(ne, 1e-30, None)
    inc = kind != "thermal"
    assert bl.zeff_includes_fast is inc
    np.testing.assert_array_equal(bl.Zeff, al if inc else th)
    np.testing.assert_array_equal(
        bl.aux["zeff"], al if kind is None
        else np.asarray(cp["zeff"], dtype=float))
