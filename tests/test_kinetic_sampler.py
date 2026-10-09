"""bouquet.kinetic_sampler: the one kinetic draw of the legacy, swb and engine
paths.  Solver-free (a trapezoid stands in for the flux integral)."""
import numpy as np
import pytest

from bouquet.kinetic_sampler import (ZEFF_DRAW_MIN, KineticBase,
                                     sample_kinetics)
from bouquet.physics import ELEMENTARY_CHARGE as EC, main_ion_density_from_zeff
from bouquet.sampling import make_rng

PSI = np.linspace(0.0, 1.0, 41)
NE = 5e19 * (1.0 - 0.8 * PSI ** 2)
TE = 3e3 * (1.0 - 0.9 * PSI ** 2)
TI = 2.5e3 * (1.0 - 0.9 * PSI ** 2)
ZF = 0.10 * NE                     # a 10 % beam (charge moment)
Z_IMP = 6.0


def _thermal(d):
    return np.trapezoid(EC * (d.ne * d.te + d.ni * d.ti), PSI)


def _base(f=0.05, zeff=None, zs=0.1, ni=None, **kw):
    ni = 0.8 * NE if ni is None else ni
    aux = {}
    if zeff is not None:
        aux = dict(aux_sigmas={"zeff": zs * np.ones_like(PSI)},
                   aux_baselines={"zeff": np.full(PSI.size, zeff)},
                   aux_length_scales={"zeff": 0.4})
    return KineticBase(psi_kin=PSI, ne=NE, te=TE, ni=ni, ti=TI,
                       sigma_ne=f * NE, sigma_te=f * TE, sigma_ni=f * ni,
                       sigma_ti=f * TI, n_ls=0.3, t_ls=0.3, **aux, **kw)


def test_same_seed_same_draw():
    b = _base()
    d1 = sample_kinetics(b, make_rng(3), _thermal, _thermal(b))
    d2 = sample_kinetics(b, make_rng(3), _thermal, _thermal(b))
    for k in ("ne", "te", "ni", "ti"):
        np.testing.assert_array_equal(getattr(d1, k), getattr(d2, k))
    assert d1.iterations >= 1 and d1.p_err_pct <= 5.0


def test_pressure_match_is_thermal_only():
    b = _base()
    seen = []

    def _rec(d):
        seen.append(d)
        return _thermal(d)
    d = sample_kinetics(b, make_rng(5), _rec, _thermal(b), p_thresh=0.02)
    assert seen[-1] is d
    assert abs(_thermal(d) / _thermal(b) - 1.0) <= 0.02


@pytest.mark.parametrize("zeff_includes_fast", [False, True])
def test_zeff_draw_is_never_below_one(zeff_includes_fast):
    # baseline Z_eff at the bottom of the window: draws push below 1 unless
    # floored
    b = _base(zeff=1.02, zs=0.3, Z_imp=Z_IMP, z_fast=ZF, z2_fast=ZF,
              zeff_includes_fast=zeff_includes_fast)
    for seed in range(5):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b),
                            p_thresh=0.5)
        assert d.zeff_primary
        assert np.all(d.zeff >= ZEFF_DRAW_MIN)
        assert np.all(d.ni >= 0.0) and np.all(d.ni <= d.ne - ZF + 1e-6)


def test_independent_ni_held_below_the_thermal_electrons():
    b = _base(f=0.3, ni=0.88 * NE, Z_imp=Z_IMP, z_fast=ZF)
    for seed in range(5):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b),
                            p_thresh=0.5)
        assert not d.zeff_primary
        assert np.all(d.ni <= np.maximum(d.ne - ZF, 0.0))


def test_quasineutral_baseline_gives_the_direct_ni():
    """ni on the Z_eff route is an increment on the baseline ni; on a
    quasineutral baseline that is the direct ni of the drawn (ne, Z_eff)."""
    zb = 1.8
    ni0 = main_ion_density_from_zeff(NE, np.full(PSI.size, zb), Z_IMP)
    b = _base(zeff=zb, ni=ni0, Z_imp=Z_IMP)
    d = sample_kinetics(b, make_rng(9), _thermal, _thermal(b), p_thresh=0.5)
    direct = main_ion_density_from_zeff(d.ne, d.zeff, Z_IMP)
    np.testing.assert_allclose(d.ni, direct, rtol=1e-9, atol=1e6)


def test_ni_from_zeff_false_draws_ni_and_passive_zeff():
    b = _base(zeff=1.8, Z_imp=Z_IMP, ni_from_zeff=False)
    d = sample_kinetics(b, make_rng(2), _thermal, _thermal(b), p_thresh=0.5)
    assert not d.zeff_primary
    assert "zeff" in d.aux and np.all(d.aux["zeff"] >= ZEFF_DRAW_MIN)


def test_zero_sigma_returns_the_base():
    b = _base(f=0.0)
    d = sample_kinetics(b, make_rng(1), _thermal, _thermal(b))
    for k in ("ne", "te", "ni", "ti"):
        np.testing.assert_allclose(getattr(d, k), getattr(b, k), rtol=1e-9)
