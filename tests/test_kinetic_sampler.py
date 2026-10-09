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
              zeff_includes_fast=zeff_includes_fast,
              clips=True)            # EXPERIMENTAL PR #56 clips, opted in
    for seed in range(5):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b),
                            p_thresh=0.5)
        assert d.zeff_primary
        assert np.all(d.zeff >= ZEFF_DRAW_MIN)
        assert np.all(d.ni >= 0.0) and np.all(d.ni <= d.ne - ZF + 1e-6)


def test_independent_ni_held_below_the_thermal_electrons():
    b = _base(f=0.3, ni=0.88 * NE, Z_imp=Z_IMP, z_fast=ZF, clips=True)
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


# ---------------------------------------------------------------------------
#  PR #56: the sigma = 0 fix the increment form brings, declared and pinned
# ---------------------------------------------------------------------------
def test_sigma0_returns_the_baseline_ni_on_a_non_quasineutral_baseline():
    """The legacy sigma = 0 violation kinetic_sampler/2 fixes.  A p-file
    baseline (several species, a beam) is not single-impurity quasineutral
    at the median Z_imp, so ``ni_of(ne, Z_eff, Z_imp) != bl.ni``.  The /1
    legacy draw took ``ni`` ABSOLUTELY from the drawn (ne, Z_eff) and so
    returned that other ``ni`` even with every sigma zero; the increment
    form returns the baseline exactly."""
    from bouquet.kinetic_sampler import KINETIC_SAMPLER_VERSION
    zb = 1.8
    ni0 = 0.80 * NE                       # != ni_of(NE, 1.8, 6) = 0.84 NE
    absolute = main_ion_density_from_zeff(NE, np.full(PSI.size, zb), Z_IMP)
    assert np.max(np.abs(absolute / ni0 - 1.0)) > 0.04   # /1 was 5 % off
    b = _base(f=0.0, zeff=zb, zs=0.0, ni=ni0, Z_imp=Z_IMP)
    for seed in range(3):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b))
        assert d.zeff_primary
        np.testing.assert_array_equal(d.ni, ni0)
        for k in ("ne", "te", "ti"):
            np.testing.assert_array_equal(getattr(d, k), getattr(b, k))
        np.testing.assert_array_equal(d.zeff, np.full(PSI.size, zb))
        assert not d.clipped
    assert KINETIC_SAMPLER_VERSION.startswith("kinetic_sampler/3")


def test_every_clip_that_fires_is_counted_on_the_draw():
    """Z_eff at the bottom of a window that zeff_bounds would let go below
    1 (thermal-numerator convention with a beam): the floor at 1 binds and
    is counted; the values are clipped exactly as before."""
    from bouquet.kinetic_sampler import CLIP_COUNTERS
    b = _base(zeff=1.0, zs=0.3, Z_imp=Z_IMP, z_fast=ZF, z2_fast=ZF,
              zeff_includes_fast=False, clips=True)
    seen = 0
    for seed in range(5):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b),
                            p_thresh=0.5)
        assert set(d.clips) == set(CLIP_COUNTERS)
        n_at_1 = int(np.count_nonzero(d.zeff == ZEFF_DRAW_MIN))
        assert d.clips["zeff_floor_1"] + d.clips["zeff_window_lo"] == n_at_1
        seen += d.clips["zeff_floor_1"]
        rec = d.record()
        assert rec["clipped"] == (sum(rec["clips"].values()) > 0)
        assert rec["version"].startswith("kinetic_sampler/3")
        assert rec["clips_enabled"] is True
    assert seen > 0


def test_clip_counting_does_not_change_the_draw_stream():
    """Counting is read-only: same seed, same draw, with or without a clip
    firing elsewhere in the stream."""
    b = _base(zeff=1.02, zs=0.3, Z_imp=Z_IMP, z_fast=ZF, z2_fast=ZF)
    d1 = sample_kinetics(b, make_rng(4), _thermal, _thermal(b), p_thresh=0.5)
    d2 = sample_kinetics(b, make_rng(4), _thermal, _thermal(b), p_thresh=0.5)
    for k in ("ne", "te", "ni", "ti", "zeff"):
        np.testing.assert_array_equal(getattr(d1, k), getattr(d2, k))
    assert d1.clips == d2.clips


def test_the_ni_ceiling_is_counted_when_it_binds():
    b = _base(f=0.3, ni=0.88 * NE, Z_imp=Z_IMP, z_fast=ZF, clips=True)
    n = 0
    for seed in range(8):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b),
                            p_thresh=0.5)
        cap = np.maximum(d.ne - ZF, 0.0)
        assert d.clips["ni_ceiling"] <= int(np.count_nonzero(d.ni == cap))
        n += d.clips["ni_ceiling"]
    assert n > 0


# ---------------------------------------------------------------------------
#  1.4.0 owner decision: the PR #56 clips are EXPERIMENTAL and opt-in
# ---------------------------------------------------------------------------
def test_default_keeps_only_the_zeff_bounds_window():
    """Default (clips off): a Z_eff draw is held in physics.zeff_bounds
    alone -- the pre-#56 window, which with a beam and the thermal
    numerator reaches below 1 -- and n_i is not clipped."""
    from bouquet.physics import zeff_bounds
    b = _base(zeff=1.0, zs=0.3, Z_imp=Z_IMP, z_fast=ZF, z2_fast=ZF,
              zeff_includes_fast=False)
    assert b.clips is False
    below = 0
    for seed in range(5):
        d = sample_kinetics(b, make_rng(seed), _thermal, _thermal(b),
                            p_thresh=0.5)
        lo, hi = zeff_bounds(d.ne, Z_IMP, ZF, ZF, False)
        assert np.all(d.zeff >= lo) and np.all(d.zeff <= hi)
        below += int(np.count_nonzero(d.zeff < ZEFF_DRAW_MIN))
        assert d.clips["zeff_floor_1"] == 0
        assert d.clips["ni_floor_0"] == 0 and d.clips["ni_ceiling"] == 0
        assert d.record()["clips_enabled"] is False
    assert below > 0          # the floor at 1 would have bound here


def test_default_passive_zeff_aux_draw_is_not_clipped():
    """Default: a passive Z_eff aux draw (ni_from_zeff=False) is the raw GP
    sample, as before PR #56; with clips=True it is held in the window."""
    b0 = _base(zeff=1.0, zs=0.3, Z_imp=Z_IMP, ni_from_zeff=False)
    b1 = _base(zeff=1.0, zs=0.3, Z_imp=Z_IMP, ni_from_zeff=False, clips=True)
    lo_seen = False
    for seed in range(5):
        d0 = sample_kinetics(b0, make_rng(seed), _thermal, _thermal(b0),
                             p_thresh=0.5)
        d1 = sample_kinetics(b1, make_rng(seed), _thermal, _thermal(b1),
                             p_thresh=0.5)
        assert np.all(d1.aux["zeff"] >= ZEFF_DRAW_MIN)
        lo_seen |= bool(np.any(d0.aux["zeff"] < ZEFF_DRAW_MIN))
        np.testing.assert_array_equal(
            np.maximum(d0.aux["zeff"], ZEFF_DRAW_MIN), d1.aux["zeff"])
    assert lo_seen


def test_clips_change_nothing_where_none_binds():
    """Same seed, a baseline far from every bound: the opt-in clips leave
    the draw bit-identical to the default."""
    b0 = _base(zeff=2.0, zs=0.05, Z_imp=Z_IMP)
    b1 = _base(zeff=2.0, zs=0.05, Z_imp=Z_IMP, clips=True)
    d0 = sample_kinetics(b0, make_rng(11), _thermal, _thermal(b0), p_thresh=0.5)
    d1 = sample_kinetics(b1, make_rng(11), _thermal, _thermal(b1), p_thresh=0.5)
    assert not d1.clipped
    for k in ("ne", "te", "ni", "ti", "zeff"):
        np.testing.assert_array_equal(getattr(d0, k), getattr(d1, k))
