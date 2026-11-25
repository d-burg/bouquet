"""Monte Carlo and invariant tests for bouquet.stiff_sampler."""

import numpy as np

from bouquet.kinetic_sampler import KineticBase, sample_kinetics
from bouquet.sampling import make_rng
from bouquet.stiff_sampler import sample_stiff_inward_channel


def test_stiff_inward_hmode_3000_draws_no_blend_explosion_and_zero_bias():
    psi = np.linspace(0.0, 1.0, 101)
    r_norm = np.sqrt(psi)
    # H-mode Te with steep barrier [0.92, 0.98] AND flat SOL foot [0.98, 1.0]
    te_core = 4000.0 + 4000.0 * (1.0 - np.minimum(psi / 0.92, 1.0) ** 1.5)
    ped_drop = np.where(
        psi <= 0.92,
        0.0,
        np.where(psi <= 0.98, -3900.0 * ((psi - 0.92) / 0.06), -3900.0),
    )
    base = np.where(psi <= 0.92, te_core, 4000.0 + ped_drop)
    sig = 0.20 * base
    dkappa_allow = np.full_like(psi, 1.2)
    ell = np.where(psi < 0.92, 0.80, 0.02)

    rng = np.random.default_rng(12345)
    draws = np.array([
        sample_stiff_inward_channel(
            psi, base, sig, dkappa_allow, ell,
            r_norm=r_norm, a_over_R0=0.33, psi_ped_top=0.92,
            has_pedestal=True, ell_kappa=0.30, enforce_ped_monotonic=True,
            rng=rng,
        )
        for _ in range(3000)
    ])

    # 1. Zero upward core/pedestal bias despite steep-barrier monotonicity & flat SOL foot
    mean_rel_err = np.abs(np.mean(draws, axis=0) / base - 1.0)
    assert float(np.max(mean_rel_err)) < 0.015
    assert float(mean_rel_err[0]) < 0.008

    # 2. Outer-core gradient fluctuations stay O(dkappa_allow), NOT 28-53x explosion
    R_over_a = 1.0 / 0.33
    rl_base = -R_over_a * np.gradient(np.log(base), r_norm)
    rl_draws = -R_over_a * np.gradient(np.log(draws), r_norm, axis=1)
    std_drl = np.std(rl_draws - rl_base[None, :], axis=0)
    outer_core = (psi >= 0.85) & (psi <= 0.92)
    assert float(np.max(std_drl[outer_core])) < 2.5 * 1.2

    # 3. No -13% outer-core marginal variance dip from two-draw blending
    log_std = np.std(np.log(draws / base[None, :]), axis=0)
    assert float(np.min(log_std[(psi >= 0.80) & (psi <= 0.92)])) > 0.185

    # 4. Steep barrier segments strictly non-increasing; nodewise positivity respected
    i_ped = int(np.searchsorted(psi, 0.92, side="right"))
    steep_seg = np.diff(base[i_ped:]) < -1.0e-3 * float(np.max(base))
    assert np.any(steep_seg)
    assert np.all(np.diff(draws[:, i_ped:], axis=1)[:, steep_seg] <= 1.0e-9)
    min_floor = 0.05 * np.maximum(base, 1.0e-4 * float(np.max(base)))
    assert np.all(draws >= min_floor[None, :] - 1.0e-12)


def test_stiff_inward_lmode_has_pedestal_false_and_zero_separatrix():
    psi = np.linspace(0.0, 1.0, 81)
    r_norm = np.sqrt(psi)
    # Gentle/flat-edge L-mode profile
    base = 3000.0 * (1.0 - 0.7 * (psi ** 2))
    sig = 0.15 * base
    dkappa_allow = np.full_like(psi, 1.5)
    ell = np.full_like(psi, 0.45)

    rng = np.random.default_rng(77)
    draws = np.array([
        sample_stiff_inward_channel(
            psi, base, sig, dkappa_allow, ell,
            r_norm=r_norm, a_over_R0=0.33, psi_ped_top=0.95,
            has_pedestal=False, ell_kappa=0.30, enforce_ped_monotonic=True,
            rng=rng,
        )
        for _ in range(1000)
    ])
    assert float(np.max(np.abs(np.mean(draws, axis=0) / base - 1.0))) < 0.015
    R_over_a = 1.0 / 0.33
    rl_base = -R_over_a * np.gradient(np.log(base), r_norm)
    rl_draws = -R_over_a * np.gradient(np.log(draws), r_norm, axis=1)
    assert float(np.max(np.std(rl_draws - rl_base[None, :], axis=0))) < 2.0 * 1.5

    # Zero separatrix base[-1] == 0.0 still satisfies strict positivity > 0
    base_zsep = base.copy()
    base_zsep[-1] = 0.0
    d_zsep = sample_stiff_inward_channel(
        psi, base_zsep, sig, dkappa_allow, ell,
        r_norm=r_norm, has_pedestal=True, rng=np.random.default_rng(9),
    )
    assert np.all(d_zsep > 0.0) and np.all(np.isfinite(d_zsep))


def test_stiff_inward_sigma0_identity_and_deterministic_rng_stream():
    psi = np.linspace(0.0, 1.0, 65)
    ne = 1.0e20 * (1.0 - 0.5 * psi)
    te = 6.0e3 * (1.0 - 0.85 * psi)
    ni = 0.9 * ne
    ti = 6.0e3 * (1.0 - 0.85 * psi)
    z0 = np.zeros_like(psi)

    # Exact sigma == 0 identity
    d0 = sample_stiff_inward_channel(
        psi, te, z0, np.ones_like(psi), np.full_like(psi, 0.2),
        rng=np.random.default_rng(5),
    )
    np.testing.assert_array_equal(d0, te)

    # Zeroing sigma_te preserves downstream ti draw seed-for-seed
    kb_te_on = KineticBase(
        psi_kin=psi, ne=ne, te=te, ni=ni, ti=ti,
        sigma_ne=z0, sigma_te=0.12 * te, sigma_ni=z0, sigma_ti=0.12 * ti,
        n_ls=0.4, t_ls=0.04, ni_from_zeff=False,
        sampling_mode="stiff_inward", gradient_policy="physics_bounds",
    )
    kb_te_off = KineticBase(
        psi_kin=psi, ne=ne, te=te, ni=ni, ti=ti,
        sigma_ne=z0, sigma_te=z0, sigma_ni=z0, sigma_ti=0.12 * ti,
        n_ls=0.4, t_ls=0.04, ni_from_zeff=False,
        sampling_mode="stiff_inward", gradient_policy="physics_bounds",
    )
    target = float(np.trapezoid(ne * te + ni * ti, psi))
    therm = lambda d: float(np.trapezoid(d.ne * d.te + d.ni * d.ti, psi))
    kd_on = sample_kinetics(kb_te_on, make_rng(2026), therm, target, p_thresh=1.0)
    kd_off = sample_kinetics(kb_te_off, make_rng(2026), therm, target, p_thresh=1.0)
    np.testing.assert_array_equal(kd_off.te, te)
    np.testing.assert_array_equal(kd_on.ti, kd_off.ti)


def test_stiff_inward_preserves_hollow_te_and_pellet_ne_in_sample_kinetics():
    psi = np.linspace(0.0, 1.0, 81)
    # Pellet off-axis density bump around psi ~ 0.72 + radiatively hollow core Te
    ne = 1.0e20 * (0.85 - 0.55 * (psi ** 2) + 0.30 * np.exp(-(((psi - 0.72) / 0.10) ** 2)))
    te = (7.0e3 * (1.0 - 0.85 * psi)) * (1.0 - 0.40 * np.exp(-((psi / 0.22) ** 2)))
    ni = 0.9 * ne
    ti = 6.5e3 * (1.0 - 0.85 * psi)

    kb = KineticBase(
        psi_kin=psi, ne=ne, te=te, ni=ni, ti=ti,
        sigma_ne=0.08 * ne, sigma_te=0.10 * te,
        sigma_ni=0.08 * ni, sigma_ti=0.10 * ti,
        n_ls=0.35, t_ls=0.04, ni_from_zeff=False,
        sampling_mode="stiff_inward", gradient_policy="physics_bounds",
        psi_ped_top=0.92, has_pedestal=True,
    )
    target = float(np.trapezoid(ne * te + ni * ti, psi))
    therm = lambda d: float(np.trapezoid(d.ne * d.te + d.ni * d.ti, psi))
    rng = make_rng(314)
    draws_te = []
    draws_ne = []
    for _ in range(40):
        kd = sample_kinetics(kb, rng, therm, target, p_thresh=0.15)
        draws_te.append(kd.te)
        draws_ne.append(kd.ne)
    mean_te = np.mean(draws_te, axis=0)
    mean_ne = np.mean(draws_ne, axis=0)
    # Hollow core Te (axis lower than psi ~ 0.25) and pellet peak preserved in ensemble mean
    assert mean_te[0] < mean_te[int(np.searchsorted(psi, 0.25))]
    assert mean_ne[int(np.searchsorted(psi, 0.72))] > mean_ne[int(np.searchsorted(psi, 0.45))]


def test_stiff_inward_grid_refinement_n501_and_phi_n_wide_pedestal():
    # 1. N=501 fine-grid regression covering all nodes up to and including psi[i_ped]:
    #    verifies standardized calibration normals + smooth beta(x) without O(N) step jump
    psi = np.linspace(0.0, 1.0, 501)
    r_norm = np.sqrt(psi)
    te_core = 4000.0 + 4000.0 * (1.0 - np.minimum(psi / 0.92, 1.0) ** 1.5)
    ped_drop = np.where(
        psi <= 0.92,
        0.0,
        np.where(psi <= 0.98, -3900.0 * ((psi - 0.92) / 0.06), -3900.0),
    )
    base = np.where(psi <= 0.92, te_core, 4000.0 + ped_drop)
    sig = 0.20 * base
    dkappa_allow = np.full_like(psi, 1.2)
    ell = np.where(psi < 0.92, 0.80, 0.02)

    i_ped = int(np.searchsorted(psi, 0.919, side="right"))  # i_ped == 460, psi[i_ped] == 0.920
    assert psi[i_ped] == 0.92
    rng = np.random.default_rng(12345)
    draws = np.array([
        sample_stiff_inward_channel(
            psi, base, sig, dkappa_allow, ell,
            r_norm=r_norm, a_over_R0=0.33, psi_ped_top=0.919,
            has_pedestal=True, ell_kappa=0.30, enforce_ped_monotonic=True,
            rng=rng,
        )
        for _ in range(1500)
    ])
    R_over_a = 1.0 / 0.33
    rl_base = -R_over_a * np.gradient(np.log(base), r_norm)
    rl_draws = -R_over_a * np.gradient(np.log(draws), r_norm, axis=1)
    std_drl = np.std(rl_draws - rl_base[None, :], axis=0)
    # Strict core up to i_ped-1 stays at ~1.2, and including i_ped itself stays < 2.5 * dkappa_allow
    assert float(np.max(std_drl[:i_ped])) < 1.35 * 1.2
    assert float(np.max(std_drl[psi <= psi[i_ped]])) < 2.5 * 1.2
    assert float(np.max(np.abs(np.mean(draws, axis=0) / base - 1.0))) < 0.015

    # 2. Wide pedestal in coord='phi_n' (Phi_N_ped = 0.56 < 0.70) is NOT clipped outward to 0.70
    phi = np.linspace(0.0, 1.0, 101)
    base_wide = np.where(
        phi <= 0.56,
        5000.0 + 3000.0 * (1.0 - (phi / 0.56) ** 1.5),
        np.maximum(5000.0 - 4800.0 * ((phi - 0.56) / 0.30), 200.0),
    )
    rng_w = np.random.default_rng(99)
    draws_w = np.array([
        sample_stiff_inward_channel(
            phi, base_wide, 0.15 * base_wide, np.full_like(phi, 1.0),
            np.where(phi < 0.56, 0.70, 0.05),
            r_norm=np.sqrt(phi), a_over_R0=0.33, psi_ped_top=0.56,
            has_pedestal=True, enforce_ped_monotonic=True, rng=rng_w,
        )
        for _ in range(400)
    ])
    # Barrier monotonicity holds across [0.57, 0.85] (would fail if i_ped were clipped to 0.70!)
    steep_zone = (phi >= 0.57) & (phi <= 0.85)
    assert np.all(np.diff(draws_w[:, steep_zone], axis=1) <= 1.0e-9)
