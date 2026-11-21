"""Tests for regional gradient bounds, flat/hollow/pellet profiles, and Tier 1 stiffness."""

import numpy as np
import pytest

from bouquet.baseline import Baseline
from bouquet.geometry.miller import miller_from_baseline
from bouquet.sampling import (
    _draw_monotonic_perturbation,
    draw_bounded_perturbation,
    make_rng,
)
from bouquet.stiffness import compute_physics_length_scales, detect_pedestal_window


def _hmode_profile(psi_N, core_val, ped_val, sep_val, psi_ped=0.92, alpha=1.3):
    psi = np.asarray(psi_N, dtype=float)
    core_part = np.maximum(1.0 - np.clip(psi / psi_ped, 0.0, 1.0) ** alpha, 0.0)
    ped_part = 0.5 * (1.0 - np.tanh((psi - 0.5 * (psi_ped + 1.0)) / (0.25 * (1.0 - psi_ped))))
    ped_part = (ped_part - ped_part[-1]) / max(ped_part[0] - ped_part[-1], 1e-12)
    return sep_val + (ped_val - sep_val) * ped_part + (core_val - ped_val) * core_part


def test_flat_sawtooth_core_bias_eliminated_by_physics_bounds():
    """_draw_monotonic_perturbation biases flat sawtooth cores upward on axis;
    draw_bounded_perturbation('physics_bounds') is unbiased."""
    psi = np.linspace(0.0, 1.0, 101)
    # Sawtooth-clamped flat core inside psi_N <= 0.28, monotonic outside
    prof = np.where(
        psi <= 0.28,
        1.0,
        0.05 + 0.95 * ((1.0 - (psi - 0.28) / 0.72) ** 1.5),
    )
    sig = 0.10 * prof

    rng_leg = make_rng(2026)
    leg_draws = np.array([
        _draw_monotonic_perturbation(psi, prof, sig, 0.25, rng=rng_leg)
        for _ in range(30)
    ])
    # Legacy global monotonicity forces sample[0] >= sample[28] on EVERY draw (~+0.11 bias)!
    leg_core_drop = float(np.mean(leg_draws[:, 0] - leg_draws[:, 28]))
    assert leg_core_drop > 0.05, "Expected legacy _draw_monotonic_perturbation positive axis bias"

    rng_pb = make_rng(2026)
    # Symmetric bounds around baseline slope rl_base (= 0 in flat sawtooth core psi <= 0.28)
    # guarantee exact antipodal symmetry delta_f -> -delta_f (zero level AND slope skew!).
    x_r = np.sqrt((psi + 0.88 * (psi ** 2.5)) / 1.88)
    rl_base = -(1.0 / 0.33) * np.gradient(prof, x_r) / np.maximum(prof, 1.0e-3)
    k_sym = np.where(psi <= 0.28, 3.5, 14.0)
    k_bounds = (rl_base - k_sym, rl_base + k_sym)
    pb_draws = np.array([
        draw_bounded_perturbation(
            psi, prof, sig, 0.25,
            policy="physics_bounds", channel="te", psi_ped_top=0.92,
            kappa_bounds=k_bounds, rng=rng_pb,
        )
        for _ in range(1200)
    ])
    assert abs(float(np.mean(pb_draws[:, 0]) - prof[0])) < 0.01
    pb_core_drop = abs(float(np.mean(pb_draws[:, 0] - pb_draws[:, 28])))
    assert pb_core_drop < 0.012, f"physics_bounds should be unbiased on flat core, got {pb_core_drop}"
    assert pb_core_drop < 0.20 * leg_core_drop
    assert np.all(pb_draws > 0.0)


def test_pellet_inverted_density_and_radiative_hollow_te_supported():
    """Pellet fueling bumps (R/L_ne < 0) and core radiative cooling (R/L_Te < 0)
    must sample smoothly without disabling H-mode pedestal barrier constraints."""
    psi = np.linspace(0.0, 1.0, 129)
    # 1. Monotonic baseline where uncertainty allows pellet outer-core density inversion
    ne_base = _hmode_profile(psi, 0.85e20, 0.82e20, 0.15e20, psi_ped=0.92)
    # 2. Radiatively hollow core Te baseline (e.g., core W/Ne radiation: Te(0) < Te(0.25))
    te_base = _hmode_profile(psi, 12.0e3, 3.5e3, 200.0, psi_ped=0.92)
    te_base *= 1.0 - 0.22 * np.exp(-((psi / 0.18) ** 2))  # hollow inside psi < 0.2

    ne_norm = ne_base / ne_base[0]
    te_norm = te_base / np.max(te_base)
    ne_bounds = (np.full_like(psi, -5.0), np.full_like(psi, 25.0))
    te_bounds = (np.full_like(psi, -4.0), np.full_like(psi, 25.0))
    rng = make_rng(42)

    saw_inverted_ne = False
    for _ in range(40):
        ne_draw = draw_bounded_perturbation(
            psi, ne_norm, 0.12 * ne_norm, 0.25,
            policy="physics_bounds", channel="ne", psi_ped_top=0.92,
            kappa_bounds=ne_bounds, rng=rng,
        ) * ne_base[0]
        te_draw = draw_bounded_perturbation(
            psi, te_norm, 0.12 * te_norm, 0.25,
            policy="physics_bounds", channel="te", psi_ped_top=0.92,
            kappa_bounds=te_bounds, rng=rng,
        ) * np.max(te_base)
        assert np.all(ne_draw > 0.0) and np.all(te_draw > 0.0)
        # Pedestal thermal barrier stays monotonically decreasing outward!
        i_ped = int(np.searchsorted(psi, 0.92))
        assert np.all(np.diff(te_draw[i_ped:]) <= 2e-4 * np.max(te_draw))
        # Check if outer-core pellet bump (ne[psi~0.85] > ne[0]) occurs naturally
        i_85 = int(np.searchsorted(psi, 0.85))
        if ne_draw[i_85] > ne_draw[0]:
            saw_inverted_ne = True
    assert saw_inverted_ne, "physics_bounds should permit pellet-like inverted ne draws"


def test_sigma_zero_identity_across_gradient_policies():
    """At sigma == 0, draw_bounded_perturbation returns the baseline bit-for-bit."""
    psi = np.linspace(0.0, 1.0, 65)
    te = _hmode_profile(psi, 1.0, 0.24, 0.02)
    z = np.zeros_like(psi)

    for gpol in ("legacy", "positivity_only", "physics_bounds"):
        draw = draw_bounded_perturbation(
            psi, te, z, 0.3, policy=gpol, channel="te", rng=make_rng(7),
        )
        np.testing.assert_array_equal(draw, te)


def test_tier1_stiffness_distinguishes_diiid_vs_sparc():
    """SPARC burning plasma (high B, high T, high Q_gB) must have smaller allowable
    gradient excursions Delta(R/L_T) and longer core Gibbs correlation lengths than DIII-D."""
    psi = np.linspace(0.0, 1.0, 129)

    def _make_bl(ne0, te0, Ip, B0, R0, a):
        ne = _hmode_profile(psi, ne0, 0.45 * ne0, 0.1 * ne0, psi_ped=0.93)
        te = _hmode_profile(psi, te0, 0.22 * te0, 100.0, psi_ped=0.93)
        j_bs = 1.5e6 * np.exp(-(((psi - 0.96) / 0.025) ** 2))
        j_ind = 2.0e6 * ((1.0 - psi) ** 1.4)
        bl = Baseline(
            psi_N=psi, j_phi=j_ind + j_bs, j_inductive=j_ind, j_BS=j_bs,
            psi_N_kinetic=psi, ne=ne, te=te, ni=0.9 * ne, ti=0.95 * te,
            Zeff=np.full_like(psi, 1.5), Ip_target=Ip, l_i_target=0.95,
            provenance="imas",
        )
        mg = miller_from_baseline(
            bl, R0_fallback=R0, a_fallback=a, B0_fallback=B0,
            kappa_edge_fallback=1.8, delta_edge_fallback=0.45,
        )
        bl.aux = {"miller_geometry": mg}
        unc_d = {
            "sigma_ne": 0.12 * ne, "sigma_te": 0.15 * te,
            "sigma_ni": 0.12 * 0.9 * ne, "sigma_ti": 0.15 * 0.95 * te,
            "sigma_jphi": 0.10 * (j_ind + j_bs),
        }
        return bl, unc_d

    bl_d3d, unc_d3d = _make_bl(6.0e19, 3.5e3, 1.2e6, 2.0, 1.67, 0.60)
    bl_sparc, unc_sparc = _make_bl(4.0e20, 20.0e3, 8.7e6, 12.2, 1.85, 0.57)

    s_d3d = compute_physics_length_scales(
        bl_d3d, unc_d3d, mode="tier1_analytical", tritium_fraction=0.0
    )
    s_sparc = compute_physics_length_scales(
        bl_sparc, unc_sparc, mode="tier1_analytical", tritium_fraction=0.5
    )

    core = (s_d3d.rho_kin >= 0.30) & (s_d3d.rho_kin <= 0.75)
    assert np.median(s_sparc.Q_gB_MW_m2[core]) > 5.0 * np.median(s_d3d.Q_gB_MW_m2[core])
    assert np.median(s_sparc.dkappa_ti[core]) < np.median(s_d3d.dkappa_ti[core])
    assert np.median(s_sparc.ti_ls[core]) > np.median(s_d3d.ti_ls[core])
    # Both H-mode plasmas drop ell(x) sharply across the pedestal
    assert np.min(s_sparc.te_ls[psi >= 0.93]) < 0.25 * np.median(s_sparc.te_ls[core])
    assert np.min(s_d3d.te_ls[psi >= 0.93]) < 0.40 * np.median(s_d3d.te_ls[core])


def test_gibbs_ls_form_chilenski_tanh_vs_pointwise():
    """gibbs_ls_form='chilenski_tanh' eliminates core ell'(psi_N) spikes while
    'pointwise' retains surface-by-surface stiffness variations."""
    psi = np.linspace(0.0, 1.0, 129)
    ne = _hmode_profile(psi, 4.0e20, 1.8e20, 0.3e20, psi_ped=0.93)
    te = _hmode_profile(psi, 20.0e3, 4.4e3, 120.0, psi_ped=0.93)
    j_bs = 1.5e6 * np.exp(-(((psi - 0.96) / 0.025) ** 2))
    j_ind = 2.0e6 * ((1.0 - psi) ** 1.4)
    bl = Baseline(
        psi_N=psi, j_phi=j_ind + j_bs, j_inductive=j_ind, j_BS=j_bs,
        psi_N_kinetic=psi, ne=ne, te=te, ni=0.9 * ne, ti=0.95 * te,
        Zeff=np.full_like(psi, 1.5), Ip_target=8.7e6, l_i_target=0.95,
        provenance="imas",
    )
    mg = miller_from_baseline(
        bl, R0_fallback=1.85, a_fallback=0.57, B0_fallback=12.2,
        kappa_edge_fallback=1.9, delta_edge_fallback=0.50,
    )
    bl.aux = {"miller_geometry": mg}
    unc_d = {
        "sigma_ne": 0.12 * ne, "sigma_te": 0.15 * te,
        "sigma_ni": 0.12 * 0.9 * ne, "sigma_ti": 0.15 * 0.95 * te,
        "sigma_jphi": 0.10 * (j_ind + j_bs),
    }

    s_tanh = compute_physics_length_scales(
        bl, unc_d, mode="tier1_analytical", gibbs_ls_form="chilenski_tanh", tritium_fraction=0.5
    )
    s_pt = compute_physics_length_scales(
        bl, unc_d, mode="tier1_analytical", gibbs_ls_form="pointwise", tritium_fraction=0.5
    )

    deep_core = psi <= 0.65
    d_ell_tanh = np.abs(np.gradient(s_tanh.te_ls, psi))[deep_core]
    assert np.max(d_ell_tanh) < 1.0e-4, f"Expected flat core plateau for chilenski_tanh, got max slope {np.max(d_ell_tanh)}"
    assert np.std(s_pt.te_ls[deep_core]) > 1.0e-3, "Expected radial structure in pointwise gibbs_ls_form"
    # Both share the same core median and pedestal minimum, and linear Chilenski Eq. 30
    # evaluates to 0.5 * (ell_core + ell_ped) at psi_ped_top (not 2 * ell_ped harmonic drop).
    core_mask = (s_tanh.rho_kin >= 0.25) & (s_tanh.rho_kin <= 0.75)
    core_med = float(np.median(s_tanh.te_ls[core_mask]))
    assert np.isclose(core_med, np.median(s_pt.te_ls[core_mask]), rtol=1e-3)
    assert np.min(s_tanh.te_ls[psi >= 0.94]) < 0.25 * core_med
    ell_at_ped_top = float(np.interp(s_tanh.pedestal.psi_ped_top, psi, s_tanh.te_ls))
    expected_mid = 0.5 * (core_med + s_tanh.pedestal.width_psiN)
    assert np.isclose(ell_at_ped_top, expected_mid, rtol=0.05)


def test_coord_phi_n_and_rho_tor_invariance():
    """Physical inverse scale lengths R0/L_s, critical gradients, excursions, and
    pedestal beta_p must be invariant when baseline.coord is 'phi_n' or 'rho_tor'."""
    psi_pol = np.linspace(0.0, 1.0, 161)
    ne = _hmode_profile(psi_pol, 7.5e19, 3.8e19, 0.8e19, psi_ped=0.93)
    te = _hmode_profile(psi_pol, 4.2e3, 1.1e3, 100.0, psi_ped=0.93)
    ni = 0.88 * ne
    ti = 0.95 * te
    zeff = np.full_like(psi_pol, 1.6)
    j_ind = 1.6e6 * ((1.0 - psi_pol) ** 1.4)
    j_bs = 0.9e6 * np.exp(-(((psi_pol - 0.96) / 0.025) ** 2))

    bl_psi = Baseline(
        psi_N=psi_pol, j_phi=j_ind + j_bs, j_inductive=j_ind, j_BS=j_bs,
        psi_N_kinetic=psi_pol, ne=ne, te=te, ni=ni, ti=ti, Zeff=zeff,
        Ip_target=1.4e6, l_i_target=0.95, provenance="imas", coord="psi_n",
    )
    # Build an H-mode Miller geometry with strong q-profile variation (q0=1.05 -> q_edge=5.5)
    mg_psi = miller_from_baseline(
        bl_psi, R0_fallback=1.67, a_fallback=0.60, B0_fallback=2.1,
        kappa_edge_fallback=1.82, delta_edge_fallback=0.45,
    )
    bl_psi.aux = {"miller_geometry": mg_psi}

    phi_grid = mg_psi.rho_tor_norm ** 2
    rho_grid = mg_psi.rho_tor_norm.copy()

    bl_phi_map = Baseline(
        psi_N=phi_grid, j_phi=j_ind + j_bs, j_inductive=j_ind, j_BS=j_bs,
        psi_N_kinetic=phi_grid, ne=ne, te=te, ni=ni, ti=ti, Zeff=zeff,
        Ip_target=1.4e6, l_i_target=0.95, provenance="reconstruction",
        coord="phi_n", psi_map=(psi_pol, phi_grid),
    )
    bl_phi_map.aux = {"miller_geometry": mg_psi}

    bl_phi_nomap = Baseline(
        psi_N=phi_grid, j_phi=j_ind + j_bs, j_inductive=j_ind, j_BS=j_bs,
        psi_N_kinetic=phi_grid, ne=ne, te=te, ni=ni, ti=ti, Zeff=zeff,
        Ip_target=1.4e6, l_i_target=0.95, provenance="imas",
        coord="phi_n", psi_map=None,
    )
    bl_phi_nomap.aux = {"miller_geometry": mg_psi}

    bl_rho = Baseline(
        psi_N=rho_grid, j_phi=j_ind + j_bs, j_inductive=j_ind, j_BS=j_bs,
        psi_N_kinetic=rho_grid, ne=ne, te=te, ni=ni, ti=ti, Zeff=zeff,
        Ip_target=1.4e6, l_i_target=0.95, provenance="imas",
        coord="rho_tor", psi_map=None,
    )
    bl_rho.aux = {"miller_geometry": mg_psi}

    unc = {
        "sigma_ne": 0.10 * ne, "sigma_te": 0.15 * te,
        "sigma_ni": 0.10 * ni, "sigma_ti": 0.15 * ti,
        "sigma_jphi": 0.10 * (j_ind + j_bs),
    }

    s_psi = compute_physics_length_scales(bl_psi, unc, current_evolution_time_s=0.3)
    s_phi = compute_physics_length_scales(bl_phi_map, unc, current_evolution_time_s=0.3)
    s_phi_fb = compute_physics_length_scales(bl_phi_nomap, unc, current_evolution_time_s=0.3)
    s_rho = compute_physics_length_scales(bl_rho, unc, current_evolution_time_s=0.3)

    for s_alt, grid_alt in ((s_phi, phi_grid), (s_phi_fb, phi_grid), (s_rho, rho_grid)):
        np.testing.assert_allclose(s_alt.psi_kin, grid_alt)
        np.testing.assert_allclose(s_alt.rlte_base, s_psi.rlte_base, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(s_alt.rlti_base, s_psi.rlti_base, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(s_alt.rlte_crit, s_psi.rlte_crit, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(s_alt.dkappa_te, s_psi.dkappa_te, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(s_alt.dkappa_ti, s_psi.dkappa_ti, rtol=1e-6, atol=1e-6)
        assert np.isclose(s_alt.pedestal.beta_p_ped, s_psi.pedestal.beta_p_ped, rtol=1e-6)
        expected_ped_run = float(np.interp(s_psi.pedestal.psi_ped_top, psi_pol, grid_alt))
        assert np.isclose(s_alt.pedestal.psi_ped_top, expected_ped_run, atol=1e-6)
        assert np.isclose(s_alt.pedestal.width_psiN, 1.0 - expected_ped_run, atol=1e-6)


def test_nonstationary_matern52_psd_cholesky_and_grid_validation():
    """Paciorek-Schervish quadratic-mean matern52 kernel must be strictly PSD
    across sharp core-to-pedestal length-scale drops, and _ell_matrices must
    reject mismatched grid sizes."""
    from bouquet.sampling import GPRProfilePerturber

    psi = np.linspace(0.0, 1.0, 129)
    ell = 0.018 + 1.60 * (0.5 * (1.0 - np.tanh((psi - 0.93) / 0.02)))
    pert = GPRProfilePerturber(kernel_func="matern52", length_scale=ell)
    K = pert._kernel(psi, psi)
    eig_min = float(np.min(np.linalg.eigvalsh(K)))
    assert eig_min > -1.0e-12, f"Expected PSD matern52 Gibbs kernel, got lambda_min={eig_min}"

    prof = _hmode_profile(psi, 1.0, 0.25, 0.03, psi_ped=0.93)
    draws = pert.generate_profiles(psi, prof, 0.12 * prof, n_samples=8, rng=make_rng(11))
    assert draws.shape == (8, 129) and np.all(np.isfinite(draws))

    with pytest.raises(ValueError, match="length_scale length"):
        pert.generate_profiles(np.linspace(0.0, 1.0, 65), prof[:65], 0.1 * prof[:65], rng=make_rng(1))


def test_pure_d_vs_dt_tritium_fraction_dilution_and_braginskii():
    """Default tritium_fraction=0.0 must keep pure-D hot-ion plasmas free of DT
    alpha heating (A_main=2.0), dilute ni==ne via Z_eff when tritium_fraction>0,
    and compute multi-species Braginskii alpha_ei without Z_eff over-scaling."""
    from bouquet.stiffness import (
        _braginskii_ei_exchange_W_m3_eV,
        _tier1_analytical_thresholds,
    )

    psi = np.linspace(0.0, 1.0, 101)
    ne = _hmode_profile(psi, 8.0e19, 4.0e19, 0.8e19, psi_ped=0.93)
    te = _hmode_profile(psi, 6.0e3, 1.5e3, 100.0, psi_ped=0.93)
    ti = _hmode_profile(psi, 15.0e3, 2.5e3, 120.0, psi_ped=0.93)  # hot-ion DIII-D / JET shot
    zeff = np.full_like(psi, 2.0)  # Carbon Z=6 -> fuel dilution (6 - 2)/(6 - 1) = 0.80
    j = 1.8e6 * (1.0 - psi)

    # Set ni == ne (as in standard IDA reconstructions before dilution)
    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=ne.copy(), ti=ti, Zeff=zeff,
        Ip_target=1.6e6, l_i_target=0.9, provenance="imas",
    )
    mg = miller_from_baseline(bl, R0_fallback=1.67, a_fallback=0.60, B0_fallback=2.1)
    bl.aux = {"miller_geometry": mg}
    unc = {
        "sigma_ne": 0.10 * ne, "sigma_te": 0.15 * te,
        "sigma_ni": 0.10 * ne, "sigma_ti": 0.15 * ti,
        "sigma_jphi": 0.10 * j,
    }

    t1_dd = _tier1_analytical_thresholds(mg, psi, ne, te, ne, ti, zeff, tritium_fraction=0.0, Z_imp=6.0)
    t1_dt_undiluted = _tier1_analytical_thresholds(mg, psi, ne, te, 0.999 * ne, ti, zeff, tritium_fraction=0.5, Z_imp=6.0)
    t1_dt_diluted = _tier1_analytical_thresholds(mg, psi, ne, te, ne, ti, zeff, tritium_fraction=0.5, Z_imp=6.0)

    assert t1_dd["p_alpha_tot_W"] == 0.0
    assert t1_dd["A_main"] == 2.0
    assert t1_dt_diluted["p_alpha_tot_W"] > 1.0e5
    assert t1_dt_diluted["A_main"] == 2.5
    # Dilution by 0.80 reduces n_fuel^2 (and P_alpha) by ~0.64x compared to undiluted ni ~ ne
    assert np.isclose(
        t1_dt_diluted["p_alpha_tot_W"] / t1_dt_undiluted["p_alpha_tot_W"],
        0.64 / (0.999 ** 2),
        rtol=0.02,
    )

    # Multi-species Braginskii: for fully stripped A_Z = 2 * Z_Z impurities (e.g. C6+, A=12)
    # in Deuterium (A=2), sum_s n_s Z_s^2 / A_s == 0.5 * n_e independent of Z_eff!
    a_ei_z1 = _braginskii_ei_exchange_W_m3_eV(ne, te, np.ones_like(psi), A_main=2.0, Z_imp=6.0, A_imp=12.0)
    a_ei_z3 = _braginskii_ei_exchange_W_m3_eV(ne, te, np.full_like(psi, 3.0), A_main=2.0, Z_imp=6.0, A_imp=12.0)
    np.testing.assert_allclose(a_ei_z3, a_ei_z1, rtol=1e-12)

    # Default compute_physics_length_scales uses pure D (tritium_fraction=0.0)
    s_default = compute_physics_length_scales(bl, unc, mode="tier1_analytical")
    s_explicit_dd = compute_physics_length_scales(bl, unc, mode="tier1_analytical", tritium_fraction=0.0)
    np.testing.assert_allclose(s_default.te_ls, s_explicit_dd.te_ls)


def test_tier1_subcritical_margin_itg_dilution_reversed_shear_and_rhostar():
    """Verify Tier-1 subcritical margin, ITG dilution stabilization, reversed-shear
    |s_hat|/q stabilization, chord line-averaged density, and saturating rho_star."""
    from bouquet.stiffness import _tier1_analytical_thresholds

    psi = np.linspace(0.0, 1.0, 101)
    ne = _hmode_profile(psi, 6.0e19, 3.0e19, 0.8e19, psi_ped=0.93)
    te_flat = np.full_like(psi, 3.0e3)  # completely subcritical flat temperature (RLTS = 0)
    te_steep = _hmode_profile(psi, 5.0e3, 1.0e3, 100.0, psi_ped=0.93)
    j = 1.4e6 * (1.0 - psi)
    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te_steep, ni=0.85 * ne, ti=te_steep,
        Zeff=np.full_like(psi, 1.8), Ip_target=1.2e6, l_i_target=0.9, provenance="imas",
    )
    mg = miller_from_baseline(bl, R0_fallback=1.70, a_fallback=0.60, B0_fallback=2.2)

    # 1. Subcritical flat profile includes (rlts_crit - 0) margin > supercritical steep excursion
    t1_flat = _tier1_analytical_thresholds(mg, psi, ne, te_flat, 0.85 * ne, te_flat, np.full_like(psi, 1.8))
    mid = 45
    assert t1_flat["dkappa_ti"][mid] > t1_flat["rlti_crit"][mid]

    # 2. Impurity dilution (higher Zeff -> smaller f_i = ni/ne) increases ITG critical gradient
    t1_clean = _tier1_analytical_thresholds(mg, psi, ne, te_steep, 0.98 * ne, te_steep, np.full_like(psi, 1.1))
    t1_dirty = _tier1_analytical_thresholds(mg, psi, ne, te_steep, 0.60 * ne, te_steep, np.full_like(psi, 3.0))
    assert float(t1_dirty["rlti_crit"][mid]) > float(t1_clean["rlti_crit"][mid])

    # 3. Reversed magnetic shear (s_hat = -1.2 in inner AT/ITB core) stabilizes ETG & ITG
    #    via |s_hat|/q rather than collapsing to 0.05
    mg_rev = miller_from_baseline(bl, R0_fallback=1.70, a_fallback=0.60, B0_fallback=2.2)
    mg_zero = miller_from_baseline(bl, R0_fallback=1.70, a_fallback=0.60, B0_fallback=2.2)
    mg_rev.s_hat = np.full_like(psi, -1.2)
    mg_zero.s_hat = np.zeros_like(psi)
    t1_rev = _tier1_analytical_thresholds(mg_rev, psi, ne, te_steep, 0.85 * ne, te_steep, np.full_like(psi, 1.8))
    t1_zero = _tier1_analytical_thresholds(mg_zero, psi, ne, te_steep, 0.85 * ne, te_steep, np.full_like(psi, 1.8))
    itb_idx = 25  # psi_N = 0.25 inner reversed-shear core
    assert float(t1_rev["rlti_crit"][itb_idx]) > float(t1_zero["rlti_crit"][itb_idx]) + 1.0
    assert float(t1_rev["rlte_crit"][itb_idx]) > float(t1_zero["rlte_crit"][itb_idx]) + 0.7

    # 4. Chord line-averaged density in Martin P_LH is invariant to edge knot clustering
    psi_clustered = np.unique(np.concatenate([np.linspace(0.0, 0.85, 35), np.linspace(0.85, 1.0, 95)]))
    mg_cl = mg.interpolate(psi_clustered)
    ne_cl = np.interp(psi_clustered, psi, ne)
    te_cl = np.interp(psi_clustered, psi, te_steep)
    t1_unif = _tier1_analytical_thresholds(mg, psi, ne, te_steep, 0.85 * ne, te_steep, np.full_like(psi, 1.8))
    t1_cl = _tier1_analytical_thresholds(mg_cl, psi_clustered, ne_cl, te_cl, 0.85 * ne_cl, te_cl, np.full_like(psi_clustered, 1.8))
    assert np.isclose(
        float(np.interp(0.5, psi_clustered, t1_cl["Qe_phys_W_m2"])),
        float(np.interp(0.5, psi, t1_unif["Qe_phys_W_m2"])),
        rtol=0.02,
    )


def test_lmode_has_pedestal_false_flat_profile_and_lognormal_fallback():
    """Verify L-mode has_pedestal=False skips edge monotonicity while applying core_mask
    across [0, 1], flat baselines do not trigger edge drop bias, and extreme-sigma
    log-normal fallback emits UserWarning while respecting bounds and unbiased mean."""
    psi = np.linspace(0.0, 1.0, 81)

    # 1. L-mode profile decreasing everywhere: has_pedestal=False permits outward edge fluctuations
    prof_l = 1.0 - 0.70 * (psi ** 1.3)
    sig_l = 0.15 * np.ones_like(psi)
    rng_l = make_rng(314)
    l_draws = np.array([
        draw_bounded_perturbation(
            psi, prof_l, sig_l, 0.15,
            policy="physics_bounds", channel="te", has_pedestal=False,
            kappa_bounds=(-8.0, 18.0), rng=rng_l,
        )
        for _ in range(80)
    ])
    # At least some L-mode draws have positive slope near the edge (psi >= 0.90)
    i_90 = int(np.searchsorted(psi, 0.90))
    assert np.any(np.diff(l_draws[:, i_90:], axis=1) > 1.0e-3)

    # 2. Constant flat profile with has_pedestal=True does NOT trigger artificial edge drop bias
    prof_flat = np.ones_like(psi)
    rng_fl = make_rng(271)
    fl_draws = np.array([
        draw_bounded_perturbation(
            psi, prof_flat, 0.10 * prof_flat, 0.25,
            policy="physics_bounds", channel="te", has_pedestal=True,
            psi_ped_top=0.90, kappa_bounds=(-4.0, 4.0), rng=rng_fl,
        )
        for _ in range(600)
    ])
    assert abs(float(np.mean(fl_draws[:, i_90] - fl_draws[:, -1]))) < 0.01

    # 3. Log-normal fallback (triggered via max_draws=0 with extreme separatrix sigma=0.25 >> f_sep=0.01):
    #    emits UserWarning, obeys pedestal monotonicity & kappa_bounds, and avoids 4.68x Ito drift bias.
    prof_h = _hmode_profile(psi, 1.0, 0.22, 0.01, psi_ped=0.92)
    sig_h = np.where(psi >= 0.90, 0.25, 0.10 * prof_h)
    ell_h = 0.03 + 0.27 * 0.5 * (1.0 - np.tanh((psi - 0.92) / 0.03))
    rng_fb = make_rng(77)
    with pytest.warns(UserWarning, match="log-normal GP rejection"):
        fb_pos = np.array([
            draw_bounded_perturbation(
                psi, prof_h, sig_h, ell_h,
                policy="positivity_only", channel="te", max_draws=0, rng=rng_fb,
            )
            for _ in range(800)
        ])
    # Unconditioned positivity_only fallback mean matches prof_h[-1] within Monte Carlo tolerance (not 4.68x!)
    assert 0.80 < float(np.mean(fb_pos[:, -1]) / prof_h[-1]) < 1.20

    rng_fb2 = make_rng(88)
    i_ped = int(np.searchsorted(psi, 0.92))
    with pytest.warns(UserWarning, match="log-normal GP rejection"):
        fb_pb = np.array([
            draw_bounded_perturbation(
                psi, prof_h, sig_h, ell_h,
                policy="physics_bounds", channel="te", psi_ped_top=0.92,
                kappa_bounds=(-6.0, 45.0), max_draws=0, rng=rng_fb2,
            )
            for _ in range(80)
        ])
    for row in fb_pb:
        assert np.all(np.diff(row[i_ped:]) <= 1.0e-4 * np.max(row))


def test_clipped_kinetic_grid_j_ls_bounded_and_smooth_q1_transition():
    """When psi_kin[-1] = 0.98 < psi_j[-1] = 1.0 and current_evolution_time_s > 0,
    j_ls must remain O(0.01..1) without edge clamping explosion and transition
    smoothly across the q = 1 sawtooth radius."""
    from bouquet.sampling import GPRProfilePerturber

    psi_j = np.linspace(0.0, 1.0, 129)
    psi_kin = np.linspace(0.0, 0.98, 101)  # clipped inside LCFS (e.g. IDA grid)
    ne = _hmode_profile(psi_kin, 7.0e19, 3.5e19, 0.9e19, psi_ped=0.92)
    te = _hmode_profile(psi_kin, 4.5e3, 1.1e3, 150.0, psi_ped=0.92)
    j = 2.2e6 * ((1.0 - psi_j) ** 1.3)

    bl = Baseline(
        psi_N=psi_j, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi_kin, ne=ne, te=te, ni=0.9 * ne, ti=0.95 * te,
        Zeff=np.full_like(psi_kin, 1.6), Ip_target=1.8e6, l_i_target=0.95,
        provenance="imas",
    )
    # Build full equilibrium MillerGeometry on [0, 1] with q0 < 1 so q=1 crossing exists
    mg_full = miller_from_baseline(
        bl, psi_N_out=psi_j, R0_fallback=1.67, a_fallback=0.60, B0_fallback=1.9,
    )
    mg_full.q = 0.82 + 3.2 * (psi_j ** 1.5)  # q=1 crossing around psi_N ~ 0.15
    bl.aux = {"miller_geometry": mg_full}

    unc = {
        "sigma_ne": 0.10 * ne, "sigma_te": 0.15 * te,
        "sigma_ni": 0.10 * 0.9 * ne, "sigma_ti": 0.15 * 0.95 * te,
        "sigma_jphi": 0.10 * j,
    }
    res = compute_physics_length_scales(bl, unc, mode="tier1_analytical", current_evolution_time_s=0.05)
    assert res.j_ls.shape == psi_j.shape
    assert np.all(np.isfinite(res.j_ls)) and np.all(res.j_ls > 0.0)
    assert float(np.max(res.j_ls)) < 1.0, f"j_ls exploded at edge: {res.j_ls[-5:]}"

    # Verify Gibbs GP current-slope std ramps smoothly across q=1 with no step-jump spike
    # above the adjacent hot resistive skin-depth zone (psi_j in [0.22, 0.35]).
    pert = GPRProfilePerturber(kernel_func="rbf", length_scale=res.j_ls)
    j_draws = pert.generate_profiles(psi_j, np.zeros_like(psi_j), np.ones_like(psi_j), n_samples=600, rng=make_rng(5))
    slope_std = np.std(np.gradient(j_draws, psi_j, axis=1), axis=0)
    i_q1 = int(np.argmin(np.abs(mg_full.q - 1.0)))
    adj_skin_ref = float(np.max(slope_std[(psi_j >= 0.22) & (psi_j <= 0.35)]))
    assert float(np.max(slope_std[max(i_q1 - 3, 0): i_q1 + 4])) <= 1.05 * adj_skin_ref

