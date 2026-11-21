"""Tests for power-balance/surrogate chi, coord='phi_n', tritium_fraction, and UncertaintyConfig hardening."""

from types import SimpleNamespace
import json
import os
import warnings

import numpy as np
import pytest

import bouquet
from bouquet.baseline import Baseline, resolve_uncertainty
from bouquet.config import (
    BouquetConfig,
    GenerationConfig,
    ImasSource,
    SolverConfig,
    UncertaintyConfig,
    with_absolute_paths,
)
from bouquet.kinetic_sampler import KineticBase, compute_power_balance_chi, sample_kinetics
from bouquet.sampling import make_rng


def test_power_balance_chi_sigma0_axis_lhopital_and_hollow_smoothness():
    psi = np.linspace(0.0, 1.0, 101)
    ne0 = 1.0e20 * (1.0 - 0.7 * (psi ** 1.5))
    # Parabolic core profile with dTe/dpsi == 0 at magnetic axis
    te0 = 1000.0 + 9000.0 * (1.0 - psi ** 2)
    chi0 = 0.5 + 1.5 * (psi ** 2)

    # 1. Exact bit-for-bit identity at zero perturbation
    chi_same = compute_power_balance_chi(psi, chi0, ne0, te0, ne0, te0)
    np.testing.assert_array_equal(chi_same, chi0)

    # 2. Exact 1/alpha scaling everywhere (including magnetic axis psi=0!) under T_draw = alpha * T_0
    alpha = 1.5
    chi_scaled = compute_power_balance_chi(psi, chi0, ne0, te0, ne0, alpha * te0)
    np.testing.assert_allclose(chi_scaled / chi0, 1.0 / alpha, rtol=1e-12, atol=1e-12)

    # 3. Steepening |dTe/dpsi| by 1.5x at fixed pedestal/edge reduces chi_e across core
    te_steep = 1000.0 + 1.5 * (te0 - 1000.0)
    chi_steep = compute_power_balance_chi(psi, chi0, ne0, te0, ne0, te_steep)
    np.testing.assert_allclose(chi_steep / chi0, 1.0 / 1.5, rtol=1e-12, atol=1e-12)

    # 4. Radiatively hollow core (dTe/dpsi > 0 inside psi < 0.25) stays strictly positive & C^1 smooth
    te_hollow = te0 * (1.0 - 0.45 * np.exp(-((psi / 0.20) ** 2)))
    chi_hollow = compute_power_balance_chi(psi, chi0, ne0, te0, ne0, te_hollow)
    assert np.all(chi_hollow > 0.0) and np.all(np.isfinite(chi_hollow))
    # Verify smooth variation under small shift in hollow depth (no maximum(g, 0) kink jump)
    te_hollow_eps = te0 * (1.0 - 0.451 * np.exp(-((psi / 0.20) ** 2)))
    chi_hollow_eps = compute_power_balance_chi(psi, chi0, ne0, te0, ne0, te_hollow_eps)
    assert np.max(np.abs(chi_hollow_eps - chi_hollow) / chi_hollow) < 0.015


def test_alpha_heating_feedback_pure_dd_zero_vs_dt_burning():
    psi = np.linspace(0.0, 1.0, 101)
    ne0 = 3.0e20 * (1.0 - 0.5 * (psi ** 2))
    ni0 = 0.9 * ne0
    te0 = 14.0e3 * (1.0 - 0.85 * psi)
    ti0 = 14.0e3 * (1.0 - 0.85 * psi)
    # Burning-plasma conductive diffusivity (~0.18 m^2/s -> P_cond comparable to P_alpha)
    chi0 = np.full_like(psi, 0.18)
    vol = 20.0 * (psi ** 1.5)
    ti_hot = 1.25 * ti0

    chi_no_alpha = compute_power_balance_chi(
        psi, chi0, ne0, te0, ne0, te0,
        alpha_heating_feedback=False, tritium_fraction=0.5,
        ni_base=ni0, ti_base=ti0, ni_draw=ni0, ti_draw=ti_hot, volume_m3=vol,
    )

    # Pure Deuterium (default tritium_fraction=0.0): P_alpha == 0 identically even at Ti0 = 14 keV!
    chi_pure_dd = compute_power_balance_chi(
        psi, chi0, ne0, te0, ne0, te0,
        alpha_heating_feedback=True,
        ni_base=ni0, ti_base=ti0, ni_draw=ni0, ti_draw=ti_hot, volume_m3=vol,
    )
    np.testing.assert_array_equal(chi_pure_dd, chi_no_alpha)

    # 50-50 D-T burning plasma (tritium_fraction=0.5): +25% Ti boosts DT alpha power & chi_e, chi_i
    chi_dt_e = compute_power_balance_chi(
        psi, chi0, ne0, te0, ne0, te0, species="e",
        alpha_heating_feedback=True, tritium_fraction=0.5,
        ni_base=ni0, ti_base=ti0, ni_draw=ni0, ti_draw=ti_hot,
        te_base=te0, te_draw=te0, volume_m3=vol,
    )
    chi_dt_i = compute_power_balance_chi(
        psi, chi0, ni0, ti0, ni0, ti0, species="i",
        alpha_heating_feedback=True, tritium_fraction=0.5,
        ni_base=ni0, ti_base=ti0, ni_draw=ni0, ti_draw=ti_hot,
        te_base=te0, te_draw=te0, volume_m3=vol,
    )
    core = (psi >= 0.15) & (psi <= 0.80)
    assert np.all(chi_dt_e[core] > 1.15 * chi_no_alpha[core])
    assert np.all(chi_dt_i[core] > 1.05 * chi0[core])


def test_surrogate_chi_sigma0_zero_onnx_calls_and_cross_species_flux_ratio(monkeypatch):
    import bouquet.surrogates.tglfnn as tglf_mod

    psi = np.linspace(0.0, 1.0, 41)
    ne = 1.0e20 * (1.0 - 0.6 * psi)
    te = 6.0e3 * (1.0 - 0.8 * psi)
    ni = 0.9 * ne
    ti = 6.0e3 * (1.0 - 0.8 * psi)
    chi_e0 = 0.7 + 0.8 * psi
    omega0 = 4.0e4 * (1.0 - psi)
    z0 = np.zeros_like(psi)

    calls = []

    def _forbid_onnx(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("ONNX surrogate must NOT be called at sigma == 0")

    monkeypatch.setattr(tglf_mod, "evaluate_tglfnn_stiffness", _forbid_onnx)

    kb_zero = KineticBase(
        psi_kin=psi, ne=ne, te=te, ni=ni, ti=ti,
        sigma_ne=z0, sigma_te=z0, sigma_ni=z0, sigma_ti=z0,
        n_ls=0.4, t_ls=0.4,
        aux_sigmas={"zeff": z0, "chi_e": z0},
        aux_baselines={"zeff": np.full_like(psi, 1.5), "chi_e": chi_e0, "omega_tor": omega0},
        chi_consistency="surrogate",
        tglfnn_model="nonexistent_model_at_sigma0",
        tritium_fraction=0.5,
    )
    target = float(np.trapezoid(ne * te + ni * ti, psi))
    kd0 = sample_kinetics(
        kb_zero, make_rng(7),
        lambda d: float(np.trapezoid(d.ne * d.te + d.ni * d.ti, psi)),
        target, p_thresh=0.20,
    )
    assert calls == []
    np.testing.assert_array_equal(kd0.aux["chi_e"], chi_e0)

    # Perturb ONLY Ti while holding (ne, te, ni, zeff) at sigma=0: surrogate must
    # still evaluate Q_NN(draw)/Q_NN(base) with omega_tor_rad_s and A_main=2.5!
    seen_calls = []

    def _mock_eval(mg, psi_arg, ne_a, te_a, ni_a, ti_a, z_a, **kw):
        seen_calls.append(kw)
        # Mock Qe_W_m2 sensitive to Ti perturbation
        ti_ratio = np.asarray(ti_a, dtype=float) / np.maximum(ti, 1.0)
        qe_w = 5.0e4 * (ti_ratio ** 2)
        qi_w = 6.0e4 * (ti_ratio ** 2)
        return SimpleNamespace(Qe_W_m2=qe_w, Qi_W_m2=qi_w)

    monkeypatch.setattr(tglf_mod, "evaluate_tglfnn_stiffness", _mock_eval)

    kb_ti = KineticBase(
        psi_kin=psi, ne=ne, te=te, ni=ni, ti=ti,
        sigma_ne=z0, sigma_te=z0, sigma_ni=z0, sigma_ti=0.15 * ti,
        n_ls=0.4, t_ls=0.4,
        ni_from_zeff=False,
        aux_baselines={"zeff": np.full_like(psi, 1.5), "chi_e": chi_e0, "omega_tor": omega0},
        chi_consistency="surrogate",
        tritium_fraction=0.5,
    )
    kd_ti = sample_kinetics(
        kb_ti, make_rng(11),
        lambda d: float(np.trapezoid(d.ne * d.te + d.ni * d.ti, psi)),
        target, p_thresh=0.50,
    )
    assert len(seen_calls) == 2
    assert seen_calls[0]["A_main"] == pytest.approx(2.5)
    np.testing.assert_allclose(seen_calls[0]["omega_tor_rad_s"], omega0)
    assert not np.allclose(kd_ti.aux["chi_e"], chi_e0)


def test_resolve_uncertainty_preserves_chi_baseline_without_dummy_sigma(tmp_path):
    psi = np.linspace(0.0, 1.0, 65)
    ne = 1.0e20 * (1.0 - 0.6 * psi)
    te = 8.0e3 * (1.0 - 0.85 * psi)
    chi_e0 = 0.8 + 1.2 * psi
    j = 1.5e6 * (1.0 - psi)

    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=0.7 * j, j_BS=0.3 * j,
        psi_N_kinetic=psi, ne=ne, te=te, ni=0.9 * ne, ti=te,
        Zeff=np.full_like(psi, 1.5), Ip_target=2.0e6, l_i_target=0.9,
        provenance="imas", aux={"chi_e": chi_e0},
    )
    cfg = BouquetConfig(
        source=ImasSource(ids_path=str(tmp_path / "dd.json")),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out"),
        generation=GenerationConfig(),
        uncertainty=UncertaintyConfig(
            ne_scalar_sigma=0.08, te_scalar_sigma=0.12,
            ni_scalar_sigma=0.08, ti_scalar_sigma=0.12,
            zeff_scalar_sigma=0.0,
            chi_consistency="power_balance",
            gradient_policy="physics_bounds",
            log_sigma_sources=False,
        ),
    )
    env = resolve_uncertainty(cfg, bl)
    assert "chi_e" in env["aux_baselines"]
    assert env["tritium_fraction"] == 0.0
    kb = KineticBase(
        psi_kin=psi, ne=ne, te=te, ni=0.9 * ne, ti=te,
        sigma_ne=env["sigma_ne"], sigma_te=env["sigma_te"],
        sigma_ni=env["sigma_ni"], sigma_ti=env["sigma_ti"],
        n_ls=env["n_ls"], t_ls=env["t_ls"],
        aux_sigmas=env["aux_sigmas"], aux_baselines=env["aux_baselines"],
        aux_length_scales=env["aux_length_scales"],
        **env["kinetic_sampling_opts"],
    )
    target = float(np.trapezoid(ne * te + 0.9 * ne * te, psi))
    kd = sample_kinetics(
        kb, make_rng(99),
        lambda d: float(np.trapezoid(d.ne * d.te + d.ni * d.ti, psi)),
        target, p_thresh=0.15,
    )
    assert "chi_e" in kd.aux
    assert not np.allclose(kd.aux["chi_e"], chi_e0)
    assert np.all(kd.aux["chi_e"] > 0.0)


def test_resolve_uncertainty_phi_n_coord_and_imas_tritium_detection(tmp_path):
    # q(psi_N) = 1 + 3.5 * psi_N^2 => Phi_N(psi_N) = (psi_N + 3.5/3 * psi_N^3) / (1 + 3.5/3)
    psi_pol = np.linspace(0.0, 1.0, 81)
    phi_n = (psi_pol + (3.5 / 3.0) * (psi_pol ** 3)) / (1.0 + 3.5 / 3.0)
    ne = 1.0e20 * (1.0 - 0.6 * psi_pol)
    te = 7.0e3 * (1.0 - 0.85 * psi_pol)
    j = 1.4e6 * (1.0 - psi_pol)

    bl_psi = Baseline(
        psi_N=psi_pol, j_phi=j, j_inductive=0.7 * j, j_BS=0.3 * j,
        psi_N_kinetic=psi_pol, ne=ne, te=te, ni=0.9 * ne, ti=te,
        Zeff=np.full_like(psi_pol, 1.5), Ip_target=1.8e6, l_i_target=0.95,
        provenance="imas", coord="psi_n", psi_map=None,
    )
    bl_phi = Baseline(
        psi_N=phi_n, j_phi=j, j_inductive=0.7 * j, j_BS=0.3 * j,
        psi_N_kinetic=phi_n, ne=ne, te=te, ni=0.9 * ne, ti=te,
        Zeff=np.full_like(phi_n, 1.5), Ip_target=1.8e6, l_i_target=0.95,
        provenance="imas", coord="phi_n", psi_map=(psi_pol, phi_n),
    )

    # 1. Test coord='phi_n' vs coord='psi_n' geometry mapping in resolve_uncertainty
    cfg_psi = BouquetConfig(
        source=ImasSource(ids_path=str(tmp_path / "missing_dd.json")),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out_psi"),
        generation=GenerationConfig(),
        uncertainty=UncertaintyConfig(
            ne_scalar_sigma=0.05, te_scalar_sigma=0.10,
            ni_scalar_sigma=0.05, ti_scalar_sigma=0.10,
            zeff_scalar_sigma=0.0,
            gradient_policy="physics_bounds",
            tritium_fraction=0.5,
            log_sigma_sources=False,
        ),
    )
    cfg_phi = BouquetConfig(
        source=ImasSource(ids_path=str(tmp_path / "missing_dd.json"), coord="phi_n"),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out_phi"),
        generation=GenerationConfig(),
        uncertainty=UncertaintyConfig(
            ne_scalar_sigma=0.05, te_scalar_sigma=0.10,
            ni_scalar_sigma=0.05, ti_scalar_sigma=0.10,
            zeff_scalar_sigma=0.0,
            gradient_policy="physics_bounds",
            tritium_fraction=0.5,
            log_sigma_sources=False,
        ),
    )
    env_psi = resolve_uncertainty(cfg_psi, bl_psi)
    env_phi = resolve_uncertainty(cfg_phi, bl_phi)
    assert env_psi["tritium_fraction"] == pytest.approx(0.5)
    assert env_phi["tritium_fraction"] == pytest.approx(0.5)
    # Mapped r_norm_kin on corresponding flux surfaces must match identically!
    np.testing.assert_allclose(
        env_phi["kinetic_sampling_opts"]["r_norm_kin"],
        env_psi["kinetic_sampling_opts"]["r_norm_kin"],
        rtol=1e-6, atol=1e-6,
    )

    # 2. Synthetic 50-50 D-T IMAS JSON auto-detects tritium_fraction == 0.5
    dt_json = tmp_path / "dt_dd.json"
    dt_json.write_text(json.dumps({
        "core_profiles": {
            "profiles_1d": [{
                "time": 1.0,
                "ion": [
                    {"label": "D", "element": [{"z_n": 1.0, "a": 2.0}], "density_thermal": (0.45 * ne).tolist()},
                    {"label": "T", "element": [{"z_n": 1.0, "a": 3.0}], "density_thermal": (0.45 * ne).tolist()},
                    {"label": "C", "element": [{"z_n": 6.0, "a": 12.0}], "density_thermal": (0.0167 * ne).tolist()},
                ],
            }]
        }
    }), encoding="utf-8")
    cfg_dt_auto = BouquetConfig(
        source=ImasSource(ids_path=str(dt_json)),
        solver=SolverConfig(mesh_path="unused"),
        output_header=str(tmp_path / "out_dt"),
        generation=GenerationConfig(),
        uncertainty=UncertaintyConfig(
            ne_scalar_sigma=0.05, te_scalar_sigma=0.10,
            ni_scalar_sigma=0.05, ti_scalar_sigma=0.10,
            zeff_scalar_sigma=0.0, log_sigma_sources=False,
        ),
    )
    assert resolve_uncertainty(cfg_dt_auto, bl_psi)["tritium_fraction"] == pytest.approx(0.5)

    # Pure-D tracked D3D-like IMAS JSON resolves tritium_fraction == 0.0
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d3d_json = os.path.join(repo_root, "examples", "D3D-like", "D3Dlike_baseline_omas.json")
    if os.path.isfile(d3d_json):
        cfg_d3d = BouquetConfig(
            source=ImasSource(ids_path=d3d_json),
            solver=SolverConfig(mesh_path="unused"),
            output_header=str(tmp_path / "out_d3d"),
            generation=GenerationConfig(),
            uncertainty=UncertaintyConfig(
                ne_scalar_sigma=0.05, te_scalar_sigma=0.10,
                ni_scalar_sigma=0.05, ti_scalar_sigma=0.10,
                zeff_scalar_sigma=0.0, log_sigma_sources=False,
            ),
        )
        assert resolve_uncertainty(cfg_d3d, bl_psi)["tritium_fraction"] == 0.0


def test_uncertainty_config_validation_checked_keys_and_abs_paths(tmp_path, monkeypatch):
    # Reject bools, inf, nan, invalid strings, and out-of-range tritium_fraction
    for bad_kw in (
        {"flux_uncertainty_rel": float("inf")},
        {"flux_uncertainty_rel": float("nan")},
        {"flux_uncertainty_rel": True},
        {"flux_uncertainty_rel": -0.01},
        {"stiff_gradient_ls": float("inf")},
        {"stiff_gradient_ls": True},
        {"stiff_gradient_ls": 0.0},
        {"current_evolution_time_s": float("inf")},
        {"current_evolution_time_s": True},
        {"current_evolution_time_s": 0.0},
        {"alpha_heating_feedback": "no"},
        {"alpha_heating_feedback": 1},
        {"tglfnn_model": ""},
        {"tglfnn_model": "   "},
        {"tglfnn_model": None},
        {"tritium_fraction": True},
        {"tritium_fraction": float("nan")},
        {"tritium_fraction": -0.05},
        {"tritium_fraction": 1.05},
    ):
        with pytest.raises(ValueError):
            UncertaintyConfig(**bad_kw)

    # Round-trip valid tritium_fraction and reject unknown/misspelled uncertainty keys
    cfg = BouquetConfig(
        source=ImasSource(ids_path="dd.json"),
        solver=SolverConfig(mesh_path="mesh.h5"),
        output_header="out",
        generation=GenerationConfig(),
        uncertainty=UncertaintyConfig(tritium_fraction=0.5, sampling_mode="stiff_inward"),
    )
    d = cfg.to_dict()
    cfg_rt = BouquetConfig.from_dict(d)
    assert cfg_rt.uncertainty.tritium_fraction == 0.5
    assert cfg_rt.uncertainty.sampling_mode == "stiff_inward"

    d_typo = cfg.to_dict()
    d_typo["uncertainty"]["sampling_mdoe"] = "stiff_inward"
    with pytest.raises(ValueError, match="sampling_mdoe.*sampling_mode"):
        BouquetConfig.from_dict(d_typo)

    d_retired = cfg.to_dict()
    d_retired["uncertainty"]["sigma_ni_from_ne"] = True
    with pytest.warns(UserWarning, match="sigma_ni_from_ne"):
        BouquetConfig.from_dict(d_retired)

    # with_absolute_paths resolves relative tglfnn_model directory paths, keeps named specs
    monkeypatch.chdir(tmp_path)
    rel_dir = tmp_path / "local_tglf_models"
    rel_dir.mkdir()
    cfg_rel = BouquetConfig(
        source=ImasSource(ids_path="dd.json"),
        solver=SolverConfig(mesh_path="mesh.h5"),
        output_header="out",
        generation=GenerationConfig(),
        uncertainty=UncertaintyConfig(tglfnn_model="local_tglf_models"),
    )
    assert with_absolute_paths(cfg_rel).uncertainty.tglfnn_model == str(rel_dir.resolve())
    assert with_absolute_paths(cfg).uncertainty.tglfnn_model == "sat2_em_d3d_azf-1_withnegD"


def test_swb_sigma0_preserves_chi_mono_zero_axis_and_public_exports():
    from bouquet.swb_draws import swb_draw

    psi = np.linspace(0.0, 1.0, 33)
    ne = 1.0e20 * (1.0 - 0.5 * psi)
    te = 5.0e3 * (1.0 - 0.8 * psi)
    chi_e0 = 0.6 + 0.9 * psi
    chi_i0 = 0.8 + 1.1 * psi
    jseed = 1.0e6 * (1.0 - psi)

    def _dummy_recipe(kin, js):
        return {
            "j_inductive": js,
            "isolated_j_BS": 0.2 * js,
            "total_j_phi": 1.2 * js,
            "ip_rel_err": 0.0,
        }

    *_, diag = swb_draw(
        None, psi, ne * te, ne, te, 0.9 * ne, te,
        None, None, None, None, None, 0.5, 0.4, 0.25,
        np.full_like(psi, 1.5), jseed, _dummy_recipe,
        coord="psi_n", rng=42, p_thresh=0.1,
        aux_baselines={"chi_e": chi_e0, "chi_i": chi_i0},
        kinetic_sampling_opts={"chi_consistency": "power_balance"},
    )
    np.testing.assert_array_equal(diag["aux"]["chi_e"], chi_e0)
    np.testing.assert_array_equal(diag["aux"]["chi_i"], chi_i0)

    # _mono handles hollow prof[0] == 0.0 without NaN / ZeroDivisionError
    te_zero_axis = 4.0e3 * psi * (1.0 - 0.5 * psi) + 100.0 * (psi > 0)
    te_zero_axis[0] = 0.0
    z0 = np.zeros_like(psi)
    kb_zero_ax = KineticBase(
        psi_kin=psi, ne=ne, te=te_zero_axis, ni=0.9 * ne, ti=te,
        sigma_ne=z0, sigma_te=z0, sigma_ni=z0, sigma_ti=z0,
        n_ls=0.4, t_ls=0.4,
    )
    target = float(np.trapezoid(ne * te_zero_axis + 0.9 * ne * te, psi))
    kd = sample_kinetics(
        kb_zero_ax, make_rng(3),
        lambda d: float(np.trapezoid(d.ne * d.te + d.ni * d.ti, psi)),
        target,
    )
    np.testing.assert_array_equal(kd.te, te_zero_axis)
    assert np.all(np.isfinite(kd.te))

    # All 14 PR-1/2/3 public symbols appear in bouquet.__all__
    expected_exports = {
        "draw_bounded_perturbation",
        "MillerGeometry",
        "build_tglf_inputs",
        "gyrobohm_flux_scales",
        "miller_from_baseline",
        "miller_from_geqdsk",
        "miller_from_imas_dd",
        "PedestalWindow",
        "PhysicsStiffnessResult",
        "bosch_hale_dt_reactivity",
        "compute_physics_length_scales",
        "detect_pedestal_window",
        "sample_stiff_inward_channel",
        "compute_power_balance_chi",
    }
    assert expected_exports.issubset(set(bouquet.__all__))
