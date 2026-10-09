"""imas_baseline="swb": config guards, the shared recipe inputs, and the sigma=0
identity between the baseline (solve B) and a draw. Fast suite: no solver."""

import inspect
import types

import numpy as np
import pytest

from bouquet import coords
from bouquet.config import (GenerationConfig, ReconstructionSource, SolverConfig,
                            swb_config_problems)
from bouquet.run import Bouquet
from bouquet import TokaMaker_interface as TI
from bouquet import swb_draws as SD
from bouquet.utils import pchip_interp

from _swb import fake_oft, swb_cfg

_cfg = swb_cfg


@pytest.fixture
def swb_oft(monkeypatch):
    fake_oft(monkeypatch)


class TestConfig:
    def test_default_is_closure(self):
        assert GenerationConfig().imas_baseline == "closure"

    def test_unknown_value_refused(self):
        with pytest.raises(ValueError, match="imas_baseline"):
            GenerationConfig(imas_baseline="hybrid")

    def test_p_fixed_reserved(self):
        with pytest.raises(ValueError, match="p_fixed"):
            GenerationConfig(imas_baseline="swb", bootstrap_kwargs={"p_fixed": 1.0})

    def test_production_settings_pass(self, swb_oft):
        assert swb_config_problems(_cfg()) == []

    @pytest.mark.parametrize("kw, word", [
        (dict(kinetic_source="fuse"), "kinetic_source"),
        (dict(recalculate_j_BS=False), "recalculate_j_BS"),
        (dict(imas_corrective_jphi=True), "imas_corrective_jphi"),
        (dict(jbs_delta_mode=True), "jbs_delta_mode"),
        (dict(anchor_pressure_to_equilibrium=True), "anchor_pressure_to_equilibrium"),
        (dict(closure_channel="sawtooth_bootstrap"), "closure_channel"),
        (dict(closure_channel="ohmic"), "closure_channel"),
        (dict(coil_drift_hard_factor=20.0), "coil_drift_hard_factor"),
        (dict(edge_pprime_pin=False), "edge_pprime_pin"),
        (dict(separatrix_pressure="legacy"), "separatrix_pressure"),
        (dict(bootstrap_kwargs={"taper_edge_jBS": True}), "swb_edge_taper_psi0 instead"),
        (dict(swb_edge_taper_psi0=1.0), "must be in (0, 1)"),
    ])
    def test_refused_by_name(self, swb_oft, kw, word):
        assert any(word in m for m in swb_config_problems(_cfg(**kw)))

    def test_threads_and_source_refused(self, swb_oft):
        assert any("nthreads" in m for m in swb_config_problems(
            _cfg(solver=SolverConfig(mesh_path="m.h5", nthreads=2))))
        rec = ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf", time=1.0)
        assert any("ImasSource" in m for m in swb_config_problems(_cfg(source=rec)))

    def test_env_modes_refused(self, swb_oft, monkeypatch):
        monkeypatch.setenv("DIFF_BS", "1")
        assert any("DIFF_BS" in m for m in swb_config_problems(_cfg()))

    def test_old_toolkit_refused(self, monkeypatch):
        monkeypatch.setattr(coords, "_swb_params", lambda: frozenset({"x"}))
        assert any("p_fixed" in m for m in swb_config_problems(_cfg()))


class _GS:
    def __init__(self, sets):
        self.coil_sets = list(sets)

    def coil_reg_term(self, coils, target=0.0, weight=1.0):
        return {"coils": dict(coils), "target": target, "weight": weight}


def test_strong_coil_reg_is_the_draw_reg():
    """The helper builds generate_bouquet's historical Step-2 terms."""
    rt = TI.strong_coil_reg(_GS(["F1A", "F6A"]), {"F1A": 5.0, "X": 9.0}, 1e4, 1.0)
    assert rt == [{"coils": {"F1A": 1.0}, "target": 5.0, "weight": 1e4},
                  {"coils": {"F6A": 1.0}, "target": 0.0, "weight": 1e4},
                  {"coils": {"#VSC": 1.0}, "target": 0.0, "weight": 1.0}]
    assert "strong_coil_reg(mygs, _initial_coils" in inspect.getsource(TI.generate_bouquet)


def test_kinetic_draw_is_shared():
    # one sampler (bouquet.kinetic_sampler) for the legacy and swb draws
    assert "sample_kinetics(" in inspect.getsource(TI.perturb_kinetic_equilibrium)
    assert "sample_kinetics(" in inspect.getsource(TI.draw_kinetics)
    assert "draw_kinetics(" in inspect.getsource(SD.swb_draw)


# ---- the sigma=0 identity -------------------------------------------------
def _baseline():
    pk = np.linspace(0.0, 1.0, 41)          # kinetic grid != run grid
    psi = np.linspace(0.0, 1.0, 33)
    sh = (1.0 - pk ** 2)
    return types.SimpleNamespace(
        psi_N=psi, psi_N_kinetic=pk,
        ne=4e19 * sh + 1e18, te=3e3 * sh + 50.0, ni=3.5e19 * sh + 1e18,
        ti=3.5e3 * sh + 50.0, Zeff=1.8 + 0.4 * pk, p_fast=2e4 * sh ** 2,
        z_fast=2e18 * sh ** 2, Z_imp=6.0,
        swb_seed_profile=1e6 * (1.0 - psi ** 1.5) ** 1.5)


def _ns(bl):
    ns = types.SimpleNamespace(baseline=bl)
    ns._zeff_eq = lambda: Bouquet._zeff_eq(ns)
    return ns


def _draw(bl, recipe, sig=0.0, sig_j=0.0, rng=None):
    pk = bl.psi_N_kinetic
    z = lambda a: sig * np.asarray(a)
    return SD.swb_draw(
        None, bl.psi_N, None, bl.ne, bl.te, bl.ni, bl.ti,
        z(bl.ne), z(bl.te), z(bl.ni), z(bl.ti), sig_j * bl.swb_seed_profile,
        0.5, 0.4, 0.25, Bouquet._zeff_eq(_ns(bl)), bl.swb_seed_profile, recipe,
        coord="psi_n", rng=rng, p_thresh=0.05, psi_N_kinetic=pk,
        p_fast=bl.p_fast, z_fast=bl.z_fast, Z_imp=bl.Z_imp)


def _fake_recipe(log):
    def recipe(kin, j_seed):
        log.append((kin, j_seed))
        j_ind = 1.04 * np.asarray(j_seed)
        j_bs = 1e5 * np.ones_like(j_ind)
        return {"j_inductive": j_ind, "isolated_j_BS": j_bs,
                "total_j_phi": j_ind + j_bs}
    return recipe


def test_sigma0_draw_inputs_are_the_baseline_inputs():
    """A sigma=0 draw hands the recipe bit-identical inputs to solve B's."""
    bl = _baseline()
    ref = Bouquet._swb_baseline_kinetics(_ns(bl))
    log = []
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    out = _draw(bl, _fake_recipe(log), rng=rng)
    kin, j_seed = log[0]
    assert j_seed is bl.swb_seed_profile
    for k in ("ne", "te", "ni", "ti", "Zeff", "p_fixed"):
        assert np.array_equal(kin[k], ref[k]), k
    assert rng.bit_generator.state == state          # no draws consumed
    assert out[6]["swb_alpha"] == pytest.approx(1.04, rel=1e-12)
    assert np.array_equal(out[5], out[6]["j_inductive"] + out[6]["j_BS"])


def test_p_fixed_is_fast_plus_carbon():
    bl = _baseline()
    ref = Bouquet._swb_baseline_kinetics(_ns(bl))
    k2e = lambda a: pchip_interp(bl.psi_N_kinetic, a, bl.psi_N)
    from bouquet.physics import impurity_pressure
    exp = k2e(bl.p_fast) + impurity_pressure(
        np.maximum(ref["ne"] - k2e(bl.z_fast), 0.0), ref["ni"], ref["ti"], 6.0)
    assert np.allclose(ref["p_fixed"], exp, rtol=1e-14, atol=0.0)
    assert np.all(ref["p_fixed"] >= 0.0)               # OFT refuses p_fixed < 0


def test_jind_redraw_nonnegative_and_seeded():
    bl = _baseline()
    seeds = []
    for _ in range(2):
        log = []
        _draw(bl, _fake_recipe(log), sig_j=0.05, rng=np.random.default_rng(3))
        seeds.append(log[0][1])
    assert not np.array_equal(seeds[0], bl.swb_seed_profile)
    assert np.all(seeds[0] >= 0.0)
    assert np.array_equal(seeds[0], seeds[1])


def test_sigma0_check_dispatches_to_the_swb_draw_check(monkeypatch):
    ns = types.SimpleNamespace(
        baseline=object(), mygs=object(),
        config=types.SimpleNamespace(generation=types.SimpleNamespace(
            single_profile_jphi=False, imas_baseline="swb")),
        _verify_sigma0_swb=lambda: {"passed": True, "swb": True})
    assert Bouquet.verify_sigma0_consistency(ns)["swb"] is True


def test_stamp_group_attrs(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bouquet.utils import stamp_group_attrs
    hdr = str(tmp_path / "a")
    with h5py.File(hdr + ".h5", "w") as hf:
        hf.create_group("scan/k/_baseline")
        hf.create_group("scan/k/3")
    stamp_group_attrs(hdr, "k", None, {"swb_alpha": 1.03, "skip": None,
                                       "coil_reg_target": {"F1A": 2.0, "F6A": -1.0}})
    stamp_group_attrs(hdr, "k", 3, {"swb_alpha": 0.99})
    with h5py.File(hdr + ".h5", "r") as hf:
        b = hf["scan/k/_baseline"].attrs
        assert b["swb_alpha"] == 1.03 and "skip" not in b
        assert [n.decode() for n in b["coil_reg_target_names"]] == ["F1A", "F6A"]
        assert list(b["coil_reg_target_values"]) == [2.0, -1.0]
        assert hf["scan/k/3"].attrs["swb_alpha"] == 0.99


@pytest.mark.parametrize("taper, saw", [(False, False), (True, False), (True, True)])
def test_edge_taper_keeps_the_channel_split(monkeypatch, swb_oft, taper, saw):
    """taper_edge_jBS tapers SWB's fixed current too: the baseline channels must
    follow, so j_phi = j_inductive + j_BS + j_NBI + j_RF + j_other holds (with the
    saw: j_other - j_sawteeth + j_saw)."""
    x = coords.swb_grid(np.linspace(0.0, 1.0, 33))
    j_ind, j_bs = 1e6 * (1 - x ** 2) + 2e4, 5e4 * x ** 4
    j_nbi, j_rf, j_oth = 2e5 * (1 - x) + 1e3, 4e4 * np.exp(-((x - .3) / .1) ** 2), 3e5 * (x < .3)
    j_st = 1e5 * (1 - x) ** 2 if saw else None     # sawteeth share of j_other
    if saw:
        j_oth = j_oth + j_st
    if saw:
        monkeypatch.setattr(coords, "_swb_params",
                            lambda: frozenset({"x", "jphi_fixed", "p_fixed", "jphi_saw"}))
    fac = np.clip((1.0 - x) / 0.05, 0.0, 1.0) if taper else np.ones_like(x)
    bl = types.SimpleNamespace(psi_N=x, j_inductive=j_ind, j_BS=j_bs, j_NBI=j_nbi, j_RF=j_rf,
                               j_other=j_oth, j_sawteeth=j_st,
                               j_phi=j_ind + j_bs + j_nbi + j_rf + j_oth, li_metrics={})
    gen = types.SimpleNamespace(bootstrap_kwargs={}, swb_saw_q=1.1 if saw else None,
                                swb_edge_taper_psi0=0.999 if taper else None)
    dj_saw = 1e4 * np.sin(np.pi * x / 0.4) * (x < 0.4)

    def solve(kin, seed, coil_reg_target=None):
        jf = np.asarray(bl.swb_jphi_fixed) * fac
        ji = 1.02 * np.asarray(seed) * fac
        res = dict(j_inductive=ji, isolated_j_BS=j_bs * fac, j_fixed=jf,
                   total_j_phi=ji + j_bs * fac + jf)
        if saw:
            res["j_saw"] = np.asarray(bl.swb_jphi_saw) * fac + dj_saw
            res["total_j_phi"] = res["total_j_phi"] + res["j_saw"]
        return res

    def state(res, seed, psi_pad=1e-3):
        st = dict(alpha=1.02, coils={"F1A": 1.0}, lcfs=None, li_3=0.9, Ip=1e6,
                  j_inductive=res["j_inductive"], j_BS=res["isolated_j_BS"],
                  j_phi=res["total_j_phi"], j_fixed=res["j_fixed"])
        if saw:
            st.update(j_saw=res["j_saw"], saw_rho_m=0.4, saw_n_dips=1)
        return st
    ns = types.SimpleNamespace(baseline=bl, config=types.SimpleNamespace(generation=gen),
                               _swb_baseline_kinetics=lambda: {
                                   k: np.ones_like(x) for k in
                                   ("ne", "te", "ni", "ti", "p_fixed")},
                               _swb_solve=solve,
                               _swb_state=state, _finish_imas_baseline=lambda its, **k: None)
    ns._swb_source_split = lambda psi_N: Bouquet._swb_source_split(ns, psi_N)
    Bouquet._swb_imas_baseline(ns)
    total = bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + bl.j_other
    if saw:
        total = total - bl.j_sawteeth + bl.j_saw
        assert np.allclose(bl.j_sawteeth, j_st * fac, rtol=1e-12, atol=1e-6)
    assert np.max(np.abs(total - bl.j_phi)) <= 1e-9 * np.max(np.abs(bl.j_phi))
    assert np.allclose(bl.swb_jphi_fixed + (j_st if saw else 0.0), j_nbi + j_rf + j_oth,
                       rtol=1e-12, atol=1e-6)  # draws: untapered
    assert np.allclose(bl.j_NBI, j_nbi * fac, rtol=1e-12, atol=1e-6)
    assert np.array_equal(bl.j_other[fac == 1.0], j_oth[fac == 1.0])   # untapered: untouched


def test_edge_taper_is_the_swb_default(swb_oft):
    from bouquet.config import swb_bootstrap_kwargs
    gc = _cfg().generation
    assert gc.swb_edge_taper_psi0 == 0.999
    gc.bootstrap_kwargs = {"diagnose_bs": True}
    assert swb_bootstrap_kwargs(gc) == {"diagnose_bs": True, "taper_edge_jBS": True,
                                        "taper_edge_psi0": 0.999}
    assert gc.bootstrap_kwargs == {"diagnose_bs": True}          # not mutated
    gc.swb_edge_taper_psi0 = None
    assert swb_bootstrap_kwargs(gc) == {"diagnose_bs": True}
