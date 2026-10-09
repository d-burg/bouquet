"""imas_baseline="swb" sawtooth q reset (GenerationConfig.swb_saw_q): config
guards, the jphi_fixed / jphi_saw split, the kwargs SWB receives, and the
current accounting j_phi = j_ind + j_BS + (fixed - j_sawteeth) + j_saw.
Fast suite: SWB is mocked."""

import sys
import types

import numpy as np
import pytest

from bouquet import coords
from bouquet import run as run_mod  # noqa: F401
from bouquet import swb as swb_mod
from bouquet import swb_draws as SD
from bouquet.config import (_BOOTSTRAP_RESERVED, BouquetConfig, GenerationConfig,
                            swb_config_problems)
from bouquet.run import Bouquet

from _swb import SWB_ARGS, fake_oft, swb_cfg

SAW_ARGS = {"jphi_saw", "saw_q_s", "saw_dq", "saw_tol", "saw_ramp", "saw_rule"}
_cfg = swb_cfg


@pytest.fixture
def saw_oft(monkeypatch):
    fake_oft(monkeypatch, SWB_ARGS | {"jphi_saw"})


# ---- config -----------------------------------------------------------------
class TestConfig:
    def test_defaults_off(self):
        gc = GenerationConfig()
        assert gc.swb_saw_q is None
        assert (gc.swb_saw_dq, gc.swb_saw_tol, gc.swb_saw_ramp, gc.swb_saw_rule) \
            == (0.03, 1e-4, 0.01, "local")

    @pytest.mark.parametrize("kw", [dict(swb_saw_q=0.0), dict(swb_saw_q=-1.0),
                                    dict(swb_saw_rule=1), dict(swb_saw_rule="outer"),
                                    dict(swb_saw_dq=0.0), dict(swb_saw_dq=-0.03),
                                    dict(swb_saw_tol=0.0), dict(swb_saw_ramp=-0.01)])
    def test_bad_values_refused(self, kw):
        with pytest.raises(ValueError, match=next(iter(kw))):
            GenerationConfig(imas_baseline="swb", **kw)

    @pytest.mark.parametrize("rule", ["fuse", "local"])
    def test_rules_accepted(self, rule):
        assert GenerationConfig(imas_baseline="swb", swb_saw_rule=rule).swb_saw_rule == rule

    @pytest.mark.parametrize("key", sorted(SAW_ARGS))
    def test_saw_kwargs_reserved(self, key):
        with pytest.raises(ValueError, match="passed explicitly"):
            GenerationConfig(imas_baseline="swb", bootstrap_kwargs={key: 1.0})

    def test_saw_relax_not_reserved(self):
        assert "saw_relax" not in (_BOOTSTRAP_RESERVED
                                   | GenerationConfig._SAW_RESERVED)

    def test_problems(self, saw_oft, monkeypatch):
        assert swb_config_problems(_cfg(swb_saw_q=1.025)) == []
        monkeypatch.setattr(coords, "_swb_params",
                            lambda: frozenset({"x", "jphi_fixed", "p_fixed"}))
        assert swb_config_problems(_cfg()) == []          # saw off: not needed
        assert any("jphi_saw" in m for m in swb_config_problems(_cfg(swb_saw_q=1.025)))

    def test_roundtrip(self):
        cfg = _cfg(swb_saw_q=1.025, swb_saw_rule="fuse")
        g2 = BouquetConfig.from_json(cfg.to_json()).generation
        assert (g2.swb_saw_q, g2.swb_saw_rule) == (1.025, "fuse")

    def test_closure_path_refuses_saw(self):
        ns = types.SimpleNamespace(
            config=types.SimpleNamespace(
                source=types.SimpleNamespace(),
                generation=GenerationConfig(swb_saw_q=1.025,
                                            reconstruction_engine="legacy")),
            _resolve_engine_defaults=lambda: None,
            _check_jbs_loop_workflow=lambda gc: None,
            _check_structured_mse_reachable=lambda cfg: None)
        with pytest.raises(ValueError, match="swb_saw_q"):
            Bouquet.prepare_baseline(ns)


# ---- swb_source_seed --------------------------------------------------------
def _profiles(n=33):
    x = np.linspace(0.0, 1.0, n)
    j_ind = 1e6 * (1.0 - x ** 1.5) ** 1.5
    j_saw = 2e4 * (np.exp(-((x - 0.05) / 0.05) ** 2) - np.exp(-((x - 0.2) / 0.05) ** 2))
    j_fix = 3e4 * np.exp(-((x - 0.4) / 0.2) ** 2) + j_saw
    return x, j_ind, j_fix, j_saw


class TestSourceSeed:
    def test_off_unchanged(self, saw_oft):
        x, j_ind, j_fix, _ = _profiles()
        out = coords.swb_source_seed(x, j_ind, j_fix)
        assert len(out) == 2
        assert np.array_equal(out[0], j_ind) and np.array_equal(out[1], j_fix)

    @pytest.mark.parametrize("params", [{"x", "jphi_fixed", "jphi_saw"},
                                        {"jphi_fixed", "jphi_saw"}])  # uniform grid
    def test_split_sums_to_old_fixed(self, monkeypatch, params):
        monkeypatch.setattr(coords, "_swb_params", lambda: frozenset(params))
        x, j_ind, j_fix, j_saw = _profiles()
        x = x ** 1.3                                       # non-uniform run grid
        seed0, fix0 = coords.swb_source_seed(x, j_ind, j_fix)
        seed, fix, saw = coords.swb_source_seed(x, j_ind, j_fix, j_saw)
        assert np.array_equal(seed, seed0)
        assert np.array_equal(saw, np.interp(coords.swb_grid(x), x, j_saw))
        np.testing.assert_allclose(fix + saw, fix0, rtol=0,
                                   atol=1e-12 * np.max(np.abs(fix0)))

    def test_old_toolkit_refused(self, monkeypatch):
        monkeypatch.setattr(coords, "_swb_params", lambda: frozenset({"x", "jphi_fixed"}))
        x, j_ind, j_fix, j_saw = _profiles()
        with pytest.raises(RuntimeError, match="jphi_saw"):
            coords.swb_source_seed(x, j_ind, j_fix, j_saw)


# ---- mocked SWB: kwargs, accounting, sigma=0 -------------------------------
class _GS:
    coil_sets = ["F1A", "F6A"]

    def __init__(self):
        self.ip = 1.0e6

    def set_isoflux(self, *a, **k): pass
    def set_coil_reg(self, *a, **k): pass
    def init_psi(self, *a, **k): pass
    def get_globals(self): return [self.ip]
    def get_coil_currents(self): return {"F1A": 1.0e3, "F6A": -2.0e3}, None
    def get_stats(self, **k): return {"l_i": 0.72}
    def coil_reg_term(self, coils, target=0.0, weight=1.0): return (coils, target, weight)


def _dj(x):
    """An Ip-neutral-looking reset current (shape only matters here)."""
    return 5e4 * (np.exp(-((x - 0.1) / 0.06) ** 2) - 0.5 * np.exp(-((x - 0.25) / 0.06) ** 2))


@pytest.fixture
def fake_swb(monkeypatch, saw_oft):
    """solve_with_bootstrap following OFT's total = a*j_ind + j_BS + jphi_fixed
    + j_saw; like the saw toolkit it returns the saw keys even when off."""
    calls = []

    def swb(mygs, ne, te, ni, ti, zeff, ip, inductive_jphi, **kw):
        calls.append(kw)
        x = kw["x"]
        j_ind = 1.0123 * np.asarray(inductive_jphi)
        j_bs = 1e5 * (1.0 - x) ** 2
        on = bool(kw.get("saw_q_s"))
        j_in = np.zeros_like(x) if kw.get("jphi_saw") is None else kw["jphi_saw"]
        j_saw = j_in + (_dj(x) if on else 0.0)
        return {"j_inductive": j_ind, "isolated_j_BS": j_bs,
                "total_j_phi": j_ind + j_bs + kw["jphi_fixed"] + j_saw,
                "j_saw": j_saw, "saw_rho_m": 0.31 if on else 0.0,
                "saw_rho_out": 0.18 if on else 0.0, "saw_n_dips": 1 if on else 0}

    mod = types.ModuleType("OpenFUSIONToolkit.TokaMaker.bootstrap")
    mod.solve_with_bootstrap = swb
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", types.ModuleType("OpenFUSIONToolkit"))
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker",
                        types.ModuleType("OpenFUSIONToolkit.TokaMaker"))
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.bootstrap", mod)
    monkeypatch.setattr(swb_mod, "_shape_from_boundary", lambda b: ())
    import bouquet.utils as U
    monkeypatch.setattr(U, "safe_trace_surf", lambda mygs, psi: np.zeros((4, 2)))
    return calls


_METHODS = ("_swb_source_split", "_swb_saw_kwargs", "_swb_solve", "_swb_state",
            "_swb_imas_baseline", "_verify_sigma0_swb", "_zeff_eq")


def _run(**gen):
    x, j_ind, _, j_saw = _profiles()
    j_nbi = 3e4 * np.exp(-((x - 0.4) / 0.2) ** 2)
    j_rf = 1e4 * np.exp(-((x - 0.5) / 0.1) ** 2)
    j_fus = 2e3 * (1.0 - x ** 2)
    j_bs = 8e4 * (1.0 - x) ** 2
    bl = types.SimpleNamespace(
        psi_N=x, j_inductive=j_ind, j_BS=j_bs, j_NBI=j_nbi, j_RF=j_rf,
        j_other=j_fus + j_saw, j_sawteeth=j_saw, Ip_target=1.0e6, li_metrics={},
        j_phi=j_ind + j_bs + j_nbi + j_rf + j_fus + j_saw, coord="psi_n",
        psi_N_kinetic=x, ne=1e19 + 0 * x, te=1e3 + 0 * x, ni=1e19 + 0 * x,
        ti=1e3 + 0 * x, Zeff=1.5 + 0 * x, p_fast=None)
    gen.setdefault("imas_baseline", "swb")
    ns = types.SimpleNamespace(
        baseline=bl, mygs=_GS(), _iso=(None, None), _boundary_RZ=None,
        config=types.SimpleNamespace(generation=GenerationConfig(**gen)),
        _reset_solver_state=lambda: None, _seed_coil_init=lambda m: None,
        _swb_baseline_kinetics=lambda: dict(
            ne=1e19 + 0 * x, te=1e3 + 0 * x, ni=1e19 + 0 * x, ti=1e3 + 0 * x,
            Zeff=1.5 + 0 * x, p_fixed=0 * x),
        _finish_imas_baseline=lambda nl, **k: None)
    for m in _METHODS:
        setattr(ns, m, types.MethodType(getattr(Bouquet, m), ns))
    return ns


class TestMockedSWB:
    def test_kwargs_off(self, fake_swb):
        ns = _run()
        jfix_old = ns.baseline.j_phi - ns.baseline.j_inductive - ns.baseline.j_BS
        ns._swb_imas_baseline()
        assert len(fake_swb) == 2                          # solve A, solve B
        for kw in fake_swb:
            assert not (SAW_ARGS & set(kw))
            assert np.array_equal(kw["jphi_fixed"], jfix_old)
        bl = ns.baseline
        assert bl.j_saw is None and bl.swb_jphi_saw is None
        assert not any(k.startswith("saw_") for k in bl.ip_closure)
        assert not (set(swb_mod._SWB_SAW_KEYS) & set(bl.swb_baseline))  # stripped though returned

    def test_kwargs_on(self, fake_swb):
        ns = _run(swb_saw_q=1.025, swb_saw_dq=0.04, swb_saw_tol=2e-4,
                  swb_saw_ramp=0.0, swb_saw_rule="fuse")
        ns._swb_imas_baseline()
        bl = ns.baseline
        for kw in fake_swb:
            assert {k: kw[k] for k in SAW_ARGS - {"jphi_saw"}} == dict(
                saw_q_s=1.025, saw_dq=0.04, saw_tol=2e-4, saw_ramp=0.0, saw_rule=1)
            assert np.array_equal(kw["jphi_saw"], bl.swb_jphi_saw)
            assert kw["jphi_fixed"] is bl.swb_jphi_fixed
        np.testing.assert_array_equal(bl.swb_jphi_saw, bl.j_sawteeth)
        assert bl.ip_closure["saw_rho_m"] == 0.31 and bl.ip_closure["saw_n_dips"] == 1
        assert bl.ip_closure["saw_rho_out"] == 0.18 == bl.swb_baseline["saw_rho_out"]
        assert bl.ip_closure["saw_map_warn"] is False      # a reset: nothing to check

    @pytest.mark.parametrize("saw_q", [None, 1.025])
    def test_accounting(self, fake_swb, saw_q):
        ns = _run(swb_saw_q=saw_q)
        bl = ns.baseline
        jfix_old = bl.j_phi - bl.j_inductive - bl.j_BS
        ns._swb_imas_baseline()
        pk = np.max(np.abs(bl.j_phi))
        if saw_q is None:
            assert np.array_equal(bl.swb_jphi_fixed, jfix_old)
            other = bl.j_other
        else:
            np.testing.assert_allclose(bl.swb_jphi_fixed + bl.swb_jphi_saw, jfix_old,
                                       rtol=0, atol=1e-12 * pk)
            np.testing.assert_allclose(bl.j_saw, bl.j_sawteeth + _dj(bl.psi_N), rtol=1e-14)
            other = bl.j_other - bl.j_sawteeth + bl.j_saw
        np.testing.assert_allclose(
            bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + other, bl.j_phi,
            rtol=0, atol=1e-12 * pk)

    @pytest.mark.parametrize("saw_q", [None, 1.025])
    def test_sigma0_repeat(self, fake_swb, saw_q):
        ns = _run(swb_saw_q=saw_q)
        ns._swb_imas_baseline()
        out = ns._verify_sigma0_swb()
        assert out["passed"]
        assert ("j_saw" in out["swb_dev"]) == (saw_q is not None)
        assert ("saw_rho_m" in out["swb_dev"]) == (saw_q is not None)
        assert ("saw_rho_out" in out["swb_dev"]) == (saw_q is not None)
        # the sigma=0 draw is solve B: strong reg toward A's coils, same saw args
        assert set(fake_swb[-1]) == set(fake_swb[1])


def _draw(res_extra):
    x = np.linspace(0.0, 1.0, 17)
    prof = 1.0 - x ** 2 + 0.1
    seed = 1e6 * (1.0 - x ** 1.5) ** 1.5

    def recipe(kin, j_seed):
        j_ind = 1.01 * j_seed
        return {"j_inductive": j_ind, "isolated_j_BS": 0 * x,
                "total_j_phi": j_ind, **res_extra}
    return SD.swb_draw(None, x, None, prof, prof, prof, prof, None, None, None,
                       None, None, 0.5, 0.4, 0.25, 1.5 + 0 * x, seed, recipe,
                       coord="psi_n", rng=0, p_thresh=0.05)[6]


def test_swb_draw_records_j_saw():
    d = _draw({"j_saw": np.ones(17), "saw_rho_m": 0.3, "saw_rho_out": 0.2,
               "saw_n_dips": 2, "saw_map_warn": False})
    assert np.array_equal(d["j_saw"], np.ones(17))
    assert (d["saw_rho_m"], d["saw_rho_out"], d["saw_n_dips"]) == (0.3, 0.2, 2)
    assert d["saw_map_warn"] is False
    assert not ({"j_saw", "saw_rho_m", "saw_rho_out", "saw_n_dips", "saw_map_warn"}
                & set(_draw({})))


def test_saw_names_are_real_toolkit_arguments():
    """Against an installed saw toolkit: the reserved names exist (no typo)."""
    from bouquet.config import _bootstrap_kwarg_names
    known = _bootstrap_kwarg_names()
    if known is None or "jphi_saw" not in known:
        pytest.skip("needs an OpenFUSIONToolkit with jphi_saw")
    assert GenerationConfig._SAW_RESERVED <= known and "saw_relax" in known


# ---- review fixes: phi_n, taper identity, sigma=0 mismatch, jphi_saw map check ----
def _swap_swb(fn):
    """Replace the fake toolkit's solve_with_bootstrap (installed by fake_swb)."""
    sys.modules["OpenFUSIONToolkit.TokaMaker.bootstrap"].solve_with_bootstrap = fn


def test_phi_n_saw_and_fixed_share_grid(fake_swb):
    ns = _run(swb_saw_q=1.025)
    bl = ns.baseline
    bl.coord = "phi_n"
    ns._swb_imas_baseline()
    for kw in fake_swb:
        assert kw["coord"] == "phi_n" and np.array_equal(kw["x"], bl.psi_N)
        assert kw["jphi_saw"].shape == kw["jphi_fixed"].shape == bl.psi_N.shape
        np.testing.assert_array_equal(kw["jphi_saw"], bl.swb_jphi_saw)
    np.testing.assert_array_equal(bl.swb_jphi_saw, bl.j_sawteeth)


def _taper(x, x0=0.9):
    return np.where(x > x0, np.cos(0.5 * np.pi * (x - x0) / (1.0 - x0)) ** 2, 1.0)


def test_saw_taper_identity(fake_swb):
    """OFT tapers every channel at the edge: j_fixed and j_saw come back tapered."""
    def swb(mygs, ne, te, ni, ti, zeff, ip, inductive_jphi, **kw):
        fake_swb.append(kw)
        x = kw["x"]
        f = _taper(x)
        j_ind = f * 1.0123 * np.asarray(inductive_jphi)
        j_bs = f * 1e5 * (1.0 - x) ** 2
        j_fix = f * kw["jphi_fixed"]
        j_saw = f * kw["jphi_saw"] + _dj(x)
        return {"j_inductive": j_ind, "isolated_j_BS": j_bs, "j_fixed": j_fix,
                "total_j_phi": j_ind + j_bs + j_fix + j_saw, "j_saw": j_saw,
                "saw_rho_m": 0.31, "saw_rho_out": 0.18, "saw_n_dips": 1}
    _swap_swb(swb)
    ns = _run(swb_saw_q=1.025)
    bl = ns.baseline
    ns._swb_imas_baseline()
    f = _taper(bl.psi_N)
    assert np.any(f < 1.0)
    np.testing.assert_allclose(bl.j_saw, f * bl.swb_jphi_saw + _dj(bl.psi_N), rtol=1e-14)
    pk = np.max(np.abs(bl.j_phi))
    np.testing.assert_allclose(
        bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + (bl.j_other - bl.j_sawteeth)
        + bl.j_saw, bl.j_phi, rtol=0, atol=1e-12 * pk)


@pytest.mark.parametrize("key", ["j_saw", "saw_rho_m", "saw_rho_out"])
def test_sigma0_fails_on_saw_mismatch(fake_swb, key):
    swb0 = sys.modules["OpenFUSIONToolkit.TokaMaker.bootstrap"].solve_with_bootstrap

    def swb(*a, **kw):
        res = swb0(*a, **kw)
        if len(fake_swb) > 2:                   # the sigma=0 draw
            res[key] = res[key] + (1.0 if key == "j_saw" else 1e-3)
        return res
    _swap_swb(swb)
    ns = _run(swb_saw_q=1.025)
    ns._swb_imas_baseline()
    out = ns._verify_sigma0_swb()
    assert not out["passed"] and out["swb_dev"][key] > 0.0
    assert all(v == 0.0 for k, v in out["swb_dev"].items() if k != key)


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_no_reset_jphi_saw_map_check(fake_swb, capsys, scale):
    """saw_n_dips 0: j_saw must be jphi_saw; a mis-mapped one warns, never raises."""
    def swb(mygs, ne, te, ni, ti, zeff, ip, inductive_jphi, **kw):
        fake_swb.append(kw)
        x = kw["x"]
        j_ind = 1.0123 * np.asarray(inductive_jphi)
        j_bs = 1e5 * (1.0 - x) ** 2
        j_saw = scale * kw["jphi_saw"]
        return {"j_inductive": j_ind, "isolated_j_BS": j_bs,
                "total_j_phi": j_ind + j_bs + kw["jphi_fixed"] + j_saw, "j_saw": j_saw,
                "saw_rho_m": 0.0, "saw_rho_out": 0.0, "saw_n_dips": 0}
    _swap_swb(swb)
    ns = _run(swb_saw_q=1.025)
    ns._swb_imas_baseline()
    ic = ns.baseline.ip_closure
    assert ic["saw_map_warn"] is (scale != 1.0)
    assert (ic["saw_map_dev"] > swb_mod.SWB_SAW_MAP_TOL) is (scale != 1.0)
    assert ("mis-map jphi_saw" in capsys.readouterr().out) is (scale != 1.0)


def test_saw_map_check_taper():
    x = np.linspace(0.0, 1.0, 33)
    f = _taper(x)
    jsaw_in = 3e4 * (1.0 - 0.5 * x)               # finite at the edge: the taper shows
    jf_in = 2e4 + 0 * x
    jf_in[-1] = 0.0                              # factor unknown on the last node
    res = {"j_saw": f * jsaw_in, "j_fixed": f * jf_in, "saw_n_dips": 0,
           "total_j_phi": 1e6 * (1.0 - x ** 2) + 1e5}
    dev, warn = swb_mod._swb_saw_map_check(res, jsaw_in, jf_in)
    assert dev < 1e-14 and not warn
    assert swb_mod._swb_saw_map_check(res, jsaw_in)[1]          # taper ignored: trips
    res["j_saw"] = jsaw_in
    assert swb_mod._swb_saw_map_check(res, jsaw_in, jf_in)[1]
    assert swb_mod._swb_saw_map_check(dict(res, saw_n_dips=2), jsaw_in, jf_in) == (None, False)
