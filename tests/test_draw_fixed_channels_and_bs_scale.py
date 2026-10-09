"""PR #70 review: the fixed channels, the bootstrap multiplier and swb_seed.

* B1 -- the IMAS delivered state holds EVERY channel the draws hold fixed
  (``j_NBI + j_RF + j_other``) exactly once.  ``_deliver_imas_state`` summed
  NBI + RF only, so ``j_other`` landed in the stored inductive residual and
  every draw added it again; both sigma=0 guards were blind to it (the
  baseline-way check took the fixed current as the stored split's residual,
  the draw route did not pass ``j_other``).
* B2 -- the bootstrap multiplier reaches the loop draws: the composer is
  linear in its scale and takes no profile, so the multiplier rides in the
  scale (range re-centred on ``bs_scale``); SWB draws keep it after SWB.  The
  draws and both guards use ONE helper (``_draw_bootstrap_scaling``).
* B4 -- ``swb_seed`` defaults to None, resolved to the toolkit's capability;
  an explicit "source" the toolkit cannot honour is refused at the SWB call.
* B5 -- the two structured-MSE closure sites record ``bs_scale_profile``.

Toy Grad-Shafranov solver and mocks (tests/test_sigma0_identity_stages.py,
tests/test_sigma0_draw_route.py, tests/test_structured_mse.py,
tests/test_mse_jbs_stage_review_fixes.py).  Synthetic only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.config import GenerationConfig
from test_sigma0_identity_stages import (  # noqa: F401  (toy is a fixture)
    _EC, _IP, _N, _NE, _NI, _PAD, _TE, _TI, _W, _X, _ZEFF, _kin_redl, _li,
    _settings, _trap, toy)

_J_NBI = 8.0e4 * np.exp(-0.5 * ((_X - 0.3) / 0.15) ** 2)
#: an "other" channel (a fusion / sawteeth entry), 10 % of the core current
_J_OTHER = 1.6e5 * np.exp(-0.5 * ((_X - 0.15) / 0.08) ** 2)


def _reconstruct(mygs, j_fixed, bs=1.0, kappa_short=0.965, n_loop=80):
    """A toy reconstruction whose request carries ``bs x Redl(F)`` (the
    rescale mode's bootstrap) and the fixed current."""
    j_ind0 = 1.6e6 * (1.0 - _X ** 2) ** 1.5
    jbs = np.zeros(_N)
    mygs.set_targets(Ip=_IP)
    for _ in range(n_loop):
        mygs.set_profiles(ffp_prof={"y": j_ind0 + jbs + j_fixed})
        mygs.solve()
        new = bs * _kin_redl(mygs)
        done = np.max(np.abs(new - jbs)) < 1e-12 * np.max(np.abs(new))
        jbs = new
        if done:
            break
    req = j_ind0 + jbs + j_fixed
    mygs.set_profiles(ffp_prof={"y": req})
    mygs.solve()
    return kappa_short * _IP / _trap(_W * req) * req


def _bouquet(mygs, bs=1.0, j_other=_J_OTHER, **gen):
    from bouquet.run import Bouquet
    gc = GenerationConfig()
    for k, v in gen.items():
        setattr(gc, k, v)
    b = Bouquet.__new__(Bouquet)
    b.mygs = mygs
    b.config = SimpleNamespace(generation=gc)
    b.baseline = SimpleNamespace(
        psi_N=_X, coord="psi_n", jBS_diff=None, j_BS=None, bs_scale=bs,
        bs_scale_profile=None, j_NBI=_J_NBI, j_RF=None, j_other=j_other,
        jphi_diff=None, Ip_target=_IP, l_i_target=_li(mygs.achieved),
        edge_pressure=None)
    return b


def _deliver(b, mode, request):
    b._deliver_imas_state(dict(mode=mode, request=request), _NE, _TE, _NI,
                          _TI, _ZEFF, _PAD, lambda a: a)
    return b.baseline


def _draw(b, s, scale_jBS=None, profile="helper", route="ip_renorm"):
    """The draw generate() runs, at zero perturbation, with generate()'s own
    bootstrap scaling (unless overridden) and fixed channels."""
    from bouquet.sampling import make_rng
    from bouquet.TokaMaker_interface import (perturb_kinetic_equilibrium,
                                             sigma0_reference_scale)
    bl = b.baseline
    rng, prof, _ = b._draw_bootstrap_scaling()
    if scale_jBS is None:
        scale_jBS = sigma0_reference_scale(rng)
    if isinstance(profile, str):
        profile = prof
    z = np.zeros(_N)
    return perturb_kinetic_equilibrium(
        b.mygs, _X, _EC * (_NE * _TE + _NI * _TI), _NE, _TE, _NI, _TI,
        np.asarray(bl.j_phi, float), z, z, z, z, z, 0.5, 0.4, 0.25, _IP,
        float(bl.l_i_target), _ZEFF, _N,
        input_jinductive=np.asarray(bl.j_inductive, float),
        l_i_tolerance=0.05, psi_pad=_PAD, constrain_sawteeth=False,
        recalculate_j_BS=True, isolate_edge_jBS=False, scale_jBS=scale_jBS,
        jBS_scale_profile=profile, floor_j_BS=False, jBS_diff=bl.jBS_diff,
        perturb_jind_in_anchor=(route == "ip_renorm"),
        accept_anchor_inband=False, j_NBI=bl.j_NBI, j_RF=bl.j_RF,
        j_other=bl.j_other, max_proxy_draws=5, p_thresh=0.05,
        rng=make_rng(7), jbs_loop=s,
        jphi_request_offset=bl.jphi_request_offset)[6]


def _identity(mygs, F, ref_jbs, d, s):
    from bouquet.jbs_loop import profile_residuals
    spk = np.asarray(d["_jbs_ctx"]["spike_used"], float)
    c = profile_residuals(spk, ref_jbs, _W, _X, _IP)
    eq = float(np.max(np.abs(mygs.achieved - F.achieved))
               / np.max(np.abs(F.achieved)))
    ok = bool(d["jbs_loop"]["converged"] and c["r_j"] <= s["rtol_j"]
              and c["r_I"] <= s["rtol_Ip"]
              and abs(_li(mygs.achieved) - _li(F.achieved)) <= s["tol_li"])
    return dict(r_j=c["r_j"], eq=eq, ok=ok)


# ---------------------------------------------------------------------------
#  B1
# ---------------------------------------------------------------------------
def test_the_delivered_state_holds_j_other_once_and_the_draw_reproduces_it(
        toy):
    req = _reconstruct(toy, _J_NBI + _J_OTHER)
    F = toy.copy_eq()
    b = _bouquet(toy)
    bl = _deliver(b, "diff", req)
    # the stored split sums to the request with each fixed channel once
    np.testing.assert_allclose(
        bl.j_inductive + bl.j_BS + _J_NBI + _J_OTHER, bl.j_phi, rtol=0,
        atol=1e-9 * np.max(np.abs(bl.j_phi)))
    s = _settings()
    toy.replace_eq(source_eq=F)
    d = _draw(b, s)
    r = _identity(toy, F, bl.j_BS, d, s)
    assert r["ok"] and r["eq"] <= 1e-10, r


def test_the_draw_route_guard_hands_the_draw_j_other_and_the_draws_scale(
        monkeypatch):
    """_sigma0_draw_route replays perturb_kinetic_equilibrium with j_other
    and with exactly the draws' scale and profile (the helper generate()
    uses)."""
    import test_sigma0_draw_route as SR
    from bouquet.TokaMaker_interface import sigma0_reference_scale
    b, calls = SR._bouquet(monkeypatch, li_draw=0.8002)
    b.baseline.j_other = 1.0e4 * np.ones_like(SR._X)
    b.baseline.bs_scale = 0.77
    blk = SR._run(b)
    k = calls[0][1]
    np.testing.assert_array_equal(k["j_other"], b.baseline.j_other)
    rng, prof, rec = b._draw_bootstrap_scaling()
    assert k["scale_jBS"] == sigma0_reference_scale(rng)
    assert k["scale_jBS"] == pytest.approx(0.77)
    assert k["jBS_scale_profile"] is None and prof is None
    assert blk["bootstrap_scaling"] == rec
    assert blk["fixed_channels"] == ["j_NBI", "j_RF", "j_other"]


# ---------------------------------------------------------------------------
#  B2
# ---------------------------------------------------------------------------
def test_a_loop_draw_at_zero_sigma_reproduces_a_rescaled_bootstrap(toy):
    """jBS_baseline_mode='rescale' with bs_scale 0.77: the baseline is
    0.77 Redl(F); the loop draw, handed generate()'s scaling, reproduces it.
    Handed what #70 gave it (scale 1, the multiplier as an after-SWB profile
    the loop never reads) it composes 1.0 Redl and misses F."""
    m = 0.77
    req = _reconstruct(toy, _J_NBI, bs=m)
    F = toy.copy_eq()
    b = _bouquet(toy, bs=m, j_other=None)
    bl = _deliver(b, "rescale", req)
    np.testing.assert_allclose(bl.j_BS, m * _kin_redl(F), rtol=1e-12)
    s = _settings()
    rng, prof, rec = b._draw_bootstrap_scaling()
    assert prof is None and rng == pytest.approx((0.99 * m, 1.01 * m))
    assert rec["applied_as"].startswith("scale_jBS")
    toy.replace_eq(source_eq=F)
    r = _identity(toy, F, bl.j_BS, _draw(b, s), s)
    assert r["ok"] and r["eq"] <= 1e-10, r
    # guard the guard: #70's hand-over misses
    toy.replace_eq(source_eq=F)
    r70 = _identity(toy, F, bl.j_BS, _draw(b, s, scale_jBS=1.0,
                                            profile=np.full(_N, m)), s)
    assert not r70["ok"] and r70["r_j"] > 0.1, r70


def test_the_scaling_rule_per_draw_kind(monkeypatch):
    b = _bouquet(SimpleNamespace(achieved=np.ones(_N)), bs=0.8)
    # loop draws: the range re-centred, no profile
    rng, prof, _ = b._draw_bootstrap_scaling()
    assert prof is None and rng == pytest.approx((0.99 * 0.8, 1.01 * 0.8))
    # ... and with no range: exactly the multiplier
    b.config.generation.jBS_scale_range = None
    assert b._draw_bootstrap_scaling()[0] == (0.8, 0.8)
    # SWB draws (loop off): the configured range, the multiplier after SWB
    b.config.generation.jBS_scale_range = (0.99, 1.01)
    b.config.generation.jbs_self_consistent = False
    rng, prof, rec = b._draw_bootstrap_scaling()
    assert rng == (0.99, 1.01) and np.all(prof == 0.8)
    assert rec["applied_as"] == "jBS_scale_profile after SWB"
    # DIFF_BS takes precedence over the loop: SWB semantics
    b.config.generation.jbs_self_consistent = True
    monkeypatch.setenv("DIFF_BS", "1")
    assert np.all(b._draw_bootstrap_scaling()[1] == 0.8)
    monkeypatch.delenv("DIFF_BS")
    # a structured s_bs(psi) cannot reach the loop composer: refused
    b.baseline.bs_scale_profile = 0.9 + 0.2 * _X
    with pytest.raises(ValueError, match="non-uniform bootstrap multiplier"):
        b._draw_bootstrap_scaling()
    b.config.generation.jbs_self_consistent = False
    np.testing.assert_array_equal(b._draw_bootstrap_scaling()[1],
                                  0.9 + 0.2 * _X)
    # bs_scale 1: nothing to carry
    b.baseline.bs_scale_profile, b.baseline.bs_scale = None, 1.0
    assert b._draw_bootstrap_scaling()[1] is None


# ---------------------------------------------------------------------------
#  B4
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("req, cap, want", [
    (None, True, "source"), (None, False, "generic"),
    ("source", True, "source"), ("source", False, "source_unavailable"),
    ("generic", True, "generic"), ("generic", False, "generic")])
def test_swb_seed_resolves_to_the_toolkit_and_refuses_only_at_an_swb_call(
        monkeypatch, req, cap, want):
    import bouquet.coords as C
    monkeypatch.setattr(C, "_SWB_PARAMS", frozenset(
        {"x", "jphi_fixed"} if cap else {"x"}), raising=False)
    b = _bouquet(SimpleNamespace(achieved=np.ones(_N)))
    b.config.generation.swb_seed = req
    mode, rec = b._resolve_swb_seed()
    assert mode == want and rec["resolved"] == want
    assert rec["requested"] == req and rec["oft_jphi_fixed"] is cap
    b._swb_seed_record = rec
    b.baseline.swb_seed_profile = None
    if want == "source_unavailable":
        with pytest.raises(RuntimeError, match="needs an OpenFUSIONToolkit"):
            b._swb_inputs(None, _X, "psi_n")
    else:
        monkeypatch.setattr(C, "psi_at", lambda mygs, x, coord: x)
        seed, kw = b._swb_inputs(None, _X, "psi_n")
        assert kw == {} and seed.shape == _X.shape


def test_swb_seed_default_is_none_and_validated():
    assert GenerationConfig().swb_seed is None
    for v in (None, "source", "generic"):
        GenerationConfig(swb_seed=v)
    with pytest.raises(ValueError, match="swb_seed"):
        GenerationConfig(swb_seed="auto")


# ---------------------------------------------------------------------------
#  B5
# ---------------------------------------------------------------------------
def test_the_ohmic_structured_mse_stage_records_the_multiplier():
    """predictor -> MSE stage -> corrector: whatever the corrector does, the
    multiplier the draws apply is the one the baseline bootstrap carries."""
    from test_structured_mse import _run_stage
    gc = GenerationConfig(closure_channel="structured",
                          jBS_baseline_mode="ohmic")
    bl, state, _n, _cyl, _ch = _run_stage(gc)
    assert bl.ip_closure["structured_mse_status"] == "applied"
    np.testing.assert_allclose(
        bl.j_BS, np.asarray(bl.bs_scale_profile) * state["j_BS_swb"],
        rtol=1e-12)


def test_the_loop_mse_stage_records_the_multiplier(monkeypatch):
    import bouquet.utils as U
    import test_mse_jbs_stage_review_fixes as R
    t = R._converging(monkeypatch)
    s_bs = 0.9 + 0.2 * R._X
    close = U.close_ip_structured

    def _close(*a, **k):
        out = close(*a, **k)
        out["s_bs"] = s_bs.copy()
        return out
    monkeypatch.setattr(U, "close_ip_structured", _close)
    nl, cur, srec = t["run"]()
    assert srec["converged"], srec["stop_reason"]
    bl = t["bl"]
    np.testing.assert_array_equal(bl.bs_scale_profile, s_bs)
    np.testing.assert_allclose(bl.j_BS, s_bs * np.asarray(cur["j_BS_swb"]),
                               rtol=1e-12)


@pytest.mark.parametrize("channels, closes", [
    (("j_NBI", "j_RF", "j_other"), True),
    (("j_NBI", "j_RF"), False)])        # the delivery before the fix
def test_the_baseline_way_guard_holds_the_draws_fixed_current(
        toy, monkeypatch, channels, closes):
    """The baseline-way sigma=0 check composes j_inductive + bootstrap +
    the fixed current the DRAWS hold.  On a split stored without j_other in
    its fixed part it now fails (it took the split's own residual as the
    fixed current and passed) and the closure gap is the j_other share."""
    from bouquet.run import Bouquet
    req = _reconstruct(toy, _J_NBI + _J_OTHER)
    F = toy.copy_eq()
    b = _bouquet(toy)
    monkeypatch.setattr(Bouquet, "DRAW_FIXED_CHANNELS", channels)
    bl = _deliver(b, "diff", req)
    monkeypatch.setattr(Bouquet, "DRAW_FIXED_CHANNELS",
                        ("j_NBI", "j_RF", "j_other"))
    s = _settings()
    toy.replace_eq(source_eq=F)
    p = _EC * (_NE * _TE + _NI * _TI)
    out = b._verify_sigma0_jbs_loop(
        s, None, {"y": np.asarray(bl.j_phi, float)}, p, _NE, _TE, _NI, _TI,
        _ZEFF, _X, _PAD, draw_route=False)
    gap = out["split_closure"]["max_abs_frac"]
    if closes:
        assert gap <= 1e-12 and out["passed_baseline_way"], out
    else:
        share = np.max(_J_OTHER) / np.max(np.abs(bl.j_phi))
        assert gap == pytest.approx(share, rel=1e-6)
        assert not out["passed_baseline_way"]


def test_a_legacy_draw_carries_the_kinetic_samplers_record(toy):
    """PR #56 B3/B4/B7 (integration hook): every legacy draw's diagnostics
    carry the shared sampler's record (version + clip counters), which
    generate_bouquet archives as the draw group's ``kinetic_sampler_json``."""
    from bouquet.kinetic_sampler import CLIP_COUNTERS, KINETIC_SAMPLER_VERSION
    req = _reconstruct(toy, _J_NBI + _J_OTHER)
    b = _bouquet(toy)
    _deliver(b, "diff", req)
    d = _draw(b, _settings())
    ks = d["kinetic_sampler"]
    assert ks["version"] == KINETIC_SAMPLER_VERSION
    assert set(ks["clips"]) == set(CLIP_COUNTERS)
    assert ks["clipped"] is False          # sigma = 0: nothing clipped
    import json
    assert json.loads(json.dumps(ks)) == ks   # archivable as JSON
