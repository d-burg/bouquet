"""Draws on the unified engine -- fast half (no GS solver).

The engine draw (:mod:`bouquet.engine_draws`) is driven by the toy
Grad-Shafranov stand-in of the Stage 2 tests (``tests/_engine_toy.py``),
and ``generate_bouquet`` / ``Bouquet.generate`` / the parallel worker by a
TokaMaker stand-in over that toy (``tests/_engine_fake_gs.py``).  Checked:

* the zero-perturbation identity BY CONSTRUCTION: the first request is the
  stored request bit for bit (delivery correction off/on, q0 row off/on),
  the closure's Ip increment is exactly zero on pass 1, the loop's pass-1
  residuals are the reconstruction's own and the loop exits after the
  required consecutive passes (the current gate is measured one pass late),
  and the delivered draw reproduces the reconstruction within the loop
  tolerances;
* the sampler draws the legacy stream (same Generator state after the
  kinetic and auxiliary channels as perturb_kinetic_equilibrium, same
  pressure) and today's toroidal inductive sigma, and a zero sigma is
  exactly the base;
* a perturbed draw moves l_i and beta_N and records the change of l_i and
  of the poloidal flux range against the reconstruction; the Ip amplitude
  is 1.0 at zero perturbation and the exact measure holds every pass; the
  q0 row acts when enabled;
* the post-hoc filters are applied to the archived draw and recorded,
  out-of-band draws are archived with in_spec=False and not counted by
  until-N; rejections carry their DRAW_REJECTION_REASONS code;
* end to end through generate_bouquet, Bouquet.generate (+ .filter(),
  verify_sigma0_consistency) and the parallel worker entry point and merge,
  on toy archives.

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import copy
import io
import os
import warnings

import numpy as np
import pytest

import _engine_toy as T
from bouquet import engine_draws as ED
from bouquet.config import GenerationConfig
from bouquet.engine import reconstruct
from bouquet.jbs_loop import (JBS_REQUIRED_CONSECUTIVE, JBSNonFinite,
                              JBSNotConverged)
from bouquet.physics import ELEMENTARY_CHARGE as EC

PSI = T.PSI
_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _kin():
    ne = 5e19 * (1 - 0.8 * PSI ** 2) + 2e18
    te = 2e3 * (1 - 0.9 * PSI ** 2) + 50.0
    return dict(ne=ne, te=te, ni=0.9 * ne, ti=1.1 * te,
                zeff=np.full(PSI.size, 1.8))


def _li0():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"]


LI0 = _li0()


def _recon(*, li=1.01, q0=None, gain=True, defect=0.02, **gc):
    """A toy reconstruction whose contract's pressure IS its kinetics'
    thermal pressure (so the sampler's pressure match is the legacy one)."""
    ad = T.ToyAdapter(li_target=LI0 * li,
                      q0_target=(None if q0 is None else q0))
    c = ad.read()
    k = _kin()
    c.kinetics = k
    c.kinetics_native = dict(psi_N=PSI, **{kk: k[kk] for kk in
                                          ("ne", "te", "ni", "ti")},
                             Zeff=k["zeff"])
    th = EC * (k["ne"] * k["te"] + k["ni"] * k["ti"])
    c.pressure = th.copy()
    c.pressure_parts = dict(thermal=th, impurity=0 * PSI, fast=0 * PSI,
                            Z_imp=None)
    b = T.ToyGS(gain=gain, defect=defect)
    b.set_inputs(pressure=c.pressure, kinetics=c.kinetics)
    eng, res, rec = _quiet(reconstruct, ad, b, T.settings(**gc), label="toy")
    assert res["converged"]
    return eng, res, rec, b


def _ctx(eng, res, *, q0_row=False, **gc):
    g = GenerationConfig(reconstruction_engine="unified", **gc)
    k = eng.c.kinetics
    native = dict(psi_N=PSI, ne=k["ne"], te=k["te"], ni=k["ni"],
                  ti=k["ti"])
    return ED.EngineDrawContext(eng, res, loop=ED.draw_loop_settings(g),
                                native=native, q0_row=q0_row)


def _unc(ctx, f=0.03, fj=None):
    k = ctx.native
    return dict(sigma_ne=f * k["ne"], sigma_te=f * k["te"],
                sigma_ni=f * k["ni"], sigma_ti=f * k["ti"],
                sigma_jphi=(f if fj is None else fj) * np.abs(ctx.request),
                n_ls=0.3, t_ls=0.3, j_ls=0.25)


@pytest.fixture(scope="module")
def recon():
    return _recon()


# ---------------------------------------------------------------------------
#  zero-perturbation identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dc", [False, True])
def test_the_first_request_is_the_stored_request_bit_for_bit(dc):
    eng, res, rec, b = _recon(engine_delivery_correction=dc)
    ctx = _ctx(eng, res)
    out = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())
    r = out["record"]
    assert r["identity"]["pass1_request_bit_identical"] is True
    assert r["identity"]["pass1_request_max_abs_diff"] == 0.0
    np.testing.assert_array_equal(out["passes"].first_request,
                                  res["state"].request)
    if dc:
        assert np.any(res["state"].delivery_correction != 0.0)


@pytest.mark.parametrize("case", ["soft_rows", "bootstrap_scalar"])
def test_the_identity_holds_for_soft_rows_and_the_scalar_preset(case):
    """The IDS-like soft rows (the Ip-row target is then the closure's
    posterior Ip, not the measured one) and a scalar preset."""
    import test_engine as TE
    if case == "soft_rows":
        ad = T.ToyAdapter(soft=True, li_target=LI0 * 1.02)
        gc = {}
    else:
        ad = T.ToyAdapter(li_target=None, jB_ind=TE._consistent_inductive())
        gc = dict(engine_preset="bootstrap_scalar", engine_rows=["Ip"])
    ad.read()
    b = T.ToyGS()
    eng, res, rec = _quiet(reconstruct, ad, b, T.settings(**gc), label="t")
    ctx = _ctx(eng, res)
    r = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())["record"]
    assert r["identity"]["pass1_request_bit_identical"]
    assert r["amplitude"]["per_pass"][0]["a_ind"] == 1.0
    assert r["loop"]["converged"]
    if case == "soft_rows":
        assert ctx.Ip_star != eng.c.Ip        # the posterior, recorded


def test_pass_one_closes_with_exactly_zero_increment(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    r = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())["record"]
    p1 = r["amplitude"]["per_pass"][0]
    assert p1["dIp_A"] == 0.0 and p1["a_ind"] == 1.0
    lp = r["loop"]
    # the pass-1 residuals ARE the reconstruction's delivered ones (Redl of
    # the same equilibrium against the same bootstrap)
    chk = rec["delivered"]["checks"]["loop"]
    # (to the re-solve's rounding: the toy's flux-range fixed point moves
    # by an ulp when the same request is solved again)
    assert lp["r_j"][0] == pytest.approx(chk["r_j"], rel=1e-6, abs=0)
    assert lp["r_I"][0] == pytest.approx(chk["r_I"], rel=1e-6, abs=0)
    assert lp["dl_i"][0] <= 1e-12
    # the loop exits after the required consecutive passes; the current
    # gate is measured one pass late, so pass 1 cannot count
    assert lp["converged"] and lp["criteria"]["current_residual"]
    assert lp["n_passes"] == JBS_REQUIRED_CONSECUTIVE + 1
    assert lp["pass_ok"] == [False, True, True]


def test_the_zero_perturbation_draw_delivers_the_reconstruction(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    s = ctx.loop
    v = _quiet(ED.verify_zero_perturbation, ctx, b)
    assert v["passed"] and v["request_bit_identical"]
    assert v["r_j"] <= s["rtol_j"] and v["r_I"] <= s["rtol_Ip"]
    assert abs(v["dl_i"]) <= s["tol_li"]
    assert v["amplitude"] == pytest.approx(1.0, abs=1e-5)
    assert v["dq0_psi_N"] == pytest.approx(T.PAD)


def test_a_zero_sigma_sample_is_exactly_the_base(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    unc = _unc(ctx, f=0.0)
    from bouquet.sampling import make_rng
    inp = ED.sample_draw_inputs(ctx, make_rng(3), unc, b.flux_integral)
    z = ctx.zero_inputs()
    for k in ("ne", "te", "ni", "ti", "zeff"):
        np.testing.assert_array_equal(inp.kinetics[k], z.kinetics[k])
    np.testing.assert_array_equal(inp.pressure, eng.c.pressure)
    np.testing.assert_array_equal(inp.jB_ind, eng.c.jB_ind)
    inp.sampler["zero_perturbation"] = True
    r = _quiet(ED.run_draw, ctx, b, inp)["record"]
    assert r["identity"]["pass1_request_bit_identical"]


def test_a_state_that_does_not_compose_its_request_is_refused(recon):
    eng, res, rec, b = recon
    bad = dict(res)
    st = copy.copy(res["state"])
    st.request = st.request * (1.0 + 1e-12)
    bad["state"] = st
    with pytest.raises(ED.EngineDrawRefused, match="does not reproduce"):
        _ctx(eng, bad)


# ---------------------------------------------------------------------------
#  the sampler: the legacy stream
# ---------------------------------------------------------------------------
class _Stop(BaseException):
    pass


class _LegacyMock:
    """Enough of a solver for perturb_kinetic_equilibrium to sample and
    stop at its first solve."""
    psi_bounds = [-0.3, 0.0]

    def __init__(self, fi):
        self.fi = fi
        self.pp = None

    def flux_integral(self, x, p):
        return self.fi(x, p)

    def set_targets(self, **k):
        pass

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        self.pp = pp_prof

    def solve(self):
        raise _Stop()


@pytest.fixture()
def _legacy_oft_names(monkeypatch):
    """The legacy perturb_kinetic_equilibrium is frozen code
    (tests/test_edge_pressure_legacy_ast.py) that imports OpenFUSIONToolkit's
    bootstrap names at its top, although the mock stops it at its first
    solve, before any of them is called.  Where OpenFUSIONToolkit is not
    installed (the fast CI suite) stub exactly those names in, each raising
    if called, so this stays a real comparison of the two streams on every
    run of the suite rather than one that skips or errors."""
    import sys
    import types
    try:
        import OpenFUSIONToolkit.TokaMaker.bootstrap  # noqa: F401
        import OpenFUSIONToolkit.TokaMaker.util  # noqa: F401
        return
    except ImportError:
        pass

    def _never(name):
        def f(*a, **k):
            raise AssertionError(f"the sampler comparison reached {name}")
        return f
    pkg = types.ModuleType("OpenFUSIONToolkit")
    sub = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    util = types.ModuleType("OpenFUSIONToolkit.TokaMaker.util")
    bs = types.ModuleType("OpenFUSIONToolkit.TokaMaker.bootstrap")
    for n in ("get_jphi_from_GS", "create_power_flux_fun"):
        setattr(util, n, _never(n))
    for n in ("solve_with_bootstrap", "find_optimal_scale"):
        setattr(bs, n, _never(n))
    sub.util, sub.bootstrap, pkg.TokaMaker = util, bs, sub
    for name, mod in (("OpenFUSIONToolkit", pkg),
                      ("OpenFUSIONToolkit.TokaMaker", sub),
                      ("OpenFUSIONToolkit.TokaMaker.util", util),
                      ("OpenFUSIONToolkit.TokaMaker.bootstrap", bs)):
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.mark.parametrize("zeff_primary", [False, True])
def test_the_sampler_draws_the_legacy_kinetic_stream(recon, zeff_primary,
                                                     _legacy_oft_names):
    from bouquet.jbs_loop import jbs_settings
    from bouquet.sampling import make_rng
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    from bouquet.utils import pchip_derivative
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    unc = _unc(ctx, f=0.05)
    k = ctx.native
    if zeff_primary:
        unc.update(aux_sigmas={"zeff": 0.1 * np.ones_like(PSI),
                               "omega_tor": 1e3 * np.ones_like(PSI)},
                   aux_baselines={"zeff": np.full(PSI.size, 1.8),
                                  "omega_tor": 1e4 * (1 - PSI)},
                   aux_length_scales={"zeff": 0.4, "omega_tor": 0.3})
    else:
        unc.update(aux_sigmas={"omega_tor": 1e3 * np.ones_like(PSI)},
                   aux_baselines={"omega_tor": 1e4 * (1 - PSI)},
                   aux_length_scales={"omega_tor": 0.3})
    m = _LegacyMock(b.flux_integral)
    r1, r2 = make_rng(7), make_rng(7)
    g = GenerationConfig(reconstruction_engine="unified")
    with pytest.raises(_Stop):
        _quiet(perturb_kinetic_equilibrium, m, PSI,
               ctx.pressure_thermal_base, k["ne"], k["te"], k["ni"],
               k["ti"], ctx.request, unc["sigma_ne"], unc["sigma_te"],
               unc["sigma_ni"], unc["sigma_ti"], unc["sigma_jphi"], 0.3,
               0.3, 0.25, 1.2e6, LI0, eng.c.kinetics["zeff"], PSI.size,
               input_jinductive=0.6 * ctx.request, p_thresh=0.05, rng=r1,
               jbs_loop=jbs_settings(g, draw=True),
               aux_sigmas=unc["aux_sigmas"],
               aux_baselines=unc["aux_baselines"],
               aux_length_scales=unc["aux_length_scales"])
    inp = ED.sample_kinetics(ctx, r2, unc, b.flux_integral, p_thresh=0.05)
    # the SAME Generator state after the kinetic and auxiliary channels
    assert r1.bit_generator.state == r2.bit_generator.state
    # ... and the same drawn pressure (increment form: rounding only)
    leg = m.pp["y"] * 0.3
    mine = pchip_derivative(PSI, inp.pressure)
    mine[-1] = 0.0
    assert np.max(np.abs(leg - mine)) <= 1e-12 * np.max(np.abs(leg))
    assert set(inp.aux) == set(unc["aux_sigmas"])
    assert inp.sampler["zeff_primary"] is zeff_primary
    # the sampler version and its clip counters ride with every engine draw
    # (PR #56 B3/B4/B7)
    from bouquet.kinetic_sampler import CLIP_COUNTERS, KINETIC_SAMPLER_VERSION
    ks = inp.sampler["kinetic_sampler"]
    assert ks["version"] == KINETIC_SAMPLER_VERSION
    assert ks["zeff_primary"] is zeff_primary
    assert set(ks["clips"]) == set(CLIP_COUNTERS)


@pytest.mark.parametrize("j_ls, bar", [(0.05, 1e-8), (0.25, 1e-3)])
def test_the_inductive_candidate_is_the_legacy_toroidal_draw(recon, j_ls,
                                                             bar):
    """The candidate's toroidal perturbation on G* IS the legacy draw of the
    same stream (sigma_jphi, j_ls) -- whatever the mean it perturbs: the
    factorised covariance does not depend on it (to rounding, which the
    near-singular kernel amplifies at the longer length scale)."""
    from bouquet.sampling import generate_perturbed_GPR, make_rng
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    unc = _unc(ctx, fj=0.05)
    unc["j_ls"] = j_ls
    fac = ctx.s_ind * ctx.parts_star["kappa"]
    other = 0.6 * ctx.request                  # a legacy-like inductive
    r1, r2 = make_rng(11), make_rng(11)
    cand, tries, _nf = ED.sample_inductive(ctx, r1, unc)
    ref = generate_perturbed_GPR(PSI, other / other[0],
                                 sigma_profile=unc["sigma_jphi"] / other[0],
                                 length_scale=j_ls, n_samples=1, rng=r2,
                                 diag_plot=False) * other[0]
    assert tries == 1
    mine = fac * (cand - np.asarray(eng.c.jB_ind))
    sig = np.max(unc["sigma_jphi"])
    assert np.max(np.abs(mine - (ref - other))) <= bar * sig
    assert r1.bit_generator.state == r2.bit_generator.state


# ---------------------------------------------------------------------------
#  a perturbed draw
# ---------------------------------------------------------------------------
def _perturbed(ctx, b, seed=12345, f=0.03, scale=1.0):
    from bouquet.sampling import make_rng
    inp = ED.sample_draw_inputs(ctx, make_rng(seed), _unc(ctx, f=f),
                                b.flux_integral, scale=scale)
    return _quiet(ED.run_draw, ctx, b, inp)


def test_a_perturbed_draw_moves_li_and_beta_and_records_the_flux_range(
        recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    out = _perturbed(ctx, b, scale=1.01)
    r = out["record"]
    assert r["loop"]["converged"]
    d = r["deltas"]
    assert abs(d["l_i_3"]) > 1e-4 and abs(d["beta_n"]) > 0.0
    assert "attribution" not in r
    # the flux range psi_b - psi_a against the reconstruction's (the toy's
    # flux range responds to the current shape)
    fr, fr0 = r["delivered"]["flux_range"], r["reference"]["flux_range"]
    # the reference is the DELIVERED measurement's, not G*'s
    assert fr0 == ED.flux_range(eng.delivered_meas)
    assert d["flux_range"] == fr - fr0 and d["flux_range"] != 0.0
    assert d["flux_range_rel"] == pytest.approx((fr - fr0) / fr0,
                                                rel=1e-14)
    assert d["l_i_1"] is not None
    for k in ("l_i_3", "l_i_1", "beta_n", "q0", "q0_psi_N", "q95", "Ip",
              "delivery_check", "flux_range"):
        assert k in r["delivered"], k
    assert r["delivered"]["delivery_check"]["ok"]


def test_the_ip_amplitude_holds_the_exact_measure_every_pass(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    r = _perturbed(ctx, b, seed=5)["record"]
    for p in r["amplitude"]["per_pass"]:
        assert p["Ip_exact_measure_A"] == pytest.approx(
            ctx.Ip_star, rel=1e-12, abs=0)
    assert r["amplitude"]["final"]["a_ind"] != 1.0
    assert ctx.Ip_star == pytest.approx(eng.c.Ip, rel=1e-9)


def test_the_q0_row_acts_when_enabled():
    from bouquet.jbs_loop import AxisRowPin  # noqa: F401
    b0 = T.ToyGS()
    b0.solve(T.ToyAdapter().read().anchor_request)
    q_anchor = b0.q_at(b0.state, T.PAD)
    eng, res, rec, b = _recon(q0=q_anchor * 0.97,
                              engine_rows=["Ip", "l_i", "q0"])
    ctx = _ctx(eng, res, q0_row=True, engine_rows=("Ip", "l_i", "q0"),
               engine_draw_q0_row=True)
    # identity with the row kept
    z = _quiet(ED.run_draw, ctx, b, ctx.zero_inputs())["record"]
    assert z["identity"]["pass1_request_bit_identical"]
    assert z["amplitude"]["per_pass"][0]["a_bs"] == 1.0
    r = _perturbed(ctx, b, seed=21)["record"]
    assert r["loop"]["converged"] and r["loop"]["criteria"]["q0_residual"]
    qr = r["q0_row"]["record"]
    assert qr["n_row_updates"] >= 1
    tol = eng.s["q0_tol"]
    assert abs(r["delivered"]["q0"] - ctx.q0_target) <= tol


def test_the_q0_row_needs_the_reconstructions_q0_row(recon):
    eng, res, rec, b = recon
    with pytest.raises(ED.EngineDrawRefused, match="no active q0 row"):
        _ctx(eng, res, q0_row=True)
    from bouquet.engine import validate_engine_settings
    with pytest.raises(ValueError, match="engine_draw_q0_row"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", engine_draw_q0_row=True))
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="legacy", engine_draw_q0_row=True))
    with pytest.raises(ValueError, match="no effect"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="legacy", engine_draw_homotopy=False))
    with pytest.raises(ValueError, match="bool"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="unified", engine_draw_homotopy=1))


# ---------------------------------------------------------------------------
#  post-hoc filters and rejection codes
# ---------------------------------------------------------------------------
def test_the_post_hoc_filters(recon):
    eng, res, rec, b = recon
    ctx = _ctx(eng, res)
    li = ctx.ref["l_i"]
    fin = dict(li=li * 1.04, q_row=0.95)
    v = ED.post_hoc_verdicts(ctx, fin, l_i_tolerance=0.05,
                             constrain_sawteeth=False)
    assert v["l_i_in_band"] and v["q0_ok"] and v["in_band"]
    v = ED.post_hoc_verdicts(ctx, fin, l_i_tolerance=0.05,
                             constrain_sawteeth=True)
    assert not v["q0_ok"] and not v["in_band"] and v["reasons"]
    v = ED.post_hoc_verdicts(ctx, dict(li=li * 1.06, q_row=1.2),
                             l_i_tolerance=0.05, constrain_sawteeth=True)
    assert not v["l_i_in_band"] and v["q0_ok"] and not v["in_band"]
    assert v["q0_psi_N"] == pytest.approx(T.PAD)


def test_rejection_codes(recon):
    from bouquet.engine import EngineClosureRefused
    from bouquet.TokaMaker_interface import (DRAW_REJECTION_REASONS,
                                             CoilSaturated,
                                             DrawAnchorSolveFailed)
    eng, res, rec, b = recon
    # loop failure: a ceiling the draw cannot meet
    ctx = _ctx(eng, res, jbs_max_passes_draw=2)
    with pytest.raises(JBSNotConverged) as ei:
        _perturbed(ctx, b)
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "jbs_not_converged"
    # non-finite bootstrap: JBSNonFinite at once
    ctx = _ctx(eng, res)
    b2 = copy.deepcopy(b)
    b2.nan_redl_at = b2.n_solves + 1
    with pytest.raises(JBSNonFinite) as ei:
        _quiet(ED.run_draw, ctx, b2, ctx.zero_inputs())
    assert ei.value.record["n_passes"] == 1
    assert ED.engine_rejection_reason(ei.value, "perturb") == "jbs_non_finite"
    # saturation under the hard coil bounds
    def _sat(stage):
        raise CoilSaturated(f"{stage}: F9A on its bound", dict(stage=stage))
    with pytest.raises(CoilSaturated):
        _quiet(ED.run_draw, ctx, copy.deepcopy(b), ctx.zero_inputs(),
               coil_guard=_sat)
    assert ED.engine_rejection_reason(CoilSaturated("x"), "perturb") \
        == "coil_saturation_jbs_loop"
    assert ED.engine_rejection_reason(CoilSaturated("x"), "post_homotopy") \
        == "coil_saturation_post_homotopy"
    # a degenerate Ip amplitude
    inp = ctx.zero_inputs()
    inp.jB_ind = np.zeros_like(inp.jB_ind)
    with pytest.raises(EngineClosureRefused) as ei:
        _quiet(ED.run_draw, ctx, copy.deepcopy(b), inp)
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "engine_closure_refused"
    # the first solve failing: the anchor analog
    b3 = copy.deepcopy(b)

    def _fail(req, n_passes=1):
        raise RuntimeError("synthetic GS failure")
    b3.solve = _fail
    with pytest.raises(DrawAnchorSolveFailed) as ei:
        _quiet(ED.run_draw, ctx, b3, ctx.zero_inputs())
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "anchor_solve_failed"
    for code in ("jbs_not_converged", "jbs_non_finite",
                 "engine_closure_refused", "anchor_solve_failed",
                 "coil_saturation_jbs_loop"):
        assert code in DRAW_REJECTION_REASONS


def test_the_until_n_verdict_ands_the_band():
    G = ED.GenerateEngineDraws.__new__(ED.GenerateEngineDraws)
    G._cur = None
    ok, why = G.until_n(True, [], {ED.DRAW_BAND_FLAG: False})
    assert not ok and why == ["engine post-hoc band"]
    ok, why = G.until_n(True, [], {ED.DRAW_BAND_FLAG: True})
    assert ok and why == []
    ok, why = G.until_n(False, ["coil"], {ED.DRAW_BAND_FLAG: True})
    assert not ok and why == ["coil"]


def test_selected_ands_the_band_flag_only_where_present(tmp_path):
    import h5py
    from bouquet.filtering import _recompute_selected
    with h5py.File(tmp_path / "f.h5", "w") as hf:
        leg = hf.create_group("a")
        leg.attrs["passes_coil_filter"] = True
        leg.attrs["passes_boundary_filter"] = True
        _recompute_selected(leg)
        assert bool(leg.attrs["selected"]) is True
        eng = hf.create_group("b")
        eng.attrs["passes_coil_filter"] = True
        eng.attrs["passes_boundary_filter"] = True
        eng.attrs[ED.DRAW_BAND_FLAG] = False
        _recompute_selected(eng)
        assert bool(eng.attrs["selected"]) is False


# ---------------------------------------------------------------------------
#  end to end: generate_bouquet on a TokaMaker stand-in
# ---------------------------------------------------------------------------
def _generate(tmp_path, monkeypatch, *, n=3, l_i_tolerance=0.05,
              n_inspec_target=None, homotopy=True, seed=12345, **extra):
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.utils import initialize_equilibrium_database
    os.makedirs(str(tmp_path), exist_ok=True)
    eng, res, rec, b = _recon()
    ctx = _ctx(eng, res)
    unc = _unc(ctx)
    G = ED.GenerateEngineDraws(ctx, unc=unc, psi_pad=T.PAD,
                               homotopy=homotopy,
                               l_i_tolerance=l_i_tolerance)
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)
    fake = FakeTokaMaker(b)
    h = str(tmp_path / "e2e")
    initialize_equilibrium_database(h)
    rej = []
    k = ctx.native
    diags = _quiet(
        generate_bouquet, fake, PSI, n, h, ctx.request, k["ne"], k["te"],
        k["ni"], k["ti"], unc["sigma_ne"], unc["sigma_te"], unc["sigma_ni"],
        unc["sigma_ti"], unc["sigma_jphi"], 0.3, 0.3, 0.25,
        float(eng.c.Ip), ctx.ref["l_i"], eng.c.kinetics["zeff"],
        input_jinductive=0.5 * ctx.request,
        baseline_j_BS=0.1 * ctx.request, l_i_tolerance=l_i_tolerance,
        psi_pad=T.PAD, constrain_sawteeth=False, isolate_edge_jBS=False,
        jBS_scale_range=(0.99, 1.01), coil_drift=0.01,
        homotopy_passes=[(0.05, 0.1), (0.01, 0.01)], seed=seed,
        capture_live_eq=False, store_achieved_jphi=True,
        jbs_loop=G.loop_settings, rejection_log=rej, draw_method=G,
        coil_filter="legacy", n_inspec_target=n_inspec_target, **extra)
    return diags, rej, h, G


def test_generate_bouquet_runs_engine_draws_end_to_end(tmp_path,
                                                        monkeypatch):
    import h5py
    diags, rej, h, G = _generate(tmp_path, monkeypatch)
    assert len(diags) == 3 and rej == []
    for i, d in enumerate(diags):
        e = d["engine"]
        assert e["version"] == ED.ENGINE_DRAW_VERSION
        assert e["loop"]["converged"] and d["jbs_loop"]["kind"] == "engine"
        assert e["post_hoc"]["in_band"] and d["in_spec"] is True
        assert e["homotopy"]["enabled"] and e["homotopy"]["homotopy_pass"] \
            == 1
        c = e["cost"]
        assert set(ED._Clock.STAGES) <= set(c)
        assert c["loop"]["solves"] == e["loop"]["n_passes"] \
            == c["loop"]["passes"]
        assert c["anchor"]["solves"] == 0
        assert c["homotopy"]["solves"] == 2        # two homotopy passes
        assert c["total"]["wall_s"] > 0.0
        back = ED.read_draw_engine(h, i)
        assert back["deltas"] == e["deltas"]
        # the kinetic sampler's version + clip counters are archived
        assert back["inputs"]["kinetic_sampler"]["version"].startswith(
            "kinetic_sampler/")
        assert "cost" in back
    with h5py.File(h + ".h5", "r") as hf:
        g0 = hf["scan/0/0"] if "scan" in hf else hf["0"]
        assert bool(g0.attrs[ED.DRAW_BAND_FLAG]) is True
        assert bool(g0.attrs["in_spec"]) is True
        assert bool(g0.attrs["jbs_converged"]) is True


def test_an_out_of_band_draw_is_archived_and_not_counted(tmp_path,
                                                          monkeypatch):
    import h5py
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=2,
                                 l_i_tolerance=1e-9, n_inspec_target=1)
    # every draw is outside a 1e-9 band: archived, in_spec False, never
    # counted -- the loop runs to its attempt cap (max(n, 5 * target))
    assert len(diags) == 5 and rej == []
    for d in diags:
        assert d["engine"]["post_hoc"]["in_band"] is False
        assert d["in_spec"] is False
        assert d["until_n_inspec"] is False
        assert "engine post-hoc band" in d["until_n_reasons"]
    with h5py.File(h + ".h5", "r") as hf:
        g0 = hf["scan/0/0"] if "scan" in hf else hf["0"]
        assert bool(g0.attrs[ED.DRAW_BAND_FLAG]) is False
        assert bool(g0.attrs["in_spec"]) is False


def test_until_n_counts_in_band_engine_draws(tmp_path, monkeypatch):
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=4,
                                 n_inspec_target=2)
    from bouquet.filtering import until_n_delivered
    assert until_n_delivered(diags) == 2 and len(diags) == 2


def test_no_homotopy_stage_when_disabled(tmp_path, monkeypatch):
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=1,
                                 homotopy=False)
    e = diags[0]["engine"]
    assert e["homotopy"]["enabled"] is False
    assert e["cost"]["homotopy"]["solves"] == 0
    assert diags[0]["homotopy_pass"] == -1 and diags[0]["in_spec"]


def test_the_same_seed_draws_the_same_engine_draws(tmp_path, monkeypatch):
    a, _r, _h, _G = _generate(tmp_path / "a", monkeypatch, n=2)
    b, _r, _h, _G = _generate(tmp_path / "b", monkeypatch, n=2)
    for x, y in zip(a, b):
        assert x["engine"]["deltas"] == y["engine"]["deltas"]
        assert x["engine"]["inputs"] == y["engine"]["inputs"]


# ---------------------------------------------------------------------------
#  Bouquet.generate / verify_sigma0_consistency / the parallel worker
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_bouquet_solver(monkeypatch):
    """The engine's backend -> a toy whose pressure and kinetics are the
    contract's (so the reconstruction and its draws agree), created once
    per stand-in solver; the draws' backend -> the same toy."""
    import bouquet.engine as be
    from _engine_fake_gs import FakeTokaMaker

    def _backend(mygs, contract, **kw):
        if getattr(mygs, "toy", None) is None:
            t = T.ToyGS(psi=contract.psi_N, Ip=contract.Ip)
            t.set_inputs(pressure=contract.pressure,
                         kinetics=contract.kinetics)
            mygs.toy = t
        return mygs.toy

    monkeypatch.setattr(be, "TokaMakerBackend", _backend)
    monkeypatch.setattr(be, "_lcfs_deviation_mm", lambda m, p: (2.5, 7.0))
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)

    def _setup(self):
        # what the real setup_solver does to the coil solve
        # (tests/test_solver_state.py runs the real one on a stand-in)
        from bouquet.solver_state import enter_bounded_coil_mode
        self.mygs = FakeTokaMaker(None)
        enter_bounded_coil_mode(self.mygs)
        return self

    import bouquet.run as br
    monkeypatch.setattr(br.Bouquet, "setup_solver", _setup)
    return _backend


def _bq(tmp_path, n=2):
    import bouquet as bq
    b = bq.Bouquet.from_geqdsk(
        os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(_EX, "DIIID_mesh.h5"), n_draws=n,
        header=str(tmp_path / "bq"), reconstruction_engine="unified")
    g = b.config.generation
    g.engine_rows = ["Ip"]
    g.seed = 12345
    return b


def test_bouquet_generate_runs_on_the_engine(tmp_path, toy_bouquet_solver):
    from bouquet.engine import load_baseline_engine
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    v = _quiet(b.verify_sigma0_consistency)
    assert v["passed"] and v["request_bit_identical"]
    assert v["invariant"] == "engine-draw"
    diags = _quiet(b.generate)
    assert len(diags) == 2 and b.draw_rejections == []
    assert all(d["engine"]["loop"]["converged"] for d in diags)
    rec = load_baseline_engine(b.config.output_header,
                               scan_key=b.config.generation.scan_key)
    assert rec is not None and rec["draws"]["version"] \
        == ED.ENGINE_DRAW_VERSION
    sel = _quiet(b.filter)
    assert sel["boundary"]["n_total"] == 2
    # the archived engine draws export like any draw: g-file bundle and a
    # perturbed IDS (the example OMAS file as the template)
    import json
    from bouquet.io.imas import write_imas_draw
    bun = _quiet(b.export_bundle, str(tmp_path / "bundle"),
                 formats=("geqdsk",), selection="all")
    assert len(bun) == 2 and all(os.path.isfile(v["geqdsk"])
                                 for v in bun.values())
    out = _quiet(write_imas_draw, b.config.output_header, 0,
                 os.path.join(_EX, "D3Dlike_baseline_omas.json"),
                 str(tmp_path / "d0.json"), time=2.3043)
    with open(out) as fh:
        ids = json.load(fh)
    assert ids["core_profiles"]["profiles_1d"]


def test_a_legacy_baseline_is_not_drawn_on_the_engine(tmp_path,
                                                       toy_bouquet_solver):
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    b.config.generation.reconstruction_engine = "legacy"
    b.config.generation.engine_rows = ("Ip", "l_i")
    with pytest.raises(NotImplementedError, match="now 'legacy'"):
        b.generate()


def test_the_parallel_worker_entry_point_and_merge(tmp_path,
                                                    toy_bouquet_solver):
    import h5py
    from bouquet.engine import load_baseline_engine
    from bouquet.parallel import merge_archives, run_shard
    b = _bq(tmp_path, n=3)
    out = str(tmp_path / "par")
    metas = [_quiet(run_shard, b.config, w, 2, n_equils_total=3,
                    seed_base=7, out_header=out, scan_key=0,
                    threads_per_worker=1, verbose=True) for w in (0, 1)]
    assert [m["n"] for m in metas] == [2, 1]
    assert metas[0]["li_target"] == metas[1]["li_target"]
    path, n = merge_archives([m["path"] for m in metas], out, scan_key=0)
    assert n == 3
    with h5py.File(path, "r") as hf:
        for i in range(3):
            g = hf[f"scan/0/{i}"]
            assert "engine_json" in g.attrs
            assert ED.DRAW_BAND_FLAG in g.attrs
    for i in range(3):
        assert ED.read_draw_engine(out, i, scan_key=0)["version"] \
            == ED.ENGINE_DRAW_VERSION
    assert load_baseline_engine(out, scan_key=0)["draws"] is not None


def test_the_probe_draw_mode_runs_on_the_toy(tmp_path, toy_bouquet_solver):
    """The solver probe's sigma0 stage and draw mode, on the stand-in (so
    the cluster run cannot fail on the probe's own bookkeeping)."""
    import json
    import sys
    sys.path.insert(0, os.path.join(_HERE, "probes"))
    import measure_engine as ME
    from bouquet.jbs_loop import jsonable
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    z = _quiet(ME._sigma0, b)
    assert z["passed"] and z["request_bit_identical"]
    assert z["cost"]["anchor"]["solves"] == 0
    d = _quiet(ME._draws, b, 2, ME.DEFAULT_SEED)
    assert d["attempts"] == 2 == d["archived"] + d["rejected"]
    assert set(d["stage_totals"]) == set(ED._Clock.STAGES)
    json.dumps(jsonable(d))


# ---------------------------------------------------------------------------
#  non-physical DRAWN kinetics: their own rejection code, before any solve
# ---------------------------------------------------------------------------
def _counting(b):
    """A backend copy that counts solves and bootstrap evaluations."""
    b2 = copy.deepcopy(b)
    n = dict(solve=0, redl=0)
    _solve, _redl = b2.solve, b2.redl

    def solve(*a, **k):
        n["solve"] += 1
        return _solve(*a, **k)

    def redl(*a, **k):
        n["redl"] += 1
        return _redl(*a, **k)
    b2.solve, b2.redl = solve, redl
    return b2, n


@pytest.mark.parametrize("name, value, rule", [
    ("ti", -1810.0, "strictly positive"), ("te", 0.0, "strictly positive"),
    ("ne", -1.0e18, "strictly positive"), ("ni", 0.0, "strictly positive"),
    ("zeff", 0.9, ">= 1")])
def test_nonphysical_drawn_kinetics_have_their_own_code(recon, name, value,
                                                        rule):
    from bouquet.TokaMaker_interface import (DRAW_REJECTION_REASONS,
                                             DrawAnchorSolveFailed)
    eng, res, _rec, b = recon
    ctx = _ctx(eng, res)
    inp = ctx.zero_inputs()
    inp.kinetics = {k: np.array(v, dtype=float, copy=True)
                    for k, v in inp.kinetics.items()}
    inp.kinetics[name][3:9] = value
    b2, n = _counting(b)
    with pytest.raises(ED.DrawKineticsNonPhysical) as ei:
        _quiet(ED.run_draw, ctx, b2, inp)
    # found BEFORE any solve and before the first bootstrap evaluation
    assert n == dict(solve=0, redl=0)
    assert not isinstance(ei.value, DrawAnchorSolveFailed)
    info = ei.value.info
    assert info["quantity"] == name and info["rule"] == rule
    assert info["psi_N"] == pytest.approx(float(PSI[3]))
    assert info["value"] == pytest.approx(value)
    assert info["n_bad"] == 6 and info["n_nodes"] == len(PSI)
    assert name in str(ei.value) and "psi_N=" in str(ei.value)
    assert ED.engine_rejection_reason(ei.value, "perturb") \
        == "kinetics_nonphysical"
    assert "kinetics_nonphysical" in DRAW_REJECTION_REASONS
    # the record carries the quantity and where
    rec = ED.GenerateEngineDraws.annotate_rejection(
        None, dict(reason="kinetics_nonphysical"), ei.value)
    assert rec["info"]["quantity"] == name
    assert rec["info"]["psi_N"] == pytest.approx(float(PSI[3]))


def test_a_non_finite_drawn_profile_is_the_same_code(recon):
    eng, res, _rec, b = recon
    ctx = _ctx(eng, res)
    inp = ctx.zero_inputs()
    inp.kinetics = {k: np.array(v, dtype=float, copy=True)
                    for k, v in inp.kinetics.items()}
    inp.kinetics["te"][5] = np.nan
    b2, n = _counting(b)
    with pytest.raises(ED.DrawKineticsNonPhysical) as ei:
        _quiet(ED.run_draw, ctx, b2, inp)
    assert n == dict(solve=0, redl=0)
    assert ei.value.info["quantity"] == "te"
    assert ei.value.info["rule"] == "finite"


def test_the_first_quantity_is_the_bootstrap_evaluations_order(recon):
    """ne, ni, te, ti, zeff -- the order evaluate_jBS checks them, so the
    quantity named is the one that evaluation named."""
    eng, res, _rec, b = recon
    ctx = _ctx(eng, res)
    inp = ctx.zero_inputs()
    inp.kinetics = {k: np.array(v, dtype=float, copy=True)
                    for k, v in inp.kinetics.items()}
    inp.kinetics["ti"][0] = -5.0
    inp.kinetics["ni"][7] = -1.0
    with pytest.raises(ED.DrawKineticsNonPhysical) as ei:
        _quiet(ED.run_draw, ctx, copy.deepcopy(b), inp)
    assert ei.value.info["quantity"] == "ni"
    assert [x["quantity"] for x in ei.value.info["all"]] == ["ni", "ti"]


def test_the_check_is_the_bootstrap_evaluations_own_domain():
    """Same verdict as evaluate_jBS's input checks on the same arrays: the
    draws rejected are the ones that evaluation refused -- no more."""
    from bouquet.physics import JBSEvaluationError, evaluate_jBS
    n = 17
    psi = np.linspace(0.0, 1.0, n)
    good = dict(ne=np.full(n, 3e19), te=np.full(n, 1e3),
                ni=np.full(n, 2.5e19), ti=np.full(n, 1.2e3),
                zeff=np.full(n, 1.8))
    assert ED.check_draw_kinetics(good, psi) is None
    cases = [("ne", 0.0), ("ne", -1.0), ("ni", 0.0), ("te", -3.0),
             ("ti", 0.0), ("ti", -1810.0), ("zeff", 0.999),
             ("ti", 1e-30), ("zeff", 1.0), ("ne", 1.0)]
    for name, v in cases:
        k = {q: a.copy() for q, a in good.items()}
        k[name][4] = v
        try:
            # the input checks run before the equilibrium is touched
            evaluate_jBS(None, psi, k["ne"], k["te"], k["ni"], k["ti"],
                         k["zeff"], psi_pad=1e-3)
            refused = None
        except JBSEvaluationError as e:
            refused = e.quantity
        except Exception:
            refused = None          # past the input checks (no equilibrium)
        try:
            ED.check_draw_kinetics(k, psi)
            mine = None
        except ED.DrawKineticsNonPhysical as e:
            mine = e.info["quantity"]
        assert mine == refused, (name, v, mine, refused)


def test_physical_draws_are_untouched(recon):
    """A draw with physical kinetics takes the path it took: same first
    request, same record (the check adds nothing to it)."""
    eng, res, _rec, b = recon
    ctx = _ctx(eng, res)
    out = _quiet(ED.run_draw, ctx, copy.deepcopy(b), ctx.zero_inputs())
    assert out["record"]["loop"]["converged"]
    assert "kinetics" not in "".join(out["record"].get("notices", []))


def _spy_store(monkeypatch):
    """Capture what store_equilibrium archives per draw (j_phi, j_BS,
    j_inductive: positional arguments 4-6)."""
    import bouquet.TokaMaker_interface as TI
    stored = []
    real = TI.store_equilibrium

    def spy(header, count, full_path, psi_N, jphi, jbs, jind, *a, **k):
        stored.append(dict(count=count, j_phi=np.array(jphi, dtype=float),
                           j_BS=np.array(jbs, dtype=float),
                           j_inductive=np.array(jind, dtype=float)))
        return real(header, count, full_path, psi_N, jphi, jbs, jind,
                    *a, **k)
    monkeypatch.setattr(TI, "store_equilibrium", spy)
    return stored


def test_the_archived_split_is_on_the_archived_state_and_never_clipped(
        tmp_path, monkeypatch):
    """An engine draw archives j_BS and the fixed parts evaluated on its
    ARCHIVED equilibrium and j_inductive as the exact residual against the
    archived j_phi -- no clip, no sliver moved into j_BS; the record says
    so (archived.split), including the negative-inductive count."""
    stored = _spy_store(monkeypatch)
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=2)
    assert len(diags) == 2 and rej == [] and len(stored) == 2
    import h5py
    from bouquet.schema import (CURRENT_SPLIT_CONVENTION_ATTR,
                                SPLIT_PRESSURE_SEPARATE)
    from bouquet.utils import _group_path, _resolve_h5
    for d, st in zip(diags, stored):
        sp = d["engine"]["archived"]["split"]
        jn, jr = np.asarray(sp["j_NBI"]), np.asarray(sp["j_RF"])
        # p'G is its own bucket (owner decision D2), archived as j_pressure
        jp = np.asarray(d["j_pressure"], dtype=float)
        np.testing.assert_array_equal(
            st["j_inductive"], st["j_phi"] - st["j_BS"] - jn - jr - jp)
        with h5py.File(_resolve_h5(h), "r") as hf:
            g = hf[_group_path(None, st["count"])]
            np.testing.assert_array_equal(g["j_pressure"][()], jp)
            assert g.attrs[CURRENT_SPLIT_CONVENTION_ATTR] == \
                SPLIT_PRESSURE_SEPARATE
        assert sp["n_negative_inductive"] == int(np.sum(
            st["j_inductive"] < 0.0))
        assert sp["min_inductive"] == float(np.min(st["j_inductive"]))
        assert "never clipped" in sp["convention"]


def test_the_archive_carries_the_parallel_parts_of_the_split(
        tmp_path, monkeypatch):
    """Every engine draw group carries ``jB_parallel/`` (schema): the
    <j.B> parts of its archived split, with j_phi = kappa (jB_inductive +
    jB_BS + jB_NBI + jB_RF) + j_pressure to round-off, jB_BS = j_BS /
    kappa (p'G is the archived j_pressure, owner decision D2) and
    jB_inductive = j_inductive / kappa -- the
    field-aligned parts only (the IDS exporter writes these, so no
    exported parallel current carries the pressure-driven term).  The
    archived state's <B^2> is moved by 3 % so kappa(archived) differs from
    the reconstruction's."""
    import h5py
    from bouquet.engine import conversion_factor, pressure_term
    from bouquet.schema import read_jB_parallel
    from bouquet.utils import _group_path, _resolve_h5
    stored = _spy_store(monkeypatch)
    fins = _final_measures(monkeypatch, geom_scale={"B2": 1.03})
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=1)
    assert len(diags) == 1 and rej == [] and len(stored) == 1
    fin, st = fins[-1], stored[0]
    with h5py.File(_resolve_h5(h), "r") as hf:
        par = read_jB_parallel(hf[_group_path(None, st["count"])])
    assert par is not None
    kap = conversion_factor(fin["geom"])
    P = pressure_term(fin["geom"])
    assert np.max(np.abs(P)) > 0.0
    np.testing.assert_allclose(par["kappa"], kap, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(par["j_pressure"], P, rtol=1e-15, atol=0.0)
    # j_BS is the field-aligned part alone; p'G is the archived j_pressure
    # (owner decision D2)
    np.testing.assert_allclose(par["jB_BS"] * kap, st["j_BS"],
                               rtol=1e-12, atol=0.0)
    with h5py.File(_resolve_h5(h), "r") as hf:
        np.testing.assert_allclose(
            hf[_group_path(None, st["count"])]["j_pressure"][()], P,
            rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(
        par["jB_inductive"], st["j_inductive"] / kap, rtol=1e-12, atol=0.0)
    comp = kap * (par["jB_inductive"] + par["jB_BS"] + par["jB_NBI"]
                  + par["jB_RF"]) + P
    np.testing.assert_allclose(comp, st["j_phi"], rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(st["j_phi"])))
    fx = G.ctx.c.jB_fix_parts
    np.testing.assert_array_equal(par["jB_NBI"], np.asarray(fx["nbi"]))


def _final_measures(monkeypatch, geom_scale=None):
    """Capture every FINAL measurement of the toy backend (the archived
    state's); with *geom_scale* = {field: factor}, the archived state's
    geometry is that of a state whose flux-surface averages moved (the toy's
    geometry does not depend on its state, so the archived and the
    reconstruction's conversion factors would otherwise coincide)."""
    fins = []
    real = T.ToyGS.measure

    def measure(self, *a, **k):
        out = real(self, *a, **k)
        if k.get("final"):
            if geom_scale:
                g = dict(out["geom"])
                for name, f in geom_scale.items():
                    g[name] = f * np.asarray(g[name], dtype=float)
                out = dict(out, geom=g)
            fins.append(out)
        return out
    monkeypatch.setattr(T.ToyGS, "measure", measure)
    return fins


def test_the_archived_split_uses_the_archived_states_kappa_and_redl(
        tmp_path, monkeypatch):
    """The archived split's bootstrap is the draw's bootstrap model ON the
    archived equilibrium: (1 + d_bs) x s_bs x scale x Redl(archived state),
    converted with the ARCHIVED state's F<1/R>/<B^2>; the fixed parts are
    converted with that same factor.  Pins the claim of 41e9682 that the
    2026-10-06 review found untested (mutants D2: the reconstruction's
    kappa; D4: the bootstrap off by 1 %): the archived state's <B^2> is moved
    by 3 % so the two conversion factors differ, and the archived j_BS,
    j_NBI and j_RF are compared with the definition at 1e-12."""
    from bouquet.engine import conversion_factor
    stored = _spy_store(monkeypatch)
    fins = _final_measures(monkeypatch, geom_scale={"B2": 1.03})
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=1)
    assert len(diags) == 1 and rej == [] and len(stored) == 1
    fin, st = fins[-1], stored[0]
    cur = G._cur
    kap = conversion_factor(fin["geom"])
    kap_recon = conversion_factor(G.ctx.geom)
    # the test discriminates: the two factors differ by ~3 %
    assert np.min(np.abs(kap / kap_recon - 1.0)) > 0.02
    dpl = cur["draw"].get("passes_post_homotopy") or cur["draw"]["passes"]
    amp = (1.0 + float(dpl.last["amp"].get("d_bs", 0.0))) * G.ctx.s_bs         * float(cur["draw"]["inputs"].scale)
    # (the pressure-driven p'G is its own bucket, j_pressure: D2)
    want_bs = amp * kap * np.asarray(fin["redl"], dtype=float)
    np.testing.assert_allclose(st["j_BS"], want_bs, rtol=1e-12, atol=0.0)
    fx = G.ctx.c.jB_fix_parts
    sp = diags[0]["engine"]["archived"]["split"]
    np.testing.assert_allclose(sp["j_NBI"], kap * np.asarray(fx["nbi"]),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        sp["j_RF"], kap * (np.asarray(fx["rf"])
                           + np.asarray(fx.get("other", 0.0))),
        rtol=1e-12, atol=0.0)
    # and a 1 % error in either input is far outside that
    assert np.max(np.abs(1.01 * want_bs - st["j_BS"])) \
        > 1e-3 * np.max(np.abs(want_bs))
    assert np.max(np.abs(amp * kap_recon * np.asarray(fin["redl"])
                         - st["j_BS"])) > 1e-2 * np.max(np.abs(want_bs))


def test_a_negative_residual_inductive_is_recorded_not_clipped(
        tmp_path, monkeypatch, capsys):
    """Force the archived bootstrap above the archived current near the
    edge: the residual inductive goes negative there and is ARCHIVED
    negative (the legacy archival would floor it at zero and move the
    sliver into j_BS); count, minimum and psi_N range are recorded and a
    console note printed."""
    stored = _spy_store(monkeypatch)
    real = ED.GenerateEngineDraws.post_hoc

    def post_hoc(self, *a, **k):
        out = real(self, *a, **k)
        sp = self._cur["final_split"]
        bump = np.zeros_like(sp["j_BS"])
        bump[-3:] = 1.0e9                       # far above any j_phi
        sp["j_BS"] = sp["j_BS"] + bump
        return out
    monkeypatch.setattr(ED.GenerateEngineDraws, "post_hoc", post_hoc)
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=1)
    st, sp = stored[0], diags[0]["engine"]["archived"]["split"]
    assert np.all(st["j_inductive"][-3:] < 0.0)
    assert sp["n_negative_inductive"] >= 3
    assert sp["min_inductive"] == float(np.min(st["j_inductive"])) < 0.0
    lo, hi = sp["negative_inductive_psi_N"]
    assert lo <= float(PSI[-3]) and hi == pytest.approx(float(PSI[-1]))
    np.testing.assert_array_equal(
        st["j_inductive"], st["j_phi"] - st["j_BS"]
        - np.asarray(sp["j_NBI"]) - np.asarray(sp["j_RF"])
        - np.asarray(diags[0]["j_pressure"]))


def test_the_post_homotopy_resplit_branch_uses_the_draws_own_fixed_parts(
        tmp_path, monkeypatch):
    """The post-homotopy check fails ONCE (a stand-in check), so the draw
    re-solves at the tight coil stage and generate_bouquet re-derives its
    split in the engine branch: j_inductive = j_phi - j_BS - the draw's own
    solved fixed parts, unclipped (not the legacy decomposition at the
    reconstruction's fixed parts) -- and the archival then re-splits on
    the archived state."""
    import bouquet.jbs_loop as JL
    real_check = JL.check_delivered
    state = dict(n=0)

    def check(*a, **k):
        out = real_check(*a, **k)
        cur = getattr(G_box.get("G"), "_cur", None)
        if cur is not None and cur["clock"].cur == "post_homotopy" \
                and state["n"] == 0:
            state["n"] += 1
            out = dict(out, ok=False)
        return out
    G_box = {}
    real_init = ED.GenerateEngineDraws.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        G_box["G"] = self
    monkeypatch.setattr(ED.GenerateEngineDraws, "__init__", init)
    monkeypatch.setattr(JL, "check_delivered", check)
    fixed_calls = []
    real_fixed = ED.GenerateEngineDraws.solved_fixed

    def solved_fixed(self):
        out = real_fixed(self)
        fixed_calls.append(np.array(out, dtype=float))
        return out
    monkeypatch.setattr(ED.GenerateEngineDraws, "solved_fixed", solved_fixed)
    import bouquet.TokaMaker_interface as TI
    legacy_calls = []
    real_dec = TI._decompose_draw_currents

    def dec(*a, **k):
        legacy_calls.append(1)
        return real_dec(*a, **k)
    monkeypatch.setattr(TI, "_decompose_draw_currents", dec)
    stored = _spy_store(monkeypatch)
    diags, rej, h, G = _generate(tmp_path, monkeypatch, n=1)
    assert state["n"] == 1 and rej == [] and len(diags) == 1
    ph = diags[0]["engine"]["post_homotopy"]
    assert ph["accepted_without_passes"] is False
    assert len(fixed_calls) == 1                 # the engine re-split ran
    assert legacy_calls == []                    # not the legacy one
    dp = G._cur["draw"]["passes_post_homotopy"]
    np.testing.assert_array_equal(fixed_calls[0],
                                  dp.last["parts"]["driven"])
    # archived on the archived state, the residual exact
    st, sp = stored[0], diags[0]["engine"]["archived"]["split"]
    np.testing.assert_array_equal(
        st["j_inductive"], st["j_phi"] - st["j_BS"]
        - np.asarray(sp["j_NBI"]) - np.asarray(sp["j_RF"])
        - np.asarray(diags[0]["j_pressure"]))


def test_generate_records_the_quantity_and_psi_n(tmp_path, monkeypatch,
                                                 capsys):
    """Through generate_bouquet: the attempt whose drawn T_i is negative is
    rejected as kinetics_nonphysical with the quantity, value and psi_N in
    its record; the other attempts are the draws they were (same sampler
    stream: the check consumes nothing)."""
    ref, _rej0, _h0, _G0 = _generate(tmp_path / "ref", monkeypatch, n=3)
    real = ED.sample_draw_inputs
    calls = dict(n=0)

    def sample(ctx, rng, unc, flux_integral, **kw):
        inp = real(ctx, rng, unc, flux_integral, **kw)
        calls["n"] += 1
        if calls["n"] == 2:
            inp.kinetics = {k: np.array(v, dtype=float, copy=True)
                            for k, v in inp.kinetics.items()}
            inp.kinetics["ti"][:4] = -1810.0
        return inp
    monkeypatch.setattr(ED, "sample_draw_inputs", sample)
    diags, rej, h, G = _generate(tmp_path / "bad", monkeypatch, n=3)
    assert len(diags) == 2 and len(rej) == 1
    r = rej[0]
    assert r["reason"] == "kinetics_nonphysical" and r["draw"] == 1
    assert r["stage"] == "perturb"
    assert r["error_type"] == "DrawKineticsNonPhysical"
    assert r["info"]["quantity"] == "ti"
    assert r["info"]["value"] == pytest.approx(-1810.0)
    assert r["info"]["psi_N"] == pytest.approx(float(PSI[0]))
    assert r["info"]["n_bad"] == 4
    assert "ti" in r["message"] and "psi_N" in r["message"]
    # attempts 1 and 3 are the reference batch's draws 1 and 3, bit for bit
    for mine, theirs in ((diags[0], ref[0]), (diags[1], ref[2])):
        np.testing.assert_array_equal(mine["j_BS"], theirs["j_BS"])
        assert mine["engine"]["delivered"]["l_i_3"] \
            == theirs["engine"]["delivered"]["l_i_3"]


# ---------------------------------------------------------------------------
#  verify_sigma0_consistency under the engine runs the generate() route
# ---------------------------------------------------------------------------
def test_the_sigma0_check_runs_the_generate_route(tmp_path,
                                                  toy_bouquet_solver,
                                                  monkeypatch):
    """The check is ONE draw through generate() (generate_bouquet with the
    engine draw, the homotopy and the post-homotopy stage), every
    perturbation zero and scale 1.0, archived into a temporary file; both
    stages are reported and both decide ``passed``; the configured archive,
    the config and the attributes generate() sets are untouched."""
    import bouquet.TokaMaker_interface as TI
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    calls = []
    real = TI.generate_bouquet

    def spy(mygs, psi_N, n_equils, header, *a, **k):
        calls.append(dict(n=n_equils, header=header,
                          scale=k.get("jBS_scale_range"),
                          engine=k.get("draw_method")))
        return real(mygs, psi_N, n_equils, header, *a, **k)

    monkeypatch.setattr(TI, "generate_bouquet", spy)
    h5 = b.config.output_header + ".h5"
    mtime = os.path.getmtime(h5) if os.path.exists(h5) else None
    b.draw_rejections = ["sentinel"]
    v = _quiet(b.verify_sigma0_consistency)
    assert len(calls) == 1 and calls[0]["n"] == 1
    assert calls[0]["scale"] == (1.0, 1.0)
    assert calls[0]["header"] != b.config.output_header
    G = calls[0]["engine"]
    assert G is not None and G.homotopy is True
    assert all(np.all(np.asarray(G.unc[k]) == 0.0) for k in
               ("sigma_ne", "sigma_te", "sigma_ni", "sigma_ti",
                "sigma_jphi"))
    assert v["route"] == "generate()" and v["passed"]
    st = v["stages"]
    assert st["loop"]["passed"] and st["loop"]["request_bit_identical"]
    assert st["archived"]["passed"]
    assert v["r_j"] == st["archived"]["r_j"]
    assert v["dl_i"] == st["archived"]["dl_i"]
    assert st["archived_coils"]["homotopy_pass"] is not None
    # the record is the archived draw's (homotopy and post-hoc included)
    assert "post_hoc" in v["record"]
    # nothing of the user's run was touched
    assert b.draw_rejections == ["sentinel"]
    assert (os.path.getmtime(h5) if os.path.exists(h5) else None) == mtime
    assert b.config.generation.n_inspec_target is None
    assert getattr(b, "_sigma0_route", None) is None


class _BlockOFT:
    """A meta-path finder refusing every ``OpenFUSIONToolkit*`` import -- the
    fast CI job's condition (OpenFUSIONToolkit not installed), reproduced
    in-process on a machine that has it."""

    def find_spec(self, name, path=None, target=None):
        if name == "OpenFUSIONToolkit" or name.startswith(
                "OpenFUSIONToolkit."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


@pytest.fixture()
def _without_oft(monkeypatch):
    import sys
    for name in [m for m in sys.modules if m == "OpenFUSIONToolkit"
                 or m.startswith("OpenFUSIONToolkit.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BlockOFT()] + sys.meta_path)
    with pytest.raises(ImportError):
        import OpenFUSIONToolkit  # noqa: F401
    with pytest.raises(ImportError):
        from OpenFUSIONToolkit.TokaMaker.bootstrap import \
            solve_with_bootstrap  # noqa: F401


def test_the_engine_sigma0_check_runs_without_openfusiontoolkit(
        tmp_path, toy_bouquet_solver, _without_oft):
    """CI: the fast suite runs without OpenFUSIONToolkit.  The engine
    route's sigma=0 check (generate() on the toy solver) must not import it:
    ``verify_sigma0_consistency`` takes OpenFUSIONToolkit only on the legacy
    route, the one that calls ``solve_with_bootstrap``.  Likewise
    ``evaluate_jBS``'s input refusals (the domain
    ``engine_draws.check_draw_kinetics`` mirrors) come before its
    OpenFUSIONToolkit import."""
    from bouquet.physics import JBSEvaluationError, evaluate_jBS
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    v = _quiet(b.verify_sigma0_consistency)
    assert v["route"] == "generate()" and v["passed"]
    n = 9
    psi = np.linspace(0.0, 1.0, n)
    ne = np.full(n, 3e19)
    ne[4] = 0.0
    with pytest.raises(JBSEvaluationError) as ei:
        evaluate_jBS(None, psi, ne, np.full(n, 1e3), np.full(n, 2.5e19),
                     np.full(n, 1.2e3), np.full(n, 1.8), psi_pad=1e-3)
    assert ei.value.quantity == "ne"
    # past the input checks the evaluation does need OpenFUSIONToolkit
    with pytest.raises(ImportError):
        evaluate_jBS(None, psi, np.full(n, 3e19), np.full(n, 1e3),
                     np.full(n, 2.5e19), np.full(n, 1.2e3), np.full(n, 1.8),
                     psi_pad=1e-3)


def test_the_sigma0_check_puts_all_solver_state_back(tmp_path,
                                                    toy_bouquet_solver,
                                                    monkeypatch):
    """The one-way coil-bound mode is entered ONCE, at ``setup_solver``
    (``bouquet.solver_state.enter_bounded_coil_mode``), BEFORE the
    reconstruction's first solve -- so the reconstruction, the check and the
    draws run one coil solve -- and recorded on the Baseline and the engine
    record.  The check itself makes no entry; afterwards the solver object
    carries exactly what it carried before: no coil-regularisation stash
    left by the route's generate(), the settings and the equilibrium object
    put back, the recorded bounds re-installed."""
    from bouquet.solver_state import COIL_SOLVE_BOUNDED, coil_solve_mode
    b = _bq(tmp_path)
    events = []
    import _engine_fake_gs as _fg
    real_bounds = _fg.FakeTokaMaker.set_coil_bounds
    real_solve = _fg.FakeTokaMaker.solve

    def set_coil_bounds(self, bnd):
        events.append(("bounds", bnd))
        return real_bounds(self, bnd)

    def solve(self, *a, **k):
        events.append(("solve",))
        return real_solve(self, *a, **k)

    monkeypatch.setattr(_fg.FakeTokaMaker, "set_coil_bounds",
                        set_coil_bounds)
    monkeypatch.setattr(_fg.FakeTokaMaker, "solve", solve)
    b.setup_solver()
    fake = b.mygs
    assert events == [("bounds", None)]           # entered at setup
    assert coil_solve_mode(fake) == COIL_SOLVE_BOUNDED
    _quiet(b.prepare_baseline)
    assert ("bounds", None) not in events[1:]     # ... and only there
    assert b.baseline.coil_solve_mode == COIL_SOLVE_BOUNDED
    assert b.baseline.engine["coil_solve_mode"] == COIL_SOLVE_BOUNDED
    del events[:]
    assert not hasattr(fake, "_strong_coil_reg")
    maxits0 = fake.settings.maxits
    eq0 = fake.copy_eq()
    v = _quiet(b.verify_sigma0_consistency)
    assert v["passed"]
    assert events[0] == ("solve",)                # no entry in the check
    assert events[-1] == ("bounds", None)         # the recorded bounds back
    assert coil_solve_mode(fake) == COIL_SOLVE_BOUNDED
    assert not hasattr(fake, "_strong_coil_reg")  # generate()'s stash gone
    assert fake.settings.maxits == maxits0
    import pickle
    assert pickle.dumps(fake.copy_eq()) == pickle.dumps(eq0)


def test_a_rejected_sigma0_draw_fails_the_check(tmp_path,
                                                toy_bouquet_solver,
                                                monkeypatch):
    """The route's homotopy fails at its first stage: the zero-perturbation
    draw is rejected, so the check FAILS with the rejection -- the old
    loop-only check could not see it."""
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    fake = b.mygs
    real = type(fake).solve

    def solve(self, *a, **k):
        G = getattr(b, "_sigma0_route", None) or {}
        G = G.get("draws")
        if G is not None and G._cur is not None and \
                G._cur["clock"].cur == "homotopy":
            raise ValueError("Error in solve: Non-finite value (NaN/Inf) "
                             "in solution")
        return real(self, *a, **k)

    monkeypatch.setattr(type(fake), "solve", solve)
    v = _quiet(b.verify_sigma0_consistency)
    assert v["passed"] is False
    assert v["rejection"]["reason"] == "homotopy_infeasible"
    assert v["stages"]["loop"]["passed"] is True   # the loop alone passes
    assert v["stages"]["archived"] is None
    # the loop-only function (kept) cannot see it
    ctx = ED.context_from_run(b._engine_run, b.config.generation,
                              b.baseline)
    assert _quiet(ED.verify_zero_perturbation, ctx, b.mygs.toy)["passed"]


@pytest.mark.parametrize("what", ["bootstrap", "l_i"])
def test_an_archived_stage_miss_fails_the_sigma0_check(tmp_path,
                                                       toy_bouquet_solver,
                                                       monkeypatch, what):
    """The archived-stage gate of the sigma=0 check (ac74b62; the 2026-10-06
    review's mutant Z3 deleted it with every test green): the LOOP stage
    passes and the draw is not rejected, but the state the draw ARCHIVES
    misses the reconstruction -- its bootstrap 1 % off lambda_BS* (r_j
    1e-3), or its l_i off by twice tol_li -- so the check FAILS."""
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    real = ED.zero_perturbation_archived_verdict

    def verdict(ctx, jbs_carried, fin):
        if what == "bootstrap":
            jbs_carried = 1.01 * np.asarray(jbs_carried, dtype=float)
        else:
            fin = dict(fin, li=float(fin["li"]) + 2.0 * ctx.loop["tol_li"])
        return real(ctx, jbs_carried, fin)
    monkeypatch.setattr(ED, "zero_perturbation_archived_verdict", verdict)
    v = _quiet(b.verify_sigma0_consistency)
    assert "rejection" not in v
    assert v["stages"]["loop"]["passed"] is True
    assert v["stages"]["archived"]["passed"] is False
    assert v["passed"] is False
    if what == "bootstrap":
        assert v["r_j"] > v["tolerances"]["rtol_j"]
    else:
        assert abs(v["dl_i"]) > v["tolerances"]["tol_li"]
    # the control: unmodified, the same check passes
    monkeypatch.setattr(ED, "zero_perturbation_archived_verdict", real)
    assert _quiet(b.verify_sigma0_consistency)["passed"] is True


def test_a_legacy_split_is_archived_with_p_g_as_j_pressure(tmp_path,
                                                          monkeypatch):
    """Owner decision D2 on the LEGACY archive writers (integration hook,
    B.md item 3): a draw whose method keeps p'G inside its in-memory
    inductive (the legacy convention: no diagnostics["j_pressure"]) is
    archived with p'G -- evaluated on the archived state
    (TokaMaker_interface.archived_pressure_term) -- taken off j_inductive
    and stored as j_pressure + current_split_convention; the _baseline the
    same way.  The engine method stands in for the legacy one (its split
    handed back in the legacy form); the evaluation is a known profile."""
    import h5py
    import bouquet.TokaMaker_interface as TI
    from bouquet.schema import (CURRENT_SPLIT_CONVENTION_ATTR,
                                SPLIT_PRESSURE_SEPARATE)
    from bouquet.utils import _group_path, _resolve_h5
    Pk = 1.0e4 * (1.0 - PSI ** 2)
    calls = []

    def fake_term(mygs, psi_N, psi_pad=1e-3, coord="psi_n", what=""):
        calls.append(what)
        return Pk.copy()
    monkeypatch.setattr(TI, "archived_pressure_term", fake_term)
    real = ED.GenerateEngineDraws.archived_split
    legacy_ind = []

    def legacy_like(self, diagnostics, j_phi, default=None):
        j_bs, j_ind = real(self, diagnostics, j_phi, default)
        P = diagnostics.pop("j_pressure")
        legacy_ind.append(j_ind + P)          # p'G in the inductive
        return j_bs, legacy_ind[-1]
    monkeypatch.setattr(ED.GenerateEngineDraws, "archived_split",
                        legacy_like)
    stored = _spy_store(monkeypatch)
    diags, rej, h, G = _generate(
        tmp_path, monkeypatch, n=1,
        baseline_split=dict(j_pressure=None,
                            inductive_includes_pressure=True))
    assert len(diags) == 1 and rej == [] and len(stored) == 1
    assert calls == ["baseline archive", "draw 0 archive"]
    st = stored[0]
    np.testing.assert_array_equal(st["j_inductive"], legacy_ind[0] - Pk)
    with h5py.File(_resolve_h5(h), "r") as hf:
        for path in (_group_path(None, st["count"]), "_baseline"):
            g = hf[path]
            np.testing.assert_array_equal(g["j_pressure"][()], Pk)
            assert g.attrs[CURRENT_SPLIT_CONVENTION_ATTR] == \
                SPLIT_PRESSURE_SEPARATE
        # the baseline's inductive (this stand-in archives the target split:
        # input_jinductive = 0.5 x the request, which carries p'G by
        # inductive_includes_pressure=True), minus p'G
        np.testing.assert_array_equal(hf["_baseline"]["j_inductive"][()],
                                      0.5 * G.ctx.request - Pk)


def test_without_the_run_convention_a_direct_call_archives_as_before(
        tmp_path, monkeypatch):
    """A direct generate_bouquet call without baseline_split never
    evaluates p'G for a legacy split (its archive keeps the pre-#64
    convention, which readers infer from the absent attr)."""
    import bouquet.TokaMaker_interface as TI
    calls = []
    monkeypatch.setattr(TI, "archived_pressure_term",
                        lambda *a, **k: calls.append(1))
    real = ED.GenerateEngineDraws.archived_split

    def legacy_like(self, diagnostics, j_phi, default=None):
        out = real(self, diagnostics, j_phi, default)
        diagnostics.pop("j_pressure")
        return out
    monkeypatch.setattr(ED.GenerateEngineDraws, "archived_split",
                        legacy_like)
    _generate(tmp_path, monkeypatch, n=1)
    assert calls == []
