"""The sigma=0 check's ``draw_route`` block -- fast half (mocked draw).

``verify_sigma0_consistency`` under the loop keeps its own check (and its
``passed``) exactly as it was; BESIDE it, ``draw_route`` runs the draw's own
code path -- :func:`perturb_kinetic_equilibrium` with the arguments
``generate()`` hands a draw -- at zero perturbation and reports it against the
baseline at the loop's own tolerances, gating nothing.  Here the draw itself is
a mock that records its arguments; the live run is in
``test_jbs_loop_solver.py`` (``pytest -m solver``).

Synthetic inputs only; no device data.
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.jbs_loop import jbs_settings

_X = np.linspace(0.0, 1.0, 33)
_XK = np.linspace(0.0, 1.05, 40)


def _jbs():
    return 3.0e5 * np.exp(-0.5 * ((_X - 0.93) / 0.03) ** 2) + 1.0e5 * (1 - _X)


class _Eq:
    def __init__(self, li):
        self.li = li
        self.replaced = []
        self.psi_bounds = (-0.1, 0.1)

    def copy_eq(self):
        return ("snap", self.li)

    def replace_eq(self, source_eq=None):
        self.replaced.append(source_eq)

    def get_stats(self, **kw):
        return {"l_i": self.li}


def _bouquet(monkeypatch, li_draw, perturb_jind_in_anchor=False,
             spike_scale=1.0, converged=True):
    import bouquet.baseline as B
    import bouquet.jbs_loop as L
    import bouquet.TokaMaker_interface as TI
    from bouquet.config import GenerationConfig
    from bouquet.run import Bouquet

    gc = GenerationConfig()
    gc.perturb_jind_in_anchor = perturb_jind_in_anchor
    gc.seed = 7
    bl = SimpleNamespace(
        Ip_target=1.0e6, l_i_target=0.80, bs_scale=1.02, psi_N=_X,
        psi_N_kinetic=_XK, ne=np.full(_XK.size, 5e19), te=np.full(_XK.size,
                                                                 2e3),
        ni=np.full(_XK.size, 4.5e19), ti=np.full(_XK.size, 2e3),
        j_phi=2.0e6 * (1 - _X) ** 2 + _jbs(), j_inductive=2.0e6 * (1 - _X) ** 2,
        j_BS=_jbs(), jBS_diff=None, p_fast=None, j_NBI=None, j_RF=None)
    b = Bouquet.__new__(Bouquet)
    b.baseline = bl
    b.mygs = _Eq(li_draw)
    b.config = SimpleNamespace(generation=gc)
    calls = []
    monkeypatch.setattr(B, "resolve_uncertainty", lambda cfg, bl_: dict(
        n_ls=0.5, t_ls=0.4, j_ls=0.25, aux_sigmas={"zeff": np.ones(_XK.size)},
        aux_baselines={"zeff": 2.0 * np.ones(_XK.size)},
        aux_length_scales={"zeff": 0.4}))
    monkeypatch.setattr(L, "residual_weights",
                        lambda eq, psi_N, psi_pad=1e-3, coord="psi_n": (np.ones_like(_X), _X,
                                                         "test"))

    def _perturb(*a, **k):
        calls.append((a, k))
        d = dict(j_BS=_jbs(), j_BS_edge=None, r2_ip_scale=(
            1.0004 if k["perturb_jind_in_anchor"] else None),
            r2_f_ind=0.7, j0_scales=[1.001],
            jbs_loop=dict(converged=converged, n_loops=1, n_passes_total=3),
            _jbs_ctx=dict(spike_used=spike_scale * _jbs()))
        return (None,) * 6 + (d,)

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _perturb)
    return b, calls


def _run(b, routes=None):
    from bouquet.TokaMaker_interface import sigma0_reference_scale  # noqa
    s = jbs_settings(b.config.generation, draw=True)
    return b._sigma0_draw_route(
        s, ("entry", 0), np.full(_X.size, 1e4), np.full(_X.size, 5e19),
        np.full(_X.size, 2e3), np.full(_X.size, 4.5e19),
        np.full(_X.size, 2e3), np.full(_X.size, 1.5), _X, 1e-3,
        np.asarray(b.baseline.j_BS, float), 0.803,
        "reconstruction li_realized_post_corrective", routes=routes)


def test_the_draw_route_is_perturb_kinetic_equilibrium_at_zero_sigma(
        monkeypatch):
    from bouquet.TokaMaker_interface import sigma0_reference_scale
    b, calls = _bouquet(monkeypatch, li_draw=0.8002)
    blk = _run(b)
    assert len(calls) == 1
    a, k = calls[0]
    # every sigma zero: kinetic (kinetic grid), j_phi (equilibrium grid), aux
    for arr in a[8:12]:
        assert arr.shape == _XK.shape and not np.any(arr)
    assert a[12].shape == _X.shape and not np.any(a[12])
    assert not np.any(k["aux_sigmas"]["zeff"])
    # generate()'s arguments: route, centre scale, generate_bouquet defaults,
    # the loop settings, a seeded Generator, the baseline inductive
    assert k["perturb_jind_in_anchor"] is False
    gc = b.config.generation
    assert k["scale_jBS"] == pytest.approx(sigma0_reference_scale(
        (gc.jBS_scale_range[0] * 1.02, gc.jBS_scale_range[1] * 1.02)))
    assert k["p_thresh"] == 0.05 and k["max_proxy_draws"] == 500
    assert k["jbs_loop"]["enabled"] and isinstance(k["rng"],
                                                   np.random.Generator)
    np.testing.assert_array_equal(k["input_jinductive"],
                                  b.baseline.j_inductive)
    # the replay started from the state the check was handed
    assert b.mygs.replaced and b.mygs.replaced[0] == ("entry", 0)
    rr = blk["routes"]["standard"]
    assert rr["passed_draw_route"] and blk["passed_draw_route"]
    assert rr["dl_i_vs_l_i_target"] == pytest.approx(2e-4)
    assert rr["dl_i_vs_delivered"] == pytest.approx(2.8e-3)
    assert rr["passes_used"] == 3 and rr["j0_scales"] == [1.001]
    # RE-SCOPED BY THE OWNER'S DECISION (the zero-perturbation identity of
    # the draws): this block used to gate nothing; it now decides
    # verify_sigma0_consistency's `passed` (test_the_draw_route_decides_...).
    assert blk["gates"].startswith("verify_sigma0_consistency's `passed`")


def test_the_configured_ip_renormalising_route_and_both_on_request(
        monkeypatch):
    b, calls = _bouquet(monkeypatch, li_draw=0.8002,
                        perturb_jind_in_anchor=True)
    blk = _run(b)
    assert list(blk["routes"]) == ["ip_renorm"]
    assert calls[0][1]["perturb_jind_in_anchor"] is True
    assert blk["routes"]["ip_renorm"]["r2_ip_scale"] == 1.0004
    b2, calls2 = _bouquet(monkeypatch, li_draw=0.8002)
    blk2 = _run(b2, routes=("standard", "ip_renorm"))
    assert [c[1]["perturb_jind_in_anchor"] for c in calls2] == [False, True]
    assert set(blk2["routes"]) == {"standard", "ip_renorm"}


@pytest.mark.parametrize("kw, why", [
    (dict(li_draw=0.8020), "l_i 2e-3 off l_i_target (tol_li 1e-3)"),
    (dict(li_draw=0.8002, spike_scale=1.01), "bootstrap 1 % off (rtol_j)"),
    (dict(li_draw=0.8002, converged=False), "loop not converged")])
def test_the_verdict_uses_the_loops_own_unchanged_tolerances(monkeypatch, kw,
                                                             why):
    b, calls = _bouquet(monkeypatch, **kw)
    blk = _run(b)
    assert blk["passed_draw_route"] is False, why
    assert blk["tolerances"] == dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3)


def test_a_raising_draw_route_is_recorded_not_propagated(monkeypatch):
    import bouquet.TokaMaker_interface as TI
    b, calls = _bouquet(monkeypatch, li_draw=0.8)

    def _boom(*a, **k):
        raise RuntimeError("no in-band draw")

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _boom)
    blk = _run(b)
    rr = blk["routes"]["standard"]
    assert rr["passed_draw_route"] is False
    assert "no in-band draw" in rr["error"]


def test_the_existing_check_and_its_passed_are_untouched():
    """The loop check "solved the baseline's way" keeps its own conjunction,
    computed before the draw route runs, under its own name.

    RE-SCOPED BY THE OWNER'S DECISION (the zero-perturbation identity of the
    draws): that conjunction used to BE ``passed``; it is now the separately
    named ``passed_baseline_way`` and no longer decides ``passed`` (which the
    draw route does -- see the next test).  Every check below is the one this
    test always made, on the renamed result."""
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet._verify_sigma0_jbs_loop)
    passed_expr = src.split("passed_baseline_way = bool(", 1)[1].split(
        ")\n", 1)[0]
    assert "draw_route" not in passed_expr
    for term in ('res["converged"]', 'cmp_["r_j"] <= settings["rtol_j"]',
                 'cmp_["r_I"] <= settings["rtol_Ip"]',
                 'dli <= settings["tol_li"]'):
        assert term in passed_expr
    assert src.index("passed_baseline_way = bool(") < src.index(
        'out["draw_route"]')


def test_the_draw_route_decides_passed():
    """With the loop on, ``passed`` REQUIRES the draw's own route(s) to
    reproduce the reconstruction state; not running them leaves it False."""
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet._verify_sigma0_jbs_loop)
    assert 'out["passed"] = bool(out["draw_route"].get("passed_draw_route"))' \
        in src
    assert "passed=False, passed_baseline_way=passed_baseline_way" in src
    # the default routes: every route the configuration can use
    assert "routes=draw_routes" in src
    dsrc = inspect.getsource(Bouquet._sigma0_draw_route)
    assert "routes = self._sigma0_draw_routes()" in dsrc


@pytest.mark.parametrize("src, jind, custom, want", [
    ("recon", False, False, ("standard", "ip_renorm")),
    ("recon", True, False, ("ip_renorm", "standard")),
    ("imas", True, False, ("ip_renorm",)),
    ("imas", True, True, ("ip_renorm", "standard"))])
def test_default_routes_are_every_route_the_configuration_can_use(
        src, jind, custom, want):
    """The g-file path runs either draw route; the modelling-source path
    refuses the standard one unless the workflow is ``custom``."""
    from bouquet.config import (GenerationConfig, ImasSource,
                                ReconstructionSource)
    from bouquet.run import Bouquet
    gc = GenerationConfig()
    gc.perturb_jind_in_anchor = jind
    if custom:
        gc.workflow = "custom"
    source = (ReconstructionSource.__new__(ReconstructionSource)
              if src == "recon" else ImasSource.__new__(ImasSource))
    b = Bouquet.__new__(Bouquet)
    b.config = SimpleNamespace(generation=gc, source=source)
    assert b._sigma0_draw_routes() == want
