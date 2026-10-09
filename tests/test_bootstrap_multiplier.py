"""The draws apply the baseline's bootstrap multiplier after SWB.

In ohmic mode the baseline is ``bl.j_BS = m * SWB(scale 1)``, with ``m`` the
closure's ``bs_scale`` or the structured ``s_bs(psi)``.  OFT applies
``scale_jBS`` inside SWB's self-consistent iteration, so
``SWB(m) != m * SWB(1)``: passing ``m`` into SWB as ``scale_jBS`` missed
``bl.j_BS`` by 8.9 % of peak at the pedestal at ``bs_scale`` 0.77 (one DIII-D
slice), and a structured profile cannot be passed at all.
"""
import sys
import types

import numpy as np
import pytest

import _swb
from bouquet.run import Bouquet
from bouquet.sampling import make_rng
from bouquet.TokaMaker_interface import (perturb_kinetic_equilibrium,
                                         smooth_jbs_transition)
from test_sigma0_identity_stages import (_EC, _IP, _N, _NE, _NI, _PAD, _TE,  # noqa: F401
                                         _TI, _X, _ZEFF, _li, _reconstruct, toy)


def _mult(bs=1.0, prof=None, n=5):
    bl = types.SimpleNamespace(bs_scale=bs, bs_scale_profile=prof,
                               psi_N=np.linspace(0, 1, n))
    return Bouquet._bootstrap_multiplier(types.SimpleNamespace(baseline=bl))


class TestMultiplier:
    def test_unit_scale_is_none(self):
        """diff mode and the geqdsk path: nothing changes."""
        assert _mult(1.0) is None

    def test_scalar_scale_is_a_flat_profile(self):
        np.testing.assert_array_equal(_mult(0.77), np.full(5, 0.77))

    def test_the_structured_profile_wins(self):
        prof = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        np.testing.assert_array_equal(_mult(0.95, prof), prof)


def _fake_swb(calls):
    """SWB whose bootstrap feeds back on its own total, so ``scale_jBS``
    enters nonlinearly, as in OFT's self-consistent iteration."""
    def swb(mygs, ne, te, ni, ti, Zeff, Ip, seed, scale_jBS=1.0, x=None,
            **kw):
        calls.append(scale_jBS)
        b = ((3.0e5 * np.exp(-0.5 * ((_X - 0.92) / 0.04) ** 2)
              + 6.0e4 * (1.0 - _X)) * (ne * te) / (ne[0] * te[0]))
        j = b.copy()
        for _ in range(200):
            j = scale_jBS * b * (1.0 + 0.4 * np.trapezoid(j, _X)
                                 / np.trapezoid(b, _X))
        return {"j_BS": j, "isolated_j_BS": j, "j_inductive": seed,
                "total_j_phi": seed + j, "scale_j0": 1.0, "scale_Ip": 1.0}
    return swb


@pytest.mark.parametrize("mult", [np.full(_N, 0.77), 0.8 + 0.4 * _X],
                         ids=["bs_scale", "s_bs_profile"])
def test_the_sigma0_draw_reproduces_the_baseline_bootstrap(toy, monkeypatch,
                                                           mult):
    calls = []
    swb = _fake_swb(calls)
    _swb.fake_oft(monkeypatch)
    monkeypatch.setattr(sys.modules["OpenFUSIONToolkit.TokaMaker.bootstrap"],
                        "solve_with_bootstrap", swb)
    req, _jbs, _fx = _reconstruct(toy)
    bl_jbs = mult * smooth_jbs_transition(
        swb(None, _NE, _TE, _NI, _TI, _ZEFF, _IP, _X)["j_BS"])
    z = np.zeros(_N)
    d = perturb_kinetic_equilibrium(
        toy, _X, _EC * (_NE * _TE + _NI * _TI), _NE, _TE, _NI, _TI,
        req, z, z, z, z, z, 0.5, 0.4, 0.25, _IP, _li(toy.achieved), _ZEFF,
        _N, input_jinductive=req - bl_jbs, l_i_tolerance=0.05, psi_pad=_PAD,
        constrain_sawteeth=False, recalculate_j_BS=True,
        isolate_edge_jBS=False, scale_jBS=1.0, floor_j_BS=False,
        jBS_scale_profile=mult, max_proxy_draws=5, p_thresh=0.05,
        rng=make_rng(7))[6]
    miss = np.max(np.abs(d["j_BS"] - bl_jbs)) / np.max(np.abs(bl_jbs))
    assert miss <= 1e-12, miss
    assert calls[1:] and all(s == 1.0 for s in calls[1:])
