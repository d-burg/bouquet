"""Review PR64 B1 at the LEGACY solve_with_bootstrap call sites (integration
hook): every SWB result is converted to the field-aligned toroidal bootstrap
``kappa <j.B>`` right after the call (TokaMaker_interface.
swb_result_toroidal), for the installed toolkit's output convention, and the
conversion is stamped.  Synthetic geometry; no solver."""
import numpy as np
import pytest

import bouquet.physics as P
import bouquet.TokaMaker_interface as TI
from bouquet import coords

N = 17
X = np.linspace(0.0, 1.0, N)


class _Geo:
    """get_profiles / get_q / sauter_fc on any grid, with a finite p'."""

    def _x(self, psi=None, npsi=None, psi_pad=None):
        return (np.asarray(psi, float) if psi is not None
                else np.linspace(0.0, 1.0, int(npsi)))

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        x = self._x(psi, npsi)
        F = 3.4 - 0.2 * x
        return (x, F, -0.05 * np.ones_like(x), np.ones_like(x),
                -2.0e4 * (1.0 - x))

    def get_q(self, psi=None, npsi=None, psi_pad=None):
        x = self._x(psi, npsi)
        ravgs = {"<R>": 1.7 + 0.05 * x, "<1/R>": 1.0 / (1.7 + 0.04 * x)}
        return (x, 1.0 + 3.0 * x, ravgs, None, None, None)

    def sauter_fc(self, psi=None, npsi=None, psi_pad=None):
        x = self._x(psi, npsi)
        B2 = 4.2 + 0.3 * x
        return (None, None, {"<|B|>": np.sqrt(B2), "<|B|^2>": B2})


def _raw():
    j = 2.0e5 * np.exp(-0.5 * ((X - 0.95) / 0.03) ** 2) + 5.0e4 * (1 - X)
    return {"jphi_bs_raw": j, "jphi_bs": j.copy(), "jphi_ind": 1e6 * (1 - X)}


@pytest.fixture()
def uniform(monkeypatch):
    """A toolkit whose SWB takes no grid (OFT's uniform grid)."""
    monkeypatch.setattr(coords, "_swb_params", lambda: frozenset())


def test_the_upstream_projection_is_undone_and_converted_with_kappa(
        uniform, monkeypatch):
    monkeypatch.setattr(P, "swb_jbs_convention",
                        lambda *a, **k: P.SWB_JBS_RAVG_OVER_F)
    g = _Geo()
    raw = _raw()
    out = TI.swb_result_toroidal(g, raw, X, scale_jbs=0.8)
    _, F, _, _, _ = g.get_profiles(npsi=N)
    rav = g.get_q(npsi=N)[2]
    B2 = g.sauter_fc(npsi=N)[-1]["<|B|^2>"]
    # the 6116d5f net factor F^2 <1/R> / (<R> <B^2>), linear: scale-free
    net = F ** 2 * rav["<1/R>"] / (rav["<R>"] * B2)
    np.testing.assert_allclose(out["jphi_bs_raw"], net * raw["jphi_bs_raw"], rtol=1e-13)
    np.testing.assert_array_equal(out["swb_raw_jphi_bs_raw"], raw["jphi_bs_raw"])
    np.testing.assert_array_equal(out["jphi_ind"], raw["jphi_ind"])
    assert out["swb_conversion"] == P.swb_conversion_record(
        P.SWB_JBS_RAVG_OVER_F)
    assert raw["jphi_bs_raw"] is not out["jphi_bs_raw"]          # input untouched


@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_a_tokamaker_jphi_toolkit_loses_p_g_at_unit_scale(uniform,
                                                          monkeypatch, scale):
    monkeypatch.setattr(P, "swb_jbs_convention",
                        lambda *a, **k: P.SWB_JBS_TOROIDAL)
    g = _Geo()
    PG = P.swb_pressure_term(g, N, 1e-3, None)
    assert np.max(np.abs(PG)) > 0.0
    raw = _raw()
    # this toolkit returns scale * (kappa<j.B> + p'G)
    kjb = raw["jphi_bs_raw"]
    raw = dict(raw, jphi_bs_raw=scale * (kjb + PG), jphi_bs=scale * (kjb + PG))
    out = TI.swb_result_toroidal(g, raw, X, scale_jbs=scale)
    np.testing.assert_allclose(out["jphi_bs_raw"], scale * kjb, rtol=1e-12,
                               atol=1e-9 * np.max(np.abs(kjb)))


@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_a_toolkit_reporting_its_p_g_apart_is_taken_as_it_splits(uniform, monkeypatch, scale,
                                                                isolate):
    """A toolkit returning the field-aligned bootstrap and p'G apart (OFT a9d0c32+: jphi_pd,
    tapered, never scaled): its bootstraps are kept, isolated ones too, with no geometry pass,
    and the legacy inductive carries the p'G."""
    monkeypatch.setattr(P, "swb_jbs_convention", lambda *a, **k: P.SWB_JBS_TOROIDAL)
    kjb, pg, taper = 1e5 * (1.0 - X), 2e4 * X, np.where(X > 0.9, 0.5, 1.0)
    raw = {"jphi_bs_raw": kjb, "jphi_bs": scale * taper * kjb, "jphi_pd_raw": pg,
           "jphi_pd": taper * pg, "jphi_total": np.ones(N), "jphi_ind": np.ones(N)}
    out = TI.swb_result_toroidal(None, raw, X, scale_jbs=scale, isolate_edge_jbs=isolate)
    np.testing.assert_array_equal(out["jphi_bs_raw"], kjb)
    np.testing.assert_array_equal(out["jphi_bs"], scale * taper * kjb)
    np.testing.assert_array_equal(out["jphi_ind"], 1.0 + taper * pg)


def test_an_isolated_spike_on_a_tokamaker_jphi_toolkit_is_refused(
        uniform, monkeypatch):
    monkeypatch.setattr(P, "swb_jbs_convention",
                        lambda *a, **k: P.SWB_JBS_TOROIDAL)
    with pytest.raises(RuntimeError, match="isolate_edge_jbs"):
        TI.swb_result_toroidal(_Geo(), _raw(), X, isolate_edge_jbs=True)


def test_an_unknown_toolkit_is_refused_not_guessed(uniform, monkeypatch):
    def _unknown(*a, **k):
        raise P.SwbConventionUnknown("unverified toolkit")
    monkeypatch.setattr(P, "swb_jbs_convention", _unknown)
    with pytest.raises(P.SwbConventionUnknown):
        TI.swb_result_toroidal(_Geo(), _raw(), X)


def test_the_archived_pressure_term_is_p_g_of_the_held_state():
    """B.md item 3: the legacy archive's jphi_pd is p'(<R> -
    F^2<1/R>/<B^2>) of the state the group archives, positive frame."""
    g = _Geo()
    got = TI.archived_pressure_term(g, X, 1e-3, "psi_n")
    want = P.swb_pressure_term(g, N, 1e-3, X)
    np.testing.assert_array_equal(got, want)
    assert np.max(np.abs(want)) > 0.0
    # positive frame: the state's own jphi <R>p' + <1/R>FF'/mu0 is > 0 here
    # with p' < 0, so p' is flipped -- p'G has the sign of (<R> - ...)
    Xc = np.clip(X, 1e-3, 1.0 - 1e-3)     # SWB's padded surfaces
    _, F, Fp, _, pp = g.get_profiles(psi=Xc)
    rav = g.get_q(psi=Xc)[2]
    jeq = rav["<R>"] * pp + rav["<1/R>"] * F * Fp / (4e-7 * np.pi)
    s = 1.0 if np.sum(jeq) >= 0 else -1.0
    B2 = g.sauter_fc(psi=Xc)[-1]["<|B|^2>"]
    np.testing.assert_allclose(
        got, s * pp * (rav["<R>"] - F ** 2 * rav["<1/R>"] / B2),
        rtol=1e-12, atol=0.0)


def test_an_unevaluable_state_keeps_the_old_convention_loudly():
    class _NoGeo:
        pass
    with pytest.warns(RuntimeWarning, match="pre-#64 convention"):
        assert TI.archived_pressure_term(_NoGeo(), X, what="draw 3") is None


# ---------------------------------------------------------------------------
#  review of #78 (B3): the edge taper multiplies every toroidal component the
#  TokaMaker-jphi toolkit returns, p'G included; the p'G taken off must carry
#  the same weight, or jphi_bs is short by (1-w)P in the taper zone
# ---------------------------------------------------------------------------
_TAPER = {"taper_edge_jphi": True, "taper_edge_psi0": 0.8}


@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_a_tapered_tokamaker_jphi_output_loses_the_tapered_p_g(
        uniform, monkeypatch, scale):
    monkeypatch.setattr(P, "swb_jbs_convention",
                        lambda *a, **k: P.SWB_JBS_TOROIDAL)
    g = _Geo()
    PG = P.swb_pressure_term(g, N, 1e-3, None)
    w = P.edge_taper_weight(X, 0.8)
    zone = (w > 0.0) & (w < 1.0)
    assert zone.sum() >= 2 and np.max(np.abs(PG[zone])) > 0.0
    kjb = _raw()["jphi_bs_raw"]
    tapered = scale * w * (kjb + PG)          # what the toolkit returns
    raw = dict(_raw(), jphi_bs_raw=tapered, jphi_bs=tapered.copy())
    out = TI.swb_result_toroidal(g, raw, X, scale_jbs=scale,
                                 bootstrap_kwargs=dict(_TAPER, iterations=2))
    tol = 1e-12 * np.max(np.abs(kjb))
    # the identity in the taper zone (and everywhere): jphi_bs_raw = w kappa<j.B>,
    # and jphi_bs_raw + w p'G is the toolkit's output
    np.testing.assert_allclose(out["jphi_bs_raw"], scale * w * kjb, rtol=0,
                               atol=tol)
    np.testing.assert_allclose(out["jphi_bs_raw"] + scale * w * PG, tapered,
                               rtol=0, atol=tol)
    # the untapered subtraction this replaces was short by (1-w)P there
    wrong = TI.swb_result_toroidal(g, raw, X, scale_jbs=scale)["jphi_bs_raw"]
    np.testing.assert_allclose(scale * w * kjb - wrong,
                               scale * (1.0 - w) * PG, rtol=0, atol=tol)
    # taper off (absent or False): unchanged
    for kw in (None, {}, {"taper_edge_jphi": False}):
        np.testing.assert_array_equal(
            TI.swb_result_toroidal(g, raw, X, scale_jbs=scale,
                                   bootstrap_kwargs=kw)["jphi_bs_raw"], wrong)


def test_the_swb_split_tapers_p_g_with_the_bootstrap(uniform, monkeypatch):
    """The swb method's split (SwbBaseline._swb_split) with
    swb_edge_taper_psi0: jphi_pd = w p'G, jphi_bs_raw = w kappa<j.B>, and the
    inductive is SWB's own -- nothing of p'G moves into it."""
    from types import SimpleNamespace

    from bouquet.swb import SwbBaseline
    monkeypatch.setattr(P, "swb_jbs_convention",
                        lambda *a, **k: P.SWB_JBS_TOROIDAL)
    g = _Geo()
    PG = P.swb_pressure_term(g, N, 1e-3, None)
    w = P.edge_taper_weight(X, 0.8)
    kjb = _raw()["jphi_bs_raw"]
    raw_ind = 1e6 * (1 - X) * w
    tapered = w * (kjb + PG)
    res = {"jphi_bs_raw": tapered, "jphi_bs": tapered.copy(),
           "jphi_ind": raw_ind}
    me = SimpleNamespace(mygs=g, config=SimpleNamespace(
        generation=SimpleNamespace(isolate_edge_jbs=False,
                                   swb_edge_taper_psi0=0.8,
                                   bootstrap_kwargs={})))
    out = SwbBaseline._swb_split(me, res, None, X, "psi_n")
    tol = 1e-12 * np.max(np.abs(kjb))
    np.testing.assert_allclose(out["jphi_pd"], w * PG, rtol=0, atol=tol)
    np.testing.assert_allclose(out["jphi_bs"], w * kjb, rtol=0,
                               atol=tol)
    np.testing.assert_allclose(out["jphi_ind"], raw_ind, rtol=0,
                               atol=tol)
    np.testing.assert_allclose(
        out["jphi_ind"] + out["jphi_bs"] + out["jphi_pd"],
        raw_ind + tapered, rtol=0, atol=tol)
