"""``fit_inductive_profile``: the inductive-amplitude fallback is LOUD.

The amplitude is the brentq root of the cylindrical l_i-proxy residual once a
sign change is bracketed; when none is found it falls back to 1.0.  That
fallback used to be silent.  It now prints and records the reason, the final
bracket and the residuals at its ends -- and nothing else moves: every output
is bit-identical to the frozen pre-change copy
(``tests/_fit_inductive_prechange.py``), with and without a fallback, with
and without the joint bootstrap rescale.

The l_i proxy is replaced by a synthetic peakedness functional of the profile
(no solver).  Synthetic profiles only.
"""
import contextlib
import io

import numpy as np
import pytest

import bouquet.TokaMaker_interface as TI

import _fit_inductive_prechange as PRE

_PSI = np.linspace(0.0, 1.0, 121)


def _profiles():
    j_BS = 3.0e5 * np.exp(-0.5 * ((_PSI - 0.94) / 0.025) ** 2)
    j_ind = 1.2e6 * (1.0 - _PSI ** 1.6) ** 1.3
    return j_ind + j_BS, j_BS


def _proxy(mygs, j, psi_pad, x=None, coord="psi_n"):
    """Peakedness: axis value over the flux-grid mean (monotone in the
    inductive amplitude when the bootstrap is edge-peaked)."""
    j = np.asarray(j, dtype=float)
    return float(j[0] / np.mean(j))


def _run(fn, target, rescale, monkeypatch, proxy=_proxy):
    monkeypatch.setattr(TI, "calc_cylindrical_li_proxy", proxy)
    jtor, jbs = _profiles()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(None, jtor, jbs, _PSI, 1e-3, target, rescale_j_BS=rescale)
    return out, buf.getvalue()


def _assert_bit_identical(new, old):
    for k in ("j_inductive_fit", "j_phi_fit", "j_BS_used"):
        np.testing.assert_array_equal(new[k], old[k])
    for k in ("fit_li", "ind_scale", "bs_scale"):
        assert new[k] == old[k] or (np.isnan(new[k]) and np.isnan(old[k])), k
    x = np.linspace(0.0, 1.0, 57)
    np.testing.assert_array_equal(new["spline"](x), old["spline"](x))


@pytest.mark.parametrize("rescale", [False, True])
def test_a_bracketed_root_is_unchanged_and_records_no_fallback(
        monkeypatch, rescale):
    jtor, _ = _profiles()
    target = _proxy(None, jtor, 1e-3)
    new, log = _run(TI.fit_inductive_profile, target, rescale, monkeypatch)
    old, _ = _run(PRE.fit_inductive_profile, target, rescale, monkeypatch)
    _assert_bit_identical(new, old)
    assert new["ind_scale"] != 1.0            # a real root, not the fallback
    assert new["ind_scale_fallback"] is False
    assert new["ind_scale_fallback_record"] is None
    assert new["n_ind_scale_fallbacks"] == 0
    assert "FALLBACK" not in log


@pytest.mark.parametrize("rescale", [False, True])
def test_an_unbracketed_target_falls_back_to_one_loudly(monkeypatch,
                                                         rescale):
    target = 100.0                         # no amplitude reaches it
    new, log = _run(TI.fit_inductive_profile, target, rescale, monkeypatch)
    old, _ = _run(PRE.fit_inductive_profile, target, rescale, monkeypatch)
    _assert_bit_identical(new, old)
    # the value and the bracket are exactly as before
    assert new["ind_scale"] == 1.0
    assert new["ind_scale_fallback"] is True
    rec = new["ind_scale_fallback_record"]
    assert rec["bracket"] == [0.5 * 0.5 ** 10, 2.0 * 2.0 ** 10]
    assert all(np.isfinite(rec["residual_at_bracket"]))
    assert rec["residual_at_bracket"][0] * rec["residual_at_bracket"][1] > 0
    assert rec["li_proxy_target"] == target
    assert rec["ind_scale_fallback"] == 1.0
    assert "no sign change" in rec["reason"]
    assert new["n_ind_scale_fallbacks"] >= 1
    assert "WARNING ind_scale FALLBACK to 1.0" in log
    assert "[0.000488281, 2048]" in log


def test_a_non_finite_residual_is_named_as_such(monkeypatch):
    new, log = _run(TI.fit_inductive_profile, 1.0, False, monkeypatch,
                    proxy=lambda m, j, p, *a: float("nan"))
    old, _ = _run(PRE.fit_inductive_profile, 1.0, False, monkeypatch,
                  proxy=lambda m, j, p, *a: float("nan"))
    _assert_bit_identical(new, old)
    assert new["ind_scale"] == 1.0 and new["ind_scale_fallback"] is True
    assert "non-finite" in new["ind_scale_fallback_record"]["reason"]
    assert "FALLBACK" in log


def test_the_reconstruction_metrics_carry_the_fallback():
    """``_reconstruction_metrics`` copies the quality record through."""
    import bouquet.baseline as bl
    rec = dict(reason="no sign change", bracket=[0.1, 10.0],
               residual_at_bracket=[0.2, 0.3], li_proxy_target=1.0,
               bs_scale=1.0, ind_scale_fallback=1.0, fit_call=1)
    q = dict(ind_scale_fallback=True, ind_scale_fallback_n=1,
             ind_scale_fallback_records=[rec])
    m = _metrics_with_quality(bl, q)
    assert m["ind_scale_fallback"] is True
    assert m["ind_scale_fallback_n"] == 1
    assert m["ind_scale_fallback_records"] == [rec]
    m0 = _metrics_with_quality(bl, {})
    assert m0["ind_scale_fallback"] is False
    assert m0["ind_scale_fallback_n"] == 0
    assert m0["ind_scale_fallback_records"] == []


class _GS:
    o_point = (1.7, 0.03)

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"q_0": 1.0, "q_95": 4.0, "beta_n": 2.0, "beta_pol": 80.0,
                "kappa": 1.8, "delta": 0.5, "W_MHD": 1.0e6, "l_i": 0.68}


class _Eqdsk:
    psi_N = np.linspace(0.0, 1.0, 51)
    Ip = 1.4e6
    li = {}


class _Src:
    psi_pad = 1e-3


def _metrics_with_quality(bl, quality):
    """``_reconstruction_metrics`` on the minimal solver/g-file doubles (as
    in tests/test_corrective_iteration_package.py)."""
    result = {
        "Ip_tokamaker": 1.4e6,
        "j_phi_fit": np.linspace(1e6, 0.0, 51),
        "eqdsk_jtor": np.linspace(1e6, 0.0, 51),
        "quality": dict(quality),
    }
    with pytest.warns(UserWarning):          # no real EFIT reference
        return bl._reconstruction_metrics(_GS(), _Eqdsk(), result, _Src(),
                                          0.6842,
                                          l_i_realized_post_corrective=0.69)
