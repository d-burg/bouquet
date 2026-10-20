"""The core-current checks of the in-spec rule (swb): a draw's solved jphi_total inside the
pedestal top within ``jphi_band_nsigma`` of the baseline's sigma_jphi, and its steepest
d jphi / d sqrt(x) there at most ``core_gradient_ratio`` times the baseline's.  They remove the
draws whose sawtooth reset leaves an unrelaxed current step."""
import h5py
import numpy as np
import pytest

import test_engine_draws as TED
from test_engine_draws import toy_bouquet_solver  # noqa: F401  (fixture)

from bouquet import filtering as F
from bouquet import swb_draws as SD
from bouquet.engine_draws import DRAW_BAND_FLAG

X = np.linspace(0.0, 1.0, 201)
RHO = np.sqrt(X)
# H-mode-like baseline: flat core to rho 0.3, a decay, the edge bootstrap peak at x ~ 0.92
J0 = (1.7e6 * np.where(RHO < 0.3, 1.0, np.exp(-((RHO - 0.3) / 0.35) ** 2))
      + 6e5 * np.exp(-((X - 0.92) / 0.03) ** 2))
SIG = 0.1 * np.abs(J0)


def _reset_step(j, rho_m=0.4):
    """Current raised inside rho_m and dropped just outside it (an unrelaxed reset)."""
    out = j.copy()
    out[RHO < rho_m] *= 1.04
    w = (RHO >= rho_m) & (RHO < rho_m + 0.02)
    out[w] *= 0.8
    return out


def test_the_pedestal_top_is_the_baseline_minimum_before_the_edge_peak():
    xp = F.pedestal_top(X, J0)
    assert 0.6 < xp < 0.9 and J0[X == xp][0] == J0[(X >= 0.5) & (X <= xp)].min()
    assert F.pedestal_top(X, 1.0 - X) == 0.5          # no edge peak (L-mode)


def test_the_metrics_flag_a_reset_step_and_pass_a_rescaled_draw():
    z, r, _ = F.core_current_metrics(X, 1.05 * J0, J0, SIG)
    assert z == pytest.approx(0.5) and r == pytest.approx(1.05)
    z, r, _ = F.core_current_metrics(X, _reset_step(J0), J0, SIG)
    assert r > 2.0 and z > 1.5


def test_the_bands_name_record_and_judge():
    b = F.InspecBands(jphi_band_nsigma=3.0, core_gradient_ratio=2.0)
    assert b.core_on and b.any_on and b.rule() == "magnetics+jphi+jgrad"
    assert F.InspecBands.from_record(b.record()) == b
    v = b.verdicts(jphi_z=1.0, gradient_ratio=2.5)
    assert v["jphi_in_band"] and not v["core_gradient_ok"] and not v["in_band"]
    assert b.flags(v) == {F.JPHI_BAND_FLAG: True, F.CORE_GRADIENT_FLAG: False,
                          DRAW_BAND_FLAG: False}
    assert not b.verdicts()["in_band"]                 # unmeasured fails
    with pytest.raises(ValueError, match="core_gradient_ratio"):
        F.InspecBands(core_gradient_ratio=0.0)


def _swb(jphi_out, bands):
    calls = []

    def recipe(kin, j_seed):
        calls.append(j_seed)
        return {"jphi_ind": jphi_out, "jphi_bs": 0 * X, "jphi_total": jphi_out}
    m = SD.SwbDraws(recipe, np.full(X.size, 1e6), np.zeros(X.size),
                    core_bands=bands, core_ref=(X, J0, SIG))
    prof = 1.0 - X ** 2 + 0.1
    inputs = dict(psi_N=X, pressure=None, ne=prof, te=prof, ni=prof, ti=prof,
                  sigma_ne=None, sigma_te=None, sigma_ni=None, sigma_ti=None,
                  n_ls=0.5, t_ls=0.4, j_ls=0.25, Zeff=1.5 + 0 * X, coord="psi_n",
                  p_thresh=0.05, psi_N_kinetic=None, p_fast=None, z_fast=None,
                  z2_fast=None, zeff_includes_fast=False, Z_imp=None,
                  aux_sigmas=None, aux_baselines=None, aux_length_scales=None,
                  ni_from_zeff=True, zeff_dne=None, isolate_edge_jbs=False)
    return m, m.draw(None, 0, 1.0, 0, inputs=inputs)[6]


def test_a_reset_step_is_out_of_spec_in_flight_and_archived(tmp_path):
    b = F.InspecBands(jphi_band_nsigma=3.0, core_gradient_ratio=2.0)
    m, d = _swb(_reset_step(J0), b)
    assert d[DRAW_BAND_FLAG] is False
    assert m.post_hoc(None, d, True) is False
    ok, why = m.until_n(True, [], d)
    assert ok is False and any("core gradient" in w for w in why)
    hdr = str(tmp_path / "a")
    with h5py.File(hdr + ".h5", "w") as hf:
        hf.create_group("scan/0/0")
    m.store_draw(hdr, 0, 0, d)
    with h5py.File(hdr + ".h5", "r") as hf:
        a = hf["scan/0/0"].attrs
        assert not a[DRAW_BAND_FLAG] and not a[F.CORE_GRADIENT_FLAG]
        assert a[F.CORE_GRADIENT_RATIO_ATTR] > 2.0
    m, d = _swb(1.02 * J0, b)
    assert d[DRAW_BAND_FLAG] is True and m.until_n(True, [], d)[0] is True
    m, d = _swb(_reset_step(J0), None)                # checks off: nothing judged
    assert DRAW_BAND_FLAG not in d


def test_refilter_judges_an_archive_from_its_profiles(tmp_path):
    p = str(tmp_path / "a.h5")
    with h5py.File(p, "w") as hf:
        bl = hf.create_group("scan/0/_baseline")
        for k, v in (("psi_N", X), ("jphi_total", J0), ("sigma_jphi", SIG)):
            bl[k] = v
        for i, j in enumerate((1.02 * J0, _reset_step(J0))):
            g = hf.create_group(f"scan/0/{i}")
            g["jphi_total"] = j
            g.attrs["passes_coil_filter"] = True
            g.attrs["passes_boundary_filter"] = True
    out = F.refilter_inspec(p, F.InspecBands(core_gradient_ratio=2.0), scan_key="0",
                            verbose=False)
    assert out["rule"] == "magnetics+jgrad" and out["n_selected"] == 1
    assert F.select_indices(p, scan_key="0") == [0]


def test_the_checks_are_refused_off_swb(tmp_path):
    b = TED._bq(tmp_path)
    b.config.generation.core_gradient_ratio = 2.0
    with pytest.raises(ValueError, match="swb draws only"):
        b._inspec_bands()
    b.config.generation.solve_method = "swb"
    bands, extra = b._inspec_bands()
    assert bands.rule() == "magnetics+jgrad" and extra["route"] == "swb"
