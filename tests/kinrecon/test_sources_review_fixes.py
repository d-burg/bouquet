"""Source-layer fixes after an independent review (synthetic inputs only).

* warm-target beam-target kernel (eq. D5): against a direct quadrature of D5 over v_r >= 0 at beam speeds
  below 6 target thermal speeds (where the earlier full-line form double-counted, up to x2);
  bound 1e-3 relative (241-point trapezoid on a smooth Gaussian-weighted integrand; measured 1e-5);
* IMAS window means: a slice of the window without the quantity is refused, not averaged in as zero;
* the Ip sign is read at the requested time (or refused), never from an arbitrary time slice;
* frame conversions refuse rho outside the geometry's grid (no silent extrapolation).
"""
import copy
import dataclasses

import numpy as np
import pytest

from bouquet.kinrecon.current.geometry import FluxSurfaceGeometry
from bouquet.kinrecon.sources import AmbiguousConventionError, NonInductiveSources, Provenance, from_imas, to_imas
from bouquet.kinrecon.sources import conventions as cv
from bouquet.kinrecon.sources import neutrons as nt
from bouquet.kinrecon.sources.conventions import E_CHARGE, M_AMU

RHO = np.linspace(0.0, 1.0, 41)


def test_warm_kernel_against_direct_d5():
    m_d = nt._A_D * M_AMU
    m_r = m_d / 2
    for ti_kev in (3.0, 6.0):
        u = np.sqrt(2 * ti_kev * 1e3 * E_CHARGE / m_d)
        for eb in (5e3, 10e3, 20e3, 80e3):
            vb = np.sqrt(2 * eb * E_CHARGE / m_d)
            vr = np.linspace(0.0, vb + 8 * u, 400001)
            sig = nt.dd_n_cross_section(0.5 * m_r * vr ** 2 / E_CHARGE / 1e3)
            k = np.exp(-((vr - vb) / u) ** 2) - np.exp(-((vr + vb) / u) ** 2)
            ref = np.trapezoid(sig * vr ** 2 * k, vr) / (np.sqrt(np.pi) * vb * u)
            got = nt._warm_kernel(np.array([vb]), np.array([u]), m_r)[0, 0]
            assert got / ref == pytest.approx(1.0, abs=1e-3), (ti_kev, eb, got / ref)
    flags = nt.validity_flags(e_cm_kev=[0.1, 10.0, 6000.0], t_kev=[0.1, 5.0])
    assert flags["cross_section_below_0.5keV_zeroed"] == 1 and flags["reactivity_below_0.2keV_zeroed"] == 1


def _sources():
    prov = Provenance(provider="test", source_code="synthetic", source_version="0", inputs={}, conventions={},
                      settings={}, assumptions=["a"], limitations=["l"])
    return NonInductiveSources(rho=RHO, time=2.5, time_window=0.0, B0=1.8, R0=1.7,
                               jdotb={"nbi_a": 3e5 * np.exp(-(RHO / 0.3) ** 2)},
                               p_fast={"D": 1e4 * (1 - RHO ** 2)}, n_fast={"D": 2e18 * (1 - RHO ** 2)},
                               provenance=prov)


def test_window_mean_refuses_missing_slices():
    dd = to_imas(_sources(), ip_sign=1, bt_sign=1)
    src0 = dd["core_sources"]["source"][0]
    blk = dict(src0["profiles_1d"][0], time=2.6)
    del blk["j_parallel"]
    src0["profiles_1d"].append(blk)                 # a slice in the window without j_parallel
    with pytest.raises(ValueError, match="missing"):
        from_imas(dd, 2.55, RHO, fast_pressure_components="imas_one_third", time_window=0.2)


def test_ip_sign_read_at_the_requested_time():
    dd = to_imas(_sources(), ip_sign=1, bt_sign=1)
    d2 = copy.deepcopy(dd)
    del d2["core_profiles"]["global_quantities"]
    d2["equilibrium"] = {"time": [1.0, 2.5], "time_slice": [{"global_quantities": {"ip": -1e6}},
                                                            {"global_quantities": {"ip": 1e6}}]}
    r = from_imas(d2, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)
    assert "time_slice[1]" in str(r.provenance.inputs) or "time_slice[1]" in str(r.provenance.conventions)
    d3 = copy.deepcopy(d2)
    del d3["equilibrium"]["time"]
    with pytest.raises(AmbiguousConventionError):
        from_imas(d3, 2.5, RHO, fast_pressure_components="imas_one_third", max_time_offset=1e-6)


def test_frame_conversion_refuses_extrapolation():
    g = FluxSurfaceGeometry.circular(0.6, 1.7, 2.0, n=51)
    g = dataclasses.replace(g, avg_b2=g.fpol ** 2 * g.avg_inv_r2)
    cv.jphi_over_r_from_jdotb(np.ones(5), np.linspace(0, 1, 5), g)
    with pytest.raises(ValueError, match="outside the geometry"):
        cv.jphi_over_r_from_jdotb(np.ones(2), np.array([0.5, 1.05]), g)
