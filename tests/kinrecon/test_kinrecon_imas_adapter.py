"""bouquet.kinrecon.adapters.imas_json on a small made-up IDS dict."""
import numpy as np
import pytest

from bouquet.kinrecon.adapters.imas_json import machine_from_imas, measurements_from_imas
from bouquet.kinrecon.measurements import probe_direction

T = np.round(np.arange(0.0, 1.0001, 0.01), 10)  # 101 samples, 10 ms apart


def _sig(values, errors, validity=0, timed=None, t=T):
    blk = {"time": list(t), "data": list(values), "data_error_upper": list(errors)}
    if validity is not None:
        blk["validity"] = validity
    if timed is not None:
        blk["validity_timed"] = list(timed)
    return blk


def _ids():
    ramp = 1.0 + T                      # value = 1 + t
    loops = [
        {"name": "L0", "position": [{"r": 1.0, "z": 0.5}], "flux": _sig(ramp, 0.1 * np.ones_like(T))},
        {"name": "L1", "position": [{"r": 1.1, "z": -0.5}], "flux": _sig(ramp, np.ones_like(T), validity=-2)},
        # sample at t=0.5 flagged invalid by validity_timed
        {"name": "L2", "position": [{"r": 1.2, "z": 0.0}],
         "flux": _sig(np.where(np.isclose(T, 0.5), 99.0, ramp), 0.1 * np.ones_like(T),
                      timed=np.where(np.isclose(T, 0.5), -2, 0))},
        # excluded the OMAS way: sigma 1e30
        {"name": "L3", "position": [{"r": 1.3, "z": 0.0}], "flux": _sig(ramp, 1e30 * np.ones_like(T))},
    ]
    probes = [{"name": "P0", "position": {"r": 1.5, "z": 0.1}, "poloidal_angle": -np.pi / 2,
               "field": _sig(0.2 * np.ones_like(T), 0.01 * np.ones_like(T))}]
    coils = [{"name": "C0", "element": [{"turns_with_sign": 2.0}, {"turns_with_sign": 3.0}],
              "current": _sig(1e3 * ramp, 10.0 * np.ones_like(T), validity=None)}]
    t_ts = np.array([0.40, 0.48, 0.52, 0.70])
    ts = [{"name": "TS0", "position": {"r": 1.9, "z": 0.0},
           "n_e": _sig([1e19, 2e19, 4e19, 8e19], [1e18, 1e18, 3e18, 1e18], None, t=t_ts),
           "t_e": _sig([100.0, 200.0, 300.0, 400.0], [10.0, 0.0, 10.0, 10.0], None, t=t_ts)},
          {"name": "TS1", "position": {"r": 2.0, "z": 0.0},
           "n_e": _sig([1e19], [1e18], None, t=[0.9]),
           "t_e": _sig([1.0], [1.0], None, t=[0.9])}]
    tg = 0.2
    mse = [{"name": "M0",
            "active_spatial_resolution": [
                {"time": 0.0, "centre": {"r": 1.8, "z": 0.0},
                 "geometric_coefficients": [1, 0, 0, 0, 0, 0, 1, 0, 0]},
                {"time": 0.5, "centre": {"r": 1.81, "z": 0.01},
                 "geometric_coefficients": [2, 0, 0, 0, 0, 0, 1, 0, 0]}],
            "polarisation_angle": _sig(np.arctan(tg) * np.ones_like(T), 1e-3 * np.ones_like(T))}]
    return {"magnetics": {"flux_loop": loops, "b_field_pol_probe": probes,
                          "ip": [_sig(1e6 * np.ones_like(T), 1e4 * np.ones_like(T), None)]},
            "pf_active": {"coil": coils},
            "thomson_scattering": {"channel": ts},
            "mse": {"channel": mse}}


def test_window_mean_and_validity():
    ms = measurements_from_imas(_ids(), time=0.5, time_window=0.1, machine="toy",
                                mse_er_corrected=True)
    g = ms.groups["flux_loop"]
    assert g.labels == ["L0", "L1", "L2", "L3"]
    # samples t = 0.45..0.55 (11 of them); mean of 1 + t is 1.5
    assert g.meta["n_samples"].tolist() == [11, 11, 10, 0]
    np.testing.assert_allclose(g.measured[0], 1.5, rtol=1e-12)
    np.testing.assert_allclose(g.sigma[0], 0.1, rtol=1e-12)
    np.testing.assert_allclose(g.measured[2], 1.5, rtol=1e-12)  # the 99 was excluded
    assert g.valid.tolist() == [True, False, True, False]
    assert np.isnan(g.measured[3]) and np.isnan(g.sigma[3])
    assert g.meta["sigma_model"] == "data_error_upper+window_mean"
    np.testing.assert_array_equal(g.geometry["r"], [1.0, 1.1, 1.2, 1.3])
    assert ms.meta["time_window_s"] == [0.45, 0.55]


def test_reductions_recorded_by_name():
    ids = _ids()
    sem = measurements_from_imas(ids, 0.5, 0.1, reduction="window_sem",
                                 groups=["flux_loop"]).groups["flux_loop"]
    np.testing.assert_allclose(sem.sigma[0], 0.1 / np.sqrt(11), rtol=1e-12)
    assert sem.meta["sigma_model"] == "data_error_upper+window_sem"
    spr = measurements_from_imas(ids, 0.5, 0.1, reduction="window_spread",
                                 groups=["flux_loop"]).groups["flux_loop"]
    vals = 1.0 + T[(T >= 0.45 - 1e-12) & (T <= 0.55 + 1e-12)]
    np.testing.assert_allclose(spr.sigma[0], np.sqrt(0.1 ** 2 + vals.var()), rtol=1e-10)
    with pytest.raises(ValueError, match="reduction"):
        measurements_from_imas(ids, 0.5, 0.1, reduction="median", groups=["flux_loop"])


def test_probe_coil_ip_geometry():
    ms = measurements_from_imas(_ids(), 0.5, 0.1, groups=["bpol_probe", "pf_current", "ip"])
    p = ms.groups["bpol_probe"]
    dr, dz = probe_direction(-np.pi / 2)
    assert (p.geometry["direction_r"][0], p.geometry["direction_z"][0]) == (dr, dz)
    np.testing.assert_allclose(p.geometry["direction_z"][0], 1.0)
    c = ms.groups["pf_current"]
    assert c.geometry["turns"].tolist() == [5.0]
    np.testing.assert_allclose(c.measured, [1.5e3])
    assert c.valid.all()
    assert ms.groups["ip"].measured.tolist() == [1e6]


def test_thomson_sparse_samples():
    ms = measurements_from_imas(_ids(), 0.5, 0.1, groups=["thomson_ne", "thomson_te"])
    ne, te = ms.groups["thomson_ne"], ms.groups["thomson_te"]
    np.testing.assert_allclose(ne.measured[0], 3e19)         # pulses at 0.48 and 0.52
    np.testing.assert_allclose(ne.sigma[0], 2e18)
    np.testing.assert_allclose(te.measured[0], 300.0)        # the sigma = 0 sample is dropped
    assert ne.valid.tolist() == [True, False]                # TS1 has no pulse in the window
    assert te.meta["n_samples"].tolist() == [1, 0]


def test_mse_tan_gamma_and_er_flag():
    ids = _ids()
    with pytest.raises(ValueError, match="mse_er_corrected"):
        measurements_from_imas(ids, 0.5, 0.1, groups=["mse"])
    # not requested explicitly: skipped rather than guessed
    assert "mse" not in measurements_from_imas(ids, 0.5, 0.1).groups
    g = measurements_from_imas(ids, 0.5, 0.1, groups=["mse"], mse_er_corrected=False).groups["mse"]
    np.testing.assert_allclose(g.measured, [0.2], rtol=1e-12)
    np.testing.assert_allclose(g.sigma, [1e-3 * (1 + 0.2 ** 2)], rtol=1e-12)
    assert g.meta["quantity"] == "tan_polarisation_angle" and g.meta["er_corrected"] is False
    np.testing.assert_array_equal(g.geometry["coefficients"][0], [2, 0, 0, 0, 0, 0, 1, 0, 0])
    assert g.geometry["r"].tolist() == [1.81]


def test_requested_absent_group_raises():
    with pytest.raises(ValueError, match="absent"):
        measurements_from_imas(_ids(), 0.5, 0.1, groups=["diamagnetic_flux"])


def test_machine_from_imas():
    b = machine_from_imas(_ids(), "toy", mesh_file="mesh.h5")
    assert b.labels("flux_loop") == ["L0", "L1", "L2", "L3"]
    np.testing.assert_allclose(b.sensors["bpol_probe"]["direction_z"], [1.0])
    assert b.sensors["pf_current"]["turns"].tolist() == [5.0]
    assert b.sensors["mse"]["coefficients"].shape == (1, 9)
