"""bouquet.kinrecon.adapters.synthetic and the named sigma models."""
import numpy as np
import pytest

from bouquet.kinrecon.adapters import SIGMA_MODELS, sigma_from_model
from bouquet.kinrecon.adapters.synthetic import synthetic_measurements
from bouquet.kinrecon.measurements import MachineBundle, read_measurements, write_measurements


def _machine(n_loops=40, n_probes=60):
    rng = np.random.default_rng(11)
    return MachineBundle(machine="toy", sensors={
        "flux_loop": {"labels": [f"L{i}" for i in range(n_loops)],
                      "r": rng.uniform(1, 2.4, n_loops), "z": rng.uniform(-1, 1, n_loops)},
        "bpol_probe": {"labels": [f"P{i}" for i in range(n_probes)],
                       "r": rng.uniform(1, 2.4, n_probes), "z": rng.uniform(-1, 1, n_probes),
                       "direction_r": np.ones(n_probes), "direction_z": np.zeros(n_probes)}})


NOISE = {"flux_loop": {"sigma_model": "relative_floor", "relative": 0.03, "floor": 1e-3},
         "bpol_probe": {"sigma_model": "relative_quadrature", "relative": 0.02, "floor": 5e-3},
         "ip": {"sigma_model": "relative", "relative": 0.01}}


def _model(m):
    rng = np.random.default_rng(5)
    return {"flux_loop": rng.normal(size=len(m.labels("flux_loop"))),
            "bpol_probe": rng.normal(scale=0.3, size=len(m.labels("bpol_probe"))),
            "ip": [1.0e6]}


def test_sigma_models():
    v = np.array([-2.0, 0.01, 1.0])
    np.testing.assert_array_equal(sigma_from_model("relative", v, relative=0.1), [0.2, 0.001, 0.1])
    np.testing.assert_array_equal(sigma_from_model("relative_floor", v, relative=0.1, floor=0.05),
                                  [0.2, 0.05, 0.1])
    np.testing.assert_allclose(sigma_from_model("relative_quadrature", v, relative=0.1, floor=0.05),
                               np.sqrt((0.1 * v) ** 2 + 0.05 ** 2))
    np.testing.assert_array_equal(sigma_from_model("absolute", v, sigma=0.3), [0.3] * 3)
    np.testing.assert_array_equal(
        sigma_from_model("efit_serror_bit", v, serror=0.03, bit=np.array([1e-3, 1e-3, 1e-2])),
        [0.06, 0.01, 0.1])
    assert set(SIGMA_MODELS) >= {"absolute", "relative", "relative_floor",
                                 "relative_quadrature", "efit_serror_bit"}
    with pytest.raises(ValueError, match="unknown"):
        sigma_from_model("nope", v)
    with pytest.raises(ValueError, match="missing"):
        sigma_from_model("relative_floor", v, relative=0.1)


def test_seeded_and_group_independent():
    m = _machine()
    model = _model(m)
    a = synthetic_measurements(m, model, NOISE, seed=42)
    b = synthetic_measurements(m, model, NOISE, seed=42)
    c = synthetic_measurements(m, model, NOISE, seed=43)
    for g in a.groups:
        np.testing.assert_array_equal(a.groups[g].measured, b.groups[g].measured)
    assert not np.array_equal(a.groups["flux_loop"].measured, c.groups["flux_loop"].measured)
    # dropping a group leaves the other groups' noise unchanged
    only = synthetic_measurements(m, {"bpol_probe": model["bpol_probe"]}, NOISE, seed=42)
    np.testing.assert_array_equal(only.groups["bpol_probe"].measured,
                                  a.groups["bpol_probe"].measured)


def test_noise_statistics():
    m = _machine(n_loops=20000, n_probes=2)
    rng = np.random.default_rng(0)
    model = {"flux_loop": rng.normal(size=20000)}
    ms = synthetic_measurements(m, model, NOISE, seed=1)
    g = ms.groups["flux_loop"]
    np.testing.assert_array_equal(g.sigma, np.maximum(0.03 * np.abs(model["flux_loop"]), 1e-3))
    z = (g.measured - model["flux_loop"]) / g.sigma
    n = z.size
    # chi^2/N of N(0,1) noise: mean 1, std sqrt(2/N); 5-sigma band
    assert abs(np.mean(z ** 2) - 1.0) < 5 * np.sqrt(2.0 / n)
    assert abs(np.mean(z)) < 5 / np.sqrt(n)


def test_noiseless_and_meta():
    m = _machine()
    model = _model(m)
    ms = synthetic_measurements(m, model, NOISE, seed=3, add_noise=False, time=0.5,
                                time_window=0.02)
    g = ms.groups["flux_loop"]
    np.testing.assert_array_equal(g.measured, model["flux_loop"])
    assert g.valid.all()
    assert g.meta["sigma_model"] == "relative_floor"
    assert g.meta["sigma_params"] == {"relative": 0.03, "floor": 1e-3}
    assert g.meta["synthetic_seed"] == 3 and g.meta["noise_added"] is False
    np.testing.assert_array_equal(g.geometry["r"], m.sensors["flux_loop"]["r"])
    assert ms.groups["ip"].labels == ["ip"]
    assert (ms.time, ms.time_window) == (0.5, 0.02)


def test_dropout():
    m = _machine()
    model = _model(m)
    ms = synthetic_measurements(m, model, NOISE, seed=9,
                                dropout={"flux_loop": 0.25, "bpol_probe": ["P3", "P7"]})
    fl = ms.groups["flux_loop"]
    assert np.count_nonzero(~fl.valid) == 10
    assert sorted(fl.meta["dropout"]) == sorted(np.array(fl.labels)[~fl.valid].tolist())
    pr = ms.groups["bpol_probe"]
    assert [pr.labels[i] for i in np.flatnonzero(~pr.valid)] == ["P3", "P7"]
    assert ms.groups["ip"].valid.all()
    # dropout does not change the noisy values themselves
    ref = synthetic_measurements(m, model, NOISE, seed=9)
    np.testing.assert_array_equal(fl.measured, ref.groups["flux_loop"].measured)
    with pytest.raises(ValueError, match="unknown channels"):
        synthetic_measurements(m, model, NOISE, dropout={"bpol_probe": ["nope"]})


def test_errors_and_file_round_trip(tmp_path):
    m = _machine()
    model = _model(m)
    with pytest.raises(ValueError, match="noise model"):
        synthetic_measurements(m, model, {"flux_loop": NOISE["flux_loop"]})
    with pytest.raises(ValueError, match="model values"):
        synthetic_measurements(m, {"flux_loop": [1.0, 2.0]}, NOISE)
    ms = synthetic_measurements(m, model, NOISE, seed=2, dropout=0.1)
    write_measurements(tmp_path / "syn.json", ms)
    back = read_measurements(tmp_path / "syn.json")
    for g in ms.groups:
        np.testing.assert_array_equal(ms.groups[g].measured, back.groups[g].measured)
        np.testing.assert_array_equal(ms.groups[g].valid, back.groups[g].valid)
        assert back.groups[g].meta["sigma_model"] == ms.groups[g].meta["sigma_model"]
