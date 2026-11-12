"""bouquet.kinrecon.measurements: per-slice file and machine bundle round trips.

All values are synthetic (made up in the test)."""
import json

import numpy as np
import pytest

from bouquet.kinrecon.interfaces import ConstraintGroup, MeasurementSet
from bouquet.kinrecon import measurements as M


def _group(name, n, rng, **geometry):
    return ConstraintGroup(
        name=name, labels=[f"{name}_{i}" for i in range(n)],
        measured=rng.normal(size=n), sigma=rng.uniform(0.1, 1.0, n),
        valid=rng.uniform(size=n) > 0.3, weight=rng.uniform(0.5, 2.0, n),
        geometry=geometry, meta={"sigma_model": "relative_floor", "sigma_params":
                                 {"relative": 0.03, "floor": np.full(n, 1e-3)}})


def _full_set():
    rng = np.random.default_rng(7)
    ms = MeasurementSet(machine="toy", time=1.25, time_window=0.01,
                        meta={"source": "unit test", "nested": {"a": [1, 2.5, None], "b": True}})
    ms.groups["flux_loop"] = _group("flux_loop", 5, rng, r=rng.uniform(1, 2, 5),
                                    z=rng.uniform(-1, 1, 5))
    # vloop shares the flux-loop channels and positions
    ms.groups["vloop"] = ConstraintGroup(
        name="vloop", labels=["flux_loop_3", "flux_loop_1"], measured=[0.1, -0.2],
        sigma=[0.01, 0.02], valid=[True, False],
        geometry={"r": ms.groups["flux_loop"].geometry["r"][[3, 1]],
                  "z": ms.groups["flux_loop"].geometry["z"][[3, 1]]})
    ang = rng.uniform(0, 2 * np.pi, 4)
    dr, dz = M.probe_direction(ang)
    ms.groups["bpol_probe"] = _group("bpol_probe", 4, rng, r=rng.uniform(1, 2, 4),
                                     z=rng.uniform(-1, 1, 4), direction_r=dr,
                                     direction_z=dz, length=rng.uniform(0.05, 0.1, 4))
    ms.groups["ip"] = ConstraintGroup(name="ip", labels=["ip"], measured=[1.0e6],
                                      sigma=[1.0e4], valid=[True])
    ms.groups["diamagnetic_flux"] = ConstraintGroup(
        name="diamagnetic_flux", labels=["diamagnetic_flux"], measured=[-2.5e-3],
        sigma=[1e-4], valid=[False], weight=[0.0])
    ms.groups["pf_current"] = _group("pf_current", 3, rng,
                                     coil_names=np.array(["C1", "C2", "C3"]),
                                     turns=np.array([10.0, 20.0, 1.0]))
    ms.groups["thomson_ne"] = _group("thomson_ne", 4, rng, r=np.full(4, 1.9),
                                     z=np.linspace(-0.2, 0.4, 4))
    te = _group("thomson_te", 4, rng, r=np.full(4, 1.9), z=np.linspace(-0.2, 0.4, 4))
    # different channel order than n_e, and one non-finite value
    order = [2, 0, 3, 1]
    te.labels = [f"thomson_ne_{i}" for i in order]
    te.geometry = {"r": np.full(4, 1.9), "z": np.linspace(-0.2, 0.4, 4)[order]}
    te.measured[1] = np.nan
    ms.groups["thomson_te"] = te
    tg = rng.uniform(-0.3, 0.3, 6)
    ms.groups["mse"] = ConstraintGroup(
        name="mse", labels=[f"ch{i}" for i in range(6)], measured=tg,
        sigma=rng.uniform(1e-3, 5e-3, 6), valid=[True] * 5 + [False],
        weight=np.ones(6), geometry={"r": rng.uniform(1.6, 2.2, 6), "z": np.zeros(6),
                                     "coefficients": rng.normal(size=(6, 9))},
        meta={"quantity": "tan_polarisation_angle", "er_corrected": True,
              "tan_gamma_uncorrected": tg + 0.01, "sigma_model": "efit_sgamma"})
    ms.groups["q"] = ConstraintGroup(name="q", labels=["q0"], measured=[1.02],
                                     sigma=[0.05], valid=[True])
    ms.groups["custom_saddle"] = _group("custom_saddle", 2, rng,
                                        corners=rng.normal(size=(2, 4, 2)))
    return ms


def _assert_same(a, b):
    assert (a.machine, a.time, a.time_window) == (b.machine, b.time, b.time_window)
    assert list(a.groups) == list(b.groups)
    _assert_tree(a.meta, b.meta)
    for name in a.groups:
        ga, gb = a.groups[name], b.groups[name]
        assert ga.labels == gb.labels, name
        for attr in ("measured", "sigma", "valid", "weight"):
            x, y = getattr(ga, attr), getattr(gb, attr)
            assert x.dtype == y.dtype, (name, attr)
            np.testing.assert_array_equal(x, y, err_msg=f"{name}.{attr}")
        assert set(ga.geometry) == set(gb.geometry), name
        for k in ga.geometry:
            np.testing.assert_array_equal(np.asarray(ga.geometry[k]), gb.geometry[k])
        _assert_tree(ga.meta, gb.meta)


def _assert_tree(x, y):
    if isinstance(x, dict):
        assert set(x) == set(y)
        for k in x:
            _assert_tree(x[k], y[k])
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray) and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            _assert_tree(u, v)
    else:
        assert x == y and type(x) is type(y)


def test_round_trip_is_exact(tmp_path):
    ms = _full_set()
    path = tmp_path / "slice.json"
    M.write_measurements(path, ms)
    back = M.read_measurements(path)
    _assert_same(ms, back)
    # and a second write is byte-identical
    M.write_measurements(tmp_path / "again.json", back)
    assert path.read_bytes() == (tmp_path / "again.json").read_bytes()


def test_file_uses_ids_names(tmp_path):
    ms = _full_set()
    M.write_measurements(tmp_path / "s.json", ms)
    doc = json.loads((tmp_path / "s.json").read_text())
    fl = {e["name"]: e for e in doc["magnetics"]["flux_loop"]}
    g = ms.groups["flux_loop"]
    e = fl["flux_loop_2"]
    assert e["flux"]["data"] == [g.measured[2]]
    assert e["flux"]["data_error_upper"] == [g.sigma[2]]
    assert e["flux"]["validity"] == (0 if g.valid[2] else M.INVALID)
    assert e["position"][0] == {"r": g.geometry["r"][2], "z": g.geometry["z"][2]}
    assert fl["flux_loop_3"]["voltage"]["data"] == [0.1]
    pr = doc["magnetics"]["b_field_pol_probe"][0]
    gp = ms.groups["bpol_probe"]
    np.testing.assert_allclose(M.probe_direction(pr["poloidal_angle"]),
                               (gp.geometry["direction_r"][0], gp.geometry["direction_z"][0]),
                               atol=1e-14)
    assert doc["pf_active"]["coil"][1]["current"]["data"] == [ms.groups["pf_current"].measured[1]]
    ch = doc["thomson_scattering"]["channel"]
    assert len(ch) == 4 and {"n_e", "t_e", "position"} <= set(ch[0])
    mse = doc["mse"]["channel"][0]
    tg = ms.groups["mse"].measured[0]
    assert mse["polarisation_angle"]["data"] == [np.arctan(tg)]
    assert len(mse["active_spatial_resolution"][0]["geometric_coefficients"]) == 9
    q = doc["equilibrium"]["time_slice"][0]["constraints"]["q"][0]
    assert q["measured"] == 1.02 and q["source"] == "q0"
    ext = doc[M.EXT]
    assert ext["time_window"] == [1.25 - 0.005, 1.25 + 0.005]
    assert ext["mse"]["er_corrected"] is True
    assert ext["groups"]["thomson_ne"]["valid"] == [bool(v) for v in ms.groups["thomson_ne"].valid]
    assert ext["sigma_models"]["mse"] == "efit_sgamma"
    assert ext["groups"]["custom_saddle"]["ids_path"] is None


def test_mse_angle_quantity_written_as_is(tmp_path):
    ms = MeasurementSet(machine="toy", time=0.0, time_window=0.0)
    ms.groups["mse"] = ConstraintGroup(name="mse", labels=["a"], measured=[0.1], sigma=[1e-3],
                                       valid=[True], geometry={"coefficients": np.ones((1, 9))})
    doc = M.to_dict(ms)
    assert doc["mse"]["channel"][0]["polarisation_angle"]["data"] == [0.1]
    assert "measured" not in doc[M.EXT]["groups"]["mse"]
    back = M.from_dict(json.loads(json.dumps(doc)))
    np.testing.assert_array_equal(back.groups["mse"].measured, [0.1])


def test_duplicate_labels_refused():
    ms = MeasurementSet(machine="toy", time=0.0, time_window=0.0)
    ms.groups["flux_loop"] = ConstraintGroup(name="flux_loop", labels=["a", "a"],
                                             measured=[1, 2], sigma=[1, 1], valid=[1, 1])
    with pytest.raises(ValueError, match="unique"):
        M.to_dict(ms)


def test_conflicting_shared_channel_geometry_refused():
    ms = MeasurementSet(machine="toy", time=0.0, time_window=0.0)
    for name, r in (("thomson_ne", 1.0), ("thomson_te", 1.5)):
        ms.groups[name] = ConstraintGroup(name=name, labels=["c0"], measured=[1.0],
                                          sigma=[0.1], valid=[True],
                                          geometry={"r": [r], "z": [0.0]})
    with pytest.raises(ValueError, match="differs"):
        M.to_dict(ms)


def test_wrong_schema_refused():
    with pytest.raises(ValueError, match="schema"):
        M.from_dict({"schema": "something/else"})


def test_probe_direction_convention():
    # DD: zero along +R, clockwise theta-like angle -> -pi/2 points along +Z
    dr, dz = M.probe_direction([0.0, -np.pi / 2, np.pi / 2, np.pi])
    np.testing.assert_allclose(dr, [1, 0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(dz, [0, 1, -1, 0], atol=1e-15)
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    np.testing.assert_allclose(M.probe_angle(*M.probe_direction(ang)), ang, atol=1e-12)


def test_machine_bundle_round_trip(tmp_path):
    mesh = tmp_path / "mesh.h5"
    mesh.write_bytes(b"not really a mesh")
    b = M.MachineBundle(
        machine="toy",
        sensors={"flux_loop": {"labels": ["L1", "L2"], "r": [1.0, 2.0], "z": [0.0, 0.5]},
                 "bpol_probe": {"labels": ["P1"], "r": [1.1], "z": [0.2],
                                "direction_r": [0.0], "direction_z": [1.0]},
                 "pf_current": {"labels": ["C1", "C2"], "turns": [10.0, 4.0]}},
        mesh_file="mesh.h5", mesh_sha256=M.file_sha256(mesh),
        channel_maps={"ARRAY_A": ["C2"], "ARRAY_B": ["C1"]}, meta={"era": "toy-1"})
    M.write_machine(tmp_path / "machine.json", b)
    back = M.read_machine(tmp_path / "machine.json", verify_mesh=True)
    assert (back.machine, back.mesh_file, back.mesh_sha256) == (b.machine, b.mesh_file,
                                                                 b.mesh_sha256)
    assert back.channel_maps == b.channel_maps and back.meta == b.meta
    assert list(back.sensors) == list(b.sensors)
    assert back.mesh_path() == str(mesh)
    for g in b.sensors:
        for k in b.sensors[g]:
            np.testing.assert_array_equal(np.asarray(b.sensors[g][k]), back.sensors[g][k])
    mesh.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mesh_sha256"):
        M.read_machine(tmp_path / "machine.json", verify_mesh=True)


def test_machine_bundle_validates_shapes():
    with pytest.raises(ValueError, match="shape"):
        M.MachineBundle(machine="toy", sensors={"flux_loop": {"labels": ["a", "b"],
                                                              "r": [1.0]}})


def test_machine_provenance_record_and_check(tmp_path):
    a, b = tmp_path / "table_a.json", tmp_path / "table_b.dat"
    a.write_text('{"x": 1}\n')
    b.write_text("1 2 3\n")
    prov = M.machine_provenance([a, b], era="era-2", era_rule="inclusive start",
                                generator={"name": "maptool", "version": "1.2.3"})
    assert prov["era"] == "era-2" and prov["generator"]["version"] == "1.2.3"
    assert prov["source_tables"]["table_a.json"]["sha256"] == M.file_sha256(a)
    bundle = M.MachineBundle(machine="toy", sensors={"flux_loop": {"labels": ["L0"]}}, meta={"provenance": prov})
    path = tmp_path / "bundle.json"
    M.write_machine(path, bundle)
    back = M.read_machine(path)
    assert back.meta["provenance"] == prov
    assert M.verify_machine_provenance(back, {"table_a.json": a, "table_b.dat": b}) == \
        {"table_a.json": True, "table_b.dat": True}
    b.write_text("1 2 4\n")
    assert M.verify_machine_provenance(back, {"table_a.json": a, "table_b.dat": b})["table_b.dat"] is False
    with pytest.raises(ValueError, match="no path"):
        M.verify_machine_provenance(back, {"table_a.json": a})
    with pytest.raises(ValueError, match="no provenance"):
        M.verify_machine_provenance(M.MachineBundle(machine="toy"), {})
    with pytest.raises(ValueError, match="unique names"):
        M.machine_provenance([a, tmp_path / "x" / ".." / "table_a.json"])
