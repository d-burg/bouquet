"""ImasSource.ida_time and the ida_hybrid timing rules (#73 and its review).

* the IDA slice follows the source-time rule (io.imas.IDA_TIME_RULE):
  nearest own slice, within half the IDA file's local step (and, paired with
  the dd time, within half the dd step of the core_profiles slice), never
  interpolated, dt recorded, refused outside; a single-slice IDA file honours
  an explicit ida_time with the 10 us floor;
* the pairing record reaches li_metrics (archived on every route);
* ``_check_replay_pairing`` tolerates rows without ``outcome`` and trusts a
  table only when it names this run's IDA file and holds the dd slice;
* ``set_slice`` / ``run_slices`` keep, apply or refuse ``ida_time`` -- never
  silently drop it.

Fully synthetic inputs (analytic profiles written to tmp files).
"""
import json
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.config import ImasSource
from bouquet.io.imas import (IMAS_SINGLE_TIME_WINDOW_S, _check_replay_pairing,
                             _hybrid_timing, read_imas_baseline)
from _imas_geometry import eq_geometry, jtor_from_jtotal

N = 33
Z_IMP = 6.0


# ---------------------------------------------------------------------------
#  the helpers in isolation
# ---------------------------------------------------------------------------
def _src(**kw):
    return ImasSource(ids_path="x.json", **kw)


def test_ida_time_defaults_to_time_and_is_matched_exactly():
    aux = {}
    t = _hybrid_timing(_src(time=1.04), 1.04, 1.04, 1.04, aux,
                       cp_times=[1.0, 1.04, 1.08],
                       ida_times=[1.0, 1.033, 1.04, 1.074])
    assert t == 1.04
    rec = aux["ida_time_match"]
    assert rec["ida_time_configured"] is None and rec["dt"] == 0.0
    assert rec["window_basis"] == "half the IDA file's local time-step"
    aux = {}
    t = _hybrid_timing(_src(time=1.04, ida_time=1.0335), 1.04, 1.04, 1.04,
                       aux, cp_times=[1.0, 1.04, 1.08],
                       ida_times=[1.0, 1.033, 1.04, 1.074])
    assert t == 1.033                                   # nearest, not 1.0335
    rec = aux["ida_time_match"]
    assert rec["ida_time_requested"] == 1.0335
    assert rec["dt"] == pytest.approx(-5e-4)
    assert rec["dd_offset"] == pytest.approx(-0.007)
    assert (aux["fuse_time_cp"], aux["fuse_time_eq"]) == (1.04, 1.04)


def test_an_ida_time_outside_the_half_step_is_refused():
    with pytest.raises(ValueError, match="no IDA slice within half the IDA"):
        _hybrid_timing(_src(time=1.04, ida_time=1.33), 1.04, 1.04, 1.04, {},
                       cp_times=[1.0, 1.04], ida_times=[0.9, 1.0, 1.1, 1.2])
    # past the end: the end interval's half step, then refused
    with pytest.raises(ValueError, match="nearest IDA slice 1.2"):
        _hybrid_timing(_src(time=1.04, ida_time=1.26), 1.04, 1.04, 1.04, {},
                       cp_times=[1.0, 1.04], ida_times=[0.9, 1.0, 1.1, 1.2])
    # within half a step: a match
    t = _hybrid_timing(_src(time=1.04, ida_time=1.06), 1.04, 1.04, 1.04, {},
                       cp_times=[1.0, 1.04], ida_times=[0.9, 1.0, 1.1, 1.2])
    assert t == 1.1


def test_paired_with_the_dd_time_the_ida_slice_must_sit_in_the_dd_window():
    """ida_time=None: the IDA slice must also lie within half the local
    core_profiles step of the core_profiles slice read."""
    with pytest.raises(ValueError, match="more than half its local"):
        _hybrid_timing(_src(time=1.0), 1.0, 1.0, 1.0, {},
                       cp_times=[0.99, 1.0, 1.01],
                       ida_times=[0.95, 1.025, 1.1])
    # the same pairing made deliberately is accepted, with a warning
    with pytest.warns(UserWarning, match="more than one local dd time-step"):
        t = _hybrid_timing(_src(time=1.0, ida_time=1.025), 1.0, 1.0, 1.0, {},
                           cp_times=[0.99, 1.0, 1.01],
                           ida_times=[0.95, 1.025, 1.1])
    assert t == 1.025


def test_a_single_slice_ida_file_honours_ida_time_with_the_10us_floor():
    aux = {}
    t = _hybrid_timing(_src(time=1.04, ida_time=1.033 + 0.5e-5), 1.04, 1.04,
                       1.04, aux, cp_times=[1.0, 1.04, 1.08],
                       ida_times=[1.033])
    assert t == 1.033
    assert aux["ida_time_match"]["half_window"] == IMAS_SINGLE_TIME_WINDOW_S
    with pytest.raises(ValueError, match="IMAS_SINGLE_TIME_WINDOW_S"):
        _hybrid_timing(_src(time=1.04, ida_time=1.0), 1.04, 1.04, 1.04, {},
                       cp_times=[1.0, 1.04, 1.08], ida_times=[1.033])
    # paired with the dd time: half the dd step decides
    t = _hybrid_timing(_src(time=1.04), 1.04, 1.04, 1.04, {},
                       cp_times=[1.0, 1.04, 1.08], ida_times=[1.033])
    assert t == 1.033
    with pytest.raises(ValueError):
        _hybrid_timing(_src(time=1.08), 1.08, 1.08, 1.08, {},
                       cp_times=[1.0, 1.04, 1.08], ida_times=[1.033])


def test_non_macro_step_warns():
    with pytest.warns(UserWarning, match="not a dd macro step"):
        _hybrid_timing(_src(time=1.02), 1.02, 1.02, 1.04, {},
                       cp_times=[1.0, 1.04], ida_times=[1.02, 1.04])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _hybrid_timing(_src(time=1.04), 1.04, 1.04 + 1e-9, 1.04, {},
                       cp_times=[1.0, 1.04], ida_times=[1.0, 1.04])


def _prov(tmp_path, rows, **extra):
    (tmp_path / "ida_provenance.json").write_text(
        json.dumps({"ida_file": "/elsewhere/run/ida.cdf",
                    "sim_times": [1.04, 1.08], "replay_pairing": rows,
                    **extra}))


def test_replay_pairing_check(tmp_path):
    ids = str(tmp_path / "dd_sim.json")
    ida = str(tmp_path / "ida.cdf")
    aux = {"fuse_time_cp": 1.04, "ida_time_used": 1.033}
    _check_replay_pairing(ids, aux, ida_path=ida)
    assert aux["pairing_consistent"] is None          # no table beside the dd
    (tmp_path / "ida_provenance.json").write_text(json.dumps({"ida_file": "ida.cdf"}))
    _check_replay_pairing(ids, aux, ida_path=ida)
    assert aux["pairing_consistent"] is None          # an older table: no replay_pairing
    rows = [{"t_sim": 1.04, "tick": 1.04 - 1e-9, "ida_time": 1.033, "outcome": "ida"},
            {"t_sim": 1.08, "tick": 1.08 - 1e-9, "ida_time": 1.074, "outcome": "ida"}]
    _prov(tmp_path, rows)
    _check_replay_pairing(ids, aux, ida_path=ida)
    assert aux["pairing_consistent"] is True          # bound by the IDA file name
    aux = {"fuse_time_cp": 1.08, "ida_time_used": 1.054}
    with pytest.warns(UserWarning, match="computed on IDA 1.074"):
        _check_replay_pairing(ids, aux, ida_path=ida)
    assert aux["pairing_consistent"] is False and aux["replayed_ida_time"] == 1.074


def test_rows_without_outcome_do_not_crash_the_reader(tmp_path):
    """Review B5: a row lacking 'outcome' raised KeyError from the reader."""
    ids, ida = str(tmp_path / "dd_sim.json"), str(tmp_path / "ida.cdf")
    _prov(tmp_path, [{"t_sim": 1.04, "ida_time": 1.0}, {"bogus": 1}])
    aux = {"fuse_time_cp": 1.04, "ida_time_used": 1.033,
           "ida_time_match": {}}
    with pytest.warns(UserWarning, match="outcome not recorded"):
        _check_replay_pairing(ids, aux, ida_path=ida)
    assert aux["pairing_consistent"] is False
    assert aux["ida_time_match"]["pairing_table"]["status"] == "checked"


def test_a_table_from_another_run_is_not_trusted(tmp_path):
    """Review B5: the table is located by directory; it must also name this
    run's IDA file and hold this dd slice, else it is not consulted."""
    ids = str(tmp_path / "dd_sim.json")
    rows = [{"t_sim": 1.04, "ida_time": 1.033, "outcome": "ida"}]
    _prov(tmp_path, rows)
    aux = {"fuse_time_cp": 1.04, "ida_time_used": 1.033, "ida_time_match": {}}
    with pytest.warns(UserWarning, match="does not describe this run"):
        _check_replay_pairing(ids, aux, ida_path=str(tmp_path / "other.cdf"))
    assert aux["pairing_consistent"] is None
    assert "not bound" in aux["ida_time_match"]["pairing_table"]["status"]
    aux = {"fuse_time_cp": 1.04, "ida_time_used": 1.033}
    _prov(tmp_path, rows, sim_times=[2.0, 2.04])        # another dd's macro steps
    with pytest.warns(UserWarning, match="sim_times do not hold"):
        _check_replay_pairing(ids, aux, ida_path=str(tmp_path / "ida.cdf"))
    assert aux["pairing_consistent"] is None
    aux = {"fuse_time_cp": 1.04, "ida_time_used": 1.033}
    (tmp_path / "dd_sim.json").write_text("{}")
    _prov(tmp_path, rows, dd_sha256="0" * 64)           # a hash, when present
    with pytest.warns(UserWarning, match="dd_sha256"):
        _check_replay_pairing(ids, aux, ida_path=str(tmp_path / "ida.cdf"))
    assert aux["pairing_consistent"] is None


# ---------------------------------------------------------------------------
#  through read_imas_baseline: a 3-slice IDA file and a 2-slice dd
# ---------------------------------------------------------------------------
_IDA_MS = (990.0, 1000.0, 1010.0)
_NE_SCALE = (0.8, 1.0, 1.3)          # each IDA slice has its own density


def _ida_cdf(path, times_ms=_IDA_MS, scales=_NE_SCALE):
    psi = np.linspace(0.0, 1.0, N)
    ne = np.array([s * 5.0e19 * (1.0 - 0.7 * psi ** 2) for s in scales])
    nc = 0.015 * ne
    zeff = 1.0 + Z_IMP * (Z_IMP - 1.0) * nc / ne
    te = np.array([3.0e3 * (1.0 - 0.9 * psi ** 2) + 50.0] * len(scales))
    with h5py.File(path, "w") as f:
        f["time"] = np.asarray(times_ms, dtype=float)
        f["psi_n"] = psi
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", 0.9 * te),
                     ("Zeff", zeff), ("n_12C6", nc)]:
            f[k] = v
        for k, v in [("n_e_err", 0.05 * ne), ("T_e_err", 0.04 * te),
                     ("T_12C6_err", 0.06 * te), ("Zeff_err", 0.03 * zeff),
                     ("n_12C6_err", 0.10 * nc)]:
            f[k] = v


def _dd(times=(1.0, 1.02)):
    psi = np.linspace(0.0, 1.0, N)
    ne = 5.0e19 * (1.0 - 0.7 * psi ** 2)
    nc = 0.015 * ne
    ni = ne - Z_IMP * nc
    ti = 2.5e3 * (1.0 - 0.9 * psi ** 2) + 50.0
    j_total = 6.0e5 * (1.0 - psi ** 2)
    p_eq = 1.602176634e-19 * (ne * ti + ni * ti + nc * ti)
    geo = eq_geometry(psi, p_eq)
    j_tor = jtor_from_jtotal(j_total, geo, -2.0)
    sl = {"time": None,
          "global_quantities": {"ip": 1.0e6, "li_1": 1.0, "li_3": 0.9},
          "boundary": {"outline": {"r": [1.2, 2.2, 1.7], "z": [0.0, 0.0, 0.8]}},
          "profiles_1d": {"psi": psi.tolist(), "pressure": p_eq.tolist(),
                          "j_tor": j_tor.tolist(), **geo}}
    cp = {"grid": {"psi": psi.tolist(), "rho_tor_norm": geo["rho_tor_norm"]},
          "j_tor": j_tor.tolist(), "j_total": j_total.tolist(),
          "j_ohmic": (0.9 * j_total).tolist(),
          "j_bootstrap": (0.1 * j_total).tolist(),
          "electrons": {"density_thermal": ne.tolist(), "temperature": ti.tolist()},
          "ion": [{"density_thermal": ni.tolist(), "temperature": ti.tolist(),
                   "label": "D", "element": [{"z_n": 1.0, "a": 2.0}]},
                  {"density_thermal": nc.tolist(), "temperature": ti.tolist(),
                   "label": "C12", "element": [{"z_n": 6.0, "a": 12.0}]}]}
    return {"equilibrium": {"time": list(times),
                            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-2.0] * len(times)},
                            "time_slice": [dict(sl, time=t) for t in times]},
            "core_profiles": {"time": list(times),
                              "profiles_1d": [dict(cp) for _ in times]}}


@pytest.fixture()
def files(tmp_path):
    cdf = str(tmp_path / "ida.cdf")
    _ida_cdf(cdf)
    ddp = tmp_path / "dd_sim.json"
    ddp.write_text(json.dumps(_dd()))
    return str(ddp), cdf


def _read(ddp, cdf, time=1.0, ks="ida_hybrid", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(
            ImasSource(ids_path=ddp, time=time, ida_path=cdf,
                       impurity_Z=Z_IMP, **kw), kinetic_source=ks)


def test_the_reader_reads_the_kinetics_at_ida_time_and_the_currents_at_time(files):
    ddp, cdf = files
    at_t = _read(ddp, cdf)
    at_ida = _read(ddp, cdf, ida_time=1.01)
    np.testing.assert_allclose(at_ida.ne, 1.3 * at_t.ne, rtol=1e-12)
    for k in ("j_BS", "Ip_target"):
        np.testing.assert_array_equal(getattr(at_ida, k), getattr(at_t, k))
    m = at_ida.li_metrics["ida_time_match"]
    assert m["ida_time_used"] == 1.01 and m["fuse_time_cp"] == 1.0
    assert m["ida_time_configured"] == 1.01 and m["dt"] == pytest.approx(0.0)
    json.dumps(at_ida.li_metrics)            # archivable as li_metrics_json
    assert at_t.li_metrics["ida_time_match"]["ida_time_used"] == 1.0


def test_the_reader_refuses_an_ida_time_off_the_ida_file(files):
    ddp, cdf = files
    with pytest.raises(ValueError, match="no IDA slice within"):
        _read(ddp, cdf, ida_time=1.3)


def test_ida_time_outside_ida_hybrid_is_refused(files):
    """Review B6: it would move only the IDA sigmas, not the kinetics."""
    ddp, cdf = files
    with pytest.raises(ValueError, match="ida_time=1.01 is set"):
        _read(ddp, cdf, ks="fuse", ida_time=1.01)
    _read(ddp, cdf, ks="fuse")                         # unset: fine


# ---------------------------------------------------------------------------
#  set_slice / run_slices
# ---------------------------------------------------------------------------
def _bouquet(ida_time=None):
    from bouquet.config import BouquetConfig, SolverConfig
    from bouquet.run import Bouquet
    return Bouquet(BouquetConfig(
        source=ImasSource(ids_path="x.json", time=1.04, ida_path="ida.cdf",
                          ida_time=ida_time),
        solver=SolverConfig(mesh_path="unused"), output_header="unused"))


def test_set_slice_keeps_applies_or_resets_ida_time():
    b = _bouquet(ida_time=1.033)
    with pytest.warns(UserWarning, match="keeps ImasSource.ida_time=1.033"):
        b.set_slice(time=1.08)
    assert b.config.source.ida_time == 1.033             # kept, not wiped
    b.set_slice(ida_time=1.074)                          # alone: applied
    assert (b.config.source.time, b.config.source.ida_time) == (1.08, 1.074)
    b.set_slice(time=1.12, ida_time=None)                # explicit reset
    assert b.config.source.ida_time is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b.set_slice(time=1.16)                           # nothing to keep


def test_set_slice_ida_time_on_a_source_without_one_raises():
    from bouquet.config import (BouquetConfig, ReconstructionSource,
                                SolverConfig)
    from bouquet.run import Bouquet
    b = Bouquet(BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g", profiles_path="p"),
        solver=SolverConfig(mesh_path="unused"), output_header="unused"))
    with pytest.raises(TypeError, match="no ida_time"):
        b.set_slice(ida_time=1.0)


def _stub_slices(b, monkeypatch):
    from types import SimpleNamespace
    seen = []
    monkeypatch.setattr(b, "setup_solver", lambda: None)
    monkeypatch.setattr(b, "filter", lambda *a, **k: None)
    monkeypatch.setattr(b, "generate", lambda *a, **k: None)
    monkeypatch.setattr(b, "selected_indices", lambda which: [])

    def prep():
        seen.append((b.config.source.time, b.config.source.ida_time))
        b.baseline = SimpleNamespace(l_i_target=1.0, Ip_target=1e6)

    monkeypatch.setattr(b, "prepare_baseline", prep)
    return seen


def test_run_slices_carries_a_per_slice_ida_time(monkeypatch):
    """Review B3: run_slices wiped ida_time on every slice."""
    b = _bouquet()
    seen = _stub_slices(b, monkeypatch)
    res = b.run_slices([1.04, 1.08, 1.12], ida_times=[1.033, 1.074, None])
    assert seen == [(1.04, 1.033), (1.08, 1.074), (1.12, None)]
    assert res[1040]["ida_time"] == 1.033 and "ida_time" not in res[1120]
    with pytest.raises(ValueError, match="ida_times must match"):
        b.run_slices([1.04, 1.08], ida_times=[1.033])


def test_run_slices_refuses_a_configured_ida_time_without_ida_times(monkeypatch):
    b = _bouquet(ida_time=1.033)
    _stub_slices(b, monkeypatch)
    with pytest.raises(ValueError, match="pass ida_times="):
        b.run_slices([1.04, 1.08])
    b2 = _bouquet()
    seen = _stub_slices(b2, monkeypatch)
    b2.run_slices([1.04, 1.08])                          # unpaired: as before
    assert seen == [(1.04, None), (1.08, None)]
