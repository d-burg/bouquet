"""The IMAS reader holds every driven core_sources current fixed, classified as
the engine's IDS adapter does (adapters._ids_driven_currents).
"""
import json
import warnings

import numpy as np

from bouquet.config import ImasSource
from bouquet.io.imas import read_imas_baseline
from bouquet.physics import jpar_to_jphi_tokamaker

from _imas_geometry import current_geom, eq_geometry, jtor_from_jtotal

N = 33
EC = 1.602176634e-19


def _src(index, name, times, prof_of_t, grid=None):
    prs = []
    for t in times:
        p = {"time": t, "j_parallel": prof_of_t(t).tolist()}
        if grid is not None:
            p["grid"] = {"rho_tor_norm": grid.tolist()}
        prs.append(p)
    return {"identifier": {"index": index, "name": name}, "profiles_1d": prs}


def _dd(saw_times=(1.0, 1.1), saw_of_t=None, extra=()):
    psi = np.linspace(0.0, 1.0, N)
    rho = np.sqrt(psi)
    shape = 1.0 - psi ** 2
    j_total = 6.3e5 * shape
    ne = 5e19 * (1 - 0.8 * psi ** 2) + 1e18
    ti = 2.5e3 * (1.0 - 0.9 * psi ** 2) + 50.0
    ni, nc = 0.9 * ne, (0.1 * ne) / 6.0
    p_eq = EC * (ne * ti + ni * ti + nc * ti)
    geo = eq_geometry(psi, p_eq)                         # rho_tor_norm = sqrt(psi_N)
    j_tor = jtor_from_jtotal(j_total, geo, -2.0)
    cs_t = [0.9, 1.0, 1.1]
    bump = lambda c, w: np.exp(-((psi - c) / w) ** 2)   # noqa: E731
    if saw_of_t is None:
        saw_of_t = lambda t: 1e4 * (t - 0.95) * (bump(0.1, 0.05) - bump(0.25, 0.05))  # noqa: E731
    sources = [
        _src(2, "beam A", cs_t, lambda t: 2e4 * t * bump(0.3, 0.2)),
        _src(2, "beam B", cs_t, lambda t: 1e4 * t * bump(0.5, 0.2)),
        _src(3, "ec", cs_t, lambda t: 5e3 * t * bump(0.4, 0.05)),
        _src(5, "ic", cs_t, lambda t: 3e3 * t * bump(0.6, 0.2)),
        _src(6, "fusion", cs_t, lambda t: 1e3 * t * shape),
        _src(7, "ohmic", cs_t, lambda t: 0.8 * j_total),
        _src(13, "bootstrap", cs_t, lambda t: 0.1 * j_total),
        _src(8, "brem", cs_t, lambda t: np.zeros(N)),     # radiation: no current
        _src(701, "sawteeth", list(saw_times), saw_of_t),
        *extra,
    ]
    return {
        "equilibrium": {
            "time": [1.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-2.0]},
            "time_slice": [{
                "time": 1.0,
                "global_quantities": {"ip": 1.0e6, "li_1": 1.0, "li_3": 0.9},
                "boundary": {"outline": {"r": [1.2, 2.2, 1.7], "z": [0.0, 0.0, 0.8]}},
                "profiles_1d": {"psi": psi.tolist(), "pressure": p_eq.tolist(),
                                "j_tor": j_tor.tolist(), **geo},
            }],
        },
        "core_profiles": {
            "time": [1.0],
            "profiles_1d": [{
                "grid": {"psi": psi.tolist(), "rho_tor_norm": rho.tolist()},
                "j_tor": j_tor.tolist(), "j_total": j_total.tolist(),
                "j_ohmic": (0.8 * j_total).tolist(),
                "j_bootstrap": (0.1 * j_total).tolist(),
                "j_non_inductive": (0.2 * j_total).tolist(),
                "electrons": {"density_thermal": ne.tolist(), "temperature": ti.tolist()},
                "ion": [{"density_thermal": ni.tolist(), "temperature": ti.tolist(),
                         "label": "D", "element": [{"z_n": 1.0, "a": 2.0}]},
                        {"density_thermal": nc.tolist(), "temperature": ti.tolist(),
                         "label": "C12", "element": [{"z_n": 6.0, "a": 12.0}]}],
            }],
        },
        "core_sources": {"time": cs_t, "source": sources},
    }


def _write(tmp_path, dd):
    p = tmp_path / "dd.json"
    p.write_text(json.dumps(dd))
    return str(p)


def _read(path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                                  p_fast_reduction="sum", **kw)


def _tor(dd, j_par):
    p1 = dd["equilibrium"]["time_slice"][0]["profiles_1d"]
    return jpar_to_jphi_tokamaker(np.asarray(j_par, float), current_geom(p1, -2.0))


def _par(dd, index, t=1.0):
    """Sum of index's j_parallel at time t, on the core_profiles grid."""
    rho = np.asarray(dd["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"])
    out = np.zeros(N)
    for s in dd["core_sources"]["source"]:
        if s["identifier"]["index"] != index:
            continue
        p = next(p for p in s["profiles_1d"] if p["time"] == t)
        j = np.asarray(p["j_parallel"], float)
        if j.size != N:
            j = np.interp(rho, np.asarray(p["grid"]["rho_tor_norm"]), j)
        out += j
    return out


def test_every_driven_source_lands_in_its_channel(tmp_path):
    dd = _dd()
    bl = _read(_write(tmp_path, dd))
    np.testing.assert_allclose(bl.j_NBI, _tor(dd, _par(dd, 2)), rtol=1e-12)
    np.testing.assert_allclose(bl.j_RF, _tor(dd, _par(dd, 3) + _par(dd, 5)),
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(bl.j_other, _tor(dd, _par(dd, 6) + _par(dd, 701)),
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(bl.j_sawteeth, _tor(dd, _par(dd, 701)),
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + bl.j_other, bl.j_phi,
        rtol=0, atol=1e-9 * np.max(np.abs(bl.j_phi)))


def test_an_aggregate_entry_is_never_added(tmp_path):
    n = N
    total = _src(1, "total", [0.9, 1.0, 1.1], lambda t: 9e4 * np.ones(n))
    dd = _dd(extra=(total,))
    bl = _read(_write(tmp_path, dd))
    ref = _read(_write(tmp_path, _dd()))
    np.testing.assert_array_equal(bl.j_other, ref.j_other)
    np.testing.assert_array_equal(bl.j_RF, ref.j_RF)


def test_without_a_sawteeth_source(tmp_path):
    dd = _dd()
    dd["core_sources"]["source"] = [s for s in dd["core_sources"]["source"]
                                    if s["identifier"]["index"] != 701]
    bl = _read(_write(tmp_path, dd))
    assert bl.j_sawteeth.shape == bl.j_other.shape and not np.any(bl.j_sawteeth)
