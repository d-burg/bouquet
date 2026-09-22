"""Every core_sources current reaches the IMAS baseline, held fixed.

The reader used to take the beam alone (on the PARENT time index) and leave
EC/LH/IC, fusion-driven, sawteeth and any unlisted source in the inductive
residual, where the draws perturb it and the ohmic closure rescales it.
"""
import json
import warnings

import numpy as np
import pytest

from bouquet.config import ImasSource
from bouquet.io.imas import read_fuse_currents, read_imas_baseline
from bouquet.physics import parallel_to_toroidal

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
    j_tor = 6.0e5 * shape
    j_total = 1.05 * j_tor                  # parallel != toroidal, as in a real dd
    ne = 5e19 * (1 - 0.8 * psi ** 2) + 1e18
    ti = 2.5e3 * (1.0 - 0.9 * psi ** 2) + 50.0
    ni, nc = 0.9 * ne, (0.1 * ne) / 6.0
    p_eq = EC * (ne * ti + ni * ti + nc * ti)
    cs_t = [0.9, 1.0, 1.1]
    bump = lambda c, w: np.exp(-((psi - c) / w) ** 2)   # noqa: E731
    if saw_of_t is None:
        saw_of_t = lambda t: 1e4 * (t - 0.95) * (bump(0.1, 0.05) - bump(0.25, 0.05))  # noqa: E731
    rho_c = np.linspace(0.0, 1.0, 17)                   # IC on a coarser rho grid
    sources = [
        _src(2, "beam A", cs_t, lambda t: 2e4 * t * bump(0.3, 0.2)),
        _src(2, "beam B", cs_t, lambda t: 1e4 * t * bump(0.5, 0.2)),
        _src(3, "ec", cs_t, lambda t: 5e3 * t * bump(0.4, 0.05)),
        _src(5, "ic", cs_t, lambda t: 3e3 * t * np.exp(-((rho_c - 0.6) / 0.2) ** 2),
             grid=rho_c),
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
                                "j_tor": j_tor.tolist()},
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
        return read_imas_baseline(ImasSource(ids_path=path, time=1.0, **kw),
                                  p_fast_reduction="sum")


def _tor(dd, j_par):
    cp = dd["core_profiles"]["profiles_1d"][0]
    return parallel_to_toroidal(np.asarray(j_par, float),
                                j_parallel_total=np.asarray(cp["j_total"], float),
                                j_tor_total=np.asarray(cp["j_tor"], float))


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


class TestChannels:
    def test_every_source_lands_in_its_fixed_channel(self, tmp_path):
        dd = _dd()
        bl = _read(_write(tmp_path, dd))
        np.testing.assert_allclose(bl.j_NBI, _tor(dd, _par(dd, 2)), rtol=1e-12)
        np.testing.assert_allclose(bl.j_RF, _tor(dd, _par(dd, 3) + _par(dd, 5)),
                                   rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(bl.j_other, _tor(dd, _par(dd, 6) + _par(dd, 701)),
                                   rtol=1e-12, atol=1e-9)
        assert bl.fuse_currents["channels"] == {
            "nbi": "j_NBI", "ec": "j_RF", "ic": "j_RF", "fusion": "j_other",
            "sawteeth": "j_other"}

    def test_the_decomposition_still_sums_to_j_tor(self, tmp_path):
        bl = _read(_write(tmp_path, _dd()))
        np.testing.assert_allclose(
            bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF + bl.j_other, bl.j_phi,
            rtol=0, atol=1e-9 * np.max(np.abs(bl.j_phi)))

    def test_sawteeth_in_ohmic_moves_it_to_the_inductive_part(self, tmp_path):
        dd = _dd()
        path = _write(tmp_path, dd)
        fixed, ohmic = _read(path), _read(path, sawteeth_in_ohmic=True)
        saw = _tor(dd, _par(dd, 701))
        assert np.max(np.abs(saw)) > 0
        np.testing.assert_allclose(ohmic.j_other, fixed.j_other - saw, atol=1e-9)
        np.testing.assert_allclose(ohmic.j_inductive, fixed.j_inductive + saw, atol=1e-9)
        np.testing.assert_array_equal(ohmic.j_phi, fixed.j_phi)
        assert ohmic.fuse_currents["channels"]["sawteeth"] == "j_inductive"


class TestSourceTimes:
    def test_a_short_source_is_taken_on_its_own_time_array(self, tmp_path):
        # sawteeth has one slice fewer ([1.0, 1.1] vs [0.9, 1.0, 1.1]): the
        # parent's nearest index (1) lands on its 1.1 s profile.
        dd = _dd()
        fc = read_fuse_currents(dd, 1.0)
        np.testing.assert_allclose(fc["sources"]["sawteeth"], _par(dd, 701, 1.0))
        assert fc["source_times"]["sawteeth"] == 1.0
        assert not np.allclose(_par(dd, 701, 1.0), _par(dd, 701, 1.1))

    def test_the_sawtooth_gate_reads_the_same_slice(self, tmp_path):
        # idle at 1.0 s, active only from 1.1 s: the old parent index said active
        on = lambda t: (t > 1.05) * 1e4 * np.linspace(1, 0, N)  # noqa: E731
        bl = _read(_write(tmp_path, _dd(saw_of_t=on)))
        assert bl.sawtooth["present"] and not bl.sawtooth["active"]


class TestNothingDroppedSilently:
    def test_an_unlisted_index_is_held_fixed_and_named(self, tmp_path):
        extra = [_src(999, "mystery", [0.9, 1.0, 1.1], lambda t: 2e3 * np.ones(N))]
        dd = _dd(extra=extra)
        with pytest.warns(UserWarning, match="unlisted.*index_999"):
            bl = read_imas_baseline(ImasSource(ids_path=_write(tmp_path, dd), time=1.0),
                                    p_fast_reduction="sum")
        np.testing.assert_allclose(
            bl.j_other, _tor(dd, _par(dd, 6) + _par(dd, 701) + _par(dd, 999)), atol=1e-9)

    def test_an_unplaceable_source_warns_and_stays_in_the_residual(self, tmp_path):
        extra = [{"identifier": {"index": 4, "name": "lh"},
                  "profiles_1d": [{"time": t, "j_parallel": [1e3] * 7}
                                  for t in (0.9, 1.0, 1.1)]}]
        dd = _dd(extra=extra)
        with pytest.warns(UserWarning, match="'lh'.*could not be read"):
            bl = read_imas_baseline(ImasSource(ids_path=_write(tmp_path, dd), time=1.0),
                                    p_fast_reduction="sum")
        assert ("lh", 4) in [(n, i) for n, i, _ in bl.fuse_currents["skipped"]]
        np.testing.assert_allclose(bl.j_RF, _tor(dd, _par(dd, 3) + _par(dd, 5)),
                                   atol=1e-9)

    def test_the_unattributed_current_is_recorded(self, tmp_path):
        dd = _dd()
        bl = _read(_write(tmp_path, dd))
        cp = dd["core_profiles"]["profiles_1d"][0]
        expect = (np.asarray(cp["j_total"]) - 0.9 * np.asarray(cp["j_total"])
                  - sum(_par(dd, i) for i in (2, 3, 5, 6, 701)))
        np.testing.assert_allclose(bl.fuse_currents["unattributed_parallel"], expect,
                                   atol=1e-9)


def test_the_draws_get_j_other():
    """generate_bouquet / perturb_kinetic_equilibrium carry j_other into every
    fixed-current sum j_NBI and j_RF enter."""
    import inspect
    import bouquet.run as brun
    import bouquet.TokaMaker_interface as tmi
    t = inspect.getsource(tmi)
    assert t.count("+ np.asarray(j_RF, dtype=float)") == t.count(
        "+ np.asarray(j_other, dtype=float)") == 5
    assert "j_other=getattr(bl, \"j_other\", None)" in inspect.getsource(brun)
