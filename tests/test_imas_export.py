"""Exact-fidelity IMAS/OMAS export from the captured live-equilibrium FSA block.

Builds a synthetic single-draw archive carrying an ``eq_fsa`` group + a minimal
template IDS, and checks that ``write_imas_draw`` converts bouquet's TokaMaker
``jphi`` components to IMAS ``j_tor`` and parallel ``<j.B>/B0`` with the draw's
own geometry (``fidelity="exact"``), that ``"exact"`` raises without a complete
captured block, and that ``"auto"`` falls back to the template's geometry.
"""
import json
import os
import warnings

import numpy as np
import pytest
import h5py

import bouquet as bq
from bouquet.io.imas import (_fuse_current_geometry, archived_pressure_term,
                             write_imas_draw)
from bouquet.physics import (jpar_to_jphi_tokamaker,
                             jphi_tokamaker_to_jpar, jphi_tokamaker_to_jtor_imas,
                             jtor_imas_to_jphi_tokamaker)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEQ = os.path.join(_HERE, "data", "d3dlike.geqdsk")

# draw current components (TokaMaker jphi), on the equilibrium psi_N grid
_PEQ = np.linspace(0.0, 1.0, 24)
_J_PHI = 8.0e5 * (1 - _PEQ**2) + 1.0e5
_J_IND = 6.0e5 * (1 - _PEQ**2)
_J_BS = 2.0e5 * np.exp(-((_PEQ - 0.9) / 0.08) ** 2)
# captured eq_fsa (its own grid); smooth geometric factors
_PF = np.linspace(0.0, 1.0, 16)
_EQ_FSA = {
    "psi_N": _PF,
    "F": np.full(16, 3.4),
    "avg_R": 1.70 + 0.04 * _PF,
    "avg_inv_R": 0.60 - 0.05 * _PF,
    "avg_B2": 4.2 - 0.3 * _PF,
    "avg_inv_R2": (0.60 - 0.05 * _PF) ** 2 * 1.02,   # ~<1/R>^2, +Bp content
    "pprime": 6.0e4 * (1.0 - 0.5 * _PF),
}
_LEGACY_KEYS = ("psi_N", "F", "avg_inv_R", "avg_B2", "avg_inv_R2")
_B0 = 2.0


def _fsa_geom_on(x):
    """The captured geometry on ``x``, with ``B0`` (the writer's ``geom``)."""
    g = {k: np.interp(x, _PF, v) for k, v in _EQ_FSA.items() if k != "psi_N"}
    g["B0"] = _B0
    return g


def _make_archive(path, with_fsa=True, legacy_fsa=False,
                  convention="pressure_in_bootstrap", j_ind=None, j_bs=None,
                  j_pressure=None):
    """``convention``: the draw's ``current_split_convention`` attr (None:
    no attr, an archive from before PR #64).  The default archive is the
    PR #64 one (p'G inside j_BS) these checks were written for."""
    j_ind = _J_IND if j_ind is None else j_ind
    j_bs = _J_BS if j_bs is None else j_bs
    with h5py.File(path, "w") as hf:
        g = hf.require_group("scan/0/0")
        if convention is not None:
            g.attrs["current_split_convention"] = convention
        if j_pressure is not None:
            g.create_dataset("j_pressure", data=j_pressure)
        g.create_dataset("psi_N", data=_PEQ)
        g.create_dataset("psi_N_kinetic", data=np.linspace(0, 1, 20))
        for k in ("n_e", "T_e", "n_i", "T_i"):
            g.create_dataset(k, data=np.linspace(1.0, 0.1, 20))
        g.create_dataset("j_phi", data=_J_PHI)
        g.create_dataset("j_inductive", data=j_ind)
        g.create_dataset("j_BS", data=j_bs)
        with open(_GEQ, "rb") as fh:
            g.create_dataset("eqdsk", data=np.void(fh.read()))
        if with_fsa:
            fg = g.create_group("eq_fsa")
            for k, v in _EQ_FSA.items():
                if legacy_fsa and k not in _LEGACY_KEYS:
                    continue                     # capture predating avg_R/pprime
                fg.create_dataset(k, data=np.asarray(v, float))


def _make_template(path, with_geometry=True):
    psi = np.linspace(-0.4, 0.6, 33)                 # Wb, monotonic
    npt = psi.size
    rho = np.sqrt(np.linspace(0.0, 1.0, npt))
    re = np.linspace(0.0, 1.0, 41)                   # equilibrium's own grid
    p1 = {"psi": np.linspace(-0.4, 0.6, 41).tolist()}
    if with_geometry:                                # COCOS 11: F, b0 < 0
        p1.update({
            "rho_tor_norm": re.tolist(),
            "f": (-3.4 + 0.02 * re).tolist(),
            "gm8": (1.70 + 0.03 * re).tolist(),
            "gm9": (0.60 - 0.04 * re).tolist(),
            "gm1": ((0.60 - 0.04 * re) ** 2 * 1.015).tolist(),
            "gm5": (4.1 - 0.2 * re).tolist(),
            "dpressure_dpsi": (-1.0e4 * (1.0 - 0.6 * re)).tolist(),
        })
    template = {
        "equilibrium": {
            "time": [0.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-_B0]},
            "time_slice": [{"global_quantities": {}, "profiles_1d": p1}],
        },
        "core_profiles": {
            "time": [0.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-_B0]},
            "profiles_1d": [{
                "grid": {"psi": psi.tolist(), "rho_tor_norm": rho.tolist()},
                "electrons": {},
                "ion": [{"element": [{"z_n": 1.0}]}],
                "j_total": (np.ones(npt) * 5e5).tolist(),
                "j_tor": (np.ones(npt) * 4e5).tolist(),
                "j_non_inductive": np.zeros(npt).tolist(),
            }],
        },
    }
    with open(path, "w") as fh:
        json.dump(template, fh)
    return psi, template


def _P(x):
    """p'G of the archived eqdsk on ``x``: what the writer subtracts."""
    with open(_GEQ, "rb") as fh:
        return archived_pressure_term(fh.read(), x)


def _check_currents(cp, psiN_t, geom, a5=None):
    """The written IDS currents are the exact conversions of the draw's jphi
    (``j_tor`` on ``a5``, default ``geom``)."""
    a5 = geom if a5 is None else a5
    jphi = np.interp(psiN_t, _PEQ, _J_PHI)
    j_ind = np.interp(psiN_t, _PEQ, _J_IND)
    j_bs = np.interp(psiN_t, _PEQ, _J_BS)
    pt = _P(psiN_t)
    assert np.allclose(cp["j_tor"], jphi_tokamaker_to_jtor_imas(jphi, a5), rtol=1e-12)
    assert np.allclose(cp["j_total"], jphi_tokamaker_to_jpar(jphi - pt, geom), rtol=1e-12)
    assert np.allclose(cp["j_ohmic"], jphi_tokamaker_to_jpar(j_ind, geom), rtol=1e-12)
    assert np.allclose(cp["j_bootstrap"], jphi_tokamaker_to_jpar(j_bs - pt, geom),
                       rtol=1e-12, atol=1e-9)
    assert np.allclose(cp["j_non_inductive"],
                       np.asarray(cp["j_total"]) - np.asarray(cp["j_ohmic"]))
    # reading the IDS back recovers bouquet's jphi (the reader's direction)
    back_total = jtor_imas_to_jphi_tokamaker(np.asarray(cp["j_tor"]), a5)
    assert np.allclose(back_total, jphi, rtol=1e-12)
    back_par = jpar_to_jphi_tokamaker(np.asarray(cp["j_total"]), geom) + pt
    assert np.allclose(back_par, jphi, rtol=1e-12)
    back_bs = jpar_to_jphi_tokamaker(np.asarray(cp["j_bootstrap"]), geom) + pt
    assert np.allclose(back_bs, j_bs, rtol=1e-10, atol=1e-6)


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestExactImasExport:
    def test_exact_uses_captured_geometry(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=True)
        tmpl = str(tmp_path / "tmpl.json"); psi, _ = _make_template(tmpl)
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="exact")

        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        geom = _fsa_geom_on(psiN_t)                      # |b0|: F > 0 here
        _check_currents(cp, psiN_t, geom)
        # the pressure term really is in play (non-vacuous)
        assert np.max(np.abs(_P(psiN_t))) > 1e3

    def test_writes_only_the_exported_slice(self, tmp_path):
        # a multi-slice template: every time series is cut to the sample
        # nearest `time`, on its own time base; static data is kept
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=True)
        tmpl = str(tmp_path / "tmpl.json"); psi, template = _make_template(tmpl)
        times = [0.5, 1.0, 1.5]
        for ids, aos in (("equilibrium", "time_slice"), ("core_profiles", "profiles_1d")):
            d = template[ids]
            d["time"] = times
            d["vacuum_toroidal_field"]["b0"] = [-3.0, -_B0, -1.0]
            d[aos] = [dict(json.loads(json.dumps(d[aos][0])), time=t) for t in times]
        template["core_profiles"]["global_quantities"] = {"ip": [1.0, 2.0, 3.0]}
        template["core_sources"] = {        # one slice fewer than its IDS
            "time": times, "source": [{"profiles_1d": [{"time": 1.0}, {"time": 1.5}]}]}
        template["pf_active"] = {"coil": [{"current": {
            "time": [0.0, 0.9, 1.2, 2.0], "data": [0.0, 1.0, 2.0, 3.0]}}]}
        template["wall"] = {"description_2d": [{"limiter": {"r": [1.0, 2.0, 3.0]}}]}
        with open(tmpl, "w") as fh:
            json.dump(template, fh)
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, time=1.1, fidelity="exact")

        dd = json.load(open(out))
        for ids, aos in (("equilibrium", "time_slice"), ("core_profiles", "profiles_1d")):
            assert dd[ids]["time"] == [1.0]
            assert dd[ids]["vacuum_toroidal_field"]["b0"] == [-_B0]
            assert [s["time"] for s in dd[ids][aos]] == [1.0]
        assert dd["core_profiles"]["global_quantities"]["ip"] == [2.0]
        assert dd["core_sources"]["time"] == [1.0]
        assert dd["core_sources"]["source"][0]["profiles_1d"] == [{"time": 1.0}]
        assert dd["pf_active"]["coil"][0]["current"] == {"time": [1.2], "data": [2.0]}
        assert dd["wall"] == template["wall"]
        # the draw is written onto the kept slice
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        geom = {k: np.interp(psiN_t, _PF, _EQ_FSA[k]) for k in _EQ_FSA if k != "psi_N"}
        geom["B0"] = _B0
        _check_currents(dd["core_profiles"]["profiles_1d"][0], psiN_t, geom)
        assert "profiles_2d" in dd["equilibrium"]["time_slice"][0]

    def test_exact_without_capture_raises(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        with pytest.raises(ValueError, match="no complete captured eq_fsa"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"),
                            scan_key=0, fidelity="exact")

    def test_exact_with_legacy_capture_raises(self, tmp_path):
        # an eq_fsa written before avg_R / pprime were captured is not enough
        arc = str(tmp_path / "run.h5"); _make_archive(arc, legacy_fsa=True)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        with pytest.raises(ValueError, match="no complete captured eq_fsa"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"),
                            scan_key=0, fidelity="exact")

    @pytest.mark.parametrize("legacy", [False, True])
    def test_auto_falls_back_to_template_geometry(self, tmp_path, legacy):
        # no eq_fsa -> the template's baseline equilibrium geometry; an older
        # one (no avg_R / pprime) -> its own kappa, the template's for j_tor
        arc = str(tmp_path / "run.h5")
        _make_archive(arc, with_fsa=legacy, legacy_fsa=legacy)
        tmpl = str(tmp_path / "tmpl.json"); psi, template = _make_template(tmpl)
        out = str(tmp_path / "draw.json")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="auto")
        assert any("template (baseline) geometry" in str(x.message) for x in w) == legacy
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        geom = _fuse_current_geometry(
            template["equilibrium"], 0,
            template["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"])
        assert geom["B0"] == -_B0 and np.all(geom["F"] < 0)   # COCOS-11 signs kept
        _check_currents(cp, psiN_t, _fsa_geom_on(psiN_t) if legacy else geom,
                        a5=geom)
        assert np.all(np.asarray(cp["j_total"]) > 0)          # signs cancel

    def test_reconstruct_without_template_geometry_raises(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl, with_geometry=False)
        with pytest.raises(ValueError, match="template equilibrium"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"),
                            scan_key=0, fidelity="reconstruct")

    def test_bad_fidelity_raises(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        with pytest.raises(ValueError, match="fidelity must be"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"),
                            scan_key=0, fidelity="bogus")

    def test_a_phi_n_archive_lands_on_the_template_phi_n_nodes(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        with h5py.File(arc, "a") as hf:
            hf.require_group("scan/0/_baseline").attrs["profile_coord"] = "phi_n"
        tmpl = str(tmp_path / "tmpl.json"); psi, template = _make_template(tmpl)
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        rho = psiN_t ** 0.4
        with open(tmpl) as fh:
            t = json.load(fh)
        t["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = rho.tolist()
        with open(tmpl, "w") as fh:
            json.dump(t, fh)
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="auto")
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        # template geometry (no eq_fsa), sampled at the template nodes
        geom = _fuse_current_geometry(template["equilibrium"], 0, rho)
        assert np.allclose(cp["j_tor"], jphi_tokamaker_to_jtor_imas(
            np.interp(rho ** 2, _PEQ, _J_PHI), geom), rtol=1e-10)


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestSplitConventions:
    """The same physical currents, archived with p'G in each bucket (schema
    current_split_convention), export to the same IMAS currents: p'G has
    zero <j.B> and comes off whichever bucket carries it (review PR64 B3)."""

    def _export(self, tmp_path, name, **kw):
        arc = str(tmp_path / f"{name}.h5")
        _make_archive(arc, **kw)
        tmpl = str(tmp_path / "tmpl.json")
        psi, _ = _make_template(tmpl)
        out = str(tmp_path / f"{name}.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="exact")
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        return cp, (psi - psi[0]) / (psi[-1] - psi[0])

    def test_every_convention_exports_the_same_currents(self, tmp_path):
        P_eq = _P(_PEQ)                       # p'G on the archive grid
        bs_fa = _J_BS - P_eq                  # the field-aligned bootstrap
        ref, x = self._export(tmp_path, "bs", convention="pressure_in_bootstrap",
                              j_bs=_J_BS)
        pre, _ = self._export(tmp_path, "pre64", convention=None,
                              j_ind=_J_IND + P_eq, j_bs=bs_fa)
        ind, _ = self._export(tmp_path, "ind", convention="pressure_in_inductive",
                              j_ind=_J_IND + P_eq, j_bs=bs_fa)
        sep, _ = self._export(tmp_path, "sep", convention="pressure_separate",
                              j_bs=bs_fa, j_pressure=P_eq)
        geom = _fsa_geom_on(x)
        Px = _P(x)
        for cp in (pre, ind, sep):
            np.testing.assert_allclose(cp["j_tor"], ref["j_tor"], rtol=1e-12)
        for cp in (pre, ind):        # sep subtracts the archived (coarser) P
            np.testing.assert_allclose(cp["j_total"], ref["j_total"], rtol=1e-12)
        # the eqdsk's own p'G (pre64 / inductive / bootstrap) on the export grid
        for cp in (pre, ind):        # the eqdsk's p'G comes off j_inductive
            np.testing.assert_allclose(cp["j_ohmic"], jphi_tokamaker_to_jpar(
                np.interp(x, _PEQ, _J_IND + P_eq) - Px, geom), rtol=1e-12)
            np.testing.assert_allclose(cp["j_bootstrap"], jphi_tokamaker_to_jpar(
                np.interp(x, _PEQ, bs_fa), geom), rtol=1e-12, atol=1e-9)
        # separate: the archived j_pressure, not the eqdsk's, comes off
        np.testing.assert_allclose(sep["j_ohmic"], jphi_tokamaker_to_jpar(
            np.interp(x, _PEQ, _J_IND), geom), rtol=1e-12)
        np.testing.assert_allclose(sep["j_bootstrap"], jphi_tokamaker_to_jpar(
            np.interp(x, _PEQ, bs_fa), geom), rtol=1e-12, atol=1e-9)
        jphi = np.interp(x, _PEQ, _J_PHI)
        np.testing.assert_allclose(sep["j_total"], jphi_tokamaker_to_jpar(
            jphi - np.interp(x, _PEQ, P_eq), geom), rtol=1e-12)
        assert np.max(np.abs(Px)) > 1e3


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestReconstructFidelityValues:
    """fidelity='reconstruct' converts with the TEMPLATE's own (baseline)
    equilibrium geometry; without a captured eq_fsa, 'auto' is the same."""

    def test_auto_without_capture_gives_the_same_values(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        a, r = str(tmp_path / "a.json"), str(tmp_path / "r.json")
        write_imas_draw(arc, 0, tmpl, a, scan_key=0, fidelity="auto")
        write_imas_draw(arc, 0, tmpl, r, scan_key=0, fidelity="reconstruct")
        assert json.load(open(a)) == json.load(open(r))
