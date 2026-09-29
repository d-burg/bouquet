"""Exact-fidelity IMAS/OMAS export from the captured live-equilibrium FSA block.

Builds a synthetic single-draw archive carrying an ``eq_fsa`` group + a minimal
template IDS, and checks that ``write_imas_draw`` converts the toroidal current
components to IMAS parallel ``<j.B>/B0`` with the draw's own geometry
(``fidelity="exact"``), that ``"exact"`` raises without a captured block, and
that ``"auto"`` falls back to the baseline-ratio reconstruction.
"""
import json
import os

import numpy as np
import pytest
import h5py

import bouquet as bq
from bouquet.io.imas import write_imas_draw
from bouquet.physics import toroidal_to_parallel

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEQ = os.path.join(_HERE, "data", "d3dlike.geqdsk")

# draw current components (toroidal j_phi), on the equilibrium psi_N grid
_PEQ = np.linspace(0.0, 1.0, 24)
_J_PHI = 8.0e5 * (1 - _PEQ**2) + 1.0e5
_J_IND = 6.0e5 * (1 - _PEQ**2)
_J_BS = 2.0e5 * np.exp(-((_PEQ - 0.9) / 0.08) ** 2)
# captured eq_fsa (its own grid); smooth geometric factors
_PF = np.linspace(0.0, 1.0, 16)
_EQ_FSA = {
    "psi_N": _PF,
    "F": np.full(16, 3.4),
    "avg_inv_R": 0.60 - 0.05 * _PF,
    "avg_B2": 4.2 - 0.3 * _PF,
    "avg_inv_R2": (0.60 - 0.05 * _PF) ** 2 * 1.02,   # ~<1/R>^2, +Bp content
}
_B0 = 2.0


def _make_archive(path, with_fsa=True):
    with h5py.File(path, "w") as hf:
        g = hf.require_group("scan/0/0")
        g.create_dataset("psi_N", data=_PEQ)
        g.create_dataset("psi_N_kinetic", data=np.linspace(0, 1, 20))
        for k in ("n_e", "T_e", "n_i", "T_i"):
            g.create_dataset(k, data=np.linspace(1.0, 0.1, 20))
        g.create_dataset("j_phi", data=_J_PHI)
        g.create_dataset("j_inductive", data=_J_IND)
        g.create_dataset("j_BS", data=_J_BS)
        with open(_GEQ, "rb") as fh:
            g.create_dataset("eqdsk", data=np.void(fh.read()))
        if with_fsa:
            fg = g.create_group("eq_fsa")
            for k, v in _EQ_FSA.items():
                fg.create_dataset(k, data=np.asarray(v, float))


def _make_template(path):
    psi = np.linspace(-0.4, 0.6, 33)                 # Wb, monotonic
    npt = psi.size
    template = {
        "equilibrium": {
            "time": [0.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-_B0]},
            "time_slice": [{"global_quantities": {}}],
        },
        "core_profiles": {
            "time": [0.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-_B0]},
            "profiles_1d": [{
                "grid": {"psi": psi.tolist()},
                "electrons": {},
                "ion": [{"element": [{"z_n": 1.0}]}],
                "j_total": (np.ones(npt) * 5e5).tolist(),
                "j_tor": (np.ones(npt) * 4e5).tolist(),
            }],
        },
    }
    with open(path, "w") as fh:
        json.dump(template, fh)
    return psi


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestExactImasExport:
    def test_exact_uses_captured_geometry(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=True)
        tmpl = str(tmp_path / "tmpl.json"); psi = _make_template(tmpl)
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="exact")

        ids = json.load(open(out))
        cp = ids["core_profiles"]["profiles_1d"][0]
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        # expected: same interp + geom + toroidal_to_parallel the code should do
        geom = {
            "F": np.interp(psiN_t, _PF, _EQ_FSA["F"]),
            "avg_inv_R": np.interp(psiN_t, _PF, _EQ_FSA["avg_inv_R"]),
            "avg_B2": np.interp(psiN_t, _PF, _EQ_FSA["avg_B2"]),
            "avg_inv_R2": np.interp(psiN_t, _PF, _EQ_FSA["avg_inv_R2"]),
            "B0": _B0,
        }
        exp_jtot = toroidal_to_parallel(np.interp(psiN_t, _PEQ, _J_PHI), geom=geom)
        exp_johm = toroidal_to_parallel(np.interp(psiN_t, _PEQ, _J_IND), geom=geom)
        exp_jbs = toroidal_to_parallel(np.interp(psiN_t, _PEQ, _J_BS), geom=geom)
        assert np.allclose(cp["j_total"], exp_jtot, rtol=1e-10)
        assert np.allclose(cp["j_ohmic"], exp_johm, rtol=1e-10)
        assert np.allclose(cp["j_bootstrap"], exp_jbs, rtol=1e-10)
        # j_tor stays the exact toroidal current (interp of the stored j_phi)
        assert np.allclose(cp["j_tor"], np.interp(psiN_t, _PEQ, _J_PHI), rtol=1e-10)

    def test_exact_without_capture_raises(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        with pytest.raises(ValueError, match="no captured eq_fsa"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"),
                            scan_key=0, fidelity="exact")

    def test_auto_falls_back_to_reconstruct(self, tmp_path):
        # no eq_fsa -> auto uses baseline ratio; parallel split differs from the
        # exact path but the file writes and j_tor is still exact
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); psi = _make_template(tmpl)
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="auto")
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        assert np.allclose(cp["j_tor"], np.interp(psiN_t, _PEQ, _J_PHI), rtol=1e-10)
        assert "j_bootstrap" in cp                       # reconstruct populated it

    def test_bad_fidelity_raises(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        with pytest.raises(ValueError, match="fidelity must be"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"),
                            scan_key=0, fidelity="bogus")


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestReconstructFidelityValues:
    """fidelity='reconstruct' converts with the TEMPLATE's ratio
    c = j_tor/j_total.  It used to read the template j_tor AFTER overwriting
    it with the draw's, so c = draw j_tor / template j_total: the exported
    j_total came out as the template's verbatim.  Check the VALUES of the
    parallel split, not just that the keys exist."""

    def test_parallel_split_uses_the_template_ratio(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); psi = _make_template(tmpl)
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="reconstruct")
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        psiN_t = (psi - psi[0]) / (psi[-1] - psi[0])
        c = 4e5 / 5e5                     # the template's j_tor / j_total
        jt = np.interp(psiN_t, _PEQ, _J_PHI)
        assert np.allclose(cp["j_tor"], jt, rtol=1e-12)
        assert np.allclose(cp["j_total"], jt / c, rtol=1e-12)
        assert np.allclose(cp["j_ohmic"], np.interp(psiN_t, _PEQ, _J_IND) / c,
                           rtol=1e-12)
        assert np.allclose(cp["j_bootstrap"], np.interp(psiN_t, _PEQ, _J_BS) / c,
                           rtol=1e-12)
        # witness: the exported total is the DRAW's, not the template's
        assert not np.allclose(cp["j_total"], 5e5)

    def test_auto_without_capture_gives_the_same_values(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        a, r = str(tmp_path / "a.json"), str(tmp_path / "r.json")
        write_imas_draw(arc, 0, tmpl, a, scan_key=0, fidelity="auto")
        write_imas_draw(arc, 0, tmpl, r, scan_key=0, fidelity="reconstruct")
        assert json.load(open(a)) == json.load(open(r))

    def test_template_without_j_tor_cannot_reconstruct(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=False)
        tmpl = str(tmp_path / "tmpl.json"); _make_template(tmpl)
        t = json.load(open(tmpl))
        del t["core_profiles"]["profiles_1d"][0]["j_tor"]
        json.dump(t, open(tmpl, "w"))
        with pytest.raises(ValueError, match="needs the template's own"):
            write_imas_draw(arc, 0, tmpl, str(tmp_path / "d.json"), scan_key=0,
                            fidelity="reconstruct")
