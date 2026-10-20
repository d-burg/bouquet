"""Phase 2: geqdsk + profiles bundle export from the archive.

Builds a synthetic 2-draw archive (real d3dlike.geqdsk bytes + profiles +
eq_fsa + coils) and checks DrawView/ScanView.extract writes geqdsk files and a
self-describing profiles JSON, honouring the selection.
"""
import json
import os

import numpy as np
import pytest
import h5py

import bouquet as bq
from bouquet.archive import BouquetArchive

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEQ = os.path.join(_HERE, "data", "d3dlike.geqdsk")
_PEQ = np.linspace(0.0, 1.0, 20)


def _make_archive(path):
    with h5py.File(path, "w") as hf:
        hf.attrs["schema_version"] = 2
        with open(_GEQ, "rb") as fh:
            raw = fh.read()
        for i in range(2):
            g = hf.require_group(f"scan/0/{i}")
            g.attrs["l_i(1)"] = 1.05 + 0.01 * i
            g.attrs["l_i(3)"] = 0.84
            g.attrs["selected"] = (i == 0)          # only draw 0 in-spec
            g.create_dataset("psi_N", data=_PEQ)
            g.create_dataset("jphi_total", data=(1 + i) * np.ones_like(_PEQ))
            g.create_dataset("n_e", data=np.linspace(5, 1, 20))
            g.create_dataset("eqdsk", data=np.void(raw))
            g.create_dataset("coil_currents", data=np.array([1.0e4, -1.0e4]))
            g.create_dataset("coil_names",
                             data=np.array(["F9A", "F9B"], dtype=h5py.string_dtype()))
            fg = g.create_group("eq_fsa")
            fg.create_dataset("psi_N", data=_PEQ)
            fg.create_dataset("avg_inv_R2", data=0.36 * np.ones_like(_PEQ))


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestBundleExport:
    def test_draw_extract_geqdsk_and_profiles(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc)
        d = BouquetArchive(arc)["0"][0]
        paths = d.extract(str(tmp_path / "out"), formats=("geqdsk", "profiles"))
        assert set(paths) == {"geqdsk", "profiles"}
        # geqdsk round-trips to a parseable equilibrium
        eq = bq.read_geqdsk(paths["geqdsk"])
        assert abs(eq.Ip) > 0
        # profiles JSON is self-describing and carries the draw state
        doc = json.load(open(paths["profiles"]))
        assert doc["count"] == 0 and doc["scan_key"] == "0"
        assert "jphi_total" in doc["profiles"] and "psi_N" in doc["profiles"]
        assert doc["units"]["jphi_total"] == "A m^-2"
        assert doc["scalars"]["l_i(1)"] == 1.05
        assert doc["coil_currents_A"] == {"F9A": 1.0e4, "F9B": -1.0e4}
        assert "eq_fsa" in doc and "avg_inv_R2" in doc["eq_fsa"]
        assert np.allclose(doc["profiles"]["jphi_total"], 1.0)

    def test_scan_extract_honours_selection(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc)
        sc = BouquetArchive(arc)["0"]
        # selected -> only draw 0
        sel = sc.extract(str(tmp_path / "sel"), formats=("geqdsk",),
                         selection="selected")
        assert list(sel) == [0]
        # all -> both draws, each with a geqdsk
        alld = sc.extract(str(tmp_path / "all"), formats=("geqdsk", "profiles"),
                          selection="all")
        assert sorted(alld) == [0, 1]
        assert all("geqdsk" in v and "profiles" in v for v in alld.values())
        # per-draw jphi_total differs in the exported profiles (draws are distinct)
        j0 = json.load(open(alld[0]["profiles"]))["profiles"]["jphi_total"]
        j1 = json.load(open(alld[1]["profiles"]))["profiles"]["jphi_total"]
        assert not np.allclose(j0, j1)

    def test_pfile_skipped_when_absent(self, tmp_path):
        arc = str(tmp_path / "run.h5"); _make_archive(arc)
        d = BouquetArchive(arc)["0"][0]
        paths = d.extract(str(tmp_path / "out"), formats=("geqdsk", "pfile"))
        assert "geqdsk" in paths and "pfile" not in paths   # no pfile stored
