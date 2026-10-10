"""OFT i-file archiving (GenerationConfig.write_ifile): storage, reader, snapshot-wrapped save."""
import os
import h5py
import numpy as np

import bouquet as bq
from bouquet.config import GenerationConfig


def _store(path, ifile):
    from bouquet.utils import store_equilibrium, store_baseline_profiles
    stem = os.path.splitext(path)[0]
    psi, one = np.linspace(0, 1, 9), np.ones(9)
    eq_path, if_path = stem + "_in.eqdsk", stem + "_in.ifile"
    with open(eq_path, "wb") as fh:
        fh.write(b"GEQDSK-BYTES")
    with open(if_path, "wb") as fh:
        fh.write(b"IFILE-BYTES")
    store_baseline_profiles(
        stem, psi, one, one, one, one, one, one, one, one, one, one, one,
        1e6, 1.0, scan_key="900", eqdsk_bytes=b"GEQDSK-BYTES",
        ifile_bytes=b"IFILE-BYTES" if ifile else None)
    store_equilibrium(
        stem, 0, eq_path, psi, one, one, one, one, one, one, one, one,
        1.0, 0.8, scan_key="900", ifile_filepath=if_path if ifile else None)


def test_ifile_stored_and_read(tmp_path):
    p = str(tmp_path / "with_ifile.h5")
    _store(p, True)
    with h5py.File(p, "r") as hf:
        assert "ifile" in hf["scan/900/0"] and "ifile" in hf["scan/900/_baseline"]
    d = bq.BouquetArchive(p)["900"][0]
    assert d.ifile_bytes == b"IFILE-BYTES"
    assert d.eqdsk_bytes == b"GEQDSK-BYTES"
    assert "ifile" not in d.profiles


def test_ifile_absent_by_default(tmp_path):
    p = str(tmp_path / "no_ifile.h5")
    _store(p, False)
    assert bq.BouquetArchive(p)["900"][0].ifile_bytes is None
    assert GenerationConfig.__dataclass_fields__["write_ifile"].default is False


def test_try_save_ifile_restores_equilibrium():
    from bouquet.utils import try_save_ifile
    calls = []

    class FakeGS:
        def copy_eq(self):
            calls.append("copy")
            return "snapshot"

        def save_ifile(self, filename, **kw):
            calls.append(("save", filename, kw))

        def replace_eq(self, source_eq=None):
            calls.append(("restore", source_eq))

    assert try_save_ifile(FakeGS(), "x.ifile", npsi=129, ntheta=257,
                          lcfs_pad=1e-3) == "x.ifile"
    assert calls == ["copy", ("save", "x.ifile", {"npsi": 129, "ntheta": 257, "lcfs_pad": 1e-3}),
                     ("restore", "snapshot")]


def test_try_save_ifile_failure_is_soft(tmp_path):
    from bouquet.utils import try_save_ifile
    path = str(tmp_path / "x.ifile")

    class FailingGS:
        def save_ifile(self, filename, **kw):
            open(filename, "wb").close()
            raise RuntimeError("trace failed")

    assert try_save_ifile(FailingGS(), path) is None
    assert not os.path.exists(path)


# ---------------------------------------------------------------------------
#  #74 review: the pressure frame, the stamp, the restore, the reader, extract
# ---------------------------------------------------------------------------
class _RecordingGS:
    """Records what each save is handed; writes a valid i-file (synthetic)."""

    def __init__(self, restore_fails=False):
        self.calls = []
        self.restore_fails = restore_fails

    def copy_eq(self):
        return "snapshot"

    def replace_eq(self, source_eq=None):
        if self.restore_fails:
            raise RuntimeError("restore exploded")

    def save_eqdsk(self, filename, **kw):
        self.calls.append(("eqdsk", kw))

    def save_ifile(self, filename, **kw):
        self.calls.append(("ifile", kw))
        _write_ifile(filename, kw["npsi"], 4, kw.get("lcfs_pressure", 0.0))


def _write_ifile(path, npsi, ntheta, p_add):
    """A Fortran sequential-unformatted i-file in OFT's layout."""
    import struct

    def rec(b):
        return struct.pack("<i", len(b)) + b + struct.pack("<i", len(b))
    psi = np.linspace(1.0, 0.0, npsi)
    p = 1e4 * psi + p_add
    body = rec(struct.pack("<ii", npsi, ntheta))
    for col in (psi, 2.0 + 0 * psi, p, 1.0 + psi):
        body += rec(np.asarray(col, "<f8").tobytes())
    for _ in range(2):
        body += rec(np.zeros(npsi * ntheta, "<f8").tobytes())
    with open(path, "wb") as fh:
        fh.write(body)


def test_the_ifile_gets_the_gfiles_lcfs_pressure_and_pad(tmp_path):
    """B1: the i-file save is handed the SAME lcfs_pressure / lcfs_pad as the
    g-file of that state; the written pressure carries p_sep."""
    from bouquet.edge_pressure import lcfs_kwargs
    from bouquet.utils import read_ifile, save_state_ifile
    gs = _RecordingGS()
    for p_sep in (0.0, 1234.5):
        gs.calls.clear()
        gs.save_eqdsk("g", lcfs_pad=1e-3, **lcfs_kwargs(p_sep))
        path, rec = save_state_ifile(gs, str(tmp_path / f"{p_sep}.ifile"),
                                     npsi=9, ntheta=4, psi_pad=1e-3,
                                     p_sep=p_sep,
                                     orientation=dict(source_current_sign=-1.0,
                                                      source_b0_sign=None))
        (_, kg), (_, ki) = gs.calls
        assert ki.get("lcfs_pressure") == kg.get("lcfs_pressure")
        assert ki["lcfs_pad"] == kg["lcfs_pad"] == 1e-3
        assert rec["ifile_written"] and rec["ifile_lcfs_pressure"] == p_sep
        assert rec["ifile_source_current_sign"] == -1.0
        assert "ifile_source_b0_sign" not in rec
        assert "positive-Ip" in rec["ifile_frame"]
        with open(path, "rb") as fh:
            d = read_ifile(fh.read())
        assert (d["npsi"], d["ntheta"]) == (9, 4)
        np.testing.assert_allclose(d["p"], 1e4 * d["psi"] + p_sep)


def test_a_failed_save_is_stamped_and_warned(tmp_path):
    import pytest
    from bouquet.utils import save_state_ifile

    class FailingGS:
        def save_ifile(self, filename, **kw):
            open(filename, "wb").close()
            raise RuntimeError("trace failed")

    path = str(tmp_path / "x.ifile")
    with pytest.warns(RuntimeWarning, match="save_ifile failed"):
        out, rec = save_state_ifile(FailingGS(), path, npsi=9, ntheta=4,
                                    psi_pad=1e-3, p_sep=10.0)
    assert out is None and not os.path.exists(path)
    assert rec["ifile_written"] is False and "trace failed" in rec["ifile_error"]


def test_a_failed_restore_is_not_swallowed(tmp_path):
    """B5: the restore error surfaces (the draw is then rejected), and no
    partial file is left behind."""
    import pytest
    from bouquet.utils import SnapshotRestoreError, try_save_ifile
    path = str(tmp_path / "x.ifile")
    with pytest.raises(SnapshotRestoreError, match="restore exploded"):
        try_save_ifile(_RecordingGS(restore_fails=True), path, npsi=9,
                       ntheta=4, lcfs_pad=1e-3)
    assert not os.path.exists(path)


def test_the_stamp_reaches_the_draw_and_baseline_groups(tmp_path):
    from bouquet.utils import stamp_group_attrs
    p = str(tmp_path / "with_ifile.h5")
    _store(p, True)
    stem = os.path.splitext(p)[0]
    rec = dict(ifile_written=True, ifile_error=None, ifile_npsi=129,
               ifile_lcfs_pressure=12.5, ifile_frame="bouquet positive-Ip")
    stamp_group_attrs(stem, "900", 0, rec)
    stamp_group_attrs(stem, "900", None, dict(rec, ifile_written=False,
                                              ifile_error="re-save failed"))
    with h5py.File(p, "r") as hf:
        d, b = hf["scan/900/0"].attrs, hf["scan/900/_baseline"].attrs
        assert bool(d["ifile_written"]) and "ifile_error" not in d
        assert d["ifile_lcfs_pressure"] == 12.5
        assert not bool(b["ifile_written"]) and b["ifile_error"] == "re-save failed"


def test_extract_writes_the_ifile_and_refuses_unknown_formats(tmp_path):
    import pytest
    p = str(tmp_path / "with_ifile.h5")
    _store(p, True)
    d = bq.BouquetArchive(p)["900"][0]
    paths = d.extract(str(tmp_path / "out"), formats=("geqdsk", "ifile"))
    with open(paths["ifile"], "rb") as fh:
        assert fh.read() == b"IFILE-BYTES"
    with pytest.raises(ValueError, match="unknown format"):
        d.extract(str(tmp_path / "out"), formats=("ifle",))
    q = str(tmp_path / "no_ifile.h5")
    _store(q, False)
    assert "ifile" not in bq.BouquetArchive(q)["900"][0].extract(
        str(tmp_path / "out2"), formats=("ifile",))


def test_the_ifile_grid_is_validated_at_config_time():
    import pytest
    for bad in (1, 0, 2.5, True, "129"):
        with pytest.raises(ValueError, match="ifile_npsi"):
            GenerationConfig(ifile_npsi=bad)
        with pytest.raises(ValueError, match="ifile_ntheta"):
            GenerationConfig(ifile_ntheta=bad)
    GenerationConfig(ifile_npsi=2, ifile_ntheta=2)


def test_generate_refuses_write_ifile_on_a_solver_without_save_ifile(tmp_path):
    """B4: one refusal up front, not a WARN per draw and a run with none."""
    import pytest
    import bouquet.TokaMaker_interface as TI
    from test_draw_rejections import _FakeGS, _N, _X
    assert not hasattr(_FakeGS(), "save_ifile")
    jphi = 1.0e6 * (1 - _X ** 2)
    ne = 5e19 * (1 - 0.8 * _X ** 2)
    te = 2e3 * (1 - 0.9 * _X ** 2) + 50.0
    with pytest.raises(RuntimeError, match="needs a TokaMaker with save_ifile"):
        TI.generate_bouquet(
            _FakeGS(), _X, 1, str(tmp_path / "nof"), jphi, ne, te,
            0.9 * ne, te, 0.05 * ne, 0.05 * te, 0.05 * ne, 0.05 * te,
            0.05 * jphi, 0.4, 0.4, 0.25, 1.0e6, 0.8, 1.5 * np.ones(_N),
            diagnostic_plots=False, seed=3, write_ifile=True)
