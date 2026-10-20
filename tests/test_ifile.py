"""OFT i-file archiving (GenerationConfig.write_ifile): storage, reader, snapshot-wrapped save."""
import os
import shutil
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


def _write_ifile(path, npsi, ntheta, p_add, extra=()):
    """A Fortran sequential-unformatted i-file in OFT's layout (+ ``extra`` records)."""
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
    for col in extra:
        body += rec(np.asarray(col, "<f8").tobytes())
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


def test_the_extended_records_reach_the_save_and_the_reader(tmp_path):
    from bouquet.utils import read_ifile, save_state_ifile
    gs = _RecordingGS()
    for on in (False, True):
        gs.calls.clear()
        _, rec = save_state_ifile(gs, str(tmp_path / f"{on}.ifile"), npsi=9, ntheta=4,
                                  psi_pad=1e-3, p_sep=0.0, profile_derivs=on, psi_bounds=on)
        kw = gs.calls[0][1]
        assert (kw.get("profile_derivs"), kw.get("psi_bounds")) == ((True, True) if on else (None, None))
        assert rec["ifile_profile_derivs"] is on and rec["ifile_psi_bounds"] is on
        assert rec["ifile_axis_pack"] == 0.0 and "axis_pack" not in kw
    gs.calls.clear()
    _, rec = save_state_ifile(gs, str(tmp_path / "ax.ifile"), npsi=9, ntheta=4, psi_pad=1e-3,
                              p_sep=0.0, axis_pack=1e-5)
    assert gs.calls[0][1]["axis_pack"] == 1e-5 == rec["ifile_axis_pack"]
    path = str(tmp_path / "ext.ifile")
    _write_ifile(path, 9, 4, 0.0, extra=(np.full(9, 2.0), np.full(9, 3.0), (1.0, 0.0)))
    d = read_ifile(open(path, "rb").read())
    assert d["ffp"].tolist() == [2.0] * 9 and d["pp"].tolist() == [3.0] * 9
    assert d["psi_bounds"].tolist() == [1.0, 0.0]
    assert not {"ffp", "pp", "psi_bounds"} & set(read_ifile(open(str(tmp_path / "False.ifile"), "rb").read()))


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



def _ifile_gs(R, fail_from):
    """The rejection tests' mock with a g-file save and an i-file save that
    fails from call *fail_from* on (0: the baseline's)."""
    class _GS(R._FakeGS):
        n_ifile = 0

        def save_eqdsk(self, filename, *a, **k):
            shutil.copy(os.path.join(os.path.dirname(__file__), "data",
                                     "g_synthetic.geqdsk"), filename)

        def save_ifile(self, filename, *a, **k):
            type(self).n_ifile += 1
            if type(self).n_ifile > fail_from:
                raise RuntimeError("trace failed")
            with open(filename, "w") as fh:
                fh.write("i")
    return _GS


def test_a_draw_whose_ifile_save_fails_is_rejected(tmp_path, monkeypatch):
    """write_ifile hands the i-file on: a draw without one is not archived."""
    import test_draw_rejections as R
    monkeypatch.setattr(R, "_FakeGS", _ifile_gs(R, 1))
    out, rej, header = R._run(tmp_path, monkeypatch, ["ok"], write_ifile=True)
    assert [r["reason"] for r in rej] == ["ifile_save_failed"]
    assert "trace failed" in rej[0]["message"] and out == []
    assert R._draw_groups(header) == []


def test_a_baseline_without_an_ifile_refuses_the_run(tmp_path, monkeypatch):
    import pytest
    import test_draw_rejections as R
    monkeypatch.setattr(R, "_FakeGS", _ifile_gs(R, 0))
    with pytest.raises(RuntimeError, match="baseline has no i-file: .*trace failed"):
        R._run(tmp_path, monkeypatch, ["ok"], write_ifile=True)


def test_compress_file_bytes_reaches_every_blob(tmp_path, monkeypatch):
    import test_draw_rejections as R
    monkeypatch.setattr(R, "_FakeGS", _ifile_gs(R, 9))
    out, rej, header = R._run(tmp_path, monkeypatch, ["ok"], write_ifile=True,
                              compress_file_bytes=True)
    assert rej == [] and len(out) == 1
    blobs = []
    with h5py.File(header + ".h5", "r") as hf:
        hf.visititems(lambda n, o: blobs.append((n.split("/")[-2], n.split("/")[-1],
                                                 o.compression, o.dtype))
                      if n.split("/")[-1] in ("eqdsk", "ifile") else None)
    assert sorted(b[:2] for b in blobs) == [("0", "eqdsk"), ("0", "ifile"),
                                            ("_baseline", "eqdsk"), ("_baseline", "ifile")]
    assert all(c == "gzip" and t == np.uint8 for *_, c, t in blobs)


def test_the_gfile_grid_reaches_every_save(tmp_path, monkeypatch):
    import pytest
    import test_draw_rejections as R
    grids = []

    class _GS(_ifile_gs(R, 9)):
        def save_eqdsk(self, filename, *a, **k):
            grids.append((k.get("nr"), k.get("nz")))
            super().save_eqdsk(filename, *a, **k)
    monkeypatch.setattr(R, "_FakeGS", _GS)
    out, rej, _ = R._run(tmp_path, monkeypatch, ["ok"], write_ifile=True,
                         eqdsk_nr=129, eqdsk_nz=193)
    assert rej == [] and len(out) == 1
    assert len(grids) == 2 and set(grids) == {(129, 193)}      # baseline + draw
    with pytest.raises(ValueError, match="eqdsk_nr"):
        GenerationConfig(eqdsk_nr=1)
    assert GenerationConfig().eqdsk_nr == GenerationConfig().eqdsk_nz == 257


class _State(dict):
    """A copy_eq() snapshot that can save itself (TokaMaker_equilibrium.save_TokaMaker)."""
    fail = False

    def save_TokaMaker(self, path):
        with open(path, "wb") as fh:
            fh.write(b"STATE")
        if self.fail:
            raise RuntimeError("h5 write failed")


def test_the_tokamaker_state_is_saved_beside_the_ifile(tmp_path):
    from bouquet.utils import save_state_ifile

    class _GS(_RecordingGS):
        def copy_eq(self):
            return _State()
    kw = dict(npsi=9, ntheta=4, psi_pad=1e-3, p_sep=0.0)
    path, rec = save_state_ifile(_GS(), str(tmp_path / "a.ifile"),
                                 state_path=str(tmp_path / "a.tmk.h5"), **kw)
    assert path and rec["tokamaker_state_written"]
    assert open(tmp_path / "a.tmk.h5", "rb").read() == b"STATE"
    _State.fail = True
    try:
        path, rec = save_state_ifile(_GS(), str(tmp_path / "b.ifile"),
                                     state_path=str(tmp_path / "b.tmk.h5"), **kw)
    finally:
        _State.fail = False
    assert path is None and not rec["ifile_written"] and not rec["tokamaker_state_written"]
    assert "save_TokaMaker failed" in rec["ifile_error"]
    assert not (tmp_path / "b.ifile").exists() and not (tmp_path / "b.tmk.h5").exists()


def test_the_archived_state_rebuilds_the_files_and_they_can_be_dropped(tmp_path, monkeypatch):
    import test_draw_rejections as R
    from bouquet.filtering import drop_file_blobs

    class _GS(_ifile_gs(R, 9)):
        def copy_eq(self):
            return _State(super().copy_eq())
    monkeypatch.setattr(R, "_FakeGS", _GS)
    out, rej, header = R._run(tmp_path, monkeypatch, ["ok"], write_ifile=True,
                              store_tokamaker_state=True)
    assert rej == [] and len(out) == 1 and not list(tmp_path.rglob("*.tmk.h5"))
    ar = bq.BouquetArchive(header + ".h5")
    d = ar[ar.scan_keys[0]][0]
    assert d.tokamaker_state_bytes == b"STATE" and d.attrs["tokamaker_state_written"]
    states = {}
    with h5py.File(header + ".h5", "r") as hf:
        hf.visititems(lambda n, o: states.update({n.split("/")[-2]: bytes(o[()])})
                      if n.split("/")[-1] == "tokamaker_state" else None)
    assert states == {"0": b"STATE", "_baseline": b"STATE"}

    calls = []

    class _Loader:
        def replace_eq(self, source_file=None):
            calls.append(open(source_file, "rb").read())

        def save_ifile(self, filename, **kw):
            calls.append(kw)

        def save_eqdsk(self, filename, **kw):
            calls.append(kw)
            with open(filename, "wb") as fh:
                fh.write(b"REBUILT")
    d.rebuild_ifile(str(tmp_path / "re.ifile"), _Loader(), npsi=513)
    assert calls[0] == b"STATE"
    assert (calls[1]["npsi"], calls[1]["ntheta"]) == (513, 257)
    assert calls[1]["lcfs_pad"] == d.attrs["ifile_lcfs_pad"]

    eq_before = d.eqdsk_bytes
    assert drop_file_blobs(header + ".h5") == 2                    # baseline + draw
    d = bq.BouquetArchive(header + ".h5")[ar.scan_keys[0]][0]
    assert d.ifile_bytes is None and d.eqdsk_bytes == eq_before
    assert d.tokamaker_state_bytes == b"STATE"
    # dropped g-files: eqdsk_bytes rebuilds one on the run's mesh (the process's solver)
    import bouquet.utils as U
    from bouquet.schema import EQDSK_DS
    from types import SimpleNamespace
    from bouquet.config import GenerationConfig
    monkeypatch.setattr(U, "tokamaker_on_mesh", lambda mesh, order: _Loader())
    monkeypatch.setattr(U, "load_config", lambda path, scan_key=None: SimpleNamespace(   # this toy run stores none
        solver=SimpleNamespace(mesh_path="mesh.h5", order=3), generation=GenerationConfig()))
    assert drop_file_blobs(header + ".h5", names=(EQDSK_DS,)) == 2
    d = bq.BouquetArchive(header + ".h5")[ar.scan_keys[0]][0]
    assert d.eqdsk_bytes == b"REBUILT" and calls[-2] == b"STATE"
    assert (calls[-1]["nr"], calls[-1]["lcfs_pad"]) == (257, d.attrs["ifile_lcfs_pad"])


def test_store_eqdsk_off_archives_the_state_and_no_gfile(tmp_path, monkeypatch):
    import pytest
    import test_draw_rejections as R

    class _GS(_ifile_gs(R, 9)):
        def copy_eq(self):
            return _State(super().copy_eq())
    monkeypatch.setattr(R, "_FakeGS", _GS)
    out, rej, header = R._run(tmp_path, monkeypatch, ["ok"], write_ifile=True,
                              store_tokamaker_state=True, store_eqdsk=False)
    assert rej == [] and len(out) == 1
    blobs = {}
    with h5py.File(header + ".h5", "r") as hf:
        hf.visititems(lambda n, o: blobs.setdefault(n.split("/")[-2], set()).add(n.split("/")[-1])
                      if isinstance(o, h5py.Dataset) else None)
    for g in ("0", "_baseline"):
        assert "tokamaker_state" in blobs[g] and "eqdsk" not in blobs[g]
    for kw in (dict(write_ifile=True, store_tokamaker_state=False),
               dict(write_ifile=False, store_tokamaker_state=True)):
        with pytest.raises(ValueError, match="store_tokamaker_state"):
            GenerationConfig(store_eqdsk=False, **kw)

