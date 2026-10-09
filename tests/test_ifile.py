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
