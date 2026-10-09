"""``jbs_loop.oft_build_info`` identifies the OFT BUILD, not only its version.

An install tree has no ``.git``, so ``git_hash`` is ``None`` in production
and a fork build and an upstream build of the same version recorded the
same block.  The block now carries ``library_sha256`` (the compiled
``liboftpy`` the package loads) and ``sources_sha256`` (the package's Python
sources) -- the digests ``tests/golden/make_golden_fixture.py`` stamps --
and still no path.  Exercised on a FAKE package and library in a temporary
directory (no OFT needed).
"""
import hashlib
import os
import sys
import types

import pytest

from bouquet import jbs_loop as J

_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"


def _fake_oft(tmp_path, monkeypatch, *, lib_where="pkg", lib_bytes=b"fake"):
    root = tmp_path / "install" / "python"
    pkg = root / "OpenFUSIONToolkit"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '9.9'\n")
    (pkg / "_core.py").write_text("x = 1\n")
    if lib_where == "pkg":
        lib = pkg / ("liboftpy" + _SUFFIX)
    elif lib_where == "bin":
        (tmp_path / "install" / "bin").mkdir()
        lib = tmp_path / "install" / "bin" / ("liboftpy" + _SUFFIX)
    else:
        lib = None
    if lib is not None:
        lib.write_bytes(lib_bytes)
    mod = types.ModuleType("OpenFUSIONToolkit")
    mod.__file__ = str(pkg / "__init__.py")
    mod.__version__ = "9.9"
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", mod)
    monkeypatch.delitem(sys.modules, "OpenFUSIONToolkit._interface",
                        raising=False)
    monkeypatch.setattr(J, "_OFT_BUILD_CACHE", {})
    return pkg, lib


@pytest.mark.parametrize("where", ["pkg", "bin"])
def test_the_library_digest_is_the_loaded_files(tmp_path, monkeypatch, where):
    pkg, lib = _fake_oft(tmp_path, monkeypatch, lib_where=where,
                         lib_bytes=b"\x7fELF fake oft build A")
    info = J.oft_build_info()
    assert info["version"] == "9.9"
    assert info["git_hash"] is None            # no checkout: as in production
    assert info["library_sha256"] == hashlib.sha256(
        b"\x7fELF fake oft build A").hexdigest()
    assert info["sources_sha256"] == J._sha256_tree(str(pkg))
    assert f"lib {info['library_sha256'][:12]}" in info["build_id"]
    for v in info.values():
        assert v is None or ("/" not in v and "\\" not in v
                             and str(tmp_path) not in v)


def test_two_builds_of_one_version_are_distinguishable(tmp_path, monkeypatch):
    _fake_oft(tmp_path / "a", monkeypatch, lib_bytes=b"upstream")
    a = J.oft_build_info()
    _fake_oft(tmp_path / "b", monkeypatch, lib_bytes=b"fork")
    b = J.oft_build_info()
    assert a["version"] == b["version"] and a["git_hash"] == b["git_hash"]
    assert a["library_sha256"] != b["library_sha256"]
    assert a["build_id"] != b["build_id"]
    # the same sources in another directory hash the same (relative paths)
    assert a["sources_sha256"] == b["sources_sha256"]


def test_the_loaded_library_wins_over_the_search(tmp_path, monkeypatch):
    _fake_oft(tmp_path, monkeypatch, lib_where="pkg", lib_bytes=b"on disk")
    loaded = tmp_path / "elsewhere.so"
    loaded.write_bytes(b"actually loaded")
    iface = types.ModuleType("OpenFUSIONToolkit._interface")
    iface.oftpy_lib = types.SimpleNamespace(_name=str(loaded))
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit._interface", iface)
    info = J.oft_build_info()
    assert info["library_sha256"] == hashlib.sha256(
        b"actually loaded").hexdigest()


def test_no_library_records_none_and_never_raises(tmp_path, monkeypatch):
    _fake_oft(tmp_path, monkeypatch, lib_where=None)
    info = J.oft_build_info()
    assert info["library_sha256"] is None
    assert info["sources_sha256"] is not None
    assert "lib " not in info["build_id"]
