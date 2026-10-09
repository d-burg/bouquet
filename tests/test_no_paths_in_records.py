"""Public records name no filesystem location.

* The self-consistent bootstrap loop record's ``oft_build`` block carries a
  build identifier (version, short commit id) and NO path: loop records are
  written into every archive and into the public golden fixture.
* The golden fixture's path guard (``tests/golden/make_golden_fixture.py``)
  catches every kind of path a record could leak -- well-known roots
  (``/usr``, ``/Volumes``, ``/opt``, ``/mnt``, ...), any absolute path at a
  token boundary, ``~`` and ``../`` relative paths, Windows drives -- inside
  string ARRAYS as well as scalar strings, and does not fire on units and
  ratios such as ``A/m^2`` or ``1/R``.  The tests below try to defeat it.

Synthetic strings and temporary files only.
"""
import json
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
if _GOLDEN not in sys.path:
    sys.path.insert(0, _GOLDEN)
import make_golden_fixture as mgf  # noqa: E402

from bouquet.jbs_loop import (jbs_settings, oft_build_info,  # noqa: E402
                              run_jbs_loop)
from test_jbs_loop import _GC, _IP, _affine_problem  # noqa: E402


# ---------------------------------------------------------------------------
#  the loop record
# ---------------------------------------------------------------------------
def _strings(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def test_the_oft_build_block_is_an_identifier_not_a_path():
    info = oft_build_info()
    assert set(info) == {"version", "git_hash", "library_sha256",
                         "sources_sha256", "build_id"}
    for v in info.values():
        if v is None:
            continue
        assert "/" not in v and "\\" not in v and "~" not in v, v
        assert not mgf.find_filesystem_paths(v), v
    if info["git_hash"] is not None:
        assert len(info["git_hash"]) <= 12


def test_the_build_token_strips_any_directory_component():
    from bouquet.jbs_loop import _build_token
    assert _build_token("/opt/oft/builds/1.2.3") == "optoftbuilds1.2.3"
    assert _build_token("~/x") == "x"
    assert _build_token("C:\\builds\\26.6") == "Cbuilds26.6"
    assert _build_token("26.6") == "26.6"
    assert _build_token("") is None and _build_token(None) is None


def test_a_loop_record_contains_no_path():
    Jstar, step, ev = _affine_problem(-0.1)
    rec = run_jbs_loop(0.5 * Jstar, step, ev, jbs_settings(_GC()), Ip=_IP,
                       meas0=dict(li=0.0))["record"]
    assert "path" not in rec["oft_build"]
    hits = [h for s in _strings(rec) for h in mgf.find_filesystem_paths(s)]
    assert not hits, hits


# ---------------------------------------------------------------------------
#  the fixture guard: try to defeat it
# ---------------------------------------------------------------------------
_LEAKS = [
    "/Users/someone/oft/python",
    "/home/someone/x",
    "/usr/local/lib/python3/site-packages/OpenFUSIONToolkit",
    "/usr",
    "/Volumes/External/runs/a.h5",
    "/opt/oft/install",
    "/mnt/homes/someone/case",
    "~/runs/case.h5",
    "~someone/runs",
    "../../private/case.h5",
    "C:\\Users\\someone\\oft",
    "D:/builds/oft",
    "/some/unlisted/root/file.h5",
    'mesh_path="/cluster/share/mesh.h5"',
    "built at: /anything/at/all",
]

_NOT_PATHS = [
    "A/m^2", "1/(<R><1/R>)", "~1/R", "psi_N/B0", "r_j/r_I", "(eps/q)^2",
    "evaluate_jBS/1 (Redl 2021 jboot1)", "09/29/2026", "~11.8 MB",
    "||jc_k - js_{k-1}||_w / ||jc_k||_w", "OpenFUSIONToolkit 26.6 git abc",
    "d/dpsi = numpy.gradient(y, psi_N) / psi_range", "<j.B>/B0",
]


@pytest.mark.parametrize("leak", _LEAKS)
def test_every_kind_of_path_is_found(leak):
    assert mgf.find_filesystem_paths(leak), leak
    # ... inside JSON, bytes, lists, tuples, dicts and numpy string arrays
    for wrapped in (json.dumps({"oft_build": {"x": leak}}), leak.encode(),
                    ["ok", leak], ("ok", (leak,)), {"k": {"v": [leak]}},
                    np.array(["F1A", leak]), np.array([b"F1A", leak.encode()]),
                    np.array(["a", leak], dtype=object)):
        assert mgf.find_filesystem_paths(wrapped), (leak, type(wrapped))


@pytest.mark.parametrize("text", _NOT_PATHS)
def test_units_ratios_and_dates_are_not_paths(text):
    assert mgf.find_filesystem_paths(text) == [], text


def _write(path, how, leak):
    with h5py.File(path, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.attrs["ok"] = "clean"
        hf.create_dataset("config_json", data=json.dumps({"a": 1}))
        g.create_dataset("coil_names",
                         data=np.array([b"F1A", b"F2A"], dtype="S8"))
        if how == "attr":
            g.attrs["note"] = leak
        elif how == "attr_array":
            g.attrs["notes"] = np.array([b"fine", leak.encode()])
        elif how == "dataset_scalar":
            g.create_dataset("text", data=leak)
        elif how == "dataset_array":
            g.create_dataset("names", data=np.array(
                [b"F1A", leak.encode()], dtype=f"S{len(leak) + 4}"))
        elif how == "vlen_array":
            dt = h5py.string_dtype()
            g.create_dataset("vl", data=np.array(["a", leak], dtype=object),
                             dtype=dt)
        elif how == "loop_json":
            g.attrs["jbs_loop_json"] = json.dumps(
                {"oft_build": {"path": leak}, "r_j": [1e-3]})
        elif how == "root_attr":
            hf.attrs["where"] = leak


@pytest.mark.parametrize("how", ["attr", "attr_array", "dataset_scalar",
                                 "dataset_array", "vlen_array", "loop_json",
                                 "root_attr"])
@pytest.mark.parametrize("leak", ["/usr/local/oft", "/Volumes/x/y",
                                  "~/oft/python", "../../oft/build",
                                  "/opt/oft", "/mnt/share/oft"])
def test_the_fixture_guard_refuses_a_path_wherever_it_is_stored(tmp_path,
                                                                 how, leak):
    p = str(tmp_path / "f.h5")
    _write(p, how, leak)
    with pytest.raises(SystemExit, match="REFUSING"):
        mgf.assert_no_filesystem_paths(p)


def test_the_fixture_guard_passes_a_clean_file(tmp_path):
    p = str(tmp_path / "f.h5")
    _write(p, "none", "")
    mgf.assert_no_filesystem_paths(p)


def test_the_scrubber_leaves_only_basenames():
    rec = {"oft_build": {"path": "/usr/local/x/OpenFUSIONToolkit"},
           "notes": ["~/a/b/c.h5", "../../d/e.h5", "A/m^2"]}
    out = mgf._scrub_paths(rec)
    assert out["oft_build"]["path"] == "OpenFUSIONToolkit"
    assert out["notes"] == ["c.h5", "e.h5", "A/m^2"]
    assert not mgf.find_filesystem_paths(out)
