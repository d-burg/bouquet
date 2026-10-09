"""The subprocess probes must import the tree under test, not an installed one.

This guards the harness itself.  The bug it pins was silent and cost two
investigations (see ``tests/_harness.py``): a probe launched as a script gets
``sys.path[0] = <repo>/tests``, finds no ``bouquet`` there, and falls through to
an editable install that may point at an entirely different checkout.  The
probe then solves against the wrong revision while the parent's assertions
describe code that was never exercised.

These tests are cheap and solve-free: they check the wiring, not the physics.
"""
from __future__ import annotations

import ast
import inspect
import os
import re
import subprocess
import sys

import pytest

import _harness


# ---------------------------------------------------------------------------
#  the helper itself
# ---------------------------------------------------------------------------
def test_repo_root_is_derived_not_hardcoded():
    """``REPO_ROOT`` must follow the checkout, so worktrees/clones all work."""
    src = inspect.getsource(_harness)
    m = re.search(r"^REPO_ROOT = (.+)$", src, re.M)
    assert m, "REPO_ROOT is gone"
    expr = m.group(1)
    assert "__file__" in expr, (
        f"REPO_ROOT is assigned from {expr.strip()!r}; it must be derived from "
        "this file's own location, never hardcoded, or a second checkout "
        "silently validates the first one")
    # and it must actually point at a tree containing the package
    assert os.path.isdir(os.path.join(_harness.REPO_ROOT, "bouquet")), (
        f"REPO_ROOT={_harness.REPO_ROOT} does not contain a bouquet/ package")


def test_subprocess_env_prepends_the_repo_root():
    env = _harness.subprocess_env()
    assert env["PYTHONPATH"].split(os.pathsep)[0] == _harness.REPO_ROOT, (
        "the repo root must come FIRST on PYTHONPATH, ahead of any editable "
        "install, or the installed copy still wins")


def test_subprocess_env_preserves_an_existing_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/caller/path")
    parts = _harness.subprocess_env()["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == _harness.REPO_ROOT
    assert "/some/caller/path" in parts, \
        "an existing PYTHONPATH was discarded rather than appended to"


def test_subprocess_env_merges_extra_vars():
    env = _harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    assert env["OMP_NUM_THREADS"] == "1" and env["MPLBACKEND"] == "Agg"


def test_the_self_check_rejects_a_foreign_bouquet(tmp_path, monkeypatch):
    """FAILURE INJECTION: point REPO_ROOT elsewhere and require a loud failure.

    Equivalent to the real bug (probe imported a bouquet from another
    checkout), and asserts the message carries BOTH paths -- the diagnostic
    whose absence made the original incidents so expensive.
    """
    monkeypatch.setattr(_harness, "REPO_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        _harness.assert_bouquet_is_repo_local()
    msg = str(exc.value)
    import bouquet
    assert os.path.abspath(bouquet.__file__) in msg, \
        "the error does not say where bouquet WAS imported from"
    assert str(tmp_path) in msg, \
        "the error does not say where it was EXPECTED"
    assert "PYTHONPATH" in msg, \
        "the error does not show the PYTHONPATH that produced the mistake"


def test_the_self_check_accepts_the_real_tree():
    """Guard the guard: it must pass in the normal configuration."""
    got = _harness.assert_bouquet_is_repo_local()
    assert got.startswith(_harness.REPO_ROOT)


# ---------------------------------------------------------------------------
#  every launch site and every probe entry point must use it
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _test_modules_launching_subprocesses():
    out = []
    for name in sorted(os.listdir(_HERE)):
        if not name.startswith("test_") or not name.endswith(".py"):
            continue
        src = open(os.path.join(_HERE, name)).read()
        if "sys.executable" in src:
            out.append((name, src))
    return out


def _test_modules_launching_themselves():
    """The launchers that run their OWN module as a script (and so need a ``__main__`` probe entry
    point); a module that only launches ``-c`` code importing no bouquet (test_workqueue) has none."""
    return [(name, src) for name, src in _test_modules_launching_subprocesses()
            if "os.path.abspath(__file__)" in src]


def test_the_parallel_test_modules_are_covered():
    """#76's subprocess-launching tests are under the guard (review B9: the matcher had been narrowed to
    let them through)."""
    names = {n for n, _ in _test_modules_launching_subprocesses()}
    assert {"test_parallel_cases.py", "test_workqueue.py"} <= names
    assert "test_parallel_cases.py" in {n for n, _ in _test_modules_launching_themselves()}


def test_there_is_at_least_one_subprocess_launcher_to_check():
    """If this ever finds nothing, the tests below became vacuous."""
    assert _test_modules_launching_subprocesses(), (
        "no test module launches a subprocess any more -- delete these "
        "wiring tests rather than letting them pass on an empty set")


def test_every_probe_launch_uses_subprocess_env():
    """A launch site added with ``dict(os.environ, ...)`` reintroduces the bug."""
    for name, src in _test_modules_launching_subprocesses():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"):
                continue
            env_kw = next((k for k in node.keywords if k.arg == "env"), None)
            assert env_kw is not None, (
                f"{name}:{node.lineno} launches a probe with no env= at all; "
                "it must use _harness.subprocess_env(...)")
            got = ast.unparse(env_kw.value)
            # env= may be the call itself, or a local bound to it a few lines
            # up; follow one level of binding before complaining.
            if "subprocess_env" not in got and isinstance(env_kw.value, ast.Name):
                # Only bindings in the ENCLOSING function count; the same name
                # is often reused for something unrelated elsewhere.
                scope = min(
                    (f for f in ast.walk(tree)
                     if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and f.lineno <= node.lineno <= (f.end_lineno or f.lineno)),
                    key=lambda f: (node.lineno - f.lineno), default=tree)
                bound = [
                    ast.unparse(a.value) for a in ast.walk(scope)
                    if isinstance(a, ast.Assign)
                    and a.lineno <= node.lineno
                    and any(isinstance(t, ast.Name) and t.id == env_kw.value.id
                            for t in a.targets)
                ]
                assert bound, (
                    f"{name}:{node.lineno} passes env={got}, which is never "
                    "assigned in this module -- cannot verify it carries the "
                    "repo root")
                assert all("subprocess_env" in b for b in bound), (
                    f"{name}:{node.lineno} passes env={got}, bound from "
                    f"{bound}; every binding must come from "
                    "_harness.subprocess_env(...)")
                continue
            assert "subprocess_env" in got, (
                f"{name}:{node.lineno} passes env={got}; use "
                "_harness.subprocess_env(...) so the child imports the tree "
                "under test rather than an editable install elsewhere")


def test_every_probe_entry_point_self_checks():
    """The ``__main__`` block must verify its own import before solving."""
    for name, src in _test_modules_launching_themselves():
        main = re.search(r'if __name__ == "__main__":(.*)\Z', src, re.S)
        assert main, f"{name} launches itself as a script but has no __main__"
        body = main.group(1)
        assert "assert_bouquet_is_repo_local" in body, (
            f"{name}'s probe entry point does not call "
            "_harness.assert_bouquet_is_repo_local(); a wrong-tree import "
            "would surface as a downstream nan instead of a clear error")
        # and it must come before any solver work in that block
        chk = body.index("assert_bouquet_is_repo_local")
        for marker in ("_oft_importable", "_probe(", "_run_r2_probe",
                       "_generate_one_ensemble"):
            if marker in body:
                assert chk < body.index(marker), (
                    f"{name}: the import self-check runs AFTER {marker}; it "
                    "must precede any solver work")


def test_harness_import_precedes_bouquet_import():
    """``ensure_repo_on_syspath`` is useless after ``bouquet`` is bound."""
    for name, src in _test_modules_launching_subprocesses():
        if "ensure_repo_on_syspath" not in src:
            continue
        ensure = src.index("ensure_repo_on_syspath")
        m = re.search(r"^(?:from|import) bouquet", src, re.M)
        if m:
            assert ensure < m.start(), (
                f"{name} imports bouquet before calling "
                "ensure_repo_on_syspath(); by then the module object is "
                "already bound and the path fix cannot take effect")


@pytest.mark.solver
def test_a_real_probe_subprocess_resolves_to_this_tree():
    """End-to-end: launch a child exactly as the probes do and ask it.

    Marked ``solver`` only because it spawns an interpreter; it runs no solve.
    """
    code = (
        "import os, sys; "
        f"sys.path.insert(0, {_HERE!r}); "
        "import _harness; "
        "print(_harness.assert_bouquet_is_repo_local())"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          env=_harness.subprocess_env(OMP_NUM_THREADS="1"),
                          capture_output=True, text=True, cwd=os.sep)
    assert proc.returncode == 0, (
        f"probe-shaped subprocess failed:\n{proc.stderr[-2000:]}")
    assert proc.stdout.strip().startswith(_harness.REPO_ROOT), (
        f"child imported {proc.stdout.strip()!r}, outside "
        f"{_harness.REPO_ROOT}")
