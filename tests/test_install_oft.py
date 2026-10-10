"""tools/install_oft.py: upstream by default, never repoints a clone (PR #60
B14).  Local git repositories only; no network, no build."""
import importlib.util
import os
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOL = os.path.join(_HERE, "..", "tools", "install_oft.py")


def _tool():
    spec = importlib.util.spec_from_file_location("install_oft", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env=dict(os.environ, GIT_AUTHOR_NAME="t",
                            GIT_AUTHOR_EMAIL="t@example.invalid",
                            GIT_COMMITTER_NAME="t",
                            GIT_COMMITTER_EMAIL="t@example.invalid"))


def _repo(path):
    os.makedirs(path)
    _git("init", "-q", "-b", "main", path)
    open(os.path.join(path, "f"), "w").write("x")
    _git("add", "f", cwd=path)
    _git("commit", "-q", "-m", "c", cwd=path)
    return path


def test_the_default_is_upstream_main_not_a_fork_branch():
    t = _tool()
    assert t.REF == "main"
    assert "hansec/OpenFUSIONToolkit" in t.REPO
    assert t.checkout.__defaults__ is None          # no hidden repo default
    assert t.install_oft.__defaults__[:2] == (t.REF, t.REPO)


def test_repository_spellings_compare_as_the_same_repository():
    t = _tool()
    assert t._same_repo("https://github.com/x/OpenFUSIONToolkit.git",
                        "https://github.com/x/OpenFUSIONToolkit/")
    assert not t._same_repo("https://github.com/x/OpenFUSIONToolkit.git",
                            "https://github.com/y/OpenFUSIONToolkit.git")


def test_an_existing_clone_of_another_repository_is_refused_untouched(
        tmp_path):
    t = _tool()
    mine = _repo(str(tmp_path / "mine"))
    other = _repo(str(tmp_path / "other"))
    prefix = str(tmp_path / "prefix")
    clone = os.path.join(prefix, "OpenFUSIONToolkit")
    os.makedirs(prefix)
    _git("clone", "-q", mine, clone)
    with pytest.raises(SystemExit, match="refusing to repoint"):
        t.checkout(prefix, other, "main")
    origin = subprocess.run(["git", "-C", clone, "remote", "get-url",
                             "origin"], capture_output=True, text=True)
    assert os.path.realpath(origin.stdout.strip()) == os.path.realpath(mine)


def test_an_existing_clone_of_the_same_repository_is_used(tmp_path):
    t = _tool()
    mine = _repo(str(tmp_path / "mine"))
    prefix = str(tmp_path / "prefix")
    os.makedirs(prefix)
    _git("clone", "-q", "--no-checkout", mine,
         os.path.join(prefix, "OpenFUSIONToolkit"))
    src, sha = t.checkout(prefix, mine, "main")
    assert os.path.isdir(src) and len(sha) == 7
