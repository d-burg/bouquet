#!/usr/bin/env python3
"""Build OpenFUSIONToolkit (OFT) at a branch or commit for bouquet.

    python tools/install_oft.py --prefix ~/oft [--ref main] [--repo URL|PATH]
                                [--libs DIR] [--jobs N]

Defaults to upstream OpenFUSIONToolkit ``main``; any other repository or
branch (e.g. a fork carrying the internal Fortran bootstrap solve) only by an
explicit ``--repo`` / ``--ref``.  An existing clone under --prefix is used
only if its ``origin`` already is ``--repo``: the script never repoints a
clone's remote (``git remote set-url``) -- give another --prefix instead.

Under --prefix: OpenFUSIONToolkit/ (clone), src_<sha>/ (worktree),
build_<sha>/ and install_<sha>/.  The external libraries (OFT's
build_libs.py: BLAS/LAPACK, HDF5, METIS, UMFPACK, ARPACK) are the long step:
--libs reuses any build_libs.py output directory (the one holding
config_cmake.sh); without it they are built once into <prefix>/libs, with
CMake too if the cmake on PATH is older than 3.27 (slow).  An
install of the same commit is reused.  Compilers come from CC/CXX/FC or PATH,
as in OFT's install instructions.  Run it with the Python that runs bouquet
(it checks the import); it prints the exports bouquet reads.  Standard
library only; it does not import bouquet.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys

#: Upstream OpenFUSIONToolkit (the repository bouquet's README names) and its
#: main branch.  A fork / feature branch is used only when named explicitly.
REPO = "https://github.com/hansec/OpenFUSIONToolkit.git"
REF = "main"
LIBS_ARGS = "--build_umfpack=1 --build_arpack=1"
DONE = ".install_oft_done"


def _run(cmd, cwd=None, log=None):
    print("+", " ".join(map(shlex.quote, cmd)) + (f"  > {log}" if log else ""), flush=True)
    if log:
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=cwd)
    if r.returncode:
        sys.exit(f"failed ({r.returncode})" + (f"; see {log}" if log else ""))


def _git(clone, *args):
    r = subprocess.run(["git", "-C", clone, *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _same_repo(a, b):
    """True when two repository spellings name the same repository (a URL up
    to a trailing ``.git`` / slash and case of the host, or the same local
    directory)."""
    def norm(u):
        u = (u or "").strip()
        if os.path.isdir(os.path.expanduser(u)):
            return os.path.realpath(os.path.expanduser(u))
        u = u.rstrip("/")
        if u.endswith(".git"):
            u = u[:-4]
        return u.lower()
    return norm(a) == norm(b)


def checkout(prefix, repo, ref):
    """A detached worktree of ``ref``; returns (source dir, short sha).

    A clone already at ``<prefix>/OpenFUSIONToolkit`` is used only when its
    ``origin`` is ``repo``; otherwise this refuses (it never runs ``git
    remote set-url`` on a clone it may not own)."""
    clone = os.path.join(prefix, "OpenFUSIONToolkit")
    if not os.path.isdir(clone):
        _run(["git", "clone", "--no-checkout", repo, clone])
    else:
        origin = _git(clone, "remote", "get-url", "origin")
        if origin is None or not _same_repo(origin, repo):
            sys.exit(f"{clone} already exists with origin {origin!r}, not "
                     f"{repo!r}; refusing to repoint it.  Use another "
                     "--prefix, or --repo with that clone's own origin.")
    _run(["git", "-C", clone, "fetch", "--quiet", "origin"])
    sha = next((s for r in (f"origin/{ref}", ref)
                if (s := _git(clone, "rev-parse", "--verify", "--short=7", r + "^{commit}"))), None)
    if sha is None:
        sys.exit(f"{ref!r} is neither a branch nor a commit of {repo}")
    src = os.path.join(prefix, f"src_{sha}")
    if not os.path.isdir(src):
        _run(["git", "-C", clone, "worktree", "add", "--detach", src, sha])
    return src, sha


def _cmake_ok():
    """A cmake >= 3.27 (what OFT needs) is on PATH."""
    if not shutil.which("cmake"):
        return False
    m = re.search(r"(\d+)\.(\d+)", subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout)
    return bool(m) and tuple(map(int, m.groups())) >= (3, 27)


def libs_config(prefix, src, libs, libs_args, jobs):
    """config_cmake.sh of ``libs``, else of <prefix>/libs (built if missing)."""
    if libs:
        cfg = os.path.join(os.path.abspath(libs), "config_cmake.sh")
        if not os.path.isfile(cfg):
            sys.exit(f"{libs} has no config_cmake.sh: not a build_libs.py output directory")
        return cfg
    libs = os.path.join(prefix, "libs")
    cfg = os.path.join(libs, "config_cmake.sh")
    if not os.path.isfile(cfg):
        if "--build_cmake" not in libs_args and not _cmake_ok():
            libs_args += " --build_cmake=1"
        os.makedirs(libs, exist_ok=True)
        _run([sys.executable, "-u", os.path.join(src, "src", "utilities", "build_libs.py"),
              f"--nthread={jobs}", *shlex.split(libs_args)],
             cwd=libs, log=os.path.join(libs, "build_libs.log"))
    return cfg


def configure_script(cfg, src, build, install):
    """config_cmake.sh pointed at this source tree and these build/install dirs."""
    text = open(cfg).read()
    text = re.sub(r"^BUILD_DIR=.*$", lambda _: f"BUILD_DIR={shlex.quote(build)}", text, flags=re.M)
    text = re.sub(r"^INSTALL_DIR=.*$", lambda _: f"INSTALL_DIR={shlex.quote(install)}", text, flags=re.M)
    # the cmake source directory is the script's last argument
    text, n = re.subn(r"(\\\n\s*)\S+\s*\Z", lambda m: m.group(1) + shlex.quote(os.path.join(src, "src")) + "\n", text)
    if n != 1:
        sys.exit(f"cannot find the source-directory argument in {cfg}")
    return text


def check(install):
    """Import OFT from ``install`` and refuse a build bouquet's Redl
    evaluator cannot run on: its geometric epsilon needs ``get_fsa`` (OFT
    v26.6+) or the fork's ``sauter_fc(return_eps=True)``.  Says which."""
    code = ("import inspect; from OpenFUSIONToolkit.TokaMaker import TokaMaker; "
            "print(hasattr(TokaMaker, 'get_fsa'), "
            "'return_eps' in inspect.signature(TokaMaker.sauter_fc).parameters)")
    env = dict(os.environ, OFT_INSTALL_DIR=install, PYTHONPATH=os.path.join(install, "python"))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"OFT import failed from {install}:\n{r.stderr[-2000:]}")
    has_fsa, has_eps = (w == "True" for w in r.stdout.split()[:2])
    if not (has_fsa or has_eps):
        sys.exit("this OFT build has neither TokaMaker.get_fsa (v26.6+) nor "
                 "sauter_fc(return_eps=True): bouquet's evaluate_jBS cannot "
                 "compute its default (geometric) epsilon on it")
    print("geometric epsilon route: "
          + ("sauter_fc(return_eps=True)" if has_eps else "get_fsa"))


def install_oft(prefix, ref=REF, repo=REPO, libs=None, jobs=2, libs_args=LIBS_ARGS, rebuild=False):
    """Build OFT ``ref`` under ``prefix`` and return its install directory."""
    prefix = os.path.abspath(os.path.expanduser(prefix))
    if os.path.isdir(os.path.expanduser(repo)):
        repo = os.path.abspath(os.path.expanduser(repo))
    os.makedirs(prefix, exist_ok=True)
    src, sha = checkout(prefix, repo, ref)
    build, install = (os.path.join(prefix, f"{d}_{sha}") for d in ("build", "install"))
    done = os.path.join(install, DONE)
    stamp = f"{sha}\nlibs={os.path.abspath(libs) if libs else '<prefix>/libs'} {libs_args}\n"
    # a marker of the older format (the sha alone) says nothing about the
    # libraries: kept, not rebuilt (the build is hours)
    if (os.path.isfile(done) and not rebuild
            and open(done).read() not in (stamp, sha + "\n")):
        print(f"{install} was built with other libraries; rebuilding")
        rebuild = True
    if rebuild or not os.path.isfile(done):
        text = configure_script(libs_config(prefix, src, libs, libs_args, jobs), src, build, install)
        script = os.path.join(prefix, f"config_cmake_{sha}.sh")
        open(script, "w").write(text)
        _run(["bash", script], cwd=prefix, log=os.path.join(prefix, f"cmake_{sha}.log"))
        _run(["make", f"-j{jobs}", "install"], cwd=build, log=os.path.join(prefix, f"make_{sha}.log"))
        open(done, "w").write(stamp)
    else:
        print(f"reusing {install} ({sha})")
    check(install)
    return install


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prefix", required=True, help="where the clone, builds and installs go")
    p.add_argument("--ref", default=REF, help=f"branch or commit (default {REF}, upstream)")
    p.add_argument("--repo", default=REPO,
                   help="OFT git URL or local clone (default: upstream, %(default)s); "
                        "a fork only when named here")
    p.add_argument("--libs", help="existing build_libs.py output directory to reuse")
    p.add_argument("--libs-args", default=LIBS_ARGS, help="build_libs.py options (default %(default)r)")
    p.add_argument("--jobs", type=int, default=2, help="make / build_libs threads (default 2)")
    p.add_argument("--rebuild", action="store_true", help="rebuild even if this commit is installed")
    a = p.parse_args(argv)
    install = install_oft(a.prefix, a.ref, a.repo, a.libs, a.jobs, a.libs_args, a.rebuild)
    print(f"export OFT_INSTALL_DIR={shlex.quote(install)}\n"
          f"export OFT_PYTHONPATH={shlex.quote(os.path.join(install, 'python'))}")


if __name__ == "__main__":
    main()
