"""Shared plumbing for the solver tests that run their probes in a subprocess.

WHY THIS EXISTS
---------------
Several solver tests keep every live-solver call behind a subprocess, because
``OFT_env`` is a per-process singleton and a module that builds a solver in the
pytest process makes ``pytest -m solver`` unrunnable alongside
``test_systematics.py``.  They launch it as::

    subprocess.run([sys.executable, os.path.abspath(__file__), ...])

Python sets ``sys.path[0]`` of a *script* to the SCRIPT'S OWN DIRECTORY -- here
``<repo>/tests`` -- not to the repository root.  There is no ``bouquet`` package
in ``tests/``, so ``import bouquet`` inside the probe falls through to whatever
is installed.  When ``bouquet`` is installed in editable/development mode, that
resolves to the checkout the install points at, which is NOT necessarily the
tree under test.

The failure mode is silent and severe: the probe solves happily against a
DIFFERENT revision of the library, and the assertions in the parent process
then describe code that was never exercised.  It has already bitten twice:

  * a branch adding ``diagnostics['r2_f_ind']`` saw ``f_ind = nan`` in every
    mode, because the probe imported an older tree with no such key -- read at
    the time as a defect in the feature itself; and
  * a four-branch integration run reported two solver failures that did not
    exist in the code being validated, costing a full re-run to disprove.

Both were the same bug, and neither announced itself.  So this module does two
things, and the second matters more than the first:

  1. :func:`subprocess_env` prepends the repo root to ``PYTHONPATH`` for the
     child, so the probe imports the tree it was launched from.
  2. :func:`assert_bouquet_is_repo_local` runs INSIDE the probe and fails
     loudly, naming both paths, if ``bouquet`` still resolved somewhere else.

(1) alone would fix today's symptom while leaving the class of bug silent; (2)
converts any future recurrence -- a launch site added without the helper, a
``PYTHONPATH`` stripped by a runner, an ``import`` that happens before the path
is set -- into an immediate, self-describing failure.
"""
from __future__ import annotations

import os
import sys

#: Repository root, derived from THIS file's location (``<repo>/tests``).
#: Never hardcoded: the tests must work from any checkout, worktree or clone.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def subprocess_env(**extra):
    """Environment for a probe subprocess, with the repo root importable.

    Prepends :data:`REPO_ROOT` to ``PYTHONPATH`` (preserving any existing
    value) so ``import bouquet`` in the child resolves to the tree under test
    rather than to an editable install pointing elsewhere.

    ``extra`` is merged last, exactly like the ``dict(os.environ, ...)`` idiom
    it replaces.
    """
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = (REPO_ROOT + os.pathsep + existing) if existing else REPO_ROOT
    return dict(os.environ, PYTHONPATH=pythonpath, **extra)


def assert_bouquet_is_repo_local():
    """Fail the probe unless ``bouquet`` was imported from this repository.

    Call at the top of every ``__main__`` probe entry point, BEFORE any solver
    work.  Raises :class:`SystemExit` with both paths in the message, so the
    parent's ``proc.stderr`` tail shows exactly what went wrong instead of a
    downstream ``nan`` or a missing diagnostics key.
    """
    import bouquet

    got = os.path.abspath(bouquet.__file__)
    expected_root = os.path.join(REPO_ROOT, "")
    if not got.startswith(expected_root):
        raise SystemExit(
            "\n".join((
                "",
                "=" * 72,
                "HARNESS ERROR: the probe subprocess imported the WRONG bouquet.",
                "",
                f"  imported from : {got}",
                f"  expected under: {REPO_ROOT}",
                f"  PYTHONPATH    : {os.environ.get('PYTHONPATH', '(unset)')}",
                "",
                "The probe would have solved against a different revision of the",
                "library than the one under test, and every assertion in the",
                "parent process would describe code that was never exercised.",
                "",
                "Usual cause: the subprocess was launched without",
                "tests/_harness.subprocess_env(), so sys.path[0] was <repo>/tests",
                "and `import bouquet` fell through to an editable install that",
                "points at another checkout.",
                "=" * 72,
            ))
        )
    return got


def ensure_repo_on_syspath():
    """Belt-and-braces for the child: put the repo root on ``sys.path`` too.

    ``subprocess_env`` is the primary mechanism; this covers a probe invoked by
    hand (``python tests/test_x.py ...``) without the helper, so running one
    directly during debugging still exercises the local tree.  Must run before
    the first ``import bouquet``.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


def golden_provenance(h5path):
    """The ``prov_*`` attrs ``tests/golden/make_golden_fixture.py`` stamps on.

    Returns ``{}`` for a fixture built before provenance was stamped (which is
    itself the useful answer: an undated fixture cannot be told apart from a
    current one, and that is how a stale one survives).
    """
    import h5py
    try:
        with h5py.File(h5path, "r") as hf:
            out = {}
            for k, v in hf.attrs.items():
                if not str(k).startswith("prov"):
                    continue
                out[str(k)] = (v.decode() if isinstance(v, bytes)
                               else (v.item() if hasattr(v, "item") else v))
            for k in ("bouquet_version", "created", "updated"):
                if k in hf.attrs:
                    v = hf.attrs[k]
                    out[k] = (v.decode() if isinstance(v, bytes)
                              else (v.item() if hasattr(v, "item") else v))
            return out
    except Exception as exc:                        # pragma: no cover
        return {"prov_read_error": str(exc)}


def golden_provenance_banner(h5path):
    """A block naming what built the fixture, for a failure message.

    A golden value can go stale because the code moved OR because the SOLVER
    underneath it moved, and the two look identical from the assertion.  The
    second is only diagnosable if the fixture says which solver build it was
    made against, so every test that compares against the fixture prints this
    when it fails.
    """
    prov = golden_provenance(h5path)
    head = f"golden fixture provenance ({os.path.basename(h5path)}):"
    if not prov:
        return (f"{head}\n  NONE STAMPED -- this fixture predates provenance "
                "stamping, so neither its bouquet revision nor its OFT build "
                "can be read off it.  Regenerate with "
                "tests/golden/make_golden_fixture.py to make the next "
                "staleness diagnosable.")
    body = "\n".join(f"  {k} = {prov[k]}" for k in sorted(prov))
    return f"{head}\n{body}"


def legacy_golden_provenance_banner(json_path):
    """:func:`golden_provenance_banner` for the slim LEGACY golden JSON
    (``tests/golden/make_golden_fixture.py --legacy-json``), which carries
    the same provenance block as the h5 fixture's manifest."""
    import json
    head = f"legacy golden provenance ({os.path.basename(json_path)}):"
    try:
        with open(json_path) as fh:
            prov = json.load(fh).get("provenance") or {}
    except Exception as exc:                        # pragma: no cover
        return f"{head}\n  unreadable: {exc}"
    flat = {}

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}_{k}" if path else str(k))
        elif node is not None:
            flat[path] = node
    _walk(prov, "")
    if not flat:
        return f"{head}\n  NONE STAMPED"
    return head + "\n" + "\n".join(f"  {k} = {flat[k]}" for k in sorted(flat))


# ---------------------------------------------------------------------------
#  Build-specific goldens: which OFT build made the fixture, which is installed
# ---------------------------------------------------------------------------
#: The regeneration recipe of the h5 golden + its manifest, verbatim from
#: ``tests/golden/README.md`` ("Build-specific goldens").
REGENERATE_H5_GOLDEN = (
    "OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_DIR "
    "--reconstruction-engine unified --verbose\n"
    "BOUQUET_OFT_BUILD_ID=<installed build> python "
    "tests/golden/make_golden_fixture.py --source "
    "RUN_DIR/D3Dlike_Hmode_golden.h5")

#: The regeneration recipe of the slim LEGACY golden JSON, verbatim from
#: ``tests/golden/README.md`` ("Build-specific goldens").
REGENERATE_LEGACY_GOLDEN = (
    "OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_DIR "
    "--reconstruction-engine legacy --verbose\n"
    "BOUQUET_OFT_BUILD_ID=<installed build> python "
    "tests/golden/make_golden_fixture.py --legacy-json --source "
    "RUN_DIR/D3Dlike_Hmode_golden.h5")


class GoldenBuildMismatchWarning(UserWarning):
    """A golden comparison passed, but against a fixture generated with a
    different OFT build than the one installed."""


def _sha8(sha):
    return sha[:8] if isinstance(sha, str) and sha else None


class BuildMatch:
    """The outcome of :func:`golden_build_check`.

    ``matches`` is the verdict: the installed OFT library's SHA-256 equals
    the one stamped in the fixture (the primary identity: the compiled
    object, measured, not stated).  ``build_id_matches`` is informational
    only (a stated name can be wrong about the build that ran; ``None`` when
    either side states none).  ``stamped`` is False for a fixture that
    carries no ``library_sha256``: it is always a mismatch, worded
    "unstamped", because its build cannot be read off it.
    """

    def __init__(self, fixture, installed):
        fixture = dict(fixture or {})
        installed = dict(installed or {})
        self.fixture_build_id = fixture.get("build_id")
        self.fixture_sha256 = fixture.get("library_sha256") or None
        self.installed_build_id = installed.get("build_id")
        self.installed_sha256 = installed.get("library_sha256") or None
        self.stamped = self.fixture_sha256 is not None
        self.matches = (self.stamped and self.installed_sha256 is not None
                        and self.fixture_sha256 == self.installed_sha256)
        self.build_id_matches = (
            None if not (self.fixture_build_id and self.installed_build_id)
            else self.fixture_build_id == self.installed_build_id)

    @staticmethod
    def _name(build_id, sha, missing):
        if sha is None:
            return f"{build_id or 'no build id'}/{missing}"
        return f"{build_id or 'no build id'}/{_sha8(sha)}"

    @property
    def fixture_name(self):
        return self._name(self.fixture_build_id, self.fixture_sha256,
                          "unstamped")

    @property
    def installed_name(self):
        return self._name(self.installed_build_id, self.installed_sha256,
                          "unavailable")

    def describe(self):
        return (f"fixture generated with {self.fixture_name}, "
                f"installed {self.installed_name}")

    def mismatch_block(self, regenerate):
        """The block a failing comparison's message starts with when the
        builds differ (owner policy, 2026-10-09)."""
        what = ("OFT build mismatch" if self.stamped
                else "OFT build mismatch (the fixture is unstamped: no "
                     "library_sha256 in its provenance)")
        return (f"{what}: {self.describe()}; goldens are build-specific "
                "(edge FF' from per-node <R> moved the lower coils ~1.4 % "
                "between builds); regenerate with:\n"
                + "\n".join("    " + ln for ln in regenerate.splitlines())
                + "\nand review the diff before committing.")

    def __repr__(self):                              # pragma: no cover
        return (f"BuildMatch(matches={self.matches}, stamped={self.stamped}, "
                f"{self.describe()})")


def _fixture_oft_stamp(manifest):
    """``provenance.oft`` of a golden manifest / legacy golden JSON, given
    as the parsed dict or a path to it (``{}`` when there is none)."""
    import json
    if isinstance(manifest, (str, os.PathLike)):
        with open(manifest) as fh:
            manifest = json.load(fh)
    prov = (manifest or {}).get("provenance") or {}
    return prov.get("oft") or {}


def installed_oft_build():
    """``{build_id, library_sha256}`` of the installed OpenFUSIONToolkit.

    ``library_sha256`` from :func:`bouquet.jbs_loop.oft_build_info`;
    ``build_id`` is the operator-stated ``BOUQUET_OFT_BUILD_ID`` when set
    (the name the fixture stamps), else ``oft_build_info``'s own.  A record
    cached before OFT became importable in this process is re-measured
    once rather than read back as "unavailable".
    """
    from bouquet import jbs_loop
    info = jbs_loop.oft_build_info()
    if not info.get("library_sha256"):
        jbs_loop._OFT_BUILD_CACHE.pop("info", None)
        info = jbs_loop.oft_build_info()
    return {"build_id": os.environ.get("BOUQUET_OFT_BUILD_ID")
            or info.get("build_id"),
            "library_sha256": info.get("library_sha256")}


def golden_build_check(manifest, installed=None):
    """Compare the OFT build a golden fixture was generated with (its
    manifest's ``provenance.oft``) with the installed one.

    *manifest*: the parsed manifest / legacy golden JSON, or its path.
    *installed*: ``{build_id, library_sha256}``; default
    :func:`installed_oft_build`.  Returns a :class:`BuildMatch`.
    """
    if installed is None:
        installed = installed_oft_build()
    return BuildMatch(_fixture_oft_stamp(manifest), installed)


class golden_comparison:
    """Run a golden comparison at its existing bars, keyed on the OFT build.

    ::

        with _harness.golden_comparison(manifest, regenerate=...):
            assert value < BAR

    * builds match, comparison passes -> pass;
    * builds match, comparison fails -> the failure, unchanged (a
      regression);
    * builds differ, comparison passes -> pass, with a
      :class:`GoldenBuildMismatchWarning` naming both builds;
    * builds differ, comparison fails -> FAIL, the message prefixed with
      :meth:`BuildMatch.mismatch_block`, then the original failure.

    Nothing is skipped or xfailed and no bar is touched: the outcome of the
    comparison is the outcome of the test.  *fixture_only*: the comparison
    reads stored values only (no solve), which the message then says, since
    the installed build cannot by itself move such a comparison.
    """

    def __init__(self, manifest, *, regenerate, fixture_only=False,
                 installed=None):
        self.match = golden_build_check(manifest, installed=installed)
        self.regenerate = regenerate
        self.fixture_only = fixture_only

    def __enter__(self):
        return self.match

    def __exit__(self, exc_type, exc, tb):
        if self.match.matches:
            return False
        if exc_type is None:
            import warnings
            warnings.warn(GoldenBuildMismatchWarning(
                f"OFT build mismatch: {self.match.describe()}; the golden "
                "comparison passed at its existing bars, but goldens are "
                "build-specific"), stacklevel=2)
            return False
        if not issubclass(exc_type, AssertionError):
            return False
        block = self.match.mismatch_block(self.regenerate)
        if self.fixture_only:
            block += ("\n(This comparison reads the stored fixture only, "
                      "with no solve: the installed build cannot by itself "
                      "explain its failure.)")
        raise AssertionError(f"{block}\n\n{exc}") from exc
