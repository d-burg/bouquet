"""Build the git-tracked golden test fixture + manifest from a full bouquet run.

The backend regression tests (``tests/test_golden_bouquet.py``) run against a
**slimmed** copy of a real bouquet ``.h5`` plus a JSON manifest of expected
values.  This script produces both, so updating the golden set on purpose is a
single, reviewable command.

What it does
------------
1. Reads a full bouquet ``.h5`` (the 30 MB shareable example artifact).
2. Writes ``D3Dlike_Hmode_golden_slim.h5`` next to this script:
   - **keeps the ``*.eqdsk`` geqdsks, gzip-compressed** (~3x; ~0.37 MB each)
     so the geqdsk read/parse/handling path -- a deliberately coarse-at-the-
     separatrix format -- is exercised by real files.  The geqdsks are stored
     as gzipped ``uint8`` under their original ``.eqdsk`` dataset names, so
     every reader (``bytes(grp[k][()])``) is unaffected.
   - drops the ``*.pfile`` byte blobs (not needed for equilibrium-handling
     tests; full p-files stay in the example artifact),
   - extracts the real ``Ip`` from each eqdsk into an ``Ip`` attr,
   - keeps every attr, ``coil_currents``, ``x_points``, both LCFS references,
     and the profile arrays the assertions need.
3. Writes ``golden_manifest.json`` of expected per-draw + baseline values with
   tolerances.

The ``--eqdsk`` flag controls geqdsk retention:
``all`` (default, ~11 MB), ``subset`` (baseline + a few representative draws,
~5 MB), or ``none`` (smallest, ~4 MB, no geqdsk-handling coverage).

Updating the golden set
-----------------------
Regenerate the full run with ``regenerate_golden_run.py`` (the recipe: the
fixture's own stored config, class API, INPUT-current archival -- NOT a plain
notebook ``generate()``, which archives the achieved current), then::

    python tests/golden/make_golden_fixture.py \
        --source /path/to/D3Dlike_Hmode_golden.h5

Review the git diff of ``golden_manifest.json`` (and the slim ``.h5``) before
committing -- the manifest diff shows exactly which physics values moved.

Note: eventually the default equilibrium interchange should migrate to
IMAS/OMAS; when it does, this fixture can store those instead of geqdsks.
"""
import argparse
import hashlib
import json
import os
import platform

import numpy as np
import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SOURCE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "bouquet", "examples", "D3D-like",
    "D3Dlike_Hmode_golden.h5"))
SLIM_NAME = "D3Dlike_Hmode_golden_slim.h5"
MANIFEST_NAME = "golden_manifest.json"
RNG_MANIFEST_NAME = "rng_stream_manifest.json"
#: the slim LEGACY golden (``--legacy-json``): the same recipe run with
#: ``reconstruction_engine="legacy"``, kept as a small JSON instead of a
#: second h5 blob, so the legacy path keeps a numeric regression record
LEGACY_JSON_NAME = "D3Dlike_Hmode_legacy_golden.json"
#: draws whose profiles the legacy JSON carries: the first N in-spec draws
#: (what ``tests/test_systematics.py`` replays, ``_N_REPLAY``)
LEGACY_REPLAY_DRAWS = 2
#: the reconstruction's LCFS reference is kept at every LEGACY_LCFS_STRIDE-th
#: point (a uniform subsample of the traced contour, ~4 mm apart), rounded
#: to 1 um -- the full trace is ~9200 points, ~0.4 MB of JSON on its own.
#: Used only as the REFERENCE side of a nearest-point boundary RMS (the
#: other side is a full trace), whose value it changes by <~0.01 mm
#: (measured on the 2026-10-05 fixture: 2.1724 vs 2.1661 mm, 1.3262 vs
#: 1.3240 mm); both values are stored per replay draw
LEGACY_LCFS_STRIDE = 8
LEGACY_LCFS_DECIMALS = 6

# Seed for the pinned draw stream.  Changing it re-pins every value in
# rng_stream_manifest.json, so don't, unless that is the intent.
RNG_STREAM_SEED = 20260804

# p-file byte blobs are always dropped (not needed for equilibrium-handling
# tests).  geqdsk retention is controlled by the --eqdsk flag.
_EQDSK_GZIP_LEVEL = 9

# Tolerances the regression test uses when comparing recomputed values to
# the manifest.  Reads/derivations are deterministic, so these are tight;
# small absolute floors guard pure float round-off.
TOLERANCES = {
    "l_i_atol": 1e-4,        # l_i(1)/l_i(3)
    "Ip_rtol": 1e-6,         # plasma current (relative)
    "drift_atol": 1e-4,      # coil drift percentages
    "bnd_atol_mm": 1e-3,     # boundary RMS/max deviation [mm]
    "coil_atol_A": 1e-3,     # per-coil current [A]
    "xpoint_atol_m": 1e-6,   # X-point R,Z [m]
}


def _is_eqdsk_name(k):
    """Both archive generations: legacy stores '<header>_count=N.eqdsk';
    the class-API schema stores a dataset simply named 'eqdsk'."""
    return k == "eqdsk" or k.endswith(".eqdsk")


def _is_pfile_name(k):
    return k == "pfile" or k.endswith(".pfile")


def _bp_to_path(bkey):
    return f"scan/{bkey}" if bkey is not None else None


def _scan_keys(hf):
    """Return list of (bkey, group_path_prefix) for each scan value."""
    if "scan" in hf:
        return [(k, f"scan/{k}") for k in sorted(hf["scan"].keys())]
    return [(None, "")]


def _ip_from_eqdsk_bytes(raw):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    from bouquet.utils import read_eqdsk_from_bytes
    from bouquet.io import read_geqdsk
    eq = read_eqdsk_from_bytes(raw, read_geqdsk)
    return float(eq.Ip)


def _boundary_devs(bl_boundary, grp):
    from scipy.spatial import cKDTree
    if bl_boundary is None or "perturbed_lcfs_ref" not in grp:
        return (np.nan, np.nan)
    pert = np.asarray(grp["perturbed_lcfs_ref"][()], dtype=float)
    tree = cKDTree(pert)
    devs, _ = tree.query(bl_boundary)
    return (float(np.sqrt(np.mean(devs ** 2)) * 1e3),
            float(np.max(devs) * 1e3))


def _select_subset(draws):
    """Pick a small, diverse set of draw indices for ``--eqdsk subset``.

    Covers the parser-relevant variation: first in-spec, first out-of-spec,
    and the min/max boundary-deviation draws.
    """
    idxs = sorted(int(k) for k in draws)
    chosen = set()
    inspec = [i for i in idxs if draws[str(i)].get("in_spec")]
    outspec = [i for i in idxs if not draws[str(i)].get("in_spec")]
    if inspec:
        chosen.add(inspec[0])
    if outspec:
        chosen.add(outspec[0])
    rms = {i: draws[str(i)].get("bnd_rms_mm", float("nan")) for i in idxs}
    fin = [i for i in idxs if rms[i] == rms[i]]  # drop NaN
    if fin:
        chosen.add(min(fin, key=lambda i: rms[i]))
        chosen.add(max(fin, key=lambda i: rms[i]))
    return chosen


def _digest(arr):
    """SHA-256 over the raw float64 bytes -- a bitwise fingerprint."""
    return hashlib.sha256(
        np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


#: Environment variables an operator sets to name the OFT build being used.
#: Nothing here is auto-scraped from a filesystem path: the fixture must not
#: carry hostnames, user names or directory layouts, and a path would tell a
#: later reader nothing a content digest does not tell them better.
_OFT_ENV = {"commit": "BOUQUET_OFT_COMMIT",
            "branch": "BOUQUET_OFT_BRANCH",
            "build_id": "BOUQUET_OFT_BUILD_ID"}


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_tree(root, suffixes=(".py",)):
    """Digest of a package's sources: sorted RELATIVE paths + contents.

    Relative, so two installs of the same revision in different directories
    (or on different machines) hash identically and no absolute path is
    embedded in the result.
    """
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(suffixes):
                continue
            full = os.path.join(dirpath, name)
            h.update(os.path.relpath(full, root).encode())
            h.update(_sha256_file(full).encode())
    return h.hexdigest()


def oft_provenance():
    """Which OpenFUSIONToolkit produced this fixture.

    The fixture's physics values are a function of the OFT build as much as of
    bouquet: an edge-localised change to the bootstrap module moves l_i(1) by
    several percent while leaving l_i(3) and the boundary untouched, which is
    exactly the failure signature a fixture with no OFT provenance cannot be
    diagnosed from.  So record it.

    Three kinds of field, deliberately:

    * **stated** -- ``commit`` / ``branch`` / ``build_id``, from the
      environment (:data:`_OFT_ENV`).  OFT embeds its revision in the compiled
      library but does not expose it to Python, so this is operator-supplied
      and may be absent.
    * **measured** -- ``sources_sha256`` (the Python package) and
      ``library_sha256`` (the compiled object).  Always available, immune to a
      mis-stated commit, and the only fields that identify the *build* rather
      than the source revision.
    * **behavioural** -- the two feature probes that actually separate the
      OFT lines bouquet has met.  A reader who has neither commit nor digest
      to compare against can still tell which line a fixture came from.
    """
    out = {k: os.environ.get(v) for k, v in _OFT_ENV.items()}
    try:
        import OpenFUSIONToolkit as _oft
    except Exception as exc:                    # pragma: no cover - no OFT
        out["available"] = False
        out["import_error"] = str(exc)
        return out
    out["available"] = True
    out["version"] = getattr(_oft, "__version__", None)
    pkg = os.path.dirname(os.path.abspath(_oft.__file__))
    try:
        out["sources_sha256"] = _sha256_tree(pkg)
    except Exception:
        out["sources_sha256"] = None
    lib = None
    for cand in ("liboftpy.so", "liboftpy.dylib"):
        for base in (os.path.join(pkg, "..", "..", "bin"),
                     os.path.join(pkg, "..", "..", "lib"), pkg):
            p = os.path.abspath(os.path.join(base, cand))
            if os.path.isfile(p):
                lib = p
                break
        if lib:
            break
    out["library_sha256"] = _sha256_file(lib) if lib else None
    try:
        import inspect
        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker_equilibrium
        import OpenFUSIONToolkit.TokaMaker.bootstrap as _bs
        out["get_q_named_ravgs"] = \
            "'<1/R^2>'" in inspect.getsource(TokaMaker_equilibrium.get_q)
        out["has_get_fsa"] = hasattr(TokaMaker_equilibrium, "get_fsa")
        out["bootstrap_second_order_stencils"] = \
            "edge_order=2" in inspect.getsource(_bs)
    except Exception:
        pass
    return out


def bouquet_provenance():
    """Which bouquet produced it: version, and the commit if this is a checkout."""
    import subprocess
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    out = {}
    try:
        import bouquet as _bq
        out["version"] = getattr(_bq, "__version__", None)
    except Exception:
        out["version"] = None
    repo = os.path.abspath(os.path.join(_HERE, "..", ".."))
    for key, args in (("commit", ["rev-parse", "HEAD"]),
                      ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
        try:
            out[key] = subprocess.run(
                ["git", "-C", repo] + args, capture_output=True, text=True,
                check=True).stdout.strip()
        except Exception:
            out[key] = None
    try:
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                               capture_output=True, text=True, check=True)
        out["dirty"] = bool(dirty.stdout.strip())
    except Exception:
        out["dirty"] = None
    return out


def source_jphi_archival(source):
    """The run's j_phi archival convention, as ``regenerate_golden_run.py``
    stamps it on the archive root (``"input"``), or ``None`` when the source
    does not say (a run not made by that recipe)."""
    if not source or not os.path.isfile(source):
        return None
    with h5py.File(source, "r") as hf:
        v = hf.attrs.get("golden_jphi_archival")
    if isinstance(v, bytes):
        v = v.decode()
    return None if v is None else str(v)


def fixture_provenance(source=None, eqdsk="all", seed=RNG_STREAM_SEED):
    """Everything needed to reproduce -- or to diagnose -- this fixture.

    Deliberately carries no hostname, user name or filesystem path: the source
    is recorded by BASENAME only, and the environment by content digests.
    """
    import datetime
    return {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": os.path.basename(__file__),
        "generator_args": {
            "source_basename": os.path.basename(source) if source else None,
            "eqdsk": eqdsk,
            "rng_stream_seed": int(seed),
            # input (the systematics replay premise) vs achieved current;
            # see regenerate_golden_run.py
            "jphi_archival": source_jphi_archival(source),
        },
        "bouquet": bouquet_provenance(),
        "oft": oft_provenance(),
        "numerics": blas_provenance(),
    }


#: Provenance keys mirrored onto the slim .h5 root attrs, flattened, so a
#: reader with only the fixture in hand (no manifest) still knows what made it.
def _flat_provenance(prov, prefix="prov"):
    flat = {}

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}_{k}")
        elif node is not None:
            flat[path] = node if isinstance(node, (str, int, float, bool)) \
                else str(node)
    _walk(prov, prefix)
    return flat


def _fixture_stamp(slim_path):
    """Read back the provenance attrs a built fixture carries (or ``{}``)."""
    try:
        with h5py.File(slim_path, "r") as hf:
            return {k: (v.decode() if isinstance(v, bytes) else
                        (v.item() if hasattr(v, "item") else v))
                    for k, v in hf.attrs.items()
                    if str(k).startswith("prov")}
    except Exception:
        return {}


def blas_provenance():
    """What the drawn values depend on besides bouquet itself.

    The draw factorises its kernel through LAPACK, so the build is part of the
    bitwise identity: reproducible on one machine, close-but-not-equal across
    machines (see ``draw_stream``).
    """
    blas = "unknown"
    try:                                        # numpy >= 1.25
        cfg = np.__config__.show(mode="dicts")
        dep = cfg["Build Dependencies"]["blas"]
        blas = f"{dep.get('name', '?')}-{dep.get('version', '?')}"
    except Exception:
        pass
    return {"platform": platform.system(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "blas": blas}


def draw_stream(psi_kin, psi_N, profiles, sigmas, seed=RNG_STREAM_SEED):
    """The seeded GPR draw stream, in the draw path's own order.

    Replays exactly what ``perturb_kinetic_equilibrium`` does for one draw --
    ne, Te, ni, Ti through ``_draw_monotonic_perturbation`` (each normalised by
    its on-axis value), then the :math:`j_\\phi` GPR candidate -- off ONE
    ``make_rng(seed)`` Generator.  No solver and no mesh, so this is cheap
    enough to pin as a golden.

    How reproducible is it?  The GP draw is a fixed-order Cholesky factor
    applied to the seeded normals, so the values are

      * **bitwise** stable for a given seed on a given machine -- the contract
        ``generate()`` relies on, and what the manifest's SHA-256 pins;
      * stable to ~1e-9 across LAPACK/libm builds: Cholesky has no discrete
        choices (no eigenvector signs, no null-space basis, no pivots), so a
        different build can only differ by rounding.  The eigh factorisation
        this replaced handed LAPACK both a sign convention and a null-space
        basis, and the same seed differed by 1.3% between the macOS and Linux
        CI builds.

    Bitwise equality across machines is still not a claim we can make --
    rounding inside LAPACK and libm is build-dependent -- so the manifest
    carries :func:`blas_provenance` and the test compares numerically
    off-machine, bitwise on-machine.

    Returns ``{channel: ndarray}`` on the grid each channel is drawn on.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    from bouquet.sampling import make_rng, _draw_monotonic_perturbation
    from bouquet.sampling import generate_perturbed_GPR

    rng = make_rng(seed)
    out = {}
    for ch, ls in (("ne", 0.5), ("te", 0.4), ("ni", 0.5), ("ti", 0.4)):
        p = np.asarray(profiles[ch], dtype=float)
        s = np.asarray(sigmas[ch], dtype=float)
        out[ch] = _draw_monotonic_perturbation(
            psi_kin, p / p[0], s / p[0], ls, rng=rng) * p[0]
    jp = np.asarray(profiles["jphi"], dtype=float)
    sj = np.asarray(sigmas["jphi"], dtype=float)
    out["jphi"] = generate_perturbed_GPR(
        psi_N, jp / jp[0], sigma_profile=sj / jp[0], length_scale=0.25,
        n_samples=1, rng=rng, diag_plot=False) * jp[0]
    return out


def build_rng_stream(source_slim=None, out_dir=_HERE, seed=RNG_STREAM_SEED):
    """Pin the seeded draw stream against the golden baseline profiles.

    Reads the baseline profiles + sigma envelopes out of the slim fixture and
    writes ``rng_stream_manifest.json``: a SHA-256 of each drawn channel plus a
    few sampled values, so the git diff shows both that something moved and
    roughly where.  This is the golden that became possible only once the seed
    actually reached the GPR -- before that the stream was OS entropy and no
    draw-level value could be pinned at all.
    """
    slim = source_slim or os.path.join(out_dir, SLIM_NAME)
    with h5py.File(slim, "r") as hf:
        bkeys = sorted(hf["scan"].keys()) if "scan" in hf else [None]
        bl = hf[f"scan/{bkeys[0]}/_baseline"] if bkeys[0] is not None \
            else hf["_baseline"]
        psi_kin = np.asarray(bl["psi_N_kinetic"][()], dtype=float)
        psi_N = np.asarray(bl["psi_N"][()], dtype=float)
        profiles = {"ne": bl["n_e"][()], "te": bl["T_e"][()],
                    "ni": bl["n_i"][()], "ti": bl["T_i"][()],
                    "jphi": bl["j_phi"][()]}
        sigmas = {"ne": bl["sigma_ne"][()], "te": bl["sigma_te"][()],
                  "ni": bl["sigma_ni"][()], "ti": bl["sigma_ti"][()],
                  "jphi": bl["sigma_jphi"][()]}

    drawn = draw_stream(psi_kin, psi_N, profiles, sigmas, seed=seed)
    man = {
        "seed": int(seed),
        "source_fixture": os.path.basename(slim),
        "grids": {"psi_N_kinetic": int(psi_kin.size),
                  "psi_N": int(psi_N.size)},
        # which machine pinned it -- the SHA-256s are only meaningful there
        "pinned_on": blas_provenance(),
        # ... and what produced the fixture the stream was drawn FROM.  The
        # stream is pure NumPy (no solver, no OFT), but its inputs are the
        # fixture's baseline profiles, so it moves when the fixture does.
        "source_provenance": _fixture_stamp(slim),
        "channels": {},
    }
    for ch, arr in drawn.items():
        a = np.asarray(arr, dtype=float)
        idx = [0, a.size // 4, a.size // 2, (3 * a.size) // 4, a.size - 1]
        man["channels"][ch] = {
            "n": int(a.size),
            "sha256": _digest(a),
            "sample_indices": idx,
            "sample_values": [float(a[i]) for i in idx],
            "min": float(a.min()), "max": float(a.max()),
        }
    path = os.path.join(out_dir, RNG_MANIFEST_NAME)
    with open(path, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    print(f"[golden] wrote {path}  (seed={seed}, "
          f"{len(man['channels'])} channels)")
    return path


# A committed fixture must not name a host, a user or a filesystem location.
# The self-consistent bootstrap record (the jbs_loop_json attrs) of an older
# build carried the OFT build's package PATH (current builds record a
# path-free build identifier instead, jbs_loop.oft_build_info), which is right
# for nobody's public fixture: keep its basename (the build identity is
# stamped separately, by content digest).  The guard below then refuses a
# fixture in which any filesystem path survived, wherever it came from.
#
# What counts as a path (each alternative is anchored so that it cannot start
# in the middle of a word, a number or a unit such as "A/m^2" or "1/R"):
#   * an absolute path under a well-known root (/Users, /home, /usr,
#     /Volumes, /opt, /mnt, /tmp, /private, /var, ...);
#   * ANY absolute path of two or more components ("/a/b") that starts at the
#     beginning of a string or after whitespace, a quote, "=", ":", "," or an
#     opening bracket;
#   * a home-relative path: "~/..." or "~user/...";
#   * a parent-relative path: "../...";
#   * a Windows drive path: "C:\..." or "C:/...".
_ABS_PATH_RE = None

_PATH_ROOTS = ("Users|home|usr|Volumes|mnt|tmp|private|var|scratch|cscratch|"
               "opt|srv|data|work|global|gpfs|lustre|net|nfs|afs|root|media|"
               "Library|Applications|System|etc|run|proj|project|projects|"
               "space|storage|nobackup")


def _abs_path_re():
    global _ABS_PATH_RE
    if _ABS_PATH_RE is None:
        import re
        _ABS_PATH_RE = re.compile(
            # a well-known root, wherever it is not glued to a word
            r"(?<![A-Za-z0-9_.])/(?:" + _PATH_ROOTS + r")(?:/[^\s\"',;]*|\b)"
            # any /a/b[/...] at a token boundary
            r"|(?:^|(?<=[\s\"'=:,(\[{]))/[A-Za-z0-9_.-]+/[^\s\"',;]*"
            # home-relative
            r"|(?<![A-Za-z0-9_.~])~(?:[A-Za-z_][A-Za-z0-9_.-]*)?/[^\s\"',;]*"
            # parent-relative
            r"|(?<![A-Za-z0-9_])\.\./[^\s\"',;]*"
            # a Windows drive
            r"|(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"',;]*")
    return _ABS_PATH_RE


def _scrub_paths(node):
    """Recursively replace filesystem paths in a JSON-like record by their
    basename."""
    if isinstance(node, dict):
        return {k: _scrub_paths(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_scrub_paths(v) for v in node]
    if isinstance(node, str) and _abs_path_re().search(node):
        return _abs_path_re().sub(
            lambda m: (os.path.basename(
                m.group(0).replace("\\", "/").rstrip("/")) or "<path>"),
            node)
    return node


def find_filesystem_paths(value):
    """Every path-like substring in *value*: a str, bytes, a JSON document, or
    any (nested) list / tuple / dict / numpy array of them -- string ARRAYS
    included, element by element.  Returns a list of the matches."""
    rx = _abs_path_re()
    out = []

    def _walk(v):
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("S", "O", "U"):
                for el in v.ravel().tolist():
                    _walk(el)
            return
        if isinstance(v, (bytes, np.bytes_)):
            v = bytes(v).decode(errors="replace")
        if isinstance(v, (str, np.str_)):
            out.extend(m.group(0) for m in rx.finditer(str(v)))
            return
        if isinstance(v, dict):
            for k, x in v.items():
                _walk(k)
                _walk(x)
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)

    _walk(value)
    return out


def _scrub_attr(key, value):
    """The attr value to store in the fixture (loop records path-scrubbed)."""
    if key == "jbs_loop_json":
        raw = value.decode() if isinstance(value, bytes) else str(value)
        return json.dumps(_scrub_paths(json.loads(raw)))
    return value


def assert_no_filesystem_paths(slim_path):
    """Refuse a fixture that names an absolute filesystem path anywhere a
    reader would see text: string attrs, string datasets (config_json) and
    the geqdsk header lines."""
    import gzip  # noqa: F401  (geqdsks are stored gzip-filtered by h5py)
    hits = []
    with h5py.File(slim_path, "r") as hf:
        def _check(where, val):
            # str, bytes, and string ARRAYS (attrs or datasets), element by
            # element -- an array used to be skipped silently
            for m in find_filesystem_paths(val):
                hits.append(f"{where}: {m[:80]}")

        for k, v in hf.attrs.items():
            _check(f"/@{k}", v)

        def _v(name, obj):
            for k, v in obj.attrs.items():
                _check(f"{name}@{k}", v)
            if isinstance(obj, h5py.Dataset):
                if _is_eqdsk_name(name.rsplit("/", 1)[-1]):
                    head = bytes(obj[()])[:4096]
                    _check(name + " (geqdsk head)", head)
                elif obj.dtype.kind in ("S", "O", "U"):
                    _check(name, obj[()])
        hf.visititems(_v)
    if hits:
        raise SystemExit("REFUSING: the fixture names filesystem paths "
                         "(public repo):\n  " + "\n  ".join(hits[:20]))


def build(source, out_dir=_HERE, eqdsk="all"):
    if eqdsk not in ("all", "subset", "none"):
        raise ValueError("eqdsk must be 'all', 'subset', or 'none'")
    slim_path = os.path.join(out_dir, SLIM_NAME)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    prov = fixture_provenance(source=source, eqdsk=eqdsk)
    manifest = {
        "source_basename": os.path.basename(source),
        "eqdsk_retention": eqdsk,
        "provenance": prov,
        "tolerances": TOLERANCES,
        "scans": {},
    }
    ip_map = {}            # full group path -> Ip (injected as attr in pass 2)
    keep_eqdsk = set()     # full .eqdsk dataset names to KEEP (gzipped)

    # ---- pass 1: read source -> manifest, Ip map, which geqdsks to keep --
    with h5py.File(source, "r") as hf:
        for bkey, prefix in _scan_keys(hf):
            parent = hf[prefix] if prefix else hf
            base = (prefix + "/") if prefix else ""
            scan_entry = {"baseline": {}, "draws": {}}

            bl = parent["_baseline"] if "_baseline" in parent else None
            bl_boundary = None
            if bl is not None and "recon_lcfs_ref" in bl:
                bl_boundary = np.asarray(bl["recon_lcfs_ref"][()], dtype=float)

            if bl is not None:
                eqk = [k for k in bl.keys() if _is_eqdsk_name(k)]
                if eqk:
                    ip = _ip_from_eqdsk_bytes(bytes(bl[eqk[0]][()]))
                    ip_map[f"{base}_baseline"] = ip
                    scan_entry["baseline"]["Ip"] = ip
                    if eqdsk in ("all", "subset"):  # baseline always kept
                        keep_eqdsk.add(f"{base}_baseline/{eqk[0]}")
                        scan_entry["baseline"]["has_eqdsk"] = True
                for k in ("Ip_target", "l_i_target"):
                    if k in bl.attrs:
                        scan_entry["baseline"][k] = float(bl.attrs[k])
                if "x_points" in bl:
                    scan_entry["baseline"]["x_points"] = \
                        np.asarray(bl["x_points"][()], dtype=float).tolist()
                if "diverted" in bl.attrs:
                    scan_entry["baseline"]["diverted"] = bool(bl.attrs["diverted"])

            draw_keys = sorted(
                int(k) for k in parent.keys()
                if k not in ("_baseline", "scan") and str(k).lstrip("-").isdigit())
            eqk_name = {}  # draw idx -> eqdsk dataset name
            for i in draw_keys:
                grp = parent[str(i)]
                rec = {}
                eqk = [k for k in grp.keys() if _is_eqdsk_name(k)]
                if eqk:
                    eqk_name[i] = eqk[0]
                    ip = _ip_from_eqdsk_bytes(bytes(grp[eqk[0]][()]))
                    ip_map[f"{base}{i}"] = ip
                    rec["Ip"] = ip
                for k in ("l_i(1)", "l_i(3)", "max_F_drift_pct",
                          "max_VSC_drift_pct", "inspec_F_max", "inspec_VSC_max",
                          "homotopy_pass", "l_i_target_used"):
                    if k in grp.attrs:
                        rec[k] = float(grp.attrs[k])
                if "in_spec" in grp.attrs:
                    rec["in_spec"] = bool(grp.attrs["in_spec"])
                rms, mx = _boundary_devs(bl_boundary, grp)
                rec["bnd_rms_mm"] = rms
                rec["bnd_max_mm"] = mx
                if "coil_currents [A]" in grp and "coil_names" in grp.attrs:
                    # legacy layout: bracketed dataset + JSON names attr
                    names = json.loads(grp.attrs["coil_names"])
                    vals = np.asarray(grp["coil_currents [A]"][()], dtype=float)
                    rec["coil_currents"] = {n: float(v)
                                            for n, v in zip(names, vals)}
                elif "coil_currents" in grp and "coil_names" in grp:
                    # schema-v2 layout: clean dataset names
                    names = [n.decode() if isinstance(n, bytes) else str(n)
                             for n in grp["coil_names"][()]]
                    vals = np.asarray(grp["coil_currents"][()], dtype=float)
                    rec["coil_currents"] = {n: float(v)
                                            for n, v in zip(names, vals)}
                if "x_points" in grp:
                    rec["x_points"] = \
                        np.asarray(grp["x_points"][()], dtype=float).tolist()
                if "diverted" in grp.attrs:
                    rec["diverted"] = bool(grp.attrs["diverted"])
                scan_entry["draws"][str(i)] = rec

            # decide which draw geqdsks to keep
            if eqdsk == "all":
                keep_draws = set(draw_keys)
            elif eqdsk == "subset":
                keep_draws = _select_subset(scan_entry["draws"])
            else:
                keep_draws = set()
            for i in keep_draws:
                if i in eqk_name:
                    keep_eqdsk.add(f"{base}{i}/{eqk_name[i]}")
                    scan_entry["draws"][str(i)]["has_eqdsk"] = True

            scan_entry["n_draws"] = len(draw_keys)
            scan_entry["n_in_spec"] = sum(
                1 for d in scan_entry["draws"].values() if d.get("in_spec"))
            scan_entry["draw_indices"] = draw_keys
            scan_entry["eqdsk_indices"] = sorted(keep_draws)
            manifest["scans"][str(bkey)] = scan_entry

    # ---- pass 2: clean rebuild (keep gzipped geqdsks, drop pfiles) -------
    if os.path.exists(slim_path):
        os.remove(slim_path)
    with h5py.File(source, "r") as src, h5py.File(slim_path, "w") as dst:
        for k in src.attrs:
            dst.attrs[k] = src.attrs[k]

        # filter flags are run-state, not generation output -- strip them so
        # the fixture is always the canonical "unfiltered" run regardless of
        # whether the source h5 had filters applied (e.g. by the notebook).
        _filter_attrs = ("passes_coil_filter", "passes_boundary_filter",
                         "selected")

        def _copy(name, obj):
            if isinstance(obj, h5py.Group):
                g = dst.require_group(name)
                for ak in obj.attrs:
                    if ak in _filter_attrs:
                        continue
                    g.attrs[ak] = _scrub_attr(ak, obj.attrs[ak])
                return
            # dataset
            if _is_pfile_name(name.rsplit("/", 1)[-1]):
                return                      # always dropped
            if _is_eqdsk_name(name.rsplit("/", 1)[-1]):
                if name not in keep_eqdsk:
                    return                  # dropped per --eqdsk
                raw = bytes(obj[()])
                arr = np.frombuffer(raw, dtype=np.uint8)
                d = dst.create_dataset(name, data=arr, compression="gzip",
                                       compression_opts=_EQDSK_GZIP_LEVEL)
                for ak in obj.attrs:
                    d.attrs[ak] = obj.attrs[ak]
                return
            d = dst.create_dataset(name, data=obj[()])
            for ak in obj.attrs:
                d.attrs[ak] = obj.attrs[ak]
        src.visititems(_copy)

        # inject the extracted Ip as a group attr (so it is asserted even
        # for groups whose geqdsk blob was dropped)
        for gpath, ip in ip_map.items():
            if gpath in dst:
                dst[gpath].attrs["Ip"] = ip

        # ... and stamp the provenance onto the fixture itself, not just the
        # manifest: a stale fixture is diagnosed from the file someone has in
        # front of them, and the manifest can go missing or be regenerated
        # separately.  Flattened because HDF5 attrs are scalars.
        for k, v in _flat_provenance(prov).items():
            dst.attrs[k] = v
        dst.attrs["prov_schema"] = 1

    assert_no_filesystem_paths(slim_path)
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    size_mb = os.path.getsize(slim_path) / 1e6
    print(f"[golden] wrote {slim_path}  ({size_mb:.2f} MB, eqdsk={eqdsk})")
    print(f"[golden] wrote {manifest_path}")
    print(f"[golden] j_phi archival: "
          f"{prov['generator_args']['jphi_archival'] or 'NOT STATED by the source'}")
    for sk, se in manifest["scans"].items():
        print(f"[golden]   scan {sk}: {se['n_draws']} draws, "
              f"{se['n_in_spec']} in-spec, "
              f"{len(se['eqdsk_indices'])} geqdsks kept, "
              f"indices {se['draw_indices']}")
    return slim_path, manifest_path


#: significant digits of the legacy JSON's float arrays (12: a relative
#: rounding of <= 5e-13, far below every comparison made against them, at
#: ~25 % fewer bytes than the exact 17-digit repr)
LEGACY_SIG_DIGITS = 12


def _f(a):
    """A float array as a JSON list, rounded to :data:`LEGACY_SIG_DIGITS`."""
    return [float(f"{v:.{LEGACY_SIG_DIGITS}g}")
            for v in np.asarray(a, dtype=float).ravel()]


def _attr_scalar(v):
    if isinstance(v, bytes):
        return v.decode()
    if hasattr(v, "item"):
        return v.item()
    return v


def build_legacy_json(source, out_dir=_HERE):
    """The slim LEGACY golden: a small JSON of a full legacy-path run.

    ``source`` is the archive of ``regenerate_golden_run.py
    --reconstruction-engine legacy``.  Kept: the stored config (verbatim, so
    the run can be regenerated from this file alone:
    ``regenerate_golden_run.py --config-from <this json>``), the
    reconstruction's scalars, coil currents, X-points, its profiles and a
    uniform subsample of its LCFS reference, every draw's scalars, and for
    the first :data:`LEGACY_REPLAY_DRAWS` in-spec draws their profiles, coil
    currents and their boundary RMS to the (subsampled) reconstruction LCFS
    -- what the legacy systematics replay (``tests/test_systematics.py``)
    needs.  No geqdsk, no p-file, no LCFS trace of a draw."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
    from bouquet.schema import read_jbs_loop, read_current_split_convention
    from bouquet.utils import _read_coil_names
    from scipy.spatial import cKDTree
    out_path = os.path.join(out_dir, LEGACY_JSON_NAME)
    prov = fixture_provenance(source=source, eqdsk="none")
    with h5py.File(source, "r") as hf:
        g = hf["scan/0"]
        cj = g["config_json"][()]
        cj = cj.decode() if isinstance(cj, bytes) else str(cj)
        # re-serialised compactly: the same values (floats round-trip
        # exactly), without the stored text's indentation
        cj = json.dumps(json.loads(cj), separators=(",", ":"))
        eng = json.loads(cj)["generation"].get("reconstruction_engine")
        if eng != "legacy":
            raise SystemExit(f"REFUSING: {os.path.basename(source)} was run "
                             f"with reconstruction_engine={eng!r}; the "
                             "legacy golden is a legacy-path run")
        bl = g["_baseline"]
        eqk = [k for k in bl.keys() if _is_eqdsk_name(k)]
        lcfs = np.asarray(bl["recon_lcfs_ref"][()], dtype=float)
        lcfs_sub = np.round(lcfs[::LEGACY_LCFS_STRIDE], LEGACY_LCFS_DECIMALS)
        loop = read_jbs_loop(bl)
        base = dict(
            attrs={k: _attr_scalar(bl.attrs[k]) for k in
                   ("Ip_target", "l_i_target", "l_i_scale", "diverted",
                    "source_kind", "jbs_converged", "jbs_n_passes")
                   if k in bl.attrs},
            Ip_eqdsk=(_ip_from_eqdsk_bytes(bytes(bl[eqk[0]][()]))
                      if eqk else None),
            edge_pressure=(json.loads(_attr_scalar(
                bl.attrs["edge_pressure_json"]))
                if "edge_pressure_json" in bl.attrs else None),
            jbs_loop=(None if loop is None else _scrub_paths(dict(
                converged=loop.get("converged"),
                n_passes=loop.get("n_passes"),
                post_corrective=loop.get("post_corrective")))),
            coil_names=list(_read_coil_names(bl)),
            coil_currents=_f(bl["coil_currents"][()]),
            x_points=np.asarray(bl["x_points"][()], dtype=float).tolist()
            if "x_points" in bl else None,
            # where the archived split keeps the pressure-driven p'G (owner
            # decision D2: its own j_pressure beside a j_inductive that does
            # not carry it; a pre-#64 archive carried it in j_inductive).
            # The replay composes the legacy path's carried inductive from
            # these, so the record says which form it holds.
            current_split_convention=read_current_split_convention(bl),
            profiles={k: _f(bl[k][()]) for k in
                      ("psi_N", "psi_N_kinetic", "n_e", "T_e", "n_i", "T_i",
                       "aux_zeff", "j_phi", "j_BS", "j_inductive",
                       "j_pressure", "pressure") if k in bl},
            recon_lcfs_ref=dict(stride=LEGACY_LCFS_STRIDE,
                                decimals=LEGACY_LCFS_DECIMALS,
                                n_full=int(lcfs.shape[0]),
                                points=lcfs_sub.tolist()))
        idxs = sorted(int(k) for k in g if k.lstrip("-").isdigit())
        summary, in_spec = {}, []
        for i in idxs:
            gi = g[str(i)]
            a = gi.attrs
            eqk = [k for k in gi.keys() if _is_eqdsk_name(k)]
            summary[str(i)] = dict(
                {k: _attr_scalar(a[k]) for k in
                 ("count", "in_spec", "l_i(1)", "l_i(3)", "l_i_target_used",
                  "max_F_drift_pct", "max_VSC_drift_pct", "homotopy_pass",
                  "jbs_converged", "jbs_n_passes") if k in a},
                Ip_eqdsk=(_ip_from_eqdsk_bytes(bytes(gi[eqk[0]][()]))
                          if eqk else None))
            if bool(a.get("in_spec")):
                in_spec.append(i)
        tree_sub = lcfs_sub
        replay = {}
        for i in in_spec[:LEGACY_REPLAY_DRAWS]:
            gi = g[str(i)]
            pert = np.asarray(gi["perturbed_lcfs_ref"][()], dtype=float)
            d_sub, _ = cKDTree(pert).query(tree_sub)
            d_full, _ = cKDTree(pert).query(lcfs)
            replay[str(i)] = dict(
                summary[str(i)],
                coil_names=list(_read_coil_names(gi)),
                coil_currents=_f(gi["coil_currents"][()]),
                current_split_convention=read_current_split_convention(
                    gi, baseline_attrs=bl.attrs),
                profiles={k: _f(gi[k][()]) for k in
                          ("n_e", "T_e", "n_i", "T_i", "aux_zeff", "j_phi",
                           "j_inductive", "j_pressure") if k in gi},
                bnd_rms_to_recon_mm=float(np.sqrt(np.mean(d_sub ** 2)) * 1e3),
                bnd_rms_to_recon_mm_full_trace=float(
                    np.sqrt(np.mean(d_full ** 2)) * 1e3))
        root = {k: _attr_scalar(hf.attrs[k]) for k in
                ("bouquet_version", "schema_version", "golden_jphi_archival")
                if k in hf.attrs}
    doc = dict(
        what=("slim LEGACY golden: the golden recipe "
              "(regenerate_golden_run.py) on reconstruction_engine='legacy'"
              "; see tests/golden/README.md"),
        source_basename=os.path.basename(source),
        provenance=prov,
        root_attrs=root,
        config_json=cj,
        baseline=base,
        n_draws=len(idxs),
        n_in_spec=len(in_spec),
        draws=summary,
        replay_draws=replay)
    hits = find_filesystem_paths(doc)
    if hits:
        raise SystemExit("REFUSING: the legacy golden names filesystem "
                         "paths (public repo):\n  " + "\n  ".join(hits[:20]))
    with open(out_path, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")
    kb = os.path.getsize(out_path) / 1e3
    print(f"[golden] wrote {out_path}  ({kb:.0f} kB; {len(idxs)} draws, "
          f"{len(in_spec)} in spec, replay draws {sorted(replay, key=int)})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=_DEFAULT_SOURCE,
                    help="full bouquet .h5 to slim (default: the D3D-like "
                         "example artifact)")
    ap.add_argument("--eqdsk", default="all",
                    choices=("all", "subset", "none"),
                    help="geqdsk retention: all (default, ~11 MB), subset "
                         "(baseline + representative draws, ~5 MB), or none")
    ap.add_argument("--rng-stream-only", action="store_true",
                    help="re-pin rng_stream_manifest.json from the EXISTING "
                         "slim fixture and stop (no --source needed)")
    ap.add_argument("--legacy-json", action="store_true",
                    help="write the slim LEGACY golden JSON "
                         f"({LEGACY_JSON_NAME}) from --source (a run of "
                         "regenerate_golden_run.py --reconstruction-engine "
                         "legacy) and stop; the h5 fixture is untouched")
    args = ap.parse_args()
    if args.legacy_json:
        if not os.path.isfile(args.source):
            raise SystemExit(f"source not found: {args.source}")
        build_legacy_json(args.source)
        raise SystemExit(0)
    if args.rng_stream_only:
        build_rng_stream()
        raise SystemExit(0)
    if not os.path.isfile(args.source):
        raise SystemExit(f"source not found: {args.source}")
    build(args.source, eqdsk=args.eqdsk)
    build_rng_stream()
