"""Regenerate the FULL golden bouquet run that ``make_golden_fixture.py`` slims.

This is the recipe of every golden, written down so it cannot be lost again:
the class API on the synthetic test fixtures at the library's DEFAULTS, with
the one archival switch the class API does not expose.

The goldens
-----------
* ``--golden gfile`` (default): the synthetic D3D-like g-file with its
  IDA-layout profile file (``tests/data/g_synthetic.geqdsk`` +
  ``tests/data/diiid_profs_synthetic.cdf``), through
  ``Bouquet.from_geqdsk``.  Run on ``--reconstruction-engine unified`` (the
  default engine: the h5 golden) and ``legacy`` (the slim legacy JSON golden).
* ``--golden ids``: the synthetic dd's 1.2 s slice run IDA-hybrid
  (``tests/data/dd_synthetic.json.gz`` + ``tests/data/diiid_profs_synthetic.cdf``),
  through ``Bouquet.from_imas(..., ida_path=...)``: the production IDS route
  (unified engine, defaults).

The recipe
----------
* **Config**: built fresh by the class constructor, every setting at its
  default except the run's identity: ``n_draws`` 20 (attempts; no until-N),
  ``seed`` 12345, ``nthreads`` 1, the engine, and for ``ids`` the slice time.
  The legacy run (``--reconstruction-engine legacy``) also switches the
  per-draw wall budget off (``draw_wall_budget_s=None``,
  :data:`LEGACY_DRAW_WALL_BUDGET_S`; owner decision 2026-10-17): its draws
  take up to about ten minutes each on the validation host, and a wall-clock
  budget would make which draws a golden archives depend on the host's speed
  and load -- a golden must be host-independent (the legacy route is
  deprecated, #90).
  Only ``output_header`` (the archive name) and ``verbose`` (log output only,
  no numerics) are set besides.  So a golden embodies the defaults of the
  code that made it.  ``--config-from FIXTURE.h5|LEGACY.json`` instead
  replays an existing golden's STORED config verbatim (what made it, not the
  current defaults).
* **Inputs**: linked (the dd gunzipped) into ``RUN_DIR`` under their
  basenames and the run made from there, so the stored config names no
  filesystem path (the fixture is public).
* **Entry point**: ``Bouquet(cfg) -> setup_solver() -> prepare_baseline() ->
  generate()``, as the example notebooks run it.
* **Archival: INPUT current** (``store_achieved_jphi=False``).
  ``Bouquet.generate()`` hard-wires ``store_achieved_jphi=True``, which
  archives the ACHIEVED flux-surface-averaged current of each converged solve
  as ``jphi_total`` (and ``jphi_ind`` as the residual).  The systematics replay
  (``tests/test_systematics.py``) feeds ``_baseline/jphi_total`` back to the
  unperturbed ``jphi-linterp`` baseline solve as its input; that is a faithful
  replay only if the archive holds the current the generator handed that
  solve, i.e. the INPUT current (``docs/CHANGES_SUMMARY.md``, 1.3.0: "a replay
  premise 'archived current reproduces archived LCFS' requires the input, not
  the achieved, current").  An achieved-current fixture makes mode 1 solve a
  slightly different baseline (see ``README.md``, "mode-1 coil drift").
  ``store_achieved_jphi`` changes what is WRITTEN, never what is solved
  (``generate_bouquet`` reads it only at the two archival sites), so this
  script wraps ``bouquet.TokaMaker_interface.generate_bouquet`` for the
  duration of ``generate()`` and flips exactly that one keyword.  The wrapper
  refuses to run if ``generate()`` stops passing the keyword as ``True``, so
  a change to the class API cannot silently change the recipe.
* **Environment**: one thread (``OMP_NUM_THREADS=1``; ``nthreads=1``), on the
  OFT build the fixture is meant to pin.  Set ``BOUQUET_OFT_COMMIT`` /
  ``BOUQUET_OFT_BRANCH`` / ``BOUQUET_OFT_BUILD_ID`` when slimming, so the
  stamp states it.

The archive gets a root attr ``golden_jphi_archival = "input"``, which the
slim builder copies with every other root attr, so the fixture itself says
how it was archived.

The sigma=0 probe (``--sigma0-probe OUT.json``) builds the SAME config,
reconstructs and runs ``Bouquet.verify_sigma0_consistency()`` (the
zero-perturbation draw must reproduce the reconstruction), and writes its
verdict; it makes no draws and no archive.  It is a separate process because
the probe leaves the solver re-anchored, and the golden run must be the
production sequence untouched.

Usage::

    OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_DIR \\
        [--golden gfile|ids] [--reconstruction-engine unified|legacy] \\
        [--config-from FIXTURE.h5|LEGACY.json] [--sigma0-probe OUT.json]
    python tests/golden/make_golden_fixture.py [--golden ids] \\
        --source RUN_DIR/<archive>.h5

``RUN_DIR`` receives the linked inputs, the full archive and a
``summary.json`` of the baseline, the per-draw loop records and the counts.
Each golden is ~45 min at one thread.
"""
import argparse
import collections
import gzip
import json
import os
import shutil
import sys
import time

import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DATA_DIR = os.path.join(_REPO, "tests", "data")
_EXAMPLE_DIR = os.path.join(_REPO, "examples", "D3D-like")

#: the inputs of each golden: the synthetic test fixtures (tests/data) and
#: the D3D-like example mesh
GFILE = "g_synthetic.geqdsk"
PROFILES = "diiid_profs_synthetic.cdf"
DD = "dd_synthetic.json"            # kept gzipped in tests/data
MESH = "DIIID_mesh.h5"
GOLDEN_INPUTS = {"gfile": (GFILE, PROFILES, MESH),
                 "ids": (DD, PROFILES, MESH)}
#: the IDS golden's slice [s] (the IDA file's own slice is 1200 ms)
IDS_TIME = 1.2

#: the run's identity -- everything else is a default
N_DRAWS = 20
SEED = 12345
NTHREADS = 1
#: the legacy golden's per-draw wall budget: off (owner decision 2026-10-17;
#: see the module doc).  The h5 goldens keep the default (600 s).
LEGACY_DRAW_WALL_BUDGET_S = None

#: archive header (= archive basename) of each golden
ARCHIVE_HEADER = "D3Dlike_Hmode_golden"
ARCHIVE_HEADERS = {"gfile": ARCHIVE_HEADER, "ids": "D3Dlike_ids_golden"}

#: The archival convention this recipe produces, stamped on the archive root.
JPHI_ARCHIVAL = "input"


def load_stored_config(fixture):
    """The stored config of a slim fixture (``scan/0/config_json``) or of the
    slim legacy golden JSON (its ``config_json`` entry)."""
    from bouquet.config import BouquetConfig
    if fixture.endswith(".json"):
        with open(fixture) as fh:
            cj = json.load(fh)["config_json"]
    else:
        with h5py.File(fixture, "r") as hf:
            cj = hf["scan/0/config_json"][()]
    return BouquetConfig.from_json(
        cj.decode() if isinstance(cj, bytes) else str(cj))


def default_config(golden, engine):
    """The golden's config at the library defaults (see the module doc):
    inputs by basename (link them first, :func:`link_inputs`),
    :data:`N_DRAWS` attempts, :data:`SEED`, :data:`NTHREADS`."""
    import bouquet as bq
    if golden == "gfile":
        b = bq.Bouquet.from_geqdsk(GFILE, profiles=PROFILES, mesh=MESH,
                                   n_draws=N_DRAWS,
                                   header=ARCHIVE_HEADERS[golden],
                                   reconstruction_engine=engine,
                                   nthreads=NTHREADS)
    elif golden == "ids":
        b = bq.Bouquet.from_imas(DD, mesh=MESH, time=IDS_TIME,
                                 ida_path=PROFILES, n_draws=N_DRAWS,
                                 header=ARCHIVE_HEADERS[golden],
                                 reconstruction_engine=engine,
                                 nthreads=NTHREADS)
    else:
        raise ValueError(golden)
    cfg = b.config
    cfg.generation.seed = SEED
    if engine == "legacy":
        cfg.generation.draw_wall_budget_s = LEGACY_DRAW_WALL_BUDGET_S
    return cfg


def apply_engine(cfg, engine):
    """Run a stored config on ``engine`` (``"stored"``: as stored).

    ``"unified"`` on a config stored by the legacy path: the field is set and
    every legacy-path field the engine never reads
    (``bouquet.engine.ENGINE_UNREAD_LEGACY_FIELDS``) that the stored config
    holds at a non-default value is put back to its default -- the engine
    refuses them on a new config, and ignores them on a stored one
    (``config._stored_config_compat`` rule (c)), so this changes nothing the
    engine solves.  Returns ``{field: (stored, applied)}`` of every change.
    """
    gc = cfg.generation
    changed = {}
    if engine == "stored" or engine == gc.reconstruction_engine:
        return changed
    from bouquet.config import GenerationConfig
    from bouquet.engine import (ENGINE_DEPENDENT_DEFAULTS,
                                ENGINE_UNREAD_LEGACY_FIELDS,
                                engine_validated_value,
                                validate_engine_settings)
    changed["reconstruction_engine"] = (gc.reconstruction_engine, engine)
    gc.reconstruction_engine = engine
    if engine == "unified":
        dflt = GenerationConfig()
        for name in ENGINE_UNREAD_LEGACY_FIELDS:
            want = getattr(dflt, name)
            if name in ENGINE_DEPENDENT_DEFAULTS:
                # resolved per engine (default None): the engine's own value
                want = engine_validated_value(name, engine, "reconstruction")
            if getattr(gc, name) != want:
                changed[name] = (getattr(gc, name), want)
                setattr(gc, name, want)
    validate_engine_settings(gc)
    return changed


def config_inputs(cfg):
    """Every input file a config names (as stored)."""
    src = cfg.source
    return [n for n in (getattr(src, "geqdsk_path", None),
                        getattr(src, "profiles_path", None),
                        getattr(src, "ids_path", None),
                        getattr(src, "ida_path", None),
                        getattr(src, "LCFS_geqdsk", None),
                        getattr(cfg.uncertainty, "ida_path", None),
                        cfg.solver.mesh_path) if n]


def link_inputs(names, run_dir):
    """Put every input in ``names`` into ``run_dir`` under its basename: a
    link to the file in ``tests/data`` (else ``examples/D3D-like``); a dd the
    repository keeps gzipped (``<name>.gz``) is unpacked."""
    for name in names:
        base = os.path.basename(name)
        dst = os.path.join(run_dir, base)
        if os.path.exists(dst):
            continue
        for d in (_DATA_DIR, _EXAMPLE_DIR):
            src = os.path.join(d, base)
            if os.path.isfile(src):
                os.symlink(src, dst)
                break
            if os.path.isfile(src + ".gz"):
                with gzip.open(src + ".gz", "rb") as fi, \
                        open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                break
        else:
            raise SystemExit("input not found in tests/data or "
                             f"examples/D3D-like: {base}")


class InputCurrentArchival:
    """Context manager: ``generate()``'s ``generate_bouquet`` call archives the
    INPUT current (``store_achieved_jphi=False``); nothing else is touched."""

    def __enter__(self):
        import bouquet.TokaMaker_interface as ti
        self._ti = ti
        self._orig = ti.generate_bouquet
        self.calls = 0
        orig = self._orig

        def _generate_bouquet(*args, **kwargs):
            if kwargs.get("store_achieved_jphi") is not True:
                raise RuntimeError(
                    "Bouquet.generate() no longer passes "
                    "store_achieved_jphi=True to generate_bouquet (got "
                    f"{kwargs.get('store_achieved_jphi')!r}); re-check the "
                    "golden recipe before regenerating")
            kwargs["store_achieved_jphi"] = False
            self.calls += 1
            return orig(*args, **kwargs)

        ti.generate_bouquet = _generate_bouquet
        return self

    def __exit__(self, *exc):
        self._ti.generate_bouquet = self._orig
        return False


def archive_counts(archive, scan_key=0):
    """``{n_archived, n_in_spec, in_spec_draws}`` of the archive's scan."""
    with h5py.File(archive, "r") as hf:
        g = hf[f"scan/{scan_key}"]
        idx = sorted(int(k) for k in g if k.lstrip("-").isdigit())
        ins = [i for i in idx if bool(g[str(i)].attrs.get("in_spec"))]
    return dict(n_archived=len(idx), n_in_spec=len(ins), in_spec_draws=ins)


def _baseline_record(bl):
    eng = getattr(bl, "engine", None)
    eng = eng if isinstance(eng, dict) else {}
    return dict(l_i_target=float(bl.l_i_target),
                Ip_target=float(bl.Ip_target),
                bistability=eng.get("bistability"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="scratch directory for the full run")
    ap.add_argument("--golden", default="gfile", choices=("gfile", "ids"),
                    help="gfile: g_synthetic.geqdsk + diiid_profs_synthetic.cdf "
                         "(default); ids: dd_synthetic.json.gz at 1.2 s, "
                         "IDA-hybrid with diiid_profs_synthetic.cdf")
    ap.add_argument("--reconstruction-engine", default="unified",
                    choices=("unified", "legacy", "stored"),
                    help="the engine (default unified; legacy: the slim "
                         "legacy JSON golden; stored: as a --config-from "
                         "config stores it)")
    ap.add_argument("--config-from", default=None,
                    help="replay this fixture's STORED config verbatim "
                         "(scan/0/config_json of a slim h5, or a legacy "
                         "golden JSON) instead of the defaults")
    ap.add_argument("--sigma0-probe", default=None, metavar="OUT.json",
                    help="reconstruct with the same config, run "
                         "verify_sigma0_consistency() and write its verdict "
                         "to OUT.json; no draws, no archive")
    ap.add_argument("--verbose", action="store_true",
                    help="stream the solver log (output only; no numerics)")
    ap.add_argument("--traceback-every", type=float, default=0.0,
                    help="dump every thread's stack to stderr every N seconds "
                         "(diagnostics for a slow solve; 0 = off)")
    args = ap.parse_args(argv)
    if args.traceback_every > 0:
        import faulthandler
        faulthandler.dump_traceback_later(args.traceback_every, repeat=True)

    sys.path.insert(0, _REPO)
    import bouquet as bq
    from bouquet.jbs_loop import jsonable

    config_from = (os.path.abspath(args.config_from) if args.config_from
                   else None)
    sigma0_out = (os.path.abspath(args.sigma0_probe) if args.sigma0_probe
                  else None)
    os.makedirs(args.run_dir, exist_ok=True)
    os.chdir(args.run_dir)
    if config_from:
        cfg = load_stored_config(config_from)
        engine_changes = apply_engine(cfg, args.reconstruction_engine)
        origin = f"stored config of {os.path.basename(config_from)}"
        link_inputs(config_inputs(cfg), ".")
    else:
        if args.reconstruction_engine == "stored":
            raise SystemExit("--reconstruction-engine stored needs "
                             "--config-from")
        link_inputs(GOLDEN_INPUTS[args.golden], ".")
        cfg = default_config(args.golden, args.reconstruction_engine)
        engine_changes = {}
        origin = f"defaults ({args.golden} golden)"
    header = ARCHIVE_HEADERS[args.golden]
    cfg.output_header = header
    cfg.verbose = bool(args.verbose)
    for k, (was, now) in engine_changes.items():
        print(f"engine: generation.{k} {was!r} -> {now!r}", flush=True)
    gc = cfg.generation
    print("bouquet from", os.path.dirname(bq.__file__), flush=True)
    print(f"config from {origin}: source {type(cfg.source).__name__}, "
          f"n_equils {gc.n_equils}, seed {gc.seed}, nthreads "
          f"{cfg.solver.nthreads}, jbs_self_consistent "
          f"{gc.jbs_self_consistent}, engine {gc.reconstruction_engine}; "
          f"archival: {JPHI_ARCHIVAL} current", flush=True)

    t0 = time.time()
    b = bq.Bouquet(cfg)
    b.setup_solver()
    bl = b.prepare_baseline()
    t1 = time.time()
    rec = (bl.reconstruction_metrics or {}).get("jbs_loop") or {}
    print("baseline done in %.0f s; loop converged=%s n_passes=%s "
          "l_i_target=%.6f" % (t1 - t0, rec.get("converged"),
                               rec.get("n_passes"), bl.l_i_target), flush=True)

    if sigma0_out:
        s0 = b.verify_sigma0_consistency()
        t2 = time.time()
        keep = {k: v for k, v in (s0 or {}).items()
                if k not in ("spike0", "record")}
        out = dict(golden=args.golden, origin=origin,
                   reconstruction_engine=gc.reconstruction_engine,
                   wall_baseline_s=t1 - t0, wall_sigma0_s=t2 - t1,
                   passed=bool((s0 or {}).get("passed")),
                   baseline=_baseline_record(bl), sigma0=keep)
        with open(sigma0_out, "w") as fh:
            json.dump(jsonable(out), fh, indent=1, default=str)
        print("DONE sigma0 probe: passed=%s (%.0f s) -> %s"
              % (out["passed"], t2 - t1, sigma0_out), flush=True)
        return

    with InputCurrentArchival() as patch:
        b.generate()
    if patch.calls != 1:
        raise RuntimeError(f"expected one generate_bouquet call, saw {patch.calls}")
    t2 = time.time()

    archive = f"{header}.h5"
    with h5py.File(archive, "a") as hf:
        hf.attrs["golden_jphi_archival"] = JPHI_ARCHIVAL
    counts = archive_counts(archive, scan_key=gc.scan_key)
    rej = collections.Counter(
        (r.get("reason") if isinstance(r, dict) else str(r))
        for r in (getattr(b, "draw_rejections", None) or []))

    out = dict(golden=args.golden, origin=origin,
               wall_baseline_s=t1 - t0, wall_generate_s=t2 - t1,
               jphi_archival=JPHI_ARCHIVAL,
               reconstruction_engine=gc.reconstruction_engine,
               engine_changes={k: list(v) for k, v in engine_changes.items()},
               attempts=int(gc.n_equils), seed=gc.seed,
               rejections=dict(rej), **counts,
               baseline=_baseline_record(bl),
               baseline_jbs_loop=jsonable(rec),
               draws=[jsonable({k: d.get(k)
                                for k in ("count", "in_spec", "jbs_loop")})
                      for d in (b.diagnostics or []) if isinstance(d, dict)])
    with open("summary.json", "w") as fh:
        json.dump(jsonable(out), fh, indent=1, default=str)
    print("DONE generate %.0f s -> %s: %d attempts, %d archived, %d in spec"
          % (t2 - t1, archive, gc.n_equils, counts["n_archived"],
             counts["n_in_spec"]), flush=True)


if __name__ == "__main__":
    main()
