"""Regenerate the FULL golden bouquet run that ``make_golden_fixture.py`` slims.

This is the recipe of every systematics golden, written down so it cannot be
lost again: the class API run from the fixture's OWN stored config, with the
one archival switch the class API does not expose.

The recipe
----------
* **Config**: the stored ``scan/0/config_json`` of an existing fixture
  (default: the git-tracked slim fixture), loaded verbatim -- n_equils,
  seed, sigmas, the ``jbs_*`` loop settings and ceilings, solver settings,
  everything.  Only ``output_header`` (the archive name) and ``verbose``
  (log output only, no numerics) are set here.  The loop setting is whatever
  the stored config says; nothing is switched on or off.
* **Entry point**: ``Bouquet(cfg) -> setup_solver() -> prepare_baseline() ->
  generate()``, as the example notebooks run it.
* **Archival: INPUT current** (``store_achieved_jphi=False``).
  ``Bouquet.generate()`` hard-wires ``store_achieved_jphi=True``, which
  archives the ACHIEVED flux-surface-averaged current of each converged solve
  as ``j_phi`` (and ``j_inductive`` as the residual).  The systematics replay
  (``tests/test_systematics.py``) feeds ``_baseline/j_phi`` back to the
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
* **Environment**: one thread (``OMP_NUM_THREADS=1``; the stored config
  carries ``solver.nthreads=1``), on the OFT build the fixture is meant to
  pin.  Set ``BOUQUET_OFT_COMMIT`` / ``BOUQUET_OFT_BRANCH`` /
  ``BOUQUET_OFT_BUILD_ID`` when slimming, so the stamp states it.

The archive gets a root attr ``golden_jphi_archival = "input"``, which the
slim builder copies with every other root attr, so the fixture itself says
how it was archived.

Usage::

    OMP_NUM_THREADS=1 python tests/golden/regenerate_golden_run.py RUN_DIR \\
        [--config-from FIXTURE.h5|LEGACY.json] \\
        [--reconstruction-engine stored|unified|legacy]
    python tests/golden/make_golden_fixture.py \\
        --source RUN_DIR/D3Dlike_Hmode_golden.h5

``RUN_DIR`` receives links to the example inputs (geqdsk, p-file, mesh) under
the basenames the stored config names, the full archive, and a
``summary.json`` of the baseline and per-draw loop records.  The D3D-like
example run is ~6.5 h at one thread.
"""
import argparse
import json
import os
import sys
import time

import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_EXAMPLE_DIR = os.path.join(_REPO, "examples", "D3D-like")
_DEFAULT_CONFIG_FROM = os.path.join(_HERE, "D3Dlike_Hmode_golden_slim.h5")
ARCHIVE_HEADER = "D3Dlike_Hmode_golden"

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


def apply_engine(cfg, engine):
    """Run the stored config on ``engine`` (``"stored"``: as stored).

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


def link_example_inputs(cfg, run_dir):
    """Link the example inputs into ``run_dir`` under the config's basenames."""
    names = [cfg.source.geqdsk_path, cfg.source.profiles_path,
             cfg.solver.mesh_path]
    for name in names:
        if not name:
            continue
        dst = os.path.join(run_dir, os.path.basename(name))
        if os.path.exists(dst):
            continue
        src = os.path.join(_EXAMPLE_DIR, os.path.basename(name))
        if not os.path.isfile(src):
            raise SystemExit(f"example input not found: {src}")
        os.symlink(src, dst)


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="scratch directory for the full run")
    ap.add_argument("--config-from", default=_DEFAULT_CONFIG_FROM,
                    help="fixture whose stored scan/0/config_json is reused "
                         "(default: the git-tracked slim fixture)")
    ap.add_argument("--reconstruction-engine", default="stored",
                    choices=("stored", "unified", "legacy"),
                    help="the engine to run the stored config on (default: "
                         "as stored).  The 2026-10 fixture is the unified-"
                         "engine default run from the legacy-stored config "
                         "(--reconstruction-engine unified); the slim legacy "
                         "JSON golden is the stored legacy config as is")
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

    cfg = load_stored_config(args.config_from)
    cfg.output_header = ARCHIVE_HEADER
    cfg.verbose = bool(args.verbose)
    engine_changes = apply_engine(cfg, args.reconstruction_engine)
    for k, (was, now) in engine_changes.items():
        print(f"engine: generation.{k} {was!r} -> {now!r}", flush=True)
    gc = cfg.generation
    print("bouquet from", os.path.dirname(bq.__file__), flush=True)
    print(f"config from {os.path.basename(args.config_from)}: n_equils "
          f"{gc.n_equils}, seed {gc.seed}, nthreads {cfg.solver.nthreads}, "
          f"jbs_self_consistent {gc.jbs_self_consistent}, ceilings "
          f"{gc.jbs_max_passes} / {gc.jbs_max_passes_draw} / "
          f"{gc.jbs_max_passes_post_homotopy}; engine "
          f"{gc.reconstruction_engine}; archival: {JPHI_ARCHIVAL} "
          "current", flush=True)

    os.makedirs(args.run_dir, exist_ok=True)
    link_example_inputs(cfg, args.run_dir)
    os.chdir(args.run_dir)

    t0 = time.time()
    b = bq.Bouquet(cfg)
    b.setup_solver()
    bl = b.prepare_baseline()
    t1 = time.time()
    rec = (bl.reconstruction_metrics or {}).get("jbs_loop") or {}
    print("baseline done in %.0f s; loop converged=%s n_passes=%s "
          "l_i_target=%.6f" % (t1 - t0, rec.get("converged"),
                               rec.get("n_passes"), bl.l_i_target), flush=True)
    with InputCurrentArchival() as patch:
        b.generate()
    if patch.calls != 1:
        raise RuntimeError(f"expected one generate_bouquet call, saw {patch.calls}")
    t2 = time.time()

    archive = f"{ARCHIVE_HEADER}.h5"
    with h5py.File(archive, "a") as hf:
        hf.attrs["golden_jphi_archival"] = JPHI_ARCHIVAL

    out = dict(wall_baseline_s=t1 - t0, wall_generate_s=t2 - t1,
               jphi_archival=JPHI_ARCHIVAL,
               reconstruction_engine=gc.reconstruction_engine,
               engine_changes={k: list(v) for k, v in engine_changes.items()},
               l_i_target=float(bl.l_i_target), Ip_target=float(bl.Ip_target),
               baseline_jbs_loop=jsonable(rec),
               draws=[jsonable({k: d.get(k)
                                for k in ("count", "in_spec", "jbs_loop")})
                      for d in (b.diagnostics or []) if isinstance(d, dict)])
    with open("summary.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("DONE generate %.0f s -> %s" % (t2 - t1, archive), flush=True)


if __name__ == "__main__":
    main()
