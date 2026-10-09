"""Out-of-tree bitwise A/B of the LEGACY path (``jbs_self_consistent=False``)
between two checkouts of bouquet -- the proof that a change gated on the
self-consistent loop leaves the legacy path bit for bit.

Usage (on a machine with OFT; nothing here runs in the test suite)::

    python tests/probes/legacy_bitwise_ab.py <tree_before> <tree_after> <outdir>

For each tree it runs on the synthetic D3D-like example with
``jbs_self_consistent=False`` and a fixed seed, each part in its OWN fresh
interpreter with ``PYTHONPATH=<tree>`` (the solver environment can be created
only once per interpreter): part ``rec`` -- the g-file reconstruction,
``verify_sigma0_consistency()`` and ``generate(n=2)``; part ``imas`` -- the
modelling-source (OMAS) diff baseline and ``generate(n=1)``.  It then compares the two runs' Baseline arrays, the sigma=0
check's numbers and the two HDF5 archives dataset by dataset and numeric
attribute by numeric attribute (``np.array_equal``; provenance/time-stamp
strings excluded), and writes ``<outdir>/legacy_ab.json`` with every
difference found.  Exit status 0 iff nothing differs.

Synthetic inputs only.
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "..", "examples",
                                        "D3D-like"))
_SKIP_ATTRS = ("config_json", "created", "timestamp", "bouquet_version",
               "generation_provenance", "git", "oft_build", "wall")


def _child(tree, outdir, tag, part):
    import numpy as np
    sys.path.insert(0, tree)
    oft = os.environ.get("OFT_PYTHONPATH")
    if oft:
        sys.path.append(oft)
    import bouquet as bq
    assert os.path.abspath(bq.__file__).startswith(os.path.abspath(tree))
    res = {}
    geq = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
    pf = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
    mesh = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
    omas = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
    if part == "rec":
        _child_rec(bq, np, res, geq, pf, mesh, outdir, tag)
    elif part == "imas":
        _child_imas(bq, np, res, omas, mesh, outdir, tag)
    else:
        raise SystemExit(f"unknown part {part!r}")
    with open(os.path.join(outdir, f"{tag}_{part}.json"), "w") as fh:
        json.dump(res, fh)


def _child_rec(bq, np, res, geq, pf, mesh, outdir, tag):
    b = bq.Bouquet.from_geqdsk(geq, profiles=pf, mesh=mesh, nthreads=1,
                               header=os.path.join(outdir, f"{tag}_rec"),
                               n_draws=2, reconstruction_engine="legacy")
    b.config.generation.jbs_self_consistent = False
    b.config.generation.seed = 20260929
    b.setup_solver()
    bl = b.prepare_baseline()
    res["recon_baseline"] = {k: np.asarray(getattr(bl, k), float).tolist()
                             for k in ("j_phi", "j_inductive", "j_BS")}
    res["recon_l_i_target"] = float(bl.l_i_target)
    s0 = b.verify_sigma0_consistency()
    res["recon_sigma0"] = {k: float(s0[k]) for k in (
        "max_dev", "rms_dev", "max_dev_frac") if k in s0}
    b.generate()


def _child_imas(bq, np, res, omas, mesh, outdir, tag):
    bi = bq.Bouquet.from_imas(omas, mesh=mesh, time=2.3043, n_draws=1,
                              nthreads=1,
                              header=os.path.join(outdir, f"{tag}_imas"),
                              reconstruction_engine="legacy")
    bi.config.generation.jbs_self_consistent = False
    bi.config.generation.seed = 20260929
    bi.setup_solver()
    bli = bi.prepare_baseline()
    res["imas_baseline"] = {k: np.asarray(getattr(bli, k), float).tolist()
                            for k in ("j_phi", "j_inductive", "j_BS")}
    res["imas_l_i_target"] = float(bli.l_i_target)
    bi.generate()


def _h5_diff(a, b):
    import h5py
    import numpy as np
    diffs = []
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        def _walk(name, oa):
            if name not in fb:
                diffs.append(f"missing in after: {name}")
                return
            ob = fb[name]
            for k in oa.attrs:
                if any(s in k for s in _SKIP_ATTRS):
                    continue
                va, vb = oa.attrs[k], ob.attrs.get(k)
                try:
                    same = np.array_equal(np.asarray(va), np.asarray(vb))
                except Exception:
                    same = (str(va) == str(vb))
                if not same:
                    diffs.append(f"attr {name}@{k}")
            for k in ob.attrs:
                if k not in oa.attrs and not any(s in k for s in _SKIP_ATTRS):
                    diffs.append(f"attr only in after: {name}@{k}")
            if isinstance(oa, h5py.Dataset):
                if not np.array_equal(np.asarray(oa[()]), np.asarray(ob[()])):
                    diffs.append(f"dataset {name}")
        fa.visititems(_walk)

        def _extra(name, ob):
            if name not in fa:
                diffs.append(f"only in after: {name}")
        fb.visititems(_extra)
    return diffs


def main(before, after, outdir):
    os.makedirs(outdir, exist_ok=True)
    for tag, tree in (("before", before), ("after", after)):
        for part in ("rec", "imas"):
            env = dict(os.environ, PYTHONPATH=tree, OMP_NUM_THREADS="1",
                       MPLBACKEND="Agg")
            p = subprocess.run([sys.executable, os.path.abspath(__file__),
                                "--child", tree, outdir, tag, part], env=env,
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit(f"{tag}/{part} run failed:\n"
                                 f"{p.stderr[-4000:]}")
    out = {"diffs": {}}
    ja, jb = {}, {}
    for part in ("rec", "imas"):
        with open(os.path.join(outdir, f"before_{part}.json")) as fa, \
                open(os.path.join(outdir, f"after_{part}.json")) as fb:
            ja.update(json.load(fa))
            jb.update(json.load(fb))
    out["diffs"]["in_memory"] = sorted(
        k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    for h in ("rec", "imas"):
        out["diffs"][h] = _h5_diff(os.path.join(outdir, f"before_{h}.h5"),
                                   os.path.join(outdir, f"after_{h}.h5"))
    out["bit_identical"] = not any(out["diffs"].values())
    with open(os.path.join(outdir, "legacy_ab.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out["bit_identical"] else 1


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        sys.exit(main(*sys.argv[1:4]))
