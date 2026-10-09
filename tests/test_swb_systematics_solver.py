"""swb draws (``solve_method="swb"``): systematics and bias, live solver.

One swb baseline (solve B) on the synthetic IMAS slice, then four generations:

  * sigma=0: after a perturbed draw, a draw with every perturbation zero
    reproduces solve B to the bit;
  * pressure-only (``sigma_jphi`` zero: the inductive seed unperturbed):
    boundary and coils bounded, no signed boundary bias;
  * production: ensemble means of l_i, q0, the boundary and the kinetic
    profiles agree with the baseline, Ip of every draw at target;
  * seeded: the same seed reproduces every archived dataset, another differs.

Every solve runs in one subprocess (tests/_harness.py), which writes every
measured number to a JSON, pass or fail.  Marked ``solver``.
"""
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")
_DD = os.path.join(_DATA, "dd_synthetic.json.gz")
_IDA = os.path.join(_DATA, "diiid_profs_synthetic.cdf")
_GEQ = os.path.join(_DATA, "g_synthetic.geqdsk")
_MESH = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like", "DIIID_mesh.h5"))
_TIME = 1.013
_SEED, _SEED_B = 20261008, 7
_N_P, _N, _N_R = 10, 16, 3            # pressure-only, production, each seeded twin
_KIN_NODES = (0.1, 0.5, 0.9)
_KIN = ("n_e", "T_e", "T_i", "n_i", "aux_zeff")

_S0_REL = 1e-9            # pressure-only inductive seed vs solve B: measured 0
_P_LCFS_MM = 1.5          # pressure-only boundary RMS per draw: measured <= 0.64 mm
_P_COIL_PCT = 4.0         # pressure-only coil change per draw: measured <= 2.1 %
_BIAS_MM = 2.0            # |signed-mean| boundary shift (test_systematics mode 2): measured <= 0.21
_K = 4.0                  # |mean - base| < _K std/sqrt(N), ~25 checks on t15 tails: measured <= 1.9
_IP_REL = 1e-4            # every draw's |Ip/Ip_target - 1|: measured <= 8e-6

pytestmark = pytest.mark.solver


def _swb_ready():
    p = os.environ.get("OFT_PYTHONPATH")
    if p and os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)
    try:
        from bouquet.coords import _swb_grid_arg, _swb_params
        return bool(_swb_grid_arg()) and {"jphi_fixed", "p_fixed"} <= _swb_params()
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  the probe (subprocess)
# ---------------------------------------------------------------------------
def _poly(pts):
    """Area, perimeter and centroid (R, Z) of a closed trace."""
    x, y = np.asarray(pts, float).T
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    c = x * yn - xn * y
    a = 0.5 * c.sum()
    return (abs(a), float(np.hypot(xn - x, yn - y).sum()),
            float(((x + xn) * c).sum() / (6 * a)), float(((y + yn) * c).sum() / (6 * a)))


def _sha(v):
    v = np.asarray(v)
    return hashlib.sha256(repr(v.tolist()).encode() if v.dtype == object
                          else v.tobytes()).hexdigest()


def _group(g, base, nodes):
    """The measured numbers of one archived group (a draw, or the baseline)."""
    from scipy.spatial import cKDTree
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.utils import _read_coil_names, read_eqdsk_from_bytes
    eq = read_eqdsk_from_bytes(g["eqdsk"][()].tobytes(), read_geqdsk)
    pk, pe = g["psi_N_kinetic"][()], g["psi_N"][()]
    seed = g["j_inductive"][()] / float(g.attrs["swb_alpha"])
    out = dict(q0=float(eq.qpsi[0]), Ip=float(abs(eq.Ip)),
               coils=dict(zip(_read_coil_names(g), map(float, g["coil_currents"][()]))),
               **{f"{k}@{x}": float(np.interp(x, pk, g[k][()]))
                  for k in _KIN if k in g for x in nodes},
               **{f"j_seed@{x}": float(np.interp(x, pe, seed)) for x in nodes})
    if base is None:
        return out
    lcfs, ref = g["perturbed_lcfs_ref"][()], base["lcfs"]
    d, _ = cKDTree(lcfs).query(ref)
    a, _, rc, zc = _poly(lcfs)
    a0, per0, rc0, zc0 = _poly(ref)
    out.update(li3=float(g.attrs["l_i(3)"]), lcfs_rms_mm=float(np.sqrt((d ** 2).mean()) * 1e3),
               dn_mm=1e3 * (a - a0) / per0, dR_mm=1e3 * (rc - rc0), dZ_mm=1e3 * (zc - zc0),
               coil_pct=max(100 * abs(out["coils"][c] - v) / max(abs(v), 1.0)
                            for c, v in base["coils"].items()),
               resamples=int(g.attrs["swb_jind_resamples"]))
    return out


def _probe(work):
    import warnings
    import h5py
    import bouquet as bq
    import bouquet.baseline as bqb
    from bouquet.run import _zero_perturbation_env
    from bouquet.swb_draws import SwbDraws
    _harness.assert_bouquet_is_repo_local()
    res = {"errors": {}}
    path = os.path.join(work, "swb_systematics.json")

    def dump():
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)

    ddp = os.path.join(work, "dd.json")
    with gzip.open(_DD, "rb") as fi, open(ddp, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    run = bq.Bouquet.from_imas(ddp, mesh=_MESH, time=_TIME, n_draws=1, ida_path=_IDA,
                               LCFS_geqdsk=_GEQ, impurity_Z=6.0, solve_method="swb",
                               header=os.path.join(work, "bl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run.prepare()
    sb = run.baseline.swb_baseline
    base = None

    def generate(tag, n, seed, env_fn=None, draw_fn=None):
        nonlocal base
        run.config.generation.seed, run.config.output_header = seed, os.path.join(work, tag)
        orig, orig_draw = bqb.resolve_uncertainty, SwbDraws.draw
        if env_fn is not None:
            bqb.resolve_uncertainty = lambda c, b: env_fn(orig(c, b))
        if draw_fn is not None:
            SwbDraws.draw = draw_fn(orig_draw)
        try:
            run.generate(n=n)
        finally:
            bqb.resolve_uncertainty, SwbDraws.draw = orig, orig_draw
        rows, hashes = [], {}
        with h5py.File(run.config.output_header + ".h5", "r") as hf:
            sc = hf["scan/0"]
            if base is None:
                b = sc["_baseline"]
                base = dict(_group(b, None, _KIN_NODES), lcfs=b["recon_lcfs_ref"][()])
                res["baseline"] = dict({k: v for k, v in base.items() if k != "lcfs"},
                                       li3=float(sb["li_3"]), Ip_live=abs(float(sb["Ip"])),
                                       Ip_target=float(run.baseline.Ip_target))
            for k in sorted((k for k in sc if k.isdigit()), key=int):
                rows.append(_group(sc[k], base, _KIN_NODES))
                hashes[k] = {n_: _sha(sc[k][n_][()]) for n_ in sc[k]
                             if isinstance(sc[k][n_], h5py.Dataset)}
        res[tag] = dict(draws=rows, hashes=hashes, n_requested=n,
                        rejections=[r.get("reason") for r in run.draw_rejections])
        dump()

    def zero_second(draw):
        """``draw`` with every perturbation of the second draw zero."""
        calls = []

        def wrapped(self, *a, inputs=None, **kw):
            calls.append(1)
            if len(calls) == 2:
                self = copy.copy(self)
                self.sigma_jind = np.zeros_like(self.sigma_jind)
                inputs = _zero_perturbation_env(inputs)
            return draw(self, *a, inputs=inputs, **kw)
        return wrapped

    no_jind = lambda e: dict(e, sigma_jphi=np.zeros_like(e["sigma_jphi"]))
    for args in (("sigma0", 2, _SEED, None, zero_second),
                 ("pressure", _N_P, _SEED, no_jind),
                 ("production", _N, _SEED), ("twin_a", _N_R, _SEED),
                 ("twin_b", _N_R, _SEED), ("twin_c", _N_R, _SEED_B)):
        try:
            generate(*args)
        except Exception:
            res["errors"][args[0]] = traceback.format_exc()[-3000:]
            dump()
    res["solve_B_coils"] = sb["coils"]
    dump()


@pytest.fixture(scope="module")
def swb_runs(tmp_path_factory):
    if not (all(os.path.isfile(p) for p in (_DD, _IDA, _GEQ, _MESH)) and _swb_ready()):
        pytest.skip("needs OFT with solve_with_bootstrap(x, jphi_fixed, p_fixed)")
    work = str(tmp_path_factory.mktemp("swb_sys"))
    env = _harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    pr = subprocess.run([sys.executable, os.path.abspath(__file__), work], env=env,
                        capture_output=True, text=True)
    path = os.path.join(work, "swb_systematics.json")
    if not os.path.isfile(path) or "baseline" not in json.load(open(path)):
        pytest.fail(f"swb probe failed (rc={pr.returncode}):\n{pr.stderr[-4000:]}")
    out = json.load(open(path))
    print("[swb-sys] " + json.dumps({k: v for k, v in out.items() if k[:4] != "twin"}))
    return out


def _draws(r, tag, n):
    assert tag not in r["errors"], r["errors"][tag]
    d = r[tag]["draws"]
    assert len(d) == n and not r[tag]["rejections"], (len(d), r[tag]["rejections"])
    return d


def _bias(draws, base, key):
    """(mean - baseline, std/sqrt(N)) of ``key`` over ``draws``."""
    x = np.array([d[key] for d in draws])
    m = x.mean() - (0.0 if base is None else base[key])
    return float(m), float(x.std(ddof=1) / np.sqrt(x.size))


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------
def test_sigma0_draw_after_a_draw_reproduces_solve_b(swb_runs):
    """Draw 2 (every perturbation zero) after a perturbed draw 1: solve B, bit for bit."""
    b = swb_runs["baseline"]
    d1, d = _draws(swb_runs, "sigma0", 2)
    assert d1["lcfs_rms_mm"] > 0
    assert all(d["coils"][c] == v for c, v in swb_runs["solve_B_coils"].items())
    assert d["lcfs_rms_mm"] == 0
    for k in ("li3", "q0", "Ip") + tuple(f"{c}@{x}" for c in _KIN[:4] + ("j_seed",)
                                         for x in _KIN_NODES):
        assert d[k] == b[k], (k, d[k], b[k])


def test_pressure_only_bounded_and_unbiased(swb_runs):
    b = swb_runs["baseline"]
    dr = _draws(swb_runs, "pressure", _N_P)
    for d in dr:
        assert all(abs(d[f"j_seed@{x}"] / b[f"j_seed@{x}"] - 1) < _S0_REL for x in _KIN_NODES)
    assert max(d["lcfs_rms_mm"] for d in dr) < _P_LCFS_MM
    assert max(d["coil_pct"] for d in dr) < _P_COIL_PCT
    for k in ("dn_mm", "dR_mm", "dZ_mm"):
        m, se = _bias(dr, None, k)
        assert abs(m) < min(_BIAS_MM, _K * se), (k, m, se)
    for k in ("li3", "q0"):
        m, se = _bias(dr, b, k)
        assert abs(m) < _K * se, (k, m, se, b[k])


def test_production_ensemble_unbiased(swb_runs):
    b = swb_runs["baseline"]
    dr = _draws(swb_runs, "production", _N)
    for d in dr:
        assert abs(d["Ip"] / b["Ip_target"] - 1) < _IP_REL, d["Ip"]
    for k in ("dn_mm", "dR_mm", "dZ_mm"):
        m, se = _bias(dr, None, k)
        assert abs(m) < min(_BIAS_MM, _K * se), (k, m, se)
    for k in ("li3", "q0") + tuple(f"{c}@{x}" for c in _KIN + ("j_seed",)
                                   for x in _KIN_NODES):
        m, se = _bias(dr, b, k)
        assert abs(m) < _K * se, (k, m, se, b[k])


def test_seeded_generations_reproduce(swb_runs):
    a, b, c = (swb_runs[t]["hashes"] for t in ("twin_a", "twin_b", "twin_c"))
    for t in ("twin_a", "twin_b", "twin_c"):
        _draws(swb_runs, t, _N_R)
    assert a == b
    for k in a:
        assert all(a[k][n] != c[k][n] for n in ("n_e", "T_e", "j_phi", "coil_currents")), k


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _swb_ready()
    _probe(sys.argv[1])
