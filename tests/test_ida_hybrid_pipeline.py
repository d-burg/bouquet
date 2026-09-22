"""The ida_hybrid kinetic path end to end: from_imas -> prepare -> generate -> archive.

The reader-level invariants (charge balance, Z_eff convention, pressure model,
beam subtraction, the pressure anchor, the psi_N(rho) guard) are unit-tested
in test_ni_fast_subtraction.py, test_fast_ion_dilution.py and
test_psi_rho_drift.py; the per-draw ni/Z_eff pairing, up to the solve, in
test_ida_ni_zeff_pair.py.  What only a live run can show is that those hold
through prepare(), the draws and the archive:

  * prepare() keeps the reader's IDA kinetics and splits j_phi exactly;
  * the sigma=0 draw reproduces the baseline bootstrap on this path;
  * every archived draw is one consistent (ne, ni, nz, Z_eff) set, stays in the
    single-impurity window, and its archived pressure is the solve pressure
    (the impurity on ne - z_fast).

Inputs: the D3D-like example dd (examples/D3D-like) with a beam density added to
D, plus a synthetic IDA .cdf of the same plasma whose two ni routes agree.
The solver runs in a subprocess (tests/_harness.py).  Marked ``solver``.
"""
import json
import os
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_DD = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_TIME = 2.2
_Z = 6.0
_FAST = 0.15          # peak beam fraction of the D density
_N_DRAWS = 3
_SEED = 20260921


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(
        not (all(os.path.isfile(p) for p in (_DD, _GEQ, _MESH)) and _oft_importable()),
        reason="needs OFT + the D3D-like example dd/mesh"),
]


# ---------------------------------------------------------------------------
#  inputs: a beam on the example dd, and the IDA fit of the same plasma
# ---------------------------------------------------------------------------
def _write_inputs(work):
    """dd with density_fast on D (total D held, as FUSE carves it) + IDA .cdf."""
    with open(_DD) as fh:
        dd = json.load(fh)
    for p in dd["core_profiles"]["profiles_1d"]:
        psi = np.asarray(p["grid"]["psi"], float)
        pn = (psi - psi[0]) / (psi[-1] - psi[0])
        d = next(i for i in p["ion"] if i["label"] == "D")
        n_tot = np.asarray(d["density_thermal"], float)
        nf = _FAST * n_tot * (1.0 - pn ** 2) ** 2
        d["density_thermal"], d["density_fast"] = (n_tot - nf).tolist(), nf.tolist()
    ddp = os.path.join(work, "dd.json")
    with open(ddp, "w") as fh:
        json.dump(dd, fh)

    ic = int(np.argmin(np.abs(np.asarray(dd["core_profiles"]["time"]) - _TIME)))
    p = dd["core_profiles"]["profiles_1d"][ic]
    psi = np.asarray(p["grid"]["psi"], float)
    pn = (psi - psi[0]) / (psi[-1] - psi[0])
    ne = np.asarray(p["electrons"]["density_thermal"], float)
    te = np.asarray(p["electrons"]["temperature"], float)
    d = next(i for i in p["ion"] if i["label"] == "D")
    c = next(i for i in p["ion"] if i["label"] != "D")
    ni_tot = np.asarray(d["density_thermal"], float) + np.asarray(d["density_fast"], float)
    nc = np.asarray(c["density_thermal"], float)
    ti = np.asarray(c["temperature"], float)
    zeff = (ni_tot + _Z ** 2 * nc) / ne     # measured: counts the beam; both routes agree
    cdf = os.path.join(work, "ida.cdf")
    with h5py.File(cdf, "w") as f:
        f["time"] = np.array([1e3 * _TIME])
        f["psi_n"] = pn
        for k, v, e in (("n_e", ne, 0.04), ("T_e", te, 0.05), ("T_12C6", ti, 0.06),
                        ("Zeff", zeff, 0.08), ("n_12C6", nc, 0.15)):
            f[k] = v[None, :]
            f[k + "_err"] = (e * v)[None, :]
    return ddp, cdf


# ---------------------------------------------------------------------------
#  the probe (subprocess): everything that needs the solver
# ---------------------------------------------------------------------------
def _probe(work):
    import warnings
    import bouquet as bq
    from bouquet.baseline import resolve_uncertainty

    ddp, cdf = _write_inputs(work)
    src = bq.ImasSource(ids_path=ddp, time=_TIME, ida_path=cdf, impurity_Z=_Z,
                        LCFS_geqdsk=_GEQ)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl_reader = bq.read_imas_baseline(src, kinetic_source="ida_hybrid")
    header = os.path.join(work, "run")
    run = bq.Bouquet.from_imas(ddp, mesh=_MESH, time=_TIME, n_draws=_N_DRAWS,
                               header=header, ida_path=cdf, LCFS_geqdsk=_GEQ,
                               impurity_Z=_Z)
    run.config.generation.seed = _SEED
    run.prepare()
    rb = run.baseline
    env = resolve_uncertainty(run.config, rb)
    if not hasattr(run.source, "psi_pad"):
        run.source.psi_pad = 1e-3
    s0 = run.verify_sigma0_consistency(tol_frac=0.02)
    run.generate()

    out = {"sigma0_dev": float(s0["max_dev_frac"]), "sigma0_passed": bool(s0["passed"]),
           "ni_from_zeff": bool(env["ni_from_zeff"]),
           "zeff_dne_set": env.get("zeff_dne") is not None,
           "Z_imp": float(rb.Z_imp), "zeff_includes_fast": bool(rb.zeff_includes_fast)}
    arr = {"env_sigma_ni": env["sigma_ni"]}
    for f in ("ne", "te", "ni", "ti", "Zeff", "z_fast", "z2_fast", "p_fast"):
        arr[f"reader_{f}"] = getattr(bl_reader, f)
        arr[f"run_{f}"] = getattr(rb, f)
    for f in ("j_phi", "j_inductive", "j_BS", "j_NBI", "j_RF", "psi_N", "psi_N_kinetic"):
        arr[f"run_{f}"] = getattr(rb, f)
    if getattr(rb, "jBS_diff", None) is not None:
        arr["run_jBS_diff"] = rb.jBS_diff
    np.savez(os.path.join(work, "probe.npz"),
             **{k: np.asarray(v, float) for k, v in arr.items() if v is not None})
    with open(os.path.join(work, "probe.json"), "w") as fh:
        json.dump(dict(out, h5=header + ".h5"), fh)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("ida_hybrid"))
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), work],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"ida_hybrid probe failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(os.path.join(work, "probe.json")) as fh:
        meta = json.load(fh)
    return meta, dict(np.load(os.path.join(work, "probe.npz")))


def _draws(meta):
    import bouquet as bq
    sc = bq.BouquetArchive(meta["h5"]).scan()
    return sc, sc.all


# ---------------------------------------------------------------------------
#  prepare()
# ---------------------------------------------------------------------------
class TestPrepared:
    @pytest.mark.parametrize("f", ["ne", "te", "ni", "ti", "Zeff", "z_fast", "z2_fast", "p_fast"])
    def test_prepare_keeps_the_readers_ida_kinetics(self, run, f):
        _, a = run
        np.testing.assert_array_equal(a[f"run_{f}"], a[f"reader_{f}"])

    def test_the_split_sums_to_j_phi(self, run):
        _, a = run
        tot = a["run_j_inductive"] + a["run_j_BS"] + a["run_j_NBI"] + a["run_j_RF"]
        tot = tot + a.get("run_jBS_diff", 0.0)
        np.testing.assert_allclose(tot, a["run_j_phi"], rtol=0,
                                   atol=1e-9 * np.max(np.abs(a["run_j_phi"])))

    def test_the_draws_derive_ni_from_the_ida_pair(self, run):
        meta, _ = run
        assert meta["ni_from_zeff"] and meta["zeff_dne_set"]
        assert meta["zeff_includes_fast"] and meta["Z_imp"] == _Z

    def test_sigma0_reproduces_the_baseline_bootstrap(self, run):
        meta, _ = run
        assert meta["sigma0_passed"], f"max dev {100 * meta['sigma0_dev']:.2f}% of peak"


# ---------------------------------------------------------------------------
#  the archive
# ---------------------------------------------------------------------------
def _k2e(prof):
    from bouquet.utils import pchip_interp
    pe = np.asarray(prof["psi_N"], float)
    pk = np.asarray(prof["psi_N_kinetic"], float)
    return lambda x: pchip_interp(pk, np.asarray(x, float), pe)  # noqa: E731


class TestArchive:
    def test_there_are_draws(self, run):
        _, d = _draws(run[0])
        assert len(d) >= 1

    def test_the_baseline_archives_the_envelope_and_the_beam(self, run):
        meta, a = run
        sc, _ = _draws(meta)
        b = sc.baseline
        np.testing.assert_array_equal(b["sigma_ni"], a["env_sigma_ni"])
        np.testing.assert_array_equal(b["z_fast"], a["run_z_fast"])
        assert b["Z_imp"] == _Z

    def test_every_draw_is_one_quasineutral_set(self, run):
        """nz >= 0 and the drawn Z_eff is the draw densities' own."""
        _, draws = _draws(run[0])
        for d in draws:
            p = d.profiles
            ne, ni, zf = p["n_e"], p["n_i"], p["z_fast"]
            conf = np.asarray(p["psi_N_kinetic"]) <= 1.0
            nz = (ne - zf - ni) / _Z
            assert np.all(nz[conf] >= -1e-9 * np.max(nz)), f"draw {d.count}"
            z2 = np.asarray(p.get("z2_fast", zf))          # D beam: z2 = z
            zeff = (ni + _Z ** 2 * nz + z2) / ne
            np.testing.assert_allclose(p["aux_zeff"][conf], zeff[conf], rtol=1e-9,
                                       err_msg=f"draw {d.count}")

    def test_every_draw_stays_in_the_single_impurity_window(self, run):
        from bouquet.physics import zeff_bounds
        _, draws = _draws(run[0])
        for d in draws:
            p = d.profiles
            lo, hi = zeff_bounds(p["n_e"], _Z, p["z_fast"], p.get("z2_fast", p["z_fast"]), True)
            z = p["aux_zeff"]
            assert np.all(z >= lo - 1e-12) and np.all(z <= hi + 1e-12), f"draw {d.count}"

    def test_the_archived_pressure_is_the_solve_pressure(self, run):
        """thermal + impurity on (ne - z_fast) + p_fast (+ p_diff), every draw."""
        from bouquet.physics import impurity_pressure
        meta, a = run
        sc, draws = _draws(meta)
        for d in [sc.baseline] + [x.profiles for x in draws]:
            k2e = _k2e(d)
            p_imp = impurity_pressure(np.maximum(k2e(d["n_e"]) - k2e(d["z_fast"]), 0.0),
                                      k2e(d["n_i"]), k2e(d["T_i"]), _Z)
            expect = d["pressure_thermal"] + p_imp + k2e(a["run_p_fast"])
            if d.get("p_diff") is not None:
                expect = expect + d["p_diff"]
            np.testing.assert_allclose(d["pressure"], expect, rtol=0,
                                       atol=1e-9 * np.max(np.abs(d["pressure"])))


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[1])
