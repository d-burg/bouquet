"""The ONE reconstruction state and the zero-perturbation draw identity --
live-solver half (``pytest -m solver``).

The design rule: the bouquet reconstruction is one equilibrium F (allowed to
differ from its input g-file / modelling-source IDS; on the g-file path it
keeps its designed l_i match to the input).  Everything recorded about the
baseline is F -- the saved baseline g-file, ``l_i_target``, the recorded q0
/ q95, the archived profiles -- and a draw with every perturbation at zero
reproduces F at the self-consistent loop's own tolerances
(``r_j <= jbs_rtol_j``, ``r_I <= jbs_rtol_Ip``, ``|dl_i| <= jbs_tol_li``),
on every draw route.  The stage-wise fast half (toy solver) is
``tests/test_sigma0_identity_stages.py``.

Runs on the synthetic D3D-like example only (the g-file + p-file, and the
OMAS data dictionary); no device data.  Every solver call is in a subprocess
probe (``OFT_env`` is a per-process singleton; see ``tests/_harness.py``).
Each probe writes EVERY measured number to its JSON, pass or fail, and does
so even when a stage raises (the error is recorded in place of the numbers).
"""
import json
import os
import subprocess
import sys
import traceback

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_OMAS = os.path.join(_EXAMPLE, "D3Dlike_baseline_omas.json")
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_TIME = 2.3043
_files_ok = all(os.path.isfile(p) for p in (_OMAS, _GEQ, _PF, _MESH))

#: the loop's own tolerances (GenerationConfig defaults) -- read, not tuned
_S = dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3, tol_q0=2e-3)
#: route R2's approved sigma=0 budget on |s-1|*f_ind (see
#: tests/test_seeded_reproducibility._S_FIND_ATOL_EXACT) -- unchanged
_S_FIND_ATOL_EXACT = 3.86e-3


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


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")


# ---------------------------------------------------------------------------
#  the probe (one interpreter per example)
# ---------------------------------------------------------------------------
def _gfile_numbers(path_or_bytes):
    """li(2) (the g-file's l_i(3) functional), q0, q95 and the FSA current
    of a g-file, read with bouquet's own reader."""
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.utils import read_eqdsk_from_bytes
    eq = (read_eqdsk_from_bytes(path_or_bytes, read_geqdsk)
          if isinstance(path_or_bytes, (bytes, bytearray))
          else read_geqdsk(path_or_bytes))
    psi = np.asarray(eq.psi_N, dtype=float)
    q = np.asarray(eq.qpsi, dtype=float)
    return dict(li2=float(eq.li.get("li(2)", float("nan"))),
                q0=float(q[0]), q95=float(np.interp(0.95, psi, q)),
                psi_N=psi.tolist(),
                jtor=np.abs(np.asarray(eq.j_tor_averaged_direct,
                                       dtype=float)).tolist(),
                Ip=float(abs(eq.Ip)))


def _live(mygs, psi_pad):
    st = mygs.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
    return dict(l_i=float(st["l_i"]), q0=float(st.get("q_0", np.nan)),
                q95=float(st.get("q_95", np.nan)), Ip=float(st["Ip"]))


def _jsonable(o):
    from bouquet.jbs_loop import jsonable
    o = jsonable(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()
                if k not in ("spike0",)}
    return o


def _probe(outdir, which):
    """``which``: "recon" (g-file + p-file) or "imas" (OMAS dd)."""
    import h5py
    import bouquet as bq
    from bouquet.utils import _baseline_group_path, DELIVERED_STATE_ATTR

    _harness.assert_bouquet_is_repo_local()
    out = dict(which=which, stages={})
    path = os.path.join(outdir, f"identity_{which}.json")

    def _stage(name, fn):
        try:
            out[name] = fn()
            out["stages"][name] = "ok"
        except Exception as e:                     # recorded, never lost
            out["stages"][name] = (f"{type(e).__name__}: {e}\n"
                                   + traceback.format_exc()[-3000:])

    try:
        if which == "recon":
            b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH,
                                       nthreads=1, n_draws=1,
                                       header=os.path.join(outdir, "rec"),
                                       reconstruction_engine="legacy")
            psi_pad = float(b.config.source.psi_pad)
        else:
            b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME,
                                     n_draws=1, nthreads=1,
                                     header=os.path.join(outdir, "imas"),
                                     reconstruction_engine="legacy")
            psi_pad = 1e-3
        g = b.config.generation
        g.seed = 12345
        assert g.jbs_self_consistent, "the loop is the default"
        b.setup_solver()
        bl = b.prepare_baseline()
        mygs = b.mygs
        ds = dict(bl.delivered_state or {})

        # ---- F as the solver holds it after the reconstruction ------------
        def _F():
            live = _live(mygs, psi_pad)
            return dict(
                live=live, l_i_target=float(bl.l_i_target),
                recorded=dict(l_i=ds.get("l_i"), q0=ds.get("q0"),
                              q95=ds.get("q95")),
                request_normalisation=ds.get("request_normalisation"),
                achieved_normalisation=ds.get("achieved_normalisation"),
                n_floored_inductive=ds.get("n_floored_inductive"),
                n_floored_target_inductive=ds.get(
                    "n_floored_target_inductive"),
                n_negative_inductive=ds.get("n_negative_inductive"),
                li_corrective_state=ds.get("li_corrective_state"),
                li_step6_matched=ds.get("li_step6_matched"),
                li_input=ds.get("li_input"),
                has_offset=bl.jphi_request_offset is not None,
                reconstruction_metrics=(
                    None if bl.reconstruction_metrics is None else
                    {k: v for k, v in bl.reconstruction_metrics.items()
                     if k != "jbs_loop"}),
                j_phi_achieved=np.asarray(ds.get("j_phi_achieved"),
                                          float).tolist()
                if ds.get("j_phi_achieved") is not None else None,
                psi_N=np.asarray(bl.psi_N, float).tolist())
        _stage("F", _F)
        snapF = mygs.copy_eq()

        # ---- F's own g-file, saved here (the like-for-like reference) ------
        def _F_gfile():
            from bouquet.utils import safe_save_eqdsk
            p = os.path.join(outdir, f"F_{which}.geqdsk")
            safe_save_eqdsk(mygs, p, nr=257, nz=257, truncate_eq=True,
                            lcfs_pad=psi_pad)
            return _gfile_numbers(p)
        _stage("F_gfile", _F_gfile)
        mygs.replace_eq(source_eq=snapF)

        # ---- the zero-perturbation identity, EVERY route -----------------
        def _s0():
            s0 = b.verify_sigma0_consistency(
                draw_routes=("standard", "ip_renorm"))
            return _jsonable(s0)
        _stage("sigma0", _s0)
        mygs.replace_eq(source_eq=snapF)

        # ---- the archive: the saved baseline g-file and recorded values ----
        def _gen():
            b.generate(n=1)
            h5 = b.config.output_header + ".h5"
            gp = _baseline_group_path(g.scan_key)
            with h5py.File(h5, "r") as hf:
                grp = hf[gp]
                attrs = {k: (grp.attrs[k].decode() if isinstance(
                    grp.attrs[k], bytes) else grp.attrs[k])
                    for k in ("l_i_target",) if k in grp.attrs}
                rec = (json.loads(grp.attrs[DELIVERED_STATE_ATTR])
                       if DELIVERED_STATE_ATTR in grp.attrs else None)
                eqb = bytes(grp["eqdsk"][()]) if "eqdsk" in grp else None
                jphi = (np.asarray(grp["j_phi"][()], float).tolist()
                        if "j_phi" in grp else None)
            return dict(attrs={k: float(v) for k, v in attrs.items()},
                        delivered_state_record=rec,
                        saved_gfile=(None if eqb is None
                                     else _gfile_numbers(eqb)),
                        archived_j_phi=jphi)
        _stage("archive", _gen)
    except Exception as e:
        out["fatal"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    finally:
        with open(path, "w") as fh:
            json.dump(out, fh, default=float)


def _run(tmp_path_factory, which):
    work = tmp_path_factory.mktemp(f"identity_{which}")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), which, str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    p = work / f"identity_{which}.json"
    if not p.exists():
        pytest.fail(f"{which} identity probe wrote no JSON (rc="
                    f"{proc.returncode}):\n{proc.stdout[-3000:]}\n"
                    f"{proc.stderr[-4000:]}")
    with open(str(p)) as fh:
        d = json.load(fh)
    d["_rc"] = proc.returncode
    return d


@pytest.fixture(scope="module")
def recon(tmp_path_factory):
    return _run(tmp_path_factory, "recon")


@pytest.fixture(scope="module")
def imas(tmp_path_factory):
    return _run(tmp_path_factory, "imas")


def _need(d, stage):
    st = d["stages"].get(stage)
    assert st == "ok", f"{d['which']} stage {stage!r} failed: {st}" + (
        f"\nfatal: {d.get('fatal')}" if d.get("fatal") else "")
    return d[stage]


def _cur_rj(a, b, psi):
    """current-weighted-free relative L2 on the common grid (the g-file's)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ---------------------------------------------------------------------------
#  1. the unperturbed draw reproduces the reconstruction, every route
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("fix", ["recon", "imas"])
@pytest.mark.parametrize("route", ["standard", "ip_renorm"])
def test_the_unperturbed_draw_reproduces_the_reconstruction(request, fix,
                                                            route):
    d = request.getfixturevalue(fix)
    s0 = _need(d, "sigma0")
    rr = s0["draw_route"]["routes"][route]
    assert rr.get("error") is None, rr
    assert rr["loop_converged"], rr
    assert rr["r_j"] <= _S["rtol_j"], rr
    assert rr["r_I"] <= _S["rtol_Ip"], rr
    assert rr["dl_i_vs_l_i_target"] <= _S["tol_li"], rr
    # reported beside them (no bar of their own in the loop): present
    for k in ("dq0_vs_reference", "dq95_vs_reference", "j_phi_vs_reference"):
        assert k in rr, (route, k)
    if route == "ip_renorm":
        # R2's own approved budget, unchanged
        assert abs(rr["r2_ip_scale"] - 1.0) * float(rr["r2_f_ind"]) \
            <= _S_FIND_ATOL_EXACT, rr


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("fix", ["recon", "imas"])
def test_the_sigma0_check_passes_on_the_draw_route_verdict(request, fix):
    """``passed`` now REQUIRES every route to reproduce the reconstruction;
    the "baseline's way" loop result is kept beside it under its own name."""
    s0 = _need(request.getfixturevalue(fix), "sigma0")
    assert "passed_baseline_way" in s0
    assert s0["passed"] == s0["draw_route"]["passed_draw_route"]
    assert s0["passed"], s0.get("passed_reason")


# ---------------------------------------------------------------------------
#  2. ONE state: recorded values, the saved baseline g-file, the archive
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("fix", ["recon", "imas"])
def test_the_recorded_baseline_metrics_are_the_delivered_state(request, fix):
    F = _need(request.getfixturevalue(fix), "F")
    # l_i_target IS the l_i of the state the reconstruction leaves (read
    # again here after the storage step's geometry queries: agreement at the
    # loop's tolerances; the probe records the exact differences)
    assert abs(F["l_i_target"] - F["live"]["l_i"]) <= _S["tol_li"], F
    assert F["recorded"]["l_i"] == F["l_i_target"], F
    assert abs(F["recorded"]["q0"] - F["live"]["q0"]) <= _S["tol_q0"], F
    assert abs(F["recorded"]["q95"] - F["live"]["q95"]) <= _S["tol_q0"], F
    assert F["has_offset"], F
    if fix == "recon":
        m = F["reconstruction_metrics"]
        assert m["li"] == F["l_i_target"]
        assert m["li_realized_post_corrective"] == F["l_i_target"]
        # the designed l_i match to the INPUT g-file is kept
        assert abs(F["l_i_target"] - F["li_input"]) < 1e-3, F


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("fix", ["recon", "imas"])
def test_the_saved_baseline_gfile_is_the_delivered_state(request, fix):
    """Re-read the baseline g-file ``generate()`` archived and compare it
    like for like (the same reader, the same estimators) with a g-file saved
    from the delivered state itself; and the archive's recorded values with
    the reconstruction's."""
    d = request.getfixturevalue(fix)
    F = _need(d, "F")
    Fg = _need(d, "F_gfile")
    a = _need(d, "archive")
    assert a["attrs"]["l_i_target"] == F["l_i_target"], a["attrs"]
    rec = a["delivered_state_record"]
    assert rec is not None and rec["l_i_target"] == F["l_i_target"], rec
    br = rec["baseline_resolve"]
    assert br is not None, rec
    # the run's baseline re-solve (the saved g-file, every draw's warm
    # start) is the delivered state
    assert abs(br["l_i"] - F["l_i_target"]) <= _S["tol_li"], br
    assert abs(br["q0"] - F["live"]["q0"]) <= _S["tol_q0"], br
    assert abs(br["q95"] - F["live"]["q95"]) <= _S["tol_q0"], br
    sg = a["saved_gfile"]
    assert abs(sg["li2"] - Fg["li2"]) <= _S["tol_li"], (sg["li2"], Fg["li2"])
    assert abs(sg["q0"] - Fg["q0"]) <= _S["tol_q0"], (sg["q0"], Fg["q0"])
    assert abs(sg["q95"] - Fg["q95"]) <= _S["tol_q0"], (sg["q95"], Fg["q95"])
    assert _cur_rj(sg["jtor"], Fg["jtor"], Fg["psi_N"]) <= _S["rtol_j"]
    # the archived baseline current is the delivered state's achieved one
    assert _cur_rj(a["archived_j_phi"], F["j_phi_achieved"],
                   F["psi_N"]) <= _S["rtol_j"]


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[2], sys.argv[1])
