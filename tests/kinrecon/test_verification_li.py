"""The solve verification compares l_i on the solver's ``'iter'`` normalization (``forward.VERIFY_LI``).

Unit (no solver): :func:`.forward.verify_states` compares the axis, q0, the ``'iter'`` l_i and Ip with the unchanged
tolerances; a change of the solver's default l_i alone (the traced-length artifact) no longer fails it; a change of
the ``'iter'`` l_i, q0, Ip or the axis beyond the tolerances still does; a missing ``'iter'`` l_i is a FAIL.

Live (``pytest -m solver``, public synthetic fixture of ``test_fit_solver.py``): on the state of a default-model
consistent solve, (a) the stats' ``l_i_iter`` equals the solver's own ``get_stats(li_normalization='iter')``;
(b) the solver's default l_i, recomputed on the traced surfaces ``psi_N = 1 - pad`` for pads 1e-4 apart, jumps by
more than the verification tolerance between adjacent pads on this unchanged state -- under the old comparison a
solve and its re-solve that land on the two sides of such a jump fail the verification; under the new one they pass;
(c) genuinely different equilibria (a changed current-profile shape at the same Ip target, and a changed Ip target)
still fail it.  ``OFT_env`` is a per-process singleton, so the live work runs in one child interpreter.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from bouquet.kinrecon import forward as FW

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_fit_solver as B  # noqa: E402

BASE = dict(R_axis=1.70, Z_axis=0.01, q_0=1.05, l_i=0.95, l_i_iter=0.80, Ip=1.0e6)


def test_verification_constants_unchanged_and_keys():
    assert FW.VERIFY_AXIS_M == 5.0e-4 and FW.VERIFY_REL == 1.0e-3
    assert FW.VERIFY_LI == "l_i_iter" and FW.VERIFY_KEYS == ("q_0", "l_i_iter", "Ip")


def test_default_li_alone_no_longer_fails():
    v = FW.verify_states(BASE, dict(BASE, l_i=BASE["l_i"] * 1.0055))
    assert v["verdict"] == "PASS" and set(v["rel"]) == {"q_0", "l_i_iter", "Ip"}


@pytest.mark.parametrize("change", [dict(l_i_iter=0.80 * 1.002), dict(q_0=1.05 * 1.002), dict(Ip=1.0e6 * 1.002),
                                    dict(R_axis=1.70 + 6.0e-4)])
def test_a_real_change_still_fails(change):
    assert FW.verify_states(BASE, dict(BASE, **change))["verdict"] == "FAIL"


def test_changes_inside_the_tolerances_pass():
    v = FW.verify_states(BASE, dict(BASE, l_i_iter=0.80 * 1.0009, q_0=1.05 * 1.0009, Ip=1.0e6 * 1.0009,
                                    R_axis=1.70 + 4.0e-4))
    assert v["verdict"] == "PASS"


def test_missing_iter_li_fails():
    a = {k: v for k, v in BASE.items() if k != "l_i_iter"}
    v = FW.verify_states(a, a)
    assert v["verdict"] == "FAIL" and "l_i_iter" in v["reason"]


# ---------------------------------------------------------------------------------------------------------- live
def child(out):
    import math
    from bouquet.kinrecon import current_model as CM
    from bouquet.kinrecon.current import sawtooth as SAW
    rec = {}
    fm, kin, src, cfg_orb, init, ip = B._setup()
    gs = fm.gs
    tr = CM.consistent_solve(fm, kin, src, dict(log_strength=float(math.log(SAW.crash_strength(B._F_TRUE, B._RATE_TRUE))),
                                                f=B._F_TRUE), cfg_orb, label="truth", initial_profiles=init)
    rec["truth"] = tr.status
    snap = fm.snapshot()
    r2 = CM.consistent_solve(fm, kin, src, {}, CM.CurrentModelSettings(r0=cfg_orb.r0), start=snap, label="default")
    rec["default"] = r2.status
    if not r2.converged:
        json.dump(B._jsonable(rec), open(out, "w"))
        return
    s0 = FW._stats(gs)
    st_iter = gs.get_stats(li_normalization="iter")
    rec["l_i_iter"], rec["l_i_iter_oft"], rec["l_i"] = s0.get("l_i_iter"), float(st_iter["l_i"]), s0.get("l_i")
    rec["verification"] = r2.solve.record["policy"].get("verification")
    mu0 = 4.0e-7 * np.pi
    ip_g, _c, vol, _pv, _df, _tf, bp_vol = gs.get_globals()
    scan = []
    for pad in np.linspace(0.004, 0.03, 261):
        try:
            _, _, _, dl, _, _ = gs.get_q(np.r_[1.0 - pad, 0.95, 0.02], compute_geo=True)
            scan.append([float(pad), float((bp_vol / vol) / (mu0 * ip_g / dl) ** 2)])
        except Exception:  # noqa: BLE001 -- a failed trace is left out of the scan
            pass
    rec["scan"] = scan
    asm = r2.assembly
    x, j, p = (np.asarray(asm[k], float) for k in ("x", "jphi", "p"))
    ipv = float(r2.solve.stats["Ip"])
    rec["base_stats"] = {k: s0.get(k) for k in ("R_axis", "Z_axis", "q_0", "l_i", "l_i_iter", "Ip")}
    base_snap = fm.snapshot()
    for name, jj, ipt in (("shape", j * (1.0 + 0.10 * (1.0 - x)), ipv), ("ip", j, ipv * 1.002)):
        fm.restore(base_snap)
        r = fm.solve((x, jj), (x, p), ipt, init="keep", label=f"changed_{name}")
        rec[f"changed_{name}"] = dict(status=r.status,
                                      stats={k: r.stats.get(k) for k in ("R_axis", "Z_axis", "q_0", "l_i", "l_i_iter",
                                                                         "Ip")})
    with open(out, "w") as fh:
        json.dump(B._jsonable(rec), fh, indent=1)


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    d = tmp_path_factory.mktemp("kinrecon_verification_li")
    out = str(d / "result.json")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (B._REPO, B._OFT, os.environ.get("PYTHONPATH", ""))
                                                      if p), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=5400)
    if proc.returncode != 0 or not os.path.isfile(out):
        pytest.fail(f"child failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    with open(out) as fh:
        return json.load(fh)


@pytest.mark.solver
@pytest.mark.skipif(B._OFT is None or not all(os.path.isfile(p) for p in (B._MESH, B._GEQ, B._IDA)),
                    reason="needs OpenFUSIONToolkit and the public synthetic fixtures")
def test_live_iter_li_and_the_length_artifact(live):
    assert live["truth"] == "consistent" and live["default"] == "consistent", live
    assert live["l_i_iter"] == pytest.approx(live["l_i_iter_oft"], rel=1e-12)
    assert live["verification"]["verdict"] == "PASS" and "l_i_iter" in live["verification"]["rel"]
    sc = np.asarray(live["scan"], float)
    assert sc.shape[0] > 200
    jump = np.abs(np.diff(sc[:, 1])) / sc[:-1, 1]
    i = int(np.argmax(jump))
    # the default l_i of ONE unchanged state jumps between adjacent traced surfaces by more than the tolerance
    assert jump[i] > FW.VERIFY_REL, jump[i]
    a = dict(live["base_stats"], l_i=float(sc[i, 1]))
    b = dict(live["base_stats"], l_i=float(sc[i + 1, 1]))
    old_rel = abs(a["l_i"] - b["l_i"]) / abs(a["l_i"])
    assert old_rel > FW.VERIFY_REL                        # the old comparison would FAIL this pair
    assert FW.verify_states(a, b)["verdict"] == "PASS"     # the new one passes it


@pytest.mark.solver
@pytest.mark.skipif(B._OFT is None or not all(os.path.isfile(p) for p in (B._MESH, B._GEQ, B._IDA)),
                    reason="needs OpenFUSIONToolkit and the public synthetic fixtures")
@pytest.mark.parametrize("name", ["shape", "ip"])
def test_live_genuinely_different_equilibrium_still_fails(live, name):
    ch = live[f"changed_{name}"]
    assert ch["status"].startswith("converged"), ch["status"]
    v = FW.verify_states(live["base_stats"], ch["stats"])
    assert v["verdict"] == "FAIL", v
    if name == "shape":
        assert v["rel"]["l_i_iter"] > FW.VERIFY_REL and v["rel"]["Ip"] < FW.VERIFY_REL


if __name__ == "__main__":
    child(sys.argv[1])
