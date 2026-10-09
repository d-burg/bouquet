"""The q0 pin acting under the loop -- live-solver half (``pytest -m solver``).

``GenerationConfig.jbs_loop_q0_corrector`` on the synthetic D3D-like example
(the OMAS data dictionary; ``jBS_baseline_mode="ohmic"``), for the two
channels that pin the on-axis safety factor:

* ``closure_channel="sawtooth_bootstrap"``;
* ``closure_channel="structured"`` (axis row admitted by the sawtooth gate,
  no l_i target -- the configuration of the existing structured loop test,
  at the shipped default pass ceiling).

Each channel is run with the flag OFF (the default: record-only, axis row
held) and ON (axis row moved once per pass from the measured q0, and
``|q0 - q0_target| <= q0_tol`` added to the loop's convergence criteria).

Flag ON must deliver: loop converged, ``|q0 - q0_target| <= q0_tol`` on the
delivered equilibrium, and the bootstrap residuals within the loop tolerances
on the last two passes.  Flag OFF must record the measured q0 residual, so the
two settings can be compared.  The probe stores, for BOTH settings of both
channels, q0 (delivered), q0_target, the residual, the residual over q0_tol
and the number of passes (plus the per-pass pin record when ON) in its JSON;
set ``BQ_Q0_PIN_PROBE_OUT`` to a directory to keep a copy of that JSON.

Every solver call lives in a subprocess probe (``OFT_env`` is a per-process
singleton -- see ``tests/_harness.py``).  Synthetic inputs only; no device
data.
"""
import json
import os
import shutil
import subprocess
import sys

import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

from test_jbs_loop_solver import _MESH, _OMAS, _TIME, solver_only

_S = dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3, tol_q0=2e-3)
_CASES = ("sawtooth", "structured")


# ---------------------------------------------------------------------------
#  the probe (subprocess)
# ---------------------------------------------------------------------------
def _probe(outdir):
    import bouquet as bq
    from bouquet.jbs_loop import JBSNotConverged, jsonable

    _harness.assert_bouquet_is_repo_local()
    b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=_TIME, n_draws=1,
                             header=os.path.join(outdir, "imas"), nthreads=1,
                             reconstruction_engine="legacy")
    b.setup_solver()
    g = b.config.generation
    g.jbs_self_consistent = True
    g.jBS_baseline_mode = "ohmic"
    g.jbs_init = "anchor"
    g.jbs_loop_on_fail = "raise"
    out = dict(q0_tol=float(g.q0_tol), tolerances=dict(_S))

    def _run(case, pin_on):
        g.jbs_loop_q0_corrector = bool(pin_on)
        if case == "sawtooth":
            g.closure_channel = "sawtooth_bootstrap"
            g.jbs_max_passes = 12                    # the default ceiling (12 since 2026-10-07)
        else:
            g.closure_channel = "structured"
            g.structured_li_target = None
            g.jbs_max_passes = 12                    # the default ceiling (12 since 2026-10-07)
        tag = f"{case}_{'on' if pin_on else 'off'}"
        try:
            blx = b.prepare_baseline()
        except JBSNotConverged as e:
            r = jsonable(e.record)
            out[tag] = dict(raised=True, error=str(e)[:2000],
                            converged=False, n_passes=r.get("n_passes"),
                            r_j=r.get("r_j"), r_I=r.get("r_I"),
                            dl_i=r.get("dl_i"), dq0=r.get("dq0"),
                            q0=r.get("q0"), q0_pin=r.get("q0_pin"),
                            max_passes=int(g.jbs_max_passes))
            return
        r = blx.li_metrics["jbs_loop"]
        ic = blx.ip_closure or {}
        q0_res = ic.get("q0_residual")
        out[tag] = dict(
            raised=False,
            converged=bool(r["converged"]), n_passes=int(r["n_passes"]),
            max_passes=int(g.jbs_max_passes),
            stop_reason=r.get("stop_reason"),
            axis_row_active=r.get("axis_row_active"),
            r_j=r["r_j"], r_I=r["r_I"], dl_i=r["dl_i"], dq0=r["dq0"],
            q0_per_pass=r["q0"],
            q0_target=ic.get("q0_target"),
            q0=ic.get("q0_solved"),
            q0_residual=q0_res,
            q0_residual_over_tol=(None if q0_res is None else
                                  abs(float(q0_res)) / float(g.q0_tol)),
            q0_pin=r.get("q0_pin"),
            criteria=r.get("criteria"),
            ip_closure_pin={k: ic[k] for k in ic
                            if str(k).startswith("q0_pin_")
                            or k == "q0_residual_over_tol"},
            closure_limited=bool(ic.get("closure_limited", False)),
            reasons=list(ic.get("closure_limited_reasons", ()) or ()),
            sawtooth_verdict=ic.get("sawtooth_verdict"),
            j_ref0_used=ic.get("j_ref0_used"),
            j_ref0_requested=ic.get("j_ref0_requested"))

    for case in _CASES:
        for pin_on in (False, True):
            _run(case, pin_on)
    with open(os.path.join(outdir, "q0_pin.json"), "w") as fh:
        json.dump(jsonable(out), fh, indent=1)


@pytest.fixture(scope="module")
def q0pin(tmp_path_factory):
    work = tmp_path_factory.mktemp("q0_pin")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"q0-pin probe failed (rc={proc.returncode}):\n"
                    f"{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}")
    src = str(work / "q0_pin.json")
    keep = os.environ.get("BQ_Q0_PIN_PROBE_OUT")
    if keep:
        os.makedirs(keep, exist_ok=True)
        shutil.copy(src, os.path.join(keep, "q0_pin.json"))
    with open(src) as fh:
        return json.load(fh)


def _summary(r):
    return {k: r.get(k) for k in ("converged", "n_passes", "q0", "q0_target",
                                  "q0_residual", "q0_residual_over_tol",
                                  "stop_reason", "error")}


# ---------------------------------------------------------------------------
#  flag ON: the pin acts, the delivered equilibrium meets both
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("case", _CASES)
def test_q0_pin_on_delivers_loop_convergence_and_the_q0_target(q0pin, case):
    r = q0pin[f"{case}_on"]
    tol = q0pin["q0_tol"]
    assert not r["raised"], _summary(r)
    assert r["axis_row_active"] is True, _summary(r)
    assert r["converged"], _summary(r)
    assert r["n_passes"] <= r["max_passes"]
    assert r["criteria"]["q0_residual"] is True
    # the delivered equilibrium meets the q0 target within the UNCHANGED tol
    assert r["q0_residual"] is not None
    assert abs(r["q0_residual"]) <= tol, _summary(r)
    assert abs(r["q0"] - r["q0_target"]) <= tol, _summary(r)
    # ... and the loop's bootstrap criteria on the last two passes
    for i in (-1, -2):
        assert r["r_j"][i] <= _S["rtol_j"] and r["r_I"][i] <= _S["rtol_Ip"]
        assert r["dl_i"][i] is not None and r["dl_i"][i] <= _S["tol_li"]
        assert r["dq0"][i] is not None and r["dq0"][i] <= _S["tol_q0"]
        assert abs(r["q0_pin"]["q0_residual"][i]) <= tol
    # the records: per pass, and in the closure-health block
    p = r["q0_pin"]
    for k in ("axis_row", "axis_current_solved", "q0", "q0_residual",
              "q0_residual_over_tol", "axis_row_next"):
        assert len(p[k]) >= r["n_passes"], k
    assert p["delivered_within_q0_tol"] is True
    assert r["ip_closure_pin"]["q0_pin_delivered_within_tol"] is True
    assert r["ip_closure_pin"]["q0_residual_over_tol"] <= 1.0
    assert r["ip_closure_pin"]["q0_pin_acted"] == (p["n_row_updates"] > 0)
    assert not any("q0 miss" in str(x) for x in r["reasons"]), r["reasons"]


# ---------------------------------------------------------------------------
#  flag OFF: record-only, the residual recorded for the comparison
# ---------------------------------------------------------------------------
@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("case", _CASES)
def test_q0_pin_off_records_the_measured_residual(q0pin, case):
    r = q0pin[f"{case}_off"]
    assert not r["raised"], _summary(r)
    assert r["converged"], _summary(r)
    assert r["axis_row_active"] is True
    assert r["q0_residual"] is not None and np.isfinite(r["q0_residual"])
    assert r["q0_target"] is not None and r["q0"] is not None
    # record-only: no pin block, no pin keys, the row held at the request
    assert r["q0_pin"] is None and r["ip_closure_pin"] == {}
    assert "q0_residual" not in (r["criteria"] or {})
    assert r["j_ref0_used"] == r["j_ref0_requested"]
    # the unchanged q0_tol flag, as before
    assert abs(r["q0_residual"]) <= q0pin["q0_tol"] or any(
        "q0" in str(x) for x in r["reasons"])


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("case", _CASES)
def test_q0_pin_on_is_no_worse_than_off_on_q0(q0pin, case):
    """The comparison the flag exists for (both numbers are in the JSON): the
    delivered |q0 - q0_target| with the pin acting is inside q0_tol, and not
    larger than the record-only residual whenever that one missed q0_tol."""
    on, off = q0pin[f"{case}_on"], q0pin[f"{case}_off"]
    tol = q0pin["q0_tol"]
    assert on["q0_residual"] is not None and off["q0_residual"] is not None, \
        (_summary(on), _summary(off))
    assert abs(on["q0_residual"]) <= tol, (_summary(on), _summary(off))
    if abs(off["q0_residual"]) > tol:
        assert abs(on["q0_residual"]) < abs(off["q0_residual"])


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _probe(sys.argv[1])
