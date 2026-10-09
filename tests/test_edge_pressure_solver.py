"""The two edge-pressure settings on the live solver (``pytest -m solver``).

The g-file example (the D3D-like g-file + p-file) on the unified engine,
with a CONSTRUCTED constant pressure added on the whole profile (a constant
fast pressure, ``BQ_ENGINE_PROBE_PSEP_ADD``; the example's own separatrix
pressure is under 1 % of the axis value) so that
``separatrix_pressure="offset"`` has something to act on.  For the four
combinations {``edge_pprime_pin`` on, off} x {``"legacy"``, ``"offset"``}:

* the reconstruction converges and the engine draw at ZERO perturbation
  reproduces it (the first request bit-identical to the stored one, the
  loop's unchanged tolerances) -- the identity holds by construction under
  every combination;
* the record carries the settings, ``p_sep`` and the axis target, and the
  solver was handed them: its axis pressure is the target; under
  ``"offset"`` its ``P'`` is the input's own over the bulk of the profile,
  under ``"legacy"`` it is inflated by ``p_axis / (p_axis - p_sep)``;
* the reporting: with nothing added back both frames are the solver's own
  statistics; under ``"offset"`` the full frame is the solver's plus
  ``1.5 p_sep V`` with ``V`` the solver's own volume;
* delivery (one seeded draw, pin off + offset): the baseline g-file and the
  draw's g-file carry the full pressure -- ``PRES`` is the inward integral
  of the file's own ``PPRIME`` from its last point, and at the edge it is
  the pressure the solve was handed (each draw's own).

Every solver call runs in a subprocess of ``tests/probes/measure_engine.py``.
Synthetic inputs only; the constant is a constructed test, labelled as such
in the probe's JSON.
"""
import json
import os
import sys

import _harness

_harness.ensure_repo_on_syspath()

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "probes"))
import measure_engine as ME  # noqa: E402

_files_ok = all(os.path.isfile(p) for p in (ME._GEQ, ME._PF, ME._MESH))
solver_only = pytest.mark.skipif(
    not (_files_ok and ME.oft_importable()),
    reason="needs OFT + the D3D-like example files; skipped when unavailable")

#: the constructed constant [Pa]: 5 % of the example's axis pressure
PSEP_ADD = 2900.0
COMBOS = [(True, "legacy"), (True, "offset"), (False, "legacy"),
          (False, "offset")]
#: the loop's own tolerances (GenerationConfig defaults) -- read, not tuned
_S = dict(rtol_j=1e-3, rtol_Ip=1e-4, tol_li=1e-3)


def _run(part, work, pin, sep, **kw):
    old = {k: os.environ.get(k) for k in ("BQ_ENGINE_PROBE_GC",
                                          "BQ_ENGINE_PROBE_PSEP_ADD")}
    os.environ["BQ_ENGINE_PROBE_GC"] = json.dumps(dict(
        edge_pprime_pin=pin, separatrix_pressure=sep))
    os.environ["BQ_ENGINE_PROBE_PSEP_ADD"] = repr(PSEP_ADD)
    try:
        return ME.run_part(part, work, **kw)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def combos(tmp_path_factory):
    out = {}
    for pin, sep in COMBOS:
        work = str(tmp_path_factory.mktemp(
            f"edge_{'pin' if pin else 'nopin'}_{sep}"))
        out[(pin, sep)] = _run("recon", work, pin, sep)
    return out


def _need(d, stage):
    st = d["stages"].get(stage)
    assert st == "ok", (f"{d['part']} stage {stage!r} failed: {st}"
                        + (f"\nfatal: {d.get('fatal')}" if d.get("fatal")
                           else ""))
    return d[stage]


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("pin,sep", COMBOS)
def test_the_reconstruction_converges_and_the_identity_holds(combos, pin,
                                                             sep):
    d = combos[(pin, sep)]
    b = _need(d, "build")
    assert b["converged"] and b["loop_converged"], b.get("delivered")
    for name, n in b["n_passes"].items():
        assert n <= b["max_passes"], (name, n)
    z = _need(d, "sigma0")
    assert z["request_bit_identical"] is True, z
    assert z["loop_converged"] is True, z
    assert z["r_j"] <= _S["rtol_j"] and z["r_I"] <= _S["rtol_Ip"], z
    assert abs(z["dl_i"]) <= _S["tol_li"], z
    assert z["passed"] is True, z


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("pin,sep", COMBOS)
def test_the_solver_was_handed_the_recorded_target(combos, pin, sep):
    d = combos[(pin, sep)]
    ep = _need(d, "build")["engine_record"]["edge_pressure"]
    assert ep["edge_pprime_pin"] is pin
    assert ep["separatrix_pressure"] == sep
    # the constructed constant is in the separatrix pressure
    assert ep["p_sep"] > PSEP_ADD
    assert ep["p_sep_applied"] == (ep["p_sep"] if sep == "offset" else 0.0)
    assert ep["pax_target"] == pytest.approx(
        ep["p_axis"] - ep["p_sep_applied"], rel=1e-14)
    ed = _need(d, "edge")
    e = ed["summary"]
    # the solver's pressure is zero at the boundary and reaches its axis
    # target (it rescales P' to it): at the innermost sampled surface it is
    # target * (p - p_sep) / (p_axis - p_sep) of the pressure it was handed
    import numpy as np
    p_at = float(np.interp(ed["psi_sampled"][0], ed["psi_N"],
                           ed["pressure_input"]))
    want_p = ep["pax_target"] * (p_at - ep["p_sep"]) \
        / (ep["p_axis"] - ep["p_sep"])
    assert e["pressure_solver_axis"] == pytest.approx(want_p, rel=1e-3)
    assert e["pressure_solver_axis"] <= ep["pax_target"]
    # ... so P' is the input's own under "offset", and inflated by
    # p_axis / (p_axis - p_sep) under "legacy"
    want = (1.0 if sep == "offset"
            else ep["p_axis"] / (ep["p_axis"] - ep["p_sep"]))
    assert e["pprime_ratio_solver_over_input_mid"] == pytest.approx(
        want, rel=5e-3)
    assert (ep["p_axis"] / (ep["p_axis"] - ep["p_sep"])) > 1.04


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("pin,sep", COMBOS)
def test_the_frames_are_the_solvers_stats_plus_the_separatrix_term(
        combos, pin, sep):
    d = combos[(pin, sep)]
    fr = _need(d, "build")["engine_record"]["edge_pressure"]["frames"]
    if sep == "legacy":
        assert fr["p_sep"] == 0.0 and fr["factor"] == 1.0
        assert fr["full"] == fr["solver"]
    else:
        assert fr["full"]["W_MHD"] == pytest.approx(
            fr["solver"]["W_MHD"] + 1.5 * fr["p_sep"] * fr["volume"],
            rel=1e-12)
        for k in ("beta_n", "beta_pol", "beta_tor"):
            assert fr["full"][k] == pytest.approx(
                fr["solver"][k] * fr["factor"], rel=1e-12)
    # the table reports both ways, each against the input's same frame
    t = _need(d, "distance")["pressure_frames"]
    assert "error" not in t, t
    for k in ("solver", "full"):
        for q in ("beta_n", "beta_p", "W_MHD_MJ"):
            assert set(t["frames"][k][q]) >= {"input", "engine", "rel_pct"}
    assert t["constructed_constant_Pa"] == PSEP_ADD
    # the solver's volume is the one the energy term uses
    assert t["volume_engine"] == fr["volume"] > 0.0


@pytest.fixture(scope="module")
def delivered(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("edge_delivered"))
    return _run("draws_recon", work, False, "offset", draws=1,
                seed=ME.DEFAULT_SEED)


@pytest.mark.solver
@solver_only
def test_a_delivered_gfile_carries_the_full_pressure(delivered):
    g = _need(delivered, "delivered_gfiles")
    files = [g["baseline"]] + list(g["draws"])
    assert len(files) >= 1
    for f in files:
        r = f["record"]
        assert r["separatrix_pressure"] == "offset"
        assert r["p_sep_applied"] == r["p_sep"] > PSEP_ADD
        # PRES is the inward integral of the file's own PPRIME
        assert f["gradient_check_max_over_axis"] <= 1e-3, f
        # ... and at its last point (the 1 - psi_pad surface) it is the
        # pressure the solve was handed there
        assert abs(f["pres_minus_input_edge_over_axis"]) <= 1e-3, f
        assert f["pres_edge"] >= r["p_sep"]
    # each draw its own separatrix pressure, not the baseline's
    for f in g["draws"]:
        assert f["record"]["p_sep"] != g["baseline"]["record"]["p_sep"]
