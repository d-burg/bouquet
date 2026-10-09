"""The two-scalar Ip + l_i preset on the live solver (``pytest -m solver``).

On the D3D-like synthetic g-file + p-file example, ``engine_preset=
"two_scalar_li"`` (one scalar on the inductive, one on the bootstrap, rows
Ip + l_i both hard):

* the reconstruction converges within the unchanged pass ceiling and both
  rows hold on the DELIVERED equilibrium (Ip through the loop residuals, the
  hard l_i at the g-file row's 1e-3);
* it is the q95 attribution study's two-scalar state: the same
  reconstruction reached the study's way (the settings' preset patched to
  the constant two-scalar basis, rows Ip + l_i) delivers the same state to
  within the loop tolerances (``jbs_tol_li``, ``jbs_tol_q0`` on q95);
* on the build the study measured (OpenFUSIONToolkit ``abbfc6f``) the
  delivered q95 is the study's 4.595594 within ``jbs_tol_q0`` and the
  q-profile rms is the study's 0.235 %; on any other build both are
  recorded in the probe JSON and printed, not asserted (the build is an
  input: the fixed build moved the structured engine's q95 by +0.004).

Both parts run at the STUDY'S OWN SETTINGS (:data:`_STUDY_SETTINGS`): the
values in force at the study's commit wherever the defaults have moved
since and the reconstruction reads them -- ``separatrix_pressure="legacy"``
(the pre-setting pressure frame; the default became ``"offset"`` on
2026-10-02) with ``edge_pprime_pin=True`` -- so "it is the study's state"
stays a like-for-like claim (owner-approved 2026-10-05).  The other defaults
changed since (``engine_mse_jacobian``, ``engine_ids_inductive``,
``engine_li_row_relaxation`` = 1.0, the draw-only fields) are not read by a
g-file reconstruction without MSE rows, or equal the study's behaviour.

Every solver call runs in a subprocess of ``tests/probes/measure_engine.py``.
Synthetic inputs only.
"""
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

#: the loop's own tolerances (GenerationConfig defaults) -- read, not tuned
_S = dict(tol_li=1e-3, tol_q0=2e-3)
#: the q95 attribution study's two-scalar state (engine/draws 924e81c,
#: OpenFUSIONToolkit abbfc6f, single-threaded): q95 and the q-profile rms
#: [%] over psi_N 0.05-0.95 against the g-file qpsi
_STUDY = dict(build="abbfc6f", q95=4.595594, q_rms_pct=0.235)
#: the settings those numbers were measured under, where today's defaults
#: differ and the g-file reconstruction reads them (see the docstring)
_STUDY_SETTINGS = dict(separatrix_pressure="legacy", edge_pprime_pin=True)


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    import json
    work = str(tmp_path_factory.mktemp("engine_2s"))
    old = os.environ.get("BQ_ENGINE_PROBE_GC")
    os.environ["BQ_ENGINE_PROBE_GC"] = json.dumps(_STUDY_SETTINGS)
    try:
        return {p: ME.run_part(p, work)
                for p in ("recon_2s", "recon_2s_patched")}
    finally:
        if old is None:
            os.environ.pop("BQ_ENGINE_PROBE_GC", None)
        else:
            os.environ["BQ_ENGINE_PROBE_GC"] = old


def _need(d, stage):
    st = d["stages"].get(stage)
    assert st == "ok", (f"{d['part']} stage {stage!r} failed: {st}"
                        + (f"\nfatal: {d.get('fatal')}" if d.get("fatal")
                           else ""))
    return d[stage]


@pytest.mark.solver
@solver_only
def test_two_scalar_li_converges_and_meets_both_rows(parts):
    b = _need(parts["recon_2s"], "build")
    assert b["converged"] and b["loop_converged"], b.get("delivered")
    for name, n in b["n_passes"].items():
        assert n <= b["max_passes"], (name, n)
    d = b["delivered"]
    assert d["ok"], d["misses"]
    li = d["checks"]["l_i"]
    assert li["hard"] and abs(li["delivered"] - li["target"]) < 1e-3, li
    assert d["checks"]["loop"]["ok"], d["checks"]["loop"]
    st = b["engine_record"]["settings"]
    assert st["preset_in_force"] == "two_scalar_li"


@pytest.mark.solver
@solver_only
def test_two_scalar_li_is_the_q95_studys_state(parts):
    a = _need(parts["recon_2s"], "distance")
    p = _need(parts["recon_2s_patched"], "distance")
    _need(parts["recon_2s_patched"], "build")
    assert abs(a["li3_matched"]["engine"] - p["li3_matched"]["engine"]) \
        <= _S["tol_li"]
    assert abs(a["q95"]["engine"] - p["q95"]["engine"]) <= _S["tol_q0"]
    q95, rms = a["q95"]["engine"], a["q_profile_rel_pct"]["rms"]
    oft = str(parts["recon_2s"].get("oft_file", ""))
    # the pinned settings reached the solve (the probe records its config)
    for part in ("recon_2s", "recon_2s_patched"):
        sp = (_need(parts[part], "build")["engine_record"]["settings"]
              .get("edge_pressure") or {})
        for k, v in _STUDY_SETTINGS.items():
            assert sp.get(k) == v, (part, k, sp)
    print(f"\n[two_scalar_li] q95 {q95:.6f} (study {_STUDY['q95']}), q rms "
          f"{rms:.3f} % (study {_STUDY['q_rms_pct']}), OFT {oft}")
    if _STUDY["build"] not in oft:
        # the study's numbers belong to one OFT build: on another the
        # comparison is NOT made -- said so (a skip), never passed silently
        pytest.skip(f"the q95 / q-rms comparison with the study needs the "
                    f"study's OFT build {_STUDY['build']!r}; this run used "
                    f"{oft!r} (the identity asserts above did run)")
    assert abs(q95 - _STUDY["q95"]) <= _S["tol_q0"], q95
    # the study quotes the rms to 3 decimals
    assert abs(rms - _STUDY["q_rms_pct"]) <= 5e-4, rms
