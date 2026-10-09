"""Rejected draws are visible: rejected, not archived, not counted.

``generate_bouquet`` runs end to end on a mock solver (no OFT) with the draw
function mocked: every attempt is rejected by the self-consistent j_BS loop --
in the draw's own loop (``JBSNotConverged``), in its loop under the hard coil
bounds (``CoilSaturated``) or in the post-homotopy stage (both kinds).  Each
rejection must carry its OWN reason code in the per-attempt records and the
printed run summary, no draw group may reach the archive, and none may count
toward the until-N target.

Synthetic inputs only; no device data.
"""
import os

import h5py
import numpy as np
import pytest

_N = 21
_X = np.linspace(0.0, 1.0, _N)
_COILS = {"F1A": 1.0e4, "F2A": -2.0e4, "F9A": 5.0e3, "F9B": -5.0e3}


class _FakeGS:
    """A permissive stand-in for a TokaMaker solver: every solve 'converges'
    to the same state (the draws are mocked, so nothing physical is asked)."""

    def __init__(self):
        self.psi_bounds = (-0.1, 0.1)
        self.coil_sets = {n: {} for n in _COILS}
        self.o_point = np.array([1.7, 0.0])
        self._coils = dict(_COILS)
        self.bounds = []

    # state
    def copy_eq(self):
        return dict(coils=dict(self._coils))

    def replace_eq(self, source_eq=None):
        if source_eq:
            self._coils = dict(source_eq["coils"])

    def get_psi(self, normalized=True):
        return np.linspace(0.0, 1.0, 50)

    def set_psi(self, psi, update_bounds=False):
        pass

    def init_psi(self, *a, **k):
        pass

    # solve
    def set_targets(self, **k):
        pass

    def set_profiles(self, **k):
        pass

    def solve(self, return_its=False):
        return (None, 10) if return_its else None

    # coils
    def get_coil_currents(self):
        return dict(self._coils), None

    def set_coil_currents(self, c):
        self._coils = dict(c)

    def set_coil_bounds(self, b=None):
        self.bounds.append(b)

    def coil_reg_term(self, coffs, target=0.0, weight=1.0):
        return (dict(coffs), target, weight)

    def set_coil_reg(self, *a, **k):
        pass

    def set_isoflux(self, *a, **k):
        pass

    # readbacks
    def get_stats(self, **k):
        return {"l_i": 0.8, "Ip": 1.0e6, "beta_pol": 0.5, "beta_n": 1.5}

    def get_globals(self):
        return (1.0e6, 0.0, 0.0)

    def get_q(self, *a, **k):
        q = np.linspace(1.2, 4.0, _N)
        return (_X, q, None, None, None, None)

    def get_xpoints(self):
        return None, False

    def trace_surf(self, psi):
        t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        return np.column_stack([1.7 + 0.6 * np.cos(t), 1.0 * np.sin(t)])

    def save_eqdsk(self, *a, **k):
        raise RuntimeError("the mock never writes an eqdsk")


def _run(tmp_path, monkeypatch, outcomes, **kw):
    import bouquet.TokaMaker_interface as TI
    from bouquet.jbs_loop import JBSNotConverged, jbs_settings

    class _GC:
        jbs_self_consistent = True
        jbs_init = "anchor"
        jbs_rtol_j = 1e-3
        jbs_rtol_Ip = 1e-4
        jbs_tol_li = 1e-3
        jbs_tol_q0 = 2e-3
        jbs_max_passes = 8
        jbs_max_passes_draw = 12
        jbs_max_passes_post_homotopy = 4
        jbs_relax = 0.7
        jbs_loop_on_fail = "raise"

    seq = list(outcomes)
    Jb = 1.0e5 * (1 - _X)

    def _perturb(mygs, psi_N, pressure, ne, te, ni, ti, input_j_phi, *a,
                 **k):
        what = seq.pop(0)
        if what == "loop":
            raise JBSNotConverged("self-consistent j_BS loop [draw anchor] "
                                  "did not converge: pass ceiling 12",
                                  dict(n_passes=12))
        if what == "loop_sat":
            raise TI.CoilSaturated("coil saturation after draw j_BS loop "
                                   "solve 3", dict(stage="hard bounds"))
        diag = dict(j_BS=Jb, j_inductive=input_j_phi - Jb, j_BS_edge=None,
                    jbs_loop=dict(enabled=True, converged=True),
                    _jbs_ctx=dict(kind="standard"), r2_ip_scale=None,
                    r2_f_ind=None, aux=None, proxy_bias_observed=None)
        return ne, te, ni, ti, None, np.asarray(input_j_phi, float), diag

    def _post_homotopy(mygs, ctx, settings, psi_N, psi_pad, Ip,
                       coil_guard=None):
        what = seq_ph.pop(0)
        if what == "ph_sat":
            raise TI.CoilSaturated("coil saturation after post-homotopy "
                                   "pass 1", dict(stage="post-homotopy"))
        raise JBSNotConverged("self-consistent j_BS loop [draw "
                              "post-homotopy] did not converge", {})

    seq_ph = [o for o in outcomes if o.startswith("ph")]
    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _perturb)
    monkeypatch.setattr(TI, "_post_homotopy_jbs", _post_homotopy)
    rej = []
    header = str(tmp_path / "rej")
    ne = 5e19 * (1 - 0.8 * _X ** 2)
    te = 2e3 * (1 - 0.9 * _X ** 2) + 50.0
    jphi = 1.0e6 * (1 - _X ** 2) + Jb
    out = TI.generate_bouquet(
        _FakeGS(), _X, len(outcomes), header, jphi, ne, te, 0.9 * ne, te,
        0.05 * ne, 0.05 * te, 0.05 * ne, 0.05 * te, 0.05 * jphi,
        0.4, 0.4, 0.25, 1.0e6, 0.8, 1.5 * np.ones(_N),
        input_jinductive=jphi - Jb, baseline_j_BS=Jb, psi_N_kinetic=None,
        diagnostic_plots=False, seed=3, jbs_loop=jbs_settings(_GC(),
                                                              draw=True),
        homotopy_passes=[(0.05, 0.10), (0.01, 0.01)],
        rejection_log=rej, **kw)
    return out, rej, header


def _draw_groups(header):
    path = f"{header}.h5"
    if not os.path.isfile(path):
        return []
    with h5py.File(path, "r") as hf:
        names = []
        hf.visit(names.append)
    return [n for n in names if n.split("/")[-1].isdigit()]


def test_loop_rejections_are_recorded_by_reason_not_archived_not_counted(
        tmp_path, monkeypatch, capsys):
    outcomes = ["loop", "loop_sat", "ph_loop", "ph_sat"]
    out, rej, header = _run(tmp_path, monkeypatch, outcomes,
                            n_inspec_target=1, max_total_draws=4)
    # rejected: one record per attempt, each with its OWN reason code
    assert [r["reason"] for r in rej] == [
        "jbs_not_converged", "coil_saturation_jbs_loop",
        "jbs_post_homotopy", "coil_saturation_post_homotopy"]
    assert [r["draw"] for r in rej] == [0, 1, 2, 3]
    assert rej[1]["info"]["stage"] == "hard bounds"
    assert rej[0]["error_type"] == "JBSNotConverged"
    # never read as a GS / homotopy failure
    assert not any(r["reason"] in ("perturb_failed", "homotopy_infeasible",
                                   "post_perturb_failed") for r in rej)
    # not archived
    assert out == []
    assert _draw_groups(header) == []
    # not counted: the until-N target (1) is reported missed, 0 delivered
    txt = capsys.readouterr().out
    assert "until-N did not reach its target: 0/1" in txt
    # the run summary names every reason
    summ = [ln for ln in txt.splitlines() if ln.startswith("[draw summary]")]
    assert len(summ) == 1
    for code in ("jbs_not_converged=1", "coil_saturation_jbs_loop=1",
                 "jbs_post_homotopy=1", "coil_saturation_post_homotopy=1"):
        assert code in summ[0], summ[0]
    assert "4 attempt(s): 0 archived, 4 rejected" in summ[0]


def test_a_masked_resample_loop_failure_has_its_own_counter():
    from bouquet.TokaMaker_interface import ANCHOR_MASKED_FAILURES
    assert "jbs_band_resample" in ANCHOR_MASKED_FAILURES
    import inspect
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    src = inspect.getsource(perturb_kinetic_equilibrium)
    assert ('"jbs_band_resample"\n                        if isinstance('
            '_rs_exc, JBSNotConverged)') in src


def test_every_rejection_path_of_the_draw_loop_records_a_reason():
    """Every `continue` exit of generate_bouquet's draw loop (the draw, the
    post-align checks, a failed g-file save, a failed state restore after
    the i-file save) records a rejection first --
    there is no silent rejection path."""
    import ast
    import inspect
    import textwrap
    from bouquet.TokaMaker_interface import generate_bouquet
    src = textwrap.dedent(inspect.getsource(generate_bouquet))
    loop = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.For)
                and ast.unparse(n.target) == "count")

    def exits(node):        # continues of THIS loop, not of nested loops
        for c in ast.iter_child_nodes(node):
            if isinstance(c, ast.Continue):
                yield c
            elif not isinstance(c, (ast.For, ast.While, ast.FunctionDef,
                                    ast.AsyncFunctionDef, ast.Lambda)):
                yield from exits(c)
    lines = src.splitlines(keepends=True)
    segs = ["".join(lines[:c.lineno - 1]) for c in exits(loop)]
    assert len(segs) == 4, len(segs)
    for seg in segs:
        assert "_reject(" in seg[-3000:]
