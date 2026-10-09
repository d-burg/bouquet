"""The structured-MSE review fixes, applied to the LOOP's MSE chord stage.

``Bouquet._structured_mse_jbs_stage`` (the MSE term under the self-consistent
bootstrap loop) shares its helpers with the legacy MSE stage, and gets the
same review fixes:

* a chord OFF the solver mesh is excluded at the first read, with its reason
  (and the stage refuses when fewer than ``min_chords`` remain); a chord that
  goes missing on a LATER read is a refusal, never a stale value;
* the field orientation is the STATED one (``ip_sign``/``bt_sign``), audited
  against the data and flagged when another fits better, never fitted;
* every GS solve the stage spends is counted (``mse_n_solves``), the restore
  re-solve of a refusal included;
* per-chord arrays and the Jacobian go to ``bl.mse_record`` (archived as
  datasets), and ``ip_closure`` keeps chord-count-independent summaries;
* the state it delivers carries what ``_structured_mse_delivered`` judges the
  DELIVERED equilibrium against;
* a bare ``Exception`` from a solve is a refusal; a bug (``TypeError``) is
  re-raised.

Mocked solver and mocked tan(gamma) algebra (no OFT).  Synthetic only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.jbs_loop import jbs_settings

from test_mse_refusal_restore import (_GC, _IP, _NCH, _W, _X, _Eq, _chords,
                                      _field_at, _jbs0, _patch_mse)


def _converging(monkeypatch, field_at=_field_at, min_chords=_NCH,
                solve_error=None, disagree=False):
    """A chord stage that converges: Redl returns the iterate, l_i constant.

    ``field_at`` replaces the field read; ``solve_error`` (an exception
    instance) is raised by the SECOND GS solve; ``disagree`` makes the
    orientation audit report that another orientation fits better.
    """
    import bouquet.mse as M
    import bouquet.utils as U
    from bouquet.run import Bouquet
    _patch_mse(monkeypatch)
    # size-aware tan(gamma)/chi2 mocks (an excluded chord shrinks the set)
    monkeypatch.setattr(M, "mse_tan_gamma",
                        lambda B, ch, sp=1.0, st=1.0:
                        np.zeros(int(ch["n_active"])))
    monkeypatch.setattr(M, "mse_chi2",
                        lambda tg, ch: (0.0, np.zeros(np.size(tg))))
    monkeypatch.setattr(M, "mse_er_terms", lambda ch: "none (mocked)")
    monkeypatch.setattr(
        U, "structured_mse_jacobian",
        lambda f, x, tg, free, step=0.02: (
            [f(np.asarray(x, float)) for _ in range(int(np.sum(free)))],
            np.zeros((np.size(tg), 2)))[1])
    if disagree:
        monkeypatch.setattr(
            M, "mse_orientation_check",
            lambda B, ch, sp, st: dict(table={"(+1,+1)": 9.0},
                                       delta_chi2=9.0, disagrees=True,
                                       best_other="(-1,+1)", note="mocked"))
    jbs0 = _jbs0()
    j_ind = 2.0e6 * (1.0 - _X) ** 2

    def _close(*a, **k):
        return dict(a=[0.0], b=[0.0], mse_chi2_model=0.0,
                    mse_objective_model=0.0, s_ind=np.ones_like(_X),
                    s_bs=np.ones_like(_X), ohm_scale_eff=1.0,
                    bs_scale_eff=1.0, Ip_hybrid=_IP, sign_pattern=None,
                    structure_ind=0.0, structure_bs=0.0, ip_residual=0.0,
                    ip_residual_pct=0.0, axis_residual=None, constraints=[])

    monkeypatch.setattr(U, "close_ip_structured", _close)
    ch = _chords()
    ch.update(min_chords=int(min_chords), n_total=_NCH,
              tgamma=np.zeros(_NCH), A5=np.zeros(_NCH), Er=np.zeros(_NCH))
    state = dict(mse=ch, mse_required=False, soft=False, psi_geom=_X,
                 basis=None, free=np.array([True, True]), x_pred=[0.0, 0.0],
                 F_pred=0.0, j_ind=j_ind, j_BS_swb=jbs0,
                 j_fixed=np.zeros_like(_X), w_lin=_W, c_signed=0.0,
                 Ip_signed=_IP, weights=None, ip_ind=0.7 * _IP,
                 ip_bs=0.3 * _IP, ip_fix=0.0)
    eq = _Eq(j_ind + jbs0)
    bl = SimpleNamespace(j_phi=eq.j.copy(), j_BS=jbs0.copy(),
                         j_inductive=j_ind.copy(), jBS_diff=None,
                         jphi_diff=None, ohm_scale=1.0, bs_scale=1.0,
                         ip_closure=dict(closure_limited=False,
                                         closure_limited_reasons=()))

    def _solve(j):
        if solve_error is not None and len(eq.solved) == 1:
            eq.solved.append(np.asarray(j, float))
            raise solve_error
        return eq.solve_jphi(j)

    s = jbs_settings(_GC(jbs_loop_on_fail="flag"))

    def run():
        return Bouquet._structured_mse_jbs_stage(
            state, bl, eq, _solve, jbs0, lambda jbs, k: dict(state),
            lambda snap: jbs0, lambda snap: dict(li=0.8, q0=None),
            lambda snap: (_W, _X), s, _IP, gate_q0=False,
            field_at=field_at,
            pre_mse=dict(j_solved=eq.j.copy(), restore=None, final={}))

    return dict(run=run, eq=eq, bl=bl, state=state)


def test_applied_stage_counts_solves_and_archives_per_chord_arrays(
        monkeypatch):
    t = _converging(monkeypatch)
    nl, cur, srec = t["run"]()
    assert srec["converged"], srec["stop_reason"]
    eq, bl = t["eq"], t["bl"]
    # every solve counted, none inferred
    assert cur["mse_n_solves"] == len(eq.solved) == srec["n_solves"]
    assert bl.ip_closure["structured_mse_n_solves"] == len(eq.solved)
    assert srec["n_fd_solves"] == 2 * 2      # two Jacobians, 2 free coeffs
    # per-chord arrays on mse_record; ip_closure keeps summaries only
    for k in ("residual_sigma_before", "residual_sigma_after",
              "tgamma_pred_before", "tgamma_pred_after", "jacobian"):
        assert k in bl.mse_record
    assert bl.mse_record["jacobian"].shape == (_NCH, 2)
    icl = bl.ip_closure
    for k in ("structured_mse_residual_sigma_before",
              "structured_mse_tgamma_pred_after", "structured_mse_jacobian"):
        assert k not in icl
    assert icl["structured_mse_jacobian_shape"] == [_NCH, 2]
    # the stated orientation, recorded with its rule
    assert icl["structured_mse_orientation"]["rule"].startswith("STATED")
    # what the delivered-equilibrium judgement needs
    for k in ("mse_chi2_before", "mse_objective_before",
              "mse_delivered_out", "mse_corrector_resolved", "mse_sign"):
        assert k in cur
    assert cur["mse"]["n_active"] == _NCH


def test_the_delivered_judgement_runs_on_the_loop_state(monkeypatch):
    from bouquet.run import Bouquet
    t = _converging(monkeypatch)
    nl, cur, srec = t["run"]()
    c2 = Bouquet._structured_mse_delivered(cur, t["bl"], t["eq"],
                                           field_at=_field_at)
    assert c2 == 0.0
    icl = t["bl"].ip_closure
    assert icl["structured_mse_delivered_worse"] is False
    assert icl["structured_mse_objective_delivered_comparable"] is True


def test_off_mesh_chord_is_excluded_with_its_reason(monkeypatch):
    def fa(R, Z):
        n = np.size(R)
        found = np.ones(n, dtype=bool)
        if n == _NCH:
            found[1] = False                     # chord 1 is off the mesh
        B = np.zeros((n, 3))
        B[~found] = np.nan
        return B, found

    t = _converging(monkeypatch, field_at=fa, min_chords=2)
    nl, cur, srec = t["run"]()
    assert srec["converged"], srec["stop_reason"]
    bl = t["bl"]
    assert cur["mse"]["n_active"] == _NCH - 1
    assert list(bl.mse_record["excluded_index"]) == [1]
    assert "off the solver mesh" in bl.mse_record["excluded_reason"][0]
    assert bl.ip_closure["structured_mse_n_excluded"] == 1


def test_too_few_chords_on_the_mesh_is_a_refusal(monkeypatch):
    def fa(R, Z):
        n = np.size(R)
        found = np.arange(n) != 1
        return np.zeros((n, 3)), found

    t = _converging(monkeypatch, field_at=fa, min_chords=_NCH)
    nl, st, srec = t["run"]()
    assert srec.get("restored_pre_mse") is True
    assert "refused" in t["bl"].ip_closure["structured_mse_status"]
    assert "remain on the solver mesh" in srec["refused"]
    # only the restore re-solve was spent, and it is counted
    assert t["state"]["mse_n_solves"] == len(t["eq"].solved) == 1


def test_a_chord_lost_on_a_later_read_is_a_refusal(monkeypatch):
    calls = {"n": 0}

    def fa(R, Z):
        calls["n"] += 1
        n = np.size(R)
        found = np.ones(n, dtype=bool)
        if calls["n"] > 1:
            found[0] = False                     # lost after the first read
        return np.zeros((n, 3)), found

    t = _converging(monkeypatch, field_at=fa)
    nl, st, srec = t["run"]()
    assert srec.get("restored_pre_mse") is True
    assert "stale field value" in srec["refused"]


def test_orientation_disagreement_is_flagged_and_the_stated_one_kept(
        monkeypatch):
    t = _converging(monkeypatch, disagree=True)
    nl, cur, srec = t["run"]()
    icl = t["bl"].ip_closure
    assert icl["structured_mse_orientation_disagrees"] is True
    assert cur["mse_sign"] == (1.0, 1.0)
    assert any("disagree with the stated field orientation" in r
               for r in icl["closure_limited_reasons"])
    assert icl["closure_limited"] is True


def test_bare_solver_exception_is_a_refusal_but_a_bug_is_not(monkeypatch):
    t = _converging(monkeypatch, solve_error=Exception("GS diverged"))
    nl, st, srec = t["run"]()
    assert srec.get("restored_pre_mse") is True
    assert "GS solve failed" in srec["refused"]
    t2 = _converging(monkeypatch, solve_error=TypeError("a bug"))
    with pytest.raises(TypeError):
        t2["run"]()
