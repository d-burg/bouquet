"""Draw-path guards of the self-consistent bootstrap loop -- fast half.

No GS solver: every solve is a mock.  The live counterparts are in
``test_jbs_loop_solver.py`` (``pytest -m solver``).

* **Coil saturation (B1).**  The homotopy treats a bounded solve whose coil
  drift reaches ``COIL_SATURATION_FRACTION`` (0.99) of its bound as
  infeasible.  The post-homotopy j_BS passes and the draw's j_BS loop passes
  under the hard coil bounds re-solve at installed bounds too, so they must
  apply the SAME criterion and reject a saturated draw -- a check added, not a
  bar moved.

Synthetic inputs only; no device data.
"""
import inspect

import numpy as np
import pytest

from bouquet.jbs_loop import jbs_settings


class _GC:
    """A GenerationConfig stand-in carrying only the loop fields."""

    def __init__(self, **kw):
        self.jbs_self_consistent = True
        self.jbs_init = "anchor"
        self.jbs_rtol_j = 1e-3
        self.jbs_rtol_Ip = 1e-4
        self.jbs_tol_li = 1e-3
        self.jbs_tol_q0 = 2e-3
        self.jbs_max_passes = 8
        self.jbs_max_passes_draw = 12
        self.jbs_max_passes_post_homotopy = 4
        self.jbs_relax = 0.7
        self.jbs_loop_on_fail = "raise"
        for k, v in kw.items():
            setattr(self, k, v)


_IP = 1.0e6
_BASE_COILS = {"F1A": 1.0e4, "F2A": -2.0e4, "F9A": 5.0e3, "F9B": -5.0e3}
_VSC = ("F9A", "F9B")


def _shape(x):
    return 3.0e5 * np.exp(-0.5 * ((x - 0.93) / 0.03) ** 2) + 1.0e5 * (1 - x)


class _CoilEq:
    """Just enough of a TokaMaker for _post_homotopy_jbs, with coil currents
    that every solve moves to ``drift_after_solve`` (fractional, on F1A)."""

    def __init__(self, drift_after_solve):
        self.psi_bounds = (-0.15, 0.12)
        self.solves = 0
        self.drift = float(drift_after_solve)
        self.coils = dict(_BASE_COILS)

    def copy_eq(self):
        return object()

    def get_stats(self, **kw):
        return {"l_i": 0.65}

    def set_targets(self, **kw):
        pass

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        pass

    def solve(self):
        self.solves += 1
        self.coils = dict(_BASE_COILS)
        self.coils["F1A"] = _BASE_COILS["F1A"] * (1.0 + self.drift)

    def get_coil_currents(self):
        return dict(self.coils), None


def _ph_setup(monkeypatch, kind):
    import bouquet.jbs_loop as L
    import bouquet.TokaMaker_interface as TI
    x = np.linspace(0.0, 1.0, 65)
    Jstar = _shape(x)
    calls = {"corr": 0}
    monkeypatch.setattr(L, "residual_weights",
                        lambda eq, psi_N, psi_pad=1e-3, coord="psi_n": (np.ones_like(x), x,
                                                         "test"))

    def _renorm(mygs, psi_N, target, Ip, pad, label=""):
        return np.asarray(target, float), 1.0

    def _corr(mygs, psi_N, target, pp, Ip, pax, pad, **kw):
        calls["corr"] += 1
        mygs.solve()                       # the corrective iteration solves
        return np.asarray(target, float) * 0.999, 3, [1.0]

    monkeypatch.setattr(TI, "_renormalize_target_to_Ip", _renorm)
    monkeypatch.setattr(TI, "_corrective_jphi_iteration", _corr)
    monkeypatch.setattr(TI, "_r2_ip_scale", lambda *a, **k: 1.0)
    j_ind = 2.0e6 * (1.0 - x) ** 2
    ctx = dict(kind=kind, compose=lambda snap: (Jstar.copy(), Jstar.copy(),
                                                None),
               spike_used=1.0015 * Jstar, cand=j_ind.copy(),
               j_ind_used=j_ind.copy(), j_fixed_eff=np.zeros_like(x),
               pres_tmp=1e4 * (1.0 - x ** 2) + 10.0, input_j_phi=j_ind + Jstar,
               r2_mode="legacy", j_phi_request=j_ind + Jstar,
               isolate_edge_jBS=False)
    return TI, x, ctx, calls, jbs_settings(_GC(), draw=True)


# ---------------------------------------------------------------------------
#  one criterion
# ---------------------------------------------------------------------------
def test_the_criterion_is_the_homotopys_099_of_the_bound():
    from bouquet.TokaMaker_interface import (COIL_SATURATION_FRACTION,
                                             _coil_saturation)
    assert COIL_SATURATION_FRACTION == 0.99
    # drifts in %, bounds as fractions -- exactly the homotopy's arithmetic
    # (0.99 * 0.01 * 100 is 0.9900000000000001 in floating point, the same
    # number the homotopy has always compared against)
    assert _coil_saturation(0.99 * 0.01 * 100.0, 0.0, 0.01, 0.01) == (True,
                                                                     False)
    assert _coil_saturation(0.9901, 0.0, 0.01, 0.01) == (True, False)
    assert _coil_saturation(0.989, 0.0, 0.01, 0.01) == (False, False)
    assert _coil_saturation(0.0, 4.951, 0.01, 0.05) == (False, True)


def test_the_homotopy_uses_the_shared_criterion():
    """One definition of "saturated": the homotopy's guard calls the same
    helper the new guards call (a second inline copy is how they drift)."""
    from bouquet.TokaMaker_interface import generate_bouquet
    src = inspect.getsource(generate_bouquet)
    assert "_coil_saturation(" in src
    assert "0.99 * _dF" not in src and "0.99 * _dVSC" not in src
    # the post-homotopy stage is handed a guard at the homotopy's final bounds
    assert "coil_guard=_ph_guard" in src
    assert "_final_drift_F_lim, _final_drift_VSC_lim" in src
    # the draw's loop is handed the hard-bound guard
    assert "coil_saturation_guard=_hard_sat_guard" in src


def test_the_guard_measures_and_raises_with_the_record():
    from bouquet.TokaMaker_interface import (CoilSaturated,
                                             _make_coil_saturation_guard)
    eq = _CoilEq(0.0)
    g = _make_coil_saturation_guard(eq, _BASE_COILS, _VSC, 0.01, 0.01,
                                    stage="test")
    eq.solve()
    assert g("pass 1")["saturated"] is False
    eq.drift = 0.0100
    eq.solve()
    with pytest.raises(CoilSaturated) as ei:
        g("pass 2")
    info = ei.value.info
    assert info["saturated"] and info["F_lim"] == 0.01
    assert info["fraction"] == 0.99 and info["stage"] == "test"
    assert [e["label"] for e in g.log] == ["pass 1", "pass 2"]


# ---------------------------------------------------------------------------
#  B1: the post-homotopy re-solves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["fixc", "standard"])
def test_a_saturating_post_homotopy_resolve_rejects_the_draw(monkeypatch,
                                                             kind):
    """The first post-homotopy pass re-solves with F1A at 0.995 % of a 1 %
    bound: the guard fires right after that re-solve and the stage raises
    CoilSaturated (generate_bouquet then rejects the draw).  Without the guard
    the same pass sequence completes and would read as a kept draw."""
    from bouquet.TokaMaker_interface import (CoilSaturated,
                                             _make_coil_saturation_guard)
    TI, x, ctx, calls, s = _ph_setup(monkeypatch, kind)
    eq = _CoilEq(0.00995)
    g = _make_coil_saturation_guard(eq, _BASE_COILS, _VSC, 0.01, 0.01,
                                    stage="post-homotopy test")
    with pytest.raises(CoilSaturated):
        TI._post_homotopy_jbs(eq, ctx, s, x, 1e-3, _IP, coil_guard=g)
    assert eq.solves == 1, "the guard must fire after the FIRST re-solve"
    assert len(g.log) == 1 and g.log[0]["saturated"]
    assert g.log[0]["label"].startswith("post-homotopy pass 1")
    # the unguarded control: the same sequence runs to completion
    eq2 = _CoilEq(0.00995)
    rec, *_ = TI._post_homotopy_jbs(eq2, ctx, s, x, 1e-3, _IP)
    assert rec["passes"]["converged"]
    assert rec["coil_saturation_guard"]["active"] is False


def test_a_post_homotopy_resolve_with_headroom_is_kept_and_logged(
        monkeypatch):
    from bouquet.TokaMaker_interface import _make_coil_saturation_guard
    TI, x, ctx, calls, s = _ph_setup(monkeypatch, "standard")
    eq = _CoilEq(0.005)
    g = _make_coil_saturation_guard(eq, _BASE_COILS, _VSC, 0.01, 0.01,
                                    stage="post-homotopy test")
    rec, spk, full, jphi = TI._post_homotopy_jbs(eq, ctx, s, x, 1e-3, _IP,
                                                 coil_guard=g)
    assert rec["passes"]["converged"]
    cg = rec["coil_saturation_guard"]
    assert cg["active"] and cg["fraction"] == 0.99
    assert cg["F_lim"] == 0.01 and cg["VSC_lim"] == 0.01
    # one check per pass, none saturated
    assert len(cg["checks"]) == rec["passes"]["n_passes"]
    assert not any(c["saturated"] for c in cg["checks"])


def test_a_draw_inside_tolerance_takes_no_pass_and_no_check(monkeypatch):
    from bouquet.TokaMaker_interface import _make_coil_saturation_guard
    TI, x, ctx, calls, s = _ph_setup(monkeypatch, "standard")
    ctx["spike_used"] = ctx["compose"](None)[0]
    eq = _CoilEq(0.00995)                     # would saturate IF it re-solved
    g = _make_coil_saturation_guard(eq, _BASE_COILS, _VSC, 0.01, 0.01,
                                    stage="post-homotopy test")
    rec, *_ = TI._post_homotopy_jbs(eq, ctx, s, x, 1e-3, _IP, coil_guard=g)
    assert rec["accepted_without_passes"] and eq.solves == 0 and not g.log


# ---------------------------------------------------------------------------
#  B1: the draw's j_BS loop passes under the hard coil bounds
# ---------------------------------------------------------------------------
def test_the_draw_loop_guards_every_pass_under_the_hard_bounds_only():
    """Every loop pass's measurement goes through the guard, which is live
    only when the hard coil bounds are installed; a saturated pass inside a
    Fix C band resample is re-raised (rejects the draw), not masked."""
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    src = inspect.getsource(perturb_kinetic_equilibrium)
    sig = inspect.signature(perturb_kinetic_equilibrium).parameters
    assert sig["coil_saturation_guard"].default is None
    assert ("if (coil_saturation_guard is not None\n"
            "                           and _stashed_bounds is not None)"
            in src)
    meas = src.split("def _jl_meas():", 1)[1].split("def _jl_eval", 1)[0]
    assert "_hard_guard(" in meas
    i_sat = src.index("except CoilSaturated:")
    i_gen = src.index('_count_masked_anchor_failure("band_resample", _rs_exc)')
    assert i_sat < i_gen
