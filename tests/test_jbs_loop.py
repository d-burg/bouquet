"""The self-consistent bootstrap loop -- fast half (no GS solver).

Two things are tested here without a live solver:

* **The kernel** (:func:`bouquet.jbs_loop.run_jbs_loop`) on synthetic
  fixed-point maps whose answer is known in closed form: the residual
  definitions, the two-consecutive-pass rule, the relaxation schedule, the
  failure policy ("raise" / "flag", early abort at the relaxation floor), and
  the three plan tests that are properties of the ITERATION rather than of the
  physics -- (c) a loop started at its fixed point returns it with first-pass
  residuals below tolerance, (d) two different initial guesses converge to the
  same fixed point, (e) non-convergence raises (default) and is recorded in
  "flag" mode.

* **The evaluator** (:func:`bouquet.physics.evaluate_jBS`) on a mock
  equilibrium with analytic flux-surface geometry: that the geometry is
  sampled on the CALLER'S surfaces (distinct clipped values only, never a
  repeated surface, psi_pad unchanged), that gradients are taken on the true
  grid, and -- the defect-A regression -- that the same physical profiles
  sampled on a uniform and on a strongly non-uniform psi_N grid give the same
  j_BS(psi_N) to interpolation accuracy, while the legacy "evenly sampled"
  reading of the non-uniform array does not.  Needs OFT's pure-Python
  ``bootstrap`` module (Redl); skipped with a reason when OFT is absent.

The live halves (bit-level agreement with ``solve_with_bootstrap``'s first
pass, the loop in every baseline mode, the sigma=0 invariants, the MSE stage)
are in ``test_jbs_loop_solver.py`` (``pytest -m solver``).

Synthetic inputs only; no device data.
"""
import numpy as np
import pytest

from bouquet.jbs_loop import (JBS_RELAX_FLOOR, JBSNotConverged,
                              check_delivered, jbs_settings, jsonable,
                              profile_residuals, run_jbs_loop,
                              validate_jbs_settings, weighted_norm)


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


_X = np.linspace(0.0, 1.0, 101)
_W = 1.0 + 0.5 * _X            # positive "Ip weights" on the grid
_IP = 1.0e6


def _shape(x=_X):
    return 3.0e5 * np.exp(-0.5 * ((x - 0.93) / 0.03) ** 2) + 1.0e5 * (1 - x)


def _affine_problem(contraction, fixed_scale=1.0):
    """A fixed-point map T(j) = J* + c (j - J*), with an l_i that follows the
    iterate: its fixed point is J* = fixed_scale*shape, reached geometrically
    at rate |c| (relaxed: |1 - omega (1 - c)|)."""
    Jstar = fixed_scale * _shape()
    state = {}

    def step(jbs, k):
        state["j"] = np.asarray(jbs, dtype=float).copy()
        li = 1.0 + 1e-7 * float(np.trapezoid(_W * jbs, _X))
        return dict(w=_W, x=_X, li=li, q0=1.0 + 1e-9 * float(jbs[0]))

    def evaluate(meas):
        return Jstar + contraction * (state["j"] - Jstar)

    return Jstar, step, evaluate


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def test_defaults_are_the_approved_values_and_on():
    from bouquet.config import GenerationConfig
    g = GenerationConfig()
    assert g.jbs_self_consistent is True
    s = jbs_settings(g)
    assert s["enabled"] is True
    assert (s["rtol_j"], s["rtol_Ip"], s["tol_li"], s["tol_q0"]) == \
        (1e-3, 1e-4, 1e-3, 2e-3)
    # ceilings: 8 baseline, 12 per draw loop, 6 post-homotopy (limits; the
    # two-consecutive rule and every tolerance above are unchanged).  The
    # post-homotopy ceiling was 4: an owner-approved change of a pass
    # ceiling on its measured need (5) plus one pass.
    assert s["max_passes"] == 12 and jbs_settings(g, draw=True)[
        "max_passes"] == 12
    assert g.jbs_max_passes_post_homotopy == 6
    assert s["post_homotopy_passes"] == 6
    assert jbs_settings(g, draw=True)["post_homotopy_passes"] == 6
    from bouquet.jbs_loop import JBS_POST_HOMOTOPY_PASSES
    assert JBS_POST_HOMOTOPY_PASSES == 6
    assert s["relax"] == 0.7 and s["relax_floor"] == 0.25
    assert s["on_fail"] == "raise" and s["init"] == "anchor"
    assert s["required_consecutive"] == 2
    # the joint relaxation defaults
    assert g.jbs_relax_current == 0.7 and s["relax_current"] == 0.7
    assert g.jbs_relax_halve_on == 3 and s["relax_halve_on"] == 3


@pytest.mark.parametrize("field, bad", [
    ("jbs_init", "sbw"), ("jbs_loop_on_fail", "ignore"),
    ("jbs_rtol_j", 0.0), ("jbs_rtol_Ip", -1e-4), ("jbs_tol_li", float("nan")),
    ("jbs_tol_q0", "x"), ("jbs_max_passes", 1), ("jbs_max_passes_draw", 2.5),
    ("jbs_max_passes_post_homotopy", 1), ("jbs_max_passes_post_homotopy", 2.5),
    ("jbs_max_passes_post_homotopy", True),
    ("jbs_relax", 0.1), ("jbs_relax", 1.5), ("jbs_self_consistent", "yes"),
    ("jbs_relax_current", 0.0), ("jbs_relax_current", 1.2),
    ("jbs_relax_current", "x"), ("jbs_relax_halve_on", 0),
    ("jbs_relax_halve_on", 2.5), ("jbs_relax_halve_on", True),
])
def test_malformed_settings_are_refused_by_name(field, bad):
    with pytest.raises(ValueError, match=field):
        validate_jbs_settings(_GC(**{field: bad}))


def test_bouquet_config_validates_the_loop_fields():
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ImasSource, SolverConfig)
    with pytest.raises(ValueError, match="jbs_loop_on_fail"):
        BouquetConfig(source=ImasSource(ids_path="x.json"),
                      solver=SolverConfig(mesh_path="m.h5"), output_header="t",
                      generation=GenerationConfig(jbs_loop_on_fail="warn"))


def test_config_round_trips_the_loop_fields():
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ImasSource, SolverConfig)
    cfg = BouquetConfig(source=ImasSource(ids_path="x.json"),
                        solver=SolverConfig(mesh_path="m.h5"), output_header="t",
                        generation=GenerationConfig(
                            reconstruction_engine="legacy",
                            jbs_self_consistent=True, jbs_init="swb",
                            jbs_loop_on_fail="flag", jbs_max_passes=5))
    back = BouquetConfig.from_dict(cfg.to_dict())
    g = back.generation
    assert (g.jbs_self_consistent, g.jbs_init, g.jbs_loop_on_fail,
            g.jbs_max_passes) == (True, "swb", "flag", 5)


def test_an_old_config_without_the_fields_loads_with_the_loop_off():
    """A stored config that predates the loop was produced by the frozen
    path; replaying it must not silently switch the bootstrap model."""
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ImasSource, SolverConfig)
    d = BouquetConfig(source=ImasSource(ids_path="x.json"),
                      solver=SolverConfig(mesh_path="m.h5"), output_header="t",
                      generation=GenerationConfig(
                          reconstruction_engine="legacy")).to_dict()
    for k in list(d["generation"]):
        if k.startswith("jbs_") and k != "jbs_delta_mode":
            del d["generation"][k]
    with pytest.warns(UserWarning, match="predates the self-consistent"):
        g = BouquetConfig.from_dict(d).generation
    assert g.jbs_self_consistent is False
    # the other loop fields take their (inert) defaults
    assert g.jbs_max_passes == 12 and g.jbs_relax == 0.7


def test_a_current_config_round_trips_the_default_on_without_a_warning():
    import warnings
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ImasSource, SolverConfig)
    d = BouquetConfig(source=ImasSource(ids_path="x.json"),
                      solver=SolverConfig(mesh_path="m.h5"), output_header="t",
                      generation=GenerationConfig()).to_dict()
    assert d["generation"]["jbs_self_consistent"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = BouquetConfig.from_dict(d)
    assert back.generation.jbs_self_consistent is True


def test_legacy_flag_round_trips():
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ImasSource, SolverConfig)
    cfg = BouquetConfig(source=ImasSource(ids_path="x.json"),
                        solver=SolverConfig(mesh_path="m.h5"), output_header="t",
                        generation=GenerationConfig(
                            reconstruction_engine="legacy",
                            jbs_self_consistent=False))
    assert BouquetConfig.from_json(cfg.to_json()).generation\
        .jbs_self_consistent is False


@pytest.mark.parametrize("kw, match", [
    (dict(single_profile_jphi=True), "single_profile_jphi"),
    (dict(recalculate_j_BS=False), "recalculate_j_BS"),
])
def test_modes_without_a_bootstrap_to_iterate_are_refused_under_the_default(
        kw, match):
    """With the loop ON by default, a mode that has no bootstrap to iterate
    is refused with the fix in the message -- never silently downgraded --
    and runs once the legacy flag is set."""
    from bouquet.config import GenerationConfig
    from bouquet.run import Bouquet
    with pytest.raises(ValueError, match=match) as ei:
        Bouquet._check_jbs_loop_workflow(GenerationConfig(**kw))
    assert "jbs_self_consistent=False" in str(ei.value)
    Bouquet._check_jbs_loop_workflow(
        GenerationConfig(jbs_self_consistent=False, **kw))


# ---------------------------------------------------------------------------
#  residuals
# ---------------------------------------------------------------------------
def test_residuals_are_current_weighted_and_unrelaxed():
    J = _shape()
    jb = 0.99 * J
    r = profile_residuals(J, jb, _W, _X, _IP)
    assert r["r_j"] == pytest.approx(0.01, rel=1e-12)
    I = float(np.trapezoid(_W * J, _X))
    assert r["r_I"] == pytest.approx(0.01 * I / _IP, rel=1e-12)
    assert r["I_BS"] == pytest.approx(I, rel=1e-12)
    # the norm weights by |w|: a change where w is large counts more
    d = np.zeros_like(_X)
    d[-1] = 1.0
    d0 = np.zeros_like(_X)
    d0[0] = 1.0
    assert weighted_norm(d, _W, _X) > weighted_norm(d0, _W, _X)


def test_check_delivered_uses_the_same_bars():
    s = jbs_settings(_GC())
    J = _shape()
    assert check_delivered(J, J * (1 - 5e-4), _W, _X, _IP, s)["ok"]
    assert not check_delivered(J, J * (1 - 5e-3), _W, _X, _IP, s)["ok"]


# ---------------------------------------------------------------------------
#  the kernel
# ---------------------------------------------------------------------------
def test_contraction_converges_on_two_consecutive_passing_passes():
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(-0.1)
    out = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                       meas0=dict(li=0.0), gate_li=True)
    rec = out["record"]
    assert out["converged"] and rec["converged"]
    ok = rec["pass_ok"]
    assert ok[-1] and ok[-2] and not any(ok[:-2] if len(ok) > 2 else [])
    assert rec["n_passes"] == len(rec["r_j"]) <= s["max_passes"]
    # delivered: within tolerance of the fixed point
    assert profile_residuals(Jstar, out["jbs_used"], _W, _X, _IP)["r_j"] \
        <= s["rtol_j"] / (1 + 0.1)
    # the record is JSON-safe and carries the plan's fields
    rj = jsonable(rec)
    for key in ("enabled", "init", "grid", "n_passes", "converged",
                "tolerances", "omega", "r_j", "r_I", "dl_i", "dq0", "I_BS",
                "jBS_peak_psiN", "jBS_peak", "wall_s",
                "evaluate_jBS_version", "oft_build"):
        assert key in rj, key
    import json
    json.dumps(rj)


def test_c_started_at_the_fixed_point_returns_it_in_one_pass():
    """(c): the first pass already meets every criterion and returns the fixed
    point unchanged; the loop stops at the minimum the two-consecutive rule
    allows (2 passes)."""
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(0.3)
    li_star = 1.0 + 1e-7 * float(np.trapezoid(_W * Jstar, _X))
    out = run_jbs_loop(Jstar.copy(), step, ev, s, Ip=_IP,
                       meas0=dict(li=li_star, q0=None), gate_li=True)
    rec = out["record"]
    assert rec["pass_ok"][0], "the first pass at the fixed point must pass"
    assert rec["r_j"][0] <= 1e-15 and rec["r_I"][0] <= 1e-15
    assert rec["n_passes"] == 2 and out["converged"]
    # (the relaxed update (1-w)J* + wJ* reproduces J* to rounding)
    np.testing.assert_allclose(out["jbs_used"], Jstar, rtol=1e-14, atol=0)


def test_d_two_initial_guesses_reach_the_same_fixed_point():
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(-0.4)
    a = run_jbs_loop(0.2 * Jstar, step, ev, s, Ip=_IP,
                     meas0=dict(li=0.0), init="anchor")
    Jstar2, step2, ev2 = _affine_problem(-0.4)
    b = run_jbs_loop(3.0 * Jstar + 1e4, step2, ev2, s, Ip=_IP,
                     meas0=dict(li=0.0), init="swb")
    assert a["converged"] and b["converged"]
    d = profile_residuals(a["jbs_used"], b["jbs_used"], _W, _X, _IP)
    assert d["r_j"] <= 2 * s["rtol_j"]
    assert b["record"]["init"] == "swb"


def _draw_problem(a_draw, a_base=1.0, eps=0.6, dL_p=0.01):
    """A perturbation draw in miniature.  The Redl bootstrap scales with the
    kinetics ``a`` and follows the geometry ``L`` (an l_i-like scalar) of the
    equilibrium it is evaluated on; ``L`` follows the solved total current
    and the pressure (the draw's pressure shifts it by ``dL_p``).  The
    baseline is self-consistent at ``a_base``.  The draw's state anchor is
    the BASELINE total (baseline inductive + baseline bootstrap) at the
    draw's pressure, exactly as in ``perturb_kinetic_equilibrium``.  Linear,
    so the fixed point is known in closed form.  Returns the two initial
    iterates: the unperturbed baseline bootstrap (the counterfactual) and the
    one a draw uses (Redl at the anchor on the draw's own kinetics)."""
    x = _X
    j_ind = 8e5 * (1 - x) ** 1.5 + 8e4
    Jshape = _shape()
    Lref, L0, mu = 1.0, 0.2, 1.0
    I = lambda j: float(np.trapezoid(_W * x * j, x)) / _IP  # noqa: E731
    Ij, IS = I(j_ind), I(Jshape)

    def redl(a, L):
        return a * Jshape * (1.0 + eps * (L - Lref))

    def fixed_L(a, dp):
        return ((L0 + dp + mu * Ij + mu * a * IS * (1.0 - eps * Lref))
                / (1.0 - mu * a * IS * eps))

    j_base = redl(a_base, fixed_L(a_base, 0.0))    # self-consistent baseline
    geom = lambda j: L0 + dL_p + mu * I(j)          # noqa: E731 (draw pressure)
    L_anchor = geom(j_ind + j_base)                 # the draw's state anchor
    st = dict(L=L_anchor)

    def step(jbs, k, relax=None):
        jc = j_ind + np.asarray(jbs, dtype=float)
        js = jc if relax is None else relax(jc)
        st["L"] = geom(js)
        return dict(w=_W, x=x, li=st["L"])

    def evaluate(meas):
        return redl(a_draw, st["L"])

    Lstar = fixed_L(a_draw, dL_p)
    return dict(step=step, evaluate=evaluate, st=st, L_anchor=L_anchor,
                old_init=j_base, new_init=redl(a_draw, L_anchor),
                Lstar=Lstar, Jstar=redl(a_draw, Lstar))


def test_draw_init_from_own_kinetics_and_from_the_baseline_reach_one_fixed_point():
    """Item 7f of the plan: a draw's loop starts from Redl at its anchor on
    its OWN perturbed kinetics.  That is initialisation only: started from the
    unperturbed baseline bootstrap instead (the counterfactual), the same draw
    reaches the same fixed point, the closed-form one, to the loop's own
    tolerances.  The own-kinetics start begins much closer to it, and the
    record says which start was used."""
    s = jbs_settings(_GC(), draw=True)
    outs = {}
    for tag in ("old", "new"):
        p = _draw_problem(a_draw=1.06)
        src = ("baseline j_BS (counterfactual)" if tag == "old" else
               "evaluate_jBS at the draw's anchor, draw's own kinetics")
        o = run_jbs_loop(p[f"{tag}_init"], p["step"], p["evaluate"], s,
                         Ip=_IP, meas0=dict(li=p["L_anchor"]), gate_li=True,
                         init_source=src)
        assert o["converged"], (tag, o["record"]["r_j"])
        assert o["record"]["init_source"] == src
        # on the closed-form fixed point, to the loop's own tolerances
        assert abs(p["st"]["L"] - p["Lstar"]) <= s["tol_li"]
        assert profile_residuals(p["Jstar"], o["jbs_used"], _W, _X,
                                 _IP)["r_j"] <= s["rtol_j"]
        outs[tag] = o
    a, b = outs["old"], outs["new"]
    assert profile_residuals(a["jbs_used"], b["jbs_used"], _W, _X,
                             _IP)["r_j"] <= s["rtol_j"]
    # the old start carries the kinetic perturbation as residual, the new
    # one only the anchor -> first-solve geometry step
    assert b["record"]["r_j"][0] < 0.5 * a["record"]["r_j"][0]
    assert b["record"]["n_passes"] <= a["record"]["n_passes"]


def test_the_init_source_defaults_to_unrecorded():
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(0.3)
    o = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP, meas0=dict(li=0.0),
                     max_passes=2, raise_on_fail=False)
    assert "init_source" in o["record"] and o["record"]["init_source"] is None
    assert o["record"]["tolerances"]["post_homotopy_passes"] == 4


def test_e_non_convergence_raises_with_the_history():
    s = jbs_settings(_GC(jbs_max_passes=4))
    Jstar, step, ev = _affine_problem(0.95)      # contracts too slowly
    with pytest.raises(JBSNotConverged) as ei:
        run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP, meas0=dict(li=0.0))
    rec = ei.value.record
    assert rec["converged"] is False and rec["n_passes"] == 4
    assert len(rec["r_j"]) == 4 and "pass ceiling" in rec["stop_reason"]
    assert ei.value.history is rec


def test_e_flag_mode_returns_the_last_iterate_and_records_it():
    from bouquet.jbs_loop import flag_reason, JBS_FLAG_PREFIX
    s = jbs_settings(_GC(jbs_max_passes=3, jbs_loop_on_fail="flag"))
    Jstar, step, ev = _affine_problem(0.95)
    out = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP, meas0=dict(li=0.0))
    assert out["converged"] is False
    assert out["record"]["jbs_converged"] is False
    assert out["record"]["fail_message"]
    assert flag_reason(out["record"]).startswith(JBS_FLAG_PREFIX)


def _growing_problem():
    J0 = _shape()
    shape2 = np.sin(np.pi * _X) * 1e5
    cnt = {"k": 0, "j": None}

    def step(jbs, k):
        cnt["k"] = k
        cnt["j"] = np.asarray(jbs, dtype=float).copy()
        return dict(w=_W, x=_X, li=1.0)

    def ev(meas):                       # a residual that grows 3x per pass
        return cnt["j"] + 1e-3 * 3.0 ** cnt["k"] * shape2

    return J0, step, ev


def test_growing_residual_halves_omega_to_the_floor_then_aborts():
    """The earlier schedule (``jbs_relax_halve_on=1``, halve on EVERY growth):
    omega 0.7 -> 0.35 -> 0.25 (floor); three growing passes AT the floor abort
    before the pass ceiling."""
    s = jbs_settings(_GC(jbs_max_passes=12, jbs_relax_halve_on=1))
    J0 = _shape()
    shape2 = np.sin(np.pi * _X) * 1e5
    cnt = {"k": 0, "j": None}

    def step(jbs, k):
        cnt["k"] = k
        cnt["j"] = np.asarray(jbs, dtype=float).copy()
        return dict(w=_W, x=_X, li=1.0)

    def ev(meas):                       # a residual that grows 3x per pass
        return cnt["j"] + 1e-3 * 3.0 ** cnt["k"] * shape2

    with pytest.raises(JBSNotConverged) as ei:
        run_jbs_loop(J0, step, ev, s, Ip=_IP, meas0=dict(li=1.0))
    rec = ei.value.record
    om = [w for w in rec["omega"] if w is not None]
    assert om[:3] == pytest.approx([0.7, 0.35, 0.25])
    assert min(om) == pytest.approx(JBS_RELAX_FLOOR)
    assert all(b > a for a, b in zip(rec["r_j"], rec["r_j"][1:]))
    assert "relaxation floor" in rec["stop_reason"]
    assert rec["n_passes"] == 6 < 12


def test_default_schedule_halves_only_on_sustained_growth():
    """Default ``jbs_relax_halve_on=3``: omega stays 0.7 through two growing
    passes and halves on the third (0.7 -> 0.35 -> 0.25); the abort rule is
    unchanged -- three growing passes AT the floor stop the loop."""
    s = jbs_settings(_GC(jbs_max_passes=12))
    J0, step, ev = _growing_problem()
    with pytest.raises(JBSNotConverged) as ei:
        run_jbs_loop(J0, step, ev, s, Ip=_IP, meas0=dict(li=1.0))
    rec = ei.value.record
    om = [w for w in rec["omega"] if w is not None]
    assert om == pytest.approx([0.7, 0.7, 0.7, 0.35, 0.35, 0.35,
                                0.25, 0.25, 0.25])
    assert rec["omega_halved_at_pass"] == [4, 7]
    assert rec["relax_halve_on"] == 3
    assert "relaxation floor" in rec["stop_reason"]
    assert rec["n_passes"] == 10 < 12


def test_an_oscillating_residual_does_not_halve_omega_by_default():
    """r_j that grows and shrinks on alternate passes (the forced response of
    the closure <-> geometry mode) never halves omega under the default; the
    earlier schedule (halve_on=1) halved it on every growth event."""
    shape2 = np.sin(np.pi * _X) * 1e5

    def make():
        cnt = {"k": 0, "j": None}

        def step(jbs, k):
            cnt["k"] = k
            cnt["j"] = np.asarray(jbs, dtype=float).copy()
            return dict(w=_W, x=_X, li=1.0)

        def ev(meas):
            amp = 2e-2 if cnt["k"] % 2 else 1e-2
            return cnt["j"] + amp * shape2
        return step, ev

    st, ev = make()
    out = run_jbs_loop(_shape(), st, ev,
                       jbs_settings(_GC(jbs_max_passes=8,
                                        jbs_loop_on_fail="flag")),
                       Ip=_IP, meas0=dict(li=1.0))
    om = [w for w in out["record"]["omega"] if w is not None]
    assert om and all(w == 0.7 for w in om)
    assert out["record"]["omega_halved_at_pass"] == []
    st, ev = make()
    out1 = run_jbs_loop(_shape(), st, ev,
                        jbs_settings(_GC(jbs_max_passes=8,
                                         jbs_loop_on_fail="flag",
                                         jbs_relax_halve_on=1)),
                        Ip=_IP, meas0=dict(li=1.0))
    assert min(w for w in out1["record"]["omega"] if w is not None) \
        == pytest.approx(JBS_RELAX_FLOOR)


# ---------------------------------------------------------------------------
#  relaxation of the solved current (jbs_relax_current, beta)
# ---------------------------------------------------------------------------
def _two_state_problem(g=-0.55, eps=0.2):
    """closure <-> geometry: each pass closes on the PREVIOUS equilibrium's
    geometry L_prev (an l_i-like scalar) and over-corrects, so the solved
    geometry swings back by the fraction ``g`` per pass -- the slow
    mode, on which omega does not act.  The bootstrap follows the geometry
    weakly (``eps``).  Linear, so the fixed point is known in closed form."""
    x = _X
    phi = x.copy()
    j_ind0 = 8e5 * (1 - x) ** 1.5 + 8e4
    Jshape = _shape()
    Lt, L0, mu = 1.0, 0.2, 1.0
    I = lambda j: float(np.trapezoid(_W * phi * j, x)) / _IP  # noqa: E731
    A, B = mu * I(j_ind0), mu * I(Jshape)
    kappa = -g / A
    st = dict(L=Lt)

    def step(jbs, k, relax=None):
        s = 1.0 + kappa * (Lt - st["L"])
        jc = s * j_ind0 + np.asarray(jbs, dtype=float)
        js = jc if relax is None else relax(jc)
        st["L"] = L0 + mu * I(js)
        st["jc"], st["js"] = jc, np.asarray(js, dtype=float)
        return dict(w=_W, x=x, li=st["L"])

    def evaluate(meas):
        return Jshape * (1.0 + eps * (st["L"] - Lt))

    Lstar = ((L0 + A * (1 + kappa * Lt) + B * (1 - eps * Lt))
             / (1 + A * kappa - B * eps))
    Jstar = Jshape * (1.0 + eps * (Lstar - Lt))
    return step, evaluate, st, Lstar, Jstar


def test_current_relaxation_changes_the_path_not_the_fixed_point():
    """The fixed point reached with beta = 0.7 and without it (beta = 1)
    agrees to the loop tolerances, and both sit on the closed-form fixed
    point; the relaxed run needs fewer passes on the oscillating mode."""
    outs = {}
    for beta in (1.0, 0.7):
        step, ev, st, Lstar, Jstar = _two_state_problem()
        s = jbs_settings(_GC(jbs_relax_current=beta, jbs_max_passes=40))
        o = run_jbs_loop(Jstar * 0.6, step, ev, s, Ip=_IP,
                         meas0=dict(li=st["L"]), gate_li=True)
        assert o["converged"], beta
        outs[beta] = (o, st["L"])
        tol = s
        # on the closed-form fixed point, to the loop's own tolerances
        assert abs(st["L"] - Lstar) <= tol["tol_li"]
        assert profile_residuals(Jstar, o["jbs_used"], _W, _X,
                                 _IP)["r_j"] <= tol["rtol_j"]
    (a, La), (b, Lb) = outs[1.0], outs[0.7]
    assert abs(La - Lb) <= s["tol_li"]
    assert profile_residuals(a["jbs_used"], b["jbs_used"], _W, _X,
                             _IP)["r_j"] <= s["rtol_j"]
    assert b["record"]["n_passes"] < a["record"]["n_passes"]
    # the record: beta, and the per-pass solved-vs-closure gap
    ra, rb = a["record"], b["record"]
    assert ra["relax_current"] == 1.0 and rb["relax_current"] == 0.7
    assert all(g == 0.0 for g in ra["current_gap"])
    assert not any(ra["current_blended"])
    assert rb["current_gap"][0] == 0.0 and rb["current_blended"][0] is False
    assert all(rb["current_blended"][1:])
    assert rb["current_gap"][-1] < rb["current_gap"][1]
    assert rb["final"]["current_gap"] == rb["current_gap"][-1]
    import json
    json.dumps(jsonable(rb))


def test_current_relaxation_is_the_identity_at_the_fixed_point():
    """(c) with beta: started at the fixed point, the blend of two equal
    currents is the identity and the loop returns it unchanged."""
    step, ev, st, Lstar, Jstar = _two_state_problem()
    st["L"] = Lstar
    s = jbs_settings(_GC())
    o = run_jbs_loop(Jstar.copy(), step, ev, s, Ip=_IP,
                     meas0=dict(li=Lstar), gate_li=True)
    assert o["converged"] and o["record"]["n_passes"] == 2
    assert max(o["record"]["current_gap"]) <= 1e-14
    assert abs(st["L"] - Lstar) <= 1e-12


def test_a_step_without_the_relaxer_is_recorded_as_unrelaxed():
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(-0.1)
    out = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP, meas0=dict(li=0.0))
    rec = out["record"]
    assert rec["relax_current"] is None
    assert rec["current_relaxation"].startswith("not applied")
    assert rec["current_gap"] == [None] * rec["n_passes"]


def test_current_relaxer_blends_against_the_previous_pass_only():
    from bouquet.jbs_loop import CurrentRelaxer
    r = CurrentRelaxer(0.7)
    a, b, c = np.full(3, 1.0), np.full(3, 2.0), np.full(3, 4.0)
    r.begin_pass()
    np.testing.assert_array_equal(r(a), a)          # first pass: no blend
    r.commit()
    r.begin_pass()
    np.testing.assert_allclose(r(b), 0.3 * a + 0.7 * b)
    # a second call in the SAME pass blends against the same base
    np.testing.assert_allclose(r(c), 0.3 * a + 0.7 * c)
    gap, blended = r.gap()
    assert blended and gap == pytest.approx(
        np.linalg.norm(0.3 * a + 0.7 * c - c) / np.linalg.norm(c))
    r.commit()
    r.begin_pass()
    np.testing.assert_allclose(r(c), 0.3 * (0.3 * a + 0.7 * c) + 0.7 * c)
    one = CurrentRelaxer(1.0)
    one.begin_pass(); one(a); one.commit(); one.begin_pass()
    np.testing.assert_array_equal(one(b), b)


def test_relaxation_changes_the_path_not_the_fixed_point():
    outs = []
    for w in (1.0, 0.7, 0.5):
        J2, st2, ev2 = _affine_problem(-0.1)
        o = run_jbs_loop(0.5 * J2, st2, ev2,
                         jbs_settings(_GC(jbs_relax=w, jbs_max_passes=16)),
                         Ip=_IP, meas0=dict(li=0.0))
        assert o["converged"]
        outs.append(o["jbs_used"])
    for o in outs[1:]:
        assert profile_residuals(outs[0], o, _W, _X, _IP)["r_j"] <= 3e-3


def test_dq0_gates_only_when_asked():
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(-0.1)
    out = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                       meas0=dict(li=0.0, q0=0.0), gate_q0=True)
    assert out["record"]["criteria"]["dq0"] is True
    assert all(v is not None for v in out["record"]["dq0"])


def test_the_first_pass_without_a_previous_l_i_cannot_pass():
    """meas0=None: pass 1 has no dl_i, so it cannot count toward convergence
    -- the rule is never satisfied by a criterion that was not measured."""
    s = jbs_settings(_GC())
    Jstar, step, ev = _affine_problem(0.0)
    li_star = 1.0 + 1e-7 * float(np.trapezoid(_W * Jstar, _X))
    out = run_jbs_loop(Jstar.copy(), step, ev, s, Ip=_IP, meas0=None)
    assert out["record"]["pass_ok"][0] is False
    assert out["record"]["n_passes"] == 3
    assert out["record"]["li"][0] == pytest.approx(li_star)


# ---------------------------------------------------------------------------
#  the evaluator on a mock equilibrium
# ---------------------------------------------------------------------------
_bs = pytest.importorskip(
    "OpenFUSIONToolkit.TokaMaker.bootstrap",
    reason="evaluate_jBS wraps OFT's pure-Python Redl implementation; "
           "OpenFUSIONToolkit is not importable here")


class _MockEq:
    """Analytic flux-surface geometry of a shaped D3D-like plasma.

    Every getter is a smooth function of psi_N and RECORDS the sample array it
    was asked for, so the tests can see exactly which surfaces the evaluator
    queried.  Dict layouts (current OFT); ``legacy=True`` returns the
    positional-array layout of older builds.
    """

    R0, a, B0 = 1.7, 0.6, 2.0

    def __init__(self, legacy=False, psi_bounds=(-0.9, 0.1)):
        self.legacy = legacy
        self.psi_bounds = np.asarray(psi_bounds, dtype=float)
        self.calls = []

    def _geo(self, psi):
        psi = np.asarray(psi, dtype=float)
        r = self.a * np.sqrt(psi)
        eps = r / self.R0
        R = self.R0 * (1.0 + 0.1 * psi)
        return psi, r, eps, R

    def get_profiles(self, psi=None, **kw):
        self.calls.append(("get_profiles", np.array(psi)))
        psi, r, eps, R = self._geo(psi)
        F = self.R0 * self.B0 * (1.0 - 0.02 * psi)
        return psi, F, 0 * F, 0 * F, 0 * F

    def sauter_fc(self, psi=None, **kw):
        self.calls.append(("sauter_fc", np.array(psi)))
        psi, r, eps, R = self._geo(psi)
        fc = 1.0 - 1.46 * np.sqrt(eps)
        rav = {"<R>": R, "<1/R>": 1.0 / R * (1 + eps ** 2 / 2), "<a>": r}
        modb = np.vstack([np.full_like(psi, self.B0),
                          (self.B0 ** 2) * (1 + eps ** 2)])
        if self.legacy:
            rav = np.vstack([rav["<R>"], rav["<1/R>"], rav["<a>"]])
        return (psi, fc, rav, modb) + ((r / R,) if kw.get("return_eps") else ())

    def get_q(self, psi=None, **kw):
        self.calls.append(("get_q", np.array(psi)))
        psi, r, eps, R = self._geo(psi)
        q = 1.05 + 2.5 * psi ** 2
        rav = {"<R>": R, "<1/R>": 1.0 / R, "<1/R^2>": 1.0 / R ** 2,
               "dV/dPsi": 2 * np.pi ** 2 * self.a ** 2 * self.R0 * np.ones_like(psi)}
        if self.legacy:
            rav = np.vstack([rav["<R>"], rav["<1/R>"], rav["dV/dPsi"]])
        return psi, q, rav, None, None, None


def _kin(x):
    """Physical H-mode-like profiles as functions of psi_N (pedestal ~0.95)."""
    ped = 0.5 * (1 - np.tanh((x - 0.95) / 0.02))
    ne = 1.0e19 * (0.3 + 3.0 * ped * (1 - 0.3 * x ** 2))
    te = 30.0 + 3000.0 * ped * (1 - 0.6 * x ** 2)
    ti = 30.0 + 2800.0 * ped * (1 - 0.6 * x ** 2)
    ni = 0.85 * ne
    zeff = 1.8 + 0 * x
    return ne, te, ni, ti, zeff


def test_geometry_is_sampled_on_the_callers_surfaces_without_repeats():
    """The IMAS-type grid [0, 1.73e-4, 6.92e-4, 1.557e-3, ...] puts three
    surfaces inside psi_pad = 1e-3: they must NOT be merged or dropped (each
    keeps its own profile value and gradient) and psi_pad must NOT change;
    only their geometry is looked up at psi_pad -- once."""
    from bouquet.physics import evaluate_jBS
    # uniform in a rho-like label: psi_N = (k/76)^2 -> [0, 1.73e-4, 6.92e-4,
    # 1.558e-3, 2.77e-3, ...]
    psi = (np.arange(77) / 76.0) ** 2
    np.testing.assert_allclose(psi[1:4], [1.73e-4, 6.92e-4, 1.558e-3],
                               rtol=2e-3)
    assert np.all(np.diff(psi) > 0)
    eq = _MockEq()
    j, d = evaluate_jBS(eq, psi, *_kin(psi), psi_pad=1e-3, smooth_axis=False)
    assert j.shape == psi.shape and np.all(np.isfinite(j))
    assert d["psi_pad"] == 1e-3
    for name, arr in eq.calls:
        assert np.all(np.diff(arr) > 0), f"{name} was handed a repeated surface"
        assert arr[0] == pytest.approx(1e-3) and arr[-1] <= 1 - 1e-3
    # the 3 points inside the pad share ONE geometry sample, nothing else
    assert d["n_geometry_surfaces"] == psi.size - 2
    np.testing.assert_array_equal(d["psi_eval"][:3], [1e-3] * 3)
    assert d["psi_eval"][3] == psi[3]
    # ... but each keeps its own profile values and TRUE-grid gradient, so
    # their Redl drives stay distinct (no surface was merged)
    assert len(np.unique(d["j_dot_B"][:3])) == 3


def test_gradients_are_on_the_true_grid_and_the_current_flux_range():
    from bouquet.physics import evaluate_jBS
    x = np.linspace(0, 1, 201)
    eq1 = _MockEq(psi_bounds=(-0.9, 0.1))
    eq2 = _MockEq(psi_bounds=(-1.9, 0.1))          # twice the flux range
    j1, _ = evaluate_jBS(eq1, x, *_kin(x), smooth_axis=False)
    j2, _ = evaluate_jBS(eq2, x, *_kin(x), smooth_axis=False)
    # the Redl drive is linear in d/dpsi = (d/dpsi_N)/Delta_psi; collisional
    # terms do not depend on Delta_psi, so j scales exactly as 1/Delta_psi
    np.testing.assert_allclose(j2, 0.5 * j1, rtol=1e-12, atol=1e-9)


def test_legacy_array_layout_gives_the_same_answer():
    from bouquet.physics import evaluate_jBS
    x = np.linspace(0, 1, 151)
    ja, _ = evaluate_jBS(_MockEq(), x, *_kin(x))
    jb, _ = evaluate_jBS(_MockEq(legacy=True), x, *_kin(x))
    np.testing.assert_array_equal(ja, jb)


def test_b_uniform_and_non_uniform_grids_agree_defect_A_regression():
    """(b): the same physical profiles on a uniform and on a rho-uniform
    (strongly non-uniform) psi_N grid give the same j_BS(psi_N) to
    interpolation accuracy.  The legacy reading -- the non-uniform array
    differentiated as if evenly sampled (``solve_with_bootstrap`` without
    ``psi_N=``, defect A) -- misses by an order of magnitude more."""
    from bouquet.physics import evaluate_jBS
    xu = np.linspace(0.0, 1.0, 1001)
    ju, _ = evaluate_jBS(_MockEq(), xu, *_kin(xu), smooth_axis=False)
    # uniform in a rho-like label; fine enough to resolve the pedestal, so
    # what is measured is the GRID MAPPING, not the finite-difference
    # resolution of either grid
    rho = np.linspace(0.0, 1.0, 801)
    xn = rho ** 2 * (0.5 + 0.5 * rho)
    jn, _ = evaluate_jBS(_MockEq(), xn, *_kin(xn), smooth_axis=False)
    # legacy: the non-uniform ARRAYS read as if on an even grid
    xl = np.linspace(0.0, 1.0, xn.size)
    kin_n = _kin(xn)
    jl, _ = evaluate_jBS(_MockEq(), xl, *kin_n, smooth_axis=False)
    # compare on the reference (uniform) grid, current-weighted, away from
    # the tracer-clipped axis and separatrix samples
    sel = (xu > 0.02) & (xu < 0.995)
    w = np.ones(sel.sum())
    ref = ju[sel]
    new = np.interp(xu[sel], xn, jn)
    old = np.interp(xu[sel], xn, jl)     # legacy values, placed at their TRUE psi_N
    e_new = weighted_norm(new - ref, w, xu[sel]) / weighted_norm(ref, w, xu[sel])
    e_old = weighted_norm(old - ref, w, xu[sel]) / weighted_norm(ref, w, xu[sel])
    assert e_new < 0.02, f"non-uniform grid disagrees by {e_new:.3%}"
    assert e_old > 10 * e_new, (e_old, e_new)


def test_evaluate_refuses_bad_grids_by_name():
    from bouquet.physics import evaluate_jBS
    x = np.linspace(0, 1, 11)
    k = _kin(x)
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_jBS(_MockEq(), x[::-1], *k)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        evaluate_jBS(_MockEq(), x * 1.1, *k)
    with pytest.raises(ValueError, match="ne has shape"):
        evaluate_jBS(_MockEq(), x, k[0][:-1], *k[1:])
    with pytest.raises(ValueError, match="psi_pad"):
        evaluate_jBS(_MockEq(), x, *k, psi_pad=0.0)


def test_draw_composer_reproduces_the_legacy_composition_rules():
    """The per-draw composition: scale on the (isolated) spike only, floor,
    jBS_diff outside delta mode; delta mode = baseline + (scale*raw - ref)."""
    from bouquet.physics import evaluate_jBS
    from bouquet.TokaMaker_interface import (_draw_jbs_composer,
                                             smooth_jbs_transition)
    x = np.linspace(0, 1, 129)
    k = _kin(x)
    base, d = evaluate_jBS(_MockEq(), x, *k)
    diff = 1e3 * np.sin(3 * x)
    comp = _draw_jbs_composer(x, *k, 1e-3, False, 0.9, False, diff, None,
                              None)
    spike, full, _ = comp(_MockEq())
    np.testing.assert_allclose(spike, 0.9 * base + diff, rtol=1e-13)
    np.testing.assert_allclose(
        full, smooth_jbs_transition(d["j_tor_full_raw"]), rtol=1e-13)
    # delta mode: at sigma=0 (same state, ref at the same scale) the spike IS
    # the baseline split, whatever the evaluator's common-mode artifacts
    raw, _ = evaluate_jBS(_MockEq(), x, *k, smooth_axis=False)
    bl = 2e5 + 0 * x
    comp_d = _draw_jbs_composer(x, *k, 1e-3, False, 0.9, False, diff,
                                0.9 * raw, bl)
    spike_d, _, _ = comp_d(_MockEq())
    np.testing.assert_allclose(spike_d, bl, rtol=0, atol=1e-9)
    # floor
    comp_f = _draw_jbs_composer(x, *k, 1e-3, False, 1.0, True, -1e9 + 0 * x,
                                None, None)
    assert np.all(comp_f(_MockEq())[0] < 0)   # diff added AFTER the floor


# ---------------------------------------------------------------------------
#  the post-homotopy check of a draw (mocked solver)
# ---------------------------------------------------------------------------
class _PHEq:
    """Just enough of a TokaMaker for _post_homotopy_jbs."""

    def __init__(self):
        self.psi_bounds = (-0.15, 0.12)
        self.solves = 0
        self.ffp = []

    def copy_eq(self):
        return object()

    def get_stats(self, **kw):
        return {"l_i": 0.65}

    def set_targets(self, **kw):
        pass

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        self.ffp.append(np.asarray(ffp_prof["y"], float).copy())

    def solve(self):
        self.solves += 1


def _ph_setup(monkeypatch, kind):
    import bouquet.jbs_loop as L
    import bouquet.TokaMaker_interface as TI
    x = np.linspace(0.0, 1.0, 65)
    Jstar = _shape(x)
    calls = {"corr": [], "renorm": 0}
    monkeypatch.setattr(L, "residual_weights",
                        lambda eq, psi_N, psi_pad=1e-3, coord="psi_n": (np.ones_like(x), x,
                                                         "test"))

    def _renorm(mygs, psi_N, target, Ip, pad, label=""):
        calls["renorm"] += 1
        return np.asarray(target, float), 1.0

    def _corr(mygs, psi_N, target, pp, Ip, pax, pad, **kw):
        calls["corr"].append((np.asarray(target, float).copy(), kw))
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
    s = jbs_settings(_GC(), draw=True)
    return TI, x, Jstar, ctx, calls, s


def test_post_homotopy_standard_draw_reaches_the_total_by_corrective_iteration(
        monkeypatch):
    """A standard draw's stored j_phi is the ACHIEVED current of its
    corrective iteration; the post-homotopy passes must reach the new total
    the same way (target renormalised to Ip + corrective iteration), never by
    handing that achieved profile back to one jphi-linterp solve."""
    TI, x, Jstar, ctx, calls, s = _ph_setup(monkeypatch, "standard")
    eq = _PHEq()
    rec, spk, full, jphi = TI._post_homotopy_jbs(eq, ctx, s, x, 1e-3, _IP)
    assert not rec["accepted_without_passes"]
    assert eq.solves == 0, "a bare jphi-linterp solve was issued"
    assert calls["corr"] and calls["renorm"] == len(calls["corr"])
    omega = s["relax"]
    jbs0 = (1 - omega) * ctx["spike_used"] + omega * Jstar
    t0, kw0 = calls["corr"][0]
    np.testing.assert_allclose(t0, ctx["j_ind_used"] + jbs0, rtol=1e-12)
    assert kw0["min_iters"] == 2 and kw0["rtol"] == 0.05
    # the delivered current is the corrective iteration's output
    np.testing.assert_allclose(jphi, calls["corr"][-1][0] * 0.999)
    assert "corrective" in rec["solve"] and "beta not applied" in rec["solve"]
    assert rec["passes"]["converged"]


def test_post_homotopy_fixc_draw_is_one_relaxed_request_solve(monkeypatch):
    TI, x, Jstar, ctx, calls, s = _ph_setup(monkeypatch, "fixc")
    eq = _PHEq()
    rec, spk, full, jphi = TI._post_homotopy_jbs(eq, ctx, s, x, 1e-3, _IP)
    assert eq.solves >= 1 and not calls["corr"]
    assert "Fix C" in rec["solve"]
    assert rec["passes"]["converged"]


def test_post_homotopy_inside_tolerance_keeps_the_draw(monkeypatch):
    TI, x, Jstar, ctx, calls, s = _ph_setup(monkeypatch, "standard")
    ctx["spike_used"] = Jstar.copy()
    eq = _PHEq()
    rec, spk, full, jphi = TI._post_homotopy_jbs(eq, ctx, s, x, 1e-3, _IP)
    assert rec["accepted_without_passes"] and eq.solves == 0
    assert not calls["corr"]


def test_the_post_homotopy_ceiling_is_not_read_with_the_loop_off():
    """The approved 4 -> 6 change of the post-homotopy ceiling cannot reach
    the frozen legacy path: with ``jbs_self_consistent=False`` the draws are
    handed no loop settings at all (``generate()`` passes ``None``), and the
    post-homotopy stage runs only on loop settings."""
    import inspect
    import re
    from bouquet.config import GenerationConfig
    from bouquet import TokaMaker_interface as TI
    from bouquet.run import Bouquet
    g = GenerationConfig(jbs_self_consistent=False)
    assert jbs_settings(g, draw=True)["enabled"] is False
    src = inspect.getsource(Bouquet.generate)
    assert re.search(r'jbs_loop=\(_jbs_draw if _jbs_draw\["enabled"\] '
                     r'else None\)', src)
    gen = inspect.getsource(TI.generate_bouquet)
    calls = [m.start() for m in re.finditer(r"_post_homotopy_jbs\(", gen)]
    assert len(calls) == 1
    # the one call sits under a guard that requires the loop settings
    head = gen[:calls[0]]
    guard = head[head.rindex("if (_jctx is not None and jbs_loop"):]
    assert "and jbs_loop" in guard and guard.count("\n") < 12
    # nothing else in the package reads the ceiling
    import glob
    import os
    root = os.path.dirname(inspect.getsourcefile(TI))
    readers = set()
    for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        with open(p) as fh:
            for ln in fh:
                if ln.lstrip().startswith("#"):
                    continue
                if re.search(r'\[\s*"post_homotopy_passes"\s*\]|'
                             r'get\(\s*"post_homotopy_passes"', ln):
                    readers.add(os.path.basename(p))
    assert readers <= {"TokaMaker_interface.py", "engine_draws.py",
                       "jbs_loop.py"}, readers
