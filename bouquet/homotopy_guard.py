"""The homotopy branch guard: an opt-in EXPERIMENTAL DIAGNOSTIC for bistable
equilibria (issue #87, still open; ``GenerationConfig.guard_bistability``,
default ``False``).

Status (1.4.1 validation).  It watches the coil-homotopy stages only.  On
the validated bistable case the draws leave the reconstruction's branch
earlier, in the draw's bootstrap loop, which it does not watch: it stamped 7
of 28 draws there and changed none ("held" never occurred).  On two
single-solution slices it flagged 5 of 62 draws, all continuous moves of
1.0-1.6 % of a, and left them unchanged.  Use it to locate where a draw
leaves its branch, not as a fix; the loop stage is #87's next target.

On a near-double-null slice the free-boundary problem can have two
neighboring solutions of the same inputs (docs/workflows.md, "Bistable
equilibria").  A draw's coil homotopy changes the objective the solver
iterates on -- the loop's coil regularization is replaced by the strong one
and the coil bounds are installed (stage 1), then tightened (stages 2, 3) --
and ONE Picard solve from the previous state can land on the other solution
although the previous state's own solution persists under the new objective
(numerical branch hopping).  The guard re-solves such a stage by
continuation from the state it started from, and acts only when a jump is
seen.

What "branch" means operationally.  The state a homotopy stage starts from
(the loop's last state for stage 1, the previous stage's for the others) is
the draw's branch.  A stage's solve that moves the magnetic axis farther than
:data:`GUARD_AXIS_FRAC` x a (``engine.AXIS_BRANCH_FRAC`` = 1 % of the
reconstruction's minor radius: the discriminator the MSE stage's
finite-difference branch check applies to ONE step) is a SUSPECTED jump.  A
continuous response to the stage moves the axis in proportion to the stage;
a jump to the other solution does not shrink with the stage.

The detector (:func:`step_displacement_frac`).  ``|axis_after - axis_before|
/ a`` of one solve, read from the solver's own o-point
(:func:`bouquet.engine.state_axis`).  At or below the threshold, or with an
axis or a unknown, nothing happens: the guard has only READ the solver (two
axis reads and one ``copy_eq`` snapshot), so the iterate path is the
unguarded one bit for bit.

The action (:func:`continue_in_halves`, only when the detector fires).  The
stage's objective is parametrized by ``s`` in [0, 1] (:func:`blend_reg`,
:func:`blend_bounds`): in stage 1, which installs the strong regularization,
every strong term keeps its weight while its target moves linearly from the
starting state's coil currents to its own (the coils are walked from where
the loop left them to the reconstruction's); each coil bound moves linearly
from the previous stage's interval (stage 1 without hard bounds: the stage's
interval widened just enough to contain the starting state) to the stage's.
``s = 1`` is EXACTLY the stage's objective.  From the starting state the guard re-solves
at ``s = 1/2``; a sub-step that again moves the axis beyond the threshold is
retried at half the step (1/4, 1/8, ``GUARD_MIN_STEP``); an accepted sub-step
becomes the new starting state.  Then:

* the walk reaches ``s = 1`` and ends within the threshold of the jumped
  state: the jump was a LEGITIMATE (continuous) response -- the jumped state
  is restored, so the draw goes on exactly as unguarded (``"legitimate"``);
* the walk reaches ``s = 1`` and ends elsewhere: the jump was branch
  hopping -- the draw goes on from the continuous state (``"held"``; the
  ONE outcome that changes the draw);
* a sub-step of ``GUARD_MIN_STEP`` still jumps: the draw's branch ends inside
  the stage (a fold) -- a boundary crossing; the jumped state is restored and
  the draw's path is stamped ``"crossed"``;
* a sub-step solve that fails (an exception, the iteration cap) is backed
  off like a jump; one that still fails at the smallest sub-step: the jumped
  state is restored (``"failed"``).

Nothing is accepted, refused or relabeled by the guard: every state it
leaves the stage in is judged by the stage's own saturation test, the
post-homotopy check and the in-spec rule, as an unguarded draw's is (the
one state it cannot hand back is a jumped state that fails to restore; the
stage then fails under the homotopy's own rule).  No
tolerance, criterion or acceptance changes; ``GUARD_AXIS_FRAC`` and
``GUARD_MIN_STEP`` decide only which states the stage's solve starts from.
"""
from __future__ import annotations

import contextlib
import time

import numpy as np

from .engine import AXIS_BRANCH_FRAC

#: The step-displacement threshold of the detector: ``engine.AXIS_BRANCH_FRAC``
#: (1 % of the reconstruction's minor radius), the discriminator the MSE
#: stage's finite-difference branch check applies to ONE coefficient step.
#: Measured on the known bistable slice: the two solutions' axes sit 17 mm
#: (2.8 % of a) apart.  A NUMERICAL-METHOD parameter (which state a stage
#: solves from), never an acceptance criterion.
GUARD_AXIS_FRAC = AXIS_BRANCH_FRAC
#: The smallest continuation sub-step (three halvings: 1/2, 1/4, 1/8, the
#: issue's "a quarter, an eighth").  A jump that persists at this step is a
#: boundary crossing.
GUARD_MIN_STEP = 0.125
#: The first sub-step.
GUARD_FIRST_STEP = 0.5
#: Version of the per-draw record.
GUARD_VERSION = "bistability_guard/1"
#: The per-draw actions, most significant first (the draw's summary action
#: is the most significant of its stages').
GUARD_ACTIONS = ("crossed", "failed", "held", "legitimate", "none")
#: What each action means (recorded with every record).
GUARD_ACTION_MEANING = {
    "none": "no stage's solve moved the axis beyond the threshold",
    "legitimate": ("a stage jumped; its continuation reached the stage's "
                   "objective within the threshold of the jumped state "
                   "(a continuous response): the jumped state was kept"),
    "held": ("a stage jumped; its continuation reached the stage's "
             "objective away from the jumped state (branch hopping): the "
             "continuous state was kept"),
    "crossed": ("a stage jumped and the jump persisted at the smallest "
                "sub-step (the branch ends inside the stage): the jumped "
                "state was kept and the path is stamped crossed"),
    "failed": ("a stage jumped and a continuation sub-step still failed "
               "to solve at the smallest sub-step: the jumped state was "
               "kept"),
}


#: The exception types a toolkit call inside the guard is expected to raise
#: when it fails (OpenFUSIONToolkit raises a plain ``Exception`` from its
#: error buffer, ``ValueError`` for an exceeded iteration cap).  Anything
#: else -- a programming error -- propagates; so does the draw's wall-budget
#: exception (:class:`bouquet.TokaMaker_interface.DrawWallBudgetExceeded`).
TOOLKIT_ERRORS = (ValueError, RuntimeError, OSError, ArithmeticError,
                  MemoryError)


def _is_budget(exc) -> bool:
    from .TokaMaker_interface import DrawWallBudgetExceeded
    return isinstance(exc, DrawWallBudgetExceeded)


def is_toolkit_error(exc) -> bool:
    """Whether *exc* is a solver / toolkit failure the guard handles (the
    toolkit's plain ``Exception``, or one of :data:`TOOLKIT_ERRORS`), not a
    programming error or the draw's wall-budget exception."""
    if _is_budget(exc):
        return False
    return type(exc) is Exception or isinstance(exc, TOOLKIT_ERRORS)


def _err(exc) -> str:
    return f"{type(exc).__name__}: {str(exc).strip()[:200]}"


def step_displacement_frac(axis_from, axis_to, minor_radius_m):
    """``|axis_to - axis_from| / a`` (two ``(R, Z)`` [m], a [m]); ``None``
    when either axis or a is unknown."""
    if axis_from is None or axis_to is None or minor_radius_m is None:
        return None
    try:
        a = float(minor_radius_m)
        d = float(np.hypot(float(axis_to[0]) - float(axis_from[0]),
                           float(axis_to[1]) - float(axis_from[1])))
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.isfinite(a) and a > 0.0 and np.isfinite(d)):
        return None
    return d / a


def jumped(frac, threshold=GUARD_AXIS_FRAC) -> bool:
    """Whether one solve's step displacement *frac* (:func:`
    step_displacement_frac`) is a suspected jump (``None``: unknown, never)."""
    return frac is not None and float(frac) > float(threshold)


def continue_in_halves(solve_from, start_state, start_axis, minor_radius_m,
                       *, threshold=GUARD_AXIS_FRAC, first_step=GUARD_FIRST_STEP,
                       min_step=GUARD_MIN_STEP):
    """Walk a stage's objective from ``s = 0`` (the starting state) to
    ``s = 1`` in sub-steps, each solved from the last accepted state.

    ``solve_from(state, s) -> (state, axis)`` solves at objective *s* from
    *state* (it may raise).  A sub-step whose axis moved more than
    ``threshold x a`` from the state it started from is rejected and retried
    at half the step; below *min_step* the jump PERSISTS.  A sub-step that
    raises (does not solve) is retried at half the step likewise; below
    *min_step* the walk FAILED; only a toolkit failure
    (:func:`is_toolkit_error`) counts as one -- the draw's wall-budget
    exception and programming errors propagate.  The step never grows again
    after a halving, so the walk takes at most
    ``log2(first_step/min_step) + 1/min_step`` solves (10 at the defaults;
    11 with the stage's own solve).

    Returns ``dict(outcome, s_reached, state, axis, steps, n_solves)``:
    ``outcome`` ``"completed"`` (``s_reached == 1``, *state* / *axis* the
    final accepted ones), ``"persistent"`` or ``"failed"`` (*state* / *axis*
    the last accepted); ``steps`` one record per sub-step solve."""
    s_done, ds = 0.0, float(first_step)
    state, axis = start_state, start_axis
    steps = []
    while s_done < 1.0:
        s = min(1.0, s_done + ds)
        try:
            new_state, new_axis = solve_from(state, s)
        except Exception as exc:                    # noqa: BLE001
            if not is_toolkit_error(exc):
                # the draw's wall budget, a programming error: never
                # stamped as a sub-step that did not solve
                raise
            # a sub-step that does not solve (the iteration cap) is backed
            # off like a jump: retried at half the step, down to min_step
            steps.append(dict(s_from=s_done, s=s, displacement_frac_a=None,
                              accepted=False, error=_err(exc)))
            if ds / 2.0 < float(min_step) - 1e-12:
                return dict(outcome="failed", s_reached=s_done, state=state,
                            axis=axis, steps=steps, n_solves=len(steps))
            ds /= 2.0
            continue
        d = step_displacement_frac(axis, new_axis, minor_radius_m)
        ok = d is not None and not jumped(d, threshold)
        steps.append(dict(s_from=s_done, s=s, displacement_frac_a=d,
                          accepted=bool(ok)))
        if ok:
            state, axis, s_done = new_state, new_axis, s
            continue
        if ds / 2.0 < float(min_step) - 1e-12:
            return dict(outcome="persistent", s_reached=s_done, state=state,
                        axis=axis, steps=steps, n_solves=len(steps))
        ds /= 2.0
    return dict(outcome="completed", s_reached=1.0, state=state, axis=axis,
                steps=steps, n_solves=len(steps))


def classify(walk, jump_axis, minor_radius_m, threshold=GUARD_AXIS_FRAC):
    """The stage's action from a :func:`continue_in_halves` *walk* and the
    jumped state's axis: ``("legitimate" | "held" | "crossed" | "failed",
    endpoint_separation_frac_a)``.  A completed walk whose endpoint is
    unknown is called ``"legitimate"`` (nothing shows a second solution)."""
    if walk["outcome"] == "failed":
        return "failed", None
    if walk["outcome"] == "persistent":
        return "crossed", None
    sep = step_displacement_frac(walk["axis"], jump_axis, minor_radius_m)
    if sep is not None and jumped(sep, threshold):
        return "held", sep
    return "legitimate", sep


def summary_action(actions) -> str:
    """The most significant of a draw's stage *actions* (:data:`GUARD_ACTIONS`
    order); ``"none"`` for none."""
    acts = [a for a in actions if a]
    for a in GUARD_ACTIONS:
        if a in acts:
            return a
    return "none"


# ---------------------------------------------------------------------------
#  the stage's objective at s (the solver's coil regularization and bounds)
# ---------------------------------------------------------------------------
def _term_parts(t):
    """``(coffs, target, weight)`` of a solver regularization term (the
    toolkit's ``coil_reg_term`` namedtuple, or a plain 3-tuple)."""
    if hasattr(t, "coffs"):
        return dict(t.coffs), float(t.target), float(t.weight)
    c, tg, w = t
    return dict(c) if isinstance(c, dict) else c, float(tg), float(w)


def term_value(coffs, currents):
    """``sum_i alpha_i I_i`` of a regularization term's coefficients
    *coffs* over the coil *currents*; ``None`` when a coil (or a virtual
    channel such as ``#VSC``, whose current the toolkit does not report) is
    missing."""
    try:
        items = coffs.items() if isinstance(coffs, dict) else \
            ((k, 1.0) for k in coffs)
        tot = 0.0
        for k, a in items:
            if k not in currents:
                return None
            tot += float(a) * float(currents[k])
        return tot
    except (TypeError, ValueError):
        return None


def blend_reg(make_term, strong_terms, start_currents, s):
    """The coil-regularization terms at objective *s* of the FIRST stage
    (the one that installs the strong regularization): every strong term at
    its own weight, its TARGET moved linearly from the starting state's
    value of the term (:func:`term_value` over *start_currents*) to its own;
    a term whose starting value is unknown (the ``#VSC`` channel) keeps its
    target.  At ``s = 0`` the strong terms hold the coils where the loop
    left them; ``s = 1`` is exactly the strong terms (the coils pulled to the
    reconstruction's).  *make_term* is the solver's ``coil_reg_term(coffs,
    target=, weight=)``."""
    s = float(s)
    out = []
    for t in strong_terms:
        c, tg, w = _term_parts(t)
        v0 = None if s >= 1.0 else term_value(c, start_currents or {})
        out.append(make_term(c, target=(tg if v0 is None
                                        else v0 + s * (tg - v0)),
                             weight=w))
    return out


def containing_bounds(target, start_values):
    """The stage's bound intervals *target* (``{name: [lo, hi]}``) widened
    just enough to contain the starting state's *start_values*
    (``{name: value}``; a symmetric ``#VSC``-like channel may be given as a
    magnitude): the ``s = 0`` end of :func:`blend_bounds` when no bounds were
    installed before the stage."""
    out = {}
    for k, (lo, hi) in target.items():
        lo, hi = float(lo), float(hi)
        v = start_values.get(k)
        if v is not None and np.isfinite(float(v)):
            v = float(v)
            if k.startswith("#") and lo <= 0.0 <= hi:
                # a virtual channel read as a magnitude: symmetric widening
                m = max(abs(v), -lo, hi)
                lo, hi = min(lo, -m), max(hi, m)
            else:
                lo, hi = min(lo, v), max(hi, v)
        out[k] = [lo, hi]
    return out


def blend_bounds(previous, target, s):
    """The coil bounds at objective *s*: each interval linear from
    *previous* (``{name: [lo, hi]}``, the ``s = 0`` end; a name missing there
    takes the target's) to *target*; ``s = 1`` is exactly *target*."""
    s = float(s)
    if s >= 1.0:
        return {k: [float(v[0]), float(v[1])] for k, v in target.items()}
    out = {}
    for k, (lo, hi) in target.items():
        plo, phi = (previous or {}).get(k, (lo, hi))
        out[k] = [float(plo) + s * (float(lo) - float(plo)),
                  float(phi) + s * (float(hi) - float(phi))]
    return out


# ---------------------------------------------------------------------------
#  the guard of one draw's homotopy (TokaMaker)
# ---------------------------------------------------------------------------
class HomotopyBranchGuard:
    """The branch guard of ONE draw's coil homotopy on a TokaMaker
    solver (built per draw by
    :meth:`bouquet.engine_draws.GenerateEngineDraws.homotopy_guard`).

    ``generate_bouquet`` calls :meth:`solve_stage` in place of the stage's
    ``mygs.solve()``; :attr:`record` is the draw's ``engine[
    "bistability_guard"]``."""

    def __init__(self, *, reference_axis, strong_reg_terms=None,
                 vsc_pair=None, baseline_coils=None,
                 reconstruction_bistability=None, budget=None,
                 threshold=GUARD_AXIS_FRAC, min_step=GUARD_MIN_STEP):
        ref = dict(reference_axis or {})
        self.reference = ref
        self.a = ref.get("minor_radius_m")
        self.strong_reg_terms = (None if strong_reg_terms is None
                                 else list(strong_reg_terms))
        #: the draw's wall-budget clock (``DrawSolveGuard``, or ``None``):
        #: the continuation runs with it paused (:meth:`solve_stage`)
        self.budget = budget
        self.vsc_pair = tuple(vsc_pair or ())
        self.baseline_coils = dict(baseline_coils or {})
        self.threshold = float(threshold)
        self.min_step = float(min_step)
        self.record = dict(
            version=GUARD_VERSION, armed=True, reason=None,
            threshold_frac_a=self.threshold, min_step=self.min_step,
            first_step=GUARD_FIRST_STEP,
            minor_radius_m=(None if self.a is None else float(self.a)),
            reconstruction_bistability=reconstruction_bistability,
            reg_blend=self.strong_reg_terms is not None,
            entry_axis=None, stages=[], fired=False, action="none",
            path=None, n_extra_solves=0, seconds=0.0,
            actions=dict(GUARD_ACTION_MEANING),
            rule=("a homotopy stage's solve that moves the magnetic axis "
                  "more than threshold_frac_a x a from the stage's starting "
                  "state is re-solved by continuation from that state "
                  "(sub-steps 1/2, 1/4, 1/8 of the stage) and stamped; only "
                  "'held' keeps a different state; the guard's verdict never "
                  "enters the in-spec rule or a rejection reason, and its "
                  "sub-steps are not charged to the draw's wall budget"))

    # ---- the stage --------------------------------------------------------
    def _vsc_magnitude(self, cur):
        """The vertical-control channel's excursion as the homotopy measures
        it: the pair's anti-series drift ``|dI_A - dI_B| / 2``."""
        if len(self.vsc_pair) != 2:
            return None
        try:
            a, b = self.vsc_pair
            return abs((float(cur[a]) - float(self.baseline_coils[a]))
                       - (float(cur[b]) - float(self.baseline_coils[b]))) / 2.0
        except (KeyError, TypeError, ValueError):
            return None

    def _start_bounds(self, mygs, bounds, prev_bounds):
        """The ``s = 0`` bounds: the previous stage's (or the loop's hard
        ones), else the stage's widened to contain the starting state -- an
        approximation: the coil-set currents read here may include the
        ``#VSC`` contribution the toolkit bounds separately, and ``#VSC``
        itself is estimated by the pair's anti-series drift."""
        if prev_bounds is not None:
            return {k: list(v) for k, v in prev_bounds.items()}
        try:
            cur, _ = mygs.get_coil_currents()
        except Exception as exc:                    # noqa: BLE001
            if not is_toolkit_error(exc):
                raise
            return {k: list(v) for k, v in bounds.items()}
        vals = {k: cur[k] for k in bounds if k in cur}
        vm = self._vsc_magnitude(cur)
        if vm is not None:
            vals["#VSC"] = vm
        return containing_bounds(bounds, vals)

    def solve_stage(self, mygs, stage_index, label, bounds, prev_bounds=None,
                    first=False):
        """Solve homotopy stage *stage_index* (its bounds *bounds* already
        installed, as is its regularization) under the guard.  *prev_bounds*:
        the bounds the starting state was solved under (``None``: none);
        *first*: the stage whose starting state is the loop's (its objective
        also swaps the loop's regularization for the strong one).  The
        stage's own failure (an exception of its solve) propagates
        untouched."""
        from .engine import axis_shift_record, state_axis
        rec = self.record
        ax0 = state_axis(mygs)
        if first and rec["entry_axis"] is None:
            rec["entry_axis"] = axis_shift_record(ax0, self.reference)
        try:
            pre = mygs.copy_eq() if hasattr(mygs, "copy_eq") else None
        except Exception as exc:                    # noqa: BLE001
            if not is_toolkit_error(exc):
                raise
            pre = None
        mygs.solve()
        ax1 = state_axis(mygs)
        d = step_displacement_frac(ax0, ax1, self.a)
        st = dict(stage=int(stage_index) + 1, label=str(label),
                  step_displacement_frac_a=d,
                  step_dR_mm=(None if d is None
                              else 1e3 * (ax1[0] - ax0[0])),
                  step_dZ_mm=(None if d is None
                              else 1e3 * (ax1[1] - ax0[1])),
                  fired=False, action="none")
        rec["stages"].append(st)
        if rec["path"] is None and d is not None:
            rec["path"] = "continuous"
        if not jumped(d, self.threshold):
            return st
        if pre is None or not hasattr(mygs, "replace_eq"):
            st["action"] = "none"
            st["note"] = ("the solver cannot snapshot its state (copy_eq / "
                          "replace_eq): a jump was seen, nothing was done")
            st["fired"] = None
            return st
        # ---- fired: continuation from the stage's starting state ----------
        t0 = time.perf_counter()
        st["fired"] = True
        rec["fired"] = True
        budget = self.budget
        if budget is not None and budget.check_wall_budget():
            # the draw's wall budget was spent by the stage's own solve: no
            # continuation; the draw goes on exactly as unguarded
            return self._finish(st, d, label, t0, action="failed",
                                reason=("the draw's wall budget was spent "
                                        "when the stage jumped: no "
                                        "continuation was run"))
        pause = (budget.budget_paused() if callable(
            getattr(budget, "budget_paused", None))
            else contextlib.nullcontext())
        with pause:
            return self._continue(mygs, st, d, label, t0, pre, ax0, ax1,
                                  bounds, prev_bounds, first)

    def _back_to_jump(self, mygs, jump, blend, bounds):
        """Put the solver back on the jumped (unguarded) state and the
        stage's own objective; the toolkit error, or ``None``."""
        try:
            if jump is not None:
                mygs.replace_eq(source_eq=jump)
            if blend and self.strong_reg_terms is not None:
                mygs.set_coil_reg(reg_terms=list(self.strong_reg_terms))
            mygs.set_coil_bounds(bounds)
        except Exception as exc:                    # noqa: BLE001
            if not is_toolkit_error(exc):
                raise
            return exc
        return None

    def _finish(self, st, d, label, t0, *, action, walk=None, sep=None,
                ax0=None, ax_end=None, blend=False, reason=None):
        rec = self.record
        st.update(action=action, reg_blended=blend,
                  seconds=float(time.perf_counter() - t0))
        if reason is not None:
            st["reason"] = str(reason)
        n = 0
        if walk is not None:
            n = int(walk["n_solves"])
            st.update(walk_outcome=walk["outcome"],
                      s_reached=float(walk["s_reached"]),
                      substeps=[dict(x) for x in walk["steps"]],
                      endpoint_separation_frac_a=sep)
        st["n_extra_solves"] = n
        st["final_step_displacement_frac_a"] = (
            None if ax_end is None
            else step_displacement_frac(ax0, ax_end, self.a))
        rec["n_extra_solves"] += n
        rec["seconds"] += st["seconds"]
        rec["action"] = summary_action([x.get("action")
                                        for x in rec["stages"]])
        if action == "crossed":
            rec["path"] = "crossed"
        print(f"  [bistability guard] {label}: the axis moved {d:.4f} a "
              f"(> {self.threshold:g} a) -> "
              + (f"continuation {walk['outcome']} at "
                 f"s={walk['s_reached']:.3f} in {n} solve(s): "
                 if walk is not None else "")
              + action + (f" ({reason})" if reason else ""), flush=True)
        return st

    def _continue(self, mygs, st, d, label, t0, pre, ax0, ax1, bounds,
                  prev_bounds, first):
        from .engine import state_axis
        rec = self.record
        blend = bool(first and rec["reg_blend"])
        strong = self.strong_reg_terms
        try:
            jump = mygs.copy_eq()
        except Exception as exc:                    # noqa: BLE001
            if not is_toolkit_error(exc):
                raise
            # nothing was changed: the solver holds the jumped state
            return self._finish(st, d, label, t0, action="failed",
                                reason="the jumped state could not be "
                                       "copied: " + _err(exc))
        try:
            mygs.replace_eq(source_eq=pre)
            b0 = self._start_bounds(mygs, bounds, prev_bounds)
            cur0 = dict(mygs.get_coil_currents()[0])
        except Exception as exc:                    # noqa: BLE001
            if not is_toolkit_error(exc):
                raise
            back = self._back_to_jump(mygs, jump, blend, bounds)
            if back is not None:
                raise RuntimeError("bistability guard: the jumped state "
                                   f"could not be restored ({back})") from exc
            return self._finish(st, d, label, t0, action="failed",
                                reason="the starting state could not be "
                                       "re-installed: " + _err(exc),
                                ax0=ax0, ax_end=state_axis(mygs))

        def solve_from(state, s):
            mygs.replace_eq(source_eq=state)
            if blend:
                mygs.set_coil_reg(reg_terms=blend_reg(
                    mygs.coil_reg_term, strong, cur0, s))
            mygs.set_coil_bounds(blend_bounds(b0, bounds, s))
            mygs.solve()
            return mygs.copy_eq(), state_axis(mygs)

        try:
            walk = continue_in_halves(solve_from, pre, ax0, self.a,
                                      threshold=self.threshold,
                                      min_step=self.min_step)
        except Exception:
            # the draw's wall budget or a programming error: the solver is
            # put back on the jumped state first, then it propagates
            self._back_to_jump(mygs, jump, blend, bounds)
            raise
        action, sep = classify(walk, ax1, self.a, self.threshold)
        reason = None
        if action == "held":
            try:
                mygs.replace_eq(source_eq=walk["state"])
                if blend and strong is not None:
                    mygs.set_coil_reg(reg_terms=list(strong))
                mygs.set_coil_bounds(bounds)
            except Exception as exc:                # noqa: BLE001
                if not is_toolkit_error(exc):
                    raise
                action = "failed"
                reason = ("the continuous state could not be installed; "
                          "the jumped state was kept: " + _err(exc))
        if action != "held":
            back = self._back_to_jump(mygs, jump, blend, bounds)
            if back is not None:
                # the one case the guard cannot undo: the solver's state is
                # unknown, so the stage fails under the homotopy's own rule
                st.update(action="failed", restore_error=_err(back))
                rec["action"] = summary_action([x.get("action")
                                                for x in rec["stages"]])
                raise RuntimeError("bistability guard: the jumped state "
                                   f"could not be restored ({back})")
        return self._finish(st, d, label, t0, action=action, walk=walk,
                            sep=sep, ax0=ax0, ax_end=state_axis(mygs),
                            blend=blend, reason=reason)


def unarmed_record(reason) -> dict:
    """The record of a draw whose guard is not armed (the setting on, the
    guard unable to judge): ``armed=False`` with the reason."""
    return dict(version=GUARD_VERSION, armed=False, reason=str(reason),
                fired=False, action="none", path=None, stages=[],
                n_extra_solves=0, seconds=0.0,
                threshold_frac_a=float(GUARD_AXIS_FRAC),
                min_step=float(GUARD_MIN_STEP))


# ---------------------------------------------------------------------------
#  counts (per generate() and per scan)
# ---------------------------------------------------------------------------
def guard_counts(records) -> dict:
    """``dict(n_armed, n_fired, by_action)`` over per-draw guard *records*
    (``None`` entries and unarmed records count as not armed)."""
    recs = [r for r in records if isinstance(r, dict)]
    armed = [r for r in recs if r.get("armed")]
    by = {a: 0 for a in GUARD_ACTIONS}
    for r in armed:
        a = r.get("action") or "none"
        by[a] = by.get(a, 0) + 1
    return dict(n_records=len(recs), n_armed=len(armed),
                n_fired=sum(1 for r in armed if r.get("fired")),
                by_action=by,
                n_extra_solves=int(sum(int(r.get("n_extra_solves") or 0)
                                       for r in armed)))


def guard_text(counts) -> str:
    """The summary-line fragment of :func:`guard_counts` (``""`` when no
    draw was armed)."""
    if not counts or not counts.get("n_armed"):
        return ""
    by = counts.get("by_action") or {}
    parts = [f"{a} {by[a]}" for a in GUARD_ACTIONS[:-1] if by.get(a)]
    return (f"bistability guard: fired on {counts['n_fired']} of "
            f"{counts['n_armed']} armed draw(s)"
            + (f" ({', '.join(parts)}; {counts['n_extra_solves']} extra "
               "solve(s))" if parts else ""))


def stamp_guard_scan(h5path_or_header, scan_key=None):
    """The scan's guard count, recounted from the archive itself (so a scan
    generated in several calls, or merged, is counted whole): scan attrs
    ``n_bistability_guard_armed`` / ``n_bistability_guard_fired`` and
    ``bistability_guard_counts_json`` (:func:`guard_counts` of the archived
    draws' group attrs).  Nothing is written when no archived draw carries a
    guard record (the setting off).  Returns the counts, or ``None``."""
    import json
    import os
    import h5py
    from .filtering import _resolve, _scan_group_path
    from .schema import (GUARD_ACTION_ATTR, GUARD_ARMED_ATTR,
                         GUARD_ARMED_SCAN_ATTR, GUARD_COUNTS_SCAN_ATTR,
                         GUARD_FIRED_ATTR, GUARD_FIRED_SCAN_ATTR)
    from .utils import _group_path, list_equilibrium_indices
    h5path = _resolve(h5path_or_header)
    if not os.path.exists(h5path):
        return None
    try:
        idx = list_equilibrium_indices(h5path, scan_key=scan_key)
    except KeyError:
        idx = []
    with h5py.File(h5path, "a") as hf:
        recs = []
        for i in idx:
            gp = _group_path(scan_key, i)
            if gp not in hf:
                continue
            a = hf[gp].attrs
            if GUARD_ARMED_ATTR not in a:
                continue
            act = a.get(GUARD_ACTION_ATTR, "none")
            recs.append(dict(
                armed=bool(a[GUARD_ARMED_ATTR]),
                fired=bool(a.get(GUARD_FIRED_ATTR, False)),
                action=(act.decode() if isinstance(act, bytes)
                        else str(act))))
        if not recs:
            return None
        c = guard_counts(recs)
        c.pop("n_extra_solves", None)
        gs = _scan_group_path(scan_key)
        if gs in hf:
            g = hf[gs]
            g.attrs[GUARD_ARMED_SCAN_ATTR] = int(c["n_armed"])
            g.attrs[GUARD_FIRED_SCAN_ATTR] = int(c["n_fired"])
            g.attrs[GUARD_COUNTS_SCAN_ATTR] = json.dumps(dict(
                c, version=GUARD_VERSION,
                note=("archived draws only; the record of every attempt "
                      "that reached the homotopy is on "
                      "Bouquet.engine_bistability_guard")))
    return c
