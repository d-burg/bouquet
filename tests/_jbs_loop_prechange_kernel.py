"""VERBATIM copy of ``bouquet.jbs_loop.run_jbs_loop`` as it was immediately
before the opt-in current-residual gate (``jbs_gate_current_residual``) was
added -- the reference the gate-OFF bit-identity test compares against.

Not a test module (no ``test_`` prefix); nothing here is collected.  Only the
kernel body is frozen: the helpers it calls (``CurrentRelaxer``,
``profile_residuals``, ``tolerances_record``, ``oft_build_info``, ...) were
not changed by that commit and are imported from the live module, so the
comparison isolates exactly the code the gate touched.  The one change to
the copied body is its relative ``from .physics import`` made absolute.  Do
not edit.

2026-10-06 (the ONE later edit, applied to keep the comparison isolating its
flag): the kernel-wide change "r_j / r_I measured against the bootstrap the
pass SOLVED" (``bouquet.jbs_loop``, docstring "Residuals") is applied here
exactly as in the live kernel -- the blended-bootstrap ``jbs_solved``, the
iterate residual kept as ``r_j_iterate`` / ``r_I_iterate`` on a loop that
relaxes the current at beta < 1, the omega schedule on the iterate
residual, ``jbs_solved`` returned.  Every other line is as frozen.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from bouquet.jbs_loop import (JBS_RELAX_FLOOR, JBS_REQUIRED_CONSECUTIVE,
                              JBS_GROWTH_ABORT_PASSES, JBSNonFinite,
                              JBSNotConverged, CurrentRelaxer, _fmt_hist,
                              _finite_or_none, _step_takes_relax,
                              oft_build_info, profile_residuals,
                              tolerances_record,
                              RESIDUAL_SOLVED_DEFINITION)


def run_jbs_loop(jbs0, step: Callable, evaluate: Callable, settings: dict, *,
                 Ip: float, meas0: Optional[dict] = None,
                 gate_li: bool = True, gate_q0: bool = False,
                 label: str = "", init: Optional[str] = None,
                 init_source: Optional[str] = None,
                 grid: str = "psi_N native",
                 on_pass: Optional[Callable] = None,
                 max_passes: Optional[int] = None,
                 raise_on_fail: Optional[bool] = None) -> dict:
    """Iterate closure <-> GS <-> Redl to the fixed point.

    Parameters
    ----------
    jbs0 : array
        Initial bootstrap profile (the one the FIRST solve is assembled with).
    step : callable ``step(jbs, k) -> meas`` or ``step(jbs, k, relax=...)``
        Closure + assembly + GS solve with bootstrap ``jbs``; returns the
        measurement dict of the NEW equilibrium: ``w`` and ``x`` (the Ip weights
        and the grid ``jbs`` lives on), optionally ``li`` and ``q0``.  Any other
        keys are carried through untouched (``meas_final``).  A step that
        accepts a ``relax`` keyword receives the pass's
        :class:`CurrentRelaxer` and solves ``relax(j_closure)`` instead of
        ``j_closure`` (the ``jbs_relax_current`` relaxation of the solved
        current); a two-argument step is called as before and the record says
        the current was not relaxed.
    evaluate : callable ``evaluate(meas) -> J``
        Redl bootstrap on the equilibrium ``step`` just produced, composed
        exactly the way ``jbs`` is (same smoothing / isolation / scale).
    settings : dict
        :func:`jbs_settings`.
    Ip : float
        Plasma current the ``r_I`` residual is normalised by.
    meas0 : dict or None
        Measurements of the starting equilibrium E_0 (``li``/``q0``), so the
        first pass has a ``dl_i``/``dq0``.
    gate_li, gate_q0 : bool
        Whether ``dl_i`` / ``dq0`` are convergence criteria here (``dq0`` only
        where an axis row / q0 target is active); they are logged either way
        whenever measured.
    on_pass : callable or None
        ``on_pass(k, meas, J, entry)`` after every pass (callers use it to
        refresh per-pass state, e.g. an axis-row target).  ``entry`` carries
        the pass's residuals, ``omega_used`` and ``omega_next`` (the
        relaxation the next iterate is built with; ``None`` when no further
        pass follows), so a caller's own per-pass update can be relaxed by the
        same factor.
    init : str or None
        The ``jbs_init`` setting recorded as ``record["init"]`` (default: the
        settings' own).
    init_source : str or None
        What ``jbs0`` actually is, in words, recorded as
        ``record["init_source"]`` (e.g. a draw's "evaluate_jBS on the draw's
        anchor equilibrium with the draw's own perturbed kinetics").  Record
        only: the fixed point does not depend on the initial iterate.
    max_passes : int or None
        Override of ``settings["max_passes"]`` (the post-homotopy stage).
    raise_on_fail : bool or None
        Override of the ``settings["on_fail"]`` policy.

    Returns
    -------
    dict with ``jbs_used`` (the bootstrap of the delivered solve), ``J_final``
    (Redl on the delivered equilibrium), ``meas_final``, ``converged`` and
    ``record`` (the logged block, JSON-safe).  On non-convergence raises
    :class:`JBSNotConverged` unless the policy is ``"flag"``.
    """
    t0 = time.perf_counter()
    K = int(settings["max_passes"] if max_passes is None else max_passes)
    need = int(settings.get("required_consecutive", JBS_REQUIRED_CONSECUTIVE))
    floor = float(settings.get("relax_floor", JBS_RELAX_FLOOR))
    n_abort = int(settings.get("growth_abort_passes", JBS_GROWTH_ABORT_PASSES))
    omega = float(settings["relax"])
    halve_on = int(settings.get("relax_halve_on", 1))
    beta = float(settings.get("relax_current", 1.0))
    takes_relax = _step_takes_relax(step)
    relaxer = CurrentRelaxer(beta) if takes_relax else None
    if raise_on_fail is None:
        raise_on_fail = (settings.get("on_fail", "raise") == "raise")

    rec = dict(
        enabled=True, label=str(label),
        init=str(init if init is not None else settings.get("init", "anchor")),
        init_source=(None if init_source is None else str(init_source)),
        grid=str(grid),
        tolerances=tolerances_record(dict(settings, max_passes=K)),
        criteria=dict(r_j=True, r_I=True, dl_i=bool(gate_li),
                      dq0=bool(gate_q0)),
        n_passes=0, converged=False, stop_reason=None,
        omega=[], r_j=[], r_I=[], dl_i=[], dq0=[], I_BS=[], I_BS_used=[],
        jBS_peak=[], jBS_peak_psiN=[], li=[], q0=[], pass_ok=[],
        wall_s=None,
        relax_current=(beta if takes_relax else None),
        current_relaxation=(
            ("solved current relaxed: js_k = (1-beta) js_k-1 + beta jc_k "
             "(k >= 1); path only, the fixed point is unchanged")
            if takes_relax else
            "not applied: this caller's step does not take the relaxer"),
        current_gap=[], current_blended=[],
        # RECORD ONLY, never gated: the unrelaxed residual of the closure
        # half, ||jc_k - js_{k-1}|| / ||jc_k|| (= current_gap / (1 - beta)
        # on a blended pass).  current_gap understates it by (1 - beta).
        current_residual_unrelaxed=[],
        current_residual_unrelaxed_definition=(
            "||jc_k - js_{k-1}||_w / ||jc_k||_w = current_gap / (1 - beta) "
            "on a blended pass: the unrelaxed closure-half residual; "
            "recorded only, NOT a convergence criterion"),
        relax_halve_on=int(halve_on), omega_halved_at_pass=[],
    )
    track_solved = bool(takes_relax and beta < 1.0)
    if track_solved:
        rec.update(
            residual_definition=RESIDUAL_SOLVED_DEFINITION,
            r_j_iterate=[], r_I_iterate=[], I_BS_used_iterate=[],
            bootstrap_blended=[])
    try:
        from bouquet.physics import EVALUATE_JBS_VERSION
        rec["evaluate_jBS_version"] = EVALUATE_JBS_VERSION
    except Exception:
        rec["evaluate_jBS_version"] = None
    rec["oft_build"] = oft_build_info()

    def _nonfinite(k, what, arr, grid):
        """Raise :class:`JBSNonFinite` at the first non-finite node of
        *arr* (pass ``k + 1``; ``k = -1`` is the initial guess)."""
        bad = ~np.isfinite(arr)
        i = int(np.argmax(bad))
        psi = None
        if grid is not None and np.shape(grid) == np.shape(arr):
            psi = float(np.asarray(grid, dtype=float)[i])
        where = (f"index {i}" + ("" if psi is None else f", psi_N={psi:.6g}")
                 + f"; {int(bad.sum())} of {bad.size} node(s)")
        when = "before pass 1" if k < 0 else f"on pass {k + 1}/{K}"
        msg = (f"self-consistent j_BS loop"
               f"{(' [' + label + ']') if label else ''}: {what} is "
               f"non-finite {when} ({where}) -- refusing to blend it into "
               "the next iterate or hand it to a GS solve")
        rec["stop_reason"] = msg
        rec["wall_s"] = float(time.perf_counter() - t0)
        rec["jbs_converged"] = False
        rec["fail_message"] = msg
        print("  [jbs-loop] " + msg, flush=True)
        raise JBSNonFinite(msg, rec, pass_number=max(k + 1, 0), index=i,
                           psi_N=psi)

    jbs = np.asarray(jbs0, dtype=float).copy()
    if not np.all(np.isfinite(jbs)):
        _nonfinite(-1, "the initial bootstrap guess jbs0", jbs,
                   (meas0 or {}).get("x"))
    jbs_solved_base = None
    jbs_solved = jbs
    prev = dict(meas0 or {})
    streak = 0
    growth_at_floor = 0
    grow_streak = 0                   # consecutive passes on which r_j grew
    omega_used_for_current = None     # omega that produced `jbs` (None: init)
    r_j_prev = None
    meas = None
    J = None
    jbs_used = jbs

    for k in range(K):
        jbs_used = jbs
        if relaxer is not None:
            relaxer.begin_pass()
            meas = step(jbs, k, relax=relaxer)
        else:
            meas = step(jbs, k)
        J = np.asarray(evaluate(meas), dtype=float)
        if J.shape != jbs.shape:
            raise ValueError(f"run_jbs_loop[{label}]: evaluate returned shape "
                             f"{J.shape}, the iterate has {jbs.shape}")
        if not np.all(np.isfinite(J)):
            rec["n_passes"] = k + 1
            _nonfinite(k, "the evaluated bootstrap J", J, meas.get("x"))
        _relaxed_now = bool(relaxer is not None and relaxer.called)
        _bs_blended = bool(_relaxed_now and relaxer.last_blended
                           and jbs_solved_base is not None
                           and jbs_solved_base.shape == jbs.shape)
        jbs_solved = ((1.0 - beta) * jbs_solved_base + beta * jbs
                      if _bs_blended else jbs)
        res_it = profile_residuals(J, jbs, meas["w"], meas["x"], Ip)
        res = (profile_residuals(J, jbs_solved, meas["w"], meas["x"], Ip)
               if _bs_blended else res_it)
        li_new = _finite_or_none(meas.get("li"))
        q0_new = _finite_or_none(meas.get("q0"))
        li_old = _finite_or_none(prev.get("li"))
        q0_old = _finite_or_none(prev.get("q0"))
        dl_i = (abs(li_new - li_old) if (li_new is not None
                                         and li_old is not None) else None)
        dq0 = (abs(q0_new - q0_old) if (q0_new is not None
                                        and q0_old is not None) else None)
        ok = (np.isfinite(res["r_j"]) and res["r_j"] <= settings["rtol_j"]
              and np.isfinite(res["r_I"])
              and res["r_I"] <= settings["rtol_Ip"])
        if gate_li:
            ok = ok and (dl_i is not None and dl_i <= settings["tol_li"])
        if gate_q0:
            ok = ok and (dq0 is not None and dq0 <= settings["tol_q0"])
        ok = bool(ok)
        rec["omega"].append(None if omega_used_for_current is None
                            else float(omega_used_for_current))
        rec["r_j"].append(res["r_j"])
        rec["r_I"].append(res["r_I"])
        rec["dl_i"].append(dl_i)
        rec["dq0"].append(dq0)
        rec["I_BS"].append(res["I_BS"])
        rec["I_BS_used"].append(res["I_BS_used"])
        rec["jBS_peak"].append(res["jBS_peak"])
        rec["jBS_peak_psiN"].append(res["jBS_peak_psiN"])
        rec["li"].append(li_new)
        rec["q0"].append(q0_new)
        rec["pass_ok"].append(ok)
        if relaxer is not None:
            _gap, _bl = relaxer.gap(meas.get("w"), meas.get("x"))
            _unrel = relaxer.unrelaxed_residual(meas.get("w"), meas.get("x"))
            relaxer.commit()
            if _relaxed_now:
                jbs_solved_base = np.asarray(jbs_solved, dtype=float).copy()
        else:
            _gap, _bl, _unrel = None, False, None
        rec["current_gap"].append(_gap)
        rec["current_blended"].append(bool(_bl))
        rec["current_residual_unrelaxed"].append(_unrel)
        if track_solved:
            rec["r_j_iterate"].append(res_it["r_j"])
            rec["r_I_iterate"].append(res_it["r_I"])
            rec["I_BS_used_iterate"].append(res_it["I_BS_used"])
            rec["bootstrap_blended"].append(_bs_blended)
        rec["n_passes"] = k + 1
        entry = dict(k=k, ok=ok, dl_i=dl_i, dq0=dq0,
                     omega_used=omega_used_for_current, **res)
        print(f"  [jbs-loop{(' ' + label) if label else ''}] pass {k + 1}/{K}: "
              f"r_j={res['r_j']:.3e} (tol {settings['rtol_j']:.0e}) "
              f"r_I={res['r_I']:.3e} (tol {settings['rtol_Ip']:.0e})"
              + ("" if dl_i is None else f" dl_i={dl_i:.2e}")
              + ("" if dq0 is None else f" dq0={dq0:.2e}")
              + f" I_BS={res['I_BS'] / 1e3:.2f} kA"
              + ("" if omega_used_for_current is None
                 else f" omega={omega_used_for_current:.3f}")
              + ("" if not _bs_blended else
                 f" (vs iterate r_j={res_it['r_j']:.3e} r_I="
                 f"{res_it['r_I']:.3e}, record only)")
              + ("" if not _bl else f" |js-jc|/|jc|={_gap:.2e}")
              + ("" if (not _bl or _unrel is None)
                 else f" (unrelaxed {_unrel:.2e}, record only)")
              + (" ok" if ok else ""), flush=True)
        # ---- a pass that can never count: stop now, not at the ceiling ----
        _never = None
        if gate_li and li_new is None:
            _never = ("the step returned no finite l_i although l_i is a "
                      "convergence criterion here (gate_li=True)")
        elif gate_q0 and q0_new is None:
            _never = ("the step returned no finite q0 although q0 is a "
                      "convergence criterion here (gate_q0=True)")
        elif (not np.any(J != 0.0)) and np.any(jbs != 0.0):
            _never = ("the evaluated bootstrap J is identically zero while "
                      "the iterate is not (r_j is infinite)")
        if _never is not None:
            rec["stop_reason"] = (f"pass {k + 1}/{K}: {_never} -- "
                                  "convergence is impossible, stopped at once "
                                  "instead of running to the pass ceiling")
            break
        streak = streak + 1 if ok else 0
        if streak >= need:
            rec["converged"] = True
            rec["stop_reason"] = (f"all active criteria met on {need} "
                                  "consecutive passes")
            if on_pass is not None:
                on_pass(k, meas, J, dict(entry, omega_next=None))
            break
        # ---- relaxation schedule: halve on SUSTAINED growth (halve_on
        # consecutive growing passes; 1 = every growth, the earlier
        # schedule), abort after n_abort growing passes at the floor --------
        if r_j_prev is not None and res_it["r_j"] > r_j_prev:
            if (omega_used_for_current is not None
                    and omega_used_for_current <= floor + 1e-15):
                growth_at_floor += 1
            else:
                growth_at_floor = 0
            grow_streak += 1
            if grow_streak >= halve_on:
                _om = max(0.5 * omega, floor)
                if _om < omega:
                    rec["omega_halved_at_pass"].append(k + 1)
                omega = _om
                grow_streak = 0
        else:
            growth_at_floor = 0
            grow_streak = 0
        r_j_prev = res_it["r_j"]
        stop = (growth_at_floor >= n_abort) or (k == K - 1)
        if on_pass is not None:
            # omega_next: the relaxation the NEXT iterate is built with (the
            # caller relaxes any per-pass update of its own -- e.g. a moved
            # constraint row -- by the same factor)
            on_pass(k, meas, J, dict(entry, omega_next=(None if stop
                                                        else omega)))
        if growth_at_floor >= n_abort:
            rec["stop_reason"] = (f"r_j grew on {n_abort} consecutive passes "
                                  f"at the relaxation floor omega={floor:g}")
            break
        if k == K - 1:
            break                       # no further solve: keep jbs as used
        jbs = (1.0 - omega) * jbs + omega * J
        omega_used_for_current = omega
        prev = meas

    rec["wall_s"] = float(time.perf_counter() - t0)
    rec["jbs_converged"] = bool(rec["converged"])
    if not rec["converged"] and rec["stop_reason"] is None:
        rec["stop_reason"] = (f"pass ceiling {K} reached without {need} "
                              "consecutive passing passes")
    rec["final"] = dict(
        r_j=rec["r_j"][-1] if rec["r_j"] else None,
        r_I=rec["r_I"][-1] if rec["r_I"] else None,
        dl_i=rec["dl_i"][-1] if rec["dl_i"] else None,
        dq0=rec["dq0"][-1] if rec["dq0"] else None,
        I_BS=rec["I_BS"][-1] if rec["I_BS"] else None,
        current_gap=rec["current_gap"][-1] if rec["current_gap"] else None,
        current_residual_unrelaxed=(rec["current_residual_unrelaxed"][-1]
                                    if rec["current_residual_unrelaxed"]
                                    else None))
    if track_solved:
        rec["final"]["r_j_iterate"] = (rec["r_j_iterate"][-1]
                                       if rec["r_j_iterate"] else None)
        rec["final"]["r_I_iterate"] = (rec["r_I_iterate"][-1]
                                       if rec["r_I_iterate"] else None)
    out = dict(jbs_used=np.asarray(jbs_used, dtype=float),
               jbs_solved=np.asarray(jbs_solved, dtype=float),
               J_final=(None if J is None else np.asarray(J, dtype=float)),
               meas_final=meas, converged=bool(rec["converged"]), record=rec)
    if not rec["converged"]:
        msg = (f"self-consistent j_BS loop{(' [' + label + ']') if label else ''}"
               f" did not converge: {rec['stop_reason']}; residual history "
               f"r_j={_fmt_hist(rec['r_j'])} r_I={_fmt_hist(rec['r_I'])}"
               + ("" if not gate_li else f" dl_i={_fmt_hist(rec['dl_i'])}")
               + ("" if not gate_q0 else f" dq0={_fmt_hist(rec['dq0'])}"))
        rec["fail_message"] = msg
        print("  [jbs-loop] " + msg, flush=True)
        if raise_on_fail:
            raise JBSNotConverged(msg, rec)
    return out

