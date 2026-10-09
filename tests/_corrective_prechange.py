"""VERBATIM copy of ``bouquet.TokaMaker_interface._corrective_jphi_iteration``
as it was immediately before ``initial_request`` / ``return_request`` were
added (the one-reconstruction-state work) -- the reference the default-kwargs
bit-identity test compares against.

Not a test module (no ``test_`` prefix); nothing here is collected.  The only
edit to the copied body is that ``q_ravg`` is imported from the live
``bouquet.physics`` (unchanged by that commit).  Do not edit.
"""
import numpy as np

from bouquet.physics import q_ravg


def _corrective_jphi_iteration(mygs, psi_N, target_jphi, pp_prof,
                                Ip_target, pax_target, psi_pad,
                                min_iters=2, max_iters=8,
                                rtol=0.05, verbose=True,
                                damping=1.0, protect_state=False):
    r"""Iterate TokaMaker input j_phi until the output matches a target.

    Uses Newton correction: ``input += (target - output)`` each step.
    Starts with *min_iters*, then checks whether the edge spike RMS
    is still improving by more than *rtol* relative per step.  Stops
    when converged or *max_iters* is reached.

    Parameters
    ----------
    mygs : TokaMaker
        GS solver (in a solved state with current profiles set).
    psi_N : ndarray
        Normalised flux grid.
    target_jphi : ndarray
        Target j_phi profile [A/m²] (e.g. j_inductive + spike_profile).
    pp_prof : dict
        Pressure gradient profile dict for ``set_profiles``.
    Ip_target : float
        Plasma current target [A].
    pax_target : float
        On-axis pressure target [Pa].
    psi_pad : float
        LCFS padding.
    min_iters : int
        Minimum iterations before checking convergence (default 2).
    max_iters : int
        Maximum iterations (default 8).
    rtol : float
        Relative improvement threshold — stop if
        ``|rms_new - rms_old| / rms_old < rtol`` (default 0.05 = 5%).
    verbose : bool
        Print per-iteration diagnostics.

    Returns
    -------
    j_phi_output : ndarray
        Converged GS output j_phi [A/m²].
    n_iters : int
        Number of iterations performed.
    edge_rms_history : list of float
        Edge RMS per iteration [A/m²].
    """
    from OpenFUSIONToolkit.TokaMaker.util import get_jphi_from_GS

    npsi = len(psi_N)
    edge_mask = psi_N > 0.9
    j_phi_input = target_jphi.copy()
    edge_rms_history = []
    # keep-best bookkeeping (protect_state=True): the imas anchor target comes
    # from ANOTHER code's flux geometry and may not be exactly achievable, so
    # undamped Newton steps can oscillate/diverge; track the best full-profile
    # RMS state and restore it at the end instead of trusting the last iterate.
    # Gate on BOTH halves of the snapshot/restore pair.  Every snapshot taken
    # here is eventually consumed by `replace_eq` (the per-iteration
    # solve-failure restore and the final keep-best restore), so a solver
    # object exposing only `copy_eq` would pass a copy_eq-only gate and then
    # raise AttributeError on the restore -- turning a recoverable solve
    # failure into a hard crash.  Matches the both-methods gate already used
    # at the warm-start snapshot site further down this module.
    _can_snap = (protect_state and hasattr(mygs, "copy_eq")
                 and hasattr(mygs, "replace_eq"))
    best = {"rms": np.inf, "eq": None, "out": None}
    full_rms_history = []
    # Seed the output with the UNCORRECTED input.  If the very first solve
    # raises, the handler below breaks out before `j_phi_output` is ever
    # assigned, and the return statement then raised NameError -- masking a
    # solve failure behind an unrelated-looking crash.  Seeding it means that
    # case degrades to "no correction was applied", which is the truthful
    # answer and matches the non-fatal intent of the break; the empty
    # `edge_rms_history` and the warning below tell the caller it happened.
    # (A later-iteration failure is unaffected: it keeps the last good
    # iterate, exactly as before.)
    j_phi_output = j_phi_input.copy()
    it = -1

    for it in range(max_iters):
        ffp = {"type": "jphi-linterp", "y": j_phi_input.copy(), "x": psi_N}
        mygs.set_targets(Ip=Ip_target, pax=pax_target)
        mygs.set_profiles(pp_prof=pp_prof, ffp_prof=ffp)
        _snap = mygs.copy_eq() if _can_snap else None
        try:
            mygs.solve()
        except (ValueError, RuntimeError) as e:
            if verbose:
                print(f"  [jphi_corr iter {it+1}] solve failed: {e}")
            if it == 0:
                # Not verbose-gated: no iterate ever succeeded, so the caller is
                # getting its own input back with no correction applied at all.
                # That is a materially different result from a converged one and
                # must not be inferable only from an empty RMS history.
                print(f"  [jphi_corr] WARNING: the FIRST corrective solve "
                      f"failed ({e}); returning the uncorrected input j_phi "
                      f"-- no corrective iteration was applied")
            if _snap is not None:
                mygs.replace_eq(source_eq=_snap)   # do not leave the diverged state
            break

        _, f, fp, _, pp = mygs.get_profiles(npsi=npsi, psi_pad=psi_pad)
        _, _, ravgs, _, _, _ = mygs.get_q(npsi=npsi, psi_pad=psi_pad)
        j_phi_output = get_jphi_from_GS(f * fp, pp, q_ravg(ravgs, "<R>"), q_ravg(ravgs, "<1/R>"))

        diff = j_phi_output - target_jphi
        rms_edge = float(np.sqrt(np.mean(diff[edge_mask]**2)))
        edge_rms_history.append(rms_edge)
        _kept = None
        if _can_snap:
            rms_full = float(np.sqrt(np.mean(diff**2)))
            full_rms_history.append(rms_full)
            _kept = rms_full < best["rms"]
            if _kept:
                best.update(rms=rms_full, eq=mygs.copy_eq(), out=j_phi_output.copy())

        if verbose:
            # Report the FULL-domain RMS too when keep-best is on: the stopping
            # rule is on the edge, but the state that gets kept is chosen on the
            # full domain, so a log showing only the edge cannot explain which
            # iterate was landed on (issue #25).
            if _can_snap:
                print(f"  [jphi_corr iter {it+1}] edge RMS = "
                      f"{rms_edge/1e6:.6f} MA/m², full RMS = "
                      f"{full_rms_history[-1]/1e6:.6f} MA/m² "
                      f"({'KEPT (new best)' if _kept else 'discarded'}; "
                      f"best {best['rms']/1e6:.6f})")
            else:
                print(f"  [jphi_corr iter {it+1}] edge RMS = {rms_edge/1e6:.6f} MA/m²")

        # Check convergence after min_iters
        if it >= min_iters - 1 and len(edge_rms_history) >= 2:
            prev_rms = edge_rms_history[-2]
            if prev_rms > 0:
                rel_change = abs(rms_edge - prev_rms) / prev_rms
                if rel_change < rtol:
                    if verbose:
                        print(f"  [jphi_corr] converged at iter {it+1} "
                              f"(rel_change={rel_change:.4f} < {rtol})")
                    break

        # Newton correction (optionally damped)
        j_phi_input = j_phi_input + damping * (target_jphi - j_phi_output)
        j_phi_input = np.maximum(j_phi_input, 0.0)

    if _can_snap and best["eq"] is not None:
        mygs.replace_eq(source_eq=best["eq"])      # land on the best state seen
        j_phi_output = best["out"]
        if verbose:
            # The kept-RMS sequence is monotone non-increasing BY CONSTRUCTION;
            # printing it is what lets a run be checked rather than trusted.
            _kept_traj = np.minimum.accumulate(np.asarray(full_rms_history))
            print(f"  [jphi_corr] keep-best landed on full RMS "
                  f"{best['rms']/1e6:.6f} MA/m² (per-iterate "
                  + " -> ".join(f"{r/1e6:.6f}" for r in full_rms_history)
                  + "; kept "
                  + " -> ".join(f"{r/1e6:.6f}" for r in _kept_traj) + ")")

    return j_phi_output, it + 1, edge_rms_history
