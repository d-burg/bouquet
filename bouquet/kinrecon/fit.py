r"""The kinetic reconstruction: a MAP fit over a small set of physical parameters.

Model: :mod:`.current_model` (ohmic + bootstrap + driven current with the sawtooth periodic state,
thermal + fast-ion pressure) iterated to consistency with the anchored forward solve
(:mod:`.forward`).  Data: the residuals of :func:`.score.score` for the fit groups

* ``flux_loop`` and ``bpol_probe``: every channel the measurement set marks valid (for the EFIT
  k-file route: the EFIT-weighted channels), scored in the measurement's own form (flux loops
  relative to the reference loop when the set says so; probes length-averaged when lengths exist);
* ``pf_current``: only coils whose measurement is a constraint (role ``"fitted"``; held coils are
  exact by construction, free coils carry no measurement constraint -- EFIT semantics);
* ``ip``;
* ``mse``: only when the group is declared E_r-corrected (``meta['er_corrected']``) -- MSE whose
  E_r status is not established is not a fit constraint;

plus one prior row ``(theta_j - mu_j)/sigma_j`` per free parameter with a Gaussian prior.  Thomson
is never a weight.  The objective is ``chi^2 = sum r^2`` (diagonal Gaussian likelihood
+ Gaussian priors); the MAP point is found by Levenberg-Marquardt with a forward-difference Jacobian.

Parameters (:data:`PARAMETERS`; default free set: ``log_strength`` and the ``a_corr`` current corrections):

=================  ==========================================  ======================================
name               meaning                                     default prior / bounds
=================  ==========================================  ======================================
``log_strength``   ln of the sawtooth strength                 N(ln Lambda_0, 1.0); Lambda_0 at f = 0.37
                   ``Lambda = -ln(1 - 0.999 f) nu`` [1/s]      and a 60 ms period; bounds ln(1e-2)..ln(1e4)
``f``              reconnection fraction at fixed strength     no Gaussian prior (uniform 0.1-1)
``s_pfast``        fast-ion pressure scale                     N(1, 0.15); bounds 0.2..3
``s_cd``           driven-current scale                        N(1, 0.20); bounds 0..3
``s_bs``           bootstrap multiplier                        N(1, 0.5); bounds 0..3
``a_corr_k``       current-correction amplitude k (fraction    N(0, 0.02) (fraction of Ip); +-0.2
                   of Ip; Gaussians on rho_tor,norm,
                   ``current_model.correction_shapes``)
=================  ==========================================  ======================================

Every prior, bound, step rule and stopping criterion is an explicit setting recorded in the
result (they are decisions for the user; the defaults are this package's proposals).

Algorithm (Levenberg-Marquardt, H. B. Nielsen's damping update with the diagonal scaling of
D. W. Marquardt, J. SIAM 11, 431 (1963)):

1. center = consistent solve at the initial parameters (from the supplied starting state); the
   channel set of the fit is fixed there (a later state that cannot model one of these channels is
   a failed trial, never a silently smaller chi^2);
2. Jacobian columns by forward differences, each a FULL CONSISTENT solve warm from the center (no
   lagged inputs); the step of parameter j starts at ``step0_j`` and is enlarged x``step_growth``
   (up to ``max_step_growths`` times) until ``|dr| >= step_noise_factor * |noise|``, where ``noise``
   is the residual change of a consistent re-solve of the first center's parameters from another
   state (measured once, with its chi^2 noise ``noise_chi2``); a column still below that is flagged
   "no information"; a column whose solve fails is retried once with the opposite step
   (``column_retry_opposite``), otherwise the fit stops as ``jacobian_failed``;
3. stopping tests at the center, with that fresh Jacobian (:func:`stopping_tests`): converged when
   (a) the UNDAMPED Gauss-Newton step is below ``step_tol_sigma`` posterior sigmas in every
   parameter, or (b) the Newton decrement ``g^T A^-1 g`` (the predicted chi^2 decrease of that step)
   is below ``max(grad_tol_chi2, noise_factor * noise_chi2)``, or (c) the last ``chi2_rel_window``
   accepted steps each decreased chi^2 by less than ``chi2_rel_tol`` (relative) while the GN step is
   below 1 sigma.  A single damped step is never convergence: the tests are always made at a new
   center with a new Jacobian.  When a test holds, the undamped Gauss-Newton step it computed is
   taken as a last trial (``final_gn_step``) and kept if chi^2 does not rise by more than the noise;
4. LM step ``(J^T J + lambda diag(J^T J)) delta = -J^T r`` shrunk as a whole to the trust caps
   ``max_step`` and clipped to the bounds; when the cap binds and the step does not change with
   lambda, lambda is raised without a solve; the trial is a FULL consistent solve warm from the
   center; a failed trial (any solve not converged or not verified, a model failure, a q-consistency
   failure, a changed channel set) is a rejected step (lambda x ``lm_up``);
5. a trial is accepted only if it lowers chi^2 by more than ``noise_factor * noise_chi2``; when no
   trial does up to ``lm_lambda_max`` the fit stops as ``stalled_no_significant_descent`` (NOT
   converged; the stopping-test values are recorded);
6. final: the accepted center is consistent and verified by construction; the Laplace covariance
   ``(J_a^T J_a)^-1`` (data and prior rows) uses a fresh consistent-solve Jacobian at that center.

Statuses: ``converged``, ``stalled_no_significant_descent``, ``max_iter``, ``failed_trials``,
``jacobian_failed``, ``failed_initial_consistency``, ``no_free_parameters``.  The number of forward
solves, wall time, every trial (predicted and realized chi^2 change, gain ratio) are recorded, and
``stopping_summary`` (:func:`stopping_summary`) states which stopping test ended the fit, the value
of test (b)'s floor and whether its noise-scaled branch set it, and whether the final Gauss-Newton
step was kept with a chi^2 increase (and by how much).

Edge taper: the whole-reconstruction tapered restart after a limit cycle in the first center is a
NON-DEFAULT option (``taper_restart=False`` by default); without it such a center is a failure.
"""

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from . import current_model as _cm
from .current import sawtooth as _saw
from .score import chi2_summary, score

#: Reference crash period of the default strength prior (proposal: median 60 ms) [s].
PRIOR_PERIOD_S = 0.060
#: Reference reconnection fraction (Bateman et al. 2006 hint) used by the prior center.
PRIOR_F = 0.37


def _lam0():
    return float(math.log(_saw.crash_strength(PRIOR_F, 1.0 / PRIOR_PERIOD_S)))


#: Parameter specifications: prior (mean, sigma; sigma None = no prior row), bounds, initial step.
PARAMETERS = {
    "log_strength": dict(prior=(_lam0(), 1.0), bounds=(math.log(1e-2), math.log(1e4)), step0=0.1,
                         unit="ln(1/s)", note="sawtooth strength along -ln(1-0.999f) nu (MSE-constrained)"),
    "f": dict(prior=(None, None), bounds=(0.1, 1.0), step0=0.05, unit="1",
              note="reconnection fraction at fixed strength (uniform prior 0.1-1: bounds only)"),
    "s_pfast": dict(prior=(1.0, 0.15), bounds=(0.2, 3.0), step0=0.05, unit="1", note="fast-ion pressure scale"),
    "s_cd": dict(prior=(1.0, 0.20), bounds=(0.0, 3.0), step0=0.05, unit="1", note="driven-current scale"),
    "s_bs": dict(prior=(1.0, 0.5), bounds=(0.0, 3.0), step0=0.05, unit="1", note="bootstrap multiplier"),
    "a_corr": dict(prior=(0.0, 0.02), bounds=(-0.2, 0.2), step0=0.005, unit="fraction of Ip",
                   note="model-discrepancy current amplitudes (one per center)"),
}


@dataclass
class FitSettings:
    """Free parameters, priors and the algorithm's settings (all recorded; all decisions)."""
    free: Sequence[str] = ("log_strength", "a_corr")
    priors: Dict[str, Any] = field(default_factory=dict)        # overrides: name -> (mean, sigma)
    bounds: Dict[str, Any] = field(default_factory=dict)        # overrides: name -> (lo, hi)
    max_iter: int = 12
    lm_lambda0: float = 1.0e-2
    lm_up: float = 10.0
    lm_down: float = 0.3
    lm_lambda_min: float = 1.0e-6
    lm_lambda_max: float = 1.0e6
    step_tol_sigma: float = 0.1          # (a) undamped GN step below this many posterior sigmas
    grad_tol_chi2: float = 1.0e-2        # (b) Newton decrement floor (chi^2 units)
    noise_factor: float = 1.0            # acceptance / (b): multiples of the measured chi^2 noise
    chi2_rel_tol: float = 1.0e-4         # (c) relative chi^2 decrease per accepted step
    chi2_rel_window: int = 2             # (c) number of consecutive accepted steps
    step_noise_factor: float = 20.0
    step_growth: float = 4.0
    max_step_growths: int = 3
    max_failed_trials: int = 4
    fit_groups: Sequence[str] = ("flux_loop", "bpol_probe", "pf_current", "ip", "mse")
    max_step: Dict[str, float] = field(default_factory=lambda: dict(log_strength=0.5, f=0.1, s_pfast=0.1, s_cd=0.1,
                                                                    s_bs=0.2, a_corr=0.01))
    fitted_coil_role: str = "fitted"
    column_retry_opposite: bool = True   # a failed column solve is retried once with the opposite step
    final_gn_step: bool = True           # take the undamped GN step of the converged test (accepted within noise)
    taper_restart: bool = False          # NON-DEFAULT: restart the whole reconstruction tapered (module docstring)

    def as_dict(self):
        return asdict(self)


def _expand(free, cfg):
    """Free parameter names expanded (``a_corr`` -> one entry per correction center)."""
    out = []
    for name in free:
        if name == "a_corr":
            out += [f"a_corr_{k}" for k in range(len(cfg.corr_centers_rho))]
        elif name in PARAMETERS:
            out.append(name)
        else:
            raise ValueError(f"unknown parameter {name!r}; known: {sorted(PARAMETERS)}")
    return out


def _spec(name, settings):
    base = PARAMETERS["a_corr" if name.startswith("a_corr_") else name]
    prior = settings.priors.get(name, base["prior"])
    bounds = settings.bounds.get(name, base["bounds"])
    return dict(prior=tuple(prior), bounds=tuple(bounds), step0=base["step0"], unit=base["unit"], note=base["note"])


def _to_params(theta, names, base):
    p = _cm.full_params(base)
    a = list(p["a_corr"])
    for n, v in zip(names, theta):
        if n.startswith("a_corr_"):
            a[int(n.split("_")[-1])] = float(v)
        else:
            p[n] = float(v)
    p["a_corr"] = tuple(a)
    return p


def _from_params(p, names):
    p = _cm.full_params(p)
    out = []
    for n in names:
        out.append(p["a_corr"][int(n.split("_")[-1])] if n.startswith("a_corr_") else p[n])
    return np.array(out, dtype=float)


# ============================================================== residuals
def fit_groups(ms, settings):
    """The groups that enter the fit and why others do not."""
    use, why = [], {}
    for g in settings.fit_groups:
        if g not in ms.groups:
            why[g] = "absent from the measurement set"
            continue
        if g == "mse" and not bool(ms.groups[g].meta.get("er_corrected", False)):
            why[g] = "MSE not declared E_r-corrected: not a fit constraint"
            continue
        use.append(g)
    return use, why


def residual_vector(fm, ms, groups, settings, coil_roles):
    """Normalized residuals of the fit groups on the solver's current state, plus the score table."""
    tab = score(fm.gs, ms, groups=groups)
    parts, index = [], []
    for g in groups:
        r = np.asarray(tab.residual(g), dtype=float)
        if g == "pf_current":
            grp = ms.groups[g]
            used = tab.used(g)
            names = [str(n) for n in grp.geometry.get("coil_names", grp.labels)]
            lab = [names[i] for i in np.nonzero(used)[0]]
            keep = np.array([coil_roles.get(n) == settings.fitted_coil_role for n in lab], dtype=bool)
            r = r[keep]
            lab = [lab_i for lab_i, k in zip(lab, keep) if k]
        else:
            grp = ms.groups[g]
            lab = [grp.labels[i] for i in np.nonzero(tab.used(g))[0]]
        parts.append(r)
        index += [(g, s) for s in lab]
    return (np.concatenate(parts) if parts else np.zeros(0)), index, tab


def _prior_rows(theta, names, settings):
    rows, lab = [], []
    for n, v in zip(names, theta):
        mu, sig = _spec(n, settings)["prior"]
        if sig is not None:
            rows.append((v - mu) / sig)
            lab.append(("prior", n))
    return np.array(rows, dtype=float), lab


def _prior_jac(names, settings):
    J = []
    for j, n in enumerate(names):
        mu, sig = _spec(n, settings)["prior"]
        if sig is not None:
            row = np.zeros(len(names))
            row[j] = 1.0 / sig
            J.append(row)
    return np.array(J).reshape(-1, len(names))


def constraints_table(tab, ms, groups):
    """``equilibrium.time_slice[:].constraints``-compatible table of one score table: per group a list
    of ``{source, measured, measured_error_upper, reconstructed, weight, chi_squared}`` over the used
    channels (IMAS DD names; MSE in the group's own quantity)."""
    names = {"mse": "mse_polarisation_angle"}
    out = {}
    for g in groups:
        grp = ms.groups[g]
        use = tab.used(g)
        model = np.asarray(tab.model[g], dtype=float)
        rows = []
        for i in np.nonzero(use)[0]:
            r = grp.weight[i] * (model[i] - grp.measured[i]) / grp.sigma[i]
            rows.append(dict(source=str(grp.labels[i]), measured=float(grp.measured[i]),
                             measured_error_upper=float(grp.sigma[i]), reconstructed=float(model[i]),
                             weight=float(grp.weight[i]), chi_squared=float(r * r)))
        out[names.get(g, g)] = rows
    return out


# ============================================================== the fit
@dataclass
class FitResult:
    status: str
    params: Dict[str, Any]
    free: Sequence[str]
    theta: np.ndarray
    sigma: np.ndarray
    covariance: np.ndarray
    chi2: Dict[str, Any]
    record: Dict[str, Any]


def _consistent(fm, kin, src, params, cfg, start, z0, label, initial_profiles=None, state_fn=None):
    return _cm.consistent_solve(fm, kin, src, params, cfg, start=start, z0=z0, label=label,
                                initial_profiles=initial_profiles, state_fn=state_fn)


def _same_index(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def lm_step(A, g, lam, cap):
    """Damped Gauss-Newton step ``(A + lam diag(A)) delta = -g`` shrunk as a whole to the trust caps.
    Returns ``(delta, capped)``."""
    Ad = A + lam * np.diag(np.maximum(np.diag(A), 1e-12))
    delta = -np.linalg.solve(Ad, g)
    scale = float(np.max(np.abs(delta) / cap)) if np.any(delta) else 0.0
    if scale > 1.0:
        return delta / scale, True
    return delta, False


def stopping_tests(A, g, settings, noise_chi2, history):
    """The stated stopping criteria at a center with a FRESH Jacobian (module docstring).

    ``A = J_a^T J_a`` (data and prior rows), ``g = J_a^T r``.  Returns a dict with the undamped
    Gauss-Newton step in posterior sigmas, the Newton decrement ``g^T A^-1 g`` (the predicted chi^2
    decrease of that step), the last relative chi^2 decreases, and which criterion (if any) holds."""
    try:
        Ainv = np.linalg.inv(A)
        dgn = -Ainv @ g
        post = np.sqrt(np.maximum(np.diag(Ainv), 0.0))
        step_sig = np.abs(dgn) / np.where(post > 0, post, np.inf)
        dec = float(g @ Ainv @ g)
    except np.linalg.LinAlgError:
        return dict(converged=False, reason="J^T J singular")
    grad_floor = max(settings.grad_tol_chi2, settings.noise_factor * noise_chi2)
    rel = [h for h in history[-settings.chi2_rel_window:]]
    out = dict(gn_step_sigma=step_sig.tolist(), newton_decrement=dec, grad_floor=grad_floor,
               grad_floor_fixed=float(settings.grad_tol_chi2), grad_floor_noise=float(settings.noise_factor * noise_chi2),
               grad_floor_noise_scaled=bool(settings.noise_factor * noise_chi2 > settings.grad_tol_chi2),
               last_rel_decreases=rel, noise_chi2=noise_chi2, test=None)
    if np.all(step_sig < settings.step_tol_sigma):
        out.update(converged=True, test="a",
                   criterion=f"undamped Gauss-Newton step < {settings.step_tol_sigma} posterior sigma in every parameter")
    elif dec <= grad_floor:
        out.update(converged=True, test="b",
                   criterion=f"Newton decrement g^T A^-1 g = {dec:.3g} <= {grad_floor:.3g} "
                             f"(max(grad_tol_chi2, noise_factor x chi^2 noise))")
    elif (len(rel) >= settings.chi2_rel_window and all(r < settings.chi2_rel_tol for r in rel)
          and np.all(step_sig < 1.0)):
        out.update(converged=True, test="c",
                   criterion=f"{settings.chi2_rel_window} accepted steps each decreased chi^2 by "
                             f"< {settings.chi2_rel_tol:g} (relative) and the GN step is < 1 sigma")
    else:
        out.update(converged=False)
    return out


def stopping_summary(status, stop, log):
    """Which stopping test ended a fit and how (recorded in every result as ``stopping_summary``).

    ``test``: ``"a"`` (undamped Gauss-Newton step), ``"b"`` (Newton decrement) or ``"c"`` (small
    relative decreases) when a test held, else ``None``; ``grad_floor`` of test (b) and whether its
    noise-scaled branch set it (``grad_floor_noise_scaled``: ``noise_factor x chi^2 noise`` above
    ``grad_tol_chi2``); ``final_gn_step``: whether the undamped Gauss-Newton step of the test was
    taken and kept, and the chi^2 change it made (``kept_with_chi2_increase`` and ``chi2_increase``
    when it was kept although chi^2 rose, within the measured noise).  Values only; no rule is
    applied here."""
    st = stop or {}
    fin = (log[-1].get("final_gn_step") if log else None)
    fgn = dict(taken=fin is not None)
    if fin is not None:
        dchi = fin.get("chi2_change")
        kept = bool(fin.get("accepted"))
        fgn.update(accepted=kept, status=fin.get("status"), chi2_change=dchi,
                   kept_with_chi2_increase=bool(kept and dchi is not None and dchi > 0),
                   chi2_increase=(float(dchi) if (kept and dchi is not None and dchi > 0) else 0.0))
    return dict(status=status, test=st.get("test") if st.get("converged") else None,
                criterion=st.get("criterion") if st.get("converged") else None,
                grad_floor=st.get("grad_floor"), grad_floor_fixed=st.get("grad_floor_fixed"),
                grad_floor_noise=st.get("grad_floor_noise"), grad_floor_noise_scaled=st.get("grad_floor_noise_scaled"),
                newton_decrement=st.get("newton_decrement"), gn_step_sigma=st.get("gn_step_sigma"),
                final_gn_step=fgn)


def reconstruct(fm, kin, src, cfg, settings=None, params0=None, start=None, initial_profiles=None, label="fit",
                state_fn=None):
    """MAP reconstruction (module docstring).

    ``fm``: :class:`.forward.ForwardModel` in reconstruction mode (with the measurement set);
    ``kin``/``src``: kinetic inputs (:class:`.kinetics.KineticInputs` or
    :class:`.current_model.KineticProfiles`) and :class:`.current_model.SourceProfiles`;
    ``cfg``: :class:`.current_model.CurrentModelSettings`; ``params0``: starting parameters
    (default: the prior means of the free parameters and :data:`current_model.DEFAULT_PARAMS`
    otherwise); ``start``: starting state (snapshot or psi on nodes); ``initial_profiles``:
    ``(x, jphi, p)`` for the first solve when ``start`` is only a guess; ``state_fn``: a supplied
    current state per pass (time-dependent path; the corrections then act on it).
    """
    t_all = time.perf_counter()
    settings = settings or FitSettings()
    names = _expand(settings.free, cfg)
    ms = fm.ms
    coil_roles = {n: v["role"] for n, v in fm.rows_record["coils"]["per_coil"].items()}
    groups, excluded = fit_groups(ms, settings)
    given = dict(params0 or {})
    base = _cm.full_params(given)
    th = np.zeros(len(names))
    for j, n in enumerate(names):
        mu = _spec(n, settings)["prior"][0]
        if n.startswith("a_corr_"):
            th[j] = base["a_corr"][int(n.split("_")[-1])] if "a_corr" in given else (mu or 0.0)
        elif n in given and given[n] is not None:
            th[j] = float(given[n])
        elif mu is not None:
            th[j] = mu
        elif base.get(n) is not None:
            th[j] = float(base[n])
        else:
            raise ValueError(f"no starting value for {n}")
    lo = np.array([_spec(n, settings)["bounds"][0] for n in names])
    hi = np.array([_spec(n, settings)["bounds"][1] for n in names])
    th = np.clip(th, lo, hi)
    cap = np.array([settings.max_step["a_corr" if n.startswith("a_corr_") else n] for n in names])
    n0 = fm.n_solves
    log = []
    rec = dict(label=label, free=names, specs={n: _spec(n, settings) for n in names}, settings=settings.as_dict(),
               model_settings=cfg.as_dict(), fit_groups=groups, excluded_groups=excluded, iterations=log,
               forward=None, kinetics_source=getattr(kin, "source", ""), sources_source=src.source,
               sources_convention=dict(src.convention),
               kinetics_provenance=dict(getattr(kin, "provenance", {}) or {}),
               kinetics_record=dict(getattr(kin, "record", {}) or {}))

    def finish(status, center, J=None, extra=None, jinfo=None):
        rec["status"] = status
        rec["stopping_summary"] = stopping_summary(status, rec.get("final_stopping_tests") or (extra or {}).get(
            "final_stopping_tests") or (log[-1].get("stopping_tests") if log else None), log)
        rec["n_solves"] = fm.n_solves - n0
        rec["wall_s"] = round(time.perf_counter() - t_all, 2)
        rec["forward"] = fm.settings_record()
        if extra:
            rec.update(extra)
        cov = np.full((len(names), len(names)), np.nan)
        sig = np.full(len(names), np.nan)
        chi = {}
        if center is not None:
            chi = center["chi2"]
            if J is not None and len(names):
                Ja = np.vstack((J, _prior_jac(names, settings)))
                try:
                    cov = np.linalg.inv(Ja.T @ Ja)
                    sig = np.sqrt(np.diag(cov))
                except np.linalg.LinAlgError:
                    rec["covariance_note"] = "J^T J singular: covariance not available"
                if jinfo is not None:
                    info_flags = {}
                    for col in jinfo.get("columns", []):
                        if col.get("noise_limited"):
                            info_flags[col["name"]] = "no information: column at the noise floor after the step growths"
                    rec["covariance_column_flags"] = info_flags
        return FitResult(status=status, params=_to_params(center["theta"] if center is not None else th, names, base),
                         free=names, theta=(center["theta"] if center is not None else th).copy(), sigma=sig,
                         covariance=cov, chi2=chi, record=rec)

    def evaluate(theta, start_state, z0, lab, init_prof=None, ref_index=None):
        p = _to_params(theta, names, base)
        res = _consistent(fm, kin, src, p, cfg, start_state, z0, lab, init_prof, state_fn)
        if not res.converged:
            return None, res
        r, idx, tab = residual_vector(fm, ms, groups, settings, coil_roles)
        if ref_index is not None and not _same_index(idx, ref_index):
            res = _cm.ConsistentResult(False, "channel_set_changed", res.passes, res.assembly, res.lag, res.solve,
                                       dict(res.record, reason="a fit channel became unmodelable (the channel set "
                                                               "is fixed at the first center)"), res.wall_s)
            return None, res
        rp, lp = _prior_rows(theta, names, settings)
        full = np.concatenate((r, rp))
        all_groups = [g for g in ms.groups if g in ("flux_loop", "bpol_probe", "ip", "pf_current", "mse")]
        all_tab = score(fm.gs, ms, groups=all_groups)
        summ = {k: dict(chi2=v[0], n=v[1], chi2_per_n=v[2]) for k, v in chi2_summary(all_tab).items()}
        fit_chi = {g: float(np.sum(r[[i for i, (gg, _) in enumerate(idx) if gg == g]] ** 2)) for g in groups}
        c = dict(theta=np.array(theta, float), params=p, res=res, r=full, r_data=r, index=idx, prior_index=lp,
                 chi2=dict(objective=float(full @ full), data=float(r @ r), prior=float(rp @ rp), fit_groups=fit_chi,
                           n_data=int(r.size), n_params=len(names), dof=int(r.size) - len(names), scored_all=summ,
                           scored_all_dof=dict(n_channels=summ["total"]["n"], n_params=len(names),
                                               dof=summ["total"]["n"] - len(names),
                                               note="chi2/n in scored_all is descriptive (sigma are "
                                                    "conventions); the n_params were fitted to the fit "
                                                    "groups, a subset of these channels")),
                 tab=tab, all_tab=all_tab, snap=fm.snapshot(), probe_correction=fm.probe_correction.copy(),
                 z0=res.assembly["orbit_z"] if res.assembly else None)
        return c, res

    if isinstance(start, np.ndarray):
        fm.set_psi(start)
        start = None
    if start is not None:
        fm.restore(start)
    snap_init = fm.snapshot()
    pc_init = fm.probe_correction.copy()
    center, res0 = evaluate(th, None, None, f"{label}:center0", initial_profiles)
    rec["center0"] = dict(status=res0.status, passes=res0.passes, wall_s=res0.wall_s,
                          passes_record=res0.record.get("passes"), reason=res0.record.get("reason"),
                          q_check=res0.record.get("q_check"))
    if (center is None and res0.status == "limit_cycle_untapered_lock" and settings.taper_restart
            and fm.case.setting.mode.value == "auto" and not fm.taper_restarts):
        # NON-DEFAULT policy option (taper_restart=True): a limit cycle in an untapered solve of this
        # reconstruction before it had a consistent state restarts the WHOLE reconstruction tapered
        fm.lock_tapered(f"limit cycle in {res0.record.get('passes', [{}])[-1].get('k')} pass of {label}:center0 "
                        "(untapered); reconstruction restarted with the edge taper (taper_restart=True)")
        fm.restore(snap_init)
        fm.probe_correction = pc_init.copy()
        center, res0 = evaluate(th, None, None, f"{label}:center0:tapered", initial_profiles)
        rec["center0_restart_tapered"] = dict(status=res0.status, passes=res0.passes, wall_s=res0.wall_s,
                                              passes_record=res0.record.get("passes"),
                                              reason=res0.record.get("reason"))
    if center is None:
        return finish("failed_initial_consistency", None)
    ref_index = center["index"]

    def restore_center(c):
        fm.restore(c["snap"])
        fm.probe_correction = c["probe_correction"].copy()

    if not names:
        restore_center(center)
        extra = _final_extra(center, fm, ms)
        out = finish("no_free_parameters", center, None, extra)
        out.record["center"] = center
        return out

    noise = dict(chi2=None, r=None, norm=None)

    def column_solve(c, theta, lab):
        """Consistent solve at theta warm from the center; returns the data residuals or None."""
        restore_center(c)
        trial, tres = evaluate(theta, c["snap"], c["z0"], lab, ref_index=ref_index)
        state_after = fm.snapshot()
        restore_center(c)
        if trial is None:
            return None, tres.status, None
        return trial["r_data"], "ok", (state_after, trial)

    def measure_noise(c, from_state):
        """chi^2 and residual noise of a consistent re-solve: the center's parameters re-solved from a
        different (nearby) starting state."""
        trial, tres = evaluate(c["theta"], from_state, c["z0"], f"{label}:noise", ref_index=ref_index)
        restore_center(c)
        if trial is None:
            return dict(status=tres.status)
        dr = trial["r_data"] - c["r_data"]
        return dict(status="ok", chi2=abs(trial["chi2"]["objective"] - c["chi2"]["objective"]), r=dr,
                    norm=float(np.linalg.norm(dr)))

    def jacobian(c, lab):
        J = np.zeros((c["r_data"].size, len(names)))
        cols = []
        for j, n in enumerate(names):
            h = _spec(n, settings)["step0"]
            info = dict(name=n, attempts=[])
            for gi in range(settings.max_step_growths + 1):
                tj = c["theta"].copy()
                direction = 1.0 if tj[j] + h <= hi[j] else -1.0
                tj[j] += direction * h
                rj, why, extra_state = column_solve(c, tj, f"{lab}:col{j}")
                if rj is None and settings.column_retry_opposite and tj[j] - 2 * direction * h >= lo[j] \
                        and tj[j] - 2 * direction * h <= hi[j]:
                    # the column solve failed: one retry with the opposite step (a backward difference); the
                    # failure is recorded
                    info["attempts"].append(dict(h=direction * h, status=why, retried_opposite=True))
                    direction = -direction
                    tj = c["theta"].copy()
                    tj[j] += direction * h
                    rj, why, extra_state = column_solve(c, tj, f"{lab}:col{j}:opp")
                if rj is None:
                    info["attempts"].append(dict(h=direction * h, status=why))
                    break
                if noise["norm"] is None:
                    m = measure_noise(c, extra_state[0])
                    rec["noise"] = dict(status=m["status"], chi2=m.get("chi2"), norm=m.get("norm"),
                                        method="consistent re-solve of the first center's parameters from the "
                                               "state of its first Jacobian column")
                    if m["status"] != "ok":
                        return None, dict(reason=f"noise measurement failed: {m['status']}", columns=cols)
                    noise.update(chi2=m["chi2"], r=m["r"], norm=m["norm"])
                dr = rj - c["r_data"]
                ok = np.linalg.norm(dr) >= settings.step_noise_factor * noise["norm"]
                info["attempts"].append(dict(h=direction * h, norm_dr=float(np.linalg.norm(dr)), ok=bool(ok)))
                J[:, j] = dr / (direction * h)
                if ok:
                    break
                h *= settings.step_growth
            info["final_h"] = info["attempts"][-1].get("h")
            info["noise_limited"] = not info["attempts"][-1].get("ok", False)
            cols.append(info)
            if info["attempts"][-1].get("status"):
                return None, dict(reason=f"column {n} failed: {info['attempts'][-1]['status']}", columns=cols)
        return J, dict(noise_norm=noise["norm"], noise_chi2=noise["chi2"],
                       noise_source="consistent re-solve (measured once)", columns=cols,
                       column_kind="consistent solves (no lagged inputs)")

    lam = settings.lm_lambda0
    fails = 0
    J = None
    jinfo = None
    status = "max_iter"
    rel_hist = []
    stop = None
    for it in range(settings.max_iter):
        J, jinfo = jacobian(center, f"{label}:it{it}")
        entry = dict(it=it, theta=center["theta"].tolist(), chi2=center["chi2"]["objective"], jacobian=jinfo)
        if J is None:
            entry["status"] = "jacobian_failed"
            log.append(entry)
            return finish("jacobian_failed", center, None)
        Ja = np.vstack((J, _prior_jac(names, settings)))
        g = Ja.T @ center["r"]
        A = Ja.T @ Ja
        stop = stopping_tests(A, g, settings, noise["chi2"] or 0.0, rel_hist)
        entry["stopping_tests"] = stop
        if stop.get("converged"):
            status = "converged"
            entry["stop_rule"] = stop["criterion"]
            # final undamped Gauss-Newton step (the stopping test computed it): taken when it does not
            # raise chi^2 by more than the measured noise; the covariance is then evaluated at the new center
            try:
                dgn = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                dgn = np.zeros_like(g)
            tt = np.clip(center["theta"] + dgn, lo, hi)
            if settings.final_gn_step and np.any(tt != center["theta"]):
                trial, tres = evaluate(tt, center["snap"], center["z0"], f"{label}:it{it}:final_gn",
                                       ref_index=ref_index)
                fin = dict(delta=(tt - center["theta"]).tolist(), status=(tres.status if trial is None else "consistent"),
                           chi2=None if trial is None else trial["chi2"]["objective"],
                           chi2_center=center["chi2"]["objective"],
                           chi2_change=(None if trial is None else
                                        trial["chi2"]["objective"] - center["chi2"]["objective"]))
                if trial is not None and trial["chi2"]["objective"] <= center["chi2"]["objective"] + \
                        settings.noise_factor * (noise["chi2"] or 0.0):
                    center = trial
                    fin["accepted"] = True
                else:
                    fin["accepted"] = False
                    restore_center(center)
                entry["final_gn_step"] = fin
            log.append(entry)
            break
        accepted = False
        last_delta = None
        while not accepted:
            try:
                delta, capped = lm_step(A, g, lam, cap)
            except np.linalg.LinAlgError:
                lam *= settings.lm_up
                continue
            tt = np.clip(center["theta"] + delta, lo, hi)
            step = tt - center["theta"]
            if last_delta is not None and np.allclose(step, last_delta, rtol=1e-6, atol=1e-12):
                # the trust cap binds and the step did not change: raise lambda without a solve
                lam *= settings.lm_up
                if lam > settings.lm_lambda_max:
                    break
                continue
            last_delta = step
            pred = float(-(g @ step) * 2.0 - step @ A @ step)       # predicted chi^2 decrease (quadratic model)
            trial, tres = evaluate(tt, center["snap"], center["z0"], f"{label}:it{it}:trial", ref_index=ref_index)
            tr = dict(lam=lam, delta=step.tolist(), capped=bool(capped), predicted_dchi2=pred,
                      status=(tres.status if trial is None else "consistent"), passes=tres.passes,
                      chi2=(None if trial is None else trial["chi2"]["objective"]), wall_s=tres.wall_s)
            entry.setdefault("trials", []).append(tr)
            if trial is None:
                fails += 1
                lam *= settings.lm_up
                restore_center(center)
                if fails >= settings.max_failed_trials:
                    log.append(entry)
                    return finish("failed_trials", center, J, jinfo=jinfo)
                continue
            dchi = center["chi2"]["objective"] - trial["chi2"]["objective"]
            tr["realized_dchi2"] = dchi
            tr["gain_ratio"] = dchi / pred if pred > 0 else None
            if dchi > settings.noise_factor * (noise["chi2"] or 0.0) and dchi > 0:
                rel_hist.append(dchi / max(center["chi2"]["objective"], 1e-300))
                center = trial
                lam = max(lam * settings.lm_down, settings.lm_lambda_min)
                accepted = True
                entry.update(accepted=True, dchi2=dchi)
            else:
                lam *= settings.lm_up
                restore_center(center)
                if lam > settings.lm_lambda_max:
                    break
        if not accepted:
            entry["note"] = (f"no trial decreased chi^2 by more than the measured noise up to lambda = "
                             f"{settings.lm_lambda_max:g}; stopping criteria NOT met")
            log.append(entry)
            status = "stalled_no_significant_descent"
            break
        log.append(entry)
    restore_center(center)
    # Laplace covariance from a FRESH Jacobian of consistent solves at the final, verified center (the
    # convergence test above already computed one at this center when it stopped there)
    if status == "converged" and J is not None and log and log[-1].get("stopping_tests") is not None \
            and log[-1]["theta"] == center["theta"].tolist():
        Jf, jf_info = J, dict(jinfo, reused="Jacobian computed at the final center for the stopping test")
    else:
        Jf, jf_info = jacobian(center, f"{label}:final")
    restore_center(center)
    extra = _final_extra(center, fm, ms)
    extra["final_jacobian"] = jf_info
    extra["final_stopping_tests"] = stop
    out = finish(status if Jf is not None else status + "_no_final_jacobian", center, Jf, extra, jinfo=jf_info)
    out.record["center"] = center
    return out


def _final_extra(center, fm, ms):
    asm = center["res"].assembly
    return dict(final_center=dict(passes=center["res"].passes, passes_record=center["res"].record.get("passes"),
                                  stats=center["res"].solve.stats, solve_status=center["res"].solve.status,
                                  verification=center["res"].solve.record["policy"].get("verification"),
                                  q_check=center["res"].record.get("q_check")),
                assembly=dict({k: v for k, v in asm.items() if k not in ("orbit_z", "state")}),
                constraints=constraints_table(center["all_tab"], ms, list(center["all_tab"].model)),
                chi2_detail=center["chi2"], boundary=fm.boundary_distance(),
                excluded_channels={k: v for k, v in center["all_tab"].excluded.items()},
                score_notes=list(center["all_tab"].notes))
