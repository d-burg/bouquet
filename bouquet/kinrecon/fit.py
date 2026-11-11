r"""The kinetic reconstruction P1: a MAP fit over a small set of physical parameters.

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

Parameters (:data:`PARAMETERS`; default free set: ``log_strength`` only):

=================  ==========================================  ======================================
name               meaning                                     default prior / bounds
=================  ==========================================  ======================================
``log_strength``   ln of the sawtooth strength                 N(ln Lambda_0, 1.0); Lambda_0 at f = 0.37
                   ``Lambda = -ln(1 - 0.999 f) nu`` [1/s]      and a 60 ms period; bounds ln(1e-2)..ln(1e4)
``f``              reconnection fraction at fixed strength     no Gaussian prior (uniform 0.1-1)
``s_pfast``        fast-ion pressure scale                     N(1, 0.15); bounds 0.2..3
``s_cd``           driven-current scale                        N(1, 0.20); bounds 0..3
``s_bs``           bootstrap multiplier                        N(1, 0.5); bounds 0..3
``a_corr_k``       model-discrepancy current amplitude k       N(0, 0.02) (fraction of Ip); +-0.2
                   (Gaussian shapes in rho, ``current_model``)
=================  ==========================================  ======================================

Every prior, bound, step rule and stopping criterion is an explicit setting recorded in the
result (they are decisions for the user; the defaults are this package's proposals:
A5, A9).

Algorithm (Levenberg-Marquardt, H. B. Nielsen's damping update with the diagonal scaling of
D. W. Marquardt, J. SIAM 11, 431 (1963)):

1. center = consistent solve at the initial parameters (from the supplied starting state);
2. measured re-solve noise: one lagged solve at the center parameters gives the residual change a
   column sees with no parameter change (``noise``);
3. Jacobian columns by forward differences, each a LAGGED solve (geometry, conductivity and
   bootstrap from the center) warm from the center; the step of parameter j starts at
   ``step0_j`` and is enlarged x4 (up to 3 times) until ``|dr| >= step_noise_factor * |noise|``
   (default 20, a rule of thumb from development checks); a step that would cross a bound is taken backwards;
4. LM step ``(J^T J + lambda diag(J^T J)) delta = -J^T r``, shrunk as a whole so that no component
   exceeds its trust-region cap ``max_step`` (default 0.5 in ln strength, 0.1 in f and the scales,
   0.2 in s_bs, 0.01 in a_corr), clipped to the bounds; the trial is a
   FULL consistent solve warm from the center; a failed trial (any solve not converged or not
   verified, a model failure) is a rejected step (lambda x 10), never an accepted state;
5. accept if chi^2 decreases; stop when an accepted step is below ``step_tol_sigma`` posterior
   sigmas in every parameter, or the chi^2 decrease is below ``max(chi2_rel_tol chi^2,
   chi2_abs_tol)``, or after ``max_iter`` iterations; the column noise of step 2 is measured once
   (``noise_once``);
6. final: the accepted center is consistent and verified by construction; the Laplace covariance
   ``(J^T J)^-1`` (with prior rows) uses a Jacobian evaluated at the final center.

The number of forward solves, wall time and every iteration are recorded.
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
    free: Sequence[str] = ("log_strength",)
    priors: Dict[str, Any] = field(default_factory=dict)        # overrides: name -> (mean, sigma)
    bounds: Dict[str, Any] = field(default_factory=dict)        # overrides: name -> (lo, hi)
    max_iter: int = 10
    lm_lambda0: float = 1.0e-2
    lm_up: float = 10.0
    lm_down: float = 0.3
    chi2_rel_tol: float = 1.0e-3
    chi2_abs_tol: float = 1.0e-2
    step_tol_sigma: float = 0.1
    noise_once: bool = True
    reuse_final_jacobian: bool = True
    step_noise_factor: float = 20.0
    step_growth: float = 4.0
    max_step_growths: int = 3
    max_failed_trials: int = 4
    fit_groups: Sequence[str] = ("flux_loop", "bpol_probe", "pf_current", "ip", "mse")
    max_step: Dict[str, float] = field(default_factory=lambda: dict(log_strength=0.5, f=0.1, s_pfast=0.1, s_cd=0.1,
                                                                    s_bs=0.2, a_corr=0.01))
    fitted_coil_role: str = "fitted"

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


def _consistent(fm, kin, src, params, cfg, start, z0, label, initial_profiles=None):
    return _cm.consistent_solve(fm, kin, src, params, cfg, start=start, z0=z0, label=label,
                                initial_profiles=initial_profiles)


def reconstruct(fm, kin, src, cfg, settings=None, params0=None, start=None, initial_profiles=None, label="p1"):
    """MAP reconstruction (module docstring).

    ``fm``: :class:`.forward.ForwardModel` in reconstruction mode (with the measurement set);
    ``kin``/``src``: :class:`.current_model.KineticProfiles` / ``SourceProfiles``;
    ``cfg``: :class:`.current_model.CurrentModelSettings`; ``params0``: starting parameters
    (default: the prior means of the free parameters and :data:`current_model.DEFAULT_PARAMS`
    otherwise); ``start``: starting state (snapshot or psi on nodes); ``initial_profiles``:
    ``(x, jphi, p)`` for the first solve when ``start`` is only a guess.
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
    n0 = fm.n_solves
    log = []
    rec = dict(label=label, free=names, specs={n: _spec(n, settings) for n in names}, settings=settings.as_dict(),
               model_settings=cfg.as_dict(), fit_groups=groups, excluded_groups=excluded, iterations=log,
               forward=None, kinetics_source=kin.source, sources_source=src.source,
               sources_convention=dict(src.convention))

    def finish(status, center, J=None, extra=None):
        rec["status"] = status
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
            if J is not None:
                Ja = np.vstack((J, _prior_jac(names, settings)))
                try:
                    cov = np.linalg.inv(Ja.T @ Ja)
                    sig = np.sqrt(np.diag(cov))
                except np.linalg.LinAlgError:
                    rec["covariance_note"] = "J^T J singular: covariance not available"
        return FitResult(status=status, params=_to_params(th, names, base), free=names, theta=th.copy(), sigma=sig,
                         covariance=cov, chi2=chi, record=rec)

    # ---- 1. center
    def evaluate(theta, start_state, z0, lab, init_prof=None):
        p = _to_params(theta, names, base)
        res = _consistent(fm, kin, src, p, cfg, start_state, z0, lab, init_prof)
        if not res.converged:
            return None, res
        r, idx, tab = residual_vector(fm, ms, groups, settings, coil_roles)
        rp, lp = _prior_rows(theta, names, settings)
        full = np.concatenate((r, rp))
        all_groups = [g for g in ms.groups if g in ("flux_loop", "bpol_probe", "ip", "pf_current", "mse")]
        all_tab = score(fm.gs, ms, groups=all_groups)
        summ = {k: dict(chi2=v[0], n=v[1], chi2_per_n=v[2]) for k, v in chi2_summary(all_tab).items()}
        fit_chi = {g: float(np.sum(r[[i for i, (gg, _) in enumerate(idx) if gg == g]] ** 2)) for g in groups}
        c = dict(theta=np.array(theta, float), params=p, res=res, r=full, r_data=r, index=idx + lp,
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
                          passes_record=res0.record.get("passes"), reason=res0.record.get("reason"))
    if center is None and res0.status == "limit_cycle_untapered_lock" and \
            fm.case.setting.mode.value == "auto" and not fm.taper_restarts:
        # edge-taper policy 'auto': a limit cycle was detected in an untapered solve of this
        # reconstruction before it had a consistent state -> the taper is applied to the WHOLE
        # reconstruction (uniform within the case): restart from the starting state, tapered
        fm.lock_tapered(f"limit cycle in {res0.record.get('passes', [{}])[-1].get('k')} pass of {label}:center0 "
                        "(untapered); reconstruction restarted with the edge taper")
        fm.restore(snap_init)
        fm.probe_correction = pc_init.copy()
        center, res0 = evaluate(th, None, None, f"{label}:center0:tapered", initial_profiles)
        rec["center0_restart_tapered"] = dict(status=res0.status, passes=res0.passes, wall_s=res0.wall_s,
                                              passes_record=res0.record.get("passes"),
                                              reason=res0.record.get("reason"))
    if center is None:
        return finish("failed_initial_consistency", None)

    def restore_center(c):
        fm.restore(c["snap"])
        fm.probe_correction = c["probe_correction"].copy()

    # ---- 2. noise of a column evaluation (lagged solve, no parameter change)
    def column(c, theta, lab):
        restore_center(c)
        p = _to_params(theta, names, base)
        res, asm, why = _cm.lagged_solve(fm, c["res"].lag, kin, src, p, cfg, z0=c["z0"], label=lab)
        if res is None or not res.converged:
            restore_center(c)
            return None, why
        r, idx, _ = residual_vector(fm, ms, groups, settings, coil_roles)
        restore_center(c)
        if [i for i in idx] != [i for i in c["index"][:len(idx)]]:
            return None, "channel set changed between the center and the column (a channel became unmodelable)"
        return r, "ok"

    noise_state = {}

    def jacobian(c, lab):
        if settings.noise_once and "nn" in noise_state:
            nn, nrms = noise_state["nn"], noise_state["rms"]
            r0n = c["r_data"]
            noise_src = "measured once at the first center (noise_once)"
        else:
            r0n, why = column(c, c["theta"], f"{lab}:noise")
            if r0n is None:
                return None, dict(reason=f"noise evaluation failed: {why}")
            noise = r0n - c["r_data"]
            nn = float(np.linalg.norm(noise))
            nrms = nn / max(1, noise.size) ** 0.5
            noise_state.update(nn=nn, rms=nrms)
            noise_src = "lagged solve at the center parameters (no parameter change)"
        J = np.zeros((c["r_data"].size, len(names)))
        cols = []
        for j, n in enumerate(names):
            h = _spec(n, settings)["step0"]
            info = dict(name=n, attempts=[])
            for g in range(settings.max_step_growths + 1):
                tj = c["theta"].copy()
                direction = 1.0 if tj[j] + h <= hi[j] else -1.0
                tj[j] += direction * h
                rj, why = column(c, tj, f"{lab}:col{j}")
                if rj is None:
                    info["attempts"].append(dict(h=h, status=why))
                    break
                dr = rj - r0n
                ok = np.linalg.norm(dr) >= settings.step_noise_factor * nn
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
        return J, dict(noise_norm=nn, noise_rms_per_channel=nrms, noise_source=noise_src, columns=cols)

    lam = settings.lm_lambda0
    last_step_small = False
    fails = 0
    J = None
    status = "max_iter"
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
        accepted = False
        while not accepted:
            Ad = A + lam * np.diag(np.maximum(np.diag(A), 1e-12))
            try:
                delta = -np.linalg.solve(Ad, g)
            except np.linalg.LinAlgError:
                lam *= settings.lm_up
                continue
            cap = np.array([settings.max_step["a_corr" if n.startswith("a_corr_") else n] for n in names])
            scale = float(np.max(np.abs(delta) / cap)) if np.any(delta) else 0.0
            if scale > 1.0:
                delta = delta / scale             # trust region: the whole step shrunk to the cap
            tt = np.clip(center["theta"] + delta, lo, hi)
            trial, tres = evaluate(tt, center["snap"], center["z0"], f"{label}:it{it}:trial")
            tr = dict(lam=lam, delta=(tt - center["theta"]).tolist(),
                      status=(tres.status if trial is None else "consistent"), passes=tres.passes,
                      chi2=(None if trial is None else trial["chi2"]["objective"]), wall_s=tres.wall_s,
                      pass_walls=[(p.get("geometry_wall_s"), p.get("assembly_wall_s"), p.get("solve_wall_s"))
                                  for p in tres.record.get("passes", [])])
            entry.setdefault("trials", []).append(tr)
            if trial is None:
                fails += 1
                lam *= settings.lm_up
                restore_center(center)
                if fails >= settings.max_failed_trials:
                    log.append(entry)
                    return finish("failed_trials", center, J)
                continue
            if trial["chi2"]["objective"] < center["chi2"]["objective"]:
                dchi = center["chi2"]["objective"] - trial["chi2"]["objective"]
                try:
                    post = np.sqrt(np.diag(np.linalg.inv(A)))
                except np.linalg.LinAlgError:
                    post = np.full(len(names), np.inf)
                step_sig = np.abs(tt - center["theta"]) / post
                center = trial
                lam *= settings.lm_down
                accepted = True
                entry.update(accepted=True, dchi2=dchi, step_in_sigma=step_sig.tolist())
                small_dchi = dchi < max(settings.chi2_rel_tol * trial["chi2"]["objective"], settings.chi2_abs_tol)
                if np.all(step_sig < settings.step_tol_sigma) or small_dchi:
                    status = "converged"
                    entry["stop_rule"] = ("step < %g posterior sigma" % settings.step_tol_sigma
                                          if np.all(step_sig < settings.step_tol_sigma) else
                                          "chi2 decrease < max(%g chi2, %g)" % (settings.chi2_rel_tol,
                                                                                 settings.chi2_abs_tol))
                    last_step_small = bool(np.all(step_sig < settings.step_tol_sigma))
            else:
                lam *= settings.lm_up
                restore_center(center)
                if lam > 1e8:
                    entry["note"] = "lambda exceeded 1e8: no descent direction found; center is the minimum"
                    status = "converged_no_descent"
                    break
        log.append(entry)
        th = center["theta"].copy()
        if status.startswith("converged"):
            break
    th = center["theta"].copy()
    restore_center(center)
    # Laplace covariance with a Jacobian at the final center (the last one is reused when the last
    # accepted step was below step_tol_sigma posterior sigmas: the difference is second order)
    if settings.reuse_final_jacobian and last_step_small and J is not None:
        Jf, jinfo = J, dict(reused="Jacobian of the last iteration (last accepted step < step_tol_sigma)")
    else:
        Jf, jinfo = jacobian(center, f"{label}:final")
    restore_center(center)
    extra = dict(final_jacobian=jinfo, final_center=dict(passes=center["res"].passes,
                                                         passes_record=center["res"].record.get("passes"),
                                                         stats=center["res"].solve.stats,
                                                         solve_status=center["res"].solve.status,
                                                         verification=center["res"].solve.record["policy"].get(
                                                             "verification")),
                 assembly=dict({k: v for k, v in center["res"].assembly.items()
                                if k not in ("orbit_z",)}),
                 constraints=constraints_table(center["all_tab"], ms, list(center["all_tab"].model)),
                 chi2_detail=center["chi2"], boundary=fm.boundary_distance(),
                 excluded_channels={k: v for k, v in center["all_tab"].excluded.items()},
                 score_notes=list(center["all_tab"].notes))
    out = finish(status if Jf is not None else status + "_no_final_jacobian", center, Jf, extra)
    out.record["center"] = center
    return out
