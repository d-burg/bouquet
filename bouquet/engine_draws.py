"""Draws on the unified reconstruction engine (``reconstruction_engine=
"unified"``; docs/engine.md, "Draws").

A draw starts from the reconstruction's DELIVERED state
(:class:`bouquet.engine.EngineState`): its geometry snapshot ``G*``, the
closure coefficients ``x*``, the relaxed bootstrap ``lambda_BS*`` and (when
on) the delivery correction ``Delta*``.  It

* perturbs the kinetics exactly as today's sampler does (the pressure-matched
  kinetic channels, the auxiliary channels) and the PARALLEL inductive
  component ``lambda_ind`` with the existing Gaussian-process sample (the
  toroidal sigma ``sigma_jphi`` and length scale ``j_ls``), and takes the
  bootstrap scale the run drew for it;
* composes the current with ``x*`` HELD::

      J = s_ind(x*) F<1/R>/<B^2> lambda_ind + s_bs(x*) F<1/R>/<B^2> lambda_BS
          + F<1/R>/<B^2> lambda_fix + p'(<R> - F^2<1/R>/<B^2>)

  on the latest solved geometry, the pressure term from the DRAW's own
  ``p'`` every pass, and closes ONE row: Ip, as a scalar amplitude on the
  inductive term in the exact (``jphi-linterp``) measure -- route R2's
  logic, zero extra solves.  With ``engine_draw_q0_row=True`` the
  reconstruction's q0 row is kept too, acting through
  :class:`bouquet.jbs_loop.AxisRowPin` on a second scalar (the bootstrap
  amplitude): the sawtooth two-scalar closure in increment form;
* runs ONE bootstrap loop (:func:`bouquet.jbs_loop.run_jbs_loop`, the draw
  ceiling ``jbs_max_passes_draw``, the current gate as a standing
  criterion, :class:`~bouquet.jbs_loop.JBSNonFinite` at once), then -- in
  ``generate_bouquet`` -- the optional coil homotopy and the existing
  post-homotopy check (:func:`post_homotopy`, dispatched through
  :func:`bouquet.TokaMaker_interface._post_homotopy_jbs`) with its
  saturation guard;
* records l_i(3), l_i(1), beta_N, q0 at its labelled radius, q95, the Ip
  amplitude, the loop record, the delivery check, the change of l_i and of
  the poloidal flux range ``psi_b - psi_a`` against the reconstruction and
  the cost by stage; the post-hoc filters (the l_i band ``l_i_tolerance``
  around the reconstruction's l_i, ``constrain_sawteeth``) are applied to
  the archived draw -- a draw outside a band is ARCHIVED with
  ``in_spec=False``, never dropped.

**Zero-perturbation identity by construction.**  Every perturbed quantity is
formed as ``base + (drawn - base)`` (kinetics, pressure, inductive, aux) so
a zero perturbation is exactly the base; the first pass composes on ``G*``
with ``x*`` and ``lambda_BS*`` (the pressure term's ``p'`` shifted by the
draw's pressure change, zero at identity) and the Ip amplitude's increment
is formed from differences that vanish exactly -- so the first request IS
the stored request, bit for bit (:class:`EngineDrawContext` refuses to run
otherwise).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Version stamp of the engine draw (recorded with every draw).
ENGINE_DRAW_VERSION = "unified-engine-draw/1"

#: The random stream of an engine draw, as recorded.
RNG_STREAM = (
    "one Generator per run (sampling.make_rng(seed)); per draw, in order: "
    "[the bootstrap scale, from the block generate_bouquet draws once before "
    "the loop]; the kinetic channels ne, Te, (Zeff -> ni | ni), Ti, redrawn "
    "together until the flux-surface integral of the thermal pressure "
    "matches the baseline's within p_thresh (perturb_kinetic_equilibrium's "
    "rejection loop, same calls, same order); the auxiliary channels in "
    "their dict order; ONE inductive candidate, redrawn only while it is "
    "negative where its mean exceeds its sigma (up to max_proxy_draws).  "
    "Identical to the legacy draw stream through the first inductive "
    "candidate; the legacy routes then consume further candidates (Fix C "
    "band resampling, the standard route's l_i pre-screen and band loop), "
    "the engine draw none -- so from the first legacy draw that resampled, "
    "the two runs' draws start at different stream offsets.")

#: What an engine draw's loop restarts from after its first solve
#: (``engine_draw_bootstrap_refresh=True``).
REFRESH_SOURCE = (
    "scale x [lambda_BS* + Redl(the draw's kinetics) on the FIRST SOLVED "
    "equilibrium (the draw's pressure) - Redl(the reconstruction's kinetics) "
    "on the starting equilibrium]: the anchor with its draw-kinetics Redl "
    "moved from G* to the geometry the first solve produced; pass 1's own "
    "Redl, so 0 extra solves and 0 extra Redl evaluations")

#: What an engine draw's post-homotopy loop starts from.
INIT_POST_HOMOTOPY = (
    "relaxed: (1 - omega) x the bootstrap the draw carries + omega x Redl on "
    "the delivered post-homotopy equilibrium (the draw's own kinetics, times "
    "its bootstrap scale)")


class EngineDrawRefused(ValueError):
    """An engine draw cannot be run honestly (the stored state does not
    reproduce the stored request, an input the engine draw cannot use, no
    physical inductive candidate).  Raised with the reason."""


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def draw_loop_settings(gc) -> dict:
    """The loop settings of an engine draw: :func:`bouquet.jbs_loop.
    jbs_settings` with ``draw=True`` (the ceiling ``jbs_max_passes_draw``,
    ``post_homotopy_passes``) and the closure-half current gate ON (decision
    9: a standing criterion, as in the engine's reconstruction)."""
    from .jbs_loop import jbs_settings
    s = dict(jbs_settings(gc, draw=True))
    s["gate_current_residual"] = True
    return s


# ---------------------------------------------------------------------------
#  the inputs of one draw
# ---------------------------------------------------------------------------
@dataclass
class EngineDrawInputs:
    """What one draw perturbs.  ``kinetics`` / ``pressure`` /
    ``pressure_thermal`` / ``jB_ind`` on the engine grid ``psi_N``;
    ``kinetics_native`` (``ne, te, ni, ti``) on the kinetic grid (archived);
    ``scale`` the bootstrap scale (1.0 = the reconstruction's)."""

    kinetics: dict
    kinetics_native: dict
    pressure: np.ndarray
    pressure_thermal: np.ndarray
    jB_ind: np.ndarray
    scale: float = 1.0
    aux: dict = field(default_factory=dict)
    sampler: dict = field(default_factory=dict)


class DrawKineticsNonPhysical(ValueError):
    """The DRAWN kinetics of an engine draw are outside the physical domain
    of the bootstrap model -- found before any solve.

    The draw is REJECTED with reason code ``"kinetics_nonphysical"``
    (:data:`~bouquet.TokaMaker_interface.DRAW_REJECTION_REASONS`); ``info``
    (JSON-safe) goes into the rejection record: the first offending
    ``quantity`` (in the order the bootstrap evaluation checks them), its
    ``value`` and ``psi_N`` there, ``n_bad`` of ``n_nodes``, the ``rule``
    broken, and ``all`` -- every offending quantity with its own first
    ``psi_N``, worst ``value`` and node count.
    """

    def __init__(self, message, info):
        super().__init__(message)
        self.info = dict(info)


#: the physical domain of the Redl inputs, in the order
#: :func:`bouquet.physics.evaluate_jBS` checks them: (name, unit, rule,
#: ``bad(array)``)
_KINETIC_DOMAIN = (
    ("ne", "m^-3", "strictly positive", lambda a: ~(a > 0.0)),
    ("ni", "m^-3", "strictly positive", lambda a: ~(a > 0.0)),
    ("te", "eV", "strictly positive", lambda a: ~(a > 0.0)),
    ("ti", "eV", "strictly positive", lambda a: ~(a > 0.0)),
    ("zeff", "", ">= 1", lambda a: ~(a >= 1.0)),
)


def check_draw_kinetics(kinetics, psi_N, label="engine draw"):
    """Refuse non-physical DRAWN kinetics before anything is solved.

    Exactly the input domain :func:`bouquet.physics.evaluate_jBS` enforces
    on the same arrays (every node finite; ``ne, ni, te, ti > 0``;
    ``zeff >= 1``), so the draws rejected are the ones that evaluation
    refused at the draw's first bootstrap evaluation -- which was reported
    as ``anchor_solve_failed`` although no solve had run.  Nothing is
    clipped or redrawn; the sampler is untouched.  Raises
    :class:`DrawKineticsNonPhysical`; returns ``None`` otherwise.
    """
    psi = np.asarray(psi_N, dtype=float)
    found = []
    for name, unit, rule, _bad in _KINETIC_DOMAIN:
        a = np.asarray(kinetics[name], dtype=float)
        if a.ndim == 0:
            a = np.full(psi.shape, float(a))
        if a.shape != psi.shape:
            return                  # a malformed input is the evaluator's
        nonfinite = ~np.isfinite(a)
        if nonfinite.any():
            i = int(np.argmax(nonfinite))
            found.append(dict(quantity=name, rule="finite", unit=unit,
                              psi_N=float(psi[i]), index=i, value=None,
                              n_bad=int(nonfinite.sum()), kind="non_finite"))
    if not found:
        for name, unit, rule, bad_of in _KINETIC_DOMAIN:
            a = np.asarray(kinetics[name], dtype=float)
            if a.ndim == 0:
                a = np.full(psi.shape, float(a))
            bad = bad_of(a)
            if bad.any():
                i = int(np.argmax(bad))
                found.append(dict(
                    quantity=name, rule=rule, unit=unit,
                    psi_N=float(psi[i]), index=i, value=float(a[i]),
                    worst=float(np.min(a[bad])), n_bad=int(bad.sum()),
                    psi_N_range=[float(psi[bad][0]), float(psi[bad][-1])],
                    kind="domain"))
    if not found:
        return
    f = found[0]
    val = "a non-finite value" if f["value"] is None else (
        f"{f['value']:.6g}" + (f" {f['unit']}" if f["unit"] else ""))
    others = "" if len(found) == 1 else (
        "; also " + ", ".join(x["quantity"] for x in found[1:]))
    raise DrawKineticsNonPhysical(
        f"{label}: drawn {f['quantity']} must be {f['rule']} on every "
        f"node; got {val} at psi_N={f['psi_N']:.6g} ({f['n_bad']} of "
        f"{psi.size} node(s)){others}.  No solve was run: the drawn "
        "kinetics are non-physical (an input sigma larger than the profile "
        "it perturbs draws through zero).",
        dict(quantity=f["quantity"], psi_N=f["psi_N"], value=f["value"],
             rule=f["rule"], unit=f["unit"], index=f["index"],
             n_bad=f["n_bad"], n_nodes=int(psi.size), all=found,
             detected="before any solve (the draw's inputs)"))


def _pchip(x, y, xn):
    from .utils import pchip_interp
    return pchip_interp(np.asarray(x, float), np.asarray(y, float),
                        np.asarray(xn, float))


# ---------------------------------------------------------------------------
#  the reconstruction a draw inherits
# ---------------------------------------------------------------------------
class EngineDrawContext:
    """Everything a draw inherits from ONE engine reconstruction (built once
    per run from the live engine and its result).

    *native* is the kinetic-grid base the sampler perturbs:
    ``dict(psi_N, ne, te, ni, ti)`` (+ optional ``z_fast``), exactly the
    arrays ``generate()`` hands the legacy sampler (the Baseline's).  On
    construction the reconstruction's composition is rebuilt from the state
    and must reproduce the stored request BIT FOR BIT, else
    :class:`EngineDrawRefused`."""

    def __init__(self, eng, res, *, loop, native, q0_row=False,
                 label="engine draw", bootstrap_refresh=False, Z_imp=None,
                 zeff_includes_fast=False):
        from .engine import _lin, complete_geometry  # noqa: F401
        from .edge_pressure import pressure_gradient, resolve_edge_pressure
        from .utils import closure_sign_convention, structured_basis_eval
        st = res["state"]
        self.eng = eng
        self.c = eng.c
        #: the reconstruction's edge-pressure settings (bouquet.
        #: edge_pressure): every draw solve and report uses the same
        self.edge = resolve_edge_pressure(
            (getattr(eng, "s", None) or {}).get("edge_pressure"))
        self.psi = np.asarray(eng.psi, dtype=float)
        self.label = str(label)
        self.loop = dict(loop)
        #: engine_draw_bootstrap_refresh: restart the loop's bootstrap after
        #: its first solve (:func:`run_draw`)
        self.bootstrap_refresh = bool(bootstrap_refresh)
        self.reconstruction_converged = bool(res.get("converged", False))
        self.Phi = structured_basis_eval(eng.basis, self.psi)
        K = self.Phi.shape[0]
        self.x = np.asarray(st.x, dtype=float).copy()
        self.s_ind = 1.0 + self.x[:K] @ self.Phi
        self.s_bs = 1.0 + self.x[K:] @ self.Phi
        self.lam = np.asarray(st.lambda_bs, dtype=float).copy()
        self.geom = st.geom
        self.request = np.asarray(st.request, dtype=float).copy()
        self.delta = (None if st.delivery_correction is None else
                      np.asarray(st.delivery_correction, dtype=float).copy())
        self.delivery_correction = self.delta is not None
        # ---- the identity: the stored state composes the stored request
        J, parts = self.compose(self.geom, self.c.jB_ind, self.lam)
        R = J if self.delta is None else J + self.delta
        if not np.array_equal(R, self.request):
            d = float(np.max(np.abs(R - self.request)))
            raise EngineDrawRefused(
                f"{self.label}: composing on the stored geometry with x* and "
                f"lambda_BS* does not reproduce the stored request (max "
                f"|diff| {d:.3e}); the zero-perturbation identity would not "
                "hold -- refusing to draw from this state")
        self.J_star = J
        self.parts_star = parts
        g = self.geom
        lin = [_lin(g, parts[k]) for k in ("ind", "bs", "fix")]
        _sgn, _ips, c_s = closure_sign_convention(*lin, g["c_affine"],
                                                  float(self.c.Ip))
        #: the Ip the reconstruction's delivered composition carries in the
        #: exact measure on G*: its linear part and its affine part
        self.lin_star = _lin(g, J)
        self.c_star = float(c_s)
        self.Ip_star = float(self.lin_star + self.c_star)
        # ---- the pressure term's p' on G* for a draw's pressure (pass 1)
        psi_q = np.asarray(g["psi_q"], dtype=float)
        self._psi_q = psi_q
        self.dPq_star = np.interp(psi_q, self.psi, pressure_gradient(
            self.psi, np.asarray(self.c.pressure, dtype=float)))
        pp = np.asarray(g["pprime"], dtype=float)
        nn = float(self.dPq_star @ self.dPq_star)
        self.sigma_p = float(pp @ self.dPq_star) / nn if nn > 0.0 else 0.0
        if not np.isfinite(self.sigma_p):
            self.sigma_p = 0.0
        # ---- the reconstruction's delivered measurements (the reference):
        # EVERY reference quantity from the one delivered measurement, the
        # flux range and the q-row radius included (not from G*, the last
        # loop pass's geometry), so a zero-perturbation draw's deltas measure
        # only its own reproduction of the delivered state
        m = eng.delivered_meas
        stats = m.get("stats") or {}
        from .engine import _full_frame
        full = _full_frame(m)
        from .physics import SOLVER_Q0_PSI_N
        self.ref = dict(
            l_i=float(m["li"]), l_i_1=_f(m.get("li_1")),
            q_row=float(m["q_row"]),
            q_row_psi_N=float(m["geom"]["psi_q"][0]),
            q0_stats=_f(stats.get("q_0")),
            q0_stats_psi_N=float(SOLVER_Q0_PSI_N),
            q95=_f(stats.get("q_95")), beta_n=_f(full.get("beta_n")),
            Ip=_f(m.get("Ip")), flux_range=flux_range(m))
        if m.get("pressure_frames") is not None:
            self.ref["pressure_frames"] = _frames(m)
        # ---- the kinetic-grid base of the sampler
        nat = dict(native)
        self.native = {k: np.asarray(v, dtype=float)
                       for k, v in nat.items() if v is not None}
        self.psi_kin = self.native["psi_N"]
        #: the baseline's impurity charge and Z_eff convention (the sampler's
        #: Z_eff window and ni derivation; bouquet.kinetic_sampler)
        self.Z_imp = None if not Z_imp else float(Z_imp)
        self.zeff_includes_fast = bool(zeff_includes_fast)
        self._kin_eq_native = {k: _pchip(self.psi_kin, self.native[k],
                                         self.psi)
                               for k in ("ne", "te", "ni", "ti")}
        self.z_fast_eq = (None if self.native.get("z_fast") is None else
                          _pchip(self.psi_kin, self.native["z_fast"],
                                 self.psi))
        self.pressure_thermal_base = np.asarray(
            self.c.pressure_parts["thermal"], dtype=float)
        # ---- the q0 row (engine_draw_q0_row)
        self.q0_row = bool(q0_row)
        if self.q0_row:
            if "q0" not in eng.rows or st.q0_target is None:
                raise EngineDrawRefused(
                    f"{self.label}: engine_draw_q0_row=True keeps the "
                    "reconstruction's q0 row, but the reconstruction had no "
                    "active q0 row (not requested, or the sawtooth gate "
                    "rejected it)")
            self.q0_target = float(st.q0_target)
            self.row0 = float(np.interp(float(psi_q[0]), self.psi, J))

    # ---- composition -----------------------------------------------------
    def compose(self, geom, jB_ind, jB_bs):
        """``(J, parts)``: the composition with ``x*`` held -- the SAME
        operations, in the same order, as the reconstruction's closure
        assembles its current (``s_ind*ind + s_bs*bs + (fix + pressure)``)."""
        from .engine import compose
        _, p = compose(geom, jB_ind, jB_bs, self.c.jB_fix)
        j_ind = self.s_ind * p["ind"]
        j_bs = self.s_bs * p["bs"]
        j_fix = p["fix"] + p["pressure"]
        J = j_ind + j_bs + j_fix
        return J, dict(ind=j_ind, bs=j_bs, fix=j_fix, driven=p["fix"],
                       pressure=p["pressure"], kappa=p["kappa"])

    def geom_for_pressure(self, pressure):
        """``G*`` with its ``p'`` shifted by the draw's pressure change (the
        first pass composes on it): ``p'* + sigma_p (dP_draw - dP*)`` on the
        row grid, ``sigma_p`` the least-squares factor between the solver's
        ``p'`` on ``G*`` and ``d p / d psi_N``.  Exactly ``G*`` at zero
        perturbation."""
        from .edge_pressure import pressure_gradient
        dPq = np.interp(self._psi_q, self.psi, pressure_gradient(
            self.psi, np.asarray(pressure, dtype=float)))
        g = dict(self.geom)
        g["pprime"] = np.asarray(self.geom["pprime"], dtype=float) \
            + self.sigma_p * (dPq - self.dPq_star)
        return g

    # ---- inputs -----------------------------------------------------------
    def eq_kinetics(self, native_draw, zeff_eq_delta=None):
        """The engine-grid kinetics of a draw: ``base + (drawn - base)``,
        each on the grid the adapter used (exactly the base at zero
        perturbation)."""
        out = {}
        for k in ("ne", "te", "ni", "ti"):
            out[k] = np.asarray(self.c.kinetics[k], dtype=float) + (
                _pchip(self.psi_kin, native_draw[k], self.psi)
                - self._kin_eq_native[k])
        z = np.asarray(self.c.kinetics["zeff"], dtype=float)
        out["zeff"] = z if zeff_eq_delta is None else z + zeff_eq_delta
        return out

    def _thermal(self, k):
        from .physics import ELEMENTARY_CHARGE as EC
        return EC * (k["ne"] * k["te"] + k["ni"] * k["ti"])

    def _impurity(self, k):
        Z = self.c.pressure_parts.get("Z_imp")
        if not Z:
            return None
        from .physics import impurity_pressure
        ne = k["ne"]
        if self.z_fast_eq is not None:
            ne = np.maximum(ne - self.z_fast_eq, 0.0)
        return impurity_pressure(ne, k["ni"], k["ti"], Z)

    def pressures(self, kin_eq):
        """``(total, thermal)`` solve pressure of a draw's kinetics: the
        adapter's own assembly (thermal + impurity + fast) in increment
        form, ``base + (drawn - base)`` per part (exactly the contract's
        pressure at zero perturbation)."""
        k0 = self.c.kinetics
        dth = self._thermal(kin_eq) - self._thermal(k0)
        th = self.pressure_thermal_base + dth
        p = np.asarray(self.c.pressure, dtype=float) + dth
        i1, i0 = self._impurity(kin_eq), self._impurity(k0)
        if i1 is not None:
            p = p + (i1 - i0)
        return p, th

    def zero_inputs(self, scale=1.0):
        """The inputs of a draw with every perturbation zero."""
        nat = {k: self.native[k].copy() for k in ("ne", "te", "ni", "ti")}
        kin = self.eq_kinetics(nat)
        p, th = self.pressures(kin)
        return EngineDrawInputs(
            kinetics=kin, kinetics_native=nat, pressure=p,
            pressure_thermal=th,
            jB_ind=np.asarray(self.c.jB_ind, dtype=float).copy(),
            scale=float(scale), sampler=dict(zero_perturbation=True))


def flux_range(m):
    """``|psi_b - psi_a|`` [Wb/rad] of a backend measurement (its geometry's
    ``dpsi_dpsiN``), or ``None``."""
    g = (m or {}).get("geom") or {}
    return _f(g.get("dpsi_dpsiN"))


def flux_range_change(fr, fr_ref):
    """The change of the poloidal flux range against the reconstruction's:
    absolute [Wb/rad] and relative (``None`` where either is missing)."""
    if fr is None or fr_ref is None or fr_ref == 0.0:
        return dict(flux_range=None, flux_range_rel=None)
    return dict(flux_range=float(fr - fr_ref),
                flux_range_rel=float((fr - fr_ref) / fr_ref))


def _frames(meas):
    """The JSON-able pressure-frames block of a final measurement
    (:func:`bouquet.edge_pressure.pressure_frames`: ``p_sep``, the volume,
    and beta / W_MHD in the solver's frame and with ``p_sep`` added
    back)."""
    pf = meas["pressure_frames"]
    return dict(p_sep=float(pf["p_sep"]), volume=float(pf["volume"]),
                factor=float(pf["factor"]),
                solver={k: float(v) for k, v in pf["solver"].items()},
                full={k: float(v) for k, v in pf["full"].items()})


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ---------------------------------------------------------------------------
#  the sampler (perturb_kinetic_equilibrium's, in increment form)
# ---------------------------------------------------------------------------
def sample_draw_inputs(ctx, rng, unc, flux_integral, *, scale=1.0,
                       p_thresh=0.05, max_pressure_iter=None,
                       max_proxy_draws=500):
    """One draw's perturbations, drawn from *rng* exactly as
    :func:`bouquet.TokaMaker_interface.perturb_kinetic_equilibrium` draws
    them (same calls, same arguments, same order; :data:`RNG_STREAM`):
    :func:`sample_kinetics` (the pressure-matched kinetic channels, then the
    auxiliary channels), then :func:`sample_inductive` (one inductive
    candidate).  Returns :class:`EngineDrawInputs`."""
    kin = sample_kinetics(ctx, rng, unc, flux_integral, p_thresh=p_thresh,
                          max_pressure_iter=max_pressure_iter)
    cand, tries, n_floor = sample_inductive(ctx, rng, unc,
                                            max_proxy_draws=max_proxy_draws)
    kin.jB_ind = cand
    kin.scale = float(scale)
    kin.sampler.update(inductive_tries=int(tries),
                       inductive_floor_nodes=int(n_floor))
    return kin


def sample_kinetics(ctx, rng, unc, flux_integral, *, p_thresh=0.05,
                    max_pressure_iter=None):
    """The kinetic half of a draw (``jB_ind`` left at the reconstruction's):
    :func:`bouquet.kinetic_sampler.sample_kinetics` -- the sampler every
    solve path shares -- matched on the engine grid's thermal pressure
    (``<.>`` the solver's flux integral), then put on the engine grid in
    increment form (exactly the base at zero perturbation).

    *unc*: ``sigma_ne, sigma_te, sigma_ni, sigma_ti`` (kinetic grid),
    ``n_ls, t_ls``, optional ``aux_sigmas, aux_baselines,
    aux_length_scales, ni_from_zeff, zeff_dne``."""
    from .kinetic_sampler import KineticBase, sample_kinetics as _sample
    nat = ctx.native
    base = KineticBase(
        psi_kin=ctx.psi_kin, ne=nat["ne"], te=nat["te"], ni=nat["ni"],
        ti=nat["ti"], sigma_ne=unc["sigma_ne"], sigma_te=unc["sigma_te"],
        sigma_ni=unc["sigma_ni"], sigma_ti=unc["sigma_ti"],
        n_ls=unc["n_ls"], t_ls=unc["t_ls"],
        aux_sigmas=unc.get("aux_sigmas") or None,
        aux_baselines=unc.get("aux_baselines") or None,
        aux_length_scales=unc.get("aux_length_scales") or None,
        Z_imp=ctx.Z_imp, z_fast=nat.get("z_fast"), z2_fast=nat.get("z2_fast"),
        zeff_includes_fast=ctx.zeff_includes_fast,
        ni_from_zeff=bool(unc.get("ni_from_zeff", True)),
        zeff_dne=unc.get("zeff_dne"))
    psi = ctx.psi
    inp = float(flux_integral(psi, ctx.pressure_thermal_base))

    def _thermal(d):
        return float(flux_integral(psi, ctx.pressures(ctx.eq_kinetics(
            d.native))[1]))

    kd = _sample(base, rng, _thermal, inp, p_thresh=p_thresh,
                 max_pressure_iter=max_pressure_iter)
    aux_out = kd.aux
    dz = None
    zb = (unc.get("aux_baselines") or {}).get("zeff")
    if "zeff" in aux_out and zb is not None:
        zb = np.asarray(zb, dtype=float)
        dz = (np.clip(_pchip(ctx.psi_kin, aux_out["zeff"], psi), 1.0, None)
              - np.clip(_pchip(ctx.psi_kin, zb, psi), 1.0, None))
    kin = ctx.eq_kinetics(kd.native, zeff_eq_delta=dz)
    p, th = ctx.pressures(kin)
    return EngineDrawInputs(
        kinetics=kin, kinetics_native=kd.native, pressure=p,
        pressure_thermal=th,
        jB_ind=np.asarray(ctx.c.jB_ind, dtype=float).copy(), aux=aux_out,
        sampler=dict(pressure_match_iterations=kd.iterations,
                     pressure_match_err_pct=kd.p_err_pct,
                     p_thresh=float(p_thresh),
                     zeff_primary=kd.zeff_primary,
                     rng_stream=RNG_STREAM))


def sample_inductive(ctx, rng, unc, *, max_proxy_draws=500):
    """``(candidate, tries, n_floor_nodes)``: ONE Gaussian-process sample of
    the PARALLEL inductive ``lambda_ind``, drawn IN TOROIDAL UNITS with the
    legacy call -- ``generate_perturbed_GPR(psi, j/j[0], sigma_jphi/j[0],
    j_ls)`` on ``j = s_ind kappa* lambda_ind``, the inductive term of the
    composition on ``G*`` -- and mapped back, ``delta lambda = delta j /
    (s_ind kappa*)``.  The toroidal perturbation is therefore the legacy
    draw's for the same normals (the factorised covariance ``sigma_i sigma_j
    K_ij`` + jitter does not depend on the mean), with today's marginal sigma
    ``sigma_jphi`` and length scale ``j_ls``.  Redrawn only while it is
    negative where its mean exceeds its sigma; where the mean is within
    sigma of zero (the floor zone) a negative excursion is clipped to zero
    (the standard route's half-Gaussian rule; nodes whose mean is itself
    negative are left alone, so a zero sigma is exactly the mean)."""
    from .sampling import generate_perturbed_GPR
    psi = ctx.psi
    lam = np.asarray(ctx.c.jB_ind, dtype=float)
    kap = np.asarray(ctx.parts_star["kappa"], dtype=float)
    fac = ctx.s_ind * kap
    jt = fac * lam
    sig = np.asarray(unc["sigma_jphi"], dtype=float)
    j0 = float(jt[0])
    if not (np.isfinite(j0) and j0 != 0.0):
        raise EngineDrawRefused(
            f"{ctx.label}: the inductive component is zero on axis; the "
            "sampler normalises by it (as the legacy sampler does)")
    mn, sn = jt / j0, sig / j0
    floor = jt <= sig
    for tries in range(1, int(max_proxy_draws) + 1):
        smp = generate_perturbed_GPR(psi, mn, sigma_profile=sn,
                                     length_scale=unc["j_ls"], n_samples=1,
                                     rng=rng, diag_plot=False)
        c = lam + (smp - mn) * j0 / fac
        neg = (c < 0.0) & (lam >= 0.0)
        if np.any(neg & ~floor):
            continue
        return (np.where(neg, 0.0, c), tries,
                int(np.sum(floor & (lam >= 0.0))))
    raise EngineDrawRefused(
        f"{ctx.label}: no inductive candidate without a negative excursion "
        f"where its mean exceeds its sigma in {int(max_proxy_draws)} tries")


# ---------------------------------------------------------------------------
#  one draw's loop
# ---------------------------------------------------------------------------
class _DrawPasses:
    """The kernel's ``step`` for an engine draw: compose with ``x*`` held
    on the latest solved geometry, close the Ip row (and the q0 row) with
    scalar increments, relax, ONE solve, measure."""

    def __init__(self, ctx, backend, inputs, *, geom, pin=None,
                 delta=None, coil_guard=None, phase="loop", on_solve=None):
        self.ctx = ctx
        self.b = backend
        self.inp = inputs
        self.geom = geom
        self.pin = pin
        self.delta = (None if delta is None
                      else np.asarray(delta, dtype=float).copy())
        self.coil_guard = coil_guard
        self.phase = str(phase)
        self.passes = []
        self.last = None
        self.first_request = None
        self.on_solve = on_solve

    def close(self, geom, jbs):
        """``(jc, parts, amp)`` on *geom*: the composition with ``x*`` held
        plus the scalar increments that close the Ip row (and the q0 row) in
        the exact measure -- increments formed from differences that are
        exactly zero at the reconstruction state."""
        from .engine import EngineClosureRefused, _lin
        from .utils import closure_sign_convention
        ctx = self.ctx
        J0, p = ctx.compose(geom, self.inp.jB_ind, jbs)
        li_ = _lin(geom, p["ind"])
        lb_ = _lin(geom, p["bs"])
        lf_ = _lin(geom, p["fix"])
        _s, _i, c_k = closure_sign_convention(li_, lb_, lf_,
                                              geom["c_affine"],
                                              float(ctx.c.Ip))
        dIp = (ctx.lin_star - _lin(geom, J0)) + (ctx.c_star - float(c_k))
        amp = dict(dIp_A=float(dIp))
        if self.pin is None:
            if not (np.isfinite(li_) and li_ != 0.0 and np.isfinite(dIp)):
                raise EngineClosureRefused(
                    f"{ctx.label}: the Ip amplitude is undefined (inductive "
                    f"Ip {li_!r}, increment {dIp!r})")
            d_ind = dIp / li_
            jc = J0 + d_ind * p["ind"]
            amp.update(d_ind=float(d_ind), a_ind=float(1.0 + d_ind))
        else:
            psi0 = float(geom["psi_q"][0])
            A = np.array([[li_, lb_],
                          [float(np.interp(psi0, ctx.psi, p["ind"])),
                           float(np.interp(psi0, ctx.psi, p["bs"]))]])
            rhs = np.array([dIp, float(self.pin.row)
                            - float(np.interp(psi0, ctx.psi, J0))])
            try:
                dd = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError as e:
                raise EngineClosureRefused(
                    f"{ctx.label}: the Ip + q0 two-scalar closure is "
                    f"singular ({e})") from e
            if not np.all(np.isfinite(dd)):
                raise EngineClosureRefused(
                    f"{ctx.label}: the Ip + q0 two-scalar closure returned "
                    f"{dd!r}")
            jc = J0 + dd[0] * p["ind"] + dd[1] * p["bs"]
            amp.update(d_ind=float(dd[0]), d_bs=float(dd[1]),
                       a_ind=float(1.0 + dd[0]), a_bs=float(1.0 + dd[1]),
                       axis_row=float(self.pin.row), axis_psi_N=psi0)
        amp["Ip_exact_measure_A"] = float(_lin(geom, jc) + float(c_k))
        amp["Ip_target_A"] = float(ctx.Ip_star)
        return jc, p, amp

    def step(self, jbs, k, relax=None):
        from .engine import _delivery_stats, complete_geometry, \
            conversion_factor
        ctx = self.ctx
        geom = self.geom
        jc, p, amp = self.close(geom, jbs)
        jint = np.asarray(relax(jc) if relax is not None else jc,
                          dtype=float)
        req = jint if self.delta is None else jint + self.delta
        if self.first_request is None:
            self.first_request = req.copy()
        n0 = int(self.b.n_solves)
        try:
            self.b.solve(req, n_passes=1)
        except Exception as e:
            if k == 0 and self.phase == "loop" and (
                    type(e) is Exception
                    or isinstance(e, (ValueError, RuntimeError))):
                from .TokaMaker_interface import DrawAnchorSolveFailed
                raise DrawAnchorSolveFailed(
                    f"{ctx.label}: the first pass (the reconstruction's "
                    "stored state composed with the draw's components, at "
                    f"the draw's pressure) failed to solve "
                    f"({type(e).__name__}: {str(e).strip()[:300]}); the "
                    "draw is REJECTED") from e
            raise
        if self.on_solve is not None:
            self.on_solve(int(self.b.n_solves) - n0)
        if self.coil_guard is not None:
            self.coil_guard(f"engine draw {self.phase} pass {k + 1}")
        m = self.b.measure()
        g1 = complete_geometry(m["geom"])
        dstat, dvec = _delivery_stats(req, m["achieved"], g1)
        self.last = dict(k=k, jc=jc, jint=jint, req=req, parts=p, amp=amp,
                         geom_composed=geom, geom=g1, m=m, dvec=dvec,
                         jbs=np.asarray(jbs, dtype=float))
        self.passes.append(dict(
            k=int(k), phase=self.phase, **amp, li=float(m["li"]),
            q_row=float(m["q_row"]), Ip=_f(m.get("Ip")), delivery=dstat,
            n_solves=int(self.b.n_solves)))
        self.geom = g1
        meas = dict(w=g1["w_lin"] * conversion_factor(g1), x=ctx.psi,
                    li=m["li"], redl=m["redl"])
        if self.pin is not None:
            meas["q0"] = m["q_row"]
            meas["axis_current_solved"] = float(np.interp(
                float(g1["psi_q"][0]), ctx.psi, jint))
        return meas

    def on_pass(self, k, meas, J, entry):
        if entry.get("omega_next") is None or self.delta is None:
            return
        # the delivery correction (engine_delivery_correction): one Newton
        # step per pass, as the reconstruction takes it
        self.delta = np.asarray(self.last["dvec"], dtype=float).copy()
        self.passes[-1]["delivery_correction_max"] = float(np.max(np.abs(
            self.delta)))


class _Clock:
    """Solves, passes and wall time by stage (anchor, loop, homotopy,
    post_homotopy, filters, archive)."""

    STAGES = ("anchor", "loop", "homotopy", "post_homotopy", "filters",
              "archive")

    def __init__(self, solve_counter=None):
        self.counter = solve_counter
        self.guarded = False
        self.t0 = time.perf_counter()
        self.stages = {s: dict(solves=0, passes=0, wall_s=0.0)
                       for s in self.STAGES}
        self.cur = None
        self._t = None
        self._n = None

    def _count(self):
        try:
            return None if self.counter is None else int(self.counter())
        except Exception:
            return None

    def start(self, stage):
        self.stop()
        self.cur = stage
        self._t = time.perf_counter()
        self._n = self._count()

    def stop(self):
        if self.cur is None:
            return
        st = self.stages[self.cur]
        st["wall_s"] += float(time.perf_counter() - self._t)
        n1 = self._count()
        if n1 is not None and self._n is not None:
            st["solves"] += n1 - self._n
        elif st.get("solves_counted") is None:
            st["solves_counted"] = False
        self.cur = None

    def add(self, stage, solves=0, passes=0):
        self.stages[stage]["solves"] += int(solves)
        self.stages[stage]["passes"] += int(passes)

    def record(self):
        out = {k: dict(v) for k, v in self.stages.items()}
        out["total"] = dict(
            solves=int(sum(v["solves"] for v in self.stages.values())),
            passes=int(sum(v["passes"] for v in self.stages.values())),
            wall_s=float(sum(v["wall_s"] for v in self.stages.values())))
        out["solve_counter"] = ("DrawSolveGuard.n_solves (every GS solve of "
                                "the draw)" if self.guarded else
                                "the engine backend's own solves (the "
                                "homotopy's direct solves are not counted "
                                "without a DrawSolveGuard)")
        return out


def run_draw(ctx, backend, inputs, *, label=None, coil_guard=None,
             clock=None, bnd_diag=None, bootstrap_refresh=None):
    """ONE engine draw from the reconstruction state (module docstring).

    *bootstrap_refresh* (default: ``ctx.bootstrap_refresh``, i.e.
    ``GenerationConfig.engine_draw_bootstrap_refresh``): after the loop's
    FIRST solve the anchor's kinetic increment is re-evaluated on that
    solved geometry and the loop RESTARTS from it (:data:`REFRESH_SOURCE`,
    ``run_jbs_loop(start_refresh=...)``) instead of blending toward the
    start computed on ``G*``.  Zero extra solves and zero extra Redl
    evaluations (pass 1's Redl is the loop's own); the first request, every
    criterion and the fixed point are unchanged -- the path only.  At zero
    perturbation the refreshed bootstrap is ``lambda_BS*`` plus the change
    of Redl between the stored and the RE-SOLVED equilibrium -- rounding on
    the toy stand-in; on the live solver the re-solve reproduces the stored
    state only to its own convergence (measured on the g-file example:
    the refresh step r_j = 1.9e-5, against jbs_rtol_j = 1e-3).

    Returns a dict: ``record`` (the JSON-safe draw record), ``jbs_used``
    (the bootstrap the delivered solve carries: the loop's ``jbs_solved``),
    ``passes`` (the :class:`_DrawPasses` of the loop, the post-homotopy stage
    continues it), ``inputs``.  Raises what the loop raises
    (:class:`~bouquet.jbs_loop.JBSNotConverged` /
    :class:`~bouquet.jbs_loop.JBSNonFinite`, a closure refusal, a failed
    first solve as :class:`~bouquet.TokaMaker_interface.
    DrawAnchorSolveFailed`), or :class:`DrawKineticsNonPhysical` before
    anything is solved -- the draw is then rejected."""
    from .jbs_loop import AxisRowPin, jsonable, run_jbs_loop
    label = str(label or ctx.label)
    # non-physical DRAWN kinetics reject the draw here, before any solve and
    # before the first bootstrap evaluation (its own reason code)
    check_draw_kinetics(inputs.kinetics, ctx.psi, label)
    clock = (clock if clock is not None
             else _Clock(lambda: int(backend.n_solves)))
    backend.set_inputs(pressure=inputs.pressure, kinetics=inputs.kinetics)
    scale = float(inputs.scale)
    # ---- anchor stage (no solve): the kinetic response of Redl on the
    # state the draw starts from, as an increment on lambda_BS*
    clock.start("anchor")
    try:
        r_d = np.asarray(backend.redl(inputs.kinetics), dtype=float)
        r_0 = np.asarray(backend.redl(ctx.c.kinetics), dtype=float)
    except Exception as e:
        clock.stop()
        from .TokaMaker_interface import DrawAnchorSolveFailed
        raise DrawAnchorSolveFailed(
            f"{label}: Redl on the starting equilibrium failed "
            f"({type(e).__name__}: {str(e).strip()[:300]})") from e
    jbs0 = scale * (ctx.lam + (r_d - r_0))
    if bootstrap_refresh is None:
        bootstrap_refresh = bool(getattr(ctx, "bootstrap_refresh", False))
    start_refresh = None
    if bootstrap_refresh:
        def start_refresh(J, meas):
            # the anchor's increment with Redl(draw kinetics) taken on the
            # FIRST SOLVED geometry instead of G* (meas["redl"]: the loop's
            # own pass-1 Redl, the draw's kinetics, solved at the draw's
            # pressure); exactly the anchor's form, so lambda_BS* at zero
            # perturbation up to the re-solve's reproduction of G*
            return scale * (ctx.lam + (np.asarray(meas["redl"], dtype=float)
                                       - r_0))
    clock.start("loop")
    pin = None
    if ctx.q0_row:
        pin = AxisRowPin(ctx.q0_target, float(ctx.eng.s["q0_tol"]),
                         ctx.row0, label=label)
    dp = _DrawPasses(ctx, backend, inputs,
                     geom=ctx.geom_for_pressure(inputs.pressure), pin=pin,
                     delta=ctx.delta, coil_guard=coil_guard,
                     on_solve=lambda n: clock.add("loop", passes=1))
    res = run_jbs_loop(
        jbs0, dp.step, lambda m: scale * np.asarray(m["redl"], dtype=float),
        ctx.loop, Ip=float(ctx.c.Ip),
        meas0=dict(li=ctx.ref["l_i"],
                   q0=(ctx.ref["q_row"] if pin is not None else None)),
        gate_li=True, gate_q0=pin is not None, label=label,
        init_source=("the reconstruction's lambda_BS* + (Redl with the "
                     "draw's kinetics - Redl with the reconstruction's) on "
                     "the starting equilibrium, times the draw's bootstrap "
                     "scale (exactly lambda_BS* at zero perturbation)"),
        on_pass=dp.on_pass, q0_pin=pin, raise_on_fail=True,
        start_refresh=start_refresh)
    if start_refresh is not None:
        res["record"]["bootstrap_refresh"]["source"] = REFRESH_SOURCE
    m_fin = backend.measure(final=True)
    clock.stop()
    if bnd_diag is not None:
        bnd_diag("after engine draw loop")
    out = _finish(ctx, backend, inputs, dp, res, m_fin, pin, label)
    from .jbs_loop import jsonable
    # the anchor and loop stages (generate() completes the clock)
    out["record"]["cost"] = jsonable(clock.record())
    return out


def _finish(ctx, backend, inputs, dp, res, m_fin, pin, label):
    from .engine import conversion_factor
    from .jbs_loop import check_delivered, jsonable
    last = dp.last
    # the bootstrap the delivered (last, relaxed) solve CARRIES -- what the
    # loop's r_j was measured against (jbs_loop "Residuals"); the iterate
    # when the last pass did not blend
    jbs_used = np.asarray(res.get("jbs_solved", res["jbs_used"]),
                          dtype=float)
    meas = res["meas_final"]
    chk = check_delivered(res["J_final"], jbs_used, meas["w"], meas["x"],
                          float(ctx.c.Ip), ctx.loop)
    stats = m_fin.get("stats") or {}
    from .engine import _full_frame
    delivered = dict(
        l_i_3=float(m_fin["li"]), l_i_1=_f(m_fin.get("li_1")),
        beta_n=_f(_full_frame(m_fin).get("beta_n")),
        q0=float(m_fin["q_row"]), q0_psi_N=float(ctx.ref["q_row_psi_N"]),
        q0_stats=_f(stats.get("q_0")),
        q0_stats_psi_N=float(ctx.ref["q0_stats_psi_N"]),
        q95=_f(stats.get("q_95")), Ip=_f(m_fin.get("Ip")),
        delivery_check=dict(r_j=float(chk["r_j"]), r_I=float(chk["r_I"]),
                            ok=bool(chk["ok"])),
        request_minus_achieved=dp.passes[-1]["delivery"])
    delivered["flux_range"] = flux_range(m_fin)
    if m_fin.get("pressure_frames") is not None:
        delivered["pressure_frames"] = _frames(m_fin)
    # the solver's uniform P' rescale (edge_pressure.P_SCALE_DEFINITION)
    delivered["p_scale"] = m_fin.get("p_scale")
    amp = last["amp"]
    dli = float(m_fin["li"]) - float(ctx.ref["l_i"])
    ident = dict(
        pass1_request_bit_identical=bool(np.array_equal(dp.first_request,
                                                        ctx.request)),
        pass1_request_max_abs_diff=float(np.max(np.abs(dp.first_request
                                                       - ctx.request))),
        zero_perturbation=bool(inputs.sampler.get("zero_perturbation",
                                                  False)))
    kap = conversion_factor(last["geom_composed"])
    j_bs_tor = np.asarray(last["parts"]["bs"], dtype=float) * (
        1.0 + float(amp.get("d_bs", 0.0)))
    rec = dict(
        version=ENGINE_DRAW_VERSION, label=label,
        route=("engine draw: x* held, the Ip row as a scalar amplitude on "
               "the inductive in the exact measure"
               + (" + the q0 row (AxisRowPin on the bootstrap amplitude)"
                  if pin is not None else "")),
        reconstruction_converged=ctx.reconstruction_converged,
        identity=ident,
        inputs=dict(scale=float(inputs.scale), **{
            k: v for k, v in inputs.sampler.items() if k != "rng_stream"}),
        rng_stream=RNG_STREAM,
        amplitude=dict(final=amp, per_pass=[
            dict(k=p["k"], a_ind=p.get("a_ind"), a_bs=p.get("a_bs"),
                 dIp_A=p.get("dIp_A"),
                 Ip_exact_measure_A=p.get("Ip_exact_measure_A"))
            for p in dp.passes]),
        loop=res["record"], passes=dp.passes,
        q0_row=(None if pin is None else dict(
            target=float(ctx.q0_target), record=pin.record())),
        delivered=delivered, reference=dict(ctx.ref),
        deltas=dict(
            l_i_3=dli,
            l_i_1=(None if (delivered["l_i_1"] is None
                            or ctx.ref["l_i_1"] is None)
                   else delivered["l_i_1"] - ctx.ref["l_i_1"]),
            beta_n=(None if (delivered["beta_n"] is None
                             or ctx.ref["beta_n"] is None)
                    else delivered["beta_n"] - ctx.ref["beta_n"]),
            q0=float(delivered["q0"] - ctx.ref["q_row"]),
            q95=(None if (delivered["q95"] is None or ctx.ref["q95"] is None)
                 else delivered["q95"] - ctx.ref["q95"]),
            **flux_range_change(delivered["flux_range"],
                                ctx.ref["flux_range"])),
        solves=dict(loop=int(backend.n_solves)))
    return dict(record=jsonable(rec), jbs_used=jbs_used, passes=dp,
                inputs=inputs, pin=pin, measure=m_fin,
                split=dict(j_phi=np.asarray(last["jint"], dtype=float),
                           j_BS=j_bs_tor,
                           j_NBI=kap * np.asarray(
                               ctx.c.jB_fix_parts["nbi"], dtype=float),
                           # the rf part plus any other driven source entry
                           # (as engine._split), so the split sums exactly
                           j_RF=kap * (np.asarray(
                               ctx.c.jB_fix_parts["rf"], dtype=float)
                               + np.asarray(ctx.c.jB_fix_parts.get(
                                   "other", 0.0), dtype=float))))


def post_homotopy(ctx, backend, draw, settings, *, coil_guard=None,
                  label=None, clock=None):
    """The post-homotopy bootstrap check of an engine draw (the homotopy
    moved coils and boundary after the loop converged) -- the existing
    check (:func:`bouquet.jbs_loop.check_delivered` on Redl of the delivered
    equilibrium against the bootstrap the draw carries, the loop
    tolerances), then at most ``settings["post_homotopy_passes"]`` further
    passes AT the current (tight) coil stage with the engine draw's own step
    (``x*`` held, the Ip amplitude, the draw's pressure), each checked by
    *coil_guard* (the homotopy's saturation criterion at those bounds).
    Raises :class:`~bouquet.jbs_loop.JBSNotConverged` when that fails.
    Returns ``(record, jbs_used, j_BS_toroidal, j_phi)`` -- the last two
    ``None`` when the draw was kept as is."""
    from .engine import complete_geometry, conversion_factor
    from .jbs_loop import (JBS_POST_HOMOTOPY_PASSES, check_delivered,
                           jsonable, run_jbs_loop)
    from .TokaMaker_interface import COIL_SATURATION_FRACTION
    label = str(label or ctx.label) + " post-homotopy"
    inputs = draw["inputs"]
    scale = float(inputs.scale)
    jbs_used = np.asarray(draw["jbs_used"], dtype=float)
    n_ph = int(settings.get("post_homotopy_passes", JBS_POST_HOMOTOPY_PASSES))
    backend.set_inputs(pressure=inputs.pressure, kinetics=inputs.kinetics)
    m = backend.measure()
    g = complete_geometry(m["geom"])
    w = g["w_lin"] * conversion_factor(g)
    J = scale * np.asarray(m["redl"], dtype=float)
    chk = check_delivered(J, jbs_used, w, ctx.psi, float(ctx.c.Ip), settings)
    rec = dict(check=jsonable(dict(chk)),
               accepted_without_passes=bool(chk["ok"]),
               solve="engine draw step (x* held, Ip amplitude), one "
                     "jphi-linterp solve per pass, beta-relaxed")
    rec["coil_saturation_guard"] = (
        dict(active=False, note="no guard supplied: re-solves unchecked")
        if coil_guard is None else
        dict(active=True, stage=getattr(coil_guard, "stage", None),
             F_lim=(getattr(coil_guard, "limits", (None, None))[0]),
             VSC_lim=(getattr(coil_guard, "limits", (None, None))[1]),
             fraction=float(COIL_SATURATION_FRACTION),
             checks=getattr(coil_guard, "log", None)))
    print(f"  [engine draw post-homotopy] r_j={chk['r_j']:.3e} "
          f"r_I={chk['r_I']:.3e} -> "
          + ("inside tolerance, draw kept" if chk["ok"] else
             f"outside tolerance, up to {n_ph} passes at the tight coil "
             "stage"), flush=True)
    if chk["ok"]:
        return rec, jbs_used, None, None
    omega = float(settings["relax"])
    jbs0 = (1.0 - omega) * jbs_used + omega * J
    prev = draw["passes"]
    dp = _DrawPasses(ctx, backend, inputs, geom=g, pin=draw.get("pin"),
                     delta=prev.delta, coil_guard=coil_guard,
                     phase="post_homotopy",
                     on_solve=(None if clock is None else
                               (lambda n: clock.add("post_homotopy",
                                                    passes=1))))
    res = run_jbs_loop(
        jbs0, dp.step, lambda mm: scale * np.asarray(mm["redl"], dtype=float),
        settings, Ip=float(ctx.c.Ip), meas0=dict(
            li=m["li"], q0=(m["q_row"] if dp.pin is not None else None)),
        gate_li=True, gate_q0=dp.pin is not None, label=label,
        init_source=INIT_POST_HOMOTOPY, on_pass=dp.on_pass, q0_pin=dp.pin,
        max_passes=n_ph, raise_on_fail=True)
    rec["passes"] = res["record"]
    rec["amplitude"] = dp.last["amp"]
    last = dp.last
    j_bs = np.asarray(last["parts"]["bs"], dtype=float) * (
        1.0 + float(last["amp"].get("d_bs", 0.0)))
    draw["jbs_used"] = np.asarray(res.get("jbs_solved", res["jbs_used"]),
                                  dtype=float)
    draw["passes_post_homotopy"] = dp
    return rec, draw["jbs_used"], j_bs, np.asarray(last["jint"], dtype=float)


# ---------------------------------------------------------------------------
#  the post-hoc filters of the archived draw
# ---------------------------------------------------------------------------
def post_hoc_verdicts(ctx, final, *, l_i_tolerance, constrain_sawteeth,
                      l_i_reference=None):
    """The engine draw's post-hoc filters on the ARCHIVED state *final*
    (a backend measurement): the l_i band ``|l_i - l_i*| <= l_i_tolerance *
    l_i*`` (``l_i_tolerance`` a FRACTION, ``l_i*`` the reconstruction's
    delivered l_i unless *l_i_reference*), and, with
    ``constrain_sawteeth``, ``q0 >= 1`` at the q-row radius (the legacy
    gate's radius, psi_N = psi_pad).  Returns the verdict dict
    (``in_band``)."""
    li_ref = float(ctx.ref["l_i"] if l_i_reference is None
                   else l_i_reference)
    li = float(final["li"])
    rel = abs(li - li_ref) / li_ref if li_ref else float("inf")
    li_ok = bool(np.isfinite(rel) and rel <= float(l_i_tolerance))
    q0 = float(final["q_row"])
    saw_ok = True if not constrain_sawteeth else bool(np.isfinite(q0)
                                                      and q0 >= 1.0)
    reasons = []
    if not li_ok:
        reasons.append(f"l_i band: |l_i - l_i*|/l_i* = {rel:.4f} > "
                       f"l_i_tolerance {float(l_i_tolerance):g}")
    if not saw_ok:
        reasons.append(f"constrain_sawteeth: q0 = {q0:.4f} < 1 at psi_N "
                       f"{ctx.ref['q_row_psi_N']:g}")
    return dict(l_i_3=li, l_i_reference=li_ref, l_i_rel=float(rel),
                l_i_tolerance=float(l_i_tolerance), l_i_in_band=li_ok,
                constrain_sawteeth=bool(constrain_sawteeth), q0=q0,
                q0_psi_N=float(ctx.ref["q_row_psi_N"]), q0_ok=saw_ok,
                in_band=bool(li_ok and saw_ok), reasons=reasons)


#: Group attribute of an archived engine draw: the post-hoc band verdict (an
#: added filter flag, ANDed into ``selected`` by
#: :func:`bouquet.filtering._recompute_selected`).
DRAW_BAND_FLAG = "passes_draw_band"


#: The solver's own text for a solve that ran out of nonlinear iterations
#: (TokaMaker: ``Exceeded "maxits"``), matched case-insensitively anywhere in
#: the exception chain.
_MAXITS_PATTERN = r'exceeded\s*"?maxits'


def solve_hit_iteration_cap(exc) -> bool:
    """Whether *exc* (or an exception it chains) is a Grad-Shafranov solve
    that stopped at its nonlinear iteration cap -- the solver's own reason
    text, not merely a message that mentions the cap."""
    import re
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if re.search(_MAXITS_PATTERN, str(e), flags=re.IGNORECASE):
            return True
        e = e.__cause__ or e.__context__
    return False


def engine_rejection_reason(exc, stage):
    """The :data:`~bouquet.TokaMaker_interface.DRAW_REJECTION_REASONS` code
    of an engine draw rejected by *exc*: non-physical drawn kinetics are
    ``kinetics_nonphysical``, a non-finite bootstrap/current is
    ``jbs_non_finite``, a closure refusal ``engine_closure_refused``, every
    other case the legacy mapping (:func:`~bouquet.TokaMaker_interface.
    _draw_rejection_reason`)."""
    from .engine import EngineClosureRefused
    from .jbs_loop import JBSNonFinite
    from .TokaMaker_interface import _draw_rejection_reason
    if isinstance(exc, DrawKineticsNonPhysical):
        return "kinetics_nonphysical"
    if isinstance(exc, JBSNonFinite):
        return "jbs_non_finite"
    if isinstance(exc, EngineClosureRefused):
        return "engine_closure_refused"
    return _draw_rejection_reason(exc, stage)


# ---------------------------------------------------------------------------
#  generate(): the hook generate_bouquet calls
# ---------------------------------------------------------------------------
def tokamaker_backend(mygs, contract, *, psi_pad, q_psi, maxits,
                      edge_pressure=None, edge_taper=None):
    """The draw's backend on a live solver (monkeypatched by the fast
    tests).  ``edge_pressure`` / ``edge_taper``: the reconstruction's
    settings (:mod:`bouquet.edge_pressure`, :func:`bouquet.engine.
    engine_edge_taper`)."""
    from .engine import TokaMakerBackend
    return TokaMakerBackend(mygs, contract, psi_pad=psi_pad, li_kind="li_3",
                            q_psi=q_psi, maxits=maxits,
                            edge_pressure=edge_pressure,
                            edge_taper=edge_taper)


class GenerateEngineDraws:
    """What ``generate_bouquet(engine_draw=...)`` runs per draw (built by
    ``Bouquet.generate`` from the live reconstruction)."""

    def __init__(self, ctx, *, unc, psi_pad, q_psi=None, maxits=None,
                 homotopy=True, l_i_tolerance=0.05, p_thresh=0.05,
                 max_proxy_draws=500, coil_reg=None):
        self.ctx = ctx
        #: the coil regularisation the reconstruction solved under
        #: (:func:`bouquet.engine.reconstruction_coil_reg`: ``terms`` and
        #: ``record``); every draw installs exactly ``terms``.  ``None`` or
        #: ``terms=None`` (a solver ``setup_solver`` did not prepare): the
        #: historical exploratory regularisation, recorded as such
        self.coil_reg = coil_reg
        self.unc = dict(unc)
        self.psi_pad = float(psi_pad)
        self.q_psi = q_psi
        self.maxits = maxits
        self.homotopy = bool(homotopy)
        self.l_i_tolerance = float(l_i_tolerance)
        self.p_thresh = float(p_thresh)
        self.max_proxy_draws = int(max_proxy_draws)
        self.loop_settings = dict(ctx.loop)
        self._cur = None
        self._cap_saved = None
        self._timer = None
        #: every solve of a draw that stopped at the cap without converging:
        #: ``draw``, ``stage`` / ``where``, ``maxits``, ``iterations`` (the
        #: cap: the solver stops there), ``seconds``, ``outcome``
        #: (``"rolled_back"`` / ``"rejected"``), ``error``
        self.cap_events = []
        #: every homotopy rollback re-solve that failed for another reason
        #: than the cap (the draw is rejected, ``homotopy_rollback_failed``)
        self.rollback_failures = []
        #: set to a dict by ``Bouquet.verify_sigma0_consistency``'s route
        #: (one zero-perturbation draw through ``generate()``): the loop
        #: stage's and the archived state's verdicts are written into it
        self.sigma0_probe = None

    # ---- the pressure the baseline re-solve and every draw use --------
    def solve_pressure(self, psi_N=None):
        return np.asarray(self.ctx.c.pressure, dtype=float).copy()

    def backend(self, mygs):
        from types import SimpleNamespace
        c = self.ctx.c
        dc = SimpleNamespace(psi_N=c.psi_N, pressure=c.pressure, Ip=c.Ip,
                             kinetics=c.kinetics)
        return tokamaker_backend(mygs, dc, psi_pad=self.psi_pad,
                                 q_psi=self.q_psi, maxits=self.maxits,
                                 edge_pressure=self.ctx.edge,
                                 edge_taper=self.ctx.eng.s.get("edge_taper"))

    def lcfs_pressure(self):
        """The separatrix pressure a written g-file of the CURRENT draw
        carries (its own ``p_sep`` under ``separatrix_pressure="offset"``,
        0 under ``"legacy"``)."""
        cur = self._cur
        if cur is None or cur.get("draw") is None:
            return self.ctx.edge.p_offset(self.ctx.c.pressure)
        return self.ctx.edge.p_offset(cur["draw"]["inputs"].pressure)

    def validate(self, *, pin_jphi, jbs_delta_mode, l_i_uncertainty,
                 recalculate_j_BS, jbs_loop):
        import os
        bad = []
        if bool(pin_jphi) or os.environ.get("PIN_JPHI", "0") == "1":
            bad.append("PIN_JPHI (pin_jphi=True or env PIN_JPHI=1)")
        if os.environ.get("DIFF_BS", "0") == "1":
            bad.append("DIFF_BS=1 (env)")
        if bool(jbs_delta_mode):
            bad.append("jbs_delta_mode=True")
        if float(l_i_uncertainty or 0.0) > 0.0:
            bad.append("l_i_uncertainty > 0 (engine draws match no l_i)")
        if not recalculate_j_BS:
            bad.append("recalculate_j_BS=False")
        if not (jbs_loop and jbs_loop.get("enabled")):
            bad.append("no self-consistent loop settings")
        if bad:
            raise EngineDrawRefused(
                "engine draws (reconstruction_engine='unified') refuse: "
                + "; ".join(bad) + " -- legacy draw modes that would be "
                "silently ignored")

    def rejection_reason(self, exc, stage):
        """The rejection code; with a cap set, a post-homotopy pass whose
        solve stopped at it is ``post_homotopy_maxits`` (its own code, never
        folded into ``jbs_post_homotopy_error``).  A capped LOOP solve keeps
        its code (``anchor_solve_failed`` / ``perturb_failed``) and is
        recorded in :attr:`cap_events` too."""
        if stage == "post_homotopy" and self.hit_cap(exc):
            self.announce_cap("post-homotopy pass", exc,
                              stage="post_homotopy",
                              seconds=self._backend_seconds())
            return "post_homotopy_maxits"
        code = engine_rejection_reason(exc, stage)
        if stage == "perturb" and self.hit_cap(exc):
            self.announce_cap("loop", exc, stage="loop",
                              seconds=self._backend_seconds())
        return code

    def annotate_rejection(self, record, exc):
        """Add what the rejection knows to its record (in place; returns
        it): for non-physical drawn kinetics, ``info`` = the offending
        quantity, its value and psi_N (:class:`DrawKineticsNonPhysical`)."""
        if isinstance(exc, DrawKineticsNonPhysical):
            record["info"] = dict(exc.info)
        return record

    def _backend_seconds(self):
        b = None if self._cur is None else self._cur.get("backend")
        v = getattr(b, "last_solve_s", None)
        return None if v is None else float(v)

    def last_homotopy_solve_seconds(self):
        """Wall time [s] of the last homotopy solve (timed while the cap is
        installed), or ``None``."""
        t = self._timer
        return None if t is None else t.get("last_s")

    # ---- draw_solve_maxits on the homotopy and post-homotopy solves ------
    def hit_cap(self, exc) -> bool:
        """A solve that stopped at ``engine_draw_solve_maxits`` (only when a
        cap is set: with ``None`` nothing is re-classified)."""
        return self.maxits is not None and solve_hit_iteration_cap(exc)

    def announce_cap(self, where, exc, *, stage=None, seconds=None,
                     outcome="rejected"):
        """Print and record a capped solve that did not converge.
        *outcome* is ``"rejected"`` (the caller rejects the draw) or
        ``"rolled_back"`` (a homotopy stage: the draw goes back to the last
        good stage, under the homotopy's own rule)."""
        ev = dict(draw=(None if self._cur is None else self._cur["count"]),
                  where=str(where),
                  stage=str(stage if stage is not None else where),
                  maxits=self.maxits, iterations=self.maxits,
                  seconds=(None if seconds is None else float(seconds)),
                  outcome=str(outcome),
                  error=f"{type(exc).__name__}: {str(exc).strip()[:300]}")
        self.cap_events.append(ev)
        what = ("draw REJECTED" if outcome == "rejected" else
                "rolled back to the last good homotopy stage")
        sec = "" if seconds is None else f" after {float(seconds):.1f} s"
        print(f"  [engine draw] {where}: the GS solve stopped at "
              f"engine_draw_solve_maxits={self.maxits}{sec} without "
              f"converging -> {what}", flush=True)
        return ev

    def announce_rollback_failed(self, exc, after):
        """Print and record a homotopy rollback re-solve that failed for a
        reason other than the cap: the draw is REJECTED
        (``homotopy_rollback_failed``) -- an engine draw never goes on from a
        failed solve, whatever its cause and whether or not a cap is set.
        *after*: what triggered the rollback (``"saturation"`` /
        ``"failed stage"``)."""
        ev = dict(draw=(None if self._cur is None else self._cur["count"]),
                  where="homotopy rollback re-solve",
                  stage="homotopy_rollback", after=str(after),
                  outcome="rejected",
                  error=f"{type(exc).__name__}: {str(exc).strip()[:300]}")
        if getattr(self, "rollback_failures", None) is None:
            self.rollback_failures = []
        self.rollback_failures.append(ev)
        print(f"  [engine draw] homotopy rollback re-solve (after a "
              f"{after}) FAILED ({ev['error']}) -> draw REJECTED "
              f"(homotopy_rollback_failed)", flush=True)
        return ev

    def cap_solver(self, mygs):
        """Set ``draw_solve_maxits`` on the solver for the homotopy stage
        (the post-homotopy passes run through the engine backend, which
        applies it per solve).  A no-op without a cap, or when the solver
        already carries it (``Bouquet.generate``'s DrawSolveGuard); the
        value found is put back by :meth:`uncap_solver`."""
        if self.maxits is None:
            return
        self._time_solves(mygs)
        if self._cap_saved is not None:
            return
        st = getattr(mygs, "settings", None)
        if st is None:
            return
        cur = getattr(st, "maxits", None)
        if cur == self.maxits:
            return
        self._cap_saved = (cur,)
        st.maxits = self.maxits
        mygs.update_settings()

    def _time_solves(self, mygs):
        """Time every ``mygs.solve`` while the homotopy cap is installed (the
        wall time of a capped stage / rollback re-solve is recorded); a pure
        pass-through otherwise, removed by :meth:`uncap_solver`."""
        if getattr(self, "_timer", None) is not None:
            return
        orig = getattr(mygs, "solve", None)
        if not callable(orig):
            return
        own = "solve" in getattr(mygs, "__dict__", {})
        t = dict(orig=orig, own=own, last_s=None)

        def solve(*a, **k):
            import time
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                t["last_s"] = time.perf_counter() - t0
        try:
            mygs.solve = solve
        except (AttributeError, TypeError):
            return
        self._timer = t

    def _untime_solves(self, mygs):
        t = getattr(self, "_timer", None)
        if t is None:
            return
        self._timer = None
        if t["own"]:
            mygs.solve = t["orig"]
        else:
            try:
                del mygs.solve
            except AttributeError:
                pass

    def uncap_solver(self, mygs):
        """Undo :meth:`cap_solver` (idempotent)."""
        self._untime_solves(mygs)
        if self._cap_saved is None:
            return
        (cur,), self._cap_saved = self._cap_saved, None
        mygs.settings.maxits = cur
        mygs.update_settings()

    # ---- the coil regularisation of a draw's loop ------------------------
    def install_coil_reg(self, mygs):
        """Install the coil regularisation a draw's loop solves under and
        return its record (``source``, ``n_terms``, ``installed``).

        The reconstruction's own term list (:attr:`coil_reg`, recorded by
        :func:`bouquet.engine.prepare_engine_baseline`): the same terms,
        targets and weights -- configured measured-coil targets at their
        configured weights, the #VSC term as configured -- so the
        zero-perturbation draw solves the stored request under the
        regularisation the reconstruction solved it under.  Only when that
        list is not known (a solver object ``setup_solver`` did not
        prepare) and ``generate_bouquet`` installed its strong one, the
        historical exploratory list (``_weak_coil_reg``, else every coil
        toward zero at weight 1 and the VSC at 1e-2), recorded as such.
        A failed install is printed and recorded (``installed=False``)."""
        reg = self.coil_reg or {}
        terms = reg.get("terms")
        recd = dict((reg.get("record") or {}))
        if terms is not None:
            out = dict(source="reconstruction ("
                       + str(recd.get("source", "?")) + ")",
                       n_terms=len(terms), installed=False)
        elif getattr(mygs, "_strong_coil_reg", None) is not None:
            terms = getattr(mygs, "_weak_coil_reg", None)
            if terms is None:
                terms = [mygs.coil_reg_term({_n: 1.0}, target=0.0,
                                            weight=1.0)
                         for _n in mygs.coil_sets]
                terms.append(mygs.coil_reg_term({"#VSC": 1.0}, target=0.0,
                                                weight=1e-2))
            out = dict(source=("historical exploratory (the "
                               "reconstruction's regularisation is not on "
                               "record)"),
                       n_terms=len(terms), installed=False)
        else:
            return dict(source="none installed (the reconstruction's "
                               "regularisation is not on record)",
                        n_terms=None, installed=False)
        try:
            mygs.set_coil_reg(reg_terms=list(terms))
            out["installed"] = True
        except Exception as _e:
            out["error"] = f"{type(_e).__name__}: {str(_e)[:200]}"
            print(f"  [engine draw] coil-regularisation install failed "
                  f"({_e}); the loop runs under the regularisation already "
                  "installed", flush=True)
        return out

    # ---- one draw ---------------------------------------------------------
    def draw(self, mygs, rng, scale, count, *, coil_guard=None,
             bnd_diag=None, solve_guard=None):
        """The legacy 7-tuple ``(ne, te, ni, ti, w_ExB, j_phi,
        diagnostics)`` of one engine draw (kinetic-grid profiles)."""
        ctx = self.ctx
        self.uncap_solver(mygs)             # (a previous draw's, if any)
        b = self.backend(mygs)
        clock = _Clock((lambda: int(b.n_solves)) if solve_guard is None
                       else (lambda: int(solve_guard.n_solves)))
        clock.guarded = solve_guard is not None
        self._cur = dict(count=int(count), clock=clock, backend=b)
        # the loop solves under the RECONSTRUCTION's own coil
        # regularisation: a draw is the reconstruction's closure perturbed
        # in its inputs, not re-regularised (generate_bouquet puts its
        # strong one back after the draw)
        self._cur["coil_reg"] = self.install_coil_reg(mygs)
        inputs = sample_draw_inputs(
            ctx, rng, self.unc, b.flux_integral, scale=float(scale),
            p_thresh=self.p_thresh, max_proxy_draws=self.max_proxy_draws)
        _hard = getattr(mygs, "_coil_drift_bounds", None)
        out = run_draw(ctx, b, inputs,
                       label=f"engine draw {int(count)}",
                       coil_guard=(coil_guard if _hard is not None
                                   else None),
                       clock=clock, bnd_diag=bnd_diag)
        clock.start("homotopy")
        self._cur["draw"] = out
        if getattr(self, "sigma0_probe", None) is not None:
            # the zero-perturbation route: the loop stage, measured here
            # before the homotopy moves anything
            self.sigma0_probe["loop"] = zero_perturbation_loop_verdict(ctx,
                                                                       out)
        sp = out["split"]
        jfix = sp["j_NBI"] + sp["j_RF"]
        rec = out["record"]
        rec["coil_reg"] = dict(self._cur["coil_reg"])
        diag = dict(
            j0_scales=[], Ip_scales=[],
            iteration_l_is=[rec["delivered"]["l_i_3"]],
            iteration_Ips=[rec["delivered"]["Ip"]],
            j_inductive=sp["j_phi"] - sp["j_BS"] - jfix, j_BS=sp["j_BS"],
            j_BS_edge=None, proxy_bias_observed=None, r2_ip_scale=None,
            r2_f_ind=None, aux=dict(inputs.aux),
            jbs_loop=_jbs_block(rec["loop"]),
            engine=rec,
            _jbs_ctx=dict(kind="engine", engine=self,
                          isolate_edge_jBS=False))
        nat = inputs.kinetics_native
        return (nat["ne"], nat["te"], nat["ni"], nat["ti"],
                np.zeros_like(ctx.psi), sp["j_phi"], diag)

    def post_homotopy(self, settings, coil_guard=None):
        """``_post_homotopy_jbs``'s engine branch."""
        cur = self._cur
        clock = cur["clock"]
        clock.start("post_homotopy")
        try:
            rec, jbs, jb_tor, jphi = post_homotopy(
                self.ctx, cur["backend"], cur["draw"], settings,
                coil_guard=coil_guard, clock=clock)
        finally:
            clock.start("homotopy")
        return rec, jb_tor, jb_tor, jphi

    def solved_fixed(self):
        """``kappa x <j.B>_fix`` on the geometry the draw's last solved
        request was composed on (the loop's, or the post-homotopy passes')."""
        d = self._cur["draw"]
        dp = d.get("passes_post_homotopy") or d["passes"]
        return np.asarray(dp.last["parts"]["driven"], dtype=float)

    def archived_split(self, diagnostics, j_phi):
        """``(j_BS, j_inductive)`` of the archived draw against its archived
        *j_phi*: the bootstrap and fixed parts of :meth:`post_hoc` (the
        archived equilibrium's), the inductive the residual -- NEVER
        clipped; a negative inductive is recorded, not altered."""
        sp = self._cur["final_split"]
        j_phi = np.asarray(j_phi, dtype=float)
        j_ind = j_phi - sp["j_BS"] - sp["j_NBI"] - sp["j_RF"]
        # the PARALLEL parts of the archived split (schema: the draw's
        # ``jB_parallel/`` subgroup; docs/archive-schema.md): the bootstrap
        # and fixed <j.B> the toroidal parts were converted from, and the
        # field-aligned inductive <j.B> = (j_inductive - P) / kappa, P the
        # pressure-driven p'(<R> - F^2<1/R>/<B^2>) of the archived state --
        # so j_phi = kappa (jB_inductive + jB_BS + jB_NBI + jB_RF) + P
        # exactly, and an IDS export carries no pressure-driven current in
        # any parallel field (io.imas.write_imas_draw)
        kap = np.asarray(sp["kappa"], dtype=float)
        P = np.asarray(sp["j_pressure"], dtype=float)
        self._cur["parallel"] = dict(
            psi_N=np.asarray(self.ctx.psi, dtype=float).copy(),
            jB_inductive=(j_ind - P) / kap,
            jB_BS=np.asarray(sp["jB_BS"], dtype=float).copy(),
            jB_NBI=np.asarray(sp["jB_NBI"], dtype=float).copy(),
            jB_RF=np.asarray(sp["jB_RF"], dtype=float).copy(),
            kappa=kap.copy(), j_pressure=P.copy())
        neg = j_ind < 0.0
        diagnostics["engine"]["archived"]["split"] = dict(
            convention=("j_phi: the archived equilibrium's achieved FSA "
                        "current; j_BS: s_bs (1 + d_bs) scale Redl and "
                        "j_NBI/j_RF: the fixed <j.B>, both times "
                        "F<1/R>/<B^2> of the archived equilibrium; "
                        "j_inductive: the residual (carries the pressure-"
                        "driven term), never clipped"),
            j_NBI=sp["j_NBI"].tolist(), j_RF=sp["j_RF"].tolist(),
            n_negative_inductive=int(np.sum(neg)),
            min_inductive=float(np.min(j_ind)),
            negative_inductive_psi_N=(
                None if not np.any(neg) else
                [float(self.ctx.psi[neg].min()),
                 float(self.ctx.psi[neg].max())]))
        if np.any(neg):
            print(f"  [engine draw split] NOTE: the residual inductive is "
                  f"negative on {int(np.sum(neg))} nodes (min "
                  f"{float(np.min(j_ind)):.3e} A/m^2); recorded, not "
                  "clipped", flush=True)
        return np.asarray(sp["j_BS"], dtype=float).copy(), j_ind

    def mark(self, stage):
        if self._cur is not None:
            self._cur["clock"].start(stage)

    def measured_drifts(self, mygs, baseline_coils, coil_drift):
        """``engine_draw_homotopy=False``: no homotopy stage; the drift of
        the loop's own delivered draw, judged at the spec (``coil_drift``)."""
        from .TokaMaker_interface import _coil_drift_pct
        cur, _ = mygs.get_coil_currents()
        return (_coil_drift_pct(cur, baseline_coils), -1, float(coil_drift),
                float(coil_drift))

    def post_hoc(self, mygs, diagnostics, in_spec, *, constrain_sawteeth,
                 l_i_target=None):
        """The post-hoc filters on the archived state; returns the draw's
        ``in_spec`` (the coil verdict AND the band)."""
        cur = self._cur
        clock = cur["clock"]
        clock.start("filters")
        b = cur["backend"]
        fin = b.measure(final=True)
        v = post_hoc_verdicts(self.ctx, fin, l_i_tolerance=self.l_i_tolerance,
                              constrain_sawteeth=constrain_sawteeth,
                              l_i_reference=l_i_target)
        v["coil_in_spec"] = bool(in_spec)
        v["in_spec"] = bool(in_spec and v["in_band"])
        stats = fin.get("stats") or {}
        from .engine import _full_frame
        rec = diagnostics["engine"]
        rec["archived"] = dict(
            l_i_3=float(fin["li"]), l_i_1=_f(fin.get("li_1")),
            beta_n=_f(_full_frame(fin).get("beta_n")),
            q0=float(fin["q_row"]),
            q0_psi_N=float(self.ctx.ref["q_row_psi_N"]),
            q95=_f(stats.get("q_95")), flux_range=flux_range(fin),
            note=("the archived (post-homotopy) state; 'delivered' is the "
                  "loop's"))
        if fin.get("pressure_frames") is not None:
            rec["archived"]["pressure_frames"] = _frames(fin)
        rec["archived"]["p_scale"] = fin.get("p_scale")
        # the archived split ON the archived equilibrium: the draw's
        # bootstrap model (x* held: s_bs (1 + d_bs) x scale x Redl) and its
        # fixed parts, both converted with THIS state's F<1/R>/<B^2>; the
        # residual against the archived j_phi is :meth:`archived_split`'s
        from .engine import composed_factor, pressure_term
        kap = composed_factor(fin["geom"])
        _w = fin["geom"].get("edge_taper")
        dpl = cur["draw"].get("passes_post_homotopy") or cur["draw"]["passes"]
        fx = self.ctx.c.jB_fix_parts
        _amp = 1.0 + float(dpl.last["amp"].get("d_bs", 0.0))
        _scale = float(cur["draw"]["inputs"].scale)
        jB_BS = (_amp * self.ctx.s_bs * _scale
                 * np.asarray(fin["redl"], dtype=float))
        jB_NBI = np.asarray(fx["nbi"], dtype=float)
        jB_RF = (np.asarray(fx["rf"], dtype=float)
                 + np.asarray(fx.get("other", 0.0), dtype=float))
        cur["final_split"] = dict(
            j_BS=(_amp * self.ctx.s_bs * kap * _scale
                  * np.asarray(fin["redl"], dtype=float)),
            j_NBI=kap * jB_NBI, j_RF=kap * jB_RF,
            jB_BS=jB_BS, jB_NBI=jB_NBI * np.ones_like(kap),
            jB_RF=jB_RF * np.ones_like(kap), kappa=kap,
            j_pressure=pressure_term(fin["geom"])
            * (1.0 if _w is None else np.asarray(_w, dtype=float)))
        rec["archived"]["deltas"] = dict(
            l_i_3=float(fin["li"]) - float(self.ctx.ref["l_i"]),
            l_i_1=(None if (rec["archived"]["l_i_1"] is None
                            or self.ctx.ref["l_i_1"] is None)
                   else rec["archived"]["l_i_1"] - self.ctx.ref["l_i_1"]),
            **flux_range_change(rec["archived"]["flux_range"],
                                self.ctx.ref["flux_range"]))
        rec["post_hoc"] = v
        if getattr(self, "sigma0_probe", None) is not None:
            self.sigma0_probe["archived"] = zero_perturbation_archived_verdict(
                self.ctx, cur["draw"]["jbs_used"], fin)
        _jl = diagnostics.get("jbs_loop") or {}
        if _jl.get("post_homotopy") is not None:
            rec["post_homotopy"] = _jl["post_homotopy"]
        diagnostics[DRAW_BAND_FLAG] = bool(v["in_band"])
        if v["reasons"]:
            print("  [engine draw post-hoc] OUT OF BAND (archived, "
                  "in_spec=False): " + "; ".join(v["reasons"]), flush=True)
        clock.start("archive")
        return v["in_spec"]

    def stored_pressures(self):
        inp = self._cur["draw"]["inputs"]
        return (np.asarray(inp.pressure_thermal, dtype=float).copy(),
                np.asarray(inp.pressure, dtype=float).copy())

    def store_draw(self, header, count, scan_key, diagnostics):
        """Write the draw's ``engine`` block (JSON attribute/dataset
        :data:`bouquet.engine.ENGINE_ATTR`) and the band flag onto its
        archived group."""
        cur = self._cur
        clock = cur["clock"]
        clock.stop()
        rec = diagnostics.get("engine")
        if rec is None:
            return
        rec["homotopy"] = dict(
            enabled=bool(self.homotopy),
            solve_maxits=self.maxits,
            cap_events=[dict(e) for e in self.cap_events
                        if e.get("draw") == cur["count"]],
            homotopy_pass=diagnostics.get("homotopy_pass"),
            homotopy_F_lim=diagnostics.get("homotopy_F_lim"),
            homotopy_VSC_lim=diagnostics.get("homotopy_VSC_lim"),
            max_F_drift_pct=diagnostics.get("max_F_drift_pct"),
            max_VSC_drift_pct=diagnostics.get("max_VSC_drift_pct"),
            note=("generate()'s coil homotopy (homotopy_passes) and the "
                  "post-homotopy bootstrap check" if self.homotopy else
                  "engine_draw_homotopy=False: no homotopy stage; the "
                  "drift of the loop's delivered draw"))
        rec["cost"] = clock.record()
        _write_draw_block(header, count, scan_key, rec,
                          diagnostics.get(DRAW_BAND_FLAG),
                          parallel=cur.get("parallel"))
        clock.start("filters")

    def until_n(self, ok, reasons, diagnostics, header=None, count=None,
                scan_key=None):
        """The until-N verdict of an engine draw: the configured coil +
        boundary verdict AND the post-hoc band (so the ledger counts what
        ``.filter()`` marks ``selected``)."""
        band = diagnostics.get(DRAW_BAND_FLAG)
        reasons = list(reasons)
        if band is not None and not band:
            reasons.append("engine post-hoc band")
        cur = self._cur
        if cur is not None and header is not None:
            cur["clock"].stop()
            rec = diagnostics.get("engine")
            if rec is not None:
                rec["cost"] = cur["clock"].record()
                _write_draw_block(header, count, scan_key, rec, band,
                                  parallel=cur.get("parallel"))
        return bool(ok and (band is None or band)), reasons

    def store_baseline(self, header, scan_key, baseline):
        """The baseline's ``engine`` block (the reconstruction record, with
        the draws' settings) on ``_baseline``."""
        from .engine import store_baseline_engine
        rec = dict(getattr(baseline, "engine", None) or {})
        if not rec:
            return
        rec["draws"] = dict(
            version=ENGINE_DRAW_VERSION, rng_stream=RNG_STREAM,
            loop=self.loop_settings, q0_row=self.ctx.q0_row,
            bootstrap_refresh=bool(self.ctx.bootstrap_refresh),
            solve_maxits=self.maxits,
            homotopy=self.homotopy, l_i_tolerance=self.l_i_tolerance,
            ip_row=("the Ip the delivered composition carries in the exact "
                    "measure on G*"), Ip_target_A=self.ctx.Ip_star,
            reference=self.ctx.ref)
        store_baseline_engine(header, rec, scan_key=scan_key)


def _jbs_block(loop_rec):
    """The draw's ``jbs_loop`` archive block, shaped like a legacy draw's."""
    from .jbs_loop import jsonable
    return jsonable(dict(
        enabled=True, converged=bool(loop_rec.get("converged")),
        kind="engine", n_loops=1,
        n_passes_total=int(loop_rec.get("n_passes", 0)),
        init_source=loop_rec.get("init_source"),
        loops=[dict(label=loop_rec.get("label"),
                    init_source=loop_rec.get("init_source"),
                    n_passes=int(loop_rec.get("n_passes", 0)),
                    converged=bool(loop_rec.get("converged")))],
        final=loop_rec))


def _write_draw_block(header, count, scan_key, record, band, parallel=None):
    import h5py
    from .engine import write_engine_json
    from .schema import write_jB_parallel
    from .utils import _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "a") as hf:
        gp = _group_path(scan_key, count)
        if gp not in hf:
            return
        grp = hf[gp]
        write_engine_json(grp, record)
        if band is not None:
            grp.attrs[DRAW_BAND_FLAG] = bool(band)
        if parallel is not None:
            write_jB_parallel(grp, parallel)


def read_draw_engine(header, count, scan_key=None):
    """The ``engine`` block of one archived draw, or ``None`` (a legacy
    draw)."""
    import h5py
    from .engine import read_engine_json
    from .utils import _group_path, _resolve_h5
    with h5py.File(_resolve_h5(header), "r") as hf:
        gp = _group_path(scan_key, count)
        if gp not in hf:
            return None
        return read_engine_json(hf[gp])


def build_generate_context(bq, env):
    """The :class:`GenerateEngineDraws` of ``Bouquet.generate`` (the live
    reconstruction of THIS session is required)."""
    run = getattr(bq, "_engine_run", None)
    if not run or run.get("baseline") is not bq.baseline:
        raise NotImplementedError(
            "engine draws (Stage 3) need the unified engine's reconstruction "
            "from prepare_baseline() in this session (its live state is what "
            "a draw inherits); call prepare_baseline() first")
    from .engine import engine_draw_maxits
    gc = bq.config.generation
    bl = bq.baseline
    ctx = context_from_run(run, gc, bl)
    unc = dict(env)
    return GenerateEngineDraws(
        ctx, unc=unc, psi_pad=run["psi_pad"], q_psi=run.get("q_psi"),
        maxits=engine_draw_maxits(gc),
        homotopy=bool(getattr(gc, "engine_draw_homotopy", True)),
        l_i_tolerance=float(gc.l_i_tolerance),
        coil_reg=run.get("coil_reg"))


def context_from_run(run, gc, bl):
    """The :class:`EngineDrawContext` of a live reconstruction and its
    Baseline (the kinetic-grid base the sampler perturbs)."""
    native = dict(psi_N=bl.psi_N_kinetic, ne=bl.ne, te=bl.te, ni=bl.ni,
                  ti=bl.ti, z_fast=getattr(bl, "z_fast", None),
                  z2_fast=getattr(bl, "z2_fast", None))
    return EngineDrawContext(
        run["engine"], run["result"], loop=draw_loop_settings(gc),
        native=native, Z_imp=getattr(bl, "Z_imp", None),
        zeff_includes_fast=bool(getattr(bl, "zeff_includes_fast", False)),
        q0_row=bool(getattr(gc, "engine_draw_q0_row", False)),
        bootstrap_refresh=bool(getattr(gc, "engine_draw_bootstrap_refresh",
                                       False)))


# ---------------------------------------------------------------------------
#  verify_sigma0_consistency under the engine
# ---------------------------------------------------------------------------
def _zero_perturbation_tolerances(ctx):
    s = ctx.loop
    return dict(rtol_j=s["rtol_j"], rtol_Ip=s["rtol_Ip"], tol_li=s["tol_li"],
                tol_q0=s["tol_q0"])


def sigma0_gate(value, bound, *, setting, absolute=True):
    """One gated quantity of the zero-perturbation check: ``value``, its
    ``bound`` and the ``setting`` the bound comes from; ``passed`` is
    ``|value| <= bound`` (``value <= bound`` with ``absolute=False``).  A
    missing or non-finite value FAILS (an unmeasured quantity is not
    reproduced); a ``None`` bound is recorded as not gated (``passed``
    ``None``) -- the caller decides whether that is allowed."""
    v = _f(value)
    b = None if bound is None else float(bound)
    if b is None:
        ok = None
    elif v is None:
        ok = False
    else:
        ok = bool((abs(v) if absolute else v) <= b)
    return dict(value=v, bound=b, setting=str(setting), passed=ok)


def _gates_pass(gates):
    """Every gate of a stage passed (a ``None`` verdict -- not gated --
    does not count against it)."""
    return all(g["passed"] is not False for g in gates.values())


def zero_perturbation_loop_verdict(ctx, d):
    """The zero-perturbation verdict of a draw's LOOP stage (the delivered
    loop state of :func:`run_draw` output *d*, before any homotopy): the
    request identity, convergence, ``r_j`` / ``r_I`` of the draw's bootstrap
    against ``lambda_BS*`` (the draw's final parallel weights), ``dl_i``,
    ``dq0`` (row radius and the solver's q0 radius), ``dq95``; ``passed``:
    the request bit-identical, the loop converged and every entry of
    ``gates`` -- ``r_j <= jbs_rtol_j``, ``r_I <= jbs_rtol_Ip``, ``|dl_i| <=
    jbs_tol_li``, ``|dq0| <= jbs_tol_q0`` (at the row radius), each recorded
    with its value and bound -- the unchanged loop tolerances."""
    from .jbs_loop import profile_residuals
    from .engine import conversion_factor
    s = ctx.loop
    rec = d["record"]
    meas = d["passes"].last
    w = meas["geom"]["w_lin"] * conversion_factor(meas["geom"])
    cmp_ = profile_residuals(d["jbs_used"], ctx.lam, w, ctx.psi,
                             float(ctx.c.Ip))
    dl = rec["deltas"]
    conv = bool(rec["loop"]["converged"])
    ident = rec["identity"]
    gates = dict(
        r_j=sigma0_gate(cmp_["r_j"], s["rtol_j"], setting="jbs_rtol_j",
                        absolute=False),
        r_I=sigma0_gate(cmp_["r_I"], s["rtol_Ip"], setting="jbs_rtol_Ip",
                        absolute=False),
        dl_i=sigma0_gate(dl["l_i_3"], s["tol_li"], setting="jbs_tol_li"),
        dq0=sigma0_gate(dl["q0"], s["tol_q0"], setting="jbs_tol_q0"))
    ok = bool(ident["pass1_request_bit_identical"] and conv
              and _gates_pass(gates))
    return dict(
        passed=ok, gates=gates,
        request_bit_identical=ident["pass1_request_bit_identical"],
        request_max_abs_diff=ident["pass1_request_max_abs_diff"],
        loop_converged=conv, n_passes=int(rec["loop"]["n_passes"]),
        r_j=float(cmp_["r_j"]), r_I=float(cmp_["r_I"]),
        dl_i=float(dl["l_i_3"]), dq0=float(dl["q0"]),
        dq0_psi_N=float(ctx.ref["q_row_psi_N"]),
        dq0_stats=(None if (rec["delivered"]["q0_stats"] is None
                            or ctx.ref["q0_stats"] is None)
                   else rec["delivered"]["q0_stats"] - ctx.ref["q0_stats"]),
        dq0_stats_psi_N=float(ctx.ref["q0_stats_psi_N"]),
        dq95=dl["q95"], amplitude=rec["amplitude"]["final"].get("a_ind"),
        solves=rec["solves"])


def zero_perturbation_archived_verdict(ctx, jbs_carried, fin):
    """The zero-perturbation verdict of a draw's ARCHIVED state (after the
    homotopy and the post-homotopy passes): *fin* is the backend's final
    measurement of that state, *jbs_carried* the bootstrap the draw carries
    there.  ``r_j`` / ``r_I`` of that bootstrap against ``lambda_BS*`` on the
    archived geometry, ``dl_i``, ``dq0`` at the row radius, ``dq95``, the
    flux-range change; ``passed``: every entry of ``gates`` (``r_j``,
    ``r_I``, ``|dl_i|``, ``|dq0|`` at the same loop tolerances, each with
    its value and bound).  The coil and boundary deltas of the archived
    state are gated by ``Bouquet.verify_sigma0_consistency`` (they need the
    archive)."""
    from .jbs_loop import profile_residuals
    from .engine import complete_geometry, conversion_factor
    s = ctx.loop
    g = complete_geometry(fin["geom"])
    w = g["w_lin"] * conversion_factor(g)
    cmp_ = profile_residuals(np.asarray(jbs_carried, dtype=float), ctx.lam,
                             w, ctx.psi, float(ctx.c.Ip))
    dli = float(fin["li"]) - float(ctx.ref["l_i"])
    stats = fin.get("stats") or {}
    q95 = _f(stats.get("q_95"))
    out = dict(
        r_j=float(cmp_["r_j"]), r_I=float(cmp_["r_I"]), dl_i=dli,
        dq0=float(fin["q_row"]) - float(ctx.ref["q_row"]),
        dq0_psi_N=float(ctx.ref["q_row_psi_N"]),
        dq95=(None if (q95 is None or ctx.ref.get("q95") is None)
              else float(q95) - float(ctx.ref["q95"])),
        **flux_range_change(flux_range(fin), ctx.ref["flux_range"]))
    out["gates"] = dict(
        r_j=sigma0_gate(out["r_j"], s["rtol_j"], setting="jbs_rtol_j",
                        absolute=False),
        r_I=sigma0_gate(out["r_I"], s["rtol_Ip"], setting="jbs_rtol_Ip",
                        absolute=False),
        dl_i=sigma0_gate(dli, s["tol_li"], setting="jbs_tol_li"),
        dq0=sigma0_gate(out["dq0"], s["tol_q0"], setting="jbs_tol_q0"))
    out["passed"] = _gates_pass(out["gates"])
    return out


def verify_zero_perturbation(ctx, backend, *, label="sigma=0 engine draw"):
    """The engine LOOP at zero perturbation (bootstrap scale 1.0) from the
    backend's current state -- the loop stage of a draw only, NOT the route
    ``generate()`` runs (no warm start, coil regularisation swap, isoflux
    re-point, homotopy or post-homotopy stage; ``Bouquet.
    verify_sigma0_consistency`` runs that route).  The request identity and
    the delivered loop state against the reconstruction
    (:func:`zero_perturbation_loop_verdict`); ``passed``: the request is
    bit-identical, the loop converged, ``r_j <= rtol_j``, ``r_I <=
    rtol_Ip``, ``|dl_i| <= tol_li`` and ``|dq0| <= tol_q0`` -- the
    unchanged loop tolerances."""
    from .jbs_loop import JBSNotConverged, jsonable
    s = ctx.loop
    out = dict(invariant="engine-draw",
               tolerances=_zero_perturbation_tolerances(ctx),
               criterion=("pass-1 request bit-identical to the stored "
                          "request, loop converged, r_j <= rtol_j, r_I <= "
                          "rtol_Ip, |l_i(draw) - l_i*| <= tol_li, "
                          "|q0(draw) - q0*| <= tol_q0 (row radius); q95 "
                          "reported"))
    try:
        d = run_draw(ctx, backend, ctx.zero_inputs(), label=label)
    except JBSNotConverged as e:
        out.update(passed=False, loop_converged=False,
                   error=f"{type(e).__name__}: {str(e)[:300]}",
                   record=jsonable(getattr(e, "record", None)))
        return out
    v = zero_perturbation_loop_verdict(ctx, d)
    out.update(v, record=d["record"])
    ok = v["passed"]
    print(f"[sigma0-check engine loop] {'PASS' if ok else 'FAIL'}: request "
          f"{'bit-identical' if v['request_bit_identical'] else 'DIFFERS'}"
          f"; loop {'converged' if v['loop_converged'] else 'NOT converged'}"
          f" in {v['n_passes']} pass(es); r_j={v['r_j']:.3e} (tol "
          f"{s['rtol_j']:.0e}) r_I={v['r_I']:.3e} (tol {s['rtol_Ip']:.0e})"
          f" |dl_i|={abs(v['dl_i']):.2e} (tol {s['tol_li']:.0e}) dq0="
          f"{v['dq0']:+.2e} (tol {s['tol_q0']:.0e}, psi_N "
          f"{ctx.ref['q_row_psi_N']:g})", flush=True)
    return out
