r"""Bootstrap current <j_BS . B>: the solver's own evaluation, ``TokaMaker.eval_bootstrap``.

The engine no longer carries a transcription of the formula.  It calls the solver's bootstrap evaluation (the
Fortran routine that TokaMaker's in-solve bootstrap uses, exposed as ``TokaMaker.eval_bootstrap``) on the current
equilibrium and on the engine's physics grid, with the options that reproduce the engine's definitions:

* coefficients: Redl et al., Phys. Plasmas 28, 022502 (2021), L31 (eqs. 10-11), L32 (eqs. 12-16), alpha
  (eqs. 20-21), L34 = L31, in Sauter's (1999) "jboot1" form (``bs_model=1``, the solver default)

  .. code-block:: text

      <j_BS . B> = -I(psi) p_e [ L31 (1/p_e) dp/dpsi + L32 (1/T_e) dT_e/dpsi
                                 + L34 alpha (1 - R_pe)/R_pe (1/T_i) dT_i/dpsi ]         (S99 eq. 5)

  with ``I(psi) = F``, ``R_pe = p_e / p``, ``p = p_e + p_i`` (thermal);
* ``p_i`` is the TOTAL thermal ion pressure: main ions plus the impurity at ``T_i``, ``(n_i + n_imp) T_i``, the
  thermal pressure the equilibrium carries (:mod:`..kinetics`, (K5)); the engine passes its own impurity density
  (``nimp``, which the solver counts in the ion pressure, ``imp_pressure`` on).  Without an impurity density the
  ion pressure is ``n_i T_i`` (``imp_pressure`` off).  The main-ion density alone enters the ion collisionality
  and ``ln Lambda_ii``;
* Coulomb logarithms ``ln_lambda_model="sauter"`` (default; solver ``ln_lambda_model=2``): Sauter, Angioni and
  Lin-Liu, Phys. Plasmas 6, 2834 (1999), eqs. (18d) and (18e), ``ln Lambda_e = 31.3 - ln(sqrt(n_e) / T_e)``,
  ``ln Lambda_ii = 30 - ln(Z^3 sqrt(n_i) / T_i^1.5)`` (n in m^-3, T in eV), ``Z = max(n_e/n_i, 1)``, no floor --
  the electron logarithm of :mod:`.conductivity`, so sigma_neo and the bootstrap see the same nu_e*.
  ``"nrl"`` (solver ``ln_lambda_model=1``, the earlier default kept for comparison): the NRL electron form and
  the same ion form, both floored at 10;
* ``nu_e* = 6.921e-18 |q| R n_e Z_eff ln Lambda_e / (T_e^2 eps^1.5)`` (S99 eq. 18b) and
  ``nu_i* = 4.90e-18 |q| R n_i Z^4 ln Lambda_ii / (T_i^2 eps^1.5)`` with ``Z^4 = (n_e / n_i) Z_eff``
  (``ion_collisionality="spike"``, the Koh multi-species form; solver ``ion_charge_model=1``, its default).
  The published forms do not fix one choice for a plasma with impurities; the spread to the main-ion charge
  (2.5-9 % of the local bootstrap current on a Z_eff = 2 deuterium-carbon profile) is carried as a model-form
  uncertainty inside the bootstrap multiplier's prior (``s_bs``, N(1, 0.5), :mod:`..fit`), not chosen from the
  data.  The earlier engine options ``"zeff"`` and ``"unity"`` have no solver counterpart and are refused;
* geometry from the solver's flux-surface contour pass (``sauter_fc``): f_t, |q|, ``R = (R_max + R_min)/2``,
  ``eps = (R_max - R_min)/(R_max + R_min)`` and ``F``; gradients by the second-order three-point stencil on the
  given samples (one-sided second order at the ends), as ``numpy.gradient(edge_order=2)``.  ``eval_bootstrap``
  returns f_t and eps of that pass (``return_geometry``), so the engine makes no second ``sauter_fc`` call for the
  conductivity on the same points.

Frame: the engine's "positive frame" of :mod:`.diffusion` (``|F|``, psi increasing outward, a co-current bootstrap
positive).  The solver's result is odd in ``F`` and in the sign of ``psi_axis - psi_boundary`` of its internal
flux (``gs.psi_bounds``); :func:`solver_jdotb` multiplies by both signs, so the result is the positive-frame value.

Equivalence with the engine's earlier Python transcription (removed; it is kept as the reference of the solver
test ``tests/kinrecon/test_bootstrap_solver.py``) is to round-off on the public fixture and on development
equilibria; the conversion to toroidal current stays in :mod:`..current_model` (identical in both).
"""

import numpy as np

EC = 1.602176634e-19

__all__ = ["solver_jdotb", "ION_COLLISIONALITY", "LN_LAMBDA_MODELS", "EC"]

#: Ion-collisionality charge model -> solver ``ion_charge_model`` (module docstring).
ION_COLLISIONALITY = {"spike": 1}
#: Coulomb-logarithm model -> solver ``ln_lambda_model``; "sauter" is the default.
LN_LAMBDA_MODELS = {"sauter": 2, "nrl": 1}
#: Solver coefficient set: Redl 2021 (L34 = L31).
BS_MODEL = 1


def solver_options(ion_collisionality="spike", ln_lambda_model="sauter"):
    """The solver's integer options for the engine's named ones; raises ``ValueError`` for a name the solver does
    not offer."""
    if ion_collisionality not in ION_COLLISIONALITY:
        raise ValueError(f"ion_collisionality must be one of {tuple(ION_COLLISIONALITY)} (the solver's "
                         "eval_bootstrap offers no counterpart of the earlier 'zeff' and 'unity' forms)")
    if ln_lambda_model not in LN_LAMBDA_MODELS:
        raise ValueError(f"ln_lambda_model must be one of {tuple(LN_LAMBDA_MODELS)}")
    return dict(bs_model=BS_MODEL, ln_lambda_model=LN_LAMBDA_MODELS[ln_lambda_model],
                ion_charge_model=ION_COLLISIONALITY[ion_collisionality])


def solver_jdotb(gs, psi_n, te, ti, ne, ni, zeff, f_signed, nimp=None,
                 ion_collisionality="spike", ln_lambda_model="sauter", return_geometry=False):
    r"""<j_BS . B> [A T / m^2] on the psi_N samples, positive frame (module docstring).

    Parameters
    ----------
    gs : TokaMaker
        The solver object holding the current (solved) equilibrium; it is not changed.
    psi_n : array (n,)
        Normalized poloidal flux of the samples (strictly increasing, inside (0, 1)).
    te, ti : array (n,)  [eV];  ne, ni : array (n,)  [m^-3];  zeff : array (n,)
        Thermal kinetic profiles on the samples (``ni``: main ions).
    f_signed : array (n,)
        ``F`` of the equilibrium on the samples, with its sign (only the sign is used).
    nimp : array (n,), optional
        Impurity density [m^-3] at ``T_i``, counted in the ion pressure; None: ion pressure ``n_i T_i``.
    return_geometry : bool
        Also return the geometry of the evaluation's own flux-surface pass (the solver's ``sauter_fc`` on the same
        points, which ``eval_bootstrap`` makes anyway): ``{"f_t": 1 - f_c, "eps": (R_max - R_min)/(R_max + R_min)}``,
        bit for bit what a separate ``gs.sauter_fc(psi=psi_n, return_eps=True)`` returns, so that call is not needed.

    Returns ``(jdotb, settings)``, or ``(jdotb, settings, geometry)`` with ``return_geometry`` (``geometry`` None
    for a solver build whose ``eval_bootstrap`` does not return it).
    """
    ops = solver_options(ion_collisionality, ln_lambda_model)
    x = np.asarray(psi_n, dtype=float)
    te, ti, ne, ni, zeff = (np.asarray(v, dtype=float) for v in (te, ti, ne, ni, zeff))
    fs = np.sign(np.asarray(f_signed, dtype=float))
    if not np.all(fs == fs[0]) or fs[0] == 0.0:
        raise ValueError("F must be nonzero and of one sign on the samples")
    bounds = np.asarray(gs.psi_bounds, dtype=float)       # (LCFS, axis) of the solver's internal flux
    sr = float(np.sign(bounds[1] - bounds[0]))
    if sr == 0.0:
        raise ValueError("psi_axis == psi_boundary")
    if nimp is None:
        out = gs.eval_bootstrap(x, te / 1.0e3, ti / 1.0e3, ne, ni, zeff, imp_pressure=False, **ops)
    else:
        nz = np.asarray(nimp, dtype=float)
        if nz.shape != ni.shape or not np.all(np.isfinite(nz)) or np.any(nz < 0):
            raise ValueError("nimp must be finite, >= 0 and on psi_n")
        out = gs.eval_bootstrap(x, te / 1.0e3, ti / 1.0e3, ne, ni, zeff, nimp=nz, imp_pressure=True, **ops)
    jb = np.asarray(out["jdotB_bs_raw"], dtype=float) * (fs[0] * sr)
    settings = dict(model="redl2021", form="jboot1 (Sauter 1999 eq. 5), L34 = L31",
                    evaluation="solver: TokaMaker.eval_bootstrap", solver_options=dict(ops, imp_pressure=nimp is not None),
                    ln_lambda_model=ln_lambda_model,
                    ln_lambda_e=("Sauter 1999 eq. 18d, no floor" if ln_lambda_model == "sauter" else "NRL, floor 10"),
                    ln_lambda_ii=("Sauter 1999 eq. 18e with Z = max(n_e/n_i, 1), no floor"
                                  if ln_lambda_model == "sauter" else "Zavg, floor 10"),
                    ion_collisionality=ion_collisionality, gradient="second-order three-point stencil on the samples",
                    ion_pressure=("(n_i + n_imp) T_i (main ions and impurity)" if nimp is not None
                                  else "n_i T_i (main ions only; no impurity density given)"),
                    frame="positive (|F|, psi increasing outward)")
    if return_geometry:
        geometry = None   # a solver build whose eval_bootstrap returns no geometry: the caller traces it itself
        if "eps" in out:
            geometry = {"f_t": np.asarray(out["f_t"], dtype=float), "eps": np.asarray(out["eps"], dtype=float)}
        return jb, settings, geometry
    return jb, settings
