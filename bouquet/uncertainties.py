
import numpy as np

# ====================================================================
#  Uncertainty envelope builder.
#  Multiply against your profile of choice to give σ(ψ_N).
# ====================================================================
def new_uncertainty_profiles(
    psi_N,
    uncertainty,
    falloff_exp=None,
    shelf=0.0,
    edge_val=0.0,
    falloff_loc=0.8,
    tail_alpha=2.5,
):
    r"""Build a 1-D uncertainty envelope over normalised flux.

    Two modes are supported:

    * **Power-law mode** (``falloff_exp`` is not ``None``):
      :math:`u(\hat{\psi}) = U\,(1 - \hat{\psi})^{\mathrm{falloff\_exp}} + S`

      where *S* is the ``shelf`` parameter — a minimum fractional
      uncertainty that prevents the envelope from decaying to zero at
      the edge.

    * **Flat + tail mode** (default, ``falloff_exp`` is ``None``):
      constant value :math:`U` for
      :math:`\hat{\psi} \le \hat{\psi}_{\rm loc}`, then a cosine
      (or cosh) decay to ``edge_val`` at :math:`\hat{\psi}=1`
      controlled by ``tail_alpha``.  The shelf is added after the
      tail decay.

    Parameters
    ----------
    psi_N : ndarray
        1-D array of normalised poloidal flux :math:`\hat{\psi}`.
    uncertainty : float
        Scalar amplitude :math:`U` of the envelope.
    falloff_exp : float or None
        Exponent for the power-law branch (``None`` selects flat + tail).
    shelf : float
        Minimum fractional uncertainty floor added to the envelope.
        Prevents the profile from decaying all the way to zero.
    edge_val : float
        Envelope value at :math:`\hat{\psi}=1` (flat + tail mode).
    falloff_loc : float
        :math:`\hat{\psi}_{\rm loc}` where the tail begins.
    tail_alpha : float
        Sharpness exponent :math:`\alpha` of the cosine/cosh tail.

    Returns
    -------
    ndarray
        1-D array of the same length as ``psi_N``.
    """
    # ---- power-law branch -------------------------------------------
    if falloff_exp is not None:
        return uncertainty * (1.0 - psi_N) ** falloff_exp + shelf

    # ---- flat + cosine/cosh tail branch -----------------------------
    profile_left = uncertainty * np.ones_like(psi_N)
    stitch_height = uncertainty                        # value at falloff_loc

    dist_to_edge = max(1.0 - falloff_loc, 1e-5)       # avoid division by zero

    if edge_val < stitch_height:
        # --- cosine decay ---
        target_cos = np.clip(
            (edge_val / stitch_height) ** (1.0 / tail_alpha), -1.0, 1.0
        )
        omega = np.arccos(target_cos) / dist_to_edge

        base_cos = np.maximum(np.cos(omega * (psi_N - falloff_loc)), 0.0)
        profile_right = stitch_height * (base_cos ** tail_alpha)
    else:
        # --- cosh rise (rare) ---
        target_cosh = (edge_val / stitch_height) ** (1.0 / tail_alpha)
        omega = np.arccosh(target_cosh) / dist_to_edge
        profile_right = stitch_height * (
            np.cosh(omega * (psi_N - falloff_loc)) ** tail_alpha
        )

    profile = np.where(psi_N <= falloff_loc, profile_left, profile_right)

    if edge_val >= 0.0:
        profile = np.maximum(profile, 0.0)

    return profile + shelf



# ====================================================================
#  Radial sigma templates (relative 1-sigma, sigma / |x|).
#
#  The median relative envelopes of IDA-lite deliveries on DIII-D slices,
#  in three regimes, sampled on 14 psi_N nodes and rounded to two
#  significant figures:
#
#    "hmode_flattop"  developed pedestal (peak pedestal, steady flat top)
#    "ramp"           L-H transition and pedestal formation
#    "lmode"          no pedestal (L-mode, including the ramp-up)
#
#  Quantities: n_e, T_e, T_i, Z_eff and the carbon density n_C each from
#  its own delivered envelope; n_i is derived per slice as
#  n_i = n_e (6 - Z_eff) / 5 with independent errors,
#  sigma_ni/n_i = sqrt((sigma_ne/n_e)^2 + (sigma_Z/(6 - Z_eff))^2).
#  Between nodes the template is piecewise linear in psi_N; past psi_N = 1
#  the separatrix value is held.  SIGMA_TEMPLATE_FLOOR is the smallest
#  relative sigma seen on psi_N in [0, 1] in any slice, rounded down to
#  2 s.f.; every node exceeds it, so it binds only for a scaled or
#  smoothed template.
#
#  The H-mode flat-top shapes are the ASSUMED default envelope of a run
#  with no measured sigmas (UncertaintyConfig.*_scalar_sigma = None) and
#  the definition of synthetic_ida_sigma; the tests/data fixtures carry
#  the template of their own phase (tests/data/apply_sigma_templates.py).
# ====================================================================
SIGMA_TEMPLATE_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                       0.9, 0.95, 0.98, 1.0)

SIGMA_TEMPLATES = {
    "hmode_flattop": {
        "ne":   (0.057, 0.034, 0.031, 0.029, 0.028, 0.028, 0.031, 0.037, 0.045, 0.051, 0.058, 0.071, 0.10, 0.15),
        "te":   (0.052, 0.039, 0.046, 0.055, 0.066, 0.078, 0.092, 0.10, 0.12, 0.13, 0.13, 0.14, 0.16, 0.19),
        "ti":   (0.089, 0.062, 0.059, 0.066, 0.076, 0.087, 0.099, 0.11, 0.13, 0.13, 0.14, 0.15, 0.18, 0.31),
        "zeff": (0.16, 0.11, 0.089, 0.086, 0.086, 0.087, 0.089, 0.093, 0.10, 0.11, 0.12, 0.14, 0.21, 0.41),
        "nc":   (0.16, 0.11, 0.10, 0.10, 0.10, 0.097, 0.093, 0.095, 0.10, 0.11, 0.12, 0.15, 0.25, 0.45),
        "ni":   (0.17, 0.11, 0.094, 0.089, 0.085, 0.081, 0.078, 0.079, 0.085, 0.090, 0.097, 0.11, 0.15, 0.22),
    },
    "ramp": {
        "ne":   (0.063, 0.039, 0.034, 0.028, 0.024, 0.022, 0.023, 0.024, 0.030, 0.036, 0.044, 0.059, 0.095, 0.16),
        "te":   (0.055, 0.048, 0.065, 0.078, 0.089, 0.10, 0.11, 0.12, 0.12, 0.12, 0.13, 0.13, 0.14, 0.20),
        "ti":   (0.15, 0.10, 0.097, 0.11, 0.11, 0.12, 0.12, 0.11, 0.11, 0.12, 0.12, 0.13, 0.15, 0.31),
        "zeff": (0.17, 0.12, 0.095, 0.088, 0.089, 0.089, 0.089, 0.090, 0.097, 0.10, 0.11, 0.13, 0.21, 0.45),
        "nc":   (0.23, 0.15, 0.15, 0.16, 0.18, 0.18, 0.17, 0.16, 0.14, 0.15, 0.15, 0.17, 0.24, 0.43),
        "ni":   (0.11, 0.085, 0.072, 0.067, 0.064, 0.062, 0.061, 0.065, 0.069, 0.076, 0.085, 0.10, 0.15, 0.27),
    },
    "lmode": {
        "ne":   (0.045, 0.031, 0.026, 0.027, 0.032, 0.037, 0.043, 0.050, 0.060, 0.063, 0.065, 0.067, 0.068, 0.069),
        "te":   (0.039, 0.035, 0.050, 0.069, 0.083, 0.094, 0.11, 0.12, 0.14, 0.14, 0.15, 0.16, 0.17, 0.17),
        "ti":   (0.074, 0.064, 0.075, 0.090, 0.10, 0.11, 0.11, 0.11, 0.12, 0.12, 0.13, 0.15, 0.18, 0.23),
        "zeff": (0.24, 0.14, 0.10, 0.11, 0.12, 0.14, 0.15, 0.15, 0.17, 0.17, 0.19, 0.21, 0.24, 0.32),
        "nc":   (0.10, 0.087, 0.078, 0.075, 0.097, 0.13, 0.14, 0.13, 0.11, 0.10, 0.12, 0.14, 0.16, 0.21),
        "ni":   (0.11, 0.081, 0.065, 0.067, 0.083, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.29, 0.37),
    },
}

SIGMA_TEMPLATE_FLOOR = {"ne": 0.013, "te": 0.025, "ti": 0.039, "zeff": 0.062,
                        "nc": 0.038, "ni": 0.035}

#: the template of the ASSUMED default envelope (sigma-less runs)
DEFAULT_SIGMA_TEMPLATE = "hmode_flattop"

# aliases -> canonical quantity key
_SIGMA_ALIASES = {
    "ne": "ne", "n_e": "ne", "density": "ne",
    "ni": "ni", "n_i": "ni",
    "te": "te", "t_e": "te",
    "ti": "ti", "t_i": "ti", "t_12c6": "ti",
    "zeff": "zeff", "z_eff": "zeff",
    "nc": "nc", "n_c": "nc", "n_12c6": "nc", "nz": "nc",
}


def sigma_template(psi_N, quantity, regime=DEFAULT_SIGMA_TEMPLATE):
    r"""Relative 1-sigma envelope :math:`\sigma_X / |X|` of a radial template.

    ``max(interp(clip(psi_N, 0, 1), SIGMA_TEMPLATE_GRID, template),
    SIGMA_TEMPLATE_FLOOR[quantity])``: piecewise linear between the 14
    nodes, the psi_N = 1 value held outside the LCFS (and the axis value
    below 0).

    Parameters
    ----------
    psi_N : array_like
        Normalised poloidal flux.  In a toroidal-flux run map the run grid
        to psi_N first (the templates are tabulated in psi_N).
    quantity : str
        ``ne``, ``te``, ``ti``, ``zeff``, ``nc`` (carbon density) or ``ni``
        (the derived main-ion density); aliases ``n_e``, ``t_e``, ``t_i``,
        ``t_12c6``, ``z_eff``, ``n_c``, ``n_12c6``, ``nz``, ``n_i``,
        ``density`` (= ``ne``).
    regime : str
        ``"hmode_flattop"`` (default), ``"ramp"`` or ``"lmode"``.
    """
    key = _SIGMA_ALIASES.get(str(quantity).strip().lower())
    if key is None:
        raise ValueError(
            f"Unknown quantity {quantity!r}; expected one of {sorted(set(_SIGMA_ALIASES))}")
    if regime not in SIGMA_TEMPLATES:
        raise ValueError(
            f"Unknown sigma template {regime!r}; expected one of {sorted(SIGMA_TEMPLATES)}")
    psi = np.clip(np.asarray(psi_N, dtype=float), 0.0, 1.0)
    rel = np.interp(psi, SIGMA_TEMPLATE_GRID, SIGMA_TEMPLATES[regime][key])
    return np.maximum(rel, SIGMA_TEMPLATE_FLOOR[key])


def synthetic_ida_sigma(psi_N, channel):
    r"""Synthetic IDA-like **fractional** 1-sigma uncertainty envelope.

    Returns :math:`\sigma_X / X` for an H-mode flat-top slice: the
    ``"hmode_flattop"`` radial template (:func:`sigma_template`; the median
    relative envelope of IDA-lite deliveries on DIII-D H-mode flat-top
    slices, 14 psi_N nodes, 2 s.f.).  Multiply by the profile to get the
    absolute sigma, e.g. ``sigma_ne = synthetic_ida_sigma(psi_N, 'ne') * ne_SI``.

    Parameters
    ----------
    psi_N : array_like
        Normalised poloidal flux (0 = axis, 1 = LCFS).  Values > 1 (SOL)
        are clamped to 1 so the envelope holds its edge value.
    channel : str
        ``ne`` / ``te`` / ``ti`` / ``ni`` / ``zeff`` / ``nc`` (aliases as for
        :func:`sigma_template`).  ``ni`` is the derived main-ion envelope
        (it was an alias of ``ne`` before 2026-10-14).

    Returns
    -------
    ndarray
        Fractional sigma, same shape as ``psi_N``, >= the template floor.

    Notes
    -----
    Values at psi_N 0 / 0.5 / 0.9 / 1.0: n_e 5.7 / 2.8 / 5.8 / 15 %,
    T_e 5.2 / 7.8 / 13 / 19 %, T_i 8.9 / 8.7 / 14 / 31 %, Z_eff 16 / 8.7 /
    12 / 41 %, n_C 16 / 9.7 / 12 / 45 %, n_i 17 / 8.1 / 9.7 / 22 %.

    Until 2026-10-14 this was a sine-basis fit (const + sin(k pi psi_N)
    harmonics + sin(pi psi_N / 2)^p edge terms) to the fractional-sigma
    shape of ONE reference case, for n_e (n_i an alias), T_e and T_i only.
    Its shapes were qualitatively right but its levels were 1.2-1.5x too low
    in the core and at mid radius and 1.6-3x too low at the separatrix
    against the flat-top median, and it had no on-axis rise; the values at
    psi_N 0 / 0.5 / 0.9 / 1.0 were n_e 2.4 / 2.4 / 4.0 / 6.0 %, T_e 3.2 /
    5.5 / 8.6 / 12 %, T_i 5.7 / 7.1 / 7.4 / 9.6 %.  docs/CHANGES_SUMMARY.md
    records the change.  The jphi_total envelope is handled separately.
    """
    return np.clip(sigma_template(psi_N, channel, "hmode_flattop"), 0.0, None)
