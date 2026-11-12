r"""Uncertainty priors for non-inductive sources (never applied silently).

Decided defaults (status ``"decided"``): the prior widths of the two source scale factors that
the reconstruction fits, for sources taken from a transport code --

* :data:`FAST_ION_PRESSURE_SCALE_SIGMA` = 0.20: fast-ion pressure scale ``s_pfast ~ N(1, 0.20)``;
* :data:`DRIVEN_CURRENT_SCALE_SIGMA` = 0.25: driven-current scale ``s_cd ~ N(1, 0.25)``;

:mod:`..fit` reads these two constants for its ``s_pfast`` / ``s_cd`` priors, and the
``"transport_code"`` entries below carry them with ``status="decided"``.  Every other width here
is a proposal (status ``"proposed"``; a user decision).  This module collects them with the
evidence each rests on, so a caller can attach them explicitly::

    s = model.compute(...)
    s.uncertainty = proposed_priors("simple_beam")

Every entry is a :class:`.provider.ScalePrior` / :class:`.provider.ShapePrior` with
``status="proposed"`` and a ``basis`` string.  Evidence (this package's development checks,
not published): the built-in model against a transport code on 11 discharges / 1363
slices (slowing-down part, given the code's own births) and on 2 discharges / 102 slices
(full chain with pencil beams); two runs of the transport code on one discharge; and the
a literature survey (quiet-plasma agreement of classical fast-ion models with
neutrons, MSE-inferred fast-ion pressure and beam-driven current within ~10-20 %).

``kind``:

``"simple_beam"``       built-in :class:`.beam.SimpleBeamModel` (bug-fixed tier 0);
``"transport_code"``    a transport-code result read through :mod:`.arrays` / :mod:`.imas`;
``"simple_ec"``         built-in :class:`.ec.SimpleECModel` with zeta0 from a tracer;
``"mhd_active"``        one-sided anomalous-loss branch for fast-ion-transporting MHD
                        (Alfven eigenmodes, fishbones, large islands), literature-based.
"""

from .provider import ScalePrior, ShapePrior

__all__ = ["proposed_priors", "PRIOR_KINDS", "FAST_ION_PRESSURE_SCALE_SIGMA", "DRIVEN_CURRENT_SCALE_SIGMA"]

PRIOR_KINDS = ("simple_beam", "transport_code", "simple_ec", "mhd_active")

#: Decided default prior width (1 sigma, fraction) of the fast-ion pressure scale (module docstring).
FAST_ION_PRESSURE_SCALE_SIGMA = 0.20
#: Decided default prior width (1 sigma, fraction) of the driven-current scale (module docstring).
DRIVEN_CURRENT_SCALE_SIGMA = 0.25

_B30 = "development check against a transport code"


def proposed_priors(kind):
    """List of priors for ``kind`` (module docstring): the two decided scale widths of the
    ``"transport_code"`` kind carry ``status='decided'``, every other entry ``status='proposed'``."""
    P = "proposed"
    D = "decided"
    if kind == "simple_beam":
        return [
            ScalePrior("p_fast", "*", 1.0, 0.20, status=P,
                       basis=f"{_B30}: built-in/transport-code stored energy 1.02 (slowing down on the code's births; "
                             "log-sd 0.08) and 1.04-1.16 per MW (full chain, log-sd 0.07, energy-mix dependent); plus "
                             "0.15 for the transport code's own error (literature survey)"),
            ScalePrior("n_fast", "*", 1.0, 0.20, status=P,
                       basis=f"{_B30}: built-in/transport-code fast-ion content 1.01 (log-sd 0.08); as p_fast"),
            ScalePrior("jdotb", "*", 0.5, 0.35, status=P,
                       basis=f"{_B30}: built-in NBCD (SI normalization, no fast-ion trapping) is 1.9-2.1x the "
                             "transport code per MW (log-sd 0.25); center 0.5 is a CALIBRATION PROPOSAL pending a "
                             "fast-ion trapping model; transport-code run-to-run log-sd 0.28"),
            ShapePrior("p_fast", "*", "shift", 0.04, status=P,
                       basis=f"{_B30}: built-in fast-pressure centroid +0.016 (slowing down only) to +0.04-0.06 "
                             "(full chain) in rho outward of the transport code"),
            ShapePrior("p_fast", "*", "width", 0.25, status=P,
                       basis=f"{_B30}: core (rho<0.2) ratio 0.92 (log-sd 0.13) vs edge (0.6-0.85) 1.15-1.39; deposition "
                             "smoothing width moves the core by x0.7-1.4"),
            ShapePrior("jdotb", "*", "shift", 0.10, status=P,
                       basis=f"{_B30}: built-in NBCD centroid +0.09-0.11 in rho outward of the transport code"),
        ]
    if kind == "transport_code":
        return [
            ScalePrior("p_fast", "*", 1.0, FAST_ION_PRESSURE_SCALE_SIGMA, status=D,
                       basis="decided default; evidence: literature survey (quiet plasmas, 0.15) widened by the "
                             "measured run-to-run spread of two runs of one transport code on one discharge "
                             f"({_B30}: stored energy per MW 0.81, log-sd 0.14, n=13)"),
            ScalePrior("n_fast", "*", 1.0, 0.20, status=P, basis="as p_fast"),
            ScalePrior("jdotb", "*", 1.0, DRIVEN_CURRENT_SCALE_SIGMA, status=D,
                       basis="decided default; evidence: literature survey (0.20) widened by the run-to-run spread "
                             f"({_B30}: NBCD per MW 0.78, log-sd 0.28, n=13, one discharge)"),
            ShapePrior("jdotb", "*", "shift", 0.05, status=P, basis="literature survey (MSE-based NBCD location)"),
        ]
    if kind == "simple_ec":
        return [
            ScalePrior("jdotb", "*", 1.0, 0.25, status=P,
                       basis="zeta0 from a ray/beam tracer: literature survey (linear tracer vs measurement chi2 ~5); "
                             f"{_B30}: tracer zeta varies 0.002-0.33 with deposition radius, so a fixed formula "
                             "without a tracer is not covered by this width"),
            ShapePrior("jdotb", "*", "shift", 0.03, status=P,
                       basis=f"{_B30}: tracer deposition width (rms) 0.02-0.05 in rho; location error recalled 0.02-0.05"),
        ]
    if kind == "mhd_active":
        return [
            ScalePrior("p_fast", "*", 0.75, 0.30, status=P, upper=1.05,
                       basis="literature survey: stored fast-ion energy 0.72 and neutrons 0.6-0.9 with Alfven "
                             "eigenmodes; one-sided"),
            ScalePrior("jdotb", "*", 0.70, 0.30, status=P, upper=1.05,
                       basis="literature survey: NBCD reduced up to ~50 % with MHD; one-sided"),
        ]
    raise ValueError(f"kind must be one of {PRIOR_KINDS}")
