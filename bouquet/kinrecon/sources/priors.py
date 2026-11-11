r"""PROPOSED default uncertainty priors for non-inductive sources (status ``"proposed"``; never applied silently).

The default prior widths are a user decision (project decision D-B3).  This module only
collects the proposals with the evidence each rests on, so a caller can attach them
explicitly::

    s = model.compute(...)
    s.uncertainty = proposed_priors("simple_beam")

Every entry is a :class:`.provider.ScalePrior` / :class:`.provider.ShapePrior` with
``status="proposed"`` and a ``basis`` string.  Evidence (kinetic-reconstruction record,
report 30): the built-in model against a transport code (ONETWO) on 11 discharges / 1363
slices (slowing-down part, given the code's own births) and on 2 discharges / 102 slices
(full chain with pencil beams); two runs of the transport code on one discharge; and the
literature survey of report 28 (quiet-plasma agreement of classical fast-ion models with
neutrons, MSE-inferred fast-ion pressure and beam-driven current within ~10-20 %).

``kind``:

``"simple_beam"``       built-in :class:`.beam.SimpleBeamModel` (bug-fixed tier 0);
``"transport_code"``    a transport-code result read through :mod:`.arrays` / :mod:`.imas`;
``"simple_ec"``         built-in :class:`.ec.SimpleECModel` with zeta0 from a tracer;
``"mhd_active"``        one-sided anomalous-loss branch for fast-ion-transporting MHD
                        (Alfven eigenmodes, fishbones, large islands), literature-based.
"""

from .provider import ScalePrior, ShapePrior

__all__ = ["proposed_priors", "PRIOR_KINDS"]

PRIOR_KINDS = ("simple_beam", "transport_code", "simple_ec", "mhd_active")

_B30 = "record report 30"


def proposed_priors(kind):
    """List of proposed priors for ``kind`` (module docstring); all ``status='proposed'``."""
    P = "proposed"
    if kind == "simple_beam":
        return [
            ScalePrior("p_fast", "*", 1.0, 0.20, status=P,
                       basis=f"{_B30}: built-in/transport-code stored energy 1.02 (slowing down on the code's births; "
                             "log-sd 0.08) and 1.04-1.16 per MW (full chain, log-sd 0.07, energy-mix dependent); plus "
                             "0.15 for the transport code's own error (report 28 literature)"),
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
            ScalePrior("p_fast", "*", 1.0, 0.20, status=P,
                       basis="report 28 literature (quiet plasmas, 0.15) widened by the measured run-to-run spread of "
                             f"two runs of one transport code on one discharge ({_B30}: stored energy per MW 0.81, "
                             "log-sd 0.14, n=13)"),
            ScalePrior("n_fast", "*", 1.0, 0.20, status=P, basis="as p_fast"),
            ScalePrior("jdotb", "*", 1.0, 0.25, status=P,
                       basis=f"report 28 literature (0.20) widened by the run-to-run spread ({_B30}: NBCD per MW 0.78, "
                             "log-sd 0.28, n=13, one discharge)"),
            ShapePrior("jdotb", "*", "shift", 0.05, status=P, basis="report 28 literature (MSE-based NBCD location)"),
        ]
    if kind == "simple_ec":
        return [
            ScalePrior("jdotb", "*", 1.0, 0.25, status=P,
                       basis="zeta0 from a ray/beam tracer: report 28 literature (linear tracer vs measurement chi2 ~5); "
                             f"{_B30}: tracer zeta varies 0.002-0.33 with deposition radius, so a fixed formula "
                             "without a tracer is not covered by this width"),
            ShapePrior("jdotb", "*", "shift", 0.03, status=P,
                       basis=f"{_B30}: tracer deposition width (rms) 0.02-0.05 in rho; location error recalled 0.02-0.05"),
        ]
    if kind == "mhd_active":
        return [
            ScalePrior("p_fast", "*", 0.75, 0.30, status=P, upper=1.05,
                       basis="report 28 literature: stored fast-ion energy 0.72 and neutrons 0.6-0.9 with Alfven "
                             "eigenmodes; one-sided"),
            ScalePrior("jdotb", "*", 0.70, 0.30, status=P, upper=1.05,
                       basis="report 28 literature: NBCD reduced up to ~50 % with MHD; one-sided"),
        ]
    raise ValueError(f"kind must be one of {PRIOR_KINDS}")
