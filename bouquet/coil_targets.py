"""Coil-current regularisation targets from measured currents.

bouquet's default regularisation pulls every coil toward ZERO at unit weight,
leaving the isoflux boundary as the only opposing constraint. The solve
therefore lands on one of many coil sets consistent with the boundary rather
than the one the machine actually ran: on a DIII-D H-mode slice the resulting
baseline sits a median 15.4 % (max 178 %) from the measured currents, and the
E-coil circuits trade current among themselves nearly freely.

This module builds a ``SolverConfig.coil_reg`` spec that pins each coil to its
measured current instead, mirroring the shipped OFT DIII-D example.

How hard to pull -- the weights
-------------------------------
A single uniform weight is the wrong shape: the measured precision is absolute
and differs by an order of magnitude between coil families, so one weight is a
tight pull on a well-measured coil and a loose one on a poorly-measured coil.
The validated form is INVERSE-VARIANCE::

    w_i = W0 * (sigma_ref / sigma_i)**2         sigma in ampere-turns
    sigma_ref = median sigma over the reference family (default "F")

so ``W0`` is the weight of a typical reference-family coil and the shape of the
pull follows the measurement.  Measured on eight DIII-D slices, in units of the
adopted per-coil tolerance sigma (|I_baseline - I_measured| / sigma):

=========================  ==============  ===========  ===========
slice                      unregularised   W0 = 10      W0 = 100
=========================  ==============  ===========  ===========
L-mode / low beta_N        6 - 23 median   1.8 - 7.6    1.4 - 2.7
H-mode, beta_N ~ 1.4       31 median       16           2.1
H-mode, beta_N >= 2.7      17 - 23 median  17 - 19      5.5 - 8.9
=========================  ==============  ===========  ===========

so: ``W0 = 100`` for beta_N <~ 0.5, ``W0 = 10`` above, and NEVER ``W0 <~ 5`` --
a weak pull resolves the inner/mid coil null space the wrong way and is not
distinguishable in strength from the historical pull toward zero.  That last
rule is enforced here rather than left in prose (see :data:`MIN_SAFE_W0`).

At high beta_N the coils and the plasma are not the same equilibrium and even
W0 = 100 leaves the baseline several sigma out; the weights cannot fix that.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np

__all__ = ["TURNFC_D3D", "DEFAULT_W0", "MIN_SAFE_W0", "coil_reg_from_measured",
           "measured_from_pf_active", "sigma_from_pf_active",
           "inverse_variance_weights"]

#: DIII-D F-coil turns::
#:
#:     TURNFC= 5*58.0, 2*55.0, 58.0, 55.0
#:            5*58.0, 2*55.0, 58.0, 55.0
#:
#: TokaMaker coil currents are ampere-turns, so a measured circuit current is
#: converted with these. E-coils are deliberately absent: the shipped D3D mesh
#: carries their turns itself (``coil_dict`` nturns sums to 61.0 for ECOILA and
#: ECOILB), which is the ``/61.0`` divisor in the OFT DIII-D example.
# Turn counts now live in the device registry (bouquet.devices); this name is kept
# as an alias so existing callers and configs keep working.
from .devices import get_device as _get_device
TURNFC_D3D = dict(_get_device("DIII-D").turns)

#: Reference weight for :func:`inverse_variance_weights`, i.e. the weight given
#: to a coil of typical measured precision.  100 is the value at which the
#: baseline lands within ~2 sigma of the machine on every L-mode and low-beta_N
#: slice checked; drop to 10 above beta_N ~ 0.5 where the boundary cost grows.
DEFAULT_W0 = 100.0

#: Below this the pull is too weak to resolve the inner/mid coil null space and
#: resolves it the WRONG way -- a baseline worse than either the unregularised
#: one or the recommended one, with no symptom.  ``W0 = 1`` in particular is
#: indistinguishable in strength from the historical pull toward zero.
MIN_SAFE_W0 = 5.0

#: The nearest ``pf_active`` sample may sit this far [s] from the requested time
#: before the readers warn that they are extrapolating.
PF_ACTIVE_TIME_TOL_S = 0.05


def _resolve_turns(turns, device):
    if turns is not None:
        return dict(turns)
    if device is None:
        raise ValueError(
            "coil_targets: give either `turns` or `device`. There is no safe default: "
            "a measured circuit current converts to solver ampere-turns with the "
            "DEVICE's turn counts, and silently using another machine's table is a "
            "1x / 58x / 61x unit error in a regularisation TARGET.")
    from .devices import DeviceSpec, resolve_device
    spec = device if isinstance(device, DeviceSpec) else resolve_device(device)
    if spec is None or not spec.turns:
        raise ValueError(f"coil_targets: device {device!r} carries no turns table")
    return dict(spec.turns)


def _pf_active_index(t, time_s, name, time_tol_s):
    j = int(np.argmin(np.abs(t - time_s)))
    dt = float(abs(t[j] - time_s))
    if time_tol_s is not None and dt > float(time_tol_s):
        warnings.warn(
            f"pf_active: nearest sample for {name!r} is {dt:.4g} s from the requested "
            f"t = {time_s:.4g} s (tolerance {float(time_tol_s):.4g} s). The requested "
            "time is probably outside the coil time base; the endpoint is being used.",
            stacklevel=3)
    return j


def measured_from_pf_active(dd_path: str, time_s: float,
                            time_tol_s: Optional[float] = PF_ACTIVE_TIME_TOL_S
                            ) -> Dict[str, float]:
    """``{name: circuit_current_A}`` from an IMAS ``pf_active`` at *time_s* [s].

    The nearest sample is taken; a request further than *time_tol_s* from any
    sample warns instead of silently returning an endpoint.
    """
    import json

    with open(dd_path) as fh:
        dd = json.load(fh)
    time_s = float(time_s)
    out = {}
    for c in dd.get("pf_active", {}).get("coil", []):
        name = c.get("name") or c.get("identifier")
        cur = c.get("current") or {}
        if not name or "data" not in cur or "time" not in cur:
            continue
        t = np.asarray(cur["time"], dtype=float)
        d = np.asarray(cur["data"], dtype=float)
        out[name] = float(d[_pf_active_index(t, time_s, name, time_tol_s)])
    return out


def sigma_from_pf_active(dd_path: str, time_s: float, turns=None, device=None,
                         time_tol_s: Optional[float] = PF_ACTIVE_TIME_TOL_S
                         ) -> Dict[str, float]:
    """``{name: sigma_A_turns}`` -- the measured 1-sigma, in solver units.

    ``data_error_upper`` is read as the 1-sigma magnitude (the same convention
    as :func:`filtering.measured_coil_currents`) and converted with the same
    turns table as the targets, so weights and targets share a unit system.
    Coils with a missing or non-positive sigma are omitted.
    """
    import json

    turns = _resolve_turns(turns, device)
    with open(dd_path) as fh:
        dd = json.load(fh)
    time_s = float(time_s)
    out = {}
    for c in dd.get("pf_active", {}).get("coil", []):
        name = c.get("name") or c.get("identifier")
        cur = c.get("current") or {}
        e = cur.get("data_error_upper")
        if not name or e is None or "time" not in cur:
            continue
        t = np.asarray(cur["time"], dtype=float)
        e = np.abs(np.asarray(e, dtype=float))
        j = _pf_active_index(t, time_s, name, time_tol_s)
        if np.isfinite(e[j]) and e[j] > 0:
            out[name] = float(e[j]) * float(turns.get(name, 1.0))
    return out


def inverse_variance_weights(sigma: Dict[str, float], W0: float = DEFAULT_W0,
                             reference_family: str = "F",
                             allow_weak: bool = False) -> Dict[str, float]:
    """``{name: W0 * (sigma_ref / sigma_i)**2}`` -- the validated weighting.

    *sigma* is per coil in the SAME units as the targets (ampere-turns; see
    :func:`sigma_from_pf_active`).  ``sigma_ref`` is the median sigma over the
    coils whose name starts with *reference_family*, so ``W0`` is the weight of
    a typical reference-family coil and a coil measured twice as well is pulled
    four times as hard.  If no coil matches the family, the median over all
    coils is used.

    Raises :class:`ValueError` for ``W0 <=`` :data:`MIN_SAFE_W0` -- see the
    module docstring for why that regime is worse than not regularising at all.
    Pass ``allow_weak=True`` to downgrade the refusal to a warning (for a
    deliberate weak-pull study).
    """
    W0 = float(W0)
    if not np.isfinite(W0) or W0 <= 0:
        raise ValueError("inverse_variance_weights: W0 must be finite and > 0")
    if W0 <= MIN_SAFE_W0:
        msg = (f"coil-target weight W0 = {W0:g} is at or below MIN_SAFE_W0 = "
               f"{MIN_SAFE_W0:g}. A pull this weak resolves the inner/mid coil null "
               "space the WRONG way: the baseline ends up worse than either the "
               "unregularised one or the recommended one, and nothing in the result "
               "looks wrong. Use W0 = 100 (beta_N <~ 0.5) or W0 = 10 above; pass "
               "allow_weak=True only for a deliberate weak-pull study.")
        if not allow_weak:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    good = {str(k): float(v) for k, v in sigma.items()
            if np.isfinite(v) and float(v) > 0}
    if not good:
        raise ValueError("inverse_variance_weights: no coil has a positive, finite sigma")
    fam = [v for k, v in good.items() if k.startswith(reference_family)]
    sigma_ref = float(np.median(fam if fam else list(good.values())))
    return {k: W0 * (sigma_ref / v) ** 2 for k, v in good.items()}


def coil_reg_from_measured(measured: Dict[str, float],
                           weights: Optional[Dict[str, float]] = None,
                           sigma: Optional[Dict[str, float]] = None,
                           W0: float = DEFAULT_W0,
                           turns: Optional[Dict[str, float]] = None,
                           device=None,
                           reference_family: str = "F",
                           allow_weak: bool = False,
                           default_weight: Optional[float] = None) -> List[dict]:
    """``SolverConfig.coil_reg`` spec pinning each coil to its measured current.

    Parameters
    ----------
    measured : {name: current}
        Measured circuit currents [A].
    weights : {name: weight}, optional
        Explicit per-coil weights.  Coils absent from it fall back to the
        inverse-variance weight when *sigma* is given, else to *W0*.
    sigma : {name: sigma_A_turns}, optional
        Measured 1-sigma per coil in solver units (see
        :func:`sigma_from_pf_active`).  Given, the weights are
        ``W0 * (sigma_ref/sigma_i)**2`` -- the form the recommendation was
        measured with.  Omitted, every coil gets the flat weight *W0*.
    W0 : float
        Reference weight; the weight of a typical reference-family coil.
        Defaults to :data:`DEFAULT_W0` = 100.  Values at or below
        :data:`MIN_SAFE_W0` are refused (see :func:`inverse_variance_weights`);
        in particular the flat ``1.0`` this helper used to default to is in
        that regime and is no longer reachable by accident.
    turns, device :
        Circuit-amps -> solver-units conversion.  Give one: ``turns`` directly,
        or a ``device`` name / :class:`DeviceSpec` whose registry turns are
        used.  There is no default -- another machine's turns table is a silent
        1x / 58x / 61x error in a regularisation target.  A coil with no entry
        converts at 1.0 (the mesh is assumed to carry its own turns, as the
        shipped D3D mesh does for the E-coils).  That assumption is about a
        MESH, not a device, so each term records the factor it was built with
        under ``"turns"`` and :meth:`Bouquet._apply_coil_reg` checks it against
        the mesh actually loaded, dropping the term if it cannot be confirmed.
    default_weight : float, optional
        DEPRECATED escape hatch for a flat weight; it bypasses the W0 floor and
        warns.  Prefer ``W0``.
    """
    turns = _resolve_turns(turns, device)
    weights = dict(weights or {})
    if default_weight is not None:
        warnings.warn(
            "coil_reg_from_measured(default_weight=...) is deprecated: it is a flat "
            "weight with no reference to the measured precision, and its old default "
            "of 1.0 is in the regime the weighting study says never to use. Pass W0 "
            "(and sigma, for inverse-variance weights) instead.", stacklevel=2)
        fallback = {k: float(default_weight) for k in measured}
    elif sigma is not None:
        fallback = inverse_variance_weights(
            sigma, W0=W0, reference_family=reference_family, allow_weak=allow_weak)
    else:
        if float(W0) <= MIN_SAFE_W0 and not allow_weak:
            raise ValueError(
                f"coil_reg_from_measured: W0 = {float(W0):g} <= MIN_SAFE_W0 = "
                f"{MIN_SAFE_W0:g}; see bouquet.coil_targets for why a weak pull is "
                "worse than none. Pass allow_weak=True for a deliberate study.")
        fallback = {k: float(W0) for k in measured}
    spec = []
    unweighted = []
    for name, i_circuit in measured.items():
        if not np.isfinite(i_circuit):
            continue
        w = weights.get(name, fallback.get(name))
        if w is None:
            # No usable sigma for this coil (sigma_from_pf_active omits any coil
            # whose data_error_upper is missing or non-positive) and no explicit
            # weight.  Dropping it here is NOT the safe option: every coil that no
            # term names is given target=0 at weight 1 by _apply_coil_reg, i.e.
            # precisely the pull toward zero these targets exist to remove, at a
            # strength the weighting study calls indistinguishable from it.  What
            # is missing is the coil's PRECISION, not its target -- so it is
            # pinned at its measured current with the flat reference weight W0,
            # the same rule every coil gets when no sigma is supplied at all.
            w = float(W0)
            unweighted.append(name)
        spec.append({
            "coils": {name: 1.0},
            "target": float(i_circuit) * float(turns.get(name, 1.0)),
            "weight": float(w),
            # the turns convention this target assumes, for _apply_coil_reg to
            # check against the mesh that is actually loaded (V50-2)
            "turns": float(turns.get(name, 1.0)),
        })
    if unweighted:
        warnings.warn(
            "coil_targets: %d coil(s) carry a measured current but no usable sigma "
            "and no explicit weight: %s. They are pinned at their MEASURED current "
            "with the flat reference weight W0 = %g rather than an inverse-variance "
            "one, because the target is known and only its precision is not. They "
            "are NOT dropped: a dropped coil reverts to the target=0, weight=1 "
            "default, which is the pull toward zero these targets exist to remove. "
            "Give them a sigma, or an explicit `weights` entry, to weight them by "
            "their own precision."
            % (len(unweighted), ", ".join(sorted(unweighted)), float(W0)),
            stacklevel=2)
    return spec
