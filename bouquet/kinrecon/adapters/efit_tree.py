"""Measurement sets from an EFIT run's stored measurement arrays (one time).

EFIT writes, for every time it fits, the measurements it used, its own fitted
(calculated) values of the same quantities, its fit weights and its per-channel
chi^2 -- in its measurement file and in the archive trees built from the same
output.  This adapter turns the arrays of ONE time into a
:class:`~bouquet.kinrecon.interfaces.MeasurementSet`.  Channels are
positional, in the order of the device's EFIT green tables, so names and
geometry come from a :class:`~bouquet.kinrecon.measurements.MachineBundle`
listed in that order (as for :mod:`.efit_kfile`).

Arrays read (EFIT output names; the lookup is case-blind, so measurement-file
names such as ``silopt`` work as well):

====================  ===========  ===========  ===========  ===========  ============
group                 measured     fitted       weight       stored chi2  stored sigma
====================  ===========  ===========  ===========  ===========  ============
``flux_loop``         ``SILOPT``   ``CSILOP``   ``FWTSI``    ``SAISIL``   ``SIGSIL``
``bpol_probe``        ``EXPMPI``   ``CMPR2``    ``FWTMP2``   ``SAIMPI``   ``SIGMPI``
``pf_current`` (F)    ``FCCURT``   ``CCBRSP``   ``FWTFC``    ``CHIFCC``   ``SIGFCC``
``pf_current`` (E)    ``ECCURT``   ``CECURR``   ``FWTEC``    ``CHIECC``   ``SIGECC``
``ip``                ``PLASMA``   ``CPASMA``   ``FWTPASMA`` ``CHIPASMA`` ``SIGPASMA``
``diamagnetic_flux``  ``DIAMAG``   ``CDFLUX``   ``FWTDIA``   ``CHIDFLUX`` ``SIGDIA``
``mse``               ``TANGAM``   ``CMGAM``    ``FWTGAM``   ``CHIGAM``   ``SIGGAM``
====================  ===========  ===========  ===========  ===========  ============

MSE geometry: ``RRGAM``, ``ZZGAM`` and the coefficients ``A1GAM`` ..
``A7GAM`` (absent ones are zero), mapped with
:func:`~bouquet.kinrecon.measurements.efit_to_imas_coefficients`.

**Uncertainty: weights are not sigmas.**  Archive trees often carry EFIT's
weights and chi^2 but not its sigma arrays.  EFIT's weight-setting step
(L. L. Lao et al., Nucl. Fusion 25, 1611 (1985) describe the code; the step
was read from its source) divides the INPUT weight of every channel by its
sigma to the power ``NSQ``; what an output file stores under the weight name
is a matter of that output's convention, so it is never assumed here.  The
caller states it (``sigma_model``):

* ``"weight_inverse"`` -- the stored weight is ``w_in / sigma`` with a stated
  input weight ``w_in`` (``input_weight``, normally 1): ``sigma = w_in / FWT``
  on channels with ``FWT > 0``.  :func:`weight_convention_check` tests the
  statement against the stored chi^2 (``chi2 = (FWT (measured - fitted))^2``
  per channel when the convention holds with ``w_in = 1``) and its result is
  recorded in each group's ``meta["weight_check"]``; with
  ``require_weight_check=True`` a group whose stored chi^2 contradicts the
  convention is refused;
* ``"stored"`` -- EFIT's stored sigma arrays (the measurement file carries
  them);
* ``"rule"`` -- EFIT's own sigma rule evaluated on the stored measured values
  with constants the caller supplies (``rule_inputs``; the same rules and
  readings as the k-file adapter, :mod:`.efit_kfile`: probes, coils and Ip
  ``max(SERROR |x|, VBIT bit)``, flux loops by ``KERSIL`` with the
  ``"siref"`` reading and the reference flux ``reference_flux``).

Groups without a stored weight (some trees carry no Ip or diamagnetic-flux
weight) take their sigma from the stored sigma array when present, else from
``rule_inputs`` when given, else they are built with every channel invalid and
the reason recorded (``meta["invalid_reason"]``) -- never with an invented
sigma.  A channel with weight 0 is invalid for magnetics and MSE (EFIT did not
fit it).  Coil currents with weight 0 are scored with weight 1 when a sigma is
available for them (``score_unfitted_coils``, the k-file adapter's scoring
convention), otherwise they are invalid with the reason recorded.

**Flux-loop storage and gauge** (stated, never inferred from a threshold).
``flux_storage`` is ``"relative"`` (loops stored relative to the reference
loop, whose own entry is then near zero) or ``"absolute"``.  The absolute flux
the solver needs is ``measured + meta["absolute_offset"]`` with
``absolute_offset = flux_scale * reference_flux``; ``reference_flux``
[source units per radian] and its ``reference_convention`` (``"measured"``:
the measured reference-loop flux; ``"fitted"``: a reference offset fitted by
EFIT, its ``FITSIREF`` option) are inputs, recorded with the value of the
reference loop's own stored entry.  Scoring stays in the storage form
(relative to ``flux_reference`` when that label is named), or -- with
``flux_offset="fitted"`` (recorded in the group's ``meta``) -- the loops are
scored absolutely plus one additive offset solved in closed form in every
table (:func:`..score.score`): the reading of data stored relative to an
offset the producing fit itself fitted (reference convention ``"fitted"``).

**Units and sign** as in :mod:`.efit_kfile`: flux loops (and their fitted
values) multiplied by ``flux_scale`` (default ``-2 pi``); coil currents as
stored (or divided by the bundle's ``turns`` for the arrays named in
``coil_current_per_turn``); Ip [A]; diamagnetic flux multiplied by
``diamagnetic_scale`` (the caller states the stored unit; default 1, i.e. Wb);
MSE ``tan(gamma)``.

**MSE E_r status** must be stated (``er_corrected=True/False``): the stored
arrays carry no uncorrected angle to decide it from.

**EFIT's own fit.**  Every group keeps EFIT's fitted values in
``meta["source_fit"]`` (scaled like the measured values) so that EFIT can be
scored on exactly the channels and sigma of the set
(:func:`source_fit_chi2`).

**Time.**  ``time`` [s] and the full averaging window ``time_window`` [s] are
stated by the caller (stored trees carry the half-width of EFIT's averaging
in their own conventions; the reading is the caller's and is recorded).

Nothing is defaulted silently: the storage form, the reference flux, the
weight convention, the input weight and the MSE status are required where they
are used, and channel counts must match the bundle (a nonzero entry beyond
the bundle's channels is refused).
"""

import numpy as np

from ..interfaces import ConstraintGroup, MeasurementSet
from ..measurements import efit_to_imas_coefficients
from . import apply_exclusions
from .efit_kfile import EFIT_SIGMA_FLOOR, _probe_n_avg, kersil_loop_sigma

#: Sigma models of this adapter (module docstring).
TREE_SIGMA_MODELS = ("weight_inverse", "stored", "rule")
#: Storage forms of the flux loops the caller can state.
FLUX_STORAGE_FORMS = ("relative", "absolute")
#: Conventions of the reference flux.
REFERENCE_CONVENTIONS = ("measured", "fitted")
#: Default relative tolerance of :func:`weight_convention_check` (a check of a stated convention, not a fit).
WEIGHT_CHECK_RTOL = 1.0e-3

#: group -> (measured, fitted, weight, stored chi2, stored sigma) array names
NAMES = {
    "flux_loop": ("SILOPT", "CSILOP", "FWTSI", "SAISIL", "SIGSIL"),
    "bpol_probe": ("EXPMPI", "CMPR2", "FWTMP2", "SAIMPI", "SIGMPI"),
    "pf_f": ("FCCURT", "CCBRSP", "FWTFC", "CHIFCC", "SIGFCC"),
    "pf_e": ("ECCURT", "CECURR", "FWTEC", "CHIECC", "SIGECC"),
    "ip": ("PLASMA", "CPASMA", "FWTPASMA", "CHIPASMA", "SIGPASMA"),
    "diamagnetic_flux": ("DIAMAG", "CDFLUX", "FWTDIA", "CHIDFLUX", "SIGDIA"),
    "mse": ("TANGAM", "CMGAM", "FWTGAM", "CHIGAM", "SIGGAM"),
}
ALL_GROUPS = ("flux_loop", "bpol_probe", "ip", "pf_current", "diamagnetic_flux", "mse")


class _Arrays:
    """Case-blind lookup of the caller's arrays; absent, empty or all-NaN entries count as absent."""

    def __init__(self, arrays):
        self._d = {}
        for k, v in dict(arrays).items():
            if v is None:
                continue
            a = np.asarray(v, dtype=float).ravel()
            if a.size == 0 or not np.any(np.isfinite(a)):
                continue
            self._d[str(k).upper()] = a

    def get(self, name):
        return self._d.get(name.upper())

    def has(self, name):
        return name.upper() in self._d


def _fit_n(a, n, name):
    """``a`` cut to ``n`` channels: entries beyond ``n`` must be zero or non-finite (padding)."""
    if a is None:
        return None
    if a.size < n:
        raise ValueError(f"array {name} has {a.size} entries for {n} channels of the machine bundle")
    extra = a[n:]
    if np.any(np.isfinite(extra) & (extra != 0.0)):
        raise ValueError(f"array {name} has nonzero entries beyond the {n} channels of the machine bundle: the "
                         "bundle does not match the arrays")
    return a[:n].copy()


def sigma_from_weight(weight, input_weight):
    """``sigma = input_weight / weight`` where ``weight > 0`` (NaN elsewhere): the ``"weight_inverse"``
    convention (module docstring)."""
    w = np.asarray(weight, dtype=float)
    wi = np.broadcast_to(np.asarray(input_weight, dtype=float), w.shape)
    out = np.full(w.shape, np.nan)
    pos = np.isfinite(w) & (w > 0)
    out[pos] = wi[pos] / w[pos]
    return out


def weight_convention_check(measured, fitted, weight, chi2, rtol=WEIGHT_CHECK_RTOL):
    """Which reading of a stored weight the stored chi^2 supports, channel by channel.

    On the channels with ``weight > 0``, ``chi2 > 0`` and a nonzero residual ``r = measured - fitted``:
    ``ratio_inverse_sigma = chi2 / (weight r)^2`` (1 when the weight is ``1/sigma``) and
    ``ratio_inverse_variance = chi2 / (weight r^2)`` (1 when the weight is ``1/sigma^2``).  Returns the
    count, the fraction of channels within ``rtol`` of 1 for each reading, their 0.5/50/99.5 percentiles and
    ``supports`` (the reading that holds on at least 99 % of the channels, else ``None``).  A record, not a
    fit: nothing is adjusted.
    """
    m, c, w, s = (np.asarray(v, dtype=float) for v in (measured, fitted, weight, chi2))
    r = m - c
    use = np.isfinite(r) & np.isfinite(w) & np.isfinite(s) & (w > 0) & (s > 0) & (r != 0.0)
    out = dict(n=int(use.sum()), rtol=float(rtol), rule="chi2 = (w r)^2 (w = 1/sigma) or w r^2 (w = 1/sigma^2)")
    if not use.any():
        out.update(supports=None, reason="no channel with weight > 0, chi2 > 0 and a nonzero residual")
        return out
    ra = s[use] / (w[use] * r[use]) ** 2
    rb = s[use] / (w[use] * r[use] ** 2)
    pct = lambda a: [float(np.percentile(a, p)) for p in (0.5, 50.0, 99.5)]  # noqa: E731
    fa = float(np.mean(np.abs(ra - 1.0) <= rtol))
    fb = float(np.mean(np.abs(rb - 1.0) <= rtol))
    out.update(frac_inverse_sigma=fa, frac_inverse_variance=fb, pct_inverse_sigma=pct(ra),
               pct_inverse_variance=pct(rb),
               supports="inverse_sigma" if fa >= 0.99 else ("inverse_variance" if fb >= 0.99 else None))
    return out


def measurements_from_efit_arrays(arrays, machine, time, time_window, *, sigma_model, groups=None,
                                  flux_storage=None, reference_flux=None, reference_convention=None,
                                  reference_flux_source=None, flux_reference=None, flux_offset=None,
                                  reference_prior=None,
                                  flux_scale=-2.0 * np.pi,
                                  input_weight=None, require_weight_check=False, weight_check_rtol=WEIGHT_CHECK_RTOL,
                                  rule_inputs=None, er_corrected=None, coil_current_per_turn=(),
                                  score_unfitted_coils=True, diamagnetic_scale=1.0, exclude=None, meta=None):
    """Build a :class:`MeasurementSet` from EFIT's stored arrays of one time (module docstring).

    Parameters
    ----------
    arrays : dict
        Array name (table in the module docstring) -> values of this time (1-D per channel; scalars for Ip
        and the diamagnetic flux).  Fitted values and stored chi^2 are optional (recorded when given).
    machine : MachineBundle
        Sensors in EFIT's channel order: ``flux_loop``, ``bpol_probe``, ``pf_current`` (labels
        ``channel_maps["BRSP"]`` + ``channel_maps["ECURRT"]`` when given), optionally ``mse``.
    time, time_window : float
        Slice time and full averaging window [s] (stated by the caller, recorded).
    sigma_model : {"weight_inverse", "stored", "rule"}
        Required (module docstring, "Uncertainty").
    groups : sequence, optional
        Groups to build (default: those whose measured array is present).
    flux_storage : {"relative", "absolute"}
        Storage form of the loop fluxes (required for ``flux_loop``).
    reference_flux : float
        Absolute flux of the reference loop [source units per radian] (required for ``flux_loop``);
        ``reference_convention`` ("measured" or "fitted") and ``reference_flux_source`` (text) are recorded.
    flux_reference : str, optional
        Label of the reference loop for relative scoring (only with ``flux_storage="relative"``).
    flux_offset : {None, "fitted"}
        ``"fitted"``: score the loops with one closed-form additive offset per table (not with
        ``flux_reference``).
    reference_prior : (float, float), optional
        With ``flux_offset="fitted"``: the MEASURED reference flux and the reference loop's sigma [source units
        per radian]; the scorer's offset then carries EFIT's prior row tying it to that measurement (the
        reference loop stays an ordinary row as well).  Recorded as ``meta["flux_offset_prior"]`` in
        measurement units.
    input_weight : float or dict, optional
        ``"weight_inverse"``: EFIT's input weight of the fitted channels (a number for every group or
        ``{group: number or array}``; required).
    require_weight_check : bool
        ``"weight_inverse"``: refuse a group whose stored chi^2 does not support ``chi2 = (w r)^2`` on at
        least 99 % of its channels (:func:`weight_convention_check` with ``weight_check_rtol``; a group
        without a stored chi^2 is then refused too).  False (default): the check is recorded only.
    rule_inputs : dict, optional
        ``"rule"`` (and groups without weights): ``SERROR``, ``VBIT``, digitizer bits ``PSIBIT``,
        ``BITMPI``, ``BITFC``, ``BITEC``, ``BITIP`` and, for the flux loops, ``KERSIL``, ``ERRSIL``,
        ``SICONT`` (``ERSIL8`` for ``KERSIL = 2``) and ``NSLREF`` (EFIT namelist names; loop radii from the
        bundle).
    er_corrected : bool
        MSE E_r status (required when ``mse`` is built).
    """
    if sigma_model not in TREE_SIGMA_MODELS:
        raise ValueError(f"sigma_model must be one of {TREE_SIGMA_MODELS} (stated, no default), got {sigma_model!r}")
    A = _Arrays(arrays)
    rule = dict(rule_inputs or {})
    if sigma_model == "weight_inverse" and input_weight is None:
        raise ValueError("sigma_model 'weight_inverse' needs input_weight (EFIT's input weight of the fitted "
                         "channels, stated by the caller)")
    if sigma_model == "rule" and not rule:
        raise ValueError("sigma_model 'rule' needs rule_inputs (SERROR, VBIT, digitizer bits, KERSIL constants)")
    present = {"flux_loop": A.has("SILOPT"), "bpol_probe": A.has("EXPMPI"), "ip": A.has("PLASMA"),
               "pf_current": A.has("FCCURT") or A.has("ECCURT"), "diamagnetic_flux": A.has("DIAMAG"),
               "mse": A.has("TANGAM") and A.has("FWTGAM") and np.any(A.get("FWTGAM") > 0)}
    if groups is None:
        groups = [g for g in ALL_GROUPS if present[g]]
    else:
        groups = list(groups)
        missing = [g for g in groups if g not in present or not present[g]]
        if missing:
            raise ValueError(f"group(s) {missing} requested but their measured arrays are absent")
    ms_meta = {"source": "efit_arrays", "time_source": "caller", "time_window_source": "caller",
               "time_window_s": [float(time) - 0.5 * float(time_window), float(time) + 0.5 * float(time_window)],
               "sigma_model": sigma_model, "sigma_model_selection": "named by the caller",
               "input_weight": input_weight if not isinstance(input_weight, np.ndarray) else input_weight.tolist(),
               "rule_inputs": {k: (np.asarray(v).tolist() if np.ndim(v) else v) for k, v in rule.items()}}
    ms_meta.update(meta or {})
    ms = MeasurementSet(machine=machine.machine, time=float(time), time_window=float(time_window), meta=ms_meta)

    def w_in(group, n):
        if isinstance(input_weight, dict):
            if group not in input_weight:
                raise ValueError(f"input_weight has no entry for group '{group}'")
            v = input_weight[group]
        else:
            v = input_weight
        return np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()

    def rule_sigma(group, value, bitkey):
        for k in ("SERROR", "VBIT", bitkey):
            if k not in rule:
                return None, f"rule needs {k} (rule_inputs)"
        bit = np.broadcast_to(np.asarray(rule[bitkey], dtype=float), value.shape)
        return (np.maximum(float(rule["SERROR"]) * np.abs(value), float(rule["VBIT"]) * np.abs(bit)),
                f"max(SERROR |x|, VBIT {bitkey}) with rule_inputs")

    def build(group, key, n, labels, geometry, scale=1.0, bitkey=None, coil=False, kersil=None, source=None):
        mname, cname, wname, chname, sname = NAMES[key]
        S = A if source is None else source
        value = _fit_n(S.get(mname), n, mname)
        fitted = _fit_n(S.get(cname), n, cname)
        fwt = _fit_n(S.get(wname), n, wname)
        chi2 = _fit_n(S.get(chname), n, chname)
        ssig = _fit_n(S.get(sname), n, sname)
        if value is None or np.count_nonzero(np.isfinite(value)) < n:
            raise ValueError(f"group '{group}': {mname} has fewer finite values than the {n} channels")
        gm = {"source": "efit_arrays", "arrays": dict(measured=mname, fitted=cname if fitted is not None else None,
                                                       weight=wname if fwt is not None else None,
                                                       chi2=chname if chi2 is not None else None,
                                                       sigma=sname if ssig is not None else None)}
        def by_rule():
            """EFIT's rule for this group: KERSIL (loops) or the bit rule; groups whose rule IS the stored
            sigma (MSE, diamagnetic flux) use it."""
            if kersil is not None:
                s, t = kersil(value)
                return (s, t) if s is not None else (None, "KERSIL rule needs KERSIL, ERRSIL, SICONT, NSLREF, VBIT, "
                                                           "PSIBIT, SERROR (rule_inputs) and PLASMA")
            if bitkey is not None:
                return rule_sigma(group, value, bitkey)
            if ssig is not None:
                return ssig.copy(), f"EFIT's stored sigma {sname} (EFIT's rule for this group)"
            return None, "no rule for this group"

        sigma, why, src = None, None, None
        if sigma_model == "weight_inverse" and fwt is not None:
            sigma = sigma_from_weight(fwt, w_in(group, n))
            src = f"input_weight / {wname} (stored weight read as input_weight / sigma, stated by the caller)"
            if fitted is not None and chi2 is not None:
                chk = weight_convention_check(value, fitted, fwt, chi2, rtol=weight_check_rtol)
                gm["weight_check"] = chk
                if require_weight_check and chk.get("supports") != "inverse_sigma":
                    raise ValueError(f"group '{group}': the stored chi^2 ({chname}) does not support the "
                                     f"inverse-sigma reading of {wname} ({chk}); refused (require_weight_check)")
            elif require_weight_check:
                raise ValueError(f"group '{group}': no stored chi^2 / fitted values to check the weight "
                                 "convention against (require_weight_check)")
            else:
                gm["weight_check"] = dict(supports=None, reason="no stored chi^2 or fitted values")
            if coil and score_unfitted_coils and np.any(fwt <= 0):
                # unfitted coils (weight 0) have no sigma in the weights: stored sigma, else EFIT's rule, recorded
                s2, t2 = (ssig.copy(), f"EFIT's stored sigma {sname}") if ssig is not None else by_rule()
                if s2 is not None:
                    sigma = np.where(fwt > 0, sigma, s2)
                    gm["sigma_rule_unfitted"] = t2
                else:
                    gm["invalid_reason_unfitted"] = f"weight 0 and no sigma available ({t2})"
        elif sigma_model == "stored" and ssig is not None:
            sigma, src = ssig.copy(), f"EFIT's stored sigma {sname}"
        elif sigma_model == "rule":
            sigma, src = by_rule()
        elif ssig is not None:
            sigma, src = ssig.copy(), f"EFIT's stored sigma {sname} (no stored weight {wname} for this group)"
        elif rule:
            sigma, src = by_rule()
        if sigma is None:
            why = (src or f"no stored weight ({wname}) or sigma ({sname}) and no rule inputs for this group")
            src = None
            sigma = np.full(n, np.nan)
        else:
            sigma = np.asarray(sigma, dtype=float) * abs(scale)
        floor = EFIT_SIGMA_FLOOR * abs(scale)
        good = np.isfinite(value) & np.isfinite(sigma) & (sigma > floor)
        if fwt is not None:
            fitted_by_efit = fwt > 0
        else:
            fitted_by_efit = np.ones(n, dtype=bool)
        if coil and score_unfitted_coils:
            valid = good
        else:
            valid = good & fitted_by_efit
        weight = np.ones(n)
        gm.update(sigma_model=sigma_model, sigma_model_id=f"efit_arrays:{sigma_model}", sigma_rule=src,
                  sigma_params={})
        if why is not None:
            gm["invalid_reason"] = why
        if fwt is not None:
            gm["efit_fit_weight"] = fwt.copy()
            gm["not_fitted_by_efit"] = [labels[i] for i in np.nonzero(~fitted_by_efit)[0]]
        if fitted is not None:
            gm["source_fit"] = fitted * scale
        return ConstraintGroup(name=group, labels=list(labels), measured=value * scale, sigma=sigma, valid=valid,
                               weight=weight, geometry=geometry, meta=gm)

    if "flux_loop" in groups:
        if flux_storage not in FLUX_STORAGE_FORMS:
            raise ValueError(f"flux_storage must be stated as one of {FLUX_STORAGE_FORMS} (no inference)")
        if reference_flux is None or not np.isfinite(float(reference_flux)):
            raise ValueError("reference_flux (absolute flux of the reference loop, per radian) is required: the "
                             "solver needs the absolute gauge")
        if reference_convention not in REFERENCE_CONVENTIONS:
            raise ValueError(f"reference_convention must be one of {REFERENCE_CONVENTIONS}")
        labels = machine.labels("flux_loop")
        n = len(labels)
        if flux_offset not in (None, "fitted"):
            raise ValueError("flux_offset must be None or 'fitted'")
        if flux_offset == "fitted" and flux_reference is not None:
            raise ValueError("flux_offset='fitted' and flux_reference are two readings of the gauge: give one")
        if flux_reference is not None:
            if flux_reference not in labels:
                raise ValueError(f"flux_reference {flux_reference!r} is not a flux-loop label")
            if flux_storage != "relative":
                raise ValueError("flux_reference names the loop relative fluxes are scored against: only with "
                                 "flux_storage='relative'")

        def kersil(value):
            if not {"KERSIL", "ERRSIL", "SICONT", "NSLREF", "VBIT", "PSIBIT", "SERROR"} <= set(rule):
                return None, None
            in1 = {"KERSIL": [float(rule["KERSIL"])], "NSLREF": [float(rule["NSLREF"])],
                   "SIREF": [float(reference_flux) if flux_storage == "relative" else 0.0]}
            consts = {k: float(rule[k]) for k in ("ERRSIL", "SICONT", "ERSIL8") if k in rule}
            ip = A.get("PLASMA")
            if ip is None:
                return None, None
            sig, krec = kersil_loop_sigma(in1, value, np.asarray(rule["PSIBIT"], float)[:n],
                                          machine.geometry("flux_loop").get("r"), float(ip[0]), float(rule["VBIT"]),
                                          float(rule["SERROR"]), consts, "siref")
            return sig, krec["rule"] + " (rule_inputs)"

        g = build("flux_loop", "flux_loop", n, labels, machine.geometry("flux_loop"), scale=flux_scale,
                  kersil=kersil)
        iref = labels.index(flux_reference) if flux_reference is not None else None
        if reference_prior is not None:
            if flux_offset != "fitted":
                raise ValueError("reference_prior applies to flux_offset='fitted'")
            r0, s0 = (float(v) for v in reference_prior)
            # the offset the scorer fits is measured - model = -flux_scale x reference flux
            g.meta["flux_offset_prior"] = [-float(flux_scale) * r0, abs(float(flux_scale)) * s0]
            g.meta["flux_offset_prior_source"] = "measured reference flux and the reference loop's sigma (caller)"
        g.meta.update(flux_scale=flux_scale, units="stored loop flux [per radian] x flux_scale",
                      flux_reference=flux_reference, flux_offset=flux_offset, coils_form=flux_storage,
                      coils_form_rule="caller",
                      coils_form_evidence=("stated by the caller" + ("" if iref is None else
                                           f"; reference loop's stored entry {g.measured[iref] / flux_scale:.6g} "
                                           "(per radian)")),
                      siref=float(reference_flux), reference_convention=reference_convention,
                      reference_flux_source=reference_flux_source or "not stated",
                      absolute_offset=float(flux_scale) * float(reference_flux) if flux_storage == "relative" else 0.0,
                      absolute_rule=("absolute flux = measured + absolute_offset; absolute_offset = flux_scale x "
                                     "reference_flux for relative storage, 0 for absolute storage"))
        ms.meta["SIREF"] = float(reference_flux)
        ms.groups["flux_loop"] = g
    if "bpol_probe" in groups:
        labels = machine.labels("bpol_probe")
        g = build("bpol_probe", "bpol_probe", len(labels), labels, machine.geometry("bpol_probe"), bitkey="BITMPI")
        g.meta.update(_probe_n_avg(machine))
        ms.groups["bpol_probe"] = g
    if "ip" in groups:
        ms.groups["ip"] = build("ip", "ip", 1, ["ip"], {}, bitkey="BITIP")
    if "pf_current" in groups:
        ms.groups["pf_current"] = _coils(A, machine, build, coil_current_per_turn, score_unfitted_coils)
    if "diamagnetic_flux" in groups:
        g = build("diamagnetic_flux", "diamagnetic_flux", 1, ["diamagnetic_flux"], {}, scale=diamagnetic_scale)
        g.meta["units"] = f"stored value x diamagnetic_scale = {diamagnetic_scale:g} -> Wb"
        ms.groups["diamagnetic_flux"] = g
    if "mse" in groups:
        if er_corrected not in (True, False):
            raise ValueError("er_corrected must be stated (True/False): the stored arrays carry no uncorrected "
                             "angle to decide it from")
        n = int(np.count_nonzero(np.isfinite(A.get("TANGAM"))))
        fw = A.get("FWTGAM")
        n = min(n, fw.size)
        if "mse" in machine.sensors and len(machine.labels("mse")) <= n:
            n = len(machine.labels("mse"))
            labels, lab_src = machine.labels("mse"), "machine bundle"
        else:
            labels, lab_src = [f"mse_{i:03d}" for i in range(n)], "generic (no matching 'mse' sensors in the bundle)"
        for k in ("RRGAM", "ZZGAM", "A1GAM", "A2GAM", "A3GAM", "A4GAM"):
            if not A.has(k):
                raise ValueError(f"MSE geometry array {k} absent")
        coef = efit_to_imas_coefficients(*(A.get(f"A{i}GAM")[:n] if A.has(f"A{i}GAM") else np.zeros(n)
                                           for i in range(1, 8)))
        geo = {"r": A.get("RRGAM")[:n], "z": A.get("ZZGAM")[:n], "coefficients": coef}
        sub = {k: (A.get(k)[:n] if A.has(k) else None) for k in NAMES["mse"]}
        g = build("mse", "mse", n, labels, geo, source=_Arrays({k: v for k, v in sub.items() if v is not None}))
        tg = g.measured
        g.valid &= np.isfinite(tg) & (np.abs(tg) > 1e-8)
        g.meta.update(quantity="tan_polarisation_angle", er_corrected=bool(er_corrected),
                      er_corrected_source="caller", er_route="caller_stated", labels_source=lab_src,
                      coefficient_source="A1GAM..A7GAM (absent ones zero)")
        ms.groups["mse"] = g
    excl = {}
    for p in ((getattr(machine, "meta", None) or {}).get("channel_exclusions"), exclude):
        for gname, d in (p or {}).items():
            excl.setdefault(gname, {}).update({str(k): str(v) for k, v in dict(d).items()})
    if excl:
        ms.meta["channel_exclusions_applied"] = apply_exclusions(ms, excl)
    ms.meta["sigma_models"] = {k: g.meta.get("sigma_model_id") for k, g in ms.groups.items()}
    return ms


def _coils(A, machine, build, per_turn, score_unfitted):
    parts = [(k, key) for k, key in (("BRSP", "pf_f"), ("ECURRT", "pf_e")) if A.has(NAMES[key][0])]
    maps = machine.channel_maps
    if not all(k in maps for k, _ in parts):
        raise ValueError("pf_current needs channel_maps['BRSP'] and channel_maps['ECURRT'] in the bundle (the "
                         "stored arrays are positional)")
    order = machine.labels("pf_current") if "pf_current" in machine.sensors else None
    groups = []
    for k, key in parts:
        labels = list(maps[k])
        geo = {}
        if order is not None:
            idx = [order.index(s) for s in labels]
            geo = machine.geometry("pf_current", idx)
        scale = 1.0
        if per_turn is True or (per_turn and k in per_turn):
            turns = geo.get("turns")
            if turns is None:
                raise ValueError("coil_current_per_turn needs 'turns' in the bundle's pf_current")
            scale = 1.0 / np.asarray(turns, dtype=float)
        groups.append(build("pf_current", key, len(labels), labels, geo, scale=scale,
                            bitkey="BITFC" if k == "BRSP" else "BITEC", coil=True))
    if len(groups) == 1:
        g = groups[0]
    else:
        a, b = groups
        meta = dict(a.meta)
        for kk in ("efit_fit_weight", "source_fit"):
            if kk in a.meta and kk in b.meta:
                meta[kk] = np.concatenate((a.meta[kk], b.meta[kk]))
            else:
                meta.pop(kk, None)
        meta["not_fitted_by_efit"] = list(a.meta.get("not_fitted_by_efit", [])) + list(b.meta.get("not_fitted_by_efit", []))
        meta["sigma_rule_by_array"] = {"BRSP": a.meta.get("sigma_rule"), "ECURRT": b.meta.get("sigma_rule")}
        meta["weight_check_by_array"] = {"BRSP": a.meta.get("weight_check"), "ECURRT": b.meta.get("weight_check")}
        for kk in ("invalid_reason", "sigma_rule_unfitted", "invalid_reason_unfitted"):
            meta[kk + "_by_array"] = {"BRSP": a.meta.get(kk), "ECURRT": b.meta.get(kk)}
            meta.pop(kk, None)
        meta.pop("weight_check", None)
        geo = {kk: np.concatenate((a.geometry[kk], b.geometry[kk])) for kk in a.geometry if kk in b.geometry}
        g = ConstraintGroup(name="pf_current", labels=a.labels + b.labels,
                            measured=np.concatenate((a.measured, b.measured)), sigma=np.concatenate((a.sigma, b.sigma)),
                            valid=np.concatenate((a.valid, b.valid)), weight=np.concatenate((a.weight, b.weight)),
                            geometry=geo, meta=meta)
    if "coil_names" not in g.geometry:
        g.geometry["coil_names"] = np.array(g.labels)
    fw = g.meta.get("efit_fit_weight")
    roles = {}
    for i, lab in enumerate(g.labels):
        is_f = lab in maps.get("BRSP", [])
        if fw is None:
            roles[lab] = "undetermined"
        elif is_f:
            roles[lab] = "fitted" if fw[i] > 0 else "free"
        else:
            roles[lab] = "fitted" if fw[i] > 0 else "undetermined"
    g.meta["coil_roles"] = roles
    g.meta["coil_role_rule"] = ("F-coils: stored weight > 0 -> 'fitted', 0 -> 'free'; E-coils: weight > 0 -> "
                                "'fitted', 0 -> 'undetermined' (whether EFIT held them at the measured current is a "
                                "namelist option the arrays do not carry: the caller states it)")
    g.meta["score_unfitted_coils"] = bool(score_unfitted)
    g.meta["coil_arrays"] = {k: list(maps[k]) for k, _ in parts}
    return g


def source_fit_chi2(ms, groups=None):
    """EFIT's own chi^2 on exactly the set's channels and sigma: ``sum ((fitted - measured) / sigma)^2`` over
    the valid channels of each group that carries ``meta["source_fit"]``.  Fitted and measured values are
    compared in the form they are stored in (relatively stored loops: EFIT's fitted values are in the same
    relative form, as its own stored chi^2 compares them); with the group's ``flux_offset="fitted"`` one
    closed-form additive offset is applied to the fitted loops as well (the same convention as the scorer,
    including the reference prior row when the group carries one; its chi^2 term is returned separately as
    ``"flux_loop_offset_prior"``).  Returns ``{group: (chi2, n, chi2/n)}``."""
    out = {}
    for name, g in ms.groups.items():
        if groups is not None and name not in groups:
            continue
        fit = g.meta.get("source_fit")
        if fit is None:
            continue
        m, f = np.asarray(g.measured, float), np.asarray(fit, float)
        prior_term = None
        if name == "flux_loop" and g.meta.get("flux_offset") == "fitted":
            u0 = g.valid & (g.weight != 0) & np.isfinite(f)
            w = (g.weight[u0] / g.sigma[u0]) ** 2
            num, den = float(np.sum(w * (m[u0] - f[u0]))), float(np.sum(w))
            pr = g.meta.get("flux_offset_prior")
            if pr is not None:
                num, den = num + pr[0] / pr[1] ** 2, den + 1.0 / pr[1] ** 2
            c = num / den if den > 0 else 0.0
            f = f + c
            if pr is not None:
                prior_term = float(((c - pr[0]) / pr[1]) ** 2)
        use = g.valid & (g.weight != 0) & np.isfinite(f)
        z = g.weight[use] * (f[use] - m[use]) / g.sigma[use]
        n = int(use.sum())
        out[name] = (float(np.sum(z ** 2)), n, float(np.sum(z ** 2) / n) if n else float("nan"))
        if prior_term is not None:
            out["flux_loop_offset_prior"] = (prior_term, 1, prior_term)
    return out


__all__ = ["measurements_from_efit_arrays", "sigma_from_weight", "weight_convention_check", "source_fit_chi2",
           "TREE_SIGMA_MODELS", "FLUX_STORAGE_FORMS", "REFERENCE_CONVENTIONS", "NAMES", "WEIGHT_CHECK_RTOL"]
