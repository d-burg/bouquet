"""Adapters that build a :class:`~bouquet.kinrecon.interfaces.MeasurementSet`.

* :mod:`.synthetic` -- from model values (plain arrays) + a sensor set + a
  noise model, for closure tests and limited-coverage studies;
* :mod:`.imas_json` -- from an IMAS-style JSON dict (OMAS / FUSE output) and a
  time window;
* :mod:`.efit_kfile` -- from an EFIT k-file namelist and a machine bundle;
* :mod:`.efit_tree` -- from an EFIT run's stored measurement arrays of one
  time (measured and fitted values, weights, chi^2) and a machine bundle.

Every group an adapter builds records the name of its uncertainty convention
in ``meta["sigma_model"]`` (and its parameters in ``meta["sigma_params"]``).
σ models are conventions, not calibrations; they are chosen explicitly and
never changed silently.  The named models below are shared by the adapters.

Conventions (shared by the adapters and the scorer; documented defaults, not
tunable choices):

* **No-data sentinels.**  A channel the source marks as having no data is
  invalid, never zero-filled: EFIT MSE ``|TGAMMA| <= 1e-8`` (EFIT writes 0
  for chords without data); OMAS / IMAS ``data_error_upper >= 1e29`` (OMAS
  writes 1e30 for an excluded channel); a sigma that is not finite and > 0.
  Invalid channels keep their position and label (values NaN where the
  source has none).
* **Exclusions are recorded, never silent.**  A channel left out of a fit or
  a score is listed with a reason: by an adapter in the group's
  ``meta["excluded"]`` (:func:`apply_exclusions`, ``(label, reason)``) or
  ``meta["invalid_reason"]``; by the scorer in ``ResidualTable.excluded``.
  Channels with fit weight 0 in the source (EFIT ``FWT* = 0``) are invalid
  for magnetics and MSE; coil currents are the stated exception (scored with
  weight 1 by default, see :mod:`.efit_kfile`).
* **Window reduction.**  Sources with time samples are reduced to one value
  per channel over the window; the IMAS route's default ``window_mean`` keeps
  the per-sample sigma undivided (no 1/sqrt(N)): the source sigma is treated
  as a convention, not as independent noise (:mod:`.imas_json`).  The EFIT
  k-file carries values EFIT already averaged over ``ITIME +/- IAVEM``.
"""

import numpy as np


def _relative(values, relative):
    return relative * np.abs(values)


def _relative_floor(values, relative, floor):
    return np.maximum(relative * np.abs(values), floor)


def _relative_quadrature(values, relative, floor):
    return np.sqrt((relative * values) ** 2 + floor ** 2)


def _absolute(values, sigma):
    return np.broadcast_to(np.asarray(sigma, dtype=float), np.shape(values)).copy()


def _efit_serror_bit(values, serror, bit):
    # Fixed-multiplier form: sigma = max(SERROR |value|, 10 * bit), with
    # SERROR and the digitizer bit (BITMPI, PSIBIT, BITFC, BITEC, BITIP) from
    # k-file namelist IN1.  The 10 is the OMAS machine-mapping constant; it
    # equals EFIT's own floor only where the k-file's VBIT is 10 (see
    # ``efit_vbit`` and adapters.efit_kfile).
    return np.maximum(serror * np.abs(values), 10.0 * np.asarray(bit, dtype=float))


def _efit_vbit(values, serror, bit, vbit):
    # EFIT-aligned form: sigma = max(SERROR |value|, VBIT * bit), VBIT read
    # from the k-file's &IN1 (EFIT's shot-number default when absent).  EFIT's
    # sigma for probes and coils; not for flux loops in general (see
    # adapters.efit_kfile).
    return np.maximum(serror * np.abs(values), float(vbit) * np.asarray(bit, dtype=float))


#: name -> (function(values, **params), required parameter names)
SIGMA_MODELS = {
    "absolute": (_absolute, ("sigma",)),
    "relative": (_relative, ("relative",)),
    "relative_floor": (_relative_floor, ("relative", "floor")),
    "relative_quadrature": (_relative_quadrature, ("relative", "floor")),
    "efit_serror_bit": (_efit_serror_bit, ("serror", "bit")),
    "efit_vbit": (_efit_vbit, ("serror", "bit", "vbit")),
}


def sigma_from_model(name, values, **params):
    """1-sigma uncertainty of ``values`` under the named model.

    ``absolute``: sigma = ``sigma`` (scalar or per channel);
    ``relative``: sigma = ``relative`` |v|;
    ``relative_floor``: sigma = max(``relative`` |v|, ``floor``);
    ``relative_quadrature``: sigma = sqrt((``relative`` v)^2 + ``floor``^2);
    ``efit_serror_bit``: sigma = max(``serror`` |v|, 10 ``bit``) (fixed
    10-bit floor, the OMAS mapping constant);
    ``efit_vbit``: sigma = max(``serror`` |v|, ``vbit`` ``bit``) (EFIT k-file
    convention for magnetics with the file's own ``VBIT``, see
    :mod:`.efit_kfile`; a named k-file model, no longer its default).
    """
    if name not in SIGMA_MODELS:
        raise ValueError(f"unknown sigma model {name!r}; known: {sorted(SIGMA_MODELS)}")
    fn, required = SIGMA_MODELS[name]
    missing = [p for p in required if p not in params]
    extra = [p for p in params if p not in required]
    if missing or extra:
        raise ValueError(f"sigma model {name!r} takes {required}; "
                         f"missing {missing}, unexpected {extra}")
    values = np.asarray(values, dtype=float)
    return np.asarray(fn(values, **params), dtype=float)


def apply_exclusions(ms, exclusions):
    """Exclude channels by default, as recorded exclusions (never by deleting them).

    ``exclusions`` is ``{group: {label: reason}}``.  Each listed channel of a group present in ``ms`` is made
    invalid (``valid = False``; its value and sigma are kept) and ``(label, reason)`` is appended to the
    group's ``meta["excluded"]`` -- the same ``(label, reason)`` form the scorer uses for
    ``ResidualTable.excluded``.  A label that is not a channel of its group raises (a misspelled exclusion
    must not be a silent no-op); groups absent from ``ms`` are skipped and reported.  Returns
    ``{group: [labels excluded]}`` plus ``"_groups_absent"`` when any were skipped.
    """
    done = {}
    for name, items in (exclusions or {}).items():
        g = ms.groups.get(name)
        if g is None:
            done.setdefault("_groups_absent", []).append(name)
            continue
        rec = g.meta.setdefault("excluded", [])
        for label, reason in dict(items).items():
            if label not in g.labels:
                raise ValueError(f"exclusion of '{label}': not a channel of group '{name}'")
            i = g.labels.index(label)
            g.valid[i] = False
            rec.append((label, str(reason)))
            done.setdefault(name, []).append(label)
    return done


__all__ = ["SIGMA_MODELS", "apply_exclusions", "sigma_from_model"]
