"""Adapters that build a :class:`~bouquet.kinrecon.interfaces.MeasurementSet`.

* :mod:`.synthetic` -- from model values (plain arrays) + a sensor set + a
  noise model, for closure tests and limited-coverage studies;
* :mod:`.imas_json` -- from an IMAS-style JSON dict (OMAS / FUSE output) and a
  time window;
* :mod:`.efit_kfile` -- from an EFIT k-file namelist and a machine bundle.

Every group an adapter builds records the name of its uncertainty convention
in ``meta["sigma_model"]`` (and its parameters in ``meta["sigma_params"]``).
σ models are conventions, not calibrations; they are chosen explicitly and
never changed silently.  The named models below are shared by the adapters.
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
    # from the k-file's &IN1 (10 when absent).  Reproduces the sigma arrays
    # EFIT writes to its m-file channel by channel; see adapters.efit_kfile.
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
    :mod:`.efit_kfile`; the k-file adapter's default).
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


__all__ = ["SIGMA_MODELS", "sigma_from_model"]
