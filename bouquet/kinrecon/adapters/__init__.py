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
    # EFIT convention for magnetics: sigma = max(SERROR |value|, 10 * bit),
    # with SERROR and the digitizer bit (BITMPI, PSIBIT, BITFC, BITEC, BITIP)
    # from k-file namelist IN1.  See adapters.efit_kfile for the source.
    return np.maximum(serror * np.abs(values), 10.0 * np.asarray(bit, dtype=float))


#: name -> (function(values, **params), required parameter names)
SIGMA_MODELS = {
    "absolute": (_absolute, ("sigma",)),
    "relative": (_relative, ("relative",)),
    "relative_floor": (_relative_floor, ("relative", "floor")),
    "relative_quadrature": (_relative_quadrature, ("relative", "floor")),
    "efit_serror_bit": (_efit_serror_bit, ("serror", "bit")),
}


def sigma_from_model(name, values, **params):
    """1-sigma uncertainty of ``values`` under the named model.

    ``absolute``: sigma = ``sigma`` (scalar or per channel);
    ``relative``: sigma = ``relative`` |v|;
    ``relative_floor``: sigma = max(``relative`` |v|, ``floor``);
    ``relative_quadrature``: sigma = sqrt((``relative`` v)^2 + ``floor``^2);
    ``efit_serror_bit``: sigma = max(``serror`` |v|, 10 ``bit``) (EFIT k-file
    convention for magnetics, see :mod:`.efit_kfile`).
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
