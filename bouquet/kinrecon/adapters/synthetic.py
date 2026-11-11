"""Synthetic measurement sets from model values, a sensor set and a noise model.

Used for closure tests (fit synthetic data made from a known equilibrium and
recover it) and for limited-coverage studies (drop sensors or whole
diagnostics and see how the error bars grow).  The model values are passed in
as plain arrays -- this module evaluates nothing itself and does not import
TokaMaker -- so any forward model can produce them.

Noise model, per group:

.. code-block:: text

    sigma_i    = sigma_model(model_i)            (named, see adapters.SIGMA_MODELS)
    measured_i = model_i + sigma_i * n_i,        n_i ~ N(0, 1)

sigma is evaluated on the MODEL (true) value, not on the noisy one, so the
noise is exactly Gaussian with the recorded sigma.  Each group draws from its
own generator, seeded by ``numpy.random.SeedSequence([seed, crc32(name)])``:
adding, removing or reordering one group never changes another group's noise
or dropout.

Dropout marks channels invalid (``valid = False``); their values are kept.
"""

import zlib

import numpy as np

from ..interfaces import ConstraintGroup, MeasurementSet
from . import sigma_from_model


def _rng(seed, name):
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(name.encode())]))


def _dropout_mask(spec, labels, rng):
    """Boolean ``keep`` mask from a dropout spec.

    ``None`` / 0: keep all; a float in (0, 1): drop ``round(f n)`` channels
    chosen at random; a sequence of labels: drop those; a boolean array:
    True = drop.
    """
    n = len(labels)
    keep = np.ones(n, dtype=bool)
    if spec is None:
        return keep
    if isinstance(spec, (float, int)) and not isinstance(spec, bool):
        if not 0.0 <= spec <= 1.0:
            raise ValueError(f"dropout fraction must be in [0, 1], got {spec}")
        k = int(round(spec * n))
        keep[rng.choice(n, size=k, replace=False)] = False
        return keep
    arr = np.asarray(spec)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"dropout mask has shape {arr.shape}, expected ({n},)")
        return ~arr
    unknown = [s for s in spec if s not in labels]
    if unknown:
        raise ValueError(f"dropout names unknown channels {unknown}")
    drop = set(spec)
    return np.array([s not in drop for s in labels])


def synthetic_measurements(machine, model, noise, seed=0, dropout=None, time=0.0,
                           time_window=0.0, add_noise=True, group_meta=None,
                           meta=None):
    """Build a :class:`MeasurementSet` from model values.

    Parameters
    ----------
    machine : MachineBundle
        Sensor set.  ``machine.sensors[name]`` gives the labels and geometry
        of group ``name``; groups absent from it (e.g. ``ip``) get the single
        label ``name`` and no geometry.
    model : dict
        Group name -> model values, one per channel, in the sensor order.
    noise : dict
        Group name -> ``{"sigma_model": <name>, **params}``
        (see :func:`~bouquet.kinrecon.adapters.sigma_from_model`).  Every
        group in ``model`` needs an entry.
    seed : int
        Base seed; recorded in each group's ``meta``.
    dropout : None, float, or dict
        A dropout spec for every group, or a dict group -> spec (see
        :func:`_dropout_mask`).
    add_noise : bool
        False returns the model values unperturbed (sigma still set) -- a
        noiseless closure test.
    group_meta : dict, optional
        Group name -> extra ``meta`` entries (e.g. ``{"mse": {"er_corrected":
        True}}``).
    """
    meta = dict(meta or {})
    meta.setdefault("source", "synthetic")
    ms = MeasurementSet(machine=machine.machine, time=float(time),
                        time_window=float(time_window), meta=meta)
    group_meta = group_meta or {}
    for name, values in model.items():
        if name not in noise:
            raise ValueError(f"no noise model given for group '{name}'")
        values = np.asarray(values, dtype=float).ravel()
        if name in machine.sensors:
            labels = machine.labels(name)
            geometry = {k: np.array(v, copy=True) for k, v in machine.geometry(name).items()}
        else:
            labels = [name] if values.size == 1 else [f"{name}_{i:03d}" for i in range(values.size)]
            geometry = {}
        if values.size != len(labels):
            raise ValueError(f"group '{name}': {values.size} model values for "
                             f"{len(labels)} sensors")
        spec = dict(noise[name])
        sigma_model = spec.pop("sigma_model")
        sigma = sigma_from_model(sigma_model, values, **spec)
        rng = _rng(seed, name)
        draws = rng.standard_normal(values.size)
        measured = values + sigma * draws if add_noise else values.copy()
        dspec = dropout.get(name) if isinstance(dropout, dict) else dropout
        keep = _dropout_mask(dspec, labels, rng)
        gmeta = {"sigma_model": sigma_model, "sigma_params": spec,
                 "synthetic_seed": int(seed), "noise_added": bool(add_noise),
                 "sigma_evaluated_on": "model"}
        if dspec is not None:
            gmeta["dropout"] = [labels[i] for i in np.flatnonzero(~keep)]
        bad_sigma = ~(np.isfinite(sigma) & (sigma > 0))
        if np.any(bad_sigma):
            # a channel with sigma <= 0 (e.g. a relative model on a zero model value) cannot be scored
            gmeta["invalid_sigma"] = [labels[i] for i in np.flatnonzero(bad_sigma)]
            keep = keep & ~bad_sigma
        if name == "mse":
            gmeta["quantity"] = "tan_polarisation_angle"
        gmeta.update(group_meta.get(name, {}))
        ms.groups[name] = ConstraintGroup(name=name, labels=labels, measured=measured,
                                          sigma=sigma, valid=keep, geometry=geometry,
                                          meta=gmeta)
    return ms
