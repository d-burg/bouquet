"""Measurement sets from an IMAS-style JSON dict, reduced to one time window.

Reads the dense IDS JSON that OMAS and FUSE (IMASdd.jl) write, already loaded
into a Python dict (``json.load``), and averages each channel over the window
``[time - time_window / 2, time + time_window / 2]`` (seconds, inclusive).
Machine-agnostic: only IMAS data-dictionary paths are used.

Groups and their sources (IMAS DD 3.41.0 paths; units from the DD):

=================  ==============================================================
``flux_loop``      ``magnetics.flux_loop[:].flux`` [Wb], ``position[0].{r,z}``
``vloop``          ``magnetics.flux_loop[:].voltage`` [V]
``bpol_probe``     ``magnetics.b_field_pol_probe[:].field`` [T] (DD3 legacy
                   ``bpol_probe`` also accepted), ``position.{r,z}``,
                   ``poloidal_angle`` -> direction (``measurements.probe_direction``)
``ip``             ``magnetics.ip[0]`` [A]
``diamagnetic_flux`` ``magnetics.diamagnetic_flux[0]`` [Wb]
``pf_current``     ``pf_active.coil[:].current`` [A]; ``turns`` = sum of
                   ``element[:].turns_with_sign``
``thomson_ne/te``  ``thomson_scattering.channel[:].n_e`` [m^-3] / ``.t_e`` [eV],
                   ``position.{r,z}``
``mse``            ``mse.channel[:].polarisation_angle`` [rad],
                   ``active_spatial_resolution[:].{centre,geometric_coefficients}``
=================  ==============================================================

**Sample selection.**  A sample counts when it lies in the window, its value
and ``data_error_upper`` are finite, ``0 < data_error_upper < 1e29`` (OMAS
marks excluded channels with sigma = 1e30) and ``validity_timed`` (when
present) is >= 0.  A channel is valid when its ``validity`` (when present) is
>= 0 (DD: negative = problem / invalid) and at least ``min_samples`` samples
count.  The DD has no validity field for Thomson, coil currents, Ip or the
diamagnetic flux; for those only the per-sample rules apply.  Invalid
channels carry NaN value and sigma.

**Window reduction** (the name is recorded in ``meta["sigma_model"]`` as
``"data_error_upper+<reduction>"``):

``window_mean``      value = mean of samples; sigma = mean of their
                     ``data_error_upper``.  Not divided by sqrt(N): the
                     source sigma is treated as systematic (calibration /
                     convention), not as independent per-sample noise.
``window_sem``       sigma = sqrt(mean(err^2) / N): independent samples.
``window_spread``    sigma = sqrt(mean(err)^2 + var(values)): adds the
                     in-window variation (population variance) in quadrature.

**MSE.**  Per sample, tan(gamma) = tan(angle) and sigma_tan =
sigma_angle (1 + tan^2) (first order); the reduction is applied to those.  The
DD has no flag for "already E_r-corrected", so ``mse_er_corrected`` must be
stated by the caller; building MSE without it is refused.  Coefficients come
from the ``active_spatial_resolution`` entry nearest the window centre when
those entries carry a ``time``, else from entry 0.
"""

import numpy as np

from ..interfaces import ConstraintGroup, MeasurementSet
from ..measurements import MachineBundle, probe_direction

REDUCTIONS = ("window_mean", "window_sem", "window_spread")

#: OMAS's sigma for a channel it excludes; anything at or above is "no data".
_SIGMA_EXCLUDED = 1e29


def _arr(blk, key):
    v = blk.get(key)
    return None if v is None else np.asarray(v, dtype=float)


def _select(blk, t0, t1, transform=None):
    """(values, errors) of the usable samples of one quantity block."""
    t = _arr(blk, "time")
    data = _arr(blk, "data")
    err = _arr(blk, "data_error_upper")
    if data is None or t is None or data.size == 0:
        return np.empty(0), np.empty(0)
    n = min(t.size, data.size)
    t, data = t[:n], data[:n]
    err = np.full(n, np.nan) if err is None else err[:n]
    ok = (t >= t0) & (t <= t1) & np.isfinite(data) & np.isfinite(err)
    ok &= (err > 0.0) & (err < _SIGMA_EXCLUDED)
    vt = _arr(blk, "validity_timed")
    if vt is not None and vt.size >= n:
        ok &= vt[:n] >= 0
    data, err = data[ok], err[ok]
    if transform is not None:
        data, err = transform(data, err)
    return data, err


def _reduce(vals, errs, reduction):
    if reduction == "window_mean":
        return vals.mean(), errs.mean()
    if reduction == "window_sem":
        return vals.mean(), np.sqrt(np.mean(errs ** 2) / vals.size)
    if reduction == "window_spread":
        return vals.mean(), np.sqrt(errs.mean() ** 2 + vals.var())
    raise ValueError(f"unknown window reduction {reduction!r}; known: {REDUCTIONS}")


def _tan_transform(angle, err):
    tg = np.tan(angle)
    return tg, err * (1.0 + tg ** 2)


def _build(name, entries, qty, t0, t1, reduction, min_samples, geom_fn,
           name_key="name", transform=None, extra_meta=None):
    labels, value, sigma, valid, nsamp = [], [], [], [], []
    geometry = {}
    for i, e in enumerate(entries):
        blk = e if qty is None else e.get(qty)
        if blk is None:
            continue
        labels.append(str(e.get(name_key) or e.get("identifier") or f"{name}_{i:03d}"))
        vals, errs = _select(blk, t0, t1, transform)
        ok = vals.size >= max(1, min_samples)
        v_ent = blk.get("validity") if isinstance(blk, dict) else None
        if v_ent is not None and np.ndim(v_ent) == 0 and v_ent < 0:
            ok = False
        if ok:
            v, s = _reduce(vals, errs, reduction)
        else:
            v, s = np.nan, np.nan
        value.append(v)
        sigma.append(s)
        valid.append(ok)
        nsamp.append(int(vals.size))
        for k, g in geom_fn(e).items():
            geometry.setdefault(k, []).append(g)
    if not labels:
        return None
    if len(set(labels)) != len(labels):
        raise ValueError(f"group '{name}': channel names are not unique")
    meta = {"sigma_model": f"data_error_upper+{reduction}",
            "sigma_params": {"min_samples": int(min_samples)},
            "n_samples": np.array(nsamp), "source": "imas_json"}
    meta.update(extra_meta or {})
    geometry = {k: np.array(v, dtype=float) for k, v in geometry.items()}
    return ConstraintGroup(name=name, labels=labels, measured=np.array(value),
                           sigma=np.array(sigma), valid=np.array(valid, bool),
                           geometry=geometry, meta=meta)


def _rz(pos):
    if isinstance(pos, list):
        pos = pos[0] if pos else {}
    return {"r": float(np.mean(pos.get("r", np.nan))),
            "z": float(np.mean(pos.get("z", np.nan)))}


def _probe_geom(e):
    g = _rz(e.get("position", {}))
    dr, dz = probe_direction(e.get("poloidal_angle", np.nan))
    g.update(direction_r=float(dr), direction_z=float(dz))
    return g


def _coil_geom(e):
    turns = [el.get("turns_with_sign", np.nan) for el in e.get("element", [])]
    return {"turns": float(np.sum(turns)) if turns else np.nan}


def _mse_geom(time):
    def fn(e):
        res = e.get("active_spatial_resolution", [])
        if not res:
            return {"r": np.nan, "z": np.nan, "coefficients": [np.nan] * 9}
        times = [r.get("time") for r in res]
        k = 0
        if all(t is not None for t in times) and len(res) > 1:
            k = int(np.argmin(np.abs(np.asarray(times, float) - time)))
        r = res[k]
        c = r.get("centre", {})
        return {"r": float(c.get("r", np.nan)), "z": float(c.get("z", np.nan)),
                "coefficients": [float(x) for x in r.get("geometric_coefficients",
                                                          [np.nan] * 9)]}
    return fn


ALL_GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current",
              "thomson_ne", "thomson_te", "mse", "vloop")


def measurements_from_imas(ids, time, time_window, machine="", groups=None,
                           reduction="window_mean", min_samples=1,
                           mse_er_corrected=None, meta=None):
    """Build a :class:`MeasurementSet` from an IMAS-style dict.

    Parameters
    ----------
    ids : dict
        Top-level dict keyed by IDS name (``magnetics``, ``pf_active``, ...).
    time, time_window : float
        Window centre and full width [s].
    groups : sequence, optional
        Groups to build (default: every group in :data:`ALL_GROUPS` whose
        source is present).  A requested group whose source is absent raises.
    reduction : str
        Window reduction, one of :data:`REDUCTIONS` (recorded by name).
    mse_er_corrected : bool, optional
        Whether the MSE angles are already E_r-corrected.  Required when MSE
        is built.
    """
    t0, t1 = time - 0.5 * time_window, time + 0.5 * time_window
    mag = ids.get("magnetics", {})
    probes = mag.get("b_field_pol_probe") or mag.get("bpol_probe") or []
    sources = {
        "flux_loop": (mag.get("flux_loop", []), "flux",
                      lambda e: _rz(e.get("position", {}))),
        "vloop": ([e for e in mag.get("flux_loop", []) if e.get("voltage")], "voltage",
                  lambda e: _rz(e.get("position", {}))),
        "bpol_probe": (probes, "field", _probe_geom),
        "ip": (mag.get("ip", [])[:1], None, lambda e: {}),
        "diamagnetic_flux": (mag.get("diamagnetic_flux", [])[:1], None, lambda e: {}),
        "pf_current": (ids.get("pf_active", {}).get("coil", []), "current", _coil_geom),
        "thomson_ne": (ids.get("thomson_scattering", {}).get("channel", []), "n_e",
                       lambda e: _rz(e.get("position", {}))),
        "thomson_te": (ids.get("thomson_scattering", {}).get("channel", []), "t_e",
                       lambda e: _rz(e.get("position", {}))),
        "mse": (ids.get("mse", {}).get("channel", []), "polarisation_angle",
                _mse_geom(time)),
    }
    requested = groups is not None
    groups = list(groups) if requested else list(ALL_GROUPS)
    ms = MeasurementSet(machine=machine, time=float(time), time_window=float(time_window),
                        meta=dict(meta or {}))
    ms.meta.setdefault("source", "imas_json")
    ms.meta["time_window_s"] = [t0, t1]
    ms.meta["window_reduction"] = reduction
    for name in groups:
        if name not in sources:
            raise ValueError(f"unknown group {name!r}")
        entries, qty, geom_fn = sources[name]
        if not entries:
            if requested:
                raise ValueError(f"group '{name}' requested but absent from the IDS dict")
            continue
        transform, extra = None, None
        if name == "mse":
            if mse_er_corrected is None:
                if requested:
                    raise ValueError("MSE present: state mse_er_corrected=True/False "
                                     "(the data dictionary does not record it)")
                continue
            transform = _tan_transform
            extra = {"quantity": "tan_polarisation_angle", "er_corrected": bool(mse_er_corrected)}
        g = _build(name, entries, qty, t0, t1, reduction, min_samples, geom_fn,
                   transform=transform, extra_meta=extra)
        if g is not None:
            ms.groups[name] = g
    return ms


def machine_from_imas(ids, machine, mesh_file=None, mesh_sha256=None):
    """A :class:`~bouquet.kinrecon.measurements.MachineBundle` holding the
    sensor geometry found in an IMAS-style dict (time-independent parts)."""
    mag = ids.get("magnetics", {})
    probes = mag.get("b_field_pol_probe") or mag.get("bpol_probe") or []
    sensors = {}

    def add(name, entries, fn):
        if not entries:
            return
        rows = [fn(e) for e in entries]
        s = {"labels": [str(e.get("name") or e.get("identifier")) for e in entries]}
        for k in rows[0]:
            s[k] = np.array([r[k] for r in rows], dtype=float)
        sensors[name] = s

    add("flux_loop", mag.get("flux_loop", []), lambda e: _rz(e.get("position", {})))
    add("bpol_probe", probes, _probe_geom)
    add("pf_current", ids.get("pf_active", {}).get("coil", []), _coil_geom)
    ts = ids.get("thomson_scattering", {}).get("channel", [])
    add("thomson_ne", ts, lambda e: _rz(e.get("position", {})))
    add("thomson_te", ts, lambda e: _rz(e.get("position", {})))
    add("mse", ids.get("mse", {}).get("channel", []), _mse_geom(0.0))
    return MachineBundle(machine=machine, sensors=sensors, mesh_file=mesh_file,
                         mesh_sha256=mesh_sha256,
                         meta={"source": "imas_json geometry"})
