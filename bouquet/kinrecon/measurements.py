"""Frozen per-slice measurement files and static machine bundles.

A :class:`~.interfaces.MeasurementSet` is written to, and read back from, one
JSON file per (device, time window).  The internal names follow the IMAS data
dictionary, so the file reads like a time-collapsed subset of the IDSs it was
taken from:

.. code-block:: text

    {
     "schema": "kinrecon.measurements/1",
     "machine": ..., "time": ..., "time_window": ...,
     "magnetics": {"flux_loop": [{"name", "position": [{"r", "z"}],
                                  "flux": {"data", "data_error_upper",
                                           "time", "validity"}}, ...],
                   "b_field_pol_probe": [...], "ip": [...],
                   "diamagnetic_flux": [...]},
     "pf_active": {"coil": [{"name", "current": {...}}, ...]},
     "thomson_scattering": {"channel": [{"name", "position", "n_e", "t_e"}]},
     "mse": {"channel": [{"name", "active_spatial_resolution":
                          [{"centre", "geometric_coefficients"}],
                          "polarisation_angle": {...}}]},
     "equilibrium": {"time_slice": [{"constraints": {"q": [...]}}]},
     "kinrecon_ext": {...}
    }

Every value array holds ONE sample (the window-reduced value) and every
``time`` array the window centre [s].  ``kinrecon_ext`` holds only what the
data dictionary lacks or cannot hold exactly:

``time_window``     ``[t0, t1]`` [s] of the averaging window
``sigma_models``    group -> name of the uncertainty convention (a readable
                    summary; the authoritative copy is each group's ``meta``)
``groups``          per group, in order: ``labels`` (the authoritative channel
                    order), ``weight``, ``meta``, any geometry the IDS layout
                    does not carry, ``valid`` where the IDS has no validity
                    field (Thomson, coil currents, Ip, diamagnetic flux), and
                    the exact measured values where the IDS stores a
                    different quantity (MSE: tan(gamma), see below)
``mse``             ``er_corrected`` and the uncorrected tan(gamma)
                    (readable summary of the ``mse`` group's ``meta``)
``meta``            :attr:`MeasurementSet.meta`

**Format choice: JSON.**  It is standard-library only (this subpackage must
not depend on h5py, see ``interfaces``), the files are KB-sized, diffable and
hashable, and Python writes floats with ``repr`` so the round trip is exact to
the bit.  Non-finite values are written as the ``NaN`` / ``Infinity`` tokens
Python's ``json`` reads back (not strict JSON).

**MSE.**  The data dictionary stores the polarisation ANGLE
(``mse.channel[:].polarisation_angle``, rad).  EFIT and the k-file store
tan(gamma).  ``meta["quantity"]`` of the MSE group says which one
``measured`` holds: ``"polarisation_angle"`` (the default when absent) is
written to the IDS block as is; for ``"tan_polarisation_angle"`` (what the
adapters produce, to keep the source's own variable and sigma) the IDS block receives
``arctan(tan gamma)`` with a linearized ``sigma / (1 + tan^2 gamma)`` for
interoperability (``d gamma = d tan(gamma) / (1 + tan^2 gamma)``, first
order), and the exact tan(gamma) and sigma go in ``kinrecon_ext``; the reader
takes the exact values.  Units follow the DD: flux loops Wb (total flux, not
per radian), probes T, currents A, Ip A, diamagnetic flux Wb, n_e m^-3,
T_e eV, polarisation angle rad, times s.

Groups not in :data:`~.interfaces.GROUP_IDS_PATHS` are written whole into
``kinrecon_ext.groups`` so that any :class:`MeasurementSet` round trips.

**Machine bundle.**  :class:`MachineBundle` is the static, per-device (or
per-era) sensor description the adapters need and the measurement file does
not repeat: flux-loop positions, probe positions and directions, coil names
(and optionally turns), other sensor geometry, and a reference to the
TokaMaker mesh file.  It is written as a separate JSON file
(:func:`write_machine` / :func:`read_machine`).
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .diagnostics import mse_from_field
from .interfaces import ConstraintGroup, MeasurementSet

SCHEMA = "kinrecon.measurements/1"
MACHINE_SCHEMA = "kinrecon.machine/1"
EXT = "kinrecon_ext"

# IDS validity written for an invalid channel.  IMAS DD 3.41.0, e.g.
# ``magnetics.flux_loop[:].flux.validity``: "0: valid from automated
# processing, 1: valid and certified by the diagnostic RO; -1 means problem
# identified in the data processing ..., -2: invalid data, should not be used".
# Any negative value reads back as invalid.
INVALID = -2

# Where each canonical group lives in the IDS-shaped file.
#   path      -- keys/indices down to the array of channel structures
#   qty       -- the quantity structure inside each channel (None: the
#                channel structure itself holds data/data_error_upper)
#   style     -- "data" (data/data_error_upper/time) or "constraint"
#                (measured/measured_error_upper/time_measurement)
#   name_key  -- the channel's name field
#   validity  -- the quantity has an IDS validity field
#   geom      -- geometry key -> path inside the channel structure
_LAYOUT = {
    "flux_loop": dict(path=("magnetics", "flux_loop"), qty="flux",
                      validity=True,
                      geom={"r": ("position", 0, "r"), "z": ("position", 0, "z")}),
    "vloop": dict(path=("magnetics", "flux_loop"), qty="voltage",
                  validity=True,
                  geom={"r": ("position", 0, "r"), "z": ("position", 0, "z")}),
    "bpol_probe": dict(path=("magnetics", "b_field_pol_probe"), qty="field",
                       validity=True,
                       geom={"r": ("position", "r"), "z": ("position", "z")}),
    "ip": dict(path=("magnetics", "ip"), qty=None, validity=False, geom={}),
    "diamagnetic_flux": dict(path=("magnetics", "diamagnetic_flux"), qty=None,
                             validity=False, geom={}),
    "pf_current": dict(path=("pf_active", "coil"), qty="current",
                       validity=False, geom={}),
    "mse": dict(path=("mse", "channel"), qty="polarisation_angle",
                validity=True,
                geom={"r": ("active_spatial_resolution", 0, "centre", "r"),
                      "z": ("active_spatial_resolution", 0, "centre", "z"),
                      "coefficients": ("active_spatial_resolution", 0,
                                       "geometric_coefficients")}),
    "thomson_ne": dict(path=("thomson_scattering", "channel"), qty="n_e",
                       validity=False,
                       geom={"r": ("position", "r"), "z": ("position", "z")}),
    "thomson_te": dict(path=("thomson_scattering", "channel"), qty="t_e",
                       validity=False,
                       geom={"r": ("position", "r"), "z": ("position", "z")}),
    "q": dict(path=("equilibrium", "time_slice", 0, "constraints", "q"),
              qty=None, style="constraint", name_key="source",
              validity=False, geom={}),
}


# ---------------------------------------------------------------------------
# tagged encoding of numpy values inside free-form dicts (meta, geometry)
# ---------------------------------------------------------------------------

def _encode(obj):
    """JSON-ready copy of ``obj``; numpy arrays become tagged dicts."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "US":
            data = obj.astype(str).tolist()
        elif obj.dtype.kind == "O":
            raise TypeError("object arrays cannot be written to a measurement file")
        else:
            data = obj.tolist()
        return {"__ndarray__": data, "dtype": obj.dtype.str if obj.dtype.kind not in "US"
                else "str", "shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        for k in obj:
            if not isinstance(k, str):
                raise TypeError(f"dict keys must be strings, got {k!r}")
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"cannot write a {type(obj).__name__} to a measurement file")


def _decode(obj):
    """Inverse of :func:`_encode`."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            dtype = str if obj["dtype"] == "str" else np.dtype(obj["dtype"])
            arr = np.array(obj["__ndarray__"], dtype=dtype)
            return arr.reshape(obj["shape"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _floats(a):
    return [float(x) for x in np.asarray(a, dtype=float).ravel()]


# ---------------------------------------------------------------------------
# conventions shared by the adapters
# ---------------------------------------------------------------------------

def probe_direction(poloidal_angle):
    """Unit vector ``(direction_r, direction_z)`` of a poloidal-field probe.

    IMAS DD 3.41.0, ``magnetics.b_field_pol_probe[:].poloidal_angle``: "Angle
    of the sensor normal vector ... with respect to horizontal plane
    (clockwise theta-like angle). Zero if sensor normal vector fully in the
    horizontal plane and oriented towards increasing major radius."  Clockwise
    in the (R, Z) plane with Z up, so

    .. code-block:: text

        direction_r = cos(angle),   direction_z = -sin(angle)

    (an angle of -pi/2 points along +Z).  The probe measures
    ``B_R direction_r + B_Z direction_z``.
    """
    a = np.asarray(poloidal_angle, dtype=float)
    return np.cos(a), -np.sin(a)


def probe_angle(direction_r, direction_z):
    """Inverse of :func:`probe_direction`, in [0, 2 pi)."""
    return np.mod(np.arctan2(-np.asarray(direction_z, float),
                             np.asarray(direction_r, float)), 2.0 * np.pi)


def efit_to_imas_coefficients(a1, a2, a3, a4, a5=None, a6=None, a7=None, a8=None):
    """Map EFIT's MSE coefficients onto the 9 IMAS geometric coefficients.

    EFIT k-file namelist INS, ``AA1GAM`` .. ``AA7GAM`` (``A8``, the B_R
    numerator, is not a k-file variable and is zero for standard chords):

    .. code-block:: text

        tan(gamma) = (A1 B_Z + A8 B_R + A5 E_R)
                   / (A2 B_phi + A3 B_R + A4 B_Z + A6 E_Z + A7 E_R)

    (the A1..A7 form as in ``bouquet/mse.py``; cf. B. W. Rice et al., Phys.
    Rev. Lett. 79, 2694 (1997) for the E-field terms).  IMAS DD 3.41.0,
    ``mse.channel[:].active_spatial_resolution[:].geometric_coefficients``:
    "coefficients of BZ, BR, Bphi, ER (numerator of the MSE angle
    expression); coefficients of BZ, BR, Bphi, ER, EZ (denominator)".  Hence

    .. code-block:: text

        c = [A1, A8, 0, A5,  A4, A3, A2, A7, A6]

    Absent coefficients are zero.  Returns an (n, 9) array.
    """
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    z = np.zeros_like(a1)

    def _c(a):
        return z if a is None else np.broadcast_to(np.asarray(a, float), a1.shape)

    return np.stack([a1, _c(a8), z, _c(a5), _c(a4), _c(a3), _c(a2), _c(a7), _c(a6)],
                    axis=-1)


def imas_to_efit_coefficients(c):
    """Inverse of :func:`efit_to_imas_coefficients`: dict ``A1`` .. ``A8``.

    Refuses a non-zero B_phi numerator coefficient (``c[:, 2]``), which the
    EFIT form has no slot for.
    """
    c = np.atleast_2d(np.asarray(c, dtype=float))
    if c.shape[-1] != 9:
        raise ValueError(f"expected 9 coefficients per channel, got {c.shape[-1]}")
    if np.any(c[:, 2] != 0.0):
        raise ValueError("non-zero B_phi numerator coefficient has no EFIT equivalent")
    return {"A1": c[:, 0], "A8": c[:, 1], "A5": c[:, 3], "A4": c[:, 4],
            "A3": c[:, 5], "A2": c[:, 6], "A7": c[:, 7], "A6": c[:, 8]}


def tan_gamma_imas(c, b_r, b_z, b_phi, e_r=0.0, e_z=0.0):
    """tan(gamma) from the 9 IMAS coefficients (DD order, see
    :func:`efit_to_imas_coefficients`) and local fields.

    A thin wrapper: the one MSE formula in kinrecon is
    :func:`bouquet.kinrecon.diagnostics.mse_from_field` (see its docstring for
    the form and source); fields are scalars or per-channel arrays.
    """
    c = np.atleast_2d(np.asarray(c, dtype=float))
    n = c.shape[0]
    B = np.column_stack([np.broadcast_to(np.asarray(x, dtype=float), (n,))
                         for x in (b_r, b_phi, b_z)])
    return mse_from_field(B, c, e_r=e_r, e_z=e_z, quantity="tan_polarisation_angle")


# ---------------------------------------------------------------------------
# nested get/set in the IDS-shaped dict
# ---------------------------------------------------------------------------

def _descend(doc, path, create):
    node = doc
    for i, key in enumerate(path):
        nxt = path[i + 1] if i + 1 < len(path) else None
        if isinstance(key, int):
            while create and len(node) <= key:
                node.append({} if not isinstance(nxt, int) else [])
            node = node[key]
        else:
            if key not in node:
                if not create:
                    raise KeyError(key)
                node[key] = [] if (nxt is None or isinstance(nxt, int)) else {}
            node = node[key]
    return node


def _get(node, path):
    for key in path:
        node = node[key]
    return node


def _set(node, path, value):
    for i, key in enumerate(path[:-1]):
        nxt = path[i + 1]
        if isinstance(key, int):
            while len(node) <= key:
                node.append({} if not isinstance(nxt, int) else [])
            node = node[key]
        else:
            node = node.setdefault(key, [] if isinstance(nxt, int) else {})
    node[path[-1]] = value


# ---------------------------------------------------------------------------
# MeasurementSet <-> file
# ---------------------------------------------------------------------------

def to_dict(ms):
    """The IDS-shaped, JSON-ready dict for a :class:`MeasurementSet`."""
    doc = {"schema": SCHEMA, "machine": ms.machine, "time": float(ms.time),
           "time_window": float(ms.time_window)}
    ext = {"time_window": [float(ms.time) - 0.5 * float(ms.time_window),
                           float(ms.time) + 0.5 * float(ms.time_window)],
           "sigma_models": {}, "groups": {}, "meta": _encode(dict(ms.meta))}
    for name, g in ms.groups.items():
        if len(set(g.labels)) != len(g.labels):
            raise ValueError(f"group '{name}': channel labels must be unique")
        gx = {"labels": [str(s) for s in g.labels], "weight": _floats(g.weight),
              "meta": _encode(dict(g.meta))}
        if "sigma_model" in g.meta:
            ext["sigma_models"][name] = g.meta["sigma_model"]
        lay = _LAYOUT.get(name)
        if lay is None:
            gx.update(ids_path=None, measured=_floats(g.measured),
                      sigma=_floats(g.sigma), valid=[bool(v) for v in g.valid],
                      geometry=_encode(dict(g.geometry)))
            ext["groups"][name] = gx
            continue
        if name == "mse" and g.meta.get("quantity") == "tan_polarisation_angle":
            tan_g, sig_t = g.measured, g.sigma
            value = np.arctan(tan_g)
            error = sig_t / (1.0 + tan_g ** 2)
            gx.update(measured=_floats(tan_g), sigma=_floats(sig_t))
        else:
            value, error = g.measured, g.sigma
        entries = _descend(doc, lay["path"], create=True)
        name_key = lay.get("name_key", "name")
        index = {e.get(name_key): e for e in entries}
        extra_geom = {}
        for i, label in enumerate(g.labels):
            label = str(label)
            entry = index.get(label)
            if entry is None:
                entry = {name_key: label}
                entries.append(entry)
                index[label] = entry
            if lay.get("style") == "constraint":
                entry.update(measured=float(value[i]),
                             measured_error_upper=float(error[i]),
                             time_measurement=float(ms.time))
            else:
                blk = {"data": [float(value[i])],
                       "data_error_upper": [float(error[i])],
                       "time": [float(ms.time)]}
                if lay["validity"]:
                    blk["validity"] = 0 if g.valid[i] else INVALID
                if lay["qty"] is None:
                    entry.update(blk)
                else:
                    entry[lay["qty"]] = blk
            for key, gpath in lay["geom"].items():
                if key not in g.geometry:
                    continue
                v = np.asarray(g.geometry[key])[i]
                v = _floats(v) if np.ndim(v) else float(v)
                try:
                    old = _get(entry, gpath)
                except (KeyError, IndexError, TypeError):
                    old = None
                if old is not None and old != v:
                    raise ValueError(f"group '{name}', channel '{label}': {key} "
                                     "differs from the value another group wrote "
                                     "for the same channel")
                _set(entry, gpath, v)
        for key, arr in g.geometry.items():
            if key not in lay["geom"]:
                extra_geom[key] = np.asarray(arr)
        if name == "bpol_probe" and {"direction_r", "direction_z"} <= set(g.geometry):
            # informational copy in the DD convention (see probe_direction)
            ang = probe_angle(np.asarray(g.geometry["direction_r"], float),
                              np.asarray(g.geometry["direction_z"], float))
            for i, label in enumerate(g.labels):
                index[str(label)]["poloidal_angle"] = float(ang[i])
        if extra_geom:
            gx["geometry"] = _encode(extra_geom)
        if not lay["validity"]:
            gx["valid"] = [bool(v) for v in g.valid]
        gx["ids_path"] = _ids_path_str(lay)
        ext["groups"][name] = gx
    if "mse" in ms.groups:
        m = ms.groups["mse"].meta
        ext["mse"] = {"er_corrected": m.get("er_corrected"),
                      "tan_gamma_uncorrected": _encode(m.get("tan_gamma_uncorrected"))}
    doc[EXT] = ext
    return doc


def _ids_path_str(lay):
    parts = []
    for p in lay["path"]:
        if isinstance(p, int):
            parts[-1] += f"[{p}]"
        else:
            parts.append(p)
    s = ".".join(parts) + "[:]"
    return s + ("." + lay["qty"] if lay["qty"] else "")


def from_dict(doc):
    """Inverse of :func:`to_dict`."""
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document (schema={doc.get('schema')!r})")
    ext = doc[EXT]
    ms = MeasurementSet(machine=doc["machine"], time=float(doc["time"]),
                        time_window=float(doc["time_window"]),
                        meta=_decode(ext.get("meta", {})))
    for name, gx in ext["groups"].items():
        labels = list(gx["labels"])
        weight = np.array(gx["weight"], dtype=float)
        meta = _decode(gx.get("meta", {}))
        geometry = _decode(gx.get("geometry", {}))
        lay = _LAYOUT.get(name)
        if lay is None:
            ms.groups[name] = ConstraintGroup(
                name=name, labels=labels, measured=np.array(gx["measured"], float),
                sigma=np.array(gx["sigma"], float), valid=np.array(gx["valid"], bool),
                weight=weight, geometry=geometry, meta=meta)
            continue
        entries = _get(doc, lay["path"])
        name_key = lay.get("name_key", "name")
        index = {e.get(name_key): e for e in entries}
        n = len(labels)
        value, error = np.empty(n), np.empty(n)
        valid = np.ones(n, dtype=bool)
        geo = {k: [None] * n for k in lay["geom"]}
        for i, label in enumerate(labels):
            entry = index[label]
            if lay.get("style") == "constraint":
                value[i] = entry["measured"]
                error[i] = entry["measured_error_upper"]
            else:
                blk = entry if lay["qty"] is None else entry[lay["qty"]]
                value[i] = blk["data"][0]
                error[i] = blk["data_error_upper"][0]
                if lay["validity"]:
                    valid[i] = blk["validity"] >= 0
            for key, gpath in lay["geom"].items():
                try:
                    geo[key][i] = _get(entry, gpath)
                except (KeyError, IndexError, TypeError):
                    geo[key][i] = None
        for key, vals in geo.items():
            present = [v is not None for v in vals]
            if all(present):
                geometry[key] = np.array(vals, dtype=float)
            elif any(present):
                raise ValueError(f"group '{name}': geometry '{key}' missing on some channels")
        if not lay["validity"]:
            valid = np.array(gx["valid"], dtype=bool)
        if "measured" in gx:
            value = np.array(gx["measured"], dtype=float)
            error = np.array(gx["sigma"], dtype=float)
        ms.groups[name] = ConstraintGroup(name=name, labels=labels, measured=value,
                                          sigma=error, valid=valid, weight=weight,
                                          geometry=geometry, meta=meta)
    return ms


def write_measurements(path, ms):
    """Write ``ms`` to the JSON measurement file ``path``."""
    doc = to_dict(ms)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def read_measurements(path):
    """Read a JSON measurement file written by :func:`write_measurements`."""
    with open(path) as f:
        return from_dict(json.load(f))


# ---------------------------------------------------------------------------
# machine bundle
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class MachineBundle:
    """Static sensor description of one device (or one era of it).

    ``sensors`` maps a canonical group name (see
    :data:`~.interfaces.GROUP_IDS_PATHS`) to ``{"labels": [...], <geometry>}``
    with the same per-channel geometry keys a :class:`ConstraintGroup` uses:
    ``r``, ``z``; ``direction_r``, ``direction_z`` for poloidal-field probes;
    ``coefficients`` (n, 9) for MSE; ``turns`` (optional) for coils.
    Channels are listed in the order the device's own files use, so that
    adapters reading unlabeled arrays (e.g. EFIT k-files) can name them.

    ``channel_maps`` optionally names the channels of a source array whose
    order differs from ``sensors`` (source-array name -> labels), e.g. coil
    currents split over two arrays in a device file.

    ``mesh_file`` is the TokaMaker mesh (relative paths resolve against the
    bundle file's directory, see :meth:`mesh_path`); ``mesh_sha256`` pins it.
    """

    machine: str
    sensors: Dict[str, Dict[str, object]] = field(default_factory=dict)
    mesh_file: str = None
    mesh_sha256: str = None
    channel_maps: Dict[str, List[str]] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)
    source_path: str = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name, s in self.sensors.items():
            if "labels" not in s:
                raise ValueError(f"sensor group '{name}' has no labels")
            labels = [str(x) for x in s["labels"]]
            if len(set(labels)) != len(labels):
                raise ValueError(f"sensor group '{name}': labels must be unique")
            s["labels"] = labels
            for key, arr in list(s.items()):
                if key == "labels":
                    continue
                arr = np.asarray(arr)
                if arr.shape[:1] != (len(labels),):
                    raise ValueError(f"sensor group '{name}': '{key}' has shape "
                                     f"{arr.shape}, expected ({len(labels)}, ...)")
                s[key] = arr

    def labels(self, group):
        """Channel labels of one sensor group."""
        return list(self.sensors[group]["labels"])

    def geometry(self, group, index=None):
        """Per-channel geometry arrays of one sensor group (labels excluded),
        optionally restricted to ``index``."""
        geo = {k: v for k, v in self.sensors[group].items() if k != "labels"}
        if index is not None:
            geo = {k: v[index] for k, v in geo.items()}
        return geo

    def mesh_path(self):
        """Absolute path of the mesh file, or None."""
        if self.mesh_file is None:
            return None
        if os.path.isabs(self.mesh_file) or self.source_path is None:
            return self.mesh_file
        return os.path.join(os.path.dirname(os.path.abspath(self.source_path)),
                            self.mesh_file)


def file_sha256(path):
    """SHA-256 hex digest of a file (for pinning the mesh in a bundle)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_machine(path, bundle):
    """Write a :class:`MachineBundle` to JSON."""
    doc = {"schema": MACHINE_SCHEMA, "machine": bundle.machine,
           "mesh_file": bundle.mesh_file, "mesh_sha256": bundle.mesh_sha256,
           "sensors": {g: _encode(dict(s)) for g, s in bundle.sensors.items()},
           "channel_maps": {k: list(v) for k, v in bundle.channel_maps.items()},
           "meta": _encode(dict(bundle.meta))}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def read_machine(path, verify_mesh=False):
    """Read a :class:`MachineBundle`; with ``verify_mesh`` the mesh file's
    SHA-256 must match the recorded one."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema") != MACHINE_SCHEMA:
        raise ValueError(f"not a {MACHINE_SCHEMA} document (schema={doc.get('schema')!r})")
    bundle = MachineBundle(machine=doc["machine"],
                           sensors={g: _decode(s) for g, s in doc["sensors"].items()},
                           mesh_file=doc.get("mesh_file"),
                           mesh_sha256=doc.get("mesh_sha256"),
                           channel_maps=doc.get("channel_maps", {}),
                           meta=_decode(doc.get("meta", {})),
                           source_path=path)
    if verify_mesh:
        if bundle.mesh_sha256 is None:
            raise ValueError("bundle records no mesh_sha256 to verify against")
        got = file_sha256(bundle.mesh_path())
        if got != bundle.mesh_sha256:
            raise ValueError(f"mesh file {bundle.mesh_path()} does not match the "
                             "bundle's mesh_sha256")
    return bundle
