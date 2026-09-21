"""Canonical HDF5 archive schema -- the single source of truth for the layout.

Schema v2 (clean break, no legacy readers):

  * profile datasets carry **bare names** (``j_phi``, ``n_e``, …) with the unit
    in ``ds.attrs["units"]`` -- not embedded in the dataset name (F16);
  * the raw g-file / p-file bytes are stored under the fixed names ``eqdsk`` /
    ``pfile`` in each group (draw and baseline), not ``{header}_{scan}_{count}``
    (F11) -- the group path already encodes the coordinates;
  * coil currents are one representation everywhere: a ``coil_currents`` values
    dataset + a ``coil_names`` string dataset (F15);
  * the layout is always ``scan/<key>/`` (F12).

``write_profile`` / ``read_profile`` are the only places that touch the
name↔unit mapping, so ad-hoc ``h5py`` users get self-describing datasets and
the package never hard-codes a bracketed string.
"""

from __future__ import annotations

SCHEMA_VERSION = 2

# Bare dataset name -> unit string (empty = dimensionless).
PROFILE_UNITS = {
    "psi_N": "",
    "psi_N_kinetic": "",
    "j_phi": "A m^-2",
    "j_BS": "A m^-2",
    "j_BS,edge": "A m^-2",
    "j_inductive": "A m^-2",
    "n_e": "m^-3",
    "T_e": "eV",
    "n_i": "m^-3",
    "T_i": "eV",
    "w_ExB": "rad/s",
    "pressure": "Pa",
    "pressure_thermal": "Pa",
    "Zeff": "",
    "z_fast": "m^-3",
    "z2_fast": "m^-3",
    "sigma_ne": "m^-3",
    "sigma_te": "eV",
    "sigma_ni": "m^-3",
    "sigma_ti": "eV",
    "sigma_jphi": "A m^-2",
    "coil_currents": "A",
}

# Fixed in-group dataset names for the opaque byte blobs (F11).
EQDSK_DS = "eqdsk"
PFILE_DS = "pfile"
COIL_VALUES_DS = "coil_currents"
COIL_NAMES_DS = "coil_names"

# Live-equilibrium flux-surface-average block (optional per-draw subgroup),
# captured from the converged TokaMaker equilibrium at generate time to enable
# an exact toroidal<->parallel current conversion at IMAS export. All on the
# eq_fsa psi_N grid. See physics.capture_equilibrium_fsa.
EQ_FSA_GROUP = "eq_fsa"
EQ_FSA_UNITS = {
    "psi_N": "",
    "F": "T m",             # R*B_phi
    "avg_inv_R": "m^-1",    # <1/R>
    "avg_inv_R2": "m^-2",   # <1/R^2> (exact quadrature; may be absent)
    "avg_B2": "T^2",        # <B^2>
    "q": "",
    "dV_dpsi": "m^3 Wb^-1",
    "f_trap": "",           # trapped fraction f_c
    "B_avg": "T",           # <|B|>
}


def is_binary_profile_source(data: bytes) -> bool:
    """True if profile-source bytes are a binary container (an IDA ``.cdf`` --
    netCDF4/HDF5 or classic netCDF3), not an Osborne text p-file.

    Drives the per-draw ``pfile`` storage policy: a TEXT p-file is rewritten
    per draw with that draw's perturbed kinetics (real per-draw data, stored),
    while a binary IDA source cannot be draw-perturbed and would only be
    duplicated verbatim -- with the new IDA-database files (~190 MB) that
    bloated a 20-draw archive to ~4 GB. Binary sources are therefore stored
    ONCE, in ``_baseline``; per-draw readers fall back there.
    """
    head = bytes(data[:8])
    return head.startswith(b"\x89HDF") or head.startswith(b"CDF")


def find_bytes_dataset(grp, kind=EQDSK_DS):
    """Name of ``grp``'s eqdsk/pfile byte blob, or ``None`` if absent.

    The single lookup used by every reader (archive, filtering, plotting):
    the schema-v2 fixed name (``eqdsk`` / ``pfile``) first, then a
    ``.{kind}``-suffix scan so pre-v2 archives with embedded
    ``{header}_{scan}_{count}.eqdsk`` names still resolve.
    """
    import h5py
    if kind in grp and isinstance(grp[kind], h5py.Dataset):
        return kind
    suffix = f".{kind}"
    for name in grp:
        if name.endswith(suffix) and isinstance(grp[name], h5py.Dataset):
            return name
    return None


def write_profile(grp, name, data):
    """Create dataset ``name`` in ``grp`` and stamp its unit attr (if known)."""
    import numpy as np
    ds = grp.create_dataset(name, data=np.asarray(data, dtype=np.float64))
    unit = PROFILE_UNITS.get(name)
    if unit:
        ds.attrs["units"] = unit
    return ds


def read_profile(grp, name):
    """Read dataset ``name`` from ``grp`` as a float ndarray (``None`` if absent)."""
    import numpy as np
    if name not in grp:
        return None
    return np.array(grp[name], dtype=float)
