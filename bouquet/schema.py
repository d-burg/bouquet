"""Canonical HDF5 archive schema -- the single source of truth for the layout.

Schema v3 (additive over v2; every v2 reader keeps working):

  * the **self-consistent bootstrap record** (``jbs_loop`` block): a draw
    group -- and the ``_baseline`` group -- whose bootstrap came from the
    self-consistent loop (``GenerationConfig.jbs_self_consistent``, the
    default) carries the attrs :data:`JBS_LOOP_ATTRS`: ``jbs_converged``
    (bool), ``jbs_n_passes`` (int) and ``jbs_loop_json`` (the full loop
    record: residual histories, relaxation, tolerances, evaluator version,
    OFT build).  A group WITHOUT them carries a frozen (legacy
    ``solve_with_bootstrap``) bootstrap: every v2 archive, and a v3 run made
    with ``jbs_self_consistent=False``.  Read it with :func:`read_jbs_loop`
    (or ``DrawView.jbs_loop`` / ``ScanView.baseline_jbs_loop``).

  No v2 dataset or attr changed name, unit or meaning, so there is nothing to
  migrate: a v2 archive reads as a v3 archive whose bootstrap is frozen
  everywhere.  See ``docs/archive-schema.md`` ("v2 -> v3").

Schema v2 (clean break, no legacy readers):

  * profile datasets carry **bare names** (``jphi_total``, ``n_e``, …) with the unit
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

SCHEMA_VERSION = 4

#: v4: current components under OpenFUSIONToolkit's boot_profs names.  An older archive's dataset
#: name -> its v4 name; :func:`stored_name` reads either, :func:`migrate_names` renames in place.
LEGACY_NAMES = {
    "j_phi": "jphi_total", "j_inductive": "jphi_ind", "j_BS": "jphi_bs", "j_BS,edge": "jphi_bs,edge",
    "j_pressure": "jphi_pd", "j_NBI": "jphi_nbi", "j_RF": "jphi_rf", "j_other": "jphi_other",
    "j_sawteeth": "jphi_saw", "swb_j_saw": "jphi_saw_out", "jBS_diff": "jphi_bs_diff",
    "jB_parallel": "jdotB", "jB_inductive": "jdotB_ind", "jB_BS": "jdotB_bs", "jB_NBI": "jdotB_nbi",
    "jB_RF": "jdotB_rf",
}
_V3_NAME = {new: old for old, new in LEGACY_NAMES.items()}


def stored_name(grp, name):
    """The name dataset (or subgroup) ``name`` is stored under in ``grp``: ``name``, or its pre-v4
    spelling in an older archive; ``name`` when neither is there."""
    old = _V3_NAME.get(name)
    return old if (name not in grp and old is not None and old in grp) else name


def migrate_names(h5):
    """Rename every pre-v4 dataset / subgroup in an open, writable h5 file or group to its v4 name
    (:data:`LEGACY_NAMES`), recursively; returns the number renamed."""
    import h5py
    n = 0
    for key in list(h5.keys()):
        obj = h5[key]
        if isinstance(obj, h5py.Group):
            n += migrate_names(obj)
        new = LEGACY_NAMES.get(key)
        if new is not None and new not in h5:
            h5.move(key, new)
            n += 1
    return n

# ---- the self-consistent bootstrap record (v3) ------------------------------
#: Group attrs of the per-draw (and ``_baseline``) ``jbs_loop`` block.
JBS_CONVERGED_ATTR = "jbs_converged"
JBS_N_PASSES_ATTR = "jbs_n_passes"
JBS_LOOP_JSON_ATTR = "jbs_loop_json"
#: Draw-group attr (legacy and swb draws): the shared kinetic sampler's
#: per-draw record as JSON (``bouquet.kinetic_sampler.KineticDraw.record``:
#: version, pressure match, clip counters, the resolved ``Z_imp`` and its
#: origin; PR #56).  Engine draws carry the
#: same record in their engine block's ``inputs.kinetic_sampler``.
KINETIC_SAMPLER_JSON_ATTR = "kinetic_sampler_json"
#: Draw-group attr (until-N runs): the draw loop's own in-spec verdict, the
#: one its ``on_inspec`` count (and a shard's ``n_inspec``) was taken on, so
#: a merge that drops the draw can take it off the recorded count.
UNTIL_N_INSPEC_ATTR = "until_n_inspec"
JBS_LOOP_ATTRS = (JBS_CONVERGED_ATTR, JBS_N_PASSES_ATTR, JBS_LOOP_JSON_ATTR)
#: Schema version that introduced the block (older archives never carry it).
JBS_LOOP_SINCE_SCHEMA = 3

#: How the bootstrap in an archive was computed -- the two values of
#: ``GenerationConfig.jbs_self_consistent``, as plot / table labels.
BOOTSTRAP_LABEL_SELF_CONSISTENT = "self-consistent Redl bootstrap"
BOOTSTRAP_LABEL_LEGACY = "frozen SWB bootstrap (legacy)"


def bootstrap_label(self_consistent) -> str:
    """Label of the bootstrap model: :data:`BOOTSTRAP_LABEL_SELF_CONSISTENT`
    for a loop run, :data:`BOOTSTRAP_LABEL_LEGACY` for the frozen path
    (``False``, or ``None`` = unknown / pre-loop archive)."""
    return (BOOTSTRAP_LABEL_SELF_CONSISTENT if bool(self_consistent)
            else BOOTSTRAP_LABEL_LEGACY)


def write_jbs_loop(grp, record) -> None:
    """Write the ``jbs_loop`` block (:data:`JBS_LOOP_ATTRS`) onto an h5 group.

    ``record`` is a loop record (``bouquet.jbs_loop``); ``None`` writes
    nothing, so a frozen-bootstrap group stays exactly as in v2.
    """
    if record is None:
        return
    import json
    from .jbs_loop import jsonable
    rec = jsonable(record)
    grp.attrs[JBS_CONVERGED_ATTR] = bool(rec.get("converged", False))
    # a draw's record totals every loop it ran (n_passes_total); a baseline's
    # is one loop (n_passes)
    n = rec.get("n_passes_total")
    if n is None:
        n = rec.get("n_passes")
    grp.attrs[JBS_N_PASSES_ATTR] = int(n or 0)
    grp.attrs[JBS_LOOP_JSON_ATTR] = json.dumps(rec)


def read_jbs_loop(grp):
    """The ``jbs_loop`` record of an h5 group as a dict, or ``None`` when the
    group carries none (a frozen bootstrap: v2, or ``jbs_self_consistent=
    False``)."""
    import json
    raw = grp.attrs.get(JBS_LOOP_JSON_ATTR)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return json.loads(str(raw))

# Bare dataset name -> unit string (empty = dimensionless).  ``psi_N`` /
# ``psi_N_kinetic`` hold the run grid in the group's ``profile_coord``
# ("psi_n" or "phi_n"; absent = "psi_n").
PROFILE_UNITS = {
    "psi_N": "",
    "psi_N_kinetic": "",
    "jphi_total": "A m^-2",
    "jphi_bs": "A m^-2",
    "jphi_bs,edge": "A m^-2",
    "jphi_ind": "A m^-2",
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
    "jphi_pd": "A m^-2",
    "jphi_saw_out": "A m^-2",
    "coil_currents": "A",
}

# Fixed in-group dataset names for the opaque byte blobs (F11).
EQDSK_DS = "eqdsk"
PFILE_DS = "pfile"
# OFT/TokaMaker i-file (inverse R,Z(psi,theta) for GPEC eq_type ldp_i); optional.
IFILE_DS = "ifile"
COIL_VALUES_DS = "coil_currents"
COIL_NAMES_DS = "coil_names"

# Live-equilibrium flux-surface-average block (optional per-draw subgroup),
# captured from the converged TokaMaker equilibrium at generate time to enable
# exact TokaMaker-jphi -> IMAS current conversions at export. All on the
# eq_fsa psi_N grid, which is always ψ_N, even in a Φ_N (profile_coord="phi_n")
# archive. See physics.capture_equilibrium_fsa.
EQ_FSA_GROUP = "eq_fsa"
EQ_FSA_UNITS = {
    "psi_N": "",
    "F": "T m",             # R*B_phi
    "pprime": "Pa Wb^-1",   # p', signed so jphi_eq > 0
    "jphi_eq": "A m^-2",    # own TokaMaker jphi <R>p' + <1/R>FF'/mu0
    "avg_R": "m",           # <R>
    "avg_inv_R": "m^-1",    # <1/R>
    "avg_inv_R2": "m^-2",   # <1/R^2> (exact quadrature; may be absent)
    "avg_B2": "T^2",        # <B^2>
    "q": "",
    "dV_dpsi": "m^3 Wb^-1",
    "f_trap": "",           # trapped fraction f_c
    "B_avg": "T",           # <|B|>
}


# The PARALLEL current parts of an engine draw (optional per-draw subgroup,
# written by the unified engine's draws since 2026-10-06): the field-aligned
# <j.B> components the archived toroidal split was converted from, on the
# subgroup's own psi_N (the draw's grid), positive (co-Ip) frame, raw <j.B>
# (NOT IMAS <j.B>/B0).  Identity, exact to round-off on that grid:
#   jphi_total = kappa (jdotB_ind + jdotB_bs + jdotB_nbi + jdotB_rf) + jphi_pd
# with kappa = F<1/R>/<B^2> (physics.field_aligned_conversion) and
# jphi_pd = p'(<R> - F^2<1/R>/<B^2>) of the archived state.  jdotB_ind
# is the field-aligned inductive ONLY (the archived toroidal jphi_ind
# minus jphi_pd, over kappa).  The IDS exporter writes these, so no
# exported parallel current carries the pressure-driven term.
JB_PARALLEL_GROUP = "jdotB"
JB_PARALLEL_UNITS = {
    "psi_N": "",
    "jdotB_ind": "T A m^-2",
    "jdotB_bs": "T A m^-2",
    "jdotB_nbi": "T A m^-2",
    "jdotB_rf": "T A m^-2",       # rf (ec/lh/ic) + other driven
    "kappa": "T^-1",           # F<1/R>/<B^2>
    "jphi_pd": "A m^-2",    # toroidal, p'(<R> - F^2<1/R>/<B^2>)
}


def write_jdotB(grp, parts):
    """Write the ``jdotB/`` subgroup (see :data:`JB_PARALLEL_GROUP`)
    into the draw group *grp*, replacing any earlier one."""
    import numpy as np
    if JB_PARALLEL_GROUP in grp:
        del grp[JB_PARALLEL_GROUP]
    sub = grp.create_group(JB_PARALLEL_GROUP)
    for name, unit in JB_PARALLEL_UNITS.items():
        if parts.get(name) is None:
            raise ValueError(f"write_jdotB: missing {name!r}")
        ds = sub.create_dataset(name, data=np.asarray(parts[name],
                                                      dtype=np.float64))
        if unit:
            ds.attrs["units"] = unit
    sub.attrs["identity"] = ("jphi_total = kappa (jdotB_ind + jdotB_bs + jdotB_nbi "
                             "+ jdotB_rf) + jphi_pd")
    return sub


def read_jdotB(grp):
    """The ``jdotB/`` subgroup of a draw group as a dict of float
    arrays, or ``None`` (a legacy draw, or an engine draw archived before
    2026-10-06)."""
    import numpy as np
    name = stored_name(grp, JB_PARALLEL_GROUP)
    if name not in grp:
        return None
    sub = grp[name]
    return {LEGACY_NAMES.get(k, k): np.asarray(sub[k][()], dtype=float) for k in sub}


# ---- where the pressure-driven current sits in the archived split ----------
#: Group attr (draw and ``_baseline``) naming where the pressure-driven
#: ``p'G = p'(<R> - F^2<1/R>/<B^2>)`` (A7 of docs/current-conventions.md) sits
#: in the archived toroidal split ``jphi_total = jphi_ind + jphi_bs + fixed (+
#: jphi_pd)``.  Its <j.B> is zero, so the IDS exporter must take it off
#: whichever bucket carries it before converting to parallel currents.
CURRENT_SPLIT_CONVENTION_ATTR = "current_split_convention"
#: Owner decision D2 (2026-10-09): ``p'G`` is its own ``jphi_pd`` dataset;
#: neither ``jphi_bs`` nor ``jphi_ind`` carries it.
SPLIT_PRESSURE_SEPARATE = "pressure_separate"
#: Every archive before PR #64 (absent attr): the residual ``jphi_ind``
#: carries ``p'G``.
SPLIT_PRESSURE_IN_INDUCTIVE = "pressure_in_inductive"
#: PR #64's convention (evaluate_jbs/3; never on main): ``jphi_bs`` carries it.
SPLIT_PRESSURE_IN_BOOTSTRAP = "pressure_in_bootstrap"
CURRENT_SPLIT_CONVENTIONS = (SPLIT_PRESSURE_SEPARATE,
                             SPLIT_PRESSURE_IN_INDUCTIVE,
                             SPLIT_PRESSURE_IN_BOOTSTRAP)
JPRESSURE_DS = "jphi_pd"


def write_current_split(grp, jphi_pd, convention=SPLIT_PRESSURE_SEPARATE):
    """Stamp *grp* (a draw group or ``_baseline``) with its split convention
    and, for :data:`SPLIT_PRESSURE_SEPARATE`, write the ``jphi_pd``
    dataset (replacing an earlier one).  ``jphi_pd`` may be None only for
    the two carried conventions."""
    import numpy as np
    if convention not in CURRENT_SPLIT_CONVENTIONS:
        raise ValueError(f"convention must be one of {CURRENT_SPLIT_CONVENTIONS},"
                         f" got {convention!r}")
    if convention == SPLIT_PRESSURE_SEPARATE:
        if jphi_pd is None:
            raise ValueError("write_current_split: the separate convention "
                             "needs the jphi_pd profile")
        if JPRESSURE_DS in grp:
            del grp[JPRESSURE_DS]
        write_profile(grp, JPRESSURE_DS, np.asarray(jphi_pd, dtype=float))
    grp.attrs[CURRENT_SPLIT_CONVENTION_ATTR] = convention


def read_current_split_convention(grp, baseline_attrs=None) -> str:
    """Where *grp*'s archived split keeps ``p'G`` (one of
    :data:`CURRENT_SPLIT_CONVENTIONS`).

    The group's own attr, else the ``_baseline`` attrs' (*baseline_attrs*);
    without either: a ``jphi_pd`` dataset means separate; a loop record
    written by ``evaluate_jbs/3`` (PR #64's evaluator) means in the
    bootstrap; anything else is an archive from before PR #64 -- in the
    inductive."""
    def _s(v):
        return v.decode() if isinstance(v, bytes) else (None if v is None
                                                         else str(v))
    v = _s(grp.attrs.get(CURRENT_SPLIT_CONVENTION_ATTR))
    if v is None and baseline_attrs is not None:
        v = _s(dict(baseline_attrs).get(CURRENT_SPLIT_CONVENTION_ATTR))
    if v is not None:
        if v not in CURRENT_SPLIT_CONVENTIONS:
            raise ValueError(f"unknown {CURRENT_SPLIT_CONVENTION_ATTR} {v!r}")
        return v
    if JPRESSURE_DS in grp:
        return SPLIT_PRESSURE_SEPARATE
    rec = read_jbs_loop(grp)
    if rec is not None and str(rec.get("evaluate_jbs_version", "")).startswith(
            "evaluate_jbs/3"):
        return SPLIT_PRESSURE_IN_BOOTSTRAP
    return SPLIT_PRESSURE_IN_INDUCTIVE


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
    """Read dataset ``name`` from ``grp`` as a float ndarray (``None`` if absent); an older archive's
    spelling of it is read too (:func:`stored_name`)."""
    import numpy as np
    name = stored_name(grp, name)
    if name not in grp:
        return None
    return np.array(grp[name], dtype=float)
