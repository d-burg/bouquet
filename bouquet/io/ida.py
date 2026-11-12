"""Reader for IDA integrated-data-analysis files (DIII-D, netCDF ``.cdf``).

The IDA ``.cdf`` is netCDF4 -- i.e. an HDF5 container -- so it is read with
**h5py alone** (already a bouquet dependency); no OMFIT / netCDF4 / OMFITnc
required. Datasets map directly: ``f['n_e'][:]``, ``f['n_e_err'][:]``, etc.

Returns the kinetic profiles together with their uncertainty (sigma) profiles,
on the IDA psi_N grid. Both the baseline reconstruction (profiles) and the
uncertainty envelope (sigmas) draw from this single read.

Operational DIII-D ``IDA_*.cdf`` layout (verified against a real IDA file):
    profiles are 2-D ``(n_time, n_radial)`` with companion ``*_err`` datasets
    (direct 1-sigma); the radial grid is ``psi_n`` (n_radial,), extending past
    the separatrix to ~1.2; ``time`` is in milliseconds. Units are already SI
    (n_e in m^-3; T_e, T_12C6 in eV).

There is no stored main-ion density.  By default (``ni_source="standard"``)
``Zeff`` is the file's visible-bremsstrahlung value and ``ni`` is derived from
``(ne, Zeff)`` under single-impurity quasineutrality with the machine impurity
charge ``impurity_Z`` (carbon Z=6 by default), its sigma the ne fraction.
EXPERIMENTAL (``bouquet.experimental.REGISTRY["ida_ion_route"]``): ``ni`` from
``Zeff`` with a propagated sigma (``ni_source="Zeff"``), from the measured
carbon density ``n_12C6`` from charge exchange recombination
(``ni_source="CER"``), or from the mean of the two (``ni_source="all"``).
With both active, their disagreement beyond statistical error widens
``sigma_ni``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import zeff_sources as _zs

# Magnitude above which a value is treated as a netCDF fill marker rather
# than data (default fills are ~9.97e36; physical densities are < 1e22).
_CARBON_FILL = 1e30


@dataclass
class IDAProfiles:
    """Kinetic profiles + sigmas read from an IDA file, at one time slice.

    Values are returned in SI (ne/ni in m^-3, Te/Ti in eV) on ``psi_N``.
    ``psi_N`` typically extends past the separatrix (~1.2) into the SOL; pass it
    through to ``generate_bouquet`` as ``psi_N_kinetic`` rather than truncating.
    """

    psi_N: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    ni: np.ndarray
    ti: np.ndarray
    Zeff: np.ndarray

    sigma_ne: np.ndarray
    sigma_te: np.ndarray
    sigma_ni: np.ndarray
    sigma_ti: np.ndarray

    time: float                     # selected slice [s]
    raw_bytes: Optional[bytes] = None   # original file bytes for archival
    # Per-radius tension between the two routes, in sigma, one per channel:
    # each is the delta that widens THAT channel's envelope, over its own
    # propagated spread. None unless both routes are live.
    ni_route_chi: Optional[np.ndarray] = None
    zeff_route_chi: Optional[np.ndarray] = None

    #: Resolved 1-sigma envelope on ``Zeff`` (the VB+CER > CER > VB ladder,
    #: route disagreement folded in).
    sigma_Zeff: Optional[np.ndarray] = None
    #: Ladder rung of ``sigma_Zeff``: "VB+CER" | "CER" | "VB", with
    #: " (ne-fraction sigma)" when neither route carries an envelope.
    sigma_Zeff_source: str = "none"
    #: Carbon-propagated dilution uncertainty, expressed as a sigma on Zeff:
    #: ``Z(Z-1)(nC/ne) sqrt((s_nC/nC)^2 + (s_ne/ne)^2)`` on the direct
    #: layout, the sample spread of ``1 + Z(Z-1) nC_s/ne_s`` on the ensemble
    #: layout.  This is the DIRECT measurement of the dilution the
    #: Zeff-primary scheme perturbs (ni = ne - Z nC), and it is markedly
    #: tighter and better behaved than the file's VB-derived ``Zeff_err``
    #: (measured on the demo shots: 1.9-2.4 % core / 4-19 % SOL, vs
    #: 8-9 % core / 44-130 % SOL for Zeff_err, whose n_e^2 sqrt(T_e)
    #: propagation + calibration + mantle-subtraction systematics dominate).
    #: Drawing Zeff with this sigma IS error propagation through
    #: ``ni = ne - Z nC``: ni is an exact function of the drawn (ne, Zeff).
    sigma_Zeff_carbon: Optional[np.ndarray] = None
    #: Provenance: "n_12C6_err" | "ensemble-samples" | "none".
    sigma_Zeff_carbon_source: str = "none"
    #: Zeff-vs-carbon cross-check, when the file also carries ``n_12C6``:
    #: relative deviation of reported Zeff (VB) from
    #: ``1 + Z(Z-1) n_C/n_e`` (CER carbon) over psi_N <= 0.9, as
    #: ``{"median": ..., "worst": ..., "psi_worst": ...}``.  Report-only --
    #: measured -1.7 % / +4.7 % / +11.3 % core-median on three DIII-D demo
    #: shots, i.e. shot-dependent and mostly within the measured sigma_Zeff;
    #: a large value flags the file, not the workflow.
    zeff_carbon_dev: Optional[dict] = None
    #: dZeff/dne of the resolved Z_eff at fixed nC: ``-w_c (Zeff_CER - 1)/ne``,
    #: zero without the CER route.  ``sigma_Zeff`` carries this ne term, so a
    #: draw deriving ni from its own (ne, Zeff) adds ``zeff_dne * dne`` and
    #: draws the rest; ni then carries ``sigma_ni`` exactly (to first order).
    zeff_dne: Optional[np.ndarray] = None
    #: The file's own safety factor on ``psi_N`` (``None`` if absent): the
    #: fit's equilibrium, from which :func:`bouquet.coords.phi_n_from_q`
    #: places the profiles in Φ_N.
    q: Optional[np.ndarray] = None
    #: Provenance of the VALUE of ``Zeff`` (and so of ``ni``), for the
    #: baseline record: ``{"convention", "source", "ni_source", "weights",
    #: "window", "n_clipped_vb", "n_clipped_cer", "n_nodes", "impurity_Z"}``.
    #: ``convention`` names the Z_eff convention (for IDA the measured Z_eff,
    #: whose numerator counts every charge the measurement sees); ``source`` names the
    #: rung that set the value ("VB+CER mean", "CER", "VB"); the
    #: ``n_clipped_*`` count the radial nodes each route was clamped to the
    #: single-impurity window ``[1, Z]`` at, before the mean -- the clip is
    #: physical (``ni >= 0``, ``ni <= ne``) and unchanged, it is only made
    #: visible.  See docs/physics-notes.md, "Kinetic assumptions".
    zeff_provenance: Optional[dict] = None
    #: The Z_eff source-rule record (:mod:`bouquet.io.zeff_sources`): every
    #: candidate with each test's value, threshold and verdict, the tension
    #: between the candidates, the decision, any override and the thresholds
    #: used.  ``None`` on ``ni_source="standard"`` (the rule is not applied);
    #: on ``"checked"`` it decides the source (and is also stored as
    #: ``zeff_provenance["zeff_rule"]``); on ``"CER"`` / ``"Zeff"`` /
    #: ``"all"`` it is evaluated and recorded beside the forced route, which
    #: it does not change (``record["forced"]``).
    zeff_rule: Optional[dict] = None


@dataclass
class IDACERProfiles:
    """Impurity (carbon CER) channels for a radial-field / rotation analysis.

    Everything needed for the impurity radial force balance
    ``E_r = (dp_C/dR)/(Z_C e n_C) - v_pol B_phi + omega R B_pol`` (see
    :func:`bouquet.physics.radial_field_from_impurity_force_balance`), read at one time slice on
    ``psi_N``. SI units: ``n_carbon`` m^-3, ``t_carbon`` eV, ``omega_tor`` rad/s,
    ``v_pol`` m/s, ``Bpol`` T, ``Rmaj`` m, ``dpsiN_dR`` 1/m. The ``sigma_*``
    fields are the measured 1-sigma envelopes (for propagating E_r uncertainty).
    ``Bpol`` is that relation's B_theta: IDA's ``Bpol_midplane`` is the vertical
    field B_Z at the outboard midplane, where B_theta = -B_Z (theta-hat = -Z-hat
    for right-handed (r, theta, phi), phi counter-clockwise from above).  That
    identity is geometric, so it holds for either I_p and B_T direction.
    ``v_pol`` is returned as the file stores it (``v_pol_12C6``), taken as
    v_theta along the same theta-hat (= -Z-hat at the outboard midplane); with
    it, E_r reproduced IDA's own E_r on the validated cases, which were all of
    the normal I_p / B_T orientation (the reversed orientations are not yet
    validated against IDA).  ``omega_tor`` and ``B_phi`` must be in the same
    lab frame: take ``B_phi = F/R`` from the INPUT (lab-frame) g-file, not
    from a bouquet-written one, which is in the positive-I_p frame (#68).
    """

    psi_N: np.ndarray
    n_carbon: np.ndarray
    t_carbon: np.ndarray
    omega_tor: np.ndarray
    v_pol: np.ndarray
    Bpol: np.ndarray
    Rmaj: np.ndarray
    dpsiN_dR: np.ndarray

    sigma_n_carbon: np.ndarray
    sigma_t_carbon: np.ndarray
    sigma_omega_tor: np.ndarray
    sigma_v_pol: np.ndarray

    time: float


def _select_time_index(time_ms: np.ndarray, time_s: Optional[float]) -> int:
    """Return the index of the slice nearest ``time_s`` (seconds)."""
    if time_ms.size == 1:
        return 0
    if time_s is None:
        avail = ", ".join(f"{t/1e3:.4f}" for t in time_ms)
        raise ValueError(
            f"IDA file has {time_ms.size} time slices; pass `time` (seconds). "
            f"Available [s]: {avail}"
        )
    return int(np.argmin(np.abs(time_ms / 1e3 - time_s)))


def ida_time_base(path: str) -> np.ndarray:
    """The IDA file's own time base [s] (stored in ms), as written -- the
    axis :func:`bouquet.io.imas._hybrid_timing` matches ``ida_time`` on
    (the half-step source-time rule) before the slice is read."""
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f["time"][:], dtype=float).ravel() / 1e3


def _carbon_tier_usable(nC, ne, sigma_nC=None, sigma_ne=None, *,
                        unit="radii", datasets="n_12C6/n_12C6_err") -> bool:
    """Is the carbon dilution data physical enough to carry the Zeff sigma tier?

    Shared by both IDA layouts (the direct ``*_err`` one and the 3-D posterior),
    because the hazard is the file's, not the layout's.  Negative nC (SOL spline
    undershoot), netCDF fill values (~1e36) and NaN holes all survive the clip
    floors downstream: a negative nC keeps its sign while its |value| is floored
    out of the relative-error denominator, and a fill value scales straight
    through, producing ~1e6-scale (direct) or ~1e17-scale (ensemble) sigmas that
    pass every downstream guard -- finite, non-negative, some > 0 -- and silently
    corrupt the Zeff ensemble while the run completes normally.

    Any bad entry drops the whole tier, loudly and with a count, matching the
    None-not-a-guessed-scalar rule elsewhere in this reader.  ``unit`` names what
    the entries are ("radii", "sample points") for the printed reason.
    """
    ok = (np.isfinite(nC) & np.isfinite(ne) & (nC > 0) & (ne > 0)
          & (np.abs(nC) < _CARBON_FILL) & (np.abs(ne) < _CARBON_FILL))
    if sigma_nC is not None:
        ok = ok & (np.isfinite(sigma_nC) & (sigma_nC >= 0)
                   & (np.abs(sigma_nC) < _CARBON_FILL))
    if sigma_ne is not None:
        ok = ok & np.isfinite(sigma_ne)
    bad = ~ok
    if not np.any(bad):
        return True
    print(f"[read_ida] carbon-tier sigma skipped: {int(bad.sum())}/{bad.size} "
          f"{unit} carry non-physical {datasets} (negative, non-finite, or fill "
          f"values); the Zeff envelope falls back a tier")
    return False


def _file_fit_points(f, t_idx):
    """The file's own impurity-density points for the slice (the points its
    fit used), from ``raw_n_12C6`` / ``raw_n_12C6_err`` / ``psin_n_12C6``;
    ``None`` when the file does not carry all three.  Fill values and
    non-finite entries are padding and are dropped, so a slice whose fit
    used no point gives an empty set
    (:func:`bouquet.io.zeff_sources.fit_points_from_arrays`)."""
    keys = ("raw_n_12C6", "raw_n_12C6_err", "psin_n_12C6")
    if not all(k in f for k in keys):
        return None
    return _zs.fit_points_from_arrays(*(f[k][t_idx] for k in keys))


#: the forced routes' candidates (what each route takes regardless of the
#: rule; "all" averages both where both have an envelope)
_FORCED_CANDIDATES = {"CER": [_zs.IMPURITY], "Zeff": [_zs.LINE],
                      "all": [_zs.IMPURITY, _zs.LINE]}


def _zeff_rule_for_slice(f, t_idx, t_sel, time_ms, *, psi_N, ne, zeff,
                         sigma_zeff, n_carbon, sigma_n_carbon,
                         sigma_zeff_carbon_rule, validity, impurity_Z, q,
                         is_ensemble, ni_source, override, fit_points,
                         raw_points, fast_density, ensemble_median,
                         crosschecks, path):
    """Gather the rule's inputs from an open IDA file at one slice and build
    the record (:func:`bouquet.io.zeff_sources.zeff_source_record`); on
    ``ni_source="checked"`` refuse when no source can be used.  Nothing is
    warned here: :func:`_zeff_rule_warn` warns once, after the route's
    arithmetic.  ``validity``: the carbon validity screen restricted to the
    plasma (:func:`bouquet.io.zeff_sources.validity_inside_boundary`), the
    precondition of I-3; ``sigma_zeff_carbon_rule``: the carbon Z_eff
    envelope under that screen (the tension's impurity sigma)."""
    psi_N, ne = np.asarray(psi_N, dtype=float), np.asarray(ne, dtype=float)
    # rho_tor,norm on the file's grid (inside the boundary) from its own q
    rho_map, rho_note = None, "none (the file carries no q)"
    if q is not None:
        try:
            rho_map = _zs.rho_map_from_q(psi_N, q)
            rho_note = "rho_tor from the file's q"
        except ValueError as exc:
            rho_note = ("none (the rho_tor map from the file's q could not "
                        f"be built: {exc})")
    nfast = fast_density
    if callable(nfast):
        nfast = np.asarray(nfast(psi_N), dtype=float)
    elif nfast is not None and not isinstance(nfast, str):
        nfast = np.asarray(nfast, dtype=float)
        if nfast.shape != psi_N.shape:
            raise ValueError(
                f"{path!r}: zeff_fast_density has shape {nfast.shape}, the "
                f"file's psi_N grid {psi_N.shape}; pass it on that grid, as "
                "f(psi_N), or as {'psi_N': ..., 'n_fast': ...} in the config")

    impurity = None
    zeff_from_carbon = None
    if n_carbon is not None:
        # the file's density is carbon (n_12C6): its Z_eff by quasineutrality
        # (and the value a copied file Z_eff would hold, L-3) uses Z = 6
        with np.errstate(divide="ignore", invalid="ignore"):
            zeff_from_carbon = 1.0 + (6.0 * 5.0 * np.asarray(n_carbon,
                                                             dtype=float)
                                      / np.clip(ne, 1e10, None))
        if float(impurity_Z) != 6.0:
            impurity = _zs.unusable_impurity(
                f"impurity_Z={float(impurity_Z):g}, but the file's impurity "
                "density is carbon ('n_12C6', Z = 6)")
        else:
            fit = (fit_points if fit_points is not None else
                   None if is_ensemble else _file_fit_points(f, t_idx))
            win = (_zs.slice_window(f["time_win_start"][:],
                                    f["time_win_end"][:], t_idx, t_sel)
                   if ("time_win_start" in f and "time_win_end" in f)
                   else None)
            impurity = _zs.impurity_tests(
                psi_N, ne, n_carbon, sigma_n_carbon, impurity_Z=impurity_Z,
                n_fast=nfast, rho_map=rho_map, fit_points=fit,
                raw_points=raw_points, slice_time=t_sel, slice_window=win,
                validity=validity)
        impurity["dataset"] = "n_12C6"

    line = None
    if zeff is not None:
        # the advisory: the core median over the file's non-placeholder
        # slices (all slices without a carbon channel)
        zs_, nes_, ncs_ = [], [], []
        for k in range(time_ms.size):
            z_k = np.asarray(f["Zeff"][k], dtype=float)
            ne_k = np.asarray(f["n_e"][k], dtype=float)
            nc_k = (np.asarray(f["n_12C6"][k], dtype=float)
                    if "n_12C6" in f else None)
            if is_ensemble:
                _c = (lambda a: np.median(a, axis=0)) if ensemble_median \
                    else (lambda a: np.mean(a, axis=0))
                z_k, ne_k = _c(z_k), _c(ne_k)
                nc_k = None if nc_k is None else _c(nc_k)
            zs_.append(z_k)
            nes_.append(ne_k)
            ncs_.append(nc_k)
        series = _zs.advisory_core_series(
            psi_N, zs_, nes_, ncs_ if "n_12C6" in f else None)
        line = _zs.line_tests(psi_N, zeff, sigma_zeff,
                              zeff_impurity=zeff_from_carbon,
                              core_median_series=series)
        line["dataset"] = "Zeff"

    names = {"CER": _zs.IMPURITY, "Zeff": _zs.LINE}
    forced = (None if ni_source == "checked" else
              dict(ni_source=str(ni_source),
                   candidates=list(_FORCED_CANDIDATES[ni_source])))
    usable_imp = impurity is not None and impurity.get("usable", True)
    try:
        rec = _zs.zeff_source_record(
            impurity, line, psi_N=psi_N,
            zeff_imp=zeff_from_carbon if usable_imp else None,
            sigma_zeff_imp=sigma_zeff_carbon_rule if usable_imp else None,
            zeff_line=zeff, sigma_zeff_line=sigma_zeff,
            override=names.get(override), time=t_sel, forced=forced,
            crosschecks=crosschecks)
    except _zs.ZeffSourceRefused as exc:      # an override that cannot hold
        raise _zs.ZeffSourceRefused(
            f"{path!r} at {t_sel:.4f} s: zeff_override={override!r}: {exc}"
        ) from None
    rec["ni_source"] = str(ni_source)
    rec["rho_map"] = rho_note
    if validity is not None:
        rec["validity_screen"] = dict(
            validity, region=f"psi_N <= {_zs.PSI_N_PLASMA_MAX:g}",
            note=("the reader's carbon validity criteria, applied inside the "
                  "boundary only (where the rule's tests are defined); the "
                  "pre-existing routes' own screen is unchanged"))

    dec = rec["decision"]
    if ni_source == "checked" and dec["source"] is None:
        raise _zs.ZeffSourceRefused(
            f"{path!r} at {t_sel:.4f} s: the Z_eff source rule "
            f"(ni_source='checked') refuses this slice: "
            f"{_zs.refusal_message(rec)}.  Name a source with "
            "zeff_override='CER' or 'Zeff' (recorded as an override), or "
            "pass ni_source='standard' (the file's Z_eff as stored, no "
            "rule).  See docs/zeff-source-rule.md", rec)
    return rec


def _zeff_rule_warn(rec, ni_source, t_sel, used, provenance):
    """Warn once per read, after the route's arithmetic.  ``"checked"``: the
    decided source's findings (unverified, tension, override, dilution,
    advisory).  A forced route: only when a candidate the route takes FAILS
    a recorded test (an unverified one is recorded, not warned), with any
    clamping the route applies; the route itself is unchanged."""
    import warnings

    if ni_source == "checked":
        rec["route_taken"] = dict(
            zeff_source=provenance.get("source"),
            n_clipped_vb=provenance.get("n_clipped_vb"),
            n_clipped_cer=provenance.get("n_clipped_cer"))
        msg = _zs.checked_warning(rec)
        if msg:
            warnings.warn(f"read_ida (ni_source='checked', {t_sel:.4f} s): "
                          + msg, _zs.ZeffSourceWarning, stacklevel=3)
        return
    rec["forced"]["candidates_used"] = list(used)
    rec["forced"]["n_clipped"] = dict(
        VB=int(provenance.get("n_clipped_vb", 0)),
        CER=int(provenance.get("n_clipped_cer", 0)))
    msgs = _zs.forced_route_failures(rec, used)
    if msgs:
        clips = [f"{k} {v}" for k, v in rec["forced"]["n_clipped"].items()
                 if v]
        if clips:
            msgs.append("the route clamps Z_eff to [1, Z] at "
                        + ", ".join(clips) + " node(s)")
    rec["forced_warnings"] = msgs
    if msgs:
        warnings.warn(
            f"read_ida (ni_source={ni_source!r}, {t_sel:.4f} s): "
            + "; ".join(msgs) + " (recorded in IDAProfiles.zeff_rule; the "
            "route is unchanged)", _zs.ZeffSourceWarning, stacklevel=3)


def zeff_rule_series(path, times, *, source=None, **kwargs):
    """One Z_eff source, named by the user, for a time series of slices of
    one IDA file.

    ``source`` (``"CER"`` or ``"Zeff"``) is required: every slice is read
    with ``ni_source="checked"`` and ``zeff_override=source`` (recorded on
    each slice), and the slices on which that source fails its tests or is
    unverified are flagged in the series record (never switched).  Without
    ``source`` the call is refused (:class:`ZeffSourceRefused`) with the
    rule's per-slice outcomes listed, so the user can name one: the rule
    does not pick a series source itself, even when every slice agrees.

    This helper is the only place the one-source-per-series provision is
    enforced: independent per-slice reads (``read_ida(ni_source="checked",
    time=t)`` or one ``ReconstructionSource(ni_source="checked", time=t)``
    per slice) each apply the rule on their own and can take different
    sources on different slices.

    Returns ``(profiles, series)``: one :class:`IDAProfiles` per time and
    the series record (:func:`bouquet.io.zeff_sources.series_source`).
    ``kwargs`` go to :func:`read_ida` (per-slice point data cannot be passed
    this way; read slice by slice for that).  A slice on which the named
    source cannot be used at all (absent, or an impurity density that fails
    I-3) is refused, naming the slice.
    """
    import warnings

    if kwargs.get("ni_source", "checked") != "checked":
        raise ValueError("zeff_rule_series applies ni_source='checked'")
    kwargs.pop("ni_source", None)
    for k in ("zeff_override", "zeff_fit_points", "zeff_raw_points",
              "zeff_rule_from"):
        if k in kwargs:
            raise ValueError(f"zeff_rule_series: {k} is per slice; read the "
                             "slices with read_ida instead")
    names = {"CER": _zs.IMPURITY, "Zeff": _zs.LINE}
    if source is not None and source not in names:
        raise ValueError(f"source={source!r} must be 'CER' or 'Zeff'")
    if source is None:
        recs = []
        for t in times:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", _zs.ZeffSourceWarning)
                try:
                    recs.append(read_ida(path, time=t, ni_source="checked",
                                         **kwargs).zeff_rule)
                except _zs.ZeffSourceRefused as exc:
                    recs.append(exc.record)
        try:
            _zs.series_source(recs, source=None)   # refuses, listing them
        except _zs.ZeffSourceRefused as exc:
            # this helper takes the reader's names, not the record's
            raise _zs.ZeffSourceRefused(
                f"{exc}.  Pass source='CER' ({_zs.IMPURITY}) or "
                f"source='Zeff' ({_zs.LINE})",
                record=getattr(exc, "record", None)) from exc
    out, recs = [], []
    for t in times:
        try:
            p = read_ida(path, time=t, ni_source="checked",
                         zeff_override=source, **kwargs)
        except _zs.ZeffSourceRefused as exc:
            raise _zs.ZeffSourceRefused(
                f"zeff_rule_series (source={source!r}), slice {t}: {exc}",
                exc.record) from None
        out.append(p)
        recs.append(p.zeff_rule)
    series = _zs.series_source(recs, source=names[source])
    series["source_route"] = source
    return out, series


def _chi(delta, var_delta):
    """Route disagreement in sigma; NaN where the spread is zero."""
    return np.sqrt(np.divide(delta ** 2, var_delta,
                             out=np.full_like(delta, np.nan),
                             where=var_delta > 0.0))


def read_ida(
    path: str,
    time: Optional[float] = None,
    sigma_mode: str = "auto",
    sigma_method: str = "percentile",   # ensemble-layout band estimator
    ensemble_median: bool = False,      # ensemble-layout central estimator
    ni_source: str = "standard",
    zeff_substitute=None,
    impurity_Z: float = 6.0,
    sigma_ni_from_ne: Optional[bool] = None,
    *,
    zeff_override: Optional[str] = None,
    zeff_fit_points=None,
    zeff_raw_points=None,
    zeff_fast_density=None,
    zeff_crosschecks: Optional[dict] = None,
    zeff_rule_from: Optional[dict] = None,
) -> IDAProfiles:
    """Read an IDA ``.cdf`` and return profiles + sigmas at ``time``.

    Parameters
    ----------
    path : str
        Path to the IDA netCDF file.
    time : float, optional
        Time slice in seconds. Required when the file holds more than one slice;
        the nearest slice is selected.
    sigma_mode : {"auto", "direct", "ensemble"}
        ``"auto"`` (default) picks the layout from the array dimensionality.
        ``"direct"`` reads the ``*_err`` datasets (2-D operational layout);
        ``"ensemble"`` reduces a 3-D posterior-sample file (mean profile + a
        sample-spread sigma). Passing a mode that contradicts the file raises.
    sigma_method : {"percentile", "std"}
        Ensemble band estimator: ``"percentile"`` -> (p84-p16)/2 (robust),
        ``"std"`` -> sample standard deviation. Unused for the direct layout.
    ensemble_median : bool
        Ensemble central estimator: sample mean (default) or median.
    ni_source : {"standard", "checked", "Zeff", "CER", "all"}
        Which measurement the main-ion density comes from.  ``"checked"``
        applies the Z_eff source rule (:mod:`bouquet.io.zeff_sources`,
        docs/zeff-source-rule.md): the impurity-density candidate (``n_12C6``)
        and the file's line-integrated ``Zeff`` are tested at the slice, each
        is qualified, unverified (a test, or part of one, could not be
        evaluated) or failed, and the source the ranking names (qualified
        impurity > qualified file value > unverified impurity, labeled and
        warned) is taken through the ``"CER"`` or ``"Zeff"`` route below;
        otherwise the read is refused
        (:class:`~bouquet.io.zeff_sources.ZeffSourceRefused`) unless
        ``zeff_override`` names a source.  The rule's validity screen and
        tests are restricted to the plasma (psi_N <= 1): an impurity density
        invalid only outside the boundary is not refused, and when the
        carbon envelope was dropped by the whole-grid screen for that reason
        alone the impurity source takes its envelope screened inside the
        boundary (``sigma_Zeff_carbon_source`` says so).  The record is
        ``IDAProfiles.zeff_rule`` (and ``zeff_provenance["zeff_rule"]``).
        Each ``"checked"`` read applies the rule on its own: the
        one-source-per-series provision is enforced only by
        :func:`zeff_rule_series`, not by independent per-slice reads.
        ``"standard"``
        (default): ``Zeff`` is the file's VB value as stored, ``ni`` follows
        from ``(ne, clip(Zeff, 1, Z))`` by single-impurity quasineutrality and
        ``sigma_ni = |ni| sigma_ne/ne``; ``sigma_Zeff`` is the file's own
        Z_eff envelope (``Zeff_err`` or the sample spread), and the Z_eff
        envelope ladder (carbon > VB > scalar) is resolved downstream.
        EXPERIMENTAL (``bouquet.experimental.REGISTRY["ida_ion_route"]``):
        ``"Zeff"`` applies quasineutrality to ``(ne, Zeff)`` with a
        propagated sigma; ``"CER"`` subtracts the measured carbon density
        ``n_12C6``; ``"all"`` takes the mean of the two. An explicit route
        needs its own ``*_err`` dataset on the direct layout and raises
        without it; ``"all"`` uses whichever routes the file supports.  On
        these three routes the rule is evaluated and recorded
        (``IDAProfiles.zeff_rule``, ``record["forced"]``) without changing
        the route; a warning
        (:class:`~bouquet.io.zeff_sources.ZeffSourceWarning`, once per read)
        is emitted only when a source the route takes FAILS a recorded test
        (an unverified one is recorded only).
    impurity_Z : float
        Impurity charge Z (carbon Z=6).
    zeff_substitute : callable, optional
        ``f(psi_N) -> Z_eff`` used in place of a MISSING ``Zeff`` dataset on
        the ``"standard"`` route only (the IDA-hybrid dd route, whose Z_eff
        is the dd's anyway: ``io.imas._merge_ida_kinetics`` passes the dd's
        thermal Z_eff).  Stamped in ``zeff_provenance`` (``source`` "dd
        (substitute: the file has no Zeff)").  Ignored when the file has
        ``Zeff``.  A file without ``Zeff`` takes the CER route alone on
        ``"all"``; ``"Zeff"``, and ``"standard"`` without a substitute,
        refuse it.
    sigma_ni_from_ne : bool, optional
        **Deprecated, no effect.**  Accepted so existing callers keep
        working; it was already a no-op before the ``ni_source`` ladder
        (``sigma_ni`` has always been propagated, never chosen by this
        flag).  Passing it emits a ``DeprecationWarning``; it will be
        removed in a future release.
    zeff_override : {"CER", "Zeff"}, optional
        ``ni_source="checked"`` only: an explicit, recorded choice of source
        that is honored whatever its tests say (``record["override"]`` keeps
        the failed tests and what the rule would have chosen).  Refused when
        the file does not carry the named source or cannot use it, and for
        an impurity density that fails I-3 (invalid inside the boundary, or
        n_i <= 0): an override cannot make those usable.
    zeff_fit_points : MeasurementPoints or dict, optional
        The impurity-density points the file's fit used at this slice
        (``value``, ``sigma`` in m^-3 at ``rho`` or ``psi_N``).  Default: the
        file's own ``raw_n_12C6`` / ``raw_n_12C6_err`` / ``psin_n_12C6``
        datasets when it carries all three; otherwise none.
        These serve the coverage test (I-1) only.
    zeff_raw_points : dict or sequence of MeasurementPoints, optional
        Raw impurity-density analyses with per-point times [s]; selected in
        the file's ``time_win_start`` / ``time_win_end`` window for the slice
        when it states one, else within +-25 ms.  Each analysis carries a
        ``kind`` label: the one labeled ``"hand_fit"`` is the reference for
        the coverage and consistency tests when present, else the one
        labeled ``"automatic"`` (the order never decides); all are recorded.
        Without raw analyses the consistency test (I-4) is not evaluated.
    zeff_fast_density : array, callable or "none", optional
        Fast-ion charge density [m^-3] on the file's ``psi_N`` grid (or
        ``f(psi_N)``), included in the rule's main-ion positivity test (I-3);
        ``"none"`` declares there are no fast ions (evaluated with fast ions
        = 0, recorded as a declaration).  Without it I-3 is evaluated
        without fast ions and the record says so (``evaluated_without``);
        this does not by itself make the impurity source unverified -- the
        impurity-derived Z_eff does not depend on the fast-ion density
        (singly charged main and fast ions), which changes the thermal n_i,
        not the choice of source.
        Points are placed in rho_tor,norm through the file's own ``q``
        (:func:`bouquet.io.zeff_sources.rho_map_from_q`); a file without a
        usable ``q`` leaves tests that need rho_tor on ``psi_N``-only points
        not evaluated, and the record says why.
    zeff_crosschecks : dict, optional
        Cross-checks to record beside the decision (``record["crosschecks"]``);
        never used to select a source.
    zeff_rule_from : dict, optional
        A rule record from an earlier read of the PROFILE file at this slice
        (its ``IDAProfiles.zeff_rule``, same ``ni_source``).  The rule is
        then not evaluated again and nothing is warned again: on
        ``"checked"`` the source is the one that record decided.  Used by
        the uncertainty re-read, so that neither a re-read of the same file
        nor a separately supplied sigma file decides differently.

    Notes
    -----
    Opens with ``h5py.File(path, "r")`` -- the file is netCDF4/HDF5, so no
    OMFIT or netCDF4 package is needed. Units are already SI; ``T_12C6`` maps to
    Ti.

    With both routes live, ``sigma_ni`` and ``sigma_Zeff`` also carry what the
    routes disagree on beyond their statistical errors. The term is one-sided,
    and ``ni_route_chi`` / ``zeff_route_chi`` report the tension behind each.
    They differ: ne cancels partly in ni and not at all in Z_eff.
    """
    import h5py

    if sigma_ni_from_ne is not None:
        import warnings
        warnings.warn(
            "read_ida(sigma_ni_from_ne=...) is deprecated and has no effect: "
            "sigma_ni is propagated from the ni_source ladder (ne, Z_eff and "
            "n_12C6 envelopes).  Drop the argument.",
            DeprecationWarning, stacklevel=2)
    if sigma_mode not in ("direct", "ensemble", "auto"):
        raise ValueError(
            f"unknown sigma_mode {sigma_mode!r}; expected 'direct', 'ensemble', or 'auto'")
    if sigma_method not in ("percentile", "std"):
        raise ValueError(
            f"unknown sigma_method {sigma_method!r}; expected 'percentile' or 'std'")
    if ni_source not in ("standard", "checked", "Zeff", "CER", "all"):
        raise ValueError(
            f"unknown ni_source {ni_source!r}; expected 'standard', "
            "'checked', 'Zeff', 'CER', or 'all'")
    if zeff_rule_from is not None:
        if ni_source == "standard" or not isinstance(zeff_rule_from, dict) \
                or zeff_rule_from.get("ni_source") != ni_source:
            raise ValueError(
                "zeff_rule_from must be the zeff_rule record of a read with "
                f"the same ni_source ({ni_source!r})")
        if zeff_override is not None:
            raise ValueError("zeff_rule_from already carries the decision "
                             "(and any override); do not pass zeff_override")
    if zeff_override is not None:
        if ni_source != "checked":
            raise ValueError(
                f"zeff_override={zeff_override!r} applies to "
                f"ni_source='checked' only (got ni_source={ni_source!r}: "
                "'CER' and 'Zeff' already name their source, 'standard' "
                "applies no rule)")
        if zeff_override not in ("CER", "Zeff"):
            raise ValueError(
                f"zeff_override={zeff_override!r} must be 'CER' (the impurity "
                "density) or 'Zeff' (the file's line-integrated Z_eff)")
    if ni_source == "CER" and float(impurity_Z) != 6.0:
        raise ValueError(
            f"ni_source={ni_source!r} requires impurity_Z=6.0, got {impurity_Z!r}: "
            "the carbon route subtracts the 'n_12C6' density, which is carbon "
            "(Z=6). For another impurity use ni_source='Zeff'")

    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    from ..physics import main_ion_density_from_zeff

    with h5py.File(path, "r") as f:
        time_ms = np.asarray(f["time"][:], dtype=float).ravel()
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)

        # Two field-validated layouts, distinguished by dimensionality:
        #   direct   : (n_time, n_radial) profiles + companion *_err datasets;
        #   ensemble : (n_time, n_samples, n_radial) posterior samples, no *_err
        #              -> profile = sample centre, sigma = sample spread.
        is_ensemble = (np.asarray(f["n_e"].shape).size == 3)
        if sigma_mode == "direct" and is_ensemble:
            raise ValueError("sigma_mode='direct' but the file is a 3-D posterior "
                             "(ensemble) IDA; use sigma_mode='auto' or 'ensemble'")
        if sigma_mode == "ensemble" and not is_ensemble:
            raise ValueError("sigma_mode='ensemble' but the file is a 2-D direct "
                             "IDA; use sigma_mode='auto' or 'direct'")
        # An explicit ni_source is strict ("CER" is checked by the ladder).
        if ni_source == "Zeff" and "Zeff" not in f:
            raise KeyError(
                f"{path!r} has no 'Zeff' dataset, so ni_source='Zeff' has no "
                "Z_eff to read; pass ni_source='CER' or 'all' to derive Z_eff "
                "from the carbon density instead")
        if ni_source == "standard" and "Zeff" not in f \
                and zeff_substitute is None:
            raise KeyError(
                f"{path!r} has no 'Zeff' dataset: ni_source='standard' reads "
                "the file's VB Z_eff; pass ni_source='CER' or 'all' to derive "
                "Z_eff from the carbon density")
        if ni_source == "Zeff" and not is_ensemble and "Zeff_err" not in f:
            raise KeyError(
                f"{path!r} has no 'Zeff_err' dataset, so the Zeff term of sigma_ni "
                "cannot be propagated; pass ni_source='CER' to derive ni from "
                "the carbon density instead")

        # The CER route's value + envelope, filled by whichever layout
        # branch runs below; None when the file carries no carbon channel.
        n_carbon = sigma_n_carbon = None
        # whether the carbon validity screen (_carbon_tier_usable) has run on
        # this file's carbon data; its verdict is sigma_Zeff_carbon (None:
        # rejected, or not run)
        _carbon_screened = False
        # the Z_eff source rule (every route but "standard"): the same
        # validity criteria restricted to the plasma (psi_N <= 1), where the
        # rule's tests are defined, and the carbon Z_eff envelope under that
        # screen -- the tension's impurity sigma, and the "checked" route's
        # carbon envelope when the whole-grid screen above dropped it for
        # values outside the boundary only
        _rule_on = ni_source != "standard"
        _rule_validity = _sigma_zc_rule = _sigma_zc_rule_source = None

        if is_ensemble:
            def _samples(key):  # (n_samples, n_radial) at the selected slice
                return np.asarray(f[key][t_idx], dtype=float)

            def _band(a):       # symmetric 1-sigma-equivalent over the sample axis
                if sigma_method == "std":
                    return np.std(a, axis=0)
                lo, hi = np.percentile(a, [16.0, 84.0], axis=0)
                return 0.5 * (hi - lo)

            def _center(a):
                return np.median(a, axis=0) if ensemble_median else np.mean(a, axis=0)

            psi_N = np.asarray(f["psi_n"][t_idx], dtype=float)[0]   # shared radial grid
            ne_s, te_s = _samples("n_e"), _samples("T_e")
            ti_s = _samples("T_12C6")
            zf_s = _samples("Zeff") if "Zeff" in f else None
            ne, te, ti = _center(ne_s), _center(te_s), _center(ti_s)
            Zeff = None if zf_s is None else _center(zf_s)
            sigma_ne, sigma_te, sigma_ti = _band(ne_s), _band(te_s), _band(ti_s)
            # Highest-fidelity tier: the posterior carries Zeff samples, so the
            # Zeff envelope is measured the same way as the kinetic ones.
            sigma_Zeff = None if zf_s is None else _band(zf_s)
            # Carbon tier: the dilution's own posterior, per sample.
            sigma_Zeff_carbon, sigma_Zeff_carbon_source = None, "none"
            if "n_12C6" in f:
                nc_s = _samples("n_12C6")
                # The CER route's own centre/spread, read whenever the
                # channel exists: the ladder below decides whether to use it.
                n_carbon, sigma_n_carbon = _center(nc_s), _band(nc_s)
                # Same screen as the direct layout, per SAMPLE: one fill value in
                # one sample at one radius is enough to put ~1e17 into the band,
                # and _band (a percentile half-width) is always finite and
                # non-negative, so nothing downstream would catch it.
                _carbon_screened = True
                if _carbon_tier_usable(nc_s, ne_s, unit="sample points",
                                       datasets="n_12C6/n_e samples"):
                    zc_s = (1.0 + impurity_Z * (impurity_Z - 1.0) * nc_s
                            / np.clip(ne_s, 1e10, None))
                    sigma_Zeff_carbon = _band(zc_s)
                    sigma_Zeff_carbon_source = "ensemble-samples"
                if _rule_on:
                    _rule_validity = _zs.validity_inside_boundary(
                        psi_N, ne_s, nc_s)
                    if sigma_Zeff_carbon is not None:
                        _sigma_zc_rule = sigma_Zeff_carbon
                        _sigma_zc_rule_source = sigma_Zeff_carbon_source
                    elif _rule_validity["ok"]:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            _sigma_zc_rule = _band(
                                1.0 + impurity_Z * (impurity_Z - 1.0) * nc_s
                                / np.clip(ne_s, 1e10, None))
                        _sigma_zc_rule_source = (
                            "ensemble-samples (validity screened inside the "
                            "boundary only)")
        else:
            def col(key):       # one radial profile at the selected time
                return np.asarray(f[key][t_idx], dtype=float)

            psi_N = np.asarray(f["psi_n"][:], dtype=float)
            ne, te = col("n_e"), col("T_e")          # m^-3, eV
            ti = col("T_12C6")                       # eV (carbon CER)
            # the VB Z_eff (dimensionless); None: the file carries none
            Zeff = col("Zeff") if "Zeff" in f else None
            sigma_ne, sigma_te, sigma_ti = col("n_e_err"), col("T_e_err"), col("T_12C6_err")
            if "n_12C6" in f:
                n_carbon = col("n_12C6")
                sigma_n_carbon = (col("n_12C6_err") if "n_12C6_err" in f
                                  else None)
            # Newer direct-layout vintages carry a measured Zeff_err profile;
            # older ones do not.  None (NOT a guessed scalar) marks the older
            # vintage, which is what drops the VB rung of the ladder below.
            sigma_Zeff = (col("Zeff_err") if ("Zeff_err" in f
                                              and Zeff is not None) else None)
            sigma_Zeff_carbon, sigma_Zeff_carbon_source = None, "none"
            if "n_12C6" in f and "n_12C6_err" in f:
                _nc, _snc = col("n_12C6"), col("n_12C6_err")
                # The propagation is only meaningful where the carbon data is
                # physical; see _carbon_tier_usable for what "physical" means
                # here and why a single bad radius drops the tier.
                _carbon_screened = True
                if _carbon_tier_usable(_nc, ne, sigma_nC=_snc, sigma_ne=sigma_ne):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _dil = (impurity_Z * (impurity_Z - 1.0) * _nc
                                / np.clip(ne, 1e10, None))
                        sigma_Zeff_carbon = _dil * np.sqrt(
                            (_snc / np.clip(_nc, 1e10, None)) ** 2
                            + (sigma_ne / np.clip(ne, 1e10, None)) ** 2)
                    sigma_Zeff_carbon_source = "n_12C6_err"
            if _rule_on and n_carbon is not None:
                _has_err = sigma_n_carbon is not None
                _rule_validity = _zs.validity_inside_boundary(
                    psi_N, ne, n_carbon, sigma_n_carbon,
                    sigma_ne if _has_err else None)
                if sigma_Zeff_carbon is not None:
                    _sigma_zc_rule = sigma_Zeff_carbon
                    _sigma_zc_rule_source = sigma_Zeff_carbon_source
                elif _has_err and _rule_validity["ok"]:
                    _nc, _snc = n_carbon, sigma_n_carbon
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _dil = (impurity_Z * (impurity_Z - 1.0) * _nc
                                / np.clip(ne, 1e10, None))
                        _sigma_zc_rule = _dil * np.sqrt(
                            (_snc / np.clip(_nc, 1e10, None)) ** 2
                            + (sigma_ne / np.clip(ne, 1e10, None)) ** 2)
                    _sigma_zc_rule_source = ("n_12C6_err (validity screened "
                                             "inside the boundary only)")

        q_ida = None
        if "q" in f:
            _q = np.asarray(f["q"][t_idx], dtype=float)
            q_ida = _q if _q.ndim == 1 else (
                np.median(_q, axis=0) if ensemble_median else np.mean(_q, axis=0))

        # Zeff-vs-carbon consistency (report-only, issue-#19-adjacent QC):
        # the file reports Zeff from visible bremsstrahlung AND n_12C6 from
        # CER; under the same single-impurity assumption they must agree as
        # Zeff = 1 + Z(Z-1) n_C/n_e.  Callahan 2019 (JINST 14 C10002) shows
        # the two techniques agree at DIII-D when C6+ dominates, so a large
        # deviation flags THIS file (non-carbon impurity, calibration, or
        # fit vintage), not the method.  No acceptance criterion -- one line.
        zeff_carbon_dev = None
        if "n_12C6" in f and Zeff is not None:
            n_c_raw = np.asarray(f["n_12C6"][t_idx], dtype=float)
            n_c = n_c_raw.mean(0) if n_c_raw.ndim == 2 else n_c_raw
            # Restrict to VALID core points: an edge-only fit vintage
            # (no psi_N <= 0.9 at all) previously crashed the argmax on an
            # empty array -- from a check documented as report-only -- and
            # negative/fill-value nC would make the "deviation" meaningless.
            _core = ((psi_N <= 0.9) & np.isfinite(n_c) & (n_c > 0)
                     & (np.abs(n_c) < _CARBON_FILL)
                     & np.isfinite(ne) & (ne > 0))
            if np.any(_core):
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_from_c = (1.0 + impurity_Z * (impurity_Z - 1.0) * n_c
                                / np.clip(ne, 1e10, None))
                    _rd = (Zeff[_core] - z_from_c[_core]) / np.clip(
                        z_from_c[_core], 1e-3, None)
                _iw = int(np.argmax(np.abs(_rd)))
                zeff_carbon_dev = {
                    "median": float(np.median(_rd)),
                    "worst": float(_rd[_iw]),
                    "psi_worst": float(psi_N[_core][_iw]),
                }
                print(f"[read_ida] Zeff(VB) vs 1+Z(Z-1)nC/ne (CER carbon), "
                      f"psi_N<=0.9: "
                      f"median {100*zeff_carbon_dev['median']:+.1f}%, "
                      f"worst {100*zeff_carbon_dev['worst']:+.1f}% at "
                      f"psi_N={zeff_carbon_dev['psi_worst']:.2f}  "
                      f"(report-only; see Callahan 2019 JINST 14 C10002)")
            else:
                print("[read_ida] Zeff-vs-carbon cross-check skipped: no "
                      "valid core (psi_N<=0.9) carbon points in this file")

        # The Z_eff source rule (bouquet.io.zeff_sources): evaluated on every
        # route but "standard" (which applies no rule, bit for bit as
        # before); it decides the source on "checked" and is recorded beside
        # the forced routes.
        zeff_rule = None
        _rule_reused = zeff_rule_from is not None
        if _rule_reused:
            # evaluated once, on the profile file's own read: reused as is
            zeff_rule = dict(zeff_rule_from)
            zeff_rule["reused"] = (
                "not re-evaluated on this read: the record of an earlier "
                "read of the profile file (zeff_rule_from)")
        elif _rule_on:
            zeff_rule = _zeff_rule_for_slice(
                f, t_idx, t_sel, time_ms, psi_N=psi_N, ne=ne, zeff=Zeff,
                sigma_zeff=sigma_Zeff, n_carbon=n_carbon,
                sigma_n_carbon=sigma_n_carbon,
                sigma_zeff_carbon_rule=_sigma_zc_rule,
                validity=_rule_validity, impurity_Z=impurity_Z,
                q=q_ida, is_ensemble=is_ensemble, ni_source=ni_source,
                override=zeff_override, fit_points=zeff_fit_points,
                raw_points=zeff_raw_points, fast_density=zeff_fast_density,
                ensemble_median=ensemble_median, crosschecks=zeff_crosschecks,
                path=path)

        if ni_source == "standard":
            # The established route (as before PR #56): the VB Z_eff as
            # stored, ni from it at the single-impurity window [1, Z] (so
            # 0 <= ni <= ne), sigma_ni the ne fraction.  The Z_eff envelope
            # is the file's own; the carbon > VB > scalar choice is made by
            # baseline.resolve_zeff_envelope.
            _substituted = Zeff is None
            if _substituted:
                # the IDA-hybrid dd route on a file with no Zeff: the caller's
                # Z_eff (the dd's) in its place, stamped below
                Zeff = np.asarray(zeff_substitute(psi_N), dtype=float)
            ni = main_ion_density_from_zeff(
                ne, np.clip(Zeff, 1.0, impurity_Z), impurity_Z)
            with np.errstate(divide="ignore", invalid="ignore"):
                _frac = np.where(ne > 0, sigma_ne / ne, 0.0)
            sigma_ni = np.abs(ni) * _frac
            sigma_Zeff_source = ("ensemble-samples" if is_ensemble else
                                 "Zeff_err" if sigma_Zeff is not None
                                 else "none")
            ni_route_chi = zeff_route_chi = zeff_dne = None
            zeff_provenance = dict(
                convention=("the caller's substitute (the file has no Zeff)"
                            if _substituted else
                            "measured (VB bremsstrahlung; includes any fast "
                            "ions the measurement sees)"),
                source=("dd (substitute: the file has no Zeff)"
                        if _substituted else "VB"), ni_source="standard",
                weights=dict(VB=1.0, CER=0.0),
                window=[1.0, float(impurity_Z)],
                n_clipped_vb=int(np.count_nonzero(
                    (np.asarray(Zeff, dtype=float) < 1.0)
                    | (np.asarray(Zeff, dtype=float) > impurity_Z))),
                n_clipped_cer=0, n_nodes=int(np.size(ne)),
                impurity_Z=float(impurity_Z),
                note=("Zeff is the stored VB value (not clipped); the window "
                      "applies to the ni derivation only"))
            return IDAProfiles(
                psi_N=psi_N, ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
                sigma_ne=sigma_ne, sigma_te=sigma_te, sigma_ni=sigma_ni,
                sigma_ti=sigma_ti, time=t_sel,
                raw_bytes=raw_bytes, sigma_Zeff=sigma_Zeff,
                sigma_Zeff_source=sigma_Zeff_source,
                sigma_Zeff_carbon=sigma_Zeff_carbon,
                sigma_Zeff_carbon_source=sigma_Zeff_carbon_source,
                zeff_carbon_dev=zeff_carbon_dev,
                ni_route_chi=None, zeff_route_chi=None, zeff_dne=None,
                q=q_ida, zeff_provenance=zeff_provenance)

        # ---- EXPERIMENTAL (PR #56): the Z_eff / n_i ladder VB+CER > CER > VB
        # Zeff_CER = 1 + Z(Z-1) nC/ne <=> ni = ne - Z nC, so ni is taken from
        # the resolved Z_eff and (ne, ni, Z_eff) stay quasineutral.  ne, Zeff_VB
        # and nC are independent (cov = 0); ne's two derivatives are summed.
        zeff_vb = Zeff                       # the file's VB measurement
        zeff_cer = None
        if n_carbon is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                zeff_cer = 1.0 + (impurity_Z * (impurity_Z - 1.0)
                                  * n_carbon / np.clip(ne, 1e10, None))

        # "checked" naming the carbon density: when the whole-grid screen
        # dropped its envelope for values outside the boundary only (the
        # rule's inside-boundary screen passed), the envelope screened
        # inside the boundary is used -- the forced "CER" route is unchanged
        _carbon_env_inside = False
        if (ni_source == "checked" and sigma_Zeff_carbon is None
                and _sigma_zc_rule is not None
                and zeff_rule["decision"]["source"] == _zs.IMPURITY):
            sigma_Zeff_carbon = _sigma_zc_rule
            sigma_Zeff_carbon_source = _sigma_zc_rule_source
            _carbon_env_inside = True

        # A route is LIVE only with a value AND a validated envelope.
        # sigma_Zeff_carbon carries the carbon screen's verdict (see
        # _carbon_tier_usable), so the ladder and the tier share ONE screen.
        _vb_ok = sigma_Zeff is not None and zeff_vb is not None
        _cer_ok = (zeff_cer is not None and sigma_Zeff_carbon is not None
                   and sigma_n_carbon is not None
                   and float(impurity_Z) == 6.0)
        # "checked": the rule's source, through the same arithmetic as the
        # forced route it names ("CER" / "Zeff"); without that source's
        # envelope (an override only) the bottom rung below takes its value
        # with the ne-fraction sigma
        _prefer_cer = False
        if ni_source == "checked":
            _src = zeff_rule["decision"]["source"]
            _prefer_cer = _src == _zs.IMPURITY
            use_vb = (not _prefer_cer) and _vb_ok
            use_cer = _prefer_cer and _cer_ok
        elif ni_source == "Zeff":
            use_vb, use_cer = True, False
        elif ni_source == "CER":
            if not _cer_ok:
                raise KeyError(
                    f"{path!r}: ni_source='CER' was asked for, but the carbon "
                    "route has no usable envelope here (no 'n_12C6' / "
                    "'n_12C6_err', or the carbon validity screen rejected "
                    "the data); pass ni_source='Zeff' or 'all'")
            use_vb, use_cer = False, True
        else:
            use_vb, use_cer = _vb_ok, _cer_ok

        # Bottom rung: neither route has a usable envelope (the oldest
        # vintages carry no *_err): take the VB value, ne-fraction sigma --
        # or, with no VB value at all, the carbon value.
        zeff_sigma_from_ne = not (use_vb or use_cer)
        zeff_from_carbon = False
        if zeff_sigma_from_ne:
            if zeff_vb is not None and not _prefer_cer:
                use_vb = True
            elif zeff_cer is not None:
                # No Zeff dataset: Z_eff DERIVED from the carbon density.
                # The same two checks the carbon route enforces (review F3):
                # the carbon validity screen (run here, without sigmas, when
                # the file has no n_12C6_err to run it with) and a carbon
                # impurity (Z_eff = 1 + Z(Z-1) n_C/n_e holds for the C6+
                # density only).  Either failing: refused, like a file with
                # no Z_eff information at all.
                _why = None
                if float(impurity_Z) != 6.0:
                    _why = (f"impurity_Z={float(impurity_Z):g}, but the "
                            "file's impurity density is carbon ('n_12C6', "
                            "Z = 6)")
                elif ni_source != "checked" and sigma_Zeff_carbon is None and (
                        _carbon_screened
                        or not _carbon_tier_usable(
                            n_carbon, ne, unit="radii",
                            datasets="n_12C6/n_e (no n_12C6_err)")):
                    _why = ("the carbon validity screen rejected its "
                            "'n_12C6' data (negative, non-finite or fill "
                            "values)")
                if _why is not None and ni_source == "checked":
                    raise _zs.ZeffSourceRefused(
                        f"{path!r}: the Z_eff source rule (ni_source="
                        "'checked') names the carbon density, but it cannot "
                        f"be used because {_why}; pass ni_source='standard' "
                        "(the file's Z_eff as stored, no rule)", zeff_rule)
                if _why is not None:
                    raise KeyError(
                        f"{path!r} carries no usable Z_eff information: no "
                        f"'Zeff' dataset, and Z_eff cannot be derived from "
                        f"the carbon density because {_why}; "
                        f"ni_source={ni_source!r} has no Z_eff to resolve. "
                        "Take the dd's Z_eff explicitly "
                        "(ImasSource.ni_source='standard') or use a file "
                        "with a Zeff dataset")
                use_cer = True
                zeff_from_carbon = True
                import warnings
                _lead = (f"read_ida: {path!r} has no 'Zeff' dataset; Z_eff is "
                         if zeff_vb is None else
                         f"read_ida: {path!r}: the carbon source named by "
                         "the Z_eff source rule has no usable envelope "
                         "here; Z_eff is ")
                warnings.warn(
                    _lead +
                    "DERIVED from the carbon density (1 + Z(Z-1) n_C/n_e, "
                    "the validity screen passed, Z = 6) with an n_e-fraction "
                    "sigma (zeff_provenance['zeff_source'] = "
                    "'carbon_density')", UserWarning, stacklevel=2)
            else:
                raise KeyError(
                    f"{path!r} carries no Z_eff information (no 'Zeff' and no "
                    f"'n_12C6' dataset): ni_source={ni_source!r} has no "
                    "Z_eff to resolve")

        w_v = 0.5 if (use_vb and use_cer) else (1.0 if use_vb else 0.0)
        w_c = 0.5 if (use_vb and use_cer) else (1.0 if use_cer else 0.0)
        zeff_tier = ("VB+CER" if (use_vb and use_cer)
                     else "CER" if use_cer else "VB")
        if zeff_sigma_from_ne:
            zeff_tier += " (ne-fraction sigma)"
        if ni_source == "all" and not (use_vb and use_cer):
            print(f"[read_ida] ni_source='all' -> Z_eff/ni tier "
                  f"'{zeff_tier}': the other route has no usable envelope in "
                  "this file, so no route-difference term enters sigma_ni")

        # Resolved Z_eff, then ni from it.  Each route is clamped to the
        # single-impurity window before the mean (Zeff_CER <= Z <=> ne - Z nC
        # >= 0), so ni(mean(Zeff)) == mean of the per-route ni stays exact.
        zeff_res = ((w_v * np.clip(zeff_vb, 1.0, impurity_Z) if w_v else 0.0)
                    + (w_c * np.clip(zeff_cer, 1.0, impurity_Z) if w_c else 0.0))
        Zeff = np.clip(zeff_res, 1.0, impurity_Z)
        ni = main_ion_density_from_zeff(ne, Zeff, impurity_Z)

        def _n_out(z):          # nodes a route is clamped at (NaN: not counted)
            z = np.asarray(z, dtype=float)
            return int(np.count_nonzero((z < 1.0) | (z > impurity_Z)))
        zeff_provenance = dict(
            convention=("thermal+impurity (measured: VB bremsstrahlung and/or "
                        "CER carbon; includes any fast ions the measurement "
                        "sees)"),
            source=("VB+CER mean" if (w_v and w_c) else
                    "CER" if w_c else "VB"),
            ni_source=str(ni_source),
            weights=dict(VB=float(w_v), CER=float(w_c)),
            window=[1.0, float(impurity_Z)],
            n_clipped_vb=_n_out(zeff_vb) if w_v else 0,
            n_clipped_cer=_n_out(zeff_cer) if w_c else 0,
            n_nodes=int(np.size(ne)),
            impurity_Z=float(impurity_Z),
            # where the resolved Z_eff comes from: the file's Zeff dataset,
            # the carbon density, or both (a file without Zeff derives it
            # from n_12C6 -- the bottom rung stamps zeff_from_carbon)
            zeff_source=("carbon_density" if not w_v else
                         "Zeff+carbon_density" if w_c else "Zeff"),
            zeff_from_carbon_fallback=bool(zeff_from_carbon),
        )
        if ni_source == "checked":
            # the rule that chose the source, archived with the route
            zeff_provenance["zeff_rule"] = zeff_rule
            zeff_provenance["zeff_override"] = (
                zeff_override if not _rule_reused else
                {_zs.IMPURITY: "CER", _zs.LINE: "Zeff"}.get(
                    (zeff_rule.get("override") or {}).get("source")))
            if _carbon_env_inside:
                zeff_provenance["carbon_envelope_screen"] = (
                    "inside the boundary only (psi_N <= 1): the whole-grid "
                    "carbon screen rejected values outside the boundary")
        if zeff_rule is not None and not _rule_reused:
            _zeff_rule_warn(
                zeff_rule, ni_source, t_sel,
                ([_zs.IMPURITY] if w_c else []) + ([_zs.LINE] if w_v else []),
                zeff_provenance)

        _inv = 1.0 / (impurity_Z - 1.0)
        # dZeff_res/dne comes from the CER route alone (VB's Z_eff has no ne
        # dependence); dni/dne then sums the explicit factor with it.
        d_ni_dne = (impurity_Z - zeff_res) * _inv
        if w_c:
            d_ni_dne = d_ni_dne + w_c * (zeff_cer - 1.0) * _inv

        # Z_eff channel: sigma_Zeff_carbon is ALREADY the CER route's total
        # (nC (+) ne), so it enters whole here.  ni must NOT reuse it -- ne
        # is carried there by d_ni_dne instead.
        z_terms, n_terms = [], []
        if w_v and sigma_Zeff is not None:
            z_terms.append(w_v * sigma_Zeff)
            n_terms.append(w_v * ne * _inv * sigma_Zeff)
        if w_c and sigma_Zeff_carbon is not None:
            z_terms.append(w_c * sigma_Zeff_carbon)
            n_terms.append(w_c * impurity_Z * sigma_n_carbon)
        n_terms.append(d_ni_dne * sigma_ne)
        var_zeff = sum(t ** 2 for t in z_terms) if z_terms else np.zeros_like(ne)
        var_ni = sum(t ** 2 for t in n_terms)

        # Route disagreement, folded into both envelopes.  Both routes are GP
        # fits, so a nonzero difference is a coherent offset in psi_N, not
        # point-to-point scatter.  max(., 0) keeps the term one-sided; the
        # resolved value is the mean of two routes, so an offset delta
        # displaces it by delta/2 -> variance excess /4.  ne cancels partly in
        # ni and not at all in Z_eff, so each delta gets its own Jacobian.
        ni_route_chi = zeff_route_chi = None
        if use_vb and use_cer:
            d_z = zeff_vb - zeff_cer
            var_dz = sigma_Zeff ** 2 + sigma_Zeff_carbon ** 2
            var_zeff = var_zeff + np.maximum(d_z ** 2 - var_dz, 0.0) / 4.0
            zeff_route_chi = _chi(d_z, var_dz)

            ni_vb = main_ion_density_from_zeff(
                ne, np.clip(zeff_vb, 1.0, impurity_Z), impurity_Z)
            ni_cer = np.maximum(ne - impurity_Z * n_carbon, 0.0)
            d_n = ni_vb - ni_cer
            var_dn = (((impurity_Z - np.clip(zeff_vb, 1.0, impurity_Z))
                       * _inv - 1.0) * sigma_ne) ** 2 \
                + (ne * _inv * sigma_Zeff) ** 2 \
                + (impurity_Z * sigma_n_carbon) ** 2
            var_ni = var_ni + np.maximum(d_n ** 2 - var_dn, 0.0) / 4.0
            ni_route_chi = _chi(d_n, var_dn)

        zeff_dne = (-w_c * (zeff_cer - 1.0) / np.clip(ne, 1e10, None)
                    if w_c else np.zeros_like(ne))

        if zeff_sigma_from_ne:
            # Nothing to propagate: both channels inherit ne's fractional
            # error.
            with np.errstate(divide="ignore", invalid="ignore"):
                _frac = np.where(ne > 0, sigma_ne / ne, 0.0)
            var_ni = (np.abs(ni) * _frac) ** 2
            var_zeff = (np.abs(Zeff) * _frac) ** 2
            zeff_dne = np.zeros_like(ne)

        sigma_ni = np.sqrt(var_ni)
        sigma_Zeff = np.sqrt(var_zeff)
        sigma_Zeff_source = zeff_tier

    return IDAProfiles(
        psi_N=psi_N,
        ne=ne, te=te, ni=ni, ti=ti, Zeff=Zeff,
        sigma_ne=sigma_ne, sigma_te=sigma_te, sigma_ni=sigma_ni, sigma_ti=sigma_ti,
        time=t_sel,
        raw_bytes=raw_bytes,
        sigma_Zeff=sigma_Zeff,
        sigma_Zeff_source=sigma_Zeff_source,
        sigma_Zeff_carbon=sigma_Zeff_carbon,
        sigma_Zeff_carbon_source=sigma_Zeff_carbon_source,
        zeff_carbon_dev=zeff_carbon_dev,
        ni_route_chi=ni_route_chi,
        zeff_route_chi=zeff_route_chi,
        zeff_dne=zeff_dne,
        q=q_ida,
        zeff_provenance=zeff_provenance,
        zeff_rule=zeff_rule,
    )


def read_ida_cer(
    path: str,
    time: Optional[float] = None,
    sigma_method: str = "percentile",
    ensemble_median: bool = False,
) -> IDACERProfiles:
    """Read the carbon-CER channels needed for a radial-field (E_r) analysis.

    Returns the impurity density / temperature, toroidal + poloidal rotation, the
    midplane poloidal field and geometry (``Rmaj``, ``dpsiN_dR``), and their
    measured 1-sigma envelopes, at ``time`` on the IDA ``psi_N`` grid. Handles
    both file layouts like :func:`read_ida`: direct (2-D + ``*_err``) and ensemble
    (3-D posterior samples -> central profile + ``sigma_method`` band). Feed the
    result to :func:`bouquet.physics.radial_field_from_impurity_force_balance`.
    ``ensemble_median`` matches :func:`read_ida`; set both alike.
    """
    import h5py

    if sigma_method not in ("percentile", "std"):
        raise ValueError(f"unknown sigma_method {sigma_method!r}")

    with h5py.File(path, "r") as f:
        time_ms = np.asarray(f["time"][:], dtype=float).ravel()
        t_idx = _select_time_index(time_ms, time)
        t_sel = float(time_ms[t_idx] / 1e3)
        is_ensemble = (np.asarray(f["n_e"].shape).size == 3)

        def _band(a):
            if sigma_method == "std":
                return np.std(a, axis=0)
            lo, hi = np.percentile(a, [16.0, 84.0], axis=0)
            return 0.5 * (hi - lo)

        def _center(a):
            return np.median(a, axis=0) if ensemble_median else np.mean(a, axis=0)

        def _pick(*names):
            """First of ``names`` present in the file, else None.

            DIII-D IDA files suffix the CER channels with the measured ion, e.g.
            ``v_pol_12C6``; older/synthetic files use the bare ``v_pol``.
            """
            return next((n for n in names if n in f), None)

        def read(key, err_key=None):
            """(value, sigma) for one channel across either layout.

            ``key=None`` (channel absent from the file) zero-fills rather than
            raising -- only v_pol is optional enough to reach here.
            """
            if key is None:
                zero = np.zeros_like(psi_N)
                return zero, zero.copy()
            if is_ensemble:
                s = np.asarray(f[key][t_idx], dtype=float)      # (n_samples, n_radial)
                return _center(s), _band(s)
            val = np.asarray(f[key][t_idx], dtype=float)
            sig = (np.asarray(f[err_key][t_idx], dtype=float)
                   if err_key and err_key in f else np.zeros_like(val))
            return val, sig

        if is_ensemble:
            psi_N = np.asarray(f["psi_n"][t_idx], dtype=float)[0]
        else:
            psi_N = np.asarray(f["psi_n"][:], dtype=float)

        missing = [k for k in ("n_12C6", "T_12C6", "omega_tor_12C6") if k not in f]
        if missing:
            raise KeyError(
                f"{path!r} has no {', '.join(missing)}: this file carries no carbon-CER "
                "measurement, so rotation / E_r cannot be derived from it")

        n_c, s_nc = read("n_12C6", "n_12C6_err")
        t_c, s_tc = read("T_12C6", "T_12C6_err")
        omg, s_om = read("omega_tor_12C6", "omega_tor_12C6_err")
        vpol, s_vp = read(_pick("v_pol_12C6", "v_pol"), _pick("v_pol_12C6_err", "v_pol_err"))
        bpol, _ = read("Bpol_midplane", "Bpol_midplane_err")
        rmaj, _ = read("Rmaj_midplane", "Rmaj_midplane_err")
        dpsidr, _ = read("dPsiN_dR_midplane", "dPsiN_dR_midplane_err")

    return IDACERProfiles(
        psi_N=psi_N, n_carbon=n_c, t_carbon=t_c, omega_tor=omg, v_pol=vpol,
        Bpol=-bpol, Rmaj=rmaj, dpsiN_dR=dpsidr,
        sigma_n_carbon=s_nc, sigma_t_carbon=s_tc,
        sigma_omega_tor=s_om, sigma_v_pol=s_vp,
        time=t_sel,
    )
