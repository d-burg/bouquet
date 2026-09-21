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

There is no stored main-ion density; ``ni`` can come from ``Zeff``
reconstructed from visible bremsstrahlung data (``ni_source="Zeff"``), from
the measured carbon density ``n_12C6`` from charge exchange recombination
(``ni_source="CER"``), or from the mean of the two (``ni_source="all"``, default).
With both active, their disagreement beyond statistical error widens
``sigma_ni``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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

    #: Measured 1-sigma envelope on ``Zeff``, when the file provides one:
    #: sample spread on the ensemble layout, the ``Zeff_err`` dataset on newer
    #: direct-layout files.  ``None`` on older direct files that carry no Zeff
    #: uncertainty -- the envelope resolver then falls back to
    #: ``UncertaintyConfig.zeff_scalar_sigma`` (measured values run ~8-9 %
    #: median in-core vs the 5 % scalar default, so the tiers are NOT
    #: interchangeable; the resolver logs which one won).
    sigma_Zeff: Optional[np.ndarray] = None
    #: Provenance of ``sigma_Zeff``: "ensemble-samples" | "Zeff_err" | "none".
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


@dataclass
class IDACERProfiles:
    """Impurity (carbon CER) channels for a radial-field / rotation analysis.

    Everything needed for the impurity radial force balance
    ``E_r = (dp_C/dR)/(Z_C e n_C) - v_pol B_phi + omega R B_pol`` (see
    :func:`bouquet.physics.radial_field_from_impurity_force_balance`), read at one time slice on
    ``psi_N``. SI units: ``n_carbon`` m^-3, ``t_carbon`` eV, ``omega_tor`` rad/s,
    ``v_pol`` m/s, ``Bpol`` T, ``Rmaj`` m, ``dpsiN_dR`` 1/m. The ``sigma_*``
    fields are the measured 1-sigma envelopes (for propagating E_r uncertainty).
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
    ni_source: str = "all",
    impurity_Z: float = 6.0,
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
    ni_source : {"Zeff", "CER", "all"}
        Which measurement the main-ion density comes from. ``"Zeff"``
        applies single-impurity quasineutrality to ``(ne, Zeff)``; ``"CER"``
        subtracts the measured carbon density ``n_12C6``; ``"all"`` takes the
        mean of the two (default). An explicit route needs its own ``*_err``
        dataset on the direct layout and raises without it; ``"all"`` uses
        whichever routes the file supports.
    impurity_Z : float
        Impurity charge Z (carbon Z=6).

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

    if sigma_mode not in ("direct", "ensemble", "auto"):
        raise ValueError(
            f"unknown sigma_mode {sigma_mode!r}; expected 'direct', 'ensemble', or 'auto'")
    if sigma_method not in ("percentile", "std"):
        raise ValueError(
            f"unknown sigma_method {sigma_method!r}; expected 'percentile' or 'std'")
    if ni_source not in ("Zeff", "CER", "all"):
        raise ValueError(
            f"unknown ni_source {ni_source!r}; expected 'Zeff', 'CER', or 'all'")
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
        # An EXPLICIT ni_source is strict: asking for a named route and
        # silently getting the other one is worse than an error.  The default
        # "all" walks the ladder below instead, once the data is in hand --
        # route availability turns on the carbon validity screen, not merely
        # on which datasets exist.
        if ni_source == "CER" and "n_12C6" not in f:
            raise KeyError(
                f"{path!r} has no 'n_12C6' dataset, so ni cannot be derived from "
                "the carbon density; pass ni_source='Zeff' to use the "
                "(ne, Zeff) quasineutrality route instead")
        if ni_source == "CER" and not is_ensemble and "n_12C6_err" not in f:
            raise KeyError(
                f"{path!r} has 'n_12C6' but no 'n_12C6_err', so the carbon term of "
                "sigma_ni cannot be propagated; pass ni_source='Zeff' to use "
                "the (ne, Zeff) quasineutrality route instead")
        if ni_source == "Zeff" and not is_ensemble and "Zeff_err" not in f:
            raise KeyError(
                f"{path!r} has no 'Zeff_err' dataset, so the Zeff term of sigma_ni "
                "cannot be propagated; pass ni_source='CER' to derive ni from "
                "the carbon density instead")

        # The CER route's value + envelope, filled by whichever layout
        # branch runs below; None when the file carries no carbon channel.
        n_carbon = sigma_n_carbon = None

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
            ti_s, zf_s = _samples("T_12C6"), _samples("Zeff")
            ne, te, ti, Zeff = (_center(ne_s), _center(te_s),
                                _center(ti_s), _center(zf_s))
            sigma_ne, sigma_te, sigma_ti = _band(ne_s), _band(te_s), _band(ti_s)
            # Highest-fidelity tier: the posterior carries Zeff samples, so the
            # Zeff envelope is measured the same way as the kinetic ones.
            sigma_Zeff = _band(zf_s)
            sigma_Zeff_source = "ensemble-samples"
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
                if _carbon_tier_usable(nc_s, ne_s, unit="sample points",
                                       datasets="n_12C6/n_e samples"):
                    zc_s = (1.0 + impurity_Z * (impurity_Z - 1.0) * nc_s
                            / np.clip(ne_s, 1e10, None))
                    sigma_Zeff_carbon = _band(zc_s)
                    sigma_Zeff_carbon_source = "ensemble-samples"
        else:
            def col(key):       # one radial profile at the selected time
                return np.asarray(f[key][t_idx], dtype=float)

            psi_N = np.asarray(f["psi_n"][:], dtype=float)
            ne, te = col("n_e"), col("T_e")          # m^-3, eV
            ti, Zeff = col("T_12C6"), col("Zeff")    # eV (carbon CER), dimensionless
            sigma_ne, sigma_te, sigma_ti = col("n_e_err"), col("T_e_err"), col("T_12C6_err")
            if "n_12C6" in f:
                n_carbon = col("n_12C6")
                sigma_n_carbon = (col("n_12C6_err") if "n_12C6_err" in f
                                  else None)
            # Newer direct-layout vintages carry a measured Zeff_err profile;
            # older ones do not.  None (NOT a guessed scalar) marks the older
            # vintage, which is what drops the VB rung of the ladder below.
            if "Zeff_err" in f:
                sigma_Zeff = col("Zeff_err")
                sigma_Zeff_source = "Zeff_err"
            else:
                sigma_Zeff = None
                sigma_Zeff_source = "none"
            sigma_Zeff_carbon, sigma_Zeff_carbon_source = None, "none"
            if "n_12C6" in f and "n_12C6_err" in f:
                _nc, _snc = col("n_12C6"), col("n_12C6_err")
                # The propagation is only meaningful where the carbon data is
                # physical; see _carbon_tier_usable for what "physical" means
                # here and why a single bad radius drops the tier.
                if _carbon_tier_usable(_nc, ne, sigma_nC=_snc, sigma_ne=sigma_ne):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        _dil = (impurity_Z * (impurity_Z - 1.0) * _nc
                                / np.clip(ne, 1e10, None))
                        sigma_Zeff_carbon = _dil * np.sqrt(
                            (_snc / np.clip(_nc, 1e10, None)) ** 2
                            + (sigma_ne / np.clip(ne, 1e10, None)) ** 2)
                    sigma_Zeff_carbon_source = "n_12C6_err"

        # Zeff-vs-carbon consistency (report-only, issue-#19-adjacent QC):
        # the file reports Zeff from visible bremsstrahlung AND n_12C6 from
        # CER; under the same single-impurity assumption they must agree as
        # Zeff = 1 + Z(Z-1) n_C/n_e.  Callahan 2019 (JINST 14 C10002) shows
        # the two techniques agree at DIII-D when C6+ dominates, so a large
        # deviation flags THIS file (non-carbon impurity, calibration, or
        # fit vintage), not the method.  No acceptance criterion -- one line.
        # It is also the spread that ni_source="all" folds into sigma_ni below.
        zeff_carbon_dev = None
        if "n_12C6" in f:
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

        # ---- the Z_eff / n_i ladder: VB+CER > CER > VB -------------------
        # ni is linear in Z_eff and the CER route is the same measurement in
        # other coordinates:
        #     Zeff_CER = 1 + Z(Z-1) nC/ne   <=>   ni = ne - Z nC   (exact)
        # so resolving Z_eff route-wise and taking ni from the result is the
        # same ni as resolving ni route-wise, and keeps the archived
        # (ne, ni, Z_eff) mutually quasineutral -- the invariant the p-file
        # export and the per-draw dilution assume.
        #
        # Independent measured variables: ne, Zeff_VB, nC, with cov = 0
        # between them (IDA stores none; VB and CER are separate
        # diagnostics).  ne enters ni both explicitly and through the CER
        # route, so its two derivatives are summed before squaring.
        zeff_vb = Zeff                       # the file's VB measurement
        zeff_cer = None
        if n_carbon is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                zeff_cer = 1.0 + (impurity_Z * (impurity_Z - 1.0)
                                  * n_carbon / np.clip(ne, 1e10, None))

        # A route is LIVE only with a value AND a validated envelope.
        # sigma_Zeff_carbon carries the carbon screen's verdict (see
        # _carbon_tier_usable), so the ladder and the tier share ONE screen.
        _vb_ok = sigma_Zeff is not None
        _cer_ok = (zeff_cer is not None and sigma_Zeff_carbon is not None
                   and sigma_n_carbon is not None
                   and float(impurity_Z) == 6.0)
        if ni_source == "Zeff":
            use_vb, use_cer = True, False
        elif ni_source == "CER":
            if not _cer_ok:
                raise KeyError(
                    f"{path!r}: ni_source='CER' was asked for, but the carbon "
                    "route has no usable envelope here (no 'n_12C6_err', "
                    "impurity_Z != 6, or the carbon validity screen rejected "
                    "the data); pass ni_source='Zeff' or 'all'")
            use_vb, use_cer = False, True
        else:
            use_vb, use_cer = _vb_ok, _cer_ok

        # Bottom rung: neither route has a usable envelope (the oldest
        # vintages carry no *_err).  The dilution value is still there, so
        # take it and fall back to an ne-fraction sigma.
        zeff_sigma_from_ne = not (use_vb or use_cer)
        if zeff_sigma_from_ne:
            if zeff_vb is not None:
                use_vb = True
            elif zeff_cer is not None:
                use_cer = True
            else:
                raise KeyError(
                    f"{path!r} carries neither a 'Zeff' nor a usable 'n_12C6' "
                    "dilution measurement, so ni cannot be derived at all")

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

        # Resolved Z_eff, then ni from it.  The single-impurity clamp applies
        # once, to the mean, so ni(mean(Zeff)) == mean(ni) stays exact.
        zeff_res = (w_v * zeff_vb if w_v else 0.0) + (w_c * zeff_cer if w_c else 0.0)
        Zeff = np.clip(zeff_res, 1.0, impurity_Z)
        ni = main_ion_density_from_zeff(ne, Zeff, impurity_Z)

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
        Bpol=bpol, Rmaj=rmaj, dpsiN_dR=dpsidr,
        sigma_n_carbon=s_nc, sigma_t_carbon=s_tc,
        sigma_omega_tor=s_om, sigma_v_pol=s_vp,
        time=t_sel,
    )
