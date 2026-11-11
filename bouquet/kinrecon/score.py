"""Score any solved TokaMaker equilibrium against a measurement set.

:func:`score` is the forward model (layer L1): it evaluates the synthetic
diagnostics of :mod:`.diagnostics` on an equilibrium and returns a
:class:`~.interfaces.ResidualTable`.  It fits nothing and changes nothing: the
equilibrium's flux, coil currents and profiles are only read (the tests check
bit-identity of ``psi`` and the coil currents before and after).

Normalized residual and chi^2 are those of :class:`~.interfaces.ResidualTable`:
``r_i = w_i (model_i - measured_i) / sigma_i`` over valid channels and
``chi^2 = sum r_i^2`` -- diagonal Gaussian likelihood, the definition of
IMAS ``equilibrium.time_slice[:].constraints.*.chi_squared`` and of OFT's
reconstruction error rows (``err = (val - model)*wt`` with ``wt = 1/sigma``,
``grad_shaf_fit.F90``), up to the sign of ``r``.

Channels that are valid in the measurement set but cannot be modeled (off the
mesh, a Thomson channel outside the LCFS, a non-finite model) are removed from
the comparison through the table's per-group ``model_valid`` mask and listed
in ``table.excluded[group]`` as ``(label, reason)``; table-wide notes (groups
skipped, MSE E_r treatment) are in ``table.notes``.  The measurement set is
neither copied nor modified: ``table.measurements`` is the input object.
"""

import numpy as np

from . import diagnostics as _diag
from .interfaces import MeasurementSet, ResidualTable

#: Groups :func:`score` can model, in evaluation order.
SCORED_GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current",
                 "mse", "thomson_ne", "thomson_te")
#: Groups with stub diagnostics; requesting them raises NotImplementedError.
STUB_GROUPS = ("q", "vloop")
#: Thomson groups and the ``kinetics`` profile each needs.
_THOMSON_PROFILE = {"thomson_ne": "n_e", "thomson_te": "t_e"}


def _model_group(eq, name, group, kinetics, notes):
    """``Synthetic`` for one group, applying its meta and the kinetics."""
    geo = group.geometry
    if name == "flux_loop":
        ref = group.meta.get("flux_reference")
        if ref is None:
            return _diag.flux_loop(eq, geo)
        if ref not in group.labels:
            raise KeyError(f"score: flux_reference '{ref}' is not a flux-loop label")
        i = group.labels.index(ref)
        notes.append(f"flux_loop: modeled relative to reference loop '{ref}'")
        return _diag.flux_loop(eq, geo, reference=(float(np.asarray(geo["r"])[i]),
                                                   float(np.asarray(geo["z"])[i])))
    if name == "bpol_probe":
        model = group.meta.get("probe_model", "finite_length")
        n_avg = int(group.meta.get("probe_n_avg", _diag.PROBE_N_AVG_DEFAULT))
        if model == "finite_length" and "length" in geo:
            notes.append(f"bpol_probe: finite-length probes (EFIT midpoint rule, n_avg = {n_avg})")
        elif model == "finite_length":
            notes.append("bpol_probe: no probe lengths in the geometry: point probes")
        else:
            notes.append("bpol_probe: point probes (meta['probe_model'] = 'point')")
        return _diag.bpol_probe(eq, geo, model=model, n_avg=n_avg)
    if name == "ip":
        return _diag.ip(eq, geo)
    if name == "diamagnetic_flux":
        return _diag.diamagnetic_flux(eq, geo)
    if name == "pf_current":
        return _diag.pf_current(eq, geo)
    if name == "mse":
        e_r = kinetics.get("e_r")
        e_z = kinetics.get("e_z")
        corrected = bool(group.meta.get("er_corrected", False))
        if corrected and e_r is not None:
            raise ValueError("score: the MSE group is already E_r-corrected "
                             "(meta['er_corrected']=True) and an E_r was supplied; "
                             "applying it would count E_r twice")
        if e_r is None:
            notes.append("mse: E_R = 0 in the model"
                         + (" (data declared E_r-corrected)" if corrected else
                            " and the data are NOT declared E_r-corrected: biased "
                            "if E_r is not small"))
        else:
            notes.append("mse: E_R from kinetics['e_r'] applied through c3 and c7")
        sp, st = group.meta.get("field_orientation", (1.0, 1.0))
        if (float(sp), float(st)) != (1.0, 1.0):
            notes.append(f"mse: stated field orientation sign_pol={sp:+g}, sign_tor={st:+g}")
        return _diag.mse(eq, geo, e_r=e_r, e_z=e_z, sign_pol=float(sp), sign_tor=float(st),
                         quantity=group.meta.get("quantity", "polarisation_angle"))
    if name in _THOMSON_PROFILE:
        prof = kinetics[_THOMSON_PROFILE[name]]
        return _diag.thomson(eq, geo, prof, outside=kinetics.get("thomson_outside", "exclude"))
    raise KeyError(name)


def score(equilibrium, measurements, kinetics=None, groups=None):
    """Residual table of *equilibrium* against *measurements*.

    Parameters
    ----------
    equilibrium : TokaMaker or TokaMaker_equilibrium
        A solved equilibrium; it is not modified.
    measurements : MeasurementSet
    kinetics : dict, optional
        ``'n_e'``, ``'t_e'``: callables of psi_N returning m^-3 and eV (needed
        for the Thomson groups); ``'e_r'``, ``'e_z'``: E field for MSE [V/m],
        a per-channel array or a callable ``f(r, z)``;
        ``'thomson_outside'``: ``'exclude'`` (default) or ``'extrapolate'``
        (see :func:`.diagnostics.thomson`).
    groups : iterable of str, optional
        Groups to score.  Default: every group present in *measurements* that
        can be modeled with what was given -- Thomson groups without their
        profile and the stub groups ``q``/``vloop`` are skipped with a note.
        Naming such a group explicitly raises instead.

    Group ``meta`` read here: flux-loop ``flux_reference`` (label of the
    reference loop; the model is then relative to it); probe ``probe_model``
    (``'finite_length'`` [default: length-averaged where the geometry carries
    ``length``] or ``'point'``) and ``probe_n_avg`` (default 25, EFIT
    ``NSMP2``); MSE ``er_corrected`` (bool), ``quantity``
    (``'polarisation_angle'`` [rad, default] or ``'tan_polarisation_angle'``)
    and ``field_orientation`` (``(sign_pol, sign_tor)``, a stated convention,
    default ``(1, 1)``).

    Returns
    -------
    ResidualTable
        ``model[name]`` aligned with the group's channels (NaN where not
        modeled), ``model_valid[name]`` False there, ``excluded`` and
        ``notes`` as described in the module docstring; ``measurements`` is
        the input set itself.
    """
    if not isinstance(measurements, MeasurementSet):
        raise TypeError("score: measurements must be a MeasurementSet")
    kinetics = dict(kinetics or {})
    explicit = groups is not None
    names = list(groups) if explicit else list(measurements.groups)
    notes = []
    todo = []
    for name in names:
        if name not in measurements.groups:
            raise KeyError(f"score: group '{name}' is not in the measurement set")
        if name in STUB_GROUPS:
            if explicit:
                raise NotImplementedError(f"score: group '{name}' has no diagnostic yet")
            notes.append(f"{name}: skipped (diagnostic not implemented)")
            continue
        if name not in SCORED_GROUPS:
            if explicit:
                raise KeyError(f"score: unknown group '{name}'")
            notes.append(f"{name}: skipped (unknown group)")
            continue
        if name in _THOMSON_PROFILE and _THOMSON_PROFILE[name] not in kinetics:
            if explicit:
                raise ValueError(f"score: group '{name}' needs kinetics['{_THOMSON_PROFILE[name]}']")
            notes.append(f"{name}: skipped (no kinetics['{_THOMSON_PROFILE[name]}'])")
            continue
        todo.append(name)

    model, model_valid, excluded = {}, {}, {}
    for name in todo:
        group = measurements.groups[name]
        syn = _model_group(equilibrium, name, group, kinetics, notes)
        value = np.asarray(syn.value, dtype=float)
        if value.shape != group.measured.shape:
            raise ValueError(f"score: group '{name}' modeled {value.shape} for "
                             f"{group.measured.shape} channels")
        ok = np.isfinite(value)
        drop = group.valid & ~ok
        if np.any(drop):
            excluded[name] = [(group.labels[i], str(syn.reason[i]) or _diag.REASON_NONFINITE)
                              for i in np.nonzero(drop)[0]]
            notes.append(f"{name}: {int(drop.sum())} valid channel(s) not modeled, "
                         "excluded from chi2 (see table.excluded)")
        model[name] = value
        model_valid[name] = ok
    return ResidualTable(measurements=measurements, model=model, model_valid=model_valid,
                         excluded=excluded, notes=notes)


def chi2_summary(table):
    """``{group: (chi2, n_valid, chi2/n)}`` plus ``'total'`` for a table."""
    out = {}
    for name in table.model:
        c, n = table.chi2(name), table.n_valid(name)
        out[name] = (c, n, c / n if n else float("nan"))
    c, n = table.chi2(), table.n_valid()
    out["total"] = (c, n, c / n if n else float("nan"))
    return out
