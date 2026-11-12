"""Score any solved TokaMaker equilibrium against a measurement set.

:func:`score` is the forward model: it evaluates the synthetic
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

Orientation (frames).  Every signed quantity is compared in the
DISCHARGE's frame.  The equilibrium may be solved in another frame (e.g. the
positive-Ip frame of a forward solve); the map between the two is STATED,
never fit: ``orientation = {"sign_ip": +-1, "sign_bt": +-1}`` multiplies the
modeled values of the groups that flip with the plasma current (Ip, flux
loops, poloidal-field probes, coil currents) and with the toroidal field
(diamagnetic flux), and enters MSE as ``(sign_pol, sign_tor)``.  The frame
used is recorded in ``table.frame``.  Scoring is refused when the
equilibrium's own Ip / B_phi sign, after the map, disagrees with the
discharge's (the measured Ip's sign, and ``meta["discharge_orientation"]``
when the set declares it) -- a sign error would otherwise show up only as
an enormous chi^2.

Comparing equilibria.  The channels that enter chi^2 can depend on the
equilibrium (off-mesh points, Thomson channels outside the LCFS), so totals of
different equilibria are not comparable unless they are taken over the same
channels: :func:`score_common` scores several equilibria on the intersection
of their modelable channels.  :func:`chi2_report` reports chi^2 with the
channel count and the degrees of freedom separately and names the sigma
conventions; on real data the sigma are conventions (e.g. EFIT's), not
standard deviations, so chi^2/dof is not a pass/fail quantity.
:func:`chi2_under_sigma_convention` gives the per-group (chi^2, N, chi^2/N)
and names the sigma convention of each group; every number it returns holds
only under those conventions.

Conventions (documented defaults of the scorer):

* a channel enters chi^2 when it is valid in the measurement set, model-valid
  and has a weight != 0; its sigma must then be finite and > 0
  (:meth:`.interfaces.ResidualTable.residual` raises otherwise);
* a channel that cannot be modeled is excluded with a recorded reason
  (``table.excluded``), never silently: off the mesh (point-in-triangle
  test), a finite-length probe whose segment leaves the mesh, a Thomson
  channel outside the LCFS (unless ``kinetics["thomson_outside"] =
  "extrapolate"``), a non-finite model value;
* no gauge is applied inside :func:`score` unless the group asks for one:
  flux loops are modeled in the form the data are stored in (relative to
  ``meta["flux_reference"]`` when it is named); with ``meta["flux_offset"] =
  "fitted"`` the loops are modeled absolutely plus ONE additive offset per
  table, solved in closed form by weighted least squares over the used
  channels (``c = sum w (measured - model) / sum w``, ``w = (weight /
  sigma)^2``; the offset is one fitted nuisance parameter, recorded in
  ``table.frame["flux_offset"]`` and in the notes) -- the reading of data
  stored relative to a reference offset that the producing fit itself fitted
  (EFIT's ``FITSIREF``), under which that fit's own fitted values reproduce
  its own chi^2.  With ``meta["flux_offset_prior"] = (c0, s0)`` (measurement
  units: the measured reference flux expressed as an offset, and the
  reference loop's sigma) the offset also carries that prior row, as EFIT's
  fit does: ``c = (sum w r + c0/s0^2) / (sum w + 1/s0^2)``; the prior row's
  chi^2 term ``((c - c0)/s0)^2`` is recorded in
  ``table.frame["flux_offset_prior_chi2"]`` (one extra datum, one extra
  parameter; it is not part of the per-channel residuals);
* E_r is never counted twice: an MSE group declared E_r-corrected with an
  E_r supplied is refused;
* an orientation mismatch between the equilibrium and the discharge is
  refused (above);
* the measurement set is neither copied nor changed; the adapters' own
  conventions (no-data sentinels, recorded exclusions, window reduction) are
  listed in :mod:`.adapters`.
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


#: Which orientation sign multiplies each group's modeled value (MSE handled separately; Thomson unsigned).
FRAME_SIGN_OF_GROUP = {"ip": "sign_ip", "flux_loop": "sign_ip", "bpol_probe": "sign_ip", "pf_current": "sign_ip",
                       "diamagnetic_flux": "sign_bt"}


def _equilibrium_signs(eq, need_ip=True, need_bt=True):
    """(sign of Ip, sign of B_phi) of a solved equilibrium (0 where not needed or, for B_phi, not available)."""
    s_ip = float(np.sign(float(eq.get_globals()[0]))) if need_ip else 0.0
    s_bt = 0.0
    if not need_bt:
        return s_ip, s_bt
    try:
        tm = _diag._tokamaker(eq)
        lim = np.asarray(getattr(tm, "lim_contour", None), dtype=float)
        if lim.ndim == 2 and lim.shape[0] > 2:
            c = lim[:, :2].mean(axis=0)
            b, _ = _diag.field_at(eq, c[None, :])
            bphi = float(np.asarray(b, dtype=float).reshape(-1, 3)[0, 1])
            s_bt = float(np.sign(bphi)) if np.isfinite(bphi) else 0.0
    except Exception:            # noqa: BLE001  (no B_phi available: the B_t check is skipped and noted)
        s_bt = 0.0
    return s_ip, s_bt


def resolve_orientation(eq, measurements, orientation=None):
    """The stated frame map and the checks of the module docstring ("Orientation").

    Returns ``{"sign_ip", "sign_bt", "source", "equilibrium_signs", "discharge_signs", "checks"}``; raises
    ``ValueError`` when the mapped equilibrium signs disagree with the discharge's.
    """
    if orientation is None:
        orientation = measurements.meta.get("orientation_map")
        src = "measurements.meta['orientation_map']" if orientation is not None else "identity (not stated)"
    else:
        src = "caller"
    orientation = dict(orientation or {})
    unknown = set(orientation) - {"sign_ip", "sign_bt"}
    if unknown:
        raise ValueError(f"orientation has unknown keys {sorted(unknown)}; use sign_ip, sign_bt")
    m_ip, m_bt = float(orientation.get("sign_ip", 1.0)), float(orientation.get("sign_bt", 1.0))
    if m_ip not in (1.0, -1.0) or m_bt not in (1.0, -1.0):
        raise ValueError("orientation signs must be +1 or -1")
    declared = dict(measurements.meta.get("discharge_orientation") or {})
    d_ip = declared.get("ip_sign")
    d_bt = declared.get("bt_sign")
    ipg = measurements.groups.get("ip")
    checks = []
    if ipg is not None and np.any(ipg.valid):
        meas = float(np.sign(np.asarray(ipg.measured, float)[np.nonzero(ipg.valid)[0][0]]))
        if d_ip is not None and float(d_ip) != meas:
            raise ValueError(f"declared discharge Ip sign {d_ip} disagrees with the measured Ip's sign {meas}")
        d_ip = meas
        checks.append("discharge Ip sign from the measured Ip")
    if d_ip is None:
        checks.append("discharge Ip sign unknown (no Ip measurement, none declared): Ip check skipped")
    # the equilibrium's signs are read only when a discharge sign is known to check them against
    e_ip, e_bt = _equilibrium_signs(eq, need_ip=d_ip is not None, need_bt=d_bt is not None)
    problems = []
    if d_ip is not None and e_ip != 0 and e_ip * m_ip != float(d_ip):
        problems.append(f"Ip: equilibrium sign {e_ip:+g} x map {m_ip:+g} != discharge sign {float(d_ip):+g}")
    if d_bt is not None and e_bt != 0 and e_bt * m_bt != float(d_bt):
        problems.append(f"B_t: equilibrium sign {e_bt:+g} x map {m_bt:+g} != discharge sign {float(d_bt):+g}")
    if d_bt is not None and e_bt == 0:
        checks.append("B_t sign of the equilibrium not available: B_t check skipped")
    if problems:
        raise ValueError("score: orientation mismatch (" + "; ".join(problems) + "); state the map with "
                         "orientation={'sign_ip': ..., 'sign_bt': ...} (or measurements.meta['orientation_map'])")
    return dict(sign_ip=m_ip, sign_bt=m_bt, source=src, equilibrium_signs=dict(ip=e_ip, bt=e_bt),
                discharge_signs=dict(ip=d_ip, bt=d_bt), checks=checks)


def _model_group(eq, name, group, kinetics, notes, frame=None):
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
        if frame is not None and (frame["sign_ip"], frame["sign_bt"]) != (1.0, 1.0):
            if "field_orientation" in group.meta and (float(sp), float(st)) != (frame["sign_ip"], frame["sign_bt"]):
                raise ValueError(f"score: MSE field_orientation {tuple(group.meta['field_orientation'])} disagrees with "
                                 f"the stated orientation map ({frame['sign_ip']:+g}, {frame['sign_bt']:+g})")
            sp, st = frame["sign_ip"], frame["sign_bt"]
        if (float(sp), float(st)) != (1.0, 1.0):
            notes.append(f"mse: stated field orientation sign_pol={sp:+g}, sign_tor={st:+g}")
        return _diag.mse(eq, geo, e_r=e_r, e_z=e_z, sign_pol=float(sp), sign_tor=float(st),
                         quantity=group.meta.get("quantity", "polarisation_angle"))
    if name in _THOMSON_PROFILE:
        prof = kinetics[_THOMSON_PROFILE[name]]
        return _diag.thomson(eq, geo, prof, outside=kinetics.get("thomson_outside", "exclude"))
    raise KeyError(name)


def score(equilibrium, measurements, kinetics=None, groups=None, orientation=None):
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

    orientation : dict, optional
        ``{"sign_ip": +-1, "sign_bt": +-1}``: the stated map from the
        equilibrium's frame to the discharge's (module docstring); default
        ``measurements.meta["orientation_map"]``, else the identity.  The
        mapped signs are checked against the discharge's; a mismatch raises.

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

    frame = resolve_orientation(equilibrium, measurements, orientation)
    if (frame["sign_ip"], frame["sign_bt"]) != (1.0, 1.0):
        notes.append(f"orientation map from {frame['source']}: sign_ip {frame['sign_ip']:+g}, "
                     f"sign_bt {frame['sign_bt']:+g} applied to the modeled values")
    model, model_valid, excluded = {}, {}, {}
    for name in todo:
        group = measurements.groups[name]
        syn = _model_group(equilibrium, name, group, kinetics, notes, frame)
        value = np.asarray(syn.value, dtype=float)
        if name in FRAME_SIGN_OF_GROUP:
            value = value * frame[FRAME_SIGN_OF_GROUP[name]]
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
    table = ResidualTable(measurements=measurements, model=model, model_valid=model_valid,
                          excluded=excluded, notes=notes, frame=frame)
    if "flux_loop" in model and measurements.groups["flux_loop"].meta.get("flux_offset") == "fitted":
        _fit_flux_offset(table)
    return table


def _fit_flux_offset(table):
    """Add the closed-form weighted-least-squares offset to the modeled loops (module docstring); the
    model must be absolute (no ``flux_reference``).  Re-callable: the previous offset is removed first."""
    g = table.measurements.groups["flux_loop"]
    if g.meta.get("flux_reference") is not None:
        raise ValueError("flux_offset='fitted' models the loops absolutely: flux_reference must be None")
    prev = float(table.frame.get("flux_offset", 0.0))
    m = np.asarray(table.model["flux_loop"], float) - prev
    use = table.used("flux_loop")
    if not np.any(use):
        table.frame["flux_offset"] = 0.0
        table.model["flux_loop"] = m
        return
    w = (g.weight[use] / g.sigma[use]) ** 2
    num, den = float(np.sum(w * (g.measured[use] - m[use]))), float(np.sum(w))
    prior = g.meta.get("flux_offset_prior")
    rule = "one additive loop offset per table, weighted least squares over the used channels"
    if prior is not None:
        c0, s0 = (float(v) for v in prior)
        if not (np.isfinite(c0) and np.isfinite(s0) and s0 > 0):
            raise ValueError("flux_offset_prior must be (finite value, sigma > 0)")
        num, den = num + c0 / s0 ** 2, den + 1.0 / s0 ** 2
        rule += " plus the prior row (c - c0)^2 / s0^2"
    c = num / den
    table.model["flux_loop"] = m + c
    table.frame["flux_offset"] = c
    table.frame["flux_offset_rule"] = rule
    if prior is not None:
        table.frame["flux_offset_prior"] = [c0, s0]
        table.frame["flux_offset_prior_chi2"] = float(((c - c0) / s0) ** 2)
    table.notes.append(f"flux_loop: fitted additive offset {c:.6g} (one nuisance parameter, closed form"
                       + (", with the reference prior row)" if prior is not None else ")"))


def score_common(equilibria, measurements, kinetics=None, groups=None, orientations=None):
    """Score several equilibria on the SAME channels: the intersection of their modelable channels.

    Each equilibrium is scored with :func:`score`; per group, a channel enters every table only if every
    equilibrium could model it (the union of the exclusions is applied to all, with the reason
    ``"not modelable in every compared equilibrium"`` added to ``excluded`` where it was modelable in this
    one).  ``orientations`` is an optional list with one orientation map per equilibrium.  Returns
    ``(tables, record)``; ``record`` gives, per group, the common channel count and the channels removed for
    comparability.  Totals of the returned tables are comparable; a channel that leaves one equilibrium's
    plasma is removed from all (it is never scored by dropping it from one of them only).
    """
    eqs = list(equilibria)
    if not eqs:
        raise ValueError("score_common needs at least one equilibrium")
    ors = list(orientations) if orientations is not None else [None] * len(eqs)
    if len(ors) != len(eqs):
        raise ValueError("one orientation map per equilibrium")
    tabs = [score(e, measurements, kinetics=kinetics, groups=groups, orientation=o) for e, o in zip(eqs, ors)]
    names = set(tabs[0].model)
    for t in tabs[1:]:
        if set(t.model) != names:
            raise ValueError("score_common: the equilibria were scored on different groups")
    record = {}
    reason = "not modelable in every compared equilibrium"
    for name in sorted(names):
        common = np.logical_and.reduce([np.asarray(t.model_valid[name], bool) for t in tabs])
        g = measurements.groups[name]
        removed = [g.labels[i] for i in np.nonzero(g.valid & ~common & np.logical_or.reduce(
            [np.asarray(t.model_valid[name], bool) for t in tabs]))[0]]
        for t in tabs:
            own = np.asarray(t.model_valid[name], bool)
            extra = g.valid & own & ~common
            if np.any(extra):
                t.excluded.setdefault(name, []).extend((g.labels[i], reason) for i in np.nonzero(extra)[0])
                t.notes.append(f"{name}: {int(extra.sum())} channel(s) removed for comparability ({reason})")
            t.model_valid[name] = common.copy()
            if name == "flux_loop" and g.meta.get("flux_offset") == "fitted":
                _fit_flux_offset(t)          # the offset is re-solved on the common channels
        record[name] = dict(n_common=int(np.count_nonzero(g.valid & common)), removed_for_comparability=removed)
    return tabs, record


def chi2_report(table, n_params=0):
    """chi^2 of a table with its channel count, parameter count and degrees of freedom stated separately.

    ``n_params`` is the number of parameters fitted to THESE channels (0 for an equilibrium scored without
    fitting).  Returns ``{"groups": {name: {"chi2", "n_channels", "sigma_model"}}, "total": {"chi2",
    "n_channels", "n_params", "dof", "chi2_per_dof"}, "interpretation": text}``.  ``chi2_per_dof`` is
    descriptive: on real data the sigma are conventions (named per group), correlations (a shared
    reference loop, time-slice-correlated relative errors) are ignored, so it is not a goodness-of-fit
    test; compare equilibria through :func:`score_common` on the same channels.
    """
    groups = {}
    for name in table.model:
        g = table.measurements.groups[name]
        groups[name] = dict(chi2=table.chi2(name), n_channels=table.n_valid(name),
                            sigma_model=g.meta.get("sigma_model"), sigma_model_id=_sigma_id(g))
    n = table.n_valid()
    dof = n - int(n_params)
    total = dict(chi2=table.chi2(), n_channels=n, n_params=int(n_params), dof=dof,
                 chi2_per_dof=(table.chi2() / dof) if dof > 0 else float("nan"))
    ids = sorted({str(v["sigma_model_id"]) for v in groups.values()})
    total["sigma_models"] = ids
    return dict(groups=groups, total=total,
                interpretation=("chi^2 under the stated sigma conventions (diagonal, correlations ignored); "
                                "not a pass/fail quantity on real data -- compare equilibria on the same channels"))


def _sigma_id(group):
    """The sigma convention a group was built under: ``meta["sigma_model_id"]`` (model and reading), else
    ``meta["sigma_model"]``, else ``"unstated"``."""
    return group.meta.get("sigma_model_id") or group.meta.get("sigma_model") or "unstated"


class Chi2UnderSigmaConvention(dict):
    """``{group: (chi2, n, chi2/n)}`` plus ``"total"``, with :attr:`sigma_conventions` ``{group: id}`` naming the
    sigma convention each number holds under (``"total"`` maps to the sorted list of the conventions it
    mixes).  A plain ``dict`` otherwise, so existing tuple unpacking keeps working."""

    sigma_conventions = None

    @property
    def mixed(self):
        """True when the total adds groups built under different sigma conventions."""
        return len(self.sigma_conventions.get("total", [])) > 1


def chi2_under_sigma_convention(table):
    """``{group: (chi2, n_valid, chi2/n)}`` plus ``'total'`` for a table, under the groups' sigma conventions.

    The returned mapping carries ``.sigma_conventions`` (``{group: sigma_model_id}``, ``"total"``: the list of
    conventions summed).  ``chi2/n`` divides by the channel count, not the degrees of freedom, and is
    descriptive only (module docstring); use :func:`chi2_report` for dof."""
    out = Chi2UnderSigmaConvention()
    conv = {}
    for name in table.model:
        c, n = table.chi2(name), table.n_valid(name)
        out[name] = (c, n, c / n if n else float("nan"))
        conv[name] = _sigma_id(table.measurements.groups[name])
    c, n = table.chi2(), table.n_valid()
    out["total"] = (c, n, c / n if n else float("nan"))
    conv["total"] = sorted(set(conv.values()))
    out.sigma_conventions = conv
    return out


#: Deprecated name of :func:`chi2_under_sigma_convention` (same return value), kept for existing callers.
chi2_summary = chi2_under_sigma_convention
