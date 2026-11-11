r"""Array provider: wrap profiles from any transport code with EXPLICIT unit/frame declarations.

Use this for ONETWO, TRANSP, RABBIT, FUSE or any code whose output has been read into numpy
arrays by the caller.  Nothing is inferred: the radial coordinate, current frame, current
units and sign convention, pressure form and units, and density units must all be stated;
an undeclared or unknown value raises :class:`.conventions.AmbiguousConventionError`.

Current frames (declared with ``current_frame``; equations in :mod:`.conventions`):

``"jdotb"``      <J.B> [A T/m^2] -- stored as is;
``"jpar_b0"``    <J.B>/B0 (IMAS ``j_parallel``) -- multiplied by B0 (C2);
``"j_r0"``       R0 <J_phi/R> (ONETWO ``<Jtor R0/R>``; AutoONETWO ``J_NBCD``) -- converted
                 with (C3)-(C4), needs ``geometry`` with ``avg_b2``;
``"j_tor"``      <J_phi/R>/<1/R> (IMAS ``j_tor``) -- (C3), (C5), needs ``avg_inv_r`` too;
``"jphi_avg"``   <J_phi> -- (C6), needs ``avg_inv_r`` and ``avg_b2``.

For the three toroidal frames a caller WITHOUT flux-surface geometry may state
``frame_approximation="large_aspect_ratio"``: then ``<J.B> = B0 j`` (exact for (C2), and
for the toroidal frames correct to the factor ``<B^2>/(B0 R0 F <1/R^2>) = (F/(B0 R0))
(1 + <B_p^2>/(F^2 <1/R^2>))``, i.e. to the diamagnetic/paramagnetic change of F and
``(B_p/B_phi)^2``, typically a few percent).  The approximation is recorded in the
provenance; it is never applied silently.

Sign (``current_sign``):

``"relative_to_ip"``  co-Ip drive positive (stored as is);
``"absolute"``        the producer's own signs; ``ip_sign`` and ``bt_sign`` (signs of Ip and of
                      B_phi in the producer's frame) are then required.  Co-Ip <J.B> has sign
                      ``ip_sign * bt_sign``; co-Ip toroidal current has sign ``ip_sign``; an
                      IMAS ``j_parallel`` divided by the producer's SIGNED B0 has co-Ip sign
                      ``ip_sign`` (DD definition, conventions C2).

Pressure (``pressure_form``, table in :mod:`.conventions`): ``"isotropic_equivalent"``,
``"energy_density"``, ``"true_components"`` (pair) or ``"imas_components"`` (pair).
Negative pressure or density values (spline undershoot in many files) are refused unless
``negative_policy="clip"``, which zeroes them and records the clipped fraction.
"""

import numpy as np

from .conventions import (CURRENT_FRAMES, PRESSURE_FORMS, AmbiguousConventionError, _geo_arrays, array_sha256,
                          jdotb_from_jphi_over_r, jdotb_from_jpar, p_fast_from_energy_density,
                          p_fast_from_imas_components, p_fast_from_true, psi_n_to_rho)
from .provider import NonInductiveSources, Provenance

__all__ = ["from_arrays", "CURRENT_UNITS", "PRESSURE_UNITS", "DENSITY_UNITS"]

#: Accepted current-density units and their factor to A/m^2 (or A T/m^2 for ``"jdotb"``).
CURRENT_UNITS = {"A/m^2": 1.0, "kA/m^2": 1.0e3, "MA/m^2": 1.0e6, "A/cm^2": 1.0e4, "A.T/m^2": 1.0}
#: Accepted pressure / energy-density units and their factor to Pa (= J/m^3).
PRESSURE_UNITS = {"Pa": 1.0, "kPa": 1.0e3, "J/m^3": 1.0, "keV/m^3": 1.602176634e-16, "MJ/m^3": 1.0e6}
#: Accepted density units and their factor to m^-3.
DENSITY_UNITS = {"m^-3": 1.0, "cm^-3": 1.0e6, "1e19 m^-3": 1.0e19, "1e20 m^-3": 1.0e20}

_SIGNS = ("relative_to_ip", "absolute")
_NEG = ("refuse", "clip")


def _need(value, name, allowed=None):
    if value is None:
        raise AmbiguousConventionError(f"{name} must be declared (no default is assumed)")
    if allowed is not None and value not in allowed:
        raise AmbiguousConventionError(f"{name}={value!r} is not one of {sorted(allowed)}")
    return value


def _sign(x):
    if x not in (1, -1):
        raise AmbiguousConventionError("ip_sign and bt_sign must be +1 or -1")
    return float(x)


def _neg(arr, policy, label, record):
    arr = np.asarray(arr, float)
    bad = arr < 0
    if not bad.any():
        return arr
    if policy == "refuse":
        raise ValueError(f"{label} has {int(bad.sum())} negative values (pass negative_policy='clip' to zero them)")
    peak = max(np.max(np.abs(arr)), 1e-300)
    record[label] = {"n_clipped": int(bad.sum()), "min_over_peak": float(arr.min() / peak)}
    return np.where(bad, 0.0, arr)


def from_arrays(rho, *, radial_coordinate, time, time_window, B0, R0, source_code, source_version,
                current=None, current_frame=None, current_units=None, current_sign=None, ip_sign=None, bt_sign=None,
                geometry=None, frame_approximation=None,
                pressure=None, pressure_form=None, pressure_units=None,
                density=None, density_units=None,
                psi_n_table=None, rho_table=None, negative_policy="refuse",
                uncertainty=None, assumptions=None, limitations=None, inputs=None, notes=""):
    r"""Build :class:`.provider.NonInductiveSources` from declared arrays.

    Parameters
    ----------
    rho
        Radial grid of every input array (1-D, strictly increasing).
    radial_coordinate
        ``"rho_tor_norm"`` or ``"psi_n"``; with ``"psi_n"``, ``psi_n_table``/``rho_table`` (the
        caller's equilibrium) map the grid to rho_tor_norm (C1).
    time, time_window
        Slice time and averaging window [s].
    B0, R0
        Reference vacuum field magnitude [T] and its major radius [m] (positive).
    source_code, source_version
        Producing code and version (``"unknown"`` is allowed but must be written out).
    current
        ``{source_name: array}`` driven current densities, or ``None``.
    current_frame, current_units, current_sign, ip_sign, bt_sign, geometry, frame_approximation
        Declarations for ``current`` (module docstring).  ``geometry`` is a
        :class:`..current.geometry.FluxSurfaceGeometry` (rho_tor_norm grid).
    pressure
        ``{species: array}`` (scalar forms) or ``{species: (par, perp)}`` (component forms).
    pressure_form, pressure_units
        Declarations for ``pressure``.
    density, density_units
        ``{species: array}`` fast-ion densities and their unit.
    negative_policy
        ``"refuse"`` (default) or ``"clip"`` for negative pressure/density samples.
    uncertainty
        List of :class:`.provider.ScalePrior` / :class:`.provider.ShapePrior` (may be empty;
        then :meth:`~.provider.NonInductiveSources.completeness` reports it).
    assumptions, limitations, inputs, notes
        Provenance text supplied by the caller (what the producing run assumed, its known
        limits, file paths/hashes).
    """
    rho_in = np.asarray(rho, float)
    if rho_in.ndim != 1 or rho_in.size < 2 or np.any(np.diff(rho_in) <= 0):
        raise ValueError("rho must be a strictly increasing 1-D grid")
    _need(radial_coordinate, "radial_coordinate", ("rho_tor_norm", "psi_n"))
    if radial_coordinate == "psi_n":
        if psi_n_table is None or rho_table is None:
            raise AmbiguousConventionError("radial_coordinate='psi_n' needs psi_n_table and rho_table "
                                           "from the caller's equilibrium")
        rho_out = psi_n_to_rho(rho_in, psi_n_table, rho_table)
        if not np.all(np.isfinite(rho_out)):
            raise ValueError("some psi_n samples fall outside the psi_n <-> rho table")
    else:
        rho_out = rho_in
    _need(negative_policy, "negative_policy", _NEG)
    conventions = {"radial_coordinate": radial_coordinate}
    clipped = {}

    jdotb = {}
    if current:
        _need(current_frame, "current_frame", CURRENT_FRAMES)
        _need(current_units, "current_units", CURRENT_UNITS)
        _need(current_sign, "current_sign", _SIGNS)
        if current_frame == "jdotb" and current_units not in ("A.T/m^2",):
            raise AmbiguousConventionError("current_frame='jdotb' takes current_units='A.T/m^2'")
        if current_frame != "jdotb" and current_units == "A.T/m^2":
            raise AmbiguousConventionError(f"current_units='A.T/m^2' is only valid for frame 'jdotb'")
        fac = CURRENT_UNITS[current_units]
        if current_sign == "absolute":
            sip, sbt = _sign(ip_sign), _sign(bt_sign)
            rel = sip * sbt if current_frame == "jdotb" else sip
        else:
            rel = 1.0
        if current_frame in ("j_r0", "j_tor", "jphi_avg"):
            if geometry is None:
                _need(frame_approximation, "frame_approximation (no geometry given for a toroidal frame)",
                      ("large_aspect_ratio",))
            elif frame_approximation is not None:
                raise AmbiguousConventionError("give either geometry or frame_approximation, not both")
        for name, arr in current.items():
            a = np.asarray(arr, float) * fac * rel
            if a.shape != rho_in.shape:
                raise ValueError(f"current[{name!r}] does not match rho")
            if current_frame == "jdotb":
                jb = a
            elif current_frame == "jpar_b0":
                jb = jdotb_from_jpar(a, B0)
            elif geometry is None:          # declared large-aspect-ratio approximation
                jb = a * float(B0)
            else:
                if current_frame == "j_r0":
                    jr = a / float(R0)
                elif current_frame == "j_tor":
                    jr = a * _geo_arrays(geometry, rho_out, ("avg_inv_r",))["avg_inv_r"]
                else:                        # jphi_avg: <J_phi> = sigma F <1/R>, <J_phi/R> = sigma F <1/R^2>
                    g = _geo_arrays(geometry, rho_out, ("avg_inv_r", "avg_inv_r2"))
                    jr = a * g["avg_inv_r2"] / g["avg_inv_r"]
                jb = jdotb_from_jphi_over_r(jr, rho_out, geometry)
            jdotb[name] = jb
        conventions.update(current_frame=current_frame, current_units=current_units, current_sign=current_sign,
                           ip_sign=ip_sign, bt_sign=bt_sign, frame_approximation=frame_approximation,
                           geometry="FluxSurfaceGeometry" if geometry is not None else None)

    p_fast, p_par, p_perp = {}, {}, {}
    if pressure:
        _need(pressure_form, "pressure_form", PRESSURE_FORMS)
        _need(pressure_units, "pressure_units", PRESSURE_UNITS)
        fac = PRESSURE_UNITS[pressure_units]
        for sp, val in pressure.items():
            if pressure_form in ("true_components", "imas_components"):
                if not (isinstance(val, (tuple, list)) and len(val) == 2):
                    raise AmbiguousConventionError(f"pressure_form={pressure_form!r} needs (par, perp) pairs")
                par = _neg(np.asarray(val[0], float) * fac, negative_policy, f"p_par[{sp}]", clipped)
                perp = _neg(np.asarray(val[1], float) * fac, negative_policy, f"p_perp[{sp}]", clipped)
                if pressure_form == "imas_components":
                    p_par[sp], p_perp[sp] = 3.0 * par, 3.0 * perp
                    p_fast[sp] = p_fast_from_imas_components(par, perp)
                else:
                    p_par[sp], p_perp[sp] = par, perp
                    p_fast[sp] = p_fast_from_true(par, perp)
            else:
                if isinstance(val, (tuple, list)):
                    raise AmbiguousConventionError(f"pressure_form={pressure_form!r} takes one array per species")
                a = _neg(np.asarray(val, float) * fac, negative_policy, f"p_fast[{sp}]", clipped)
                p_fast[sp] = a if pressure_form == "isotropic_equivalent" else p_fast_from_energy_density(a)
        conventions.update(pressure_form=pressure_form, pressure_units=pressure_units)

    n_fast = {}
    if density:
        _need(density_units, "density_units", DENSITY_UNITS)
        for sp, val in density.items():
            n_fast[sp] = _neg(np.asarray(val, float) * DENSITY_UNITS[density_units], negative_policy,
                              f"n_fast[{sp}]", clipped)
        conventions.update(density_units=density_units)

    if not (jdotb or p_fast or n_fast):
        raise ValueError("nothing to wrap: give current, pressure or density")
    all_arrays = list(jdotb.values()) + list(p_fast.values()) + list(n_fast.values())
    prov = Provenance(
        provider="arrays", source_code=str(source_code), source_version=str(source_version),
        inputs=dict(inputs or {}, sha256=array_sha256(rho_in, *all_arrays)),
        conventions=conventions,
        settings={"negative_policy": negative_policy, "clipped": clipped},
        assumptions=list(assumptions or []) + (
            ["toroidal-frame current converted with the large-aspect-ratio approximation <J.B> = B0 j"]
            if frame_approximation == "large_aspect_ratio" else []),
        limitations=list(limitations or []), notes=notes)
    return NonInductiveSources(rho=rho_out, time=time, time_window=time_window, B0=B0, R0=R0, jdotb=jdotb,
                               p_fast=p_fast, n_fast=n_fast, p_par=p_par, p_perp=p_perp,
                               uncertainty=list(uncertainty or []), provenance=prov)
