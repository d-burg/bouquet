r"""Built-in zeroth-order neutral-beam model: deposition, classical fast ions, beam-driven current.

The "FUSE-equivalent, bugs fixed" tier.  It follows the structure of FUSE's ``ActorSimpleNB``
plus IMAS.jl's classical fast-ion profiles (Meneghini et al., arXiv:2409.05894 (2024); FUSE
and IMAS.jl are Apache-2.0, General Atomics) but is RE-DERIVED here from the physics
literature (:mod:`.slowing_down`, :mod:`.nbcd`) and differs from FUSE where FUSE is known to
be wrong:

==========================================  =======================================  ===================================
item                                        FUSE (v1.1.2 / IMAS.jl 7.1.6)            here
==========================================  =======================================  ===================================
birth energy of the E/2, E/3 components     all slow down from the full E (open PR   each from its own E/k (S6-S12)
                                            #1201)
attenuation of the E/2, E/3 components      ion-impact and CX cross sections at the  each at its own E/k
                                            full E (PR #1201)
energy fractions                            ``beam_current_fraction`` used as the    power and current (particle)
                                            power fraction (PR #1202)                fractions distinguished (B1-B2)
NBCD normalization                          ``0.1 e (S m v R xi) tau / (A m_p)``:    SI fast-ion current
                                            an uncommented factor 0.1 times the      ``Z_b e S tau_s v0 xi0 I_1`` (S11),
                                            major radius R (the momentum source is   no numerical factor
                                            an angular momentum)
deposition smoothing at the axis            Gaussian cut at rho = 0 and              axis-mirrored Gaussian (PR #1200
                                            renormalized (nonzero on-axis slope)     behavior); volume-normalized
pitch-angle scattering exponent             main-ion mass for all ions               exact species sums (S5)
beam-ion Coulomb logarithm                  = electron one                           NRL beam-ion form (default)
ion-impact ionization                       on n_i = n_e with Z = 1, no impurities   main ions + each impurity (Z)
trapped-electron correction                 large-aspect-ratio fit                   G = L31 (Redl; general geometry)
inward orbit (banana) shift of births       ``banana_width (1 - xi) x                NOT included: births stay on the
                                            banana_shift_fraction`` (default 0.5)    flux surface of the ionization point
NBCD output                                 returned                                 opt-in only, flagged NOT validated
==========================================  =======================================  ===================================

**Beam-driven current is not validated and is off by default.**  The fast-ion current here is
the cylindrical Gaffey result (S11).  With the birth-point pitch alone, fast ions born in the
trapped region of pitch space are counted as passing and passing ions carry their birth pitch
rather than the transit average; against a transport code this overestimated the driven current
by 1.5-2.4x per MW for tangential injection.  Since the second development pass the DEFAULT of
the current (``trapping="zero_orbit_width"``, below) removes the current of births on trapped
orbits and gives passing births their transit-averaged pitch, which brought the co-injection
development sample to 0.95 of the transport code (median); it needs the caller's
:class:`.trapping.FluxSurfaceField`, and without one (or for a beam given as a deposition, which
has no birth positions) the birth pitch is used and the record says so (a missing-input rule).
Pitch scattering across the trapped-passing boundary, finite orbit width and first-orbit loss are
still missing; a bounce-averaged fast-ion model (Cordey, Nucl. Fusion 16, 499 (1976), recalled)
is not implemented.  :class:`SimpleBeamModel` therefore
returns NO driven current unless ``compute_nbcd=True``, and then marks every beam's ``jdotb``
as not validated in the output (``not_validated``), in its provenance and in its limitations;
reading it raises until the caller acknowledges it
(:meth:`.provider.NonInductiveSources.acknowledge_not_validated`).  No calibration factor is
applied.  The fast-ion pressure, density and heating are unaffected by this flag.

**The stored-energy validation depends on the beam-ion Coulomb logarithm.**  The fast-ion
stored energy agrees with a transport code to about 2 % (ratio 1.02) with the default separate
NRL beam-ion Coulomb logarithm (``ln_lambda_i="nrl_beam_ion"``, chosen before the comparison);
with the beam-ion logarithm set equal to the electron one the same comparison gives about 1.17
(ion heating 0.97 against 0.87).  The validation is therefore a statement about the model WITH
that default, not about the slowing-down model alone.

**Dropped FUSE feature (declared difference).**  FUSE ``ActorSimpleNB`` shifts each birth
inward by ``banana_width (1 - xi) banana_shift_fraction`` (default fraction 0.5), a crude
finite-orbit correction of the deposition.  It is not reproduced here (births are mapped to
the flux surface of their ionization point); this may contribute to an outward bias of the
fast-ion pressure near the edge.  Listed in :data:`FUSE_DIFFERENCES` and in every provenance
record.

Fractions of a beam's NEUTRAL flux by energy component k = 1, 2, 3 (energies E/k):

.. code-block:: text

    current (= particle) fraction  c_k = Gamma_k / sum Gamma_j
    power fraction                 p_k = (c_k / k) / sum_j (c_j / j)                        (B1)
    inverse                        c_k = k p_k / sum_j (j p_j)                              (B2)

(the neutral "current" of a component is e Gamma_k; its power is Gamma_k E/k; definitions of
IMAS ``nbi.unit[:].beam_current_fraction`` / ``beam_power_fraction``).  The particle rate of
component k is ``Gamma_k = p_k P / (e E/k)``.

Pencil-beam deposition (straight line from the source, attenuated by the local inverse mean
free path; Fowler, Holmes and Rome, NFREYA, ORNL/TM-6845 (1979), recalled):

.. code-block:: text

    dI_k/ds = -I_k sum_proc lambda_proc^-1(E_k/A_b, n, T) ,   births dN_k = -dI_k             (B3)
    xi0 = v_phi / v = R_tan cos(theta_v) / R                                                  (B4)

(B4: angular momentum about the symmetry axis is conserved along a straight line,
``R v_phi = R_tan v cos(theta_v)``; the poloidal-field contribution to v.b is neglected).
Births are binned on the rho grid (cloud-in-cell) and optionally spread with a Gaussian in
rho of width ``smoothing_width``, mirrored at the axis and normalized over volume, which
conserves the number of births.  The width stands in for the finite beam cross-section and
orbit width that a pencil beam lacks; it is a declared setting, not a fitted one.

Cross sections (``FreyaCrossSections``): the FREYA fits as transcribed in IMAS.jl
``src/physics/fast.jl`` (Apache-2.0; "pulled from freya_fsgxn.f90"; the electron-impact
rate and proton-impact fits are attributed there to Freeman and Jones, Culham CLM-R137
(1974); the charge-exchange form is Riviere's, Nucl. Fusion 11, 363 (1971); the impurity
form is the FREYA one; original papers recalled, not checked):

.. code-block:: text

    electron impact:  lambda^-1 = n_e <sigma v>_e(T_e) / v_b,
                      ln<sigma v>_e[cm^3/s] = sum_i a_i (ln T_e[eV])^i   (T_e clamped to [1, 1e5] eV)   (B5)
    proton impact:    lambda^-1 = n_i sigma_p(E/A),  ln sigma_p[cm^2] = sum_i b_i (ln E[keV/u])^i
                      (zero below 0.1 keV/u)                                                            (B6)
    charge exchange:  sigma_cx[cm^2] = 0.6937e-14 (1 - 0.155 log10 E[eV/u])^2 / (1 + 0.1112e-14 E^3.3)  (B7)
    impurity impact:  sigma_Z[cm^2] = 1e-17 46 Z (32 Z / E[keV/u]) (1 - exp(-E/(32 Z)))                (B8)

Validity: hydrogenic beams (A_b <= 3), ground-state atoms only -- beam-stopping enhancement
by excited states (multi-step ionization, typically +10-30 % on the stopping rate at
fusion-relevant densities; Janev, Boley and Post, Nucl. Fusion 29, 2125 (1989), recalled)
is NOT included, so shine-through is overestimated and the deposition is slightly too
deep.  A caller can pass its own ``cross_sections`` object with the same
``inverse_mfp`` method (e.g. a Suzuki/ADAS beam-stopping coefficient).

Two OPTIONS, both off by default (the defaults reproduce the model above bit for bit):

* ``deposition="beamlets"`` (default ``"pencil"``): each beam with a finite-beam description
  (:class:`.beam_optics.BeamOptics`: source size, focus, divergence, apertures; IMAS schema) is
  deposited as ``n_beamlets`` straight beamlets per divergence component, sampled
  deterministically from its source rectangle, focal lengths and divergence and clipped at its
  apertures (equations O1-O3 of :mod:`.beam_optics`).  Each beamlet is attenuated with B3
  along its own line, its pitch from its own angular momentum about the symmetry axis
  (``xi0 = L_z / R``, L_z conserved along the line, the generalization of B4), and all births
  are binned together as above (the ``smoothing_width`` still applies to every beamlet; it then
  stands for the orbit width only).  A beam without a description stays a pencil beam, and
  every missing item of a description follows a stated rule; both are recorded per beam in
  ``meta["beams"][name]``.  The number of beamlets is the caller's and must be chosen by a
  convergence test (on-axis value, peaking and stored energy against the number).
* ``trapping="zero_orbit_width"`` (THE DEFAULT; ``"none"`` restores the birth pitch), for the
  driven current only: births on trapped orbits carry no current and passing births carry their
  transit-averaged pitch (equations T1-T4 of :mod:`.trapping`); it uses ``field``
  (:class:`.trapping.FluxSurfaceField`) in :meth:`SimpleBeamModel.compute`.  Without a field, or
  for a beam given as a deposition, it is not applied and the record says so ("requested, not
  applied: ...").  It acts only when ``compute_nbcd=True``; with ``compute_nbcd=False`` (the
  default) nothing of it appears in the output or the records.  The pressure, density and heating
  do not depend on it.
  The driven current stays opt-in and flagged NOT validated (pitch scattering across the
  trapped-passing boundary, finite orbit width and first-orbit loss are not included); a
  record flags counter-injected power, for which the net current is ill-conditioned without
  first-orbit loss.

LIMITATIONS (recorded in provenance): pencil beams (no divergence or source size except via
the smoothing width; see the ``deposition`` option), straight-line orbits of the neutrals, no finite-orbit width of the
fast ions (births are mapped to the flux surface of the ionization point), no prompt or
orbit loss, no fast-ion CX loss, no anomalous fast-ion transport (Alfven eigenmodes,
sawteeth, tearing modes), no beam-beam collisions, steady state per slice (no time history),
a 1-D current model (the fast-ion parallel current is equated with <J.B>/B0, using the
birth-point pitch: an order-one error near the trapped-passing boundary, not O(eps^2)),
collisionless trapped-electron correction, fast-ion trapping only in the zero-orbit-width limit
and only where a field is given (see above; the provenance entry "no fast-ion trapping" is
replaced when the trapping acts and kept otherwise), no inward orbit
shift of the births (see above).  Cross-section fit clamps (T_e outside [1, 1e5] eV; proton
impact zeroed below 0.1 keV/u) are counted per call and recorded in provenance
(``inputs["cross_section_clamps"]``).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .beam_optics import BeamOptics, HALTON_BASES, beamlet_sample
from .conventions import E_CHARGE, M_AMU, array_sha256
from .nbcd import shielding_factor
from .provider import SOURCES_VERSION, NonInductiveSources, Provenance
from .slowing_down import IonSpecies, classical_moments
from .trapping import FluxSurfaceField, classify_births

__all__ = ["NeutralBeam", "volume_weights", "volume_integral", "power_fractions_from_current", "current_fractions_from_power", "FreyaCrossSections",
           "pencil_beam_deposition", "beamlet_deposition", "BeamDeposition", "SimpleBeamModel", "BEAM_LIMITATIONS",
           "FUSE_DIFFERENCES", "DEPOSITION_MODES", "TRAPPING_MODES"]

DEPOSITION_MODES = ("pencil", "beamlets")
TRAPPING_MODES = ("none", "zero_orbit_width")

BEAM_LIMITATIONS = [
    "pencil beams: no divergence or source size beyond the declared rho smoothing width",
    "no finite-orbit width: births mapped to the flux surface of the ionization point",
    "no prompt/orbit loss and no fast-ion charge-exchange loss",
    "no anomalous fast-ion transport (Alfven eigenmodes, sawteeth, tearing modes)",
    "no beam-beam or beam-fast-ion collisions",
    "steady state per slice (no time history of the beam power)",
    "ground-state stopping cross sections (no multi-step enhancement): shine-through overestimated",
    "collisionless trapped-electron correction G = L31(f_t)",
    "1-D current model: fast-ion parallel current equated with <J.B>/B0 with the birth-point pitch (order-one "
    "error near the trapped-passing boundary)",
    "no fast-ion trapping: the cylindrical (Gaffey) fast-ion current counts trapped fast ions as passing, "
    "which overestimates NBCD (1.5-2.4x a transport code per MW for tangential injection); the driven current is "
    "therefore opt-in and flagged NOT validated",
    "no inward orbit (banana) shift of the births (FUSE ActorSimpleNB applies one; dropped here)",
]
FUSE_DIFFERENCES = [
    "each energy component born at its own energy E/k, with its own cross sections (FUSE PR #1201 behavior)",
    "power fractions and current (particle) fractions distinguished, eqs. B1-B2 (FUSE PR #1202 behavior)",
    "NBCD normalization is the SI fast-ion current, no 0.1 R factor",
    "axis-mirrored, volume-normalized deposition smoothing (FUSE PR #1200 behavior)",
    "exact species sums in critical energy and pitch-angle scattering; beam-ion Coulomb logarithm separate",
    "ion-impact ionization on main ions and impurities with their own densities and charges",
    "DROPPED: FUSE's inward banana shift of births (banana_width (1 - xi) banana_shift_fraction, default 0.5) is not "
    "applied; births stay on the flux surface of the ionization point",
    "NBCD is opt-in (compute_nbcd=True) and flagged NOT validated (no fast-ion trapping); FUSE returns it",
]
#: Replaces the first entry of BEAM_LIMITATIONS with ``deposition="beamlets"``.
BEAMLET_LIMITATION = ("finite beams from beamlets (source size, focus, divergence, apertures as described; missing "
                      "items by the stated rules, recorded per beam); beams without a description stay pencil beams")
#: Replaces the "no fast-ion trapping" entry of BEAM_LIMITATIONS with ``trapping="zero_orbit_width"``.
TRAPPING_LIMITATION = ("zero-orbit-width fast-ion trapping in the driven current: trapped births carry no current, "
                       "passing births their transit-averaged pitch; no pitch scattering across the trapped-passing "
                       "boundary, no finite orbit width, no first-orbit loss; the driven current stays opt-in and "
                       "flagged NOT validated")


def power_fractions_from_current(c):
    """p_k = (c_k/k) / sum_j (c_j/j) (eq. B1)."""
    c = np.asarray(c, float)
    k = np.arange(1, c.size + 1)
    if np.any(c < 0) or c.sum() <= 0:
        raise ValueError("fractions must be >= 0 and not all zero")
    x = c / k
    return x / x.sum()


def current_fractions_from_power(p):
    """c_k = k p_k / sum_j (j p_j) (eq. B2)."""
    p = np.asarray(p, float)
    k = np.arange(1, p.size + 1)
    if np.any(p < 0) or p.sum() <= 0:
        raise ValueError("fractions must be >= 0 and not all zero")
    x = k * p
    return x / x.sum()


@dataclass
class NeutralBeam:
    """One beam (or beamlet group).  Exactly one of ``power_fractions`` / ``current_fractions``.

    ``power`` -- injected neutral power [W]; ``energy`` -- full energy [eV]; ``A``, ``Z`` --
    mass number and charge of the beam ions; ``direction`` -- +1 co-Ip, -1 counter-Ip.
    Geometry: either ``tangency_radius`` [m] with ``source_r``, ``source_z`` [m] and
    ``vertical_angle`` [rad, positive upward] (pencil beam), or ``deposition`` = dict with
    ``rho`` (grid), ``shape`` (births per volume, any normalization; one array, or a list of
    three for the components), ``xi0`` (array or scalar, signed relative to Ip) and optionally
    ``p2_xi0`` and ``shine_through`` (fraction per component, default stated as 0).
    """

    name: str
    power: float
    energy: float
    A: float
    Z: float
    direction: int
    power_fractions: Optional[Sequence[float]] = None
    current_fractions: Optional[Sequence[float]] = None
    tangency_radius: Optional[float] = None
    source_r: Optional[float] = None
    source_z: Optional[float] = None
    vertical_angle: Optional[float] = None
    deposition: Optional[dict] = None
    species_label: Optional[str] = None

    def __post_init__(self):
        if (self.power_fractions is None) == (self.current_fractions is None):
            raise ValueError(f"{self.name}: give exactly one of power_fractions / current_fractions")
        f = np.asarray(self.power_fractions if self.power_fractions is not None else self.current_fractions, float)
        if f.size != 3 or abs(f.sum() - 1.0) > 1e-6 or np.any(f < 0):
            raise ValueError(f"{self.name}: three non-negative energy fractions summing to 1 are required")
        if self.direction not in (1, -1):
            raise ValueError(f"{self.name}: direction must be +1 (co-Ip) or -1 (counter-Ip)")
        if not (self.power >= 0 and self.energy > 0 and self.A > 0 and self.Z > 0):
            raise ValueError(f"{self.name}: power >= 0, energy, A, Z > 0 required")
        pencil = self.tangency_radius is not None
        if pencil == (self.deposition is not None):
            raise ValueError(f"{self.name}: give either pencil geometry (tangency_radius, source_r, source_z, "
                             "vertical_angle) or a deposition dict")
        if pencil and None in (self.source_r, self.source_z, self.vertical_angle):
            raise ValueError(f"{self.name}: pencil geometry needs source_r, source_z and vertical_angle")
        if pencil and not (0 <= self.tangency_radius < self.source_r):
            raise ValueError(f"{self.name}: need 0 <= tangency_radius < source_r")
        if self.species_label is None:
            self.species_label = f"beam_A{self.A:g}_Z{self.Z:g}"

    @property
    def p_frac(self):
        """Power fractions (full, half, third)."""
        if self.power_fractions is not None:
            return np.asarray(self.power_fractions, float)
        return power_fractions_from_current(self.current_fractions)

    @property
    def c_frac(self):
        """Current (particle) fractions (full, half, third)."""
        if self.current_fractions is not None:
            return np.asarray(self.current_fractions, float)
        return current_fractions_from_power(self.power_fractions)

    @property
    def energies(self):
        """Component energies E/k [eV]."""
        return self.energy / np.arange(1, 4)

    @property
    def particle_rates(self):
        """Neutral particle rates Gamma_k = p_k P / (e E/k) [1/s]."""
        return self.p_frac * self.power / (E_CHARGE * self.energies)


class FreyaCrossSections:
    r"""Inverse mean free paths of a hydrogenic neutral (eqs. B5-B8; coefficients from IMAS.jl/FREYA)."""

    name = "freya_imasjl_7.1.6"
    _CFIONE = (-3.173850e+01, 1.143818e+01, -3.833998, 0.7046692, -0.07431486, 0.004153749, -9.486967e-05)
    _CFIONP = (-4.203309e+01, 3.557321, -1.045134, 0.3139238, -0.07454475, 0.008459113, -3.495444e-04)

    @staticmethod
    def _poly(c, x):
        out = np.zeros_like(x)
        for a in reversed(c):
            out = out * x + a
        return out

    def inverse_mfp(self, e_per_amu_ev, v_beam, ne, te_ev, hydrogenic, impurities, log=None):
        """Total lambda^-1 [1/m] (arrays on the path).  ``hydrogenic`` / ``impurities``: lists of (n, Z).

        ``log`` (a dict, optional) accumulates how often the fits' clamps acted: ``te_clamped`` (points with
        T_e outside [1, 1e5] eV) and ``proton_impact_zeroed`` (calls below 0.1 keV/u).
        """
        te_in = np.asarray(te_ev, float)
        te = np.clip(te_in, 1.0, 1.0e5)
        if log is not None:
            log["te_clamped"] = log.get("te_clamped", 0) + int(np.count_nonzero((te_in < 1.0) | (te_in > 1.0e5)))
        sv_e = np.exp(self._poly(self._CFIONE, np.log(te))) * 1e-6              # m^3/s
        lam = np.asarray(ne, float) * sv_e / v_beam
        e = float(e_per_amu_ev)
        lnek = np.log(e / 1000.0)
        sig_p = 0.0 if lnek <= -2.30258 else float(np.exp(self._poly(self._CFIONP, np.array(lnek)))) * 1e-4
        if log is not None and lnek <= -2.30258:
            log["proton_impact_zeroed"] = log.get("proton_impact_zeroed", 0) + 1
        sig_cx = 0.6937e-14 * (1.0 - 0.155 * np.log10(e)) ** 2 / (1.0 + 0.1112e-14 * e ** 3.3) * 1e-4
        for n, _z in hydrogenic:
            lam = lam + np.asarray(n, float) * (sig_p + sig_cx)
        ekev = e / 1000.0
        for n, z in impurities:
            sig_z = 1e-17 * 46.0 * z * (32.0 * z / ekev) * (1.0 - np.exp(-ekev / (32.0 * z))) * 1e-4
            lam = lam + np.asarray(n, float) * sig_z
        return lam


@dataclass
class BeamDeposition:
    """Result of :func:`pencil_beam_deposition` for one beam: per component k (index 0, 1, 2)."""

    births: np.ndarray        # (3, n_rho) birth rate density [m^-3 s^-1]
    xi0: np.ndarray           # (3, n_rho) birth-weighted <xi0>, signed relative to Ip
    p2_xi0: np.ndarray        # (3, n_rho) birth-weighted <P_2(xi0)>
    shine_through: np.ndarray  # (3,) transmitted fraction of each component
    path: Dict[str, np.ndarray] = field(default_factory=dict)
    xi_cd: Optional[np.ndarray] = None        # (3, n_rho) birth-weighted current pitch (T4), trapping option only
    trapped_births: Optional[np.ndarray] = None  # (3,) fraction of births on trapped orbits, trapping option only
    info: Dict = field(default_factory=dict)  # beamlet record (deposition="beamlets" only)


def volume_weights(rho, dvol_drho):
    """Trapezoid weights in the volume coordinate: sum_j f_j w_j = int f dV (V = cumulative trapezoid of V').

    Unlike trapezoid weights in rho, the axis node has a non-zero weight (half the first cell volume),
    so births binned onto the axis node are not lost.
    """
    rho = np.asarray(rho, float)
    fv = np.asarray(dvol_drho, float)
    dV = 0.5 * (fv[1:] + fv[:-1]) * np.diff(rho)
    w = np.zeros(rho.size)
    w[:-1] += 0.5 * dV
    w[1:] += 0.5 * dV
    return w


def volume_integral(rho, dvol_drho, f):
    """int f dV with the weights of :func:`volume_weights` (the integral births are conserved in)."""
    return np.asarray(f, float) @ volume_weights(rho, dvol_drho)


def _bin(rho_grid, dvol_drho, rho_pts, weights, width, mirror_axis):
    """Births per volume on ``rho_grid`` from point births (cloud-in-cell, optional Gaussian spread).

    Conserves ``sum(weights)`` exactly in :func:`volume_integral`.
    """
    n = rho_grid.size
    dv = volume_weights(rho_grid, dvol_drho)
    out = np.zeros((weights.shape[0], n))
    if width and width > 0:
        # each birth spread as a Gaussian in rho (mirrored at 0), normalized to unit volume integral
        z = rho_grid[None, :]
        kern = np.exp(-0.5 * ((z - rho_pts[:, None]) / width) ** 2)
        if mirror_axis:
            kern = kern + np.exp(-0.5 * ((z + rho_pts[:, None]) / width) ** 2)
        norm = kern @ dv
        kern = kern / np.where(norm > 0, norm, 1.0)[:, None]
        return weights @ kern, dv
    h = np.diff(rho_grid)
    idx = np.clip(np.searchsorted(rho_grid, rho_pts) - 1, 0, n - 2)
    t = np.clip((rho_pts - rho_grid[idx]) / h[idx], 0.0, 1.0)
    for k in range(weights.shape[0]):
        np.add.at(out[k], idx, weights[k] * (1 - t))
        np.add.at(out[k], idx + 1, weights[k] * t)
    return out / dv[None, :], dv


def pencil_beam_deposition(beam, rho_of_rz, rho, dvol_drho, ne, te_ev, hydrogenic, impurities,
                           cross_sections=None, n_steps=4000, smoothing_width=0.0, mirror_axis=True, clamp_log=None,
                           classifier=None):
    r"""Attenuate a pencil beam (eq. B3) and return births per component on ``rho``.

    ``rho_of_rz(R, Z)`` gives rho_tor_norm (NaN outside the plasma) from the caller's
    equilibrium; ``ne``, ``te_ev`` are arrays on ``rho``; ``hydrogenic`` / ``impurities`` are
    lists of ``(n_array_on_rho, Z)``.  The beam travels from ``(source_r, 0, source_z)`` toward
    the co-Ip toroidal direction for ``direction = +1`` (internal positive frame).  ``clamp_log``
    (a dict) accumulates the cross-section fits' clamp counts.  ``classifier(rho, R, Z, xi0)``
    (trapping option only; default None, which changes nothing) returns ``(trapped, xi_cd)`` per
    birth segment; the result then also carries ``xi_cd`` and ``trapped_births``.
    """
    if beam.tangency_radius is None:
        raise ValueError("pencil_beam_deposition needs pencil geometry")
    xs = FreyaCrossSections() if cross_sections is None else cross_sections
    rho = np.asarray(rho, float)
    Rs, Zs, Rt, th = beam.source_r, beam.source_z, beam.tangency_radius, beam.vertical_angle
    sin_a = Rt / Rs
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    d = np.array([-cos_a * np.cos(th), sin_a * np.cos(th), np.sin(th)])   # co-direction (+phi) in the frame
    L = 2.0 * Rs * cos_a + 2.0 * abs(Zs) + 1.0
    s = np.linspace(0.0, L, n_steps)
    X, Y, Z = Rs + d[0] * s, d[1] * s, Zs + d[2] * s
    R = np.hypot(X, Y)
    rr = np.asarray(rho_of_rz(R, Z), float)
    inside = np.isfinite(rr) & (rr <= rho[-1])
    rq = np.where(inside, rr, 0.0)
    ne_s = np.where(inside, np.interp(rq, rho, ne), 0.0)
    te_s = np.where(inside, np.interp(rq, rho, te_ev), 1.0)
    hyd_s = [(np.where(inside, np.interp(rq, rho, n), 0.0), z) for n, z in hydrogenic]
    imp_s = [(np.where(inside, np.interp(rq, rho, n), 0.0), z) for n, z in impurities]
    xi_pt = beam.direction * Rt * np.cos(th) / R
    rates = beam.particle_rates
    both = inside[1:] & inside[:-1]
    seg_in = inside[1:] | inside[:-1]          # boundary segments count: their attenuation is a birth too
    rmid = np.where(both, 0.5 * (rq[1:] + rq[:-1]), np.where(inside[:-1], rq[:-1], rq[1:]))
    xmid = 0.5 * (xi_pt[1:] + xi_pt[:-1])
    W = np.zeros((3, s.size - 1))
    shine = np.zeros(3)
    for k, Ek in enumerate(beam.energies):
        vb = np.sqrt(2.0 * Ek * E_CHARGE / (beam.A * M_AMU))
        try:
            lam_all = xs.inverse_mfp(Ek / beam.A, vb, ne_s[inside], te_s[inside], [(n[inside], z) for n, z in hyd_s],
                                     [(n[inside], z) for n, z in imp_s], log=clamp_log)
        except TypeError:                      # a caller's cross-section object without the log argument
            lam_all = xs.inverse_mfp(Ek / beam.A, vb, ne_s[inside], te_s[inside], [(n[inside], z) for n, z in hyd_s],
                                     [(n[inside], z) for n, z in imp_s])
        lam = np.zeros_like(s)
        lam[inside] = lam_all
        tau = np.concatenate([[0.0], np.cumsum(0.5 * (lam[1:] + lam[:-1]) * np.diff(s))])
        I = rates[k] * np.exp(-tau)
        W[k] = np.where(seg_in, I[:-1] - I[1:], 0.0)
        shine[k] = np.exp(-tau[-1])
    pts = seg_in & (W.sum(0) > 0)
    births, dv = _bin(rho, np.asarray(dvol_drho, float), rmid[pts], W[:, pts], smoothing_width, mirror_axis)
    xw, _ = _bin(rho, np.asarray(dvol_drho, float), rmid[pts], W[:, pts] * xmid[pts], smoothing_width, mirror_axis)
    p2w, _ = _bin(rho, np.asarray(dvol_drho, float), rmid[pts], W[:, pts] * 0.5 * (3 * xmid[pts] ** 2 - 1),
                  smoothing_width, mirror_axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        xi0 = np.where(births > 0, xw / births, 0.0)
        p2 = np.where(births > 0, p2w / births, 0.0)
    out = BeamDeposition(births=births, xi0=xi0, p2_xi0=p2, shine_through=shine,
                         path={"R": R, "Z": Z, "rho": rr, "s": s})
    if classifier is not None:
        Rm, Zm = 0.5 * (R[1:] + R[:-1]), 0.5 * (Z[1:] + Z[:-1])
        out.xi_cd, out.trapped_births = _current_pitch(rho, np.asarray(dvol_drho, float), rmid[pts], Rm[pts], Zm[pts],
                                                       xmid[pts], W[:, pts], births, smoothing_width, mirror_axis,
                                                       classifier)
    return out


def _current_pitch(rho, dvol_drho, rp, Rp, Zp, xp, Wp, births, width, mirror_axis, classifier):
    """Birth-weighted current pitch xi_cd (T4 of :mod:`.trapping`) on ``rho`` and the trapped fraction of births."""
    trapped, xcd = classifier(rp, Rp, Zp, xp)
    cw, _ = _bin(rho, dvol_drho, rp, Wp * xcd[None, :], width, mirror_axis)
    tot = Wp.sum(1)
    tf = np.where(tot > 0, (Wp * trapped[None, :]).sum(1) / np.where(tot > 0, tot, 1.0), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xi_cd = np.where(births > 0, cw / births, 0.0)
    return np.clip(xi_cd, -1.0, 1.0), tf


def _beam_frame(beam):
    """Source center S and the unit vectors d (axis), e_h, e_v of the beam frame (module docstring of beam_optics)."""
    Rs, Zs, Rt, th = beam.source_r, beam.source_z, beam.tangency_radius, beam.vertical_angle
    sin_a = Rt / Rs
    cos_a = np.sqrt(1.0 - sin_a ** 2)
    d = np.array([-cos_a * np.cos(th), sin_a * np.cos(th), np.sin(th)])
    e_v = np.array([cos_a * np.sin(th), -sin_a * np.sin(th), np.cos(th)])
    e_h = np.array([sin_a, cos_a, 0.0])
    return np.array([Rs, 0.0, Zs]), d, e_h, e_v, cos_a


def beamlet_deposition(beam, optics, rho_of_rz, rho, dvol_drho, ne, te_ev, hydrogenic, impurities, *, n_beamlets,
                       cross_sections=None, n_steps=4000, smoothing_width=0.0, mirror_axis=True, clamp_log=None,
                       classifier=None, batch=64, chunk=20000):
    r"""Attenuate a finite beam as beamlets (eqs. O1-O3 of :mod:`.beam_optics`, B3 per beamlet); births on ``rho``.

    Arguments as :func:`pencil_beam_deposition`; ``optics`` a :class:`.beam_optics.BeamOptics`, ``n_beamlets``
    the number per divergence component.  The beam's particle rates are shared by the transmitted beamlets in
    proportion to their weights (the power is the launched power, after the apertures).  ``info`` records the
    sampling, the transmission through the apertures and each beamlet's closest approach in rho.
    """
    if beam.tangency_radius is None:
        raise ValueError("beamlet_deposition needs pencil geometry (the beam axis)")
    xs = FreyaCrossSections() if cross_sections is None else cross_sections
    rho = np.asarray(rho, float)
    dv_arr = np.asarray(dvol_drho, float)
    smp = beamlet_sample(optics, n_beamlets)
    S, d, eh, ev, cos_a = _beam_frame(beam)
    dj = d[None, :] + np.tan(smp["a_h"])[:, None] * eh[None, :] + np.tan(smp["a_v"])[:, None] * ev[None, :]
    dj = dj / np.linalg.norm(dj, axis=1)[:, None]
    P0 = S[None, :] + smp["y_h"][:, None] * eh[None, :] + smp["y_v"][:, None] * ev[None, :]
    keep = np.ones(dj.shape[0], bool)
    for ap in optics.apertures:
        t = ap.distance / (dj @ d)
        keep &= ap.passes(smp["y_h"] + t * (dj @ eh), smp["y_v"] + t * (dj @ ev))
    w_all = smp["weight"]
    info = {"n_beamlets_per_component": int(n_beamlets), "n_sampled": int(w_all.size), "n_transmitted": int(keep.sum()),
            "transmission": float(w_all[keep].sum() / w_all.sum()),
            "sampling": f"Halton bases {HALTON_BASES}, points 1..{int(n_beamlets)} per divergence component",
            "optics": optics.record()}
    n = rho.size
    zeros3 = np.zeros((3, n))
    if not keep.any():
        info["note"] = "fully clipped by the apertures: nothing deposited"
        return BeamDeposition(births=zeros3.copy(), xi0=zeros3.copy(), p2_xi0=zeros3.copy(), shine_through=np.zeros(3),
                              xi_cd=None if classifier is None else zeros3.copy(),
                              trapped_births=None if classifier is None else np.zeros(3), info=info)
    wt = w_all[keep] / w_all[keep].sum()
    P0, dj = P0[keep], dj[keep]
    L = 2.0 * beam.source_r * cos_a + 2.0 * abs(beam.source_z) + 1.0
    s = np.linspace(0.0, L, n_steps)
    ds = np.diff(s)
    rates = beam.particle_rates
    births, xw, p2w = zeros3.copy(), zeros3.copy(), zeros3.copy()
    cw = zeros3.copy() if classifier is not None else None
    trapped_w, born_w = np.zeros(3), np.zeros(3)
    shine = np.zeros(3)
    rho_min = np.full(wt.size, np.nan)
    for b0 in range(0, wt.size, batch):
        sl = slice(b0, b0 + batch)
        p0, dd, ww = P0[sl], dj[sl], wt[sl]
        X = p0[:, 0, None] + dd[:, 0, None] * s[None, :]
        Y = p0[:, 1, None] + dd[:, 1, None] * s[None, :]
        Z = p0[:, 2, None] + dd[:, 2, None] * s[None, :]
        R = np.hypot(X, Y)
        rr = np.asarray(rho_of_rz(R.ravel(), Z.ravel()), float).reshape(R.shape)
        inside = np.isfinite(rr) & (rr <= rho[-1])
        if np.any(inside, axis=1).any():
            rho_min[sl] = np.where(inside.any(1), np.min(np.where(inside, rr, np.inf), axis=1), np.nan)
        rq = np.where(inside, rr, 0.0)
        ne_s = np.where(inside, np.interp(rq, rho, ne), 0.0)
        te_s = np.where(inside, np.interp(rq, rho, te_ev), 1.0)
        hyd_s = [(np.where(inside, np.interp(rq, rho, nn), 0.0), z) for nn, z in hydrogenic]
        imp_s = [(np.where(inside, np.interp(rq, rho, nn), 0.0), z) for nn, z in impurities]
        Lz = p0[:, 0] * dd[:, 1] - p0[:, 1] * dd[:, 0]           # angular momentum per unit speed (conserved)
        xi_pt = beam.direction * Lz[:, None] / R
        both = inside[:, 1:] & inside[:, :-1]
        seg_in = inside[:, 1:] | inside[:, :-1]
        rmid = np.where(both, 0.5 * (rq[:, 1:] + rq[:, :-1]), np.where(inside[:, :-1], rq[:, :-1], rq[:, 1:]))
        xmid = 0.5 * (xi_pt[:, 1:] + xi_pt[:, :-1])
        W = np.zeros((3,) + seg_in.shape)
        for k, Ek in enumerate(beam.energies):
            vb = np.sqrt(2.0 * Ek * E_CHARGE / (beam.A * M_AMU))
            args = (Ek / beam.A, vb, ne_s[inside], te_s[inside], [(nn[inside], z) for nn, z in hyd_s],
                    [(nn[inside], z) for nn, z in imp_s])
            try:
                lam_all = xs.inverse_mfp(*args, log=clamp_log)
            except TypeError:                  # a caller's cross-section object without the log argument
                lam_all = xs.inverse_mfp(*args)
            lam = np.zeros(R.shape)
            lam[inside] = lam_all
            tau = np.concatenate([np.zeros((R.shape[0], 1)),
                                  np.cumsum(0.5 * (lam[:, 1:] + lam[:, :-1]) * ds[None, :], axis=1)], axis=1)
            I = rates[k] * ww[:, None] * np.exp(-tau)
            W[k] = np.where(seg_in, I[:, :-1] - I[:, 1:], 0.0)
            shine[k] += float(np.sum(ww * np.exp(-tau[:, -1])))
        pts = seg_in & (W.sum(0) > 0)
        rp, xp, Wp = rmid[pts], xmid[pts], W[:, pts]
        if classifier is not None:
            trapped, xcd = classifier(rp, 0.5 * (R[:, 1:] + R[:, :-1])[pts], 0.5 * (Z[:, 1:] + Z[:, :-1])[pts], xp)
            trapped_w += (Wp * trapped[None, :]).sum(1)
            born_w += Wp.sum(1)
        for c0 in range(0, rp.size, chunk):
            c = slice(c0, c0 + chunk)
            bb, _ = _bin(rho, dv_arr, rp[c], Wp[:, c], smoothing_width, mirror_axis)
            births += bb
            bb, _ = _bin(rho, dv_arr, rp[c], Wp[:, c] * xp[c], smoothing_width, mirror_axis)
            xw += bb
            bb, _ = _bin(rho, dv_arr, rp[c], Wp[:, c] * 0.5 * (3 * xp[c] ** 2 - 1), smoothing_width, mirror_axis)
            p2w += bb
            if classifier is not None:
                bb, _ = _bin(rho, dv_arr, rp[c], Wp[:, c] * xcd[c], smoothing_width, mirror_axis)
                cw += bb
    with np.errstate(invalid="ignore", divide="ignore"):
        xi0 = np.where(births > 0, xw / births, 0.0)
        p2 = np.where(births > 0, p2w / births, 0.0)
        xi_cd = None if cw is None else np.clip(np.where(births > 0, cw / births, 0.0), -1.0, 1.0)
    info["rho_closest_approach"] = {"min": float(np.nanmin(rho_min)) if np.any(np.isfinite(rho_min)) else None,
                                    "weighted_mean": (float(np.nansum(wt * rho_min) / np.sum(wt[np.isfinite(rho_min)]))
                                                      if np.any(np.isfinite(rho_min)) else None)}
    tf = None if classifier is None else np.where(born_w > 0, trapped_w / np.where(born_w > 0, born_w, 1.0), 0.0)
    return BeamDeposition(births=births, xi0=xi0, p2_xi0=p2, shine_through=shine, xi_cd=xi_cd, trapped_births=tf,
                          info=info)


def _beam_record(b):
    rec = {k: (list(map(float, v)) if isinstance(v, (tuple, list, np.ndarray)) else v)
           for k, v in b.__dict__.items() if k != "deposition"}
    rec["deposition_given"] = b.deposition is not None
    return rec


class SimpleBeamModel:
    r"""Provider: neutral beams -> :class:`.provider.NonInductiveSources` (module docstring).

    Settings (all recorded in provenance; the defaults are the bug-fixed model):
    ``smoothing_width`` (rho units, pencil beams only; 0 = cloud-in-cell binning),
    ``mirror_axis`` (True), ``shielding`` (``"redl_l31"``), ``ln_lambda_e`` (``"nrl_ei"``),
    ``ln_lambda_i`` (``"nrl_beam_ion"``), ``cross_sections`` (FREYA fits), ``n_steps`` (4000),
    ``compute_nbcd`` (False: no driven current is returned; True: returned and flagged NOT
    validated, module docstring), ``zeff_check_rtol`` (None: the Z_eff given and the one
    implied by the ion list are compared and the difference recorded; a number: refuse when
    the relative difference exceeds it).

    Options (module docstring): ``deposition`` (``"pencil"``, the default | ``"beamlets"``), with
    ``optics`` (``{beam name: :class:`.beam_optics.BeamOptics`}``; a beam without an entry stays a
    pencil beam, recorded) and ``n_beamlets`` (required with ``"beamlets"``: per divergence
    component, chosen by the caller's convergence test); ``trapping`` (``"zero_orbit_width"``, the
    default | ``"none"``), which acts on the driven current only, i.e. only with
    ``compute_nbcd=True``: it uses ``field`` in :meth:`compute`; without a field, or for a beam given
    as a deposition, the birth pitch is used and the record says "requested, not applied" with the
    reason.  The settings and records of an option appear only when it acts (trapping: when
    ``compute_nbcd=True``), so records with ``compute_nbcd=False`` are those of the model before the
    options.
    """

    #: Reason attached to every built-in driven current (module docstring).
    NBCD_NOT_VALIDATED = ("built-in NBCD: cylindrical fast-ion current with birth-point pitch, no fast-ion "
                          "trapping; 1.5-2.4x a transport code per MW in tests; no calibration applied")
    #: Reason attached with ``trapping="zero_orbit_width"`` (module docstring of :mod:`.trapping`).
    NBCD_NOT_VALIDATED_TRAPPING = (
        "built-in NBCD with zero-orbit-width fast-ion trapping: trapped births carry no current, passing births "
        "their transit-averaged pitch; NOT included: pitch scattering across the trapped-passing boundary during "
        "slowing down, finite orbit width, first-orbit loss (counter injection makes the net current "
        "ill-conditioned); not compared with a bounce-averaged model; no calibration applied")
    #: Appended to NBCD_NOT_VALIDATED when the (default) trapping was requested but could not act on a beam.
    TRAPPING_NOT_APPLIED = "; trapping requested (the default) but NOT applied to this beam: {why}"

    def __init__(self, beams, *, smoothing_width, mirror_axis=True, shielding="redl_l31", ln_lambda_e="nrl_ei",
                 ln_lambda_i="nrl_beam_ion", cross_sections=None, n_steps=4000, compute_nbcd=False,
                 zeff_check_rtol=None, deposition="pencil", optics=None, n_beamlets=None,
                 trapping="zero_orbit_width"):
        if not beams:
            raise ValueError("no beams given")
        names = [b.name for b in beams]
        if len(set(names)) != len(names):
            raise ValueError("beam names must be unique")
        self.beams = list(beams)
        self.smoothing_width = float(smoothing_width)
        self.mirror_axis = bool(mirror_axis)
        self.shielding = shielding
        self.ln_lambda_e = ln_lambda_e
        self.ln_lambda_i = ln_lambda_i
        self.cross_sections = FreyaCrossSections() if cross_sections is None else cross_sections
        self.n_steps = int(n_steps)
        self.compute_nbcd = bool(compute_nbcd)
        self.zeff_check_rtol = None if zeff_check_rtol is None else float(zeff_check_rtol)
        self._clamp_log = {}
        if deposition not in DEPOSITION_MODES:
            raise ValueError(f"deposition must be one of {DEPOSITION_MODES}")
        if trapping not in TRAPPING_MODES:
            raise ValueError(f"trapping must be one of {TRAPPING_MODES}")
        self.deposition = deposition
        self.trapping = trapping
        if deposition == "pencil":
            if optics is not None or n_beamlets is not None:
                raise ValueError("optics and n_beamlets apply only with deposition='beamlets'")
            self.optics, self.n_beamlets = {}, None
        else:
            if n_beamlets is None or int(n_beamlets) < 1:
                raise ValueError("deposition='beamlets' needs n_beamlets >= 1 (chosen by a convergence test)")
            self.n_beamlets = int(n_beamlets)
            self.optics = dict(optics or {})
            for k, v in self.optics.items():
                if k not in names:
                    raise ValueError(f"optics given for {k!r}, which is not a beam")
                if not isinstance(v, BeamOptics):
                    raise ValueError(f"optics[{k!r}] must be a BeamOptics")

    def settings(self):
        out = {"smoothing_width": self.smoothing_width, "mirror_axis": self.mirror_axis, "shielding": self.shielding,
               "ln_lambda_e": self.ln_lambda_e, "ln_lambda_i": self.ln_lambda_i,
               "cross_sections": getattr(self.cross_sections, "name", type(self.cross_sections).__name__),
               "n_steps": self.n_steps,
               "compute_nbcd": self.compute_nbcd if not self.compute_nbcd else "True (output flagged NOT validated)",
               "zeff_check_rtol": self.zeff_check_rtol, "banana_shift": "not applied (declared difference from FUSE)",
               "fast_ion_density_lower_limit": "v = 0 (no thermalization cutoff; slowing_down module docstring)"}
        if self.deposition != "pencil":            # option keys only when on (default records unchanged)
            out.update(deposition=self.deposition, n_beamlets=self.n_beamlets,
                       beamlet_sampling=f"Halton bases {HALTON_BASES}, points 1..n per divergence component",
                       beamlet_power_rule="beam power = launched power (after apertures), shared by transmitted "
                                          "beamlets",
                       beamlet_smoothing="smoothing_width applied to every beamlet")
        if self.trapping != "none" and self.compute_nbcd:     # acts on the driven current only
            out.update(trapping=self.trapping + " (driven current only; trapping module T1-T4; birth pitch where "
                                                "no field or no birth positions, recorded)")
        return out

    def _classifier(self, field):
        if self.trapping == "none" or not self.compute_nbcd or field is None:
            return None
        if not isinstance(field, FluxSurfaceField):
            raise ValueError("field must be a FluxSurfaceField")
        return lambda r, R, Z, x: classify_births(field, r, R, Z, x)

    def deposit(self, beam, rho, dvol_drho, ne, te_ev, ions, rho_of_rz, field=None):
        """Births, <xi0>, <P_2(xi0)> and shine-through of one beam on ``rho`` (with the options: beamlets, xi_cd).
        A beam given as a deposition has no birth positions, so trapping does not act on it (``xi_cd`` is None)."""
        classifier = self._classifier(field)
        if beam.deposition is not None:
            dep = beam.deposition
            g = np.asarray(dep["rho"], float)
            shapes = dep["shape"]
            shapes = [shapes] * 3 if np.ndim(shapes[0]) == 0 else list(shapes)
            if len(shapes) != 3:
                raise ValueError(f"{beam.name}: deposition shape must be one array or three")
            if "shine_through" not in dep:
                raise ValueError(f"{beam.name}: deposition dict must state shine_through (per component)")
            shine = np.broadcast_to(np.asarray(dep["shine_through"], float), (3,)).copy()
            dv = np.asarray(dvol_drho, float)
            births = np.zeros((3, rho.size))
            for k in range(3):
                sh = np.interp(rho, g, np.asarray(shapes[k], float), left=0.0, right=0.0)
                if np.any(sh < 0):
                    raise ValueError(f"{beam.name}: negative deposition shape")
                norm = volume_integral(rho, dv, sh)
                if norm <= 0:
                    raise ValueError(f"{beam.name}: deposition shape integrates to zero")
                births[k] = sh / norm * beam.particle_rates[k] * (1.0 - shine[k])
            xi = np.interp(rho, g, np.broadcast_to(np.asarray(dep["xi0"], float), g.shape))
            p2 = (np.interp(rho, g, np.broadcast_to(np.asarray(dep["p2_xi0"], float), g.shape))
                  if "p2_xi0" in dep else 0.5 * (3 * xi ** 2 - 1))
            return BeamDeposition(births=births, xi0=np.tile(xi, (3, 1)), p2_xi0=np.tile(p2, (3, 1)),
                                  shine_through=shine)
        if rho_of_rz is None:
            raise ValueError(f"{beam.name}: pencil geometry needs rho_of_rz from the caller's equilibrium")
        hyd = [(ion.n, ion.Z) for ion in ions if ion.Z <= 1.0 + 1e-9]
        imp = [(ion.n, ion.Z) for ion in ions if ion.Z > 1.0 + 1e-9]
        if self.deposition == "beamlets" and beam.name in self.optics:
            return beamlet_deposition(beam, self.optics[beam.name], rho_of_rz, rho, dvol_drho, ne, te_ev, hyd, imp,
                                      n_beamlets=self.n_beamlets, cross_sections=self.cross_sections,
                                      n_steps=self.n_steps, smoothing_width=self.smoothing_width,
                                      mirror_axis=self.mirror_axis, clamp_log=self._clamp_log, classifier=classifier)
        if classifier is not None:
            return pencil_beam_deposition(beam, rho_of_rz, rho, dvol_drho, ne, te_ev, hyd, imp, self.cross_sections,
                                          self.n_steps, self.smoothing_width, self.mirror_axis,
                                          clamp_log=self._clamp_log, classifier=classifier)
        return pencil_beam_deposition(beam, rho_of_rz, rho, dvol_drho, ne, te_ev, hyd, imp, self.cross_sections,
                                      self.n_steps, self.smoothing_width, self.mirror_axis,
                                      clamp_log=self._clamp_log)

    def compute(self, rho, *, time, ne, te_ev, ions, zeff, ft, B0, R0, dvol_drho, rho_of_rz=None, uncertainty=None,
                time_window=0.0, inputs=None, eps=None, field=None):
        r"""Driven current, fast-ion pressure/density and heating of all beams at one slice.

        ``rho`` -- rho_tor_norm grid; ``ne`` [m^-3], ``te_ev`` [eV], ``zeff``, ``ft`` (trapped
        fraction, for the shielding; ``eps`` instead for ``shielding="start_cordey_fit"``) and
        ``dvol_drho`` [m^3] are arrays on it; ``ions`` a list of
        :class:`.slowing_down.IonSpecies`; ``B0`` [T] and ``R0`` [m] the reference field and radius;
        ``field`` -- a :class:`.trapping.FluxSurfaceField` from the same equilibrium (used by the
        trapping of the driven current, the default; without it the birth pitch is used, recorded).
        """
        trap_req = self.trapping != "none" and self.compute_nbcd
        trap = self._classifier(field) is not None
        applied, not_applied = [], {}
        rho = np.asarray(rho, float)
        ne, te = np.asarray(ne, float), np.asarray(te_ev, float)
        zeff = np.broadcast_to(np.asarray(zeff, float), rho.shape)
        # Z_eff (used in the shielding) against the one implied by the ion list (used in E_c, R_D and stopping)
        zeff_ions = sum(np.asarray(i.n, float) * i.Z ** 2 for i in ions) / ne
        zrel = float(np.max(np.abs(zeff_ions / zeff - 1.0)))
        if self.zeff_check_rtol is not None and zrel > self.zeff_check_rtol:
            raise ValueError(f"Z_eff given and Z_eff of the ion list (sum n Z^2 / n_e) differ by up to {zrel:.3g} "
                             f"(relative) > zeff_check_rtol = {self.zeff_check_rtol:g}")
        self._clamp_log = {}
        jdotb, p_fast, n_fast, p_par, p_perp, heating = {}, {}, {}, {}, {}, {}
        diag = {}
        for b in self.beams:
            dep = self.deposit(b, rho, dvol_drho, ne, te, ions, rho_of_rz, field=field)
            use_cd = trap and dep.xi_cd is not None
            if use_cd:
                applied.append(b.name)
            elif trap_req:
                not_applied[b.name] = ("no field (FluxSurfaceField) given: birth pitch used (missing input)"
                                       if not trap else "beam given as a deposition has no birth positions: birth "
                                                        "pitch used (missing input)")
            jf = np.zeros_like(rho)
            tot = {k: np.zeros_like(rho) for k in ("n_fast", "p_fast", "p_par", "p_perp", "q_e", "q_i")}
            for k, Ek in enumerate(b.energies):
                m = classical_moments(dep.births[k], Ek, ne, te, ions, b.A, b.Z, xi0=dep.xi0[k],
                                      ln_lambda_e=self.ln_lambda_e, ln_lambda_i=self.ln_lambda_i,
                                      p2_xi0=dep.p2_xi0[k])
                if use_cd:        # T4: the current pitch replaces the birth pitch in S11 (current only)
                    jf += classical_moments(dep.births[k], Ek, ne, te, ions, b.A, b.Z, xi0=dep.xi_cd[k],
                                            ln_lambda_e=self.ln_lambda_e, ln_lambda_i=self.ln_lambda_i)["j_fast"]
                else:
                    jf += m["j_fast"]
                for key in tot:
                    tot[key] += m[key]
            if self.compute_nbcd:
                Fb = shielding_factor(zeff, b.Z, self.shielding, ft=ft, eps=eps)
                jdotb[b.name] = jf * Fb * B0      # <J.B> = j_par B0 (module LIMITATIONS); NOT validated
            sp = b.species_label
            for d, key in ((p_fast, "p_fast"), (n_fast, "n_fast"), (p_par, "p_par"), (p_perp, "p_perp")):
                d[sp] = d.get(sp, 0.0) + tot[key]
            heating[b.name] = (tot["q_e"], tot["q_i"])
            births_int = [float(volume_integral(rho, dvol_drho, dep.births[k])) for k in range(3)]
            diag[b.name] = {"shine_through": dep.shine_through.tolist(), "births_per_s": births_int,
                            "power_fractions": b.p_frac.tolist(), "current_fractions": b.c_frac.tolist()}
            if self.deposition == "beamlets":          # option records only when on
                if b.name in self.optics:
                    diag[b.name].update(deposition="beamlets", beamlets=dep.info)
                elif b.deposition is not None:
                    diag[b.name].update(deposition="given deposition (beamlets not applicable)")
                else:
                    diag[b.name].update(deposition="pencil: no finite-beam description given (missing input)")
            if use_cd:
                diag[b.name].update(trapped_birth_fraction=np.asarray(dep.trapped_births, float).tolist())
            elif trap_req:
                diag[b.name].update(trapping="requested, not applied: " + not_applied[b.name])
        extra_meta = {}
        extra_limitations = []
        if trap_req:
            extra_meta["trapping"] = {"requested": self.trapping, "applied_to": applied, "not_applied": not_applied}
            if not_applied:
                extra_limitations.append("trapping requested (the default) but not applied to "
                                         + ", ".join(sorted(not_applied)) + ": birth pitch used (missing input)")
            p_ctr = float(sum(b.power for b in self.beams if b.direction == -1))
            p_all = float(sum(b.power for b in self.beams))
            extra_meta["counter_injection"] = {
                "counter_power_W": p_ctr, "total_power_W": p_all, "flag": p_ctr > 0,
                "note": "counter-injected power present: the net driven current is a difference of co and counter "
                        "currents and is ill-conditioned without first-orbit loss (not modeled)" if p_ctr > 0
                        else "no counter-injected power"}
            if field is not None:
                extra_meta["trapping_field"] = field.record()
            if p_ctr > 0:
                extra_limitations.append(f"counter-injected power present ({p_ctr:.4g} of {p_all:.4g} W): net driven "
                                         "current ill-conditioned without first-orbit loss (not modeled)")
        prov = Provenance(
            provider="simple_beam", source_code="kinrecon.sources.beam", source_version=SOURCES_VERSION,
            inputs=dict(inputs or {}, time=time, beams=[_beam_record(b) for b in self.beams],
                        sha256=array_sha256(rho, ne, te, zeff, dvol_drho, *[i.n for i in ions]),
                        zeff_vs_ions_max_rel_difference=zrel, cross_section_clamps=dict(self._clamp_log),
                        nbcd="computed, NOT validated" if self.compute_nbcd else "not computed (compute_nbcd=False)"),
            conventions={"current": "<J.B> = j_par B0, signed relative to Ip",
                         "pressure": "isotropic equivalent (C8); true components in p_par/p_perp"},
            settings=self.settings(),
            assumptions=["classical slowing down, steady state", "beam fractions as given (neutral-beam fractions)",
                         "thermal ions Maxwellian with v_thi << v; electrons with v << v_the"],
            limitations=list(BEAM_LIMITATIONS),
            notes="Re-derived zeroth-order model; differences from FUSE ActorSimpleNB: " + "; ".join(FUSE_DIFFERENCES)
                  + ".  Credits: FUSE / IMAS.jl (Apache-2.0, General Atomics) for the model structure and the FREYA "
                    "cross-section fits.")
        if self.deposition == "beamlets":
            prov.limitations[0] = BEAMLET_LIMITATION
        if applied:
            prov.limitations = [TRAPPING_LIMITATION if x.startswith("no fast-ion trapping") else x
                                for x in prov.limitations]
        if trap_req:
            prov.limitations = prov.limitations + extra_limitations

        def reason_of(name):
            if name in applied:
                return self.NBCD_NOT_VALIDATED_TRAPPING
            if name in not_applied:
                return self.NBCD_NOT_VALIDATED + self.TRAPPING_NOT_APPLIED.format(why=not_applied[name])
            return self.NBCD_NOT_VALIDATED
        reason = (self.NBCD_NOT_VALIDATED_TRAPPING if applied and not not_applied else
                  self.NBCD_NOT_VALIDATED_TRAPPING + " (not applied to: " + ", ".join(sorted(not_applied)) + ")"
                  if applied else self.NBCD_NOT_VALIDATED + (self.TRAPPING_NOT_APPLIED.format(
                      why="no field or no birth positions") if not_applied else ""))
        if self.compute_nbcd:
            prov.limitations.insert(0, "NOT VALIDATED: the returned beam-driven current (" + reason + ")")
        return NonInductiveSources(rho=rho, time=time, time_window=time_window, B0=B0, R0=R0, jdotb=jdotb,
                                   p_fast=p_fast, n_fast=n_fast, p_par=p_par, p_perp=p_perp, heating=heating,
                                   uncertainty=list(uncertainty or []), provenance=prov,
                                   meta=dict({"beams": diag}, **extra_meta),
                                   not_validated={f"jdotb:{k}": reason_of(k) for k in jdotb})
