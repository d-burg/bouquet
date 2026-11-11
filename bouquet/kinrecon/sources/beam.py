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
the cylindrical Gaffey result (S11) with the birth-point pitch: fast ions born in the trapped
region of pitch space are counted as passing, and passing ions carry their birth pitch rather
than the transit average.  Against a transport code this overestimated the driven current by
1.5-2.4x per MW for tangential injection, consistent with an independent Monte-Carlo estimate
of the trapping effect (x1.5-2.0); a bounce-averaged fast-ion model (Cordey, Nucl. Fusion 16,
499 (1976), recalled) is the remedy and is not implemented.  :class:`SimpleBeamModel` therefore
returns NO driven current unless ``compute_nbcd=True``, and then marks every beam's ``jdotb``
as not validated in the output (``not_validated``), in its provenance and in its limitations;
reading it raises until the caller acknowledges it
(:meth:`.provider.NonInductiveSources.acknowledge_not_validated`).  No calibration factor is
applied.  The fast-ion pressure, density and heating are unaffected by this flag.

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

LIMITATIONS (recorded in provenance): pencil beams (no divergence or source size except via
the smoothing width), straight-line orbits of the neutrals, no finite-orbit width of the
fast ions (births are mapped to the flux surface of the ionization point), no prompt or
orbit loss, no fast-ion CX loss, no anomalous fast-ion transport (Alfven eigenmodes,
sawteeth, tearing modes), no beam-beam collisions, steady state per slice (no time history),
a 1-D current model (the fast-ion parallel current is equated with <J.B>/B0, using the
birth-point pitch: an order-one error near the trapped-passing boundary, not O(eps^2)),
collisionless trapped-electron correction, NO FAST-ION TRAPPING (see above), no inward orbit
shift of the births (see above).  Cross-section fit clamps (T_e outside [1, 1e5] eV; proton
impact zeroed below 0.1 keV/u) are counted per call and recorded in provenance
(``inputs["cross_section_clamps"]``).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .conventions import E_CHARGE, M_AMU, array_sha256
from .nbcd import shielding_factor
from .provider import SOURCES_VERSION, NonInductiveSources, Provenance
from .slowing_down import IonSpecies, classical_moments

__all__ = ["NeutralBeam", "volume_weights", "volume_integral", "power_fractions_from_current", "current_fractions_from_power", "FreyaCrossSections",
           "pencil_beam_deposition", "BeamDeposition", "SimpleBeamModel", "BEAM_LIMITATIONS", "FUSE_DIFFERENCES"]

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
                           cross_sections=None, n_steps=4000, smoothing_width=0.0, mirror_axis=True, clamp_log=None):
    r"""Attenuate a pencil beam (eq. B3) and return births per component on ``rho``.

    ``rho_of_rz(R, Z)`` gives rho_tor_norm (NaN outside the plasma) from the caller's
    equilibrium; ``ne``, ``te_ev`` are arrays on ``rho``; ``hydrogenic`` / ``impurities`` are
    lists of ``(n_array_on_rho, Z)``.  The beam travels from ``(source_r, 0, source_z)`` toward
    the co-Ip toroidal direction for ``direction = +1`` (internal positive frame).  ``clamp_log``
    (a dict) accumulates the cross-section fits' clamp counts.
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
    return BeamDeposition(births=births, xi0=xi0, p2_xi0=p2, shine_through=shine,
                          path={"R": R, "Z": Z, "rho": rr, "s": s})


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
    """

    #: Reason attached to every built-in driven current (module docstring).
    NBCD_NOT_VALIDATED = ("built-in NBCD: cylindrical fast-ion current with birth-point pitch, no fast-ion "
                          "trapping; 1.5-2.4x a transport code per MW in tests; no calibration applied")

    def __init__(self, beams, *, smoothing_width, mirror_axis=True, shielding="redl_l31", ln_lambda_e="nrl_ei",
                 ln_lambda_i="nrl_beam_ion", cross_sections=None, n_steps=4000, compute_nbcd=False,
                 zeff_check_rtol=None):
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

    def settings(self):
        return {"smoothing_width": self.smoothing_width, "mirror_axis": self.mirror_axis, "shielding": self.shielding,
                "ln_lambda_e": self.ln_lambda_e, "ln_lambda_i": self.ln_lambda_i,
                "cross_sections": getattr(self.cross_sections, "name", type(self.cross_sections).__name__),
                "n_steps": self.n_steps,
                "compute_nbcd": self.compute_nbcd if not self.compute_nbcd else "True (output flagged NOT validated)",
                "zeff_check_rtol": self.zeff_check_rtol, "banana_shift": "not applied (declared difference from FUSE)",
                "fast_ion_density_lower_limit": "v = 0 (no thermalization cutoff; slowing_down module docstring)"}

    def deposit(self, beam, rho, dvol_drho, ne, te_ev, ions, rho_of_rz):
        """Births, <xi0>, <P_2(xi0)> and shine-through of one beam on ``rho``."""
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
        return pencil_beam_deposition(beam, rho_of_rz, rho, dvol_drho, ne, te_ev, hyd, imp, self.cross_sections,
                                      self.n_steps, self.smoothing_width, self.mirror_axis,
                                      clamp_log=self._clamp_log)

    def compute(self, rho, *, time, ne, te_ev, ions, zeff, ft, B0, R0, dvol_drho, rho_of_rz=None, uncertainty=None,
                time_window=0.0, inputs=None, eps=None):
        r"""Driven current, fast-ion pressure/density and heating of all beams at one slice.

        ``rho`` -- rho_tor_norm grid; ``ne`` [m^-3], ``te_ev`` [eV], ``zeff``, ``ft`` (trapped
        fraction, for the shielding; ``eps`` instead for ``shielding="start_cordey_fit"``) and
        ``dvol_drho`` [m^3] are arrays on it; ``ions`` a list of
        :class:`.slowing_down.IonSpecies`; ``B0`` [T] and ``R0`` [m] the reference field and radius.
        """
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
            dep = self.deposit(b, rho, dvol_drho, ne, te, ions, rho_of_rz)
            jf = np.zeros_like(rho)
            tot = {k: np.zeros_like(rho) for k in ("n_fast", "p_fast", "p_par", "p_perp", "q_e", "q_i")}
            for k, Ek in enumerate(b.energies):
                m = classical_moments(dep.births[k], Ek, ne, te, ions, b.A, b.Z, xi0=dep.xi0[k],
                                      ln_lambda_e=self.ln_lambda_e, ln_lambda_i=self.ln_lambda_i,
                                      p2_xi0=dep.p2_xi0[k])
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
        if self.compute_nbcd:
            prov.limitations.insert(0, "NOT VALIDATED: the returned beam-driven current (" + self.NBCD_NOT_VALIDATED + ")")
        return NonInductiveSources(rho=rho, time=time, time_window=time_window, B0=B0, R0=R0, jdotb=jdotb,
                                   p_fast=p_fast, n_fast=n_fast, p_par=p_par, p_perp=p_perp, heating=heating,
                                   uncertainty=list(uncertainty or []), provenance=prov, meta={"beams": diag},
                                   not_validated={f"jdotb:{k}": self.NBCD_NOT_VALIDATED for k in jdotb})
