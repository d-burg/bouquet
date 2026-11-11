r"""Built-in zeroth-order electron-cyclotron current drive (ECCD) estimate.

No ray tracing: the deposition location ``rho0`` and width are SUPPLIED (from a ray/beam
tracer, a cold-resonance estimate, or a scan), and the driven current follows from a stated
dimensionless efficiency.  The efficiency is the conventional one

.. code-block:: text

    zeta = (e^3 / eps0^2) I_EC n_e R0 / (P_EC T_e)        (T_e in J)
         = 3.27 I_EC[A] R0[m] n_e[1e19 m^-3] / (P_EC[W] T_e[keV])                          (E1)

evaluated at the deposition location (Luce et al., Phys. Rev. Lett. 83, 4550 (1999);
Petty et al., Nucl. Fusion 42, 1366 (2002); recalled; the 3.27 coefficient is reproduced
from SI constants in the tests).  Two declared models for zeta:

``"constant"``      ``zeta = zeta0``;
``"fisch_boozer"``  ``zeta = zeta0 * 6 / (5 + Z_eff(rho0))`` -- the Z dependence of the
                    Fisch-Boozer high-velocity limit, efficiency proportional to 1/(5 + Z)
                    (Fisch and Boozer, Phys. Rev. Lett. 45, 720 (1980); recalled), normalized
                    so that zeta0 is the Z_eff = 1 value.

``zeta0`` has NO default: it must come from a tracer calculation, a measurement or the
user.  For reference, FUSE ``ActorSimpleEC`` uses ``I = c 0.09 T_e[keV] P / ((5 + Z_eff) R0
n_e20)`` with a per-case scale ``c``; in the notation above that is
``"fisch_boozer"`` with ``zeta0 = 32.7 * 0.09 * c / 6 = 0.4905 c`` (E1 with n_e20 = n_e19/10).

Profile: an axis-mirrored Gaussian in rho of standard deviation ``width``, normalized so the
driven toroidal current is ``I_EC`` (C7).  With ``geometry`` the frame conversion is exact
(C3-C4); without it the declared large-aspect-ratio relation ``<J.B> = B0 j_R0`` is used.

LIMITATIONS: no ray tracing, refraction, Doppler/relativistic resonance broadening or
multiple passes; full single-pass absorption; no trapped-electron (Ohkawa) degradation of
zeta beyond what zeta0 carries; no dependence on launch angle; Gaussian shape.
"""

from dataclasses import dataclass

import numpy as np

from .conventions import E_CHARGE, EPS0, array_sha256, jdotb_from_jphi_over_r
from .provider import SOURCES_VERSION, NonInductiveSources, Provenance

__all__ = ["ECLauncher", "SimpleECModel", "eccd_efficiency_coefficient", "EC_LIMITATIONS", "ZETA_MODELS"]

ZETA_MODELS = ("constant", "fisch_boozer")
EC_LIMITATIONS = [
    "no ray tracing: deposition location and width supplied by the caller",
    "full single-pass absorption; no refraction, relativistic broadening or multiple passes",
    "efficiency from a stated zeta model; no trapped-electron degradation beyond zeta0",
    "Gaussian deposition profile",
]


def eccd_efficiency_coefficient():
    """The coefficient of E1 in practical units: (e^3/eps0^2) * 1e19 / (1e3 e) ~= 3.27."""
    return E_CHARGE ** 3 / EPS0 ** 2 * 1.0e19 / (1.0e3 * E_CHARGE)


@dataclass
class ECLauncher:
    """One EC launcher: ``power`` [W] absorbed, ``rho0`` and ``width`` (rho units), ``direction`` (+1 co-Ip)."""

    name: str
    power: float
    rho0: float
    width: float
    direction: int

    def __post_init__(self):
        if not (self.power >= 0 and 0 <= self.rho0 <= 1 and self.width > 0):
            raise ValueError(f"{self.name}: power >= 0, 0 <= rho0 <= 1, width > 0 required")
        if self.direction not in (1, -1):
            raise ValueError(f"{self.name}: direction must be +1 or -1")


class SimpleECModel:
    """Provider for ECCD (module docstring).  ``zeta0`` and ``zeta_model`` are required."""

    def __init__(self, launchers, *, zeta0, zeta_model):
        if zeta_model not in ZETA_MODELS:
            raise ValueError(f"zeta_model must be one of {ZETA_MODELS}")
        if not (zeta0 > 0):
            raise ValueError("zeta0 must be positive")
        self.launchers = list(launchers)
        self.zeta0 = float(zeta0)
        self.zeta_model = zeta_model

    def current(self, launcher, ne0, te0_ev, zeff0, R0):
        """I_EC [A] from E1 at the deposition point (signed relative to Ip)."""
        zeta = self.zeta0 * (6.0 / (5.0 + zeff0) if self.zeta_model == "fisch_boozer" else 1.0)
        return launcher.direction * zeta * EPS0 ** 2 * launcher.power * te0_ev * E_CHARGE / (E_CHARGE ** 3 * ne0 * R0)

    def compute(self, rho, *, time, ne, te_ev, zeff, B0, R0, dvol_drho, geometry=None, uncertainty=None,
                time_window=0.0, inputs=None):
        rho = np.asarray(rho, float)
        dv = np.asarray(dvol_drho, float)
        jdotb, diag = {}, {}
        for L in self.launchers:
            ne0 = float(np.interp(L.rho0, rho, ne))
            te0 = float(np.interp(L.rho0, rho, te_ev))
            z0 = float(np.interp(L.rho0, rho, np.broadcast_to(zeff, rho.shape)))
            I = self.current(L, ne0, te0, z0, R0)
            shape = np.exp(-0.5 * ((rho - L.rho0) / L.width) ** 2) + np.exp(-0.5 * ((rho + L.rho0) / L.width) ** 2)
            # <J_phi/R> = I * shape / (int shape dV / 2pi)   (C7)
            norm = np.trapezoid(shape * dv, rho) / (2.0 * np.pi)
            jphi_r = I * shape / norm
            if geometry is not None:
                jdotb[L.name] = jdotb_from_jphi_over_r(jphi_r, rho, geometry)
            else:
                jdotb[L.name] = B0 * R0 * jphi_r       # declared large-aspect-ratio relation <J.B> = B0 j_R0
            diag[L.name] = {"I_EC": I, "ne0": ne0, "te0_ev": te0, "zeff0": z0}
        prov = Provenance(
            provider="simple_ec", source_code="kinrecon.sources.ec", source_version=SOURCES_VERSION,
            inputs=dict(inputs or {}, time=time, launchers=[L.__dict__ for L in self.launchers],
                        sha256=array_sha256(rho, ne, te_ev, dv)),
            conventions={"current": "<J.B>, signed relative to Ip",
                         "frame": "exact (C3-C4)" if geometry is not None else "large-aspect-ratio <J.B> = B0 j_R0"},
            settings={"zeta0": self.zeta0, "zeta_model": self.zeta_model},
            assumptions=["zeta0 supplied by the caller (" + ("decided" if inputs and inputs.get("zeta0_decided")
                                                             else "basis to be stated by the caller") + ")"],
            limitations=list(EC_LIMITATIONS))
        return NonInductiveSources(rho=rho, time=time, time_window=time_window, B0=B0, R0=R0, jdotb=jdotb,
                                   p_fast={}, n_fast={}, uncertainty=list(uncertainty or []), provenance=prov,
                                   meta={"launchers": diag})
