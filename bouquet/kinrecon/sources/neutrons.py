r"""D-D neutron-rate ESTIMATES: thermonuclear and classical beam-target (diagnostic use only).

A neutron rate from the provider's own fast-ion distribution is the cheapest measured check
of the fast-ion content (report 28 of the project record).  These are estimates, clearly
labeled: no beam-beam reactions, the isotropic part of the classical distribution only (the
rate of a fast particle on an isotropic target does not depend on its pitch), and either a
cold (stationary) target or a Maxwellian target at T_i (eq. D5).

D(d,n)3He cross section and Maxwellian reactivity: Bosch and Hale, Nucl. Fusion 32, 611
(1992), eqs. 8-9 (cross section, Table IV coefficients) and eqs. 12-14 (reactivity, Table
VII coefficients); coefficients recalled, then cross-checked against each other in the
tests (the reactivity fit and the Maxwellian average of the cross-section fit agree within
the fits' stated accuracy).

.. code-block:: text

    sigma(E) = S(E) / (E exp(B_G / sqrt(E))),  S = A1 + E (A2 + E (A3 + E (A4 + E A5)))      [mb, E = CM keV]   (D1)
    <sigma v>(T) = C1 theta sqrt(xi / (m_r c^2 T^3)) exp(-3 xi),
    theta = T / (1 - T (C2 + T (C4 + T C6)) / (1 + T (C3 + T (C5 + T C7)))),
    xi = (B_G^2 / (4 theta))^(1/3)                                                  [cm^3/s, T keV]   (D2)
    R_TN  = (1/2) n_D^2 <sigma v>                                                     (identical particles)   (D3)
    R_BT  = n_D S tau_s int_0^v0 sigma(E_cm(v)) v^3 / (v^3 + v_c^3) dv,  E_cm = m_r v^2 / 2           (D4)

D4 is the classical slowing-down distribution of :mod:`.slowing_down` (eq. S6, l = 0)
reacting with a stationary target; ``m_r = m_b m_D/(m_b + m_D)`` is the reduced mass.  With a
Maxwellian target of temperature T_i the product ``sigma(v) v`` in D4 is replaced by its
average over the target velocities (isotropic Maxwellian, thermal speed ``u = sqrt(2 T_i/m_t)``):

.. code-block:: text

    <sigma v_r>(v_b) = (1/(sqrt(pi) v_b u)) int_0^inf sigma(v_r) v_r^2
                       [exp(-(v_r - v_b)^2/u^2) - exp(-(v_r + v_b)^2/u^2)] dv_r            (D5)

(the angular integral of a shifted Maxwellian, standard; it tends to sigma(v_b) v_b as u -> 0).
Validity of D1: 0.5-5000 keV; of D2: 0.2-100 keV.
"""

import numpy as np

from .conventions import E_CHARGE, M_AMU

__all__ = ["dd_n_cross_section", "dd_n_reactivity", "thermonuclear_rate", "beam_target_rate"]

_BG = 31.3970                      # sqrt(keV)
_A = (5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9)
_MRC2 = 937814.0                   # keV
_C = (5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0, -2.964e-6, 0.0, 0.0)
_A_D = 2.01410177812               # u (deuteron atomic mass; nuclear mass differs by m_e, negligible here)


def dd_n_cross_section(e_cm_kev):
    """D(d,n)3He cross section [m^2] at CM energy [keV] (eq. D1); 0 below 0.5 keV."""
    e = np.asarray(e_cm_kev, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        S = _A[0] + e * (_A[1] + e * (_A[2] + e * (_A[3] + e * _A[4])))
        sig_mb = S / (e * np.exp(_BG / np.sqrt(e)))
    return np.where(e >= 0.5, sig_mb, 0.0) * 1e-31


def dd_n_reactivity(t_kev):
    """Maxwellian D(d,n)3He reactivity [m^3/s] at ion temperature [keV] (eq. D2); 0 below 0.2 keV."""
    t = np.asarray(t_kev, float)
    c1, c2, c3, c4, c5, c6, c7 = _C
    tt = np.maximum(t, 0.2)
    theta = tt / (1.0 - tt * (c2 + tt * (c4 + tt * c6)) / (1.0 + tt * (c3 + tt * (c5 + tt * c7))))
    xi = (_BG ** 2 / (4.0 * theta)) ** (1.0 / 3.0)
    sv = c1 * theta * np.sqrt(xi / (_MRC2 * tt ** 3)) * np.exp(-3.0 * xi)
    return np.where(t >= 0.2, sv, 0.0) * 1e-6


def thermonuclear_rate(n_d, ti_kev):
    """Thermonuclear D-D neutron rate density [m^-3 s^-1] = n_D^2 <sigma v>/2 (eq. D3)."""
    return 0.5 * np.asarray(n_d, float) ** 2 * dd_n_reactivity(ti_kev)


_GLX, _GLW = np.polynomial.legendre.leggauss(64)


def _warm_kernel(v, u, m_r):
    """<sigma v_r> over a Maxwellian target (eq. D5) at beam speeds ``v`` (1-D) and target speed ``u`` (array)."""
    t = np.linspace(-6.0, 6.0, 241)
    vr = v[None, :, None] + u[:, None, None] * t[None, None, :]             # (n_rho, n_v, n_t)
    vr = np.abs(vr)
    sig = dd_n_cross_section(0.5 * m_r * vr ** 2 / E_CHARGE / 1e3)
    vb, uu = v[None, :, None], u[:, None, None]
    kern = np.exp(-((vr - vb) / uu) ** 2) - np.exp(-((vr + vb) / uu) ** 2)
    integ = np.trapezoid(sig * vr ** 2 * kern, t, axis=-1) * u[:, None]      # dv_r = u dt
    return integ / (np.sqrt(np.pi) * v[None, :] * u[:, None])


def beam_target_rate(S, E0_ev, A_b, tau_s, v_c, n_target, A_target=_A_D, ti_kev=None):
    """Classical beam-target D-D neutron rate density [m^-3 s^-1] for one component (eqs. D4-D5).

    ``S`` [m^-3 s^-1], ``tau_s`` [s], ``v_c`` [m/s] and ``n_target`` [m^-3] are arrays on rho
    (from :func:`.slowing_down.classical_moments`: ``tau_s``, and ``v_c = v0 w``).  ``ti_kev``
    (array, keV) switches on the Maxwellian-target average D5; ``None`` = cold target.
    """
    S, tau_s, v_c, nt = (np.asarray(a, float) for a in (S, tau_s, v_c, n_target))
    m_b = A_b * M_AMU
    m_t = A_target * M_AMU
    m_r = m_b * m_t / (m_b + m_t)
    v0 = np.sqrt(2.0 * E0_ev * E_CHARGE / m_b)
    x = 0.5 * (_GLX + 1.0)                     # v / v0 on (0, 1)
    v = v0 * x
    if ti_kev is None:
        e_cm_kev = 0.5 * m_r * v ** 2 / E_CHARGE / 1e3
        sv = np.broadcast_to(dd_n_cross_section(e_cm_kev) * v, S.shape + v.shape)
    else:
        u = np.sqrt(2.0 * np.asarray(ti_kev, float) * 1e3 * E_CHARGE / m_t) * np.ones(S.shape)
        sv = _warm_kernel(v, u, m_r)
    integrand = sv * v[None, :] ** 2 / (v[None, :] ** 3 + v_c[..., None] ** 3)
    integral = 0.5 * v0 * np.sum(integrand * _GLW, axis=-1)
    return nt * S * tau_s * integral
