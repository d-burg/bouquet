"""Build the synthetic IMAS-path and g-file test fixtures (FIXTURE_SPEC.md).

    dd_synthetic.json.gz          OMAS/IMAS JSON, five time slices: equilibrium,
                                  core_profiles and core_sources at every time -- spec 1
    diiid_profs_synthetic.cdf     an IDA-layout ("direct") kinetic fit at the dd's
                                  pedestal-forming slice (1.2 s, T_IDA) -- spec 2
    diiid_profs_synthetic_pfast.cdf  the same kinetic fit (same psi_N grid, same
                                  kinetics and sigmas) with the q of
                                  g_synthetic_pfast.geqdsk: the IDA file that
                                  belongs to the total-pressure g-file, for the
                                  routes that pair the dd with a g-file -- spec 2b
    g_synthetic.geqdsk            the kinetic-EFIT-like equilibrium that IDA fit
                                  belongs to (COCOS 1), THERMAL pressure only (the
                                  pressure the g-file + IDA route reads from that
                                  file) -- spec 3
    g_synthetic_pfast.geqdsk      the same slice and current model with the TOTAL
                                  pressure (thermal + fast ions) -- spec 3b
    p_synthetic.peqdsk            an Osborne p-file of that slice (n_e, T_e, n_i,
                                  T_i, n_b, the beam pressure p_b, n_z1, omeg, ptot)
                                  that belongs to g_synthetic_pfast -- spec 3b
    g_synthetic_lmode.geqdsk      a separate upper-single-null L-mode equilibrium
    ida_synthetic_lmode.cdf       and the IDA-layout kinetic fit that belongs to it

Every kinetic sigma (the IDA files' ``*_err``, the dd's core_profiles
``*_error_upper``) is the radial template of the fixture's phase, applied by the
functions of ``apply_sigma_templates.py`` (which also rewrites the envelopes of
existing fixtures in place, without a solve).

Run (OpenFUSIONToolkit on the path, or ``OFT_PYTHONPATH`` set; one thread):

    python tests/data/make_synthetic_fixtures.py OUT_DIR [--mesh PATH]

Everything comes from the PARAMETERS tables below (every number rounded to at
most two significant figures) and is re-solved self-consistently with
TokaMaker on the D3D-like example mesh (``examples/D3D-like/DIIID_mesh.h5``):

* **Profiles** are functions of psi_N: ``hmode`` = ``bouquet.utils.Hmode_profiles``
  (edge, ped, core, expin, expout, widthp, xphalf), ``poly`` = edge + (core -
  edge)(1 - psi_N^a)^b (no pedestal term), optionally times an axis factor
  ``dip`` = 1 - h exp(-psi_N / w).  Units: densities 1e19 m^-3, temperatures keV,
  rotation 1e4 rad/s, the beam (fast-ion) pressure kPa (``p_par + 2 p_perp``, the
  'sum' convention), currents MA/m^2 (<J.B>/B0, except ``JPHI_LMODE``, a plain
  <j_phi>).  Z_eff is a profile; the carbon density follows as
  n_C = n_e (Z_eff - 1) / 30 and the thermal deuterons as
  n_D = n_e - 6 n_C - n_fast (quasineutrality; Z_eff counts the beam deuterons).
* **The dd** (``SLICES``): a five-slice discharge -- ramp-up L-mode, the L-H
  transition, a forming pedestal with strong beam current, the peak pedestal
  (hot ions, low density), and a dense flat-top -- in one lower-single-null
  shape (``LCFS_DD``).  Each slice is its own free-boundary solve whose current
  is j_total = A j_ohmic_shape + j_BS + j_beam (<J.B>/B0): j_BS is
  ``bouquet.physics.evaluate_jBS`` (Redl, default eps definition) on the
  converged equilibrium, the pressure is thermal + beam (its gradient clamped to
  zero on the final psi_N node, :func:`equilibrium_pressure`), the ohmic amplitude A
  gives the slice's Ip, and Newton-corrected ``jphi-linterp`` passes make the
  solve's own <j_phi> equal kappa B0 j_total + p'G (docs/current-conventions.md,
  A7) to the printed residual r_j.  ``core_profiles.j_tor`` of every slice is
  built from its ``j_total`` on the PREVIOUS equilibrium slice (the first slice:
  on itself), as FUSE does (``bouquet.io.imas._jtor_from_jpar``).
* **The g-files of the T_IDA slice** (1.2 s: pedestal forming, ~0.1 MA of
  beam current), both solved with a kinetic-EFIT-like current model -- the
  slice's ohmic shape (amplitude re-solved for Ip) plus the Redl bootstrap, no
  beam current channel (``G_OHMIC``):
  ``g_synthetic_pfast.geqdsk`` carries the slice's TOTAL pressure (thermal +
  beam; it differs from the dd's T_IDA slice by the current model only) and
  belongs to the p-file ``p_synthetic.peqdsk``, whose ``pb`` block is that
  beam pressure (:func:`write_pfile`); ``g_synthetic.geqdsk`` carries the
  THERMAL pressure only, e (n_e T_e + n_i T_i) with n_i from Z_eff by
  single-impurity quasineutrality, i.e. exactly what the g-file + IDA route
  reads from the IDA file (which has no fast-ion channel), and belongs to the
  IDA file.  The thermal-only solve continues from the total-pressure state
  (``PRESSURE_THERMAL``).
* **The L-mode case** (``LMODE``, ``JPHI_LMODE``, ``LCFS_LMODE``): an
  upper-single-null L-mode with q0 < 1; its current is a prescribed smooth
  <j_phi> profile, its pressure is thermal (no beam).

Before any scenario is solved a **sanity gate** (:func:`sanity_gate`) refuses it
(raises, with the numbers) when T_e or T_i at the separatrix exceeds GATE_TSEP x the
slice's reference value (``sep``), j_total(1) > GATE_JSEP x its peak, the j_BS peak
lies inside psi_N < GATE_JBS_PED on a ``pedestal`` slice, or the core of j_total
is not a bounded smooth one: on psi_N < GATE_CORE_X its maximum may exceed its axis
value by at most GATE_CORE_RISE (a shallow off-axis maximum is allowed), and its
second difference on a GATE_CURV_H grid may change sign at most GATE_CURV_CHANGES
times (|d2j/dpsi_N^2| < GATE_CURV_EPS x peak counts as zero) -- no humps or dives; the gate is evaluated on the composition of
the live state before the scenario's first pass (``--gate-only`` evaluates it after
a short fresh-start solve of every scenario and writes nothing but ``gate_dump.npz``)
and again on the converged state (a hard self-check).

A free-boundary solve that stalls (a limit cycle just above nl_tol, which is
never changed) is retried with smaller under-relaxation, then with a halved
Newton step on the requested current, then (first pass only) from a fresh psi.
``--mesh PATH`` runs the solves on another TokaMaker mesh of the same machine
(the fixtures do not depend on the mesh beyond solver accuracy).

The script prints every consistency number of FIXTURE_SPEC.md section 1 and
exits non-zero if a hard one fails.  It writes nothing outside OUT_DIR.
"""
import gzip
import json
import os
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MESH = os.path.join(REPO, "examples", "D3D-like", "DIIID_mesh.h5")   # default; --mesh PATH overrides
sys.path.insert(0, REPO)
if os.environ.get("OFT_PYTHONPATH"):
    sys.path.append(os.environ["OFT_PYTHONPATH"])
sys.path.insert(0, HERE)
from apply_sigma_templates import (IDA_REGIME, add_dd_errors, dd_regime,  # noqa: E402
                                   ida_errors, write_dd)


def P(fam, p, dip=None):
    """A profile: family ``"hmode"`` / ``"poly"``, its parameters, an optional
    axis factor (h, w)."""
    return {"fam": fam, "p": tuple(float(v) for v in p),
            "dip": None if dip is None else tuple(float(v) for v in dip)}


# =============================================================================
#  PARAMETERS (every number rounded to <= 2 significant figures)
# =============================================================================
B0 = -2.0               # T, vacuum field at R0 (signed: COCOS 11 dd; g-file COCOS 1)
R0 = 1.7                # m
Z_IMP = 6.0             # carbon
A_D, A_C = 2.0, 12.0    # atomic masses written to the dd

# Plasma boundaries: 16 isoflux points (R - R0 [m], Z [m]) and the X-point
# (a saddle constraint).
LCFS_DD = dict(  # lower single null
    isoflux=((0.54, 0.2), (0.45, 0.44), (0.28, 0.69), (-0.018, 0.92), (-0.38, 0.84),
             (-0.55, 0.49), (-0.59, 0.2), (-0.6, -0.031), (-0.59, -0.26), (-0.53, -0.54),
             (-0.38, -0.91), (-0.018, -0.9), (0.24, -0.64), (0.4, -0.45), (0.51, -0.25),
             (0.56, -0.031)),
    xpoint=(-0.27, -1.1))
LCFS_LMODE = dict(  # upper single null
    isoflux=((-0.62, 0.087), (-0.6, 0.32), (-0.56, 0.6), (-0.43, 1.0), (-0.01, 0.92),
             (0.24, 0.69), (0.4, 0.47), (0.49, 0.29), (0.56, 0.07), (0.56, -0.16),
             (0.48, -0.44), (0.3, -0.72), (-0.01, -0.92), (-0.38, -0.85), (-0.56, -0.49),
             (-0.6, -0.21)),
    xpoint=(-0.42, 1.1))

# The dd: five slices.  ne/te/ti/zeff/omega kinetics, nfast/pfast the beam
# deuterons, jbeam the beam current, johm the ohmic SHAPE (amplitude solved).
SLICES = (
    dict(name="dd0", t=0.7, ip=0.95e6, pedestal=False,    # ramp-up L-mode, beam just on
         sep=dict(te=0.043, ti=0.043),      # T_i = T_e x (T_i/T_e of the first measured slice), T_i(1) = T_e(1)
         ne=P("hmode", (0.44, 0.67, 1.5, 1.0, 0.89, 0.1, 1.0)),
         te=P("hmode", (0.043, 0.15, 1.1, 1.3, 1.3, 0.1, 0.98)),
         ti=P("hmode", (0.043, 0.37, 1.5, 1.1, 3.9, 0.1, 1.0)),
         zeff=P("poly", (1.69, 1.74, 1.0, 1.0)),            # linear (3 s.f.: at 2 s.f. it is flat)
         omega=P("hmode", (0.78, 2.3, 8.1, 1.0, 1.4, 0.41, 1.0)),
         nfast=P("hmode", (0.0, 0.0058, 0.035, 1.0, 2.1, 0.22, 0.8)),
         pfast=P("hmode", (0.0, 0.12, 0.93, 1.0, 1.9, 0.18, 0.8)),
         jbeam=P("hmode", (0.0, 0.00074, 0.0036, 1.0, 1.5, 0.22, 0.8)),
         johm=P("hmode", (0.039, 0.12, 0.87, 2.2, 1.6, 0.083, 1.0))),
    dict(name="dd1", t=1.1, ip=1.0e6, pedestal=False,     # L-H transition
         sep=dict(te=0.13, ti=0.38),
         ne=P("hmode", (0.71, 0.95, 2.1, 1.0, 1.6, 0.1, 1.0)),
         te=P("hmode", (0.13, 0.42, 3.1, 1.0, 1.6, 0.1, 1.0)),
         ti=P("hmode", (0.38, 0.61, 5.7, 1.0, 2.9, 0.1, 1.0)),
         zeff=P("hmode", (1.5, 1.8, 1.5, 3.4, 12.0, 0.11, 0.99)),
         omega=P("hmode", (1.2, 2.5, 19.0, 1.0, 2.6, 0.1, 1.0)),
         nfast=P("hmode", (0.0, 0.2, 0.92, 1.3, 2.1, 0.6, 0.87)),
         pfast=P("hmode", (0.0, 6.7, 46.0, 1.0, 1.9, 0.51, 0.8)),
         jbeam=P("hmode", (0.0, 0.033, 0.19, 1.0, 2.0, 0.46, 0.8)),
         johm=P("hmode", (0.037, 0.16, 1.2, 1.8, 2.3, 0.15, 1.0))),
    dict(name="dd2", t=1.2, ip=1.0e6, pedestal=True,      # pedestal forming, strong beam current (the IDA / g-file time)
         sep=dict(te=0.13, ti=0.26),
         ne=P("hmode", (0.67, 1.1, 2.4, 1.0, 1.5, 0.056, 1.0)),
         te=P("hmode", (0.13, 0.65, 3.9, 1.0, 1.6, 0.071, 0.99)),
         ti=P("hmode", (0.26, 0.94, 8.8, 1.0, 2.7, 0.078, 1.0)),
         zeff=P("hmode", (2.3, 2.1, 1.4, 2.7, 4.1, 0.025, 0.95)),
         omega=P("hmode", (0.26, 3.9, 26.0, 1.0, 2.2, 0.061, 0.99)),
         nfast=P("hmode", (0.0, 0.077, 1.0, 1.1, 1.5, 0.16, 1.0)),
         pfast=P("hmode", (0.0, 4.9, 58.0, 1.0, 2.0, 0.41, 1.0)),
         jbeam=P("hmode", (0.0, 0.051, 0.25, 1.0, 1.6, 0.48, 0.8)),
         johm=P("hmode", (0.0076, 0.16, 1.1, 2.1, 2.9, 0.11, 0.98))),
    dict(name="dd3", t=1.4, ip=1.0e6, pedestal=True,      # peak pedestal, hot ions, low density
         sep=dict(te=0.088, ti=0.25),
         ne=P("hmode", (0.5, 2.5, 2.5, 1.3, 12.0, 0.059, 0.97)),
         te=P("hmode", (0.088, 1.0, 4.3, 1.0, 1.4, 0.05, 0.97)),
         ti=P("hmode", (0.25, 1.5, 6.8, 1.0, 2.1, 0.057, 0.98)),
         zeff=P("hmode", (1.8, 3.1, 1.5, 1.6, 2.1, 0.038, 0.98)),
         omega=P("hmode", (0.0, 5.4, 36.0, 1.0, 1.9, 0.037, 0.97)),
         nfast=P("hmode", (0.0, 0.032, 0.59, 1.0, 1.5, 0.016, 0.99)),
         pfast=P("hmode", (0.0, 0.96, 24.0, 1.0, 1.6, 0.044, 1.0)),
         jbeam=P("hmode", (0.0, 0.0036, 0.1, 1.0, 1.4, 0.018, 0.99)),
         johm=P("hmode", (0.0, 0.15, 1.3, 2.0, 4.0, 0.067, 0.97))),
    dict(name="dd4", t=3.0, ip=1.0e6, pedestal=True,      # dense flat-top
         sep=dict(te=0.078, ti=0.17),
         ne=P("hmode", (1.0, 3.6, 7.1, 1.0, 1.4, 0.05, 0.98)),
         te=P("hmode", (0.078, 0.53, 2.7, 1.0, 1.3, 0.052, 0.97)),
         ti=P("hmode", (0.17, 0.82, 2.8, 1.0, 1.4, 0.064, 0.97)),
         zeff=P("hmode", (1.5, 3.8, 2.6, 1.1, 0.91, 0.017, 0.98)),
         omega=P("hmode", (0.0, 1.9, 10.0, 1.0, 1.1, 0.02, 0.97)),
         nfast=P("hmode", (0.0, 0.012, 0.14, 1.0, 1.1, 0.015, 1.0)),
         pfast=P("hmode", (0.0, 0.28, 4.9, 1.0, 1.2, 0.031, 1.0)),
         jbeam=P("hmode", (0.0, 0.0012, 0.012, 1.1, 0.71, 0.041, 0.87)),
         johm=P("hmode", (0.0051, 0.22, 1.3, 1.2, 2.6, 0.061, 0.96))),
)
T_IDA = 1.2             # s, the dd slice the IDA file, p-file and both H-mode g-files belong to
# g_synthetic.geqdsk's pressure: "thermal" = e (n_e T_e + n_i T_i), n_i =
# n_e (Z_IMP - Z_eff)/(Z_IMP - 1) (bouquet.physics.main_ion_density_from_zeff,
# the IDA reader's n_i; no impurity term, no fast ions) -- the pressure the
# g-file + IDA route rebuilds from the IDA file.  g_synthetic_pfast.geqdsk keeps
# the total (thermal + beam), as the dd slices.
PRESSURE_THERMAL = "thermal"
N_PFILE = 201           # p-file psinorm nodes (uniform on [0, 1])
# g_synthetic.geqdsk's EFIT-like current: ohmic + Redl bootstrap, no beam channel.
# "fit": the T_IDA slice's own ohmic shape (amplitude re-solved for Ip);
# "sigma_neo": a steady-state neoclassical ohmic current (uniform loop voltage).
G_OHMIC = "fit"

# The L-mode case (thermal pressure only, no beam)
IP_LMODE = 1.3e6        # A
T_LMODE = 1.0           # s, the IDA file's time
LMODE = dict(name="lmode", pedestal=False,
             sep=dict(te=0.088, ti=0.17),
             ne=P("hmode", (1.5, 2.1, 4.4, 1.0, 0.58, 0.1, 1.0)),
             te=P("hmode", (0.088, 0.45, 3.0, 1.2, 1.5, 0.15, 0.98)),
             ti=P("hmode", (0.17, 0.55, 4.0, 1.0, 1.7, 0.16, 1.0)),
             zeff=P("hmode", (1.4, 1.5, 1.6, 3.6, 3.2, 0.1, 1.0)),
             omega=P("hmode", (0.0, 2.1, 12.0, 1.0, 1.2, 0.1, 0.98)))
# its <j_phi> SHAPE [MA/m^2] (amplitude set by IP_LMODE): monotone, separatrix
# value 0 (the real one is negative)
JPHI_LMODE = P("hmode", (0.0, 0.11, 2.4, 1.1, 1.7, 0.011, 0.98))

SOL_DECAY = 0.05        # IDA nodes past psi_N = 1: value(1) exp(-(psi_N - 1)/SOL_DECAY); Z_eff held
# Sigma envelopes: the radial templates of the fixture's phase
# (apply_sigma_templates.py: IDA_REGIME, DD_REGIME; FIXTURE_SPEC.md sections 1-2).
# omega has no template: its (unread) envelope stays this constant fraction.
OMEGA_SIGMA = 0.10

# Grids and solver settings
N_SOLVE = 513           # psi_N nodes of the FF' / j_BS fixed point
N_EQ = 257              # equilibrium.time_slice profiles_1d (uniform psi_N)
N_CP = 257              # core_profiles, UNIFORM in rho_tor_norm (spec 1)
N_GEQDSK = 257          # g-file R, Z grid
N_IDA_CORE, N_IDA_EDGE = 90, 60   # IDA nodes: [0, 0.9) step 0.01, [0.9, 1.2]
PSI_PAD = 1e-3          # flux-surface geometry clip at the axis / LCFS
GFILE_PAD = 1e-4        # save_eqdsk lcfs_pad (as examples/D3D-like)
ISO_W, SADDLE_W = 200.0, 200.0
N_PASS = 10             # Newton-corrected jphi-linterp passes per equilibrium (at most)
MAXITS = 200            # TokaMaker NL iteration cap (nl_tol stays the default 1e-6)
URF_LADDER = (0.2, 0.1, 0.05)   # under-relaxation tried in turn on a stalled solve
NEWTON_DAMP = 1.0       # step of the Newton correction of the jphi-linterp request
STEP_HALVINGS = 3       # halvings of that step tried on a pass whose solve stalls
VSC_REG = 1e-2          # regularisation weight of the vertical-control coil pair (others: 1)
XPOINT_ISOFLUX = False  # the X-point is a saddle constraint; True also makes it an isoflux point
# sanity gate (refuse a scenario rather than solve it)
GATE_TSEP = 2.0         # T_e(1), T_i(1) <= GATE_TSEP x the slice's reference separatrix value
GATE_JSEP = 0.15        # j_total(1) <= GATE_JSEP x max j_total
GATE_JBS_PED = 0.85     # pedestal slices: the j_BS peak at psi_N >= GATE_JBS_PED
GATE_CORE_X = 0.3       # bounded smooth core of j_total on psi_N < GATE_CORE_X:
GATE_CORE_RISE = 0.15   # max j_total <= (1 + GATE_CORE_RISE) x j_total(0) (shallow off-axis maximum)
GATE_CURV_H = 0.01      # d2j/dpsi_N^2 from second differences on this uniform psi_N grid ...
GATE_CURV_EPS = 0.5     # ... |d2j/dpsi_N^2| < GATE_CURV_EPS x max j_total counts as zero (no sign) ...
GATE_CURV_CHANGES = 1   # ... and its sign changes at most this many times (no humps or dives)

EC = 1.602176634e-19
MU0 = 4.0e-7 * np.pi
_XS = np.linspace(0.0, 1.0, 4001)


# =============================================================================
#  analytic profiles
# =============================================================================
def _hmode(p):
    from bouquet.utils import Hmode_profiles
    edge, ped, core, expin, expout, widthp, xphalf = p
    return Hmode_profiles(edge=edge, ped=ped, core=core, rgrid=_XS.size, expin=expin,
                          expout=expout, widthp=widthp, xphalf=xphalf)


def prof(x, spec, sol="decay"):
    """Profile ``spec`` at psi_N ``x``.  Past psi_N = 1: ``"decay"`` (an
    exponential of length SOL_DECAY) or ``"hold"`` (the separatrix value)."""
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, 0.0, 1.0)
    if spec["fam"] == "hmode":
        v = np.interp(xc, _XS, _hmode(spec["p"]))
        v1 = _hmode(spec["p"])[-1]
    elif spec["fam"] == "poly":
        edge, core, a, b = spec["p"]
        v = edge + (core - edge) * (1.0 - xc ** a) ** b
        v1 = edge
    else:
        raise ValueError(spec["fam"])
    if spec["dip"] is not None:
        h, w = spec["dip"]
        v = v * (1.0 - h * np.exp(-xc / w))
    if sol == "hold":
        return v
    return np.where(x > 1.0, v1 * np.exp(-(x - 1.0) / SOL_DECAY), v)


def equilibrium_pressure(x, p):
    """``(p_eq, dp)``: the pressure the equilibria carry on the grid ``x`` (psi_N,
    ending at 1).  dp = dp/dpsi_N is the kinetic pressure's gradient at every node
    except the last, where it is clamped to 0 (on the linear-interpolated p' profile
    the solver uses, a one-cell drop inside the pedestal width: a pressure gradient
    AT the separatrix stalls the free-boundary solves); p_eq is its exact integral
    (trapezoid of the piecewise-linear dp) from the kinetic separatrix value p(1).
    The kinetic profiles are untouched.  A modelling choice of these fixtures."""
    x = np.asarray(x, dtype=float)
    dp = np.gradient(np.asarray(p, dtype=float), x)
    dp[-1] = 0.0
    seg = 0.5 * (dp[1:] + dp[:-1]) * np.diff(x)
    p_eq = p[-1] - np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return p_eq, dp


def kinetics(x, sc):
    """SI kinetic profiles of scenario ``sc`` at psi_N ``x`` (n [m^-3], T [eV],
    p [Pa], currents [A/m^2]); n_D is the thermal deuterium density."""
    return _kinetics(x, sc)


def _kinetics(x, sc):
    zero = np.zeros(np.shape(x))
    ne = 1e19 * prof(x, sc["ne"])
    zeff = prof(x, sc["zeff"], sol="hold")
    nc = ne * (zeff - 1.0) / (Z_IMP ** 2 - Z_IMP)
    nf = 1e19 * prof(x, sc["nfast"]) if sc.get("nfast") else zero
    nd = ne - Z_IMP * nc - nf
    te, ti = 1e3 * prof(x, sc["te"]), 1e3 * prof(x, sc["ti"])
    pf = 1e3 * prof(x, sc["pfast"]) if sc.get("pfast") else zero
    p_th = EC * (ne * te + (nd + nc) * ti)
    # the thermal pressure the g-file + IDA route reads: n_i from Z_eff (= n_D +
    # n_fast here: Z_eff counts the beam deuterons), no impurity term
    from bouquet.physics import main_ion_density_from_zeff
    ni_r = main_ion_density_from_zeff(ne, zeff, Z_IMP)
    p_th_reader = EC * (ne * te + ni_r * ti)
    if sc.get("pressure") == PRESSURE_THERMAL:
        p = p_th_reader
    elif sc.get("pressure") is None:
        p = p_th + pf
    else:
        raise ValueError(f"unknown pressure model {sc.get('pressure')!r}")
    return dict(ne=ne, te=te, ti=ti, nc=nc, nf=nf, nd=nd, pf=pf, p_th=p_th, p=p,
                ni_reader=ni_r, p_th_reader=p_th_reader,
                omega=1e4 * prof(x, sc["omega"]), zeff=zeff,
                j_beam=1e6 * prof(x, sc["jbeam"]) if sc.get("jbeam") else zero,
                j_ohm_shape=1e6 * prof(x, sc["johm"]) if sc.get("johm") else None)


def sigma_neo(te, ne, zeff, f_t, nu_e_star):
    """Sauter, Angioni & Lin-Liu (1999) neoclassical parallel conductivity
    [1/(Ohm m)] (Eqs. 13a, 13b, 18a, 18d; T_e in eV, n_e in m^-3)."""
    ln_lam = 31.3 - np.log(np.sqrt(ne) / te)
    n_z = 0.58 + 0.74 / (0.76 + zeff)
    s_sp = 1.9012e4 * te ** 1.5 / (zeff * n_z * ln_lam)
    x = f_t / (1.0 + (0.55 - 0.1 * f_t) * np.sqrt(nu_e_star)
               + 0.45 * (1.0 - f_t) * nu_e_star / zeff ** 1.5)
    f33 = 1.0 - (1.0 + 0.36 / zeff) * x + 0.59 / zeff * x ** 2 - 0.23 / zeff * x ** 3
    return s_sp * f33


# =============================================================================
#  TokaMaker
# =============================================================================
def make_solver(mesh=None):
    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    mygs = TokaMaker(OFT_env(nthreads=1))
    mp, ml, mr, cd, cnd = load_gs_mesh(mesh or MESH)
    mygs.setup_mesh(mp, ml, mr)
    mygs.setup_regions(cond_dict=cnd, coil_dict=cd)
    mygs.setup(order=3, F0=abs(B0) * R0)
    mygs.settings.maxits = MAXITS
    mygs.settings.pm = False
    mygs.update_settings()
    mygs.set_coil_vsc({"F9A": 1.0, "F9B": -1.0})
    reg = [mygs.coil_reg_term({n: 1.0}, target=0.0, weight=1.0) for n in mygs.coil_sets]
    reg.append(mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=VSC_REG))
    mygs.set_coil_reg(reg_terms=reg)
    return mygs


def set_shape(mygs, lcfs):
    """Isoflux on the boundary table, a saddle constraint at the X-point."""
    pts = np.array([(R0 + dr, z) for dr, z in lcfs["isoflux"]], dtype=float)
    xp = np.array([[R0 + lcfs["xpoint"][0], lcfs["xpoint"][1]]], dtype=float)
    iso = np.vstack([pts, xp]) if XPOINT_ISOFLUX else pts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mygs.set_isoflux(iso, weights=np.full(len(iso), ISO_W))
    mygs.set_saddle_constraints(xp, np.array([SADDLE_W]))
    r, z = pts[:, 0], pts[:, 1]
    a = 0.5 * (r.max() - r.min())
    mygs._bq_shape = dict(rc=0.5 * (r.max() + r.min()), zc=0.5 * (z.max() + z.min()), a=a,
                          kappa=0.5 * (z.max() - z.min()) / a)
    return dict(rc=0.5 * (r.max() + r.min()), zc=0.5 * (z.max() + z.min()), a=a,
                kappa=0.5 * (z.max() - z.min()) / a)


def fsa(mygs, x):
    """Flux-surface quantities of the live equilibrium at psi_N ``x`` (the
    geometry clipped to [PSI_PAD, 1 - PSI_PAD]), in the positive-current
    frame (``pp`` signed so the solve's own <j_phi> is positive)."""
    from bouquet.physics import q_ravg
    xc = np.ascontiguousarray(np.clip(x, PSI_PAD, 1.0 - PSI_PAD), dtype=float)
    _, F, Fp, P_, Pp = mygs.get_profiles(psi=xc.copy())
    _, q, rv, *_ = mygs.get_q(psi=xc.copy())
    _fc, _geo, bf = mygs.sauter_fc(psi=xc.copy())[-3:]
    B2 = np.asarray(bf["<|B|^2>"] if isinstance(bf, dict) else np.asarray(bf)[1], float)
    R = np.asarray(q_ravg(rv, "<R>"), float)
    iR = np.asarray(q_ravg(rv, "<1/R>"), float)
    if not (isinstance(rv, dict) and rv.get("<1/R^2>") is not None):
        raise RuntimeError("this OpenFUSIONToolkit's get_q lacks <1/R^2>")
    iR2 = np.asarray(rv["<1/R^2>"], float)
    F, Fp, Pp = (np.asarray(a, float) for a in (F, Fp, Pp))
    jeq = R * Pp + iR * F * Fp / MU0
    s = 1.0 if float(np.sum(jeq)) >= 0.0 else -1.0
    return dict(x=np.asarray(x, float), F=np.abs(F), FFp=F * Fp, Pp_raw=Pp, s=s,
                pp=s * Pp, R=R, iR=iR, iR2=iR2, B2=B2, q=np.abs(np.asarray(q, float)),
                P=np.asarray(P_, float), jeq=s * jeq)


def compose(mygs, x, K, sc, ip):
    """The current of scenario ``sc`` on the LIVE geometry at psi_N ``x``:
    ``(J, parts, G)``, J in TokaMaker <j_phi> with the amplitude A of its
    ohmic (or prescribed) part solved so that the FSA current is ``ip``.

    model "fuse": J = kappa|B0|(A j_ohm_shape + j_BS + j_beam) + p'G
    model "efit": J = kappa|B0|(A j_ohm_shape + j_BS) + p'G  (G_OHMIC "fit"; or
                  A sigma_neo with G_OHMIC "sigma_neo"), no beam
    model "jphi": J = A JPHI shape (a prescribed <j_phi>)."""
    from scipy.integrate import trapezoid
    from bouquet.physics import evaluate_jBS
    from bouquet.utils import Ip_fsa_weights, fsa_current_geometry
    G = fsa(mygs, x)
    kap = G["F"] * G["iR"] / G["B2"]
    pG = G["pp"] * (G["R"] - G["F"] ** 2 * G["iR"] / G["B2"])
    w, c = Ip_fsa_weights(fsa_current_geometry(mygs, x, psi_pad=PSI_PAD),
                          "jphi-linterp", pprime_sign=G["s"])
    zero = np.zeros_like(x)
    if sc["model"] == "jphi":
        shape = 1e6 * prof(x, sc["jphi"])
        A = (ip - c) / trapezoid(w * shape, x)
        return A * shape, dict(A=A, kappa=kap, pG=pG, j_bs_tor=zero, j_bs_par=zero,
                               j_beam_tor=zero, j_ohm_tor=A * shape, j_ohm_par=zero,
                               j_ohm_shape=shape, w_par=w), G
    jbs_tor, diag = evaluate_jBS(mygs, x, K["ne"], K["te"], K["nd"], K["ti"],
                                 K["zeff"], psi_pad=PSI_PAD)
    kap0 = G["F"] * G["iR"] / G["B2"]
    # <J.B>/|B0| of the bootstrap current the solve uses: evaluate_jBS smooths
    # its toroidal current near the axis (smooth_axis), not diag["jdotB_bs"]
    jbs_par = np.asarray(jbs_tor, float) / (kap0 * abs(B0))
    if sc["model"] == "efit" and G_OHMIC == "sigma_neo":
        sig = sigma_neo(K["te"], K["ne"], K["zeff"], np.asarray(diag["f_T"], float),
                        np.asarray(diag["nu_e_star"], float))
        shape = 1e6 * sig / np.max(sig)                       # uniform loop voltage
        jbeam = zero
    elif sc["model"] == "efit":
        shape, jbeam = K["j_ohm_shape"], zero                 # the slice's ohmic shape, no beam
    else:
        shape, jbeam = K["j_ohm_shape"], K["j_beam"]
    jb = kap * abs(B0) * jbeam
    jo1 = kap * abs(B0) * shape
    A = (ip - c - trapezoid(w * (jbs_tor + jb + pG), x)) / trapezoid(w * jo1, x)
    J = A * jo1 + jbs_tor + jb + pG
    return J, dict(A=A, kappa=kap, pG=pG, j_bs_tor=jbs_tor, j_bs_par=jbs_par,
                   j_beam_tor=jb, j_ohm_tor=A * jo1, j_ohm_par=A * shape,
                   j_ohm_shape=shape, w_par=w * kap * abs(B0)), G


def sanity_gate(sc, x, K, parts):
    """``(failures, numbers)`` of the sanity gate for the composition ``parts``
    of scenario ``sc`` on psi_N ``x`` (see the module docstring).  j_total is
    <J.B>/B0 (ohmic + bootstrap + beam); for the prescribed-<j_phi> model, that
    <j_phi>."""
    bad, num = [], {}
    for k in ("te", "ti"):
        ref = (sc.get("sep") or {}).get(k)
        v = float(K[k][-1]) / 1e3
        num[k + "_sep_keV"] = v
        if ref is not None and v > GATE_TSEP * ref:
            bad.append(f"{k}(1) = {v:.3g} keV > {GATE_TSEP:g} x reference {ref:.3g} keV")
    if sc["model"] == "jphi":
        jt = np.asarray(parts["j_ohm_tor"], float)
    else:
        jt = np.asarray(parts["j_ohm_par"] + parts["j_bs_par"] + K["j_beam"], float)
    pk = float(np.max(jt))
    num["jtot_sep_over_peak"] = float(jt[-1] / pk)
    if jt[-1] > GATE_JSEP * pk:
        bad.append(f"j_total(1) = {jt[-1]:.3e} A/m^2 = {jt[-1] / pk:.3f} x peak > {GATE_JSEP:g}")
    if sc["model"] != "jphi":
        xb = float(x[int(np.argmax(parts["j_bs_par"]))])
        num["jbs_peak_psiN"] = xb
        if sc.get("pedestal") and xb < GATE_JBS_PED:
            bad.append(f"j_BS peak at psi_N {xb:.3f} < {GATE_JBS_PED:g} on a pedestal slice")
    xg = np.arange(0.0, GATE_CORE_X + 0.5 * GATE_CURV_H, GATE_CURV_H)
    jg = np.interp(xg, x, jt)
    rise = float(jg.max() / jg[0] - 1.0) if jg[0] > 0 else np.inf
    num["core_rise_over_axis"] = rise
    num["core_max_at_psiN"] = float(xg[int(np.argmax(jg))])
    if rise > GATE_CORE_RISE:
        bad.append(f"j_total on psi_N < {GATE_CORE_X:g} peaks at psi_N {num['core_max_at_psiN']:.3f}, "
                   f"{rise:.3f} above its axis value (> {GATE_CORE_RISE:g})")
    d2 = (jg[2:] - 2.0 * jg[1:-1] + jg[:-2]) / GATE_CURV_H ** 2 / pk
    sg = np.sign(d2[np.abs(d2) > GATE_CURV_EPS])
    nchg = int(np.sum(sg[1:] != sg[:-1]))
    num["core_curv_sign_changes"] = nchg
    if nchg > GATE_CURV_CHANGES:
        bad.append(f"d2j/dpsi_N^2 changes sign {nchg} times on psi_N < {GATE_CORE_X:g} "
                   f"(> {GATE_CURV_CHANGES}; dead band {GATE_CURV_EPS:g} x peak)")
    return bad, num


def _gate_line(label, when, bad, num):
    return (f"[{label}] sanity gate ({when}): " + ("PASS" if not bad else "FAIL: " + "; ".join(bad))
            + "  [" + ", ".join(f"{k} {v:.3g}" for k, v in num.items()) + "]")


def _solve_damped(mygs, setup, what, retries):
    """``setup(); mygs.solve()``; a solve that stalls in a limit cycle just
    above the (unchanged) nl_tol is retried from the pre-solve state with a
    smaller under-relaxation factor (solver-side damping only)."""
    urf0, its0 = mygs.settings.urf, mygs.settings.maxits
    try:
        for urf in URF_LADDER:
            snap = mygs.copy_eq()
            mygs.settings.urf = urf
            mygs.settings.maxits = MAXITS
            mygs.update_settings()
            setup()
            try:
                mygs.solve()
                return urf
            except ValueError as exc:
                retries.append((what, urf))
                print(f"[{what}] solve with urf {urf:g} failed ({str(exc).strip()}); "
                      "restoring the pre-solve state", flush=True)
                mygs.replace_eq(source_eq=snap)
        raise RuntimeError(f"[{what}] no converged solve on urf ladder {URF_LADDER}")
    finally:
        mygs.settings.urf, mygs.settings.maxits = urf0, its0
        mygs.update_settings()


def solve_scenario(mygs, sc, ip, label, init=None, n_pass=None, gate_raise=True):
    """Self-consistent solve of scenario ``sc`` at plasma current ``ip``;
    ``init`` (a shape record) starts from a fresh psi.  Returns the record of
    the converged state (the solver is left in it)."""
    x = np.linspace(0.0, 1.0, N_SOLVE)
    K = kinetics(x, sc)
    p_eq, dp = equilibrium_pressure(x, K["p"])     # p'(1) = 0, p(1) kinetic
    pp_prof = {"type": "linterp", "x": x, "y": dp / np.min(dp)}
    pax = float(p_eq[0] - p_eq[-1])                # p(LCFS) is the offset
    retries = []

    def fresh_start(shape):
        from OpenFUSIONToolkit.TokaMaker.util import create_power_flux_fun
        mygs.init_psi(shape["rc"], shape["zc"], 0.9 * shape["a"], shape["kappa"], 0.3)
        _solve_damped(mygs, lambda: (mygs.set_targets(Ip=ip, pax=pax), mygs.set_profiles(
            ffp_prof=create_power_flux_fun(N_SOLVE, 1.5, 1.5), pp_prof=pp_prof)),
            f"{label} start", retries)
        mygs._bq_offset = 0.0

    if init is not None:
        fresh_start(init)
    # Newton-corrected jphi-linterp passes: the request is moved by (composed -
    # achieved) each pass, so the converged state's own <j_phi> is the
    # composition evaluated on that state's geometry (and j_BS).
    hist = []
    J, parts, G = compose(mygs, x, K, sc, ip)
    bad, num = sanity_gate(sc, x, K, parts)          # on the live state, before this scenario's passes
    print(_gate_line(label, "pre-solve", bad, num), flush=True)
    if bad and gate_raise:
        raise RuntimeError(f"[{label}] sanity gate REFUSES this scenario: " + "; ".join(bad)
                           + "  [" + ", ".join(f"{k} {v:.4g}" for k, v in num.items()) + "]")
    request = J + getattr(mygs, "_bq_offset", 0.0)  # carry the last state's offset
    # the request of the last converged state: before pass 0, the live state's
    # own <j_phi> (a continuation path from it to the first request)
    req_prev = fsa(mygs, x)["jeq"]
    restarted = init is not None
    stopped = None
    for it in range(N_PASS if n_pass is None else n_pass):
        # A free-boundary solve can stall in a limit cycle (the coil fit and the
        # X-point flip between iterates just above nl_tol).  The urf ladder is
        # tried first; if every rung stalls, the Newton step of this pass is
        # halved (a line search on the request, at most STEP_HALVINGS times).
        # If that fails too, the passes stop at the last converged state, whose
        # r_j is reported.  nl_tol is never changed.
        for h in range(STEP_HALVINGS + 1):
            try:
                _solve_damped(mygs, lambda: (mygs.set_targets(Ip=ip, pax=pax), mygs.set_profiles(
                    ffp_prof={"type": "jphi-linterp", "x": x, "y": request.copy()},
                    pp_prof=pp_prof)), f"{label} pass {it}" + (f" step/{2 ** h}" if h else ""),
                    retries)
                break
            except RuntimeError:
                if h < STEP_HALVINGS:
                    request = req_prev + 0.5 * (request - req_prev)
                    continue
                if it == 0 and not restarted:
                    # the jump from the previous scenario stalled on every
                    # step: restart this scenario from a fresh psi (once)
                    print(f"[{label}] pass 0 from the previous state stalled; "
                          "restarting from a fresh psi", flush=True)
                    restarted = True
                    fresh_start(mygs._bq_shape)
                    req_prev = fsa(mygs, x)["jeq"]
                    J, parts, G = compose(mygs, x, K, sc, ip)
                    request = J.copy()
                    stopped = "retry"
                    break
                if it == 0:
                    raise RuntimeError(f"[{label}] pass 0: no converged solve on any step")
                stopped = it
                break
        if stopped == "retry":
            stopped = None
            for h in range(STEP_HALVINGS + 1):
                try:
                    _solve_damped(mygs, lambda: (mygs.set_targets(Ip=ip, pax=pax), mygs.set_profiles(
                        ffp_prof={"type": "jphi-linterp", "x": x, "y": request.copy()},
                        pp_prof=pp_prof)), f"{label} pass 0 (fresh)" + (f" step/{2 ** h}" if h else ""),
                        retries)
                    break
                except RuntimeError:
                    if h == STEP_HALVINGS:
                        raise RuntimeError(f"[{label}] pass 0: no converged solve on any step")
                    request = req_prev + 0.5 * (request - req_prev)
        if stopped is not None:
            print(f"[{label}] pass {it}: no converged solve on any step; the passes stop at "
                  f"the converged state of pass {it - 1}", flush=True)
            break
        req_prev = request.copy()
        J, parts, G = compose(mygs, x, K, sc, ip)
        r = G["jeq"] - J
        hist.append(float(np.max(np.abs(r)) / np.max(np.abs(G["jeq"]))))
        print(f"[{label}] pass {it}: r_j max {hist[-1]:.2e} at psi_N {x[np.argmax(np.abs(r))]:.3f}, "
              f"amplitude {parts['A']:.4f}", flush=True)
        mygs._bq_offset = request - G["jeq"]  # the request-minus-achieved offset
        if it >= 2 and hist[-1] < 2e-4 and hist[-1] > 0.8 * hist[-2]:
            break                           # at the solver's floor
        request = request + NEWTON_DAMP * (J - G["jeq"])
    r = G["jeq"] - J                                # the converged state
    gate = sanity_gate(sc, x, K, parts)
    print(_gate_line(label, "converged state", *gate), flush=True)
    r_j = float(np.max(np.abs(r)) / np.max(np.abs(G["jeq"])))
    r_j_rms = float(np.sqrt(np.mean(r ** 2)) / np.max(np.abs(G["jeq"])))
    st = mygs.get_stats(li_normalization="iter")
    st1 = mygs.get_stats(li_normalization="std")
    rec = dict(label=label, sc=sc, ip_target=float(ip), x=x, K=K, J=J, G=G, r_j=r_j, stopped=stopped,
               r_j_rms=r_j_rms, hist=hist, retries=retries, Ip=float(st["Ip"]),
               li_3=float(st["l_i"]), li_1=float(st1["l_i"]), q0=float(st["q_0"]),
               q95=float(st["q_95"]), pax=pax,
               psi_bounds=np.asarray(mygs.psi_bounds, float).copy(), gate=gate, **parts)
    print(f"[{label}] Ip {rec['Ip']:.6e}  l_i(3) {rec['li_3']:.4f}  q0 {rec['q0']:.3f}  "
          f"q95 {rec['q95']:.3f}  amplitude {parts['A']:.4f}  "
          f"r_j max {r_j:.2e} rms {r_j_rms:.2e}  stalled solves {len(retries)}  (passes: "
          + ", ".join(f"{a:.1e}" for a in hist) + ")", flush=True)
    return rec


def save_gfile(mygs, path, rec):
    """``save_eqdsk`` (COCOS 7, Bt > 0), then COCOS 1 with Bt = B0 < 0 (the
    Bt-only flip: negate BCENTR and FPOL, keep CURRENT, PSIRZ, PPRIME,
    FFPRIM and QPSI)."""
    from bouquet.io.geqdsk import read_geqdsk
    tmp = path + ".cocos7"
    mygs.save_eqdsk(tmp, nr=N_GEQDSK, nz=N_GEQDSK, lcfs_pad=GFILE_PAD, rcentr=R0,
                    run_info="synthetic", lcfs_pressure=float(rec["K"]["p"][-1]))
    g = read_geqdsk(tmp, cocos=7).cocosify(1, copy=True)
    os.remove(tmp)
    if B0 < 0:
        g._raw["BCENTR"] *= -1
        g._raw["FPOL"] = g._raw["FPOL"] * -1
        if hasattr(g, "_cache"):
            g._cache.clear()
    g.save(path)
    return read_geqdsk(path, cocos=1)


# =============================================================================
#  the dd
# =============================================================================
def _rho_map(x, q):
    """rho_tor_norm(psi_N) = sqrt(Phi/Phi_edge), Phi = int q dpsi, by the
    antiderivative of a cubic spline of q (exact for the tabulated q's spline)."""
    from scipy.interpolate import CubicSpline
    phi = CubicSpline(x, q).antiderivative()(x)
    return np.sqrt(np.clip(phi / phi[-1], 0.0, 1.0)), phi


def equilibrium_slice(mygs, rec, t, keep_gfile):
    """IMAS ``equilibrium.time_slice`` (COCOS 11) of the live state ``rec``."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import CubicSpline
    from bouquet.physics import jphi_tokamaker_to_jtor_imas
    from bouquet.utils import fsa_current_geometry
    x = np.linspace(0.0, 1.0, N_EQ)
    G = fsa(mygs, x)
    pb = rec["psi_bounds"]
    dpsi11 = 2.0 * np.pi * abs(float(pb[1]) - float(pb[0]))     # Wb, > 0
    psi_ax11 = -2.0 * np.pi * float(pb[1])                        # axis (psi_bounds[1])
    psi11 = psi_ax11 + dpsi11 * x
    pressure, dp = equilibrium_pressure(x, kinetics(x, rec["sc"])["p"])
    dpdpsi = dp / dpsi11                    # exact pair: pressure = p(1) - trapz(dpdpsi, psi)
    rho, _ = _rho_map(x, G["q"])
    geom = {"avg_R": G["R"], "avg_inv_R": G["iR"], "avg_inv_R2": G["iR2"],
            "pprime": -2.0 * np.pi * dpdpsi}
    j_tor = jphi_tokamaker_to_jtor_imas(G["jeq"], geom)
    gi = fsa_current_geometry(mygs, x, psi_pad=PSI_PAD)
    area = cumulative_trapezoid(gi["dA_dpsiN"], x, initial=0.0)
    sgn_q = float(np.sign(rec["ip_target"] * B0))
    gf = save_gfile(mygs, keep_gfile, rec)
    bR, bZ = np.asarray(gf.boundary_R, float), np.asarray(gf.boundary_Z, float)
    return {
        "time": float(t),
        "global_quantities": {"ip": rec["ip_target"], "li_3": rec["li_3"], "li_1": rec["li_1"],
                              "magnetic_axis": {"r": float(gf.R_mag), "z": float(gf.Z_mag)}},
        "boundary": {"outline": {"r": bR.tolist(), "z": bZ.tolist()}},
        "profiles_1d": {
            "psi": psi11.tolist(), "q": (sgn_q * G["q"]).tolist(),
            "pressure": pressure.tolist(), "dpressure_dpsi": dpdpsi.tolist(),
            "j_tor": j_tor.tolist(), "rho_tor_norm": rho.tolist(),
            "f": (np.sign(B0) * G["F"]).tolist(), "gm1": G["iR2"].tolist(),
            "gm5": G["B2"].tolist(), "gm8": G["R"].tolist(), "gm9": G["iR"].tolist(),
            "area": area.tolist()},
        "_x": x,
    }


def core_profiles_slice(rec, ts, eq, k_pair):
    """core_profiles.profiles_1d of state ``rec`` (equilibrium slice ``ts``) on
    a rho-uniform grid; ``j_tor`` from ``j_total`` on slice ``k_pair``."""
    from scipy.interpolate import CubicSpline, PchipInterpolator
    from bouquet.io.imas import _fuse_current_geometry, _jtor_from_jpar
    p1 = ts["profiles_1d"]
    x_eq = np.asarray(ts["_x"], float)
    psi11 = np.asarray(p1["psi"], float)
    # psi_N at the rho-uniform nodes: invert Phi_N(psi_N) of this slice on a
    # dense grid (Phi_N is ~linear in psi_N at the axis, so the inverse is smooth)
    xd = np.linspace(0.0, 1.0, 20001)
    phid = CubicSpline(x_eq, np.abs(np.asarray(p1["q"], float))).antiderivative()(xd)
    phid /= phid[-1]
    rho = np.linspace(0.0, 1.0, N_CP)
    xc = np.interp(rho ** 2, phid, xd)
    xc[0], xc[-1] = 0.0, 1.0
    psi_cp = psi11[0] + (psi11[-1] - psi11[0]) * xc
    K = kinetics(xc, rec["sc"])
    at = lambda a: PchipInterpolator(rec["x"], a)(xc)              # noqa: E731
    j_ohm = rec["A"] * K["j_ohm_shape"]
    j_bs = at(rec["j_bs_par"])
    j_beam = K["j_beam"]
    j_total = j_ohm + j_bs + j_beam
    j_tor = _jtor_from_jpar(j_total, _fuse_current_geometry(eq, k_pair, rho))
    area_cp = np.interp(xc, x_eq, np.asarray(p1["area"], float))
    zero = np.zeros(N_CP)
    pf3 = K["pf"] / 3.0      # 'sum' convention: p_par + 2 p_perp = p_fast
    ion_d = {"label": "D", "element": [{"z_n": 1.0, "a": A_D}],
             "density_thermal": K["nd"].tolist(), "density_fast": K["nf"].tolist(),
             "temperature": K["ti"].tolist(),
             "pressure_fast_parallel": pf3.tolist(),
             "pressure_fast_perpendicular": pf3.tolist(),
             "rotation_frequency_tor": K["omega"].tolist()}
    ion_c = {"label": "C", "element": [{"z_n": Z_IMP, "a": A_C}],
             "density_thermal": K["nc"].tolist(), "density_fast": zero.tolist(),
             "temperature": K["ti"].tolist(),
             "pressure_fast_parallel": zero.tolist(),
             "pressure_fast_perpendicular": zero.tolist(),
             "rotation_frequency_tor": K["omega"].tolist()}
    grid = {"rho_tor_norm": rho.tolist(), "psi": psi_cp.tolist(),
            "psi_norm": xc.tolist(), "area": area_cp.tolist()}
    cp1 = {"time": float(ts["time"]), "grid": grid,
           "electrons": {"density_thermal": K["ne"].tolist(), "temperature": K["te"].tolist(),
                         "density_fast": zero.tolist(),
                         "pressure_fast_parallel": zero.tolist(),
                         "pressure_fast_perpendicular": zero.tolist()},
           "ion": [ion_d, ion_c], "zeff": K["zeff"].tolist(),
           "j_total": j_total.tolist(), "j_tor": j_tor.tolist(),
           "j_bootstrap": j_bs.tolist(), "j_ohmic": j_ohm.tolist(),
           "j_non_inductive": (j_bs + j_beam).tolist()}
    add_dd_errors(cp1, dd_regime(ts["time"]))      # the IMAS *_error_upper envelopes
    extra = dict(xc=xc, rho=rho, psi=psi_cp, K=K, j_ohm=j_ohm, j_bs=j_bs, j_beam=j_beam,
                 j_total=j_total, j_tor=j_tor)
    return cp1, extra


def core_sources(cps):
    """core_sources: one beam (2), ohmic (7), bootstrap (13) and a zero
    sawteeth (701) entry, a profile at every core_profiles time."""
    def src(index, name, key):
        return {"identifier": {"index": index, "name": name},
                "profiles_1d": [{"time": float(t),
                                 "grid": {"rho_tor_norm": e["rho"].tolist(), "psi": e["psi"].tolist()},
                                 "j_parallel": (np.zeros(N_CP) if key is None else e[key]).tolist()}
                                for t, e in cps]}
    return {"time": [float(t) for t, _ in cps],
            "source": [src(2, "beam", "j_beam"), src(7, "ohmic", "j_ohm"),
                       src(13, "bootstrap", "j_bs"), src(701, "sawteeth", None)]}


def write_ida(path, g, sc, t):
    """IDA "direct" layout (FIXTURE_SPEC.md section 2) of scenario ``sc`` on
    the g-file's q; the sigmas are the template of the file's phase
    (``IDA_REGIME``)."""
    import h5py
    x = np.round(np.concatenate([np.linspace(0.0, 0.9, N_IDA_CORE, endpoint=False),
                                 np.linspace(0.9, 1.2, N_IDA_EDGE)]), 6)
    K = kinetics(x, sc)
    qg = np.abs(np.asarray(g.qpsi, float))
    q = np.interp(np.clip(x, 0.0, 1.0), np.linspace(0.0, 1.0, qg.size), qg)
    data = {"n_e": K["ne"], "T_e": K["te"], "T_12C6": K["ti"], "Zeff": K["zeff"],
            "n_12C6": K["nc"], "omega_tor_12C6": K["omega"]}
    with h5py.File(path, "w") as f:
        f["time"] = np.array([1e3 * t])
        f["psi_n"] = x
        err = ida_errors(x, data, IDA_REGIME[os.path.basename(path)])
        err["omega_tor_12C6_err"] = OMEGA_SIGMA * np.abs(data["omega_tor_12C6"])
        for k, v in data.items():
            f[k] = v[None, :]
            f[k + "_err"] = err[k + "_err"][None, :]
        f["q"] = q[None, :]
    return dict(x=x, K=K, zeff=K["zeff"], q=q)


#: the p-file's ion-species footer (Osborne / OMFIT order: impurities, main ion, beam ion)
PFILE_SPECIES = dict(N=(6.0, 1.0, 1.0), Z=(Z_IMP, 1.0, 1.0), A=(A_C, A_D, A_D))


def write_pfile(path, sc):
    """Osborne p-file (the DIII-D kinetic-EFIT layout ``bouquet.io.pfile``
    reads) of scenario ``sc`` on a uniform psinorm grid of N_PFILE nodes:
    ne, te, ni (the THERMAL main ions, n_D), ti, nb (the beam ions, n_fast),
    pb (the beam pressure [kPa]: the dd's fast-ion fields p_par = p_perp =
    p_fast/3 isotropised by ``bouquet.physics.isotropize_fast_pressure`` with
    the dd's 'sum' rule), nz1 (carbon), omeg (toroidal rotation [kRad/s]),
    ptot (``PFile.compute_pressure``: ne Te + (ni + nz1) Ti + pb) and the
    ``N Z A`` footer.  Densities 1e20 m^-3, temperatures keV."""
    from bouquet.io.pfile import PFile
    from bouquet.physics import isotropize_fast_pressure
    x = np.linspace(0.0, 1.0, N_PFILE)
    K = kinetics(x, sc)
    pb = isotropize_fast_pressure(K["pf"] / 3.0, K["pf"] / 3.0, "sum")
    pf = PFile.new()
    for key, v in (("ne", K["ne"] / 1e20), ("te", K["te"] / 1e3), ("ni", K["nd"] / 1e20),
                   ("ti", K["ti"] / 1e3), ("nb", K["nf"] / 1e20), ("pb", pb / 1e3),
                   ("nz1", K["nc"] / 1e20), ("omeg", K["omega"] / 1e3)):
        pf.set_profile(key, x, v)
    pf.compute_pressure()
    pf.set_ion_species(**PFILE_SPECIES)
    pf.save(path)
    return dict(x=x, K=K, pb=pb)


# =============================================================================
#  self-checks (FIXTURE_SPEC.md section 1)
# =============================================================================
def r_out(g, psi_N):
    """Outboard-midplane R [m] of the surfaces psi_N of a parsed g-file (the
    rule of tests/test_phi_imas_solver.py)."""
    from scipy.interpolate import RectBivariateSpline
    s = RectBivariateSpline(g.Z_grid, g.R_grid, g.psi_N_RZ)
    r = np.linspace(g.R_mag, g.R_grid[-1], 4000)
    pn = s(g.Z_mag, r)[0]
    k = int(np.argmax(pn >= 1.0)) + 1
    return np.interp(psi_N, np.maximum.accumulate(pn[:k]), r[:k])


def pressure_consistency(g_path, prof_path):
    """``(axis_rel, profile_max_rel, p_g_axis, p_prof_axis)``: the g-file's
    pressure vs the pressure its profile file gives as the g-file route reads
    it -- e (n_e T_e + n_i T_i) [+ the p-file's ``pb``] -- with n_i from Z_eff
    (IDA file, ``main_ion_density_from_zeff``) or the p-file's own ``ni``;
    the profile maximum over psi_N <= 0.95 of |p_g/p_prof - 1|."""
    import h5py
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.io.pfile import read_pfile
    from bouquet.physics import main_ion_density_from_zeff
    g = read_geqdsk(g_path)
    pg = np.abs(np.asarray(g.pres, float))
    xg = np.linspace(0.0, 1.0, pg.size)
    if prof_path.endswith(".cdf"):
        with h5py.File(prof_path, "r") as f:
            x = np.asarray(f["psi_n"][()], float)
            ne, te, ti, zf = (np.asarray(f[k][0], float) for k in ("n_e", "T_e", "T_12C6", "Zeff"))
        ni = main_ion_density_from_zeff(ne, np.clip(zf, 1.0, Z_IMP), Z_IMP)
        pp = EC * (ne * te + ni * ti)
    else:
        pf = read_pfile(prof_path)
        x = np.asarray(pf.psinorm_for("ne"), float)
        pp = (EC * 1e20 * 1e3 * (np.asarray(pf.ne) * np.asarray(pf.te) + np.asarray(pf.ni) * np.asarray(pf.ti))
              + 1e3 * np.asarray(pf.pb))
    m = (x >= 0.0) & (x <= 0.95)
    rel = np.interp(x[m], xg, pg) / pp[m] - 1.0
    return float(pg[0] / pp[0] - 1.0), float(np.max(np.abs(rel))), float(pg[0]), float(pp[0])


def self_checks(out_dir, recs, ida, gfile_dd_t, ida_l, gl_path, pfile=None):
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import CubicSpline
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.io.imas import (_paired_current_geometry, _psi_rho_drift,
                                 NI_FAST_GATE_PSI_N, _jtor_from_jpar,
                                 _fuse_current_geometry, _net_current)
    from bouquet.physics import main_ion_density_from_zeff
    dd = json.load(gzip.open(os.path.join(out_dir, "dd_synthetic.json.gz"), "rt"))
    gpath = os.path.join(out_dir, "g_synthetic.geqdsk")
    ipath = os.path.join(out_dir, "diiid_profs_synthetic.cdf")
    eq = dd["equilibrium"]
    hard = []
    print("\n==== self-checks (FIXTURE_SPEC.md section 1) ====")
    for r in recs:
        jo = r["j_ohm_par"] if r["sc"]["model"] != "jphi" else r["j_ohm_tor"]
        print(f"  solve {r['label']:6s}: Ip {r['Ip']:.6e} (|Ip/IP-1| {abs(r['Ip'] / r['ip_target'] - 1):.1e}), "
              f"l_i(3) {r['li_3']:.4f}, q0 {r['q0']:.3f}, q95 {r['q95']:.3f}, "
              f"r_j max {r['r_j']:.2e} rms {r['r_j_rms']:.2e}, passes {len(r['hist'])}, "
              f"stalled solves {len(r['retries'])}, passes stopped early: {r['stopped'] is not None}, "
              f"min j_ohmic {np.min(jo):.3e} A/m^2")
        hard.append((f"{r['label']} Ip", abs(r["Ip"] / r["ip_target"] - 1) < 1e-4))
        print("  " + _gate_line(r["label"], "converged state", *r["gate"]))
        hard.append((f"{r['label']} sanity gate", not r["gate"][0]))
    # -- per core_profiles slice: pairing, composition, signs, seed
    srcs = {s["identifier"]["index"]: s for s in dd["core_sources"]["source"]}
    t_eq = [float(t) for t in eq["time"]]
    for i, cp in enumerate(dd["core_profiles"]["profiles_1d"]):
        t = float(dd["core_profiles"]["time"][i])
        jt, jtor = np.asarray(cp["j_total"]), np.asarray(cp["j_tor"])
        geom, meta = _paired_current_geometry(eq, cp, t, jt, jtor)
        want = t_eq[max(i - 1, 0)]
        others = []
        for k, te in enumerate(t_eq):
            g_k = _fuse_current_geometry(eq, k, np.asarray(cp["grid"]["rho_tor_norm"]))
            ok = np.abs(jtor) > 1e-3 * np.max(np.abs(jtor))
            others.append(f"{te:.1f}: {np.median(np.abs(_jtor_from_jpar(jt, g_k)[ok] / jtor[ok] - 1)):.1e}")
        print(f"  cp t = {t:.1f}: j_tor = A6(j_total) paired with slice {meta['time']:.1f} s "
              f"(built on {want:.1f}), median rel mismatch {meta['jtor_mismatch']:.1e} (tol 1e-3); "
              f"on each slice: {'; '.join(others)}")
        hard.append((f"cp {t} pairing slice", abs(meta["time"] - want) < 1e-9))
        hard.append((f"cp {t} pairing mismatch", meta["jtor_mismatch"] <= 1e-3))
        sj = {k: np.asarray(s["profiles_1d"][i]["j_parallel"]) for k, s in srcs.items()}
        ts_ok = all(abs(s["profiles_1d"][i]["time"] - t) < 1e-12 for s in srcs.values())
        comp = np.asarray(cp["j_ohmic"]) + np.asarray(cp["j_bootstrap"]) + sj[2]
        e = float(np.max(np.abs(comp - jt)) / np.max(np.abs(jt)))
        e7 = float(np.max(np.abs(sj[7] - np.asarray(cp["j_ohmic"]))))
        e13 = float(np.max(np.abs(sj[13] - np.asarray(cp["j_bootstrap"]))))
        hard += [(f"cp {t} j_total composition", e < 1e-12), (f"cp {t} source times", ts_ok),
                 (f"cp {t} sources = cp", e7 == 0.0 and e13 == 0.0)]
        area = np.asarray(cp["grid"]["area"])
        signs = {n: _net_current(np.asarray(v, float), area) > 0 for n, v in
                 (("j_tor", jtor), ("j_total", jt), ("j_bootstrap", cp["j_bootstrap"]), ("beam", sj[2]))}
        jo = np.asarray(cp["j_ohmic"])
        print(f"           j_total = ohmic + bootstrap + beam max rel {e:.1e}; source times = cp {ts_ok}; "
              f"nets co-Ip {signs}; beam min {sj[2].min():.3e}; ohmic min {jo.min():.3e}, "
              f"ohmic[0] {jo[0]:.4e}, ohmic(rho=1) {jo[-1]:.3e} A/m^2")
        hard += [(f"cp {t} sign {n}", ok) for n, ok in signs.items()]
        hard.append((f"cp {t} seed >= 0, seed[0] > 0", bool(np.all(jo >= 0) and jo[0] > 0)))
        ni = np.asarray(cp["ion"][0]["density_thermal"])
        hard.append((f"cp {t} n_D > 0", bool(ni.min() > 0)))
    # -- per equilibrium slice
    for k, ts in enumerate(eq["time_slice"]):
        p1 = ts["profiles_1d"]
        psi, pr = np.asarray(p1["psi"]), np.asarray(p1["pressure"])
        dpw = np.asarray(p1["dpressure_dpsi"])
        integ = pr[-1] - np.concatenate([np.cumsum((0.5 * (dpw[1:] + dpw[:-1]) * np.diff(psi))[::-1])[::-1], [0.0]])
        e_dp = float(np.max(np.abs(integ - pr)) / np.max(np.abs(pr)))
        e_spl = float(np.max(np.abs(CubicSpline(psi, pr).derivative()(psi)[:-1] - dpw[:-1])) / np.max(np.abs(dpw)))
        q = np.abs(np.asarray(p1["q"]))
        phi = cumulative_trapezoid(q, psi, initial=0.0)
        rho = np.asarray(p1["rho_tor_norm"])
        e_rho = float(np.max(np.abs(np.sqrt(phi / phi[-1]) - rho)))
        mono = bool(np.all(np.diff(rho) > 0))
        gm1, gm9, gm5, gm8, f = (np.asarray(p1[n]) for n in ("gm1", "gm9", "gm5", "gm8", "f"))
        jen = float(np.min(gm1 / gm9 ** 2 - 1))
        b2 = float(np.min(gm5 / (f ** 2 * gm1) - 1))
        r8 = float(np.min(gm8 * gm9 - 1))
        sgn = bool(np.sign(np.sum(p1["j_tor"])) == np.sign(ts["global_quantities"]["ip"]))
        print(f"  eq t = {ts['time']:.1f}: ip {ts['global_quantities']['ip']:.3e}, li_3 "
              f"{ts['global_quantities']['li_3']:.4f}; pressure = p(1) - trapz(dpressure_dpsi) {e_dp:.1e} "
              f"(dpressure_dpsi(LCFS) {dpw[-1]:.1e}; vs a spline derivative of pressure, interior "
              f"{e_spl:.1e}); rho_tor_norm "
              f"vs sqrt(trapz Phi) {e_rho:.1e}, increasing {mono}; min(gm1/gm9^2-1) {jen:.2e}, "
              f"min(gm5/(f^2 gm1)-1) {b2:.2e}, min(gm8 gm9-1) {r8:.2e}; j_tor co-Ip {sgn}")
        hard += [(f"slice {k} dpressure_dpsi", e_dp < 1e-9), (f"slice {k} rho monotone", mono),
                 (f"slice {k} Jensen", jen >= 0), (f"slice {k} gm5", b2 >= 0),
                 (f"slice {k} gm8 gm9", r8 >= 0), (f"slice {k} sign", sgn)]
    # -- the IDA time: drift, n_i, IDA internals, q
    i_ida = [round(t, 9) for t in dd["core_profiles"]["time"]].index(round(T_IDA, 9))
    cp = dd["core_profiles"]["profiles_1d"][i_ida]
    x_cp = np.asarray(cp["grid"]["psi_norm"])
    d = _psi_rho_drift(x_cp, cp["grid"]["rho_tor_norm"], gpath)
    print(f"  psi_N(rho) dd (t = {T_IDA}) vs g-file: "
          f"{', '.join(f'rho {k}: {v:+.4f}' for k, v in d['gate'].items())}; "
          f"max {d['max_abs']:.4f} (tol 2e-2) -> {'OK' if not d['exceeds'] else 'EXCEEDS'}")
    nd, nf = (np.asarray(cp["ion"][0][k], float) for k in ("density_thermal", "density_fast"))
    ni_ida = main_ion_density_from_zeff(ida["K"]["ne"], np.clip(ida["zeff"], 1, Z_IMP), Z_IMP)
    pts = np.asarray(NI_FAST_GATE_PSI_N)
    a = np.interp(pts, ida["x"], ni_ida)
    b = np.interp(pts, x_cp, nd + nf)
    dev = np.abs(a - b) / np.abs(b)
    print(f"  IDA n_i vs dd (n_D + n_fast) at psi_N {list(pts)}: "
          f"{', '.join(f'{v:.1e}' for v in dev)}; max {dev.max():.1e} (tol 1e-2)")
    for lab, I in (("dd IDA", ida), ("L-mode IDA", ida_l)):
        ne_i, nc_i, z_i = I["K"]["ne"], I["K"]["nc"], I["zeff"]
        ni_i = ne_i - Z_IMP * nc_i
        e_z = float(np.max(np.abs((ni_i + Z_IMP ** 2 * nc_i) / ne_i - z_i)))
        print(f"  {lab}: min(n_e - 6 n_C) {ni_i.min():.3e} >= 0; |Z_eff - (n_i + 36 n_C)/n_e| max {e_z:.1e}")
        hard.append((f"{lab} n_i >= 0", ni_i.min() >= 0))
    for lab, I, gp in (("dd IDA", ida, gpath), ("L-mode IDA", ida_l, gl_path)):
        g = read_geqdsk(gp)
        gq = np.interp(np.clip(I["x"], 0, 1), np.linspace(0, 1, np.size(g.qpsi)), np.abs(g.qpsi))
        print(f"  {lab} q vs its g-file q: max |dq| {np.max(np.abs(gq - I['q'])):.1e}; "
              f"g-file Ip {g.Ip:.4e}, B_center {g.B_center:+.3f}, q0 {abs(g.qpsi[0]):.3f}")
    # -- the outboard-midplane movement dd vs g-file (spec 4; the tests re-solve)
    g = read_geqdsk(gpath)
    gd = read_geqdsk(gfile_dd_t)
    from bouquet.coords import phi_n_from_q
    nodes = ida["x"][(ida["x"] >= 0.1) & (ida["x"] <= 0.98)]
    dr_psi = 1e3 * (r_out(gd, nodes) - r_out(g, nodes))
    _, phi_g = phi_n_from_q(np.linspace(0, 1, np.size(g.qpsi)), g.qpsi)
    _, phi_d = phi_n_from_q(np.linspace(0, 1, np.size(gd.qpsi)), gd.qpsi)
    ph = np.interp(nodes, np.linspace(0, 1, np.size(g.qpsi)), phi_g)
    psi_d_at = np.interp(ph, phi_d, np.linspace(0, 1, np.size(gd.qpsi)))
    dr_phi = 1e3 * (r_out(gd, psi_d_at) - r_out(g, nodes))
    print(f"  outboard midplane, dd (t = {T_IDA}) vs g-file, IDA nodes psi_N 0.1-0.98: "
          f"psi_N-held max |dR| {np.max(np.abs(dr_psi)):.1f} mm at psi_N "
          f"{nodes[np.argmax(np.abs(dr_psi))]:.2f}; Phi_N-held max |dR| "
          f"{np.max(np.abs(dr_phi)):.1f} mm (estimate: the tests re-solve with IDA kinetics)")
    # -- the reader
    try:
        import bouquet as bq
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            ddp = os.path.join(out_dir, "_dd_check.json")
            with open(ddp, "w") as fh:
                json.dump(dd, fh)
            src = bq.ImasSource(ids_path=ddp, time=T_IDA, ida_path=ipath, LCFS_geqdsk=gpath,
                                impurity_Z=Z_IMP)
            bl = bq.read_imas_baseline(src, kinetic_source="ida_hybrid")
            os.remove(ddp)
        conv = (bl.li_metrics or {}).get("imas_current_conversion")
        print(f"  reader (ida_hybrid): current conversion "
              f"{conv if isinstance(conv, str) else (conv or {}).get('method')}")
        seed = np.asarray(bl.j_inductive) - np.asarray(bl.j_pressure)
        print(f"  reader swb seed (j_inductive - j_pressure): min {seed.min():.3e}, seed[0] {seed[0]:.3e}, "
              f"seed at psi_N 0.9 / 0.99: {np.interp(0.9, bl.psi_N, seed):.3e} / "
              f"{np.interp(0.99, bl.psi_N, seed):.3e}, max {seed.max():.3e}")
        hard.append(("reader seed >= 0", bool(seed.min() >= 0 and seed[0] > 0)))
        for w in wl:
            if "Quarantined" in type(w.message).__name__:
                continue
            print(f"  reader WARNING: {type(w.message).__name__}: {str(w.message)[:300]}")
    except Exception as exc:                      # pragma: no cover
        print(f"  reader check FAILED: {type(exc).__name__}: {exc}")
        hard.append(("reader read", False))
    # -- pressure consistency of each g-file with its profile file (2 % on axis)
    pairs = (("H-mode thermal", "g_synthetic.geqdsk", "diiid_profs_synthetic.cdf"),
             ("H-mode total", "g_synthetic_pfast.geqdsk", "p_synthetic.peqdsk"),
             ("L-mode", "g_synthetic_lmode.geqdsk", "ida_synthetic_lmode.cdf"))
    for lab, gn, pn in pairs:
        a, pm, pg0, pp0 = pressure_consistency(os.path.join(out_dir, gn), os.path.join(out_dir, pn))
        print(f"  pressure {lab}: {gn} axis {pg0 / 1e3:.2f} kPa vs {pn} {pp0 / 1e3:.2f} kPa "
              f"(thermal{' + pb' if pn.endswith('.peqdsk') else ''}): {100 * a:+.2f} % "
              f"(tol 2 %); max |rel| over psi_N <= 0.95 {100 * pm:.2f} %")
        hard.append((f"{lab} pressure consistency", abs(a) < 0.02))
    if pfile is not None:
        from bouquet.io.pfile import read_pfile
        from bouquet.physics import isotropize_fast_pressure
        pf = read_pfile(os.path.join(out_dir, "p_synthetic.peqdsk"))
        pf.compute_quasineutrality()
        xz, zf = pf.compute_zeff()
        ez = float(np.max(np.abs(zf - np.interp(xz, pfile["x"], pfile["K"]["zeff"]))))
        print(f"  p-file read back: |Z_eff(p-file) - Z_eff| max {ez:.1e} (tol 1e-2); "
              f"pb axis {pf.pb[0]:.2f} kPa, ptot axis {pf.ptot[0]:.2f} kPa")
        hard.append(("p-file Z_eff", ez < 1e-2))
        par = sum(np.asarray(s_.get("pressure_fast_parallel", np.zeros(len(x_cp))), float)
                  for s_ in [cp["electrons"]] + list(cp["ion"]))
        perp = sum(np.asarray(s_.get("pressure_fast_perpendicular", np.zeros(len(x_cp))), float)
                   for s_ in [cp["electrons"]] + list(cp["ion"]))
        p_dd = isotropize_fast_pressure(perp, par, "sum")
        e_pb = float(np.max(np.abs(np.interp(pfile["x"], x_cp, p_dd) - pfile["pb"])) / np.max(p_dd))
        print(f"  p-file pb vs the dd's t = {T_IDA} fast pressure ('sum'), on the p-file grid: "
              f"max |diff| {e_pb:.1e} of its peak ({np.max(p_dd) / 1e3:.2f} kPa)")
    lrec = [r for r in recs if r["label"] == "lmode"][0]
    print(f"  L-mode q0 {lrec['q0']:.3f} (< 1: {lrec['q0'] < 1})")
    hard.append(("L-mode q0 < 1", lrec["q0"] < 1))
    bad = [n for n, ok in hard if not ok]
    print("==== hard checks: " + ("all passed" if not bad else "FAILED: " + ", ".join(bad)))
    return not bad


# =============================================================================
def pfast_ida_only(out_dir):
    """Write ``diiid_profs_synthetic_pfast.cdf`` (spec 2b) next to an EXISTING
    ``g_synthetic_pfast.geqdsk`` in ``out_dir``, without a solve: the kinetics
    of the thermal H-mode scenario (exactly those of
    ``diiid_profs_synthetic.cdf``) on the IDA grid, the q of that g-file, the
    sigma template of IDA_REGIME."""
    from bouquet.io.geqdsk import read_geqdsk
    k_ida = [round(s["t"], 9) for s in SLICES].index(round(T_IDA, 9))
    sg = dict(SLICES[k_ida], name="g", model="efit", jbeam=None)
    sgt = dict(sg, name="g_thermal", pressure=PRESSURE_THERMAL)
    g_pf = read_geqdsk(os.path.join(out_dir, "g_synthetic_pfast.geqdsk"), cocos=1)
    write_ida(os.path.join(out_dir, "diiid_profs_synthetic_pfast.cdf"), g_pf, sgt, T_IDA)
    return 0


def gate_only(out_dir, mesh=None, n_pass=3):
    """Evaluate the sanity gate of every scenario after a fresh start and
    ``n_pass`` composition passes (no fixture is written; ``gate_dump.npz``
    holds the compositions).  Returns 0 if every gate passes."""
    os.makedirs(out_dir, exist_ok=True)
    mygs = make_solver(mesh)
    scs = [(dict(s, model="fuse"), s["ip"], s["name"], LCFS_DD) for s in SLICES]
    k_ida = [round(s["t"], 9) for s in SLICES].index(round(T_IDA, 9))
    scs.append((dict(SLICES[k_ida], name="g", model="efit", jbeam=None), SLICES[k_ida]["ip"], "g", LCFS_DD))
    scs.append((dict(SLICES[k_ida], name="g_thermal", model="efit", jbeam=None, pressure=PRESSURE_THERMAL),
                SLICES[k_ida]["ip"], "g_thermal", LCFS_DD))
    scs.append((dict(LMODE, model="jphi", jphi=JPHI_LMODE), IP_LMODE, "lmode", LCFS_LMODE))
    dump, fails = {}, []
    for sc, ip, label, lcfs in scs:
        rec = solve_scenario(mygs, sc, ip, label, init=set_shape(mygs, lcfs), n_pass=n_pass,
                             gate_raise=False)
        if rec["gate"][0]:
            fails.append(label)
        for k in ("x", "j_bs_par", "j_ohm_par", "w_par", "A", "q0", "li_3"):
            dump[f"{label}_{k}"] = rec[k]
        dump[f"{label}_j_beam"] = rec["K"]["j_beam"]
    np.savez(os.path.join(out_dir, "gate_dump.npz"), **dump)
    print("==== gate-only: " + ("all passed" if not fails else "FAILED: " + ", ".join(fails)))
    return 0 if not fails else 1


def main(out_dir, mesh=None):
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "_work")      # diagnostic states and per-slice g-files
    os.makedirs(work, exist_ok=True)
    mygs = make_solver(mesh)
    print(f"mesh: {os.path.basename(mesh or MESH)}; profile grid {N_SOLVE} psi_N nodes; "
          f"equilibrium slices {N_EQ}; core_profiles {N_CP} (rho-uniform); g-files {N_GEQDSK}x{N_GEQDSK}",
          flush=True)
    recs, slices = [], []
    # -- the dd's five slices, an evolution solved in time order
    shape = set_shape(mygs, LCFS_DD)
    for k, s in enumerate(SLICES):
        sc = dict(s, model="fuse")
        rec = solve_scenario(mygs, sc, s["ip"], s["name"], init=shape if k == 0 else None)
        recs.append(rec)
        slices.append(equilibrium_slice(mygs, rec, s["t"], os.path.join(work, f"{s['name']}.geqdsk")))
    k_ida = [round(s["t"], 9) for s in SLICES].index(round(T_IDA, 9))
    eq = {"time": [float(s["t"]) for s in SLICES],
          "vacuum_toroidal_field": {"r0": R0, "b0": [B0] * len(SLICES)},
          "time_slice": [{k: v for k, v in ts.items() if not k.startswith("_")} for ts in slices]}
    cps, cp1s = [], []
    for k, (rec, ts) in enumerate(zip(recs, slices)):
        cp1, extra = core_profiles_slice(rec, ts, eq, max(k - 1, 0))
        cp1s.append(cp1)
        cps.append((ts["time"], extra))
        G = rec["G"]                     # r_j of the STORED composition on its own slice
        jeq = np.interp(extra["xc"], rec["x"], G["jeq"])
        jc = (np.interp(extra["xc"], rec["x"], rec["kappa"]) * abs(B0) * extra["j_total"]
              + np.interp(extra["xc"], rec["x"], rec["pG"]))
        r = np.abs(jeq - jc) / np.max(np.abs(jeq))
        print(f"[{rec['label']}] r_j of the stored core_profiles j_total on its own slice: "
              f"max {r.max():.2e} (at psi_N {extra['xc'][np.argmax(r)]:.3f}), rms {np.sqrt(np.mean(r ** 2)):.2e}")
    dd = {"equilibrium": eq,
          "core_profiles": {"ids_properties": {"comment": "synthetic test dd; p_fast_reduction=sum"},
                            "time": [float(s["t"]) for s in SLICES], "profiles_1d": cp1s},
          "core_sources": core_sources(cps)}
    write_dd(dd, os.path.join(out_dir, "dd_synthetic.json.gz"))
    # -- the T_IDA slice's g-files, EFIT-like current: first the TOTAL pressure
    # (g_synthetic_pfast + its p-file) ...
    sg = dict(SLICES[k_ida], name="g", model="efit", jbeam=None)
    rec_g = solve_scenario(mygs, sg, SLICES[k_ida]["ip"], "g")
    recs.append(rec_g)
    g_pf = save_gfile(mygs, os.path.join(out_dir, "g_synthetic_pfast.geqdsk"), rec_g)
    pfile = write_pfile(os.path.join(out_dir, "p_synthetic.peqdsk"), sg)
    # the state the thermal-only solve continues from (solved after the L-mode,
    # so that the L-mode solve sees the same solver history as before)
    snap_g = (mygs.copy_eq(), np.array(mygs._bq_offset, copy=True))
    # -- the L-mode g-file and its IDA file
    sl = dict(LMODE, model="jphi", jphi=JPHI_LMODE)
    shape_l = set_shape(mygs, LCFS_LMODE)
    rec_l = solve_scenario(mygs, sl, IP_LMODE, "lmode", init=shape_l)
    recs.append(rec_l)
    gl_path = os.path.join(out_dir, "g_synthetic_lmode.geqdsk")
    gl = save_gfile(mygs, gl_path, rec_l)
    ida_l = write_ida(os.path.join(out_dir, "ida_synthetic_lmode.cdf"), gl, sl, T_LMODE)
    # -- ... then the THERMAL-only g-file the IDA fit belongs to, continued
    # from the total-pressure state (same boundary, coils, Ip, mesh, settings)
    mygs.replace_eq(source_eq=snap_g[0])
    set_shape(mygs, LCFS_DD)
    mygs._bq_offset = snap_g[1]
    sgt = dict(sg, name="g_thermal", pressure=PRESSURE_THERMAL)
    rec_gt = solve_scenario(mygs, sgt, SLICES[k_ida]["ip"], "g_thermal")
    recs.append(rec_gt)
    g = save_gfile(mygs, os.path.join(out_dir, "g_synthetic.geqdsk"), rec_gt)
    ida = write_ida(os.path.join(out_dir, "diiid_profs_synthetic.cdf"), g, sgt, T_IDA)
    write_ida(os.path.join(out_dir, "diiid_profs_synthetic_pfast.cdf"), g_pf, sgt, T_IDA)
    ok = self_checks(out_dir, recs, ida, os.path.join(work, f"{SLICES[k_ida]['name']}.geqdsk"),
                     ida_l, gl_path, pfile=pfile)
    np.savez(os.path.join(work, "states.npz"), **{
        f"{r['label']}_{k}": v for r in recs for k, v in
        dict(x=r["x"], J=r["J"], jeq=r["G"]["jeq"], q=r["G"]["q"], j_ohm_tor=r["j_ohm_tor"],
             j_bs_tor=r["j_bs_tor"], j_beam_tor=r["j_beam_tor"], pG=r["pG"],
             kappa=r["kappa"], j_bs_par=r["j_bs_par"], j_ohm_par=r["j_ohm_par"],
             A=r["A"], li_3=r["li_3"], q0=r["q0"], q95=r["q95"], r_j=r["r_j"],
             r_j_rms=r["r_j_rms"], Ip=r["Ip"], n_pass=len(r["hist"]),
             n_retry=len(r["retries"]),
             stopped=-1 if r["stopped"] is None else r["stopped"]).items()})
    print(f"(diagnostic states and the per-slice g-files in {work})")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", nargs="?", default="synthetic_fixtures_out")
    ap.add_argument("--mesh", default=None, help="TokaMaker mesh for the solves (default: the "
                    "D3D-like example mesh); the fixtures written do not depend on it")
    ap.add_argument("--gate-only", action="store_true", help="evaluate the sanity gate of every "
                    "scenario after a short fresh-start solve; write only gate_dump.npz")
    ap.add_argument("--pfast-ida-only", action="store_true", help="write only "
                    "diiid_profs_synthetic_pfast.cdf next to an existing g_synthetic_pfast.geqdsk "
                    "in out_dir (no solve)")
    a = ap.parse_args()
    if a.pfast_ida_only:
        sys.exit(pfast_ida_only(a.out_dir))
    sys.exit(gate_only(a.out_dir, a.mesh) if a.gate_only else main(a.out_dir, a.mesh))
