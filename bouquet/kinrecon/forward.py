r"""Anchored forward solve: one magnetics-constrained TokaMaker equilibrium for a fully specified
current and pressure profile (the anchored recipe with an isoflux boundary target, weight factor 10).

What one solve is
-----------------
Given a flux-surface-averaged toroidal current density ``<j_phi>(psi_N)`` and a full-frame pressure
``p(psi_N)`` (both on knots from psi_N = 0 to 1), TokaMaker solves the free-boundary
Grad-Shafranov equation with

* the current profile as ``jphi-linterp`` (TokaMaker renormalizes it to the Ip target; the
  solver's ``ffp_scale`` is recorded) and ``P'`` = ``dp/dpsi_N`` as ``linterp`` scaled to the axis target
  ``pax = p_axis - p_sep`` (the "offset" separatrix-pressure frame of :mod:`.pressure`);
* in-solve least-squares rows (TokaMaker ``gs_fit_isoflux``):

  - flux loops and poloidal-field probes that the measurement set marks valid (for the EFIT
    k-file route: EFIT-weighted channels only -- channels EFIT gave zero weight broke the fit on
    every case tested), each with weight ``1/sigma`` of the measurement set;
  - flux loops in the ABSOLUTE gauge: ``target = measured + meta['absolute_offset']`` (the k-file
    ``SIREF`` route, :func:`.adapters.efit_kfile.flux_gauge`); the relative form as outer passes is
    ill-posed (the offset is almost unobservable in the coil fit; measured).
    Scoring stays in the measurement's own form;
  - probes at their center points; with ``probe_in_solve="corrected_point"`` (default) the target
    of each probe is shifted by ``b_point - b_length_averaged`` evaluated on the previous state, so
    that the in-solve objective equals the finite-length one used in scoring to first order (the
    correction is a lagged input and is iterated to consistency by the caller,
    :mod:`.current_model`);
  - coils by role (:data:`COIL_ROLES`): ``"held"`` coils by hard bounds ``measured +/- 1 mA``
    (EFIT ``IECURR = 1``; held E-coils make the coil-only problem well conditioned: condition
    number 12-21 against 2.7e4-3.1e4 when free), ``"fitted"`` coils by measurement rows
    ``weight = lambda / (mu0 sigma)`` acting on the coil's TOTAL current including its share of
    the vertical-stability coil (VSC), ``"free"`` coils (EFIT weight 0) unconstrained; plus a VSC
    gauge row (target 0);
  - isoflux rows on every ``isoflux_stride``-th point of a supplied REFERENCE boundary (from a
    prior magnetic equilibrium, e.g. EFIT01), weight ``anchor_factor * 2 pi median(1/sigma_loop)``
    (default factor 10; converged and verified on 5/5 magnetics cases at 10 and 30; working
    window 3-30, measured).  The default is CONVERGENCE-SELECTED: 10 is the center of the window
    in which the solves converge, not a value derived from the measurement uncertainty, and it is
    kept provisionally; results quote their sensitivity to it across the 3-30 window.  This makes
    the solve "magnetics + reference boundary", not a free-boundary shape reconstruction;
* the edge-P' taper policy ``auto`` (:func:`.edge.solve_with_edge_taper`) with one
  :class:`.edge.EdgeTaperCase` per forward model, so the taper is uniform within a case;
* OFT's default solver settings, never set: ``nl_tol``, ``urf``, ``maxits`` are read back and
  recorded and compared with the defaults at every solve (record key ``G7``);
* continuation: a solve starts from the current state of the solver (``init="keep"``), from a
  saved state (:meth:`ForwardModel.snapshot`), or -- first solve of a case only -- from a supplied
  reference ``psi`` on the mesh nodes;
* verified convergence: the solver flag AND a warm re-solve that reproduces the axis within
  0.5 mm and q0, l_i, Ip within 1e-3 (relative).  A solve that is not verified is a failure;
* the nonlinear-residual history of every solve is captured from the solver's own output
  (file descriptor 1) and classified with :func:`.edge.classify_residual_history`.

Returns a :class:`SolveResult` with the status, the stats, the history and a complete settings
record; :meth:`ForwardModel.settings_record` holds the case-level settings (rows, weights, roles).

Units and frames: TokaMaker psi per radian; flux-loop targets for ``set_flux_constraints`` are
``2 pi psi_TM = -psi_IMAS`` (OFT ``fit_flux_eval`` evaluates ``+2 pi psi_TM``, :mod:`.diagnostics`
``-2 pi psi_TM``); coil rows ``weight * mu0 * (I - target)`` with ``I`` in A (OFT
``tokamaker_set_coil_regmat`` multiplies targets by mu0; checked in the Fortran).  Plasma current must be positive (TokaMaker refuses a negative Ip target): a
reversed-Ip discharge is solved in the positive frame and the caller states the map.

Device data (mesh, coil names, VSC gains, the reference boundary, F0) are all supplied by the
caller; nothing device-specific is defined here.
"""

import ctypes
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from . import diagnostics as _diag
from . import edge as _edge
from . import pressure as _pressure

# gfortran buffers unit 6 unless this is set before the OFT library loads; the residual history is
# read from that output (see FortranStdout).  setdefault: an explicit user value is respected.  If
# OFT was imported before this module without it, the capture is unreliable (outputs of several
# solves arrive together): recorded in every settings record, and each solve's history is checked
# against its iteration count (an inconsistent history is never passed to the cycle detector).
HISTORY_CAPTURE_NOTE = ("OK: GFORTRAN_UNBUFFERED_PRECONNECTED set before OFT loaded"
                        if ("OpenFUSIONToolkit" not in sys.modules
                            or os.environ.get("GFORTRAN_UNBUFFERED_PRECONNECTED"))
                        else "UNRELIABLE: OpenFUSIONToolkit was imported before bouquet.kinrecon.forward without "
                             "GFORTRAN_UNBUFFERED_PRECONNECTED; solver output is buffered")
os.environ.setdefault("GFORTRAN_UNBUFFERED_PRECONNECTED", "y")

MU0 = 4.0e-7 * np.pi
#: Coil roles understood by :class:`ForwardModel` (k-file adapter ``meta['coil_roles']``).
COIL_ROLES = ("held", "fitted", "free")
#: Verified-convergence criterion (decided, fixed): warm re-solve reproduces these.
VERIFY_AXIS_M = 5.0e-4
VERIFY_REL = 1.0e-3


@dataclass
class ForwardSettings:
    """Every setting of the anchored forward solve.  Weights, sigma use, constraint sets and the
    anchor are decisions (each needs the user's confirmation).

    ``vsc`` (coil name -> gain of the virtual vertical-stability coil) is device data and must be
    given by the caller.
    """
    vsc: Dict[str, float] = field(default_factory=dict)
    vsc_gauge_weight: float = 1.0e2
    coil_lambda: float = 1.0                    # coil row weight = lambda / (mu0 sigma)
    coil_rows_vsc_aware: bool = True            # rows act on the total current incl. the VSC share
    held_halfwidth_A: float = 1.0e-3            # hard bounds of "held" coils
    anchor_factor: float = 10.0                 # isoflux weight = factor * 2 pi median(1/sigma_loop); convergence-selected
    isoflux_stride: int = 4
    flux_gauge: str = "absolute"                # loops in the solve: "absolute" (SIREF route)
    probe_in_solve: str = "corrected_point"     # or "point"
    probe_n_avg: int = _diag.PROBE_N_AVG_DEFAULT
    ip_target: str = "measured"                 # Ip target: the measured Ip of the set
    separatrix_pressure: str = _pressure.SEPARATRIX_PRESSURE_DEFAULT
    verify: bool = True
    nthreads: int = 1
    edge_taper: Any = field(default_factory=_edge.EdgeTaperSetting)
    detector_version: str = _edge.DETECTOR_VERSION_DEFAULT   # "v2": corrected classification (non-default)
    stall_policy: str = "report"                # "taper": treat a flat stall as taper-eligible (non-default)

    def as_dict(self):
        d = asdict(self)
        d["edge_taper"] = self.edge_taper.as_dict()
        d["verify_criterion"] = dict(axis_m=VERIFY_AXIS_M, q0_li_ip_rel=VERIFY_REL, rule="warm re-solve reproduces axis, q0, l_i and Ip")
        return d


@dataclass
class SolveResult:
    """Outcome of one forward solve (``converged`` means flag AND verification)."""
    status: str
    converged: bool
    verified: Optional[bool]
    stats: Dict[str, float]
    record: Dict[str, Any]
    wall_s: float


# ============================================================== residual-history capture
#: Debugging aid: set to a list to keep every captured solver output (None: not kept).
CAPTURE_LOG = None


class FortranStdout:
    """Capture file descriptor 1 (where the Fortran solver writes) for one solve, then echo it to
    the real stdout (or drop it with ``echo=False``).  Python and C stdio are flushed on entry and
    exit."""

    def __init__(self, echo=False):
        self.echo = echo
        self.text = ""

    def __enter__(self):
        self._libc = ctypes.CDLL(None)
        sys.stdout.flush()
        self._libc.fflush(None)
        self._saved = os.dup(1)
        self._tmp = tempfile.TemporaryFile()
        os.dup2(self._tmp.fileno(), 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        self._libc.fflush(None)
        os.dup2(self._saved, 1)
        os.close(self._saved)
        self._tmp.seek(0)
        raw = self._tmp.read()
        self._tmp.close()
        self.text = raw.decode("utf-8", errors="replace")
        if CAPTURE_LOG is not None:
            CAPTURE_LOG.append(self.text)
        if self.echo:
            os.write(1, raw)
        return False


_NUM = r"[-+]?\d\.\d{4}E[-+]\d{2}"
_ITER = re.compile(rf"^\s*(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$")


def parse_history(text):
    """Residual history of ONE TokaMaker nonlinear solve from its output.

    The solver prints, per iteration (``grad_shaf.F90``, after ``Starting non-linear GS solver``; the
    LAST such block of the captured output is parsed -- profile updates may print more than one):
    iteration, ffp_scale, p_scale, SQRT(nl_res), O-point R, O-point Z, vcontrol/mu0; column 4 is
    the norm compared with ``nl_tol``.  Accepted only if the counter runs 1, 2, 3, ... without gaps.
    """
    blocks = text.split("Starting non-linear GS solver")
    if len(blocks) < 2:
        return dict(ok=False, reason="no solver block found", residuals=[], n=0)
    its, res, oz = [], [], []
    for ln in blocks[-1].splitlines():
        m = _ITER.match(ln)
        if m:
            its.append(int(m.group(1)))
            res.append(float(m.group(4)))
            oz.append((float(m.group(5)), float(m.group(6))))
    ok = its == list(range(1, len(its) + 1)) and len(its) > 0
    return dict(ok=ok, reason=None if ok else "iteration counter not 1..n (lines lost or merged)",
                residuals=res, n=len(res), o_point_tail=oz[-6:], n_blocks=len(blocks) - 1)


# ============================================================== forward model
def _stats(gs):
    out = {}
    try:
        st = gs.get_stats()
        for k in ("Ip", "l_i", "q_0", "q_95", "beta_n", "beta_pol", "beta_tor", "W_MHD", "vol", "P_ax"):
            if k in st:
                out[k] = float(st[k])
    except Exception as exc:  # noqa: BLE001
        out["stats_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        out["R_axis"], out["Z_axis"] = (float(v) for v in gs.o_point)
    except Exception:  # noqa: BLE001
        pass
    return out


def verify_states(a, b):
    """Verified-convergence check of two stats dicts (solve and its warm re-solve)."""
    try:
        d_ax = float(np.hypot(a["R_axis"] - b["R_axis"], a["Z_axis"] - b["Z_axis"]))
        rel = {k: abs(a[k] - b[k]) / max(abs(a[k]), 1e-300) for k in ("q_0", "l_i", "Ip")}
    except (KeyError, TypeError) as exc:
        return dict(verdict="FAIL", reason=f"missing quantity: {exc}")
    ok = d_ax <= VERIFY_AXIS_M and all(v <= VERIFY_REL for v in rel.values())
    return dict(verdict="PASS" if ok else "FAIL", axis_m=d_ax, rel=rel,
                criterion=dict(axis_m=VERIFY_AXIS_M, rel=VERIFY_REL))


def orientation_record(measurements, F0):
    """Frames of every scored quantity for a forward solve, and the refusals.

    TokaMaker solves with a POSITIVE plasma-current target, and the forward solve uses the measured
    values directly as constraint rows in the discharge's frame.  :mod:`.score` can compare an
    equilibrium solved in another frame through a stated orientation map (``score(...,
    orientation=...)``), but the rows of THIS forward solve carry no such map.  Hence:

    * measured Ip < 0 is REFUSED here: the solved state would carry the opposite poloidal field, and
      every loop, probe and the Ip itself would flip sign against the rows;
    * B_t of either sign is allowed: ``F0`` is passed with the discharge's sign, so ``B_phi`` (MSE
      denominator, diamagnetic flux) is physical, and the MSE group's ``field_orientation`` must be
      ``(1, 1)`` (no map needed) -- anything else is refused as inconsistent.
    """
    rec = dict(frames=dict(ip="signed, discharge frame (must be > 0)", flux_loop="signed, IMAS -2 pi psi_TM",
                           bpol_probe="signed, B.n in the right-handed (R, phi, Z) frame",
                           diamagnetic_flux="signed with F0 (B_t direction) -- physical because F0 is signed",
                           mse="stated field_orientation (must be (1, 1) here)"),
               F0_sign=float(np.sign(F0)))
    ipg = measurements.groups.get("ip")
    if ipg is not None:
        ip = float(np.asarray(ipg.measured, float)[0])
        rec["ip_sign"] = float(np.sign(ip))
        if ip <= 0:
            raise ValueError("measured Ip <= 0: a reversed-Ip discharge cannot be reconstructed here (TokaMaker "
                             "solves to a positive Ip and loops/probes/Ip have no orientation map in score)")
    mse = measurements.groups.get("mse")
    if mse is not None:
        fo = tuple(float(v) for v in mse.meta.get("field_orientation", (1.0, 1.0)))
        rec["mse_field_orientation"] = fo
        if fo != (1.0, 1.0):
            raise ValueError(f"MSE field_orientation {fo} with a solve in the discharge's own frame (Ip > 0, "
                             "signed F0): only (1, 1) is consistent")
    return rec


def psi_on_nodes(nodes_rz, r_grid, z_grid, psi_grid, map_sign):
    """Initial-state helper: a reference equilibrium's psi(R, Z) grid on the mesh nodes.

    ``psi_grid`` has shape ``(len(z_grid), len(r_grid))`` (g-file ``PSIRZ`` layout); bicubic
    interpolation (``scipy.interpolate.RectBivariateSpline``, clamped to the grid box), times
    ``map_sign`` (the caller's stated map from the reference's flux convention to TokaMaker's,
    e.g. ``-sign(Ip)`` for a COCOS-1 g-file solved in the positive frame).  Used only to start the
    first solve of a case; it is an initial state, not a setting.
    """
    from scipy.interpolate import RectBivariateSpline
    r = np.clip(np.asarray(nodes_rz)[:, 0], r_grid[0], r_grid[-1])
    z = np.clip(np.asarray(nodes_rz)[:, 1], z_grid[0], z_grid[-1])
    spl = RectBivariateSpline(np.asarray(z_grid, float), np.asarray(r_grid, float), np.asarray(psi_grid, float),
                              kx=3, ky=3, s=0)
    return float(map_sign) * spl.ev(z, r)


class ForwardModel:
    """One TokaMaker object, set up once per case, plus the case's constraint rows.

    Parameters
    ----------
    mesh : tuple
        ``(pts, lc, reg, coil_dict, cond_dict)`` as returned by OFT ``load_gs_mesh``.
    F0 : float
        Vacuum ``R B_phi`` [T m] (sign as the discharge); changed between the slices of a series
        with :meth:`set_F0`.
    measurements : MeasurementSet
        Needs ``flux_loop``, ``bpol_probe``, ``ip`` and ``pf_current`` groups for the magnetics
        rows; the flux-loop group must carry ``absolute_offset`` (and ``coils_form`` not
        ``"undetermined"``) when ``settings.flux_gauge == "absolute"``.
    reference_boundary : array (n, 2)
        The anchor: (R, Z) points of a prior equilibrium's last closed surface.
    coil_roles : dict, optional
        Coil name -> role (:data:`COIL_ROLES`); default ``measurements.groups['pf_current']
        .meta['coil_roles']``.  Coils of the mesh without a role or a measurement are free.
    settings : ForwardSettings
    oft_env : OFT_env, optional
        An existing OFT environment (one per process); created with ``settings.nthreads`` if None.
    case_id : str
        Label of the case (edge-taper case lock, records).
    ip, generation_coil_reg : float, optional
        GENERATION MODE (``measurements=None``; synthetic truth states for closure tests): no
        magnetics or coil-measurement rows; the Ip target ``ip`` [A]; every coil regularized toward
        zero with the raw OFT weight ``generation_coil_reg`` (the closure tests use 1e-2);
        isoflux rows on ``reference_boundary`` with weight 1.  Never used for a
        reconstruction.

    :meth:`reconfigure` rebuilds the rows on the same TokaMaker object (one OFT environment and
    one TokaMaker object per process is the safe use of OFT).
    """

    def __init__(self, mesh, F0, measurements, reference_boundary, coil_roles=None, settings=None,
                 oft_env=None, case_id="case", ip=None, generation_coil_reg=None):
        from OpenFUSIONToolkit import OFT_env
        from OpenFUSIONToolkit.TokaMaker import TokaMaker
        import OpenFUSIONToolkit
        self.settings = settings or ForwardSettings()
        s = self.settings
        if s.flux_gauge != "absolute":
            raise ValueError("only flux_gauge='absolute' is implemented (the relative form as outer passes is "
                             "ill-posed: the offset is almost unobservable in the coil fit)")
        if s.probe_in_solve not in ("corrected_point", "point"):
            raise ValueError("probe_in_solve must be 'corrected_point' or 'point'")
        if not s.vsc:
            raise ValueError("settings.vsc (coil -> VSC gain) must be given: it is device data")
        self.env = oft_env if oft_env is not None else OFT_env(nthreads=s.nthreads)
        self.gs = TokaMaker(self.env)
        pts, lc, reg, coil_dict, cond_dict = mesh
        self.gs.setup_model(pts, lc, reg, cond_dict=cond_dict, coil_dict=coil_dict, order=2, F0=float(F0))
        self.gs.set_coil_vsc(dict(s.vsc))
        self.F0 = float(F0)
        self.defaults = self._read_settings()
        self.oft_path = os.path.dirname(os.path.abspath(OpenFUSIONToolkit.__file__))
        self.n_solves = 0
        self.history = []
        self.taper_restarts = []
        self.reconfigure(measurements, reference_boundary, coil_roles, case_id=case_id, ip=ip,
                         generation_coil_reg=generation_coil_reg)

    def reconfigure(self, measurements, reference_boundary, coil_roles=None, case_id="case", ip=None,
                    generation_coil_reg=None):
        """(Re)build the constraint rows and start a new edge-taper case on the same solver."""
        self.ms = measurements
        self.case = _edge.EdgeTaperCase(case_id=case_id, setting=self.settings.edge_taper)
        self.case_id = case_id
        if measurements is None:
            if ip is None or generation_coil_reg is None:
                raise ValueError("generation mode (measurements=None) needs ip and generation_coil_reg")
            self._setup_generation(reference_boundary, float(ip), float(generation_coil_reg))
        else:
            self._setup_rows(reference_boundary, coil_roles)
        self.probe_correction = np.zeros(self._probe["n"])

    def _setup_generation(self, reference_boundary, ip, weight):
        gs = self.gs
        regs = [gs.coil_reg_term({n: 1.0}, target=0.0, weight=weight) for n in gs.coil_sets]
        regs.append(gs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=weight))
        self._coils = dict(regs=regs, bounds={})
        self._fl = dict(mask=None, loc=np.zeros((0, 2)), target=np.zeros(0), weight=np.zeros(0))
        self._probe = dict(n=0, has_length=False)
        b = np.asarray(reference_boundary, float)
        if np.allclose(b[0], b[-1]):
            b = b[:-1]
        bi = b[::int(self.settings.isoflux_stride)].copy()
        self._iso = dict(points=bi, weight=1.0)
        self.reference_boundary = b
        self.ip_measured = ip
        self.rows_record = dict(mode="generation (no measurements; synthetic truth only)",
                                coils=dict(rule=f"all coils and the VSC regularized toward 0, raw weight {weight:g}"),
                                anchor=dict(n_points=int(len(bi)), weight=1.0), ip_target=dict(value=ip))

    # ---------------------------------------------------------------- setup
    def _read_settings(self):
        g = self.gs.settings
        return dict(nl_tol=float(g.nl_tol), urf=float(g.urf), maxits=int(g.maxits))

    def _setup_rows(self, reference_boundary, coil_roles):
        ms, s, gs = self.ms, self.settings, self.gs
        rec = dict(notes=[], orientation=orientation_record(ms, self.F0))
        # ---- flux loops
        fl = ms.groups["flux_loop"]
        vfl = np.asarray(fl.valid, bool) & np.isfinite(fl.measured) & np.isfinite(fl.sigma) & (fl.sigma > 0)
        loc_fl = np.column_stack((fl.geometry["r"], fl.geometry["z"]))
        on = _diag.on_mesh(gs, loc_fl)
        off = vfl & ~on
        vfl = vfl & on
        form = fl.meta.get("coils_form")
        off_abs = fl.meta.get("absolute_offset")
        if off_abs is None or form == "undetermined":
            raise ValueError("flux-loop group has no absolute gauge (meta 'absolute_offset' / 'coils_form'); "
                             "an absolute flux reference (k-file SIREF) is required by flux_gauge='absolute'")
        tgt_imas = np.asarray(fl.measured, float) + float(off_abs)
        self._fl = dict(mask=vfl, loc=loc_fl[vfl], target=-tgt_imas[vfl],
                        weight=1.0 / np.asarray(fl.sigma, float)[vfl])
        rec["flux_loops"] = dict(n_rows=int(vfl.sum()), n_off_mesh_excluded=int(off.sum()),
                                 gauge="absolute: target = measured + absolute_offset (SIREF route)",
                                 coils_form=form, weight="1/sigma of the measurement set",
                                 sigma_model=fl.meta.get("sigma_model"),
                                 scoring_form=("relative to " + str(fl.meta.get("flux_reference"))
                                               if fl.meta.get("flux_reference") else "absolute"))
        # ---- probes
        pr = ms.groups["bpol_probe"]
        vpr = np.asarray(pr.valid, bool) & np.isfinite(pr.measured) & np.isfinite(pr.sigma) & (pr.sigma > 0)
        loc_pr = np.column_stack((pr.geometry["r"], pr.geometry["z"]))
        nrm = np.column_stack((pr.geometry["direction_r"], pr.geometry["direction_z"]))
        nrm = nrm / np.hypot(nrm[:, 0], nrm[:, 1])[:, None]
        on = _diag.on_mesh(gs, loc_pr)
        excl = vpr & ~on
        has_len = "length" in pr.geometry
        seg_bad = np.zeros_like(vpr)
        if has_len:
            segp, ends = _diag.probe_segment_points(pr.geometry, s.probe_n_avg)
            for i in np.nonzero(vpr & on)[0]:
                pts_i = np.vstack((segp[i], ends[i]))
                if not np.all(_diag.on_mesh(gs, pts_i)):
                    seg_bad[i] = True
        vpr = vpr & on & ~seg_bad
        self._probe = dict(mask=vpr, loc=loc_pr[vpr], norm=nrm[vpr], measured=np.asarray(pr.measured, float)[vpr],
                           weight=1.0 / np.asarray(pr.sigma, float)[vpr], sigma=np.asarray(pr.sigma, float)[vpr],
                           n=int(vpr.sum()), has_length=has_len,
                           geometry={k: np.asarray(v)[vpr] for k, v in pr.geometry.items()})
        rec["probes"] = dict(n_rows=int(vpr.sum()), n_off_mesh_excluded=int(excl.sum()),
                             n_segment_off_mesh_excluded=int(seg_bad.sum()),
                             in_solve=(s.probe_in_solve if has_len else "point (no lengths in the geometry)"),
                             weight="1/sigma of the measurement set", sigma_model=pr.meta.get("sigma_model"))
        # ---- coils
        pf = ms.groups.get("pf_current")
        roles = dict(coil_roles if coil_roles is not None else (pf.meta.get("coil_roles", {}) if pf else {}))
        names = [str(n) for n in (pf.geometry.get("coil_names", pf.labels) if pf else [])]
        meas = {n: float(pf.measured[i]) for i, n in enumerate(names)} if pf else {}
        sig = {n: float(pf.sigma[i]) for i, n in enumerate(names)} if pf else {}
        regs, bounds, crec = [], {}, {}
        for n in gs.coil_sets:
            role = roles.get(n, "free")
            if role not in COIL_ROLES:
                raise ValueError(f"coil {n}: role {role!r} not in {COIL_ROLES} (state it explicitly)")
            if role in ("held", "fitted") and n not in meas:
                raise ValueError(f"coil {n} has role {role!r} but no measurement")
            if role == "held":
                bounds[n] = [meas[n] - s.held_halfwidth_A, meas[n] + s.held_halfwidth_A]
                crec[n] = dict(role="held", bounds_halfwidth_A=s.held_halfwidth_A)
            elif role == "fitted":
                w = s.coil_lambda / (MU0 * sig[n])
                coffs = {n: 1.0}
                if s.coil_rows_vsc_aware and n in s.vsc:
                    coffs["#VSC"] = float(s.vsc[n])
                regs.append(gs.coil_reg_term(coffs, target=meas[n], weight=w))
                crec[n] = dict(role="fitted", sigma_A=sig[n], equivalent_sigma_A=sig[n] / s.coil_lambda,
                               vsc_aware=bool(s.coil_rows_vsc_aware and n in s.vsc))
            else:
                crec[n] = dict(role="free", has_measurement=n in meas)
        regs.append(gs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=float(s.vsc_gauge_weight)))
        self._coils = dict(regs=regs, bounds=bounds)
        rec["coils"] = dict(per_coil=crec, vsc=dict(s.vsc), vsc_gauge_weight=s.vsc_gauge_weight,
                            coil_lambda=s.coil_lambda, row_rule="weight = lambda/(mu0 sigma), target = measured",
                            n_held=sum(1 for v in crec.values() if v["role"] == "held"),
                            n_fitted=sum(1 for v in crec.values() if v["role"] == "fitted"),
                            n_free=sum(1 for v in crec.values() if v["role"] == "free"))
        # ---- isoflux anchor
        b = np.asarray(reference_boundary, float)
        if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] < 8:
            raise ValueError("reference_boundary must be an (n >= 8, 2) array of (R, Z)")
        if np.allclose(b[0], b[-1]):
            b = b[:-1]
        bi = b[::int(s.isoflux_stride)].copy()
        med = float(np.median(self._fl["weight"])) if self._fl["weight"].size else 1.0
        wiso = float(s.anchor_factor) * 2.0 * np.pi * med
        self._iso = dict(points=bi, weight=wiso)
        self.reference_boundary = b
        rec["anchor"] = dict(n_points=int(len(bi)), stride=int(s.isoflux_stride), weight=wiso,
                             anchor_factor=float(s.anchor_factor),
                             rule="weight = factor * 2 pi median(1/sigma_loop) over the loop rows")
        # ---- Ip
        ipg = ms.groups["ip"]
        self.ip_measured = float(np.asarray(ipg.measured, float)[0])
        if self.ip_measured <= 0:
            rec["notes"].append("measured Ip < 0: solved to |Ip| in the positive frame (caller states the map)")
        rec["ip_target"] = dict(source="measured Ip of the set", value_sign=float(np.sign(self.ip_measured)))
        self.rows_record = rec

    def settings_record(self):
        """Case-level settings: OFT build and defaults, every row and weight, coil roles, anchor."""
        return dict(settings=self.settings.as_dict(), oft_python=self.oft_path, oft_defaults=self.defaults,
                    oft_settings_now=self._read_settings(), F0=self.F0, rows=self.rows_record,
                    edge_taper_case=self.case.as_dict(), n_solves=self.n_solves,
                    history_capture=HISTORY_CAPTURE_NOTE, taper_restarts=list(self.taper_restarts))

    def lock_tapered(self, reason):
        """Lock this case's edge taper ON (the ``auto`` policy after a detected limit cycle, applied to
        the WHOLE reconstruction so that the case stays uniform): every later solve is tapered."""
        if self.case.setting.mode is not _edge.EdgeTaperMode.AUTO:
            raise RuntimeError("lock_tapered applies only to the 'auto' edge-taper mode")
        self.case.locked = True
        self.case.spec = self.case.setting.spec
        self.case.decided_by = str(reason)
        self.taper_restarts.append(str(reason))

    def set_F0(self, F0):
        """Vacuum ``R0 B_phi`` [T m] for the following solves (a time series: each slice's own
        measured value).  It is passed to TokaMaker with the profiles of every solve, so it also
        replaces the value carried by a restored snapshot.  A change of sign (B_t reversal) is
        refused: the orientation record is made once per case."""
        F0 = float(F0)
        if not np.isfinite(F0) or F0 == 0.0 or np.sign(F0) != np.sign(self.F0):
            raise ValueError(f"F0 = {F0!r}: must be finite, nonzero and of the case's sign ({np.sign(self.F0):+.0f})")
        self.F0 = F0

    # ---------------------------------------------------------------- state
    def snapshot(self):
        """A copy of the solver's equilibrium (psi, coil currents, profiles, targets)."""
        return self.gs.copy_eq()

    def restore(self, snap):
        self.gs.replace_eq(snap)

    def set_psi(self, psi_nodes):
        self.gs.set_psi(np.ascontiguousarray(psi_nodes, dtype=float))

    # ---------------------------------------------------------------- rows
    def _apply_rows(self):
        gs = self.gs
        f = self._fl
        if f["loc"].shape[0]:
            gs.set_flux_constraints(np.ascontiguousarray(f["loc"]), np.ascontiguousarray(f["target"]),
                                    weights=np.ascontiguousarray(f["weight"]))
        else:
            gs.set_psi_constraints(None, None)
        p = self._probe
        if p["n"]:
            tgt = p["measured"] + (self.probe_correction if self.settings.probe_in_solve == "corrected_point"
                                   and p["has_length"] else 0.0)
            gs.set_mirnov_constraints(np.ascontiguousarray(p["loc"]), np.ascontiguousarray(p["norm"]),
                                      np.ascontiguousarray(tgt), weights=np.ascontiguousarray(p["weight"]))
        else:
            gs.set_mirnov_constraints(None, None, None)
        gs.set_isoflux_constraints(np.ascontiguousarray(self._iso["points"]),
                                   weights=np.full(len(self._iso["points"]), self._iso["weight"]))
        gs.set_coil_reg(reg_terms=list(self._coils["regs"]))
        gs.set_coil_bounds(dict(self._coils["bounds"]) if self._coils["bounds"] else None)

    def update_probe_correction(self):
        """``b_point - b_length_averaged`` on the current state (lagged input of the next solve).
        Returns the change relative to the probe sigma (max over probes)."""
        p = self._probe
        if not (p["n"] and p["has_length"] and self.settings.probe_in_solve == "corrected_point"):
            return 0.0
        point = _diag.bpol_probe(self.gs, p["geometry"], model="point").value
        avg = _diag.bpol_probe(self.gs, p["geometry"], model="finite_length", n_avg=self.settings.probe_n_avg).value
        new = point - avg
        if not np.all(np.isfinite(new)):
            raise RuntimeError("probe correction not finite on the current state")
        change = float(np.max(np.abs(new - self.probe_correction) / p["sigma"]))
        self.probe_correction = new
        return change

    # ---------------------------------------------------------------- one solve
    def solve(self, jphi, pressure, ip=None, init="keep", label="solve", lock_case=True):
        """One anchored solve under the edge-taper policy, then the verified-convergence check.

        ``jphi`` = ``(x, <j_phi>)`` [A/m^2] and ``pressure`` = ``(x, p_full)`` [Pa] on knots from
        psi_N = 0 to 1 (the same or different knots); ``ip`` [A] defaults to the measured Ip.
        ``init``: ``"keep"`` (continue from the solver's state), a snapshot, or an array of psi
        on the mesh nodes.  ``lock_case=False`` runs the edge-taper policy WITHOUT the case lock
        (only for a starting-state solve with other profiles, which must not decide the case's
        taper).  Returns a :class:`SolveResult`; a failed solve leaves the solver in its
        failed state (the caller restores a snapshot).
        """
        t_all = time.perf_counter()
        s, gs = self.settings, self.gs
        xj, yj = (np.asarray(v, float) for v in jphi)
        xp, pp = (np.asarray(v, float) for v in pressure)
        for xx in (xj, xp):
            if xx[0] != 0.0 or xx[-1] != 1.0 or np.any(np.diff(xx) <= 0):
                raise ValueError("profile knots must increase strictly from exactly 0 to exactly 1")
        if not (np.all(np.isfinite(yj)) and np.all(np.isfinite(pp))):
            raise ValueError("non-finite profile values")
        ipv = abs(self.ip_measured if ip is None else float(ip))
        base = _pressure.solver_inputs(xp, pp, pprime=_pressure.pressure_gradient(xp, pp),
                                       separatrix_pressure=s.separatrix_pressure)
        if isinstance(init, str):
            if init != "keep":
                raise ValueError("init must be 'keep', a snapshot or a psi array")
            start = self.snapshot()
        elif isinstance(init, np.ndarray):
            self.set_psi(init)
            start = self.snapshot()
        else:
            self.restore(init)
            start = self.snapshot()
        attempts = []

        def one(taper):
            self.restore(start)
            if taper is None:
                x_, y_, pax, trec = xp, base["pprime"], base["pax"], None
            else:
                x_, y_, pax, trec = _pressure.solver_inputs_with_taper(
                    xp, pp, base["pprime"], 1.0, taper, separatrix_pressure=s.separatrix_pressure)
            gs.set_targets(Ip=ipv, pax=float(pax))
            gs.set_profiles(ffp_prof={"type": "jphi-linterp", "x": xj, "y": yj}, foffset=self.F0,
                            pp_prof={"type": "linterp", "x": np.asarray(x_, float), "y": np.asarray(y_, float)})
            self._apply_rows()
            now = self._read_settings()
            t0 = time.perf_counter()
            err, its = None, None
            with FortranStdout() as cap:
                try:
                    res = gs.solve(return_its=True)
                    its = int(res[1]) if isinstance(res, tuple) else None
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
            self.n_solves += 1
            h = parse_history(cap.text)
            expect = its if err is None else (now["maxits"] if "maxits" in (err or "").lower() or
                                              "iteration" in (err or "").lower() else None)
            consistent = bool(h["ok"] and (expect is None or h["n"] == expect))
            info = dict(error=err, its=its, wall_s=round(time.perf_counter() - t0, 4), settings_at_solve=now,
                        G7=("PASS" if now == self.defaults else "FAIL"),
                        history=_edge.summarize_history(h["residuals"]), history_consistent=consistent,
                        taper=None if trec is None else dict(pax_ratio=trec["taper"].get("pax_ratio"),
                                                             variant=trec["taper_variant"]))
            info.update(_stats(gs))
            if err is not None or not consistent:
                # solver output kept for diagnosis (warnings, the iteration lines)
                info["output_tail"] = cap.text[-3000:]
                info["n_surface_warnings"] = cap.text.count("flux-surface evaluation failed")
            attempts.append(info)
            self.history.append(dict(label=label + ("" if taper is None else ":tapered"),
                                     residuals=list(h["residuals"])))
            return _edge.SolveAttempt(converged=err is None, residuals=(h["residuals"] if consistent else []),
                                      payload=None, info=info)

        out = _edge.solve_with_edge_taper(one, setting=None if lock_case else s.edge_taper,
                                          case=self.case if lock_case else None,
                                          label=f"{self.case_id}:{label}", tol=self.defaults["nl_tol"],
                                          detector_kw=dict(version=s.detector_version),
                                          stall_policy=s.stall_policy)
        pol = out.record
        flag = pol["status"] != _edge.STATUS_FAILED
        stats = _stats(gs) if flag else {}
        ver = None
        if flag and s.verify:
            a = dict(stats)
            t0 = time.perf_counter()
            with FortranStdout() as cap:
                try:
                    gs.solve()
                    err = None
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
            self.n_solves += 1
            b = _stats(gs)
            v = verify_states(a, b) if err is None else dict(verdict="FAIL", reason=err)
            v["wall_s"] = round(time.perf_counter() - t0, 4)
            v["warm_its"] = parse_history(cap.text).get("n")
            ver = v["verdict"] == "PASS"
            stats = b
            pol["verification"] = v
        converged = bool(flag and (ver is not False))
        status = pol["status"] if converged else ("failed_verification" if flag else "failed")
        cycle_locked = bool(not flag and (pol.get("case_lock") or {}).get("locked") is False and any(
            (a.get("detector") or {}).get("verdict") == "limit_cycle" for a in pol["attempts"]))
        if cycle_locked:
            status = "failed_limit_cycle_untapered_lock"
        rec = dict(label=label, status=status, policy=pol, attempts=attempts,
                   ffp_scale_solver=None, pax=base["pax"], p_sep=base["p_sep"], ip_target=ipv)
        try:
            if converged:
                rec["ffp_scale_solver"] = float(gs.ffp_scale)
        except Exception:  # noqa: BLE001
            pass
        return SolveResult(status=status, converged=converged, verified=ver, stats=stats, record=rec,
                           wall_s=round(time.perf_counter() - t_all, 4))

    # ---------------------------------------------------------------- geometry metrics
    def boundary_distance(self):
        """First-order normal distance [m] of the solved LCFS from the reference boundary points:
        ``(psi_N - 1) |psi_b - psi_a| / (R |B_p|)`` at each reference point; points with
        ``|B_p| < 0.2 median`` (X-point region) are excluded.  Returns rms, max, mean, counts."""
        gs = self.gs
        pb = self.reference_boundary
        pn = np.asarray(_diag.psi_n_at(gs, pb)[0], float)
        B, _ = _diag.field_at(gs, pb)
        bp = np.hypot(B[:, 0], B[:, 2])
        lo, hi = gs.psi_bounds
        dist = (pn - 1.0) * abs(hi - lo) / np.maximum(pb[:, 0] * bp, 1e-30)
        good = np.isfinite(dist) & (bp > 0.2 * np.nanmedian(bp))
        return dict(rms_m=float(np.sqrt(np.mean(dist[good] ** 2))), max_m=float(np.max(np.abs(dist[good]))),
                    mean_m=float(np.mean(dist[good])), n_used=int(good.sum()), n=int(len(pb)),
                    method="(psi_N - 1) |psi_b - psi_a| / (R |B_p|) at the reference points; |B_p| < 0.2 median "
                           "excluded")
