r"""Anchored forward solve: one magnetics-constrained TokaMaker equilibrium for a fully specified
current and pressure profile (the anchored recipe with an isoflux boundary target, weight factor
:data:`ISOFLUX_WEIGHT_FACTOR` = 3, an interim value).

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
    ``SIREF`` route, :func:`.adapters.efit_kfile.flux_gauge`), from a MEASURED reference flux (a gauge
    built from another fit's fitted reference is refused unless ``fitted_reference_gauge=True``); the
    relative form as outer passes is ill-posed (the offset is almost unobservable in the coil fit;
    measured).  Scoring stays in the measurement's own form; when the set asks for a fitted additive
    offset (``meta['flux_offset'] = 'fitted'``) it is fitted in the score (:func:`.score.score`), one
    per table, with the reference prior row;
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
    prior magnetic equilibrium), weight ``anchor_factor * 2 pi median(1/sigma_loop)`` (default
    factor :data:`ISOFLUX_WEIGHT_FACTOR`; see there for what the value trades).  This makes the
    solve "magnetics + reference boundary", not a free-boundary shape reconstruction; every result
    records how far the achieved boundary moved from the reference (:meth:`ForwardModel.boundary_distance`);
* the plasma source at the last closed flux surface integrated by TokaMaker's cut-cell quadrature
  (setting ``lcfs_cut_cells``) when the solver build provides it (:attr:`ForwardSettings.lcfs_cut_cells`,
  default ``"if_available"``; :func:`solver_build`): each mesh cell cut by the boundary contributes
  only its part inside, so the discrete equations stay continuous when ``P'`` or the current is
  finite at the boundary.  A build without the setting solves as before (point test of each
  quadrature point), and every result records which of the two was in effect and the build;
* the edge-P' taper (:func:`.edge.solve_with_edge_taper`) only when the caller asks for it
  (:attr:`ForwardSettings.edge_taper`, default ``off``), with one :class:`.edge.EdgeTaperCase` per
  forward model, so the taper is uniform within a case.  The taper changes the INPUT pressure
  gradient in the outermost layer; it is a workaround for builds without the cut-cell setting;
* OFT's default solver settings, never set: ``nl_tol``, ``urf``, ``maxits`` are read from the
  TokaMaker wrapper's Python-side settings (the values ``update_settings`` sends; the solver itself
  offers no getter) and recorded and compared with the defaults at every solve (record key ``G7``).
  ``lcfs_cut_cells`` is an opt-in of the quadrature, not a tolerance: it is set as described above
  and recorded beside them, never part of the ``G7`` comparison.  The one exception is the iteration
  limit ``maxits``, and only as an explicit, recorded, non-default setting
  (:attr:`ForwardSettings.solver_maxits`, default None = OFT's own limit untouched): when given, it
  is set once in the wrapper's settings and sent to the solver, and ``G7`` compares with OFT's
  defaults with that one value replaced; ``nl_tol`` and ``urf`` are never set.  Every solve's
  iteration count is logged (:meth:`ForwardModel.iteration_record`);
* continuation: a solve starts from the current state of the solver (``init="keep"``), from a
  saved state (:meth:`ForwardModel.snapshot`), or -- first solve of a case only -- from a supplied
  reference ``psi`` on the mesh nodes;
* verified convergence: the solver flag AND a warm re-solve that reproduces the axis within
  0.5 mm and q0, l_i, Ip within 1e-3 (relative).  A solve that is not verified is a failure.  The l_i
  compared is the solver's ``'iter'`` normalization, ``2 Bp_vol / ((mu0 Ip)^2 R_axis)``
  (:data:`VERIFY_LI`, stats key ``l_i_iter``), not the solver's default l_i: the default divides by the
  length of the traced psi_N = 0.99 surface of a diverted equilibrium, and that traced length can jump by
  a few tenths of a percent between states that do not move, which made unchanged states fail the
  verification on l_i alone.  The default l_i is still computed and recorded (``l_i``), and the
  consistency loop's own state test still uses it;
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
import hashlib
import os
import re
import sys
import tempfile
import time
import warnings
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
#: The quantities the verification compares to ``VERIFY_REL`` (stats keys).  ``l_i_iter`` is the solver's ``'iter'``
#: normalization of l_i, ``2 Bp_vol / ((mu0 Ip)^2 R_axis)``, which uses no traced surface length (module docstring).
VERIFY_LI = "l_i_iter"
VERIFY_KEYS = ("q_0", VERIFY_LI, "Ip")
#: Default isoflux (boundary-anchor) weight factor: the isoflux rows carry weight
#: ``factor * 2 pi median(1/sigma_loop)``.  INTERIM value 3 (the earlier default was 10, convergence-selected as
#: the center of the 3-30 window in which the anchored solves converged on the magnetics cases).  What it
#: trades: a stiff anchor holds the boundary on the reference and pushes any disagreement between the model's
#: interior and the reference boundary into the sensor misfit (loops and probes); a weak anchor lets the sensors
#: move the boundary.  Measured on four first-slice fits with a kinetic interior: 10 -> 3 lowered the loop
#: chi^2/N by 0.2-1.0 and the probe chi^2/N by 0.1-0.6 and moved the fitted boundary 4-8 mm rms from the
#: reference (1.5-4 mm at 10); 3 is the lowest factor at which all four converged (1 failed on one).  The
#: factor is not derived from the boundary's own uncertainty; it is a recorded option
#: (:attr:`ForwardSettings.anchor_factor`) so that it can be varied, and every result records the boundary
#: displacement it produced (:meth:`ForwardModel.boundary_distance`).
ISOFLUX_WEIGHT_FACTOR = 3.0
#: Points of the limiter contour per meter when the strike points of a solved state are located.
STRIKE_SEARCH_POINTS_PER_M = 500
#: psi_N of the recorded ``q_90`` (q on the psi_N = 0.9 surface, beside TokaMaker's ``q_95``).
Q_90_PSIN = 0.9

#: TokaMaker setting of the cut-cell quadrature of the plasma integrals at the last closed flux surface.
CUT_CELL_SETTING = "lcfs_cut_cells"
#: Exported library symbol of the cut-cell rule (gfortran name of the module procedure).  Its presence in the
#: loaded library is checked against the Python package's settings: the Python settings structure and the library
#: must come from ONE build, or the settings passed to the solver are misaligned.
CUT_CELL_LIBRARY_SYMBOL = "__oft_gs_MOD_gs_cut_cell_rule"
#: Values of :attr:`ForwardSettings.lcfs_cut_cells`: ``"if_available"`` (default) on when the build provides the
#: setting, otherwise the build's own point test; ``"on"`` required (a build without it is refused); ``"off"``
#: the point test also on a build that provides the setting.
LCFS_CUT_CELL_OPTIONS = ("if_available", "on", "off")
LCFS_CUT_CELLS_DEFAULT = "if_available"
#: What the recorded solver-setting values are (``oft_defaults``, ``oft_settings_now``, ``settings_at_solve``,
#: ``iteration_limit``): the TokaMaker wrapper's Python-side settings, the values ``update_settings`` sends to the
#: solver; the solver offers no getter, so they are not read from the solver itself.
SETTINGS_SOURCE_NOTE = ("the TokaMaker wrapper's Python-side settings (the values update_settings sends to the "
                        "solver; the solver has no getter): in_effect / now = the wrapper's value, oft_default = "
                        "this instance's value at construction")
_NOTICES = set()


def check_solver_maxits(value):
    """:attr:`ForwardSettings.solver_maxits` checked: None (OFT's own iteration limit, never set) or a whole number
    >= 1 (an int, or a float with an integral value; a bool or a string is refused).  Returns None or the int."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"solver_maxits must be None or a whole number >= 1, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"solver_maxits must be None or a whole number >= 1, got {value!r}") from None
    if not np.isfinite(v) or v != int(v) or int(v) < 1:
        raise ValueError(f"solver_maxits must be None or a whole number >= 1, got {value!r}")
    return int(v)


def _notice_once(key, text):
    """One message per process and reason (a solver-build fact, not a per-solve event)."""
    if key not in _NOTICES:
        _NOTICES.add(key)
        warnings.warn(text, RuntimeWarning, stacklevel=3)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_BUILD_CACHE = {}


def solver_build():
    """Identity of the loaded OpenFUSIONToolkit build and whether it provides the cut-cell setting.

    Read without a TokaMaker instance (cached per process; the library cannot change once loaded):

    * ``provides_lcfs_cut_cells``: the Python package's TokaMaker settings have ``lcfs_cut_cells`` AND its settings
      structure has the field AND the loaded library exports the cut-cell rule
      (:data:`CUT_CELL_LIBRARY_SYMBOL`).  The three must agree; any disagreement (a Python package of one build
      with the library of another) raises ``RuntimeError``, because the settings structure would be misaligned;
    * ``liboftpy_path``, ``liboftpy_sha256``: the library file that was loaded and its hash.  The hash is the
      reliable build identifier: the package version and the revision in the solver's banner are the same for a
      patched and an unpatched build of one revision;
    * ``tokamaker_python_sha256``: hashes of the TokaMaker Python files that define the settings and the structure;
    * ``oft_version``: the package's version string (not build-specific, see above).
    """
    if "build" in _BUILD_CACHE:
        return _copy_build(_BUILD_CACHE["build"])
    import OpenFUSIONToolkit
    from OpenFUSIONToolkit import _interface as oft_interface
    from OpenFUSIONToolkit.TokaMaker import _core as tm_core
    from OpenFUSIONToolkit.TokaMaker import _interface as tm_interface
    lib = oft_interface.oftpy_lib
    lib_path = getattr(lib, "_name", None)
    in_python = CUT_CELL_SETTING in vars(tm_core.tokamaker_settings())
    in_struct = any(f[0] == CUT_CELL_SETTING for f in tm_interface.tokamaker_settings_cstruct._fields_)
    in_library = hasattr(lib, CUT_CELL_LIBRARY_SYMBOL)
    if not in_python == in_struct == in_library:
        raise RuntimeError(f"inconsistent OpenFUSIONToolkit build: '{CUT_CELL_SETTING}' in the Python settings "
                           f"{in_python}, in the settings structure {in_struct}, cut-cell rule in the loaded library "
                           f"{lib_path} {in_library}; the Python package and the library must come from one build")
    py_files = {n: os.path.abspath(m.__file__) for n, m in (("TokaMaker/_core.py", tm_core),
                                                             ("TokaMaker/_interface.py", tm_interface))}
    build = dict(provides_lcfs_cut_cells=bool(in_python), library_symbol=CUT_CELL_LIBRARY_SYMBOL,
                 oft_python=os.path.dirname(os.path.abspath(OpenFUSIONToolkit.__file__)),
                 oft_version=getattr(OpenFUSIONToolkit, "__version__", None),
                 liboftpy_path=lib_path,
                 liboftpy_sha256=_sha256(lib_path) if lib_path and os.path.isfile(lib_path) else None,
                 tokamaker_python_sha256={n: _sha256(p) for n, p in py_files.items()},
                 identifier_note="liboftpy_sha256 identifies the build; oft_version and the solver banner's revision "
                                 "do not distinguish a patched from an unpatched build of one revision")
    _BUILD_CACHE["build"] = build
    return _copy_build(build)


def _copy_build(build):
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in build.items()}


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
    anchor_factor: float = ISOFLUX_WEIGHT_FACTOR   # isoflux weight = factor * 2 pi median(1/sigma_loop); interim
    isoflux_stride: int = 4
    flux_gauge: str = "absolute"                # loops in the solve: "absolute" (SIREF route)
    probe_in_solve: str = "corrected_point"     # or "point"
    probe_n_avg: int = _diag.PROBE_N_AVG_DEFAULT
    ip_target: str = "measured"                 # Ip target: the measured Ip of the set
    separatrix_pressure: str = _pressure.SEPARATRIX_PRESSURE_DEFAULT
    verify: bool = True
    nthreads: int = 1
    lcfs_cut_cells: str = LCFS_CUT_CELLS_DEFAULT   # LCFS_CUT_CELL_OPTIONS; "off" is a recorded opt-out
    edge_taper: Any = field(default_factory=_edge.EdgeTaperSetting)   # default mode "off": an opt-in workaround
    detector_version: str = _edge.DETECTOR_VERSION_DEFAULT   # "v2": corrected classification (non-default)
    stall_policy: str = "report"                # "taper": treat a flat stall as taper-eligible (non-default)
    fitted_reference_gauge: bool = False        # True: accept a gauge built from ANOTHER fit's fitted reference
    #                                             (non-default; the default takes it from a measured reference)
    solver_maxits: Optional[int] = None         # None: OFT's own nonlinear iteration limit, never set (default);
    #                                             a whole number >= 1: that limit, sent once to the solver and used by
    #                                             every solve of this forward model -- an explicit, recorded,
    #                                             non-default setting; nl_tol, urf and every criterion are untouched

    def as_dict(self):
        d = asdict(self)
        d["edge_taper"] = self.edge_taper.as_dict()
        d["verify_criterion"] = dict(axis_m=VERIFY_AXIS_M, q0_li_ip_rel=VERIFY_REL, li=VERIFY_LI,
                                     rule="warm re-solve reproduces axis, q0, l_i ('iter' normalization, "
                                          "2 Bp_vol/((mu0 Ip)^2 R_axis)) and Ip")
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
#: Speed options read by the forward solve (:class:`.speed.SpeedSettings`; this process switch is off outside a speed
#: scope, and the fit and series entry points set it on by default).
SPEED_OPTIONS = dict(reuse_stats=False)


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
    ``vsc_A``: column 7 of the last line, the vertical-stability coil current [A per unit gain] of the state the
    solve leaves (printed after that iteration's coil solve, 5 significant digits; None when not accepted).
    """
    blocks = text.split("Starting non-linear GS solver")
    if len(blocks) < 2:
        return dict(ok=False, reason="no solver block found", residuals=[], n=0)
    its, res, oz, vsc = [], [], [], []
    for ln in blocks[-1].splitlines():
        m = _ITER.match(ln)
        if m:
            its.append(int(m.group(1)))
            res.append(float(m.group(4)))
            oz.append((float(m.group(5)), float(m.group(6))))
            vsc.append(float(m.group(7)))
    ok = its == list(range(1, len(its) + 1)) and len(its) > 0
    return dict(ok=ok, reason=None if ok else "iteration counter not 1..n (lines lost or merged)",
                residuals=res, n=len(res), o_point_tail=oz[-6:], n_blocks=len(blocks) - 1,
                vsc_A=(vsc[-1] if ok else None))


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
        # q on the psi_N = 0.9 surface: the same trace (TokaMaker get_q) and sign as get_stats' q_95
        out["q_90"] = float(np.asarray(gs.get_q(psi=np.array([Q_90_PSIN]))[1], float)[0])
    except Exception as exc:  # noqa: BLE001
        out["q_90_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        out["R_axis"], out["Z_axis"] = (float(v) for v in gs.o_point)
    except Exception:  # noqa: BLE001
        pass
    try:
        # l_i on the solver's 'iter' normalization (the formula of TokaMaker get_stats(li_normalization='iter')):
        # 2 Bp_vol / ((mu0 Ip)^2 R_axis) -- no traced surface length; the quantity the verification compares
        ip_g, _c, _v, _pv, _df, _tf, bp_vol = gs.get_globals()
        out[VERIFY_LI] = float(2.0 * bp_vol / ((MU0 * ip_g) ** 2 * float(gs.o_point[0])))
    except Exception as exc:  # noqa: BLE001
        out[VERIFY_LI + "_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def verify_states(a, b):
    """Verified-convergence check of two stats dicts (solve and its warm re-solve): the axis within
    :data:`VERIFY_AXIS_M` and :data:`VERIFY_KEYS` (q0, the ``'iter'`` l_i, Ip) within :data:`VERIFY_REL` (relative).
    A missing quantity is a FAIL."""
    try:
        d_ax = float(np.hypot(a["R_axis"] - b["R_axis"], a["Z_axis"] - b["Z_axis"]))
        rel = {k: abs(a[k] - b[k]) / max(abs(a[k]), 1e-300) for k in VERIFY_KEYS}
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


def read_cut_cells(gs):
    """``lcfs_cut_cells`` of a TokaMaker instance's settings: the Python-side value that ``update_settings`` sends to
    the solver; None when its build has no such setting.  The solver offers no way to read the value back from its
    own (Fortran) settings, so this is what can be recorded; whether the value reached the solver is shown by
    behavior (the live tests: a case that converges only with cut cells stalls under ``"off"``)."""
    v = vars(gs.settings)
    return bool(v[CUT_CELL_SETTING]) if CUT_CELL_SETTING in v else None


def configure_cut_cells(gs, option, build):
    """Apply :attr:`ForwardSettings.lcfs_cut_cells` to a set-up TokaMaker instance; returns the record.

    ``build`` is :func:`solver_build`.  The setting is present when ``lcfs_cut_cells`` is an attribute of the
    instance's settings (no exception is used to find out: on a build without it, assigning the attribute raises).
    ``"if_available"``: on when present, otherwise nothing is set and the solve uses the build's point test;
    ``"on"``: a build without the setting is refused; ``"off"``: set to False when present.  When present, the value
    is sent to the solver with ``update_settings()``.  Running without the cut cells (either reason) gives one
    message per process.  The record states the option, whether the build provides the setting, the value in
    effect (None without the setting) and why.
    """
    if option not in LCFS_CUT_CELL_OPTIONS:
        raise ValueError(f"lcfs_cut_cells must be one of {LCFS_CUT_CELL_OPTIONS}")
    provided = CUT_CELL_SETTING in vars(gs.settings)
    if provided != bool(build["provides_lcfs_cut_cells"]):
        raise RuntimeError(f"the TokaMaker settings ('{CUT_CELL_SETTING}' present: {provided}) disagree with the "
                           f"build record ({build['provides_lcfs_cut_cells']})")
    if option == "on" and not provided:
        raise RuntimeError(f"lcfs_cut_cells='on' but this OpenFUSIONToolkit build ({build.get('liboftpy_path')}) has "
                           f"no '{CUT_CELL_SETTING}' setting")
    want = bool(provided and option != "off")
    if provided:
        setattr(gs.settings, CUT_CELL_SETTING, want)
        gs.update_settings()
    if not provided:
        reason = ("the solver build does not provide the setting: the plasma source at the boundary uses the build's "
                  "point test of each quadrature point")
        _notice_once("not_provided", f"bouquet.kinrecon: this OpenFUSIONToolkit build ({build.get('liboftpy_path')}) "
                     f"has no '{CUT_CELL_SETTING}' setting; the forward solves run without the cut-cell quadrature at "
                     "the last closed flux surface (as before). Solves with a pressure gradient or current that is "
                     "finite at the boundary may then cycle or stall; the edge taper (ForwardSettings.edge_taper) is "
                     "the opt-in workaround for such a build.")
    elif not want:
        reason = "turned off by the option lcfs_cut_cells='off'"
        _notice_once("off", f"bouquet.kinrecon: lcfs_cut_cells='off': the forward solves run without the cut-cell "
                     "quadrature although the solver build provides it.")
    else:
        reason = "on: the solver build provides the setting"
    return dict(option=option, provided=bool(provided), requested=want if provided else None,
                in_effect=read_cut_cells(gs), reason=reason, liboftpy_sha256=build.get("liboftpy_sha256"),
                in_effect_source=("the Python-side settings value sent by update_settings (the solver has no "
                                  "getter for its own settings)"))


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
    reference_points : dict, optional
        Points of the reference equilibrium that its boundary polygon does not carry, used only to
        RECORD the displacement of the achieved boundary (:meth:`boundary_distance`), never as rows:
        ``"x_points"`` -- list of (R, Z) [m], ``"strike_points"`` -- dict name -> (R, Z) [m] on the
        limiter (names ``"<lower|upper>_<inner|outer>"``); no defaults (without them the solved
        X-points and strike points are listed without a displacement).

    :meth:`reconfigure` rebuilds the rows on the same TokaMaker object (one OFT environment and
    one TokaMaker object per process is the safe use of OFT).
    """

    def __init__(self, mesh, F0, measurements, reference_boundary, coil_roles=None, settings=None,
                 oft_env=None, case_id="case", ip=None, generation_coil_reg=None, reference_points=None):
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
        if s.lcfs_cut_cells not in LCFS_CUT_CELL_OPTIONS:
            raise ValueError(f"lcfs_cut_cells must be one of {LCFS_CUT_CELL_OPTIONS}")
        maxits = check_solver_maxits(s.solver_maxits)
        self.solver_build = solver_build()
        if s.lcfs_cut_cells == "on" and not self.solver_build["provides_lcfs_cut_cells"]:
            raise RuntimeError(f"lcfs_cut_cells='on' but this OpenFUSIONToolkit build "
                               f"({self.solver_build['liboftpy_path']}) has no '{CUT_CELL_SETTING}' setting")
        self.env = oft_env if oft_env is not None else OFT_env(nthreads=s.nthreads)
        self.gs = TokaMaker(self.env)
        pts, lc, reg, coil_dict, cond_dict = mesh
        self.gs.setup_model(pts, lc, reg, cond_dict=cond_dict, coil_dict=coil_dict, order=2, F0=float(F0))
        self.gs.set_coil_vsc(dict(s.vsc))
        self.cut_cells = configure_cut_cells(self.gs, s.lcfs_cut_cells, self.solver_build)
        self.F0 = float(F0)
        self.defaults = self._read_settings()
        self.iteration_limit = self._configure_iteration_limit(maxits)
        self.oft_path = os.path.dirname(os.path.abspath(OpenFUSIONToolkit.__file__))
        self.n_solves = 0
        self.iteration_log = []
        self.history = []
        self.taper_restarts = []
        self.reconfigure(measurements, reference_boundary, coil_roles, case_id=case_id, ip=ip,
                         generation_coil_reg=generation_coil_reg, reference_points=reference_points)

    def reconfigure(self, measurements, reference_boundary, coil_roles=None, case_id="case", ip=None,
                    generation_coil_reg=None, reference_points=None):
        """(Re)build the constraint rows and start a new edge-taper case on the same solver."""
        self.ms = measurements
        self.case = _edge.EdgeTaperCase(case_id=case_id, setting=self.settings.edge_taper)
        self.case_id = case_id
        self.reference_points = dict(reference_points or {})
        self._inner_dof = None
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

    def _configure_iteration_limit(self, maxits):
        """Apply :attr:`ForwardSettings.solver_maxits` (already checked) once, after OFT's defaults were read.
        None: nothing is set and ``G7`` compares with OFT's defaults.  A number: ``maxits`` only is set in the
        TokaMaker wrapper's Python-side settings and sent to the solver (``update_settings``); the wrapper's value
        is re-read (the solver itself has no getter, so this checks the value that was sent, not the solver), and
        ``G7`` compares with OFT's defaults with that value replaced; ``nl_tol`` and ``urf`` stay OFT's.  Returns
        the record: ``setting``; ``in_effect`` and ``oft_default`` = the wrapper's value after the setting and this
        instance's value at construction (:data:`SETTINGS_SOURCE_NOTE`)."""
        oft = int(self.defaults["maxits"])
        if maxits is None:
            self.expected_settings = dict(self.defaults)
            return dict(setting=None, in_effect=oft, oft_default=oft, values_source=SETTINGS_SOURCE_NOTE,
                        rule="OFT's own nonlinear iteration limit (never set)")
        self.gs.settings.maxits = int(maxits)
        self.gs.update_settings()
        self.expected_settings = dict(self.defaults, maxits=int(maxits))
        now = self._read_settings()
        if now != self.expected_settings:
            raise RuntimeError(f"solver settings after setting maxits = {maxits}: {now}; expected "
                               f"{self.expected_settings}")
        return dict(setting=int(maxits), in_effect=int(now["maxits"]), oft_default=oft,
                    values_source=SETTINGS_SOURCE_NOTE,
                    rule="explicit non-default setting ForwardSettings.solver_maxits: the nonlinear iteration limit "
                         "sent to the solver for every solve of this forward model; nl_tol and urf are OFT's")

    def _log_iterations(self, label, its, kind, at_limit):
        """One entry of :attr:`iteration_log` per solver call counted in :attr:`n_solves`."""
        self.iteration_log.append(dict(label=str(label), its=None if its is None else int(its), kind=kind,
                                       stopped_at_limit=bool(at_limit)))

    def iteration_record(self, start=0):
        """Iteration counts of the solves logged since index ``start`` of :attr:`iteration_log` (a fit or a series
        passes its own start; one entry per solver call counted in :attr:`n_solves`, warm verification re-solves
        included).  ``its``: the counts in call order (None where the solver output could not be read);
        ``kind``: ``"solve"`` or ``"verify"`` per entry; ``n_over_oft_default``: solves that needed more iterations
        than OFT's own limit (this instance's default at construction; possible only with
        :attr:`ForwardSettings.solver_maxits` above it); ``n_stopped_at_limit``: solves that ended with the solver's
        iteration-limit error (its output parsed; not a read of the solver's setting); ``over_oft_default``: label,
        count and kind of each solve above OFT's limit or stopped at the limit.  A record only."""
        log = self.iteration_log[int(start):]
        oft = int(self.iteration_limit["oft_default"])
        its = [e["its"] for e in log]
        known = [i for i in its if i is not None]
        flagged = [dict(e) for e in log if (e["its"] is not None and e["its"] > oft) or e["stopped_at_limit"]]
        return dict(limit=dict(self.iteration_limit), n_solves=len(log), its=its, kind=[e["kind"] for e in log],
                    n_unknown=sum(1 for i in its if i is None), max_its=(max(known) if known else None),
                    n_over_oft_default=sum(1 for i in known if i > oft),
                    n_stopped_at_limit=sum(1 for e in log if e["stopped_at_limit"]),
                    over_oft_default=flagged)

    def _setup_rows(self, reference_boundary, coil_roles):
        ms, s, gs = self.ms, self.settings, self.gs
        rec = dict(notes=[], orientation=orientation_record(ms, self.F0))
        # ---- flux loops
        fl = ms.groups["flux_loop"]
        conv = fl.meta.get("reference_convention")
        if conv == "fitted" and not s.fitted_reference_gauge:
            raise ValueError("the flux-loop gauge was built from a FITTED reference (another fit's solution); the "
                             "forward solve takes its gauge from the measured reference -- pass the measured "
                             "reference flux (reference_convention='measured'), or set "
                             "ForwardSettings(fitted_reference_gauge=True) (recorded)")
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
                                 reference_convention=conv or "measured (k-file SIREF)",
                                 scoring_offset=("fitted in the score (one additive offset per table, with the "
                                                 "reference prior row when present)"
                                                 if fl.meta.get("flux_offset") == "fitted" else None),
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
        """Case-level settings: OFT build and defaults, the cut-cell setting, every row and weight, coil roles,
        anchor, edge taper."""
        return dict(settings=self.settings.as_dict(), oft_python=self.oft_path, oft_defaults=self.defaults,
                    oft_settings_now=self._read_settings(), oft_settings_expected=dict(self.expected_settings),
                    F0=self.F0, rows=self.rows_record,
                    solver=self.solver_record(),
                    edge_taper_case=self.case.as_dict(), n_solves=self.n_solves,
                    history_capture=HISTORY_CAPTURE_NOTE, taper_restarts=list(self.taper_restarts))

    def solver_record(self):
        """The solver build (:func:`solver_build`), the cut-cell setting (option, provided, value in effect, read
        back now) and the edge-taper setting (mode and parameters): recorded with every solve, fit and series."""
        return dict(build=_copy_build(self.solver_build),
                    lcfs_cut_cells=dict(self.cut_cells, now=read_cut_cells(self.gs)),
                    edge_taper=self.settings.edge_taper.as_dict(),
                    iteration_limit=dict(self.iteration_limit, now=int(self.gs.settings.maxits)))

    def coil_current_record(self, solve=None):
        """RECORD that steers nothing: the coil currents of the solver's present state and the vertical-stability
        coil (VSC) current.  TokaMaker's ``get_coil_currents`` returns each coil's TOTAL current, its own current plus
        its VSC gain times the VSC current; the VSC current has no getter and is the value the solver printed for the
        state that ``solve`` (the :class:`SolveResult` that produced the present state) left (``record["vsc_A"]``,
        :func:`parse_history`; 5 significant digits).  Per coil with a nonzero gain: the gain, the total, the VSC
        share (gain x VSC current), the coil's own current (total - share) and share / total.  Without ``solve``
        the VSC current and the split are not stated.  Never raises."""
        try:
            cur, _reg = self.gs.get_coil_currents()
            tot = {str(k): float(v) for k, v in dict(cur).items()}
            vsc = None if solve is None else solve.record.get("vsc_A")
            per = {}
            for n, g in self.settings.vsc.items():
                if float(g) == 0.0 or str(n) not in tot:
                    continue
                t = tot[str(n)]
                share = None if vsc is None else float(g) * float(vsc)
                per[str(n)] = dict(gain=float(g), total_A=t, vsc_share_A=share,
                                   own_A=(None if share is None else t - share),
                                   share_of_total=(None if share is None or t == 0.0 else share / t))
            return dict(computed=True, vsc_A=vsc, vsc_coils=per, totals_A=tot,
                        vsc_source=("solver output: last iteration line of the solve that left this state (5 "
                                    "significant digits)" if vsc is not None else "not stated (no solve record)"),
                        rule="total = own + gain x VSC current (get_coil_currents returns the totals)")
        except Exception as exc:  # noqa: BLE001 -- a record must never end a fit
            return dict(computed=False, reason=f"{type(exc).__name__}: {exc}"[:300])

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
        last_stats = [None]      # the stats of the latest attempt's state (the state the policy leaves)

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
            cut_now = read_cut_cells(gs)
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
            at_limit = err is not None and ("maxits" in err.lower() or "iteration" in err.lower())
            expect = its if err is None else (now["maxits"] if at_limit else None)
            consistent = bool(h["ok"] and (expect is None or h["n"] == expect))
            self._log_iterations(label + ("" if taper is None else ":tapered"),
                                 its if its is not None else (h["n"] if consistent else None), "solve", at_limit)
            info = dict(error=err, its=its, wall_s=round(time.perf_counter() - t0, 4), settings_at_solve=now,
                        G7=("PASS" if now == self.expected_settings else "FAIL"),
                        lcfs_cut_cells_at_solve=cut_now,
                        lcfs_cut_cells_check=("PASS" if cut_now == self.cut_cells["in_effect"] else "FAIL"),
                        history=_edge.summarize_history(h["residuals"]), history_consistent=consistent,
                        vsc_A=h["vsc_A"],
                        taper=None if trec is None else dict(pax_ratio=trec["taper"].get("pax_ratio"),
                                                             variant=trec["taper_variant"]))
            st_now = _stats(gs)
            last_stats[0] = st_now
            info.update(st_now)
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
        if flag and SPEED_OPTIONS["reuse_stats"] and last_stats[0] is not None:
            # opt-in: the solver still holds the state of the latest attempt, whose stats were just evaluated; they are
            # reused (the same evaluation of the same state) instead of evaluated a second time
            stats = dict(last_stats[0])
        else:
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
            hv = parse_history(cap.text)
            v["warm_its"] = hv.get("n")
            v["vsc_A"] = hv.get("vsc_A")
            self._log_iterations(label + ":verify", v["warm_its"], "verify",
                                 err is not None and ("maxits" in err.lower() or "iteration" in err.lower()))
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
                   ffp_scale_solver=None, pax=base["pax"], p_sep=base["p_sep"], ip_target=ipv,
                   solver=dict(lcfs_cut_cells=read_cut_cells(gs), provides_lcfs_cut_cells=self.cut_cells["provided"],
                               lcfs_cut_cells_option=s.lcfs_cut_cells,
                               liboftpy_sha256=self.solver_build["liboftpy_sha256"],
                               edge_taper_mode=s.edge_taper.mode.value,
                               edge_taper_case_locked=self.case.locked if lock_case else None))
        try:
            if converged:
                rec["ffp_scale_solver"] = float(gs.ffp_scale)
        except Exception:  # noqa: BLE001
            pass
        # the VSC current of the state this solve leaves (the warm re-solve when it ran, else the latest attempt)
        rec["vsc_A"] = ((pol.get("verification") or {}).get("vsc_A") if ver is not None
                        else (attempts[-1].get("vsc_A") if attempts else None))
        return SolveResult(status=status, converged=converged, verified=ver, stats=stats, record=rec,
                           wall_s=round(time.perf_counter() - t_all, 4))

    # ---------------------------------------------------------------- inner (coil) degrees of freedom
    def inner_dof(self):
        """Effective number of parameters the coil solve inside every forward solve fits to the data rows.

        The coil currents of the non-held coils and the VSC are solved inside each Picard step of TokaMaker
        (``gs_fit_isoflux``) by linear least squares on the rows of :meth:`_apply_rows` at fixed plasma current:
        flux loops ``w (2 pi psi)``, probes ``w B.n``, isoflux rows ``w (psi_j - psi_0)`` and the coil rows
        ``weight mu0 (coffs . I)``.  Its design matrix ``A`` holds, per unknown, the vacuum response of each row
        to a unit coil current (``vac_solve`` of that coil alone; the VSC column is the gain-weighted sum); the
        hat matrix ``H = A (A^T A)^-1 A^T`` (rank-revealing SVD) is plasma-independent.  It leaves out the plasma's
        own response to the coil currents (the Picard iterations re-solve the plasma with the coils): measured by
        finite differences on a synthetic case (each data row moved by one sigma, the anchored solve repeated at
        fixed profiles), the full response is about 1 % below this vacuum trace.  The trace of ``H`` splits by row
        kind: ``flux_loop``, ``bpol_probe``, ``fitted_coil`` (data rows), ``isoflux`` and ``vsc_gauge`` (not
        data).  Returns ``{"by_kind", "n_rows", "rank", "n_unknowns", "condition", "method"}``, cached until the
        rows are rebuilt (:meth:`reconfigure`); the solver's equilibrium is restored afterwards.  None in
        generation mode (no data rows)."""
        if self.ms is None:
            return None
        if self._inner_dof is not None:
            return self._inner_dof
        gs = self.gs
        fl, pr, iso = self._fl, self._probe, self._iso
        held = set(self._coils["bounds"])
        coils = [n for n in gs.coil_sets if n not in held]
        snap = self.snapshot()
        cols = {}
        try:
            for c in coils:
                gs.set_coil_currents({c: 1.0})
                with FortranStdout():          # the vacuum solves' iteration output is not kept
                    eq = gs.vac_solve()
                col = {}
                col["flux_loop"] = (2.0 * np.pi * _diag.psi_at(eq, fl["loc"])[0] * fl["weight"]
                                    if fl["loc"].shape[0] else np.zeros(0))
                if pr["n"]:
                    B, _ = _diag.field_at(eq, pr["loc"])
                    col["bpol_probe"] = (B[:, 0] * pr["norm"][:, 0] + B[:, 2] * pr["norm"][:, 1]) * pr["weight"]
                else:
                    col["bpol_probe"] = np.zeros(0)
                piso = _diag.psi_at(eq, iso["points"])[0]
                col["isoflux"] = (piso[1:] - piso[0]) * iso["weight"]
                cols[c] = col
        finally:
            gs.set_coil_currents({})
            self.restore(snap)
        names = coils + ["#VSC"]
        cols["#VSC"] = {k: sum(float(g) * cols[n][k] for n, g in self.settings.vsc.items() if n in cols)
                        for k in ("flux_loop", "bpol_probe", "isoflux")}
        regs = self._coils["regs"]
        R = np.array([[float(r.coffs.get(n, 0.0)) * float(r.weight) * MU0 for n in names] for r in regs])
        kinds = (["flux_loop"] * fl["loc"].shape[0] + ["bpol_probe"] * int(pr["n"])
                 + ["isoflux"] * (iso["points"].shape[0] - 1)
                 + ["fitted_coil" if any(k != "#VSC" for k in r.coffs) else "vsc_gauge" for r in regs])
        A = np.vstack([np.column_stack([cols[n][k] for n in names]).reshape(-1, len(names))
                       for k in ("flux_loop", "bpol_probe", "isoflux")] + [R.reshape(-1, len(names))])
        if not np.all(np.isfinite(A)):
            raise RuntimeError("inner coil design matrix is not finite")
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        rank = int(np.sum(s > s[0] * 1.0e-12))
        h = np.sum(U[:, :rank] ** 2, axis=1)
        kinds = np.array(kinds)
        self._inner_dof = dict(
            by_kind={k: float(h[kinds == k].sum()) for k in ("flux_loop", "bpol_probe", "fitted_coil", "isoflux",
                                                             "vsc_gauge")},
            n_rows={k: int(np.count_nonzero(kinds == k)) for k in ("flux_loop", "bpol_probe", "fitted_coil",
                                                                    "isoflux", "vsc_gauge")},
            rank=rank, n_unknowns=len(names), unknowns=names, condition=float(s[0] / s[rank - 1]),
            method="trace of the hat matrix of the in-solve linear coil least squares (vacuum response per unit coil "
                   "current, rows and weights as installed); the data rows are flux_loop, bpol_probe, fitted_coil")
        return self._inner_dof

    def inner_dof_data(self, groups):
        """Sum of :meth:`inner_dof` over the data-row kinds whose measurement groups are in ``groups`` (flux
        loops -> ``flux_loop``, probes -> ``bpol_probe``, fitted coils -> ``pf_current``).  0.0 when not
        available (generation mode)."""
        d = self.inner_dof()
        if d is None:
            return 0.0
        gmap = dict(flux_loop="flux_loop", bpol_probe="bpol_probe", fitted_coil="pf_current")
        return float(sum(v for k, v in d["by_kind"].items() if k in gmap and gmap[k] in set(groups)))

    # ---------------------------------------------------------------- geometry metrics
    def boundary_distance(self):
        """Displacement of the solved boundary from the reference (anchor) boundary.

        Sign convention (every signed displacement of this class): POSITIVE when the solved boundary lies OUTSIDE the
        anchor boundary (the solved plasma extends beyond it), negative when it lies inside.

        * ``rms_m``, ``max_m``, ``mean_m``: first-order normal distance at each reference point,
          ``(1 - psi_N) |psi_b - psi_a| / (R |B_p|)`` (psi_N of the solved equilibrium at the reference point: below 1
          where the reference point lies inside the solved boundary, i.e. where the solved boundary lies outside);
          points with ``|B_p| < 0.2 median`` (X-point region) are excluded;
        * ``displacement_mm`` (:meth:`boundary_displacement`): the same rms and maximum in mm, the outboard and
          inboard midplane displacement, the X-point(s) and strike points.
        """
        gs = self.gs
        pb = self.reference_boundary
        pn = np.asarray(_diag.psi_n_at(gs, pb)[0], float)
        B, _ = _diag.field_at(gs, pb)
        bp = np.hypot(B[:, 0], B[:, 2])
        lo, hi = gs.psi_bounds
        dist = (1.0 - pn) * abs(hi - lo) / np.maximum(pb[:, 0] * bp, 1e-30)     # positive: solved boundary outside
        good = np.isfinite(dist) & (bp > 0.2 * np.nanmedian(bp))
        out = dict(rms_m=float(np.sqrt(np.mean(dist[good] ** 2))), max_m=float(np.max(np.abs(dist[good]))),
                   mean_m=float(np.mean(dist[good])), n_used=int(good.sum()), n=int(len(pb)),
                   method="(1 - psi_N) |psi_b - psi_a| / (R |B_p|) at the reference points; |B_p| < 0.2 median "
                          "excluded",
                   sign="positive: the solved boundary lies outside the anchor boundary")
        try:
            out["displacement_mm"] = self.boundary_displacement(out)
        except Exception as exc:  # noqa: BLE001
            out["displacement_mm"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])
        return out

    def _psi_bry(self):
        """The solver's LCFS flux value [Wb/rad] (the ``psi_bounds`` entry that maps to psi_N = 1)."""
        gs = self.gs
        lo, hi = (float(v) for v in gs.psi_bounds)
        n_lo = float(np.asarray(gs.abspsi_to_normalized(np.array([lo])), float).ravel()[0])
        return lo if abs(n_lo - 1.0) < abs(n_lo) else hi

    def _psi_direct(self):
        """``psi(pts)`` [Wb/rad] with TokaMaker's own interpolator, for points that lie inside the mesh region by
        construction (the horizontal line from the axis to just beyond the boundary, the limiter contour), without
        the per-point mesh test of :mod:`.diagnostics` (whose cost grows with points x triangles and is paid again
        for every new point set).  Non-finite values come back as NaN.  Returns ``(function, release)``."""
        fe = self.gs.get_field_eval("PSI")

        def f(pts):
            v = np.asarray(fe.eval(np.ascontiguousarray(np.asarray(pts, float).reshape(-1, 2))), float).reshape(-1)
            return np.where(np.isfinite(v), v, np.nan)
        return f, fe

    def _lcfs_crossing(self, r_ref, z, r_axis, psi_fn):
        """R [m] where the solved psi_N = 1 crosses the horizontal line Z = z on the side of ``r_ref`` (from the
        axis outward), by a scan in 1 mm steps from the axis and bisection; NaN when there is none."""
        gs = self.gs
        sgn = 1.0 if r_ref >= r_axis else -1.0
        r = r_axis + sgn * np.arange(0.002, 1.5 * abs(r_ref - r_axis) + 0.1, 0.001)
        r = r[r > 0.0]
        pn = np.asarray(gs.abspsi_to_normalized(psi_fn(np.column_stack((r, np.full_like(r, z))))), float)
        ok = np.isfinite(pn)
        idx = np.nonzero(ok[:-1] & ok[1:] & (pn[:-1] < 1.0) & (pn[1:] >= 1.0))[0]
        if not idx.size:
            return np.nan
        i = int(idx[0])
        a, b = r[i], r[i + 1]
        for _ in range(30):
            m = 0.5 * (a + b)
            if float(np.asarray(gs.abspsi_to_normalized(psi_fn(np.array([[m, z]]))), float).ravel()[0]) < 1.0:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    def _strike_points(self, xpts, psi_fn):
        """Limiter crossings of the solved psi = psi_b nearest each X-point, on its inner and outer side and on
        the far side of the X-point from the axis; ``{"<lower|upper>_<inner|outer>": (R, Z)}``."""
        gs = self.gs
        lim = getattr(gs, "lim_contour", None)
        if lim is None or not len(xpts):
            return {}
        lim = np.asarray(lim, float)[:, :2]
        seg = np.hypot(*np.diff(lim, axis=0).T)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        n = max(200, int(s[-1] * STRIKE_SEARCH_POINTS_PER_M))
        ss = np.linspace(0.0, s[-1], n)
        pts = np.column_stack((np.interp(ss, s, lim[:, 0]), np.interp(ss, s, lim[:, 1])))
        v = psi_fn(pts) - self._psi_bry()
        ok = np.isfinite(v)
        idx = np.nonzero(ok[:-1] & ok[1:] & (v[:-1] * v[1:] < 0))[0]
        cross = np.array([pts[i] + (pts[i + 1] - pts[i]) * v[i] / (v[i] - v[i + 1]) for i in idx]).reshape(-1, 2)
        za = float(gs.o_point[1])
        out = {}
        for xp in xpts:
            pos = "lower" if xp[1] < za else "upper"
            far = cross[(cross[:, 1] < xp[1]) if pos == "lower" else (cross[:, 1] > xp[1])]
            for side, sel in (("inner", far[:, 0] < xp[0]), ("outer", far[:, 0] >= xp[0])):
                cand = far[sel]
                if len(cand):
                    k = int(np.argmin(np.hypot(cand[:, 0] - xp[0], cand[:, 1] - xp[1])))
                    out[f"{pos}_{side}"] = (float(cand[k, 0]), float(cand[k, 1]))
        return out

    def boundary_displacement(self, normal=None):
        """Displacement of the solved boundary from the reference boundary, in mm (recorded with every solve and fit
        result through :meth:`boundary_distance`):

        Sign: positive when the solved boundary lies OUTSIDE the anchor boundary (:meth:`boundary_distance`).

        * ``rms``, ``max``, ``mean``: the first-order normal distances of :meth:`boundary_distance` (X-point region
          excluded; ``mean`` signed);
        * ``outboard_midplane``, ``inboard_midplane``: signed distance of the solved LCFS crossing from the reference
          polygon's crossing along the horizontal line through the solved magnetic axis, positive outside the anchor
          (outboard: the solved crossing at larger R; inboard: at smaller R);
        * ``x_points``: per reference X-point (``reference_points["x_points"]``; none by default: the polygon
          alone does not say whether its extreme points are X-points) the distance to the nearest solved X-point
          (TokaMaker ``get_xpoints``) on the same side of the axis, with both positions; the solved X-points are
          listed in any case (``x_points_solved``);
        * ``strike_points``: per reference strike point (``reference_points["strike_points"]``; none by default)
          the distance to the solved strike point of the same name (limiter crossing of psi = psi_b nearest the
          solved X-point); the solved strike points are listed in any case.
        """
        gs = self.gs
        nd = normal if normal is not None else self.boundary_distance()
        out = dict(rms=1e3 * nd["rms_m"], max=1e3 * nd["max_m"], mean=1e3 * nd["mean_m"],
                   anchor_factor=float(self.settings.anchor_factor),
                   sign="positive: the solved boundary lies outside the anchor boundary")
        b = self.reference_boundary
        ra, za = (float(v) for v in gs.o_point)
        p, q = b, np.roll(b, -1, axis=0)
        sel = (p[:, 1] - za) * (q[:, 1] - za) <= 0
        rx = []
        for a, c in zip(p[sel], q[sel]):
            if c[1] != a[1]:
                rx.append(a[0] + (za - a[1]) / (c[1] - a[1]) * (c[0] - a[0]))
        rx = np.array(sorted(rx))
        psi_fn, fe = self._psi_direct()
        if rx.size >= 2:
            r_out = self._lcfs_crossing(rx[-1], za, ra, psi_fn)
            r_in = self._lcfs_crossing(rx[0], za, ra, psi_fn)
            out["outboard_midplane"] = 1e3 * (r_out - rx[-1])
            out["inboard_midplane"] = 1e3 * (rx[0] - r_in)
            out["midplane_z_m"] = za
        xs, _div = gs.get_xpoints()
        xs = np.zeros((0, 2)) if xs is None else np.asarray(xs, float).reshape(-1, 2)
        ref_x = self.reference_points.get("x_points")
        x_src = ("reference_points" if ref_x is not None else
                 "none given: solved X-points listed without a displacement")
        out["x_points_solved"] = xs.tolist()
        xrec, used = [], []
        for xr in (ref_x or []):
            xr = np.asarray(xr, float)
            pos = "lower" if xr[1] < za else "upper"
            cand = xs[(xs[:, 1] < za) if pos == "lower" else (xs[:, 1] > za)] if len(xs) else xs
            if not len(cand):
                xrec.append(dict(position=pos, reference=xr.tolist(), solved=None, distance=None,
                                 note="no solved X-point on this side"))
                continue
            k = int(np.argmin(np.hypot(cand[:, 0] - xr[0], cand[:, 1] - xr[1])))
            xrec.append(dict(position=pos, reference=xr.tolist(), solved=cand[k].tolist(),
                             distance=1e3 * float(np.hypot(*(cand[k] - xr))), dR=1e3 * float(cand[k, 0] - xr[0]),
                             dZ=1e3 * float(cand[k, 1] - xr[1])))
            used.append(cand[k])
        out["x_points"] = xrec
        out["x_points_reference"] = x_src
        if ref_x is None:
            # strike points of the solved state's own X-points nearest the plasma (one per side of the axis)
            for pos, sel in (("lower", xs[:, 1] < za), ("upper", xs[:, 1] > za)):
                if len(xs) and np.any(sel):
                    cand = xs[sel]
                    used.append(cand[int(np.argmin(np.abs(cand[:, 1] - za)))])
        sp = self._strike_points(used, psi_fn)
        del fe
        ref_sp = self.reference_points.get("strike_points") or {}
        srec = {}
        for name, xy in sp.items():
            r = dict(solved=list(xy))
            if name in ref_sp:
                rr = np.asarray(ref_sp[name], float)
                r.update(reference=rr.tolist(), distance=1e3 * float(np.hypot(xy[0] - rr[0], xy[1] - rr[1])))
            srec[name] = r
        for name in ref_sp:
            if name not in srec:
                srec[name] = dict(reference=list(ref_sp[name]), solved=None, distance=None,
                                  note="no solved strike point of this name")
        out["strike_points"] = srec
        out["strike_points_reference"] = ("reference_points" if ref_sp else
                                          "none given: solved strike points listed without a displacement")
        return out
