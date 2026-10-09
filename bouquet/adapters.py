"""Source adapters for the unified reconstruction engine (:mod:`bouquet.engine`).

An adapter turns ONE input type into the engine's contract
(:class:`EngineContract`): kinetics, the fixed pressure, the PARALLEL current
components, the boundary, the measurement rows and the sign frame.  It is the
only place (with the exporters) where a current convention is converted:

* the engine stores every current component as the flux-surface-averaged
  parallel current ``<j.B>`` [T A/m^2];
* the solver's variable -- the plain flux-surface average ``<j_phi>`` that
  TokaMaker's ``jphi-linterp`` profile consumes -- is formed in exactly one
  place, the engine's composition, with the field-aligned conversion
  ``<j.B> F<1/R>/<B^2>`` plus the pressure-driven term
  ``p'(<R> - F^2<1/R>/<B^2>)`` (identity (I2) of
  docs/engine.md);
* a g-file carries ``<j_phi>`` (its ``p'`` and ``FF'``) and is converted to
  ``<j.B>`` HERE, with the g-file's own flux surfaces; an IDS carries IMAS
  ``<j.B>/B0`` and is multiplied by ``|B0|`` HERE.

Two adapters:

``GFileAdapter``
    ``<j.B>_in = F p' + F' <B^2>/mu0`` on the g-file's own traced surfaces
    (identity (I0)), minus the Redl bootstrap on the anchor equilibrium, minus
    the fixed driven parts, smoothed with the EXISTING inductive basis of
    :func:`bouquet.TokaMaker_interface.fit_inductive_profile` (smoothing spline
    + PCHIP, zero edge anchor) -- with NO amplitude search.  Rows: Ip (exact);
    l_i(3) = the reader's ``li(2)`` key, HARD, absolute tolerance
    :func:`bouquet.engine.gfile_li_row_tol`; optional q0 (the g-file's own q
    at the measurement radius).
``IdsAdapter``
    the parallel residual ``j_total - j_bootstrap - sum(driven)`` BY
    DEFINITION (IMAS ``<j.B>/B0``), the source's ``j_ohmic`` a stamped
    cross-check (``inductive="j_ohmic"`` / ``"auto"`` use it explicitly);
    every driven ``core_sources`` ``j_parallel`` held fixed.  Rows: Ip
    (soft, the preset's ``sigma_Ip``); ``li_3`` (soft, the preset's ``sigma_li``); optional q0
    (the source's own q at the measurement radius, when the sawtooth gate
    admits it); optional MSE chords (E_r-corrected data only).

Each adapter works in two steps so that everything that does not need a
Grad-Shafranov solve can be read -- and tested -- without one: ``read()``
parses the source; ``finalize(anchor)`` uses the ANCHOR equilibrium (one solve
of the source's own total current, run by the engine) for the Redl bootstrap
the g-file inductive is taken against and for converting user-supplied
toroidal fixed parts.

Synthetic inputs only in the tests; nothing here names a device or a pulse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import coords

#: Version stamp of the contract (recorded with every engine run).
CONTRACT_VERSION = "engine-contract/1"

#: Physical constants the conversions use (the evaluator's own).
_MU0 = 4.0e-7 * np.pi


class EngineInputRefused(ValueError):
    """An adapter refused its input: the engine cannot be given it honestly.

    Raised with the reason (a sign disagreement, a missing current, a raw-E_r
    MSE request, a malformed row) instead of substituting a default."""


@dataclass
class EngineContract:
    """What every source adapter hands the engine (see the module docstring).

    Currents are PARALLEL ``<j.B>`` [T A/m^2] on :attr:`psi_N`, in bouquet's
    positive-current frame; pressure in Pa, densities in m^-3, temperatures in
    eV.  :attr:`rows` holds every measurement row the source can supply; the
    engine settings decide which are active.
    """

    kind: str                              # "gfile" | "ids"
    psi_N: np.ndarray                      # the engine's current grid
    kinetics: dict                         # ne, te, ni, ti, zeff on psi_N
    kinetics_native: dict                  # psi_N + the same on the source grid
    pressure: np.ndarray                   # thermal + impurity + fast [Pa]
    pressure_parts: dict                   # thermal / impurity / fast
    jB_ind: Optional[np.ndarray]           # parallel inductive (after finalize)
    jB_fix: np.ndarray                     # parallel fixed driven (sum)
    jB_fix_parts: dict                     # nbi / rf
    boundary: np.ndarray                   # (N, 2) LCFS points [m]
    Ip: float                              # |Ip| [A] (positive frame)
    rows: dict                             # Ip / l_i / q0 / mse
    signs: dict                            # current_sign, b0_sign, frame
    anchor_request: np.ndarray             # jphi-linterp current of E_0
    provenance: dict = field(default_factory=dict)
    jB_bs_anchor: Optional[np.ndarray] = None   # Redl <j.B> on E_0
    version: str = CONTRACT_VERSION

    def record(self) -> dict:
        """The JSON-safe summary the engine record carries."""
        from .jbs_loop import jsonable
        rows = {}
        for k, v in (self.rows or {}).items():
            if v is None:
                rows[k] = None
            elif k == "mse":
                rows[k] = dict(n_active=int(v["chords"]["n_active"]),
                               er_corrected=bool(v["chords"]["er_corrected"]),
                               er_terms=v.get("er_terms"),
                               orientation=v.get("orientation"),
                               excluded=[[int(i), str(r)] for i, r in
                                         v["chords"].get("excluded", ())])
            else:
                rows[k] = {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, np.ndarray)}
        return jsonable(dict(
            version=self.version, kind=self.kind, n_psi=int(self.psi_N.size),
            Ip=float(self.Ip), rows=rows, signs=dict(self.signs),
            provenance=dict(self.provenance),
            I_ind_parallel_max=(None if self.jB_ind is None
                                else float(np.max(np.abs(self.jB_ind)))),
            jB_fix_max=float(np.max(np.abs(self.jB_fix)))))


# ---------------------------------------------------------------------------
#  validation
# ---------------------------------------------------------------------------
def validate_contract(c: EngineContract, *, final: bool = True) -> None:
    """Refuse a malformed contract (shapes, finiteness, positivity, rows).

    ``final=False`` accepts a contract whose inductive is not formed yet (the
    g-file's ``read()`` output, before the anchor)."""
    who = f"engine contract [{c.kind}]"
    psi = np.asarray(c.psi_N, dtype=float)
    if psi.ndim != 1 or psi.size < 3 or np.any(np.diff(psi) <= 0.0) \
            or not np.all(np.isfinite(psi)):
        raise EngineInputRefused(f"{who}: psi_N must be a strictly increasing "
                                 "finite 1-D grid with >= 3 points")
    n = psi.size

    def _arr(name, a, positive=False, nonneg=False):
        a = np.asarray(a, dtype=float)
        if a.shape != (n,):
            raise EngineInputRefused(f"{who}: {name} has shape {a.shape}, "
                                     f"expected ({n},) on psi_N")
        if not np.all(np.isfinite(a)):
            raise EngineInputRefused(f"{who}: {name} is not finite")
        if positive and not np.all(a > 0.0):
            raise EngineInputRefused(f"{who}: {name} must be strictly "
                                     "positive on every node")
        if nonneg and not np.all(a >= 0.0):
            raise EngineInputRefused(f"{who}: {name} must be >= 0")
        return a

    for k in ("ne", "te", "ni", "ti"):
        _arr(f"kinetics[{k!r}]", c.kinetics[k], positive=True)
    z = _arr("kinetics['zeff']", c.kinetics["zeff"])
    if np.any(z < 1.0):
        raise EngineInputRefused(f"{who}: zeff must be >= 1")
    _arr("pressure", c.pressure, nonneg=True)
    _arr("jB_fix", c.jB_fix)
    _arr("anchor_request", c.anchor_request)
    if final:
        if c.jB_ind is None:
            raise EngineInputRefused(f"{who}: the inductive component was "
                                     "never formed (finalize() not run)")
        _arr("jB_ind", c.jB_ind)
    b = np.asarray(c.boundary, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] < 4 \
            or not np.all(np.isfinite(b)):
        raise EngineInputRefused(f"{who}: boundary must be (N >= 4, 2) finite")
    if not (np.isfinite(c.Ip) and c.Ip > 0.0):
        raise EngineInputRefused(f"{who}: Ip must be positive (the frame is "
                                 f"positive), got {c.Ip!r}")
    rows = c.rows or {}
    if rows.get("Ip") is None:
        raise EngineInputRefused(f"{who}: the Ip row is mandatory")
    li = rows.get("l_i")
    if li is not None:
        if not (np.isfinite(li["target"]) and li["target"] > 0.0):
            raise EngineInputRefused(f"{who}: l_i target must be positive")
        if li["hard"] and not (li.get("tol") and li["tol"] > 0.0):
            raise EngineInputRefused(f"{who}: a hard l_i row needs a tol")
        if not li["hard"] and not (li.get("sigma") and li["sigma"] > 0.0):
            raise EngineInputRefused(f"{who}: a soft l_i row needs a sigma")
    q0 = rows.get("q0")
    if q0 is not None and not (np.isfinite(q0["target"]) and q0["target"] > 0):
        raise EngineInputRefused(f"{who}: q0 target must be positive")


# ---------------------------------------------------------------------------
#  the existing inductive basis, without its amplitude search
# ---------------------------------------------------------------------------
def inductive_basis(psi_N, residual, *, k=3, psi_bridge=0.99,
                    core_exact_psi=0.30):
    """The smoothing of :func:`bouquet.TokaMaker_interface.fit_inductive_profile`
    applied to *residual*, WITHOUT the amplitude (l_i proxy) search.

    A verbatim copy of that function's basis construction on its
    ``shelf_psi_N=0``, ``rescale_j_BS=False`` path (the zero edge anchor at
    psi_N = 1, the smoothing spline with ``s = n var * 0.1``, the ~32-point
    subsample with the raw-residual core window below *core_exact_psi*, the
    PCHIP and the ``>= 0`` floors), so the engine's g-file inductive is
    ``basis`` where the legacy's is ``ind_scale * basis``.  Kept as a copy
    rather than a refactor so the legacy function is untouched; a fast test
    asserts the two agree to rounding.
    """
    from scipy.interpolate import PchipInterpolator, UnivariateSpline
    psi_N = np.asarray(psi_N, dtype=float)
    residual = np.asarray(residual, dtype=float)
    mask_core = psi_N <= psi_bridge
    psi_trusted = np.concatenate([psi_N[mask_core], [1.0]])
    res_trusted = np.concatenate([residual[mask_core], [0.0]])
    _s_factor = len(psi_trusted) * np.var(res_trusted) * 0.1
    _smooth_spline = UnivariateSpline(psi_trusted, res_trusted, k=k,
                                      s=_s_factor)
    _n_sub = min(32, len(psi_N))
    _psi_sub = np.linspace(psi_N[0], psi_N[-1], _n_sub)
    if core_exact_psi and core_exact_psi > 0.0:
        _psi_core = np.linspace(psi_N[0], core_exact_psi, 16)
        _psi_sub = np.unique(np.concatenate([_psi_core, _psi_sub]))
        _raw_sub = np.interp(_psi_sub, psi_trusted, res_trusted)
        _spl_sub = _smooth_spline(_psi_sub)
        _w_core = np.clip((core_exact_psi - _psi_sub)
                          / (0.35 * core_exact_psi), 0.0, 1.0)
        _res_sub = _w_core * _raw_sub + (1.0 - _w_core) * _spl_sub
    else:
        _res_sub = _smooth_spline(_psi_sub)
    _res_sub = np.maximum(_res_sub, 0.0)
    _pchip = PchipInterpolator(_psi_sub, _res_sub)
    basis = _pchip(psi_N)
    return np.maximum(basis, 0.0)


# ---------------------------------------------------------------------------
#  g-file: <j.B> from the g-file's own surfaces (identity I0)
# ---------------------------------------------------------------------------
def gfile_parallel_current(eqdsk):
    """``(jB_in, parts)``: the g-file's ``<j.B>`` in the positive frame.

    ``<j.B> = F p' + F' <B^2>/mu0`` (identity (I0)) with ``p'``, ``FF'``,
    ``F`` and ``<B^2>`` from the reader's OWN traced flux surfaces
    (``eqdsk.averages``: ``PPRIME``, ``FFPRIM``, ``F``, ``Btot**2``) and the
    COCOS factor the reader applies to its own ``<j_phi>``
    (``Jt_GS = -sigma_Bp (p'<R> + FF'<1/R>/mu0) (2 pi)^exp_Bp``), so that
    ``p'_+ <R> + FF'_+ <1/R>/mu0`` reproduces ``sgn * Jt_GS`` exactly.  The
    positive frame is ``sgn = sign(Ip)``; the input is REFUSED when its
    ``<j_phi>`` does not carry the sign of its own Ip in the bulk.

    ``parts`` carries ``F``, ``pprime``, ``ffprime`` (positive frame),
    ``R_avg``, ``inv_R``, ``B2``, ``jphi_in`` (= ``sgn * Jt_GS``) and the
    identity residual of the frame factor.
    """
    from .io.geqdsk import _cocos_params
    avg = eqdsk.averages
    cc = _cocos_params(eqdsk.cocos)
    fac = -float(cc["sigma_Bp"]) * (2.0 * np.pi) ** float(cc["exp_Bp"])
    Ip = float(eqdsk.Ip)
    if not (np.isfinite(Ip) and Ip != 0.0):
        raise EngineInputRefused(f"g-file adapter: unusable Ip {Ip!r}")
    sgn = 1.0 if Ip > 0.0 else -1.0
    jt = np.asarray(avg["Jt_GS"], dtype=float)
    if not np.all(np.isfinite(jt)):
        raise EngineInputRefused("g-file adapter: the g-file's <j_phi> is not "
                                 "finite")
    if float(np.median(sgn * jt)) <= 0.0:
        raise EngineInputRefused(
            "g-file adapter: the g-file's flux-surface current <j_phi> does "
            "not carry the sign of its own Ip in the bulk (median of "
            f"sign(Ip) * <j_phi> = {float(np.median(sgn * jt)):.4g}); check "
            "the COCOS the file was read with")
    F = np.abs(np.asarray(avg["F"], dtype=float))
    pp = sgn * fac * np.asarray(avg["PPRIME"], dtype=float)
    ffp = sgn * fac * np.asarray(avg["FFPRIM"], dtype=float)
    R_avg = np.asarray(avg["R"], dtype=float)
    inv_R = np.asarray(avg["1/R"], dtype=float)
    B2 = np.asarray(avg["Btot**2"], dtype=float)
    for nm, a in (("F", F), ("<B^2>", B2), ("<1/R>", inv_R), ("<R>", R_avg)):
        if not (np.all(np.isfinite(a)) and np.all(a > 0.0)):
            raise EngineInputRefused(f"g-file adapter: the g-file surface "
                                     f"average {nm} is not finite and "
                                     "positive on every surface")
    jphi_in = sgn * jt
    ident = pp * R_avg + ffp * inv_R / _MU0 - jphi_in
    scale = float(np.max(np.abs(jphi_in))) or 1.0
    jB = F * pp + (ffp / F) * B2 / _MU0
    return jB, dict(F=F, pprime=pp, ffprime=ffp, R_avg=R_avg, inv_R=inv_R,
                    B2=B2, jphi_in=jphi_in, sign=sgn,
                    frame_identity_max_rel=float(np.max(np.abs(ident)) / scale))


def _gfile_geometry_from_parts(parts):
    """The composition geometry (:func:`bouquet.engine.compose`) built from a
    g-file's own surface averages."""
    return dict(F=parts["F"], R_avg=parts["R_avg"], inv_R=parts["inv_R"],
                B2=parts["B2"], pprime=parts["pprime"])


def mse_rows(gc, signs=None):
    """The MSE row block of a :class:`GenerationConfig`, or ``None``.

    The engine takes E_r-CORRECTED pitch angles only (decision 19): the block
    must say ``er_corrected=True`` and carry no ``Er``; a raw-E_r modelling
    request is REFUSED (:class:`EngineInputRefused`), never modelled.

    **Orientation is stated, never fitted** (:mod:`bouquet.mse`): the block's
    ``ip_sign`` / ``bt_sign`` are the directions of Ip and B_t in the
    right-handed ``(R, phi, Z)`` frame of the A-coefficients.  *signs* is the
    SOURCE's declared orientation in that frame (``dict(ip_sign, bt_sign,
    basis)``, from the adapter).  A block that states neither is completed
    from the source; a block that states one the source contradicts is
    REFUSED -- two statements of one discharge's orientation cannot both
    hold, and choosing one silently would hide which.  A source that cannot
    state a sign it is asked for is refused too."""
    md = getattr(gc, "mse_data", None)
    if md is None:
        return None
    from .mse import (MSE_ORIENTATION_KEYS, MSEDataUnusable, mse_chords,
                      mse_er_terms)
    md = dict(md)
    sg = dict(signs or {})
    filled = []
    for key in MSE_ORIENTATION_KEYS:
        src = sg.get(key)
        if key in md and md[key] is not None:
            if src is not None and float(md[key]) != float(src):
                raise EngineInputRefused(
                    f"engine MSE rows: mse_data[{key!r}] = {md[key]!r} but "
                    f"the source declares {float(src):+.0f} "
                    f"({sg.get('basis', 'source orientation')}); the "
                    "discharge's orientation is stated twice and the two "
                    "disagree -- refusing to choose one")
        else:
            if src is None:
                raise EngineInputRefused(
                    f"engine MSE rows: mse_data carries no {key!r} and the "
                    "source does not declare it either "
                    f"({sg.get('basis', 'no source orientation')}); state it "
                    "in the block (+1 or -1)")
            md[key] = float(src)
            filled.append(key)
    try:
        ch = mse_chords(md, min_chords=int(gc.structured_mse_min_chords),
                        sigma_sys=float(gc.structured_mse_sigma_sys))
    except MSEDataUnusable as e:
        raise EngineInputRefused(f"engine MSE rows: {e}") from e
    if not ch["er_corrected"] or ch["er_applied"]:
        raise EngineInputRefused(
            "engine MSE rows: the unified engine accepts only E_r-CORRECTED "
            "tan(gamma) (mse_data['er_corrected']=True, no 'Er'); a raw-E_r "
            "modelling request (A5*Er in the forward model) is refused -- "
            "correct the pitch angles upstream")
    return dict(chords=ch, er_terms=mse_er_terms(ch),
                orientation=dict(ip_sign=float(ch["ip_sign"]),
                                 bt_sign=float(ch["bt_sign"]),
                                 filled_from_source=filled,
                                 source_basis=sg.get("basis")),
                required=bool(getattr(gc, "structured_mse_required", False)))


def _q0_psi(psi_N, psi_pad):
    """The radius the q0 row is measured at: the psi_pad-clipped axis sample
    (the closure's axis-row radius, :func:`bouquet.utils.fsa_current_geometry`)."""
    return float(np.clip(float(np.asarray(psi_N, float)[0]), float(psi_pad),
                         1.0 - float(psi_pad)))


class GFileAdapter:
    """The g-file (+ p-file / IDA profiles) adapter."""

    kind = "gfile"

    def __init__(self, source, config, *, li_row_tol=None):
        self.source = source
        self.config = config
        self.li_row_tol = li_row_tol
        self.eqdsk = None
        self._c = None

    def read(self) -> EngineContract:
        """Everything that needs no solve: the contract minus the inductive."""
        from .baseline import _load_kinetic_profiles, _resolve_fixed
        from .io.geqdsk import read_geqdsk
        from .physics import ELEMENTARY_CHARGE as _EC, impurity_pressure
        from .utils import pchip_interp
        src, cfg = self.source, self.config
        if src.profile_overrides:
            raise NotImplementedError("profile_overrides is not yet applied")
        eqdsk = read_geqdsk(src.geqdsk_path, cocos=src.cocos)
        self.eqdsk = eqdsk
        psi_N = np.asarray(eqdsk.psi_N, dtype=float)
        kin = _load_kinetic_profiles(src)
        # run grids, as the legacy reconstruction builds them
        # (coords.gfile_run_grids); psi_N keeps the g-file's psi_N (the
        # inductive basis and the q-row radius are psi_N constructs)
        self.coord = coords.run_coord(getattr(src, "coord", coords.PSI))
        self._psi_g = psi_N
        try:
            x_run, psi_kin, _in, kin = coords.gfile_run_grids(
                eqdsk, kin, src.profiles_path, self.coord)
        except ValueError as exc:
            raise EngineInputRefused(f"g-file adapter: {exc}") from exc
        psi_kin = np.asarray(psi_kin, dtype=float)
        psi_map = None if _in is None else (psi_N, x_run)

        def to_eq(a):
            return pchip_interp(psi_kin, a, x_run)

        ne, te, ni, ti = (to_eq(kin[k]) for k in ("ne", "te", "ni", "ti"))
        zeff = np.clip(to_eq(kin["Zeff"]), 1.0, None)
        fc = cfg.fixed_components
        fc_x = coords.to_run_grid(fc.psi_N, getattr(fc, "coord", coords.RUN),
                                  psi_map)
        p_fast_kin = _resolve_fixed(fc.p_fast, fc_x, psi_kin)
        p_fast = (to_eq(p_fast_kin) if fc.p_fast is not None
                  else np.zeros_like(psi_N))
        Z_imp = getattr(fc, "Z_imp", None)
        p_th = _EC * (ne * te + ni * ti)
        p_imp = (impurity_pressure(ne, ni, ti, Z_imp) if Z_imp
                 else np.zeros_like(psi_N))
        jB_in, parts = gfile_parallel_current(eqdsk)
        # the discharge's orientation in the right-handed (R, phi, Z) frame
        # of the MSE A-coefficients: the file's Ip and B_t signs, carried by
        # its COCOS's sigma_RpZ (-1: (R, Z, phi) is right-handed, phi flips)
        from .io.geqdsk import _cocos_params
        _srpz = float(_cocos_params(eqdsk.cocos)["sigma_RpZ"])
        _bc = float(eqdsk.B_center)
        self.orientation = dict(
            ip_sign=_srpz * float(parts["sign"]),
            bt_sign=(None if not (np.isfinite(_bc) and _bc != 0.0)
                     else _srpz * (1.0 if _bc > 0.0 else -1.0)),
            basis=(f"g-file CURRENT and BCENTR signs in its COCOS "
                   f"{int(eqdsk.cocos)} (sigma_RpZ {_srpz:+.0f})"))
        # user fixed parts are TOROIDAL (FixedComponentsConfig); converted to
        # parallel in finalize() with the anchor geometry
        self._fix_tor = dict(nbi=_resolve_fixed(fc.j_NBI, fc_x, x_run),
                             rf=_resolve_fixed(fc.j_RF, fc_x, x_run))
        psi_pad = float(src.psi_pad)
        from .engine import gfile_li_row_tol
        tol = (gfile_li_row_tol() if self.li_row_tol is None
               else float(self.li_row_tol))
        li_in = float(eqdsk.li.get("li(2)", float("nan")))
        q_psi = _q0_psi(psi_N, psi_pad)
        qpsi = np.abs(np.asarray(eqdsk.qpsi, dtype=float))
        q_t = float(np.interp(q_psi, psi_N, qpsi))
        from .utils import q0_gate_admits
        adm, basis = q0_gate_admits(False, float(qpsi[0]), q_t,
                                    float(getattr(cfg.generation, "q0_gate",
                                                  1.1)))
        rows = dict(
            Ip=dict(target=abs(float(eqdsk.Ip)), sigma=None, hard=True,
                    source="g-file CURRENT"),
            l_i=dict(target=li_in, kind="li_3", hard=True, tol=float(tol),
                     sigma=None, tol_origin=(
                         "bouquet.TokaMaker_interface._rematch_li_request "
                         "li_tol default (the step-5 / re-match secant's "
                         "absolute l_i tolerance)"),
                     source="g-file li(2) key (the li(3)/'iter' functional)"),
            q0=dict(target=q_t, psi=q_psi, admitted=bool(adm),
                    gate_basis=basis, q0_source_axis=float(qpsi[0]),
                    source=("g-file qpsi interpolated at the measurement "
                            "radius (like radii)")),
            mse=mse_rows(cfg.generation, self.orientation),
        )
        c = EngineContract(
            kind="gfile", psi_N=x_run,
            kinetics=dict(ne=ne, te=te, ni=ni, ti=ti, zeff=zeff),
            kinetics_native=dict(psi_N=psi_kin, ne=kin["ne"], te=kin["te"],
                                 ni=kin["ni"], ti=kin["ti"], Zeff=kin["Zeff"],
                                 raw_bytes=kin.get("raw_bytes"),
                                 p_fast=p_fast_kin,
                                 zeff_provenance=kin.get("zeff_provenance")),
            pressure=p_th + p_imp + p_fast,
            pressure_parts=dict(thermal=p_th, impurity=p_imp, fast=p_fast,
                                Z_imp=Z_imp),
            jB_ind=None, jB_fix=np.zeros_like(psi_N),
            jB_fix_parts=dict(nbi=np.zeros_like(psi_N),
                              rf=np.zeros_like(psi_N)),
            boundary=np.column_stack([eqdsk.boundary_R, eqdsk.boundary_Z]),
            Ip=abs(float(eqdsk.Ip)), rows=rows,
            signs=dict(current_sign=float(parts["sign"]),
                       b0_sign=self.orientation["bt_sign"],
                       ip_sign_RphiZ=self.orientation["ip_sign"],
                       bt_sign_RphiZ=self.orientation["bt_sign"],
                       orientation_basis=self.orientation["basis"],
                       frame="positive: |Ip|, F = |R B_phi|, <j.B> > 0 "
                             "co-current"),
            anchor_request=np.abs(np.asarray(eqdsk.j_tor_averaged_direct,
                                             dtype=float)),
            provenance=dict(
                inductive=("<j.B>_in (I0, g-file surfaces) - Redl <j.B> on "
                           "the anchor - fixed parts, smoothed with the "
                           "fit_inductive_profile basis (no amplitude search)"),
                frame_identity_max_rel=parts["frame_identity_max_rel"],
                kinetics="p-file / IDA profiles PCHIP-regridded onto the "
                         "g-file psi_N (as the legacy reconstruction)",
                kinetics_sigma=("resolved from the Baseline by "
                                "baseline.resolve_uncertainty (unchanged)"),
                electron_charge="physics.ELEMENTARY_CHARGE",
                anchor=("one solve of |eqdsk.j_tor_averaged_direct| at the "
                        "full pressure (the legacy anchor request)")),
        )
        self._jB_in = jB_in
        self._parts = parts
        validate_contract(c, final=False)
        self._c = c
        return c

    def finalize(self, jB_bs_anchor, anchor_geom) -> EngineContract:
        """Subtract Redl on the anchor and the fixed parts; smooth."""
        from .engine import conversion_factor
        c = self._c
        if c is None:
            raise RuntimeError("GFileAdapter.finalize before read()")
        kap = conversion_factor(anchor_geom)
        jB_nbi = np.asarray(self._fix_tor["nbi"], float) / kap
        jB_rf = np.asarray(self._fix_tor["rf"], float) / kap
        jB_fix = jB_nbi + jB_rf
        resid = self._jB_in - np.asarray(jB_bs_anchor, float) - jB_fix
        src = self.source
        # on the g-file's psi_N (its bridge and core thresholds are psi_N)
        jB_ind = inductive_basis(self._psi_g, resid, k=int(src.n_k),
                                 psi_bridge=float(src.psi_bridge))
        c.jB_ind = jB_ind
        c.jB_fix = jB_fix
        c.jB_fix_parts = dict(nbi=jB_nbi, rf=jB_rf)
        c.jB_bs_anchor = np.asarray(jB_bs_anchor, float).copy()
        c.provenance.update(
            residual_negative_nodes=int(np.sum(resid < 0.0)),
            smoothing=dict(k=int(src.n_k), psi_bridge=float(src.psi_bridge),
                           core_exact_psi=0.30),
            inductive_vs_residual_rms_rel=float(
                np.sqrt(np.mean((jB_ind - resid) ** 2))
                / (float(np.max(np.abs(self._jB_in))) or 1.0)))
        validate_contract(c)
        return c


# ---------------------------------------------------------------------------
#  IDS
# ---------------------------------------------------------------------------
#: How the IDS adapter forms the inductive component.  "residual" (the
#: default, owner decision 2026-10-02): the parallel residual
#: ``j_total - j_bootstrap - sum(driven)`` BY DEFINITION, with the source's
#: ``j_ohmic`` compared against it and stamped as a cross-check (no
#: threshold, no warning).  "j_ohmic": the source's, warning above the
#: tolerance below.  "auto": ``j_ohmic`` unless the split misses by more
#: than that tolerance, then the residual with a warning.
IDS_INDUCTIVE_CHOICES = ("auto", "j_ohmic", "residual")

#: ``GenerationConfig.imas_li3_radius`` values (the IDS l_i row target's
#: normalisation radius; :func:`resolve_li3_radius`).
IMAS_LI3_RADIUS_CHOICES = ("auto", "geometric", "axis")
#: ``"auto"``: the relative agreement within which li_3 recomputed from the
#: source's own equilibrium with a radius must reproduce the stored li_3
#: for that radius to be taken (0.5 %; the two candidate radii differ by
#: ~3 % on these plasmas; owner-approved 2026-10-07).
LI3_RADIUS_MATCH_TOL = 0.005

_MU0_LI = 4.0e-7 * np.pi


class Li3RadiusRefused(EngineInputRefused):
    """The stored li_3 matches neither normalisation radius."""


def _li3_geometry(ts):
    """``(R_axis, R_geo, R_geo_source, Bp2v, Ip, missing)`` of one IDS
    equilibrium time slice *ts* (a dict): the magnetic axis, the geometric
    radius ``(R_out + R_in)/2`` of the boundary (``profiles_1d`` r_outboard
    / r_inboard at the last surface, else ``boundary.geometric_axis.r``,
    else the outline's extremes), and ``int B_p^2 dV`` from the source's own
    flux-surface averages (COCOS 11: ``B_p = |grad psi| / (2 pi R)``, so
    ``<B_p^2> = (dpsi/drho_tor)^2 gm2 / (2 pi)^2`` with ``gm2 =
    <|grad rho_tor|^2 / R^2>``, integrated with ``dvolume_dpsi`` -- or the
    gradient of ``volume`` -- over psi).  ``missing`` names what could not
    be read (``None`` entries then)."""
    gq = ts.get("global_quantities", {}) or {}
    p1 = ts.get("profiles_1d", {}) or {}
    bd = ts.get("boundary", {}) or {}
    missing = []

    def _arr(name):
        v = p1.get(name)
        if v is None:
            return None
        a = np.asarray(v, dtype=float)
        return a if (a.ndim == 1 and a.size >= 2
                     and np.all(np.isfinite(a))) else None

    R_axis = None
    ma = gq.get("magnetic_axis") or {}
    if ma.get("r") is not None and np.isfinite(float(ma["r"])):
        R_axis = float(ma["r"])
    else:
        missing.append("global_quantities.magnetic_axis.r")
    R_geo, geo_src = None, None
    ro, ri = _arr("r_outboard"), _arr("r_inboard")
    if ro is not None and ri is not None:
        R_geo = 0.5 * (float(ro[-1]) + float(ri[-1]))
        geo_src = "profiles_1d r_outboard/r_inboard at the last surface"
    elif (bd.get("geometric_axis") or {}).get("r") is not None:
        R_geo = float(bd["geometric_axis"]["r"])
        geo_src = "boundary.geometric_axis.r"
    elif (bd.get("outline") or {}).get("r") is not None:
        r = np.asarray(bd["outline"]["r"], dtype=float)
        R_geo = 0.5 * (float(np.max(r)) + float(np.min(r)))
        geo_src = "boundary.outline (max R + min R) / 2"
    else:
        missing.append("a boundary radius (profiles_1d r_outboard/"
                       "r_inboard, boundary.geometric_axis, outline)")
    psi = _arr("psi")
    gm2, dpdr = _arr("gm2"), _arr("dpsi_drho_tor")
    dv = _arr("dvolume_dpsi")
    if dv is None and psi is not None and _arr("volume") is not None:
        dv = np.gradient(_arr("volume"), psi)
    Bp2v = None
    if psi is None or gm2 is None or dpdr is None or dv is None:
        missing.extend("profiles_1d " + n for n, a in (
            ("psi", psi), ("gm2", gm2), ("dpsi_drho_tor", dpdr),
            ("dvolume_dpsi (or volume)", dv)) if a is None)
    else:
        from scipy.integrate import trapezoid
        bp2 = dpdr ** 2 * gm2 / (2.0 * np.pi) ** 2
        Bp2v = abs(float(trapezoid(bp2 * dv, psi)))
    Ip = gq.get("ip")
    Ip = None if Ip is None else abs(float(Ip))
    if not Ip:
        missing.append("global_quantities.ip")
        Ip = None
    return R_axis, R_geo, geo_src, Bp2v, Ip, missing


def resolve_li3_radius(ts, setting="auto", tol=LI3_RADIUS_MATCH_TOL):
    """The normalisation radius of the source's stored li_3 and the factor
    that puts it at the measurement's radius (the magnetic axis).

    Returns a record: ``setting``, ``choice`` (``"axis"`` /
    ``"geometric"``), ``status`` (``"matched"`` -- auto found it;
    ``"stated"`` -- the setting named it; ``"undetermined"`` -- auto could
    not recompute li_3 from this source, the target is used UNRESCALED, as
    before the setting existed, printed and recorded), the stored
    ``li3_source``, the recomputed ``li3_axis`` / ``li3_geometric`` and the
    ratios ``stored / recomputed`` (``None`` when not computable), the
    radii, ``factor = R_src / R_axis`` and the rescaled ``target``.
    ``"auto"`` with NEITHER ratio within *tol* of 1 is REFUSED
    (:class:`Li3RadiusRefused`) naming both -- never silently rescaled;
    ``"geometric"`` without both radii is refused."""
    if setting not in IMAS_LI3_RADIUS_CHOICES:
        raise ValueError(f"imas_li3_radius must be one of "
                         f"{IMAS_LI3_RADIUS_CHOICES}, got {setting!r}")
    gq = ts.get("global_quantities", {}) or {}
    li_src = gq.get("li_3")
    li_src = None if li_src is None else float(li_src)
    R_axis, R_geo, geo_src, Bp2v, Ip, missing = _li3_geometry(ts)
    rec = dict(setting=str(setting), li3_source=li_src, R_axis=R_axis,
               R_geo=R_geo, R_geo_source=geo_src, Bp2v=Bp2v,
               match_tol=float(tol), missing=list(missing),
               li3_axis=None, li3_geometric=None, ratio_axis=None,
               ratio_geometric=None,
               definition=("li_3(R) = 2 int B_p^2 dV / ((mu0 Ip)^2 R) from "
                           "the source's own flux-surface averages; ratio = "
                           "stored / recomputed; the measurement "
                           "(utils.li_achieved) normalises by R_axis"))
    if Bp2v is not None and Ip:
        for name, R in (("axis", R_axis), ("geometric", R_geo)):
            if R:
                li = 2.0 * Bp2v / ((_MU0_LI * Ip) ** 2 * R)
                rec["li3_" + name] = float(li)
                if li_src is not None and li > 0.0:
                    rec["ratio_" + name] = float(li_src / li)

    def _done(choice, status):
        R_src = R_axis if choice == "axis" else R_geo
        fac = (1.0 if choice == "axis" else float(R_src) / float(R_axis))
        rec.update(choice=choice, status=status, R_source=R_src,
                   factor=float(fac),
                   target=(None if li_src is None else li_src * fac))
        return rec

    if setting == "axis":
        return _done("axis", "stated")
    if setting == "geometric":
        if R_geo is None or R_axis is None:
            raise Li3RadiusRefused(
                "IDS adapter: imas_li3_radius='geometric' needs both the "
                "boundary's geometric radius and the magnetic axis to "
                "rescale li_3 to the measurement's radius; missing: "
                + "; ".join(missing))
        return _done("geometric", "stated")
    if li_src is None:
        return _done("axis", "no li_3 in the source (no l_i row)")
    ra, rg = rec["ratio_axis"], rec["ratio_geometric"]
    if ra is None or rg is None:
        msg = ("IDS adapter: imas_li3_radius='auto' cannot tell the radius "
               "the source's li_3 was normalised with (li_3 cannot be "
               "recomputed from this source: missing " + "; ".join(missing)
               + "); the l_i row's target is the stored li_3 UNRESCALED, as "
               "before the setting existed -- state imas_li3_radius="
               "'axis' or 'geometric' if known")
        # printed and recorded (status "undetermined"), not raised as a
        # warning: nothing changes against the code before the setting
        print("[engine] NOTE " + msg, flush=True)
        return _done("axis", "undetermined")
    ok = {n: abs(r - 1.0) for n, r in (("axis", ra), ("geometric", rg))
          if abs(r - 1.0) <= tol}
    if not ok:
        raise Li3RadiusRefused(
            "IDS adapter: the source's li_3 = %.6g matches neither "
            "normalisation radius within %.2g %%: recomputed from its own "
            "equilibrium, li_3(R_axis = %.4f m) = %.6g (stored/recomputed "
            "%.5f) and li_3(R_geo = %.4f m, %s) = %.6g (stored/recomputed "
            "%.5f); state imas_li3_radius='axis' or 'geometric' if the "
            "convention is known -- the target is never rescaled silently"
            % (li_src, 100.0 * tol, R_axis, rec["li3_axis"], ra, R_geo,
               geo_src, rec["li3_geometric"], rg))
    return _done(min(ok, key=ok.get), "matched")
#: The default of :class:`IdsAdapter` and of
#: ``GenerationConfig.engine_ids_inductive``.  Evidence (validation, all
#: slices of two IDS source families): on self-consistent sources residual
#: and ``j_ohmic`` are indistinguishable (|dl_i| <= 1.8e-3); where the
#: source's own split is locally inconsistent the residual is closer to the
#: source's <j_phi>.
IDS_INDUCTIVE_DEFAULT = "residual"
#: Net mismatch ``j_ohmic - (j_total - j_bootstrap - sum(driven))`` as a
#: fraction of the total current above which ``inductive="auto"`` takes the
#: residual instead of ``j_ohmic`` and ``inductive="j_ohmic"`` warns (the
#: number is stamped either way; it plays no role under "residual").
#: Owner-set 2026-10-02: a source whose current split is self-consistent
#: agrees to <= 0.8 %; the one seen inconsistent (a slice on a sawtooth
#: step of the source's own model) missed by 7 %.
IDS_INDUCTIVE_MISMATCH_TOL = 0.02
#: IMAS ``core_sources`` identifier indices and how the engine's IDS adapter
#: treats an entry's ``j_parallel``.  The table is the IMAS data dictionary's
#: ``core_sources.source[:].identifier`` enumeration (verified against the
#: data-dictionary structures shipped with omas, DD 3.13-3.41 and develop_3,
#: and the IMASdd identifier table used by FUSE, which adds
#: 409 time_derivative and 701 sawteeth).  Never "everything except ohmic and
#: bootstrap": an AGGREGATE entry (a total, or a combination of other
#: entries) would be counted twice.
IDS_OHMIC_SOURCE_INDEX = 7               # j_ohmic (core_profiles) instead
IDS_BOOTSTRAP_SOURCE_INDEX = 13          # j_bootstrap (core_profiles) instead
IDS_RF_SOURCE_INDICES = (3, 4, 5)       # ec, lh, ic
#: DRIVEN primary entries, held fixed, by part: nbi (2) -> "nbi"; ec / lh /
#: ic (3, 4, 5) -> "rf"; fusion-alpha current drive (6), runaways (501) and
#: a model's sawtooth redistribution entry (701 sawteeth, FUSE) -> "other".
IDS_DRIVEN_SOURCE_PARTS = {2: "nbi", 3: "rf", 4: "rf", 5: "rf",
                           6: "other", 501: "other", 701: "other"}
#: AGGREGATE entries -- a total or a combination of other entries -- NEVER
#: added (they would double-count their constituents): 1 total,
#: 100 auxiliary (all auxiliary H&CD), 101-107 the ic/nbi/fusion/ec/lh
#: combinations, 200 radiation (total), 202 cyclotron_synchrotron
#: (201 + 9), 203 impurity_radiation (line + bremsstrahlung).  An ignored
#: one carrying a non-zero j_parallel is stamped (provenance
#: "ignored_sources") and warned about.
IDS_AGGREGATE_SOURCE_INDICES = (1, 100, 101, 102, 103, 104, 105, 106, 107,
                                200, 202, 203)
#: Entries describing the BOOTSTRAP (or ohmic) current under another name:
#: the engine's bootstrap is Redl / the source's core_profiles j_bootstrap,
#: so holding one fixed would double-count it.  401 "neoclassical" is
#: where a source may publish its bootstrap.  Ignored like an aggregate
#: (stamped and warned when non-zero).
IDS_BOOTSTRAP_LIKE_SOURCE_INDICES = (401,)
def _ids_source_slice(s, isrc, t_slice, n_time, base_times=None, *,
                      t_cp=None, cp_half=None, rec=None, who="IDS adapter"):
    """``(profile, how)``: the entry's ``profiles_1d`` at the core_sources
    slice *isrc* (time *t_slice*), or ``(None, why)`` when the entry has no
    slice within HALF a local time-step of that time -- the rule of
    :func:`bouquet.io.imas._source_slice_at`, which this calls (one rule
    for both paths): the entry's own half-step (*base_times*, the
    core_profiles time base, for a single-time entry) AND, owner decision
    2026-10-06, within *cp_half* (half the local core_profiles step) of the
    core_profiles slice time *t_cp*.  An entry carrying its own per-slice
    times is matched BY TIME to its nearest own slice; one without them
    must have exactly the IDS's number of slices, or it cannot be aligned
    and is refused -- never the first slice taken in place of a missing
    one.  *rec* receives the match record (dt, windows, bracketing own
    times, status).  A refusal names *who* called."""
    from .io.imas import _source_slice_at
    try:
        return _source_slice_at(s, isrc, t_slice, n_time, base_times,
                                t_cp=t_cp, cp_half=cp_half, rec=rec)
    except ValueError as exc:
        raise EngineInputRefused(
            str(exc).replace("IMAS reader:", f"{who}:", 1)) from None


#: (announce_key, kind, name, index) of every unknown / aggregate-entry
#: warning already issued: once per source file (review PR70 B8 -- the
#: legacy reader and the engine adapter run the same rule on the same file)
_DRIVEN_WARNED = set()


def _warn_driven_once(announce_key, tag, msg, stacklevel=4):
    """Warn *msg* once per *announce_key* and *tag* (every call when the
    key is None)."""
    import warnings
    if announce_key is not None:
        key = (announce_key,) + tuple(tag)
        if key in _DRIVEN_WARNED:
            return
        _DRIVEN_WARNED.add(key)
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


def _ids_driven_currents(srcs, isrc, n, sgn, base_times=None, off=None, *,
                         t_cp=None, cp_half=None, matches=None,
                         announce_key=None, who="IDS adapter"):
    """The driven ``j_parallel`` of the ``core_sources`` slice *isrc*, by
    the explicit identifier classification above, split into ``nbi`` /
    ``rf`` / ``other`` (positive frame).  Returns ``(parts, used, ignored)``:
    the contributing entries, and every entry with a non-zero
    ``j_parallel`` that was NOT added (aggregate or bootstrap-like) with its
    reason.  An UNKNOWN index carrying a non-zero ``j_parallel`` is held
    fixed under "other" with a warning (its nature cannot be told from the
    enumeration).  A DRIVEN (or unknown) entry carrying a non-zero
    ``j_parallel`` with no slice within half a local time-step of the slice
    time is REFUSED (:class:`EngineInputRefused`; owner-approved 2026-10-06
    -- before, it was dropped to zero and stamped); an aggregate or
    bootstrap-like one, never added, is stamped as before.

    Refinement (2026-10-06): "carries current" is judged on the entry's own
    slices BRACKETING the slice time
    (:func:`bouquet.io.imas._entry_bracketing_slices`), not its whole time
    history.  A driven (or unknown) entry with no slice in the window and
    no current on its bracketing slices is OFF at that time, not missing:
    it contributes zero, is not warned about, and its record (name, index,
    reason) is appended to the list *off* when one is given (provenance
    "off_sources").  An entry identically zero over its whole history is
    skipped silently, as before.

    Owner decision 2026-10-06: the matched own slice must also lie within
    *cp_half* (half the local core_profiles step) of the core_profiles
    slice time *t_cp* (default: the core_sources slice time; with no
    *cp_half* given, half the local step of *base_times* there, when it
    has one); a driven (or unknown) entry whose own record begins AFTER
    the slice time is OFF there -- zero, appended to *off* with
    ``reason="off_before_record"`` and its ``first_own_time``, and
    announced (print + warning) once per *announce_key* and entry; past its
    last own time it is still refused when that slice carries current.
    *matches*, a list, receives every entry's match record (dt, windows,
    bracketing own times, status).  *who* names the caller in every
    announcement and refusal ("IDS adapter", "IMAS reader"); the
    unknown-index and not-added warnings are issued once per *announce_key*
    (the source file) and entry, whichever caller reads it first."""
    from .io.imas import (_announce_off_before, _entry_off_before_record,
                          _half_local_step)
    parts = {k: np.zeros(n) for k in ("nbi", "rf", "other")}
    used, ignored = [], []
    tb = srcs.get("time")
    n_time = None if not tb else len(tb)
    t_slice = (None if not tb else float(np.asarray(tb, dtype=float)[isrc]))
    if t_cp is None:
        t_cp = t_slice
    if cp_half is None and t_cp is not None and t_slice is not None:
        cp_half = _half_local_step(base_times, t_cp, t_slice)
    for s in srcs.get("source", []):
        idn = s.get("identifier", {}) or {}
        idx = idn.get("index")
        if idx in (IDS_OHMIC_SOURCE_INDEX, IDS_BOOTSTRAP_SOURCE_INDEX):
            continue
        erec = {}
        if matches is not None:
            matches.append(erec)
        q, how = _ids_source_slice(s, isrc, t_slice, n_time, base_times,
                                   t_cp=t_cp, cp_half=cp_half, rec=erec,
                                   who=who)
        if q is None:
            from .io.imas import (_carries_current, _entry_off_near,
                                  _entry_time_refusal)
            erec["reason"] = how
            if any(_carries_current(qq) for qq in s.get("profiles_1d", [])):
                if idx in IDS_AGGREGATE_SOURCE_INDICES or \
                        idx in IDS_BOOTSTRAP_LIKE_SOURCE_INDICES:
                    # never added anyway: stamped, as at a matched time
                    erec["status"] = "ignored"
                    ignored.append(dict(name=idn.get("name"), index=idx,
                                        reason=how))
                    continue
                why_off = _entry_off_near(s, t_slice)
                if why_off is not None:
                    # idle on its slices bracketing this time: off here,
                    # not missing -- contributes zero, stamped
                    erec["status"] = "off_idle"
                    if off is not None:
                        off.append(dict(name=idn.get("name"), index=idx,
                                        reason=why_off))
                    continue
                first = _entry_off_before_record(s, t_slice)
                if first is not None:
                    # no record before its first own time (which carries
                    # current): OFF here (owner decision 2026-10-06),
                    # stamped and announced once
                    erec.update(status="off_before_record",
                                first_own_time=first)
                    if off is not None:
                        off.append(dict(name=idn.get("name"), index=idx,
                                        reason="off_before_record",
                                        first_own_time=first))
                    _announce_off_before(who, idn, first, t_slice,
                                         key=announce_key)
                    continue
                raise EngineInputRefused(_entry_time_refusal(who, idn, how))
            erec["status"] = "zero"
            continue
        if q.get("j_parallel") is None:
            continue
        jp = np.asarray(q["j_parallel"], dtype=float)
        if jp.shape != (n,) or not np.all(np.isfinite(jp)):
            raise EngineInputRefused(
                f"{who}: core_sources {idn.get('name')!r} (index "
                f"{idx}) j_parallel is malformed or not finite")
        if not np.any(jp != 0.0):
            continue
        if idx in IDS_AGGREGATE_SOURCE_INDICES:
            ignored.append(dict(name=idn.get("name"), index=idx,
                                reason="aggregate entry (a total or a "
                                       "combination of other entries)",
                                j_parallel_max_abs=float(np.max(np.abs(jp)))))
            continue
        if idx in IDS_BOOTSTRAP_LIKE_SOURCE_INDICES:
            ignored.append(dict(name=idn.get("name"), index=idx,
                                reason="bootstrap-like entry (the bootstrap "
                                       "is core_profiles j_bootstrap / Redl)",
                                j_parallel_max_abs=float(np.max(np.abs(jp)))))
            continue
        kind = IDS_DRIVEN_SOURCE_PARTS.get(idx)
        rec = dict(name=idn.get("name"), index=idx, slice=how,
                   matched_time=erec.get("matched_time"), dt=erec.get("dt"))
        if kind is None:
            kind = "other"
            rec["unclassified"] = True
            _warn_driven_once(
                announce_key, ("unknown", idn.get("name"), idx),
                f"{who}: core_sources {idn.get('name')!r} carries a "
                f"non-zero j_parallel under identifier index {idx!r}, which "
                "is not a known driven source (nbi, ec, lh, ic, fusion, "
                "runaways, sawteeth) nor a known aggregate: it is held FIXED "
                "as a driven current under 'other' -- check that it is not a "
                "sum of other entries")
        parts[kind] = parts[kind] + sgn * jp
        rec["part"] = kind
        used.append(rec)
    if ignored:
        _warn_driven_once(
            announce_key, ("not_added",) + tuple(
                (d["name"], d["index"]) for d in ignored),
            f"{who}: core_sources entries carrying a non-zero j_parallel "
            "were NOT added to the driven current: "
            + "; ".join(f"{d['name']!r} (index {d['index']}): {d['reason']}"
                        for d in ignored))
    return parts, used, ignored


def _ids_inductive_mismatch(j_ohm, residual, j_tot, rho):
    """``j_ohmic`` against the parallel residual: the NET difference as a
    fraction of the total current (flux-area proxy ``int j rho drho`` on the
    toroidal-flux radius, so no equilibrium geometry is needed at read
    time) and the rms of the difference over the rms of ``j_total``."""
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    d = j_ohm - residual
    tot = _trapz(j_tot * rho, rho)
    net = (_trapz(d * rho, rho) / tot) if tot != 0.0 else float("nan")
    rms_t = float(np.sqrt(np.mean(j_tot ** 2)))
    rms = (float(np.sqrt(np.mean(d ** 2))) / rms_t if rms_t > 0.0
           else float("nan"))
    return float(net), float(rms)


def _ids_b0(dd, ie, ic):
    """``|B0|`` of the IMAS ``<j.B>/B0`` normalisation (core_profiles, then
    equilibrium ``vacuum_toroidal_field.b0``); REFUSED when absent -- the
    engine never treats ``<j.B>/B0`` as raw ``<j.B>``."""
    for ids_name, it in (("core_profiles", ic), ("equilibrium", ie)):
        vtf = dd.get(ids_name, {}).get("vacuum_toroidal_field")
        if vtf and vtf.get("b0") is not None:
            b0 = np.atleast_1d(np.asarray(vtf["b0"], dtype=float))
            if b0.size:
                v = abs(float(b0[min(it, b0.size - 1)]))
                if np.isfinite(v) and v > 0.0:
                    return v, ids_name
    raise EngineInputRefused(
        "IDS adapter: no vacuum_toroidal_field.b0 in core_profiles or "
        "equilibrium -- the parallel currents are <j.B>/B0 and cannot be "
        "converted without it")


class IdsAdapter:
    """The IMAS / OMAS modelling-source adapter.

    *baseline* is the :class:`~bouquet.baseline.Baseline` the legacy reader
    (:func:`bouquet.io.imas.read_imas_baseline`) already built: its kinetics,
    fast pressure, impurity charge and sawtooth facts are reused unchanged;
    the parallel currents are re-read here, because the reader keeps only
    their toroidal conversion."""

    kind = "ids"

    def __init__(self, source, config, baseline, *,
                 inductive=IDS_INDUCTIVE_DEFAULT,
                 psi_pad=1e-3, inductive_tol=IDS_INDUCTIVE_MISMATCH_TOL):
        if inductive not in IDS_INDUCTIVE_CHOICES:
            raise ValueError(f"IdsAdapter: inductive must be one of "
                             f"{IDS_INDUCTIVE_CHOICES}, got {inductive!r}")
        if not (np.isfinite(inductive_tol) and inductive_tol >= 0.0):
            raise ValueError("IdsAdapter: inductive_tol must be a finite "
                             f"non-negative fraction, got {inductive_tol!r}")
        self.source = source
        self.config = config
        self.bl = baseline
        self.inductive = inductive
        self.inductive_tol = float(inductive_tol)
        self.psi_pad = float(psi_pad)
        self._c = None

    def read(self) -> EngineContract:
        from .io.imas import (_load_dd, _nearest_index, read_imas_geometry,
                              source_current_sign)
        from .physics import ELEMENTARY_CHARGE as _EC, impurity_pressure
        from .utils import STRUCTURED_PRESETS, pchip_interp, q0_gate_admits
        src, cfg, bl = self.source, self.config, self.bl
        gc = cfg.generation
        # the parsed dd shared with the reader (cached per file; READ-ONLY:
        # nothing below writes into it -- tests/test_imas_dd_cache.py)
        dd = _load_dd(src.ids_path)
        T = src.time
        eq = dd["equilibrium"]
        ie = _nearest_index(eq["time"], T, "equilibrium")
        gq = eq["time_slice"][ie]["global_quantities"]
        cps = dd["core_profiles"]
        ic = _nearest_index(cps["time"], T, "core_profiles")
        cp = cps["profiles_1d"][ic]
        # ONE normalisation: the factor the READER applied to this source's
        # currents (Baseline.source_current_sign -- sign(ip) at the slice the
        # currents come from, or ImasSource.current_orientation when the
        # source needs it stated; the reader has already refused a source
        # whose currents disagree with it).  The adapter re-reads the dd's
        # parallel currents and must put them in the SAME frame, so it takes
        # that factor rather than deriving a second one from ip.  A baseline
        # that carries none (not produced by the reader) falls back to
        # sign(ip), the reader's own "auto" rule.
        _sgn_bl = getattr(bl, "source_current_sign", None)
        if _sgn_bl is None:
            sgn = source_current_sign(gq["ip"])
            sgn_origin = "auto: sign(equilibrium ip) (no reader record)"
        else:
            sgn = float(_sgn_bl)
            if sgn not in (1.0, -1.0):
                raise EngineInputRefused(
                    "IDS adapter: the baseline's source_current_sign is "
                    f"{_sgn_bl!r}, not +1 or -1")
            sgn_origin = str(getattr(bl, "source_current_sign_origin", None)
                             or "the reader's normalisation "
                                "(Baseline.source_current_sign)")
        B0, b0_from = _ids_b0(dd, ie, ic)
        # IMAS is COCOS 11: (R, phi, Z) right-handed, so the dd's own ip and
        # b0 signs ARE the orientation in the A-coefficients' frame.  That
        # statement needs the source to be self-consistent: when the current
        # factor the reader applied differs from sign(ip) (an
        # ImasSource.current_orientation override on a source whose currents
        # and ip are stored in different orientations) the source states the
        # direction of Ip twice and the two disagree, so the adapter states
        # NONE -- an MSE row then needs the block's own ip_sign (mse_rows
        # refuses a sign nobody states; it is never guessed).
        _b0s = getattr(bl, "source_b0_sign", None)
        _ip_dd = float(source_current_sign(gq["ip"]))
        _ip_ok = (_ip_dd == float(sgn))
        self.orientation = dict(
            ip_sign=(float(sgn) if _ip_ok else None),
            bt_sign=(None if _b0s is None else float(_b0s)),
            basis=("IDS equilibrium global_quantities.ip and "
                   "vacuum_toroidal_field.b0 signs (COCOS 11)" if _ip_ok
                   else "vacuum_toroidal_field.b0 sign (COCOS 11); the Ip "
                        f"direction is NOT stated: the source's ip sign is "
                        f"{_ip_dd:+.0f} but its currents were read with the "
                        f"factor {float(sgn):+.0f} ({sgn_origin})"),
            current_sign_origin=sgn_origin)
        psi = np.asarray(cp["grid"]["psi"], dtype=float)
        psi_N = (psi - psi[0]) / (psi[-1] - psi[0])
        # the run grid: the reader's (psi_N, or Phi_N in a toroidal-flux run,
        # the same nodes relabelled); psi_cp keeps the nodes' psi_N for the
        # q-row radius
        self.coord = coords.check_coord(getattr(bl, "coord", coords.PSI))
        psi_cp = psi_N
        if self.coord == coords.PHI:
            x_run = np.asarray(bl.psi_N, float)
            if x_run.shape != psi_N.shape:
                raise EngineInputRefused("IDS adapter: the core_profiles grid "
                                         "differs from the reader's")
            psi_N = x_run
        elif not np.allclose(psi_N, np.asarray(bl.psi_N, float), rtol=0.0,
                             atol=1e-12):
            raise EngineInputRefused("IDS adapter: the core_profiles grid "
                                     "differs from the reader's")
        n = psi_N.size

        def _cur(name, required=False):
            if name not in cp or cp[name] is None:
                if required:
                    raise EngineInputRefused(
                        f"IDS adapter: core_profiles carries no {name}")
                return None
            a = sgn * np.asarray(cp[name], dtype=float)
            if a.shape != (n,) or not np.all(np.isfinite(a)):
                raise EngineInputRefused(f"IDS adapter: core_profiles {name} "
                                         "is malformed or not finite")
            return a

        j_ohm = _cur("j_ohmic")
        j_boot = _cur("j_bootstrap")
        j_tot = _cur("j_total")
        # driven parallel currents, by the explicit IMAS identifier
        # classification (IDS_DRIVEN_SOURCE_PARTS: beams -> "nbi", the
        # reader's j_NBI before its toroidal conversion; ec/lh/ic -> "rf";
        # fusion, runaways, sawteeth -> "other"; an unknown index -> "other"
        # with a warning), all held fixed; aggregates and bootstrap-like
        # entries are never added (stamped in provenance["ignored_sources"]);
        # a driven entry with no slice near the time and no current on its
        # bracketing slices is off there (provenance["off_sources"])
        # the core_sources slice: nearest the core_profiles slice read and
        # within half its local step, else refused; entries matched within
        # that half-step too; an entry starting after the slice is off
        # (owner decision 2026-10-06; io.imas.core_sources_slice -- the
        # reader's rule)
        from .io.imas import _cp_window, core_sources_slice
        srcs = dd.get("core_sources", {})
        try:
            isrc, t_src, slice_rec = core_sources_slice(
                srcs, cps.get("time"), ic, T, who="IDS adapter")
        except ValueError as exc:
            raise EngineInputRefused(str(exc)) from None
        t_cp = slice_rec["core_profiles_time"]
        cp_half = (None if (t_cp is None or t_src is None) else
                   _cp_window(cps.get("time"), srcs.get("time"), t_cp,
                              t_src)[0])
        driven_off = []
        entry_matches = []
        fix_parts, driven_used, driven_ignored = _ids_driven_currents(
            srcs, isrc, n, sgn, cps.get("time"), off=driven_off,
            t_cp=t_cp, cp_half=cp_half, matches=entry_matches,
            announce_key=str(src.ids_path))
        nbi = fix_parts["nbi"]
        driven = fix_parts["nbi"] + fix_parts["rf"] + fix_parts["other"]
        # The inductive current is the parallel residual j_total -
        # j_bootstrap - driven BY DEFINITION ("residual", the default); the
        # source's j_ohmic is compared against it and the numbers stamped,
        # a cross-check with no threshold.  The explicit "auto" / "j_ohmic"
        # use j_ohmic and keep the tolerance: a net mismatch above
        # inductive_tol (fraction of the total current) means the source's
        # components are not mutually consistent at this slice (seen: one
        # slice on a sawtooth step of the source's model, where j_ohmic sat
        # 8 points off its neighbours while j_bootstrap did not); "auto" then
        # takes the residual, loudly.  Nothing in the source is altered, and
        # the numbers are stamped whichever inductive is used.
        has_ohm = j_ohm is not None and np.any(j_ohm != 0.0)
        can_resid = j_tot is not None and j_boot is not None
        residual = (j_tot - j_boot - driven) if can_resid else None
        consistency = dict(checked=bool(has_ohm and can_resid),
                           tol=(None if self.inductive == "residual"
                                else self.inductive_tol),
                           net_frac=None, rms_frac=None, action="unchecked")
        if consistency["checked"]:
            _rho = cp["grid"].get("rho_tor_norm")
            rho = (np.asarray(_rho, dtype=float) if _rho is not None
                   else np.sqrt(psi_N))
            if rho.shape != (n,) or not np.all(np.isfinite(rho)):
                rho = np.sqrt(psi_N)
            net, rms = _ids_inductive_mismatch(j_ohm, residual, j_tot, rho)
            consistency.update(net_frac=net, rms_frac=rms)
        mode = self.inductive
        if mode == "residual" and not can_resid:
            missing = [k for k, a in (("j_total", j_tot),
                                      ("j_bootstrap", j_boot)) if a is None]
            raise EngineInputRefused(
                "IDS adapter: inductive='residual' (the default: the "
                "inductive current is j_total - j_bootstrap - driven by "
                "definition) but the source carries no "
                + " / ".join(missing) + "; "
                + ("it does carry j_ohmic: set inductive='j_ohmic' "
                   "(GenerationConfig.engine_ids_inductive='j_ohmic') to use "
                   "the source's j_ohmic explicitly" if has_ohm else
                   "it carries no j_ohmic either (inductive='j_ohmic' would "
                   "need one), so no inductive current can be formed"))
        if mode == "auto":
            mode = "j_ohmic" if has_ohm else "residual"
            if (consistency["checked"] and np.isfinite(consistency["net_frac"])
                    and abs(consistency["net_frac"]) > self.inductive_tol):
                mode = "residual"
                consistency["action"] = "fallback_to_residual"
                import warnings
                warnings.warn(
                    "IDS adapter: the source's current split does not add "
                    f"up at this slice: j_ohmic differs from j_total - "
                    f"j_bootstrap - driven by {100 * consistency['net_frac']:+.2f} "
                    f"% of the total current (rms {100 * consistency['rms_frac']:.2f} "
                    f"% of j_total), above the {100 * self.inductive_tol:.1f} % "
                    "tolerance; using the residual as the inductive current "
                    "(inductive='j_ohmic' forces the source's j_ohmic).",
                    UserWarning, stacklevel=2)
        if mode == "j_ohmic":
            if not has_ohm:
                raise EngineInputRefused("IDS adapter: inductive='j_ohmic' "
                                         "but the source carries no j_ohmic")
            ind = j_ohm
            if consistency["action"] == "unchecked" and consistency["checked"]:
                consistency["action"] = (
                    "kept_j_ohmic" if abs(consistency["net_frac"])
                    <= self.inductive_tol else "kept_j_ohmic_over_tol")
                if consistency["action"] == "kept_j_ohmic_over_tol":
                    import warnings
                    warnings.warn(
                        "IDS adapter: inductive='j_ohmic' kept although the "
                        "source's current split misses by "
                        f"{100 * consistency['net_frac']:+.2f} % of the total "
                        f"current (tolerance {100 * self.inductive_tol:.1f} %)",
                        UserWarning, stacklevel=2)
        else:
            if not can_resid:
                raise EngineInputRefused(
                    "IDS adapter: the source carries no j_ohmic and no "
                    "j_total / j_bootstrap to form the parallel residual; "
                    "refusing (no inductive current can be formed)")
            ind = residual
            if (self.inductive == "residual"
                    and consistency["action"] == "unchecked"
                    and consistency["checked"]):
                # the cross-check: stamped, no threshold, no warning
                consistency["action"] = "residual_by_definition"
        # user-supplied driven currents are defined in the positive-Ip frame
        # (co-current positive) on both source paths, exactly as the reader
        # takes them: they are NOT multiplied by the source's factor (only
        # the dd's own currents are)
        fc = cfg.fixed_components
        # fc.psi_N in the run coordinate (a psi_n input via the dd's map)
        fc_x = coords.to_run_grid(fc.psi_N, getattr(fc, "coord", coords.RUN),
                                  None if self.coord == coords.PSI
                                  else (psi_cp, psi_N))
        user_fix = {}
        if fc.j_NBI is not None:
            user_fix["nbi"] = np.interp(psi_N, fc_x, fc.j_NBI) \
                if fc.psi_N is not None else np.asarray(fc.j_NBI, float)
        if fc.j_RF is not None:
            user_fix["rf"] = np.interp(psi_N, fc_x, fc.j_RF) \
                if fc.psi_N is not None else np.asarray(fc.j_RF, float)
        self._user_fix_tor = user_fix
        # kinetics + pressure exactly as the reader resolved them (thermal +
        # impurity + fast; NO p_diff -- the engine does not anchor pressure)
        k2e = lambda a: pchip_interp(bl.psi_N_kinetic, a, psi_N)  # noqa: E731
        ne, te, ni, ti = k2e(bl.ne), k2e(bl.te), k2e(bl.ni), k2e(bl.ti)
        zeff = np.clip(k2e(bl.Zeff), 1.0, None)
        p_th = _EC * (ne * te + ni * ti)
        p_fast = (k2e(bl.p_fast) if bl.p_fast is not None
                  else np.zeros(n))
        p_imp = np.zeros(n)
        if getattr(bl, "Z_imp", None):
            zf = getattr(bl, "z_fast", None)
            ne_th = ne if zf is None else np.maximum(ne - k2e(zf), 0.0)
            p_imp = impurity_pressure(ne_th, ni, ti, bl.Z_imp)
        preset = STRUCTURED_PRESETS["li_soft_onesided"]
        # the row's target at the MEASUREMENT's radius (R_axis): the stored
        # li_3's own radius resolved (GenerationConfig.imas_li3_radius) and
        # the factor R_src / R_axis applied and recorded
        li3_radius = resolve_li3_radius(
            eq["time_slice"][ie],
            str(getattr(gc, "imas_li3_radius", "auto")))
        li3 = (None if gq.get("li_3") is None else li3_radius["target"])
        eqp1 = eq["time_slice"][ie]["profiles_1d"]
        psi_eq = np.asarray(eqp1["psi"], dtype=float)
        psiN_eq = (psi_eq - psi_eq[0]) / (psi_eq[-1] - psi_eq[0])
        o = np.argsort(psiN_eq)
        q_psi = _q0_psi(psi_cp, self.psi_pad)
        q_t = None
        if "q" in eqp1:
            q_t = float(abs(np.interp(q_psi, psiN_eq[o],
                                      np.asarray(eqp1["q"], float)[o])))
        saw = dict(getattr(bl, "sawtooth", None) or {})
        adm, basis = q0_gate_admits(bool(saw.get("active")), saw.get("q0_dd"),
                                    q_t, float(getattr(gc, "q0_gate", 1.1)))
        rows = dict(
            Ip=dict(target=float(bl.Ip_target), hard=False,
                    sigma=float(preset["structured_ip_sigma_frac"])
                    * float(bl.Ip_target),
                    sigma_origin=("utils.STRUCTURED_PRESETS['li_soft_onesided']"
                                  "['structured_ip_sigma_frac']"),
                    source="equilibrium global_quantities.ip"),
            l_i=(None if li3 is None else dict(
                target=float(li3), kind="li_3", hard=False, tol=None,
                sigma=float(preset["structured_li_sigma"]),
                sigma_origin=("utils.STRUCTURED_PRESETS['li_soft_onesided']"
                              "['structured_li_sigma']"),
                source=("equilibrium global_quantities.li_3 x R_src / "
                        "R_axis (imas_li3_radius)"),
                li3_source=li3_radius["li3_source"],
                radius_choice=li3_radius["choice"],
                radius_status=li3_radius["status"],
                radius_factor=li3_radius["factor"])),
            q0=(None if q_t is None else dict(
                target=q_t, psi=q_psi, admitted=bool(adm), gate_basis=basis,
                q0_source_axis=saw.get("q0_dd"),
                source=("equilibrium profiles_1d.q interpolated at the "
                        "measurement radius (like radii)"))),
            mse=mse_rows(gc, self.orientation),
        )
        _F0, boundary = read_imas_geometry(src)
        anchor = np.asarray(bl.j_phi, dtype=float)
        c = EngineContract(
            kind="ids", psi_N=psi_N,
            kinetics=dict(ne=ne, te=te, ni=ni, ti=ti, zeff=zeff),
            kinetics_native=dict(psi_N=np.asarray(bl.psi_N_kinetic, float),
                                 ne=bl.ne, te=bl.te, ni=bl.ni, ti=bl.ti,
                                 Zeff=bl.Zeff, p_fast=bl.p_fast),
            pressure=p_th + p_imp + p_fast,
            pressure_parts=dict(thermal=p_th, impurity=p_imp, fast=p_fast,
                                Z_imp=getattr(bl, "Z_imp", None)),
            jB_ind=B0 * ind, jB_fix=B0 * driven,
            jB_fix_parts={k: B0 * v for k, v in fix_parts.items()},
            boundary=np.asarray(boundary, dtype=float),
            Ip=float(bl.Ip_target), rows=rows,
            signs=dict(current_sign=float(sgn),
                       current_sign_origin=sgn_origin,
                       b0_sign=getattr(bl, "source_b0_sign", None),
                       ip_sign_RphiZ=self.orientation["ip_sign"],
                       bt_sign_RphiZ=self.orientation["bt_sign"],
                       orientation_basis=self.orientation["basis"],
                       frame="positive: |Ip|, F = |r0 b0|, <j.B> > 0 "
                             "co-current"),
            anchor_request=anchor,
            provenance=dict(
                inductive=(f"{mode}: " + ("core_profiles j_ohmic"
                                          if mode == "j_ohmic" else
                                          "j_total - j_bootstrap - driven "
                                          "core_sources j_parallel")
                           + f" (IMAS <j.B>/B0) x |B0| = {B0:.6g} T "
                             f"(from {b0_from})"),
                inductive_consistency=dict(consistency),
                fixed=("core_sources j_parallel of the DRIVEN entries by "
                       "the IMAS identifier (adapters.IDS_DRIVEN_SOURCE_PARTS; "
                       "an unknown index under 'other' with a warning; "
                       "aggregates and bootstrap-like entries never) x "
                       "|B0|, held fixed; parts nbi / rf (ec, lh, ic) / "
                       "other"),
                driven_sources=list(driven_used),
                ignored_sources=list(driven_ignored),
                off_sources=list(driven_off),
                source_time_match=dict(core_sources=slice_rec,
                                       entries=entry_matches),
                pressure=("e (ne Te + ni Ti) + impurity + fast (no p_diff)"),
                electron_charge="physics.ELEMENTARY_CHARGE",
                kinetics_sigma=("resolved from the Baseline by "
                                "baseline.resolve_uncertainty (unchanged)"),
                anchor=("one solve of the source j_tor (the legacy first "
                        "solve); it only seeds geometry and Redl"),
                li3_radius=dict(li3_radius),
                B0=float(B0)),
        )
        validate_contract(c)
        self._c = c
        return c

    def finalize(self, jB_bs_anchor, anchor_geom) -> EngineContract:
        """Convert user-supplied TOROIDAL fixed parts with the anchor
        geometry (they replace the source's, as in the legacy reader)."""
        from .engine import conversion_factor
        c = self._c
        if c is None:
            raise RuntimeError("IdsAdapter.finalize before read()")
        if self._user_fix_tor:
            kap = conversion_factor(anchor_geom)
            parts = dict(c.jB_fix_parts)
            for k, v in self._user_fix_tor.items():
                parts[k] = np.asarray(v, float) / kap
            c.jB_fix_parts = parts
            c.jB_fix = sum(np.asarray(v, float) for v in parts.values())
            c.provenance["fixed"] = ("FixedComponentsConfig toroidal parts "
                                     "converted with the anchor's "
                                     "F<1/R>/<B^2>")
        c.jB_bs_anchor = np.asarray(jB_bs_anchor, float).copy()
        validate_contract(c)
        return c
