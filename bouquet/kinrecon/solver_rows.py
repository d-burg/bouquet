"""Magnetic measurement rows evaluated by the solver's own reconstruction constraints.

The modeled signals of the groups ``flux_loop``, ``bpol_probe``, ``ip``, ``diamagnetic_flux``, ``pf_current`` and
``mse`` (:data:`SOLVER_GROUPS`) are taken from OpenFUSIONToolkit's reconstruction module
(``OpenFUSIONToolkit.TokaMaker.reconstruction``): one constraint object per channel, set up once per solver object
and measurement set, and evaluated on the current equilibrium by ``reconstruction.eval_error`` (column 2, the
modeled signal).  :func:`.score.score` keeps everything the solver does not have or defines differently:

* the residual and chi^2 definitions, weights, valid masks and reports (:class:`.interfaces.ResidualTable`);
* the IMAS flux sign: the solver's flux-loop row is ``+2 pi psi_TM``, the IMAS quantity ``-2 pi psi_TM``
  (:func:`.diagnostics.flux_loop`), so the row is negated; a loop stored relative to a reference loop is the
  difference of two rows (the solver has no relative form); the fitted flux offset stays in :mod:`.score`;
* the off-mesh rule: the solver moves an off-mesh point to the nearest boundary node, the engine excludes the
  channel with a recorded reason, so the engine's own mesh test (:func:`.diagnostics.on_mesh`, and for finite-length
  probes the segment rule of :func:`.diagnostics.bpol_probe`) decides which channels are built at all;
* the orientation map: the modeled values are multiplied by the stated frame signs in :mod:`.score`, and the MSE
  orientation ``(sign_pol, sign_tor)`` enters the coefficients (``c0, c1, c4, c5`` times ``sign_pol``, ``c2, c6``
  times ``sign_tor``; exact);
* the plasma-current row is built with a placeholder value (the solver refuses Ip <= 0 as a VALUE; its modeled
  signal is the equilibrium's signed current) -- every row is written with value 0 (Ip: 1) and error 1, since only
  the modeled signal is read;
* Thomson groups (not Grad-Shafranov constraints) and the stub groups.

Coil rows are the solver's total coil current (record type 15, ``total=True``): the coil's own current plus its
share of the virtual vertical-stability coil, the engine's definition (``get_coil_currents``).  Finite-length probes
use the solver's EFIT-convention average (``length``, ``n_avg``), the engine's rule.

The solver evaluates the ACTIVE equilibrium of a ``TokaMaker`` object.  A ``TokaMaker_equilibrium`` (a copy) is
therefore scored by the engine's own diagnostics (:mod:`.diagnostics`), which stay as that path and as the reference
of the equivalence tests.

Lifetime of the solver's constraint set.  The solver holds ONE constraint set per ``TokaMaker`` object, which any
reconstruction object replaces when it sets up its own.  Outside :func:`keep_constraints`, every evaluation sets the
set up, evaluates it and destroys it again, so nothing is left behind and nothing else can have replaced it.  Inside
:func:`keep_constraints` (the scope :func:`.fit.reconstruct` and :func:`.series.run_series` run in, where the engine
is the only user of the solver's reconstruction constraints), the set of the specification last evaluated stays set
up between evaluations and is replaced only when another specification is evaluated (or the solver object's
pointer changed); it is destroyed when the outermost scope ends, also on an exception.  The values are the same bit
for bit: the set holds only the constraint definitions and the mesh cells of their points (located once, from the
same start, as a fresh set-up locates them); every evaluation reads the solver's CURRENT equilibrium (flux, coil
currents, profiles).  The :class:`_Rows` of each specification are cached per solver object (the
:data:`_CACHE_MAX` most recently used), so alternating specifications (the fit groups and all groups of a fit) do not
build new reconstruction objects on every call.
"""

import contextlib
import hashlib
import os
import shutil
import tempfile
import weakref

import numpy as np

from . import diagnostics as _diag

#: Groups whose modeled signal comes from the solver's reconstruction constraints.
SOLVER_GROUPS = ("flux_loop", "bpol_probe", "ip", "diamagnetic_flux", "pf_current", "mse")
_MSE_QUANTITY = {"polarisation_angle": "angle", "tan_polarisation_angle": "tan"}
#: Per solver object: ``{signature: _Rows}``, most recently used last (at most :data:`_CACHE_MAX` entries).
_CACHE = weakref.WeakKeyDictionary()
_CACHE_MAX = 8
#: Depth of nested :func:`keep_constraints` scopes (0: every evaluation sets up and destroys the set).
_KEEP = dict(depth=0)
#: Per solver object: the :class:`_Rows` whose constraint set the solver holds (only inside :func:`keep_constraints`).
_LIVE = weakref.WeakKeyDictionary()


def available(eq):
    """True when *eq* is a ``TokaMaker`` object (its active equilibrium is what the solver evaluates)."""
    return hasattr(eq, "lim_contour") and hasattr(eq, "_tMaker_ptr")


def _point_reasons(eq, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    reason = np.full(pts.shape[0], "", dtype=object)
    good = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] > 0.0)
    reason[~good] = _diag.REASON_BAD_GEOMETRY
    inside = np.zeros(pts.shape[0], dtype=bool)
    if np.any(good):
        inside[good] = _diag.on_mesh(eq, pts[good])
    reason[good & ~inside] = _diag.REASON_OFF_MESH
    return reason


def _probe_spec(eq, geo, model, n_avg):
    """Per-probe ``(reason, (loc, norm, length, n_avg) or None)`` under the engine's rules (diagnostics.bpol_probe)."""
    pts = _diag._points(geo)
    n = pts.shape[0]
    nr = np.atleast_1d(np.asarray(geo["direction_r"], dtype=float)).ravel()
    nz = np.atleast_1d(np.asarray(geo["direction_z"], dtype=float)).ravel()
    if nr.shape != (n,) or nz.shape != (n,):
        raise ValueError("bpol_probe: direction arrays must match 'r'")
    norm = np.hypot(nr, nz)
    if np.any(~np.isfinite(norm) | (norm <= 0.0)):
        raise ValueError("bpol_probe: probe direction must be a finite, non-zero vector")
    if model not in _diag.PROBE_MODELS:
        raise ValueError(f"bpol_probe: model must be one of {_diag.PROBE_MODELS}")
    finite = model == "finite_length" and "length" in geo
    if not finite:
        reason = _point_reasons(eq, pts)
        L = np.zeros(n)
    else:
        seg, ends = _diag.probe_segment_points(geo, n_avg)
        m = seg.shape[1]
        rp = _point_reasons(eq, seg.reshape(-1, 2)).reshape(n, m)
        re = _point_reasons(eq, ends.reshape(-1, 2)).reshape(n, 2)
        L = np.atleast_1d(np.asarray(geo["length"], dtype=float)).ravel()
        reason = np.full(n, "", dtype=object)
        good = np.all(np.isfinite(pts), axis=1) & (pts[:, 0] > 0.0)
        for i in range(n):
            if not good[i]:
                reason[i] = _diag.REASON_BAD_GEOMETRY
                continue
            bad = [r for r in rp[i] if r]
            if bad or any(re[i]):
                reason[i] = _diag.REASON_SEGMENT_OFF_MESH if np.any(np.abs(seg[i] - pts[i]) > 0) else \
                    (bad[0] if bad else _diag.REASON_OFF_MESH)
    spec = []
    for i in range(n):
        if reason[i]:
            spec.append(None)
            continue
        nrm = (float(nr[i] / norm[i]), 0.0, float(nz[i] / norm[i]))
        spec.append(((float(pts[i, 0]), float(pts[i, 1])), nrm, float(L[i]) if (finite and L[i] != 0.0) else None,
                     int(n_avg) if (finite and L[i] != 0.0) else None))
    return reason, spec


def _field_values(e, pts):
    n = pts.shape[0]
    if e is None:
        return np.zeros(n)
    v = e(pts[:, 0], pts[:, 1]) if callable(e) else e
    return np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()


def build_specs(eq, measurements, names, kinetics, frame, flux_reference):
    """Channel specifications and the engine's per-channel reasons for the solver groups in *names*."""
    specs, reasons = {}, {}
    for name in names:
        group = measurements.groups[name]
        geo = group.geometry
        if name == "flux_loop":
            pts = _diag._points(geo)
            r = _point_reasons(eq, pts)
            s = [None if r[i] else (float(pts[i, 0]), float(pts[i, 1])) for i in range(pts.shape[0])]
            ref = None
            if flux_reference is not None:
                rp = np.asarray(flux_reference, dtype=float).reshape(1, 2)
                rr = _point_reasons(eq, rp)
                if rr[0]:
                    r = np.where(r == "", "reference loop: " + rr[0], r)
                    s = [None] * len(s)
                else:
                    ref = (float(rp[0, 0]), float(rp[0, 1]))
            specs[name] = dict(points=s, reference=ref)
            reasons[name] = r
        elif name == "bpol_probe":
            model = group.meta.get("probe_model", "finite_length")
            n_avg = int(group.meta.get("probe_n_avg", _diag.PROBE_N_AVG_DEFAULT))
            r, s = _probe_spec(eq, geo, model, n_avg)
            specs[name], reasons[name] = dict(probes=s), r
        elif name in ("ip", "diamagnetic_flux"):
            specs[name], reasons[name] = dict(), np.full(1, "", dtype=object)
        elif name == "pf_current":
            names_c = list(geo["coil_names"])
            missing = [c for c in names_c if c not in _diag._tokamaker(eq).coil_sets]
            if missing:
                raise KeyError(f"pf_current: coil sets not in the equilibrium: {missing}")
            specs[name], reasons[name] = dict(coils=names_c), np.full(len(names_c), "", dtype=object)
        elif name == "mse":
            pts = _diag._points(geo)
            n = pts.shape[0]
            r = _point_reasons(eq, pts)
            er, ez = _field_values(kinetics.get("e_r"), pts), _field_values(kinetics.get("e_z"), pts)
            sp, st = frame["mse_signs"]
            c = np.asarray(geo["coefficients"], dtype=float).reshape(-1, 9)
            if c.shape[0] != n:
                raise ValueError("mse: one row of 9 coefficients per channel is required")
            cs = c * np.array([sp, sp, st, 1.0, sp, sp, st, 1.0, 1.0])
            q = group.meta.get("quantity", "polarisation_angle")
            if q not in _MSE_QUANTITY:
                raise ValueError(f"mse: quantity must be one of {tuple(_MSE_QUANTITY)}")
            bad_e = (r == "") & ~(np.isfinite(er) & np.isfinite(ez))
            r[bad_e] = _diag.REASON_NONFINITE
            s = [None if r[i] else ((float(pts[i, 0]), float(pts[i, 1])), tuple(float(v) for v in cs[i]),
                                    float(er[i]), float(ez[i])) for i in range(n)]
            specs[name], reasons[name] = dict(chords=s, quantity=_MSE_QUANTITY[q]), r
    return specs, reasons


class _Rows:
    """A reconstruction object holding one constraint per built channel, and the row of each."""

    def __init__(self, gs, specs):
        from OpenFUSIONToolkit.TokaMaker.reconstruction import reconstruction
        self.dir = tempfile.mkdtemp(prefix="kr_rows_")
        self._finalize_dir = weakref.finalize(self, _remove_dir, self.dir, os.getpid())
        self.live_ptr = None              # the solver pointer the set was set up on while it is kept (else None)
        self.recon = reconstruction(gs, in_filename=os.path.join(self.dir, "fit.in"),
                                    out_filename=os.path.join(self.dir, "fit.out"))
        rc, cons = self.recon, {}
        if "flux_loop" in specs:
            fl = specs["flux_loop"]
            for i, p in enumerate(fl["points"]):
                if p is not None:
                    rc.add_flux_loop(p, 0.0, 1.0)
                    cons[("flux_loop", i)] = rc._flux_loops[-1]
            if fl["reference"] is not None:
                rc.add_flux_loop(fl["reference"], 0.0, 1.0)
                cons[("flux_loop", "ref")] = rc._flux_loops[-1]
        for i, p in enumerate(specs.get("bpol_probe", {}).get("probes", [])):
            if p is not None:
                rc.add_Mirnov(p[0], p[1], 0.0, 1.0, length=p[2], n_avg=p[3])
                cons[("bpol_probe", i)] = rc._mirnovs[-1]
        if "pf_current" in specs:
            coils = specs["pf_current"]["coils"]
            uniq = list(dict.fromkeys(coils))
            rc.set_coil_currents({c: 0.0 for c in uniq}, {c: 1.0 for c in uniq}, total=True)
            by_id = {int(con.ind): con for con in rc._coil_current_cons}
            for i, c in enumerate(coils):
                cons[("pf_current", i)] = by_id[int(gs.coil_sets[c]["id"])]
        if "ip" in specs:
            rc.set_Ip(1.0, 1.0)
            cons[("ip", 0)] = rc._Ip_con
        if "diamagnetic_flux" in specs:
            rc.set_DFlux(0.0, 1.0)
            cons[("diamagnetic_flux", 0)] = rc._Dflux_con
        if "mse" in specs:
            m = specs["mse"]
            for i, ch in enumerate(m["chords"]):
                if ch is not None:
                    rc.add_MSE(ch[0], ch[1], 0.0, 1.0, quantity=m["quantity"], E_R=ch[2], E_Z=ch[3])
                    cons[("mse", i)] = rc._mse_cons[-1]
        order = {id(con): k for k, con in enumerate(rc.get_constraints())}
        self.rows = {key: order[id(con)] for key, con in cons.items()}
        self.n_rows = len(order)
        self.projected = []

    def evaluate(self, gs):
        """Modeled signal of every row on the current equilibrium of *gs* (module docstring, "Lifetime").  Outside
        :func:`keep_constraints` the set is set up and destroyed around the evaluation; inside, it is set up only
        when the solver does not hold this object's set (first use, another specification evaluated since, or a
        new solver pointer) and is kept afterwards."""
        if not self.n_rows:
            return np.zeros(0)
        if _KEEP["depth"] <= 0:
            self.recon.setup_constraints()
            try:
                self.projected = list(self.recon.projected)
                return self.recon.eval_error()[:, 1].copy()
            finally:
                self.recon.destroy_constraints()       # leave no constraint set behind (score changes nothing)
        ptr = _pointer(gs)
        live = _LIVE.get(gs)
        if live is not self or self.live_ptr != ptr:
            _LIVE.pop(gs, None)
            if live is not None:
                live.release()
            self.release()
            try:
                self.recon.setup_constraints()
            except BaseException:
                self.recon.destroy_constraints()       # a set-up that failed leaves no set behind
                raise
            self.live_ptr = ptr
            _LIVE[gs] = self
            self.projected = list(self.recon.projected)
        try:
            return self.recon.eval_error()[:, 1].copy()
        except BaseException:
            _LIVE.pop(gs, None)
            self.release()                             # an evaluation that failed leaves no set behind
            raise

    def release(self):
        """Destroy this object's constraint set in the solver if it holds it (kept sets only)."""
        if self.live_ptr is None:
            return
        self.live_ptr = None
        self.recon.destroy_constraints()


def _remove_dir(path, pid):
    """Remove a constraint-file directory, in the process that made it only (not in a forked worker)."""
    if os.getpid() == pid:
        shutil.rmtree(path, True)


def _pointer(gs):
    p = getattr(gs, "_tMaker_ptr", None)
    return getattr(p, "value", p)


@contextlib.contextmanager
def keep_constraints():
    """Scope in which the solver's constraint set is kept between evaluations (module docstring, "Lifetime"); the
    kept sets are destroyed when the outermost scope ends.  Changes no value.  Usable as a decorator."""
    _KEEP["depth"] += 1
    try:
        yield
    finally:
        _KEEP["depth"] -= 1
        if _KEEP["depth"] <= 0:
            _KEEP["depth"] = 0
            release_all()


def release_all():
    """Destroy every kept constraint set (module docstring, "Lifetime")."""
    for gs in list(_LIVE.keys()):
        rows = _LIVE.pop(gs, None)
        if rows is not None:
            rows.release()


def _signature(specs):
    return hashlib.sha256(repr(sorted(specs.items(), key=lambda kv: kv[0])).encode()).hexdigest()


def rows_for(gs, specs):
    """The cached :class:`_Rows` of *gs* for *specs* (built on the first use of the specification; the
    :data:`_CACHE_MAX` most recently used specifications are kept per solver object)."""
    sig = _signature(specs)
    per = _CACHE.get(gs)
    if per is None:
        per = _CACHE[gs] = {}
    rows = per.pop(sig, None)
    if rows is None:
        rows = _Rows(gs, specs)
    per[sig] = rows                                    # most recently used last
    while len(per) > _CACHE_MAX:
        old = per.pop(next(iter(per)))
        if _LIVE.get(gs) is old:
            del _LIVE[gs]
        old.release()
    return rows


def evaluate(eq, measurements, names, kinetics, frame, flux_reference=None):
    """``{group: Synthetic}`` for the solver groups in *names* (module docstring), on the active equilibrium of the
    ``TokaMaker`` object *eq*; also returns the record ``{"projected": n}``."""
    specs, reasons = build_specs(eq, measurements, names, kinetics, frame, flux_reference)
    rows = rows_for(eq, specs)
    sig = rows.evaluate(eq)
    out = {}
    for name in names:
        r = reasons[name].copy()
        n = r.shape[0]
        val = np.full(n, np.nan)
        for i in range(n):
            k = rows.rows.get((name, i))
            if k is not None:
                val[i] = sig[k]
        if name == "flux_loop":
            kref = rows.rows.get(("flux_loop", "ref"))
            # solver row +2 pi psi_TM; IMAS -2 pi psi_TM (relative: minus the reference loop's IMAS value)
            val = -val if kref is None else (-val) - (-sig[kref])
        bad = (r == "") & ~np.isfinite(val)
        val[bad] = np.nan
        r[bad] = _diag.REASON_NONFINITE
        out[name] = _diag.Synthetic(val, r)
    return out, dict(projected=len(rows.projected), rows=rows.n_rows)
