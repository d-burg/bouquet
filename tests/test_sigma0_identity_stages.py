"""Zero-perturbation identity of the draw pipeline, stage by stage (fast half).

The design rule: the bouquet reconstruction is ONE equilibrium F (allowed to
differ from its input), and a draw with every perturbation at zero reproduces
F -- its equilibrium, current, bootstrap and l_i -- within the loop's own
tolerances.  With the self-consistent loop on, every stage of a draw is the
identity at zero perturbation:

* the reconstruction stores its split as F's jphi-linterp REQUEST, normalised
  to Ip in the 'exact' FSA current measure on F (``_deliver_request_split``),
  with the draws' own sigma=0 bootstrap composition and the inductive as the
  residual -- so the state anchor, the anchor loop and route R2 re-solve F,
  and R2's Ip bookkeeping reads a scale of 1;
* the standard route targets ACHIEVED currents, so it perturbs
  ``j_inductive - jphi_request_offset``, roots its amplitude in the same
  measure (1 at zero perturbation), and starts its corrective iteration from
  ``target + offset`` = F's request;
* the g-file reconstruction ends on the l_i re-match of the corrective
  iteration's landed request (``_rematch_li_request``), a single-solve state.

Everything here runs on a TOY Grad-Shafranov solver: a jphi-linterp "solve"
normalises the request to Ip in the solver's own measure and realises an
achieved current ``M @ request`` whose SHAPE differs from the request near
the edge (the representation deviation that made the g-file baseline carry
three states), l_i/q0/q95 are functionals of the achieved current, and the
toy bootstrap depends on the equilibrium (so the loop has a genuine fixed
point).  The real ``perturb_kinetic_equilibrium`` runs on it end to end.
The live-solver halves are in ``tests/test_sigma0_identity_solver.py``.

Synthetic inputs only; no device data.
"""
import os
import sys
import types

import numpy as np
import pytest

from bouquet.config import GenerationConfig
from bouquet.jbs_loop import jbs_settings, profile_residuals

_HERE = os.path.dirname(os.path.abspath(__file__))
_N = 61
_X = np.linspace(0.0, 1.0, _N)
_IP = 1.2e6
_PAD = 1e-3
#: the 'exact' measure's per-surface Ip weights (current-weighted norm too)
_W = 1.6 * (0.3 + _X) * (1.2 - 0.5 * _X)
#: the toy SOLVER normalises in a measure 1e-4 away from the exact one --
#: the real measure's runtime self-check is 1e-5..4e-4 of Ip
_W_SOLVER = _W * (1.0 + 1.0e-4 * _X)


def _trap(y, x=_X):
    from scipy.integrate import trapezoid
    return float(trapezoid(np.asarray(y, float), np.asarray(x, float)))


def _representation_matrix():
    """achieved = M @ request: identity in the core, a shape-changing edge
    overshoot (a smoothing kernel blended in over psi_N > 0.8)."""
    n = _N
    K = np.exp(-0.5 * ((_X[:, None] - _X[None, :]) / 0.04) ** 2)
    K /= K.sum(axis=1, keepdims=True)
    blend = np.clip((_X - 0.8) / 0.2, 0.0, 1.0)[:, None]
    M = (1.0 - 0.25 * blend) * np.eye(n) + 0.25 * blend * K
    M[-1, -1] += 0.04                       # edge overshoot
    return M


_M = _representation_matrix()


def _li(A):
    return 0.45 + 0.9 * _trap(A * _W * (1.0 - _X) ** 1.3) / _trap(A * _W)


class _Snap:
    def __init__(self, req, ach):
        self.req = None if req is None else req.copy()
        self.achieved = None if ach is None else ach.copy()


class ToyGS:
    """Just enough of a TokaMaker for the draw pipeline (see module doc)."""

    def __init__(self):
        self.req = None
        self.achieved = None
        self._ffp = None
        self._Ip = _IP
        self.psi_bounds = (-0.2, 0.1)
        self.n_solves = 0

    # -- profiles / solve ---------------------------------------------------
    def set_targets(self, Ip=None, pax=None, **kw):
        if Ip is not None:
            self._Ip = float(Ip)

    def set_profiles(self, ffp_prof=None, pp_prof=None, **kw):
        if ffp_prof is not None:
            self._ffp = np.asarray(ffp_prof["y"], dtype=float).copy()

    def solve(self):
        y = self._ffp
        self.req = y * self._Ip / _trap(_W_SOLVER * y)
        self.achieved = _M @ self.req
        self.n_solves += 1

    # -- snapshots -----------------------------------------------------------
    def copy_eq(self):
        return _Snap(self.req, self.achieved)

    def replace_eq(self, source_eq=None):
        self.req = source_eq.req.copy()
        self.achieved = source_eq.achieved.copy()

    def get_psi(self, *a, **k):
        return self.req.copy()

    def set_psi(self, psi, update_bounds=True):
        self.req = np.asarray(psi, float).copy()
        self.achieved = _M @ self.req

    # -- measurements ----------------------------------------------------------
    def get_stats(self, lcfs_pad=None, li_normalization=None, **kw):
        A = self.achieved
        return {"l_i": _li(A), "Ip": self._Ip,
                "q_0": 0.6 + 1.0e6 / A[0],
                "q_95": 3.0 + 2.0e6 / _trap(_W * A) * 1.5
                        - 1.0e-7 * np.interp(0.95, _X, A)}

    def _grid(self, psi, npsi):
        if psi is not None:
            return np.asarray(psi, dtype=float)
        if npsi is None or int(npsi) == _N:
            return _X.copy()
        return np.linspace(0.0, 1.0, int(npsi))

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        x = self._grid(psi, npsi)
        A = np.interp(x, _X, self.achieved)
        # get_jphi_from_GS(f*fp, pp, <R>, <1/R>) = f*fp*<1/R>/mu0 + <R>*pp = A
        return (x, A, np.ones_like(A), np.zeros_like(A), np.zeros_like(A))

    def get_q(self, psi=None, npsi=None, psi_pad=None):
        x = self._grid(psi, npsi)
        ravgs = {"<R>": np.ones_like(x), "<1/R>": _MU0 * np.ones_like(x),
                 "dV/dPsi": 1.0 + x}
        return (x, 1.0 + 3.0 * x, ravgs, None, None, None)

    def sauter_fc(self, psi=None, npsi=None, psi_pad=None):
        """The flux-surface averages the SWB conversion reads
        (physics._swb_geometry): a flat <|B|^2>.  The toy's p' is zero, so
        the pressure-driven p'G the conversion takes off a TokaMaker-jphi
        toolkit's SWB output is exactly zero here."""
        x = self._grid(psi, npsi)
        one = np.ones_like(x)
        return (None, None, {"<|B|>": one, "<|B|^2>": one})

    def flux_integral(self, psi_N, prof):
        return _trap(prof, psi_N)

    def get_coil_currents(self):
        return ({"C1": 1.0}, None)

    def get_globals(self):
        return (self._Ip,)


def _toy_redl(eq, psi_N, ne, te, ni, ti, Zeff, psi_pad=1e-3,
              isolate_edge=False, smooth_axis=True, coord="psi_n"):
    """Bootstrap that depends on the kinetics AND on the equilibrium."""
    kin = (np.asarray(ne, float) * np.asarray(te, float)) / (
        np.asarray(ne, float)[0] * np.asarray(te, float)[0])
    base = (3.0e5 * np.exp(-0.5 * ((psi_N - 0.92) / 0.04) ** 2)
            + 6.0e4 * (1.0 - psi_N)) * (0.5 + 0.5 * kin / kin.max())
    J = base * (1.0 + 0.8 * (_li(eq.achieved) - 0.75))
    return J.copy(), {"j_tor_full_raw": J.copy(), "I_BS": _trap(J)}


def _oft_modules():
    """(get_jphi_from_GS, mu0) of OFT when its Python layer imports, else a
    stub of the three names the draw imports (the formula, and
    find_optimal_scale's scale-1 acceptance as OFT implements it)."""
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.append(os.path.abspath(cand))
    try:
        import OpenFUSIONToolkit.TokaMaker.util as U  # noqa: F401
        import OpenFUSIONToolkit.TokaMaker.bootstrap as B  # noqa: F401
        return float(U.mu0), None
    except Exception:
        return 4.0e-7 * np.pi, "stub"


_MU0, _OFT_KIND = _oft_modules()


@pytest.fixture()
def toy(monkeypatch):
    """Patch the measure, the residual weights and Redl onto the toy."""
    import bouquet.jbs_loop as L
    import bouquet.physics as P
    import bouquet.TokaMaker_interface as TI
    import bouquet.utils as U
    if _OFT_KIND == "stub":
        mu0 = _MU0
        util = types.ModuleType("OpenFUSIONToolkit.TokaMaker.util")
        util.get_jphi_from_GS = (lambda ffp, pp, R, iR:
                                 ffp * (iR / mu0) + R * pp)
        util.mu0 = mu0
        boot = types.ModuleType("OpenFUSIONToolkit.TokaMaker.bootstrap")

        def _fos(mygs, psi_N, pressure, ffp_prof, pp_prof, j_inductive,
                 Ip_target, psi_pad, spike_prof=None, tolerance=0.01,
                 max_iter=5, diagnostic_plots=False, verbose=True):
            psi_eval = np.clip(psi_N, psi_pad, 1.0 - psi_pad)
            ffp_prof["y"] = j_inductive + spike_prof
            mygs.set_targets(Ip=Ip_target, pax=pressure[0])
            mygs.set_profiles(ffp_prof=ffp_prof, pp_prof=pp_prof)
            mygs.solve()
            _, f, fp, _, pp = mygs.get_profiles(psi=psi_eval)
            out0 = f[0] * fp[0]
            inp0 = j_inductive[0] + spike_prof[0]
            if abs(inp0 - out0) / out0 < tolerance:
                return 1.0, f * fp
            raise AssertionError("stub find_optimal_scale: scale != 1")

        boot.find_optimal_scale = _fos
        boot.solve_with_bootstrap = None
        pkg = types.ModuleType("OpenFUSIONToolkit")
        sub = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
        sub.util, sub.bootstrap, pkg.TokaMaker = util, boot, sub
        for name, mod in (("OpenFUSIONToolkit", pkg),
                          ("OpenFUSIONToolkit.TokaMaker", sub),
                          ("OpenFUSIONToolkit.TokaMaker.util", util),
                          ("OpenFUSIONToolkit.TokaMaker.bootstrap", boot)):
            monkeypatch.setitem(sys.modules, name, mod)
    geom = lambda eq, psi, **k: {"toy": True}               # noqa: E731
    prof = lambda g, conv, eq=None, pprime_sign=1.0: np.asarray(  # noqa
        eq.achieved, float)
    wts = lambda g, convention="jphi-linterp", pprime_sign=1.0: (  # noqa
        _W.copy(), 0.0)
    for mod in (TI, U):
        monkeypatch.setattr(mod, "fsa_current_geometry", geom)
        monkeypatch.setattr(mod, "eq_jphi_profile", prof)
        monkeypatch.setattr(mod, "Ip_fsa_weights", wts)
    monkeypatch.setattr(L, "residual_weights",
                        lambda eq, psi_N, psi_pad=1e-3, coord="psi_n": (_W.copy(), _X,
                                                         "toy"))
    monkeypatch.setattr(P, "evaluate_jBS", _toy_redl)
    monkeypatch.delenv("BOUQUET_R2_IP_MODE", raising=False)
    return ToyGS()


# ---------------------------------------------------------------------------
#  the kinetics and a reconstruction on the toy
# ---------------------------------------------------------------------------
_NE = 5.0e19 * (1.0 - 0.8 * _X ** 2)
_TE = 3.0e3 * (1.0 - 0.9 * _X ** 1.5) + 50.0
_NI = 0.9 * _NE
_TI = 0.9 * _TE
_ZEFF = 1.6 * np.ones(_N)
# the draws' eV -> J factor under the self-consistent loop
from bouquet.physics import ELEMENTARY_CHARGE as _EC  # noqa: E402


def _kin_redl(eq):
    return _toy_redl(eq, _X, _NE, _TE, _NI, _TI, _ZEFF)[0]


def _reconstruct(mygs, j_fixed=None, jphi_diff=None, kappa_short=1.0,
                 n_loop=60):
    """A toy reconstruction: inductive shape + self-consistent bootstrap,
    iterated to Redl(F) == the bootstrap in the request.  The request is
    handed over ``kappa_short`` short of Ip in the exact measure (the source
    total read 0.965 Ip on the real modelling-source example).  Returns
    ``(request, j_bs, fixed_eff)`` with ``mygs`` on F."""
    j_ind0 = 1.6e6 * (1.0 - _X ** 2) ** 1.5
    fixed = np.zeros(_N) if j_fixed is None else np.asarray(j_fixed, float)
    jd = np.zeros(_N) if jphi_diff is None else np.asarray(jphi_diff, float)
    jbs = np.zeros(_N)
    mygs.set_targets(Ip=_IP)
    for _ in range(n_loop):
        req = j_ind0 + jbs + fixed + jd
        mygs.set_profiles(ffp_prof={"y": req})
        mygs.solve()
        jbs_new = _kin_redl(mygs)
        if np.max(np.abs(jbs_new - jbs)) < 1e-9 * np.max(np.abs(jbs_new)):
            jbs = jbs_new
            break
        jbs = jbs_new
    req = j_ind0 + jbs + fixed + jd
    mygs.set_profiles(ffp_prof={"y": req})
    mygs.solve()
    # the stored total is short of Ip in the exact measure (the solver makes
    # it up uniformly -- the modelling-source path's defect)
    k = kappa_short * _IP / _trap(_W * req)
    return k * req, jbs, fixed + jd


def _delivered(mygs, request, jbs, fixed_eff):
    """What baseline.py / run.py store with the loop on."""
    from bouquet.TokaMaker_interface import (_deliver_request_split,
                                             _request_offset)
    dv = _deliver_request_split(mygs, _X, _PAD, _IP, request, jbs, fixed_eff)
    off, n_fl = _request_offset(dv["j_inductive"], dv["achieved"], jbs,
                                fixed_eff)
    return dv, off, n_fl


def _draw(mygs, route, input_j_phi, j_ind, l_i_target, settings,
          offset=None, j_NBI=None, jphi_diff=None):
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    from bouquet.sampling import make_rng
    z = np.zeros(_N)
    pressure = _EC * (_NE * _TE + _NI * _TI)
    return perturb_kinetic_equilibrium(
        mygs, _X, pressure, _NE, _TE, _NI, _TI,
        np.asarray(input_j_phi, float), z, z, z, z, z, 0.5, 0.4, 0.25,
        _IP, float(l_i_target), _ZEFF, _N,
        input_jinductive=np.asarray(j_ind, float), l_i_tolerance=0.05,
        psi_pad=_PAD, constrain_sawteeth=False, recalculate_j_BS=True,
        isolate_edge_jBS=False, scale_jBS=1.0, floor_j_BS=False,
        jBS_diff=None, perturb_jind_in_anchor=(route == "ip_renorm"),
        accept_anchor_inband=False, j_NBI=j_NBI, jphi_diff=jphi_diff,
        max_proxy_draws=5, p_thresh=0.05, rng=make_rng(7),
        jbs_loop=settings, jphi_request_offset=offset)[6]


def _settings(**over):
    gc = GenerationConfig()
    for k, v in over.items():
        setattr(gc, k, v)
    return jbs_settings(gc, draw=True)


def _identity(mygs, F, jbs_F, d, s):
    """The draw vs F at the loop's tolerances (and the equilibrium itself)."""
    spk = np.asarray(d["_jbs_ctx"]["spike_used"], float)
    c = profile_residuals(spk, jbs_F, _W, _X, _IP)
    li_d = _li(mygs.achieved)
    return dict(r_j=c["r_j"], r_I=c["r_I"], dl_i=abs(li_d - _li(F.achieved)),
                eq=float(np.max(np.abs(mygs.achieved - F.achieved))
                         / np.max(np.abs(F.achieved))),
                converged=bool(d["jbs_loop"]["converged"]),
                ok=(d["jbs_loop"]["converged"] and c["r_j"] <= s["rtol_j"]
                    and c["r_I"] <= s["rtol_Ip"]
                    and abs(li_d - _li(F.achieved)) <= s["tol_li"]))


# ---------------------------------------------------------------------------
#  1. the stored split is the reconstruction's own request, Ip-normalised
# ---------------------------------------------------------------------------
def test_delivered_split_sums_to_the_normalised_request_and_resolves_to_F(toy):
    req, jbs, fx = _reconstruct(toy, kappa_short=0.965)
    F = toy.copy_eq()
    dv, off, n_fl = _delivered(toy, req, jbs, fx)
    # normalised in the exact measure ON F: Ip exactly, factor 1/0.965
    assert _trap(_W * dv["request"]) == pytest.approx(_IP, rel=1e-12)
    assert dv["kappa"] == pytest.approx(1.0 / 0.965, rel=1e-12)
    # the parts sum to it exactly (the inductive is the residual)
    np.testing.assert_allclose(dv["j_inductive"] + jbs + fx, dv["request"],
                               rtol=0, atol=1e-9 * np.max(dv["request"]))
    # one solve of the stored request IS F (shape-only normalisation)
    toy.set_targets(Ip=_IP)
    toy.set_profiles(ffp_prof={"y": dv["request"]})
    toy.solve()
    np.testing.assert_allclose(toy.achieved, F.achieved, rtol=1e-12)
    # the standard route's achieved-convention mean + offset = the request
    assert n_fl == 0
    mean_t = dv["j_inductive"] - off
    np.testing.assert_allclose(mean_t + jbs + fx + off, dv["request"],
                               rtol=1e-12)
    np.testing.assert_allclose(mean_t + jbs + fx, dv["achieved"], rtol=1e-12)


def test_request_offset_floors_the_target_inductive_and_counts_it():
    from bouquet.TokaMaker_interface import _request_offset
    j_ind = np.array([3.0, 2.0, 1.0, 0.2])
    ach = np.array([4.0, 3.0, 1.5, 0.1])
    jbs = np.array([0.5, 0.5, 0.4, 0.3])
    off, n = _request_offset(j_ind, ach, jbs, np.zeros(4))
    assert n == 1
    np.testing.assert_allclose(j_ind - off, [3.5, 2.5, 1.1, 0.0])


# ---------------------------------------------------------------------------
#  2. Ip bookkeeping: R2's scale and the standard root read 1 at sigma=0
# ---------------------------------------------------------------------------
def test_r2_scale_is_one_on_the_normalised_split_in_the_same_measure(toy):
    """The measure the split is normalised in IS the measure R2 roots in, so
    the scale is 1 to rounding -- and on the un-normalised split it is the
    defect the diagnosis measured (the whole shortfall charged to the
    inductive)."""
    from bouquet.TokaMaker_interface import _AnchorIpRenorm
    req, jbs, fx = _reconstruct(toy, kappa_short=0.965)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    aip = _AnchorIpRenorm(toy, _X, dv["request"], _IP, _PAD, mode="exact")
    s = aip.solve_scale(dv["j_inductive"], jbs + fx)
    assert abs(s - 1.0) <= 1e-12
    # guard the guard: the old storage (the short total, inductive residual
    # of it) reads the shortfall on the inductive alone
    j_ind_old = req - jbs - fx
    s_old = aip.solve_scale(j_ind_old, jbs + fx)
    f_ind = _trap(_W * j_ind_old) / _IP
    assert s_old - 1.0 == pytest.approx((1.0 - 0.965) / f_ind, rel=1e-9)
    # the standard route's root on its achieved-convention candidate
    a = aip.solve_scale(dv["j_inductive"] - off, jbs + fx)
    assert abs(a - 1.0) <= 1e-12


# ---------------------------------------------------------------------------
#  3. the corrective iteration: identity from the reconstruction's request
# ---------------------------------------------------------------------------
def test_corrective_iteration_started_from_the_request_stays_on_F(toy):
    from bouquet.TokaMaker_interface import (_corrective_jphi_iteration,
                                             _corrective_output_jphi)
    req, jbs, fx = _reconstruct(toy)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    F = toy.copy_eq()
    target = dv["achieved"]
    pp = {"type": "linterp", "y": np.zeros(_N), "x": _X}
    out, n, hist, landed = _corrective_jphi_iteration(
        toy, _X, target, pp, _IP, 1.0e4, _PAD, min_iters=2, max_iters=8,
        rtol=0.05, verbose=False, initial_request=target + off,
        return_request=True)
    # The first iterate IS F.  What moves it at all is the corrective
    # target's Ip normalisation (issue #29's renormalisation, in the exact
    # measure) against the solver's own: the target is F's achieved current
    # x kappa_achieved, which F carries only up to the measure's self-check
    # (|1 - kappa_achieved| = 4.4e-4 on this toy; 1e-5..4e-4 measured on real
    # cases).  The Newton step then injects that uniform residual into the
    # request; its effect on the state is second order: bounded here at a
    # tenth of the self-check.
    sc = abs(1.0 - dv["kappa_achieved"])
    assert 0.0 < sc < 1e-3
    assert np.max(np.abs(toy.achieved - F.achieved)) <= 0.1 * sc * np.max(
        F.achieved)
    np.testing.assert_allclose(_corrective_output_jphi(toy, _X, _PAD),
                               out, rtol=1e-12)
    assert landed is not None
    # guard the guard: started from the TARGET (the historical start) the
    # first iterate solves the achieved current as a request and leaves F
    toy.replace_eq(F)
    _corrective_jphi_iteration(toy, _X, target, pp, _IP, 1.0e4, _PAD,
                               min_iters=2, max_iters=1, rtol=0.05,
                               verbose=False)
    assert np.max(np.abs(toy.achieved - F.achieved)) > 1e-4 * np.max(
        F.achieved)


def test_corrective_iteration_default_kwargs_are_the_prechange_function(
        toy):
    """Legacy bit-identity: with the new keywords at their defaults the
    function is the frozen pre-change copy, iterate for iterate."""
    import _corrective_prechange as PRE
    from bouquet.TokaMaker_interface import _corrective_jphi_iteration
    req, jbs, fx = _reconstruct(toy)
    F = toy.copy_eq()
    target = 1.02 * (_M @ (req * _IP / _trap(_W_SOLVER * req)))
    pp = {"type": "linterp", "y": np.zeros(_N), "x": _X}
    for protect in (False, True):
        toy.replace_eq(F)
        a = _corrective_jphi_iteration(toy, _X, target, pp, _IP, 1e4, _PAD,
                                       min_iters=2, max_iters=6, rtol=0.05,
                                       verbose=False, protect_state=protect)
        sa = toy.copy_eq()
        toy.replace_eq(F)
        b = PRE._corrective_jphi_iteration(toy, _X, target, pp, _IP, 1e4,
                                           _PAD, min_iters=2, max_iters=6,
                                           rtol=0.05, verbose=False,
                                           protect_state=protect)
        assert len(a) == 3 and len(b) == 3
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1] and a[2] == b[2]
        np.testing.assert_array_equal(sa.achieved, toy.achieved)
        # the legacy reconstruction's call asks for return_request=True: the
        # first three results and the landed state are still the pre-change
        # ones (the request is a recorded copy)
        toy.replace_eq(F)
        c = _corrective_jphi_iteration(toy, _X, target, pp, _IP, 1e4, _PAD,
                                       min_iters=2, max_iters=6, rtol=0.05,
                                       verbose=False, protect_state=protect,
                                       return_request=True)
        assert len(c) == 4
        np.testing.assert_array_equal(c[0], b[0])
        assert c[1] == b[1] and c[2] == b[2]
        np.testing.assert_array_equal(toy.achieved, sa.achieved)
        # ... and the request it reports re-solves to the landed state
        toy.set_profiles(ffp_prof={"y": c[3]})
        toy.solve()
        np.testing.assert_array_equal(toy.achieved, sa.achieved)


# ---------------------------------------------------------------------------
#  4. the g-file reconstruction's l_i re-match (the ONE state it delivers)
# ---------------------------------------------------------------------------
def test_rematch_puts_the_landed_request_back_on_the_l_i_target(toy):
    from bouquet.TokaMaker_interface import _rematch_li_request
    req, jbs, fx = _reconstruct(toy)
    li0 = _li(toy.achieved)
    target = li0 - 0.006                   # the corrective drift, reversed
    pp = {"type": "linterp", "y": np.zeros(_N), "x": _X}
    rm = _rematch_li_request(toy, _X, req - jbs, jbs, pp, _IP, 1.0e4,
                             target, _PAD)
    assert rm["converged"] and abs(rm["li"] - target) < 1e-3
    assert rm["li"] == pytest.approx(_li(toy.achieved), abs=0)
    # the state it leaves IS one solve of the request it reports
    F = toy.copy_eq()
    toy.set_profiles(ffp_prof={"y": rm["request"]})
    toy.solve()
    np.testing.assert_allclose(toy.achieved, F.achieved, rtol=1e-12)
    np.testing.assert_allclose(rm["request"],
                               rm["factor"] * (req - jbs) + jbs, rtol=1e-12)


def test_rematch_is_the_identity_on_an_already_matched_state(toy):
    from bouquet.TokaMaker_interface import _rematch_li_request
    req, jbs, fx = _reconstruct(toy)
    n0 = toy.n_solves
    pp = {"type": "linterp", "y": np.zeros(_N), "x": _X}
    rm = _rematch_li_request(toy, _X, req - jbs, jbs, pp, _IP, 1.0e4,
                             _li(toy.achieved), _PAD)
    assert rm["n_solves"] == 0 and toy.n_solves == n0
    assert rm["factor"] == 1.0
    np.testing.assert_array_equal(rm["request"], (req - jbs) + jbs)


# ---------------------------------------------------------------------------
#  5. the whole draw at zero perturbation, both routes, on the toy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["gfile", "imas"])
@pytest.mark.parametrize("route", ["ip_renorm", "standard"])
def test_the_unperturbed_draw_reproduces_the_reconstruction(toy, route,
                                                            scenario):
    if scenario == "imas":
        j_nbi = 8.0e4 * np.exp(-0.5 * ((_X - 0.3) / 0.15) ** 2)
        jd = 4.0e4 * np.sin(2.0 * np.pi * _X) * (1.0 - _X)
        req, jbs, fx = _reconstruct(toy, j_fixed=j_nbi, jphi_diff=jd,
                                    kappa_short=0.965)
    else:
        j_nbi, jd = None, None
        req, jbs, fx = _reconstruct(toy, kappa_short=0.997)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    F = toy.copy_eq()
    jbs_F = _kin_redl(F)
    j_phi = dv["request"] - (0.0 if jd is None else jd)
    s = _settings()
    d = _draw(toy, route, j_phi, dv["j_inductive"], _li(F.achieved), s,
              offset=off, j_NBI=j_nbi, jphi_diff=jd)
    r = _identity(toy, F, jbs_F, d, s)
    assert r["ok"], r
    if route == "ip_renorm":
        # every stage the identity BY CONSTRUCTION: to rounding
        assert r["r_j"] <= 1e-10 and r["dl_i"] <= 1e-12, r
        assert r["eq"] <= 1e-10, r
        assert abs(d["r2_ip_scale"] - 1.0) <= 1e-10
    else:
        # identical up to the corrective target's exact-measure Ip
        # normalisation vs the solver's own (see the corrective test): a
        # tenth of that self-check
        sc = abs(1.0 - dv["kappa_achieved"])
        assert r["eq"] <= 0.1 * sc and r["r_j"] <= 0.1 * sc, (r, sc)
        # the l_i stage's core rescale is exactly 1 (find_optimal_scale's
        # first trial accepted) on every candidate solve
        assert d["j0_scales"] and all(v == 1.0 for v in d["j0_scales"]), d


@pytest.mark.parametrize("route", ["ip_renorm", "standard"])
@pytest.mark.parametrize("flag", ["jbs_gate_current_residual",
                                  "jbs_loop_q0_corrector"])
def test_identity_holds_with_each_opt_in_flag_on(toy, route, flag):
    req, jbs, fx = _reconstruct(toy, kappa_short=0.98)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    F = toy.copy_eq()
    s = _settings(**{flag: True})
    d = _draw(toy, route, dv["request"], dv["j_inductive"], _li(F.achieved),
              s, offset=off)
    r = _identity(toy, F, _kin_redl(F), d, s)
    assert r["ok"], r


def test_guard_the_guard_the_old_storage_misses_the_reconstruction(toy):
    """Before the change: the modelling-source split was the SHORT source
    total, and the draw's R2 renormalisation charged the shortfall to the
    inductive -- the toy reproduces the defect's sign and size class."""
    req, jbs, fx = _reconstruct(toy, kappa_short=0.965)
    F = toy.copy_eq()
    s = _settings()
    d = _draw(toy, "ip_renorm", req, req - jbs - fx, _li(F.achieved), s)
    r = _identity(toy, F, _kin_redl(F), d, s)
    assert d["r2_ip_scale"] > 1.03
    # the draw is a different, more peaked equilibrium (the toy's l_i is
    # less sensitive than a real one, so the size is asserted on the state)
    assert r["eq"] > 1e-3 and r["dl_i"] > 1e-4, r


def test_guard_the_guard_the_achieved_split_as_a_request_leaves_F(toy):
    """Before the change on the g-file path: the stored split was F's
    ACHIEVED current, and a single-solve route fed it as a request lands on
    a different equilibrium (the diagnosis' J)."""
    req, jbs, fx = _reconstruct(toy)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    F = toy.copy_eq()
    s = _settings()
    d = _draw(toy, "ip_renorm", dv["achieved"], dv["achieved"] - jbs,
              _li(F.achieved), s)
    r = _identity(toy, F, _kin_redl(F), d, s)
    assert r["eq"] > 1e-4, r
