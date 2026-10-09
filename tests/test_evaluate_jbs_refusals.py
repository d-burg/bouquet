"""evaluate_jBS refuses what it cannot evaluate -- and changes nothing else.

The Redl expressions are undefined for non-physical kinetic input
(``Z_eff < 1``, non-positive densities or temperatures) and on a surface the
flux-surface tracer failed on (it returns an all-zero row).  The historical
evaluator mapped the resulting NaN to ``j_BS = 0`` at that node with no
signal.  It now raises :class:`bouquet.physics.JBSEvaluationError` naming the
quantity and the ``psi_N`` location, and keeps the historical treatment only
at the END nodes (geometry on the clipped axis / separatrix surface,
identified by coordinate).

The bit-identity half compares the new function against a verbatim copy of
the evaluator as it was before this change (:func:`_evaluate_jBS_reference`
below) on the synthetic mock equilibrium of ``test_jbs_loop`` over several
grids and options: for every accepted input the output is identical to the
last bit.

Synthetic inputs only; no device data.  Needs OFT's pure-Python ``bootstrap``
module (Redl); skipped with a reason when OFT is absent.
"""
import numpy as np
import pytest

_bs = pytest.importorskip(
    "OpenFUSIONToolkit.TokaMaker.bootstrap",
    reason="evaluate_jBS wraps OFT's pure-Python Redl implementation; "
           "OpenFUSIONToolkit is not importable here")

from bouquet.physics import (EVALUATE_JBS_VERSION, JBSEvaluationError,  # noqa: E402
                             _EC, _SAUTER_MODB_INDEX, _SAUTER_RAVG_INDEX,
                             _sauter_avg, evaluate_jBS,
                             jpar_to_jphi_tokamaker,
                             jphi_tokamaker_pressure_term, q_ravg)
from test_jbs_loop import _MockEq, _kin  # noqa: E402


# ---------------------------------------------------------------------------
#  the reference: the evaluator exactly as it was before the refusals
# ---------------------------------------------------------------------------
def _evaluate_jBS_reference(mygs, psi_N, ne, te, ni, ti, zeff, *, psi_pad=1e-3,
                 isolate_edge=False, smooth_axis=True):
    """Verbatim copy of evaluate_jBS as it was BEFORE the refusals (the
    historical nan_to_num at every node); docstring dropped."""
    import OpenFUSIONToolkit.TokaMaker.bootstrap as _oft_bs

    psi_N = np.asarray(psi_N, dtype=float)
    n = psi_N.size
    if psi_N.ndim != 1 or n < 3:
        raise ValueError("evaluate_jBS: psi_N must be 1-D with >= 3 points")
    if not np.all(np.isfinite(psi_N)):
        raise ValueError("evaluate_jBS: psi_N contains non-finite values")
    if np.any(np.diff(psi_N) <= 0.0):
        raise ValueError("evaluate_jBS: psi_N must be strictly increasing")
    if psi_N[0] < 0.0 or psi_N[-1] > 1.0:
        raise ValueError(
            f"evaluate_jBS: psi_N must lie in [0, 1] (got {psi_N[0]}, "
            f"{psi_N[-1]})")
    psi_pad = float(psi_pad)
    if not (0.0 < psi_pad < 0.5):
        raise ValueError(f"evaluate_jBS: psi_pad must be in (0, 0.5), got "
                         f"{psi_pad!r}")

    def _prof(a, name):
        a = np.asarray(a, dtype=float)
        if a.ndim == 0:
            a = np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(f"evaluate_jBS: {name} has shape {a.shape}, "
                             f"expected ({n},) to match psi_N")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"evaluate_jBS: {name} contains non-finite "
                             "values")
        return a

    ne = _prof(ne, "ne")
    te = _prof(te, "te")
    ni = _prof(ni, "ni")
    ti = _prof(ti, "ti")
    zeff = _prof(zeff, "zeff")

    # ---- geometry on the caller's surfaces (distinct clipped values only) ---
    psi_eval = np.clip(psi_N, psi_pad, 1.0 - psi_pad)
    psi_u, inv = np.unique(psi_eval, return_inverse=True)
    psi_u = np.ascontiguousarray(psi_u, dtype=float)
    _, F_u, _, _, pp_u = mygs.get_profiles(psi=psi_u.copy())
    # a live TokaMaker exposes sauter_fc; a copy_eq() snapshot
    # (TokaMaker_equilibrium) exposes the same routine as calc_sauter_fc
    _sfc = getattr(mygs, "sauter_fc", None)
    if _sfc is None:
        _sfc = getattr(mygs, "calc_sauter_fc")
    fc_u, r_sau, modb = _sfc(psi=psi_u.copy())[-3:]
    _, q_u, ravgs_q, *_rest = mygs.get_q(psi=psi_u.copy())
    F = np.asarray(F_u, dtype=float)[inv]
    f_T = (1.0 - np.asarray(fc_u, dtype=float))[inv]
    eps = (_sauter_avg(r_sau, "<a>", _SAUTER_RAVG_INDEX)
           / _sauter_avg(r_sau, "<R>", _SAUTER_RAVG_INDEX))[inv]
    avg_inv_R = _sauter_avg(r_sau, "<1/R>", _SAUTER_RAVG_INDEX)[inv]
    avg_B2 = _sauter_avg(modb, "<|B|^2>", _SAUTER_MODB_INDEX)[inv]
    q = np.asarray(q_u, dtype=float)[inv]
    R_avg = np.asarray(q_ravg(ravgs_q, "<R>"), dtype=float)[inv]
    inv_R_q = np.asarray(q_ravg(ravgs_q, "<1/R>"), dtype=float)[inv]
    dV_dpsi = np.abs(np.asarray(q_ravg(ravgs_q, "dV/dPsi"), dtype=float))[inv]

    # ---- gradients on the TRUE grid, current flux range ---------------------
    bounds = np.asarray(mygs.psi_bounds, dtype=float)
    psi_range = float(bounds[1] - bounds[0])
    if psi_range == 0.0 or not np.isfinite(psi_range):
        raise ValueError(f"evaluate_jBS: degenerate psi_bounds {bounds}")

    def _d(y):
        return np.gradient(y, psi_N, edge_order=2) / psi_range

    dn_e = _d(ne)
    dT_e = _d(te)
    dn_i = _d(ni)
    dT_i = _d(ti)

    # ---- collisionality (verbatim SWB physics) ------------------------------
    ln_le, ln_lii = _oft_bs.calculate_ln_lambda(
        te, ti, ne, ni, zeff,
        electron_lnLambda_model="NRL", ion_lnLambda_model="Zavg")
    Zdom = 1.0                         # deuterium main ion
    Zavg = ne / ni
    Zion = (Zdom ** 2 * Zavg * zeff) ** 0.25
    nu_i_star = (4.90e-18 * np.abs(q) * R_avg * ni
                 * Zion ** 4 * ln_lii / (ti ** 2 * eps ** 1.5))
    nu_e_star = (6.921e-18 * np.abs(q) * R_avg * ne
                 * zeff * ln_le / (te ** 2 * eps ** 1.5))

    j_dot_B, _coeffs = _oft_bs.redl_bootstrap(
        psi_N=psi_N, Te=te, Ti=ti, ne=ne, ni=ni,
        pe=_EC * (ne * te), pi=_EC * (ni * ti),
        Zeff=zeff, R=R_avg, q=q, eps=eps, fT=f_T, I_psi=F,
        dT_e_dpsi=dT_e, dT_i_dpsi=dT_i,
        dn_e_dpsi=dn_e, dn_i_dpsi=dn_i,
        ln_lambda_e=ln_le, ln_lambda_ii=ln_lii,
        nu_e_star_override=nu_e_star, nu_i_star_override=nu_i_star,
        use_legacy_L34=False, use_sign_q=True, formula_form="jboot1")
    j_dot_B = np.nan_to_num(np.asarray(j_dot_B, dtype=float), nan=0.0)

    geom = {"F": F, "avg_inv_R": avg_inv_R, "avg_B2": avg_B2,
            "avg_R": R_avg, "pprime": np.asarray(pp_u, dtype=float)[inv]}
    p_term = jphi_tokamaker_pressure_term(geom)
    j_tor_full = np.nan_to_num(jpar_to_jphi_tokamaker(j_dot_B, geom) + p_term,
                               nan=0.0)

    if isolate_edge:
        # SWB isolates the spike on its OWN projection <j.B> R_avg/F (the
        # shelf/mask detection is value-dependent), and bouquet then converts
        # the masked spike by the per-surface factor.  Same order here.
        swb_proj = j_dot_B * (R_avg / F)
        res = _oft_bs.analyze_bootstrap_edge_spike(psi_N, swb_proj)
        masked = np.asarray(res["masked_spike"], dtype=float)
        j_tor_sel = np.nan_to_num(jpar_to_jphi_tokamaker(
            masked * F / R_avg, geom) + p_term, nan=0.0)
    else:
        j_tor_sel = j_tor_full

    if smooth_axis:
        from bouquet.TokaMaker_interface import smooth_jbs_transition
        j_out = smooth_jbs_transition(j_tor_sel)
    else:
        j_out = np.asarray(j_tor_sel, dtype=float).copy()

    # FSA current of the delivered profile ('fsa' measure: V'/2pi <1/R>).
    w_fsa = dV_dpsi / (2.0 * np.pi) * inv_R_q * abs(psi_range)
    from scipy.integrate import trapezoid as _trap
    I_BS = float(_trap(w_fsa * j_out, psi_N))

    diag = dict(
        psi_eval=psi_eval, psi_pad=psi_pad,
        n_geometry_surfaces=int(psi_u.size),
        f_T=f_T, nu_e_star=nu_e_star, nu_i_star=nu_i_star, q=q, eps=eps,
        R_avg=R_avg, F=F, avg_inv_R=avg_inv_R, avg_B2=avg_B2,
        ln_lambda_e=np.asarray(ln_le, dtype=float),
        ln_lambda_ii=np.asarray(ln_lii, dtype=float),
        dpsi=psi_range, j_dot_B=j_dot_B,
        j_tor_full_raw=j_tor_full, j_tor_raw=np.asarray(j_tor_sel, float),
        I_BS=I_BS, isolate_edge=bool(isolate_edge),
        smooth_axis=bool(smooth_axis), version=EVALUATE_JBS_VERSION,
    )
    return j_out, diag



# ---------------------------------------------------------------------------
#  bit-identity on every accepted input
# ---------------------------------------------------------------------------
def _grids():
    rho = np.linspace(0.0, 1.0, 201)
    return {
        "uniform-11": np.linspace(0.0, 1.0, 11),
        "uniform-151": np.linspace(0.0, 1.0, 151),
        "uniform-1001": np.linspace(0.0, 1.0, 1001),
        # IMAS-type: three surfaces inside psi_pad
        "imas-77": (np.arange(77) / 76.0) ** 2,
        "rho-uniform-201": rho ** 2 * (0.5 + 0.5 * rho),
        # no node on either clipped end
        "interior": np.linspace(0.02, 0.98, 97),
    }


_DIAG_ARRAYS = ("psi_eval", "f_T", "nu_e_star", "nu_i_star", "q", "eps",
                "R_avg", "F", "avg_inv_R", "avg_B2", "ln_lambda_e",
                "ln_lambda_ii", "j_dot_B", "j_tor_full_raw", "j_tor_raw")


#: (grid, isolate_edge, smooth_axis, legacy layout).  isolate_edge=True runs
#: OFT's spike analyser (~1 s per call on the mock), so it is taken on the two
#: production-like grids only; on the 11-point grid the analyser does not
#: terminate (an OFT property, independent of this change), so it is omitted.
_CASES = ([(g, False, sm, lg) for g in sorted(_grids())
           for sm in (True, False) for lg in (False, True)]
          + [(g, True, True, lg) for g in ("imas-77", "rho-uniform-201")
             for lg in (False, True)])


@pytest.mark.parametrize("grid, isolate_edge, smooth_axis, legacy", _CASES)
def test_every_accepted_input_is_bit_identical_to_the_historical_evaluator(
        grid, isolate_edge, smooth_axis, legacy):
    x = _grids()[grid]
    k = _kin(x)
    kw = dict(isolate_edge=isolate_edge, smooth_axis=smooth_axis)
    j_ref, d_ref = _evaluate_jBS_reference(_MockEq(legacy=legacy), x, *k,
                                           **kw)
    j_new, d_new = evaluate_jBS(_MockEq(legacy=legacy), x, *k, **kw)
    np.testing.assert_array_equal(j_new, j_ref)
    for key in _DIAG_ARRAYS:
        np.testing.assert_array_equal(np.asarray(d_new[key]),
                                      np.asarray(d_ref[key]), err_msg=key)
    assert d_new["I_BS"] == d_ref["I_BS"]
    assert d_new["n_nonfinite_zeroed_at_ends"] == 0


def test_bit_identity_holds_for_scalar_zeff_and_other_flux_ranges():
    x = (np.arange(77) / 76.0) ** 2
    ne, te, ni, ti, _z = _kin(x)
    for bounds in ((-0.9, 0.1), (0.4, -0.3), (-2.5, -0.5)):
        for zeff in (1.0, 1.8, 3.2):
            j_ref, _ = _evaluate_jBS_reference(
                _MockEq(psi_bounds=bounds), x, ne, te, ni, ti, zeff)
            j_new, _ = evaluate_jBS(_MockEq(psi_bounds=bounds), x, ne, te,
                                    ni, ti, zeff)
            np.testing.assert_array_equal(j_new, j_ref)


# ---------------------------------------------------------------------------
#  the singular END nodes keep the historical treatment -- and only they
# ---------------------------------------------------------------------------
class _AxisLimitEq(_MockEq):
    """The innermost (clipped axis) surface returns f_c marginally above 1,
    i.e. f_T < 0 -- the f_T -> 0 limit of the axis overshot by the tracer.
    Redl's sqrt(f_T) is NaN there: a singularity of the axis surface by
    construction, which the historical code zeroed."""

    def sauter_fc(self, psi=None, **kw):
        psi_, fc, rav, modb, *eps = super().sauter_fc(psi=psi, **kw)
        fc = np.array(fc, dtype=float)
        fc[np.asarray(psi, dtype=float) <= 1e-3] = 1.0 + 1e-9
        return (psi_, fc, rav, modb, *eps)


@pytest.mark.parametrize("grid", ["uniform-151", "imas-77"])
def test_the_clipped_axis_surface_keeps_the_historical_zeroing(grid):
    x = _grids()[grid]
    k = _kin(x)
    with np.errstate(invalid="ignore"):
        j_ref, d_ref = _evaluate_jBS_reference(_AxisLimitEq(), x, *k,
                                               smooth_axis=False)
        j_new, d_new = evaluate_jBS(_AxisLimitEq(), x, *k, smooth_axis=False)
    np.testing.assert_array_equal(j_new, j_ref)
    n_end = int(np.count_nonzero(x <= 1e-3))
    assert np.all(j_new[:n_end] == 0.0)
    assert d_new["n_nonfinite_zeroed_at_ends"] >= n_end


def test_a_non_finite_redl_value_off_the_end_nodes_is_refused(monkeypatch):
    """The coordinate test, not the value: the same NaN is zeroed at an end
    node and refused one node inside it."""
    x = np.linspace(0.0, 1.0, 101)
    k = _kin(x)
    orig = _bs.redl_bootstrap

    def poisoned(at):
        def redl(**kw):
            j, c = orig(**kw)
            j = np.array(j, dtype=float)
            j[at] = np.nan
            return j, c
        return redl

    monkeypatch.setattr(_bs, "redl_bootstrap", poisoned(0))
    j, d = evaluate_jBS(_MockEq(), x, *k, smooth_axis=False)
    assert j[0] == 0.0 and d["n_nonfinite_zeroed_at_ends"] >= 1
    monkeypatch.setattr(_bs, "redl_bootstrap", poisoned(100))
    j, d = evaluate_jBS(_MockEq(), x, *k, smooth_axis=False)
    assert j[100] == 0.0
    monkeypatch.setattr(_bs, "redl_bootstrap", poisoned(1))
    with pytest.raises(JBSEvaluationError, match=r"<j_BS\.B>.*psi_N=0\.01 ") \
            as ei:
        evaluate_jBS(_MockEq(), x, *k, smooth_axis=False)
    assert ei.value.index == 1 and ei.value.psi_N == pytest.approx(0.01)


# ---------------------------------------------------------------------------
#  refusals of non-physical input, by name and location
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which, value, idx", [
    ("zeff", 0.99, 40), ("te", 0.0, 55), ("te", -5.0, 55),
    ("ti", 0.0, 70), ("ne", -1e18, 30), ("ni", 0.0, 90),
    # the axis and separatrix nodes are NOT exempt from the input domain
    ("te", 0.0, 100), ("ne", 0.0, 0), ("zeff", 0.5, 0),
])
def test_non_physical_input_is_refused_by_name_and_location(which, value,
                                                            idx):
    x = np.linspace(0.0, 1.0, 101)
    ne, te, ni, ti, zeff = [np.array(a, dtype=float) for a in _kin(x)]
    prof = dict(ne=ne, te=te, ni=ni, ti=ti, zeff=zeff)
    prof[which][idx] = value
    # the historical evaluator returned a silently zeroed node here
    with np.errstate(all="ignore"):
        j_ref, _ = _evaluate_jBS_reference(
            _MockEq(), x, prof["ne"], prof["te"], prof["ni"], prof["ti"],
            prof["zeff"], smooth_axis=False)
    assert np.all(np.isfinite(j_ref))
    with pytest.raises(JBSEvaluationError, match=which) as ei:
        evaluate_jBS(_MockEq(), x, prof["ne"], prof["te"], prof["ni"],
                     prof["ti"], prof["zeff"], smooth_axis=False)
    e = ei.value
    assert isinstance(e, ValueError)
    assert e.quantity == which and e.index == idx
    assert e.psi_N == pytest.approx(x[idx]) and e.n_bad == 1
    assert f"psi_N={x[idx]:.6g}" in str(e)


class _FailedTraceEq(_MockEq):
    """The tracer fails on one surface: get_q and sauter_fc return a ZERO row
    there (zero-initialised outputs; a failed trace CYCLEs)."""

    def __init__(self, bad_psi, **kw):
        super().__init__(**kw)
        self.bad_psi = float(bad_psi)

    def _zero(self, psi, arr):
        arr = np.array(arr, dtype=float)
        hit = np.isclose(np.asarray(psi, dtype=float), self.bad_psi,
                         rtol=0, atol=1e-12)
        if arr.ndim == 1:
            arr[hit] = 0.0
        else:
            arr[:, hit] = 0.0
        return arr

    def sauter_fc(self, psi=None, **kw):
        psi_, fc, rav, modb, *eps = super().sauter_fc(psi=psi, **kw)
        rav = {k: self._zero(psi, v) for k, v in rav.items()}
        return (psi_, self._zero(psi, fc), rav, self._zero(psi, modb),
                *(self._zero(psi, e) for e in eps))

    def get_q(self, psi=None, **kw):
        psi_, q, rav, *rest = super().get_q(psi=psi, **kw)
        rav = {k: self._zero(psi, v) for k, v in rav.items()}
        return (psi_, self._zero(psi, q), rav, *rest)


@pytest.mark.parametrize("bad_psi", [0.5, 1e-3, 1.0 - 1e-3])
def test_a_failed_trace_zero_row_is_refused_everywhere(bad_psi):
    """Also on the clipped axis / separatrix surface: a failed trace is not a
    singularity by construction."""
    x = np.linspace(0.0, 1.0, 101)
    k = _kin(x)
    with np.errstate(all="ignore"):
        j_ref, _ = _evaluate_jBS_reference(_FailedTraceEq(bad_psi), x, *k,
                                           smooth_axis=False)
    i = int(np.argmin(np.abs(np.clip(x, 1e-3, 1 - 1e-3) - bad_psi)))
    assert j_ref[i] == 0.0                  # the historical silent zero
    with pytest.raises(JBSEvaluationError, match="failed") as ei:
        evaluate_jBS(_FailedTraceEq(bad_psi), x, *k, smooth_axis=False)
    assert ei.value.index == i


def test_a_negative_trapped_fraction_inside_the_plasma_is_refused():
    class _Bad(_MockEq):
        def sauter_fc(self, psi=None, **kw):
            psi_, fc, rav, modb, *eps = super().sauter_fc(psi=psi, **kw)
            fc = np.array(fc, dtype=float)
            fc[np.isclose(np.asarray(psi), 0.3)] = 1.0 + 1e-9
            return (psi_, fc, rav, modb, *eps)
    x = np.linspace(0.0, 1.0, 101)
    with pytest.raises(JBSEvaluationError, match="f_T") as ei:
        evaluate_jBS(_Bad(), x, *_kin(x))
    assert ei.value.index == 30


# ---------------------------------------------------------------------------
#  the toroidal conversion is the exact TokaMaker jphi (A7, incl. p'G)
# ---------------------------------------------------------------------------
def test_the_toroidal_conversion_is_the_exact_one_with_the_pressure_term():
    """Pins the convention: TokaMaker
    ``jphi = F<1/R><j.B>/<B^2> + p'(<R> - F^2<1/R>/<B^2>)`` -- the field-
    aligned part and the pressure-driven p'G on the bootstrap, as the IMAS
    reader and OFT's own SWB output carry it."""
    x = np.linspace(0.0, 1.0, 151)
    _j, d = evaluate_jBS(_MockEq(), x, *_kin(x), smooth_axis=False)
    exp = (d["F"] * d["avg_inv_R"] * d["j_dot_B"] / d["avg_B2"]
           + d["pprime"] * (d["R_avg"] - d["F"] ** 2 * d["avg_inv_R"]
                            / d["avg_B2"]))
    np.testing.assert_allclose(d["j_tor_full_raw"], exp, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(d["p_term"], d["pprime"] * (
        d["R_avg"] - d["F"] ** 2 * d["avg_inv_R"] / d["avg_B2"]))
