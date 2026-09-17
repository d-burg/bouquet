"""closure_channel="structured": the minimal-norm radial Ip closure.

Solve-free coverage of ``utils.close_ip_structured`` -- the SHIPPED algebra, not
a re-derivation of it (same rule as ``test_ohmic_closure.py``: an earlier
generation of those tests re-derived the scale formulas locally and so validated
the tester rather than the code).

What is proved here:

* **it CONTAINS the scalar channels.**  One constant basis function with the
  inductive coefficient hard-pinned (``W_ind = inf``) reproduces
  ``close_ip("bootstrap")`` to machine precision; the same basis with the axis
  row added reproduces ``close_ip_q0`` exactly, and does so for EVERY weight
  set, because two constraints on two unknowns leave nothing for the objective
  to choose.  A large finite ``W_ind`` converges to the same answer
  monotonically, which is what says the ``inf`` shortcut is a limit and not a
  special case.
* **Ip is exact by construction**, on the real 4-Gaussian basis, with and
  without the axis row, for both shipped weight sets.
* **the KKT solution IS the norm-minimiser** -- checked against
  ``scipy.optimize`` on a small case, and against a direct search over the
  constraint null space.
* **the refusals fire**: multipliers outside [0.2, 5], a singular/degenerate
  constraint system, a constraint row the basis cannot move, non-finite input,
  bad weights and bad basis specs.

No solver anywhere: the geometry is the same synthetic
``fsa_current_geometry``-shaped dict the ohmic-closure tests use.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from bouquet.utils import (Ip_fsa_weights, close_ip, close_ip_q0,
                           close_ip_structured, closure_health,
                           structured_basis_eval,
                           STRUCTURED_BASIS_DEFAULT,
                           STRUCTURED_WEIGHTS_PHYSICS,
                           STRUCTURED_WEIGHTS_UNIFORM)

_N = 201
_CONST = dict(kind="constant")


def _geom():
    """A plausible fsa_current_geometry() dict for a D3D-like shape."""
    psi = np.linspace(0.01, 0.99, _N)
    R0, a = 1.7, 0.6
    r = a * np.sqrt(psi)
    R_avg = R0 + 0.1 * r ** 2 / a
    inv_R = (1.0 / R0) * (1.0 + 0.05 * (r / a) ** 2)
    inv_R2 = inv_R ** 2 * (1.0 + 0.02 * (r / a) ** 2)
    dV_dpsi = 4.0 * np.pi ** 2 * R0 * r * (a / (2.0 * np.sqrt(psi) + 1e-12))
    pprime = -8.0e3 * (1.0 - psi)
    return {"psi_N": psi, "psi_q": psi, "R_avg": R_avg, "inv_R": inv_R,
            "inv_R2": inv_R2, "dV_dpsi": dV_dpsi, "dpsi_dpsiN": 0.9,
            "pprime": pprime}


def _parts(bs_amp=3.0e5):
    """(psi, w, c, j_ind, j_bs, j_fix, lin, Ip_signed) -- a closable hybrid.

    Ip_signed is deliberately NOT the raw component sum: the whole point of a
    closure is that the three sources miss the target (here by ~4 %, the size
    the FUSE core_profiles total actually misses by).
    """
    g = _geom()
    w, c = Ip_fsa_weights(g, convention="jphi-linterp")
    psi = g["psi_N"]
    j_ind = 8.0e5 * (1.0 - psi) ** 1.5
    j_bs = bs_amp * np.exp(-((psi - 0.9) / 0.06) ** 2) + 2.0e4 * (1.0 - psi)
    j_fix = 1.0e5 * (1.0 - psi) ** 3
    lin = lambda j: float(trapezoid(w * np.asarray(j, float), psi))
    raw = lin(j_ind) + lin(j_bs) + lin(j_fix) + c
    Ip_signed = 1.04 * raw
    return psi, w, c, j_ind, j_bs, j_fix, lin, Ip_signed


def _axis(psi, j_ind, j_bs, j_fix, q0_pull=0.97):
    """An axis row at the psi_pad-clipped axis sample, as run.py builds it."""
    p0 = float(psi[0])
    at = lambda j: float(np.interp(p0, psi, np.asarray(j, float)))
    j_ind0, j_bs0, j_fix0 = at(j_ind), at(j_bs), at(j_fix)
    # a reference axis current a few % away from what the raw components give
    j_ref0 = q0_pull * (j_ind0 + j_bs0 + j_fix0)
    return dict(psi=p0, j_ind0=j_ind0, j_bs0=j_bs0, j_fix0=j_fix0,
                j_ref0=j_ref0)


# ---------------------------------------------------------------------------
class TestBasis:
    def test_gaussian_basis_is_peak_normalised_at_its_centre(self):
        spec = STRUCTURED_BASIS_DEFAULT
        Phi = structured_basis_eval(spec, np.asarray(spec["centres"], float))
        np.testing.assert_allclose(np.diag(Phi), 1.0, rtol=0, atol=1e-15)

    def test_constant_basis_is_one_function_identically_one(self):
        Phi = structured_basis_eval(_CONST, np.linspace(0, 1, 17))
        assert Phi.shape == (1, 17)
        np.testing.assert_array_equal(Phi, 1.0)

    def test_default_basis_spans_core_to_pedestal(self):
        c = np.asarray(STRUCTURED_BASIS_DEFAULT["centres"], float)
        assert c.size == 4 and c[0] < 0.2 and c[-1] > 0.9
        assert np.all(np.diff(c) > 0)

    @pytest.mark.parametrize("spec, match", [
        (dict(kind="chebyshev"), "unknown structured basis kind"),
        (dict(kind="gaussian", centres=[0.2, 0.8], widths=[0.1]), "same length"),
        (dict(kind="gaussian", centres=[0.2], widths=[0.0]), "strictly positive"),
        (dict(kind="gaussian", centres=[np.nan], widths=[0.1]), "non-finite"),
    ])
    def test_bad_basis_specs_raise(self, spec, match):
        with pytest.raises(ValueError, match=match):
            structured_basis_eval(spec, np.linspace(0, 1, 11))

    def test_scalar_width_broadcasts(self):
        Phi = structured_basis_eval(
            dict(kind="gaussian", centres=[0.3, 0.7], widths=0.2),
            np.linspace(0, 1, 21))
        assert Phi.shape == (2, 21)


# ---------------------------------------------------------------------------
class TestContainsTheScalarChannels:
    """The channel must be a strict generalisation, not a different answer."""

    def test_constant_basis_pinned_ind_reproduces_close_ip_bootstrap(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ref_ohm, ref_bs = close_ip("bootstrap", Ip_s, c,
                                   lin(j_ind), lin(j_bs), lin(j_fix))
        out = close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, basis=_CONST,
            weights=dict(name="pin-ind", ind=(np.inf,), bs=(1.0,)))
        np.testing.assert_allclose(out["s_ind"], ref_ohm, rtol=0, atol=1e-15)
        np.testing.assert_allclose(out["s_bs"], ref_bs, rtol=1e-12)
        assert out["ohm_scale_eff"] == pytest.approx(ref_ohm, rel=1e-12)
        assert out["bs_scale_eff"] == pytest.approx(ref_bs, rel=1e-12)
        # a constant multiplier has, by definition, no structure
        assert out["structure_ind"] == pytest.approx(0.0, abs=1e-12)
        assert out["structure_bs"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_basis_pinned_bs_reproduces_close_ip_ohmic(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ref_ohm, ref_bs = close_ip("ohmic", Ip_s, c,
                                   lin(j_ind), lin(j_bs), lin(j_fix))
        out = close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, basis=_CONST,
            weights=dict(name="pin-bs", ind=(1.0,), bs=(np.inf,)))
        np.testing.assert_allclose(out["s_ind"], ref_ohm, rtol=1e-12)
        np.testing.assert_allclose(out["s_bs"], ref_bs, rtol=0, atol=1e-15)

    @pytest.mark.parametrize("ratio", [1e2, 1e4, 1e6, 1e8, 1e10])
    def test_large_finite_W_ind_converges_to_the_bootstrap_channel(self, ratio):
        """The inf shortcut is a LIMIT, not a special case."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ref_ohm, ref_bs = close_ip("bootstrap", Ip_s, c,
                                   lin(j_ind), lin(j_bs), lin(j_fix))
        out = close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, basis=_CONST,
            weights=dict(name="soft-pin", ind=(ratio,), bs=(1.0,)))
        # error falls like 1/ratio; assert the bound rather than a fixed number
        assert abs(float(out["s_ind"][0]) - ref_ohm) < 50.0 / ratio
        assert abs(float(out["s_bs"][0]) - ref_bs) < 50.0 / ratio

    def test_convergence_to_the_scalar_answer_is_monotone_in_W(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ref_ohm, _ = close_ip("bootstrap", Ip_s, c,
                              lin(j_ind), lin(j_bs), lin(j_fix))
        errs = []
        for ratio in (1e2, 1e3, 1e4, 1e5, 1e6):
            out = close_ip_structured(
                psi, w, c, Ip_s, j_ind, j_bs, j_fix, basis=_CONST,
                weights=dict(name="soft-pin", ind=(ratio,), bs=(1.0,)))
            errs.append(abs(float(out["s_ind"][0]) - ref_ohm))
        assert all(b < a for a, b in zip(errs, errs[1:])), errs

    @pytest.mark.parametrize("wts", [
        dict(name="a", ind=(1.0,), bs=(1.0,)),
        dict(name="b", ind=(1000.0,), bs=(0.001,)),
        dict(name="c", ind=(0.01,), bs=(7.0,)),
    ])
    def test_constant_basis_with_axis_row_reproduces_close_ip_q0(self, wts):
        """Two constraints, two unknowns: the objective cannot change the
        answer, so EVERY weight set must land on close_ip_q0 exactly."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix)
        ref_ohm, ref_bs = close_ip_q0(
            Ip_s, c, lin(j_ind), lin(j_bs), lin(j_fix),
            ax["j_ind0"], ax["j_bs0"], ax["j_fix0"], ax["j_ref0"])
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  basis=_CONST, weights=wts, axis=ax)
        np.testing.assert_allclose(out["s_ind"], ref_ohm, rtol=1e-10)
        np.testing.assert_allclose(out["s_bs"], ref_bs, rtol=1e-10)
        assert out["axis_residual"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
class TestIpIsExactByConstruction:
    @pytest.mark.parametrize("wts", [STRUCTURED_WEIGHTS_PHYSICS,
                                     STRUCTURED_WEIGHTS_UNIFORM])
    @pytest.mark.parametrize("with_axis", [False, True])
    def test_closed_hybrid_integrates_to_Ip(self, wts, with_axis):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix) if with_axis else None
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  weights=wts, axis=ax)
        direct = lin(out["s_ind"] * j_ind + out["s_bs"] * j_bs + j_fix) + c
        assert direct == pytest.approx(Ip_s, rel=1e-12)
        assert abs(out["ip_residual_pct"]) < 1e-9
        assert out["constraints"][0] == "Ip"
        assert (len(out["constraints"]) == 2) is with_axis

    def test_effective_scalars_satisfy_the_same_scalar_closure(self):
        """This identity is what makes closure_health applicable unchanged."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  axis=_axis(psi, j_ind, j_bs, j_fix))
        closed = (out["ohm_scale_eff"] * lin(j_ind)
                  + out["bs_scale_eff"] * lin(j_bs) + lin(j_fix) + c)
        assert closed == pytest.approx(Ip_s, rel=1e-12)
        h = closure_health(out["ohm_scale_eff"], out["bs_scale_eff"], Ip_s, c,
                           lin(j_ind), lin(j_bs), lin(j_fix))
        assert set(h) >= {"closure_limited", "f_BS_closed", "f_BS_unscaled"}
        assert h["f_BS_closed"] == pytest.approx(
            abs(lin(out["s_bs"] * j_bs)) / abs(Ip_s), rel=1e-12)

    def test_axis_row_is_matched_exactly(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix)
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix, axis=ax)
        phi0 = out["basis_phi0"]
        j0 = ((1.0 + out["a"] @ phi0) * ax["j_ind0"]
              + (1.0 + out["b"] @ phi0) * ax["j_bs0"] + ax["j_fix0"])
        assert j0 == pytest.approx(ax["j_ref0"], rel=1e-10)


# ---------------------------------------------------------------------------
class TestItIsTheNormMinimiser:
    def _objective(self, out, a, b):
        return float(np.sum(out["weights_ind"] * a ** 2)
                     + np.sum(out["weights_bs"] * b ** 2))

    def test_matches_scipy_constrained_minimisation(self):
        opt = pytest.importorskip("scipy.optimize")
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix)
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix, axis=ax)
        K = out["basis_K"]
        Phi = structured_basis_eval(out["basis"], psi)
        phi0 = out["basis_phi0"]
        W = np.concatenate([out["weights_ind"], out["weights_bs"]])

        def f(x):
            return float(np.sum(W * x ** 2))

        def con_ip(x):
            s_i = 1.0 + x[:K] @ Phi
            s_b = 1.0 + x[K:] @ Phi
            return (lin(s_i * j_ind + s_b * j_bs + j_fix) + c - Ip_s) / abs(Ip_s)

        def con_ax(x):
            return ((1.0 + x[:K] @ phi0) * ax["j_ind0"]
                    + (1.0 + x[K:] @ phi0) * ax["j_bs0"]
                    + ax["j_fix0"] - ax["j_ref0"]) / abs(ax["j_ref0"])

        r = opt.minimize(f, np.zeros(2 * K), method="SLSQP",
                         constraints=[{"type": "eq", "fun": con_ip},
                                      {"type": "eq", "fun": con_ax}],
                         options=dict(maxiter=800, ftol=1e-14))
        assert r.success, r.message
        kkt = self._objective(out, out["a"], out["b"])
        # SLSQP can only tie or lose against the exact KKT minimiser
        assert kkt <= f(r.x) * (1.0 + 1e-6)
        np.testing.assert_allclose(np.concatenate([out["a"], out["b"]]), r.x,
                                   rtol=2e-4, atol=2e-6)

    def test_no_feasible_perturbation_lowers_the_objective(self):
        """Direct null-space search: move along the constraint manifold and
        the trust-weighted norm can only go up."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix)
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix, axis=ax)
        K = out["basis_K"]
        Phi = structured_basis_eval(out["basis"], psi)
        phi0 = out["basis_phi0"]
        C = np.vstack([
            np.concatenate([[lin(Phi[k] * j_ind) for k in range(K)],
                            [lin(Phi[k] * j_bs) for k in range(K)]]),
            np.concatenate([phi0 * ax["j_ind0"], phi0 * ax["j_bs0"]])])
        # null space of C: directions that keep BOTH constraints satisfied
        _, _, Vt = np.linalg.svd(C)
        null = Vt[C.shape[0]:]
        x0 = np.concatenate([out["a"], out["b"]])
        base = self._objective(out, out["a"], out["b"])
        rng = np.random.default_rng(11)
        for _ in range(300):
            d = null.T @ rng.standard_normal(null.shape[0])
            for eps in (1e-3, 1e-2, 1e-1):
                x = x0 + eps * d / max(np.linalg.norm(d), 1e-300)
                assert self._objective(out, x[:K], x[K:]) >= base - 1e-12 * abs(base)

    def test_the_physics_prior_moves_structure_where_it_says_it_does(self):
        """Trusting the ohmic in the core must push the CORE correction onto
        the bootstrap, and vice versa at the edge -- otherwise the weights are
        decoration."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        phys = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                   weights=STRUCTURED_WEIGHTS_PHYSICS)
        unif = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                   weights=STRUCTURED_WEIGHTS_UNIFORM)
        # innermost coefficient: the physics prior lets j_ind move LESS
        assert abs(phys["a"][0]) < abs(unif["a"][0])
        # outermost coefficient: the physics prior lets j_BS move LESS
        assert abs(phys["b"][-1]) < abs(unif["b"][-1])


# ---------------------------------------------------------------------------
class TestStructureNumber:
    def test_zero_for_a_constant_multiplier(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        out = close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, basis=_CONST,
            weights=dict(name="pin-ind", ind=(np.inf,), bs=(1.0,)))
        assert out["structure_bs"] == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_when_the_closure_is_genuinely_radial(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  axis=_axis(psi, j_ind, j_bs, j_fix))
        assert out["structure_ind"] > 1e-3 or out["structure_bs"] > 1e-3
        assert np.isfinite(out["structure_ind"])
        assert np.isfinite(out["structure_bs"])


# ---------------------------------------------------------------------------
class TestKKTScaling:
    """The degeneracy test is about the CONSTRAINT ROWS, not about the prior.

    The refusal used to be taken on the bordered matrix
    ``[[2 diag(W/max W), Cn'], [Cn, 0]]``, whose smallest singular value is
    bounded by the prior's own dynamic range -- so any ``W_max/W_min`` past
    ~``1/cond_rtol`` was reported as "the constraint rows are degenerate".
    That is a false refusal AND a false diagnosis, and it capped the branch's
    asymmetric prior at a sigma ratio of a few hundred.

    Three things are pinned here: genuinely degenerate rows are STILL refused
    (at any weight range), a wide weight range is solved exactly, and the
    answer on well-conditioned cases is the same minimiser the bordered system
    gave.  ``cond_rtol`` itself is unchanged.
    """

    def _rows(self, psi, w, c, j_ind, j_bs, j_fix, Ip_s, ax):
        """The SAME constraint rows close_ip_structured builds, from the
        shipped basis evaluator -- so the reference below is a reference for
        the SOLVE, not a second implementation of the rows."""
        Phi = structured_basis_eval(dict(STRUCTURED_BASIS_DEFAULT), psi)
        K = Phi.shape[0]
        _lin = lambda y: float(trapezoid(w * np.asarray(y, float), psi))
        A = np.array([_lin(Phi[k] * j_ind) for k in range(K)])
        B = np.array([_lin(Phi[k] * j_bs) for k in range(K)])
        deficit = Ip_s - c - _lin(j_ind) - _lin(j_bs) - _lin(j_fix)
        phi0 = structured_basis_eval(dict(STRUCTURED_BASIS_DEFAULT),
                                     np.array([ax["psi"]]))[:, 0]
        C = np.array([np.concatenate([A, B]),
                      np.concatenate([phi0 * ax["j_ind0"],
                                      phi0 * ax["j_bs0"]])])
        d = np.array([deficit,
                      ax["j_ref0"] - ax["j_ind0"] - ax["j_bs0"] - ax["j_fix0"]])
        return C, d

    @pytest.mark.parametrize("wt", [
        dict(name="a", ind=(1.0e2, 10.0, 3.0, 1.0), bs=(1.0, 1.0, 1.0, 1.0)),
        dict(name="b", ind=(4.0, 4.0, 4.0, 4.0), bs=(4.0, 11.0, 44.0, 100.0)),
        dict(name="c", ind=(100.0, 6.25, 6.25, 6.25),
             bs=(4.0, 11.0, 44.0, 100.0)),
    ])
    def test_is_the_same_minimiser_the_bordered_system_gave(self, wt):
        """``x = W^-1 C' (C W^-1 C')^-1 d`` is the closed-form stationary point
        of the bordered KKT system.  The substituted solve must reproduce it --
        this is the "unchanged on well-conditioned cases" evidence."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix)
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  axis=ax, weights=wt)
        C, d = self._rows(psi, w, c, j_ind, j_bs, j_fix, Ip_s, ax)
        Wv = np.concatenate([np.asarray(wt["ind"], float),
                             np.asarray(wt["bs"], float)])
        Wi = np.diag(1.0 / Wv)
        x_ref = Wi @ C.T @ np.linalg.solve(C @ Wi @ C.T, d)
        x = np.concatenate([out["a"], out["b"]])
        assert np.max(np.abs(x - x_ref)) <= 1.0e-12 * np.max(np.abs(x_ref))
        # and it is a MINIMISER, not just a feasible point
        assert float(x @ np.diag(Wv) @ x) <= float(
            x_ref @ np.diag(Wv) @ x_ref) * (1.0 + 1.0e-12)

    @pytest.mark.parametrize("ratio", [1.0e3, 1.0e6, 1.0e9, 1.0e12])
    def test_a_wide_trust_weight_range_is_solved_not_refused(self, ratio):
        """``W_max/W_min`` past ~1/cond_rtol used to be refused as a degenerate
        constraint system.  The rows are independent at every ratio and the
        constraints are met exactly.

        The superseded test is rebuilt here and asserted to have refused the
        wide cases -- so this is a regression test against the FIX, not just a
        statement that the current code works."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix)
        wt = dict(name="wide", ind=(1.0, 1.0, 1.0, ratio),
                  bs=(1.0, 1.0, 1.0, 1.0))
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  axis=ax, weights=wt)
        C, d = self._rows(psi, w, c, j_ind, j_bs, j_fix, Ip_s, ax)
        x = np.concatenate([out["a"], out["b"]])
        resid = C @ x - d
        assert np.all(np.abs(resid) <= 1.0e-12 * np.abs(d))
        assert abs(out["ip_residual_pct"]) < 1.0e-9
        # the row conditioning is a property of the ROWS: same at every ratio
        flat = close_ip_structured(
            psi, w, c, Ip_s, j_ind, j_bs, j_fix, axis=ax,
            weights=dict(name="flat", ind=(1.0,) * 4, bs=(1.0,) * 4))
        assert out["constraint_cond"] == pytest.approx(flat["constraint_cond"],
                                                       rel=1e-12)
        # what the bordered matrix would have said
        Wv = np.concatenate([np.asarray(wt["ind"], float),
                             np.asarray(wt["bs"], float)])
        rn = np.max(np.abs(C), axis=1)
        Cn = C / rn[:, None]
        n, m = Cn.shape[1], Cn.shape[0]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = 2.0 * np.diag(Wv / Wv.max())
        M[:n, n:], M[n:, :n] = Cn.T, Cn
        sv_old = np.linalg.svd(M, compute_uv=False)
        would_have_refused = sv_old[-1] <= 1.0e-6 * sv_old[0]
        assert would_have_refused == (ratio >= 1.0e6)

    @pytest.mark.parametrize("ratio", [1.0, 1.0e4, 1.0e10])
    def test_genuinely_degenerate_rows_are_still_refused_at_any_range(self,
                                                                      ratio):
        """The case the refusal is FOR: j_BS == j_ind makes the Ip row and the
        axis row read the same combination.  A wide prior must not rescue it
        and a narrow one must not be needed to catch it."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        j2 = j_ind.copy()
        ax = _axis(psi, j_ind, j2, j_fix)
        with pytest.raises(RuntimeError, match="degenerate on this basis"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j2, j_fix,
                                basis=_CONST, axis=ax,
                                weights=dict(name="u", ind=(ratio,),
                                             bs=(1.0,)))

    def test_a_nearly_degenerate_pair_is_still_refused(self):
        """Rows proportional to within cond_rtol -- the floor itself, not just
        the exactly-singular case."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        j2 = j_ind * (1.0 + 1.0e-9)
        ax = _axis(psi, j_ind, j2, j_fix)
        with pytest.raises(RuntimeError, match="degenerate on this basis"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j2, j_fix,
                                basis=_CONST, axis=ax,
                                weights=dict(name="u", ind=(1.0,), bs=(1.0,)))

    def test_cond_rtol_default_is_unchanged(self):
        """The fix must not move the acceptance value, only what it is
        applied to."""
        import inspect
        from bouquet.utils import close_ip_structured_soft
        for fn in (close_ip_structured, close_ip_structured_soft):
            assert inspect.signature(fn).parameters["cond_rtol"].default \
                == 1.0e-6

    def test_the_row_conditioning_is_reported_separately(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        out = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                  axis=_axis(psi, j_ind, j_bs, j_fix))
        assert out["constraint_cond"] >= 1.0
        assert len(out["constraint_singular_values"]) == 2
        assert np.all(np.isfinite(out["constraint_singular_values"]))


# ---------------------------------------------------------------------------
class TestRefusals:
    def test_refuses_when_a_multiplier_leaves_the_scale_bounds(self):
        """A deficit far too large for the basis to absorb gently."""
        psi, w, c, j_ind, j_bs, j_fix, lin, _ = _parts()
        raw = lin(j_ind) + lin(j_bs) + lin(j_fix) + c
        with pytest.raises(RuntimeError, match="outside \\[0.2, 5\\]"):
            close_ip_structured(psi, w, c, 6.0 * raw, j_ind, j_bs, j_fix)

    def test_refuses_a_degenerate_constraint_pair(self):
        """Ip row and axis row proportional: with j_BS == j_ind the two rows
        both read 'move the same combination', and no (Ip, q0) split exists."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        j2 = j_ind.copy()
        ax = _axis(psi, j_ind, j2, j_fix)
        with pytest.raises(RuntimeError, match="singular KKT"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j2, j_fix,
                                basis=_CONST, axis=ax,
                                weights=dict(name="u", ind=(1.0,), bs=(1.0,)))

    def test_refuses_a_constraint_row_the_free_coefficients_cannot_move(self):
        """j_BS carries no current and only the bs coefficient is free: the Ip
        row is identically zero.  This is close_ip's 'j_BS integrates to ~0'
        refusal, reached through the structured machinery."""
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        with pytest.raises(RuntimeError, match="identically zero"):
            close_ip_structured(
                psi, w, c, Ip_s, j_ind, np.zeros_like(j_bs), j_fix,
                basis=_CONST,
                weights=dict(name="pin-ind", ind=(np.inf,), bs=(1.0,)))

    def test_refuses_when_every_weight_is_infinite(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        with pytest.raises(RuntimeError, match="every trust weight is"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                basis=_CONST,
                                weights=dict(name="all", ind=(np.inf,),
                                             bs=(np.inf,)))

    def test_refuses_non_finite_inputs(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        bad = j_ind.copy(); bad[3] = np.nan
        with pytest.raises(RuntimeError, match="non-finite j_ind"):
            close_ip_structured(psi, w, c, Ip_s, bad, j_bs, j_fix)
        with pytest.raises(RuntimeError, match="non-finite c_affine"):
            close_ip_structured(psi, w, np.nan, Ip_s, j_ind, j_bs, j_fix)

    def test_refuses_a_non_finite_axis_row(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        ax = _axis(psi, j_ind, j_bs, j_fix); ax["j_ref0"] = np.inf
        with pytest.raises(RuntimeError, match="non-finite axis row"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix, axis=ax)

    def test_refuses_bad_weights(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        with pytest.raises(ValueError, match="strictly positive"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                weights=dict(name="neg", ind=(1, 1, 1, -1),
                                             bs=(1, 1, 1, 1)))
        with pytest.raises(ValueError, match="basis functions but"):
            close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                weights=dict(name="short", ind=(1.0, 2.0),
                                             bs=(1.0, 2.0, 3.0, 4.0)))

    def test_refuses_mismatched_grids(self):
        psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
        with pytest.raises(ValueError, match="share one grid"):
            close_ip_structured(psi, w, c, Ip_s, j_ind[:-1], j_bs, j_fix)


# ---------------------------------------------------------------------------
class TestShippedDefaults:
    def test_defaults_are_the_documented_ones(self):
        assert STRUCTURED_BASIS_DEFAULT["kind"] == "gaussian"
        assert tuple(STRUCTURED_BASIS_DEFAULT["centres"]) == \
            (0.15, 0.45, 0.75, 0.95)
        assert tuple(STRUCTURED_BASIS_DEFAULT["widths"]) == (0.2,) * 4
        assert STRUCTURED_WEIGHTS_PHYSICS["name"] == "physics-prior"
        assert STRUCTURED_WEIGHTS_UNIFORM["name"] == "uniform"
        # the prior really is opposite-running: ohmic trusted in, bs trusted out
        wi = np.asarray(STRUCTURED_WEIGHTS_PHYSICS["ind"], float)
        wb = np.asarray(STRUCTURED_WEIGHTS_PHYSICS["bs"], float)
        assert np.all(np.diff(wi) < 0) and np.all(np.diff(wb) > 0)
        assert np.all(np.asarray(STRUCTURED_WEIGHTS_UNIFORM["ind"], float) == 1)

    def test_config_exposes_the_knobs(self):
        from bouquet.config import GenerationConfig
        g = GenerationConfig()
        assert g.closure_channel == "bootstrap"          # default unchanged
        assert g.structured_basis is None                # None -> shipped
        assert g.structured_weights is None

    def test_knobs_round_trip_through_config_serialisation(self):
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t")
        cfg.generation.closure_channel = "structured"
        cfg.generation.structured_basis = dict(kind="gaussian",
                                               centres=[0.2, 0.8],
                                               widths=[0.15, 0.15])
        cfg.generation.structured_weights = dict(name="custom", ind=[5.0, 1.0],
                                                 bs=[1.0, 5.0])
        back = BouquetConfig.from_dict(cfg.to_dict()).generation
        assert back.closure_channel == "structured"
        assert back.structured_basis == cfg.generation.structured_basis
        assert back.structured_weights == cfg.generation.structured_weights


class TestWorkflowWhitelists:
    """"structured" must be accepted by BOTH gates -- the pre-SWB dispatch
    guard and the workflow validator -- or the campaign burns a full
    solve_with_bootstrap sequence per slice and then refuses a valid channel.
    """

    def _config(self, channel):
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t")
        cfg.generation.jBS_baseline_mode = "ohmic"
        cfg.generation.perturb_jind_in_anchor = True
        cfg.generation.workflow = "custom"      # downgrade baseline-only refusal
        cfg.generation.closure_channel = channel
        return cfg

    def _validate(self, cfg):
        from bouquet.run import Bouquet
        Bouquet(cfg)._validate_workflow()

    def test_validator_accepts_structured(self, capsys):
        self._validate(self._config("structured"))
        out = capsys.readouterr().out
        assert "closure_channel" not in out

    def test_validator_still_refuses_a_typo(self):
        cfg = self._config("structured_")
        cfg.generation.workflow = "auto"     # no 'custom' downgrade
        with pytest.raises(ValueError, match="closure_channel"):
            self._validate(cfg)

    def test_presolve_dispatch_guard_lists_structured(self):
        """The guard is a literal tuple inside the ohmic block; assert on the
        source so a future edit that drops the channel is caught here rather
        than three hours into a campaign."""
        import inspect
        from bouquet import run as _run
        src = inspect.getsource(_run)
        assert src.count('"sawtooth_bootstrap",\n                                "structured"') == 2


# ---------------------------------------------------------------------------
#  the post-closure Ip round-trip gate: an ALGEBRA check, not an acceptance
# ---------------------------------------------------------------------------
class TestIpRoundtripGate:
    """``utils.ip_roundtrip_gate`` -- the gate the assembled hybrid must pass.

    The soft structured channel's posterior Ip differs from the measurement BY
    DESIGN, so the gate's reference is the closure's own Ip there.  The
    TOLERANCE is untouched: what changed is what the round trip is measured
    against.  These tests pin both halves of that statement.
    """

    IP = 1.0e6
    SIG = 0.005 * IP          # structured_ip_sigma_frac = 0.005

    def test_the_tolerance_is_unchanged(self):
        from bouquet.utils import IP_ROUNDTRIP_TOL_PCT

        assert IP_ROUNDTRIP_TOL_PCT == 0.05

    def test_soft_posterior_0p3pct_off_the_measurement_passes(self):
        """The case the old gate refused: a posterior 0.3 % from Ip_meas whose
        assembled hybrid reproduces THAT posterior exactly."""
        from bouquet.utils import ip_roundtrip_gate

        post = self.IP * 1.003
        g = ip_roundtrip_gate(post, self.IP, posterior=post, sigma_Ip=self.SIG)
        assert abs(g["err_pct"]) < 1e-9
        assert g["reference"] == pytest.approx(post, rel=1e-15)
        assert g["measured_residual_pct"] == pytest.approx(0.3, rel=1e-9)
        assert g["residual_sigma_Ip"] == pytest.approx(0.6, rel=1e-9)

    def test_the_same_number_on_a_hard_channel_still_refuses(self):
        """No posterior -> the reference is the measurement and the message is
        the one it always was."""
        from bouquet.utils import ip_roundtrip_gate

        with pytest.raises(RuntimeError, match="of Ip_target after closure "
                                               "-- algebra error, refusing"):
            ip_roundtrip_gate(self.IP * 1.003, self.IP)

    def test_a_real_assembly_error_is_still_refused_on_the_soft_channel(self):
        """The reference moved; the budget did not.  A hybrid that misses its
        OWN posterior by 0.06 % is still an algebra error."""
        from bouquet.utils import ip_roundtrip_gate

        post = self.IP * 1.003
        with pytest.raises(RuntimeError, match="posterior Ip after closure"):
            ip_roundtrip_gate(post * 1.0006, self.IP, posterior=post,
                              sigma_Ip=self.SIG)

    def test_the_achieved_sigma_is_the_delivered_hybrids_own_distance(self):
        """``residual_sigma_Ip`` is ACHIEVED: it reads off the closed hybrid,
        not off the posterior the closure predicted."""
        from bouquet.utils import ip_roundtrip_gate

        post = self.IP * 1.003
        g = ip_roundtrip_gate(post * 1.0002, self.IP, posterior=post,
                              sigma_Ip=self.SIG)
        assert g["residual_sigma_Ip"] == pytest.approx(
            (post * 1.0002 - self.IP) / self.SIG, rel=1e-9)
        assert g["measured_residual_pct"] == pytest.approx(0.3, rel=1e-9)

    def test_a_posterior_equal_to_the_measurement_is_the_old_gate(self):
        from bouquet.utils import ip_roundtrip_gate

        g = ip_roundtrip_gate(self.IP * 1.0004, self.IP, posterior=self.IP)
        assert g["err_pct"] == pytest.approx(0.04, rel=1e-9)
        assert g["residual_sigma_Ip"] is None

    def test_beyond_one_sigma_is_a_flag_not_a_refusal(self):
        """Health, not acceptance: the slice is delivered and marked."""
        from bouquet.utils import (SOFT_IP_FLAG_PREFIX, closure_health,
                                   ip_roundtrip_gate)

        post = self.IP * 1.008                       # 1.6 sigma
        g = ip_roundtrip_gate(post, self.IP, posterior=post, sigma_Ip=self.SIG)
        assert g["residual_sigma_Ip"] == pytest.approx(1.6, rel=1e-9)
        h = closure_health(1.0, 1.0, self.IP, 0.0, 0.6e6, 0.2e6, 0.2e6,
                           soft_ip_residual_sigma=g["residual_sigma_Ip"])
        assert h["closure_limited"] is True
        assert any(str(r).startswith(SOFT_IP_FLAG_PREFIX)
                   for r in h["closure_limited_reasons"])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"),
                                     -float("inf")])
    def test_a_non_finite_round_trip_is_refused_not_passed(self, bad):
        """``abs(nan) > tol`` is False.  A NaN in the assembled profile -- the
        exact class of defect this gate exists to catch -- used to pass it and
        be recorded as ``err_pct = nan``."""
        from bouquet.utils import ip_roundtrip_gate

        with pytest.raises(RuntimeError, match="refusing"):
            ip_roundtrip_gate(bad, self.IP)
        with pytest.raises(RuntimeError, match="refusing"):
            ip_roundtrip_gate(bad, self.IP, posterior=self.IP,
                              sigma_Ip=self.SIG)

    @pytest.mark.parametrize("bad", [float("nan"), 0.0])
    def test_an_unusable_reference_is_refused(self, bad):
        from bouquet.utils import ip_roundtrip_gate

        with pytest.raises(RuntimeError, match="no usable reference|refusing"):
            ip_roundtrip_gate(self.IP, bad)

    def test_a_non_finite_posterior_falls_back_to_the_measurement(self):
        """Already guarded by ``np.isfinite`` -- pinned so the NaN hardening
        above cannot accidentally turn it into a refusal."""
        from bouquet.utils import ip_roundtrip_gate

        g = ip_roundtrip_gate(self.IP, self.IP, posterior=float("nan"))
        assert g["reference_name"] == "Ip_target"
        assert g["err_pct"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("kw", [
        dict(ohm_scale=float("nan")), dict(bs_scale=float("nan")),
        dict(ip_bs=float("nan")), dict(c_affine=float("inf")),
    ])
    def test_closure_health_flags_a_non_finite_input(self, kw):
        """Same hole, same shape: every test in closure_health is
        ``abs(x) > threshold``, so a NaN scale gave closure_limited=False with
        f_BS_closed=nan and an EMPTY reason tuple."""
        from bouquet.utils import closure_health

        base = dict(ohm_scale=1.0, bs_scale=1.0, Ip_target_signed=self.IP,
                    c_affine=0.0, ip_ind=0.6e6, ip_bs=0.2e6, ip_fix=0.2e6)
        base.update(kw)
        h = closure_health(**base)
        assert h["closure_limited"] is True
        assert any("unreadable" in str(r)
                   for r in h["closure_limited_reasons"])

    def test_inside_one_sigma_is_not_flagged(self):
        from bouquet.utils import closure_health

        h = closure_health(1.0, 1.0, self.IP, 0.0, 0.6e6, 0.2e6, 0.2e6,
                           soft_ip_residual_sigma=0.6)
        assert h["closure_limited"] is False
        assert h["closure_limited_reasons"] == ()

    def test_hard_channels_are_untouched_by_the_new_argument(self):
        """Default ``None`` -> the record every other channel has always had."""
        from bouquet.utils import closure_health

        args = (0.9, 0.8, self.IP, 0.0, 0.6e6, 0.2e6, 0.2e6)
        assert closure_health(*args) == closure_health(
            *args, soft_ip_residual_sigma=None)

    def test_the_run_gate_passes_a_posterior_only_on_the_soft_channel(self):
        """The reference switch is guarded by ``_soft_ip``; assert on the
        source so a future edit cannot quietly widen it to hard channels."""
        import inspect

        from bouquet import run as _run

        src = inspect.getsource(_run)
        assert 'posterior=(_q0_extra.get("structured_ip_posterior")' in src
        assert "if _soft_ip else None)" in src
