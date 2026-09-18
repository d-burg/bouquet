"""l_i as the structured closure's SECOND global measurement.

``closure_channel="structured"`` closes Ip exactly and picks, among the
profiles that do, the one that departs least from "trust the sources".  Ip is
ONE number against 2K coefficients: it cannot see radial redistribution, which
is precisely what the closure-cloud campaign found it gets wrong.  ``l_i`` is
the other global number a magnetics reconstruction reports and it sees exactly
that, so it is added here in two forms -- HARD
(:func:`~bouquet.utils.close_ip_structured` with ``li_target``) and SOFT
(:func:`~bouquet.utils.close_ip_structured_soft`, the posterior mode of Ip and
l_i as Gaussian measurements).

What is proved here:

* **the discrete l_i form is EXACT, not a proxy.**  With psi the per-radian
  poloidal flux, Ampere's law on a flux surface gives ``<B_p^2> = 2 pi mu0
  I(psi) / V'``, so ``int B_p^2 dV = 2 pi mu0 int I dpsi`` with the
  flux-surface geometry cancelling identically.  The solver-marked tests at the
  bottom check that against TokaMaker's own ``get_stats`` on the synthetic
  D3D-like anchor: measured **+0.0208 % on Bp_vol and +0.0215 % on both l_i
  normalisations** for the equilibrium's own GS current profile, against a
  1 % bar.
* **which perimeter is used is load-bearing.**  ``l_i(1) ~ L^2``, and
  ``get_stats``' ``dl`` sits +0.8 % above the traced LCFS contour's own
  perimeter on this anchor (+1.6 % on real diverted equilibria), i.e. 0.013
  (0.032) of l_i(1).  An EFIT a-file ``LI`` is l_i(1) on the CONTOUR
  perimeter, so that is what the closure measures; a solver test pins the
  convention and the ratio.  Calibrated out-of-tree against an independent
  g-file l_i estimator (one estimator on both equilibria) on the
  anchor's saved g-file: closure l_i(1) **+0.00038**, the ``li_achieved``
  readback **+0.00020**, l_i(3) +0.0013 / +0.0012 -- against **+0.01337** for
  ``get_stats('std')``.
* **the hard row is exact, not merely linearised.**  Because the Ip row pins
  ``Ip(x)``, ``l_i = G S(x) sgn / Ip^2`` is LINEAR in the coefficients, so the
  "linearisation about s == 1" taken along the Ip-closed manifold hits the
  target to machine precision.  Linearising about the anchor's own ``Ip0``
  instead would leave the ``(Ip0/Ip_target)^2`` mismatch in the row -- 0.3 % of
  l_i on a 4 % Ip deficit, which is tested for explicitly.
* **the analytic gradient matches finite differences**, at the anchor and away
  from it, in both normalisations.
* **the soft solver CONTAINS the hard one and the scalar channels**:
  ``Ip_sigma -> 0`` (and ``Ip_sigma=None``) reproduces
  ``close_ip_structured`` to 1e-10; the constant basis with a pinned prior
  reproduces ``close_ip("bootstrap")`` / ``close_ip("ohmic")`` and, with the
  axis row, ``close_ip_q0``; ``li_sigma -> 0`` reproduces the hard l_i answer.
* **the soft answer IS the posterior mode** -- checked against
  ``scipy.optimize``.
* **the refusals fire**: an unreachable l_i target, a missing ``li_geom``, an
  unknown ``li_kind``, a soft l_i with no sigma, bad sigmas, a degenerate hard
  constraint system.

Only the last class needs a solver; everything else is pure NumPy on the same
synthetic ``fsa_current_geometry``-shaped dict the other closure tests use.
"""
import json
import os
import subprocess
import sys

# Must precede the `bouquet` import below: puts the repo root on sys.path so a
# probe invoked directly still exercises THIS tree.  See tests/_harness.py.
import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from bouquet.utils import (Ip_fsa_affine_profile, Ip_fsa_weights, close_ip,
                           close_ip_q0, close_ip_structured,
                           close_ip_structured_soft, li_closure_geometry,
                           li_prefactor, li_value, sigma_from_weights,
                           structured_basis_eval, structured_li_gradient,
                           structured_li_model, structured_li_of,
                           LI_KINDS, STRUCTURED_WEIGHTS_PHYSICS,
                           STRUCTURED_WEIGHTS_UNIFORM)
from test_structured_closure import _geom, _axis, _CONST

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))

#: The acceptance the l_i identity had to clear before being wired in: the
#: discrete form must reproduce TokaMaker's OWN get_stats l_i on the anchor.
#: Measured +0.0208-0.0215 %.  TIGHTENED from 1e-2 to 1e-3 (adversarial
#: review): at 1 % the bar sat ~46x above the physics and a 0.5 % drift in the
#: psi Jacobian, the volume read or the perimeter would have passed silently.
#: 1e-3 keeps a ~4.6x margin over the measured agreement and was verified
#: against a live solver run of every assertion below.  Not to be widened: a
#: miss means the 2 pi mu0 identity, the psi Jacobian or the
#: (vol, perimeter, R_axis) read has moved.
_LI_SELF_CONSISTENCY = 1.0e-3

#: A DIFFERENT comparison: the ARCHIVED baseline total against the
#: equilibrium's own GS profile.  These are two profiles that genuinely
#: differ, so this is not the identity above and must not borrow its bar --
#: sharing one constant is what forced that constant up to 46x the physics.
#: Measured 0.1506 % on l_i(3) in a live solver run, with l_i(1) inside the
#: same bar.  TIGHTENED from the shared 1e-2 to 3e-3, ~2x the measurement.
_LI_ARCHIVED_AGREEMENT = 3.0e-3


def _oft_importable():
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  fixture: the same synthetic geometry the structured tests use, with the
#  current amplitudes scaled so l_i lands in the physical range
# ---------------------------------------------------------------------------
#: ``l_i`` depends on the current amplitude only (``l_i ~ 1/j``: scaling
#: dpsi/dpsi_N or the volume element scales I and Ip together and cancels), and
#: ``test_structured_closure``'s geometry fixture is not calibrated against its
#: current fixture.  So the components carry a single overall factor chosen once
#: to put ``l_i(1)`` near 0.69 -- an ordinary H-mode value -- which is what makes
#: a +-2 % l_i target a small perturbation here as it is on a real slice.
_LI_CURRENT_SCALE = 11.0


def _li_parts(scale=_LI_CURRENT_SCALE, bs_amp=3.0e5):
    """``(psi, w, c, j_ind, j_bs, j_fix, lin, Ip_signed, li_geom)``.

    ``j_ind`` carries a finite EDGE value (unlike the structured fixture's
    ``(1-psi)^1.5``): a component that vanishes at the boundary makes its own
    multiplier there unbounded for any finite current change, which is a
    property of that fixture and not of the closure.
    """
    g = dict(_geom())
    g["pprime"] = g["pprime"] * scale          # keep the affine term in step
    psi = g["psi_N"]
    w, c = Ip_fsa_weights(g, convention="jphi-linterp")
    j_ind = scale * (8.0e5 * (1.0 - psi) ** 1.5 + 8.0e4)
    j_bs = scale * (bs_amp * np.exp(-((psi - 0.9) / 0.06) ** 2)
                    + 2.0e4 * (1.0 - psi))
    j_fix = scale * (1.0e5 * (1.0 - psi) ** 3 + 1.0e4)
    lin = lambda j: float(trapezoid(w * np.asarray(j, float), psi))  # noqa: E731
    Ip_signed = 1.04 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
    li_geom = dict(
        psi_N=psi, dpsi_dpsiN=g["dpsi_dpsiN"],
        # self-consistent with the fixture's own circular geometry
        vol=float(trapezoid(g["dV_dpsi"], psi) * g["dpsi_dpsiN"]),
        perimeter=2.0 * np.pi * 0.6, R_axis=1.7,
        affine_cum=Ip_fsa_affine_profile(g, convention="jphi-linterp"),
        psi_pad=1.0e-3,
    )
    return psi, w, c, j_ind, j_bs, j_fix, lin, Ip_signed, li_geom


def _model(kind="li_1", basis=None):
    psi, w, c, ji, jb, jf, lin, Ip_s, lg = _li_parts()
    Phi = structured_basis_eval(basis, psi)
    return (structured_li_model(psi, w, Phi, ji, jb, jf, lg, Ip_s, kind),
            (psi, w, c, ji, jb, jf, lin, Ip_s, lg))


def _li_of_closure(out, model):
    """The exact model l_i of a returned closure's coefficients."""
    return structured_li_of(model, np.concatenate([out["a"], out["b"]]))[0]


# ---------------------------------------------------------------------------
class TestAffineProfile:
    def test_cumulative_affine_ends_at_the_scalar_c(self):
        g = _geom()
        _w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        prof = Ip_fsa_affine_profile(g, convention="jphi-linterp")
        assert prof[0] == 0.0
        assert prof[-1] == pytest.approx(c, rel=1e-12)

    def test_fsa_convention_has_no_affine_profile(self):
        prof = Ip_fsa_affine_profile(_geom(), convention="fsa")
        np.testing.assert_array_equal(prof, 0.0)

    def test_pprime_sign_flips_it(self):
        g = _geom()
        a = Ip_fsa_affine_profile(g, pprime_sign=1.0)
        b = Ip_fsa_affine_profile(g, pprime_sign=-1.0)
        np.testing.assert_allclose(a, -b, rtol=1e-14)

    @pytest.mark.parametrize("bad, match", [
        (dict(inv_R2=None), "<1/R\\^2>"),
        (dict(pprime=None), "P'"),
    ])
    def test_missing_pieces_raise(self, bad, match):
        with pytest.raises(ValueError, match=match):
            Ip_fsa_affine_profile(dict(_geom(), **bad))

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="unknown convention"):
            Ip_fsa_affine_profile(_geom(), convention="area")


# ---------------------------------------------------------------------------
class TestTheDiscreteForm:
    """The algebra behind ``int B_p^2 dV = 2 pi mu0 int I dpsi``."""

    def test_li_value_matches_the_coefficient_model_at_s_equals_one(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        I0 = cumulative_trapezoid(w * (ji + jb + jf), psi,
                                  initial=0.0) + lg["affine_cum"]
        assert li_value(psi, I0, lg, "li_1") == pytest.approx(
            structured_li_of(m)[0], rel=1e-14)

    def test_the_two_normalisations_differ_only_by_their_prefactor(self):
        m1, (_psi, _w, _c, _ji, _jb, _jf, _lin, _Ip, lg) = _model("li_1")
        m3, _ = _model("li_3")
        ratio = li_prefactor(lg, "li_3") / li_prefactor(lg, "li_1")
        assert structured_li_of(m3)[0] == pytest.approx(
            structured_li_of(m1)[0] * ratio, rel=1e-14)
        # and the prefactors are the documented ones
        mu0 = 4.0e-7 * np.pi
        assert li_prefactor(lg, "li_1") == pytest.approx(
            2.0 * np.pi * lg["perimeter"] ** 2 / (mu0 * lg["vol"]), rel=1e-14)
        assert li_prefactor(lg, "li_3") == pytest.approx(
            4.0 * np.pi / (mu0 * lg["R_axis"]), rel=1e-14)

    def test_li_is_positive_for_a_negative_current_convention(self):
        """q and Ip carry a COCOS sign; l_i must not."""
        psi, w, c, ji, jb, jf, lin, Ip_s, lg = _li_parts()
        Phi = structured_basis_eval(None, psi)
        pos = structured_li_of(structured_li_model(
            psi, w, Phi, ji, jb, jf, lg, Ip_s, "li_1"))[0]
        lg_n = dict(lg, affine_cum=-lg["affine_cum"])
        neg = structured_li_of(structured_li_model(
            psi, w, Phi, -ji, -jb, -jf, lg_n, -Ip_s, "li_1"))[0]
        assert pos > 0 and neg > 0
        assert neg == pytest.approx(pos, rel=1e-14)

    def test_li_lands_in_the_physical_range_on_the_fixture(self):
        """Guards the fixture itself: a fixture with l_i ~ 10 would make every
        target below a physical perturbation of it."""
        m1, _ = _model("li_1")
        m3, _ = _model("li_3")
        assert 0.4 < structured_li_of(m1)[0] < 1.5
        assert 0.4 < structured_li_of(m3)[0] < 1.5

    @pytest.mark.parametrize("kind", LI_KINDS)
    def test_gradient_matches_finite_differences(self, kind):
        m, _ = _model(kind)
        rng = np.random.default_rng(11)
        for x0 in (np.zeros(8), 0.02 * rng.standard_normal(8)):
            g = structured_li_gradient(m, x0)
            h = 1.0e-6
            fd = np.empty(8)
            for k in range(8):
                e = np.zeros(8); e[k] = h
                fd[k] = (structured_li_of(m, x0 + e)[0]
                         - structured_li_of(m, x0 - e)[0]) / (2.0 * h)
            np.testing.assert_allclose(g, fd, rtol=2e-6)

    def test_the_Ip_response_term_is_what_makes_li_nonlinear(self):
        """With Ip held fixed l_i is exactly linear in the coefficients; the
        second term of the gradient is the whole nonlinearity."""
        m, _ = _model()
        rng = np.random.default_rng(3)
        x = 0.05 * rng.standard_normal(8)
        # a direction that does not move Ip is a direction along which the
        # exact l_i is affine
        gIp = np.concatenate([m["Ip_ind"], m["Ip_bs"]])
        d = rng.standard_normal(8)
        d -= gIp * (d @ gIp) / (gIp @ gIp)        # project out the Ip response
        f = lambda t: structured_li_of(m, x + t * d)[0]  # noqa: E731
        lin_check = f(0.0) + 0.5 * (f(1.0) - f(-1.0))
        assert f(1.0) == pytest.approx(lin_check, rel=1e-12)

    def test_the_model_grid_must_match_the_geometry(self):
        psi, w, c, ji, jb, jf, lin, Ip_s, lg = _li_parts()
        Phi = structured_basis_eval(None, psi)
        with pytest.raises(ValueError, match="different grid"):
            structured_li_model(psi, w, Phi, ji, jb, jf,
                                dict(lg, affine_cum=lg["affine_cum"][:-1]),
                                Ip_s)

    def test_unknown_normalisation_raises(self):
        _m, (_psi, _w, _c, _ji, _jb, _jf, _lin, _Ip, lg) = _model()
        with pytest.raises(ValueError, match="unknown li_kind"):
            li_prefactor(lg, "li_2")


# ---------------------------------------------------------------------------
class TestHardLiRow:
    @pytest.mark.parametrize("kind", LI_KINDS)
    @pytest.mark.parametrize("bump", [1.02, 0.98, 0.95])
    def test_the_target_is_hit_exactly(self, kind, bump):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model(kind)
        base = structured_li_of(m)[0]
        out = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                  li_target=base * bump, li_kind=kind,
                                  li_geom=lg)
        # the row, the recorded prediction and an INDEPENDENT re-evaluation of
        # the model on the returned coefficients must all agree
        assert out["li_predicted"] == pytest.approx(base * bump, rel=1e-12)
        assert _li_of_closure(out, m) == pytest.approx(base * bump, rel=1e-12)
        assert abs(out["li_predictor_residual"]) < 1e-12 * base
        # ... and Ip is still exact
        assert abs(out["ip_residual_pct"]) < 1e-10

    def test_the_row_is_exact_not_merely_linearised(self):
        """The Ip-closed-manifold linearisation point is load-bearing.  Taking
        the gradient at the ANCHOR's own Ip0 instead leaves the
        ``(Ip0/Ip_target)^2`` factor in the row; on this fixture's 4 % Ip
        deficit that is a ~0.3 % l_i error, i.e. ~60x the corrector's own
        tolerance budget, spent for nothing."""
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        base, Ip0 = structured_li_of(m)
        target = base * 1.02
        out = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                  li_target=target, li_geom=lg)
        assert abs(_li_of_closure(out, m) / target - 1.0) < 1e-12
        # what the anchor-Ip linearisation would have produced
        naive_rhs = target - base
        grad0 = structured_li_gradient(m)
        x = np.concatenate([out["a"], out["b"]])
        assert abs(grad0 @ x - naive_rhs) / target > 1e-3, (
            "the two linearisation points have become indistinguishable on "
            "this fixture -- give it a real Ip deficit again")

    def test_it_composes_with_the_axis_row(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        ax = _axis(psi, ji, jb, jf, q0_pull=0.99)
        plain = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, axis=ax)
        target = _li_of_closure(plain, m) * 1.01
        out = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, axis=ax,
                                  li_target=target, li_geom=lg)
        assert len(out["constraints"]) == 3
        assert abs(out["ip_residual_pct"]) < 1e-10
        assert abs(out["axis_residual"]) < 1e-6 * abs(ax["j_ref0"])
        assert _li_of_closure(out, m) == pytest.approx(target, rel=1e-12)

    def test_more_rows_than_free_coefficients_is_refused(self):
        """``svd`` returns only min(m, n) singular values, so a test on the
        SMALLEST returned one cannot see an over-constrained system: the
        documented ``{"kind": "constant"}`` one-liner (K = 1, so two
        coefficients) carrying Ip + an axis row + an l_i row is m = 3 > n = 2,
        every returned singular value is comfortably nonzero, and the solve
        would hand back the LEAST-SQUARES answer labelled "hard-KKT" -- with
        Ip no longer exact, which is the one property this channel is built
        on.  Refused on RANK, against the unchanged ``cond_rtol``."""
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model(basis=_CONST)
        ax = _axis(psi, ji, jb, jf, q0_pull=0.99)
        target = structured_li_of(m)[0] * 1.01
        with pytest.raises(RuntimeError, match="singular KKT system"):
            close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, axis=ax,
                                li_target=target, li_geom=lg, basis=_CONST)

    def test_the_same_three_rows_on_a_basis_that_can_carry_them_still_solve(
            self):
        """The well-posed twin of the refusal above: m = 3 <= n = 8 on the
        shipped basis solves, with Ip exact to the bar this module already
        uses."""
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        ax = _axis(psi, ji, jb, jf, q0_pull=0.99)
        plain = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, axis=ax)
        target = _li_of_closure(plain, m) * 1.01
        out = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, axis=ax,
                                  li_target=target, li_geom=lg)
        assert len(out["constraints"]) == 3
        assert abs(out["ip_residual_pct"]) < 1e-10
        assert _li_of_closure(out, m) == pytest.approx(target, rel=1e-12)

    def test_no_li_target_is_bit_identical_to_the_previous_channel(self):
        """The l_i row is opt-in; a caller who does not set it must get the
        SAME answer the channel gave before it existed."""
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        a = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf)
        b = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, li_geom=lg)
        np.testing.assert_array_equal(a["s_ind"], b["s_ind"])
        np.testing.assert_array_equal(a["s_bs"], b["s_bs"])
        assert a["constraints"] == ("Ip",) and a["li_target"] is None

    def test_the_weights_still_decide_where_the_current_comes_from(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        target = structured_li_of(m)[0] * 0.97
        phys = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                   li_target=target, li_geom=lg,
                                   weights=STRUCTURED_WEIGHTS_PHYSICS)
        unif = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                   li_target=target, li_geom=lg,
                                   weights=STRUCTURED_WEIGHTS_UNIFORM)
        # both hit the same target ...
        for out in (phys, unif):
            assert _li_of_closure(out, m) == pytest.approx(target, rel=1e-12)
        # ... by visibly different profiles
        assert np.max(np.abs(phys["s_bs"] - unif["s_bs"])) > 1e-3

    def test_an_unreachable_target_is_refused_not_clamped(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        base = structured_li_of(m)[0]
        with pytest.raises(RuntimeError, match="outside"):
            close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                li_target=base * 1.5, li_geom=lg)

    @pytest.mark.parametrize("kwargs, exc, match", [
        (dict(li_target=0.7), ValueError, "without li_geom"),
        (dict(li_target=0.7, li_kind="li_2"), ValueError, "unknown li_kind"),
        (dict(li_target=np.nan), ValueError, "without li_geom"),
    ])
    def test_bad_li_arguments_raise(self, kwargs, exc, match):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        if "li_geom" not in kwargs and "unknown" in match:
            kwargs = dict(kwargs, li_geom=lg)
        with pytest.raises(exc, match=match):
            close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, **kwargs)

    def test_a_nonfinite_target_raises(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(RuntimeError, match="non-finite li_target"):
            close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                li_target=np.nan, li_geom=lg)


# ---------------------------------------------------------------------------
class TestSigmaLadder:
    def test_sigma_is_the_inverse_square_root_of_the_weight(self):
        s = sigma_from_weights(STRUCTURED_WEIGHTS_PHYSICS, 4)
        np.testing.assert_allclose(
            s["ind"], 1.0 / np.sqrt(np.asarray(
                STRUCTURED_WEIGHTS_PHYSICS["ind"], float)), rtol=1e-14)
        assert s["name"] == "physics-prior"

    def test_an_infinite_weight_becomes_a_zero_sigma(self):
        s = sigma_from_weights(dict(name="pin", ind=(np.inf,), bs=(1.0,)), 1)
        assert s["ind"][0] == 0.0 and s["bs"][0] == 1.0

    def test_a_zero_weight_becomes_an_infinite_sigma(self):
        s = sigma_from_weights(dict(name="free", ind=(0.0,), bs=(4.0,)), 1)
        assert np.isinf(s["ind"][0]) and s["bs"][0] == 0.5

    def test_negative_weights_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            sigma_from_weights(dict(name="bad", ind=(-1.0,), bs=(1.0,)), 1)


# ---------------------------------------------------------------------------
class TestSoftContainsTheHardSolvers:
    """The soft channel must be a strict generalisation, not a rival answer."""

    def test_hard_Ip_reproduces_close_ip_structured_exactly(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        hard = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf)
        soft = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf)
        np.testing.assert_allclose(soft["s_ind"], hard["s_ind"], rtol=0,
                                   atol=1e-12)
        np.testing.assert_allclose(soft["s_bs"], hard["s_bs"], rtol=0,
                                   atol=1e-12)

    def test_sigma_Ip_to_zero_converges_to_the_hard_answer(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        hard = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf)
        errs = []
        for frac in (1e-2, 1e-4, 1e-6, 1e-8):
            soft = close_ip_structured_soft(psi, w, c, Ip_s,
                                            frac * abs(Ip_s), ji, jb, jf)
            errs.append(max(
                float(np.max(np.abs(soft["s_ind"] - hard["s_ind"]))),
                float(np.max(np.abs(soft["s_bs"] - hard["s_bs"])))))
        # quadratic in sigma_Ip (3e-4 -> 3e-8 -> 3e-12) until it reaches the
        # floating-point floor, which it does by sigma_Ip = 1e-6 Ip; past that
        # the remaining wobble is rounding, not a different answer.
        assert errs[0] > errs[1] > errs[2], errs
        assert max(errs[2:]) < 1e-10, errs

    def test_constant_basis_with_a_pinned_inductive_is_close_ip_bootstrap(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        ref_ohm, ref_bs = close_ip("bootstrap", Ip_s, c,
                                   lin(ji), lin(jb), lin(jf))
        out = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                       basis=_CONST, sigma_ind=(0.0,),
                                       sigma_bs=(1.0,))
        np.testing.assert_allclose(out["s_ind"], ref_ohm, rtol=0, atol=1e-15)
        np.testing.assert_allclose(out["s_bs"], ref_bs, rtol=1e-12)

    def test_constant_basis_with_a_pinned_bootstrap_is_close_ip_ohmic(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        ref_ohm, ref_bs = close_ip("ohmic", Ip_s, c, lin(ji), lin(jb), lin(jf))
        out = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                       basis=_CONST, sigma_ind=(1.0,),
                                       sigma_bs=(0.0,))
        np.testing.assert_allclose(out["s_ind"], ref_ohm, rtol=1e-12)
        np.testing.assert_allclose(out["s_bs"], ref_bs, rtol=0, atol=1e-15)

    def test_constant_basis_with_a_hard_axis_row_is_close_ip_q0(self):
        """Two hard rows on two unknowns: the prior has nothing left to choose,
        so EVERY sigma ladder must give close_ip_q0's answer."""
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        ax = _axis(psi, ji, jb, jf, q0_pull=0.99)
        ref = close_ip_q0(Ip_s, c, lin(ji), lin(jb), lin(jf),
                          ax["j_ind0"], ax["j_bs0"], ax["j_fix0"],
                          ax["j_ref0"])
        for sig in ((1.0,), (0.1,), (7.0,)):
            out = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                           basis=_CONST, sigma_ind=sig,
                                           sigma_bs=sig, axis=ax,
                                           axis_sigma=None)
            assert out["s_ind"][0] == pytest.approx(ref[0], rel=1e-12), sig
            assert out["s_bs"][0] == pytest.approx(ref[1], rel=1e-12), sig

    def test_sigma_li_to_zero_converges_to_the_hard_li_answer(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        target = structured_li_of(m)[0] * 0.98
        hard = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf,
                                   li_target=target, li_geom=lg)
        errs = []
        for s in (1e-2, 1e-4, 1e-6):
            soft = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                            li_target=target, li_sigma=s,
                                            li_geom=lg)
            errs.append(float(np.max(np.abs(soft["s_ind"] - hard["s_ind"]))))
        assert all(b < a for a, b in zip(errs, errs[1:])), errs
        assert errs[-1] < 1e-8, errs


# ---------------------------------------------------------------------------
class TestSoftIsThePosteriorMode:
    def test_a_synthetic_case_with_a_known_li_reaches_it(self):
        """Tight sigmas on both measurements: the posterior mode must sit on
        the l_i that generated the case, and on Ip."""
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        truth = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf)
        li_true = _li_of_closure(truth, m)
        out = close_ip_structured_soft(psi, w, c, Ip_s, 1e-6 * abs(Ip_s),
                                       ji, jb, jf, li_target=li_true,
                                       li_sigma=1e-6, li_geom=lg)
        assert _li_of_closure(out, m) == pytest.approx(li_true, rel=1e-8)
        assert abs(out["ip_residual_pct"]) < 1e-6

    def test_the_returned_point_minimises_the_stated_objective(self):
        """Against scipy, on the full 4-Gaussian basis with both measurements
        soft -- the objective is written out here independently of the solver."""
        from scipy.optimize import minimize

        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        li_t = structured_li_of(m)[0] * 1.03
        s_ip, s_li = 0.004 * abs(Ip_s), 0.02
        sig = sigma_from_weights(STRUCTURED_WEIGHTS_PHYSICS, 4)
        out = close_ip_structured_soft(psi, w, c, Ip_s, s_ip, ji, jb, jf,
                                       sigma_ind=sig["ind"],
                                       sigma_bs=sig["bs"], li_target=li_t,
                                       li_sigma=s_li, li_geom=lg)

        sigv = np.concatenate([sig["ind"], sig["bs"]])

        def cost(x):
            li_x, ip_x = structured_li_of(m, x)
            return (float(np.sum((x / sigv) ** 2))
                    + ((ip_x - Ip_s) / s_ip) ** 2
                    + ((li_x - li_t) / s_li) ** 2)

        x_ours = np.concatenate([out["a"], out["b"]])
        ref = minimize(cost, np.zeros(8), method="Nelder-Mead",
                       options=dict(maxiter=200000, maxfev=200000,
                                    xatol=1e-12, fatol=1e-14))
        assert cost(x_ours) <= ref.fun * (1.0 + 1e-6)
        assert out["objective"] == pytest.approx(cost(x_ours), rel=1e-10)

    def test_the_residual_z_scores_are_reported(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        li_t = structured_li_of(m)[0] * 1.03
        s_ip, s_li = 0.005 * abs(Ip_s), 0.02
        out = close_ip_structured_soft(psi, w, c, Ip_s, s_ip, ji, jb, jf,
                                       li_target=li_t, li_sigma=s_li,
                                       li_geom=lg)
        assert out["residual_sigma_Ip"] == pytest.approx(
            out["ip_residual"] / s_ip, rel=1e-12)
        assert out["residual_sigma_li"] == pytest.approx(
            (out["li_predicted"] - li_t) / s_li, rel=1e-12)
        # a measurement pulled against a prior does not land exactly on target
        assert abs(out["residual_sigma_li"]) > 1e-6
        assert out["solver"] == "soft-GaussNewton"

    def test_a_soft_axis_row_is_pulled_not_pinned(self):
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        ax = _axis(psi, ji, jb, jf, q0_pull=0.99)
        hard = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                        axis=ax, axis_sigma=None)
        soft = close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                        axis=ax,
                                        axis_sigma=0.05 * abs(ax["j_ref0"]))
        assert abs(hard["axis_residual"]) < 1e-6 * abs(ax["j_ref0"])
        assert abs(soft["axis_residual"]) > 1e-6 * abs(ax["j_ref0"])
        assert abs(soft["residual_sigma_axis"]) < 1.0

    def test_it_converges_in_one_step_when_Ip_is_hard(self):
        """With Ip pinned the whole problem is linear, so Gauss-Newton is a
        single least-squares solve; more than a couple of iterations would mean
        the l_i model had stopped being linear in the coefficients."""
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        out = close_ip_structured_soft(
            psi, w, c, Ip_s, None, ji, jb, jf,
            li_target=structured_li_of(m)[0] * 0.99, li_sigma=0.05,
            li_geom=lg)
        assert out["n_iter"] <= 2


# ---------------------------------------------------------------------------
class TestSoftRefusals:
    def test_a_soft_li_with_no_sigma_is_refused(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(ValueError, match="close_ip_structured"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                     li_target=0.7, li_sigma=None, li_geom=lg)

    @pytest.mark.parametrize("kwargs, match", [
        (dict(li_target=0.7, li_sigma=0.0), "finite and positive"),
        (dict(li_target=0.7, li_sigma=-1.0), "finite and positive"),
        (dict(li_target=0.7, li_sigma=np.inf), "finite and positive"),
    ])
    def test_bad_li_sigmas_raise(self, kwargs, match):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(ValueError, match=match):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                     li_geom=lg, **kwargs)

    @pytest.mark.parametrize("ip_sigma", [0.0, -1.0, np.inf, np.nan])
    def test_bad_Ip_sigmas_raise(self, ip_sigma):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(ValueError, match="Ip_sigma"):
            close_ip_structured_soft(psi, w, c, Ip_s, ip_sigma, ji, jb, jf)

    def test_a_negative_prior_sigma_raises(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(ValueError, match="non-negative"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                     sigma_ind=(-1.0, 1.0, 1.0, 1.0))

    def test_a_wrong_length_ladder_raises(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(ValueError, match="sigmas"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                     sigma_bs=(1.0, 1.0))

    def test_every_coefficient_pinned_is_refused(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(RuntimeError, match="no coefficient is free"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji, jb, jf,
                                     basis=_CONST, sigma_ind=(0.0,),
                                     sigma_bs=(0.0,))

    def test_a_multiplier_off_the_scale_bounds_is_refused(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        raw = lin(ji) + lin(jb) + lin(jf) + c
        with pytest.raises(RuntimeError, match="outside"):
            close_ip_structured_soft(psi, w, c, 6.0 * raw, None, ji, jb, jf)

    def test_a_zero_constraint_row_is_refused(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(RuntimeError, match="identically zero"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji,
                                     np.zeros_like(jb), jf, basis=_CONST,
                                     sigma_ind=(0.0,), sigma_bs=(1.0,))

    def test_mismatched_grids_raise(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        with pytest.raises(ValueError, match="share one grid"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, ji[:-1], jb, jf)

    def test_nonfinite_input_raises(self):
        _m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        bad = ji.copy(); bad[3] = np.nan
        with pytest.raises(RuntimeError, match="non-finite j_ind"):
            close_ip_structured_soft(psi, w, c, Ip_s, None, bad, jb, jf)
        with pytest.raises(RuntimeError, match="non-finite c_affine"):
            close_ip_structured_soft(psi, w, np.nan, Ip_s, None, ji, jb, jf)


# ---------------------------------------------------------------------------
class TestConfigSurface:
    def test_the_knobs_exist_with_the_documented_defaults(self):
        from bouquet.config import GenerationConfig

        g = GenerationConfig()
        assert g.structured_li_target is None
        assert g.structured_li_sigma is None
        assert g.structured_li_kind == "li_1"
        assert g.structured_ip_sigma is None
        assert g.structured_ip_sigma_frac is None
        assert g.structured_soft is False
        assert g.structured_li_tol == 0.005
        assert g.structured_li_max_corrector_steps == 1

    def test_the_two_ip_sigma_spellings_are_mutually_exclusive(self):
        """A campaign runner that does not know Ip passes a fraction; a caller
        that does passes amps.  Setting both would make the recorded sigma_Ip
        ambiguous, so it is refused instead of one being picked.

        Asserted by TRIGGERING the refusal, not by reading the source: the
        old test passed against a file with the guard deleted and a comment
        left behind.
        """
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        from bouquet.run import Bouquet

        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t")
        cfg.generation.jBS_baseline_mode = "ohmic"
        cfg.generation.perturb_jind_in_anchor = True
        cfg.generation.closure_channel = "structured"
        cfg.generation.structured_ip_sigma = 1.0e4
        cfg.generation.structured_ip_sigma_frac = 0.005
        with pytest.raises(ValueError, match="mutually exclusive"):
            Bouquet(cfg)._validate_workflow()

    def test_the_default_channel_is_unchanged(self):
        from bouquet.config import GenerationConfig

        assert GenerationConfig().closure_channel == "bootstrap"


# ---------------------------------------------------------------------------
#  solver: the validation the discrete l_i form was accepted on
# ---------------------------------------------------------------------------
def _probe(outdir):
    """Reproduce get_stats' own l_i from the enclosed-current identity.

    Subprocess entry point (``OFT_env`` is a per-process singleton, so this
    module must never build a solver in the pytest process).  Results land in
    ``<outdir>/li.json``.
    """
    import numpy as np
    import bouquet as bq
    from bouquet.utils import (fsa_current_geometry, eq_jphi_profile,
                               li_closure_geometry, li_value)

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "li"), n_draws=1)
    b.setup_solver()
    bl = b.prepare_baseline()
    mygs = b.mygs
    psi_pad = float(getattr(b.config.source, "psi_pad", 1e-3))
    psi_N = np.asarray(bl.psi_N, dtype=float)
    snap = mygs.copy_eq()

    st_1 = snap.get_stats(lcfs_pad=psi_pad, li_normalization="std")
    st_3 = snap.get_stats(lcfs_pad=psi_pad, li_normalization="iter")
    Ip, _centroid, vol, _pvol, _dflux, _tflux, Bp_vol = snap.get_globals()

    geom = fsa_current_geometry(snap, psi_N)
    lg = li_closure_geometry(snap, geom, psi_pad=psi_pad)
    from bouquet.utils import Ip_fsa_weights, li_achieved
    w_lin, _c = Ip_fsa_weights(geom, convention="jphi-linterp")
    from scipy.integrate import cumulative_trapezoid

    out = {"li_1_stats": float(st_1["l_i"]), "li_3_stats": float(st_3["l_i"]),
           "Bp_vol": float(Bp_vol), "vol_globals": float(vol),
           "Ip_globals": float(Ip), "vol_geom": float(lg["vol"]),
           "perimeter": float(lg["perimeter"]),
           "perimeter_source": str(lg["perimeter_source"]),
           "perimeter_get_stats_dl": float(lg["perimeter_get_stats_dl"]),
           "perimeter_ratio_dl_over_L": float(lg["perimeter_ratio_dl_over_L"]),
           "R_axis": float(lg["R_axis"]),
           "dpsi_dpsiN": float(lg["dpsi_dpsiN"])}
    for kind in LI_KINDS:
        v, _info = li_achieved(snap, li_kind=kind, psi_pad=psi_pad,
                               perimeter=lg["perimeter"])
        out[f"li_achieved_{kind}"] = float(v)
    # the SAME geometry with get_stats' own dl, so the identity can be checked
    # separately from the perimeter convention
    lg_dl = dict(lg, perimeter=float(lg["perimeter_get_stats_dl"]))

    for tag, j in (("eq_own", eq_jphi_profile(geom, "jphi-linterp", eq=snap)),
                   ("archived", np.asarray(bl.j_phi, dtype=float))):
        I_enc = cumulative_trapezoid(w_lin * np.asarray(j, float), psi_N,
                                     initial=0.0) + lg["affine_cum"]
        S = float(lg["dpsi_dpsiN"]) * float(np.trapezoid(I_enc, psi_N))
        sgn = float(np.sign(I_enc[-1]))
        mu0 = 4.0e-7 * np.pi
        out[tag] = {
            "Ip_enclosed": float(I_enc[-1]),
            "Bp_vol_identity": 2.0 * np.pi * mu0 * S * sgn,
            "li_1": li_value(psi_N, I_enc, lg, "li_1"),
            "li_1_with_get_stats_dl": li_value(psi_N, I_enc, lg_dl, "li_1"),
            "li_3": li_value(psi_N, I_enc, lg, "li_3"),
        }

    # reading the snapshot must not have moved the live solver
    out["Ip_after"] = float(mygs.get_globals()[0])
    with open(os.path.join(outdir, "li.json"), "w") as fh:
        json.dump(out, fh)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    work = tmp_path_factory.mktemp("li")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"l_i probe failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-4000:]}")
    with open(str(work / "li.json")) as fh:
        return json.load(fh)


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like mesh/baseline; skipped when unavailable")


@pytest.mark.solver
@solver_only
def test_the_identity_reproduces_the_solvers_own_Bp_volume_integral(measured):
    """``int B_p^2 dV = 2 pi mu0 int I dpsi`` against TokaMaker's own Bp_vol.
    Measured +0.0208 %."""
    got = float(measured["eq_own"]["Bp_vol_identity"])
    err = abs(got / float(measured["Bp_vol"]) - 1.0)
    assert err <= _LI_SELF_CONSISTENCY, (
        f"the enclosed-current identity gives Bp_vol = {got:.6e} against the "
        f"solver's {measured['Bp_vol']:.6e} ({100 * err:.4f}%, bar "
        f"{100 * _LI_SELF_CONSISTENCY:.2f}%)")


@pytest.mark.solver
@solver_only
def test_the_discrete_form_reproduces_get_stats_li_3(measured):
    """l_i(3) carries no perimeter, so the identity is directly comparable with
    ``get_stats(li_normalization='iter')``.  Measured +0.0215 %."""
    got = float(measured["eq_own"]["li_3"])
    ref = float(measured["li_3_stats"])
    err = abs(got / ref - 1.0)
    assert err <= _LI_SELF_CONSISTENCY, (
        f"identity gives l_i(3) {got:.6f} against get_stats' {ref:.6f} "
        f"({100 * err:.4f}%, bar {100 * _LI_SELF_CONSISTENCY:.2f}%)")


@pytest.mark.solver
@solver_only
def test_the_discrete_form_reproduces_get_stats_li_1_on_its_own_perimeter(
        measured):
    """l_i(1) is proportional to L^2, so it can only be compared with
    ``get_stats(li_normalization='std')`` on get_stats' OWN ``dl``.  Fed that
    perimeter, the identity reproduces it to +0.0215 % -- which is the check
    that the IDENTITY is right, separately from which perimeter is the right
    one (the next test)."""
    got = float(measured["eq_own"]["li_1_with_get_stats_dl"])
    ref = float(measured["li_1_stats"])
    err = abs(got / ref - 1.0)
    assert err <= _LI_SELF_CONSISTENCY, (
        f"identity gives l_i(1) {got:.6f} on get_stats' own dl against its "
        f"{ref:.6f} ({100 * err:.4f}%, bar "
        f"{100 * _LI_SELF_CONSISTENCY:.2f}%)")


@pytest.mark.solver
@solver_only
def test_the_perimeter_is_the_lcfs_contour_not_get_stats_dl(measured):
    """The calibration that decides whether the closure's l_i(1) is on the same
    scale as a g-file (EFIT-convention) estimator.

    ``get_stats`` takes ``dl`` from ``get_q``; that reading is ABOVE the
    perimeter of the traced ``1 - psi_pad`` contour -- the surface
    ``save_eqdsk`` writes as ``RBBBS``/``ZBBBS`` -- by +0.83 % on this anchor
    (+1.6 % on the real diverted DIII-D equilibria the calibration study used).
    Since ``l_i(1) ~ L^2`` that is +1.7 % (+3.2 %) of l_i(1), i.e. 0.013 (0.032)
    in absolute terms: several times any tolerance here, and it would be read
    as core current peaking that is not there.  So the closure measures its own
    perimeter and records both."""
    assert "traced LCFS contour" in measured["perimeter_source"], \
        measured["perimeter_source"]
    ratio = float(measured["perimeter_ratio_dl_over_L"])
    assert 1.0 < ratio < 1.05, (
        f"get_stats' dl / traced LCFS perimeter = {ratio:.5f}; if OFT has "
        "changed how get_q reports dl, re-run the g-file calibration before "
        "trusting either")
    # ... and the two conventions differ in l_i(1) by ~ratio^2, as they must
    both = (float(measured["eq_own"]["li_1_with_get_stats_dl"])
            / float(measured["eq_own"]["li_1"]))
    assert both == pytest.approx(ratio ** 2, rel=1e-9)


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("kind", LI_KINDS)
def test_li_achieved_agrees_with_the_current_identity(measured, kind):
    """``li_achieved`` reads the SOLVE's own l_i from TokaMaker's exact volume
    integrals; the closure's model computes it from the enclosed current.  Two
    independent routes to the same number, normalised the same way -- they must
    agree to the identity's own accuracy."""
    got = float(measured[f"li_achieved_{kind}"])
    ref = float(measured["eq_own"][kind])
    err = abs(got / ref - 1.0)
    assert err <= _LI_SELF_CONSISTENCY, (kind, got, ref, err)


@pytest.mark.solver
@solver_only
def test_li_achieved_li_3_is_exactly_get_stats_iter(measured):
    """l_i(3) has no perimeter in it, so there is nothing for the closure's
    convention to change and the two must agree to rounding."""
    assert measured["li_achieved_li_3"] == pytest.approx(
        measured["li_3_stats"], rel=1e-12)


@pytest.mark.solver
@solver_only
def test_the_geometry_is_read_off_the_calls_it_claims(measured):
    """vol from get_globals, R_axis from o_point -- not R_geo, which get_stats
    does NOT use for l_i(3)."""
    assert measured["vol_geom"] == measured["vol_globals"]
    assert 1.0 < measured["R_axis"] < 3.0
    assert 3.0 < measured["perimeter"] < 12.0


@pytest.mark.solver
@solver_only
def test_the_archived_profile_agrees_too(measured):
    """The archived baseline total is not bit-identical to the equilibrium's
    own GS profile, so its l_i differs -- but only at the level the profiles
    themselves differ, not at the level of a wrong formula.

    A DIFFERENT quantity from the identity above, and so its own bar: the
    identity reproduces TokaMaker's own l_i from the same profile (0.02 %),
    while this compares two profiles that genuinely differ (0.15 %).  Sharing
    one constant is what forced that constant to be 46x the physics.
    """
    err = abs(float(measured["archived"]["li_3"])
              / float(measured["li_3_stats"]) - 1.0)
    assert err <= _LI_ARCHIVED_AGREEMENT, err
    err = abs(float(measured["archived"]["li_1_with_get_stats_dl"])
              / float(measured["li_1_stats"]) - 1.0)
    assert err <= _LI_ARCHIVED_AGREEMENT, err


@pytest.mark.solver
@solver_only
def test_reading_the_snapshot_did_not_perturb_the_live_solver(measured):
    assert measured["Ip_after"] == measured["Ip_globals"]


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[1])
# ---------------------------------------------------------------------------
#  the corrector's gain law, and what the record says it delivered
# ---------------------------------------------------------------------------
#: The corrector's row update inverts a LOG-GAIN, not a proportionality: the
#: closure freezes ``dpsi_dpsiN`` at the anchor, but the solved equilibrium's
#: own ``Delta_psi`` moves with ``sqrt(l_i)``, so l_i enters the achieved
#: equilibrium twice and ``achieved ~ row**2``.  The out-of-tree campaign
#: (2026-09-14, 133 corrected hard slices) measured
#: ``d(achieved)/d(row) = 2.14`` against
#: the ``A1/T = 0.96`` the old proportional update assumed -- a factor ~2.2
#: overshoot that flipped the residual's sign on 133/133 slices.  These tests
#: run the update against a SYNTHETIC power-law map, where the exact answer is
#: known, so they pin the algebra rather than re-measure the campaign.
class TestLiCorrectorGainLaw:
    T = 0.9

    @staticmethod
    def _map(p, T=0.9, frac=0.97):
        """A synthetic ``achieved = C * row**p`` response.

        ``C`` is set so that the predictor row (``row == T``, imposed exactly)
        lands a fraction *frac* of the way to the target -- the ordinary miss
        the corrector exists to remove.
        """
        C = float(frac) * float(T) / float(T) ** p
        return lambda row: C * float(row) ** p

    def test_a_square_law_map_closes_in_one_step(self):
        """p = 2 is the shipped, parameter-free exponent, and on a map that
        really is quadratic the single step is EXACT -- not merely closer."""
        from bouquet.utils import LI_GAIN_EXPONENT, li_corrector_row

        assert LI_GAIN_EXPONENT == 2.0
        for T in (0.55, 0.9, 1.35):
            A = self._map(2.0, T=T)
            row1 = T                       # the hard predictor imposes row == T
            a1 = A(row1)
            row2 = li_corrector_row(T, a1)
            assert abs(A(row2) - T) < 1e-12
            # and it is the SQUARE ROOT of the ratio the old update applied
            assert row2 == pytest.approx(T * np.sqrt(T / a1), rel=1e-14)

    def test_the_old_proportional_update_overshoots_by_the_measured_factor(self):
        """The superseded update is this one at exponent 1.  On the quadratic
        map it overshoots and flips the residual's sign, exactly as the
        campaign measured."""
        from bouquet.utils import li_corrector_row

        T = self.T
        A = self._map(2.0, T=T)
        a1 = A(T)                          # a predictor that landed 3 % low
        old = li_corrector_row(T, a1, row=T, exponent=1.0)
        new = li_corrector_row(T, a1, row=T)
        pre = a1 - T
        assert pre < 0.0
        assert A(old) - T > 0.0            # sign flipped
        assert abs(A(old) - T) > abs(pre)  # and bigger than what it removed
        assert abs(A(new) - T) < 1e-12

    def test_a_row_2p2_map_leaves_a_residual_the_secant_then_closes(self):
        """Where the slice's own exponent is not 2 the parameter-free step
        lands close but not inside; the SECOND step reads the exponent off the
        two measured pairs and closes it exactly."""
        from bouquet.utils import li_corrector_row, li_gain_exponent_secant

        p_true = 2.2
        T = self.T
        # a 10 % low predictor -- the campaign's tail, where a p = 2 step on a
        # p = 2.2 slice does NOT land inside structured_li_tol = 0.005
        A = self._map(p_true, T=T, frac=0.90)
        row1, a1 = T, A(T)
        row2 = li_corrector_row(T, a1)
        a2 = A(row2)
        # what a p = 2 step leaves on a p = 2.2 map, in closed form
        expected = A(T * (T / a1) ** 0.5) - T
        assert a2 - T == pytest.approx(expected, rel=1e-12)
        assert abs(a2 - T) > 0.005                       # outside li_tol
        assert abs(a2 - T) < 0.15 * abs(a1 - T)          # but far better

        p_sec = li_gain_exponent_secant(row1, a1, row2, a2)
        assert p_sec == pytest.approx(p_true, rel=1e-12)
        row3 = li_corrector_row(T, a2, row=row2, exponent=p_sec)
        assert abs(A(row3) - T) < 1e-12

    def test_the_secant_refuses_a_degenerate_pair(self):
        """No step to measure, a non-positive value, or an exponent outside
        the sanity gate -> None, and the caller falls back to p = 2."""
        from bouquet.utils import (LI_GAIN_EXPONENT_BOUNDS,
                                   li_gain_exponent_secant)

        assert li_gain_exponent_secant(0.9, 0.8, 0.9, 0.8) is None  # no step
        assert li_gain_exponent_secant(0.9, 0.8, 1.0, -0.1) is None
        assert li_gain_exponent_secant(0.9, np.nan, 1.0, 0.85) is None
        # p outside the bounds is refused, not imposed
        lo, hi = LI_GAIN_EXPONENT_BOUNDS
        assert li_gain_exponent_secant(1.0, 1.0, 1.1, 1.1 ** (hi + 1.0)) is None
        assert li_gain_exponent_secant(1.0, 1.0, 1.1, 1.1 ** (lo / 2.0)) is None

    @pytest.mark.parametrize("kwargs", [
        dict(li_target=0.0, li_achieved_value=0.9),
        dict(li_target=0.9, li_achieved_value=-0.2),
        dict(li_target=0.9, li_achieved_value=np.nan),
        dict(li_target=0.9, li_achieved_value=0.9, exponent=0.0),
        dict(li_target=0.9, li_achieved_value=0.9, row=0.0),
    ])
    def test_a_row_that_cannot_be_formed_raises(self, kwargs):
        """A NaN or negative row would be IMPOSED as a constraint, so the
        corrector must never form one silently."""
        from bouquet.utils import li_corrector_row

        with pytest.raises(ValueError):
            li_corrector_row(**kwargs)

    def test_the_docstring_cites_the_measurement_it_rests_on(self):
        """p = 2 is parameter-free, but the reason to believe it is measured
        out of tree.  The provenance travels with the constant -- the campaign
        is cited generically (this is a public repository, so no run path or
        discharge identifier appears), together with the measured gain and the
        campaign-calibrated exponent named as NOT adopted."""
        from bouquet import utils

        import inspect

        src = inspect.getsource(utils)
        assert "out of tree" in src
        assert "133 corrected hard" in src        # the sample it rests on
        assert "2.14" in src                      # the measured gain
        # and the campaign-calibrated 2.17 is named as NOT adopted
        assert "2.17" in src


# ---------------------------------------------------------------------------
#  the corrector itself, on a stubbed solver: bookkeeping and health
# ---------------------------------------------------------------------------
class _FakeSnap:
    """Enough of a TokaMaker ``copy_eq()`` snapshot for the corrector."""

    def __init__(self, li, q0=1.05):
        self._li = float(li)
        self._q0 = float(q0)

    def get_q(self, psi=None, compute_geo=False):
        return (np.asarray(psi, dtype=float), np.full(np.size(psi), self._q0),
                None, None)

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"l_i": self._li * 1.008}      # the get_stats 'dl' reading


class _FakeGS:
    def __init__(self, world):
        self._world = world

    def copy_eq(self):
        return _FakeSnap(self._world["li"], self._world.get("q0", 1.05))


class _FakeBaseline:
    def __init__(self):
        self.ip_closure = {"closure_limited": False,
                           "closure_limited_reasons": ()}
        self.ohm_scale = 1.0
        self.bs_scale = 1.0


#: an admitted sawtooth axis row, as run.py's predictor builds it -- needed by
#: any case where the corrector actually takes its q0 step
_AXIS_ROW = dict(psi=0.0, j_ind0=7.0e5, j_bs0=3.0e5, j_fix0=0.0, j_ref0=9.7e5)


def _stub_corrector(monkeypatch, p_true, target, C=None, li_sigma=None,
                    max_steps=1, soft=False, ip_sigma=None, ip_post=None,
                    state_extra=None, ip_of=None, roundtrip_gate=None,
                    bl=None, solver_raises=None, bs_eff=1.0, q0=1.05):
    """Run ``_close_ip_structured_corrector`` against ``achieved = C row**p``.

    The closure solvers and ``li_achieved`` are stubbed: what is under test is
    the corrector's row update and its record, not the KKT algebra (which the
    rest of this module covers).  ``C`` defaults to a value that makes the
    predictor land 3 % low, i.e. an ordinary miss.
    """
    from bouquet import utils
    from bouquet.run import Bouquet

    if C is None:
        C = 0.97 * target / target ** p_true
    world = {"li": C * target ** p_true, "row": float(target), "solves": 0,
             "q0": q0}

    def _fake_solver(*a, **kw):
        if solver_raises is not None:
            raise solver_raises
        row = kw["li_target"]
        world["row"] = float(row)
        _post = 1.0e6 if ip_post is None else float(ip_post)
        return dict(s_ind=np.ones(5), s_bs=np.ones(5), a=[1.0], b=[1.0],
                    ohm_scale_eff=1.0, bs_scale_eff=float(bs_eff),
                    structure_ind=0.0,
                    structure_bs=0.0, ip_residual_pct=0.0,
                    Ip_hybrid=_post, ip_residual=_post - 1.0e6,
                    residual_sigma_Ip=(None if not ip_sigma
                                       else (_post - 1.0e6) / float(ip_sigma)),
                    li_predicted=float(row))

    def _fake_li_achieved(eq, li_kind="li_1", psi_pad=1e-3, perimeter=None):
        return float(eq._li), {"perimeter": perimeter}

    monkeypatch.setattr(utils, "close_ip_structured", _fake_solver)
    monkeypatch.setattr(utils, "close_ip_structured_soft", _fake_solver)
    monkeypatch.setattr(utils, "li_achieved", _fake_li_achieved)

    def _solve_jphi(j):
        world["solves"] += 1
        world["li"] = C * world["row"] ** p_true
        return 7

    bl = _FakeBaseline() if bl is None else bl
    state = dict(
        q0_target=1.05, psi_q=np.linspace(0.0, 1.0, 5),
        psi_geom=np.linspace(0.0, 1.0, 5),
        j_ind=np.ones(5), j_BS_swb=np.ones(5), j_fixed=np.zeros(5),
        axis=None, w_lin=np.ones(5), c_signed=0.0, Ip_signed=1.0e6,
        # linear Ip parts, consistent with Ip_signed so the refreshed
        # closure-health block is clean unless a case makes it dirty
        ip_ind=7.0e5, ip_bs=3.0e5, ip_fix=0.0,
        basis=None, weights=None, q0_tol=0.01, gated=False, soft=soft,
        ip_sigma=ip_sigma, sigma_ind=None, sigma_bs=None,
        li_target=float(target), li_sigma=li_sigma, li_kind="li_1",
        li_geom={"perimeter": 4.2}, psi_pad=1e-3, li_tol=0.005,
        li_max_corrector_steps=max_steps,
    )
    state.update(state_extra or {})
    Bouquet._close_ip_structured_corrector(state, bl, _FakeGS(world),
                                           _solve_jphi, ip_of=ip_of,
                                           roundtrip_gate=roundtrip_gate)
    return bl.ip_closure, world


class TestCorrectorBookkeeping:
    def test_predictor_and_corrected_are_recorded_separately(self, monkeypatch):
        """The field whose name reads like the delivered residual must BE the
        delivered residual.  ``structured_li_solved_residual`` used to be
        written once at the predictor stage and never refreshed."""
        T = 0.9
        rec, world = _stub_corrector(monkeypatch, 2.0, T)

        assert world["solves"] == 1
        assert rec["n_extra_solves"] == 1
        pre = rec["structured_li_residual_predictor"]
        post = rec["structured_li_residual_corrected"]
        assert pre == pytest.approx(-0.03 * T, rel=1e-9)
        assert abs(post) < 1e-12                      # p = 2 map, one step
        assert rec["structured_li_achieved_predictor"] == pytest.approx(
            T + pre, rel=1e-12)
        assert rec["structured_li_achieved_corrected"] == pytest.approx(
            T + post, rel=1e-12)
        # old names are aliases of the CORRECTED values
        assert rec["structured_li_solved_residual"] == post
        assert rec["structured_li_residual"] == post
        assert rec["structured_li_solved"] == rec[
            "structured_li_achieved_corrected"]
        # the predictor's achieved keeps its own (already unambiguous) name
        assert rec["structured_li_solved_predictor"] == rec[
            "structured_li_achieved_predictor"]

    def test_the_gain_law_used_is_recorded(self, monkeypatch):
        from bouquet.utils import LI_GAIN_EXPONENT

        rec, _ = _stub_corrector(monkeypatch, 2.0, 0.9)
        assert rec["structured_li_gain_exponents"] == [LI_GAIN_EXPONENT]
        assert "p = 2" in rec["structured_li_gain_law"]
        assert rec["structured_li_max_corrector_steps"] == 1
        # the row really is the square root of the ratio
        T, A1 = 0.9, rec["structured_li_achieved_predictor"]
        assert rec["structured_li_corrector_row"] == pytest.approx(
            T * np.sqrt(T / A1), rel=1e-12)
        assert rec["structured_li_corrector_ratio"] == pytest.approx(
            T / A1, rel=1e-12)
        assert len(rec["structured_li_corrector_rows"]) == 2   # pre + post

    def test_zero_extra_solves_reports_the_predictor_as_the_delivered_value(
            self, monkeypatch):
        """With nothing to correct the predictor IS the delivered equilibrium,
        so corrected == predictor rather than None."""
        T = 0.9
        rec, world = _stub_corrector(monkeypatch, 2.0, T,
                                     C=T / T ** 2.0)    # lands exactly on T
        assert world["solves"] == 0
        assert rec["n_extra_solves"] == 0
        assert (rec["structured_li_residual_corrected"]
                == rec["structured_li_residual_predictor"])
        assert rec["closure_limited"] is False

    def test_sigma_li_is_the_achieved_corrected_residual(self, monkeypatch):
        """``structured_residual_sigma_li`` is the ACHIEVED distance in sigma
        units; the posterior's model-space fit to its own row lives under
        ``structured_residual_sigma_li_model`` and is written by the
        predictor."""
        rec, _ = _stub_corrector(monkeypatch, 2.2, 0.9, li_sigma=0.04,
                                 soft=True)
        assert rec["structured_residual_sigma_li"] == pytest.approx(
            rec["structured_li_residual_corrected"] / 0.04, rel=1e-12)
        import inspect as _i

        from bouquet.run import Bouquet
        src = _i.getsource(Bouquet._close_ip_structured_predictor)
        assert "structured_residual_sigma_li_model=out.get" in src

    def test_a_missed_hard_row_is_flagged_never_retried(self, monkeypatch):
        """A hard l_i row that the single corrector step could not meet is a
        closure-health FLAG.  The cost ceiling is the point: no extra solve is
        spent, and the existing reasons are preserved."""
        T = 0.9
        rec, world = _stub_corrector(monkeypatch, 2.6, T)
        assert world["solves"] == 1                    # still ONE, not a loop
        assert abs(rec["structured_li_residual_corrected"]) > 0.005
        assert rec["closure_limited"] is True
        assert any("l_i misses its hard row" in r
                   for r in rec["closure_limited_reasons"])

    def test_the_soft_channel_is_not_flagged_on_the_hard_tolerance(
            self, monkeypatch):
        """The soft row is a measurement with a sigma, not a pin; its miss is
        reported in sigma units, not as a closure-limited slice."""
        rec, _ = _stub_corrector(monkeypatch, 2.6, 0.9, li_sigma=0.04,
                                 soft=True)
        assert abs(rec["structured_li_residual_corrected"]) > 0.005
        assert rec["closure_limited"] is False

    def test_the_optional_second_step_is_off_by_default(self, monkeypatch):
        """Default behaviour changes from the previous release ONLY by the
        gain law: still at most one extra solve."""
        from bouquet.config import GenerationConfig

        assert GenerationConfig().structured_li_max_corrector_steps == 1
        _rec, world = _stub_corrector(monkeypatch, 2.6, 0.9)
        assert world["solves"] == 1

    def test_the_optional_second_step_closes_the_tail(self, monkeypatch):
        """Enabled, the second step fires only when the first missed, reads
        the slice's own exponent off the two measured pairs, and closes."""
        T = 0.9
        rec, world = _stub_corrector(monkeypatch, 2.6, T, max_steps=2)
        assert world["solves"] == 2
        assert rec["n_extra_solves"] == 2
        assert abs(rec["structured_li_residual_corrected"]) < 1e-12
        exps = rec["structured_li_gain_exponents"]
        assert len(exps) == 2
        assert exps[0] == 2.0
        assert exps[1] == pytest.approx(2.6, rel=1e-9)   # measured, not assumed
        assert rec["closure_limited"] is False
        assert "2 corrector solves" in rec["sawtooth_verdict"]
        assert len(rec["structured_li_corrector_rows"]) == 3

    def test_the_second_step_is_not_spent_when_the_first_closed(self,
                                                               monkeypatch):
        """Conditional, not unconditional: on a p = 2 slice the first step
        already lands and the second solve is never taken."""
        _rec, world = _stub_corrector(monkeypatch, 2.0, 0.9, max_steps=2)
        assert world["solves"] == 1


class TestCorrectorKeepsThePosteriorIp:
    """A corrector re-solve re-runs the SAME closure, so on the soft channel it
    lands on a NEW posterior Ip.  It must be recorded as one -- never snapped
    back to the measurement, never gated on."""

    IP = 1.0e6
    SIG = 0.005 * 1.0e6

    def test_the_posterior_is_refreshed_from_the_corrector_solve(
            self, monkeypatch):
        post = self.IP * 1.003
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9, li_sigma=0.04,
                                     soft=True, ip_sigma=self.SIG,
                                     ip_post=post)
        assert world["solves"] == 1
        assert rec["structured_ip_posterior"] == pytest.approx(post, rel=1e-12)
        assert rec["structured_ip_measured_residual_pct"] == pytest.approx(
            0.3, rel=1e-9)
        assert rec["structured_residual_sigma_Ip"] == pytest.approx(
            0.6, rel=1e-9)
        # inside 1 sigma -> reported, not flagged
        assert rec["closure_limited"] is False

    def test_beyond_one_sigma_the_corrector_flags_it(self, monkeypatch):
        from bouquet.utils import SOFT_IP_FLAG_PREFIX

        rec, _ = _stub_corrector(monkeypatch, 2.0, 0.9, li_sigma=0.04,
                                 soft=True, ip_sigma=self.SIG,
                                 ip_post=self.IP * 1.008)
        assert rec["structured_residual_sigma_Ip"] == pytest.approx(
            1.6, rel=1e-9)
        assert rec["closure_limited"] is True
        assert sum(str(r).startswith(SOFT_IP_FLAG_PREFIX)
                   for r in rec["closure_limited_reasons"]) == 1

    def test_the_hard_channel_records_no_soft_ip_residual(self, monkeypatch):
        rec, _ = _stub_corrector(monkeypatch, 2.0, 0.9, ip_post=self.IP)
        assert rec["structured_ip_posterior"] == pytest.approx(self.IP,
                                                               rel=1e-12)
        assert "structured_residual_sigma_Ip" not in rec
        assert rec["closure_limited"] is False


# ---------------------------------------------------------------------------
class TestCorrectorRefusalAndHealthPaths:
    """The corrector's delivery side: what it flags, what it keeps, what it
    re-checks.  Every branch here decides which equilibrium ships, and none of
    them was reached by the suite before.
    """

    IP = 1.0e6
    SIG = 0.005 * 1.0e6

    # ---- B4: a readback that could not be taken --------------------------
    def test_a_non_finite_q0_is_not_reported_as_predictor_accepted(
            self, monkeypatch):
        """``abs(nan) > tol`` is False, so a NaN q0 used to make want_q0 False
        and print "predictor accepted, no extra solve" with closure_limited
        untouched.  The usable-reference guards could not catch it: they were
        only consulted after want_* had been decided."""
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9,
                                     C=0.9 / 0.9 ** 2.0,   # l_i lands exactly
                                     q0=float("nan"),
                                     state_extra=dict(gated=True))
        assert world["solves"] == 0
        assert rec["n_extra_solves"] == 0
        assert "not finite" in rec["sawtooth_verdict"]
        assert "predictor accepted" not in rec["sawtooth_verdict"]
        assert rec["closure_limited"] is True
        assert any("q0 is not finite" in str(r)
                   for r in rec["closure_limited_reasons"])

    def test_a_non_finite_li_is_not_reported_as_predictor_accepted(
            self, monkeypatch):
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9, C=float("nan"))
        assert world["solves"] == 0
        assert "not finite" in rec["sawtooth_verdict"]
        assert rec["closure_limited"] is True
        assert any("l_i is not finite" in str(r)
                   for r in rec["closure_limited_reasons"])

    def test_a_finite_readback_inside_tolerance_is_still_clean(self,
                                                               monkeypatch):
        """The guard must not make every slice closure-limited."""
        T = 0.9
        rec, world = _stub_corrector(monkeypatch, 2.0, T, C=T / T ** 2.0,
                                     q0=1.05, state_extra=dict(gated=True))
        assert world["solves"] == 0
        assert rec["sawtooth_verdict"] == "structured predictor (0 extra solves)"
        assert rec["closure_limited"] is False
        assert rec["closure_limited_reasons"] == ()

    # ---- A2, on this channel: a missed q0 is a FLAG -----------------------
    def test_a_missed_q0_is_flagged_after_the_corrector(self, monkeypatch):
        """The q0 corrector has flagged this since the A2 fix; here the
        residual was recorded and read by nothing, so a slice the corrector
        could not land on q0 was indistinguishable from one it did."""
        T = 0.9
        rec, world = _stub_corrector(
            monkeypatch, 2.0, T, C=T / T ** 2.0, q0=1.10,
            state_extra=dict(gated=True, axis=_AXIS_ROW))
        assert world["solves"] == 1          # it did take its one step
        assert rec["closure_limited"] is True
        assert any("q0 misses its row" in str(r)
                   for r in rec["closure_limited_reasons"])
        # flag only: the bar itself is untouched and recorded
        assert rec["closure_limited_thresholds"]["q0_tol"] == 0.01

    def test_a_non_finite_q0_residual_is_flagged_after_the_corrector(
            self, monkeypatch):
        T = 0.9
        rec, _ = _stub_corrector(monkeypatch, 2.0, T, C=T / T ** 2.0,
                                 q0=np.nan, state_extra=dict(gated=True))
        assert rec["closure_limited"] is True
        assert any("q0 residual is not finite" in str(r)
                   for r in rec["closure_limited_reasons"])

    def test_a_q0_only_corrector_refusal_is_flagged(self, monkeypatch):
        """With no l_i target the refusal was recorded in
        ``structured_corrector_refusal`` and the run then reported a clean
        closure on the predictor equilibrium."""
        rec, world = _stub_corrector(
            monkeypatch, 2.0, 0.9,
            q0=1.10,                      # the miss that asks for the step
            solver_raises=RuntimeError("close_ip_structured: singular KKT "
                                       "system"),
            state_extra=dict(gated=True, axis=_AXIS_ROW,
                             li_target=None))
        assert world["solves"] == 0
        assert "refused" in rec["sawtooth_verdict"]
        assert rec["closure_limited"] is True
        assert any("the corrector step was refused" in str(r)
                   for r in rec["closure_limited_reasons"])

    # ---- B8 / "keep the predictor" ---------------------------------------
    @pytest.mark.parametrize("exc", [
        RuntimeError("close_ip_structured: singular KKT system"),
        ValueError("close_ip_structured: 1 basis functions but 4/4 weights"),
    ])
    def test_a_refused_re_solve_keeps_the_predictor(self, monkeypatch, exc):
        """The docstring promises the last accepted equilibrium is KEPT.  The
        ValueError half of that promise was not kept: ``close_ip_structured``
        raises ValueError on a bad basis or weight spec and only RuntimeError
        was caught."""
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9, solver_raises=exc)
        assert world["solves"] == 0
        assert rec["n_extra_solves"] == 0
        assert rec["sawtooth_verdict"] == ("structured predictor kept "
                                           "(corrector step refused)")
        assert str(exc)[:20] in rec["structured_corrector_refusal"]
        # the predictor's own residual is still the delivered one
        assert rec["structured_li_residual_corrected"] == rec[
            "structured_li_residual_predictor"]

    def test_a_row_update_refusal_is_caught_too(self, monkeypatch):
        """``li_corrector_row`` raises ValueError on a non-positive or
        non-finite l_i and used to be called OUTSIDE the try."""
        from bouquet import utils

        def _boom(*a, **kw):
            raise ValueError("li_corrector_row: non-finite l_i")
        monkeypatch.setattr(utils, "li_corrector_row", _boom)
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9)
        assert world["solves"] == 0
        assert "refused" in rec["sawtooth_verdict"]
        assert "li_corrector_row" in rec["structured_corrector_refusal"]

    # ---- B9: the health record describes what was delivered ---------------
    def test_a_corrector_that_deepens_the_downscale_is_flagged(self,
                                                               monkeypatch):
        """The predictor lands a healthy bs_scale, the corrector solve returns
        one under bs_scale_min.  The record used to keep the predictor's."""
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9, bs_eff=0.41)
        assert world["solves"] == 1
        assert rec["bs_scale"] == pytest.approx(0.41)
        assert rec["closure_limited"] is True
        assert any("bs_scale" in str(r)
                   for r in rec["closure_limited_reasons"])
        # f_BS_closed follows the DELIVERED bootstrap
        assert rec["f_BS_closed"] == pytest.approx(0.41 * 3.0e5 / 1.0e6,
                                                   rel=1e-12)

    def test_the_health_block_is_refreshed_even_with_no_corrector_solve(
            self, monkeypatch):
        T = 0.9
        rec, _ = _stub_corrector(monkeypatch, 2.0, T, C=T / T ** 2.0)
        assert rec["f_BS_closed"] == pytest.approx(3.0e5 / 1.0e6, rel=1e-12)
        assert rec["raw_components_ip_mismatch_pct"] == pytest.approx(
            0.0, abs=1e-12)

    def test_a_soft_flag_does_not_stack_across_the_refresh(self, monkeypatch):
        from bouquet.utils import SOFT_IP_FLAG_PREFIX

        bl = _FakeBaseline()
        bl.ip_closure["closure_limited_reasons"] = (
            SOFT_IP_FLAG_PREFIX + " (z_Ip = +1.90)",)
        bl.ip_closure["closure_limited"] = True
        rec, _ = _stub_corrector(monkeypatch, 2.0, 0.9, li_sigma=0.04,
                                 soft=True, ip_sigma=self.SIG,
                                 ip_post=self.IP * 1.008, bl=bl)
        assert sum(str(r).startswith(SOFT_IP_FLAG_PREFIX)
                   for r in rec["closure_limited_reasons"]) == 1
        assert "+1.60" in " ".join(rec["closure_limited_reasons"])

    def test_a_stale_soft_flag_is_kept_when_no_corrector_solve_ran(
            self, monkeypatch):
        """Nothing new to say about the posterior -> the predictor's flag
        stands rather than being silently dropped by the refresh."""
        from bouquet.utils import SOFT_IP_FLAG_PREFIX

        T = 0.9
        bl = _FakeBaseline()
        bl.ip_closure["closure_limited_reasons"] = (
            SOFT_IP_FLAG_PREFIX + " (z_Ip = +1.90)",)
        bl.ip_closure["closure_limited"] = True
        rec, world = _stub_corrector(monkeypatch, 2.0, T, C=T / T ** 2.0,
                                     li_sigma=0.04, soft=True,
                                     ip_sigma=self.SIG, bl=bl)
        assert world["solves"] == 0
        assert rec["closure_limited"] is True
        assert any(str(r).startswith(SOFT_IP_FLAG_PREFIX)
                   for r in rec["closure_limited_reasons"])

    # ---- B10: the assembly gate, re-run on the delivered hybrid -----------
    def test_the_roundtrip_gate_re_runs_after_the_corrector(self, monkeypatch):
        from bouquet.utils import ip_roundtrip_gate

        seen = []

        def _gate(ip, posterior=None, sigma_Ip=None):
            seen.append((float(ip), posterior, sigma_Ip))
            return ip_roundtrip_gate(ip, self.IP, posterior=posterior,
                                     sigma_Ip=sigma_Ip)
        post = self.IP * 1.003
        rec, world = _stub_corrector(
            monkeypatch, 2.0, 0.9, li_sigma=0.04, soft=True,
            ip_sigma=self.SIG, ip_post=post,
            ip_of=lambda j: post, roundtrip_gate=_gate)
        assert world["solves"] == 1
        assert rec["Ip_hybrid"] == pytest.approx(post, rel=1e-12)
        # the SOFT reference is the refreshed posterior, not the measurement
        assert seen == [(post, pytest.approx(post, rel=1e-12), self.SIG)]
        assert rec["structured_roundtrip_post_corrector_reference"] == \
            "the closure's own posterior Ip"
        assert abs(rec["structured_roundtrip_post_corrector_err_pct"]) < 1e-9

    def test_a_hard_channel_re_gates_against_the_measurement(self,
                                                             monkeypatch):
        from bouquet.utils import ip_roundtrip_gate

        seen = []

        def _gate(ip, posterior=None, sigma_Ip=None):
            seen.append((posterior, sigma_Ip))
            return ip_roundtrip_gate(ip, self.IP, posterior=posterior,
                                     sigma_Ip=sigma_Ip)
        rec, _ = _stub_corrector(monkeypatch, 2.0, 0.9,
                                 ip_of=lambda j: self.IP, roundtrip_gate=_gate)
        assert seen == [(None, None)]
        assert rec["structured_roundtrip_post_corrector_reference"] == \
            "Ip_target"

    def test_a_bad_assembly_after_the_corrector_is_refused(self, monkeypatch):
        """The corrector IS an assembly; the gate that exists to catch a bad
        one has to see it."""
        from bouquet.utils import ip_roundtrip_gate

        with pytest.raises(RuntimeError, match="algebra error"):
            _stub_corrector(monkeypatch, 2.0, 0.9,
                            ip_of=lambda j: self.IP * 1.01,
                            roundtrip_gate=lambda ip, posterior=None,
                            sigma_Ip=None: ip_roundtrip_gate(
                                ip, self.IP, posterior=posterior,
                                sigma_Ip=sigma_Ip))


# ---------------------------------------------------------------------------
class TestTheGateProductionActuallyPasses:
    """The WIRING, not a wrapper written for the test.

    Every test above builds its own one-positional closure over the
    measurement, so the suite pinned the corrector's contract and never the
    object ``run.py`` hands it.  Production passed
    ``utils.ip_roundtrip_gate`` itself -- whose ``Ip_measured`` is a REQUIRED
    positional -- so the first corrector that ran died with a ``TypeError``
    after paying for its solves, on the channel's headline configuration.
    These tests use ``Bouquet._structured_roundtrip_gate``, which is what the
    call site now builds, with no wrapper in between.
    """

    IP = 1.0e6
    SIG = 0.005 * 1.0e6

    def _gate(self):
        from bouquet.run import Bouquet
        return Bouquet._structured_roundtrip_gate(self.IP)

    def test_the_raw_utils_gate_does_not_satisfy_the_correctors_contract(self):
        """Why the helper exists: binding the corrector's call against the
        unbound function is the TypeError, and a signature check says so
        without paying for a solve."""
        import inspect
        from bouquet.utils import ip_roundtrip_gate

        with pytest.raises(TypeError):
            inspect.signature(ip_roundtrip_gate).bind(
                self.IP, posterior=None, sigma_Ip=None)
        # ... and the bound helper does satisfy it
        inspect.signature(self._gate()).bind(
            self.IP, posterior=None, sigma_Ip=None)

    def test_the_hard_channel_re_gates_through_the_production_object(
            self, monkeypatch):
        rec, world = _stub_corrector(monkeypatch, 2.0, 0.9,
                                     ip_of=lambda j: self.IP,
                                     roundtrip_gate=self._gate())
        assert world["solves"] == 1
        assert rec["structured_roundtrip_post_corrector_reference"] == \
            "Ip_target"
        assert abs(rec["structured_roundtrip_post_corrector_err_pct"]) < 1e-9

    def test_the_soft_channel_re_gates_against_its_own_posterior(
            self, monkeypatch):
        """The pass-through the one-line q0-style closure would have lost: the
        soft channel's reference is the posterior, not the measurement, and
        the tolerance is the same one."""
        post = self.IP * 1.003            # 0.3 %: outside the 0.05 % budget
        rec, world = _stub_corrector(          # if the reference were Ip
            monkeypatch, 2.0, 0.9, li_sigma=0.04, soft=True,
            ip_sigma=self.SIG, ip_post=post, ip_of=lambda j: post,
            roundtrip_gate=self._gate())
        assert world["solves"] == 1
        assert rec["structured_roundtrip_post_corrector_reference"] == \
            "the closure's own posterior Ip"
        assert abs(rec["structured_roundtrip_post_corrector_err_pct"]) < 1e-9

    def test_a_refused_step_is_gated_against_the_predictors_posterior(
            self, monkeypatch):
        """A refused corrector step delivers the PREDICTOR, so ``rec`` never
        gains a posterior of its own.  The re-gate used to fall back to
        Ip_target there and call a 0.3 % soft posterior an algebra error on
        the hard channel's 0.05 % budget; the predictor's posterior, already
        on ``bl.ip_closure``, is the reference."""
        post = self.IP * 1.003
        bl = _FakeBaseline()
        bl.ip_closure["structured_ip_posterior"] = post
        bl.j_phi = np.ones(5)      # the predictor's assembly, still delivered
        rec, world = _stub_corrector(
            monkeypatch, 2.0, 0.9, li_sigma=0.04, soft=True,
            ip_sigma=self.SIG, ip_of=lambda j: post, bl=bl,
            solver_raises=RuntimeError("refused"),
            roundtrip_gate=self._gate())
        assert world["solves"] == 0
        assert "structured_corrector_refusal" in rec
        assert rec["structured_roundtrip_post_corrector_reference"] == \
            "the closure's own posterior Ip"
        assert abs(rec["structured_roundtrip_post_corrector_err_pct"]) < 1e-9

    def test_a_refused_step_with_a_bad_assembly_is_still_refused(
            self, monkeypatch):
        """... and the fallback reference must not have softened the gate."""
        post = self.IP * 1.003
        bl = _FakeBaseline()
        bl.ip_closure["structured_ip_posterior"] = post
        bl.j_phi = np.ones(5)      # the predictor's assembly, still delivered
        with pytest.raises(RuntimeError, match="algebra error"):
            _stub_corrector(
                monkeypatch, 2.0, 0.9, li_sigma=0.04, soft=True,
                ip_sigma=self.SIG, ip_of=lambda j: post * 1.01, bl=bl,
                solver_raises=RuntimeError("refused"),
                roundtrip_gate=self._gate())

    def test_a_bad_assembly_is_still_refused_through_it(self, monkeypatch):
        """Binding the measurement must not have softened the gate."""
        with pytest.raises(RuntimeError, match="algebra error"):
            _stub_corrector(monkeypatch, 2.0, 0.9,
                            ip_of=lambda j: self.IP * 1.01,
                            roundtrip_gate=self._gate())

    def test_the_call_site_builds_its_gate_with_that_helper(self):
        """The one mechanical link the tests above cannot reach: the helper is
        only worth having if the structured call site uses it.  Driving that
        line needs a full IMAS forward solve, so this reads it."""
        import inspect
        from bouquet.run import Bouquet

        src = inspect.getsource(Bouquet)
        i = src.index("self._close_ip_structured_corrector(")
        assert "_structured_roundtrip_gate(" in src[i:i + 400]


# ---------------------------------------------------------------------------
class TestSoftSolverConvergenceRefusals:
    """The two ``RuntimeError``s that decide what the posterior mode delivers
    when it fails.  Nothing reached either of them: the suite only ever
    exercised solves that converged, so the branches that choose between
    "here is an answer" and "there is no answer" were untested.
    """

    def _case(self):
        psi, w, c, ji, jb, jf, lin, Ip_s, lg = _li_parts()
        m, _ = _model()
        return (psi, w, c, ji, jb, jf, Ip_s, lg,
                structured_li_of(m)[0] * 1.03)

    def test_a_run_out_of_iterations_is_refused_not_returned(self):
        """`max_iter` exhausted must raise, never hand back the last iterate:
        a half-converged posterior is not a posterior."""
        psi, w, c, ji, jb, jf, Ip_s, lg, target = self._case()
        with pytest.raises(RuntimeError,
                           match="did not converge in 1 iterations"):
            close_ip_structured_soft(psi, w, c, Ip_s, 0.005 * abs(Ip_s),
                                     ji, jb, jf, li_target=target,
                                     li_sigma=0.03, li_geom=lg, max_iter=1)

    def test_the_refusal_names_the_objective_it_stopped_at(self):
        psi, w, c, ji, jb, jf, Ip_s, lg, target = self._case()
        with pytest.raises(RuntimeError, match=r"objective [0-9.]+e"):
            close_ip_structured_soft(psi, w, c, Ip_s, 0.005 * abs(Ip_s),
                                     ji, jb, jf, li_target=target,
                                     li_sigma=0.03, li_geom=lg, max_iter=2)

    def test_damping_that_cannot_find_a_descent_step_is_refused(self,
                                                                monkeypatch):
        """The other terminal branch.  It is defensive -- no in-spec input
        found in review reaches it -- so the step solver is made to return a
        uselessly uphill direction and the SHIPPED damping loop, gradient test
        and refusal run unmodified around it."""
        real = np.linalg.lstsq

        def _uphill(A, b, rcond=None):
            d = real(A, b, rcond=rcond)
            return (np.full(np.shape(d[0]), 1.0e3),) + tuple(d[1:])
        monkeypatch.setattr(np.linalg, "lstsq", _uphill)
        psi, w, c, ji, jb, jf, Ip_s, lg, target = self._case()
        with pytest.raises(RuntimeError,
                           match="could not find a descent step"):
            close_ip_structured_soft(psi, w, c, Ip_s, 0.005 * abs(Ip_s),
                                     ji, jb, jf, li_target=target,
                                     li_sigma=0.03, li_geom=lg)

    def test_a_converged_solve_is_unaffected_by_either_guard(self):
        psi, w, c, ji, jb, jf, Ip_s, lg, target = self._case()
        out = close_ip_structured_soft(psi, w, c, Ip_s, 0.005 * abs(Ip_s),
                                       ji, jb, jf, li_target=target,
                                       li_sigma=0.03, li_geom=lg)
        assert 1 <= out["n_iter"] < 100
        assert np.isfinite(out["objective"])
