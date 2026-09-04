"""Unit tests for bouquet.physics -- pure numpy, no OFT/solver required.

Covers the two convention reductions used by the baseline resolvers and the
per-draw bootstrap recompute:

  * isotropize_fast_pressure -- anisotropic fast pressure -> scalar GS pressure
  * parallel_to_toroidal     -- FSA parallel current <j.B> -> toroidal
                                <j_phi/R>/<1/R>, ratio + analytic methods
"""

import numpy as np
import pytest

from bouquet.physics import (isotropize_fast_pressure, parallel_to_toroidal,
                             toroidal_to_parallel)


def _fsa_metrics_circular(R0=1.7, a=0.55, F=3.4, Bp0=0.35, npol=20000):
    """FSA metrics on one analytic flux surface via the *proper* operator
    <A> = oint (A/B_p) dl / oint dl/B_p  (Wesson 4th ed. sec 4.4).

    Concentric circular surface R=R0+a*cos(th), Z=a*sin(th) (finite aspect
    ratio eps=a/R0), with a theta-varying poloidal field Bp = Bp0/(1+eps*cos)
    (the large-aspect-ratio 1/R fall-off) so the FSA is genuinely weighted, not
    a plain theta-average. Returns a geom dict + the raw per-angle fields for
    independent-path checks.
    """
    th = np.linspace(0.0, 2 * np.pi, npol, endpoint=False)
    R = R0 + a * np.cos(th)
    dl = a                                   # |d(R,Z)/dth| for a circle (const)
    Bp = Bp0 / (1.0 + (a / R0) * np.cos(th))
    Bphi = F / R
    B2 = Bphi**2 + Bp**2
    w = dl / Bp                              # FSA weight  dl/B_p
    def fsa(A):
        return np.sum(A * w) / np.sum(w)
    geom = {
        "F": F,
        "avg_inv_R": fsa(1.0 / R),
        "avg_inv_R2": fsa(1.0 / R**2),
        "avg_B2": fsa(B2),
    }
    return geom, dict(th=th, R=R, Bp=Bp, Bphi=Bphi, B2=B2, F=F, fsa=fsa)


# ---------------------------------------------------------------------------
# Physics benchmark: proper-FSA quadrature vs the conversion formula, and the
# <B_phi^2> = F^2 <1/R^2> identity. Verified against Wesson sec 4.4 (FSA def,
# field-aligned + Pfirsch-Schlueter split) and the IMAS/EUROfusion convention
# j_phi == <J^phi>/<1/R>.
# ---------------------------------------------------------------------------
class TestFSAQuadratureBenchmark:
    def test_forward_matches_independent_fsa_quadrature(self):
        # For a field-aligned current j = lambda*B: the code's formula output
        # must equal <j_phi/R>/<1/R> computed by DIRECT FSA quadrature (a fully
        # independent path from the closed-form formula).
        geom, f = _fsa_metrics_circular()
        lam = 4.2e5                                  # lambda = <j.B>/<B^2>
        jdotB = lam * geom["avg_B2"]                 # <j.B> on the surface
        # independent direct path: j_phi = lambda*Bphi, then FSA of j_phi/R
        j_tor_direct = f["fsa"]((lam * f["Bphi"]) / f["R"]) / geom["avg_inv_R"]
        j_tor_formula = parallel_to_toroidal(np.array([jdotB]), geom=geom)[0]
        assert np.isclose(j_tor_formula, j_tor_direct, rtol=1e-10)

    def test_bphi2_identity(self):
        # <B_phi^2> = F^2 <1/R^2> exactly (B_phi = F/R), the identity the
        # analytic bracket relies on.
        geom, f = _fsa_metrics_circular()
        avg_Bphi2 = f["fsa"](f["Bphi"] ** 2)
        assert np.isclose(avg_Bphi2, f["F"] ** 2 * geom["avg_inv_R2"], rtol=1e-12)

    def test_finite_aspect_ratio_correction_is_real(self):
        # sanity: at eps~0.32 the geometric factor departs from the cylinder
        # limit by a non-trivial amount (so the test isn't vacuous)
        geom, _ = _fsa_metrics_circular()
        cyl = geom["F"] * geom["avg_inv_R2"] / (geom["avg_B2"] * geom["avg_inv_R"])
        assert not np.isclose(cyl, 1.0, atol=1e-3)   # genuine O(eps^2)+Bp effect


# ---------------------------------------------------------------------------
# toroidal_to_parallel -- the IDS write-back inverse
# ---------------------------------------------------------------------------
class TestToroidalToParallel:
    def test_round_trip_exact(self):
        # forward(inverse) == identity to machine precision with full geom
        geom, _ = _fsa_metrics_circular()
        jdotB = np.array([9.1e5, 4.0e5, -1.5e5])
        j_tor = parallel_to_toroidal(jdotB, geom=geom)
        back = toroidal_to_parallel(j_tor, geom=geom)
        assert np.allclose(back, jdotB, rtol=1e-11)

    def test_round_trip_with_b0(self):
        # IMAS input/output normalised by B0 must also round-trip
        geom, _ = _fsa_metrics_circular()
        B0 = 2.0
        j_par_imas = np.array([1.0e6, 3.0e5])
        j_tor = parallel_to_toroidal(j_par_imas, geom={**geom, "B0": B0})
        back = toroidal_to_parallel(j_tor, geom={**geom, "B0": B0})
        assert np.allclose(back, j_par_imas, rtol=1e-11)

    def test_bracket_one_fallback_round_trips(self):
        # without <1/R^2> both directions use bracket=1, so they still invert
        # each other exactly (self-consistent, just not machine-exact physics)
        geom, _ = _fsa_metrics_circular()
        g = {"F": geom["F"], "avg_inv_R": geom["avg_inv_R"], "avg_B2": geom["avg_B2"]}
        jdotB = np.array([7.7e5])
        back = toroidal_to_parallel(parallel_to_toroidal(jdotB, geom=g), geom=g)
        assert np.allclose(back, jdotB, rtol=1e-11)

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            toroidal_to_parallel(np.array([1.0]), geom={"F": 3.4, "avg_B2": 4.0})


# ---------------------------------------------------------------------------
# Exact <1/R^2> flux-surface quadrature (headless, via a mock equilibrium)
# ---------------------------------------------------------------------------
class _MockCircleEquil:
    """Minimal mygs stand-in: concentric circular surfaces psi_hat=(r/a0)^2,
    R=R0+r*cos, Z=r*sin, with an analytic large-aspect-ratio poloidal field
    Bp = Bp0*R0/R and Bphi = F/R, so <1/R^2> etc. have quadrature references."""
    def __init__(self, R0=1.7, a0=0.5, F=3.4, Bp0=0.35, npts=1200):
        self.R0, self.a0, self.F, self.Bp0, self.npts = R0, a0, F, Bp0, npts

    def get_field_eval(self, field_type):
        assert field_type == "B"
        R0, F, Bp0 = self.R0, self.F, self.Bp0
        class _E:
            def eval(self, p):
                R, Z = float(p[0]), float(p[1])
                Bp = Bp0 * R0 / R                    # |B_p| shape ~ 1/R
                # split Bp into (R,Z) comps along the circle tangent direction;
                # magnitude is what matters for the FSA weight
                return np.array([Bp * 0.6, F / R, Bp * 0.8])  # hypot(.6,.8)=1
        return _E()

    def trace_surf(self, psi):
        r = self.a0 * np.sqrt(psi)
        th = np.linspace(0, 2 * np.pi, self.npts, endpoint=False)
        return np.column_stack([self.R0 + r * np.cos(th), r * np.sin(th)])


class TestExactInvR2Quadrature:
    def test_fsa_matches_analytic_reference(self):
        from bouquet.physics import _capture_exact_inv_R2
        from bouquet.utils import safe_trace_surf
        eq = _MockCircleEquil()
        psi = np.array([0.25, 0.64])                 # r = a0*sqrt(psi)
        inv_R2, inv_R_chk, _ = _capture_exact_inv_R2(eq, psi, safe_trace_surf)
        # independent reference: same proper FSA on a dense analytic circle
        for k, ps in enumerate(psi):
            r = eq.a0 * np.sqrt(ps)
            th = np.linspace(0, 2 * np.pi, 40000, endpoint=False)
            R = eq.R0 + r * np.cos(th)
            w = (r * (2 * np.pi / th.size)) / (eq.Bp0 * eq.R0 / R)   # dl/Bp
            ref_inv_R2 = np.sum((1 / R**2) * w) / np.sum(w)
            assert np.isclose(inv_R2[k], ref_inv_R2, rtol=1e-3)

    def test_native_read_accepts_valid_extended_sauter(self):
        # forward-compat: when sauter_fc grows <1/R^2> (geo row 3) and <B_phi^2>
        # (bfield row 2), the native reader returns <1/R^2> and passes the
        # F^2<1/R^2>==<B_phi^2> identity + Jensen.
        from bouquet.physics import _native_fsa_inv_R2
        n = 20
        avg_R = np.full(n, 1.7); inv_R = np.full(n, 0.60); a = np.full(n, 0.5)
        inv_R2 = inv_R**2 * 1.03                      # > <1/R>^2 (Jensen ok)
        F = np.full(n, 3.4)
        Babs = np.full(n, 2.1); B2 = np.full(n, 4.4)
        bphi2 = F**2 * inv_R2                          # exact identity
        geo = np.vstack([avg_R, inv_R, a, inv_R2])     # extended geo
        bfield = np.vstack([Babs, B2, bphi2])          # extended bfield
        got = _native_fsa_inv_R2(geo, bfield, F, inv_R)
        assert got is not None and np.allclose(got, inv_R2)

    def test_native_read_rejects_absent_and_wrong(self):
        from bouquet.physics import _native_fsa_inv_R2
        n = 20
        inv_R = np.full(n, 0.60); F = np.full(n, 3.4)
        geo3 = np.vstack([np.full(n, 1.7), inv_R, np.full(n, 0.5)])   # no 4th row
        assert _native_fsa_inv_R2(geo3, np.vstack([np.full(n, 2.1), np.full(n, 4.4)]),
                                  F, inv_R) is None
        # 4th row present but violates Jensen (< <1/R>^2) -> rejected
        bad = np.vstack([np.full(n, 1.7), inv_R, np.full(n, 0.5), inv_R**2 * 0.5])
        assert _native_fsa_inv_R2(bad, np.vstack([np.full(n, 2.1), np.full(n, 4.4)]),
                                  F, inv_R) is None
        # 4th row passes Jensen but <B_phi^2> identity fails -> rejected
        cand = inv_R**2 * 1.03
        geo4 = np.vstack([np.full(n, 1.7), inv_R, np.full(n, 0.5), cand])
        bf_wrong = np.vstack([np.full(n, 2.1), np.full(n, 4.4), F**2 * cand * 1.5])
        assert _native_fsa_inv_R2(geo4, bf_wrong, F, inv_R) is None

    def test_self_check_inv_R_consistent(self):
        # the quadrature's own <1/R> must match a direct dense average (the
        # gate that guards against bad B_p ordering / trace shape at runtime)
        from bouquet.physics import _capture_exact_inv_R2
        from bouquet.utils import safe_trace_surf
        eq = _MockCircleEquil()
        psi = np.array([0.49])
        _, inv_R_chk, _ = _capture_exact_inv_R2(eq, psi, safe_trace_surf)
        r = eq.a0 * 0.7
        th = np.linspace(0, 2 * np.pi, 40000, endpoint=False)
        R = eq.R0 + r * np.cos(th)
        w = 1.0 / (eq.Bp0 * eq.R0 / R)
        assert np.isclose(inv_R_chk[0], np.sum(w / R) / np.sum(w), rtol=1e-3)


# ---------------------------------------------------------------------------
# isotropize_fast_pressure
# ---------------------------------------------------------------------------
class TestIsotropizeFastPressure:
    def setup_method(self):
        self.p_perp = np.array([3.0e4, 2.0e4, 0.0])
        self.p_par = np.array([1.5e4, 2.0e4, 0.0])

    def test_trace(self):
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="trace")
        assert np.allclose(out, (2 * self.p_perp + self.p_par) / 3.0)

    def test_mean(self):
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="mean")
        assert np.allclose(out, (self.p_perp + self.p_par) / 2.0)

    def test_perp(self):
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="perp")
        assert np.allclose(out, self.p_perp)

    def test_isotropic_input_is_identity(self):
        # p_perp == p_par -> every reduction returns the same scalar pressure
        p = np.array([1.0e4, 5.0e3])
        for method in ("trace", "mean", "perp"):
            assert np.allclose(isotropize_fast_pressure(p, p, method=method), p)

    def test_trace_preserves_energy_density(self):
        # w = (p_par + 2 p_perp)/2 = (3/2) p_scalar for the trace reduction
        out = isotropize_fast_pressure(self.p_perp, self.p_par, method="trace")
        w = 0.5 * (self.p_par + 2 * self.p_perp)
        assert np.allclose(1.5 * out, w)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            isotropize_fast_pressure(np.zeros(3), np.zeros(4))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown p_fast reduction"):
            isotropize_fast_pressure(self.p_perp, self.p_par, method="rms")


# ---------------------------------------------------------------------------
# parallel_to_toroidal -- ratio method
# ---------------------------------------------------------------------------
class TestParallelToToroidalRatio:
    def test_component_scales_with_total_ratio(self):
        j_par_tot = np.array([2.0e6, 1.0e6, 5.0e5])
        j_tor_tot = np.array([1.8e6, 0.95e6, 4.6e5])
        j_comp = np.array([1.0e5, 2.0e5, 3.0e5])
        out = parallel_to_toroidal(
            j_comp, j_parallel_total=j_par_tot, j_tor_total=j_tor_tot)
        assert np.allclose(out, j_comp * j_tor_tot / j_par_tot)

    def test_zero_crossing_filled_from_neighbours(self):
        # surface where the total parallel current passes through zero:
        # the ill-defined ratio must be interpolated, not propagated as inf
        j_par_tot = np.array([2.0e6, 0.0, 1.0e6])
        j_tor_tot = np.array([1.0e6, 0.0, 0.5e6])
        j_comp = np.ones(3)
        out = parallel_to_toroidal(
            j_comp, j_parallel_total=j_par_tot, j_tor_total=j_tor_tot)
        assert np.all(np.isfinite(out))
        assert np.isclose(out[1], 0.5)  # interpolated between c=0.5 and c=0.5

    def test_all_zero_total_raises(self):
        with pytest.raises(ValueError, match="cannot form ratio"):
            parallel_to_toroidal(
                np.ones(3), j_parallel_total=np.zeros(3), j_tor_total=np.zeros(3))


# ---------------------------------------------------------------------------
# parallel_to_toroidal -- analytic (field-aligned) method
# ---------------------------------------------------------------------------
class TestParallelToToroidalAnalytic:
    def test_cylinder_limit_is_identity(self):
        # B_p -> 0 and no R variation: j_tor == j_parallel exactly
        R0, B0 = 1.7, 2.0
        F = R0 * B0
        j_par = np.array([1.0e6, 5.0e5, 2.0e5])
        geom = {
            "F": np.full(3, F),
            "avg_inv_R": np.full(3, 1.0 / R0),
            "avg_B2": np.full(3, B0**2),
            "avg_inv_R2": np.full(3, 1.0 / R0**2),
        }
        out = parallel_to_toroidal(j_par * B0, geom=geom)  # raw <j.B>
        assert np.allclose(out, j_par)

    def test_imas_b0_normalisation(self):
        # IMAS convention input <j.B>/B0 with geom["B0"] must equal raw <j.B>
        R0, B0 = 1.7, 2.0
        geom = {
            "F": R0 * B0, "avg_inv_R": 1.0 / R0,
            "avg_B2": B0**2, "avg_inv_R2": 1.0 / R0**2,
        }
        j_par_imas = np.array([1.0e6])
        raw = parallel_to_toroidal(j_par_imas * B0, geom=dict(geom))
        viaB0 = parallel_to_toroidal(j_par_imas, geom={**geom, "B0": B0})
        assert np.allclose(raw, viaB0)

    def test_exact_formula_on_shaped_surface(self):
        # j_tor = <j.B> F <1/R^2> / (<B^2> <1/R>) reproduced exactly when
        # <1/R^2> is supplied
        R0, eps, F = 1.7, 0.36, 3.4
        th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        Rs = R0 * (1 + eps * np.cos(th))
        inv_R, inv_R2 = np.mean(1 / Rs), np.mean(1 / Rs**2)
        B2 = F**2 * inv_R2 * 1.008  # ~0.8% poloidal-field content
        jB = 1.0e6
        out = parallel_to_toroidal(
            np.array([jB]),
            geom={"F": F, "avg_inv_R": inv_R, "avg_B2": B2, "avg_inv_R2": inv_R2},
        )
        assert np.isclose(out[0], jB * F * inv_R2 / (B2 * inv_R), rtol=1e-12)

    def test_missing_inv_R2_error_is_order_Bp_over_B_squared(self):
        # dropping <1/R^2> must only cost the <B_p^2>/<B^2> bracket (<~1%)
        R0, eps, F = 1.7, 0.36, 3.4
        th = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        Rs = R0 * (1 + eps * np.cos(th))
        inv_R, inv_R2 = np.mean(1 / Rs), np.mean(1 / Rs**2)
        bp_frac = 0.008  # <B_p^2>/<B_phi^2>
        B2 = F**2 * inv_R2 * (1 + bp_frac)
        jB = np.array([1.0e6])
        geom = {"F": F, "avg_inv_R": inv_R, "avg_B2": B2}
        exact = parallel_to_toroidal(jB, geom={**geom, "avg_inv_R2": inv_R2})
        approx = parallel_to_toroidal(jB, geom=geom)
        rel = abs(approx[0] - exact[0]) / abs(exact[0])
        assert rel < 0.01
        assert np.isclose(rel, bp_frac, rtol=0.05)

    def test_cocos_sign_safety(self):
        # flipping the signs of F and <j.B> together (opposite-helicity COCOS)
        # must not flip or distort the toroidal output magnitude
        R0, F = 1.7, 3.4
        geom = {"F": F, "avg_inv_R": 1.0 / R0, "avg_B2": (F / R0) ** 2}
        out_pos = parallel_to_toroidal(np.array([1.0e6]), geom=dict(geom))
        out_neg = parallel_to_toroidal(np.array([-1.0e6]), geom={**geom, "F": -F})
        assert np.isclose(out_pos[0], out_neg[0])

    def test_missing_geom_key_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            parallel_to_toroidal(np.ones(2), geom={"F": 1.0})

    def test_no_method_selected_raises(self):
        with pytest.raises(ValueError, match="ratio method"):
            parallel_to_toroidal(np.ones(2))


# ---------------------------------------------------------------------------
# impurity derivation -- effective_impurity_charge / main_ion_density_from_zeff
# ---------------------------------------------------------------------------
from bouquet.physics import effective_impurity_charge, main_ion_density_from_zeff


class TestImpurityDerivation:
    def test_round_trip_consistent_baseline(self):
        # build a consistent (ne, ni, zeff) set from known carbon Z=6, then
        # recover Z_imp and re-derive ni exactly
        Z = 6.0
        psi = np.linspace(0, 1, 50)
        ne = 5e19 * (1 - 0.8 * psi**2)
        zeff = 2.0 + 0.3 * psi          # 2.0 .. 2.3
        ni = ne * (Z - zeff) / (Z - 1.0)
        Z_rec = effective_impurity_charge(ne, ni, zeff)
        assert Z_rec is not None
        assert np.isclose(Z_rec, Z, rtol=1e-10)
        ni_rec = main_ion_density_from_zeff(ne, zeff, Z_rec)
        assert np.allclose(ni_rec, ni)

    def test_no_dilution_returns_none(self):
        # the IDA ni = ne workflow carries no dilution information
        ne = np.full(20, 4e19)
        assert effective_impurity_charge(ne, ne.copy(), np.full(20, 1.8)) is None

    def test_physical_bounds(self):
        # for 1 <= zeff <= Z_imp: 0 <= ni <= ne and nz >= 0
        Z = 6.0
        ne = np.full(11, 5e19)
        zeff = np.linspace(1.0, Z, 11)
        ni = main_ion_density_from_zeff(ne, zeff, Z)
        nz = (ne - ni) / Z
        assert np.all(ni >= -1e-6) and np.all(ni <= ne + 1e-6)
        assert np.all(nz >= -1e-6)
        assert np.isclose(ni[0], ne[0])     # zeff=1 -> pure plasma
        assert np.isclose(ni[-1], 0.0)      # zeff=Z -> fully diluted

    def test_invalid_z_imp_raises(self):
        with pytest.raises(ValueError, match="exceed 1"):
            main_ion_density_from_zeff(np.ones(3), np.ones(3), 1.0)

    def test_zeff_floor_ignored_in_median(self):
        # surfaces with zeff <= 1 or negligible dilution must not poison Z_imp
        Z = 6.0
        ne = np.full(30, 5e19)
        zeff = np.full(30, 2.0)
        ni = ne * (Z - zeff) / (Z - 1.0)
        zeff_noisy = zeff.copy(); zeff_noisy[:3] = 0.99   # bad edge points
        ni_noisy = ni.copy(); ni_noisy[3:6] = ne[3:6]     # zero-dilution points
        Z_rec = effective_impurity_charge(ne, ni_noisy, zeff_noisy)
        assert Z_rec is not None and np.isclose(Z_rec, Z, rtol=1e-9)


class TestImpurityRoundTrip:
    def test_ida_derivation_recovers_Z_imp(self):
        # IDA reader path: (ne, Zeff, Z=6) -> ni; effective_impurity_charge
        # back-derives Z_imp = 6 from the resulting (ne, ni, Zeff)
        Z = 6.0
        ne = 5e19 * (1 - 0.7 * np.linspace(0, 1, 40)**2)
        zeff = 1.8 + 0.4 * np.linspace(0, 1, 40)
        ni = main_ion_density_from_zeff(ne, np.clip(zeff, 1, Z), Z)
        assert np.all((ni >= 0) & (ni <= ne))
        assert np.isclose(effective_impurity_charge(ne, ni, zeff), Z, rtol=1e-9)

    def test_tungsten_Z(self):
        # a high-Z machine: ni stays close to ne (tiny W density dilutes a lot)
        ZW = 74.0
        ne = np.full(20, 1e20)
        zeff = np.full(20, 1.6)
        ni = main_ion_density_from_zeff(ne, zeff, ZW)
        assert np.all(ni < ne) and np.all(ni > 0.98 * ne)  # ni/ne = (74-1.6)/73
        assert np.isclose((ne[0] - ni[0]) / ne[0], (zeff[0] - 1) / (ZW - 1), rtol=1e-9)


# ---------------------------------------------------------------------------
# fast_pressure_residual / infer_fast_pressure
# ---------------------------------------------------------------------------
class TestFastPressure:
    def setup_method(self):
        from bouquet.physics import impurity_pressure
        self.psi = np.linspace(0, 1, 64)
        self.ne = np.full(64, 5e19)
        self.te = np.full(64, 1500.0)
        self.ni = np.full(64, 4.5e19)
        self.ti = np.full(64, 1500.0)
        self.Zimp = 6.0
        e = 1.602176634e-19
        self._p_th = (e * (self.ne * self.te + self.ni * self.ti)
                      + impurity_pressure(self.ne, self.ni, self.ti, self.Zimp))

    def test_residual_recovers_added_fast(self):
        from bouquet.physics import fast_pressure_residual
        p_tot = self._p_th + 3000.0
        pf = fast_pressure_residual(self.psi, self.ne, self.te, self.ni, self.ti,
                                    self.Zimp, self.psi, p_tot)
        assert np.allclose(pf, 3000.0, atol=1e-6)

    def test_residual_clips_negative(self):
        from bouquet.physics import fast_pressure_residual
        pf = fast_pressure_residual(self.psi, self.ne, self.te, self.ni, self.ti,
                                    self.Zimp, self.psi, self._p_th - 5000.0)
        assert np.all(pf >= 0.0)

    def test_infer_valid_core_peaked(self):
        from bouquet.physics import infer_fast_pressure
        fast = 8000.0 * np.exp(-(self.psi / 0.3) ** 2)   # core-peaked beam
        pf, info = infer_fast_pressure(self.psi, self.ne, self.te, self.ni,
                                       self.ti, self.Zimp, self.psi, self._p_th + fast)
        assert info["valid"] is True
        assert np.isclose(info["peak_psi_N"], 0.0, atol=0.05)
        assert np.isclose(pf[0], 8000.0, rtol=0.02)

    def test_infer_invalid_thermal_exceeds_total_on_axis(self):
        from bouquet.physics import infer_fast_pressure
        # total below thermal near axis (magnetics EFIT) -> invalid -> zeros
        bad = self._p_th.copy()
        bad[self.psi <= 0.15] -= 6000.0
        pf, info = infer_fast_pressure(self.psi, self.ne, self.te, self.ni,
                                       self.ti, self.Zimp, self.psi, bad)
        assert info["valid"] is False
        assert np.allclose(pf, 0.0)
        assert "not a kinetic-EFIT" in info["message"]


# ---------------------------------------------------------------------------
# radial_field_from_impurity_force_balance  (E_r via impurity force balance)
# ---------------------------------------------------------------------------
class TestRadialField:
    def test_three_terms_and_units(self):
        from bouquet.physics import radial_field_from_impurity_force_balance
        psi = np.linspace(0, 1, 50)
        n = np.full(50, 1e19); t = np.full(50, 1000.0)   # flat -> no diamagnetic
        omega = np.full(50, 1e5); vpol = np.full(50, 1e3)
        Bpol = np.full(50, 0.2); Rmaj = np.full(50, 1.8); dpsidr = np.full(50, 2.0)
        Bphi = np.full(50, -2.0)
        Er, info = radial_field_from_impurity_force_balance(psi, n, t, omega, vpol, Bpol, Rmaj, dpsidr, Bphi, Z_imp=6.0)
        assert np.allclose(info["diamagnetic"], 0.0, atol=1.0)         # flat p
        assert np.allclose(info["toroidal"], 1e5 * 1.8 * 0.2)          # v_phi B_theta
        assert np.allclose(info["poloidal"], -1e3 * (-2.0))            # -v_theta B_phi
        assert np.allclose(Er, info["diamagnetic"] + info["toroidal"] + info["poloidal"])

    def test_diamagnetic_sign_outward_decreasing_pressure(self):
        from bouquet.physics import radial_field_from_impurity_force_balance
        psi = np.linspace(0, 1, 60)
        n = np.full(60, 1e19); t = 2000.0 * (1 - psi ** 2) + 50.0     # decreasing outward
        z = np.zeros(60)
        Er, info = radial_field_from_impurity_force_balance(psi, n, t, z, z, z, np.full(60, 1.8),
                                         np.full(60, 2.0), z, Z_imp=6.0)
        # dp/dR < 0, positive charge -> diamagnetic E_r <= 0
        assert np.all(info["diamagnetic"][1:-1] <= 1e-9)

    def test_sigma_propagation_positive(self):
        from bouquet.physics import radial_field_from_impurity_force_balance
        psi = np.linspace(0, 1, 40)
        n = np.full(40, 1e19); t = np.full(40, 1000.0)
        omega = np.full(40, 1e5); vpol = np.full(40, 1e3)
        Bpol = np.full(40, 0.2); Rmaj = np.full(40, 1.8); dpsidr = np.full(40, 2.0)
        Bphi = np.full(40, -2.0)
        _, info = radial_field_from_impurity_force_balance(psi, n, t, omega, vpol, Bpol, Rmaj, dpsidr, Bphi,
                                        sigma_omega_tor=np.full(40, 1e4),
                                        sigma_v_pol=np.full(40, 1e2))
        # toroidal sigma = |R Bpol| * sigma_omega ; poloidal sigma = |Bphi| * sigma_vpol
        exp = np.sqrt((1.8 * 0.2 * 1e4) ** 2 + (2.0 * 1e2) ** 2)
        assert np.allclose(info["sigma"], exp)


class TestFloorInductiveSplit:
    """j_inductive >= 0 convention: negative pedestal residuals are floored and
    absorbed into j_BS with the total exactly preserved (201586@4200 regression:
    a negative GPR mean rejected all 500 candidate draws)."""

    def test_floor_and_absorb(self):
        import numpy as np
        from bouquet.baseline import floor_inductive_split
        psi = np.linspace(0, 1, 11)
        j_ind = np.array([1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.02, -0.01, 0.0]) * 1e6
        j_bs = np.full(11, 0.3e6)
        tot = j_ind + j_bs
        ji2, jb2 = floor_inductive_split(j_ind, j_bs, psi)
        assert (ji2 >= 0).all()
        assert np.allclose(ji2 + jb2, tot)               # sum preserved exactly
        assert ji2[8] == 0.0 and jb2[8] == tot[8]        # deficit absorbed

    def test_noop_when_nonnegative(self):
        import numpy as np
        from bouquet.baseline import floor_inductive_split
        j_ind = np.linspace(1.5e6, 0.0, 9)
        j_bs = np.full(9, 0.2e6)
        ji2, jb2 = floor_inductive_split(j_ind, j_bs)
        assert np.array_equal(ji2, j_ind) and np.array_equal(jb2, j_bs)


def test_isotropize_sum_recovers_imas_per_dof_fast_pressure():
    """IMAS.jl stores pressa/3 in each directional field; 'sum' recovers pressa,
    'trace' returns pressa/3 (the defect seen on FUSE dds)."""
    from bouquet.physics import isotropize_fast_pressure
    pressa = np.array([3.0e4, 1.5e4, 0.0])
    p_perp = p_par = pressa / 3.0
    assert np.allclose(isotropize_fast_pressure(p_perp, p_par, method="sum"), pressa)
    assert np.allclose(isotropize_fast_pressure(p_perp, p_par, method="trace"), pressa / 3.0)
