"""jBS_baseline_mode='ohmic': the FSA current measure and the workflow guard.

Solve-free coverage of the machinery the ohmic Ip closure is built on:

* ``Ip_fsa_weights`` is AFFINE -- ``I_p[J] = trapezoid(w J, psi_N) + c`` --
  and the P'-term constant ``c`` is counted exactly once (the first closure
  implementation triple-counted it via per-component integral calls).
* The two profile conventions must agree exactly on the equilibrium's own
  Grad-Shafranov profile: integrating ``eq_jphi_profile(conv)`` under the
  matching measure gives the SAME current for 'jphi-linterp' and 'fsa'.
  That identity is what the run-time roundtrip gate validates on a real
  geometry; here it is checked algebraically on a synthetic one.
* The workflow validator accepts the new mode on the IMAS path but refuses
  to generate() with it (baseline-only until the draw path's sigma=0
  reproduction is verified), with the usual workflow='custom' downgrade.

No solver: the geometry is a synthetic ``fsa_current_geometry``-shaped dict.
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import trapezoid

from bouquet.utils import Ip_fsa_weights, Ip_fsa_integral, eq_jphi_profile

_N = 101


def _geom():
    """A plausible fsa_current_geometry() dict for a D3D-like shape."""
    psi = np.linspace(0.01, 0.99, _N)
    R0, a = 1.7, 0.6
    r = a * np.sqrt(psi)                        # minor-radius proxy
    R_avg = R0 + 0.1 * r ** 2 / a               # Shafranov-ish outward shift
    inv_R = (1.0 / R0) * (1.0 + 0.05 * (r / a) ** 2)
    inv_R2 = inv_R ** 2 * (1.0 + 0.02 * (r / a) ** 2)
    dV_dpsi = 4.0 * np.pi ** 2 * R0 * r * (a / (2.0 * np.sqrt(psi) + 1e-12))
    pprime = -8.0e3 * (1.0 - psi)               # falls to 0 at the edge
    FFp = -0.4 * (1.0 - psi) ** 2
    return {
        "psi_N": psi, "psi_q": psi, "R_avg": R_avg,
        "inv_R": inv_R, "inv_R2": inv_R2, "dV_dpsi": dV_dpsi,
        "dpsi_dpsiN": 0.9, "pprime": pprime, "FFp": FFp,
    }


class TestAffineMeasure:
    def test_integral_is_weights_dot_profile_plus_c(self):
        g = _geom()
        rng = np.random.default_rng(3)
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        for _ in range(3):
            j = 1e6 * rng.standard_normal(_N)
            direct = Ip_fsa_integral(None, g["psi_N"], j,
                                     convention="jphi-linterp", geom=g)
            assert direct == pytest.approx(
                float(trapezoid(w * j, g["psi_N"])) + c, rel=1e-12)

    def test_c_is_counted_once_not_per_component(self):
        """Summing per-component Ip_fsa_integral calls over-counts c -- the
        linear-parts closure exists precisely to avoid that."""
        g = _geom()
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        assert c != 0.0
        parts = [np.full(_N, 2.0e5), np.full(_N, 1.0e5), np.full(_N, 5.0e4)]
        total = Ip_fsa_integral(None, g["psi_N"], sum(parts),
                                convention="jphi-linterp", geom=g)
        summed = sum(Ip_fsa_integral(None, g["psi_N"], p,
                                     convention="jphi-linterp", geom=g)
                     for p in parts)
        assert summed - total == pytest.approx(2.0 * c, rel=1e-9)

    def test_fsa_convention_is_purely_linear(self):
        _, c = Ip_fsa_weights(_geom(), convention="fsa")
        assert c == 0.0

    def test_conventions_agree_on_the_equilibrium_profile(self):
        """The roundtrip identity behind the run-time 0.5% gate, exactly."""
        g = _geom()
        for sign in (1.0, -1.0):
            ips = {}
            for conv in ("jphi-linterp", "fsa"):
                prof = eq_jphi_profile(g, convention=conv, pprime_sign=sign)
                ips[conv] = Ip_fsa_integral(None, g["psi_N"], prof,
                                            convention=conv,
                                            pprime_sign=sign, geom=g)
            assert ips["fsa"] == pytest.approx(ips["jphi-linterp"], rel=1e-12)

    def test_pprime_sign_moves_only_the_affine_term(self):
        g = _geom()
        wp, cp = Ip_fsa_weights(g, convention="jphi-linterp", pprime_sign=1.0)
        wm, cm = Ip_fsa_weights(g, convention="jphi-linterp", pprime_sign=-1.0)
        np.testing.assert_array_equal(wp, wm)
        assert cm == pytest.approx(-cp, rel=1e-12)

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="unknown convention"):
            Ip_fsa_weights(_geom(), convention="cylindrical")

    def test_jphi_linterp_requires_inv_R2_and_pprime(self):
        g = _geom(); g["inv_R2"] = None
        with pytest.raises(ValueError, match="<1/R\\^2>"):
            Ip_fsa_weights(g, convention="jphi-linterp")
        g = _geom(); g["pprime"] = None
        with pytest.raises(ValueError, match="P'"):
            Ip_fsa_weights(g, convention="jphi-linterp")


class TestClosureAlgebra:
    """The channel scale formulas close Ip exactly on the affine measure."""

    def _parts(self):
        g = _geom()
        w, c = Ip_fsa_weights(g, convention="jphi-linterp")
        psi = g["psi_N"]
        j_ind = 8.0e5 * (1.0 - psi) ** 1.5
        j_bs = 3.0e5 * np.exp(-((psi - 0.9) / 0.06) ** 2)   # pedestal hump
        j_fix = 1.0e5 * (1.0 - psi) ** 3
        lin = lambda j: float(trapezoid(w * j, psi))
        return g, w, c, j_ind, j_bs, j_fix, lin

    def test_bootstrap_channel_closes_exactly(self):
        g, w, c, j_ind, j_bs, j_fix, lin = self._parts()
        Ip_t = 1.1 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)  # 10% deficit
        s_bs = (Ip_t - c - lin(j_ind) - lin(j_fix)) / lin(j_bs)
        closed = Ip_fsa_integral(None, g["psi_N"],
                                 j_ind + s_bs * j_bs + j_fix,
                                 convention="jphi-linterp", geom=g)
        assert closed == pytest.approx(Ip_t, rel=1e-12)

    def test_ohmic_channel_closes_exactly(self):
        g, w, c, j_ind, j_bs, j_fix, lin = self._parts()
        Ip_t = 0.93 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
        s_ohm = (Ip_t - c - lin(j_bs) - lin(j_fix)) / lin(j_ind)
        closed = Ip_fsa_integral(None, g["psi_N"],
                                 s_ohm * j_ind + j_bs + j_fix,
                                 convention="jphi-linterp", geom=g)
        assert closed == pytest.approx(Ip_t, rel=1e-12)


class TestWorkflowGuard:
    def _config(self, mode, workflow="auto"):
        from bouquet.config import (BouquetConfig, ImasSource, SolverConfig)
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t")
        cfg.generation.jBS_baseline_mode = mode
        cfg.generation.perturb_jind_in_anchor = True     # diff+C baseline rule
        cfg.generation.workflow = workflow
        return cfg

    def _validate(self, cfg):
        from bouquet.run import Bouquet
        Bouquet(cfg)._validate_workflow()

    def test_diff_and_rescale_pass(self):
        for mode in ("diff", "rescale"):
            self._validate(self._config(mode))

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="jBS_baseline_mode"):
            self._validate(self._config("hybrid"))

    def test_ohmic_mode_refuses_generate(self):
        """Baseline-only until the draw path's sigma=0 reproduction is
        verified: generate()'s validator must refuse it."""
        with pytest.raises(ValueError, match="baseline-only"):
            self._validate(self._config("ohmic"))

    def test_custom_workflow_downgrades_to_warning(self, capsys):
        self._validate(self._config("ohmic", workflow="custom"))
        assert "baseline-only" in capsys.readouterr().out


class TestDefaults:
    def test_closure_channel_default_and_baseline_fields(self):
        from bouquet.config import GenerationConfig
        from bouquet.baseline import Baseline
        assert GenerationConfig().closure_channel == "bootstrap"
        assert Baseline.__dataclass_fields__["ohm_scale"].default == 1.0
        assert Baseline.__dataclass_fields__["ip_closure"].default is None
