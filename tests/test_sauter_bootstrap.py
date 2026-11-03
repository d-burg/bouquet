"""The Sauter bootstrap model (``GenerationConfig.bootstrap_model="sauter"``).

``physics.sauter_bootstrap`` is Sauter, Angioni and Lin-Liu, Phys. Plasmas 6,
2834 (1999), Eqs. 14-17, with the ``alpha`` sign correction of the erratum,
Phys. Plasmas 9, 5140 (2002), in the ``jboot1`` form of OpenFUSIONToolkit's
``redl_bootstrap``.  ``evaluate_jBS(bootstrap_model="sauter")`` swaps the
coefficient fit and nothing else: ``nu_e*``, ``nu_i*``, ``eps``, ``f_T``, the
gradients, the ``kappa`` conversion and ``p'G`` are the Redl evaluation's.

* the coefficients against an independent implementation: OMFIT's
  ``sauter_bootstrap(version="jboot1")``, frozen on synthetic profiles in
  ``tests/data/sauter_omfit_reference.json`` (``make_sauter_reference.py``;
  nu_e* 0.012-110, Zeff 1.2-3.5) and, where omfit_classes is importable,
  recomputed live;
* the closed-form limits (banana regime, the erratum's sign at high nu_i*);
* ``evaluate_jBS`` on the mock equilibrium of ``test_jbs_loop``: only the fit
  changes, the default is bit for bit the Redl evaluator, the version tag
  says which fit ran;
* the setting: validated, carried by the loop settings and the records,
  refused on the toolkit routes (Redl only) instead of being ignored.

Synthetic inputs only.
"""
import json
import os

import numpy as np
import pytest

from bouquet.config import GenerationConfig, check_bootstrap_model_route
from bouquet.physics import (BOOTSTRAP_MODEL_DEFAULT, BOOTSTRAP_MODELS,
                             EVALUATE_JBS_VERSION, check_bootstrap_model,
                             evaluate_jBS, evaluate_jbs_version,
                             sauter_bootstrap)
from test_jbs_loop import _MockEq, _kin

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
X = np.linspace(0.0, 1.0, 151)


def _ref():
    with open(os.path.join(_DATA, "sauter_omfit_reference.json")) as f:
        return json.load(f)


def _ours(r):
    i = {k: np.asarray(v, dtype=float) for k, v in r["inputs"].items()}
    return sauter_bootstrap(
        Te=i["Te"], Ti=i["Ti"], ne=i["ne"], ni=i["ni"],
        pe=np.asarray(r["pe"]), pi=np.asarray(r["pi"]),
        Zeff=np.asarray(r["Zeff_used"]), q=i["q"], fT=i["fT"],
        I_psi=i["I_psi"], dT_e_dpsi=i["dT_e_dpsi"], dT_i_dpsi=i["dT_i_dpsi"],
        dn_e_dpsi=i["dn_e_dpsi"], dn_i_dpsi=i["dn_i_dpsi"],
        nu_e_star=np.asarray(r["nu_e_star"]),
        nu_i_star=np.asarray(r["nu_i_star"]))


def _check_against(r):
    j, co = _ours(r)
    i = {k: np.asarray(v, dtype=float) for k, v in r["inputs"].items()}
    pe, pi = np.asarray(r["pe"]), np.asarray(r["pi"])
    dp = (i["ne"] * i["dT_e_dpsi"] + i["Te"] * i["dn_e_dpsi"]
          + i["ni"] * i["dT_i_dpsi"] + i["Ti"] * i["dn_i_dpsi"]) * (
        1.602176634e-19)
    R_pe = pe / (pe + pi)
    np.testing.assert_allclose(j, r["j_dot_B"], rtol=1e-12)
    np.testing.assert_allclose(co["alpha"], r["alpha"], rtol=1e-12)
    np.testing.assert_allclose(co["alpha0"], r["alpha0"], rtol=1e-12)
    # the three bracket terms separately: L31, L32 and L34 alpha
    np.testing.assert_allclose(co["L31"] * dp / pe, r["term1"], rtol=1e-12)
    np.testing.assert_allclose(co["L32"] * i["dT_e_dpsi"] / i["Te"],
                               r["term2"], rtol=1e-12)
    np.testing.assert_allclose(
        co["L34"] * co["alpha"] * (1 - R_pe) / R_pe * i["dT_i_dpsi"] / i["Ti"],
        r["term3"], rtol=1e-12)


# ---------------------------------------------------------------------------
#  the coefficients against OMFIT's implementation
# ---------------------------------------------------------------------------
def test_matches_the_frozen_omfit_jboot1_reference():
    r = _ref()
    # the reference spans the banana regime to a collisional edge
    assert min(r["nu_e_star"]) < 0.05 and max(r["nu_e_star"]) > 10.0
    assert min(r["Zeff_used"]) < 1.3 and max(r["Zeff_used"]) > 3.0
    _check_against(r)


def test_matches_omfit_live():
    """Recompute the reference with the installed OMFIT: it still agrees with
    the frozen file and with ours (skipped without omfit_classes)."""
    pytest.importorskip("omfit_classes.utils_fusion")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_sauter_reference",
        os.path.join(_DATA, "make_sauter_reference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    live = mod.build()
    np.testing.assert_allclose(live["j_dot_B"], _ref()["j_dot_B"], rtol=1e-12)
    _check_against(live)


# ---------------------------------------------------------------------------
#  closed-form limits
# ---------------------------------------------------------------------------
def _flat(fT, nu_e, nu_i, Z):
    one = np.ones(1)
    return sauter_bootstrap(
        Te=1e3 * one, Ti=1e3 * one, ne=1e20 * one, ni=1e20 * one,
        pe=1.6e4 * one, pi=1.6e4 * one, Zeff=Z * one, q=2.0 * one,
        fT=fT * one, I_psi=3.0 * one, dT_e_dpsi=-1e3 * one,
        dT_i_dpsi=-1e3 * one, dn_e_dpsi=-1e19 * one, dn_i_dpsi=-1e19 * one,
        nu_e_star=nu_e * one, nu_i_star=nu_i * one)[1]


@pytest.mark.parametrize("fT,Z", [(0.3, 1.0), (0.6, 2.0), (0.8, 3.5)])
def test_banana_limit(fT, Z):
    """nu* = 0: f_teff = f_T, L34 = L31 = F31(f_T) (Eq. 14a) and
    alpha = alpha0 (Eq. 17b)."""
    co = _flat(fT, 0.0, 0.0, Z)
    F31 = ((1 + 1.4 / (Z + 1)) * fT - 1.9 / (Z + 1) * fT ** 2
           + 0.3 / (Z + 1) * fT ** 3 + 0.2 / (Z + 1) * fT ** 4)
    np.testing.assert_allclose(co["L31"], F31, rtol=1e-14)
    np.testing.assert_allclose(co["L34"], co["L31"], rtol=1e-14)
    a0 = -1.17 * (1 - fT) / (1 - 0.22 * fT - 0.19 * fT ** 2)
    np.testing.assert_allclose(co["alpha"], a0, rtol=1e-14)


def test_untrapped_plasma_carries_no_bootstrap():
    co = _flat(0.0, 0.3, 0.3, 1.5)
    for k in ("L31", "L32", "L34"):
        assert abs(float(co[k][0])) == 0.0


def test_alpha_has_the_2002_erratum_sign():
    """At large nu_i* the f_T^6 terms dominate Eq. 17a: alpha -> +0.315/0.15
    = +2.1 with the erratum (Sauter 2002), -2.1 without it."""
    co = _flat(0.6, 0.1, 1e8, 1.5)
    np.testing.assert_allclose(co["alpha"], 0.315 / 0.15, rtol=1e-3)


# ---------------------------------------------------------------------------
#  evaluate_jBS: only the fit changes
# ---------------------------------------------------------------------------
@pytest.fixture
def oft():
    return pytest.importorskip(
        "OpenFUSIONToolkit.TokaMaker.bootstrap",
        reason="evaluate_jBS wraps OFT's Coulomb logarithms and Redl; "
               "OpenFUSIONToolkit is not importable here")


def test_default_is_redl_bit_for_bit(oft):
    assert BOOTSTRAP_MODEL_DEFAULT == "redl"
    j0, d0 = evaluate_jBS(_MockEq(), X, *_kin(X))
    j1, d1 = evaluate_jBS(_MockEq(), X, *_kin(X), bootstrap_model="redl")
    np.testing.assert_array_equal(j0, j1)
    np.testing.assert_array_equal(d0["j_dot_B"], d1["j_dot_B"])
    assert d0["bootstrap_model"] == "redl"
    assert d0["version"] == EVALUATE_JBS_VERSION == evaluate_jbs_version()
    assert "Redl 2021 jboot1" in EVALUATE_JBS_VERSION
    assert "OPT-IN" not in EVALUATE_JBS_VERSION


@pytest.mark.parametrize("eps_definition", [None, "a_over_R"])
def test_sauter_changes_only_the_fit(oft, eps_definition):
    kin = _kin(X)
    jr, dr = evaluate_jBS(_MockEq(), X, *kin, eps_definition=eps_definition)
    js, ds = evaluate_jBS(_MockEq(), X, *kin, eps_definition=eps_definition,
                          bootstrap_model="sauter")
    # the shared inputs are identical
    for k in ("nu_e_star", "nu_i_star", "eps", "f_T", "q", "F", "avg_inv_R",
              "avg_B2", "R_nu_star", "j_pressure"):
        np.testing.assert_array_equal(ds[k], dr[k], err_msg=k)
    assert ds["bootstrap_model"] == "sauter"
    assert ds["version"].startswith("evaluate_jBS/4 (OPT-IN Sauter 1999")
    assert ds["version"] == evaluate_jbs_version(
        ds["eps_definition"], "sauter")
    # <j.B> is sauter_bootstrap on those inputs
    ne, te, ni, ti, zeff = (np.broadcast_to(np.asarray(a, float), X.shape)
                            for a in kin)
    dpsi = lambda y: np.gradient(y, X, edge_order=2) / ds["dpsi"]
    jB, _co = sauter_bootstrap(
        Te=te, Ti=ti, ne=ne, ni=ni, pe=1.602176634e-19 * ne * te,
        pi=1.602176634e-19 * ni * ti, Zeff=zeff, q=ds["q"], fT=ds["f_T"],
        I_psi=ds["F"], dT_e_dpsi=dpsi(te), dT_i_dpsi=dpsi(ti),
        dn_e_dpsi=dpsi(ne), dn_i_dpsi=dpsi(ni), nu_e_star=ds["nu_e_star"],
        nu_i_star=ds["nu_i_star"])
    inner = (X > 1e-3) & (X < 1 - 1e-3)
    np.testing.assert_allclose(ds["j_dot_B"][inner], jB[inner], rtol=1e-12)
    # the same kappa conversion
    kap = ds["F"] * ds["avg_inv_R"] / ds["avg_B2"]
    np.testing.assert_allclose(ds["j_tor_full_raw"][inner],
                               (kap * ds["j_dot_B"])[inner], rtol=1e-12)
    # and it is a different fit
    assert np.max(np.abs(ds["j_dot_B"] / dr["j_dot_B"] - 1)[inner]) > 1e-2


def test_unknown_model_is_refused(oft):
    with pytest.raises(ValueError, match="bootstrap_model"):
        evaluate_jBS(_MockEq(), X, *_kin(X), bootstrap_model="sauter2002")
    with pytest.raises(ValueError, match="bootstrap_model"):
        check_bootstrap_model("Sauter")
    assert sorted(BOOTSTRAP_MODELS) == ["redl", "sauter"]


# ---------------------------------------------------------------------------
#  the setting
# ---------------------------------------------------------------------------
def test_setting_reaches_the_loop_settings_and_records():
    from bouquet.engine import eps_record
    from bouquet.jbs_loop import jbs_settings
    assert "bootstrap_model" not in jbs_settings(GenerationConfig())
    s = jbs_settings(GenerationConfig(bootstrap_model="sauter"))
    assert s["bootstrap_model"] == "sauter"
    rec = eps_record(None, "sauter")
    assert rec["bootstrap_model"] == "sauter"
    assert "OPT-IN Sauter" in rec["evaluate_jBS_version"]
    # a default record keeps its pre-option keys
    assert "bootstrap_model" not in eps_record(None)


def test_engine_settings_carry_the_model():
    from bouquet.engine import engine_settings
    gc = GenerationConfig(reconstruction_engine="unified",
                          bootstrap_model="sauter")
    assert engine_settings(gc)["bootstrap_model"] == "sauter"


def test_refused_where_the_toolkit_bootstrap_would_ignore_it():
    with pytest.raises(ValueError, match="Redl only"):
        GenerationConfig(solve_method="swb", bootstrap_model="sauter")
    with pytest.raises(ValueError, match="Redl only"):
        GenerationConfig(reconstruction_engine="legacy",
                         jbs_self_consistent=False, bootstrap_model="sauter")
    # an edit after construction is caught where the run is resolved
    gc = GenerationConfig(bootstrap_model="sauter")
    object.__setattr__(gc, "jbs_self_consistent", False)
    object.__setattr__(gc, "reconstruction_engine", "legacy")
    with pytest.raises(ValueError, match="Redl only"):
        check_bootstrap_model_route(gc)
    # Redl is accepted everywhere
    GenerationConfig(solve_method="swb", bootstrap_model="redl")
