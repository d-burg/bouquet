"""The inverse aspect ratio of the Redl collisionalities in ``evaluate_jBS``.

Owner decision E4 (2026-10-09): the default is the geometric
``eps = (R_max - R_min)/(2<R>)`` (``evaluate_jBS/4``), computed on EVERY
OpenFUSIONToolkit build -- from ``sauter_fc(return_eps=True)`` where the
installed build has it (the fork), else from ``get_fsa``'s ``R_min`` /
``R_max`` over the ``sauter_fc`` ``<R>`` (OpenFUSIONToolkit main) -- and the
``<a>/<R>`` of ``evaluate_jBS/1..3`` is the explicit opt-in
``eps_definition="a_over_R"``.  It never raises for the missing fork option.

The mock (``test_jbs_loop._MockEq``) has shaped surfaces whose half-width is
``0.85 <a>``, so the two definitions differ by 15 % and a test sees which one
was used.  Synthetic inputs only.
"""
import numpy as np
import pytest

pytest.importorskip(
    "OpenFUSIONToolkit.TokaMaker.bootstrap",
    reason="evaluate_jBS wraps OFT's pure-Python Redl implementation; "
           "OpenFUSIONToolkit is not importable here")

from bouquet.physics import (EPS_DEFINITION_DEFAULT, EPS_DEFINITIONS,  # noqa: E402
                             EVALUATE_JBS_VERSION, evaluate_jBS,
                             evaluate_jbs_version, geometric_eps)
from test_jbs_loop import _MockEq, _kin  # noqa: E402

X = np.linspace(0.0, 1.0, 151)


def _eps_expected(eq, x, definition):
    psi_eval = np.clip(x, 1e-3, 1.0 - 1e-3)
    _p, r, _e, R = eq._geo(psi_eval)
    return (eq.HALF_WIDTH * r / R) if definition == "geometric" else r / R


def test_the_default_is_the_geometric_eps_and_says_so():
    assert EPS_DEFINITION_DEFAULT == "geometric"
    assert EVALUATE_JBS_VERSION.startswith("evaluate_jBS/4")
    assert "(R_max-R_min)/(2<R>)" in EVALUATE_JBS_VERSION
    eq = _MockEq()
    _j, d = evaluate_jBS(eq, X, *_kin(X))
    np.testing.assert_allclose(d["eps"], _eps_expected(eq, X, "geometric"),
                               rtol=1e-13)
    assert d["eps_definition"] == "geometric"
    assert d["eps_formula"] == EPS_DEFINITIONS["geometric"]
    assert d["version"] == EVALUATE_JBS_VERSION


def test_without_the_fork_option_it_comes_from_get_fsa_and_never_raises():
    """OpenFUSIONToolkit main: ``sauter_fc`` has no ``return_eps`` (the
    mock ignores it and returns the 4-tuple) -- the epsilon is computed from
    ``get_fsa``, not refused."""
    eq = _MockEq(fork=False)
    j, d = evaluate_jBS(eq, X, *_kin(X))
    assert np.all(np.isfinite(j))
    assert d["eps_route"] == "get_fsa"
    assert any(name == "get_fsa" for name, _a in eq.calls)


class _MainSauter(_MockEq):
    """``sauter_fc`` with OpenFUSIONToolkit main's exact signature (no
    ``return_eps``, no ``**kwargs``): passing the keyword would raise."""

    def sauter_fc(self, psi=None, psi_pad=0.02, npsi=50):
        return super().sauter_fc(psi=psi)


def test_a_signature_without_return_eps_is_never_called_with_it():
    j, d = evaluate_jBS(_MainSauter(), X, *_kin(X))
    assert d["eps_route"] == "get_fsa" and np.all(np.isfinite(j))


def test_the_fork_route_is_used_where_the_build_has_it():
    eq = _MockEq(fork=True)
    _j, d = evaluate_jBS(eq, X, *_kin(X))
    assert d["eps_route"] == "sauter_fc(return_eps=True)"
    assert not any(name == "get_fsa" for name, _a in eq.calls)
    np.testing.assert_allclose(d["eps"], _eps_expected(eq, X, "geometric"),
                               rtol=1e-13)


def test_the_two_routes_agree_to_rounding_where_both_exist():
    """Same surfaces, both routes: the fork's ``sauter_fc(return_eps=True)``
    and ``get_fsa``'s ``R_min``/``R_max`` over the ``sauter_fc`` ``<R>``.
    (Live: ``test_jbs_loop_solver`` repeats this on a solved equilibrium and
    skips -- the comparison only, not the evaluation -- on builds without the
    fork option.)"""
    eq = _MockEq(fork=True)
    psi_u = np.unique(np.clip(X, 1e-3, 1.0 - 1e-3))
    e_fork, r1 = geometric_eps(eq, psi_u)
    R = eq.sauter_fc(psi=psi_u)[2]["<R>"]
    e_fsa, r2 = geometric_eps(_MockEq(fork=False), psi_u, R_avg=R)
    assert (r1, r2) == ("sauter_fc(return_eps=True)", "get_fsa")
    np.testing.assert_allclose(e_fsa, e_fork, rtol=1e-14, atol=0.0)
    j_main, _ = evaluate_jBS(_MockEq(fork=False), X, *_kin(X))
    j_fork, _ = evaluate_jBS(_MockEq(fork=True), X, *_kin(X))
    np.testing.assert_allclose(j_main, j_fork, rtol=1e-12, atol=1e-9)


def test_the_opt_in_is_the_a_over_R_of_evaluate_jBS_3_and_is_named():
    eq = _MockEq()
    j_geo, d_geo = evaluate_jBS(_MockEq(), X, *_kin(X), smooth_axis=False)
    j_aR, d_aR = evaluate_jBS(eq, X, *_kin(X), smooth_axis=False,
                              eps_definition="a_over_R")
    np.testing.assert_allclose(d_aR["eps"], _eps_expected(eq, X, "a_over_R"),
                               rtol=1e-13)
    assert d_aR["eps_definition"] == "a_over_R"
    assert d_aR["eps_route"] == "sauter_fc <a>/<R>"
    assert "OPT-IN eps = <a>/<R>" in d_aR["version"]
    assert d_aR["version"] == evaluate_jbs_version("a_over_R")
    assert d_aR["version"] != EVALUATE_JBS_VERSION
    assert not any(name == "get_fsa" for name, _a in eq.calls)
    # nu* ~ eps^-3/2: the geometric eps (0.85x) raises nu* by 0.85^-1.5
    inner = (X > 0.01) & (X < 0.99)
    np.testing.assert_allclose(
        d_geo["nu_e_star"][inner] / d_aR["nu_e_star"][inner],
        np.full(inner.sum(), 0.85 ** -1.5), rtol=1e-10)
    # ... so the bootstrap moves (the default physics change is visible)
    assert np.max(np.abs(j_geo - j_aR)) > 1e-3 * np.max(np.abs(j_aR))


def test_an_unknown_definition_is_refused():
    with pytest.raises(ValueError, match="eps_definition"):
        evaluate_jBS(_MockEq(), X, *_kin(X), eps_definition="inverse_aspect")


class _NoFsaNoFork(_MainSauter):
    get_fsa = None


def test_a_build_with_neither_route_is_refused_by_name_not_silently():
    """Pre-v26.6 OFT (no ``get_fsa``) without the fork option: the default
    cannot be computed -- a named refusal that points at the opt-in, which
    still works there."""
    with pytest.raises(RuntimeError, match="a_over_R"):
        evaluate_jBS(_NoFsaNoFork(), X, *_kin(X))
    j, d = evaluate_jBS(_NoFsaNoFork(), X, *_kin(X),
                        eps_definition="a_over_R")
    assert np.all(np.isfinite(j)) and d["eps_definition"] == "a_over_R"
