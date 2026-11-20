"""``bouquet.kinrecon.current.bootstrap``: the adapter to the solver's ``eval_bootstrap`` (no solver needed).

A stand-in solver object records what the adapter sends: the option integers that reproduce the engine's
definitions, temperatures in keV, the impurity density in the ion pressure, and the frame signs.  The numerical
equivalence with the engine's earlier Python transcription is the solver test ``test_bootstrap_solver.py``.
``evaluate_geometry`` is checked to trace the Sauter surfaces only once when the evaluation returns their geometry
(its ``sauter_fc`` calls counted on a stand-in), and to store that geometry's f_t and eps each in its own place.
"""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon.current import bootstrap as _bs
from bouquet.kinrecon.current.bootstrap import EC, solver_jdotb, solver_options


class _FakeGS:
    def __init__(self, bounds=(0.1, 0.5)):
        self.psi_bounds = np.array(bounds, dtype=float)
        self.calls = []

    def eval_bootstrap(self, psi, te, ti, ne, ni, Zeff, nimp=None, ln_lambda_model=None, imp_pressure=None,
                       bs_model=None, ion_charge_model=None):
        self.calls.append(dict(psi=psi, te=te, ti=ti, ne=ne, ni=ni, Zeff=Zeff, nimp=nimp,
                               ln_lambda_model=ln_lambda_model, imp_pressure=imp_pressure, bs_model=bs_model,
                               ion_charge_model=ion_charge_model))
        return {"psi_n": np.asarray(psi), "jdotB_bs_raw": np.linspace(1.0, 2.0, len(psi))}


def _profiles(n=12):
    x = np.linspace(0.02, 0.98, n)
    te = 3.0e3 * (1 - x ** 1.5) ** 1.5 + 100.0
    ti = 2.5e3 * (1 - x ** 1.4) ** 1.4 + 120.0
    ne = 5.0e19 * (1 - 0.6 * x ** 2) + 0.5e19
    zeff = 1.8 + 0.2 * x
    ni = ne * (6.0 - zeff) / 5.0
    return x, te, ti, ne, ni, zeff


def test_options_reproduce_the_engine_definitions():
    assert solver_options() == dict(bs_model=1, ln_lambda_model=2, ion_charge_model=1)
    assert solver_options(ln_lambda_model="nrl")["ln_lambda_model"] == 1
    for bad in ("zeff", "unity", "koh"):
        with pytest.raises(ValueError):
            solver_options(ion_collisionality=bad)
    with pytest.raises(ValueError):
        solver_options(ln_lambda_model="spitzer")


def test_call_units_impurity_and_frame():
    x, te, ti, ne, ni, zeff = _profiles()
    nimp = ne * (zeff - 1.0) / 30.0
    F = np.full_like(x, 3.4)
    gs = _FakeGS((0.1, 0.5))                         # psi_axis > psi_boundary
    jb, s = solver_jdotb(gs, x, te, ti, ne, ni, zeff, F, nimp=nimp)
    c = gs.calls[-1]
    np.testing.assert_array_equal(c["te"], te / 1.0e3)
    np.testing.assert_array_equal(c["ti"], ti / 1.0e3)
    np.testing.assert_array_equal(c["nimp"], nimp)
    assert c["imp_pressure"] is True and (c["bs_model"], c["ln_lambda_model"], c["ion_charge_model"]) == (1, 2, 1)
    np.testing.assert_array_equal(jb, np.linspace(1.0, 2.0, x.size))
    assert "impurity" in s["ion_pressure"] and s["evaluation"].startswith("solver")
    # without an impurity density: no impurity in the ion pressure
    jb0, s0 = solver_jdotb(gs, x, te, ti, ne, ni, zeff, F)
    assert gs.calls[-1]["nimp"] is None and gs.calls[-1]["imp_pressure"] is False and "only" in s0["ion_pressure"]
    # the result is odd in F and in the sign of psi_axis - psi_boundary: the positive frame either way
    jb1, _ = solver_jdotb(gs, x, te, ti, ne, ni, zeff, -F, nimp=nimp)
    np.testing.assert_array_equal(jb1, -jb)
    jb2, _ = solver_jdotb(_FakeGS((0.5, 0.1)), x, te, ti, ne, ni, zeff, -F, nimp=nimp)
    np.testing.assert_array_equal(jb2, jb)


def test_geometry_of_the_evaluation_is_passed_through():
    """``return_geometry``: f_t and eps of the solver's pass unchanged (frame signs apply to the current only); None
    from a solver build that returns no geometry."""
    x, te, ti, ne, ni, zeff = _profiles()
    F = np.full_like(x, -3.4)

    class _GeoGS(_FakeGS):
        def eval_bootstrap(self, psi, *a, **kw):
            out = super().eval_bootstrap(psi, *a, **kw)
            return dict(out, f_t=np.linspace(0.1, 0.6, len(psi)), eps=np.linspace(0.01, 0.3, len(psi)))

    jb, _, geo = solver_jdotb(_GeoGS((0.5, 0.1)), x, te, ti, ne, ni, zeff, F, return_geometry=True)
    np.testing.assert_array_equal(geo["f_t"], np.linspace(0.1, 0.6, x.size))
    np.testing.assert_array_equal(geo["eps"], np.linspace(0.01, 0.3, x.size))
    np.testing.assert_array_equal(jb, solver_jdotb(_GeoGS((0.5, 0.1)), x, te, ti, ne, ni, zeff, F)[0])
    assert solver_jdotb(_FakeGS(), x, te, ti, ne, ni, zeff, F, return_geometry=True)[2] is None


class _CountingGS(_FakeGS):
    """A stand-in solver for ``evaluate_geometry``: ``eval_bootstrap`` returns the geometry of its pass or not (a build
    without those outputs), and every ``sauter_fc`` call is counted (a second, identical call changes no value, so
    only a count shows it)."""

    def __init__(self, with_geometry, ft=lambda n: np.full(n, 0.3), eps=lambda n: np.full(n, 0.2),
                 fc_sep=lambda n: np.full(n, 0.6), eps_sep=lambda n: np.full(n, 0.25)):
        super().__init__((0.5, 0.1))
        self.with_geometry = with_geometry
        self.ft, self.eps, self.fc_sep, self.eps_sep = ft, eps, fc_sep, eps_sep
        self.sauter_calls = []

    def eval_bootstrap(self, psi, *a, **kw):
        out = super().eval_bootstrap(psi, *a, **kw)
        if self.with_geometry:
            out = dict(out, f_t=self.ft(len(psi)), eps=self.eps(len(psi)))
        return out

    def sauter_fc(self, psi, return_eps=False):
        self.sauter_calls.append(np.array(psi, dtype=float))
        n = len(psi)
        return None, self.fc_sep(n), None, None, self.eps_sep(n)

    def get_fsa(self, psi):
        one = np.ones(len(psi))
        return {"F": -3.4 * one, "<1/R^2>": 0.3 * one, "<Bp^2>": 0.04 * one, "q": 1.5 * one, "<1/R>": 0.55 * one,
                "<R>": 1.8 * one, "R_min": 1.2 * one, "R_max": 2.3 * one, "psi_boundary": 0.1, "psi_axis": 0.5}

    def get_q(self, psi):
        return psi, np.full(len(psi), 1.1)


def _evaluate_geometry(monkeypatch, gs):
    """``current_model.evaluate_geometry`` on the stand-in ``gs``, its other inputs replaced by simple stand-ins."""
    from bouquet.kinrecon import current_model as CM

    xs = np.linspace(0.05, 1.0, 20)
    layer = SimpleNamespace(psi_n=np.array([0.996, 0.999]))
    monkeypatch.setattr(CM, "from_tokamaker", lambda gs, r0, **kw: SimpleNamespace(meta=dict(
        outer_layer_geometry=layer, psi_n_edge=0.995, psi_n_samples=xs, rho_samples=np.sqrt(xs))))
    monkeypatch.setattr(CM, "rho_map_from_geometry", lambda geom, lay: SimpleNamespace(
        psin_of_rho=lambda r: np.asarray(r, float) ** 2, rho_of_psin=lambda p: np.sqrt(np.asarray(p, float))))
    monkeypatch.setattr(CM, "neoclassical_conductivity", lambda ne, *a, **kw: (np.ones_like(ne), {}))

    def profiles_at(psi_n, rho):
        x = np.asarray(psi_n, float)
        zeff = 1.8 + 0.2 * x
        ne = 5.0e19 * (1 - 0.6 * x ** 2) + 0.5e19
        return dict(te=3.0e3 * (1 - x) + 100.0, ti=2.5e3 * (1 - x) + 120.0, ne=ne, ni=ne * (6.0 - zeff) / 5.0,
                    zeff=zeff)

    return CM.evaluate_geometry(gs, SimpleNamespace(profiles_at=profiles_at), CM.CurrentModelSettings(r0=1.7))


def test_evaluate_geometry_makes_no_separate_sauter_fc_call_when_the_bootstrap_returned_the_geometry(monkeypatch):
    """``current_model.evaluate_geometry``: NO separate ``sauter_fc`` call when the bootstrap evaluation returned the
    geometry of its own pass (f_t and eps taken from it), and exactly one, on the same points, when it did not."""
    gs = _CountingGS(with_geometry=True)
    lag = _evaluate_geometry(monkeypatch, gs)
    assert len(gs.calls) == 1 and gs.sauter_calls == []
    np.testing.assert_array_equal(lag["phys"]["ft"], np.full(len(gs.calls[0]["psi"]), 0.3))
    np.testing.assert_array_equal(lag["phys"]["eps"], np.full(len(gs.calls[0]["psi"]), 0.2))

    gs = _CountingGS(with_geometry=False)
    lag = _evaluate_geometry(monkeypatch, gs)
    assert len(gs.calls) == 1 and len(gs.sauter_calls) == 1
    np.testing.assert_array_equal(gs.sauter_calls[0], gs.calls[0]["psi"])
    n = len(gs.calls[0]["psi"])
    np.testing.assert_array_equal(lag["phys"]["ft"], 1.0 - np.full(n, 0.6))
    np.testing.assert_array_equal(lag["phys"]["eps"], np.full(n, 0.25))


def test_evaluate_geometry_stores_f_t_and_eps_of_the_evaluation_not_interchanged(monkeypatch):
    """The ``ft`` and ``eps`` that ``evaluate_geometry`` stores are the bootstrap evaluation's ``f_t`` and ``eps``
    respectively (and, in the fallback, ``1 - f_c`` and ``eps`` of the separate pass), never interchanged: the
    stand-in's two profiles differ on every point."""
    def ft(n):
        return np.linspace(0.05, 0.70, n)

    def eps(n):
        return np.linspace(0.01, 0.35, n) ** 2

    def fc_sep(n):
        return np.linspace(0.95, 0.30, n)

    gs = _CountingGS(with_geometry=True, ft=ft, eps=eps)
    lag = _evaluate_geometry(monkeypatch, gs)
    n = len(gs.calls[0]["psi"])
    assert np.all(np.abs(ft(n) - eps(n)) > 0.01)
    np.testing.assert_array_equal(lag["phys"]["ft"], ft(n))
    np.testing.assert_array_equal(lag["phys"]["eps"], eps(n))
    gs = _CountingGS(with_geometry=False, fc_sep=fc_sep, eps_sep=eps)
    lag = _evaluate_geometry(monkeypatch, gs)
    assert np.all(np.abs((1.0 - fc_sep(n)) - eps(n)) > 0.01)
    np.testing.assert_array_equal(lag["phys"]["ft"], 1.0 - fc_sep(n))
    np.testing.assert_array_equal(lag["phys"]["eps"], eps(n))


def test_refusals():
    x, te, ti, ne, ni, zeff = _profiles()
    F = np.full_like(x, 3.4)
    with pytest.raises(ValueError):
        solver_jdotb(_FakeGS(), x, te, ti, ne, ni, zeff, F, nimp=-ne * 0.01)
    with pytest.raises(ValueError):
        solver_jdotb(_FakeGS(), x, te, ti, ne, ni, zeff, np.where(x < 0.5, F, -F))
    with pytest.raises(ValueError):
        solver_jdotb(_FakeGS((0.3, 0.3)), x, te, ti, ne, ni, zeff, F)
    with pytest.raises(ValueError):
        solver_jdotb(_FakeGS(), x, te, ti, ne, ni, zeff, F, ion_collisionality="zeff")


def test_sauter_coulomb_logarithm_is_the_default_everywhere():
    """The bootstrap uses Sauter's Coulomb logarithms by default, in the module and in the current model's settings
    (the NRL pair is a comparison option only)."""
    from bouquet.kinrecon.current_model import CurrentModelSettings
    assert inspect.signature(_bs.solver_jdotb).parameters["ln_lambda_model"].default == "sauter"
    assert CurrentModelSettings(r0=1.7).bootstrap_ln_lambda_model == "sauter"
    assert CurrentModelSettings(r0=1.7).ion_collisionality == "spike"
    assert EC == 1.602176634e-19
