"""``bouquet.kinrecon.current.bootstrap``: the adapter to the solver's ``eval_bootstrap`` (no solver needed).

A stand-in solver object records what the adapter sends: the option integers that reproduce the engine's
definitions, temperatures in keV, the impurity density in the ion pressure, and the frame signs.  The numerical
equivalence with the engine's earlier Python transcription is the solver test ``test_bootstrap_solver.py``.
"""
import inspect

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
