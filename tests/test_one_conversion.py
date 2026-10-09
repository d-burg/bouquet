"""One ``<j.B>`` -> ``<j_phi>`` conversion in the package.

Every site that turns a field-aligned parallel current ``<j.B>`` into the
toroidal current the solver consumes uses the engine's factor

    kappa = F <1/R> / <B^2>            (bouquet.physics.field_aligned_conversion)

-- the solver-consistent one: OFT's ``jphi-linterp`` consumes the plain
flux-surface average ``<j_phi>``, and for ``j = lambda(psi) B`` (``lambda =
<j.B>/<B^2>``, ``B_phi = F/R``) ``<j_phi> = lambda F <1/R> = kappa <j.B>``
exactly.  Before 2026-10-06 the legacy sites used ``<j.B>/(F<1/R>)`` (no
``<1/R^2>``) and the IDS exporter ``<j.B>F<1/R^2>/(<B^2><1/R>)``.

Checked here, on the synthetic example's own flux-surface averages (the
golden fixture's captured ``eq_fsa`` block):

* ``physics.parallel_to_toroidal`` / ``toroidal_to_parallel``,
  ``physics.evaluate_jBS`` (its toroidal output), ``physics.
  _swb_jbs_to_toroidal`` (SWB's ``R_avg/F`` output) and
  ``engine.conversion_factor`` agree with kappa bit for bit;
* the installed toolkit's SWB output convention is identified, never
  guessed (``physics.swb_jbs_convention``), and a toolkit that returns the
  bootstrap as TokaMaker jphi has ``p'G`` taken off it;
* the old legacy factor exceeded kappa by exactly the bracket
  ``<B^2>/<B_phi^2>`` times the Jensen ratio ``<1/R^2>/<1/R>^2``;
* the engine's composition is unchanged: its request is rebuilt bit for bit
  from the literal kappa formula (this fails if kappa itself moved), and the
  engine's TokaMaker backend consumes ``evaluate_jBS``'s PARALLEL output,
  never its toroidal one.

Synthetic inputs only; no solver, no device data.
"""
import os
import sys
import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet import physics  # noqa: E402
from bouquet.engine import compose, conversion_factor  # noqa: E402

_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden",
                       "D3Dlike_Hmode_golden_slim.h5")


def _example_fsa():
    """The first archived draw's captured flux-surface averages of the
    synthetic g-file example (the golden fixture)."""
    with h5py.File(_GOLDEN, "r") as hf:
        scan = hf["scan/0"]
        keys = sorted((k for k in scan if k.isdigit()
                       and "eq_fsa" in scan[k]), key=int)
        assert keys, "the golden fixture carries no eq_fsa block"
        g = scan[keys[0]]["eq_fsa"]
        return {k: np.asarray(g[k][()], dtype=float) for k in g}


@pytest.fixture(scope="module")
def fsa():
    return _example_fsa()


def _kappa_literal(F, inv_R, B2):
    # the engine's factor, written out independently of the package
    return F * inv_R / B2


def _jdotB(psi):
    """A bootstrap-like parallel current [T A/m^2] with a pedestal peak."""
    return 2.0 * (1.5e5 * (1.0 - psi) + 1.2e6 * np.exp(
        -0.5 * ((psi - 0.96) / 0.015) ** 2))


# ---------------------------------------------------------------------------
#  the conversion functions
# ---------------------------------------------------------------------------
def test_parallel_to_toroidal_is_kappa_and_its_inverse_is_exact(fsa):
    F, iR, B2 = fsa["F"], fsa["avg_inv_R"], fsa["avg_B2"]
    jB = _jdotB(fsa["psi_N"])
    geom = {"F": F, "avg_inv_R": iR, "avg_B2": B2}
    kap = _kappa_literal(F, iR, B2)
    jt = physics.parallel_to_toroidal(jB, geom=geom)
    np.testing.assert_array_equal(jt, jB * kap)
    np.testing.assert_array_equal(
        physics.field_aligned_conversion(F, iR, B2), kap)
    np.testing.assert_array_equal(
        conversion_factor({"F": F, "inv_R": iR, "B2": B2}), kap)
    back = physics.toroidal_to_parallel(jt, geom=geom)
    np.testing.assert_allclose(back, jB, rtol=1e-14, atol=0.0)
    # the IMAS normalisation <j.B>/B0 round-trips the same way
    B0 = 2.1
    jt0 = physics.parallel_to_toroidal(jB / B0, geom={**geom, "B0": B0})
    np.testing.assert_allclose(jt0, jt, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        physics.toroidal_to_parallel(jt0, geom={**geom, "B0": B0}), jB / B0,
        rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("fn", ["parallel_to_toroidal",
                                "toroidal_to_parallel"])
def test_a_geometry_carrying_inv_R2_is_refused(fsa, fn):
    geom = {"F": fsa["F"], "avg_inv_R": fsa["avg_inv_R"],
            "avg_B2": fsa["avg_B2"], "avg_inv_R2": fsa["avg_inv_R2"]}
    with pytest.raises(ValueError, match="avg_inv_R2"):
        getattr(physics, fn)(np.ones_like(fsa["F"]), geom=geom)
    # an explicit None is the absent key
    getattr(physics, fn)(np.ones_like(fsa["F"]),
                         geom={**geom, "avg_inv_R2": None})


def test_the_old_legacy_factor_was_bracket_times_jensen_high(fsa):
    """``<j.B>/(F<1/R>)`` over ``kappa <j.B>`` is ``<B^2>/(F^2<1/R>^2)`` =
    ``[<B^2>/<B_phi^2>] [<1/R^2>/<1/R>^2]`` (``<B_phi^2> = F^2 <1/R^2>``),
    and both factors are >= 1 (``<B^2> >= <B_phi^2>``; Jensen).  On this
    example at psi_N ~ 0.97: bracket ~1.4 %, Jensen ~5.2 %, ~6.7 % in all
    (the moved legacy bootstrap at the pedestal)."""
    F, iR, iR2, B2 = (fsa["F"], fsa["avg_inv_R"], fsa["avg_inv_R2"],
                      fsa["avg_B2"])
    old = 1.0 / (F * iR)
    ratio = old / _kappa_literal(F, iR, B2)
    bracket = B2 / (F ** 2 * iR2)
    jensen = iR2 / iR ** 2
    np.testing.assert_allclose(ratio, bracket * jensen, rtol=1e-12, atol=0.0)
    assert np.all(bracket >= 1.0) and np.all(jensen >= 1.0 - 1e-12)


# ---------------------------------------------------------------------------
#  evaluate_jBS and _swb_jbs_to_toroidal, on a mock of the example
# ---------------------------------------------------------------------------
class _ExampleEq:
    """A mygs stand-in serving the example's captured averages: the
    primitives ``evaluate_jBS`` (``psi=``) and ``_swb_jbs_to_toroidal``
    (``npsi=``, ``psi_pad=`` or ``psi=``) read.  ``<R>`` and ``<a>`` are not captured;
    plausible synthetic values stand in (neither enters the conversion)."""

    def __init__(self, fsa):
        self.f = fsa
        self.psi_bounds = [-0.5, 0.0]

    def _at(self, key, psi):
        return np.interp(psi, self.f["psi_N"], self.f[key])

    def _grid(self, psi, npsi, psi_pad):
        if psi is not None:
            return np.asarray(psi, dtype=float)
        g = np.linspace(psi_pad, 1.0 - psi_pad, int(npsi))
        np.testing.assert_allclose(g, self.f["psi_N"], rtol=0, atol=1e-15)
        return self.f["psi_N"].copy()

    def _R(self, psi):
        return 1.03 / self._at("avg_inv_R", psi)

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        p = self._grid(psi, npsi, psi_pad)
        F = self._at("F", p)
        z = np.zeros_like(p)
        return p, F, z, z, z

    def sauter_fc(self, psi=None, npsi=None, psi_pad=None, return_eps=False):
        p = self._grid(psi, npsi, psi_pad)
        r = {"<R>": self._R(p), "<1/R>": self._at("avg_inv_R", p),
             "<a>": 0.6 * np.sqrt(p) + 1e-3}
        modb = np.array([self._at("B_avg", p), self._at("avg_B2", p)])
        out = (p, self._at("f_trap", p), r, modb)
        return out + ((r["<a>"] / r["<R>"],) if return_eps else ())

    def get_q(self, psi=None, npsi=None, psi_pad=None, compute_geo=False):
        p = self._grid(psi, npsi, psi_pad)
        r = {"<R>": self._R(p), "<1/R>": self._at("avg_inv_R", p),
             "dV/dPsi": np.abs(self._at("dV_dpsi", p)) + 1.0}
        return p, self._at("q", p), r, None, None, None


@pytest.fixture
def fake_redl(monkeypatch, fsa):
    """OFT's Redl module replaced by one that returns a prescribed
    ``<j.B>`` (the conversion, not the Redl physics, is under test)."""
    state = {}

    def redl_bootstrap(psi_N=None, **kw):
        jb = _jdotB(np.asarray(psi_N, dtype=float))
        state["jB"] = jb
        return jb.copy(), {}

    def calculate_ln_lambda(te, ti, ne, ni, zeff, **kw):
        return np.full_like(te, 17.0), np.full_like(te, 17.0)

    bs = types.ModuleType("OpenFUSIONToolkit.TokaMaker.bootstrap")
    bs.redl_bootstrap = redl_bootstrap
    bs.calculate_ln_lambda = calculate_ln_lambda
    tm = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    tm.bootstrap = bs
    top = types.ModuleType("OpenFUSIONToolkit")
    top.TokaMaker = tm
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", top)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker", tm)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker.bootstrap",
                        bs)
    return state


def test_evaluate_jBS_toroidal_output_is_kappa(fsa, fake_redl):
    eq = _ExampleEq(fsa)
    psi = np.linspace(0.0, 1.0, 201)
    n = psi.size
    kin = dict(ne=np.full(n, 5e19), te=np.full(n, 1e3), ni=np.full(n, 4e19),
               ti=np.full(n, 1e3), zeff=np.full(n, 2.0))
    _j, d = physics.evaluate_jBS(eq, psi, kin["ne"], kin["te"], kin["ni"],
                                 kin["ti"], kin["zeff"], smooth_axis=False)
    kap = _kappa_literal(d["F"], d["avg_inv_R"], d["avg_B2"])
    np.testing.assert_array_equal(d["j_dot_B"], fake_redl["jB"])
    np.testing.assert_array_equal(d["j_tor_full_raw"], d["j_dot_B"] * kap)
    np.testing.assert_array_equal(_j, d["j_dot_B"] * kap)
    assert "kappa" in d["version"]


def test_swb_jbs_to_toroidal_is_kappa(fsa):
    """SWB's ``<j.B> R_avg/F`` output, converted, IS ``kappa <j.B>`` with
    kappa the physics module's one factor (restored 2026-10-09: PR #64
    deleted the conversion and this pin)."""
    eq = _ExampleEq(fsa)
    p = fsa["psi_N"]
    jB = _jdotB(p)
    R_avg = eq._R(p)
    swb = jB * R_avg / fsa["F"]            # SWB's own R_avg/F projection
    out = physics._swb_jbs_to_toroidal(eq, swb, 1e-3,
                                       convention=physics.SWB_JBS_RAVG_OVER_F)
    kap = physics.field_aligned_conversion(fsa["F"], fsa["avg_inv_R"],
                                           fsa["avg_B2"])
    np.testing.assert_allclose(out, jB * kap, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(
        out, jB * _kappa_literal(fsa["F"], fsa["avg_inv_R"], fsa["avg_B2"]),
        rtol=1e-14, atol=0.0)
    # SWB's raw output is NOT that: R_avg/F over kappa = <R><B^2>/(F^2<1/R>)
    assert np.max(np.abs(swb / out - 1.0)) > 1e-2


def test_swb_jbs_to_toroidal_on_the_grid_swb_was_given(fsa):
    """With the grid SWB was called on (``x`` / ``psi_N``), the geometry is
    read on SWB's own clipped surfaces; on a grid equal to OFT's uniform one
    the result is the uniform call's."""
    eq = _ExampleEq(fsa)
    p = fsa["psi_N"]
    swb = _jdotB(p) * eq._R(p) / fsa["F"]
    a = physics._swb_jbs_to_toroidal(eq, swb, 1e-3,
                                     convention=physics.SWB_JBS_RAVG_OVER_F)
    b = physics._swb_jbs_to_toroidal(eq, swb, 1e-3, psi=p,
                                     convention=physics.SWB_JBS_RAVG_OVER_F)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="shape"):
        physics._swb_jbs_to_toroidal(eq, swb, 1e-3, psi=p[:-1],
                                     convention=physics.SWB_JBS_RAVG_OVER_F)


class _PressureEq(_ExampleEq):
    """The example's averages with a non-zero p' and FF' (positive jphi)."""

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        p = self._grid(psi, npsi, psi_pad)
        F = self._at("F", p)
        pp = -4.0e5 * (1.0 - p) - 2.0e6 * np.exp(-0.5 * ((p - 0.95) / 0.02) ** 2)
        Fp = -0.02 * (1.0 - p) / F
        return p, F, Fp, np.zeros_like(p), pp


def test_a_toroidal_swb_output_has_the_pressure_term_taken_off(fsa):
    """A toolkit whose SWB returns TokaMaker jphi (``kappa <j.B> + p'G``,
    A7): the conversion takes ``p'G`` off -- the field-aligned bootstrap is
    left, and ``p'G`` is the third bucket (D2) -- with p' signed so the
    equilibrium's own jphi is positive."""
    eq = _PressureEq(fsa)
    p = fsa["psi_N"]
    F, iR, B2 = fsa["F"], fsa["avg_inv_R"], fsa["avg_B2"]
    R = eq._R(p)
    _, _, Fp, _, pp = eq.get_profiles(psi=p)
    sign = 1.0 if np.sum(R * pp + iR * F * Fp / (4e-7 * np.pi)) >= 0 else -1.0
    assert sign == -1.0                    # the mock is in the reversed frame
    P = sign * pp * (R - F ** 2 * iR / B2)
    assert np.max(np.abs(P)) > 1e3
    kjB = _jdotB(p) * _kappa_literal(F, iR, B2)
    out = physics._swb_jbs_to_toroidal(eq, kjB + P, 1e-3, psi=p,
                                       convention=physics.SWB_JBS_TOROIDAL)
    np.testing.assert_allclose(out, kjB, rtol=1e-12, atol=1e-9 * kjB.max())
    np.testing.assert_allclose(physics.swb_pressure_term(eq, p.size, 1e-3,
                                                         psi=p), P,
                               rtol=1e-14, atol=0.0)


def test_the_swb_output_convention_is_identified_not_guessed():
    """The capability check: the upstream projection line -> R_avg/F; a
    toolkit with ``solve_with_bootstrap(x, coord)`` -> toroidal; anything
    else (a Fortran-routed call, another projection) is refused."""
    def upstream(mygs, ne, Te, ni, Ti, Zeff, Ip_target, psi_N=None):
        j_BS_neo = R_avg = f = 1.0
        j_BS_final = j_BS_neo * (R_avg / f)
        return j_BS_final

    def fork(mygs, ne, Te, ni, Ti, Zeff, Ip_target, x=None, coord="psi_n",
             jphi_fixed=None):
        return None

    def routed(mygs, ne, Te, ni, Ti, Zeff, Ip_target, use_python_solve=False):
        j_BS_neo = R_avg = f = 1.0
        return j_BS_neo * (R_avg / f)   # the python path is not the default

    def other(mygs, ne, Te, ni, Ti, Zeff, Ip_target):
        j_BS_neo = b = 1.0
        return j_BS_neo / b

    assert physics.swb_jbs_convention(upstream) == physics.SWB_JBS_RAVG_OVER_F
    assert physics.swb_jbs_convention(fork) == physics.SWB_JBS_TOROIDAL
    for fn in (routed, other):
        with pytest.raises(physics.SwbConventionUnknown):
            physics.swb_jbs_convention(fn)
    rec = physics.swb_conversion_record(physics.SWB_JBS_RAVG_OVER_F)
    assert rec["swb_jbs_convention"] == physics.SWB_JBS_RAVG_OVER_F
    assert "kappa" in rec["swb_jbs_conversion"]
    with pytest.raises(ValueError, match="convention"):
        physics._swb_jbs_to_toroidal(None, np.ones(3), 1e-3,
                                     convention="R/F?")


def test_the_installed_toolkit_has_a_known_swb_convention():
    """On the toolkit this suite runs against, the convention is one of the
    two known ones (upstream OFT: R_avg/F)."""
    bs = pytest.importorskip("OpenFUSIONToolkit.TokaMaker.bootstrap")
    conv = physics.swb_jbs_convention(bs.solve_with_bootstrap)
    assert conv in physics.SWB_JBS_CONVENTIONS


# ---------------------------------------------------------------------------
#  the engine path is unchanged by construction
# ---------------------------------------------------------------------------
def test_engine_request_is_rebuilt_bit_for_bit_from_the_literal_kappa():
    """The engine's delivered request, recomposed with the LITERAL
    ``F<1/R>/<B^2>`` on the stored geometry, bit for bit: this fails if the
    engine's conversion moved (it did not: the engine already used kappa,
    and the one-conversion change only routed the legacy sites to it)."""
    import contextlib
    import io
    import _engine_toy as T
    from bouquet.engine import reconstruct
    from bouquet.utils import structured_basis_eval
    b0 = T.ToyGS()
    b0.solve(T.ToyAdapter().read().anchor_request)
    b = T.ToyGS()
    ad = T.ToyAdapter(li_target=b0.state["li"] * 1.01)
    ad.read()
    with contextlib.redirect_stdout(io.StringIO()):
        eng, res, _rec = reconstruct(ad, b, T.settings(), label="toy")
    st = res["state"]
    g = st.geom
    kap = _kappa_literal(np.asarray(g["F"], float),
                         np.asarray(g["inv_R"], float),
                         np.asarray(g["B2"], float))
    _, parts = compose(g, eng.c.jB_ind, st.lambda_bs, eng.c.jB_fix)
    np.testing.assert_array_equal(parts["kappa"], kap)
    Phi = structured_basis_eval(eng.basis, eng.psi)
    K = Phi.shape[0]
    s_ind = 1.0 + st.x[:K] @ Phi
    s_bs = 1.0 + st.x[K:] @ Phi
    ind = 1.0 * kap * np.asarray(eng.c.jB_ind, float)
    bs = 1.0 * kap * np.asarray(st.lambda_bs, float)
    fix = kap * np.asarray(eng.c.jB_fix, float)
    R = s_ind * ind + s_bs * bs + (fix + parts["pressure"])
    np.testing.assert_array_equal(R, st.request)


def test_engine_backend_consumes_the_parallel_redl_output(monkeypatch):
    """The TokaMaker backend's bootstrap is ``evaluate_jBS``'s ``j_dot_B``
    (with the shared axis repair); the toroidal output never reaches it."""
    import _engine_toy as T
    from bouquet.engine import TokaMakerBackend
    from bouquet.TokaMaker_interface import smooth_jbs_transition
    c = T.ToyAdapter().read()
    jB = _jdotB(np.asarray(c.psi_N, float))

    def fake_eval(*a, **k):
        return np.full_like(jB, np.nan), dict(j_dot_B=jB.copy())

    monkeypatch.setattr(physics, "evaluate_jBS", fake_eval)

    class _GS:
        def copy_eq(self):
            return object()

    be = TokaMakerBackend(_GS(), c)
    np.testing.assert_array_equal(be.redl(), smooth_jbs_transition(jB))
