"""The inverse aspect ratio of the Redl collisionalities in ``evaluate_jBS``,
and the major radius ``R`` that goes with it in ``nu*``.

Owner decision E7 (2026-10-09): the default is ``eps_definition =
"r_over_R_geo"``, ``eps = (R_max - R_min)/(R_max + R_min)`` -- the surface's
half-width over its geometric major radius ``R_geo = (R_max + R_min)/2``, as
in Sauter (1999) / Redl (2021) (``eps = r/R0``), OMFIT's ``sauter_bootstrap``
and FUSE -- and the ``R`` in ``nu_e*`` / ``nu_i*`` (Sauter Eqs. 18b/18c) is
the SAME ``R_geo``.  The two older forms stay as named opt-ins, each with the
``R`` its own convention used (``<R>`` from ``get_q``):
``"half_width_over_fsa_R"`` (``(R_max - R_min)/(2<R>)``) and ``"a_over_R"``
(``<a>/<R>``, bit for bit the pre-change evaluator).

Decision 2026-10-09 (build independence): both half-width forms take
``R_min``, ``R_max`` and ``<R>`` from ``get_fsa`` on EVERY OpenFUSIONToolkit
build.  The internal-solve toolkit's ``sauter_fc(return_eps=True)`` agrees
with that only to ~1.4e-4 (its ``<R>`` is a cut-cell quadrature), so it is
only RECORDED beside ours (``diag["eps_fork_diagnostic"]``) with a
cross-build sanity bar of 1e-3 (:data:`bouquet.physics.EPS_ROUTE_SANITY_RTOL`)
-- a diagnostic, not a physics acceptance criterion.

Two geometries:

* the mock ``test_jbs_loop._MockEq``: shaped surfaces whose half-width is
  ``0.85 <a>``, whose geometric centre lies outside ``<R>`` (``R_geo =
  <R>(1 + 0.05 psi)``) and whose fork epsilon is 1.4e-4 off, so all three
  epsilons, both ``nu*`` radii and the fork value differ and a test sees
  which one was used;
* the synthetic D3D-like g-file ``tests/data/d3dlike.geqdsk``, its flux
  surfaces traced here (contours of a quintic spline of ``psirz``, the
  dl/B_p-weighted averages ``<R>`` and ``<a>`` on them) -- real diverted
  geometry, no solve, no OpenFUSIONToolkit needed.

Synthetic inputs only.
"""
import os

import numpy as np
import pytest

from bouquet.physics import (EPS_DEFINITION_DEFAULT, EPS_DEFINITIONS,
                             EPS_ROUTE_SANITY_RTOL, EVALUATE_JBS_VERSION,
                             NU_STAR_R, evaluate_jBS, evaluate_jbs_version,
                             fork_eps_diagnostic, geometric_eps, redl_eps)
from test_jbs_loop import _MockEq, _kin

X = np.linspace(0.0, 1.0, 151)
_HW = "half_width_over_fsa_R"
_GFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                      "d3dlike.geqdsk")


@pytest.fixture
def oft():
    """OFT's pure-Python Redl (``evaluate_jBS`` past its input checks)."""
    return pytest.importorskip(
        "OpenFUSIONToolkit.TokaMaker.bootstrap",
        reason="evaluate_jBS wraps OFT's pure-Python Redl implementation; "
               "OpenFUSIONToolkit is not importable here")


def _mock_geometry(eq, x):
    """``(R_min, R_max, <R>, <a>)`` of the mock on the evaluator's surfaces
    (``psi_N`` clipped at the default ``psi_pad``)."""
    psi_eval = np.clip(x, 1e-3, 1.0 - 1e-3)
    f = eq.get_fsa(psi=psi_eval)
    _p, r, _e, R = eq._geo(psi_eval)
    return f["R_min"], f["R_max"], R, r


def _eps_expected(eq, x, definition):
    r_min, r_max, R, a = _mock_geometry(eq, x)
    return {"r_over_R_geo": (r_max - r_min) / (r_max + r_min),
            _HW: (r_max - r_min) / (2.0 * R),
            "a_over_R": a / R}[definition]


def _nu_R_expected(eq, x, definition):
    r_min, r_max, R, _a = _mock_geometry(eq, x)
    return 0.5 * (r_min + r_max) if definition == "r_over_R_geo" else R


# ---------------------------------------------------------------------------
#  the default, its stamps and its nu* radius (mock)
# ---------------------------------------------------------------------------
def test_the_default_is_r_over_R_geo_and_says_so(oft):
    assert EPS_DEFINITION_DEFAULT == "r_over_R_geo"
    assert EPS_DEFINITIONS["r_over_R_geo"] == "(R_max - R_min)/(R_max + R_min)"
    assert EVALUATE_JBS_VERSION.startswith("evaluate_jBS/4")
    assert "eps = (R_max-R_min)/(R_max+R_min)" in EVALUATE_JBS_VERSION
    assert "nu* R = R_geo = (R_max + R_min)/2" in EVALUATE_JBS_VERSION
    assert "OPT-IN" not in EVALUATE_JBS_VERSION
    eq = _MockEq()
    _j, d = evaluate_jBS(eq, X, *_kin(X))
    np.testing.assert_allclose(d["eps"], _eps_expected(eq, X, "r_over_R_geo"),
                               rtol=1e-13)
    assert d["eps_definition"] == "r_over_R_geo"
    assert d["eps_formula"] == EPS_DEFINITIONS["r_over_R_geo"]
    assert d["eps_route"] == "get_fsa"
    assert d["nu_star_R"] == NU_STAR_R["r_over_R_geo"]
    assert d["version"] == EVALUATE_JBS_VERSION
    assert d["eps_fork_diagnostic"] is None          # not a fork build
    assert any(name == "get_fsa" for name, _a in eq.calls)


@pytest.mark.parametrize("definition", sorted(EPS_DEFINITIONS))
def test_every_definition_is_build_independent(oft, definition):
    """The same surfaces on an OFT-main-like build and on a fork-like build
    (whose ``sauter_fc(return_eps=True)`` is 1.4e-4 off, as the real fork's
    is) give the SAME bootstrap to the last bit: no definition reads the
    fork's value."""
    j_main, d_main = evaluate_jBS(_MockEq(fork=False), X, *_kin(X),
                                  eps_definition=definition)
    j_fork, d_fork = evaluate_jBS(_MockEq(fork=True), X, *_kin(X),
                                  eps_definition=definition)
    np.testing.assert_array_equal(j_fork, j_main)
    np.testing.assert_array_equal(d_fork["eps"], d_main["eps"])
    assert d_fork["eps_route"] == d_main["eps_route"]
    np.testing.assert_allclose(d_main["eps"],
                               _eps_expected(_MockEq(), X, definition),
                               rtol=1e-13)


@pytest.mark.parametrize("definition", sorted(EPS_DEFINITIONS))
def test_the_nu_star_R_is_R_geo_for_the_default_and_fsa_R_otherwise(
        oft, definition):
    """(d) On the mock ``<R>`` and ``R_geo`` differ (by up to 5 %), so the R
    in ``nu*`` is visible: ``diag["R_nu_star"]``, and the R the Redl call
    receives."""
    eq = _MockEq()
    seen = {}
    real = oft.redl_bootstrap

    def _spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    oft.redl_bootstrap = _spy
    try:
        _j, d = evaluate_jBS(eq, X, *_kin(X), eps_definition=definition)
    finally:
        oft.redl_bootstrap = real
    np.testing.assert_allclose(d["R_nu_star"], _nu_R_expected(eq, X,
                                                               definition),
                               rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(seen["R"]), d["R_nu_star"])
    assert d["nu_star_R"] == NU_STAR_R[definition]
    # R_avg (get_q's <R>) is untouched: it still feeds the p'G term and the
    # SWB projection under every definition
    _p, _r, _e, R_fsa = eq._geo(np.clip(X, 1e-3, 1 - 1e-3))
    np.testing.assert_allclose(d["R_avg"], R_fsa, rtol=1e-14)
    inner = (X > 0.3) & (X < 0.99)
    if definition == "r_over_R_geo":
        assert np.all(d["R_nu_star"][inner] > 1.01 * d["R_avg"][inner])
    else:
        assert d["R_nu_star"] is d["R_avg"]


def test_the_nu_star_ratio_between_definitions_is_R_times_eps_minus_3_2(oft):
    """``nu* ~ R eps^-3/2``: between any two definitions on the same
    surfaces and kinetics the collisionalities differ by exactly
    ``(R_1/R_2) (eps_1/eps_2)^-3/2`` -- both factors, not only the eps one."""
    ds = {k: evaluate_jBS(_MockEq(), X, *_kin(X), smooth_axis=False,
                          eps_definition=k)[1] for k in EPS_DEFINITIONS}
    inner = (X > 0.01) & (X < 0.99)
    ref = ds["a_over_R"]
    for k, d in ds.items():
        want = (d["R_nu_star"] / ref["R_nu_star"]) * (
            d["eps"] / ref["eps"]) ** -1.5
        for nu in ("nu_e_star", "nu_i_star"):
            np.testing.assert_allclose(d[nu][inner] / ref[nu][inner],
                                       want[inner], rtol=1e-12)
    # the half-width opt-in on the mock: 0.85 x <a>/<R>, same R
    np.testing.assert_allclose(
        ds[_HW]["nu_e_star"][inner] / ref["nu_e_star"][inner],
        np.full(inner.sum(), 0.85 ** -1.5), rtol=1e-10)


def test_the_version_string_distinguishes_the_three_definitions():
    """(e) Each definition's tag names its epsilon and its ``nu*`` R; every
    one stays in the ``evaluate_jBS/4`` family (p'G separate), and the opt-ins
    are marked."""
    tags = {k: evaluate_jbs_version(k) for k in EPS_DEFINITIONS}
    assert len(set(tags.values())) == 3
    for k, v in tags.items():
        assert v.startswith("evaluate_jBS/4 (")
        assert "p'G separate as j_pressure" in v
        assert EPS_DEFINITIONS[k].replace(" ", "") in v
        assert f"nu* R = {NU_STAR_R[k]}" in v
        assert ("OPT-IN" in v) == (k != EPS_DEFINITION_DEFAULT)
    assert tags["r_over_R_geo"] == EVALUATE_JBS_VERSION
    assert "(R_max-R_min)/(2<R>)" in tags[_HW]
    assert "<a>/<R> (the /3 definition)" in tags["a_over_R"]


# ---------------------------------------------------------------------------
#  the fork's own epsilon: recorded, never used (mock)
# ---------------------------------------------------------------------------
class _MainSauter(_MockEq):
    """``sauter_fc`` with OpenFUSIONToolkit main's exact signature (no
    ``return_eps``, no ``**kwargs``): passing the keyword would raise."""

    def sauter_fc(self, psi=None, psi_pad=0.02, npsi=50):
        return super().sauter_fc(psi=psi)


@pytest.mark.parametrize("definition", sorted(EPS_DEFINITIONS))
def test_a_signature_without_return_eps_is_never_called_with_it(
        oft, definition):
    j, d = evaluate_jBS(_MainSauter(), X, *_kin(X),
                        eps_definition=definition)
    assert np.all(np.isfinite(j)) and d["eps_fork_diagnostic"] is None


def test_the_fork_value_is_recorded_beside_ours_as_a_cross_build_check(oft):
    """CROSS-BUILD SANITY CHECK (a diagnostic, not a physics bar): where the
    build has ``sauter_fc(return_eps=True)``, its ``(R_max - R_min)/(2<R>)``
    is stamped beside the ``get_fsa`` value of the same formula, with their
    largest relative difference and the documented 1e-3 bar.  The real fork
    and ``get_fsa`` agree to ~1.4e-4 (cut-cell ``<R>``), never to rounding,
    which is why no evaluation uses the fork's value."""
    assert EPS_ROUTE_SANITY_RTOL == 1.0e-3
    eq = _MockEq(fork=True)
    _j, d = evaluate_jBS(eq, X, *_kin(X))
    fd = d["eps_fork_diagnostic"]
    assert fd is not None and fd["sanity_rtol"] == EPS_ROUTE_SANITY_RTOL
    np.testing.assert_allclose(fd["eps_get_fsa"], _eps_expected(eq, X, _HW),
                               rtol=1e-13)
    np.testing.assert_allclose(fd["eps_sauter_fc_return_eps"],
                               fd["eps_get_fsa"], rtol=EPS_ROUTE_SANITY_RTOL)
    np.testing.assert_allclose(fd["max_rel_diff"], eq.FORK_EPS_OFFSET,
                               rtol=1e-6)
    assert fd["sanity_ok"]
    # ... and the evaluation itself used get_fsa
    assert d["eps_route"] == "get_fsa"
    np.testing.assert_allclose(d["eps"], _eps_expected(eq, X, "r_over_R_geo"),
                               rtol=1e-13)


def test_a_gross_fork_disagreement_is_flagged_not_hidden():
    class _BadFork(_MockEq):
        FORK_EPS_OFFSET = 1.0e-2

    psi_u = np.unique(np.clip(X, 1e-3, 1.0 - 1e-3))
    fd = fork_eps_diagnostic(_BadFork(fork=True), psi_u)
    assert not fd["sanity_ok"] and fd["max_rel_diff"] > 5e-3
    assert fork_eps_diagnostic(_MockEq(fork=False), psi_u) is None


def test_the_half_width_opt_in_is_get_fsa_on_every_build():
    psi_u = np.unique(np.clip(X, 1e-3, 1.0 - 1e-3))
    for fork in (False, True):
        eq = _MockEq(fork=fork)
        e, route = geometric_eps(eq, psi_u)
        assert route == "get_fsa"
        f = eq.get_fsa(psi=psi_u)
        np.testing.assert_allclose(
            e, (f["R_max"] - f["R_min"]) / (2.0 * f["<R>"]), rtol=1e-15)


# ---------------------------------------------------------------------------
#  the a_over_R opt-in (bit identity with the pre-change evaluator lives in
#  test_evaluate_jbs_refusals / test_one_conversion)
# ---------------------------------------------------------------------------
def test_the_a_over_R_opt_in_is_named_and_never_traces_get_fsa(oft):
    eq = _MockEq()
    j_def, _ = evaluate_jBS(_MockEq(), X, *_kin(X), smooth_axis=False)
    j_aR, d_aR = evaluate_jBS(eq, X, *_kin(X), smooth_axis=False,
                              eps_definition="a_over_R")
    assert not any(name == "get_fsa" for name, _a in eq.calls)
    np.testing.assert_allclose(d_aR["eps"], _eps_expected(eq, X, "a_over_R"),
                               rtol=1e-13)
    assert d_aR["eps_definition"] == "a_over_R"
    assert d_aR["eps_route"] == "sauter_fc <a>/<R>"
    assert d_aR["version"] == evaluate_jbs_version("a_over_R")
    assert d_aR["version"] != EVALUATE_JBS_VERSION
    # the default physics change is visible in the bootstrap
    assert np.max(np.abs(j_def - j_aR)) > 1e-3 * np.max(np.abs(j_aR))


def test_an_unknown_definition_is_refused_and_the_draft_name_is_explained():
    with pytest.raises(ValueError, match="eps_definition"):
        evaluate_jBS(_MockEq(), X, *_kin(X), eps_definition="inverse_aspect")
    with pytest.raises(ValueError, match="half_width_over_fsa_R"):
        evaluate_jbs_version("geometric")


class _NoFsaNoFork(_MainSauter):
    get_fsa = None


def test_a_build_without_get_fsa_is_refused_by_name_not_silently(oft):
    """Pre-v26.6 OFT (no ``get_fsa``): neither half-width form can be
    computed -- named refusals that point at the opt-in which still works
    there."""
    for definition in ("r_over_R_geo", _HW):
        with pytest.raises(RuntimeError, match="a_over_R"):
            evaluate_jBS(_NoFsaNoFork(), X, *_kin(X),
                         eps_definition=definition)
    j, d = evaluate_jBS(_NoFsaNoFork(), X, *_kin(X),
                        eps_definition="a_over_R")
    assert np.all(np.isfinite(j)) and d["eps_definition"] == "a_over_R"


# ---------------------------------------------------------------------------
#  real diverted geometry: the synthetic D3D-like g-file, surfaces traced
# ---------------------------------------------------------------------------
class _GfileSurfaces:
    """``get_fsa`` / ``sauter_fc`` of the flux surfaces of a g-file, traced
    here: contours of a quintic spline of ``psirz`` (on a 4x refined grid)
    enclosing the magnetic axis; ``R_min``/``R_max`` their extremes,
    ``<R>`` and ``<a>`` (distance from the axis) averaged with the
    flux-surface weight ``dl/B_p = R dl/|grad psi|``.  Only what
    :func:`bouquet.physics.redl_eps` reads."""

    def __init__(self, path):
        import contourpy
        from scipy.interpolate import RectBivariateSpline
        from bouquet.io.geqdsk import _read_geqdsk
        g = _read_geqdsk(path)
        R = np.linspace(g["RLEFT"], g["RLEFT"] + g["RDIM"], g["NW"])
        Z = np.linspace(g["ZMID"] - g["ZDIM"] / 2, g["ZMID"] + g["ZDIM"] / 2,
                        g["NH"])
        psi_n = (g["PSIRZ"] - g["SIMAG"]) / (g["SIBRY"] - g["SIMAG"])
        self.sp = RectBivariateSpline(Z, R, psi_n, kx=5, ky=5)
        Rf = np.linspace(R[0], R[-1], 4 * R.size)
        Zf = np.linspace(Z[0], Z[-1], 4 * Z.size)
        self._cg = contourpy.contour_generator(
            Rf, Zf, self.sp(Zf, Rf), name="serial", line_type="Separate")
        self.R0, self.Z0 = float(g["RMAXIS"]), float(g["ZMAXIS"])
        rb = np.asarray(g["RBBBS"], dtype=float)
        #: the boundary's a / R_geo
        self.boundary_eps = (rb.max() - rb.min()) / (rb.max() + rb.min())

    def _surface(self, x):
        for seg in self._cg.lines(float(x)):
            if np.hypot(*(seg[0] - seg[-1])) > 1e-9:
                continue                               # open (a leg)
            th = np.unwrap(np.arctan2(seg[:, 1] - self.Z0,
                                      seg[:, 0] - self.R0))
            if abs(th[-1] - th[0]) > 6.0:              # encloses the axis
                return seg
        raise AssertionError(f"no closed surface around the axis at {x}")

    def _row(self, x):
        s = self._surface(x)
        r, z = s[:, 0], s[:, 1]
        rm, zm = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1])
        w = np.hypot(np.diff(r), np.diff(z)) * rm / np.hypot(
            self.sp(zm, rm, dx=1, grid=False),
            self.sp(zm, rm, dy=1, grid=False))
        return (r.min(), r.max(), np.sum(w * rm) / np.sum(w),
                np.sum(w * np.hypot(rm - self.R0, zm - self.Z0)) / np.sum(w))

    def _rows(self, psi):
        return np.array([self._row(x) for x in np.asarray(psi, dtype=float)])

    def get_fsa(self, psi=None, **kw):
        t = self._rows(psi)
        return {"psi_norm": np.asarray(psi), "R_min": t[:, 0],
                "R_max": t[:, 1], "<R>": t[:, 2]}

    def sauter_fc(self, psi=None, **kw):
        t = self._rows(psi)
        rav = {"<R>": t[:, 2], "<1/R>": 1.0 / t[:, 2], "<a>": t[:, 3]}
        return (np.asarray(psi), np.zeros(len(t)), rav, None)


@pytest.fixture(scope="module")
def gfile():
    pytest.importorskip("contourpy")
    return _GfileSurfaces(_GFILE)


def _eps_all(eq, psi):
    psi = np.asarray(psi, dtype=float)
    r_sau = eq.sauter_fc(psi=psi)[2]
    return {k: redl_eps(eq, psi, k, r_sau) for k in EPS_DEFINITIONS}


def test_the_default_eps_at_the_lcfs_is_the_boundary_a_over_R_geo(gfile):
    """(a) On the outermost surface the evaluator samples (``1 - psi_pad`` =
    0.999) the default epsilon is the boundary's ``a/R_geo`` to 1e-3
    (measured 7.9e-4: the 0.999 surface sits just inside the boundary);
    ``(R_max - R_min)/(2<R>)`` is ~7 % and ``<a>/<R>`` ~55 % above it there,
    because the dl/B_p weight pulls ``<R>`` toward the X-point."""
    e = _eps_all(gfile, [0.999])
    eps, R_geo, route = e["r_over_R_geo"]
    assert route == "get_fsa"
    np.testing.assert_allclose(eps[0], gfile.boundary_eps, rtol=1e-3)
    assert e[_HW][0][0] > 1.05 * gfile.boundary_eps
    assert e["a_over_R"][0][0] > 1.4 * gfile.boundary_eps
    # the default's nu* R is the surface's geometric centre; the others'
    # None (= get_q's <R>, the caller's)
    f = gfile.get_fsa(psi=[0.999])
    np.testing.assert_allclose(R_geo, 0.5 * (f["R_min"] + f["R_max"]),
                               rtol=1e-15)
    assert e[_HW][1] is None and e["a_over_R"][1] is None


def test_the_three_definitions_are_ordered_at_the_edge(gfile):
    """(b) Near the edge ``<a>/<R> > (R_max - R_min)/(2<R>) > (R_max -
    R_min)/(R_max + R_min)``; toward the axis ``<R> -> R_geo`` so the two
    half-width forms converge, while ``<a>/<R>`` stays above both by the
    elongation (``<a>`` includes the vertical extent) -- it does NOT
    converge to them."""
    edge = [0.9, 0.95, 0.98, 0.99, 0.999]
    e = _eps_all(gfile, edge)
    old, hw, new = e["a_over_R"][0], e[_HW][0], e["r_over_R_geo"][0]
    assert np.all(old > hw) and np.all(hw > new)
    # the <R> upturn: the gap grows toward the separatrix
    assert np.all(np.diff(hw / new) > 0.0)
    assert hw[-1] / new[-1] > 1.05
    core = [0.01, 0.02, 0.05]
    c = _eps_all(gfile, core)
    np.testing.assert_allclose(c[_HW][0], c["r_over_R_geo"][0], rtol=1e-3)
    assert np.all(c["a_over_R"][0] > 1.1 * c["r_over_R_geo"][0])


# ---------------------------------------------------------------------------
#  GenerationConfig.eps_definition reaches every evaluation, and is stamped
# ---------------------------------------------------------------------------
def test_the_config_field_defaults_to_r_over_R_geo_and_is_validated():
    from bouquet.config import GenerationConfig
    assert GenerationConfig().eps_definition == "r_over_R_geo"
    for name in EPS_DEFINITIONS:
        assert GenerationConfig(eps_definition=name).eps_definition == name
    with pytest.raises(ValueError, match="eps_definition"):
        GenerationConfig(eps_definition="inverse_aspect")
    with pytest.raises(ValueError, match="half_width_over_fsa_R"):
        GenerationConfig(eps_definition="geometric")


def test_it_is_engine_independent_not_an_engine_unread_field():
    """It configures physics.evaluate_jBS on every path: not an engine-only
    field (refused under legacy) nor an engine-unread legacy field (refused
    under unified)."""
    from bouquet.config import GenerationConfig
    from bouquet.engine import (ENGINE_FIELD_DEFAULTS,
                                ENGINE_UNREAD_LEGACY_FIELDS,
                                validate_engine_settings)
    assert "eps_definition" not in ENGINE_FIELD_DEFAULTS
    assert "eps_definition" not in ENGINE_UNREAD_LEGACY_FIELDS
    for eng in ("legacy", "unified"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine=eng, eps_definition="a_over_R"))


def test_the_engine_settings_and_record_carry_it():
    from bouquet.config import GenerationConfig
    from bouquet.engine import engine_settings, eps_record
    for name in EPS_DEFINITIONS:
        s = engine_settings(GenerationConfig(reconstruction_engine="unified",
                                             eps_definition=name))
        assert s["eps_definition"] == name
        rec = eps_record(name)
        assert rec == dict(eps_definition=name,
                           eps_formula=EPS_DEFINITIONS[name],
                           nu_star_R=NU_STAR_R[name],
                           evaluate_jBS_version=evaluate_jbs_version(name))
    assert eps_record(None)["eps_definition"] == "r_over_R_geo"


def _spy_evaluate(monkeypatch, j):
    import bouquet.physics as physics
    seen = []

    def _spy(*a, **k):
        seen.append(k.get("eps_definition"))
        return j.copy(), dict(j_dot_B=j.copy(), j_tor_full_raw=j.copy(),
                              I_BS=1.0)

    monkeypatch.setattr(physics, "evaluate_jBS", _spy)
    return seen


@pytest.mark.parametrize("name", sorted(EPS_DEFINITIONS))
def test_the_engine_backend_and_its_draws_evaluate_with_it(monkeypatch, name):
    """The reconstruction backend, and the draws' backend built from the
    reconstruction's settings (``GenerateEngineDraws.backend``), pass the
    definition to every Redl evaluation."""
    from types import SimpleNamespace
    import _engine_toy as T
    from bouquet.engine import TokaMakerBackend
    from bouquet.engine_draws import GenerateEngineDraws
    c = T.ToyAdapter().read()
    seen = _spy_evaluate(monkeypatch, np.ones(len(c.psi_N)))

    class _GS:
        def copy_eq(self):
            return object()

    TokaMakerBackend(_GS(), c, eps_definition=name).redl()
    assert seen == [name]
    gd = object.__new__(GenerateEngineDraws)
    gd.ctx = SimpleNamespace(c=c, edge=None, coord="psi_n",
                             eng=SimpleNamespace(s={"eps_definition": name,
                                                    "edge_taper": None}))
    gd.psi_pad, gd.q_psi, gd.maxits = 1e-3, None, None
    be = gd.backend(_GS())
    assert be.eps_definition == name
    be.redl()
    assert seen == [name, name]


@pytest.mark.parametrize("name", sorted(EPS_DEFINITIONS))
def test_the_legacy_draw_composer_evaluates_with_it(monkeypatch, name):
    from bouquet.TokaMaker_interface import _draw_jbs_composer
    x = np.linspace(0.0, 1.0, 21)
    seen = _spy_evaluate(monkeypatch, np.ones_like(x))
    k = _kin(x)
    _draw_jbs_composer(x, *k, 1e-3, False, 1.0, False, None, None, None,
                       eps_definition=name)(object())
    assert seen == [name]


def test_the_loop_settings_carry_a_non_default_and_the_record_names_it():
    """``jbs_settings`` (the legacy loops' and the engine's loop settings)
    carries a non-default definition -- the default dict keeps its keys --
    and every loop record's ``evaluate_jBS_version`` is that definition's."""
    from bouquet.jbs_loop import jbs_settings, run_jbs_loop
    from test_jbs_loop import _GC, _IP, _affine_problem
    assert "eps_definition" not in jbs_settings(_GC())
    for name in EPS_DEFINITIONS:
        g = _GC()
        g.eps_definition = name
        s = jbs_settings(g)
        assert s.get("eps_definition", EPS_DEFINITION_DEFAULT) == name
        assert ("eps_definition" in s) == (name != EPS_DEFINITION_DEFAULT)
        Jstar, step, ev = _affine_problem(-0.1)
        rec = run_jbs_loop(0.5 * Jstar, step, ev, s, Ip=_IP,
                           meas0=dict(li=0.0))["record"]
        assert rec["evaluate_jBS_version"] == evaluate_jbs_version(name)


def test_the_archive_stamp_round_trips(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bouquet.engine import eps_record
    from bouquet.utils import (_baseline_group_path, load_bootstrap_eps,
                               stamp_bootstrap_eps)
    p = str(tmp_path / "a.h5")
    with h5py.File(p, "w") as hf:
        hf.create_group(_baseline_group_path(None))
    assert load_bootstrap_eps(p) is None             # predates the record
    rec = eps_record("half_width_over_fsa_R")
    stamp_bootstrap_eps(p, record=rec)
    assert load_bootstrap_eps(p) == rec
