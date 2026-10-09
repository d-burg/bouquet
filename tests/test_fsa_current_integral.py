"""The flux-surface-averaged plasma-current measure (``utils.Ip_fsa_integral``).

Two halves.

**Fast** (pure NumPy): the algebra of :func:`bouquet.utils.Ip_fsa_weights` --
that the measure is affine with the ``P'`` term in the constant, that the
``fsa`` reading has no constant, that a circular large-aspect-ratio geometry
integrates to the analytic answer, and that every unsupported combination
raises instead of returning a plausible number.

**Solver** (``pytest -m solver``, in a subprocess): the validation the measure
was accepted on -- integrate the SOLVED equilibrium's OWN current profile and
compare with ``compute_area_integral(calc_jtor_plasma)``.  Required: 0.1 %.
Measured on the synthetic D3D-like baseline: **+0.0071 %** in both conventions.
The same run pins the three properties the implementation depends on:

  * the CLIPPED sampling grid gives a non-degenerate geometry -- which is the
    property ``fsa_current_geometry``'s clipping exists to guarantee, the only
    one bouquet's answer depends on, and the one that must hold on EVERY build.
    On builds that reproduce the trap, ``get_q(psi=...)`` collapses SILENTLY
    onto the magnetic axis when the sample grid includes ``psi_N = 0`` --
    ``<R>`` constant to 2e-15 across all 257 surfaces, ``dV/dPsi`` constant, no
    exception.  That demonstration is BUILD-dependent (not OFT-version
    dependent -- the same OFT commit has been measured both ways on different
    machines' builds), so it is gated on a measurement and skips, loudly, where
    the build does not reproduce it.  The clipping stays either way;
  * ``dV/dPsi`` is per DIMENSIONAL psi (``int dV/dPsi dpsi`` recovers the
    volume to -0.25 %; the ``dpsi_N`` reading is out by +291 %);
  * ``compute_flux_integral`` is ``int_plasma f dA`` for a profile that
    vanishes at the LCFS, on every build.  ``FI(1)`` depends on the build:
    the plasma cross-section where ``gs_flux_int`` is plasma-masked, the
    whole region-1 (limiter) area where it covers every ``reg == 1`` cell with
    the profile pinned at its LCFS value outside the plasma (``FI(1) =
    2.8385 m^2`` against ``1.7901 m^2``, where 7dc254b's "+12.9 % convention
    bias" came from).  Each is pinned against an independent measurement of
    its domain at 1 %.

Runs on the synthetic D3D-like example (no proprietary data).
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

from bouquet.utils import Ip_fsa_integral, Ip_fsa_weights

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")
_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))

#: The step-1 acceptance the measure had to clear before being wired in.
#: Measured +0.0071 % -- 14x margin.  Not to be widened: a miss means the
#: V'/<1/R> plumbing or the psi_N -> psi Jacobian has moved.
_SELF_CONSISTENCY = 1.0e-3


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
#  fast: the algebra, on a synthetic geometry
# ---------------------------------------------------------------------------
def _circular_geom(npsi=401, R0=3.0, a=0.3):
    """Large-aspect-ratio circular flux surfaces with a known area element.

    ``r = a*sqrt(psi_N)`` (so ``dA/dpsi_N = pi a^2`` is constant) and the FSA
    of any function of R is taken at R0 -- exact in the ``a/R0 -> 0`` limit,
    which is enough to give the current integral a closed form.
    """
    psi_N = np.linspace(0.0, 1.0, npsi)
    ones = np.ones_like(psi_N)
    dpsi_dpsiN = 2.0
    inv_R = ones / R0
    # dA/dpsi_N = (V'/2pi) <1/R> |dpsi/dpsi_N| = pi a^2  =>  V' = 2 pi^2 a^2 R0
    dV_dpsi = (np.pi * a ** 2) * (2.0 * np.pi) * R0 / dpsi_dpsiN * ones
    return {
        "psi_N": psi_N, "psi_q": psi_N,
        "R_avg": R0 * ones, "inv_R": inv_R, "inv_R2": ones / R0 ** 2,
        "dV_dpsi": dV_dpsi, "dpsi_dpsiN": dpsi_dpsiN,
        "pprime": np.zeros_like(psi_N),
        "dA_dpsiN": dV_dpsi / (2.0 * np.pi) * inv_R * dpsi_dpsiN,
    }


def test_uniform_current_integrates_to_j_times_area():
    """A flat profile over a known cross-section: I_p = J * A, both readings."""
    geom = _circular_geom()
    area = float(np.pi * 0.3 ** 2)
    J = np.full_like(geom["psi_N"], 1.5e6)
    for convention in ("fsa", "jphi-linterp"):
        got = Ip_fsa_integral(None, geom["psi_N"], J, convention=convention,
                              geom=geom)
        assert got == pytest.approx(1.5e6 * area, rel=1e-12), convention


def test_fsa_reading_has_no_constant_term_and_jphi_reading_does():
    """The P' term is independent of J, so it must live in ``c``, not ``w``."""
    geom = _circular_geom()
    _w, c = Ip_fsa_weights(geom, convention="fsa")
    assert c == 0.0
    geom_p = dict(geom, pprime=np.full_like(geom["psi_N"], 2.0e5))
    # <R><1/R^2>/<1/R> == 1 exactly in this geometry, so even a finite P'
    # contributes nothing -- the correction is a shaping effect.
    _w, c = Ip_fsa_weights(geom_p, convention="jphi-linterp")
    assert c == pytest.approx(0.0, abs=1e-6)
    # ... but bend <1/R^2> away from <1/R>^2 and it must show up.
    geom_s = dict(geom_p, inv_R2=geom["inv_R2"] * 1.1)
    _w, c = Ip_fsa_weights(geom_s, convention="jphi-linterp")
    assert abs(c) > 1.0e3


def test_the_measure_is_affine_in_the_profile():
    """What ``_AnchorIpRenorm.solve_scale`` roots analytically instead of by
    bisection.  If this ever stops holding the analytic root is wrong."""
    geom = dict(_circular_geom(), pprime=np.full_like(_circular_geom()["psi_N"],
                                                      2.0e5))
    geom["inv_R2"] = geom["inv_R2"] * 1.07
    rng = np.random.default_rng(7)
    j1 = 1e6 * (1.0 - geom["psi_N"] ** 2) + 1e4 * rng.standard_normal(geom["psi_N"].size)
    j2 = 3e5 * np.exp(-((geom["psi_N"] - 0.95) / 0.03) ** 2)
    _w, c = Ip_fsa_weights(geom, convention="jphi-linterp")
    ip = lambda p: Ip_fsa_integral(None, geom["psi_N"], p,  # noqa: E731
                                   convention="jphi-linterp", geom=geom)
    for a in (0.5, 1.0, 2.75):
        assert ip(a * j1 + j2) == pytest.approx(a * (ip(j1) - c) + ip(j2),
                                                rel=1e-12)


def test_unsupported_combinations_raise_rather_than_guess():
    geom = _circular_geom()
    with pytest.raises(ValueError, match="unknown convention"):
        Ip_fsa_weights(geom, convention="area")
    with pytest.raises(ValueError, match="<1/R\\^2>"):
        Ip_fsa_weights(dict(geom, inv_R2=None), convention="jphi-linterp")
    with pytest.raises(ValueError, match="P'"):
        Ip_fsa_weights(dict(geom, pprime=None), convention="jphi-linterp")


def test_r2_mode_resolution():
    """The A/B switch.  'ratio' (and 7dc254b's spelling, 'anchor') was
    retired per issue #35: both must now raise with the retirement pointer,
    not silently resolve to a different measure."""
    from bouquet.TokaMaker_interface import _r2_ip_mode, _R2_IP_MODE_DEFAULT

    saved = os.environ.pop("BOUQUET_R2_IP_MODE", None)
    try:
        assert _r2_ip_mode() == _R2_IP_MODE_DEFAULT
        for given, want in (("exact", "exact"), ("fsa", "fsa"),
                            (" legacy ", "legacy")):
            os.environ["BOUQUET_R2_IP_MODE"] = given
            assert _r2_ip_mode() == want, given
        for retired in ("ratio", "anchor", "ANCHOR"):
            os.environ["BOUQUET_R2_IP_MODE"] = retired
            with pytest.raises(ValueError, match="retired"):
                _r2_ip_mode()
        os.environ["BOUQUET_R2_IP_MODE"] = "exakt"
        with pytest.raises(ValueError, match="BOUQUET_R2_IP_MODE"):
            _r2_ip_mode()
    finally:
        os.environ.pop("BOUQUET_R2_IP_MODE", None)
        if saved is not None:
            os.environ["BOUQUET_R2_IP_MODE"] = saved


# ---------------------------------------------------------------------------
#  solver: the validation the measure was accepted on
# ---------------------------------------------------------------------------
def _probe(outdir):
    """Everything the measure is validated by, on the solved D3D-like anchor.

    Subprocess entry point (``OFT_env`` is a per-process singleton, so this
    module must never build a solver in the pytest process).  Results land in
    ``<outdir>/fsa.json``.
    """
    import numpy as np
    import bouquet as bq
    from bouquet.utils import (fsa_current_geometry, Ip_fsa_integral,
                               eq_jphi_profile)

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "fsa"), n_draws=1,
                               reconstruction_engine="legacy")
    b.setup_solver()
    bl = b.prepare_baseline()
    mygs = b.mygs
    psi_N = np.asarray(bl.psi_N, dtype=float)
    J = np.asarray(bl.j_phi, dtype=float)
    Ip_true = float(mygs.compute_area_integral(mygs.calc_jtor_plasma()))
    vol_true = float(mygs.get_stats(
        lcfs_pad=float(getattr(b.config.source, "psi_pad", 1e-3)))["vol"])

    out = {"Ip_true": Ip_true, "vol_true": vol_true,
           "flux_integral_of_one": float(
               mygs.compute_flux_integral(psi_N, np.ones_like(psi_N))),
           # a profile that VANISHES at the LCFS: whatever a build does
           # off-plasma (nothing, or the LCFS value held), it adds zero there
           "flux_integral_of_1_minus_psiN": float(
               mygs.compute_flux_integral(psi_N, 1.0 - psi_N)),
           # the mesh area of the plasma region (reg 1: plasma + vacuum
           # inside the limiter), measured independently of gs_flux_int
           "region1_area": float(mygs.compute_area_integral(
               np.ones(int(mygs.np)), reg_mask=1))}

    snap = mygs.copy_eq()
    for tag, obj in (("live", mygs), ("snapshot", snap)):
        geom = fsa_current_geometry(obj, psi_N)
        sgn = 1.0 if float(np.dot(
            eq_jphi_profile(geom, "jphi-linterp", eq=obj), J)) > 0 else -1.0
        out[tag] = {"pprime_sign": sgn,
                    "plasma_area": float(np.trapezoid(geom["dA_dpsiN"], psi_N)),
                    "plasma_integral_of_1_minus_psiN": float(np.trapezoid(
                        (1.0 - psi_N) * geom["dA_dpsiN"], psi_N)),
                    "vol_dpsi": float(np.trapezoid(
                        geom["dV_dpsi"], psi_N) * geom["dpsi_dpsiN"]),
                    "vol_dpsiN": float(np.trapezoid(geom["dV_dpsi"], psi_N))}
        for conv in ("jphi-linterp", "fsa"):
            j_eq = eq_jphi_profile(geom, conv, eq=obj, pprime_sign=sgn)
            out[tag][conv] = {
                "self": Ip_fsa_integral(obj, psi_N, j_eq, convention=conv,
                                        pprime_sign=sgn, geom=geom),
                "archived": Ip_fsa_integral(obj, psi_N, J, convention=conv,
                                            pprime_sign=sgn, geom=geom),
            }

    # the silent get_q collapse the clipping exists to avoid, and -- the part
    # that holds on EVERY build -- the clipped geometry it produces instead.
    try:
        fsa_current_geometry(mygs, psi_N, psi_pad=0.0)
        out["collapse_guard"] = None
    except RuntimeError as exc:
        out["collapse_guard"] = str(exc)
    ravgs = mygs.get_q(psi=np.ascontiguousarray(psi_N))[2]
    R_raw = np.asarray(ravgs["<R>"] if isinstance(ravgs, dict) else ravgs[0],
                       dtype=float)
    out["unclipped_R_span"] = float(np.ptp(R_raw))
    # A surface the tracer fails on comes back as a zero row (get_q's output
    # arrays are zero-initialised and a failed trace CYCLEs), so the raw span
    # is not a collapse detector on its own -- see _unclipped_collapses.
    R_pos = R_raw[R_raw > 0.0]
    out["unclipped_R_span_traced"] = float(np.ptp(R_pos)) if R_pos.size else 0.0
    out["unclipped_R_traced_mean"] = float(np.mean(R_pos)) if R_pos.size else 0.0
    out["unclipped_n_untraced"] = int(R_raw.size - R_pos.size)
    out["unclipped_n_surfaces"] = int(R_raw.size)
    clipped = fsa_current_geometry(mygs, psi_N, want_pprime=False)
    out["clipped_R_span"] = float(np.ptp(clipped["R_avg"]))
    out["clipped_R_mean"] = float(np.mean(clipped["R_avg"]))
    out["clipped_psi_q_ends"] = [float(clipped["psi_q"][0]),
                                 float(clipped["psi_q"][-1])]

    out["Ip_after"] = float(mygs.compute_area_integral(mygs.calc_jtor_plasma()))
    with open(os.path.join(outdir, "fsa.json"), "w") as fh:
        json.dump(out, fh)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    work = tmp_path_factory.mktemp("fsa")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(work)],
        # See tests/_harness.py: a script's sys.path[0] is <repo>/tests, so
        # without this the probe can import an editable-installed bouquet from
        # a different checkout and validate the wrong revision silently.
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"FSA probe failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-4000:]}")
    with open(str(work / "fsa.json")) as fh:
        return json.load(fh)


solver_only = pytest.mark.skipif(
    not (_files_ok and _oft_importable()),
    reason="needs OFT + the D3D-like mesh/baseline; skipped when unavailable")


@pytest.mark.solver
@solver_only
@pytest.mark.parametrize("convention", ["jphi-linterp", "fsa"])
def test_measure_reproduces_the_equilibriums_own_current(measured, convention):
    """THE validation: integrate the solved equilibrium's own j_phi profile and
    demand its true I_p back, to 0.1 %.  Measured +0.0071 %."""
    got = float(measured["live"][convention]["self"])
    err = abs(got / float(measured["Ip_true"]) - 1.0)
    assert err <= _SELF_CONSISTENCY, (
        f"[{convention}] Ip_fsa_integral of the equilibrium's own profile is "
        f"{got:.6e} against a true Ip of {measured['Ip_true']:.6e} "
        f"({100 * err:.4f}%, bar {100 * _SELF_CONSISTENCY:.1f}%)")


@pytest.mark.solver
@solver_only
def test_the_snapshot_geometry_equals_the_live_geometry(measured):
    """``_AnchorIpRenorm`` evaluates every getter on a ``copy_eq`` snapshot; if
    that were not bit-faithful the frozen-anchor discipline would be a lie."""
    for conv in ("jphi-linterp", "fsa"):
        for what in ("self", "archived"):
            assert measured["snapshot"][conv][what] == \
                measured["live"][conv][what], f"{conv}/{what}"
    assert measured["Ip_after"] == measured["Ip_true"], \
        "reading the snapshot's getters perturbed the live solver"


@pytest.mark.solver
@solver_only
def test_dV_dpsi_is_per_dimensional_psi(measured):
    """The Jacobian half of the formula.  The dpsi reading recovers the volume;
    the dpsi_N reading is out by ~300 %, which is how a missing Jacobian would
    present."""
    vol = float(measured["vol_true"])
    assert abs(measured["live"]["vol_dpsi"] / vol - 1.0) <= 0.01
    assert measured["live"]["vol_dpsiN"] / vol > 2.0


#: A collapsed ``<R>`` is constant to ~2e-15 of its own magnitude; a healthy
#: D3D-like geometry spans ~1.7 m.  Anything between the two is neither, so the
#: build is classified on a threshold ten orders of magnitude clear of both.
#: This is a build-detection threshold, not a physics tolerance.
_COLLAPSE_REL = 1.0e-9


def _unclipped_collapses(measured):
    """Does THIS build exhibit the axis collapse?

    Measured, never inferred from a version string.  ``OpenFUSIONToolkit
    .__version__`` does not move with the surface-tracing code at all, and the
    collapse has been observed to depend on the *build* (compiler, ISA,
    platform) and not only on the source revision -- the same OFT commit
    collapses on one machine's build and traces every surface on another's.

    The raw span is not the detector: a surface the tracer fails on (the exact
    separatrix is one, on some builds) returns ``<R> = 0``, which by itself
    makes the span look healthy while every surface that WAS traced is still
    pinned to the axis.  Judge only the traced surfaces.
    """
    if measured["unclipped_n_surfaces"] - measured["unclipped_n_untraced"] < 2:
        return False
    return (measured["unclipped_R_span_traced"]
            <= _COLLAPSE_REL * measured["unclipped_R_traced_mean"])


@pytest.mark.solver
@solver_only
def test_the_clipping_is_what_makes_the_geometry_well_posed(measured):
    """The invariant, on EVERY OFT build: with the clipping in place the
    sampled geometry is the real one.

    This is the assertion bouquet's answer actually rests on.  Whether an
    UNCLIPPED grid would have collapsed is a property of the OFT build (see
    the test below); that the CLIPPED grid does not is a property of bouquet,
    and it must hold on the legacy builds and the fixed ones alike.
    """
    assert measured["clipped_R_span"] > _COLLAPSE_REL * measured["clipped_R_mean"], (
        f"the clipped grid produced a CONSTANT <R> "
        f"({measured['clipped_R_mean']:.6f} m across the profile) -- "
        "fsa_current_geometry's whole premise has failed on this build")
    # ... and it is a physical span, not merely non-zero.  On the D3D-like
    # baseline <R> runs 1.743 m (axis) -> 1.518 m (edge): a span of 0.224 m,
    # 13.3 % of the mean, measured identically on every build tried.
    assert measured["clipped_R_span"] > 0.1 * measured["clipped_R_mean"], (
        f"<R> spans only {measured['clipped_R_span']:.4f} m about a mean of "
        f"{measured['clipped_R_mean']:.4f} m -- expected ~0.224 m")
    lo, hi = measured["clipped_psi_q_ends"]
    assert lo > 0.0 and hi < 1.0, \
        f"the sampling grid was not clipped off both endpoints: [{lo}, {hi}]"


@pytest.mark.solver
@solver_only
def test_get_q_collapses_silently_on_an_unclipped_grid(measured):
    """The trap the clipping was written for, on the builds that have it.

    An exact ``psi_N = 0`` sample makes every surface return the AXIS values,
    with no exception raised by OFT -- the tracer starts from the axis, never
    leaves it, and ``pt_last`` carries that onto every later surface.  Where it
    bites, it is worth 37 % on the measure and is completely silent.

    It is **not** a property of an OFT revision, so it is not something a
    version check could gate on and not something an OFT upgrade retires: the
    same OFT commit has been measured collapsing on one build and tracing every
    surface on another.  The clipping therefore stays unconditionally, and this
    test asks the build in front of it rather than assuming.  A skip here means
    "this build did not reproduce the trap", never "the trap is fixed".
    """
    if not _unclipped_collapses(measured):
        pytest.skip(
            "this build does not reproduce the axis collapse: an unclipped "
            f"grid traced {measured['unclipped_n_surfaces'] - measured['unclipped_n_untraced']}"
            f"/{measured['unclipped_n_surfaces']} surfaces with a <R> span of "
            f"{measured['unclipped_R_span_traced']:.4f} m.  Nothing is retired "
            "by this -- the collapse is build-dependent, fsa_current_geometry "
            "still clips, and the guard is covered without a solver by "
            "test_the_collapse_guard_fires_on_a_constant_R_geometry.")
    assert measured["collapse_guard"], \
        "fsa_current_geometry accepted psi_pad=0 instead of raising"
    assert "collapsed" in measured["collapse_guard"]


# ---------------------------------------------------------------------------
#  fast: the collapse GUARD itself, on every build and with no solver
# ---------------------------------------------------------------------------
class _ConstantRGeom:
    """An ``eq`` whose ``get_q`` reports the same ``<R>`` on every surface.

    Exactly what a collapsed surface tracer returns.  Keeping this as a stub
    means the guard in :func:`bouquet.utils.fsa_current_geometry` is exercised
    on every OFT build -- including the ones that do not reproduce the collapse
    for the solver test above to trigger.
    """

    psi_bounds = (0.0, 1.5)

    def __init__(self, R0=1.7655, n=257):
        self._R0 = float(R0)
        self._n = int(n)

    def get_q(self, psi=None):
        x = np.asarray(psi, dtype=float)
        ones = np.ones_like(x)
        return (None, None, {"<R>": self._R0 * ones,
                             "<1/R>": ones / self._R0,
                             "<1/R^2>": ones / self._R0 ** 2,
                             "dV/dPsi": 12.0 * ones}, None, None, None)

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        x = np.asarray(psi, dtype=float)
        z = np.zeros_like(x)
        return (x, z, z, z, z)


def test_the_collapse_guard_fires_on_a_constant_R_geometry():
    """A constant ``<R>`` must raise, not be integrated into a plausible I_p.

    The collapse is silent at the OFT boundary -- no exception, no warning,
    just wrong numbers (37 % on the measure when it was found) -- so this
    guard is the only thing standing between it and a passing threshold gate.
    """
    from bouquet.utils import fsa_current_geometry

    psi_N = np.linspace(0.0, 1.0, 257)
    with pytest.raises(RuntimeError, match="collapsed"):
        fsa_current_geometry(_ConstantRGeom(), psi_N)


def test_a_genuinely_sheared_geometry_does_not_trip_the_guard():
    """The other half: the guard must not fire on a real geometry.  A stub
    with a realistic ``<R>`` shear passes through and keeps its arrays."""
    from bouquet.utils import fsa_current_geometry

    class _Sheared(_ConstantRGeom):
        def get_q(self, psi=None):
            x = np.asarray(psi, dtype=float)
            ones = np.ones_like(x)
            R = self._R0 * (1.0 - 0.25 * x)          # axis -> edge shear
            return (None, None, {"<R>": R, "<1/R>": ones / R,
                                 "<1/R^2>": ones / R ** 2,
                                 "dV/dPsi": 12.0 * ones}, None, None, None)

    psi_N = np.linspace(0.0, 1.0, 129)
    geom = fsa_current_geometry(_Sheared(), psi_N)
    assert np.ptp(geom["R_avg"]) > 0.1
    # the grid the getters were called on is clipped, the returned psi_N is not
    assert geom["psi_q"][0] > 0.0 and geom["psi_q"][-1] < 1.0
    assert geom["psi_N"][0] == 0.0 and geom["psi_N"][-1] == 1.0


#: Agreement demanded of the mesh flux integral against an independently
#: measured area / plasma integral (the bar PR #56 set for FI(1)).
_FLUX_INT_RTOL = 1.0e-2


@pytest.mark.solver
@solver_only
def test_compute_flux_integral_is_the_plasma_integral(measured):
    """``compute_flux_integral`` is ``int_plasma f dA`` for a profile that
    vanishes at the LCFS -- on EVERY build: a limiter-wide ``gs_flux_int``
    holds the profile at its LCFS value off-plasma, which is zero here.
    Compared with the plasma integral bouquet computes itself from the
    traced geometry, ``int (1 - psi_N) dA/dpsi_N dpsi_N``."""
    fi = float(measured["flux_integral_of_1_minus_psiN"])
    ref = float(measured["live"]["plasma_integral_of_1_minus_psiN"])
    assert abs(fi / ref - 1.0) <= _FLUX_INT_RTOL, (
        f"compute_flux_integral(1 - psi_N) = {fi:.5f} m^2 vs the plasma "
        f"integral {ref:.5f} m^2")


@pytest.mark.solver
@solver_only
def test_compute_flux_integral_of_one_is_the_area_its_build_integrates(
        measured):
    """FI(1) is the area of the domain this build's ``gs_flux_int`` covers,
    to the same bar, against an INDEPENDENT measurement of that domain:

    * a plasma-masked build (OpenFUSIONToolkit with the plasma-only
      ``gs_flux_int``): the plasma cross-section from the traced geometry;
    * a limiter-wide build (OpenFUSIONToolkit main at the time of writing:
      every ``reg == 1`` cell, the profile held at its LCFS value outside the
      plasma -- defect 3 of ``_AnchorIpRenorm``, FI(1) = 1.59x the plasma
      area on this example): the region-1 mesh area,
      ``compute_area_integral(1, reg_mask=1)``.

    Matching neither fails.  bouquet's own measures do not depend on which
    build it is (the pressure match is a ratio of one integral, and the I_p
    measure is ``Ip_fsa_integral``); this pins the property both ways."""
    fi_one = float(measured["flux_integral_of_one"])
    plasma = float(measured["live"]["plasma_area"])
    region = float(measured["region1_area"])
    # the two candidate domains are far apart on this example, so the
    # classification cannot be ambiguous at the bar
    assert abs(region / plasma - 1.0) > 10 * _FLUX_INT_RTOL
    d_plasma = abs(fi_one / plasma - 1.0)
    d_region = abs(fi_one / region - 1.0)
    assert min(d_plasma, d_region) <= _FLUX_INT_RTOL, (
        f"compute_flux_integral(1) = {fi_one:.5f} m^2 matches neither the "
        f"plasma cross-section {plasma:.5f} m^2 ({100 * d_plasma:.2f} %) nor "
        f"the region-1 mesh area {region:.5f} m^2 ({100 * d_region:.2f} %)")


if __name__ == "__main__":
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _probe(sys.argv[1])
