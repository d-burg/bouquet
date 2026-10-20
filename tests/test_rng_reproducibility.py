"""The reproducibility contract: one seed governs every draw.

``GenerationConfig.seed`` used to reach only NumPy's legacy global RNG, while
every GPR draw site called ``generate_perturbed_GPR(..., rng=None)`` and got a
fresh ``np.random.default_rng()`` -- OS entropy per draw.  Seeded ensembles
were therefore not regenerable, and no draw-level value could be pinned.

These are the solver-free halves of the contract:

  * ``make_rng`` behaves as the single seed -> Generator entry point;
  * the samplers honour an injected Generator (same seed -> bitwise identical,
    different seed -> different, no rng -> independent);
  * every draw call site in the perturbation path passes ``rng=`` -- a source
    check, because the defect was silent at runtime (unseeded draws still
    *look* fine) and only a structural assertion prevents its return;
  * the committed goldens pin the seeded draw stream bitwise
    (``tests/golden/rng_stream_manifest.json`` from the g-file golden's
    baseline, ``tests/golden/ids_rng_stream_manifest.json`` from the IDS
    golden's).

The live-solver half -- two seeded ``generate()`` runs producing bitwise
identical archives -- is in ``test_seeded_reproducibility.py`` (``solver``).
"""
import ast
import hashlib
import json
import os

import numpy as np
import pytest

from bouquet.sampling import (make_rng, generate_perturbed_GPR,
                              _draw_monotonic_perturbation)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
#: golden -> (slim fixture, rng-stream manifest), as
#: tests/golden/make_golden_fixture.GOLDEN_FILES names them
_RNG_GOLDENS = {
    "gfile": ("D3Dlike_Hmode_golden_slim.h5", "rng_stream_manifest.json"),
    "ids": ("D3Dlike_ids_golden_slim.h5", "ids_rng_stream_manifest.json"),
}


@pytest.fixture(params=sorted(_RNG_GOLDENS))
def rng_golden(request):
    """``(slim, rng manifest)`` paths of one golden; skipped when absent."""
    slim, man = (os.path.join(_GOLDEN_DIR, n)
                 for n in _RNG_GOLDENS[request.param])
    if not (os.path.isfile(slim) and os.path.isfile(man)):
        pytest.skip(f"golden {request.param!r} not built; run "
                    "tests/golden/make_golden_fixture.py")
    return slim, man

_PSI = np.linspace(0.0, 1.0, 65)
_PROFILE = 1.0 - 0.85 * _PSI ** 2
_SIGMA = 0.05 * np.ones_like(_PSI)

# Cross-machine agreement of a seeded draw.  The Cholesky factorisation has no
# discrete choices for a LAPACK build to make, so the residue is rounding
# accumulation only: measured ~1e-9 on the golden's own channels under a
# sub-ulp kernel perturbation (the stand-in for a different LAPACK/libm build).
# 1e-7 leaves two decades of headroom while still catching any real change to
# the draw.  It is NOT a convergence or physics tolerance -- see the golden
# test's docstring.
_RTOL = 1e-7


# ---------------------------------------------------------------------------
#  make_rng -- the one seed-consumption point
# ---------------------------------------------------------------------------
class TestMakeRng:
    def test_int_seed_is_deterministic(self):
        assert make_rng(7).standard_normal(4).tolist() == \
            make_rng(7).standard_normal(4).tolist()

    def test_distinct_seeds_give_distinct_streams(self):
        assert make_rng(7).standard_normal(8).tolist() != \
            make_rng(8).standard_normal(8).tolist()

    def test_none_is_fresh_entropy(self):
        assert make_rng(None).standard_normal(8).tolist() != \
            make_rng(None).standard_normal(8).tolist()

    def test_generator_passes_through_unchanged(self):
        """A caller that already owns a stream can inject it."""
        g = np.random.default_rng(3)
        assert make_rng(g) is g

    def test_returns_a_generator(self):
        for s in (None, 0, 12345):
            assert isinstance(make_rng(s), np.random.Generator)


# ---------------------------------------------------------------------------
#  the samplers honour an injected Generator
# ---------------------------------------------------------------------------
class TestSamplersHonourRng:
    def test_gpr_same_seed_is_bitwise_identical(self):
        a = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(42))
        b = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_gpr_different_seed_differs(self):
        a = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(42))
        b = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1, rng=make_rng(43))
        assert not np.allclose(a, b)

    def test_gpr_without_rng_is_not_reproducible(self):
        """rng=None is the documented OS-entropy mode -- keep it that way."""
        a = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1)
        b = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                   n_samples=1)
        assert not np.allclose(a, b)

    def test_monotonic_draw_same_seed_is_bitwise_identical(self):
        """The rejection loop must draw from the RUN's stream, not a new one:
        it consumes a variable number of draws, so a per-attempt generator
        would desynchronise every later channel even if the seed were set."""
        kw = dict(psi_N=_PSI, normalised_profile=_PROFILE,
                  sigma_profile=_SIGMA, length_scale=0.3)
        np.testing.assert_array_equal(
            _draw_monotonic_perturbation(**kw, rng=make_rng(11)),
            _draw_monotonic_perturbation(**kw, rng=make_rng(11)))

    def test_monotonic_draw_different_seed_differs(self):
        kw = dict(psi_N=_PSI, normalised_profile=_PROFILE,
                  sigma_profile=_SIGMA, length_scale=0.3)
        assert not np.allclose(
            _draw_monotonic_perturbation(**kw, rng=make_rng(11)),
            _draw_monotonic_perturbation(**kw, rng=make_rng(12)))

    def test_one_generator_sequences_a_multi_channel_draw(self):
        """Two channels off ONE Generator reproduce as a *sequence*: replaying
        only the second call must NOT match, or the stream is not shared."""
        rng = make_rng(5)
        first = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                       n_samples=1, rng=rng)
        second = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                        n_samples=1, rng=rng)
        assert not np.allclose(first, second)
        rng2 = make_rng(5)
        np.testing.assert_array_equal(
            first, generate_perturbed_GPR(_PSI, _PROFILE,
                                          sigma_profile=_SIGMA, n_samples=1,
                                          rng=rng2))
        np.testing.assert_array_equal(
            second, generate_perturbed_GPR(_PSI, _PROFILE,
                                           sigma_profile=_SIGMA, n_samples=1,
                                           rng=rng2))


# ---------------------------------------------------------------------------
#  structural: no draw site may fall back to OS entropy
# ---------------------------------------------------------------------------
_DRAW_FUNCS = {"generate_perturbed_GPR", "_draw_monotonic_perturbation"}
_PERTURB_MODULE = os.path.join(_HERE, "..", "bouquet",
                               "TokaMaker_interface.py")


def _draw_calls(path):
    """(lineno, func, passes_rng) for every draw call in a module."""
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in _DRAW_FUNCS:
            yield node.lineno, name, any(k.arg == "rng" for k in node.keywords)


def test_every_draw_site_threads_the_generator():
    """The regression guard for the whole bug class.

    An unseeded draw site is invisible at runtime -- it produces perfectly
    good-looking numbers -- so the only durable protection is asserting that
    no call site omits ``rng=``.  Nine sites existed when this was written.
    """
    calls = list(_draw_calls(_PERTURB_MODULE))
    assert calls, "found no draw call sites; did the module move?"
    missing = [(ln, fn) for ln, fn, has_rng in calls if not has_rng]
    assert not missing, (
        f"draw call sites without an explicit rng= (they would fall back to "
        f"OS entropy and break the seed contract): {missing}")


def test_generate_bouquet_consumes_the_seed_once():
    """``seed`` must become a Generator, not just np.random.seed()."""
    with open(_PERTURB_MODULE) as fh:
        src = fh.read()
    assert "rng = make_rng(seed)" in src
    # and the two formerly-legacy draws must come off it
    assert "rng.uniform(" in src, "scale_jbs draw is not on the run Generator"
    assert "rng.normal(" in src, "l_i target draw is not on the run Generator"


# ---------------------------------------------------------------------------
#  the draw must not depend on LAPACK's freedom of basis
# ---------------------------------------------------------------------------
class TestDrawIsFactorizationStable:
    r"""The GP kernel is factorised by Cholesky precisely so that there is
    nothing for a LAPACK build to decide.

    The previous ``eigh``-based factorisation handed LAPACK two conventions --
    the near-null-subspace basis and every eigenvector's sign -- and both
    reached the draw: the same seed produced 1.3%-different profiles between
    the macOS and Linux CI builds, and per-vector sign canonicalisation
    provably could not close the gap (near-degenerate pairs can still swap or
    rotate; flat-sigma kernels are mirror-symmetric and exercise exactly that).
    A fixed-order Cholesky of ``K + jitter*I`` has no discrete choices at all,
    so cross-build variation collapses to rounding accumulation -- measured at
    ~1e-9 on the golden's own channels and ~4e-10 on the flat-sigma matern52
    config that was the eigh era's worst case.

    These tests pin (i) the structural property that ``eigh`` is out of the
    draw path, (ii) the measured stability, with two decades of margin, and
    (iii) the sampling law, which the jitter must not visibly perturb.
    """

    def _draw(self, sigma=None, kf="rbf", ls=0.25):
        return generate_perturbed_GPR(_PSI, _PROFILE,
                                      sigma_profile=_SIGMA if sigma is None else sigma,
                                      length_scale=ls, kernel_func=kf,
                                      n_samples=1, rng=make_rng(20260804))

    def test_the_draw_path_does_not_use_eigh(self, monkeypatch):
        """Structural guard: any eigen-decomposition reintroduces basis freedom,
        so the draw must not touch one.  Reverting to the eigh factorisation
        (canonicalised or not) fails here immediately."""
        def boom(*_a, **_k):
            raise AssertionError("np.linalg.eigh reached the GP draw path")
        monkeypatch.setattr(np.linalg, "eigh", boom)
        self._draw()                               # must succeed without eigh

    @pytest.mark.parametrize("kf,ls", [("rbf", 0.25), ("matern52", 0.4)],
                             ids=["rbf", "matern52-flat(eigh-era worst)"])
    def test_sub_ulp_kernel_perturbation_does_not_move_the_draw(
            self, monkeypatch, kf, ls):
        """A different LAPACK/libm build differs from this one by rounding --
        modelled as a 1e-15 relative symmetric perturbation of the kernel
        before factorisation.  Measured residue: 1.2e-9 (rbf) / 2.5e-11
        (matern52) on this grid; asserted with two decades of margin.  The raw
        eigh factorisation moved up to 2.3e-5 under the same perturbation on
        flat-sigma kernels."""
        reference = self._draw(kf=kf, ls=ls)       # BEFORE any patching
        real = np.linalg.cholesky

        def noisy(A, _r=real):
            rs = np.random.default_rng(0)
            E = rs.standard_normal(A.shape)
            return _r(A + 1e-15 * np.abs(A).max() * 0.5 * (E + E.T))

        monkeypatch.setattr(np.linalg, "cholesky", noisy)
        moved = np.abs(self._draw(kf=kf, ls=ls) - reference).max() \
            / np.abs(reference).max()
        assert moved < _RTOL, f"draw moved {moved:.2e} under a sub-ulp kernel change"

    def test_law_is_unchanged_by_the_factorisation(self):
        """Marginal 1sigma must still equal the experimental envelope."""
        n = 20000
        draws = generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=_SIGMA,
                                       length_scale=0.25, n_samples=n,
                                       rng=make_rng(4))
        # MC noise floor on a std from n samples is 1/sqrt(2n) ~ 5e-3
        np.testing.assert_allclose(draws.std(axis=0, ddof=1), _SIGMA,
                                   rtol=0.05)

    def test_jitter_inflation_is_below_the_mc_floor(self):
        """The regularisation inflates each marginal std by sqrt(1+eps/sigma^2);
        pin it far below anything a user could observe.  Computed directly from
        the factor the sampler builds, not from draws (the MC floor is 5e-3)."""
        from bouquet.sampling import GPRProfilePerturber, _CHOL_JITTER
        p = GPRProfilePerturber(kernel_func="rbf", length_scale=0.25)
        K = p._kernel(_PSI, _PSI) * np.outer(_SIGMA, _SIGMA)
        n = _PSI.size
        jit = _CHOL_JITTER * n * K.diagonal().max()
        L = np.linalg.cholesky(K + jit * np.eye(n))
        std = np.sqrt(np.einsum("ij,ij->i", L, L))
        rel = np.abs(std - _SIGMA).max() / _SIGMA.max()
        assert rel < 1e-6, f"jitter inflates sigma by {rel:.2e}"

    def test_all_zero_sigma_is_exactly_unperturbed(self):
        """sigma == 0 everywhere must return the profile bit-exactly -- the
        sigma=0 consistency guard depends on it."""
        out = generate_perturbed_GPR(_PSI, _PROFILE,
                                     sigma_profile=np.zeros_like(_PSI),
                                     length_scale=0.25, n_samples=1,
                                     rng=make_rng(3))
        np.testing.assert_array_equal(out, _PROFILE)

    @pytest.mark.parametrize("zero_sigma", [False, True], ids=["drawn", "zero"])
    def test_rng_consumption_never_depends_on_the_data(self, zero_sigma):
        """z is drawn full-length on EVERY path (including sigma==0), or later
        channels in a seeded multi-channel run would shift."""
        sig = np.zeros_like(_PSI) if zero_sigma else _SIGMA
        a, b = make_rng(7), make_rng(7)
        generate_perturbed_GPR(_PSI, _PROFILE, sigma_profile=sig,
                               n_samples=1, rng=a)
        b.standard_normal((_PSI.size, 1))
        assert a.standard_normal() == b.standard_normal()


# ---------------------------------------------------------------------------
#  committed golden: the seeded draw stream
# ---------------------------------------------------------------------------
def test_seeded_draw_stream_matches_golden(rng_golden):
    """Pin the seeded draw stream against the golden baseline profiles.

    The first draw-level golden the package can hold at all: before the seed
    reached the GPR the stream was OS entropy.  Re-pin deliberately with
    ``python tests/golden/make_golden_fixture.py --rng-stream-only`` and review
    the manifest diff.

    Asserted in two strengths, because the draw's reproducibility genuinely
    has two strengths (see ``draw_stream``):

      * **numerically, everywhere** -- ``_RTOL``.  The GP draw factorises the
        kernel with ``eigh`` and applies it with a ``gemm``; neither is
        bit-specified across LAPACK/BLAS builds.  ``eigh``'s two real degrees
        of freedom -- eigenvector sign, and an arbitrary basis for the
        numerically-null subspace -- are removed in the sampler, which is what
        brings a cross-machine draw from ~1e-1 to ~1e-9; the residue is
        floating-point reduction order and is not removable.
      * **bitwise, on the machine that pinned it** -- the SHA-256, which is the
        contract seeded ``generate()`` runs actually depend on.  Skipped
        elsewhere: a mismatch there would be reporting the BLAS, not bouquet.

    A value moving by more than ``_RTOL`` is therefore a real change in the
    sampler, not a platform difference.
    """
    import h5py
    import sys
    sys.path.insert(0, _GOLDEN_DIR)
    from make_golden_fixture import draw_stream, blas_provenance

    slim, rng_manifest = rng_golden
    with open(rng_manifest) as fh:
        man = json.load(fh)
    assert man["source_fixture"] == os.path.basename(slim)

    with h5py.File(slim, "r") as hf:
        bkeys = sorted(hf["scan"].keys()) if "scan" in hf else [None]
        bl = (hf[f"scan/{bkeys[0]}/_baseline"] if bkeys[0] is not None
              else hf["_baseline"])
        psi_kin = np.asarray(bl["psi_N_kinetic"][()], dtype=float)
        psi_N = np.asarray(bl["psi_N"][()], dtype=float)
        profiles = {"ne": bl["n_e"][()], "te": bl["T_e"][()],
                    "ni": bl["n_i"][()], "ti": bl["T_i"][()],
                    "jphi": bl["jphi_total"][()]}
        sigmas = {"ne": bl["sigma_ne"][()], "te": bl["sigma_te"][()],
                  "ni": bl["sigma_ni"][()], "ti": bl["sigma_ti"][()],
                  "jphi": bl["sigma_jphi"][()]}

    assert psi_kin.size == man["grids"]["psi_N_kinetic"]
    assert psi_N.size == man["grids"]["psi_N"]

    drawn = draw_stream(psi_kin, psi_N, profiles, sigmas, seed=man["seed"])
    assert set(drawn) == set(man["channels"])

    same_machine = man.get("pinned_on") == blas_provenance()

    for ch, exp in man["channels"].items():
        a = np.ascontiguousarray(drawn[ch], dtype=np.float64)
        assert a.size == exp["n"], ch
        # readable first: which value moved
        for i, v in zip(exp["sample_indices"], exp["sample_values"]):
            np.testing.assert_allclose(float(a[i]), v, rtol=_RTOL,
                                       err_msg=f"{ch}[{i}]")
        np.testing.assert_allclose(float(a.min()), exp["min"], rtol=_RTOL,
                                   err_msg=f"{ch} min")
        np.testing.assert_allclose(float(a.max()), exp["max"], rtol=_RTOL,
                                   err_msg=f"{ch} max")
        # then exhaustive -- only where bitwise is a claim we can make
        if same_machine:
            assert hashlib.sha256(a.tobytes()).hexdigest() == exp["sha256"], (
                f"{ch}: draw stream changed bitwise on the machine that pinned "
                f"it (sampled values still matched to {_RTOL:g}, so the change "
                f"is elsewhere in the profile)")


def test_golden_draw_stream_actually_perturbs(rng_golden):
    """Guard the guard: a golden of an all-zero-sigma stream would pass for
    the wrong reason."""
    import h5py
    with h5py.File(rng_golden[0], "r") as hf:
        bkeys = sorted(hf["scan"].keys()) if "scan" in hf else [None]
        bl = (hf[f"scan/{bkeys[0]}/_baseline"] if bkeys[0] is not None
              else hf["_baseline"])
        for k in ("sigma_ne", "sigma_te", "sigma_ni", "sigma_ti",
                  "sigma_jphi"):
            assert float(np.max(np.abs(bl[k][()]))) > 0.0, k
