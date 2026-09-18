"""The ONE-SIDED (asymmetric Tikhonov) inductive prior in the structured closure.

``sigma_ind_up`` gives the inductive multipliers a second prior width, used
while a coefficient is POSITIVE; the ordinary ladder keeps the negative side.
A tight up-side sigma therefore lets ``s_ind`` FALL freely at that radius while
resisting a rise -- see ``utils.close_ip_structured``'s docstring for the
empirical motivation (the l_i channels lose on the under-shoot slices because
the MSE chords penalise inductive current ADDED at mid-radius), the supporting
mechanism (FUSE's collapsed edge T_e -> high edge resistivity -> too-fast
inward current diffusion -> mid-radius inductive current more likely OVER- than
under-estimated) and the circularity caveat that goes with both.

What is proved here:

* **the symmetric limit is EXACT, not approximate.**  ``sigma_ind_up=None`` is
  byte-identical to the channel as it shipped (that is what the pre-existing
  573 tests check, all of which pass unchanged); and passing an up ladder that
  EQUALS the down ladder reproduces the symmetric answer bit for bit, in both
  solvers.  Nothing is "close to" anything here: ``array_equal``.
* **an EXCESS slice is untouched.**  When the closure has to REMOVE inductive
  current every ``a_k`` is negative, the up side is never consulted, and the
  answer is bit-identical to the symmetric one.  This is the half of the
  population the prior must not disturb.
* **a DEFICIT slice is redirected.**  When the closure has to ADD current, a
  tight mid-radius ``sigma_ind_up`` drives the mid-radius coefficients to
  numerical zero (four orders below their symmetric values) and the current
  goes to the radii the prior leaves free instead.
* **the refusals fire**: a cycling sign pattern, an iteration that will not
  settle inside the cap, an up ladder that disagrees with the down ladder
  about which coefficients exist, and ``sigma_ind_up = inf`` on the hard
  solver (which would mean a trust weight of 0).

No solver anywhere -- the same synthetic geometry the rest of the structured
tests use, imported rather than re-derived.
"""
import numpy as np
import pytest

from bouquet.utils import (SIGN_ITER_MAX, _one_sided_sign_iterate,
                           close_ip_structured, close_ip_structured_soft)

from test_structured_closure import _axis, _parts

# The ladder the campaign runs (lib/run_closure_cloud.py): mid-radius and
# mantle free to fall, mid-radius resisted on the way up.
SIG_IND_DOWN = [0.10, 0.40, 0.40, 0.40]
SIG_IND_UP = [0.10, 0.10, 0.10, 0.40]
SIG_BS = [0.50, 0.30, 0.15, 0.10]
#: the same shape with the mid-radius up side effectively closed, so "resists a
#: rise" can be asserted as "does not rise" rather than as an inequality
SIG_IND_UP_TIGHT = [0.10, 1.0e-3, 1.0e-3, 0.40]
MID = (1, 2)            # the psi_N = 0.45 / 0.75 basis functions

WEIGHTS = dict(name="sigma-midfree",
               ind=[1.0 / s ** 2 for s in SIG_IND_DOWN],
               bs=[1.0 / s ** 2 for s in SIG_BS])


def _case(sign="deficit"):
    """``(hard_kwargs, soft_kwargs)`` for a deficit (+4 %) or excess (-4 %) slice.

    ``_parts()`` ships a 4 % DEFICIT (the size the FUSE core_profiles total
    actually misses by); the excess case is the same hybrid with the target
    moved 4 % the other way, so the two differ ONLY in the sign of the
    mismatch -- which is the gate this prior is built around.
    """
    psi, w, c, j_ind, j_bs, j_fix, _lin, Ip = _parts()
    raw = Ip / 1.04
    Ip_t = Ip if sign == "deficit" else 0.96 * raw
    hard = dict(psi_N=psi, w_lin=w, c_affine=c, Ip_target_signed=Ip_t,
                j_ind=j_ind, j_bs=j_bs, j_fix=j_fix, weights=WEIGHTS)
    soft = dict(psi_N=psi, w_lin=w, c_affine=c, Ip_target_signed=Ip_t,
                Ip_sigma=0.005 * abs(Ip_t), j_ind=j_ind, j_bs=j_bs,
                j_fix=j_fix, sigma_ind=SIG_IND_DOWN, sigma_bs=SIG_BS)
    return psi, hard, soft


# ---------------------------------------------------------------------------
# the symmetric limit
# ---------------------------------------------------------------------------
class TestSymmetricLimit:
    def test_none_leaves_the_record_saying_so(self):
        _psi, hard, soft = _case("deficit")
        h = close_ip_structured(**hard)
        s = close_ip_structured_soft(**soft)
        for out in (h, s):
            assert out["one_sided_ind"] is False
            assert out["sigma_ind_up"] is None
            assert out["sign_pattern"] is None
            assert out["n_sign_iter"] == 0
            assert out["weights_ind_up"] is None

    def test_hard_up_equals_down_is_bit_identical(self):
        _psi, hard, _soft = _case("deficit")
        base = close_ip_structured(**hard)
        same = close_ip_structured(sigma_ind_up=SIG_IND_DOWN, **hard)
        assert np.array_equal(base["a"], same["a"])
        assert np.array_equal(base["b"], same["b"])
        assert np.array_equal(base["s_ind"], same["s_ind"])
        assert np.array_equal(base["s_bs"], same["s_bs"])
        # it took a confirming second solve, and said so
        assert same["one_sided_ind"] is True
        assert same["n_sign_iter"] == 2

    def test_soft_up_equals_down_is_bit_identical(self):
        _psi, _hard, soft = _case("deficit")
        base = close_ip_structured_soft(**soft)
        same = close_ip_structured_soft(sigma_ind_up=SIG_IND_DOWN, **soft)
        assert np.array_equal(base["a"], same["a"])
        assert np.array_equal(base["b"], same["b"])
        assert base["objective"] == same["objective"]
        assert base["prior_chi2"] == same["prior_chi2"]

    def test_symmetric_limit_holds_with_the_axis_row_and_a_tight_ladder(self):
        """The equality is about the PRIOR, so it must survive the q0 row."""
        psi, hard, soft = _case("deficit")
        ax = _axis(psi, hard["j_ind"], hard["j_bs"], hard["j_fix"])
        base = close_ip_structured(axis=ax, **hard)
        same = close_ip_structured(axis=ax, sigma_ind_up=SIG_IND_DOWN, **hard)
        assert np.array_equal(base["a"], same["a"])
        b0 = close_ip_structured_soft(axis=ax, axis_sigma=None, **soft)
        b1 = close_ip_structured_soft(axis=ax, axis_sigma=None,
                                      sigma_ind_up=SIG_IND_DOWN, **soft)
        assert np.array_equal(b0["a"], b1["a"])


# ---------------------------------------------------------------------------
# the two signs of the mismatch
# ---------------------------------------------------------------------------
class TestExcessSliceUntouched:
    """The over-shoot half of the population the prior must NOT disturb."""

    def test_hard_excess_is_bit_identical_to_symmetric(self):
        _psi, hard, _soft = _case("excess")
        base = close_ip_structured(**hard)
        one = close_ip_structured(sigma_ind_up=SIG_IND_UP, **hard)
        assert np.all(base["a"] < 0.0)          # it really is removing current
        assert np.array_equal(base["a"], one["a"])
        assert np.array_equal(base["b"], one["b"])
        assert np.array_equal(base["s_ind"], one["s_ind"])
        assert one["sign_pattern"] == (False, False, False, False)
        assert one["n_sign_iter"] == 1          # the all-down start is stable

    def test_soft_excess_is_bit_identical_to_symmetric(self):
        _psi, _hard, soft = _case("excess")
        base = close_ip_structured_soft(**soft)
        one = close_ip_structured_soft(sigma_ind_up=SIG_IND_UP, **soft)
        assert np.all(base["a"] < 0.0)
        assert np.array_equal(base["a"], one["a"])
        assert np.array_equal(base["b"], one["b"])
        assert one["n_sign_iter"] == 1

    def test_an_arbitrarily_tight_up_side_still_changes_nothing(self):
        """Nothing about the excess answer may depend on the up-side width."""
        _psi, hard, _soft = _case("excess")
        base = close_ip_structured(**hard)
        for up in (SIG_IND_UP, SIG_IND_UP_TIGHT, [1.0e-6] * 4):
            one = close_ip_structured(sigma_ind_up=up, **hard)
            assert np.array_equal(base["a"], one["a"])


class TestDeficitSliceRedirected:
    """The under-shoot half: added inductive current is pushed off mid-radius."""

    def test_hard_deficit_puts_no_positive_coefficient_at_mid_radius(self):
        _psi, hard, _soft = _case("deficit")
        base = close_ip_structured(**hard)
        one = close_ip_structured(sigma_ind_up=SIG_IND_UP_TIGHT, **hard)
        assert np.all(base["a"] > 0.0)          # symmetric: it adds everywhere
        scale = float(np.max(np.abs(base["a"])))
        for k in MID:
            # abs(): the claim is "driven to numerical ZERO", so a coefficient
            # that over-corrected into a large NEGATIVE value has to fail this
            # too.  Without it the assertion passed for any undershoot.
            assert abs(one["a"][k]) <= 1.0e-4 * scale, (
                f"mid-radius basis {k} still carries {one['a'][k]:.3e} "
                f"against a symmetric {base['a'][k]:.3e}")
        # and the deficit did not simply go unclosed
        assert abs(one["ip_residual_pct"]) < 1.0e-6

    def test_soft_deficit_puts_no_positive_coefficient_at_mid_radius(self):
        _psi, _hard, soft = _case("deficit")
        base = close_ip_structured_soft(**soft)
        one = close_ip_structured_soft(sigma_ind_up=SIG_IND_UP_TIGHT, **soft)
        assert np.all(base["a"] > 0.0)
        scale = float(np.max(np.abs(base["a"])))
        for k in MID:
            assert abs(one["a"][k]) <= 1.0e-4 * scale

    def test_the_multiplier_rise_is_cut_at_the_resisted_radius(self):
        """The statement in profile space, which is what the chords see.

        Made at psi_N = 0.45 only, and deliberately so: the basis is four
        peak-normalised Gaussians of width 0.2, so the MANTLE function
        (centre 0.95) still has real amplitude at 0.75.  Once the prior pushes
        the added current out to the mantle, ``s_ind(0.75)`` can end up HIGHER
        than the symmetric answer even though the 0.75 coefficient itself has
        gone to zero -- a fact about basis overlap, not a failure of the
        prior, and asserting the profile there would only be asserting the
        basis.  The per-coefficient statement is the one that carries, and it
        is made above.
        """
        psi, hard, _soft = _case("deficit")
        base = close_ip_structured(**hard)
        one = close_ip_structured(sigma_ind_up=SIG_IND_UP_TIGHT, **hard)
        s_base = float(np.interp(0.45, psi, base["s_ind"]))
        s_one = float(np.interp(0.45, psi, one["s_ind"]))
        assert s_base > 1.02                     # symmetric lifts it clearly
        assert s_one < s_base
        # the current has to go somewhere: the radii left free take it
        assert one["a"][3] > base["a"][3]

    def test_the_campaign_ladder_shifts_current_off_mid_radius(self):
        """The ACTUAL ladder the runner uses, not just the extreme limit."""
        _psi, hard, _soft = _case("deficit")
        base = close_ip_structured(**hard)
        one = close_ip_structured(sigma_ind_up=SIG_IND_UP, **hard)
        for k in MID:
            assert one["a"][k] < base["a"][k]
        assert one["a"][3] > base["a"][3]
        assert one["sign_pattern"] == (True, True, True, True)
        # exactly 2: the all-down start solves once, flips every coefficient
        # up, and the second solve confirms the pattern.  The old
        # `1 <= n <= SIGN_ITER_MAX` window admitted any behaviour the cap
        # allows, including one that never consulted the up side at all.
        assert one["n_sign_iter"] == 2

    def test_an_arbitrarily_tight_up_ladder_is_solved_not_refused(self):
        """"Resist a rise HARD" is the documented intent of ``sigma_ind_up``,
        and it used to stop working past a sigma ratio of a few hundred: the
        singularity test was taken on the whole bordered matrix, whose (1,1)
        block is the prior, so a tight up side was refused as *degenerate
        constraint rows*.  Nothing in the docs or the tests warned of it, and
        the two rows here are manifestly independent.

        The tightest ladder below is a trust-weight range of ~1e13.  Ip stays
        exact throughout and the resisted coefficients go monotonically to
        zero -- the limit the prior claims, reached rather than refused.
        """
        _psi, hard, _soft = _case("deficit")
        base = close_ip_structured(**hard)
        scale = float(np.max(np.abs(base["a"])))
        prev = [abs(base["a"][k]) for k in MID]
        for tighten in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6):
            one = close_ip_structured(
                sigma_ind_up=[tighten * s for s in SIG_IND_DOWN], **hard)
            assert abs(one["ip_residual_pct"]) < 1.0e-9
            now = [abs(one["a"][k]) for k in MID]
            assert all(n < p for n, p in zip(now, prev)), (
                f"tightening to x{tighten:g} did not reduce {now} below "
                f"{prev}")
            prev = now
        assert all(v <= 1.0e-4 * scale for v in prev)   # numerical zero

    @pytest.mark.parametrize("up", ["campaign", "tight"])
    def test_a_tight_up_ladder_still_hits_a_hard_li_target(self, up):
        """The campaign ladder uses ``sigma_ind_up`` AND an l_i row, and the
        sign iteration re-solves a KKT that CONTAINS that row -- nothing
        checked the two together.  The l_i row is exact along the Ip-closed
        manifold, so the one-sided prior must not cost any of that exactness:
        it chooses among feasible points, it does not move the feasible set."""
        from test_li_closure import _li_parts, _li_of_closure, _model
        m, (psi, w, c, ji, jb, jf, lin, Ip_s, lg) = _model()
        base = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, li_geom=lg,
                                   weights=WEIGHTS)
        target = _li_of_closure(base, m) * 1.02
        ladder = SIG_IND_UP if up == "campaign" else SIG_IND_UP_TIGHT
        out = close_ip_structured(psi, w, c, Ip_s, ji, jb, jf, li_geom=lg,
                                  weights=WEIGHTS, li_target=target,
                                  sigma_ind_up=ladder)
        assert out["one_sided_ind"] is True
        assert abs(out["ip_residual_pct"]) < 1.0e-9
        assert out["li_predicted"] == pytest.approx(target, rel=1e-12)
        assert _li_of_closure(out, m) == pytest.approx(target, rel=1e-12)

    def test_the_record_carries_the_ladder_and_the_pattern(self):
        _psi, hard, soft = _case("deficit")
        for out in (close_ip_structured(sigma_ind_up=SIG_IND_UP, **hard),
                    close_ip_structured_soft(sigma_ind_up=SIG_IND_UP, **soft)):
            assert out["one_sided_ind"] is True
            assert np.allclose(out["sigma_ind_up"], SIG_IND_UP)
            assert len(out["sign_pattern"]) == 4
            assert np.allclose(out["weights_ind_up"],
                               [1.0 / s ** 2 for s in SIG_IND_UP])


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------
class TestSignIterationRefusals:
    def test_a_two_cycle_is_detected_and_named(self):
        """Alternating patterns must raise, not return the face it stops on."""
        flip = {(False,) * 4: np.array([1.0, 1.0, -1.0, -1.0, 0, 0, 0, 0]),
                (True, True, False, False):
                    np.array([-1.0, -1.0, 1.0, 1.0, 0, 0, 0, 0]),
                (False, False, True, True):
                    np.array([1.0, 1.0, -1.0, -1.0, 0, 0, 0, 0])}
        with pytest.raises(RuntimeError, match="CYCLING"):
            _one_sided_sign_iterate(lambda p: (flip[p],), 4, "unit-test")

    def test_a_pattern_that_never_settles_hits_the_cap(self):
        """A fresh pattern every time: the cap, not the cycle check, fires."""
        seq = iter(range(1, 64))

        def _never(_pat):
            n = next(seq)
            bits = [(n >> i) & 1 for i in range(4)]
            return (np.array([1.0 if b else -1.0 for b in bits] + [0.0] * 4),)

        with pytest.raises(RuntimeError, match="did not settle"):
            _one_sided_sign_iterate(_never, 4, "unit-test")

    def test_the_cap_is_the_documented_constant(self):
        assert SIGN_ITER_MAX == 8

    def test_a_stable_pattern_returns_the_solve_count(self):
        out, pat, it = _one_sided_sign_iterate(
            lambda p: (np.array([-1.0] * 8),), 4, "unit-test")
        assert pat == (False,) * 4 and it == 1
        assert np.array_equal(out[0], np.array([-1.0] * 8))


class TestLadderRefusals:
    def test_a_pin_on_one_side_only_is_refused_hard(self):
        _psi, hard, _soft = _case("deficit")
        with pytest.raises(ValueError, match="disagrees with the down-side"):
            close_ip_structured(sigma_ind_up=[0.0, 0.10, 0.10, 0.40], **hard)

    def test_a_pin_on_one_side_only_is_refused_soft(self):
        _psi, _hard, soft = _case("deficit")
        with pytest.raises(ValueError, match="disagrees with the down-side"):
            close_ip_structured_soft(sigma_ind_up=[0.0, 0.10, 0.10, 0.40],
                                     **soft)

    def test_an_unpenalised_up_side_is_refused_soft_unless_it_matches(self):
        _psi, _hard, soft = _case("deficit")
        with pytest.raises(ValueError, match="disagrees with the down-side"):
            close_ip_structured_soft(
                sigma_ind_up=[0.10, np.inf, 0.10, 0.40], **soft)

    def test_infinite_up_sigma_is_refused_on_the_hard_solver(self):
        _psi, hard, _soft = _case("deficit")
        with pytest.raises(ValueError, match="trust weight of 0"):
            close_ip_structured(sigma_ind_up=[0.10, np.inf, 0.10, 0.40],
                                **hard)

    def test_a_wrong_length_ladder_is_refused(self):
        _psi, hard, soft = _case("deficit")
        with pytest.raises(ValueError, match="ind_up"):
            close_ip_structured(sigma_ind_up=[0.10, 0.10], **hard)
        with pytest.raises(ValueError, match="ind_up"):
            close_ip_structured_soft(sigma_ind_up=[0.10, 0.10], **soft)

    def test_a_negative_up_sigma_is_refused(self):
        _psi, hard, _soft = _case("deficit")
        with pytest.raises(ValueError, match="non-negative"):
            close_ip_structured(sigma_ind_up=[0.10, -0.10, 0.10, 0.40],
                                **hard)

    def test_a_matching_pin_on_both_sides_is_accepted(self):
        """The check is about DISAGREEMENT, not about pins as such."""
        _psi, hard, _soft = _case("deficit")
        w = dict(WEIGHTS)
        w["ind"] = [np.inf] + w["ind"][1:]
        out = close_ip_structured(sigma_ind_up=[0.0, 0.10, 0.10, 0.40],
                                  **{**hard, "weights": w})
        assert out["a"][0] == 0.0
        assert out["one_sided_ind"] is True


# ---------------------------------------------------------------------------
# the config / run.py wiring
# ---------------------------------------------------------------------------
class TestConfigWiring:
    def test_the_field_defaults_to_none(self):
        from bouquet.config import GenerationConfig

        assert GenerationConfig().structured_sigma_ind_up is None

    def test_both_solvers_are_handed_the_field(self):
        """Assert on the source, so a future edit cannot drop one of them."""
        import inspect

        from bouquet import run as _run

        src = inspect.getsource(_run)
        assert 'getattr(gc, "structured_sigma_ind_up", None)' in src
        # both predictor calls plus the corrector state it is stashed in
        assert src.count("sigma_ind_up=sig_ind_up") == 3
        assert src.count('sigma_ind_up=state.get("sigma_ind_up")') == 2
