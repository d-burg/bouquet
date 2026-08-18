r"""The inductive share ``f_ind`` and the algebra the sigma=0 bound rests on.

Route R2 charges the whole :math:`I_p` miss to the inductive amplitude alone
(``j_BS`` is held fixed, by design), so the scale ``s`` that
``_AnchorIpRenorm.solve_scale`` returns is the Ip-space residual DIVIDED by the
inductive share:

.. math::
    s - 1 = \frac{-\Delta}{\int w\,j_{\rm ind}},\qquad
    |s-1|\,f_{\rm ind} = \left|\frac{\Delta}{I_p^{\rm demand}}\right| .

That identity is what lets ``tests/test_seeded_reproducibility`` state the
sigma=0 acceptance on ``|s-1| * f_ind`` -- the residual its budget is actually
derived in -- instead of on ``|s-1|``, whose stringency scales with
``1/f_ind`` and therefore with the operating point (issue #23).

**Solve-free.**  ``_AnchorIpRenorm`` only ever touches the frozen snapshot
through ``get_q`` / ``get_profiles`` / ``psi_bounds`` / ``get_stats``, so a stub
exercising those four is enough to pin the algebra exactly -- and it pins it
without a GS solve, which is the point: this is the part that must not drift
silently, and it should be checked on every run of the suite, not only under
``-m solver``.  The measured NUMBERS live with the solver probe.
"""
import numpy as np
import pytest

from bouquet.TokaMaker_interface import _AnchorIpRenorm, _r2_f_ind, _fmt_s_and_find
from bouquet.utils import Ip_fsa_weights


# ---------------------------------------------------------------------------
#  a stub equilibrium: large-aspect-ratio circular, analytic everywhere
# ---------------------------------------------------------------------------
_R0, _A, _NPSI = 3.0, 0.3, 401
_DPSI = 2.0                              # psi_bounds span, so dpsi/dpsi_N = 2


class _StubEq:
    """The four accessors ``_AnchorIpRenorm`` reads off a ``copy_eq`` snapshot.

    ``r = a*sqrt(psi_N)`` so ``dA/dpsi_N`` is constant and the current integral
    has a closed form; the FSA of any function of R is taken at ``R0`` (exact as
    ``a/R0 -> 0``).  ``P'`` and ``FF'`` are finite and shaped so the affine
    measure's constant term ``c`` is genuinely non-zero -- a stub with ``c == 0``
    would let a linear-vs-affine slip pass.
    """

    psi_bounds = (0.0, _DPSI)

    def __init__(self, pprime_amp=2.0e5, ffp_amp=0.4):
        self._pp_amp = float(pprime_amp)
        self._ffp_amp = float(ffp_amp)

    def _grid(self, psi):
        return np.asarray(psi, dtype=float)

    def get_q(self, psi=None):
        x = self._grid(psi)
        ones = np.ones_like(x)
        dV_dpsi = (np.pi * _A ** 2) * (2.0 * np.pi) * _R0 / _DPSI * ones
        ravgs = {
            # a tiny R-shear so fsa_current_geometry's axis-collapse guard
            # (constant <R> across surfaces) does not fire on the stub
            "<R>": _R0 * (1.0 + 0.02 * x),
            "<1/R>": ones / _R0,
            "<1/R^2>": 1.07 * ones / _R0 ** 2,   # != <1/R>^2 -> c != 0
            "dV/dPsi": dV_dpsi,
        }
        return (None, None, ravgs, None, None, None)

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        x = self._grid(psi)
        F = np.full_like(x, 5.0)
        Fp = self._ffp_amp * (1.0 - x)
        P = np.zeros_like(x)
        Pp = self._pp_amp * (1.0 - x ** 2)
        return (x, F, Fp, P, Pp)

    def compute_flux_integral(self, psi_vals, field_vals):
        # The retired ratio mode's measure (issue #35); kept on the stub so
        # the retirement guard below can prove the class no longer uses it.
        from scipy.integrate import trapezoid
        return float(trapezoid(np.asarray(field_vals, dtype=float)
                               * (np.pi * _A ** 2), np.asarray(psi_vals)))


class _StubGS:
    """Just enough ``TokaMaker`` for ``_AnchorIpRenorm.__init__``."""

    def __init__(self, Ip=1.0e6, eq=None):
        self._Ip = float(Ip)
        self._eq = eq if eq is not None else _StubEq()

    def copy_eq(self):
        return self._eq

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"Ip": self._Ip}


def _psi_grid():
    return np.linspace(0.0, 1.0, _NPSI)


def _split(psi_N, ind_frac=0.75):
    """An inductive + 'other' split with a controllable inductive weight."""
    j_ind = 1.0e6 * (1.0 - psi_N ** 2)
    j_oth = 3.0e5 * np.exp(-((psi_N - 0.95) / 0.03) ** 2) + 5.0e4
    return ind_frac * j_ind, j_oth


# ---------------------------------------------------------------------------
#  the identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["exact", "fsa"])
def test_s_minus_one_times_f_ind_is_the_ip_space_residual(mode):
    """``|s-1| * f_ind == |Delta / demand|`` to machine precision, in both
    surviving modes (ratio retired, issue #35).

    This is the whole justification for stating the sigma=0 acceptance on the
    product.  #23 verified it to 2.2e-16 on real probe output; here it is
    verified on a geometry where every term is known analytically, so a change
    to the measure cannot quietly satisfy it by coincidence.
    """
    psi_N = _psi_grid()
    j_ind, j_oth = _split(psi_N)
    total = j_ind + j_oth
    Ip_target = 1.0e6
    air = _AnchorIpRenorm(_StubGS(Ip=Ip_target), psi_N, total, Ip_target,
                          1e-3, mode=mode)

    s = air.solve_scale(j_ind, j_oth)
    f_ind = air.inductive_share(j_ind)

    # Delta measured against the demand this mode roots against.  (The
    # retired ratio mode calibrated its demand and rooted on the limiter-area
    # flux integral; both surviving modes root the FSA measure on Ip_target.)
    demand = air._target                                       # noqa: SLF001
    delta = air._Ip_of(total) - demand                         # noqa: SLF001

    assert f_ind > 0.0
    assert abs(s - 1.0) * f_ind == pytest.approx(
        abs(delta / demand), rel=1e-12, abs=1e-15), (
        f"[{mode}] the identity |s-1|*f_ind == |Delta/Ip| does not hold: "
        f"s={s!r}, f_ind={f_ind!r}, Delta/Ip={delta / demand!r}")


def test_f_ind_is_the_share_of_the_demand_the_inductive_profile_carries():
    """``f_ind`` is ``int(w*j_ind)/Ip_demand`` -- the LINEAR part only.

    The affine constant ``c`` (the ``P'`` term) belongs to neither component,
    so folding it in would make ``f_ind`` depend on the pressure and break the
    identity above.  Pinned against ``Ip_fsa_weights`` directly.
    """
    from bouquet.utils import fsa_current_geometry
    from scipy.integrate import trapezoid

    psi_N = _psi_grid()
    j_ind, j_oth = _split(psi_N)
    Ip_target = 1.0e6
    gs = _StubGS(Ip=Ip_target)
    air = _AnchorIpRenorm(gs, psi_N, j_ind + j_oth, Ip_target, 1e-3,
                          mode="exact")

    geom = fsa_current_geometry(gs.copy_eq(), psi_N)
    w, c = Ip_fsa_weights(geom, convention="jphi-linterp",
                          pprime_sign=air._pprime_sign)   # noqa: SLF001
    assert c != 0.0, "the stub's P' term vanished; the affine case is untested"
    assert air.inductive_share(j_ind) == pytest.approx(
        float(trapezoid(w * j_ind, psi_N)) / Ip_target, rel=1e-12)


def test_f_ind_scales_with_the_inductive_amplitude_not_the_bootstrap():
    """Doubling j_ind doubles f_ind; changing j_other does not move it.

    The physical content: f_ind is the inductive SHARE, which is why the same
    Ip-space residual reads larger in s-space on a high-bootstrap archive.
    """
    psi_N = _psi_grid()
    j_ind, j_oth = _split(psi_N)
    air = _AnchorIpRenorm(_StubGS(), psi_N, j_ind + j_oth, 1.0e6, 1e-3,
                          mode="exact")
    f1 = air.inductive_share(j_ind)
    assert air.inductive_share(2.0 * j_ind) == pytest.approx(2.0 * f1, rel=1e-12)
    # j_other is not an argument at all -- state the contract explicitly so a
    # future signature change that starts consuming it fails here.
    assert air.inductive_share(j_ind) == pytest.approx(f1, rel=0, abs=0)


def test_low_inductive_share_inflates_s_for_the_same_residual():
    """The #23 finding, reproduced in miniature: two archives with the SAME
    Ip-space residual report |s-1| in a ratio of their inductive shares.

    A constant bar on |s-1| is therefore a bar whose stringency tracks
    1/f_ind -- which is why the acceptance moved onto the product.
    """
    psi_N = _psi_grid()
    Ip_target = 1.0e6
    resid_target = 2.0e-3 * Ip_target       # the SAME Delta for both archives
    out = {}
    for tag, frac in (("high_f_ind", 0.9), ("low_f_ind", 0.45)):
        j_ind, j_oth = _split(psi_N, ind_frac=frac)
        air = _AnchorIpRenorm(_StubGS(Ip=Ip_target), psi_N, j_ind + j_oth,
                              Ip_target, 1e-3, mode="exact")
        # Trim j_other by a flat offset until the split carries exactly
        # demand + resid_target.  The measure is affine, so the offset is
        # closed-form; nothing about j_ind (hence f_ind) moves.
        total = j_ind + j_oth
        want = air._target + resid_target                      # noqa: SLF001
        unit = air._Ip_of(np.ones_like(psi_N)) - air._c        # noqa: SLF001
        j_oth_off = j_oth + (want - air._Ip_of(total)) / unit  # noqa: SLF001
        s = air.solve_scale(j_ind, j_oth_off)
        f = air.inductive_share(j_ind)
        out[tag] = (abs(s - 1.0), f, abs(s - 1.0) * f)

    d_hi, f_hi, p_hi = out["high_f_ind"]
    d_lo, f_lo, p_lo = out["low_f_ind"]
    assert f_lo < f_hi
    # products agree (same residual); the raw |s-1| do not
    assert p_lo == pytest.approx(p_hi, rel=1e-9)
    assert d_lo / d_hi == pytest.approx(f_hi / f_lo, rel=1e-9)
    assert d_lo > 1.5 * d_hi, (
        "the stub did not reproduce the denominator effect, so it cannot "
        "demonstrate why the bound was rebased")


# ---------------------------------------------------------------------------
#  the reporting contract
# ---------------------------------------------------------------------------
def test_r2_f_ind_returns_none_without_a_frozen_anchor():
    """Legacy / fallback root: no cached measure, so no share.  ``None``, not a
    plausible-looking 1.0 -- a wrong denominator would silently rescale the
    acceptance bound."""
    assert _r2_f_ind(None, np.ones(5)) is None


def test_the_qc_fragment_always_carries_both_numbers():
    """Reporting |s-1| without f_ind is the failure mode #23 identified, so the
    formatter is pinned: it prints both, and the product, or says n/a."""
    txt = _fmt_s_and_find(0.9968, 0.7614)
    assert "|s-1|" in txt and "f_ind=0.7614" in txt and "|s-1|*f_ind" in txt
    assert "2.435e-03" in txt or "2.436e-03" in txt   # 3.2e-3 * 0.7614
    assert _fmt_s_and_find(0.9968, None).endswith("f_ind=n/a")
    assert _fmt_s_and_find(0.9968, float("nan")).endswith("f_ind=n/a")


def test_the_qc_fragment_stamps_the_measure_mode():
    """``f_ind`` is normalised by the MODE's own Ip demand, so the log must say
    which mode produced it.

    ``exact`` and ``fsa`` differ by the affine ``P'`` constant, so their
    products are not interchangeable either.  (The worked ~2.1% example was
    exact-vs-ratio before the ratio retirement, issue #35.)
    """
    for mode in ("exact", "fsa"):
        txt = _fmt_s_and_find(0.9968, 0.7614, mode)
        assert f"[mode={mode}]" in txt, (
            f"the QC fragment does not stamp mode={mode!r}; an fsa-mode f_ind "
            "is not comparable with an exact-mode one")
    # the stamp must survive the n/a branch too -- that is where a reader has
    # least context and most need of it
    assert "[mode=fsa]" in _fmt_s_and_find(0.9968, None, "fsa")
    # omitted mode stays backward-compatible (no stray tag)
    assert "[mode=" not in _fmt_s_and_find(0.9968, 0.7614)


def test_both_R2_invariant_log_sites_pass_the_mode():
    """Guard the wiring, not just the formatter: an unstamped call site would
    print exactly the ambiguous line this stamp exists to remove."""
    import inspect
    import re
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium

    src = inspect.getsource(perturb_kinetic_equilibrium)
    calls = re.findall(r"_fmt_s_and_find\(([^)]*)\)", src)
    assert calls, "the R2 QC fragment is no longer emitted at all"
    for args in calls:
        assert args.count(",") >= 2, (
            f"_fmt_s_and_find({args}) is called without a mode argument; the "
            "[R2-invariant] line would not say which measure produced f_ind")
        assert "_r2_mode" in args, (
            f"_fmt_s_and_find({args}) does not pass the resolved _r2_mode")


# ---------------------------------------------------------------------------
#  the retirement (issue #35)
# ---------------------------------------------------------------------------
def test_ratio_mode_is_retired():
    """Constructing the anchor in the retired mode must raise, not silently
    build a calibrated demand.  Direct construction bypasses _r2_ip_mode's
    env-var guard, so the class needs its own."""
    psi_N = _psi_grid()
    j_ind, j_oth = _split(psi_N)
    total = j_ind + j_oth
    with pytest.raises(ValueError, match="retired"):
        _AnchorIpRenorm(_StubGS(Ip=1.0e6), psi_N, total, 1.0e6, 1e-3,
                        mode="ratio")


# ---------------------------------------------------------------------------
#  the floored-zone annotation (issue #35 item 1)
# ---------------------------------------------------------------------------
def test_floored_zone_note_annotates_only_floored_baselines():
    """On a floored baseline the QC line must say so -- the sigma=0 invariant
    carries an expected floor there (the amplitude root redistributes the
    floor-absorbed current) and a reader comparing against the bar needs
    that context on the line itself.  Un-floored baselines (the typical
    case) must be byte-unchanged."""
    from bouquet.TokaMaker_interface import _floored_zone_note

    j = np.linspace(1.0, 0.1, 12)
    assert _floored_zone_note(j) == ""
    assert _floored_zone_note(None) == ""

    j_floored = j.copy(); j_floored[-3:] = 0.0
    note = _floored_zone_note(j_floored)
    assert "3 floored" in note and "#35" in note
