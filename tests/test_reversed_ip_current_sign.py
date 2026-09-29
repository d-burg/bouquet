"""Reversed-current sources: the IMAS reader brings every current into one frame.

bouquet solves every TokaMaker anchor to ``|Ip|`` with ``F0 = |r0*b0|`` -- a
positive-current frame -- so every bootstrap it recomputes on the anchor (the
legacy ``solve_with_bootstrap`` path and the self-consistent loop's
``evaluate_jBS``) is positive.  A dd written for a reversed-current discharge
(``ip < 0`` in the dd's own COCOS) carries NEGATIVE current profiles.  The
reader used to keep that sign, so the inductive and fixed currents stayed
negative while the recomputed bootstrap was positive: the bootstrap was added
AGAINST Ip, ``closure_sign_convention`` re-signed the Ip target and the P'
constant but never the bootstrap, and ``unrenormalise_q0`` mixed the two
frames into a negative q0 target.

The fix is at the reader: every current profile is multiplied by
``sign(equilibrium ip)`` (``Baseline.source_current_sign``).  These tests pin
it WITHOUT a solver:

  * a mirrored copy of the source -- all four (Ip, B0) orientations
    (``tests/_mirror_dd.py``) -- reads to a Baseline that is BIT-IDENTICAL to
    the original's.  Bitwise, not approximate: the normalisation is a
    multiplication by +-1 (exact in IEEE arithmetic) and every operation the
    reader applies to a current afterwards (the ratio conversion
    ``j_tor/j_total``, the residual ``j_phi - j_BS - j_NBI - j_RF``, the
    ``jphi_diff`` difference) is odd in it, so there is no rounding to allow for;
  * the reader records the sign, logs it for a reversed source, and leaves an
    ``ip > 0`` read exactly as it was;
  * everything downstream that meets a recomputed (positive) bootstrap -- the
    closure's current-direction pairing, ``close_ip``, the q0 target, the
    FUSE-total error, the SWB/FUSE peak ratio, the ``floor_j_BS`` clip and
    ``floor_inductive_split`` -- sees identical inputs for every orientation,
    with a witness that the pre-fix (dd-frame) inputs did not.

The live-solver counterpart (the full forward solve of every mirror, bitwise)
is ``test_reversed_ip_solver.py``.  No proprietary data: the only dds here are
the shipped synthetic D3D-like example and dds built in-process.
"""
import json
import warnings

import numpy as np
import pytest
from scipy.integrate import trapezoid

import _mirror_dd as mdd
from bouquet.config import FixedComponentsConfig, ImasSource
from bouquet.io.imas import read_imas_baseline, source_current_sign
from test_imas_p_fast_convention import _minimal_dd


#: Every array field of the Baseline the IMAS reader fills.
_ARRAYS = ("psi_N", "j_phi", "j_inductive", "j_BS", "j_NBI", "j_RF",
           "jphi_diff", "psi_N_kinetic", "ne", "te", "ni", "ti", "Zeff",
           "p_fast", "z_fast", "p_equilibrium", "p_diff")
_SCALARS = ("Ip_target", "l_i_target", "Z_imp", "provenance", "p_fast_meta",
            "li_metrics")


def _read(path, time=mdd.EXAMPLE_TIME, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(ImasSource(ids_path=path, time=time), **kw)


def _example():
    with open(mdd.EXAMPLE_DD) as fh:
        return json.load(fh)


def _write(tmp_path, dd, name):
    p = tmp_path / name
    p.write_text(json.dumps(dd))
    return str(p)


def _assert_same_baseline(ref, bl, what):
    for f in _ARRAYS:
        a, b = getattr(ref, f), getattr(bl, f)
        if a is None or b is None:
            assert a is None and b is None, (what, f)
            continue
        assert np.array_equal(np.asarray(a), np.asarray(b)), (
            f"{what}: {f} differs (max |diff| "
            f"{float(np.max(np.abs(np.asarray(a) - np.asarray(b))))})")
    for f in _SCALARS:
        assert getattr(ref, f) == getattr(bl, f), (what, f)
    # aux (rotation, E_r, chi, zeff) is not a current and is not mirrored
    assert ref.aux.keys() == bl.aux.keys()
    for k in ref.aux:
        assert np.array_equal(ref.aux[k], bl.aux[k]), (what, "aux", k)
    # the sawtooth gate inputs: identical except the dd's own q0, recorded raw
    # in the dd's COCOS -- the gate reads it only as |q0_dd|
    sa, sb = dict(ref.sawtooth), dict(bl.sawtooth)
    qa, qb = sa.pop("q0_dd"), sb.pop("q0_dd")
    assert sa == sb, what
    assert (qa is None and qb is None) or abs(qa) == abs(qb), what


# ---------------------------------------------------------------------------
#  1. mirrored sources read to the SAME baseline, bit for bit
# ---------------------------------------------------------------------------
class TestMirroredReadIsBitIdentical:
    @pytest.mark.parametrize("time", [2.1, 2.2, mdd.EXAMPLE_TIME])
    @pytest.mark.parametrize("orient", mdd.ORIENTATIONS,
                             ids=[mdd.tag(*o) for o in mdd.ORIENTATIONS])
    def test_example_dd_every_orientation(self, tmp_path, orient, time):
        dd = _example()
        ref = _read(_write(tmp_path, dd, "ref.json"), time=time)
        bl = _read(_write(tmp_path, mdd.mirror_dd(dd, *orient), "m.json"),
                   time=time)
        _assert_same_baseline(ref, bl, mdd.tag(*orient))

    @pytest.mark.parametrize("orient", mdd.ORIENTATIONS,
                             ids=[mdd.tag(*o) for o in mdd.ORIENTATIONS])
    def test_minimal_dd_every_orientation(self, tmp_path, orient):
        dd = _minimal_dd()
        ref = _read(_write(tmp_path, dd, "ref.json"), time=1.0,
                    p_fast_reduction="trace")
        bl = _read(_write(tmp_path, mdd.mirror_dd(dd, *orient), "m.json"),
                   time=1.0, p_fast_reduction="trace")
        _assert_same_baseline(ref, bl, mdd.tag(*orient))

    def test_every_current_is_in_the_positive_frame(self, tmp_path):
        """The reversed-current mirror's currents carry Ip's positive sign --
        they were NEGATIVE here before the fix (the dd-frame read)."""
        dd = mdd.mirror_dd(_example(), -1.0, 1.0)
        bl = _read(_write(tmp_path, dd, "rev.json"))
        psi = np.asarray(bl.psi_N)
        assert trapezoid(bl.j_phi, psi) > 0.0
        assert trapezoid(bl.j_inductive, psi) > 0.0
        assert trapezoid(bl.j_BS, psi) > 0.0
        assert float(np.min(bl.j_BS)) >= 0.0   # the example's bootstrap is >= 0

    def test_user_supplied_fixed_currents_take_the_dd_orientation(self, tmp_path):
        """A FixedComponentsConfig current replaces a dd quantity, so it is
        given in the dd's own orientation and normalised with it."""
        dd = _example()
        psi_fc = np.linspace(0.0, 1.0, 21)
        jnbi = 2.0e5 * (1.0 - psi_fc ** 2)
        jrf = 5.0e4 * np.exp(-((psi_fc - 0.3) / 0.1) ** 2)
        ref = _read(_write(tmp_path, dd, "ref.json"),
                    fixed=FixedComponentsConfig(j_NBI=jnbi, j_RF=jrf,
                                                psi_N=psi_fc))
        bl = _read(_write(tmp_path, mdd.mirror_dd(dd, -1.0, 1.0), "rev.json"),
                   fixed=FixedComponentsConfig(j_NBI=-jnbi, j_RF=-jrf,
                                               psi_N=psi_fc))
        for f in ("j_NBI", "j_RF", "j_inductive", "j_phi"):
            assert np.array_equal(getattr(ref, f), getattr(bl, f)), f


# ---------------------------------------------------------------------------
#  2. the reader records (and logs) what it did; ip > 0 is untouched
# ---------------------------------------------------------------------------
class TestSignIsRecorded:
    @pytest.mark.parametrize("orient", mdd.ORIENTATIONS,
                             ids=[mdd.tag(*o) for o in mdd.ORIENTATIONS])
    def test_baseline_records_the_source_orientation(self, tmp_path, orient):
        s_ip, s_b0 = orient
        bl = _read(_write(tmp_path, mdd.mirror_dd(_example(), *orient), "m.json"))
        # the example itself is Ip > 0, B0 < 0
        assert bl.source_current_sign == s_ip
        assert bl.source_b0_sign == -s_b0

    def test_reversed_source_is_logged_and_positive_is_silent(self, tmp_path,
                                                              capsys):
        dd = _example()
        _read(_write(tmp_path, dd, "ref.json"))
        assert "source ip" not in capsys.readouterr().out
        _read(_write(tmp_path, mdd.mirror_dd(dd, -1.0, 1.0), "rev.json"))
        out = capsys.readouterr().out
        assert "source ip = -1.2346 MA < 0" in out
        assert "multiplied by -1" in out and "source_current_sign = -1" in out

    def test_source_current_sign_helper(self):
        assert source_current_sign(1.2e6) == 1.0
        assert source_current_sign(-1.2e6) == -1.0
        # no orientation to read: the read is left untouched
        assert source_current_sign(0.0) == 1.0
        assert source_current_sign(float("nan")) == 1.0

    def test_positive_read_is_the_pre_fix_read(self, tmp_path):
        """For ip > 0 the factor is exactly +1.0: the Baseline currents are the
        dd's own arrays run through the same formulas as before, bit for bit."""
        from bouquet.physics import parallel_to_toroidal

        dd = _example()
        bl = _read(_write(tmp_path, dd, "ref.json"))
        eq = dd["equilibrium"]
        ie = int(np.argmin(np.abs(np.asarray(eq["time"]) - mdd.EXAMPLE_TIME)))
        cp = dd["core_profiles"]["profiles_1d"][ie]
        jtot = np.asarray(cp["j_total"], float)
        jtor = np.asarray(cp["j_tor"], float)
        j_bs = parallel_to_toroidal(np.asarray(cp["j_bootstrap"], float),
                                    j_parallel_total=jtot, j_tor_total=jtor)
        jnbi_par = np.zeros_like(jtor)
        for s in dd["core_sources"]["source"]:
            if s["identifier"]["index"] == 2:
                jnbi_par = jnbi_par + np.asarray(s["profiles_1d"][ie]["j_parallel"], float)
        j_nbi = parallel_to_toroidal(jnbi_par, j_parallel_total=jtot,
                                     j_tor_total=jtor)
        assert bl.source_current_sign == 1.0
        assert np.array_equal(bl.j_phi, jtor)
        assert np.array_equal(bl.j_BS, j_bs)
        assert np.array_equal(bl.j_NBI, j_nbi)
        assert np.array_equal(bl.j_inductive, jtor - j_bs - j_nbi - np.zeros_like(jtor))

    def test_dd_whose_currents_oppose_its_own_ip_is_refused(self, tmp_path):
        """No sign convention can repair a dd whose core_profiles total opposes
        its own equilibrium ip -- the reader REFUSES instead of closing on it
        (it used to warn and continue with the mixed frame)."""
        dd = _example()
        for c in dd["core_profiles"]["profiles_1d"]:
            for k in ("j_tor", "j_total", "j_ohmic", "j_bootstrap"):
                c[k] = [-v for v in c[k]]
        with pytest.raises(ValueError,
                           match="disagree in sign with its plasma current"):
            read_imas_baseline(ImasSource(ids_path=_write(tmp_path, dd, "bad.json"),
                                          time=mdd.EXAMPLE_TIME))


# ---------------------------------------------------------------------------
#  3. downstream: everything that meets a recomputed (positive) bootstrap
# ---------------------------------------------------------------------------
def _closure_inputs(bl, j_bs_recomputed):
    """The pieces the ohmic-mode closure forms, on a synthetic positive measure:
    the reader's inductive/fixed currents and a bootstrap recomputed on the
    positive anchor (stood in for by the ORIGINAL orientation's j_BS)."""
    psi = np.asarray(bl.psi_N, float)
    w = 1.0 + 0.5 * psi                     # positive FSA weights (any will do)
    lin = lambda j: float(trapezoid(w * np.asarray(j, float), psi))
    j_fixed = np.asarray(bl.j_phi) - np.asarray(bl.j_inductive) - np.asarray(bl.j_BS)
    c = 0.03 * bl.Ip_target                 # the anchor's positive P' constant
    return lin(bl.j_inductive), lin(j_bs_recomputed), lin(j_fixed), c


class TestDownstreamSeesOneFrame:
    @pytest.fixture
    def pair(self, tmp_path):
        dd = _example()
        ref = _read(_write(tmp_path, dd, "ref.json"))
        rev = _read(_write(tmp_path, mdd.mirror_dd(dd, -1.0, 1.0), "rev.json"))
        return ref, rev

    def test_closure_pairing_and_scales_are_identical(self, pair):
        from bouquet.utils import close_ip, closure_sign_convention

        ref, rev = pair
        swb = np.asarray(ref.j_BS)          # recomputed on the positive anchor
        out = []
        for bl in (ref, rev):
            ii, ib, ifx, c = _closure_inputs(bl, swb)
            sgn, tgt, cs = closure_sign_convention(ii, ib, ifx, c, bl.Ip_target)
            assert sgn == 1.0
            out.append((sgn, tgt, cs) + tuple(close_ip("bootstrap", tgt, cs,
                                                        ii, ib, ifx,
                                                        scale_bounds=(-1e9, 1e9))))
        assert out[0] == out[1]

    def test_witness_the_dd_frame_read_closed_differently(self, pair):
        """What the fix removes: with the dd-frame (negative) inductive and
        fixed currents against a positive recomputed bootstrap, the pairing
        flips (sgn = -1) and the bootstrap scale that closes Ip is NOT the
        normal-orientation one -- the posterior asked for s_bs ~ -1."""
        from bouquet.utils import close_ip, closure_sign_convention

        ref, _ = pair
        swb = np.asarray(ref.j_BS)
        ii, ib, ifx, c = _closure_inputs(ref, swb)
        sgn, tgt, cs = closure_sign_convention(ii, ib, ifx, c, ref.Ip_target)
        good = close_ip("bootstrap", tgt, cs, ii, ib, ifx, scale_bounds=(-1e9, 1e9))
        sgn_b, tgt_b, cs_b = closure_sign_convention(-ii, ib, -ifx, c, ref.Ip_target)
        bad = close_ip("bootstrap", tgt_b, cs_b, -ii, ib, -ifx,
                       scale_bounds=(-1e9, 1e9))
        assert sgn_b == -1.0
        assert bad[1] < 0.0 < good[1]

    def test_q0_target_is_positive_and_identical(self, pair):
        """q0_target = q0_anchor * j_achieved(0)/j_requested(0): the anchor's
        q0 and axis current are positive-frame, so the source's requested
        axis current must be too (it came out negative before the fix)."""
        from bouquet.utils import unrenormalise_q0

        ref, rev = pair
        q0_anchor, j_ach0 = 0.98, 1.05 * float(ref.j_phi[0])
        q = [unrenormalise_q0(q0_anchor, j_ach0, float(bl.j_phi[0]))
             for bl in (ref, rev)]
        assert q[0] == q[1] > 0.0
        # witness: the dd-frame axis current gave a negative target
        assert unrenormalise_q0(q0_anchor, j_ach0, -float(ref.j_phi[0])) < 0.0

    def test_fuse_total_error_and_peak_ratio_are_identical(self, pair):
        """``fuse_total_err_pct`` (|lin + c| vs Ip) and
        ``swb_over_fuse_jBS_peak`` (SWB peak / max(FUSE peak, 1)) read the
        source total and bootstrap: identical, and the ratio is O(1)."""
        ref, rev = pair
        swb = np.asarray(ref.j_BS)
        res = []
        for bl in (ref, rev):
            psi = np.asarray(bl.psi_N, float)
            w = 1.0 + 0.5 * psi
            tot = float(trapezoid(w * np.asarray(bl.j_phi), psi)) + 0.03 * bl.Ip_target
            err = 100.0 * (abs(tot) - bl.Ip_target) / bl.Ip_target
            ratio = swb.max() / max(np.asarray(bl.j_BS).max(), 1.0)
            res.append((err, ratio))
        assert res[0] == res[1]
        assert res[0][1] == pytest.approx(1.0)
        # witness: the dd-frame bootstrap peak is <= 0 and floored at 1 A/m^2
        assert swb.max() / max((-np.asarray(ref.j_BS)).max(), 1.0) > 1e3

    def test_floor_j_BS_and_inductive_floor_are_identical(self, pair):
        """The ``floor_j_BS`` clip (>= 0) and ``floor_inductive_split``
        (j_inductive >= 0) both assume the positive frame.  On the reader's
        output they act identically for a mirrored source -- before the fix
        the clip would have ZEROED a dd-frame bootstrap and the inductive floor
        would have moved the whole (negative) inductive into j_BS."""
        from bouquet.baseline import floor_inductive_split

        ref, rev = pair
        for fn in (lambda bl: np.clip(np.asarray(bl.j_BS), 0.0, None),
                   lambda bl: floor_inductive_split(bl.j_inductive, bl.j_BS)[0],
                   lambda bl: floor_inductive_split(bl.j_inductive, bl.j_BS)[1]):
            assert np.array_equal(fn(ref), fn(rev))
        # witness: on the dd-frame arrays the clip empties the bootstrap
        assert not np.any(np.clip(-np.asarray(ref.j_BS), 0.0, None))


# ---------------------------------------------------------------------------
#  4. what bouquet shows and stores for the source
# ---------------------------------------------------------------------------
class TestRecordsAndOverlays:
    def test_input_overlay_is_in_the_same_frame(self, tmp_path):
        """plot_input_vs_recon overlays the dd's equilibrium j_tor on the
        solved (positive-frame) profile: identical for every orientation."""
        from bouquet.plotting import _imas_input_profiles

        dd = _example()
        ref = _imas_input_profiles(ImasSource(ids_path=_write(tmp_path, dd, "r.json"),
                                              time=mdd.EXAMPLE_TIME))
        for orient in mdd.ORIENTATIONS:
            got = _imas_input_profiles(ImasSource(
                ids_path=_write(tmp_path, mdd.mirror_dd(dd, *orient), "m.json"),
                time=mdd.EXAMPLE_TIME))
            assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
            assert np.array_equal(ref[3], got[3]), mdd.tag(*orient)
            assert np.array_equal(np.abs(ref[2]), np.abs(got[2]))   # |q|

    def test_archive_stamp(self, tmp_path):
        import h5py
        from bouquet.utils import CURRENT_FRAME, stamp_source_orientation

        p = str(tmp_path / "a.h5")
        with h5py.File(p, "w") as hf:
            hf.create_group("scan/0/_baseline")
        stamp_source_orientation(p, scan_key=0, current_sign=-1.0, b0_sign=-1.0)
        with h5py.File(p, "r") as hf:
            a = hf["scan/0/_baseline"].attrs
            assert a["source_current_sign"] == -1.0
            assert a["source_b0_sign"] == -1.0
            assert a["current_frame"] == CURRENT_FRAME
        # no baseline group -> nothing written, nothing raised
        q = str(tmp_path / "b.h5")
        with h5py.File(q, "w") as hf:
            hf.create_group("scan/0")
        stamp_source_orientation(q, scan_key=0, current_sign=-1.0)
        with h5py.File(q, "r") as hf:
            assert "_baseline" not in hf["scan/0"]
