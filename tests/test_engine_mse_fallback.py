"""The unified engine's MSE stage when it fails, and the chord chi^2 of the
delivered fit.

* Any failure inside the stage (or in the delivery of its fit) with
  ``structured_mse_required=False`` restores the converged reconstruction
  without MSE, delivers it (bit for bit the run without the MSE row) and
  flags ``mse_stage_failed``; with ``required=True`` it raises.  Through
  ``Bouquet.prepare_baseline`` the baseline is then delivered, not ``None``
  (the code before the fallback set it to ``None`` and re-raised: the
  mutants below).
* The delivered fit records chi^2 / N against the raw (E_r-corrected)
  chords with the stage's own weights and the chi^2 of the reconstruction
  without MSE, flagging ``mse_worse_than_without`` and
  ``mse_chi2_per_chord_high`` (``GenerationConfig.mse_chi2n_flag``, default
  10.0, owner-approved 2026-10-07; a flag only); under
  ``jbs_loop_on_fail="flag"`` a non-converged fit carries
  ``mse_converged=False`` and the same records.

Synthetic inputs only; no solver, no device data.
"""
import numpy as np
import pytest

import _engine_toy as T
from bouquet.engine import UnifiedEngine
from test_engine_mse_refresh import _mse_data, _quiet, _run


class OffsetAtDeliveryToy(T.ToyGS):
    """The toy whose DELIVERY measurement reads tan(gamma) shifted by
    ``shift_sigma`` sigma_eff on every chord (*sigma*: the chords'
    sigma_eff): the delivered fit is worse than the state the stage ended
    on (and than the reconstruction without MSE)."""

    shift_sigma = 40.0
    sigma = None

    def measure(self, want_chords=False, final=False):
        out = super().measure(want_chords=want_chords, final=final)
        if final and "B_chords" in out:
            B = np.asarray(out["B_chords"], dtype=float).copy()
            B[:, 2] = B[:, 2] + self.shift_sigma * self.sigma * B[:, 1]
            out["B_chords"] = B
        return out


# ---------------------------------------------------------------------------
#  B. the fallback to the reconstruction without MSE
# ---------------------------------------------------------------------------
class FailingToy(T.ToyGS):
    """The toy whose solve number *fail_at* raises (a failed GS solve)."""

    fail_at = None

    def solve(self, request, n_passes=1):
        if self.fail_at is not None and self.n_solves + 1 >= self.fail_at \
                and self.n_solves < self.fail_at:
            self.n_solves += 1
            raise RuntimeError("injected solve failure")
        super().solve(request, n_passes=n_passes)


def _no_mse_reference(ch):
    eng, res, rec = _run(T.ToyGS(chords=ch), rows=("Ip", "l_i"))
    assert res["converged"]
    return rec


@pytest.mark.parametrize("where", ["fd", "mse_pass"])
def test_a_failed_mse_stage_delivers_the_reconstruction_without_mse(where):
    md, ch = _mse_data()
    ref = _no_mse_reference(ch)
    n_loop = ref["solves"]["anchor"] + ref["solves"]["loop"]
    b = FailingToy(chords=ch)
    # inside the FD Jacobian, or on the second pass of the MSE loop
    b.fail_at = n_loop + (4 if where == "fd" else 1 + 8 + 2)
    eng, res, rec = _run(b, md=md, ch=ch)
    assert rec["mse"]["stage_failed"] and not rec["mse"]["applied"]
    assert rec["mse"]["failure"]["message"] == "injected solve failure"
    assert "mse_stage_failed" in rec["mse"]["flags"]
    assert any(f.startswith("MSE: mse_stage_failed") and "NOT applied" in f
               for f in rec["flags"])
    # the delivered state IS the reconstruction without the MSE row
    assert res["converged"] and rec["delivered"]["ok"]
    assert "mse" not in rec["delivered"]["checks"]
    np.testing.assert_array_equal(rec["state"]["request"],
                                  ref["state"]["request"])
    assert rec["state"]["x"] == ref["state"]["x"]
    assert rec["delivered"]["checks"]["l_i"]["delivered"] == \
        ref["delivered"]["checks"]["l_i"]["delivered"]
    assert eng.state.mse_J is None
    ph = rec["phases"][-1]
    assert ph["name"] == "mse" and ph["jacobian"]["failed"]
    s = rec["solves"]
    assert s["total"] == sum(v for k, v in s.items() if k != "total"), s
    assert "mse_fd" not in s and s["mse_failed"] >= 1


def _ceiling_after_real_passes(monkeypatch):
    """The MSE loop runs its real passes (Broyden updates included) and
    then raises ``JBSNotConverged`` with its record, as a pass ceiling does
    under ``jbs_loop_on_fail="raise"``: the route on which the failed
    phase's Jacobian record used to lack ``n_free``."""
    from bouquet.jbs_loop import JBSNotConverged
    real = UnifiedEngine._mse_loop

    def _loop(self, *a, **k):
        out = real(self, *a, **k)
        raise JBSNotConverged("injected: pass ceiling reached",
                              dict(out["record"], converged=False))

    monkeypatch.setattr(UnifiedEngine, "_mse_loop", _loop)


@pytest.mark.parametrize("scheme", ["fd_broyden", "fd_chord"])
def test_a_failed_mse_stage_keeps_the_jacobian_record_it_took(monkeypatch,
                                                              scheme):
    """A stage that fails in its passes, AFTER the FD Jacobian was taken,
    keeps that Jacobian's record on its failed phase with every key a
    delivered phase's carries (``n_free``, the FD's ``n_solves``, the
    scheme, the Broyden updates, the refresh block) beside the failure; the
    solves spent are ``n_solves_spent`` = ``solves["mse_failed"]``.  Before
    the fix the record was ``{applied, failed, where, reason, n_solves}``
    and reading ``n_free`` raised ``KeyError``."""
    md, ch = _mse_data()
    _ceiling_after_real_passes(monkeypatch)
    eng, res, rec = _run(T.ToyGS(chords=ch), md=md, ch=ch,
                         engine_mse_jacobian=scheme)
    assert rec["mse"]["stage_failed"] and res["converged"]
    assert "pass ceiling" in rec["mse"]["failure"]["message"]
    ph = rec["phases"][1]
    assert ph["name"] == "mse" and ph is rec["phases"][-1]
    fd = ph["jacobian"]
    # the Jacobian: one base solve + one per free coefficient (2K = 8)
    assert fd["n_free"] == 8 and fd["n_solves"] == 1 + 8
    assert fd["failed"] and not fd["applied"] and fd["jacobian_taken"]
    assert fd["n_pass_solves"] >= 1
    assert fd["n_solves_spent"] == rec["solves"]["mse_failed"] \
        == fd["n_solves"] + fd["n_pass_solves"] + fd["refresh"]["n_solves"]
    if scheme == "fd_broyden":
        assert fd["n_broyden_updates"] >= 1
    else:
        assert fd["n_broyden_updates"] == 0
    assert ("Broyden" in fd["scheme"]) == (scheme == "fd_broyden")
    assert np.shape(fd["J_initial"]) == (int(ch["n_active"]), 8)
    assert fd["refresh"]["rounds"] == [] and not fd["refresh"][
        "fresh_J_stationary"]
    # the record carried by the exception is the failed phase's record
    assert ph["record"]["converged"] is False


def test_a_stage_that_fails_before_its_jacobian_still_records_n_free():
    """Inside the FD itself (no Jacobian taken): ``n_free`` and the solves
    spent, ``jacobian_taken`` False."""
    md, ch = _mse_data()
    ref = _no_mse_reference(ch)
    b = FailingToy(chords=ch)
    b.fail_at = ref["solves"]["anchor"] + ref["solves"]["loop"] + 4
    eng, res, rec = _run(b, md=md, ch=ch)
    fd = rec["phases"][-1]["jacobian"]
    assert fd["n_free"] == 8
    assert fd["failed"] and not fd["jacobian_taken"]
    assert fd["n_solves"] == fd["n_solves_spent"] \
        == rec["solves"]["mse_failed"]


def test_a_failed_mse_stage_with_mse_required_raises():
    md, ch = _mse_data()
    ref = _no_mse_reference(ch)
    b = FailingToy(chords=ch)
    b.fail_at = ref["solves"]["anchor"] + ref["solves"]["loop"] + 4
    with pytest.raises(RuntimeError, match="injected solve failure"):
        _run(b, md=md, ch=ch, required=True)


def test_mutant_without_the_fallback_raises(monkeypatch):
    """The code before the fallback: the stage's exception propagated."""
    def _reraise(self, saved, exc, phases, *, where):
        raise exc

    monkeypatch.setattr(UnifiedEngine, "_mse_stage_failed", _reraise)
    md, ch = _mse_data()
    ref = _no_mse_reference(ch)
    b = FailingToy(chords=ch)
    b.fail_at = ref["solves"]["anchor"] + ref["solves"]["loop"] + 4
    with pytest.raises(RuntimeError, match="injected solve failure"):
        _run(b, md=md, ch=ch)


def _toy_md_for_gfile():
    """The toy's chord block with its orientation left to the source."""
    md, _ch = _mse_data()
    return {k: v for k, v in md.items() if k not in ("ip_sign", "bt_sign")}


@pytest.mark.parametrize("required", [False, True])
def test_prepare_baseline_keeps_the_no_mse_baseline_when_not_required(
        monkeypatch, required):
    """Bouquet level: an MSE stage that fails leaves a baseline (flagged)
    when MSE is not required -- the code before the fallback left
    ``bq.baseline = None`` and raised -- and raises when it is."""
    import bouquet as bq
    import bouquet.engine as be
    import test_engine_wiring as TW

    def _backend(mygs, contract, **kw):
        return T.ToyGS(psi=contract.psi_N, Ip=contract.Ip,
                       chords=kw.get("chords"))

    def _boom(self, *a, **k):
        raise RuntimeError("injected FD failure")

    monkeypatch.setattr(be, "TokaMakerBackend", _backend)
    monkeypatch.setattr(be, "_lcfs_deviation_mm",
                        lambda mygs, pts: (2.5, 7.0))
    monkeypatch.setattr(UnifiedEngine, "_mse_fd", _boom)
    b = bq.Bouquet.from_geqdsk(TW._GEQ, profiles=TW._PF, mesh=TW._MESH,
                               n_draws=1, reconstruction_engine="unified")
    g = b.config.generation
    g.engine_rows = ["Ip", "mse"]
    g.mse_data = _toy_md_for_gfile()
    g.structured_mse_required = required
    b.mygs = TW._FakeGS()
    if required:
        with pytest.raises(RuntimeError, match="injected FD failure"):
            _quiet(b.prepare_baseline)
        assert b.baseline is None
        return
    bl = _quiet(b.prepare_baseline)
    assert b.baseline is bl and bl is not None
    assert bl.engine["mse"]["stage_failed"]
    m = bl.reconstruction_metrics
    assert m["closure_limited"]
    assert any("mse_stage_failed" in r for r in m["closure_limited_reasons"])


# ---------------------------------------------------------------------------
#  B. chi^2 / N and the two flags
# ---------------------------------------------------------------------------
def test_chi2_per_chord_is_recorded_and_flagged_above_the_threshold():
    md, ch = _mse_data(alt=6.0)          # data no profile fits: chi2/N > 10
    eng, res, rec = _run(T.ToyGS(chords=ch), md=md, ch=ch)
    assert res["converged"]
    sm, c = rec["mse"], rec["delivered"]["checks"]["mse"]
    n = int(ch["n_active"])
    assert sm["n_chords"] == c["n_chords"] == n
    assert sm["chi2_per_chord"] == pytest.approx(c["chi2"] / n, rel=0,
                                                 abs=0)
    assert sm["chi2_per_chord"] > 10.0 and sm["chi2n_flag"] == 10.0
    assert sm["chi2_per_chord_high"] and c["chi2_per_chord_high"]
    assert "mse_chi2_per_chord_high" in sm["flags"]
    assert any("mse_chi2_per_chord_high" in f and "never acceptance" in f
               for f in rec["flags"])
    # the pre-MSE chi2: the no-MSE reconstruction's, against the same chords
    assert sm["chi2_per_chord_pre_mse"] > sm["chi2_per_chord"]
    assert not sm["worse_than_without"]
    # a flag only: the result is delivered and converged either way, and
    # the threshold is read from the config
    eng2, res2, rec2 = _run(T.ToyGS(chords=ch), md=md, ch=ch,
                            mse_chi2n_flag=1.0e6)
    assert res2["converged"] and not rec2["mse"]["chi2_per_chord_high"]
    np.testing.assert_array_equal(rec2["state"]["request"],
                                  rec["state"]["request"])


def test_a_fit_that_lands_within_the_threshold_is_not_flagged():
    md, ch = _mse_data()
    eng, res, rec = _run(T.ToyGS(chords=ch), md=md, ch=ch)
    sm = rec["mse"]
    assert sm["chi2_per_chord"] <= 10.0 and not sm["chi2_per_chord_high"]
    assert not sm["worse_than_without"] and sm["flags"] == []


def test_a_delivered_fit_worse_than_without_mse_is_flagged_and_unconverged():
    """Under jbs_loop_on_fail='flag' the delivery whose chords read worse
    than the no-MSE reconstruction is delivered, flagged
    mse_worse_than_without, with mse_converged=False and the chi2 records;
    under 'raise' the failed delivery falls back to the no-MSE state."""
    md, ch = _mse_data()
    b = OffsetAtDeliveryToy(chords=ch)
    b.sigma = np.asarray(ch["sigma_eff"], dtype=float)
    eng, res, rec = _run(b, md=md, ch=ch, jbs_loop_on_fail="flag")
    sm = rec["mse"]
    assert sm["applied"] and sm["mse_converged"] is False
    assert not res["converged"]
    assert sm["worse_than_without"] and "mse_worse_than_without" in sm[
        "flags"]
    assert sm["chi2"] > sm["chi2_pre_mse"]
    assert any("mse_worse_than_without" in f for f in rec["flags"])
    for k in ("chi2_per_chord", "chi2_per_chord_pre_mse", "chi2n_flag",
              "chi2_per_chord_high"):
        assert k in sm
    # the same delivery under 'raise': the fit's delivery failed, so the
    # reconstruction without MSE is delivered instead (not required)
    b2 = OffsetAtDeliveryToy(chords=ch)
    b2.sigma = b.sigma
    eng2, res2, rec2 = _run(b2, md=md, ch=ch)
    assert rec2["mse"]["stage_failed"]
    assert rec2["mse"]["failure"]["where"] == "delivery"
    s = rec2["solves"]
    assert s["total"] == sum(v for k, v in s.items() if k != "total"), s
    assert "mse" not in rec2["delivered"]["checks"] and res2["converged"]


def test_the_threshold_is_validated_and_engine_only():
    from bouquet.config import GenerationConfig
    from bouquet.engine import engine_settings, validate_engine_settings
    assert GenerationConfig().mse_chi2n_flag == 10.0
    assert engine_settings(GenerationConfig())["mse_chi2n_flag"] == 10.0
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, "10"):
        with pytest.raises(ValueError, match="mse_chi2n_flag"):
            validate_engine_settings(GenerationConfig(mse_chi2n_flag=bad))
    validate_engine_settings(GenerationConfig(mse_chi2n_flag=25))
    with pytest.raises(ValueError, match="mse_chi2n_flag"):
        validate_engine_settings(GenerationConfig(
            reconstruction_engine="legacy", mse_chi2n_flag=5.0))
    validate_engine_settings(GenerationConfig(reconstruction_engine="legacy"))
