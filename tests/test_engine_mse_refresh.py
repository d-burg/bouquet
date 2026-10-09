"""The unified engine's MSE stage: the Jacobian refresh at convergence.

* At each MSE loop's convergence the chord Jacobian is RE-TAKEN by the same
  finite differences and the Gauss-Newton step it implies is judged by the
  stage's own criterion (``MSE_CHORD_OFFSET_TOL_SIGMA``); the refresh is
  recorded (``jacobian_refresh_rel_change``, ``refresh_step_norm``).  On a
  toy whose pitch angles respond NON-linearly to the current (its Jacobian
  changes along the path) the first refresh step exceeds the criterion, the
  loop continues with the refreshed Jacobian and the stage ends at the
  fresh-Jacobian stationary point; the mutant that stops at the old-Jacobian
  point leaves a fresh-Jacobian step above the criterion.

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import copy
import dataclasses
import io
import warnings

import numpy as np
import pytest

import _engine_toy as T
from bouquet.engine import UnifiedEngine, reconstruct
from bouquet.jbs_loop import MSE_CHORD_OFFSET_TOL_SIGMA, JBSNotConverged


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _li0():
    b = T.ToyGS()
    b.solve(T.ToyAdapter().read().anchor_request)
    return b.state["li"]


LI0 = _li0()


# ---------------------------------------------------------------------------
#  toys
# ---------------------------------------------------------------------------
_REF = {}


def _bz_ref(ch):
    """B_Z of the anchor equilibrium at the chords (the nonlinear toy's
    reference point)."""
    key = tuple(np.round(np.asarray(ch["R"], dtype=float), 12))
    if key not in _REF:
        b = T.ToyGS(chords=ch)
        b.solve(T.ToyAdapter().read().anchor_request)
        _REF[key] = T.ToyGS.field_at_chords(b, b.state)[0][:, 2].copy()
    return _REF[key]


class NonlinearToy(T.ToyGS):
    """The toy with a pitch-angle response quadratic in the enclosed
    current: ``B_Z -> B_Z (1 + gamma (B_Z / B_Z,anchor - 1))``, so
    ``d tan(gamma) / d x`` changes along the path from the anchor."""

    gamma = 1.0

    def field_at_chords(self, st):
        B, found = super().field_at_chords(st)
        B = B.copy()
        bz = B[:, 2]
        B[:, 2] = bz * (1.0 + self.gamma * (bz / _bz_ref(self.chords) - 1.0))
        return B, found


def _mse_data(toy_cls=T.ToyGS, amp=0.04, sig=0.01, n=8, alt=0.0):
    """Pitch angles of the toy (class *toy_cls*) solved with a mid-radius
    bump on the anchor request, E_r-corrected, ``ip_sign = bt_sign = +1``
    (the toy's own frame); *alt* adds an alternating offset of that many
    sigma (data no current profile fits)."""
    from bouquet.mse import mse_chords
    ch0 = T.toy_chords(n)
    bt = toy_cls(chords=ch0)
    req = T.ToyAdapter().read().anchor_request
    bt.solve(req * (1.0 + amp * np.exp(-0.5 * ((T.PSI - 0.45) / 0.15) ** 2)))
    B, _found = bt.field_at_chords(bt.state)
    tg = B[:, 2] / B[:, 1]
    s = np.abs(tg) * sig
    tg = tg + alt * s * (-1.0) ** np.arange(n)
    md = dict(R=ch0["R"], Z=ch0["Z"], tgamma=tg, sigma=s,
              weight=np.ones(n), A1=ch0["A1"], A2=ch0["A2"], A3=ch0["A3"],
              A4=ch0["A4"], er_corrected=True, ip_sign=1.0, bt_sign=1.0)
    return md, mse_chords(md, min_chords=4)


def _run(backend, *, rows=("Ip", "l_i", "mse"), required=False, md=None,
         ch=None, **gc):
    mse = (None if "mse" not in rows else
           dict(chords=ch, er_terms="toy", required=required))
    ad = T.ToyAdapter(li_target=LI0 * 1.01, mse=mse)
    ad.read()
    kw = dict(engine_rows=list(rows))
    if "mse" in rows:
        kw.update(mse_data=md, structured_mse_required=required)
    kw.update(gc)
    eng, res, rec = _quiet(reconstruct, ad, backend, T.settings(**kw),
                           label="toy")
    return eng, res, rec


def _mse_phase(rec):
    return [p for p in rec["phases"] if p["name"] == "mse"][-1]


# ---------------------------------------------------------------------------
#  A. the Jacobian refresh at convergence
# ---------------------------------------------------------------------------
def test_the_jacobian_refresh_is_recorded_and_the_record_names_the_method():
    md, ch = _mse_data()
    eng, res, rec = _run(T.ToyGS(chords=ch), md=md, ch=ch)
    assert res["converged"]
    fd = _mse_phase(rec)["jacobian"]
    rf = fd["refresh"]
    assert len(rf["rounds"]) >= 1 and rf["fresh_J_stationary"]
    last = rf["rounds"][-1]
    for k in ("jacobian_refresh_rel_change", "refresh_step_norm",
              "refresh_step_dtg_max_sigma", "refresh_step_rel_change",
              "old_J_step_dtg_max_sigma"):
        assert np.isfinite(last[k])
    # the toy's Jacobian does move between the no-MSE state and the fit
    assert last["jacobian_refresh_rel_change"] > 0.0
    assert last["refresh_step_dtg_max_sigma"] <= MSE_CHORD_OFFSET_TOL_SIGMA
    assert fd["jacobian_refresh_rel_change"] == \
        last["jacobian_refresh_rel_change"]
    assert fd["refresh_step_norm"] == last["refresh_step_norm"]
    # one base solve + one per free coefficient per refresh, counted apart
    assert rec["solves"]["mse_refresh"] == rf["n_solves"] \
        == len(rf["rounds"]) * (1 + fd["n_free"])
    assert rec["solves"]["mse_fd"] == 1 + fd["n_free"]
    # the fixed-Jacobian path says so (it used to say "Broyden-updated")
    assert "held fixed" in fd["scheme"] and "Broyden" not in fd["scheme"]
    assert "Broyden" not in rec["row_update"]
    assert "held fixed" in rec["row_update"]
    assert rec["mse"]["applied"] and rec["mse"]["mse_converged"]


def test_the_broyden_scheme_is_named_where_it_is_used():
    md, ch = _mse_data()
    eng, res, rec = _run(T.ToyGS(chords=ch), md=md, ch=ch,
                         engine_mse_jacobian="fd_broyden")
    assert "Broyden updates every pass" in _mse_phase(rec)["jacobian"][
        "scheme"]
    assert "Broyden updates every pass" in rec["row_update"]


def _nonlinear_case():
    md, ch = _mse_data(NonlinearToy)
    return md, ch


def test_a_jacobian_that_changes_converges_to_the_fresh_J_stationary_point():
    md, ch = _nonlinear_case()
    eng, res, rec = _run(NonlinearToy(chords=ch), md=md, ch=ch)
    assert res["converged"]
    rounds = _mse_phase(rec)["jacobian"]["refresh"]["rounds"]
    # the first refresh found the old-Jacobian point off the fresh one ...
    assert rounds[0]["refresh_step_dtg_max_sigma"] \
        > MSE_CHORD_OFFSET_TOL_SIGMA
    assert rounds[0]["continued"] and not rounds[0]["ok"]
    assert "previous_loop_record" in rounds[0]
    # ... and the stage ended where the fresh-Jacobian step is within it
    assert len(rounds) >= 2 and rounds[-1]["ok"]
    assert rounds[-1]["refresh_step_dtg_max_sigma"] \
        <= MSE_CHORD_OFFSET_TOL_SIGMA
    assert _mse_phase(rec)["jacobian"]["refresh"]["fresh_J_stationary"]


def test_mutant_without_the_continuation_stops_off_the_fresh_J_point(
        monkeypatch):
    """The mutant measures the fresh-Jacobian step but never continues
    (the stage then ends where the code before the refresh ended): the
    fresh-Jacobian step at its end exceeds the criterion."""
    orig = UnifiedEngine._refresh_jacobian

    def _no_continuation(self, out):
        keep = (self.b.snapshot(), copy.deepcopy(self.state),
                copy.deepcopy(self._mse_phase))
        r = orig(self, out)
        self.b.restore(keep[0])
        for f in dataclasses.fields(keep[1]):        # in place
            setattr(self.state, f.name, getattr(keep[1], f.name))
        self._mse_phase = keep[2]
        r["record"] = dict(r["record"], mutant_fresh_step_ok=r["ok"])
        r["ok"] = True
        return r

    monkeypatch.setattr(UnifiedEngine, "_refresh_jacobian",
                        _no_continuation)
    md, ch = _nonlinear_case()
    eng, res, rec = _run(NonlinearToy(chords=ch), md=md, ch=ch)
    rounds = _mse_phase(rec)["jacobian"]["refresh"]["rounds"]
    assert len(rounds) == 1
    assert not rounds[0]["mutant_fresh_step_ok"]
    assert rounds[0]["refresh_step_dtg_max_sigma"] \
        > MSE_CHORD_OFFSET_TOL_SIGMA


def test_a_fresh_J_step_that_never_settles_is_not_converged(monkeypatch):
    """A cap of refreshes reached with the fresh step still above the
    criterion: NOT converged -- raised under on_fail='raise' when MSE is
    required (a failed stage otherwise: the no-MSE fallback), delivered
    flagged (mse_converged=False) under 'flag'."""
    import bouquet.engine as be
    monkeypatch.setattr(be, "MSE_JACOBIAN_MAX_REFRESHES", 1)
    md, ch = _nonlinear_case()
    with pytest.raises(JBSNotConverged, match="fresh-Jacobian"):
        _run(NonlinearToy(chords=ch), md=md, ch=ch, required=True)
    # not required: the failed stage falls back to the reconstruction
    # without MSE (tests/test_engine_mse_fallback.py)
    eng, res, rec = _run(NonlinearToy(chords=ch), md=md, ch=ch)
    assert rec["mse"]["stage_failed"] and res["converged"]
    assert "fresh-Jacobian" in rec["mse"]["failure"]["message"]
    eng, res, rec = _run(NonlinearToy(chords=ch), md=md, ch=ch,
                         jbs_loop_on_fail="flag")
    assert not res["converged"]
    assert "fresh-Jacobian" in _mse_phase(rec)["record"]["stop_reason"]
    # the delivered non-converged fit carries mse_converged=False and the
    # chi2 records (tests/test_engine_mse_fallback.py)
    assert rec["mse"]["applied"] and rec["mse"]["mse_converged"] is False
    for k in ("chi2_per_chord", "chi2_per_chord_pre_mse",
              "worse_than_without", "chi2_per_chord_high"):
        assert k in rec["mse"]
