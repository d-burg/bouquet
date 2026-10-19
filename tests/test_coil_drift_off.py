"""Coil handling off (``generate_bouquet(coil_drift=None)``), issue #95 --
the fast half (a toy solver; no OFT).

With coil handling off the coil-handling block of ``generate_bouquet`` never
runs, and the per-draw loop read names only that block bound (the warm-start
snapshot, the SKIP_* switches, the boundary diagnostic): every run raised
``UnboundLocalError`` before archiving a draw.  Here the run must archive its
draws, and each archived draw must say what it is: no homotopy ran
(``homotopy_pass`` -1, both limits 0) and the coil drifts were not measured
(NaN), so the legacy drift verdict ``coil_drift_in_spec`` is False.  The
archived ``in_spec`` is the in-spec rule's verdict with the CONFIGURED coil
filter: under the legacy +/-drift filter (what this toy run resolves to) a
NaN drift fails it, so ``in_spec`` is False here; under the default chi2
filter a coil-off draw's coil currents are judged like any other draw's
(see the live half, ``tests/test_coil_drift_off_solver.py``).
"""
import math

import h5py
import numpy as np

import test_engine_draws as TED


def _generate_coil_off(tmp_path, monkeypatch, **kw):
    import bouquet.TokaMaker_interface as TI
    real = TI.generate_bouquet

    def coil_off(mygs, *a, **k):
        k["coil_drift"] = None
        return real(mygs, *a, **k)
    monkeypatch.setattr(TI, "generate_bouquet", coil_off)
    try:
        return TED._generate(tmp_path / "coil_off", monkeypatch, **kw)
    finally:
        monkeypatch.setattr(TI, "generate_bouquet", real)


def test_coil_off_archives_every_draw(tmp_path, monkeypatch):
    diags, rej, h, _ = _generate_coil_off(tmp_path, monkeypatch, n=2)
    assert len(diags) == 2 and rej == []
    with h5py.File(h + ".h5", "r") as hf:
        assert {"0", "1", "_baseline"} <= set(hf.keys())
        assert int(hf.attrs["n_stored"]) == 2
        for k in ("0", "1"):
            assert "eqdsk" in hf[k] and "coil_currents" in hf[k]


def test_coil_off_draw_stamps(tmp_path, monkeypatch):
    diags, _, h, _ = _generate_coil_off(tmp_path, monkeypatch, n=2)
    with h5py.File(h + ".h5", "r") as hf:
        for k in ("0", "1"):
            a = hf[k].attrs
            assert int(a["homotopy_pass"]) == -1
            assert float(a["homotopy_F_lim"]) == 0.0
            assert float(a["homotopy_VSC_lim"]) == 0.0
            assert math.isnan(float(a["max_F_drift_pct"]))
            assert math.isnan(float(a["max_VSC_drift_pct"]))
            # unmeasured drifts fail the legacy drift verdict ...
            assert not bool(a["coil_drift_in_spec"])
            # ... and, under the legacy coil filter this run uses, the rule
            assert not bool(a["in_spec"])
            assert str(a["in_spec_source"]) == "inspec_rule"
    for d in diags:
        assert d["in_spec_coil_filter"] == "legacy"
        assert np.isnan(d["max_F_drift_pct"]) and np.isnan(d["max_VSC_drift_pct"])
        assert not bool(d["in_spec"])


def test_coil_on_is_unchanged_by_the_fix(tmp_path, monkeypatch):
    """The default path (coil_drift=0.01) still measures and stamps its
    drifts (the names bound before the branch are overwritten in it)."""
    diags, rej, h, _ = TED._generate(tmp_path / "coil_on", monkeypatch, n=2)
    assert len(diags) == 2 and rej == []
    with h5py.File(h + ".h5", "r") as hf:
        for k in ("0", "1"):
            a = hf[k].attrs
            assert int(a["homotopy_pass"]) >= 0
            assert np.isfinite(float(a["max_F_drift_pct"]))
