"""The solver's uniform P' rescale ``p_scale`` is recorded on every
delivered state.

TokaMaker builds the pressure by integrating the handed ``P'`` inward from
the boundary from zero and rescales ``P'`` uniformly so the axis value is
``pax``; that factor (OpenFUSIONToolkit's ``p_scale``) is ~1.02-1.03 on a
pedestal with the edge ``P'`` pin -- the size of the separatrix correction
-- and was recorded nowhere.  Now (``edge_pressure.solver_p_scale``):

* the engine's final measurement carries it, so the engine record's
  ``edge_pressure`` block and the Baseline's ``delivered_state`` do;
* every archived edge-pressure record (``edge_pressure_json``) has a
  ``p_scale`` key: each draw's own (read on the draw's solved state), the
  baseline's from its reconstruction's record (``None`` when not known);
* the engine draw record carries it for the loop's delivered state and the
  archived one.

Solver-free (stand-ins report a fixed ``p_scale``); synthetic inputs only.
"""
import contextlib
import io
import types
import warnings

import numpy as np
import pytest

import _engine_toy as T
import test_engine_draws as TD
from _engine_fake_gs import FakeTokaMaker
from bouquet import edge_pressure as EP
from bouquet import engine_draws as ED

P_SCALE = 1.0273


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def test_solver_p_scale_reads_the_solver_or_none():
    assert EP.solver_p_scale(types.SimpleNamespace(p_scale=1.02)) == 1.02
    assert EP.solver_p_scale(types.SimpleNamespace()) is None
    assert EP.solver_p_scale(types.SimpleNamespace(p_scale=float("nan"))) \
        is None
    assert EP.solver_p_scale(types.SimpleNamespace(p_scale="x")) is None


def test_every_archive_record_has_the_key():
    p = np.linspace(5e4, 2e3, 11)
    assert EP.archive_record(None, p)["p_scale"] is None
    assert EP.archive_record(None, p, p_scale=1.03)["p_scale"] == 1.03
    assert "P_SCALE_DEFINITION" in dir(EP) and "p_scale" in \
        EP.P_SCALE_DEFINITION


@pytest.fixture
def toy_reports_p_scale(monkeypatch):
    """Every toy final measurement reports the solver's p_scale, as
    TokaMakerBackend.measure(final=True) does."""
    orig = T.ToyGS.measure

    def _measure(self, want_chords=False, final=False):
        out = orig(self, want_chords=want_chords, final=final)
        if final:
            out["p_scale"] = P_SCALE
        return out

    monkeypatch.setattr(T.ToyGS, "measure", _measure)


def test_the_engine_reconstruction_records_its_p_scale(toy_reports_p_scale):
    from bouquet.engine import _delivered_state
    eng, res, rec, _toy = TD._recon()
    assert rec["edge_pressure"]["p_scale"] == P_SCALE
    ds, _off = _delivered_state(eng, res, rec, "reconstruction")
    assert ds["edge_pressure"]["p_scale"] == P_SCALE


def test_the_engine_record_says_none_when_the_backend_reports_none():
    eng, res, rec, _toy = TD._recon()
    assert "p_scale" in rec["edge_pressure"]
    assert rec["edge_pressure"]["p_scale"] is None


class _PFake(FakeTokaMaker):
    p_scale = P_SCALE


def test_every_archived_draw_and_the_baseline_carry_p_scale(
        tmp_path, monkeypatch, toy_reports_p_scale):
    from bouquet.engine import _delivered_state
    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.utils import initialize_equilibrium_database
    eng, res, rec, toy = TD._recon()
    ds, _off = _delivered_state(eng, res, rec, "reconstruction")
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)
    fake = _PFake(toy)
    ctx = TD._ctx(eng, res)
    unc = TD._unc(ctx)
    G = ED.GenerateEngineDraws(ctx, unc=unc, psi_pad=T.PAD)
    h = str(tmp_path / "arch")
    initialize_equilibrium_database(h)
    k = ctx.native
    diags = _q(
        generate_bouquet, fake, T.PSI, 2, h, ctx.request, k["ne"], k["te"],
        k["ni"], k["ti"], unc["sigma_ne"], unc["sigma_te"], unc["sigma_ni"],
        unc["sigma_ti"], unc["sigma_jphi"], 0.3, 0.3, 0.25,
        float(eng.c.Ip), ctx.ref["l_i"], eng.c.kinetics["zeff"],
        input_jinductive=0.5 * ctx.request,
        baseline_j_BS=0.1 * ctx.request, l_i_tolerance=0.05,
        psi_pad=T.PAD, constrain_sawteeth=False, isolate_edge_jBS=False,
        jBS_scale_range=(0.99, 1.01), coil_drift=0.01,
        homotopy_passes=[(0.05, 0.1), (0.01, 0.01)], seed=12345,
        capture_live_eq=False, store_achieved_jphi=True,
        jbs_loop=G.loop_settings, rejection_log=[], draw_method=G,
        coil_filter="legacy", delivered_state=ds)
    assert len(diags) == 2
    # the baseline's: from its reconstruction's record
    assert EP.load_record(h)["p_scale"] == P_SCALE
    for i in range(2):
        # each draw's own, read on its solved state
        assert EP.load_record(h, count=i)["p_scale"] == P_SCALE
        er = ED.read_draw_engine(h, i)
        assert er["delivered"]["p_scale"] == P_SCALE
        assert er["archived"]["p_scale"] == P_SCALE
