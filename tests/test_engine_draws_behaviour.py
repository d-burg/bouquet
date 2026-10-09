"""Engine draws: behaviour the fast suite did not execute (the 2026-10-04
review's surviving mutants and stand-ins that enforced what they asserted).

On the toy Grad-Shafranov stand-in (``tests/_engine_toy.py``), each test runs
the package's own code path and observes its effect:

* the draw starts its loop from ``lambda_BS*`` plus the KINETIC increment of
  Redl on the starting state (mutant D3);
* with ``engine_delivery_correction`` on, every pass's correction is the
  previous pass's own ``intended - achieved`` (mutant D7);
* ``engine_draws.post_homotopy`` itself -- its check, and its passes when the
  homotopy moved the state (before, the fast tests replaced it wholesale;
  mutant D9);
* the homotopy rollback restores the last good stage's STATE (a stand-in
  whose ``set_psi`` really restores, and whose failed solve leaves a
  corrupted state behind, as the solver's does; mutant H1 was caught only by
  the frozen-AST snapshot).

Synthetic inputs only; no solver.
"""
import copy

import numpy as np
import pytest

import _engine_toy as T
import test_engine_draws as TD
import test_engine_draw_refresh_and_cap as TC
from bouquet import engine_draws as ED

PSI = T.PSI


@pytest.fixture(scope="module")
def recon():
    return TD._recon()


def _inputs(ctx, b, seed, f=0.03):
    from bouquet.sampling import make_rng
    return ED.sample_draw_inputs(ctx, make_rng(seed), TD._unc(ctx, f=f),
                                 b.flux_integral, scale=1.0)


# ---------------------------------------------------------------------------
#  D3: the loop's starting bootstrap carries the kinetic Redl increment
# ---------------------------------------------------------------------------
def test_the_draw_starts_from_the_kinetic_increment_of_redl(recon,
                                                            monkeypatch):
    import bouquet.jbs_loop as L
    eng, res, rec, b = recon
    ctx = TD._ctx(eng, res)
    b = copy.deepcopy(b)
    inp = _inputs(ctx, b, 11)
    # expected, on the starting state (nothing is solved before the loop)
    probe = copy.deepcopy(b)
    probe.set_inputs(pressure=inp.pressure, kinetics=inp.kinetics)
    r_d = probe.redl(inp.kinetics)
    r_0 = probe.redl(ctx.c.kinetics)
    assert np.max(np.abs(r_d - r_0)) > 1e-3 * np.max(np.abs(ctx.lam))
    seen = []
    real = L.run_jbs_loop

    def spy(jbs0, *a, **k):
        seen.append(np.asarray(jbs0, dtype=float).copy())
        return real(jbs0, *a, **k)

    monkeypatch.setattr(L, "run_jbs_loop", spy)
    TD._quiet(ED.run_draw, ctx, b, inp)
    np.testing.assert_array_equal(seen[0], 1.0 * (ctx.lam + (r_d - r_0)))


# ---------------------------------------------------------------------------
#  D7: the draw's delivery correction is updated every pass
# ---------------------------------------------------------------------------
def test_the_draws_delivery_correction_follows_each_pass(monkeypatch):
    import bouquet.engine as E
    eng, res, rec, b = TD._recon(engine_delivery_correction=True)
    ctx = TD._ctx(eng, res)
    d0 = np.asarray(res["state"].delivery_correction, dtype=float)
    assert np.any(d0 != 0.0)
    dvecs = []
    real = E._delivery_stats

    def spy(req, A, geom):
        s, v = real(req, A, geom)
        dvecs.append(np.asarray(v, dtype=float).copy())
        return s, v

    monkeypatch.setattr(E, "_delivery_stats", spy)
    out = TD._quiet(ED.run_draw, ctx, copy.deepcopy(b), _inputs(ctx, b, 5))
    passes = out["passes"].passes
    upd = [(k, p["delivery_correction_max"]) for k, p in enumerate(passes)
           if "delivery_correction_max" in p]
    assert upd, "no pass updated the correction"
    for k, v in upd:
        assert v == float(np.max(np.abs(dvecs[k]))), (k, v)
    # and it moved off the reconstruction's correction
    assert any(v != float(np.max(np.abs(d0))) for _k, v in upd)


# ---------------------------------------------------------------------------
#  D9: engine_draws.post_homotopy's own check and passes
# ---------------------------------------------------------------------------
def _draw_then_move(recon, amp):
    eng, res, rec, b = recon
    ctx = TD._ctx(eng, res)
    b = copy.deepcopy(b)
    d = TD._quiet(ED.run_draw, ctx, b, _inputs(ctx, b, 3))
    if amp:
        # the "homotopy" moves the equilibrium: the same request solved
        # with a bent core (coils / boundary moved it)
        st = b.state
        b.solve(np.asarray(st["R"], dtype=float)
                * (1.0 + amp * np.exp(-0.5 * (PSI / 0.3) ** 2)))
    return ctx, b, d


def test_post_homotopy_keeps_an_unmoved_draw_without_passes(recon):
    ctx, b, d = _draw_then_move(recon, 0.0)
    n0 = b.n_solves
    rec, jbs, jb_tor, jphi = TD._quiet(ED.post_homotopy, ctx, b, d,
                                       ctx.loop)
    assert rec["accepted_without_passes"] is True and rec["check"]["ok"]
    assert "passes" not in rec and jb_tor is None and jphi is None
    assert b.n_solves == n0


def test_post_homotopy_runs_its_passes_when_the_homotopy_moved_the_draw(
        recon):
    from bouquet.engine import complete_geometry, conversion_factor
    from bouquet.jbs_loop import check_delivered
    ctx, b, d = _draw_then_move(recon, 0.06)
    s = ctx.loop
    n0 = b.n_solves
    rec, jbs, jb_tor, jphi = TD._quiet(ED.post_homotopy, ctx, b, d, s)
    assert rec["accepted_without_passes"] is False
    assert rec["check"]["ok"] is False
    assert "passes" in rec and rec["passes"]["converged"]
    assert 1 <= rec["passes"]["n_passes"] <= s["post_homotopy_passes"]
    assert b.n_solves > n0 and jphi is not None and jb_tor is not None
    # the delivered state now passes the check it failed
    m = b.measure()
    g = complete_geometry(m["geom"])
    chk = check_delivered(np.asarray(m["redl"], dtype=float), jbs,
                          g["w_lin"] * conversion_factor(g), ctx.psi,
                          float(ctx.c.Ip), s)
    assert chk["ok"], chk


def test_post_homotopy_rejects_what_its_passes_cannot_recover(recon):
    from bouquet.jbs_loop import JBSNotConverged
    ctx, b, d = _draw_then_move(recon, 0.06)
    s = dict(ctx.loop, post_homotopy_passes=1)
    with pytest.raises(JBSNotConverged):
        TD._quiet(ED.post_homotopy, ctx, b, d, s)


# ---------------------------------------------------------------------------
#  H1: the rollback restores the last good stage's state
# ---------------------------------------------------------------------------
def test_the_rollback_restores_the_last_good_stage(tmp_path, monkeypatch):
    """Pass 2 fails leaving a corrupted state behind (as the solver does);
    the rollback must put back the pass-1 STATE (set_psi) before its
    re-solve.  The archived draw is therefore the pass-1 state re-solved
    once (set_psi alone leaves the flux-surface averages stale, so the
    rollback solves on purpose): the reference is that same operation run
    on an untouched copy of the pass-1 state, compared exactly.  Comparing
    against the pass-1 l_i itself is one solve short: the toy's fixed-point
    loop moves by a few ulp on a re-solve from its own converged state, and
    which way depends on the CPU's exp/pow code paths.  The stand-in's
    get_psi / set_psi really save and restore its state."""
    from _engine_fake_gs import FakeTokaMaker
    snaps = {}
    box = dict(n_h=0, li_pass1=None, li_resolve1=None)

    def get_psi(self, normalized=True):
        tok = len(snaps) + 1
        snaps[tok] = self.toy.snapshot()
        out = np.zeros(16)
        out[0] = tok
        return out

    def set_psi(self, psi, update_bounds=False):
        tok = int(np.asarray(psi)[0])
        if tok in snaps:
            self.toy.restore(snaps[tok])

    monkeypatch.setattr(FakeTokaMaker, "get_psi", get_psi)
    monkeypatch.setattr(FakeTokaMaker, "set_psi", set_psi)

    class _Fail(dict):
        """``n in fail`` is asked once per homotopy solve (the harness's
        counter): on pass 1 record the state's l_i after it; on pass 2
        corrupt the state, then fail."""

        def __contains__(self, n):
            box["n_h"] = n
            return dict.__contains__(self, n)

    real_solve = FakeTokaMaker.solve

    def solve(self, *a, **k):
        r = real_solve(self, *a, **k)
        if box["n_h"] == 1 and box["li_pass1"] is None:
            box["li_pass1"] = float(self.toy.state["li"])
            # the rollback's own operation (restore, then one solve) on an
            # untouched copy of the pass-1 state
            probe = copy.deepcopy(self.toy)
            probe.solve(np.asarray(probe.state["R"]))
            box["li_resolve1"] = float(probe.state["li"])
        return r

    monkeypatch.setattr(FakeTokaMaker, "solve", solve)
    err = "Error in solve: Non-finite value (NaN/Inf) in solution"
    fail = _Fail({2: err})
    # corrupt before raising: wrap the harness's own raise
    orig_contains = _Fail.__contains__

    def contains(self, n):
        hit = orig_contains(self, n)
        if hit:
            fake = box["fake"]
            # a corrupted SHAPE (a uniform factor would be undone by the
            # imposed Ip)
            fake.toy.solve(np.asarray(fake.toy.state["R"]) * (
                1.0 + 0.3 * np.exp(-0.5 * (PSI / 0.3) ** 2)))
        return hit

    _Fail.__contains__ = contains
    real_init = FakeTokaMaker.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        box["fake"] = self

    monkeypatch.setattr(FakeTokaMaker, "__init__", init)
    diags, rej, G, fake, seen = TC._generate_capped(
        tmp_path, monkeypatch, maxits=40, fail=fail)
    assert rej == [] and len(diags) == 1
    assert diags[0]["homotopy_pass"] == 0          # rolled back to pass 1
    assert len(seen) == 3                          # pass 1, pass 2, rollback
    arch = diags[0]["engine"]["archived"]
    assert box["li_pass1"] is not None
    # the archive holds the restored-and-re-solved pass-1 state, exactly
    assert arch["l_i_3"] == box["li_resolve1"]
    # and the corrupted pass-2 shape did not leak through: l_i moved back
    # to within re-solve noise of pass 1 (the mutant without set_psi is off
    # by ~6e-2 relative)
    assert abs(arch["l_i_3"] - box["li_pass1"]) < 1e-10 * box["li_pass1"]


# ---------------------------------------------------------------------------
#  the archive stage's failed get_stats (``eq_stats_iter = None``, a
#  statement the frozen-AST normaliser removes): the draw is archived with
#  l_i NaN and an edge-pressure record WITHOUT frames -- never with frames
#  computed from stale statistics, never a crash
# ---------------------------------------------------------------------------
def test_a_failed_archive_get_stats_archives_no_frames(tmp_path, monkeypatch):
    import h5py
    from _engine_fake_gs import FakeTokaMaker
    from bouquet import edge_pressure as EP
    box = dict(G=None)
    real = FakeTokaMaker.get_stats

    def get_stats(self, *a, **k):
        G = box["G"]
        if G is not None and G._cur is not None and \
                G._cur["clock"].cur in ("archive", "filters"):
            raise ValueError("degenerate equilibrium (stand-in)")
        return real(self, *a, **k)

    monkeypatch.setattr(FakeTokaMaker, "get_stats", get_stats)
    real_init = ED.GenerateEngineDraws.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        box["G"] = self

    monkeypatch.setattr(ED.GenerateEngineDraws, "__init__", init)
    diags, rej, G, fake, seen = TC._generate_capped(
        tmp_path, monkeypatch, maxits=None, fail={})
    assert rej == [] and len(diags) == 1
    h = str(tmp_path / "e2e")
    rec = EP.load_record(h, count=0)
    assert rec is not None and rec["frames"] is None
    with h5py.File(h + ".h5", "r") as hf:
        root = hf["scan/0"] if "scan" in hf else hf
        assert np.isnan(float(root["0"].attrs["l_i(3)"]))
        assert np.isnan(float(root["0"].attrs["l_i(1)"]))
