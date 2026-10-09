"""The engine sigma=0 check under a CONFIGURED coil regularisation, and its
widened gate -- fast half (no GS solver; the toy stand-in of
``tests/_engine_fake_gs.py``).

* The engine reconstruction records the coil regularisation it solved under
  (``Bouquet._engine_run["coil_reg"]``: the term list ``_apply_coil_reg``
  installed -- configured measured-coil targets at their configured
  weights) and every engine draw's loop solves under exactly that list, the
  zero-perturbation draw included.  Before 2026-10-06 the draw installed the
  WEAK exploratory list (every weight 1.0, a configured #VSC term dropped,
  the VSC at 1e-2 toward zero), identical to the reconstruction's only when
  ``SolverConfig.coil_reg`` is empty -- which is every example and test.
* ``passed`` gates, besides r_j / r_I / |dl_i|, the q0 change at the row
  radius (``jbs_tol_q0``) at both stages and, on the archived state, the
  coil-current drift against the hard coil bound the route rejects at and
  the LCFS rms deviation against the in-spec boundary cut; every gated
  quantity is recorded with its bound.

Synthetic inputs only; no device data.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_engine_draws as TD  # noqa: E402
from bouquet import engine_draws as ED  # noqa: E402

toy_bouquet_solver = TD.toy_bouquet_solver
_quiet = TD._quiet

#: measured-target style terms: a flat W0 = 100 term, an inverse-variance
#: weighted one, and a configured #VSC constraint
SPEC = [{"coils": {"F1A": 1.0}, "target": 2.0e4, "weight": 100.0},
        {"coils": {"F2A": 1.0}, "target": -1.5e4, "weight": 37.5},
        {"coils": {"#VSC": 1.0}, "target": 0.0, "weight": 50.0}]


@pytest.fixture
def configured(tmp_path, toy_bouquet_solver, monkeypatch):
    """A toy Bouquet whose solver setup runs the real ``_apply_coil_reg``
    with :data:`SPEC` configured, recording every coil-reg install."""
    import bouquet.run as br
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.solver_state import enter_bounded_coil_mode

    def _setup(self):
        self.mygs = FakeTokaMaker(None)
        self.mygs.installs = []
        real = self.mygs.set_coil_reg

        def set_coil_reg(reg_terms=None, **kw):
            self.mygs.installs.append(list(reg_terms))
            return real(reg_terms=reg_terms, **kw)
        self.mygs.set_coil_reg = set_coil_reg
        self._apply_coil_reg(self.mygs)
        enter_bounded_coil_mode(self.mygs)
        return self

    monkeypatch.setattr(br.Bouquet, "setup_solver", _setup)
    b = TD._bq(tmp_path)
    b.config.solver.coil_reg = [dict(t) for t in SPEC]
    b.setup_solver()
    _quiet(b.prepare_baseline)
    # what each draw's loop actually solves under: the regularisation on
    # the solver when the draw samples its inputs (right after the install)
    seen = []
    real_sample = ED.sample_draw_inputs

    def sample(*a, **k):
        seen.append(list(b.mygs.reg))
        return real_sample(*a, **k)
    monkeypatch.setattr(ED, "sample_draw_inputs", sample)
    return b, seen


def test_the_reconstruction_records_its_own_configured_coil_reg(configured):
    b, _ = configured
    cr = b._engine_run["coil_reg"]
    assert cr["record"]["source"] == "configured"
    # exactly the list _apply_coil_reg installed at setup
    assert cr["terms"] == b.mygs.installs[0]
    by = {tuple(t["coils"]): t for t in cr["record"]["terms"]}
    assert by[("F1A",)]["weight"] == 100.0 and by[("F1A",)]["target"] == 2e4
    assert by[("F2A",)]["weight"] == 37.5
    assert by[("#VSC",)] == dict(coils={"#VSC": 1.0}, target=0.0,
                                 weight=50.0)
    assert b.baseline.engine["coil_reg"]["source"] == "configured"


def test_every_draw_loop_solves_under_the_reconstructions_term_list(
        configured):
    """The mutant (the old weak build: weights 1.0, the configured #VSC
    term replaced by a 1e-2 pull toward zero) fails here."""
    b, seen = configured
    recon = b._engine_run["coil_reg"]["terms"]
    v = _quiet(b.verify_sigma0_consistency)
    assert len(seen) == 1 and seen[0] == recon
    assert v["coil_reg"]["source"] == "reconstruction (configured)"
    assert v["coil_reg"]["installed"] is True
    diags = _quiet(b.generate)
    assert len(seen) == 1 + len(diags) and len(diags) == 2
    for s in seen:
        assert s == recon
        w = {t[0]: t[2] for t in s}
        assert w[("F1A",)] == 100.0 and w[("#VSC",)] == 50.0
    for d in diags:
        assert d["engine"]["coil_reg"]["source"] == \
            "reconstruction (configured)"


def test_the_sigma0_check_passes_with_coil_reg_configured(configured):
    b, _ = configured
    v = _quiet(b.verify_sigma0_consistency)
    assert v["passed"] is True
    g = v["gates"]
    for stage in ("loop", "archived"):
        assert set(g[stage]) == {"r_j", "r_I", "dl_i", "dq0"}
        assert g[stage]["dq0"]["bound"] == v["tolerances"]["tol_q0"]
        assert g[stage]["dq0"]["setting"] == "jbs_tol_q0"
        assert all(x["passed"] for x in g[stage].values())
    geo = g["archived_geometry"]
    assert set(geo) == {"coil_F_drift_pct", "coil_VSC_drift_pct",
                        "boundary_rms_mm"}
    hp = b.config.generation.homotopy_passes[-1]
    assert geo["coil_F_drift_pct"]["bound"] == pytest.approx(100 * hp[0])
    assert geo["coil_VSC_drift_pct"]["bound"] == pytest.approx(100 * hp[1])
    assert geo["coil_F_drift_pct"]["value"] == 0.0     # the toy has no coils
    cut, _src = b._boundary_cut(quiet=True)
    assert geo["boundary_rms_mm"]["bound"] == cut
    assert geo["boundary_rms_mm"]["value"] is not None
    assert all(x["passed"] for x in geo.values())


@pytest.mark.parametrize("stage", ["loop", "archived"])
def test_a_q0_miss_fails_the_widened_gate(tmp_path, toy_bouquet_solver,
                                          monkeypatch, stage):
    """|dq0| = 2 jbs_tol_q0 at the row radius, everything else reproduced:
    the stage FAILS on its q0 gate alone (r_j / r_I / l_i still pass)."""
    b = TD._bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    if stage == "loop":
        real = ED.zero_perturbation_loop_verdict

        def verdict(ctx, d):
            import copy
            d = dict(d, record=copy.deepcopy(d["record"]))
            d["record"]["deltas"]["q0"] = 2.0 * ctx.loop["tol_q0"]
            return real(ctx, d)
        monkeypatch.setattr(ED, "zero_perturbation_loop_verdict", verdict)
    else:
        real = ED.zero_perturbation_archived_verdict

        def verdict(ctx, jbs_carried, fin):
            fin = dict(fin, q_row=float(fin["q_row"])
                       + 2.0 * ctx.loop["tol_q0"])
            return real(ctx, jbs_carried, fin)
        monkeypatch.setattr(ED, "zero_perturbation_archived_verdict",
                            verdict)
    v = _quiet(b.verify_sigma0_consistency)
    st = v["stages"][stage]
    assert st["passed"] is False and v["passed"] is False
    gq = st["gates"]["dq0"]
    assert gq["passed"] is False
    assert abs(gq["value"]) == pytest.approx(2.0 * gq["bound"],
                                             abs=0.1 * gq["bound"])
    assert all(st["gates"][k]["passed"] for k in ("r_j", "r_I", "dl_i"))
    # the control: unmodified, the same check passes
    monkeypatch.setattr(ED, "zero_perturbation_" + stage + "_verdict", real)
    assert _quiet(b.verify_sigma0_consistency)["passed"] is True


def test_a_q0_gate_unit():
    """The gate itself: |value| <= bound passes, beyond fails, an unmeasured
    value fails, no bound is recorded as not gated."""
    g = ED.sigma0_gate(1.9e-3, 2e-3, setting="jbs_tol_q0")
    assert g == dict(value=1.9e-3, bound=2e-3, setting="jbs_tol_q0",
                     passed=True)
    assert ED.sigma0_gate(-2.1e-3, 2e-3, setting="x")["passed"] is False
    assert ED.sigma0_gate(None, 2e-3, setting="x")["passed"] is False
    assert ED.sigma0_gate(float("nan"), 2e-3, setting="x")["passed"] is False
    assert ED.sigma0_gate(5.0, None, setting="x")["passed"] is None
    assert ED._gates_pass({"a": ED.sigma0_gate(5.0, None, setting="x")})
    assert not ED._gates_pass({"a": ED.sigma0_gate(5.0, 1.0, setting="x")})


@pytest.mark.parametrize("which", ["coil_F", "coil_VSC", "boundary"])
def test_a_coil_or_boundary_miss_fails_the_check(tmp_path,
                                                 toy_bouquet_solver,
                                                 monkeypatch, which):
    """The archived state's coils / LCFS moved: the loop and archived
    bootstrap stages pass, the geometry gate fails, so the check fails."""
    import bouquet.run as br
    b = TD._bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    real = br.Bouquet._sigma0_archived_geometry

    def geo(self, header, scan_key):
        out = real(self, header, scan_key)
        if which == "coil_F":
            out["max_F_drift_pct"] = 3.0          # > the 1 % final stage
        elif which == "coil_VSC":
            out["max_VSC_drift_pct"] = 3.0
        else:
            out["boundary_rms_mm"] = 1.0e3
        return out
    monkeypatch.setattr(br.Bouquet, "_sigma0_archived_geometry", geo)
    v = _quiet(b.verify_sigma0_consistency)
    assert v["stages"]["loop"]["passed"] and v["stages"]["archived"]["passed"]
    key = dict(coil_F="coil_F_drift_pct", coil_VSC="coil_VSC_drift_pct",
               boundary="boundary_rms_mm")[which]
    assert v["gates"]["archived_geometry"][key]["passed"] is False
    assert v["passed"] is False


def test_an_unreadable_archived_geometry_fails(tmp_path, toy_bouquet_solver,
                                               monkeypatch):
    import bouquet.run as br
    b = TD._bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    real = br.Bouquet._sigma0_archived_geometry

    def geo(self, header, scan_key):
        out = real(self, header, scan_key)
        out["max_F_drift_pct"] = None
        return out
    monkeypatch.setattr(br.Bouquet, "_sigma0_archived_geometry", geo)
    v = _quiet(b.verify_sigma0_consistency)
    assert v["gates"]["archived_geometry"]["coil_F_drift_pct"]["passed"] \
        is False
    assert v["passed"] is False


def test_the_hard_bound_is_the_configured_hard_factor_when_set(
        tmp_path, toy_bouquet_solver):
    b = TD._bq(tmp_path)
    gc = b.config.generation
    gc.coil_drift_hard_factor = 3.0
    geo = dict(max_F_drift_pct=0.0, max_VSC_drift_pct=0.0,
               boundary_rms_mm=0.0)
    b.mygs = None
    g = b._sigma0_geometry_gates(geo)
    assert g["coil_F_drift_pct"]["bound"] == pytest.approx(
        100 * 3.0 * gc.coil_drift)
    assert g["coil_F_drift_pct"]["setting"] == \
        "coil_drift_hard_factor x coil_drift"
    gc.coil_drift_hard_factor = None
    gc.engine_draw_homotopy = False
    g = b._sigma0_geometry_gates(geo)
    assert g["coil_VSC_drift_pct"]["bound"] == pytest.approx(
        100 * gc.coil_drift)


def test_without_a_record_the_draw_keeps_the_historical_list():
    """A solver ``setup_solver`` did not prepare (nothing on record): the
    draw falls back to the historical exploratory list and SAYS so."""
    from _engine_fake_gs import FakeTokaMaker
    G = ED.GenerateEngineDraws.__new__(ED.GenerateEngineDraws)
    G.coil_reg = None
    fake = FakeTokaMaker(None)
    fake._strong_coil_reg = ["strong"]
    rec = G.install_coil_reg(fake)
    assert rec["source"].startswith("historical exploratory")
    assert rec["installed"] is True
    assert ((("#VSC",), 0.0, 1e-2)) in fake.reg
    G.coil_reg = dict(terms=["a", "b"], record=dict(source="configured"))
    rec = G.install_coil_reg(fake)
    assert rec == dict(source="reconstruction (configured)", n_terms=2,
                       installed=True)
    assert fake.reg == ["a", "b"]
