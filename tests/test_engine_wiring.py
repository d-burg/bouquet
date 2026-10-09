"""``Bouquet.prepare_baseline()`` under ``reconstruction_engine="unified"``,
end to end WITHOUT a solver: the real adapters on the repository's synthetic
inputs (the D3D-like g-file + p-file; the OMAS file), the engine, and the
Baseline the rest of the package consumes -- with the solver replaced by the
toy backend (``tests/_engine_toy.py``) and the few live-solver reads the
metrics make by a stand-in.  What this checks is the WIRING (the adapter ->
engine -> Baseline plumbing, the recorded fields, the failure semantics);
the physics is the toy's, so no fidelity number is asserted.

Synthetic inputs only; no solver, no device data.
"""
import contextlib
import io
import os
import warnings

import numpy as np
import pytest

import _engine_toy as T
from _engine_fake_gs import point_without_solver

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")


class _FakeGS:
    """The live-solver reads outside the engine's backend (metrics)."""

    psi_bounds = [-0.3, 0.0]
    o_point = [1.72, 0.02]

    def __init__(self):
        self.calls = []

    def set_isoflux(self, pts, weights=None):
        self.calls.append(("set_isoflux", len(pts),
                           None if weights is None else float(weights[0])))

    def init_psi(self, *a):
        self.calls.append(("init_psi",) + tuple(float(v) for v in a))

    def get_psi(self, normalized=True):
        return np.zeros(8)

    def get_profiles(self, psi=None, npsi=None, psi_pad=None):
        n = len(psi)
        return (np.asarray(psi), np.full(n, 3.4), np.full(n, -0.1),
                np.zeros(n), np.zeros(n))

    def get_stats(self, lcfs_pad=None, li_normalization="std"):
        return dict(Ip=1.2e6, l_i=0.65, q_0=1.25, q_95=4.6, beta_n=1.3,
                    beta_pol=64.0, kappa=1.7, delta=0.3, W_MHD=5.9e5)


class _ResettableFakeGS(_FakeGS):
    """_FakeGS plus the surface the REAL reset uses (Bouquet._install_solver /
    _reset_solver_state / CaseGeometry.apply), recording the reset steps."""

    coil_sets = {"F1A": 0, "F2A": 1}

    def copy_eq(self):
        return "clean"

    def replace_eq(self, source_eq=None):
        self.calls.append(("replace_eq", source_eq))

    def coil_reg_term(self, coffs, target=0.0, weight=1.0):
        return (tuple(coffs), float(target), float(weight))

    def set_coil_reg(self, reg_terms=None, **kw):
        self.calls.append(("set_coil_reg",))

    def set_coil_bounds(self, b):
        self.calls.append(("set_coil_bounds",))

    def set_profiles(self, **kw):
        self.calls.append(("foffset", kw["foffset"]) if "foffset" in kw else ("set_profiles",))

    def set_saddle_constraints(self, pts, weights=None):
        self.calls.append(("set_saddle_constraints", len(pts)))


@pytest.fixture
def toy_solver(monkeypatch):
    import bouquet.engine as be
    made = {}

    def _backend(mygs, contract, **kw):
        b = T.ToyGS(psi=contract.psi_N, Ip=contract.Ip)
        made["backend"] = b
        made["kw"] = kw
        return b

    monkeypatch.setattr(be, "TokaMakerBackend", _backend)
    monkeypatch.setattr(be, "_lcfs_deviation_mm", lambda mygs, pts: (2.5, 7.0))
    return made


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def test_a_gfile_engine_baseline_is_a_complete_baseline(toy_solver, monkeypatch):
    import bouquet as bq
    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, n_draws=1,
                               reconstruction_engine="unified")
    g = b.config.generation
    g.engine_rows = ["Ip"]
    b.mygs = _FakeGS()
    point_without_solver(monkeypatch)
    bl = _quiet(b.prepare_baseline)
    assert b.baseline is bl and bl.provenance == "reconstruction"
    # the solver was set up as the legacy reconstruction does it
    kinds = [c[0] for c in b.mygs.calls]
    assert kinds[:2] == ["set_isoflux", "init_psi"]
    assert b.mygs.calls[0][2] == 200.0
    # the split sums to the delivered request, which one solve reproduces
    st = bl.engine["state"]
    # ... with the pressure-driven p'G as its own bucket (owner decision D2)
    from bouquet.schema import SPLIT_PRESSURE_SEPARATE
    assert bl.current_split_convention == SPLIT_PRESSURE_SEPARATE
    np.testing.assert_allclose(bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF
                               + bl.j_pressure,
                               bl.j_phi, rtol=0, atol=1e-9 * np.max(bl.j_phi))
    np.testing.assert_array_equal(bl.j_phi, np.asarray(st["request"]))
    np.testing.assert_array_equal(toy_solver["backend"].state["R"], bl.j_phi)
    # the recorded state and the targets
    assert bl.l_i_target == toy_solver["backend"].state["li"]
    assert bl.delivered_state["l_i"] == bl.l_i_target
    assert bl.delivered_state["convention"].startswith("unified engine")
    assert bl.jphi_request_offset is not None
    assert bl.engine["converged"] is True
    m = bl.reconstruction_metrics
    assert m["jbs_loop"]["converged"] is True and not m.get("closure_limited")
    assert m["boundary_rms_mm"] == 2.5 and m["engine"]["solves"]["anchor"] == 2
    assert bl.recon["request_jphi"] is not None
    assert bl.eqdsk_bytes and bl.pfile_bytes
    assert bl.psi_N_kinetic.size == bl.ne.size


def test_an_ids_engine_baseline_is_a_complete_baseline(toy_solver,
                                                       monkeypatch):
    import bouquet as bq
    b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=2.3043, n_draws=1,
                             reconstruction_engine="unified")
    g = b.config.generation
    g.engine_rows = ["Ip"]
    b._install_solver(_ResettableFakeGS())        # the real point_solver / reset
    bl = _quiet(b.prepare_baseline)
    assert bl.provenance == "imas"
    # the solver was reset to the clean state and pointed at THIS slice's
    # geometry (F0, LCFS isoflux) before the engine's init_psi
    kinds = [c[0] for c in b.mygs.calls]
    assert kinds[:5] == ["replace_eq", "set_coil_reg", "foffset", "set_isoflux",
                         "init_psi"]
    assert b.mygs.calls[0] == ("replace_eq", "clean")
    from bouquet.io.imas import read_imas_geometry
    F0, lcfs = read_imas_geometry(b.config.source)
    assert b.mygs.calls[2] == ("foffset", F0)
    assert b.mygs.calls[3][1:] == (len(lcfs), 500.0)
    # the legacy anchors are not carried into the engine's baseline
    assert bl.jBS_diff is None and bl.jphi_diff is None and bl.p_diff is None
    icl = bl.ip_closure
    assert icl["engine"] is True and icl["jbs_converged"] is True
    assert not icl["closure_limited"]
    assert bl.li_metrics["tokamaker_li_3"] == bl.l_i_target
    # ... with the pressure-driven p'G as its own bucket (owner decision D2)
    from bouquet.schema import SPLIT_PRESSURE_SEPARATE
    assert bl.current_split_convention == SPLIT_PRESSURE_SEPARATE
    np.testing.assert_allclose(bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF
                               + bl.j_pressure,
                               bl.j_phi, rtol=0, atol=1e-9 * np.max(bl.j_phi))
    assert bl.engine["contract"]["kind"] == "ids"


def test_a_failed_engine_build_leaves_no_baseline(toy_solver, monkeypatch):
    import bouquet as bq
    import bouquet.engine as be
    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, n_draws=1,
                               reconstruction_engine="unified")
    b.config.generation.engine_rows = ["Ip"]
    b.mygs = _FakeGS()
    point_without_solver(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(be, "reconstruct", _boom)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        _quiet(b.prepare_baseline)
    assert b.baseline is None


def test_the_engine_needs_a_solver():
    import bouquet as bq
    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, n_draws=1,
                               reconstruction_engine="unified")
    with pytest.raises(ValueError, match="setup_solver"):
        b.prepare_baseline()


def test_a_flagged_ids_engine_baseline_warns(toy_solver, monkeypatch):
    """jbs_loop_on_fail="flag" with a loop that cannot converge: the IDS
    engine baseline is delivered flagged closure_limited (ip_closure) AND
    warned about -- as loudly as a flagged g-file baseline."""
    import bouquet as bq
    b = bq.Bouquet.from_imas(_OMAS, mesh=_MESH, time=2.3043, n_draws=1,
                             reconstruction_engine="unified")
    g = b.config.generation
    g.engine_rows = ["Ip"]
    g.jbs_loop_on_fail = "flag"
    g.jbs_max_passes = 2
    b.mygs = _FakeGS()
    point_without_solver(monkeypatch)
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.warns(RuntimeWarning, match="engine IDS baseline: NOT "
                                                "converged"):
            bl = b.prepare_baseline()
    assert bl.ip_closure["jbs_converged"] is False
    assert bl.ip_closure["closure_limited"] is True
