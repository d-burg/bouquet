"""The per-baseline solver geometry is recorded and archived (#76 B2).

Since #76 every baseline on a reused solver solves at ITS OWN vacuum F0
(``bouquet.solver_geometry``); before, a reused solver kept the first
baseline's F0.  These tests pin the record of that rule on the baseline and
in the archive (``_baseline`` attr ``solver_geometry_json``), through the real
``point_solver -> _reset_solver_state -> CaseGeometry.apply`` path on the
engine's toy stand-in solver.  Synthetic example data only.
"""
import copy
import os

import numpy as np
import pytest

from test_engine_draws import _bq, _quiet, toy_bouquet_solver  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")


def _gfile_F0():
    from bouquet.io.geqdsk import read_geqdsk
    g = read_geqdsk(os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"))
    return abs(g.R_center * g.B_center)


def test_the_case_geometry_record_names_its_origins():
    from bouquet.config import SolverConfig, ReconstructionSource
    from bouquet.solver_geometry import F0_RULE, case_geometry
    src = ReconstructionSource(
        geqdsk_path=os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles_path=os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk"))
    sc = SolverConfig(mesh_path=os.path.join(_EX, "DIIID_mesh.h5"))
    rec = case_geometry(src, sc).record()
    assert rec["solver_F0"] == rec["F0_source"] == pytest.approx(_gfile_F0(), rel=0, abs=0)
    assert rec["F0_origin"].startswith("source: g-file")
    assert rec["F0_rule"] == F0_RULE
    assert rec["isoflux_origin"] == "source boundary" and rec["isoflux_n"] > 10
    assert rec["isoflux_weights"]["min"] == rec["isoflux_weights"]["max"] == 500.0
    assert rec["saddle_n"] == 0 and rec["saddle_origin"] is None

    sc2 = copy.deepcopy(sc)
    sc2.F0 = 1.1 * _gfile_F0()
    sc2.saddle_targets = [[1.3, -1.1]]
    with pytest.warns(UserWarning, match="overrides the source's F0"):
        rec2 = case_geometry(src, sc2).record()
    assert rec2["solver_F0"] == pytest.approx(1.1 * _gfile_F0())
    assert rec2["F0_source"] == pytest.approx(_gfile_F0())
    assert rec2["F0_origin"] == "SolverConfig.F0"
    assert rec2["saddle_n"] == 1 and rec2["saddle_points"] == [[1.3, -1.1]]


def test_a_source_without_F0_is_refused_with_a_remedy(monkeypatch):
    import bouquet.io.imas as bi
    from bouquet.config import ImasSource, SolverConfig
    from bouquet.solver_geometry import case_geometry
    monkeypatch.setattr(bi, "read_imas_geometry",
                        lambda s: (None, np.zeros((4, 2))))
    src = ImasSource.__new__(ImasSource)          # no file access needed
    with pytest.raises(ValueError, match="set SolverConfig.F0"):
        case_geometry(src, SolverConfig(mesh_path="m.h5"))


def test_every_baseline_records_and_archives_its_own_geometry(tmp_path, toy_bouquet_solver):
    """Two baselines on ONE solver: each is stamped with its own F0 and the
    second knows it ran on a reused solver."""
    from bouquet.utils import load_solver_geometry
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    rec = b.baseline.solver_geometry
    assert rec["solver_F0"] == pytest.approx(_gfile_F0(), rel=0, abs=0)
    assert rec["solver_use"] == 1
    assert rec["coil_reg"]["source"] == "default" and rec["coil_reg"]["n_terms"] > 0
    assert b.baseline.engine["solver_geometry"] == rec
    _quiet(b.generate)
    assert load_solver_geometry(b.config.output_header, 0) == rec

    # a second baseline on the same solver at another F0
    b.config.solver.F0 = 1.05 * _gfile_F0()
    b.set_case(header=str(tmp_path / "second"), scan_key=1)
    _quiet(b.prepare_baseline)
    rec2 = b.baseline.solver_geometry
    assert rec2["solver_F0"] == pytest.approx(1.05 * _gfile_F0())
    assert rec2["F0_origin"] == "SolverConfig.F0" and rec2["solver_use"] == 2
    _quiet(b.generate)
    assert load_solver_geometry(str(tmp_path / "second"), 1)["solver_F0"] \
        == pytest.approx(1.05 * _gfile_F0())
