"""The reader reports when the dd's psi_N(rho) is not the LCFS g-file's.

FUSE holds replayed profiles fixed in rho_tor_norm while it solves its own
equilibrium.  On shot 174956 t=1.191 that moved rho=0.8 from psi_N 0.70 (EFIT)
to 0.77, and the IDA-vs-dd ni cross-check reported the shift as a 5.8 % ni
mismatch.  The guard names the geometry instead.
"""
import json
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet.io.geqdsk as geq
from bouquet.io.imas import PSI_RHO_DRIFT_TOL, PSI_RHO_GATE_RHO, _psi_rho_drift
from test_ni_fast_subtraction import N, _build, _read

PSI = np.linspace(0.0, 1.0, N)


class _G:
    def __init__(self, rho):
        self.rhovn = rho


@pytest.fixture
def gfile(monkeypatch):
    """read_geqdsk stub: rho_tor_norm(psi_N) = psi_N ** power on 65 points."""
    def use(power):
        monkeypatch.setattr(geq, "read_geqdsk",
                            lambda path: _G(np.linspace(0.0, 1.0, 65) ** power))
        return "g.stub"
    return use


class TestDrift:
    def test_same_map_is_quiet(self, gfile):
        d = _psi_rho_drift(PSI, np.sqrt(PSI), gfile(0.5))
        assert d["max_abs"] < 1e-3 and not d["exceeds"]

    def test_a_shifted_map_is_measured_in_psi_N(self, gfile):
        d = _psi_rho_drift(PSI, np.sqrt(PSI), gfile(0.45))
        pts = np.asarray(PSI_RHO_GATE_RHO)
        expect = np.interp(pts, np.sqrt(PSI), PSI) - pts ** (1 / 0.45)
        np.testing.assert_allclose(list(d["gate"].values()), expect, atol=2e-3)
        assert d["exceeds"] and d["max_abs"] > PSI_RHO_DRIFT_TOL

    def test_no_usable_rho_grid(self, gfile):
        assert _psi_rho_drift(PSI, np.zeros(N), gfile(0.5)) is None


def _with_rho(scale_ni=1.0):
    def mutate(dd):
        cp = dd["core_profiles"]["profiles_1d"][0]
        cp["grid"]["rho_tor_norm"] = np.sqrt(PSI).tolist()
        cp["ion"][0]["density_thermal"] = (
            scale_ni * np.asarray(cp["ion"][0]["density_thermal"])).tolist()
    return mutate


def _read_warned(ddp, cdf, g):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bl = _read(ddp, cdf, LCFS_geqdsk=g)
    return bl, [str(x.message) for x in w]


class TestReader:
    def test_the_reader_records_and_warns(self, tmp_path, gfile):
        ddp, cdf, *_ = _build(tmp_path, dd_mutate=_with_rho())
        bl, msgs = _read_warned(ddp, cdf, gfile(0.45))
        assert bl.aux["psi_rho_drift"]["exceeds"]
        assert any("psi_N(rho)" in m and "IDA kinetics" in m for m in msgs)

    def test_quiet_on_a_matching_g_file(self, tmp_path, gfile):
        ddp, cdf, *_ = _build(tmp_path, dd_mutate=_with_rho())
        bl, msgs = _read_warned(ddp, cdf, gfile(0.5))
        assert not bl.aux["psi_rho_drift"]["exceeds"]
        assert not any("psi_N(rho)" in m for m in msgs)

    @pytest.mark.parametrize("power, named", [(0.45, True), (0.5, False)])
    def test_the_ni_mismatch_names_the_drift_only_when_there_is_one(
            self, tmp_path, gfile, power, named):
        ddp, cdf, *_ = _build(tmp_path, dd_mutate=_with_rho(scale_ni=1.05))
        bl, _ = _read_warned(ddp, cdf, gfile(power))
        meta = bl.aux["ni_fast_meta"]
        assert meta["agrees"] is False
        assert ("psi_N(rho) drift" in meta["evidence"]) is named

    def test_no_g_file_no_guard(self, tmp_path):
        ddp, cdf, *_ = _build(tmp_path, dd_mutate=_with_rho())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bl = _read(ddp, cdf)
        assert "psi_rho_drift" not in bl.aux
