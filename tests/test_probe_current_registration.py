"""The probe compares the achieved current with its reference ON ONE GRID.

``_corrective_output_jphi`` samples the achieved current on the solver's
uniform grid ``linspace(psi_pad, 1 - psi_pad, n)``; only the LENGTH of the
``psi_N`` handed in is used.  The probe compared those samples index for
index with profiles on ``psi_N``.  On a reconstruction grid (uniform) the two
differ by at most ``psi_pad``; on an IDS ``core_profiles`` grid (dense near
the axis) the comparison was between the current at one radius and the
source's at another, and a state that reproduces its source EXACTLY read
some ten per cent of the peak in the core.

Solver-free: a stand-in whose achieved current is a known function of
psi_N, sampled exactly as the solver samples it.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "probes"))
import measure_engine as ME  # noqa: E402

PAD = 1e-3


def _j(x):
    """A peaked current with an edge pedestal [A/m^2]."""
    x = np.asarray(x, float)
    return 1.2e6 * (1.0 - x ** 1.5) ** 1.2 + 2.5e5 * np.exp(
        -((x - 0.95) / 0.03) ** 2)


def _ids_grid(n=101):
    """Dense near the axis, as a rho_tor-uniform grid is in psi_N."""
    return np.linspace(0.0, 1.0, n) ** 2


def _uniform_output(mygs, psi, pad):
    from bouquet.TokaMaker_interface import corrective_output_grid
    return _j(corrective_output_grid(len(psi), pad))


class _GS:
    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"l_i": 0.9, "q_95": 4.0}

    def get_q(self, psi=None):
        psi = np.asarray(psi, float)
        return psi, 1.0 + 3.0 * psi ** 2


def _dd(tmp_path, sign=1.0):
    psi = np.linspace(0.0, 1.0, 201)
    sl = [dict(profiles_1d=dict(psi=list(-psi), q=list(1.0 + 3.0 * psi ** 2),
                                j_tor=list(sign * _j(psi))),
               global_quantities=dict(li_3=0.9, ip=sign * 1.0e6))]
    p = tmp_path / "dd.json"
    p.write_text(json.dumps(dict(equilibrium=dict(time=[1.0],
                                                  time_slice=sl))))
    return str(p)


def _run_ids(tmp_path, monkeypatch, psi, sign=1.0, stamp_sign=True):
    import bouquet.engine as E
    import bouquet.io.imas as IM
    import bouquet.TokaMaker_interface as TI
    monkeypatch.setattr(TI, "_corrective_output_jphi", _uniform_output)
    monkeypatch.setattr(E, "_lcfs_deviation_mm", lambda mygs, bnd: (1.0, 2.0))
    monkeypatch.setattr(IM, "read_imas_geometry",
                        lambda src: (1.0, np.zeros((4, 2))))
    b = SimpleNamespace(mygs=_GS(), config=SimpleNamespace(
        source=SimpleNamespace(ids_path=_dd(tmp_path, sign), time=1.0)))
    bl = SimpleNamespace(psi_N=psi, j_phi=_j(psi))
    if stamp_sign:
        bl.source_current_sign = sign
    return ME._distance_ids(b, bl, PAD)


def test_the_helpers_state_the_grid():
    from bouquet.TokaMaker_interface import (corrective_output_grid,
                                             index_registration_error,
                                             register_corrective_output)
    u = corrective_output_grid(51, PAD)
    np.testing.assert_allclose(u, np.linspace(PAD, 1 - PAD, 51))
    psi = _ids_grid(51)
    A = _j(u)
    reg = register_corrective_output(A, psi, PAD)
    # registered: the profile at psi_N, to the interpolation
    assert np.max(np.abs(reg - _j(psi))) / 1.2e6 < 0.02
    # index for index: an artefact of tens of per cent of the peak
    art = index_registration_error(_j(psi), psi, PAD)
    assert np.max(np.abs(art)) / 1.2e6 > 0.10
    # and nothing on one grid
    np.testing.assert_allclose(index_registration_error(A, u, PAD), 0.0,
                               atol=1e-6)


def test_an_exact_state_reads_zero_on_an_ids_grid(tmp_path, monkeypatch):
    out = _run_ids(tmp_path, monkeypatch, _ids_grid())
    d = out["jphi_vs_equilibrium_jtor_pct_of_peak"]
    assert d["core"]["rms"] < 0.2 and d["core"]["max"] < 0.5
    assert d["edge"]["rms"] < 1.0
    r = out["requested_minus_achieved_pct_of_peak"]
    assert r["core"]["rms"] < 0.2
    g = out["comparison_grid"]
    assert g["max_grid_offset"] > 0.2
    # the artefact the old form reported, on this exact state
    assert g["index_for_index_artefact_pct_of_peak"]["rms"] > 5.0


def test_negative_control_index_for_index_reads_ten_per_cent(tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(ME, "_registered", lambda A, psi, pad: np.asarray(A))
    out = _run_ids(tmp_path, monkeypatch, _ids_grid())
    assert out["jphi_vs_equilibrium_jtor_pct_of_peak"]["core"]["rms"] > 5.0


def test_a_uniform_grid_moves_by_the_pad_only(tmp_path, monkeypatch):
    out = _run_ids(tmp_path, monkeypatch, np.linspace(0.0, 1.0, 129))
    g = out["comparison_grid"]
    assert g["max_grid_offset"] == pytest.approx(PAD)
    assert g["index_for_index_artefact_pct_of_peak"]["max"] < 1.0
    assert out["jphi_vs_equilibrium_jtor_pct_of_peak"]["core"]["rms"] < 0.05


@pytest.mark.parametrize("stamp", [True, False])
def test_a_reversed_current_source_is_compared_in_the_solve_frame(
        tmp_path, monkeypatch, stamp):
    """The source stores j_tor negative; the solve (and the achieved
    current) is in the positive frame.  Compared signed, an exact state
    read ~200 % of the peak."""
    out = _run_ids(tmp_path, monkeypatch, _ids_grid(), sign=-1.0,
                   stamp_sign=stamp)
    assert out["source_current_sign"] == -1.0
    assert out["jphi_vs_equilibrium_jtor_pct_of_peak"]["core"]["rms"] < 0.2
