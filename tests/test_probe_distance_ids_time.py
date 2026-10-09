"""The engine probe's IDS distance table reads the SOURCE's slice time.

``tests/probes/measure_engine.py::_distance_ids`` used to compare against the
synthetic example's constant slice time whatever the source said -- on any
other dd (or another slice of the same dd) it silently measured the wrong
slice.  Checked here on a two-slice synthetic dd with the solver calls
stubbed (no solver, no device data).
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


class _GS:
    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"l_i": 0.9, "q_95": 4.0}

    def get_q(self, psi=None):
        psi = np.asarray(psi, float)
        return psi, 1.0 + 3.0 * psi ** 2


def _dd(tmp_path):
    psi = np.linspace(0.0, 1.0, 11)
    sl = []
    for li3, qs in ((0.71, 1.0), (0.83, 1.2)):
        sl.append(dict(
            profiles_1d=dict(psi=list(-psi), q=list(qs + 3.0 * psi ** 2),
                             j_tor=list(1e6 * (1 - psi ** 2))),
            global_quantities=dict(li_3=li3)))
    p = tmp_path / "dd.json"
    p.write_text(json.dumps(dict(equilibrium=dict(time=[1.0, 2.0],
                                                  time_slice=sl))))
    return str(p)


@pytest.mark.parametrize("t, li3", [(1.0, 0.71), (2.0, 0.83), (None, 0.71),
                                    (1.9, 0.83)])
def test_distance_ids_uses_the_source_time(tmp_path, monkeypatch, t, li3):
    import bouquet.engine as E
    import bouquet.io.imas as IM
    import bouquet.TokaMaker_interface as TI
    path = _dd(tmp_path)
    psi = np.linspace(0.0, 1.0, 21)
    monkeypatch.setattr(TI, "_corrective_output_jphi",
                        lambda mygs, p, pad: 1e6 * (1 - np.asarray(p) ** 2))
    monkeypatch.setattr(E, "_lcfs_deviation_mm", lambda mygs, bnd: (1.0, 2.0))
    monkeypatch.setattr(IM, "read_imas_geometry",
                        lambda src: (1.0, np.zeros((4, 2))))
    b = SimpleNamespace(mygs=_GS(), config=SimpleNamespace(
        source=SimpleNamespace(ids_path=path, time=t)))
    bl = SimpleNamespace(psi_N=psi, j_phi=1e6 * (1 - psi ** 2))
    out = ME._distance_ids(b, bl, 1e-3)
    assert out["li3"]["input"] == pytest.approx(li3)
    assert out["slice"]["time_requested"] == t
    assert out["slice"]["time_of_slice"] == (1.0 if li3 == 0.71 else 2.0)
