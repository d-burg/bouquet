"""The frozen legacy path (``jbs_self_consistent=False``) is bit for bit
after the engine-base consistency changes -- a fast form of
``tests/probes/legacy_bitwise_ab.py`` for what those changes touch.

* The electron-charge constant: ``generate_bouquet``'s pressure handed to
  every draw is built with the frozen ``ELEMENTARY_CHARGE_LEGACY`` exactly as
  the historical expression ``1.6022e-19 * (ne*te + ni*ti)`` did (same grid
  and on a separate kinetic grid), and with ``ELEMENTARY_CHARGE`` only when
  the loop is on.
* The inductive-amplitude fallback: bit-identical to its frozen pre-change
  copy (``tests/test_fit_inductive_fallback_loud.py``).
* The anchor rejection is loop-only (``tests/test_draw_anchor_solve_refused``)
  and the q0 changes are labels (``tests/test_q0_like_radii.py``).

Mocked draw (no solver); synthetic inputs only.
"""
import contextlib
import io
import os

import numpy as np
import pytest

from bouquet.physics import ELEMENTARY_CHARGE, ELEMENTARY_CHARGE_LEGACY
from bouquet.utils import pchip_interp


def _captured_pressure(tmp_path, monkeypatch, jbs_loop, kin_grid):
    import bouquet.TokaMaker_interface as TI
    import test_draw_rejections as R

    cap = {}

    def _perturb(mygs, psi_N, pressure, *a, **k):
        cap["pressure"] = np.array(pressure, dtype=float, copy=True)
        raise RuntimeError("captured (A/B stop)")

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _perturb)
    X = R._X
    kin = None if not kin_grid else np.linspace(0.0, 1.0, 31)
    xk = X if kin is None else kin
    ne = 5e19 * (1 - 0.8 * xk ** 2)
    te = 2e3 * (1 - 0.9 * xk ** 2) + 50.0
    ni, ti = 0.9 * ne, 0.95 * te
    Jb = 1.0e5 * (1 - X)
    jphi = 1.0e6 * (1 - X ** 2) + Jb
    with contextlib.redirect_stdout(io.StringIO()):
        TI.generate_bouquet(
            R._FakeGS(), X, 1, os.path.join(str(tmp_path), "ab"), jphi,
            ne, te, ni, ti, 0.05 * ne, 0.05 * te, 0.05 * ni, 0.05 * ti,
            0.05 * jphi, 0.4, 0.4, 0.25, 1.0e6, 0.8, 1.5 * np.ones(len(X)),
            input_jinductive=jphi - Jb, baseline_j_BS=Jb, psi_N_kinetic=kin,
            diagnostic_plots=False, seed=3, jbs_loop=jbs_loop,
            homotopy_passes=[(0.05, 0.10), (0.01, 0.01)], rejection_log=[])
    if kin is None:
        thermal = ne * te + ni * ti
    else:
        f = lambda a: pchip_interp(kin, a, X)          # noqa: E731
        thermal = f(ne) * f(te) + f(ni) * f(ti)
    return cap["pressure"], thermal


@pytest.mark.parametrize("kin_grid", [False, True])
def test_legacy_generate_pressure_is_the_historical_expression(
        tmp_path, monkeypatch, kin_grid):
    p, thermal = _captured_pressure(tmp_path, monkeypatch, None, kin_grid)
    # bit for bit the historical literal
    np.testing.assert_array_equal(p, 1.6022e-19 * thermal)
    np.testing.assert_array_equal(p, ELEMENTARY_CHARGE_LEGACY * thermal)


@pytest.mark.parametrize("kin_grid", [False, True])
def test_loop_generate_pressure_uses_the_one_constant(tmp_path, monkeypatch,
                                                       kin_grid):
    from bouquet.jbs_loop import jbs_settings
    from test_mse_refusal_restore import _GC
    p, thermal = _captured_pressure(tmp_path, monkeypatch,
                                    jbs_settings(_GC(), draw=True), kin_grid)
    np.testing.assert_array_equal(p, ELEMENTARY_CHARGE * thermal)
    # the deliberate consistency change, and its size
    rel = p / (ELEMENTARY_CHARGE_LEGACY * thermal) - 1.0
    np.testing.assert_allclose(rel, -1.4584e-5, rtol=1e-4, atol=0.0)
