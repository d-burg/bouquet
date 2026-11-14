"""Current corrections of the filter as a source over the interval (in-window form), against the steady form.

Analytic separatrix-like geometry of ``test_coordinates_kinetics_series.py`` WITH its non-inductive (edge
bootstrap-like) current, the package's interval solver (:func:`bouquet.kinrecon.series.interval_solver`), no OFT.

* A mid-radius correction (one Gaussian at rho_tor,norm 0.5, amplitude 1 % of Ip) switched on for 20 ms on a
  relaxed state, the state differenced against the same interval without it: the axis current barely moves,
  while the zero-net-current steady response (:func:`bouquet.kinrecon.current_model.correction_response`, what
  the earlier treatment added per absorbed fit) moves the axis current by about the amplitude.  Bound (fixed
  before the first run): the 20 ms axis change is below 5 % of the steady one, and the steady one is at least
  half of the amplitude times the axis current (so the comparison is not between two negligible numbers).
* Convergence: with the conductivity scaled down by 1e3, the in-window response approaches the steady
  response of the same interval solver as the interval grows -- monotonically over 5e-5, 5e-4, 5e-3 and
  5e-2 s, and to 1e-6 of its span after 5e-2 s.  Bounds fixed before the first run; the interval set was
  changed after the first run, which had used 5e-3 to 5 s on a guessed resistive time: the response had
  converged to 4e-11 by 5e-2 s and the longer intervals only accumulated round-off (up to 4e-8, from the
  linearly drifting boundary flux), so their sequence was not monotone.  The code's steady form (built through
  ``build_solver``'s outer layer) is compared with that same-solver steady response: bound 1e-3 of the span,
  fixed before the first run (the two constructions integrate the layer's conductance and source separately).
* The net current is unchanged: the interval is driven at the same Ip boundary value with and without the
  source, and the domain-edge current change is returned by the layer (``Delta I(c) + F (A Delta rate +
  Delta B) = 0`` to 1e-9 of Ip).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import test_coordinates_kinetics_series as GEO  # noqa: E402
from bouquet.kinrecon import current_model as CM  # noqa: E402
from bouquet.kinrecon import series as SE  # noqa: E402
from bouquet.kinrecon.current import CurrentState  # noqa: E402

IP0 = GEO.IP
AMPS = (0.0, 0.01, 0.0)          # one Gaussian at rho 0.5, 1 % of Ip


def _setup(sigma_scale):
    lag = GEO._lag(q0=1.1, sigma_scale=sigma_scale)
    assert np.max(np.abs(lag["jbs"])) > 0.0                      # the fixture carries non-inductive current
    cfg = CM.CurrentModelSettings(r0=GEO.R0)
    rf = CurrentState.uniform_faces(cfg.n_cells)
    st0 = CM.build_solver(lag, lag["jbs"], IP0).steady_state(rf)
    st0 = CurrentState(rf, st0.psi, st0.psi_boundary, 0.0)
    shapes = CM.correction_shapes(lag, cfg, IP0)
    jsrc = lag["jbs"] + sum(a * g for a, g in zip(AMPS, shapes))
    return lag, cfg, rf, st0, shapes, jsrc


def _in_window(lag, st0, jsrc, dt, dt_max):
    """End states over [0, dt] with and without the correction source (same Ip boundary value)."""
    sol_s, _ = SE.interval_solver(lag, lag, jsrc, jsrc, 0.0, dt, IP0, IP0)
    sol_0, _ = SE.interval_solver(lag, lag, lag["jbs"], lag["jbs"], 0.0, dt, IP0, IP0)
    return sol_s.advance(st0, dt, dt=dt_max), sol_0.advance(st0, dt, dt=dt_max), sol_s, sol_0


def _gauge_free(st):
    return np.asarray(st.psi, float) - float(st.psi_boundary)


def _axis_jdotb(lag, st):
    return float(CM.build_solver(lag, lag["jbs"], IP0).profiles(st)["jdotb"][0])


def test_in_window_correction_moves_the_axis_far_less_than_the_steady_form():
    lag, cfg, rf, st0, shapes, jsrc = _setup(1.0)
    e_s, e_0, _, _ = _in_window(lag, st0, jsrc, 0.02, SE.SeriesSettings().dt_max)
    j0 = _axis_jdotb(lag, st0)
    d_window = _axis_jdotb(lag, CurrentState(rf, st0.psi + (e_s.psi - e_0.psi),
                                             st0.psi_boundary + (e_s.psi_boundary - e_0.psi_boundary), 0.0)) - j0
    dpsi, dpb = CM.correction_response(lag, shapes, AMPS, rf)
    d_steady = _axis_jdotb(lag, CurrentState(rf, st0.psi + dpsi, st0.psi_boundary + dpb, 0.0)) - j0
    assert abs(d_steady) >= 0.5 * sum(AMPS) * abs(j0), (d_steady, j0)
    assert d_steady < 0.0                                         # the steady return sits on the axis
    assert abs(d_window) <= 0.05 * abs(d_steady), (d_window, d_steady)
    # the source changed the current where it acts: the gauge-free state difference is not negligible
    diff = _gauge_free(e_s) - _gauge_free(e_0)
    assert np.max(np.abs(diff)) > 1e-6 * np.ptp(_gauge_free(st0))


def test_in_window_response_converges_to_the_steady_response():
    lag, cfg, rf, st0, shapes, jsrc = _setup(1.0e-3)
    errs = []
    ref = None
    for dt in (5.0e-5, 5.0e-4, 5.0e-3, 5.0e-2):
        e_s, e_0, sol_s, sol_0 = _in_window(lag, st0, jsrc, dt, dt / 50.0)
        if ref is None:
            ref = _gauge_free(sol_s.steady_state(rf)) - _gauge_free(sol_0.steady_state(rf))
        d = _gauge_free(e_s) - _gauge_free(e_0)
        errs.append(float(np.max(np.abs(d - ref)) / np.ptp(ref)))
    assert all(a > b for a, b in zip(errs[:-1], errs[1:])), errs
    assert errs[-1] <= 1e-6, errs
    dpsi, dpb = CM.correction_response(lag, shapes, AMPS, rf)
    code = dpsi - dpb
    assert np.max(np.abs(code - ref)) <= 1e-3 * np.ptp(ref)


def test_in_window_correction_keeps_the_net_current():
    lag, cfg, rf, st0, shapes, jsrc = _setup(1.0)
    e_s, e_0, sol_s, sol_0 = _in_window(lag, st0, jsrc, 0.02, SE.SeriesSettings().dt_max)
    p_s, p_0 = sol_s.profiles(e_s), sol_0.profiles(e_0)
    assert p_s["ip"] == pytest.approx(IP0, rel=1e-12) and p_0["ip"] == pytest.approx(IP0, rel=1e-12)
    a_s, b_s, f_s = SE._layer_integrals(lag, lag["sigma"], jsrc)
    a_0, b_0, f_0 = SE._layer_integrals(lag, lag["sigma"], lag["jbs"])
    d_layer = f_s * (a_s * p_s["v_loop_boundary"] / (2 * np.pi) + b_s) - f_0 * (a_0 * p_0["v_loop_boundary"] / (2 * np.pi)
                                                                               + b_0)
    d_edge = p_s["ip_domain_edge"] - p_0["ip_domain_edge"]
    assert abs(d_edge + d_layer) <= 1e-9 * IP0                    # what crosses the domain edge the layer returns
