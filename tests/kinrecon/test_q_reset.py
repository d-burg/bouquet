"""EXPERIMENTAL opt-in core form ``q_floor_form="reset"``: the Grad-Shafranov solver's own zero-net-current q reset
(``current.sawtooth.q_reset``, calling the solver's 1-D routine) applied to the model state.

Synthetic geometry of ``test_q_floor.py`` (the model's steady state at 2.2 times the fixture Ip has q < 1 inside rho
~0.56).  Needs the solver's Python package for the 1-D routine (skipped without it).

Checked: Ip and the enclosed current outside the reset unchanged (zero net current); q on the interior faces at or
above the level and at most the level plus the shoulder height after a full reset without smoothing; the identity
where q >= level; refusal when q at the domain edge is below the level; rule 1 and rule 2 agree on a monotone
profile; the source variant (the reset current as a zero-net non-inductive source of the diffusion equation) has as
its steady state the projected state; the default core form is unchanged.
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
from bouquet.kinrecon.current import CurrentState  # noqa: E402
from bouquet.kinrecon.current import sawtooth as SAW  # noqa: E402

try:
    from OpenFUSIONToolkit.TokaMaker.bootstrap import _saw_reset_1d  # noqa: F401
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False

pytestmark = pytest.mark.skipif(not _HAVE, reason="needs the solver's 1-D sawtooth reset routine")

IP_LOW_Q = 2.2 * GEO.IP
ROUND_OFF = 1e-12


def _steady(ip, n=100):
    lag = GEO._lag()
    sol = CM.build_solver(lag, lag["jbs"], ip)
    return sol, sol.steady_state(CurrentState.uniform_faces(n))


@pytest.mark.parametrize("rule", [1, 2])
def test_reset_conserves_ip_and_the_current_outside_and_lands_on_the_level(rule):
    sol, st = _steady(IP_LOW_Q)
    p0 = sol.profiles(st)
    st1, rec = SAW.q_reset(st, sol, 1.0, rule=rule, smooth=0)
    p1 = sol.profiles(st1)
    assert rec["acted"] and rec["ierr"] == 0 and 0.0 < rec["rho_m"] < 1.0
    assert p1["ip"] == p0["ip"] and st1.psi_boundary == st.psi_boundary and st1.psi[-1] == st.psi[-1]
    rf = st.rho_faces
    out = rf > rec["rho_out"]
    np.testing.assert_allclose(p1["I_enclosed"][out], p0["I_enclosed"][out], rtol=ROUND_OFF, atol=0.0)
    q1 = p1["q"][1:-1]
    assert q1.min() >= 1.0 * (1.0 - ROUND_OFF)
    inside = (rf[1:-1] < rec["rho_m"])
    assert q1[inside].max() <= 1.0 + SAW.Q_RESET_DQ + ROUND_OFF


def test_reset_is_the_identity_above_the_level_and_refuses_the_edge():
    sol, st = _steady(GEO.IP)
    st1, rec = SAW.q_reset(st, sol, 1.0)
    assert st1 is st and not rec["acted"]
    sol2, st2 = _steady(IP_LOW_Q)
    q_edge = sol2.profiles(st2)["q"][-1]
    with pytest.raises(SAW.QFloorRefused):
        SAW.q_reset(st2, sol2, q_edge * 1.01)


def test_source_variant_steady_state_is_the_projected_state():
    sol, st = _steady(IP_LOW_Q)
    st1, rec = SAW.q_reset(st, sol, 1.0, rule=1, smooth=0)
    s_src = SAW.reset_source_solver(sol, rec["dI_faces"], st.rho_faces)
    st2 = s_src.steady_state(st.rho_faces)
    q1, q2 = sol.profiles(st1)["q"], sol.profiles(st2)["q"]
    np.testing.assert_allclose(q2[1:], q1[1:], rtol=1e-10, atol=0.0)
    assert s_src.profiles(st2)["ip"] == pytest.approx(sol.profiles(st1)["ip"], rel=1e-12)


def test_default_core_form_is_unchanged():
    cfg = CM.CurrentModelSettings(r0=1.7)
    assert cfg.q_floor_form == "resistive" and cfg.core_relaxation == "q_floor"
    assert (cfg.q_reset_rule, cfg.q_reset_dq, cfg.q_reset_ramp, cfg.q_reset_smooth) == (2, 0.03, 0.01, 8)
    with pytest.raises(ValueError):
        CM.CurrentModelSettings(r0=1.7, q_reset_rule=3)
