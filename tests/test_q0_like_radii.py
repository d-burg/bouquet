"""q0 is compared at LIKE radii, and every reported q0 says where it lives.

TokaMaker's ``get_stats()['q_0']`` is q at psi_N = 0.02 (its ``axis_pad``),
the first traced surface -- not the magnetic axis, where a g-file's
``qpsi[0]`` lives.  The reconstruction metrics used to compare the two.  Now
the g-file is read at the solver's radius (``q0_efit``, ``q0_err_pct``), the
axis value and the old solver-vs-axis error are kept under honest names
(``q0_efit_axis``, ``q0_err_pct_vs_axis``), and ``q0_psi_N`` records the
radius.  No q0 target, gate or tolerance changes.

Synthetic doubles only (no solver).
"""
import numpy as np
import pytest

from bouquet.physics import SOLVER_Q0_PSI_N

_PSI = np.linspace(0.0, 1.0, 65)
_QPSI = 0.95 + 3.0 * _PSI ** 2          # q rises off axis
#: the solver's q at psi_N 0.02: the g-file's q there, 1e-5 relative off
_Q0_SOLVER = float(np.interp(0.02, _PSI, _QPSI)) * (1.0 + 1e-5)


class _GS:
    o_point = (1.7, 0.03)

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"q_0": _Q0_SOLVER, "q_95": 3.6, "beta_n": 2.0,
                "beta_pol": 80.0, "kappa": 1.8, "delta": 0.5,
                "W_MHD": 1.0e6, "l_i": 0.68}


class _Eqdsk:
    psi_N = _PSI
    qpsi = _QPSI
    Ip = 1.4e6
    li = {"li(2)": 0.68, "li(1)_EFIT": 0.9}
    betas = {"beta_n": 2.0, "beta_p": 0.8}
    geometry = {"kappa": np.array([1.8]), "delta": np.array([0.5])}
    R_mag = 1.7
    Z_mag = 0.03
    pres = np.zeros(len(_PSI))

    def volume_integral(self, what):
        return np.linspace(0.0, 1.0e6 / 1.5, len(_PSI))


class _Src:
    psi_pad = 1e-3


def _metrics():
    import bouquet.baseline as bl
    result = {"Ip_tokamaker": 1.4e6,
              "j_phi_fit": np.linspace(1e6, 0.0, len(_PSI)),
              "eqdsk_jtor": np.linspace(1e6, 0.0, len(_PSI)),
              "quality": {}}
    return bl._reconstruction_metrics(_GS(), _Eqdsk(), result, _Src(),
                                      0.68,
                                      l_i_realized_post_corrective=0.68)


def test_the_solver_q0_radius_is_named():
    assert SOLVER_Q0_PSI_N == 0.02


def test_reconstruction_q0_is_compared_at_the_solver_radius():
    m = _metrics()
    q_like = float(np.interp(0.02, _PSI, _QPSI))
    assert m["q0_psi_N"] == 0.02
    assert m["q0"] == _Q0_SOLVER
    assert m["q0_efit"] == pytest.approx(q_like, rel=1e-15)
    assert m["q0_err_pct"] == pytest.approx(
        100.0 * (_Q0_SOLVER - q_like) / q_like, rel=1e-12)
    # the axis value and the old (unlike-radius) comparison, honestly named
    assert m["q0_efit_axis"] == _QPSI[0]
    assert m["q0_err_pct_vs_axis"] == pytest.approx(
        100.0 * (_Q0_SOLVER - _QPSI[0]) / _QPSI[0], rel=1e-12)
    # like radii agree to 0.001 %; the axis comparison reads the q rise
    assert abs(m["q0_err_pct"]) < 1e-2 < abs(m["q0_err_pct_vs_axis"])


def test_the_verdict_does_not_read_q0():
    """Labelling only: the PASS/CHECK verdict never involved q0."""
    m = _metrics()
    assert m["verdict"] in ("PASS", "CHECK")
    import inspect

    import bouquet.baseline as bl
    src = inspect.getsource(bl._reconstruction_metrics)
    verdict = src.split("ok = bool(", 1)[1].split(")\n", 1)[0]
    assert "q0" not in verdict


def test_delivered_states_label_their_q0_radius():
    import inspect

    import bouquet.baseline as bl
    import bouquet.run as R
    import bouquet.TokaMaker_interface as TI
    assert "q0_psi_N=float(m.get(\"q0_psi_N\", SOLVER_Q0_PSI_N))" in \
        inspect.getsource(bl._deliver_reconstruction_state)
    src_run = inspect.getsource(R)
    assert src_run.count("q0_psi_N=float(SOLVER_Q0_PSI_N)") == 2
    assert "q0_psi_N=float(SOLVER_Q0_PSI_N)" in inspect.getsource(
        TI.generate_bouquet)
