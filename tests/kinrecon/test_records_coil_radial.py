"""Fast tests (no solver) of two records that steer nothing:

* the producing equilibrium's midplane table on the kinetic inputs (``assemble_kinetic_inputs(...,
  midplane_radius=...)``): kept on the result, stated in its record ("not supplied" without it), refused when it is
  not two strictly increasing finite arrays, and without any effect on the profiles (compared bit for bit);
* the vertical-stability coil current read from the solver's iteration lines (``forward.parse_history``: column 7
  of the last line of the last solver block), checked on constructed solver output;
* a forward solve (on a stub solver) whose solver output is not captured (buffered Fortran output): a recorded
  failure of that solve with the reason stated, and the same solve with its output captured accepted;
* the fit's two record helpers when the forward model or the kinetic inputs do not provide what they read.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from bouquet.kinrecon import edge as EDGE
from bouquet.kinrecon import fit as FIT
from bouquet.kinrecon import forward as FW
from bouquet.kinrecon import kinetics as KI


def _inputs():
    x = np.linspace(0.0, 1.0, 51)
    ne = 4e19 * (1.0 - 0.8 * x ** 2)
    te = 2e3 * (1.0 - 0.9 * x ** 2)
    return x, ne, te, 0.8 * te, dict(zeff_candidates={"vb": ("zeff", np.full_like(x, 1.8))}, zeff_source="vb",
                                     z_imp=6.0, n_fast="none",
                                     ti_coverage=dict(points=dict(value=[1.5e3], psi_N=[0.2]), core_psi_n_max=None))


def test_midplane_table_is_kept_and_stated_and_changes_no_profile():
    x, ne, te, ti, kw = _inputs()
    mp = np.linspace(0.0, 1.02, 30)
    mr = 1.7 + 0.6 * np.sqrt(mp)
    a = KI.assemble_kinetic_inputs(x, ne, te, ti, **kw)
    b = KI.assemble_kinetic_inputs(x, ne, te, ti, midplane_radius=(mp, mr, "producer"), **kw)
    assert a.midplane_radius is None and a.record["midplane_radius"] == "not supplied"
    assert b.midplane_radius["source"] == "producer"
    assert np.array_equal(b.midplane_radius["psi_n"], mp) and np.array_equal(b.midplane_radius["r_m"], mr)
    assert b.record["midplane_radius"].startswith("supplied: producer, 30 points")
    for k in ("x", "ne", "te", "ti", "zeff", "ni", "nimp", "n_fast"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert np.array_equal(a.thermal_pressure_at(rho=x), b.thermal_pressure_at(rho=x))
    c = KI.assemble_kinetic_inputs(x, ne, te, ti, midplane_radius=(mp, mr), mapping_source="eq", **kw)
    assert c.midplane_radius["source"] == "eq"


@pytest.mark.parametrize("bad", ["psi_flat", "r_down", "nan", "length"])
def test_midplane_table_refused_unless_strictly_increasing_and_finite(bad):
    x, ne, te, ti, kw = _inputs()
    mp = np.linspace(0.0, 1.0, 10)
    mr = 1.7 + 0.5 * mp
    if bad == "psi_flat":
        mp[3] = mp[2]
    elif bad == "r_down":
        mr[-1] = mr[-3]
    elif bad == "nan":
        mr[4] = np.nan
    else:
        mr = mr[:-1]
    with pytest.raises(ValueError, match="midplane_radius"):
        KI.assemble_kinetic_inputs(x, ne, te, ti, midplane_radius=(mp, mr), **kw)


def _line(i, vsc):
    return f"  {i:4d}" + "".join(f"{v:12.4E}" for v in (1.0, 2.0, 1e-3 / i, 1.7, 0.01, vsc))


def test_vsc_current_is_column_7_of_the_last_line_of_the_last_block():
    text = "\n".join(["Starting non-linear GS solver", _line(1, 9.0), _line(2, 8.0), " Timing: 1",
                      "Starting non-linear GS solver", _line(1, -12.5), _line(2, -12.34), _line(3, -1.2346E+03),
                      " Timing: 2"])
    h = FW.parse_history(text)
    assert h["ok"] and h["n"] == 3 and h["vsc_A"] == -1234.6
    bad = "\n".join(["Starting non-linear GS solver", _line(1, 1.0), _line(3, 2.0)])
    assert FW.parse_history(bad)["vsc_A"] is None
    assert FW.parse_history("no solver output")["n"] == 0


class _StubGS:
    """A stand-in for the solver object: ``solve`` writes ``text`` to file descriptor 1 (where the Fortran solver
    writes; nothing when its output is buffered) and returns without an error."""

    def __init__(self, text):
        self.text = text
        self.settings = SimpleNamespace(nl_tol=1e-6, urf=0.2, maxits=40)
        self.ffp_scale = 1.0

    def set_targets(self, **kw):
        pass

    def set_profiles(self, **kw):
        pass

    def solve(self, return_its=False):
        if self.text:
            os.write(1, self.text.encode())
        return (0, 3) if return_its else 0

    def get_stats(self):
        return {"Ip": 1.0e6, "l_i": 0.9, "q_0": 1.1}

    def copy_eq(self):
        return None

    def replace_eq(self, snap):
        pass


def _stub_forward_model(text):
    fm = object.__new__(FW.ForwardModel)
    fm.settings = FW.ForwardSettings(verify=False)
    fm.gs = _StubGS(text)
    fm._apply_rows = lambda: None
    fm.defaults = dict(nl_tol=1e-6, urf=0.2, maxits=40)
    fm.expected_settings = dict(fm.defaults)
    fm.cut_cells = dict(in_effect=None, provided=False)
    fm.n_solves, fm.iteration_log, fm.history = 0, [], []
    fm.case_id = "stub"
    fm.case = EDGE.EdgeTaperCase(case_id="stub")
    fm.ip_measured, fm.F0 = 1.0e6, 1.0
    fm.solver_build = dict(liboftpy_sha256="stub")
    return fm


def _solve(fm):
    x = np.linspace(0.0, 1.0, 11)
    return fm.solve((x, 1.0e6 * (1.0 - x ** 2)), (x, 1.0e4 * (1.0 - 0.9 * x ** 2)))


def test_solve_without_solver_output_is_a_recorded_failure_not_a_key_error():
    """Buffered Fortran output: the solver returns without an error but nothing of its output is captured.  The solve
    is a failure with the reason stated (status ``failed``, the attempt's ``error``); no VSC current is guessed."""
    res = _solve(_stub_forward_model(""))
    assert res.status == "failed" and res.converged is False and res.verified is None
    att = res.record["attempts"]
    assert len(att) == 1 and att[0]["error"] == FW.NO_SOLVER_HISTORY and att[0]["vsc_A"] is None
    assert att[0]["history_consistent"] is False and res.record["vsc_A"] is None
    pol = res.record["policy"]
    assert pol["status"] == EDGE.STATUS_FAILED and pol["attempts"][0]["info"]["error"] == FW.NO_SOLVER_HISTORY
    assert pol["attempts"][0]["detector"]["verdict"] == "insufficient_history" and pol["taper_applied"] is False
    assert "vsc_A" in FW.NO_SOLVER_HISTORY and "GFORTRAN_UNBUFFERED_PRECONNECTED" in FW.NO_SOLVER_HISTORY
    h = FW.parse_history("")
    assert h["n_blocks"] == 0 and h["vsc_A"] is None and h["o_point_tail"] == []


def test_solve_with_its_solver_output_is_accepted_with_the_vsc_current():
    """The same stub with the output captured (the control of the test above): accepted, the VSC current read."""
    text = "\n".join(["Starting non-linear GS solver", _line(1, -10.0), _line(2, -11.0), _line(3, -12.34), ""])
    res = _solve(_stub_forward_model(text))
    assert res.status == EDGE.STATUS_CONVERGED and res.converged is True
    att = res.record["attempts"]
    assert att[0]["error"] is None and att[0]["history_consistent"] is True
    assert res.record["vsc_A"] == -12.34


def test_fit_record_helpers_without_the_methods_or_the_table():
    class Bare:
        pass
    assert FIT._coil_currents(Bare(), None)["computed"] is False
    assert FIT._radial_coordinate(Bare(), Bare())["computed"] is False

    class WithMethod:
        def midplane_radius_record(self, table):
            return dict(table=table)
    assert FIT._radial_coordinate(WithMethod(), Bare()) == dict(table=None)


def test_separatrix_te_record_reads_the_profile_where_our_separatrix_sits():
    """``fit.separatrix_te_record``: T_e of the kinetic inputs at our separatrix radius on the producing equilibrium's
    midplane table; a record, no extrapolation, never raises."""
    from scipy.interpolate import PchipInterpolator

    class Kin:
        coordinate = "rho_tor_norm"
        x = np.linspace(0.0, 1.0, 101)

        def _f(self, name):
            y = dict(te=1000.0 * (1.0 - self.x) + 80.0, ne=5.0e19 * (1.0 - 0.9 * self.x))[name]
            return PchipInterpolator(self.x, y)

    kin = Kin()
    psi = np.linspace(0.0, 1.0, 51)
    kin.midplane_radius = dict(psi_n=psi, r_m=1.70 + 0.55 * np.sqrt(psi), rho_tor_norm=np.sqrt(psi), source="toy")
    rad = dict(computed=True, edge_r_ours_m=2.25 - 0.011)            # our separatrix 11 mm inside the table's
    rec = FIT.separatrix_te_record(rad, kin)
    rho = (2.25 - 0.011 - 1.70) / 0.55
    assert rec["computed"] and rec["inside_table"]
    assert rec["separatrix_minus_table_edge_mm"] == pytest.approx(-11.0, abs=1e-6)
    assert rec["coordinate_at_our_separatrix"] == pytest.approx(rho, abs=2e-4)
    assert rec["te_eV"] == pytest.approx(1000.0 * (1.0 - rho) + 80.0, rel=2e-3)
    assert rec["te_table_edge_eV"] == pytest.approx(80.0) and rec["te_over_table_edge"] > 1.2
    out = FIT.separatrix_te_record(dict(computed=True, edge_r_ours_m=2.26), kin)   # outside the table: not extrapolated
    assert out["computed"] and out["inside_table"] is False and out["te_eV"] is None
    del kin.midplane_radius["rho_tor_norm"]                         # no table coordinate for rho inputs: stated
    assert FIT.separatrix_te_record(rad, kin)["computed"] is False
    assert FIT.separatrix_te_record(dict(computed=False), kin)["computed"] is False
    assert FIT.separatrix_te_record(rad, object())["computed"] is False
