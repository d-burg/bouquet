"""Fast tests (no solver) of two records that steer nothing:

* the producing equilibrium's midplane table on the kinetic inputs (``assemble_kinetic_inputs(...,
  midplane_radius=...)``): kept on the result, stated in its record ("not supplied" without it), refused when it is
  not two strictly increasing finite arrays, and without any effect on the profiles (compared bit for bit);
* the vertical-stability coil current read from the solver's iteration lines (``forward.parse_history``: column 7
  of the last line of the last solver block), checked on constructed solver output;
* the fit's two record helpers when the forward model or the kinetic inputs do not provide what they read.
"""
import numpy as np
import pytest

from bouquet.kinrecon import fit as FIT
from bouquet.kinrecon import forward as FW
from bouquet.kinrecon import kinetics as KI


def _inputs():
    x = np.linspace(0.0, 1.0, 51)
    ne = 4e19 * (1.0 - 0.8 * x ** 2)
    te = 2e3 * (1.0 - 0.9 * x ** 2)
    return x, ne, te, 0.8 * te, dict(zeff_candidates={"vb": ("zeff", np.full_like(x, 1.8))}, zeff_source="vb",
                                     z_imp=6.0, n_fast="none")


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


def test_fit_record_helpers_without_the_methods_or_the_table():
    class Bare:
        pass
    assert FIT._coil_currents(Bare(), None)["computed"] is False
    assert FIT._radial_coordinate(Bare(), Bare())["computed"] is False

    class WithMethod:
        def midplane_radius_record(self, table):
            return dict(table=table)
    assert FIT._radial_coordinate(WithMethod(), Bare()) == dict(table=None)
