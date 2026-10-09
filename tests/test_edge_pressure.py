"""The pressure handed to the solver: one helper, two settings
(``bouquet.edge_pressure``).

* THE PRE-CHANGE SETTINGS ARE THE OLD ARRAYS, BIT FOR BIT (frozen-copy
  pattern): at ``edge_pprime_pin=True, separatrix_pressure="legacy"``
  (``EP.PRE_CHANGE_EDGE_PRESSURE``) the helper's ``P'`` node values, its
  ``pp_prof`` dict and its axis target equal the inline statements every
  site carried before the helper existed (frozen below, verbatim), on
  uniform and non-uniform grids; the engine backend hands the solver exactly
  those; the engine draw's first-pass pressure-term shift is unchanged.
* THE DEFAULT IS ``"offset"`` (owner-approved physics change, 2026-10-02):
  the defaults are NOT the pre-change settings; they differ from them in
  the axis target only (``p_axis - p_sep``), never in a ``P'`` node, and
  coincide with them when ``p_sep = 0``.  A stored config that predates the
  setting loads as ``"legacy"``.  (That every legacy solve path is the
  frozen pre-change code with the helper written back inline is
  ``tests/test_edge_pressure_legacy_ast.py`` and
  ``tests/test_engine_draws_legacy_ast.py``.)
* the two settings are validated loudly, everywhere they can be set;
* ``edge_pprime_pin=False`` keeps the profile's own derivative at the last
  node and changes nothing else; ``separatrix_pressure="offset"`` passes
  ``p_axis - p_sep`` and changes no ``P'`` node;
* the reporting formulas (:func:`bouquet.edge_pressure.pressure_frames`):
  with ``p_sep = 0`` both frames ARE the solver's own stats; with ``p_sep``
  they are the solver's own formulas evaluated with ``int p dV + p_sep V``;
* the zero-perturbation identity of the engine draw holds by construction
  under every combination of the two settings;
* delivery: a written g-file is asked to carry each draw's OWN ``p_sep``
  under ``"offset"`` and nothing under ``"legacy"``; the record is archived
  on the baseline and on every draw.

Solver-free; synthetic inputs only.
"""
import contextlib
import io
import os
import warnings

import numpy as np
import pytest

import _engine_toy as T
import test_engine_draws as TD
from bouquet import edge_pressure as EP
from bouquet import engine_draws as ED
from bouquet.config import GenerationConfig
from bouquet.utils import pchip_derivative

COMBOS = [(True, "legacy"), (True, "offset"), (False, "legacy"),
          (False, "offset")]


# ---------------------------------------------------------------------------
#  the frozen inline statements (verbatim from the pre-helper sites)
# ---------------------------------------------------------------------------
def _frozen_pp_dict(psi_N, pres_tmp, _pr):
    _pp = {"type": "linterp",
           "y": pchip_derivative(psi_N, pres_tmp) / _pr, "x": psi_N}
    _pp["y"][-1] = 0.0
    return _pp


def _frozen_pp_array(psi_N, pres_tmp, psi_range):
    pprime_tmp = pchip_derivative(psi_N, pres_tmp) / psi_range
    pprime_tmp[-1] = 0.0
    return pprime_tmp


def _frozen_pax(pressure):
    return float(pressure[0])


def _profiles():
    out = []
    for n, grid in ((65, "uniform"), (129, "uniform"), (101, "packed")):
        x = np.linspace(0.0, 1.0, n)
        if grid == "packed":
            x = 1.0 - (1.0 - x) ** 1.7
            x[0], x[-1] = 0.0, 1.0
        ped = 0.5 * (1.0 - np.tanh((x - 0.95) / 0.03))
        for p in (6.0e4 * (1.0 - x ** 2) ** 1.5,                  # zero edge
                  5.0e4 * (1.0 - x ** 2) ** 2 + 2.5e3,             # p_sep 5 %
                  3.0e4 * (1.0 - 0.6 * x ** 2) * ped + 4.0e2,      # pedestal
                  T.pressure(x)):
            out.append((x, p))
    return out


#: every way of handing the helper the PRE-CHANGE settings (pin on,
#: "legacy"); since 2026-10-02 these are not the defaults
def _pre_change_edges():
    return (EP.EdgePressure.pre_change(), EP.EdgePressure(True, "legacy"),
            GenerationConfig(separatrix_pressure="legacy"),
            dict(EP.PRE_CHANGE_EDGE_PRESSURE),
            dict(separatrix_pressure="legacy"))


@pytest.mark.parametrize("case", range(len(_profiles())))
@pytest.mark.parametrize("psi_range", [0.2731, -0.2731, 1.0])
def test_the_pre_change_settings_are_the_inline_statements_bit_for_bit(
        case, psi_range):
    x, p = _profiles()[case]
    for edge in _pre_change_edges():
        y = EP.solver_pprime(x, p, psi_range, edge)
        ref = _frozen_pp_array(x, p, psi_range)
        assert y.dtype == ref.dtype and np.array_equal(y, ref)
        assert y[-1] == 0.0
        d = EP.solver_pp_profile(x, p, psi_range, edge)
        dref = _frozen_pp_dict(x, p, psi_range)
        assert list(d) == list(dref) and d["type"] == dref["type"]
        assert np.array_equal(d["y"], dref["y"]) and d["x"] is x
        pax = EP.solver_pax(p, edge)
        assert type(pax) is float and pax == _frozen_pax(p)
        assert EP.solver_pressure(p, edge) is p
        assert EP.applied_offset(p, edge) == 0.0
    # the same through the settings object's methods
    e = EP.EdgePressure.pre_change()
    assert np.array_equal(e.pprime(x, p, psi_range),
                          _frozen_pp_array(x, p, psi_range))
    assert e.pax(p) == _frozen_pax(p) and e.p_offset(p) == 0.0
    assert e.is_pre_change and not e.offset and not e.is_default


# ---------------------------------------------------------------------------
#  the settings
# ---------------------------------------------------------------------------
def test_the_defaults_are_pin_on_and_offset_not_the_pre_change_settings():
    g = GenerationConfig()
    assert g.edge_pprime_pin is True and g.separatrix_pressure == "offset"
    assert EP.EDGE_PRESSURE_DEFAULTS == dict(edge_pprime_pin=True,
                                             separatrix_pressure="offset")
    assert EP.PRE_CHANGE_EDGE_PRESSURE == dict(edge_pprime_pin=True,
                                               separatrix_pressure="legacy")
    assert EP.resolve_edge_pressure(g) == EP.EdgePressure()
    for d in (EP.resolve_edge_pressure(None), EP.EdgePressure(),
              EP.resolve_edge_pressure(g), EP.resolve_edge_pressure({})):
        assert d.record() == EP.EDGE_PRESSURE_DEFAULTS
        assert d.is_default and d.offset and not d.is_pre_change
    pc = EP.EdgePressure.pre_change()
    assert pc.record() == EP.PRE_CHANGE_EDGE_PRESSURE
    assert pc.is_pre_change and not pc.is_default and not pc.offset
    assert EP.resolve_edge_pressure(
        GenerationConfig(separatrix_pressure="legacy")) == pc


@pytest.mark.parametrize("case", range(len(_profiles())))
def test_the_default_is_offset_on_the_arrays(case):
    """At the defaults the axis target is ``p_axis - p_sep`` and ``p_sep``
    is added back; no ``P'`` node differs from the pre-change settings', and
    with ``p_sep = 0`` neither does the axis target."""
    x, p = _profiles()[case]
    pc = EP.EdgePressure.pre_change()
    p_sep = float(p[-1])
    for edge in (None, EP.EdgePressure(), GenerationConfig(),
                 dict(EP.EDGE_PRESSURE_DEFAULTS)):
        assert np.array_equal(EP.solver_pprime(x, p, 0.2731, edge),
                              EP.solver_pprime(x, p, 0.2731, pc))
        assert EP.solver_pax(p, edge) == float(p[0]) - p_sep
        assert EP.applied_offset(p, edge) == p_sep
        assert np.array_equal(EP.solver_pressure(p, edge), p - p_sep)
        assert EP.describe(edge, p)["separatrix_pressure"] == "offset"
        if p_sep == 0.0:
            assert EP.solver_pax(p, edge) == EP.solver_pax(p, pc)
            assert EP.lcfs_kwargs(EP.applied_offset(p, edge)) == {}
        else:
            assert EP.solver_pax(p, edge) != EP.solver_pax(p, pc)
            assert EP.lcfs_kwargs(EP.applied_offset(p, edge)) == \
                {"lcfs_pressure": p_sep}
    # the profile set exercises both
    assert any(q[-1] == 0.0 for _x, q in _profiles())
    assert any(q[-1] > 0.0 for _x, q in _profiles())


def test_a_stored_config_without_the_setting_loads_as_legacy():
    """A config written before the setting existed was produced with the
    full axis pressure as the target: it reloads with ``"legacy"`` (and says
    so), so replaying it reproduces what it recorded; a current config
    carries the field and keeps it."""
    import bouquet as bq
    from bouquet.config import BouquetConfig
    ex = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples", "D3D-like")
    b = bq.Bouquet.from_geqdsk(
        os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1)
    d = b.config.to_dict()
    assert d["generation"]["separatrix_pressure"] == "offset"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert BouquetConfig.from_dict(d).generation.separatrix_pressure \
            == "offset"
    del d["generation"]["separatrix_pressure"]
    with pytest.warns(UserWarning, match="separatrix_pressure='legacy'"):
        back = BouquetConfig.from_dict(d)
    assert back.generation.separatrix_pressure == "legacy"
    assert EP.resolve_edge_pressure(back.generation).is_pre_change


@pytest.mark.parametrize("bad", [dict(edge_pprime_pin=1),
                                 dict(edge_pprime_pin="on"),
                                 dict(edge_pprime_pin=None),
                                 dict(separatrix_pressure="Offset"),
                                 dict(separatrix_pressure=None),
                                 dict(separatrix_pressure=True),
                                 dict(separatrix_pressure="thermal")])
def test_a_malformed_setting_is_refused_by_name(bad):
    name = list(bad)[0]
    with pytest.raises(ValueError, match=name):
        GenerationConfig(**bad)
    with pytest.raises(ValueError, match=name):
        EP.resolve_edge_pressure(bad)
    with pytest.raises(ValueError, match=name):
        EP.EdgePressure(**bad)
    # set after construction: refused where the settings are read
    g = GenerationConfig()
    setattr(g, name, bad[name])
    with pytest.raises(ValueError, match=name):
        EP.resolve_edge_pressure(g)
    x, p = _profiles()[1]
    with pytest.raises(ValueError, match=name):
        EP.solver_pprime(x, p, 1.0, g)
    with pytest.raises(ValueError, match=name):
        EP.solver_pax(p, g)


def test_an_unknown_settings_key_is_refused():
    with pytest.raises(ValueError, match="unknown key"):
        EP.resolve_edge_pressure(dict(edge_pin=False))


def test_the_whole_config_validates_and_round_trips_the_settings(tmp_path):
    import bouquet as bq
    from bouquet.config import BouquetConfig
    ex = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples", "D3D-like")
    b = bq.Bouquet.from_geqdsk(
        os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1)
    g = b.config.generation
    g.edge_pprime_pin, g.separatrix_pressure = False, "offset"
    d = b.config.to_dict()
    assert d["generation"]["edge_pprime_pin"] is False
    assert d["generation"]["separatrix_pressure"] == "offset"
    back = BouquetConfig.from_dict(d)
    assert back.generation.edge_pprime_pin is False
    assert back.generation.separatrix_pressure == "offset"
    # a malformed value is refused when the whole config is (re)built
    d["generation"]["separatrix_pressure"] = "shifted"
    with pytest.raises(ValueError, match="separatrix_pressure"):
        BouquetConfig.from_dict(d)
    d["generation"]["separatrix_pressure"] = "offset"
    d["generation"]["edge_pprime_pin"] = "off"
    with pytest.raises(ValueError, match="edge_pprime_pin"):
        BouquetConfig.from_dict(d)


def test_the_engine_settings_and_record_carry_the_settings():
    s = T.settings()
    assert s["edge_pressure"] == EP.EDGE_PRESSURE_DEFAULTS
    s = T.settings(edge_pprime_pin=False, separatrix_pressure="offset")
    assert s["edge_pressure"] == dict(edge_pprime_pin=False,
                                      separatrix_pressure="offset")
    eng, res, rec, b = TD._recon(edge_pprime_pin=False,
                                 separatrix_pressure="offset")
    ep = rec["edge_pressure"]
    p = np.asarray(eng.c.pressure, float)
    assert ep["edge_pprime_pin"] is False
    assert ep["separatrix_pressure"] == "offset"
    assert ep["p_sep"] == float(p[-1]) == ep["p_sep_applied"]
    assert ep["p_axis"] == float(p[0])
    assert ep["pax_target"] == float(p[0]) - float(p[-1])
    assert rec["settings"]["edge_pressure"] == s["edge_pressure"]
    # the defaults (offset, pin on): recorded and applied
    eng, res, rec, b = TD._recon()
    ep = rec["edge_pressure"]
    p = np.asarray(eng.c.pressure, float)
    assert ep["separatrix_pressure"] == "offset" and ep["edge_pprime_pin"]
    assert ep["p_sep"] == float(p[-1]) == ep["p_sep_applied"] > 0.0
    assert ep["pax_target"] == float(p[0]) - float(p[-1])
    # the pre-change settings: recorded too, nothing applied
    eng, res, rec, b = TD._recon(separatrix_pressure="legacy")
    ep = rec["edge_pressure"]
    assert ep["p_sep"] == float(np.asarray(eng.c.pressure)[-1]) > 0.0
    assert ep["p_sep_applied"] == 0.0
    assert ep["pax_target"] == ep["p_axis"]


# ---------------------------------------------------------------------------
#  what each setting changes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", range(len(_profiles())))
def test_pin_off_keeps_the_profiles_own_last_derivative(case):
    x, p = _profiles()[case]
    on = EP.solver_pprime(x, p, 0.3, dict(edge_pprime_pin=True))
    off = EP.solver_pprime(x, p, 0.3, dict(edge_pprime_pin=False))
    raw = pchip_derivative(x, p) / 0.3
    assert np.array_equal(off, raw)
    assert np.array_equal(on[:-1], off[:-1]) and on[-1] == 0.0
    # the axis target does not depend on the pin (either separatrix setting)
    assert EP.solver_pax(p, dict(edge_pprime_pin=False,
                                 separatrix_pressure="legacy")) == float(p[0])
    for sep in ("legacy", "offset"):
        assert EP.solver_pax(p, dict(edge_pprime_pin=False,
                                     separatrix_pressure=sep)) == \
            EP.solver_pax(p, dict(edge_pprime_pin=True,
                                  separatrix_pressure=sep))


@pytest.mark.parametrize("case", range(len(_profiles())))
@pytest.mark.parametrize("pin", [True, False])
def test_offset_moves_the_axis_target_only(case, pin):
    x, p = _profiles()[case]
    leg = dict(edge_pprime_pin=pin, separatrix_pressure="legacy")
    off = dict(edge_pprime_pin=pin, separatrix_pressure="offset")
    assert np.array_equal(EP.solver_pprime(x, p, 0.3, leg),
                          EP.solver_pprime(x, p, 0.3, off))
    assert EP.solver_pax(p, off) == float(p[0]) - float(p[-1])
    assert EP.applied_offset(p, off) == float(p[-1])
    assert EP.separatrix_pressure_of(p) == float(p[-1])
    sp = EP.solver_pressure(p, off)
    assert sp[0] == EP.solver_pax(p, off) and sp[-1] == 0.0
    assert np.array_equal(sp, p - p[-1])
    d = EP.describe(off, p)
    assert d["p_sep"] == d["p_sep_applied"] == float(p[-1])
    assert d["pax_target"] == float(p[0]) - float(p[-1])
    d = EP.describe(leg, p)
    assert d["p_sep"] == float(p[-1]) and d["p_sep_applied"] == 0.0


def test_offset_refuses_a_target_that_is_not_positive():
    x = np.linspace(0.0, 1.0, 33)
    off = dict(separatrix_pressure="offset")
    with pytest.raises(ValueError, match="not positive"):
        EP.solver_pax(np.full(33, 1.0e3), off)
    with pytest.raises(ValueError, match="not positive"):
        EP.solver_pressure(1.0e3 * (0.5 + x), off)
    p = 1.0e3 * (1.0 - x)
    p[-1] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        EP.solver_pax(p, off)
    # legacy never looks at the edge value; the default (offset) does
    assert EP.solver_pax(p, dict(separatrix_pressure="legacy")) == 1.0e3
    with pytest.raises(ValueError, match="not finite"):
        EP.solver_pax(p, None)


def test_offset_refuses_a_negative_separatrix_pressure():
    """A negative p_sep (e.g. a legacy draw's perturbed edge n_e or T_e
    below zero) used to RAISE the axis target and write a negative boundary
    PRES (review finding 14): refused under "offset"; "legacy" never reads
    the edge value."""
    x = np.linspace(0.0, 1.0, 33)
    p = 1.0e4 * (1.0 - x) - 50.0                 # p_sep = -50 Pa
    for fn in (EP.applied_offset, EP.solver_pax, EP.solver_pressure):
        with pytest.raises(ValueError, match="negative"):
            fn(p, dict(separatrix_pressure="offset"))
    assert EP.solver_pax(p, dict(separatrix_pressure="legacy")) == p[0]
    p0 = 1.0e4 * (1.0 - x)                       # p_sep = 0 exactly: fine
    assert EP.applied_offset(p0, None) == 0.0
    assert EP.solver_pax(p0, None) == p0[0]


# ---------------------------------------------------------------------------
#  the reporting formulas
# ---------------------------------------------------------------------------
_MU0 = 4.0e-7 * np.pi


def _solver_stats(pvol, *, vol=19.3, Ip=1.2e6, dl=7.9, F0=3.4, R_geo=1.69,
                  a_geo=0.59, P_ax=5.8e4):
    """The solver's own formulas for the pressure-integral statistics."""
    bt = 100.0 * (2.0 * pvol * _MU0 / vol) / (F0 / R_geo) ** 2
    return dict(vol=vol, P_ax=P_ax, W_MHD=pvol * 1.5,
                beta_pol=100.0 * (2.0 * pvol * _MU0 / vol)
                / (Ip * _MU0 / dl) ** 2,
                beta_tor=bt, beta_n=bt * a_geo * (F0 / R_geo) / (Ip / 1e6),
                l_i=0.9, q_95=4.2)


def test_with_no_separatrix_pressure_both_frames_are_the_solvers_stats():
    st = _solver_stats(3.1e5)
    fr = EP.pressure_frames(st, 0.0)
    assert fr["factor"] == 1.0 and fr["p_sep"] == 0.0
    for k in ("W_MHD", "beta_pol", "beta_tor", "beta_n", "P_ax"):
        assert fr["solver"][k] == st[k] == fr["full"][k]
    assert fr["volume"] == st["vol"]


@pytest.mark.parametrize("p_sep", [382.6, 2.9e3, -150.0])
def test_the_full_frame_is_the_solvers_formulas_with_p_sep_added(p_sep):
    pvol, vol = 3.1e5, 19.3
    st = _solver_stats(pvol, vol=vol)
    fr = EP.pressure_frames(st, p_sep)
    want = _solver_stats(pvol + p_sep * vol, vol=vol)
    assert fr["full"]["W_MHD"] == pytest.approx(st["W_MHD"]
                                                + 1.5 * p_sep * vol,
                                                rel=1e-14)
    for k in ("W_MHD", "beta_pol", "beta_tor", "beta_n"):
        assert fr["full"][k] == pytest.approx(want[k], rel=1e-13)
        assert fr["solver"][k] == st[k]
    assert fr["full"]["P_ax"] == st["P_ax"] + p_sep
    assert fr["p_sep_volume"] == p_sep * vol
    assert fr["factor"] == pytest.approx(1.0 + p_sep * vol / pvol, rel=1e-15)


def test_the_frames_need_a_positive_pressure_integral():
    st = _solver_stats(0.0)
    assert EP.pressure_frames(st, 0.0)["full"]["W_MHD"] == 0.0
    with pytest.raises(ValueError, match="not positive"):
        EP.pressure_frames(st, 100.0)


def test_the_inputs_frames_are_those_of_its_own_edge_pressure():
    V, pvol, pe = 19.3, 3.1e5, 2.5e3
    fr = EP.input_pressure_frames(V, pvol, pe, betas=dict(beta_n=2.0))
    assert fr["full"]["W_MHD"] == 1.5 * pvol
    assert fr["solver"]["W_MHD"] == pytest.approx(1.5 * (pvol - pe * V))
    assert fr["solver"]["beta_n"] == pytest.approx(
        2.0 * (pvol - pe * V) / pvol)
    z = EP.input_pressure_frames(V, pvol, 0.0, betas=dict(beta_n=2.0))
    assert z["solver"] == z["full"]


# ---------------------------------------------------------------------------
#  the engine backend
# ---------------------------------------------------------------------------
class _RecordingGS:
    """Records what a backend hands the solver."""

    def __init__(self, dpsi=0.2731):
        self.calls = []
        self._dpsi = dpsi

    @property
    def psi_bounds(self):
        return [-self._dpsi, 0.0]

    def set_targets(self, Ip=None, pax=None):
        self.calls.append(("targets", Ip, pax))

    def set_profiles(self, pp_prof=None, ffp_prof=None):
        self.calls.append(("profiles", pp_prof, ffp_prof))

    def solve(self):
        self.calls.append(("solve",))


def _backend(edge):
    from types import SimpleNamespace
    from bouquet.engine import TokaMakerBackend
    x, p = _profiles()[2]
    gs = _RecordingGS()
    c = SimpleNamespace(psi_N=x, pressure=p, Ip=1.2e6, kinetics=None)
    return TokaMakerBackend(gs, c, edge_pressure=edge), gs, x, p


@pytest.mark.parametrize("edge", [EP.EdgePressure.pre_change(),
                                  dict(EP.PRE_CHANGE_EDGE_PRESSURE),
                                  dict(separatrix_pressure="legacy")])
def test_the_engine_backend_at_the_pre_change_settings_hands_the_old_arrays(
        edge):
    b, gs, x, p = _backend(edge)
    req = 1.0e6 * (1.0 - x ** 2)
    b.solve(req, n_passes=2)
    psi_range = gs.psi_bounds[1] - gs.psi_bounds[0]
    ref = _frozen_pp_array(x, p, psi_range)
    targets = [c for c in gs.calls if c[0] == "targets"]
    profs = [c for c in gs.calls if c[0] == "profiles"]
    assert len(targets) == len(profs) == 2 and b.n_solves == 2
    for t, pr in zip(targets, profs):
        assert t[1] == 1.2e6 and t[2] == _frozen_pax(p)
        assert type(t[2]) is float
        assert pr[1]["type"] == "linterp" and pr[1]["x"] is b.psi
        assert np.array_equal(pr[1]["y"], ref)
        assert pr[2]["type"] == "jphi-linterp"
        assert np.array_equal(pr[2]["y"], req)
    assert b.p_sep() == 0.0


@pytest.mark.parametrize("pin,sep", COMBOS)
def test_the_engine_backend_follows_the_settings(pin, sep):
    b, gs, x, p = _backend(dict(edge_pprime_pin=pin, separatrix_pressure=sep))
    b.solve(1.0e6 * (1.0 - x ** 2))
    psi_range = gs.psi_bounds[1] - gs.psi_bounds[0]
    raw = pchip_derivative(x, p) / psi_range
    y = [c for c in gs.calls if c[0] == "profiles"][0][1]["y"]
    assert np.array_equal(y[:-1], raw[:-1])
    assert y[-1] == (0.0 if pin else raw[-1])
    pax = [c for c in gs.calls if c[0] == "targets"][0][2]
    assert pax == (float(p[0]) - float(p[-1]) if sep == "offset"
                   else float(p[0]))
    assert b.p_sep() == (float(p[-1]) if sep == "offset" else 0.0)
    # a draw's own pressure: its own p_sep
    b.set_inputs(pressure=p + 123.0, kinetics=None)
    assert b.p_sep() == (float(p[-1] + 123.0) if sep == "offset" else 0.0)


# ---------------------------------------------------------------------------
#  the engine draws
# ---------------------------------------------------------------------------
def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.mark.parametrize("sep", ["legacy", "offset"])
def test_the_draws_first_pass_pressure_shift_is_unchanged(sep):
    """At the pre-change settings and at the defaults ("offset" changes the
    axis target only, never a ``P'``)."""
    eng, res, rec, b = TD._recon(separatrix_pressure=sep)
    ctx = TD._ctx(eng, res)
    assert ctx.edge == EP.EdgePressure(True, sep)
    assert ctx.edge == (EP.EdgePressure.pre_change() if sep == "legacy"
                        else EP.EdgePressure())
    p = np.asarray(eng.c.pressure, float)
    # exactly G* at zero perturbation
    g0 = ctx.geom_for_pressure(p)
    assert np.array_equal(g0["pprime"], ctx.geom["pprime"])
    # the frozen expression for a perturbed pressure
    p2 = p * (1.0 + 0.05 * np.sin(3.0 * ctx.psi))
    dPq = np.interp(ctx._psi_q, ctx.psi, pchip_derivative(ctx.psi, p2))
    ref = np.asarray(ctx.geom["pprime"], dtype=float) \
        + ctx.sigma_p * (dPq - ctx.dPq_star)
    assert np.array_equal(ctx.geom_for_pressure(p2)["pprime"], ref)
    assert np.array_equal(ctx.dPq_star, np.interp(
        ctx._psi_q, ctx.psi, pchip_derivative(ctx.psi, p)))


@pytest.mark.parametrize("pin,sep", COMBOS)
def test_the_zero_perturbation_identity_holds_under_every_combination(
        pin, sep):
    eng, res, rec, b = TD._recon(edge_pprime_pin=pin,
                                 separatrix_pressure=sep)
    ctx = TD._ctx(eng, res)
    assert ctx.edge == EP.EdgePressure(pin, sep)
    out = _q(ED.run_draw, ctx, b, ctx.zero_inputs())
    r = out["record"]
    assert r["identity"]["pass1_request_bit_identical"] is True
    assert r["identity"]["pass1_request_max_abs_diff"] == 0.0
    np.testing.assert_array_equal(out["passes"].first_request,
                                  res["state"].request)
    v = _q(ED.verify_zero_perturbation, ctx, b)
    assert v["passed"] and v["request_bit_identical"]
    # a zero-perturbation draw's pressure is the contract's, so its p_sep is
    np.testing.assert_array_equal(ctx.zero_inputs().pressure,
                                  np.asarray(eng.c.pressure, float))
    assert ctx.edge.p_offset(ctx.zero_inputs().pressure) == \
        rec["edge_pressure"]["p_sep_applied"]


def _generate(tmp_path, monkeypatch, pin, sep, n=3):
    from _engine_fake_gs import FakeTokaMaker
    from bouquet.TokaMaker_interface import generate_bouquet
    from bouquet.utils import initialize_equilibrium_database
    os.makedirs(str(tmp_path), exist_ok=True)
    eng, res, rec, b = TD._recon(edge_pprime_pin=pin,
                                 separatrix_pressure=sep)
    ctx = TD._ctx(eng, res)
    unc = TD._unc(ctx)
    G = ED.GenerateEngineDraws(ctx, unc=unc, psi_pad=T.PAD)
    monkeypatch.setattr(ED, "tokamaker_backend",
                        lambda mygs, c, **kw: mygs.toy)
    fake = FakeTokaMaker(b)
    saves = []
    real_save = fake.save_eqdsk

    def _save(filename, **kw):
        saves.append(dict(kw))
        return real_save(filename, **kw)
    fake.save_eqdsk = _save
    h = str(tmp_path / "e2e")
    initialize_equilibrium_database(h)
    k = ctx.native
    diags = _q(
        generate_bouquet, fake, T.PSI, n, h, ctx.request, k["ne"], k["te"],
        k["ni"], k["ti"], unc["sigma_ne"], unc["sigma_te"], unc["sigma_ni"],
        unc["sigma_ti"], unc["sigma_jphi"], 0.3, 0.3, 0.25,
        float(eng.c.Ip), ctx.ref["l_i"], eng.c.kinetics["zeff"],
        input_jinductive=0.5 * ctx.request,
        baseline_j_BS=0.1 * ctx.request, l_i_tolerance=0.05,
        psi_pad=T.PAD, constrain_sawteeth=False, isolate_edge_jBS=False,
        jBS_scale_range=(0.99, 1.01), coil_drift=0.01,
        homotopy_passes=[(0.05, 0.1), (0.01, 0.01)], seed=12345,
        capture_live_eq=False, store_achieved_jphi=True,
        jbs_loop=G.loop_settings, rejection_log=[], draw_method=G,
        coil_filter="legacy")
    return diags, saves, h, eng


@pytest.mark.parametrize("pin,sep", COMBOS)
def test_a_written_gfile_carries_each_draws_own_separatrix_pressure(
        tmp_path, monkeypatch, pin, sep):
    import h5py
    diags, saves, h, eng = _generate(tmp_path, monkeypatch, pin, sep)
    assert len(diags) == 3
    p_bl = float(np.asarray(eng.c.pressure, float)[-1])
    with h5py.File(h + ".h5", "r") as hf:
        root = hf["scan/0"] if "scan" in hf else hf
        p_draw = [float(np.asarray(root[str(i)]["pressure"])[-1])
                  for i in range(3)]
    # the baseline re-save, then one save per draw
    assert len(saves) == 4
    if sep == "legacy":
        assert all("lcfs_pressure" not in kw for kw in saves)
    else:
        assert saves[0]["lcfs_pressure"] == p_bl
        for kw, pd in zip(saves[1:], p_draw):
            assert kw["lcfs_pressure"] == pd
        # each draw's own, not the baseline's
        assert len({p_bl, *p_draw}) == 4
    # the archived records
    bl = EP.load_record(h)
    assert bl["edge_pprime_pin"] is pin and bl["separatrix_pressure"] == sep
    assert bl["p_sep"] == p_bl
    assert bl["p_sep_applied"] == (p_bl if sep == "offset" else 0.0)
    for i, pd in enumerate(p_draw):
        r = EP.load_record(h, count=i)
        assert r["separatrix_pressure"] == sep and r["p_sep"] == pd
        assert r["p_sep_applied"] == (pd if sep == "offset" else 0.0)


def test_the_lcfs_keyword_is_absent_when_nothing_is_added_back():
    assert EP.lcfs_kwargs(0.0) == {}
    assert EP.lcfs_kwargs(382.6) == {"lcfs_pressure": 382.6}


def test_the_archive_record_reports_both_frames():
    x, p = _profiles()[1]
    st = _solver_stats(3.1e5)
    r = EP.archive_record(dict(separatrix_pressure="offset"), p, stats=st)
    assert r["p_sep_applied"] == float(p[-1])
    assert r["frames"]["full"]["W_MHD"] == pytest.approx(
        st["W_MHD"] + 1.5 * float(p[-1]) * st["vol"])
    assert r["frames"]["solver"]["W_MHD"] == st["W_MHD"]
    # the default is "offset": the same as the explicit setting
    assert EP.archive_record(None, p, stats=st) == r
    r = EP.archive_record(EP.EdgePressure.pre_change(), p, stats=st)
    assert r["p_sep_applied"] == 0.0
    assert r["frames"]["full"] == r["frames"]["solver"]
    # stats without a volume: recorded, not raised
    r = EP.archive_record(None, p, stats=dict(l_i=0.9))
    assert r["frames"] is None and "frames_error" in r


def test_plot_input_vs_recon_reports_the_full_pressure():
    """plot_input_vs_recon compared the input's FULL pressure with the
    solver-frame pressure (zero at psi_N = 1), so under "offset" a uniform
    p_sep gap looked like a reconstruction error (review finding 13): the
    solved pressure is now shown in the reported frame."""
    from types import SimpleNamespace
    from bouquet.plotting import _reported_pressure
    x = np.linspace(0.0, 1.0, 9)
    p_solver = 1.0e4 * (1.0 - x)
    bl = SimpleNamespace(edge_pressure=EP.describe(
        dict(separatrix_pressure="offset"), p_solver + 300.0))
    np.testing.assert_array_equal(_reported_pressure(bl, p_solver),
                                  p_solver + 300.0)
    for e in (EP.describe(dict(separatrix_pressure="legacy"),
                          p_solver + 300.0), None):
        np.testing.assert_array_equal(
            _reported_pressure(SimpleNamespace(edge_pressure=e), p_solver),
            p_solver)


@pytest.mark.parametrize("change", [dict(separatrix_pressure="legacy"),
                                    dict(edge_pprime_pin=False)])
def test_changing_the_edge_settings_after_the_baseline_is_refused(
        tmp_path, change, toy_bouquet_solver):
    """The legacy draws read the CONFIG's edge settings at generate(), the
    engine draws the reconstruction's: a setting changed after
    prepare_baseline() would mix pressure frames or be silently ignored
    (review finding 16) -- refused, naming both values."""
    b = TD._bq(tmp_path)
    b.setup_solver()
    TD._quiet(b.prepare_baseline)
    for k, v in change.items():
        setattr(b.config.generation, k, v)
    with pytest.raises(ValueError, match="differs from the baseline"):
        b.generate()
    with pytest.raises(ValueError, match="differs from the baseline"):
        b.verify_sigma0_consistency()


toy_bouquet_solver = TD.toy_bouquet_solver


def test_a_negative_separatrix_pressure_refuses_the_legacy_baseline(
        tmp_path, monkeypatch):
    """Disclosed 2026-10-06 (finding 4 of the second-pass review): the
    negative-p_sep refusal of c709aae reaches the BASELINE on the legacy
    path (the default until 2026-10-06), not only the draws.  Kept (owner
    rule: failures loud; a negative p_sep is unphysical input) and NAMED: the error says the
    baseline was refused, why, and how to build it as before.  Legacy IMAS
    baseline whose solve pressure at psi_N = 1 is made negative (the
    reader's baseline with a negative pressure offset); the refusal comes
    before any solve (a stand-in solver)."""
    import bouquet as bq
    import bouquet.baseline as B
    from _engine_fake_gs import FakeTokaMaker
    _ex = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "examples", "D3D-like")
    b = bq.Bouquet.from_imas(
        os.path.join(_ex, "D3Dlike_baseline_omas.json"),
        mesh=os.path.join(_ex, "DIIID_mesh.h5"), time=2.3043, n_draws=1,
        header=str(tmp_path / "bq"), reconstruction_engine="legacy")
    assert b.config.generation.reconstruction_engine == "legacy"
    assert b.config.generation.separatrix_pressure == "offset"
    real = B.resolve_baseline

    def resolve(config, mygs):
        bl = real(config, mygs)
        p_edge = 1.602176634e-19 * (bl.ne[-1] * bl.te[-1]
                                    + bl.ni[-1] * bl.ti[-1])
        # p_sep -> about -(edge pressure); the axis stays far above zero
        bl.p_diff = np.full(np.asarray(bl.psi_N_kinetic).size,
                            -3.0 * float(p_edge) - 100.0)
        return bl
    monkeypatch.setattr(B, "resolve_baseline", resolve)
    solves = []

    class _Fake(FakeTokaMaker):
        psi_bounds = (-1.0, 0.0)            # no toy state behind it

        def solve(self, *a, **k):
            solves.append(1)
            raise AssertionError("the refusal comes before any solve")
    b.mygs = _Fake(None)
    with pytest.raises(EP.NegativeSeparatrixPressure) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.prepare_baseline()
    msg = str(ei.value)
    assert msg.startswith("prepare_baseline REFUSED THE BASELINE (source "
                          "ImasSource, reconstruction_engine='legacy')")
    assert "pressure at the separatrix" in msg and "unphysical" in msg
    assert "separatrix_pressure='legacy'" in msg
    assert "is negative" in msg and "Pa)" in msg
    assert isinstance(ei.value, ValueError)
    assert solves == [] and b.baseline is None
