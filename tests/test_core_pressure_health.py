"""The core-pressure hollowness record describes; it never gates.

``physics.core_pressure_health`` measures how far a pressure profile rises
above its innermost-node (axis) value inside a stated core window, and over
what radial extent.  ``physics.core_pressure_hollow_record`` assembles those
measurements for the INPUT pressure (total and thermal species only) and the
ACHIEVED pressure of a converged equilibrium into the per-slice record that
``Baseline.core_pressure_hollow`` and the archived ``li_metrics`` carry.

Three properties are under test:

* the NUMBERS are right on synthetic profiles with known answers, including
  non-uniform grids, a grid whose first node is off-axis and a descending
  grid, and bad data reads "not evaluated" with a reason -- never a false
  "not hollow";
* ``is_hollow`` follows the documented reporting threshold on those numbers
  and nothing else; and
* the record is REPORT-ONLY: the pressure composition the solver receives is
  bit-identical with and without the record code (checked by executing the
  real source blocks with the record statements removed), the inputs are not
  written to, and no filter / in-spec / until-N code reads it.
"""
import ast
import copy
import inspect
import json
import textwrap
import warnings
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import bouquet.physics as physics
from bouquet.physics import (CORE_HOLLOW_MAX_REF_PSI_N, CORE_HOLLOW_RISE_FRAC,
                             CORE_PSI_N, CorePressureHollowWarning,
                             core_pressure_health, core_pressure_hollow_record)


def _psi(n=601):
    return np.linspace(0.0, 1.0, n)


def _monotone(psi_N, p0=1.0e5):
    """Core-peaked, strictly decreasing off-axis."""
    return p0 * (1.0 - psi_N ** 2) ** 2 + 1.0e3


def _hollow(psi_N, p0=1.0e5):
    """p = p0 (1 - psi)^2 (1 + 4 psi): dp/dpsi = p0 (1 - psi)(2 - 12 psi), so
    the pressure rises from the axis to a maximum at psi = 1/6 of
    p0 * 125/108.  Known answers: rise_frac = 17/108, rise_extent = 1/6."""
    return p0 * (1.0 - psi_N) ** 2 * (1.0 + 4.0 * psi_N)


HOLLOW_RISE = 17.0 / 108.0
HOLLOW_PSI_MAX = 1.0 / 6.0


# ---------------------------------------------------------------------
#  The measure
# ---------------------------------------------------------------------

def test_monotone_profile_is_not_hollow():
    psi_N = _psi()
    rec = core_pressure_health(psi_N, _monotone(psi_N))
    assert rec["evaluated"] is True and rec["reason"] is None
    assert rec["is_hollow"] is False
    assert rec["rise_frac"] == 0.0
    assert rec["psi_N_of_max"] == 0.0
    assert rec["rise_extent"] == 0.0
    assert rec["positive_gradient_extent"] == 0.0
    assert rec["positive_gradient_psi_N_max"] is None
    assert rec["n_core_nodes"] == int(np.count_nonzero(psi_N <= CORE_PSI_N))


def test_hollow_profile_is_hollow_with_the_known_numbers():
    psi_N = _psi(601)                      # psi = 1/6 is node 100
    rec = core_pressure_health(psi_N, _hollow(psi_N))
    assert rec["is_hollow"] is True
    assert rec["rise_frac"] == pytest.approx(HOLLOW_RISE, rel=1e-12)
    assert rec["psi_N_of_max"] == pytest.approx(HOLLOW_PSI_MAX, abs=1e-12)
    assert rec["rise_extent"] == pytest.approx(HOLLOW_PSI_MAX, abs=1e-12)
    # every interval inside (0, 1/6) climbs, none outside it does
    assert rec["positive_gradient_extent"] == pytest.approx(HOLLOW_PSI_MAX,
                                                            abs=1e-12)
    assert rec["positive_gradient_psi_N_max"] == pytest.approx(HOLLOW_PSI_MAX,
                                                               abs=1e-12)
    assert rec["p_ref"] == pytest.approx(1.0e5, rel=1e-15)
    assert rec["p_core_max"] == pytest.approx(1.0e5 * 125.0 / 108.0, rel=1e-12)


def test_single_node_noise_wiggle_is_recorded_but_below_the_threshold():
    """A 1e-3 relative bump on one node near the axis of a monotone profile:
    the numbers see it, the documented 1 % threshold does not call it
    hollow."""
    psi_N = _psi(601)
    p = _monotone(psi_N)
    p[2] = p[2] * (1.0 + 1.0e-3)
    rec = core_pressure_health(psi_N, p)
    expected = (p[2] - p[0]) / p[0]
    assert 0.0 < expected < 1.0e-3
    assert rec["rise_frac"] == pytest.approx(expected, rel=1e-12)
    assert rec["psi_N_of_max"] == pytest.approx(psi_N[2], abs=1e-15)
    # the one climbing secant (node 1 -> node 2) is counted
    assert rec["positive_gradient_extent"] == pytest.approx(
        psi_N[2] - psi_N[1], rel=1e-12)
    assert rec["is_hollow"] is False
    assert CORE_HOLLOW_RISE_FRAC == 0.01
    # the same numbers with a threshold under the wiggle flip only the boolean
    low = core_pressure_health(psi_N, p, rise_threshold=1.0e-4)
    assert low["is_hollow"] is True
    for k in ("rise_frac", "psi_N_of_max", "rise_extent",
              "positive_gradient_extent", "p_ref", "p_core_max"):
        assert low[k] == rec[k]


def test_flat_core_is_not_hollow():
    psi_N = _psi(401)
    p = np.where(psi_N <= 0.6, 5.0e4, 5.0e4 * (1.0 - (psi_N - 0.6)))
    rec = core_pressure_health(psi_N, p)
    assert rec["evaluated"] is True
    assert rec["is_hollow"] is False
    assert rec["rise_frac"] == 0.0
    assert rec["psi_N_of_max"] == 0.0          # first occurrence on a tie
    assert rec["positive_gradient_extent"] == 0.0


@pytest.mark.parametrize("grid", ["rho_uniform", "axis_clustered_random"])
def test_non_uniform_grid(grid):
    if grid == "rho_uniform":
        psi_N = np.linspace(0.0, 1.0, 301) ** 2
    else:
        rng = np.random.default_rng(3)
        psi_N = np.unique(np.concatenate(
            [[0.0, 1.0], rng.uniform(0.0, 1.0, 700) ** 1.5]))
    p = _hollow(psi_N)
    rec = core_pressure_health(psi_N, p)
    assert rec["is_hollow"] is True
    # the grid samples the true maximum only approximately
    k = int(np.argmax(np.where(psi_N <= CORE_PSI_N, p, -np.inf)))
    assert rec["psi_N_of_max"] == psi_N[k]
    assert rec["rise_frac"] == pytest.approx((p[k] - p[0]) / p[0], rel=1e-12)
    assert rec["rise_frac"] == pytest.approx(HOLLOW_RISE, rel=2e-3)
    assert rec["psi_N_of_max"] == pytest.approx(HOLLOW_PSI_MAX, abs=5e-3)
    # positive-gradient extent is the grid width from the axis to the
    # outermost climbing node, which is the node of the maximum here
    assert rec["positive_gradient_extent"] == pytest.approx(
        rec["psi_N_of_max"] - psi_N[0], rel=1e-12)


def test_descending_grid_gives_the_same_numbers():
    psi_N = _psi(601)
    p = _hollow(psi_N)
    up = core_pressure_health(psi_N, p)
    down = core_pressure_health(psi_N[::-1], p[::-1])
    for k in ("is_hollow", "rise_frac", "psi_N_of_max", "rise_extent",
              "p_ref", "p_core_max", "n_core_nodes"):
        assert down[k] == up[k]
    assert down["positive_gradient_extent"] == pytest.approx(
        up["positive_gradient_extent"], rel=1e-12)


def test_first_node_off_axis_is_used_and_recorded():
    psi_N = np.linspace(0.01, 1.0, 600)
    p = _hollow(psi_N)
    rec = core_pressure_health(psi_N, p)
    assert rec["evaluated"] is True
    assert rec["psi_N_ref"] == pytest.approx(0.01, abs=1e-15)
    assert rec["p_ref"] == pytest.approx(float(_hollow(np.array([0.01]))[0]),
                                         rel=1e-15)
    assert rec["rise_frac"] == pytest.approx((rec["p_core_max"] - p[0]) / p[0],
                                             rel=1e-12)
    assert rec["rise_extent"] == pytest.approx(rec["psi_N_of_max"] - 0.01,
                                               abs=1e-15)
    assert rec["is_hollow"] is True


def test_first_node_too_far_from_the_axis_is_not_evaluated():
    """A hollow inside the first node would be invisible, so the answer must
    not read as 'not hollow'."""
    psi_N = np.linspace(0.2, 1.0, 200)
    rec = core_pressure_health(psi_N, _monotone(psi_N))
    assert rec["evaluated"] is False
    assert rec["is_hollow"] is None
    assert "cannot stand for the axis" in rec["reason"]
    assert rec["psi_N_ref"] == pytest.approx(0.2)
    assert CORE_HOLLOW_MAX_REF_PSI_N == 0.05


@pytest.mark.parametrize("where", ["pressure_core", "pressure_edge", "psi_N"])
def test_nan_input_is_not_evaluated_never_not_hollow(where):
    psi_N = _psi(201)
    p = _hollow(psi_N)
    if where == "pressure_core":
        p[5] = np.nan
    elif where == "pressure_edge":
        p[-3] = np.inf
    else:
        psi_N = psi_N.copy()
        psi_N[7] = np.nan
    rec = core_pressure_health(psi_N, p)
    assert rec["evaluated"] is False
    assert rec["is_hollow"] is None             # not False
    assert rec["rise_frac"] is None
    assert "non-finite" in rec["reason"]


@pytest.mark.parametrize("case,match", [
    ("short", "at least 3 points"),
    ("shape", "at least 3 points"),
    ("nonmonotone", "not strictly monotone"),
    ("nonpositive_axis", "not positive"),
    ("tiny_core", "node(s) inside"),
    ("strings", "not numeric"),
])
def test_degenerate_input_is_not_evaluated_with_a_reason(case, match):
    psi_N = _psi(51)
    p = _monotone(psi_N)
    if case == "short":
        psi_N, p = psi_N[:2], p[:2]
    elif case == "shape":
        p = p[:-1]
    elif case == "nonmonotone":
        psi_N = psi_N.copy()
        psi_N[[3, 4]] = psi_N[[4, 3]]
    elif case == "nonpositive_axis":
        p = p - p[0]
    elif case == "tiny_core":
        psi_N = np.array([0.0, 0.3, 0.6, 0.8, 1.0])
        p = _monotone(psi_N)
    elif case == "strings":
        p = ["a"] * psi_N.size
    rec = core_pressure_health(psi_N, p)
    assert rec["evaluated"] is False
    assert rec["is_hollow"] is None
    assert match in rec["reason"]


def test_component_shares_add_to_one_and_describe_the_rise():
    psi_N = _psi(601)
    thermal = _monotone(psi_N)
    fast = 3.0e4 * _hollow(psi_N) / 1.0e5
    rec = core_pressure_health(psi_N, thermal + fast,
                               components={"electron_thermal": 0.5 * thermal,
                                           "ion_thermal": 0.5 * thermal,
                                           "fast": fast})
    assert rec["is_hollow"] is True
    sh = rec["component_shares"]
    assert sum(sh.values()) == pytest.approx(1.0, abs=1e-10)
    assert sh["fast"] > 1.0
    assert sh["electron_thermal"] < 0.0
    # a malformed component gets None rather than failing the record
    rec2 = core_pressure_health(psi_N, thermal + fast,
                                components={"fast": np.zeros(3)})
    assert rec2["evaluated"] is True and rec2["component_shares"] == {"fast": None}


def test_inputs_are_never_written_to():
    psi_N = _psi(301)
    p = _hollow(psi_N)
    comps = {"electron_thermal": 0.4 * p, "ion_thermal": 0.6 * p}
    arrays = (psi_N, p, *comps.values())
    for a in arrays:
        a.flags.writeable = False            # any in-place write would raise
    snap = [a.tobytes() for a in arrays]
    core_pressure_health(psi_N, p, components=comps)
    core_pressure_hollow_record(psi_N, p, input_components=comps,
                                achieved_total=p, warn=False)
    core_pressure_health(psi_N[::-1], p[::-1])
    assert [a.tobytes() for a in arrays] == snap


# ---------------------------------------------------------------------
#  The per-slice record
# ---------------------------------------------------------------------

def test_record_evaluates_total_thermal_input_and_achieved_separately():
    psi_N = _psi(601)
    base = _monotone(psi_N)
    fast = 3.0e4 * _hollow(psi_N) / 1.0e5
    comps = {"electron_thermal": 0.5 * base, "ion_thermal": 0.4 * base,
             "impurity": 0.1 * base, "fast": fast}
    total = base + fast
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rec = core_pressure_hollow_record(psi_N, total, input_components=comps,
                                          achieved_total=_monotone(psi_N))
    assert rec["input"]["total"]["is_hollow"] is True
    assert rec["input"]["total"]["composition"] == sorted(comps)
    th = rec["input"]["thermal"]
    assert th["evaluated"] is True and th["is_hollow"] is False
    assert th["composition"] == ["electron_thermal", "ion_thermal", "impurity"]
    assert rec["achieved"]["total"]["evaluated"] is True
    assert rec["achieved"]["total"]["is_hollow"] is False
    assert rec["achieved"]["thermal"]["evaluated"] is False
    assert "not separable" in rec["achieved"]["thermal"]["reason"]
    d = rec["definition"]
    assert d["rise_threshold"] == CORE_HOLLOW_RISE_FRAC
    assert d["psi_N_core"] == CORE_PSI_N
    assert "report-only" in d["kind"] and "gates nothing" in d["kind"]
    # one report-only warning, for the input total; no causal language
    hw = [x for x in w if issubclass(x.category, CorePressureHollowWarning)]
    assert len(hw) == 1 and rec["warned"] is True
    msg = str(hw[0].message)
    assert "input total" in msg and "Report-only" in msg
    for word in ("artifact", "artefact", "unphysical", "caused", "because"):
        assert word not in msg.lower()


def test_record_without_components_or_achieved_says_so():
    psi_N = _psi(101)
    rec = core_pressure_hollow_record(psi_N, _monotone(psi_N), warn=False)
    assert rec["input"]["total"]["evaluated"] is True
    assert rec["input"]["thermal"]["evaluated"] is False
    assert rec["achieved"]["total"]["evaluated"] is False
    assert rec["achieved"]["total"]["is_hollow"] is None
    rec2 = core_pressure_hollow_record(psi_N, _monotone(psi_N), warn=False,
                                       achieved_reason="get_profiles failed: x")
    assert rec2["achieved"]["total"]["reason"] == "get_profiles failed: x"


def test_nan_input_record_is_not_evaluated_and_does_not_warn():
    psi_N = _psi(101)
    p = _hollow(psi_N)
    p[3] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorePressureHollowWarning)
        rec = core_pressure_hollow_record(psi_N, p, achieved_total=p)
    for side in ("input", "achieved"):
        assert rec[side]["total"]["evaluated"] is False
        assert rec[side]["total"]["is_hollow"] is None
    assert rec["warned"] is False


def test_record_is_json_serializable():
    psi_N = _psi(257)
    p = _hollow(psi_N)
    rec = core_pressure_hollow_record(
        psi_N, p, input_components={"electron_thermal": 0.5 * p,
                                    "ion_thermal": 0.5 * p},
        achieved_total=p, warn=False)
    assert json.loads(json.dumps(rec)) == rec     # strict: no numpy leaks


# ---------------------------------------------------------------------
#  Archive: recorded next to the closure-health block, old archives read
# ---------------------------------------------------------------------

def _store(tmp_path, name, meta):
    from bouquet.utils import store_baseline_profiles
    header = str(tmp_path / name)
    psi = np.linspace(0.0, 1.0, 11)
    one = np.ones_like(psi)
    store_baseline_profiles(
        header, psi, one, one, one, one, one, one,
        one, one, one, one, one,
        Ip_target=1.0e6, l_i_target=0.9, scan_key=0, baseline_meta=meta)
    return header + ".h5"


def test_record_round_trips_through_the_archive(tmp_path):
    from bouquet.utils import load_baseline_profiles
    psi_N = _psi(101)
    rec = core_pressure_hollow_record(psi_N, _hollow(psi_N),
                                      achieved_total=_hollow(psi_N), warn=False)
    meta = {"tokamaker_li_1": 0.9, "core_pressure_hollow": rec,
            "ip_closure": {"closure_limited": False}, "closure_limited": False}
    bl = load_baseline_profiles(_store(tmp_path, "a", meta), scan_key=0)
    assert bl["core_pressure_hollow"] == rec
    assert bl["li_metrics"]["core_pressure_hollow"] == rec
    assert bl["closure_limited"] is False


def test_older_archives_without_the_record_still_read(tmp_path):
    from bouquet.utils import load_baseline_profiles
    bl = load_baseline_profiles(_store(tmp_path, "none", None), scan_key=0)
    assert "core_pressure_hollow" not in bl and "li_metrics" not in bl
    bl = load_baseline_profiles(
        _store(tmp_path, "old", {"tokamaker_li_1": 0.9}), scan_key=0)
    assert "core_pressure_hollow" not in bl
    assert bl["li_metrics"] == {"tokamaker_li_1": 0.9}


def test_filter_decisions_are_identical_with_and_without_the_record(tmp_path):
    from bouquet.filtering import (filter_coil_currents, read_filter_flags,
                                   select_indices)
    psi_N = _psi(101)
    rec = core_pressure_hollow_record(psi_N, _hollow(psi_N), warn=False)
    out = {}
    for name, meta in (("with", {"core_pressure_hollow": rec}),
                       ("without", None)):
        h5 = _store(tmp_path, name, meta)
        with h5py.File(h5, "a") as hf:
            for i, (F, V) in enumerate([(0.5, 0.5), (3.0, 0.5), (1.0, 1.9),
                                        (0.1, 2.5)]):
                d = hf.create_group(f"scan/0/{i}")
                d.attrs["max_F_drift_pct"] = F
                d.attrs["max_VSC_drift_pct"] = V
                d.attrs["inspec_F_max"] = 0.02
                d.attrs["inspec_VSC_max"] = 0.02
                d.attrs["in_spec"] = bool(F <= 2 and V <= 2)
                d.attrs["passes_boundary_filter"] = True
        summ, _ = filter_coil_currents(h5, scan_key=0, plot=False)
        out[name] = (summ["n_pass"], read_filter_flags(h5, scan_key=0),
                     list(select_indices(h5, scan_key=0)))
    assert out["with"] == out["without"]
    assert out["with"][0] == 2                    # the fixture is not vacuous


def test_no_decision_code_reads_the_record():
    """Filter, in-spec, until-N and sampling code must not consume it."""
    import importlib
    for mod in ("bouquet.filtering", "bouquet.parallel", "bouquet.sampling",
                "bouquet.archive"):
        src = inspect.getsource(importlib.import_module(mod))
        for name in ("core_pressure_hollow", "core_pressure_health",
                     "is_hollow", "rise_frac"):
            assert name not in src, f"{mod} reads {name}"
    # generate_bouquet (draws, in-spec, until-N) does not read it either
    from bouquet.TokaMaker_interface import generate_bouquet
    gsrc = inspect.getsource(generate_bouquet)
    assert "core_pressure_hollow" not in gsrc and "is_hollow" not in gsrc


# ---------------------------------------------------------------------
#  Bit-identity of the pressure the solver receives
# ---------------------------------------------------------------------
#
# The record's call sites add statements to the pressure-composition blocks
# (per-component copies ``_pc[...] = ...``).  These tests execute the REAL
# source of each block twice -- as written, and with every ``_pc`` statement
# removed from the AST -- and require the composed pressure and its P' to be
# bit-identical.  The record-building ``try`` blocks are then executed with
# the record function forced to fail, to show a failure cannot escape.

def _is_pc_stmt(node):
    def _pc(t):
        return ((isinstance(t, ast.Name) and t.id == "_pc")
                or (isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Name) and t.value.id == "_pc"))
    if isinstance(node, ast.Assign):
        return all(_pc(t) for t in node.targets)
    if isinstance(node, ast.AugAssign):
        return _pc(node.target)
    return False


class _StripPc(ast.NodeTransformer):
    def __init__(self):
        self.removed = 0

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse"):
            stmts = getattr(node, field, None)
            if (isinstance(stmts, list) and stmts
                    and isinstance(stmts[0], ast.stmt)):
                keep = [s for s in stmts if not _is_pc_stmt(s)]
                self.removed += len(stmts) - len(keep)
                setattr(node, field, keep or [ast.Pass()])
        return node


def _function_body(func):
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return tree.body[0].body


def _region(body, start_pred, stop_pred):
    i0 = next(i for i, s in enumerate(body) if start_pred(s))
    i1 = next(i for i, s in enumerate(body) if i > i0 and stop_pred(s))
    return body[i0:i1]


def _assigns(name):
    def pred(s):
        return (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Name)
                and s.targets[0].id == name)
    return pred


def _run(stmts, ns, strip):
    mod = ast.Module(body=copy.deepcopy(stmts), type_ignores=[])
    tr = _StripPc()
    if strip:
        mod = tr.visit(mod)
    ast.fix_missing_locations(mod)
    exec(compile(mod, "<region>", "exec"), ns)
    return tr.removed


def _kinetics(psi_N):
    ne = 5.0e19 * (1.0 - 0.8 * psi_N ** 2) + 1.0e18
    te = 3.0e3 * (1.0 - psi_N ** 1.5) + 50.0
    ni = 0.85 * ne
    ti = 2.5e3 * (1.0 - psi_N ** 2) + 40.0
    return ne, te, ni, ti


def test_reconstruction_pressure_composition_is_bit_identical():
    from bouquet.TokaMaker_interface import reconstruct_equilibrium
    from bouquet.utils import pchip_derivative
    body = _function_body(reconstruct_equilibrium)
    region = _region(body, _assigns("pres_tmp"), _assigns("ffp_prof"))
    psi_N = np.linspace(0.0, 1.0, 129)
    ne, te, ni, ti = _kinetics(psi_N)
    p_fast = 2.0e4 * _hollow(psi_N) / 1.0e5
    results = {}
    for strip in (False, True):
        ns = {"__name__": "bouquet.TokaMaker_interface",
              "__package__": "bouquet", "np": np,
              "pchip_derivative": pchip_derivative,
              "ne": ne.copy(), "te": te.copy(), "ni": ni.copy(),
              "ti": ti.copy(), "p_fast": p_fast.copy(), "Z_imp": 6.0,
              "eqdsk": SimpleNamespace(psi_N=psi_N.copy()),
              "mygs": SimpleNamespace(psi_bounds=(-0.4, 0.15))}
        removed = _run(region, ns, strip)
        if strip:
            assert removed >= 3, "the _pc statements were not found"
            assert "_pc" not in ns
        else:
            assert set(ns["_pc"]) == {"electron_thermal", "ion_thermal",
                                      "fast", "impurity"}
        results[strip] = (ns["pres_tmp"].tobytes(),
                          ns["pprime_tmp"].tobytes(),
                          ns["pp_prof"]["y"].tobytes(),
                          ns["pp_prof"]["x"].tobytes())
        for a, b in ((ns["ne"], ne), (ns["te"], te), (ns["ni"], ni),
                     (ns["ti"], ti), (ns["p_fast"], p_fast)):
            assert a.tobytes() == b.tobytes()
    assert results[False] == results[True]


def test_imas_forward_solve_pressure_composition_is_bit_identical():
    from bouquet.run import Bouquet
    from bouquet.utils import pchip_derivative, pchip_interp
    body = _function_body(Bouquet._forward_solve_imas_baseline)
    region = _region(
        body, _assigns("p_total"),
        lambda s: isinstance(s, ast.FunctionDef) and s.name == "solve_jphi")
    psi_N = np.linspace(0.0, 1.0, 101)
    psi_k = np.linspace(0.0, 1.0, 51)
    ne, te, ni, ti = _kinetics(psi_N)
    bl = SimpleNamespace(psi_N_kinetic=psi_k,
                         p_fast=2.0e4 * _hollow(psi_k) / 1.0e5,
                         Z_imp=6.0, z_fast=1.0e18 * (1.0 - psi_k),
                         p_diff=500.0 * np.sin(3.0 * psi_k))
    results = {}
    for strip in (False, True):
        ns = {"__name__": "bouquet.run", "__package__": "bouquet", "np": np,
              "EC": 1.602176634e-19, "ne": ne.copy(), "te": te.copy(),
              "ni": ni.copy(), "ti": ti.copy(), "bl": bl,
              "k2e": lambda a: pchip_interp(psi_k, a, psi_N)}
        removed = _run(region, ns, strip)
        if strip:
            assert removed >= 4, "the _pc statements were not found"
        else:
            assert set(ns["_pc"]) == {"electron_thermal", "ion_thermal",
                                      "fast", "impurity", "anchor_diff"}
        results[strip] = (ns["p_total"].tobytes(),
                          pchip_derivative(psi_N, ns["p_total"]).tobytes())
    assert results[False] == results[True]


def _health_try(func):
    """The ``try`` statement in *func* that builds the record."""
    for node in ast.walk(ast.Module(body=_function_body(func),
                                    type_ignores=[])):
        if (isinstance(node, ast.Try)
                and "core_pressure_hollow_record" in ast.unparse(node)):
            return node
    raise AssertionError(f"no record-building try block in {func.__name__}")


def test_a_failing_record_cannot_escape_either_call_site(monkeypatch):
    from bouquet.TokaMaker_interface import reconstruct_equilibrium
    from bouquet.run import Bouquet

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(physics, "core_pressure_hollow_record", boom)
    psi_N = _psi(51)
    p = _monotone(psi_N)

    ns = {"__name__": "bouquet.TokaMaker_interface", "__package__": "bouquet",
          "np": np, "eqdsk": SimpleNamespace(psi_N=psi_N), "pres_tmp": p,
          "_pc": {}, "_p_ach": p, "quality": {}}
    _run([_health_try(reconstruct_equilibrium)], ns, strip=False)
    assert "synthetic failure" in \
        ns["quality"]["core_pressure_hollow"]["unavailable"]

    class _Gs:
        def get_profiles(self, psi=None):
            raise RuntimeError("no equilibrium")
    ns = {"__name__": "bouquet.run", "__package__": "bouquet", "np": np,
          "psi_N": psi_N, "p_total": p, "_pc": {}, "mygs": _Gs(),
          "core_pressure_hollow_record": boom}
    _run([_health_try(Bouquet._forward_solve_imas_baseline)], ns, strip=False)
    assert "synthetic failure" in ns["_cph"]["unavailable"]

    # and with the real record: a failing get_profiles is recorded, not raised
    ns["core_pressure_hollow_record"] = core_pressure_hollow_record
    _run([_health_try(Bouquet._forward_solve_imas_baseline)], ns, strip=False)
    ach = ns["_cph"]["achieved"]["total"]
    assert ach["evaluated"] is False and "no equilibrium" in ach["reason"]
    assert ns["_cph"]["input"]["total"]["evaluated"] is True


# The positional field order of ``Baseline`` before the hollowness record was
# added.  ``Baseline`` is a public dataclass exported from ``bouquet`` and is
# not keyword-only, so a new field must be appended after all of these: a
# mid-list insertion would silently shift every later positional argument
# (e.g. an auxiliary-profile dict landing in ``core_pressure_hollow``).
_BASELINE_FIELDS_BEFORE_HOLLOW_RECORD = (
    "psi_N", "j_phi", "j_inductive", "j_BS", "psi_N_kinetic", "ne", "te",
    "ni", "ti", "Zeff", "Ip_target", "l_i_target", "provenance", "l_i_scale",
    "j_NBI", "j_RF", "p_fast", "p_fast_meta", "bs_scale", "ohm_scale",
    "ip_closure", "sawtooth", "jBS_diff", "p_equilibrium", "p_diff", "Z_imp",
    "z_fast", "jphi_diff", "eqdsk_bytes", "pfile_bytes", "recon",
    "li_metrics", "aux", "reconstruction_metrics", "reconstruction_log",
)


def test_baseline_positional_slots_unchanged_by_hollow_record():
    """Every pre-existing field keeps its positional slot; the new record is
    appended and defaults to None when callers pass the old argument list."""
    import dataclasses

    from bouquet import Baseline

    names = [f.name for f in dataclasses.fields(Baseline)]
    n_old = len(_BASELINE_FIELDS_BEFORE_HOLLOW_RECORD)
    assert tuple(names[:n_old]) == _BASELINE_FIELDS_BEFORE_HOLLOW_RECORD
    # the record follows the old fields; anything appended after it (other
    # lines add their own records the same way) must also carry a default,
    # so the old positional argument list still constructs a Baseline
    assert names[n_old] == "core_pressure_hollow"
    _missing = dataclasses.MISSING
    for f in dataclasses.fields(Baseline)[n_old:]:
        assert (f.default is not _missing
                or f.default_factory is not _missing), f.name

    # one distinct sentinel per pre-existing slot, passed positionally
    sentinels = [object() for _ in _BASELINE_FIELDS_BEFORE_HOLLOW_RECORD]
    b = Baseline(*sentinels)
    for name, val in zip(_BASELINE_FIELDS_BEFORE_HOLLOW_RECORD, sentinels):
        assert getattr(b, name) is val, name
    assert b.core_pressure_hollow is None
