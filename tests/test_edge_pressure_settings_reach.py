"""What the frozen-copy tests of the edge-pressure helper do NOT compare, and
the tests that close it.

``tests/test_edge_pressure_legacy_ast.py`` and
``tests/test_engine_draws_legacy_ast.py`` undo the helper substitution
before comparing with the frozen code.  The normaliser
(``tests/_edge_pressure_ast.py``) rewrites ``solver_pprime(psi, p, r,
<settings>)`` to the old inline statements from its FIRST THREE arguments
and removes every ``_edge = ...`` binding and every ``edge_pressure=``
keyword.  So those tests prove "the code is the frozen code, with a helper
call wherever the inline statements were" -- and say nothing about WHICH
settings each call is handed.  A site passing a settings object other
than the configured one, a site passing none while its neighbours pass the
configured one, or a new function that calls the helper outside the
compared set would all pass them.  Bit-identity of the legacy path at the
PRE-CHANGE settings (``edge_pprime_pin=True, separatrix_pressure="legacy"``,
``EP.PRE_CHANGE_EDGE_PRESSURE``; NOT the defaults since 2026-10-02, whose
separatrix setting is ``"offset"``) needs three more things, asserted
here:

1. every helper call of the package hands the helper the function's ONE
   settings object (``_edge``, or ``self.edge`` on the engine backend),
   positionally, and nothing else;
2. that object is bound only by ``resolve_edge_pressure`` of the configured
   settings (the function's own ``edge_pressure`` parameter, the
   generation config, or the draw context), every function taking an
   ``edge_pressure`` parameter defaults it to ``None``, and every call of
   such a function in the package passes it on -- so a configuration with
   the pre-change settings resolves to the pre-change settings at every
   site, and the default configuration to the defaults (checked on the
   objects themselves, not only on the source);
3. the helper at the pre-change settings equals the old inline expressions
   BIT FOR BIT
   on the arrays the legacy path itself produced: the reconstruction's and
   every archived draw's solve pressure of the stored reference archive, on
   their own grids and their own flux ranges.

And every function of the package that calls the helper is one a
frozen-copy test (or, for the engine backend, a numeric test) covers: a new
site in a new function fails here until it is added to a compared set.

Solver-free.  Reads the stored reference archive; writes nothing.

(Retired: the frozen-copy AST tests and their snapshot files
(tests/test_*_legacy_ast.py, tests/data/*_prechange.py.txt) are no longer
part of the suite; the comparisons against them are removed.)
"""
import ast
import glob
import os

import numpy as np
import pytest

from bouquet import edge_pressure as EP
from bouquet.config import GenerationConfig
from bouquet.utils import pchip_derivative

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "bouquet")
_ARCHIVE = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")

#: helper -> index of its settings argument
HELPERS = {"solver_pp_profile": 3, "solver_pprime": 3, "solver_pax": 1,
           "solver_pressure": 1}
#: the settings object of a function, as every site names it
SETTINGS_NAMES = {"_edge", "self.edge"}
#: what a settings object may be resolved from
RESOLVE_FROM = {"edge_pressure", "self.config.generation",
                "config.generation", "gc", "ctx.get('edge_pressure')",
                "(getattr(eng, 's', None) or {}).get('edge_pressure')"}
#: functions whose solve path is compared with frozen code
#: (tests/test_edge_pressure_legacy_ast.py NAMES and
#: tests/test_engine_draws_legacy_ast.py), by outermost name
AST_COMPARED = {
    ("TokaMaker_interface.py", "_std_candidate_solve"),
    ("TokaMaker_interface.py", "perturb_kinetic_equilibrium"),
    ("TokaMaker_interface.py", "reconstruct_equilibrium"),
    ("TokaMaker_interface.py", "generate_bouquet"),
    ("TokaMaker_interface.py", "_post_homotopy_jbs"),
    ("run.py", "Bouquet._forward_solve_imas_baseline"),
    ("run.py", "Bouquet._verify_sigma0_jbs_loop"),
    ("run.py", "Bouquet._sigma0_draw_route"),
    ("run.py", "Bouquet.verify_sigma0_consistency"),
}
#: the engine backend: compared numerically with a recording solver
#: (tests/test_edge_pressure.py), not on the legacy path
NUMERIC_COMPARED = {("engine.py", "TokaMakerBackend.solve")}
#: functions that take the settings as ``edge_pressure=``
TAKES_SETTINGS = {"_std_candidate_solve", "perturb_kinetic_equilibrium",
                  "generate_bouquet", "reconstruct_equilibrium",
                  "_reconstruction_metrics", "TokaMakerBackend",
                  "tokamaker_backend"}


def _files():
    out = sorted(glob.glob(os.path.join(_PKG, "**", "*.py"), recursive=True))
    out = [p for p in out if os.path.basename(p) != "edge_pressure.py"]
    assert len(out) > 15
    return out


def _walk(node, qual, visit):
    for ch in ast.iter_child_nodes(node):
        q = qual
        if isinstance(ch, (ast.FunctionDef, ast.ClassDef)):
            q = qual + [ch.name]
        visit(ch, qual)
        _walk(ch, q, visit)


def _outer(qual):
    """``Class.method`` or the top-level function a node lives in."""
    return ".".join(qual[:2]) if qual and qual[0][:1].isupper() else (
        qual[0] if qual else "")


def _collect():
    calls, binds, kws, defs = [], [], [], []
    for path in _files():
        name = os.path.relpath(path, _PKG)
        with open(path) as fh:
            tree = ast.parse(fh.read())

        def visit(ch, qual, name=name):
            where = (name, _outer(qual))
            if isinstance(ch, ast.Call):
                fn = ch.func.id if isinstance(ch.func, ast.Name) else (
                    ch.func.attr if isinstance(ch.func, ast.Attribute)
                    else None)
                if isinstance(ch.func, ast.Name) and fn in HELPERS:
                    calls.append((where, fn, ch))
                if fn in TAKES_SETTINGS:
                    kws.append((where, fn, ch))
            if isinstance(ch, ast.Assign):
                for t in ch.targets:
                    if ast.unparse(t) in SETTINGS_NAMES:
                        binds.append((where, ast.unparse(t), ch))
            if isinstance(ch, ast.FunctionDef):
                a = ch.args
                names = [x.arg for x in a.args]
                if "edge_pressure" in names:
                    i = names.index("edge_pressure")
                    first = len(a.args) - len(a.defaults)
                    defs.append((where, ch.name,
                                 a.defaults[i - first] if i >= first
                                 else None))
                for x, dflt in zip(a.kwonlyargs, a.kw_defaults):
                    if x.arg == "edge_pressure":
                        defs.append((where, ch.name, dflt))
        _walk(tree, [], visit)
    return calls, binds, kws, defs


def test_every_helper_call_is_handed_the_functions_one_settings_object():
    calls, _b, _k, _d = _collect()
    assert len(calls) >= 50          # not vacuous: the sites are found
    bad = []
    for where, fn, ch in calls:
        i = HELPERS[fn]
        # coord=: the run coordinate a Phi_N run tags the profile with
        ok = (len(ch.args) == i + 1
              and all(k.arg == "coord" for k in ch.keywords)
              and ast.unparse(ch.args[i]) in SETTINGS_NAMES)
        if not ok:
            bad.append((where, ast.unparse(ch)))
    assert not bad, bad


def test_the_settings_object_is_only_ever_the_resolved_configuration():
    _c, binds, _k, defs = _collect()
    assert len(binds) >= 12
    bad = []
    for where, tgt, node in binds:
        v = node.value
        src = ast.unparse(v)
        if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id == "resolve_edge_pressure"
                and len(v.args) == 1 and not v.keywords
                and ast.unparse(v.args[0]) in RESOLVE_FROM):
            continue
        # the draw method's settings (the resolved ones, or the engine draw
        # context's own: test_the_engine_branch_binding_is_gated_on_the_engine)
        if src == "_m.edge(_edge)" and where[1] == "generate_bouquet":
            continue
        bad.append((where, ast.unparse(node)))
    assert not bad, bad
    # every edge_pressure parameter defaults to None (-> the defaults)
    assert {name for _w, name, _d in defs} == (
        TAKES_SETTINGS - {"TokaMakerBackend"}) | {"__init__"}
    for where, name, default in defs:
        assert (isinstance(default, ast.Constant)
                and default.value is None), (where, name)


def test_the_engine_branch_binding_is_gated_on_the_engine():
    """``_edge = _m.edge(_edge)``: the legacy and swb methods keep the resolved
    settings; only the engine method returns its draw context's own."""
    from types import SimpleNamespace
    from bouquet.draw_methods import DrawMethod
    from bouquet.engine_draws import GenerateEngineDraws
    from bouquet.swb_draws import SwbDraws
    edge = object()
    assert DrawMethod().edge(edge) is edge
    assert SwbDraws.edge is DrawMethod.edge
    eng = GenerateEngineDraws.__new__(GenerateEngineDraws)
    eng.ctx = SimpleNamespace(edge="ctx")
    assert eng.edge(edge) == "ctx"


def test_every_caller_passes_the_settings_on():
    """No call of a function that takes ``edge_pressure`` leaves it out (a
    site that did would run at the defaults whatever was configured --
    since 2026-10-02 at "offset" even under a "legacy" configuration)."""
    _c, _b, kws, _d = _collect()
    assert len(kws) >= 8
    bad = [(where, fn) for where, fn, ch in kws
           if "edge_pressure" not in [k.arg for k in ch.keywords]]
    assert not bad, bad


def test_every_function_that_calls_the_helper_is_a_compared_one():
    calls, _b, _k, _d = _collect()
    seen = {where for where, _fn, _ch in calls}
    extra = seen - AST_COMPARED - NUMERIC_COMPARED
    assert not extra, (
        "helper call(s) in function(s) no frozen-copy or numeric test "
        f"compares: {sorted(extra)}")


def test_the_default_configuration_resolves_to_the_defaults_everywhere():
    """The objects the bindings above resolve from, at the default
    configuration."""
    from bouquet.engine import engine_settings
    gc = GenerationConfig()
    d = EP.EdgePressure()
    assert d.is_default and d.offset and not d.is_pre_change
    assert EP.resolve_edge_pressure(None) == d
    assert EP.resolve_edge_pressure(gc) == d
    assert EP.resolve_edge_pressure({}.get("edge_pressure")) == d
    assert EP.resolve_edge_pressure(dict(edge_pressure=d)["edge_pressure"]) \
        is d
    gu = GenerationConfig(reconstruction_engine="unified")
    assert EP.resolve_edge_pressure(
        engine_settings(gu)["edge_pressure"]) == d


def test_a_pre_change_configuration_resolves_to_the_pre_change_settings():
    """The same routes, configured with ``separatrix_pressure="legacy"``
    (pin at its default, on): every one resolves to the pre-change
    settings -- the settings at which the frozen-copy tests prove the legacy
    paths bit for bit."""
    from bouquet.engine import engine_settings
    pc = EP.EdgePressure.pre_change()
    assert pc.is_pre_change and not pc.is_default
    gc = GenerationConfig(separatrix_pressure="legacy")
    assert EP.resolve_edge_pressure(gc) == pc
    rec = EP.resolve_edge_pressure(gc).record()
    assert rec == EP.PRE_CHANGE_EDGE_PRESSURE
    assert EP.resolve_edge_pressure(rec) == pc
    assert EP.resolve_edge_pressure(dict(edge_pressure=pc)["edge_pressure"]) \
        is pc
    assert EP.resolve_edge_pressure(EP.describe(pc)) == pc
    gu = GenerationConfig(reconstruction_engine="unified",
                          separatrix_pressure="legacy")
    assert EP.resolve_edge_pressure(
        engine_settings(gu)["edge_pressure"]) == pc


# ---------------------------------------------------------------------------
#  numeric bit-identity on the arrays the legacy path produced
# ---------------------------------------------------------------------------
def _legacy_path_inputs():
    """(label, psi_N, pressure, flux range) of the stored reference
    archive: the reconstruction and every archived draw, each with the flux
    range of its own stored equilibrium."""
    import h5py
    from bouquet.io.geqdsk import read_geqdsk
    from bouquet.schema import find_bytes_dataset
    from bouquet.utils import read_eqdsk_from_bytes
    out = []
    with h5py.File(_ARCHIVE, "r") as hf:
        scan = hf["scan"]
        for sk in scan:
            for name in scan[sk]:
                g = scan[sk][name]
                if not hasattr(g, "keys") or "pressure" not in g:
                    continue
                nm = find_bytes_dataset(g, "eqdsk")
                eq = read_eqdsk_from_bytes(bytes(g[nm][()]), read_geqdsk)
                out.append((f"{sk}/{name}",
                            np.array(g["psi_N"][()], dtype=float),
                            np.array(g["pressure"][()], dtype=float),
                            float(eq.psi_boundary) - float(eq.psi_axis)))
    return out


_INPUTS = _legacy_path_inputs() if os.path.isfile(_ARCHIVE) else []


def test_the_reference_archive_supplies_the_inputs():
    assert len(_INPUTS) >= 10
    assert any(lab.endswith("_baseline") for lab, *_ in _INPUTS)
    for _lab, x, p, r in _INPUTS:
        assert x.size == p.size >= 65 and np.all(np.isfinite(p))
        assert np.isfinite(r) and r != 0.0


@pytest.mark.parametrize("i", range(len(_INPUTS)))
@pytest.mark.parametrize("flip", [1.0, -1.0])
def test_the_helper_is_the_old_inline_code_on_the_legacy_paths_arrays(
        i, flip):
    _lab, x, p, r = _INPUTS[i]
    r = flip * r                      # either orientation of the flux
    # the pre-change settings, four ways (not the defaults since 2026-10-02)
    for edge in (EP.EdgePressure.pre_change(),
                 GenerationConfig(separatrix_pressure="legacy"),
                 dict(EP.PRE_CHANGE_EDGE_PRESSURE),
                 dict(separatrix_pressure="legacy")):
        # the old inline statements, verbatim
        old = pchip_derivative(x, p) / r
        old[-1] = 0.0
        old_pp = {"type": "linterp", "y": pchip_derivative(x, p) / r,
                  "x": x}
        old_pp["y"][-1] = 0.0
        old_pax = p[0]
        new = EP.solver_pprime(x, p, r, edge)
        assert new.dtype == old.dtype and new.shape == old.shape
        assert new.tobytes() == old.tobytes()
        pp = EP.solver_pp_profile(x, p, r, edge)
        assert list(pp) == list(old_pp) and pp["type"] == old_pp["type"]
        assert pp["y"].tobytes() == old_pp["y"].tobytes() and pp["x"] is x
        pax = EP.solver_pax(p, edge)
        # the old sites passed the array element; the helper its float:
        # the same IEEE double
        assert np.float64(pax).tobytes() == np.float64(old_pax).tobytes()
        assert EP.solver_pressure(p, edge) is p
        assert EP.applied_offset(p, edge) == 0.0
        assert EP.lcfs_kwargs(EP.applied_offset(p, edge)) == {}
        assert EP.describe(edge, p)["p_sep_applied"] == 0.0


def test_no_inline_site_anywhere_in_the_package():
    """The whole package (every module, not a list of them): no ``...[-1] =
    0.0`` on a profile and no ``pax=<p>[0]`` outside the helper.  The one
    exception is the g-file READER's own edge extrapolation of the file's
    ``PPRIME`` / ``FFPRIM`` arrays (``io/geqdsk.py``), which is not a
    profile handed to the solver."""
    import re
    # (a zero, written 0 / 0. / 0.0, assigned to a last element)
    pin = re.compile(r"\[\s*-1\s*\]\s*=\s*0(?:\.0*)?\s*(?:#.*)?$")
    pax = re.compile(r"pax\s*=\s*(?:float\()?[\w.\[\]\"']+\[\s*0\s*\]")
    assert pin.search("pp['y'][-1] = 0.0") and pin.search("y[-1] = 0")
    assert not pin.search("r[0] = r[-1] = 0.5 * (r[0] + r[-1])")
    bad, reader = [], []
    for path in _files():
        name = os.path.relpath(path, _PKG)
        with open(path) as fh:
            for i, ln in enumerate(fh, 1):
                if ln.lstrip().startswith("#"):
                    continue
                if pin.search(ln.rstrip()) or pax.search(ln):
                    (reader if name == os.path.join("io", "geqdsk.py")
                     else bad).append(f"{name}:{i}: {ln.strip()}")
    assert not bad, bad
    assert reader == [r for r in reader if "prof[-1] = 0.0" in r]
    assert len(reader) == 1, reader


# ---------------------------------------------------------------------------
#  behavioural: the VALUE each executed legacy call site hands on
# ---------------------------------------------------------------------------
# The checks above are on the source: a site that keeps the keyword but
# passes ``None`` (``edge_pressure=None``) passes them, and since 2026-10-02
# ``resolve_edge_pressure(None)`` is "offset" -- so such a site would run
# "offset" under a "legacy" configuration (the review's surviving mutants
# S1 / S2).  These run the sites and check the object actually received.
@pytest.mark.parametrize("sep", ["legacy", "offset"])
def test_the_draw_loop_redo_receives_the_draws_settings(monkeypatch, sep,
                                                        toy):
    """perturb_kinetic_equilibrium's self-consistent-loop redo
    (``_gs_step`` -> ``_std_candidate_solve``), executed on the toy stand-in
    of tests/test_sigma0_identity_stages.py: every redo is handed the draw's
    own settings object, at the configured value."""
    import test_sigma0_identity_stages as SI
    import bouquet.TokaMaker_interface as TI
    seen = []
    real = TI._std_candidate_solve

    def spy(*a, **k):
        seen.append(k.get("edge_pressure"))
        return real(*a, **k)

    monkeypatch.setattr(TI, "_std_candidate_solve", spy)
    edge = EP.resolve_edge_pressure(dict(separatrix_pressure=sep))
    req, jbs, fx = SI._reconstruct(toy, kappa_short=0.98)
    dv, off, _n = SI._delivered(toy, req, jbs, fx)
    real_pk = TI.perturb_kinetic_equilibrium

    def pk(*a, **k):
        k["edge_pressure"] = edge
        return real_pk(*a, **k)

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", pk)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        SI._draw(toy, "standard", dv["request"], dv["j_inductive"],
                 SI._li(toy.copy_eq().achieved), SI._settings(), offset=off)
    redo = seen[1:]                    # [0]: the candidate's first solve
    assert redo, "the loop took no redo pass: the site was not executed"
    assert all(e is edge for e in seen), seen


@pytest.mark.parametrize("sep", ["legacy", "offset"])
def test_the_legacy_sigma0_route_hands_the_configured_settings(monkeypatch,
                                                               sep):
    """Bouquet._sigma0_draw_route (the legacy sigma=0 draw route) runs
    perturb_kinetic_equilibrium with the CONFIGURATION's settings, at their
    value -- the stand-in harness of tests/test_sigma0_draw_route.py."""
    import test_sigma0_draw_route as SR
    b, calls = SR._bouquet(monkeypatch, li_draw=0.8002)
    b.config.generation.separatrix_pressure = sep
    SR._run(b)
    assert len(calls) == 1
    got = calls[0][1]["edge_pressure"]
    assert got is not None
    assert got == EP.resolve_edge_pressure(b.config.generation)
    assert got.separatrix_pressure == sep


from test_sigma0_identity_stages import toy  # noqa: E402,F401  (fixture)
