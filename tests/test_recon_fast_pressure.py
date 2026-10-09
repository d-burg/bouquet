"""Contract tests for the fast-ion pressure in ``reconstruct_equilibrium``.

Background
----------
``reconstruct_equilibrium`` used to build its Grad-Shafranov pressure as
thermal-only, single-ion::

    pres_tmp = e * (ne*te + ni*ti)

while every consumer of that reconstruction solved a HIGHER pressure --
``perturb_kinetic_equilibrium`` (per draw) and the state anchor both add
``p_fast`` and the impurity term.  The reconstruction therefore tuned its
inductive-current knob against a lower-pressure equilibrium than the draws it
feeds, and its reported ``beta_n`` / ``beta_p`` / ``W_MHD`` were low by ~20%
on a beam-heated shot.

These tests lock in the contract.  They are deliberately cheap: they do NOT
run a GS solve (that needs OFT, a mesh and ~40 s), so they cannot catch a
numerical regression inside the solve.  What they do catch is the plumbing
and grid contract, which is what actually drifted.
"""
from __future__ import annotations

import ast
import inspect
import re
import textwrap

import numpy as np
import pytest


def test_reconstruct_equilibrium_accepts_p_fast_and_Z_imp():
    """The kwargs exist and default to None (default-off)."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    params = inspect.signature(reconstruct_equilibrium).parameters
    for name in ("p_fast", "Z_imp"):
        assert name in params, f"reconstruct_equilibrium lost its {name} kwarg"
        assert params[name].default is None, (
            f"{name} must default to None so the pre-fix behaviour is "
            "reproduced bitwise when it is not supplied"
        )


def test_impurity_pressure_is_zero_for_falsy_Z_imp():
    """The guarded impurity call is safe on impurity-free (geqdsk) sources."""
    from bouquet.physics import impurity_pressure

    ne = np.linspace(5e19, 1e19, 16)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 16)
    for Z in (None, 0, 0.0):
        out = impurity_pressure(ne, ni, ti, Z)
        assert np.all(out == 0.0), f"impurity_pressure({Z!r}) should be zeros"


def test_impurity_pressure_matches_the_one_zeff_model():
    """nz = (ne - ni)/Z_imp thermalized at ti -- the term single-ion omits.

    Note: ``physics`` uses the CODATA elementary charge
    (``physics.ELEMENTARY_CHARGE``, 1.602176634e-19).  Under the
    self-consistent loop the thermal terms use the same constant; only the
    frozen legacy path keeps ``ELEMENTARY_CHARGE_LEGACY = 1.6022e-19`` for its
    thermal terms (a 1.5e-5 relative difference to the impurity term there).
    Harmless for recon-vs-draw consistency (both call THIS function), so the
    test pins the physics module's own constant.
    """
    from bouquet.physics import impurity_pressure, _EC as EC

    ne = np.linspace(5e19, 1e19, 16)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 16)
    Z = 6.0
    expected = EC * ((ne - ni) / Z) * ti
    np.testing.assert_allclose(impurity_pressure(ne, ni, ti, Z), expected,
                               rtol=1e-12, atol=0.0)


def _materialize_nested(outer, name, env):
    """Compile and return a nested ``def`` out of ``outer``'s REAL source.

    The kin->eq helpers are closures inside multi-hundred-line functions that
    cannot be called without a solve.  Rather than re-implement them in the
    test (which is how the previous version of this contract test ended up
    comparing ``pchip_interp`` to itself and became unfailable), this lifts the
    nested ``def`` verbatim out of the enclosing function's source and executes
    it with its free variables supplied through ``env``.

    So the object under test IS the shipped code: if someone edits
    ``baseline._resolve_reconstruction.to_eq`` to call ``np.interp``, the
    materialized function calls ``np.interp`` too, and the comparison fails.
    """
    src = textwrap.dedent(inspect.getsource(outer))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.name == name
                and node is not tree.body[0]):
            mod = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(mod)
            ns = dict(env)
            exec(compile(mod, f"<{outer.__name__}.{name}>", "exec"), ns)
            return ns[name]
    raise LookupError(
        f"nested def {name!r} no longer exists inside {outer.__name__} -- the "
        "kin->eq contract test can no longer see the code it guards; re-point "
        "it rather than deleting it"
    )


def test_recon_and_draw_kin_to_eq_agree_on_the_same_inputs():
    """The recon path's kin->eq map must equal the draw path's, bit for bit.

    Executes the ACTUAL nested helpers -- ``baseline._resolve_reconstruction``'s
    ``to_eq`` (reconstruction path) and ``perturb_kinetic_equilibrium``'s
    ``_kin_to_eq`` (draw path) -- lifted from their real source, and compares
    them on identical inputs.  If either drifts to a different interpolant the
    recon and the draws again solve slightly different fast pressures, which is
    exactly the regression this file exists to prevent.
    """
    import bouquet.baseline as _baseline
    import bouquet.TokaMaker_interface as _tmi
    from bouquet.utils import pchip_interp

    psi_kin = np.linspace(0.0, 1.0, 150)
    psi_eq = np.linspace(0.0, 1.0, 129)

    draw_k2e = _materialize_nested(
        _tmi.perturb_kinetic_equilibrium, "_kin_to_eq",
        dict(np=np, pchip_interp=pchip_interp,
             _dual_grid=True, psi_kin=psi_kin, psi_N=psi_eq),
    )
    recon_k2e = _materialize_nested(
        _baseline._resolve_reconstruction, "to_eq",
        dict(np=np, pchip_interp=pchip_interp,
             x_kin=psi_kin, x_run=psi_eq),
    )

    # A smooth core profile, a pedestal-like profile (where a linear regrid
    # differs most from PCHIP), and a non-monotone one.
    profiles = {
        "parabolic": 2.1e4 * (1.0 - psi_kin ** 2) ** 2,
        "pedestal": 1.0 / (1.0 + np.exp((psi_kin - 0.95) / 0.02)),
        "wiggly": np.sin(6.0 * np.pi * psi_kin) + 2.0,
    }
    for label, arr in profiles.items():
        a = draw_k2e(arr)
        b = recon_k2e(arr)
        np.testing.assert_array_equal(
            a, b, err_msg=f"recon and draw kin->eq disagree on {label!r}"
        )
        assert a.shape == psi_eq.shape

    # Shape-preserving and non-negative for a non-negative input.
    out = draw_k2e(profiles["parabolic"])
    assert np.all(out >= -1e-9), "kin->eq regrid must not undershoot below zero"


def test_every_kin_to_eq_site_routes_through_pchip_interp():
    """No kin->eq site may use a different interpolant.

    The two helpers exercised above are ``def``s; the remaining sites are
    lambdas/defs inside ``generate_bouquet`` that cannot be materialized as
    cleanly.  Assert structurally that each one calls ``pchip_interp`` and that
    no linear regrid (``np.interp`` / ``interp1d``) has crept into the kin->eq
    role next to it.
    """
    import bouquet.baseline as _baseline
    import bouquet.TokaMaker_interface as _tmi

    # (module, nested-helper name) pairs that perform a kin->eq regrid.
    sites = [
        (_tmi, "_kin_to_eq"), (_tmi, "_kin2eq"), (_tmi, "_k2e"), (_tmi, "_to_eq"),
        (_baseline, "to_eq"),
    ]
    trees = {}
    for mod, _ in sites:
        trees.setdefault(mod, ast.parse(inspect.getsource(mod)))

    seen = set()
    for mod, name in sites:
        bodies = []
        for node in ast.walk(trees[mod]):
            # `def name(...)` form
            if isinstance(node, ast.FunctionDef) and node.name == name:
                bodies.append(node)
            # `name = lambda ...` form
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
                if any(isinstance(t, ast.Name) and t.id == name
                       for t in node.targets):
                    bodies.append(node.value)
        assert bodies, (
            f"no kin->eq helper named {name!r} found in {mod.__name__}; the "
            "site was renamed or removed -- re-point this contract test"
        )
        for body in bodies:
            called = {c.func.id for c in ast.walk(body)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            attr_called = {
                f"{c.func.value.id}.{c.func.attr}" for c in ast.walk(body)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
            }
            assert "pchip_interp" in called, (
                f"{mod.__name__}.{name} (line {body.lineno}) no longer regrids "
                f"through pchip_interp; it calls {sorted(called | attr_called)}"
            )
            forbidden = {"np.interp", "interp1d", "_interp1d"}
            assert not (forbidden & (called | attr_called)), (
                f"{mod.__name__}.{name} (line {body.lineno}) uses a LINEAR "
                "regrid in the kin->eq role"
            )
            seen.add((mod.__name__, name))

    assert len(seen) == len(sites), "some kin->eq site went unchecked"


def _pressure_assembly_source():
    """The pressure-assembly hunk of ``reconstruct_equilibrium`` (step 4)."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    src = inspect.getsource(reconstruct_equilibrium)
    blk = re.search(r"# ---- 4\. Pressure and GS profiles ----(.*?)"
                    r"pprime_tmp = ", src, re.S)
    assert blk, "the step-4 pressure assembly is gone from reconstruct_equilibrium"
    return blk.group(1)


def test_p_fast_enters_the_recon_pressure_additively():
    """Sign contract, asserted against the SHIPPED assembly, not local algebra.

    A larger pressure means a larger Shafranov shift, so R_axis moves outboard
    and l_i(3) ~ 1/R_axis falls.  If p_fast were ever subtracted (or made to
    replace the thermal term) the mechanism behind the fix is inverted and the
    reconstruction's l_i knob compensates the wrong way.

    The previous version of this test built `thermal + p_fast` out of local
    numpy and asserted that a sum exceeds its addend -- true of arithmetic,
    and true whatever bouquet does, so it could not fail.
    """
    body = _pressure_assembly_source()
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))

    # the thermal base, then p_fast ADDED to it (not subtracted, not assigned)
    # (its eV -> J factor is physics.thermal_pressure_charge(jbs_loop): the
    # one ELEMENTARY_CHARGE under the loop, the frozen 1.6022e-19 otherwise)
    assert re.search(r"pres_tmp\s*=\s*thermal_pressure_charge\(jbs_loop\)"
                     r"\s*\*\s*\(ne\s*\*\s*te\s*\+\s*"
                     r"ni\s*\*\s*ti\)", code), (
        "the thermal single-ion base of the recon pressure changed shape")
    m = re.search(r"pres_tmp\s*=\s*pres_tmp\s*([+-])\s*np\.asarray\(p_fast",
                  code)
    assert m, (
        "p_fast is no longer folded into pres_tmp as `pres_tmp = pres_tmp "
        "+/- np.asarray(p_fast, ...)`; the recon may have stopped solving the "
        "fast pressure the draws solve")
    assert m.group(1) == "+", (
        "p_fast is SUBTRACTED from the reconstruction pressure; it must be "
        "added, as perturb_kinetic_equilibrium and the state anchor do")

    # and the impurity term likewise added, under its Z_imp guard
    assert re.search(r"pres_tmp\s*=\s*pres_tmp\s*\+\s*impurity_pressure\(",
                     code), "the impurity term is no longer added to pres_tmp"

    # numeric corroboration of the sign, using the real physics helper
    ne = np.linspace(5e19, 1e19, 32)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 32)
    from bouquet.physics import impurity_pressure
    assert np.all(impurity_pressure(ne, ni, ti, 6.0) >= 0.0), (
        "the impurity term must be non-negative, or 'additive' lowers the "
        "solve pressure")


class _DummyEqdsk:
    """Just enough of a geqdsk for the p_fast shape guard (no solve)."""

    def __init__(self, n=129):
        self.psi_N = np.linspace(0.0, 1.0, n)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(2.1e4, id="python-scalar"),
        pytest.param(np.float64(2.1e4), id="numpy-scalar"),
        pytest.param(np.array(2.1e4), id="0d-array"),
        pytest.param(np.array([2.1e4]), id="length-1-array"),
        pytest.param(np.zeros(150), id="kinetic-grid-length"),
        pytest.param(np.zeros((129, 1)), id="2d-column"),
    ],
)
def test_reconstruct_equilibrium_rejects_non_equilibrium_grid_p_fast(bad):
    """A scalar/length-1/wrong-length p_fast must raise, not broadcast.

    Before the guard, ``pres_tmp + np.asarray(p_fast)`` silently broadcast a
    scalar into a flat pressure offset, so the recon solved a different
    pressure from the draws (which regrid and would have raised) -- reviving
    the very inconsistency this module fixes.
    """
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    eqdsk = _DummyEqdsk(129)
    with pytest.raises(ValueError, match=r"p_fast must be a 1-D array"):
        # mygs=None is safe: the guard runs before anything touches the solver.
        reconstruct_equilibrium(
            None, eqdsk, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            p_fast=bad,
        )


def test_reconstruct_equilibrium_p_fast_error_names_the_expected_shape():
    """The ValueError must tell the caller what to do, not just that it failed."""
    from bouquet.TokaMaker_interface import reconstruct_equilibrium

    eqdsk = _DummyEqdsk(129)
    with pytest.raises(ValueError) as exc:
        reconstruct_equilibrium(
            None, eqdsk, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            p_fast=np.zeros(150),
        )
    msg = str(exc.value)
    assert "(129,)" in msg and "(150,)" in msg, "error must show both shapes"
    assert "pchip_interp" in msg, "error must point at the kin->eq remedy"


def test_Z_imp_is_scalar_by_design_and_rejects_arrays():
    """Z_imp needs no shape guard -- it is a scalar and coercion fails loudly.

    Documents why the p_fast guard has no Z_imp twin: ``impurity_pressure``
    does ``float(Z_imp)``, so a non-scalar raises rather than broadcasting.
    """
    from bouquet.physics import impurity_pressure

    ne = np.linspace(5e19, 1e19, 16)
    ni = ne * 0.9
    ti = np.linspace(3e3, 2e2, 16)
    with pytest.raises((TypeError, ValueError)):
        impurity_pressure(ne, ni, ti, np.array([6.0, 6.0]))


def _resolve_reconstruction_source():
    import bouquet.baseline as _baseline
    return inspect.getsource(_baseline._resolve_reconstruction)


def test_baseline_regrids_p_fast_before_handing_it_to_the_recon():
    """``_resolve_reconstruction`` must pass p_fast through ``to_eq``, not raw.

    Guards the specific bug shape: ``Baseline.p_fast`` is contractually on the
    KINETIC grid, so passing that array straight through would feed the recon a
    profile of the wrong length (and, before the shape guard, silently
    broadcast a length-1 one).

    Asserted on the real call site.  The previous version of this test only
    re-did the regrid itself with local numpy and never looked at
    ``_resolve_reconstruction``, so it passed regardless of what baseline.py
    actually handed over.
    """
    src = _resolve_reconstruction_source()

    # p_fast_eq is built by regridding the kinetic-grid array, and is None
    # (not zeros) when the fixed component is unset, so the default-off path
    # provably does not enter the new branch.
    assert re.search(r"p_fast_eq\s*=\s*to_eq\(p_fast_kin\)\s*"
                     r"if\s+fc\.p_fast\s+is\s+not\s+None\s+else\s+None", src), (
        "p_fast_eq is no longer `to_eq(p_fast_kin) if fc.p_fast is not None "
        "else None`; the recon may be receiving a kinetic-grid array, or "
        "zeros where None is required for the bitwise default-off path")

    # and it is p_fast_eq -- not p_fast_kin/p_fast -- that reaches the recon
    call = re.search(r"result\s*=\s*reconstruct_equilibrium\((.*?)\n\s*\)",
                     src, re.S)
    assert call, "could not find the reconstruct_equilibrium call site"
    args = call.group(1)
    assert re.search(r"p_fast\s*=\s*p_fast_eq", args), (
        "reconstruct_equilibrium is not being passed p_fast=p_fast_eq; a "
        "kinetic-grid array here is the original bug")

    # the returned Baseline keeps the KINETIC-grid array (unchanged contract)
    assert re.search(r"p_fast\s*=\s*p_fast_kin", src), (
        "Baseline.p_fast must stay on the kinetic grid for downstream consumers")

    # numeric corroboration that the two grids really are different lengths,
    # i.e. that the regrid above is load-bearing rather than decorative
    from bouquet.baseline import _resolve_fixed
    from bouquet.utils import pchip_interp

    psi_src = np.linspace(0.0, 1.0, 64)
    psi_kin = np.linspace(0.0, 1.0, 150)
    psi_eq = np.linspace(0.0, 1.0, 129)
    p_fast_kin = _resolve_fixed(2.1e4 * (1.0 - psi_src ** 2) ** 2,
                                psi_src, psi_kin)
    assert p_fast_kin.shape == psi_kin.shape
    p_fast_eq = pchip_interp(psi_kin, p_fast_kin, psi_eq)
    assert p_fast_eq.shape == psi_eq.shape != p_fast_kin.shape
    assert np.all(np.isfinite(p_fast_eq)) and p_fast_eq.max() > 0.0


def test_Z_imp_activates_symmetrically_on_the_recon_and_draw_paths():
    """The recon's Z_imp and the draws' Z_imp must come from the SAME source.

    ``_resolve_reconstruction`` reads ``getattr(fc, "Z_imp", None)`` for the
    reconstruction.  The draws instead read ``Baseline.Z_imp`` (run.py hands it
    to generate_bouquet and the forward / sigma=0 solves read it directly).
    If the returned Baseline does not carry the same value, then the day
    FixedComponentsConfig gains a Z_imp field the reconstruction starts adding
    impurity pressure while the draws do not -- recreating the recon-vs-draw
    inconsistency this module removes, through the getattr meant to guard it.
    """
    src = _resolve_reconstruction_source()

    assert re.search(r"Z_imp_recon\s*=\s*getattr\(fc,\s*[\"']Z_imp[\"'],\s*None\)",
                     src), "the recon's Z_imp source changed shape"

    # Resolve the two call sites structurally -- a plain substring search for
    # "Z_imp=Z_imp_recon" is satisfied by the reconstruct_equilibrium call
    # alone and would not notice the Baseline losing it.
    def _kwarg_source(callee):
        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == callee):
                for kw in node.keywords:
                    if kw.arg == "Z_imp":
                        return getattr(kw.value, "id", ast.dump(kw.value))
                return None      # call found, kwarg absent
        raise AssertionError(f"no call to {callee} in _resolve_reconstruction")

    assert _kwarg_source("reconstruct_equilibrium") == "Z_imp_recon", (
        "the reconstruction is no longer passed Z_imp=Z_imp_recon")
    assert _kwarg_source("Baseline") == "Z_imp_recon", (
        "the returned Baseline does not carry Z_imp=Z_imp_recon, so the recon "
        "and the draws would activate impurity pressure independently -- the "
        "recon would add impurity pressure the draws never see")

    # Baseline must actually have the field for the draws to read.
    from bouquet.baseline import Baseline
    assert "Z_imp" in getattr(Baseline, "__dataclass_fields__", {}), \
        "Baseline lost its Z_imp field; the draw path reads it"

    # Both are None today on this path, so activation is symmetric AND off.
    from bouquet.config import FixedComponentsConfig
    assert not hasattr(FixedComponentsConfig, "Z_imp"), (
        "FixedComponentsConfig gained a Z_imp field -- that is fine, but "
        "re-verify that the recon and the draws now both see it (this test's "
        "premise was that both are inert)")


# ---------------------------------------------------------------------------
#  the sigma=0 anchor's psi re-init must not lose the state it was handed
# ---------------------------------------------------------------------------
def test_sigma0_reinit_restores_state_before_a_failure_propagates():
    """``verify_sigma0_consistency``'s psi re-init needs a state guard.

    ``init_psi`` DISCARDS the reconstruction's converged state and installs a
    cold analytic psi.  Before the re-init was added, a failed solve here left
    ``mygs`` on that converged state -- benign, which is why the failure was
    allowed to propagate untouched.  With the re-init in place and no guard, a
    failure would instead hand the caller a cold, non-converged psi.

    The exception must STAY fatal (this is a verification routine; a silent
    fallback would defeat its purpose) -- so this asserts restore-then-re-raise,
    not swallow.
    """
    import bouquet.run as _run

    src = inspect.getsource(_run.Bouquet.verify_sigma0_consistency)

    tree = ast.parse(textwrap.dedent(src))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    names = [n.func.attr for n in calls]
    assert "init_psi" in names, "the sigma=0 psi re-init is gone"

    # the snapshot must be taken, and taken BEFORE init_psi discards the state
    assert "copy_eq" in names, (
        "init_psi runs without a copy_eq snapshot; a failed solve would leave "
        "the caller on the cold psi this method installed")
    order = {n: src.index(f".{n}(") for n in ("copy_eq", "init_psi")}
    assert order["copy_eq"] < order["init_psi"], (
        "the snapshot is taken AFTER init_psi has already discarded the state")

    # the solve is guarded, and the handler restores and then RE-RAISES
    handler = re.search(r"try:\s*\n\s*mygs\.solve\(\)\s*\n\s*except\s+"
                        r"(?:\w+)?\s*:?(?P<body>.*?)(?:\n\s{0,8}\S|\Z)",
                        src, re.S)
    assert handler, "mygs.solve() in the sigma=0 anchor is not wrapped at all"
    body = handler.group("body")
    assert "replace_eq" in body, (
        "the failure path does not restore the snapshot; the caller is left on "
        "a cold, non-converged psi")
    assert re.search(r"^\s*raise\s*$", body, re.M), (
        "the failure is being swallowed -- verify_sigma0_consistency must stay "
        "fatal, it only needs to restore state on the way out")
