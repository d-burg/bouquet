"""jbs_delta_mode sigma=0 reference: scale semantics + the scoping regression.

The first version of the issue-#44 fix passed ``scale_jBS`` -- a per-draw
local that is only bound INSIDE the draw loop -- into the pre-loop cache
call.  Python scoping made that an UnboundLocalError, which the cache
block's blanket ``except Exception`` swallowed into a silent
cache-disable fallback: jbs_delta_mode never used its reference again and
nothing went red.  These tests pin both the correct center semantics and
the scoping property that failed.
"""
import ast
import inspect

import pytest

import bouquet.TokaMaker_interface as tmi
from bouquet.TokaMaker_interface import sigma0_reference_scale


class TestReferenceScale:
    def test_no_range_means_unit_scale(self):
        """Without a configured range every draw runs at 1.0 -- the
        reference must match or the delta stops telescoping at sigma=0."""
        assert sigma0_reference_scale(None) == 1.0

    def test_center_is_the_range_midpoint(self):
        # run.py re-centres the configured range on bl.bs_scale, so for
        # gc.jBS_scale_range=(0.85, 1.15) and bs_scale=0.70 the passed
        # range is (0.595, 0.805) and the uniform draws' mean is 0.70.
        assert sigma0_reference_scale((0.595, 0.805)) == pytest.approx(0.70)

    def test_asymmetric_range_uses_its_own_mean(self):
        """An asymmetric range's uniform mean is still the midpoint --
        NOT bs_scale -- so the reference follows the draws, not the label."""
        assert sigma0_reference_scale((0.8, 1.2)) == pytest.approx(1.0)
        assert sigma0_reference_scale((0.9, 1.2)) == pytest.approx(1.05)

    def test_cache_call_uses_the_helper(self):
        """The pre-loop SWB cache must take its scale from
        sigma0_reference_scale, not from any per-draw local."""
        src = inspect.getsource(tmi)
        assert "_scale_ref = sigma0_reference_scale(jBS_scale_range)" in src


class TestNoLoadBeforeStore:
    def test_generate_bouquet_locals_are_bound_before_use(self):
        """The regression class itself: no local of generate_bouquet may be
        loaded on a line before its first (lexical) binding.  This is what
        made the original fix an UnboundLocalError swallowed by the cache
        block's except."""
        tree = ast.parse(inspect.getsource(tmi))
        gb = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "generate_bouquet")
        params = {a.arg for a in gb.args.args + gb.args.kwonlyargs}
        if gb.args.vararg:
            params.add(gb.args.vararg.arg)
        if gb.args.kwarg:
            params.add(gb.args.kwarg.arg)
        # names bound inside nested scopes (defs, lambdas, comprehensions --
        # all have their own scope in py3) don't shadow the outer function
        nested = [n for n in ast.walk(gb)
                  if isinstance(n, (ast.FunctionDef, ast.Lambda,
                                    ast.GeneratorExp, ast.ListComp,
                                    ast.SetComp, ast.DictComp))
                  and n is not gb]

        def in_nested(node):
            return any(f.lineno <= node.lineno <= (f.end_lineno or f.lineno)
                       for f in nested)

        stores, loads = {}, {}
        for node in ast.walk(gb):
            if isinstance(node, ast.Name) and not in_nested(node):
                book = (stores if isinstance(node.ctx, (ast.Store,))
                        else loads if isinstance(node.ctx, ast.Load)
                        else None)
                if book is not None:
                    book.setdefault(node.id, node.lineno)
                    book[node.id] = min(book[node.id], node.lineno)
        offenders = sorted(
            (name, loads[name], stores[name])
            for name in loads
            if name in stores and name not in params
            and loads[name] < stores[name])
        assert not offenders, (
            "locals loaded before their first binding (UnboundLocalError "
            f"risk, the issue-#44 fix regression): {offenders}")
