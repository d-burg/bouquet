"""The numpy compatibility helper (``bouquet.kinrecon._numpy_compat``): numpy 2 calls the trapezoid-rule integrator
``trapezoid``, numpy 1 ``trapz``.  Both branches of the helper are exercised here with stand-in modules (one numpy is
installed per environment); the installed numpy's own branch is checked as well.  Exact comparisons: the helper
returns numpy's function itself, so there is nothing to approximate."""
import re
import types
from pathlib import Path

import numpy as np

from bouquet.kinrecon import _numpy_compat as NC

KINRECON = Path(__file__).resolve().parents[2] / "bouquet" / "kinrecon"


def _f2(y, x=None):
    return "numpy-2 name"


def _f1(y, x=None):
    return "numpy-1 name"


def test_numpy2_branch_takes_trapezoid():
    mod = types.SimpleNamespace(trapezoid=_f2, trapz=_f1)
    assert NC.pick_trapezoid(mod) is _f2


def test_numpy1_branch_falls_back_to_trapz():
    mod = types.SimpleNamespace(trapz=_f1)
    assert NC.pick_trapezoid(mod) is _f1


def test_installed_numpy_branch_is_numpys_own_function():
    own = getattr(np, "trapezoid", None)
    assert NC.trapezoid is (own if own is not None else np.trapz)
    x = np.array([0.0, 1.0, 3.0])
    assert NC.trapezoid(x, x) == 4.5          # int_0^3 x dx, exact for a linear integrand


def test_package_calls_the_helper_only():
    pat = re.compile(r"\bnp\.(trapezoid|trapz)\(")
    hits = [f"{p.relative_to(KINRECON)}:{i}" for p in KINRECON.rglob("*.py") if p.name != "_numpy_compat.py"
            for i, line in enumerate(p.read_text().splitlines(), 1) if pat.search(line)]
    assert hits == []
