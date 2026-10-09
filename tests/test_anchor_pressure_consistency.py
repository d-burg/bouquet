"""Every state-anchor solve must use the full solve pressure (issue #35).

Defect 1 of #35: `perturb_kinetic_equilibrium` assembled `pres_tmp` (thermal
+ p_fast + impurity + p_diff) and used it at every solve site EXCEPT the
state anchor, which anchored at the thermal-only `pressure` argument.  The
anchor was therefore a genuinely different equilibrium from the
reconstruction that produced its own `input_j_phi` -- on a 27 kPa-p_fast
case the archived j_phi integrated +4.1 % of Ip high on the anchor
geometry, and the sigma=0 R2 invariant read 1058 % of its bar.  A fifth
instance of the same class sat in `generate_bouquet`'s jBS-delta cache
anchor.  Measured with the fix, the four p_fast demo archives collapse from
+1.1..+4.1 % split bias to the controls' +0.07..0.19 % band, with the
controls bit-unchanged.

These tests are structural (solve-free): they pin the two thermal-anchor
spellings out of the module so the sites cannot quietly revert.  The
solver-side behaviour is pinned by the sigma=0 R2 invariant itself
(`tests/test_seeded_reproducibility.py`).
"""
import re
from pathlib import Path

import pytest

_SRC = (Path(__file__).resolve().parents[1] / "bouquet"
        / "TokaMaker_interface.py")

#: The spellings of a thermal-only anchor, whitespace-tolerant: the inline
#: forms the module used before ``bouquet.edge_pressure`` (which must not come
#: back) and the same mistake written through that module's helper (every
#: P' / axis target of the module is now built there).
_THERMAL_PAX = re.compile(
    r"pax\s*=\s*pressure\s*\[\s*0\s*\]"
    r"|solver_pax\(\s*pressure\s*,")
_THERMAL_PP = re.compile(
    r"pchip_derivative\(\s*psi_N\s*,\s*pressure\s*\)"
    r"|solver_pp(?:rime|_profile)\(\s*psi_N\s*,\s*pressure\s*,")


def _code_lines(src):
    """The module with comment-only lines stripped, so a historical note
    quoting the legacy spelling cannot fail the guard (Copilot review)."""
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))


def test_no_solve_site_anchors_at_thermal_only_pressure():
    code = _code_lines(_SRC.read_text())

    offenders = _THERMAL_PAX.findall(code) + _THERMAL_PP.findall(code)

    assert not offenders, (
        f"TokaMaker_interface.py anchors a solve at the thermal-only "
        f"`pressure` argument again: {offenders}.  Use pres_tmp (per-draw "
        "sites) or pressure_solve (baseline sites) -- issue #35 Defect 1."
    )


@pytest.mark.parametrize("variant", [
    "pax=pressure[0]",
    "pax = pressure[ 0 ]",
    "pchip_derivative(psi_N, pressure)",
    "pchip_derivative( psi_N , pressure )",
    "pax=solver_pax(pressure, _edge)",
    "solver_pp_profile(psi_N, pressure, psi_range, _edge)",
    "solver_pprime( psi_N , pressure , psi_range, _edge)",
])
def test_the_guard_regexes_catch_the_stock_spellings(variant):
    """Negative control: the exact pre-fix spellings (and spaced variants)
    must match, or the test above can pass vacuously."""
    assert _THERMAL_PAX.search(variant) or _THERMAL_PP.search(variant)


@pytest.mark.parametrize("ok", [
    "pax=pres_tmp[0]",
    "pax=float(pressure_solve[0])",
    "pchip_derivative(psi_N, pres_tmp)",
    "pchip_derivative(psi_N, pressure_solve)",
    "pax=solver_pax(pres_tmp, _edge)",
    "pax=solver_pax(pressure_solve, _edge)",
    "solver_pp_profile(psi_N, pres_tmp, psi_range, _edge)",
    "solver_pp_profile(psi_N, pressure_solve, _psi_range_b, _edge)",
    "solver_pprime(psi_N, pres_tmp, psi_range, _edge)",
])
def test_the_guard_regexes_pass_the_fixed_spellings(ok):
    """The consistent spellings must NOT match -- a guard that also fires on
    the fix would just get deleted."""
    assert not (_THERMAL_PAX.search(ok) or _THERMAL_PP.search(ok))


def test_the_anchor_sites_use_the_full_pressure():
    """Positive assertion: the per-draw anchor uses pres_tmp and the
    jBS-delta cache anchor uses pressure_solve, as calls, not comments."""
    code = _code_lines(_SRC.read_text())

    # (through the one helper that builds every P' and axis target since
    # bouquet.edge_pressure: solver_pp_profile / solver_pprime / solver_pax)
    assert re.search(
        r"solver_pp(?:rime|_profile)\(\s*psi_N\s*,\s*pres_tmp\s*,", code)
    assert re.search(
        r"solver_pp(?:rime|_profile)\(\s*psi_N\s*,\s*pressure_solve\s*,",
        code)
    assert re.search(r"solver_pax\(\s*pres_tmp\s*,", code)
    assert re.search(r"solver_pax\(\s*pressure_solve\s*,", code)
    # and no inline P' / axis-target construction is left in the module
    assert not re.search(r"\[\s*-1\s*\]\s*=\s*0\.0", code)
    assert not re.search(r"pax\s*=\s*(?:float\()?\w+\[\s*0\s*\]", code)
