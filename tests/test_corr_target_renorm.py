"""The corrector target must carry Ip_target before the corrector runs (#29).

Measured on eight cases, the reconstruction's corrector target carried
-3.9 % to +18.7 % of Ip (golden +6.0 %) while every solver output sat at
Ip to <= 0.06 %: the corrector was chasing a total the solver is constrained
to refuse.  `_renormalize_target_to_Ip` scales the target uniformly so its
physical current integral equals Ip_target -- AFFINE-exact, because the
measure is int(w*J) + c and a plain ratio misses by (1-s)*c.

Solve-free: reuses the analytic large-aspect-ratio stub from
test_r2_inductive_share, which exposes exactly the accessors
fsa_current_geometry / eq_jphi_profile need.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from bouquet.TokaMaker_interface import _renormalize_target_to_Ip
from bouquet.utils import fsa_current_geometry, Ip_fsa_weights, eq_jphi_profile
from test_r2_inductive_share import _StubGS, _psi_grid, _split


class _StubSolver(_StubGS):
    """fsa_current_geometry/eq_jphi_profile read these off the SOLVER (live
    mygs), not a copy_eq snapshot -- delegate to the stub equilibrium."""
    psi_bounds = _StubGS(1.0).copy_eq().psi_bounds

    def get_q(self, *a, **k):
        return self._eq.get_q(*a, **k)

    def get_profiles(self, *a, **k):
        return self._eq.get_profiles(*a, **k)


def _measure(gs, psi, j):
    geom = fsa_current_geometry(gs, psi)
    probe = eq_jphi_profile(geom, "jphi-linterp", eq=gs)
    sign = 1.0 if float(np.dot(probe, j)) > 0.0 else -1.0
    w, c = Ip_fsa_weights(geom, convention="jphi-linterp", pprime_sign=sign)
    return float(trapezoid(w * j, psi)) + c, c


def test_renormalised_target_carries_ip_target_exactly():
    psi = _psi_grid(); j_ind, j_oth = _split(psi); target = j_ind + j_oth
    gs = _StubSolver(Ip=1.0e6)
    Ip_before, c = _measure(gs, psi, target)
    assert c != 0.0, "stub's affine constant vanished; the affine case is untested"
    assert abs(Ip_before / 1.0e6 - 1.0) > 1e-3, "fixture target already at Ip"

    scaled, factor = _renormalize_target_to_Ip(gs, psi, target, 1.0e6, 1e-3)

    Ip_after, _ = _measure(gs, psi, scaled)
    assert Ip_after == pytest.approx(1.0e6, rel=1e-12)
    assert factor != 1.0


def test_plain_ratio_would_miss_by_the_affine_constant():
    """Negative control for the affine-exact claim."""
    psi = _psi_grid(); j_ind, j_oth = _split(psi); target = j_ind + j_oth
    gs = _StubSolver(Ip=1.0e6)
    Ip_before, c = _measure(gs, psi, target)
    naive = target * (1.0e6 / Ip_before)
    Ip_naive, _ = _measure(gs, psi, naive)
    expected_miss = (1.0 - 1.0e6 / Ip_before) * c
    assert Ip_naive - 1.0e6 == pytest.approx(expected_miss, rel=1e-9)
    assert abs(Ip_naive - 1.0e6) > 1.0   # the miss is real on this stub (> 1 A)


def test_scaling_is_uniform_so_the_shape_and_li_are_untouched():
    psi = _psi_grid(); j_ind, j_oth = _split(psi); target = j_ind + j_oth
    scaled, factor = _renormalize_target_to_Ip(_StubSolver(Ip=1.0e6), psi,
                                               target, 1.0e6, 1e-3)
    np.testing.assert_allclose(scaled / target, factor, rtol=1e-13)


def test_measure_failure_leaves_the_target_untouched():
    class _Broken:
        def get_q(self, *a, **k):
            raise RuntimeError("no geometry")
        def get_stats(self, *a, **k):
            return {"Ip": 1.0e6}
        psi_bounds = (0.0, 1.0)
    psi = _psi_grid(); j_ind, j_oth = _split(psi); target = j_ind + j_oth
    scaled, factor = _renormalize_target_to_Ip(_Broken(), psi, target,
                                               1.0e6, 1e-3)
    assert factor == 1.0
    np.testing.assert_array_equal(scaled, target)


# ---------------------------------------------------------------------------
#  structural guard: every corrector call site renormalises first
# ---------------------------------------------------------------------------
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "bouquet" / "TokaMaker_interface.py"


def _code(src):
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def test_every_corrector_call_site_renormalises_its_target_first():
    """Each `_corrective_jphi_iteration(` call in TokaMaker_interface must be
    preceded (within the same block) by a `_renormalize_target_to_Ip(` call.
    A third site that bypasses it would reintroduce #29 silently."""
    code = _code(_SRC.read_text())
    calls = [m.start() for m in re.finditer(r"_corrective_jphi_iteration\(", code)
             if not code[max(0, m.start() - 4):m.start()].endswith("def ")]
    assert len(calls) >= 2, "expected the recon and per-draw call sites"
    for pos in calls:
        window = code[max(0, pos - 1500):pos]
        assert "_renormalize_target_to_Ip(" in window, (
            f"corrector call at offset {pos} is not preceded by a target "
            "renormalisation (issue #29)")


def test_the_guard_is_not_satisfied_by_a_comment():
    """Negative control: a renormalise mention in a comment must not count."""
    fake = ("# _renormalize_target_to_Ip( would go here\n"
            "out = _corrective_jphi_iteration(mygs, psi_N, t, pp, Ip, pax, pad)\n")
    code = _code(fake)
    assert "_renormalize_target_to_Ip(" not in code
