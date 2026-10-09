"""One electron-charge constant (``physics.ELEMENTARY_CHARGE``).

The modelling-source forward solve used 1.602176634e-19 while the draws used
1.6022e-19, so a sigma=0 draw's thermal pressure sat 1.46e-5 relative above
the reconstruction's.  There is now ONE constant, read everywhere from
``bouquet.physics``.  The frozen legacy path (``jbs_self_consistent=False``)
keeps its historical value, named ``ELEMENTARY_CHARGE_LEGACY`` and selected
only through ``thermal_pressure_charge``, so it stays bit for bit.

Synthetic inputs only.
"""
import os
import re

import numpy as np
import pytest

import bouquet.physics as P

_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "bouquet")


def test_the_constant_and_the_frozen_legacy_value():
    assert P.ELEMENTARY_CHARGE == 1.602176634e-19
    assert P.ELEMENTARY_CHARGE_LEGACY == 1.6022e-19
    rel = P.ELEMENTARY_CHARGE_LEGACY / P.ELEMENTARY_CHARGE - 1.0
    assert rel == pytest.approx(1.4584e-5, rel=1e-4)


@pytest.mark.parametrize("loop,expected", [
    (None, "legacy"), (False, "legacy"), ({}, "legacy"),
    ({"enabled": False}, "legacy"), ({"enabled": True}, "exact"),
    (True, "exact")])
def test_the_run_picks_the_constant_by_the_loop_flag(loop, expected):
    want = (P.ELEMENTARY_CHARGE if expected == "exact"
            else P.ELEMENTARY_CHARGE_LEGACY)
    assert P.thermal_pressure_charge(loop) == want


def test_every_module_reads_the_one_constant():
    import bouquet.io.imas as imas
    import bouquet.io.pfile as pfile
    import bouquet.sampling as sampling
    assert P._EC is P.ELEMENTARY_CHARGE or P._EC == P.ELEMENTARY_CHARGE
    assert imas._EC == P.ELEMENTARY_CHARGE
    assert pfile._NT_TO_KPA == P.ELEMENTARY_CHARGE * 1e20
    # back-compat export: the frozen legacy value, never the physics one
    assert sampling.EC == P.ELEMENTARY_CHARGE_LEGACY


def test_no_other_electron_charge_literal_in_the_package():
    """Only the two definitions in physics.py may spell the number."""
    pat = re.compile(r"1\.602\d*e-19")
    hits = []
    for root, _dirs, files in os.walk(_PKG):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    code = line.split("#", 1)[0]
                    if pat.search(code):
                        hits.append((os.path.relpath(path, _PKG), i,
                                     line.strip()))
    allowed = {("physics.py", "ELEMENTARY_CHARGE = 1.602176634e-19"),
               ("physics.py", "ELEMENTARY_CHARGE_LEGACY = 1.6022e-19")}
    extra = [h for h in hits if (h[0], h[2]) not in allowed]
    assert not extra, extra
    assert len(hits) == 2, hits


def test_the_loop_draw_pressure_uses_the_exact_constant():
    """A sigma=0 self-consistent draw on the toy solver builds its thermal
    pressure with ELEMENTARY_CHARGE (the pressure-match integral it forms)."""
    from _pytest.monkeypatch import MonkeyPatch

    import test_sigma0_identity_stages as T
    mp = MonkeyPatch()
    try:
        toy = T.toy._get_wrapped_function()(mp)
        seen = []
        toy.flux_integral = lambda psi, y: (seen.append(np.array(y, float)),
                                            float(np.sum(y)))[1]
        req, jbs, fx = T._reconstruct(toy, kappa_short=0.997)
        dv, off, _n = T._delivered(toy, req, jbs, fx)
        F = toy.copy_eq()
        T._draw(toy, "standard", dv["request"], dv["j_inductive"],
                T._li(F.achieved), T._settings(), offset=off)
    finally:
        mp.undo()
    thermal = T._NE * T._TE + T._NI * T._TI
    # [0] the target handed in, [1] the draw's own pres_tmp (sigma=0: the
    # same kinetics after the grid round trip, so equal to rounding -- and
    # 1.46e-5 away from what the frozen legacy factor would give)
    np.testing.assert_array_equal(seen[0], P.ELEMENTARY_CHARGE * thermal)
    np.testing.assert_allclose(seen[1], P.ELEMENTARY_CHARGE * thermal,
                               rtol=1e-14, atol=0.0)
    assert np.max(np.abs(seen[1] / (P.ELEMENTARY_CHARGE_LEGACY * thermal)
                         - 1.0)) > 1e-5


def test_the_reconstruction_and_generate_pressures_are_loop_gated():
    """Source check: every thermal-pressure site of the g-file
    reconstruction, the draw and generate_bouquet takes its factor from
    thermal_pressure_charge (the frozen value only when the loop is off)."""
    import inspect

    import bouquet.TokaMaker_interface as TI
    # reconstruct_equilibrium: the anchor's and the step-4 pressure, plus
    # the step-4 per-species copies of the core-pressure hollowness record
    for fn, n in ((TI.reconstruct_equilibrium, 4),
                  (TI.perturb_kinetic_equilibrium, 1),
                  (TI.generate_bouquet, 1)):
        src = inspect.getsource(fn)
        assert src.count("thermal_pressure_charge(jbs_loop)") == n, fn
        assert not re.search(r"\bEC \*", src), fn
