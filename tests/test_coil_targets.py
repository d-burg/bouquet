"""Coil regularisation targets from measured currents."""
import pytest

from bouquet.coil_targets import TURNFC_D3D, coil_reg_from_measured


def _by_coil(spec):
    return {list(t["coils"])[0]: t for t in spec}


class TestTargets:
    def test_applies_turnfc_to_f_coils(self):
        by = _by_coil(coil_reg_from_measured({"F6A": 1000.0, "F1A": 1000.0}))
        assert by["F6A"]["target"] == pytest.approx(55.0 * 1000.0)
        assert by["F1A"]["target"] == pytest.approx(58.0 * 1000.0)

    def test_ecoils_convert_at_unity(self):
        """The D3D mesh already carries E-coil turns (nturns sums to 61.0)."""
        spec = coil_reg_from_measured({"ECOILA": -2464.0})
        assert spec[0]["target"] == pytest.approx(-2464.0)

    def test_per_coil_weight_override(self):
        by = _by_coil(coil_reg_from_measured(
            {"F5A": 1.0, "F6A": 1.0}, weights={"F5A": 100.0}, default_weight=2.0))
        assert by["F5A"]["weight"] == 100.0
        assert by["F6A"]["weight"] == 2.0

    def test_skips_non_finite(self):
        assert coil_reg_from_measured({"F1A": float("nan")}) == []

    def test_turnfc_matches_dprobe(self):
        """5*58, 2*55, 58, 55 per EFIT dprobe.dat_20200406, both A and B sets."""
        for side in ("A", "B"):
            got = [TURNFC_D3D[f"F{i}{side}"] for i in range(1, 10)]
            assert got == [58.0]*5 + [55.0, 55.0] + [58.0, 55.0]


class TestApplyCoilReg:
    """_apply_coil_reg must honour the config AND survive the solver reset."""

    class _GS:
        def __init__(self, sets): self.coil_sets = list(sets); self.installed = None
        def coil_reg_term(self, coils, target=0.0, weight=1.0):
            return {"coils": dict(coils), "target": target, "weight": weight}
        def set_coil_reg(self, reg_terms=None): self.installed = list(reg_terms)

    def _run(self, spec):
        from bouquet.run import Bouquet
        gs = self._GS(["F1A", "F6A"])
        obj = Bouquet.__new__(Bouquet)
        obj.config = type("C", (), {"solver": type("S", (), {"coil_reg": spec})()})()
        Bouquet._apply_coil_reg(obj, gs)
        return gs.installed

    def test_default_is_unchanged_when_unset(self):
        got = self._run([])
        assert all(t["target"] == 0.0 for t in got)
        assert any("#VSC" in t["coils"] for t in got)

    def test_configured_targets_are_installed(self):
        got = self._run([{"coils": {"F1A": 1.0}, "target": 1234.0, "weight": 9.0}])
        by = {list(t["coils"])[0]: t for t in got}
        assert by["F1A"]["target"] == 1234.0 and by["F1A"]["weight"] == 9.0

    def test_unnamed_coils_still_get_a_zero_target(self):
        got = self._run([{"coils": {"F1A": 1.0}, "target": 1234.0, "weight": 9.0}])
        by = {list(t["coils"])[0]: t for t in got}
        assert by["F6A"]["target"] == 0.0
        assert "#VSC" in by
