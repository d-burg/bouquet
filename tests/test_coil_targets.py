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

    def test_ignores_coils_absent_from_the_mesh(self):
        """A measurement source is not mesh-specific: DIII-D pf_active has 24
        circuits, the shipped mesh models 20. coil_reg_term raises KeyError on
        an unknown coil, which would kill setup_solver."""
        spec = [{"coils": {"F1A": 1.0}, "target": 5.0, "weight": 1.0},
                {"coils": {"E567UP": 1.0}, "target": 7.0, "weight": 1.0}]
        with pytest.warns(UserWarning, match="E567UP"):
            got = self._run(spec)
        names = {c for t in got for c in t["coils"]}
        assert "E567UP" not in names
        by = {list(t["coils"])[0]: t for t in got}
        assert by["F1A"]["target"] == 5.0

    def test_unnamed_coils_still_get_a_zero_target(self):
        got = self._run([{"coils": {"F1A": 1.0}, "target": 1234.0, "weight": 9.0}])
        by = {list(t["coils"])[0]: t for t in got}
        assert by["F6A"]["target"] == 0.0
        assert "#VSC" in by


class TestCoilInit:
    """SolverConfig.coil_init seeds the inverse iterate; it must not constrain."""

    class _GS:
        def __init__(self, sets):
            self.coil_sets = list(sets)
            self._cur = {k: 0.0 for k in sets}
            self.set_calls = []
        def init_psi(self, *a): pass
        def get_coil_currents(self): return dict(self._cur), None
        def set_coil_currents(self, cur):
            self.set_calls.append(dict(cur)); self._cur = dict(cur)

    def test_seeds_only_known_coils_and_keeps_the_rest(self):
        """A 24-circuit measurement against a 20-set mesh: unknown names are
        dropped, known ones are seeded, unlisted known coils keep their value."""
        gs = self._GS(["F1A", "F6A", "ECOILA"])
        gs._cur["ECOILA"] = -5.0
        ci = {"F1A": 1000.0, "E567UP": 99.0}           # E567UP not in mesh
        known = set(gs.coil_sets)
        use = {k: float(v) for k, v in ci.items() if k in known}
        cur, _ = gs.get_coil_currents(); cur.update(use); gs.set_coil_currents(cur)
        assert gs.set_calls[-1] == {"F1A": 1000.0, "F6A": 0.0, "ECOILA": -5.0}
        assert "E567UP" not in gs.set_calls[-1]

    def test_config_field_defaults_to_none(self):
        from bouquet.config import SolverConfig
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(SolverConfig)}
        assert "coil_init" in f
        assert SolverConfig(mesh_path="x").coil_init is None


def test_turns_come_from_the_device_registry():
    from bouquet.coil_targets import TURNFC_D3D
    from bouquet.devices import get_device
    assert TURNFC_D3D == get_device("DIII-D").turns
    assert TURNFC_D3D["F1A"] == 58.0 and TURNFC_D3D["F6A"] == 55.0 and "ECOILA" not in TURNFC_D3D
