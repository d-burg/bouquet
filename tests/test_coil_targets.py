"""Coil regularisation targets from measured currents."""
import json

import numpy as np
import pytest

from bouquet.coil_targets import (DEFAULT_W0, MIN_SAFE_W0, TURNFC_D3D,
                                  coil_reg_from_measured,
                                  inverse_variance_weights)

D3D = {"device": "DIII-D"}


def _by_coil(spec):
    return {list(t["coils"])[0]: t for t in spec}


class TestTargets:
    def test_applies_turnfc_to_f_coils(self):
        by = _by_coil(coil_reg_from_measured({"F6A": 1000.0, "F1A": 1000.0}, **D3D))
        assert by["F6A"]["target"] == pytest.approx(55.0 * 1000.0)
        assert by["F1A"]["target"] == pytest.approx(58.0 * 1000.0)

    def test_ecoils_convert_at_unity(self):
        """The D3D mesh already carries E-coil turns (nturns sums to 61.0)."""
        spec = coil_reg_from_measured({"ECOILA": -2464.0}, **D3D)
        assert spec[0]["target"] == pytest.approx(-2464.0)

    def test_per_coil_weight_override(self):
        by = _by_coil(coil_reg_from_measured(
            {"F5A": 1.0, "F6A": 1.0}, weights={"F5A": 100.0}, W0=20.0, **D3D))
        assert by["F5A"]["weight"] == 100.0
        assert by["F6A"]["weight"] == 20.0

    def test_skips_non_finite(self):
        assert coil_reg_from_measured({"F1A": float("nan")}, **D3D) == []

    def test_turnfc_values(self):
        """5*58, 2*55, 58, 55 on both the A and B sets."""
        for side in ("A", "B"):
            got = [TURNFC_D3D[f"F{i}{side}"] for i in range(1, 10)]
            assert got == [58.0] * 5 + [55.0, 55.0] + [58.0, 55.0]


class TestTurnsMustBeStated:
    """A measured circuit current converts with the DEVICE's turns; using
    another machine's table silently is a 1x / 58x / 61x error in a TARGET."""

    def test_no_device_and_no_turns_is_refused(self):
        with pytest.raises(ValueError, match="no safe default"):
            coil_reg_from_measured({"F1A": 1000.0})

    def test_explicit_turns_win(self):
        spec = coil_reg_from_measured({"PF1": 1000.0}, turns={"PF1": 3.0})
        assert spec[0]["target"] == pytest.approx(3000.0)

    def test_turns_come_from_the_device_registry(self):
        from bouquet.devices import get_device
        assert TURNFC_D3D == get_device("DIII-D").turns
        assert TURNFC_D3D["F1A"] == 58.0 and TURNFC_D3D["F6A"] == 55.0
        assert "ECOILA" not in TURNFC_D3D

    def test_unknown_device_is_loud(self):
        with pytest.raises(KeyError, match="unknown device"):
            coil_reg_from_measured({"F1A": 1.0}, device="NOT-A-TOKAMAK")


class TestInverseVarianceWeights:
    """w_i = W0 * (sigma_ref/sigma_i)^2 -- the form the recommendation was
    measured with.  A flat weight is the wrong shape: the measured precision is
    absolute and differs ~10x between coil families."""

    SIG = {"F1A": 400.0, "F6A": 400.0, "F9A": 800.0, "ECOILA": 4200.0}

    def test_reference_family_median_sets_the_scale(self):
        w = inverse_variance_weights(self.SIG, W0=100.0)
        assert w["F1A"] == pytest.approx(100.0)          # at the F-coil median
        assert w["F9A"] == pytest.approx(25.0)           # 2x sigma -> 1/4 weight
        assert w["ECOILA"] < w["F9A"]                    # 10x sigma -> ~1/100

    def test_weight_scales_with_W0(self):
        a = inverse_variance_weights(self.SIG, W0=10.0)
        b = inverse_variance_weights(self.SIG, W0=100.0)
        assert all(b[k] == pytest.approx(10.0 * a[k]) for k in a)

    def test_equal_sigmas_reduce_to_a_flat_W0(self):
        w = inverse_variance_weights({"F1A": 7.0, "F2A": 7.0}, W0=42.0)
        assert w == {"F1A": pytest.approx(42.0), "F2A": pytest.approx(42.0)}

    def test_drops_coils_with_no_usable_sigma(self):
        w = inverse_variance_weights({**self.SIG, "F2A": 0.0, "F3A": float("nan")})
        assert "F2A" not in w and "F3A" not in w

    def test_a_weak_pull_is_refused_not_silently_shipped(self):
        """The PR's own measurements say a weak pull resolves the inner/mid coil
        null space the WRONG way -- worse than not regularising, with no symptom.
        W0 = 1 (the old shipped default_weight) is squarely in that regime."""
        with pytest.raises(ValueError, match="MIN_SAFE_W0"):
            inverse_variance_weights(self.SIG, W0=1.0)
        with pytest.raises(ValueError, match="MIN_SAFE_W0"):
            coil_reg_from_measured({"F1A": 1.0}, W0=MIN_SAFE_W0, **D3D)
        with pytest.warns(UserWarning, match="MIN_SAFE_W0"):
            inverse_variance_weights(self.SIG, W0=1.0, allow_weak=True)

    def test_the_shipped_default_is_the_recommended_weight(self):
        assert DEFAULT_W0 == 100.0 and MIN_SAFE_W0 == 5.0
        by = _by_coil(coil_reg_from_measured({"F1A": 1000.0, "F9A": 1000.0},
                                             sigma=self.SIG, **D3D))
        assert by["F1A"]["weight"] == pytest.approx(DEFAULT_W0)
        assert by["F9A"]["weight"] == pytest.approx(DEFAULT_W0 / 4.0)
        # ... and with no sigma the fallback is the flat W0, still above the floor
        flat = _by_coil(coil_reg_from_measured({"F1A": 1000.0}, **D3D))
        assert flat["F1A"]["weight"] == pytest.approx(DEFAULT_W0)

    def test_deprecated_flat_weight_warns(self):
        with pytest.warns(UserWarning, match="deprecated"):
            by = _by_coil(coil_reg_from_measured({"F1A": 1.0}, default_weight=1.0, **D3D))
        assert by["F1A"]["weight"] == 1.0


class TestCoilsWithNoSigma:
    """A coil measured but with no usable sigma must not be RELEASED.

    ``sigma_from_pf_active`` omits any coil whose ``data_error_upper`` is missing
    or non-positive, so it survives into ``measured`` but not into ``sigma``.
    Dropping it from the spec is the one thing that must not happen: every coil
    no term names is given target=0 at weight 1 by ``_apply_coil_reg``, i.e. the
    pull toward zero the whole module exists to remove.
    """

    SIG = {"F1A": 100.0, "F2A": 100.0}

    def test_the_coil_is_pinned_at_its_measurement_at_the_flat_reference_weight(self):
        meas = {"F1A": 1000.0, "F2A": 1000.0, "F6A": 2000.0}   # F6A has no sigma
        with pytest.warns(UserWarning, match="F6A"):
            by = _by_coil(coil_reg_from_measured(meas, sigma=self.SIG, **D3D))
        assert set(by) == {"F1A", "F2A", "F6A"}
        assert by["F6A"]["target"] == pytest.approx(2000.0 * 55.0)
        assert by["F6A"]["weight"] == pytest.approx(DEFAULT_W0)
        assert by["F6A"]["weight"] > 1.0      # not the toward-zero default strength

    def test_the_warning_says_what_happened_and_what_would_have(self):
        with pytest.warns(UserWarning) as rec:
            coil_reg_from_measured({"F1A": 1.0, "F6A": 1.0}, sigma=self.SIG, **D3D)
        msg = str(rec[0].message)
        assert "F6A" in msg and "no usable sigma" in msg
        assert "target=0, weight=1" in msg     # the outcome that was avoided

    def test_an_explicit_weight_covers_the_gap_without_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            by = _by_coil(coil_reg_from_measured(
                {"F1A": 1.0, "F6A": 1.0}, sigma=self.SIG,
                weights={"F6A": 7.0}, **D3D))
        assert by["F6A"]["weight"] == 7.0


class TestPfActiveReaders:
    @staticmethod
    def _dd(tmp_path):
        dd = {"pf_active": {"coil": [
            {"name": "F1A", "current": {"time": [1.0, 2.0, 3.0],
                                        "data": [10.0, 20.0, 30.0],
                                        "data_error_upper": [7.0, 7.0, 7.0]}},
            {"name": "ECOILA", "current": {"time": [1.0, 2.0, 3.0],
                                           "data": [-1.0, -2.0, -3.0],
                                           "data_error_upper": [69.0, 69.0, 69.0]}},
        ]}}
        p = tmp_path / "dd.json"
        p.write_text(json.dumps(dd))
        return str(p)

    def test_reads_the_nearest_sample(self, tmp_path):
        from bouquet.coil_targets import measured_from_pf_active
        assert measured_from_pf_active(self._dd(tmp_path), 2.01)["F1A"] == 20.0

    def test_warns_when_the_request_is_off_the_time_base(self, tmp_path):
        from bouquet.coil_targets import measured_from_pf_active
        with pytest.warns(UserWarning, match="outside the coil time base"):
            out = measured_from_pf_active(self._dd(tmp_path), 9.0)
        assert out["F1A"] == 30.0                       # endpoint, but said so

    def test_sigma_is_turns_converted_into_solver_units(self, tmp_path):
        from bouquet.coil_targets import sigma_from_pf_active
        sig = sigma_from_pf_active(self._dd(tmp_path), 2.0, device="DIII-D")
        assert sig["F1A"] == pytest.approx(7.0 * 58.0)
        assert sig["ECOILA"] == pytest.approx(69.0)     # mesh carries E turns


class TestApplyCoilReg:
    """_apply_coil_reg must honour the config AND survive the solver reset."""

    class _GS:
        def __init__(self, sets): self.coil_sets = list(sets); self.installed = None
        def coil_reg_term(self, coils, target=0.0, weight=1.0):
            return {"coils": dict(coils), "target": target, "weight": weight}
        def set_coil_reg(self, reg_terms=None): self.installed = list(reg_terms)

    def _run_gs(self, spec, sets=("F1A", "F6A")):
        from bouquet.run import Bouquet
        gs = self._GS(sets)
        obj = Bouquet.__new__(Bouquet)
        obj.config = type("C", (), {"solver": type("S", (), {"coil_reg": spec})()})()
        Bouquet._apply_coil_reg(obj, gs)
        return gs

    def _run(self, spec, sets=("F1A", "F6A")):
        return self._run_gs(spec, sets).installed

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

    def test_a_dropped_multi_coil_term_names_every_coil_it_releases(self):
        """A term is dropped WHOLE, so a difference constraint also releases its
        on-mesh coil back to target=0 -- the warning must say so, not name only
        the off-mesh coil."""
        spec = [{"coils": {"F1A": 1.0, "E567UP": -1.0}, "target": 0.0, "weight": 5.0}]
        with pytest.warns(UserWarning) as rec:
            got = self._run(spec)
        msg = str(rec[0].message)
        assert "F1A" in msg and "E567UP" in msg and "target=0" in msg
        by = {list(t["coils"])[0]: t for t in got}
        assert by["F1A"]["target"] == 0.0 and by["F1A"]["weight"] == 1.0

    def test_unnamed_coils_still_get_a_zero_target(self):
        got = self._run([{"coils": {"F1A": 1.0}, "target": 1234.0, "weight": 9.0}])
        by = {list(t["coils"])[0]: t for t in got}
        assert by["F6A"]["target"] == 0.0
        assert "#VSC" in by


class TestTurnsConventionIsCheckedAgainstTheMesh:
    """The turns factor is keyed by DEVICE; which side carries the turns is a
    property of the MESH. Exactly one of them must, so the claim is checkable:
    factor != 1 needs net_turns == 1, factor == 1 needs net_turns != 1."""

    class _GS:
        def __init__(self, sets):
            # TokaMaker's real shape: {name: {"id", "net_turns", "sub_coils"}}
            self.coil_sets = sets
            self.installed = None

        def coil_reg_term(self, coils, target=0.0, weight=1.0):
            return {"coils": dict(coils), "target": target, "weight": weight}

        def set_coil_reg(self, reg_terms=None): self.installed = list(reg_terms)

    #: the shipped D3D mesh: F-coil sets carry one turn, the E-coils carry 61
    SHIPPED = {"F1A": {"net_turns": 1.0}, "ECOILA": {"net_turns": 61.0}}

    def _run(self, spec, sets=None):
        from bouquet.run import Bouquet
        gs = self._GS(dict(self.SHIPPED if sets is None else sets))
        obj = Bouquet.__new__(Bouquet)
        obj.config = type("C", (), {"solver": type("S", (), {"coil_reg": spec})()})()
        Bouquet._apply_coil_reg(obj, gs)
        return {list(t["coils"])[0]: t for t in gs.installed}

    def test_the_shipped_convention_is_confirmed_and_nothing_is_dropped(self):
        import warnings
        spec = coil_reg_from_measured({"F1A": 1000.0, "ECOILA": 1000.0}, **D3D)
        assert [t["turns"] for t in spec] == [58.0, 1.0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            by = self._run(spec)
        assert by["F1A"]["target"] == pytest.approx(58000.0)
        assert by["ECOILA"]["target"] == pytest.approx(1000.0)

    def test_an_unconfirmed_implicit_factor_is_dropped_not_applied(self):
        """A second mesh of the same machine that splits the E-coil and does NOT
        carry its turns would take the same implicit 1.0 and be pinned to a
        target wrong by the whole turn count -- at the configured weight."""
        spec = coil_reg_from_measured({"F1A": 1000.0, "E567UP": 1000.0}, **D3D)
        sets = {"F1A": {"net_turns": 1.0}, "E567UP": {"net_turns": 1.0}}
        with pytest.warns(UserWarning) as rec:
            by = self._run(spec, sets=sets)
        msg = "\n".join(str(r.message) for r in rec)
        assert "E567UP" in msg and "net_turns = 1" in msg and "target=0, weight=1" in msg
        assert by["E567UP"]["target"] == 0.0 and by["E567UP"]["weight"] == 1.0
        assert by["F1A"]["target"] == pytest.approx(58000.0)     # unaffected

    def test_turns_counted_twice_is_refused_as_well(self):
        spec = coil_reg_from_measured({"F1A": 1000.0}, **D3D)
        with pytest.warns(UserWarning, match="net_turns = 58"):
            by = self._run(spec, sets={"F1A": {"net_turns": 58.0}})
        assert by["F1A"]["target"] == 0.0

    def test_a_mesh_that_cannot_report_net_turns_is_unconfirmed(self):
        spec = coil_reg_from_measured({"F1A": 1000.0}, **D3D)
        with pytest.warns(UserWarning, match="does not report net_turns"):
            by = self._run(spec, sets={"F1A": {}})
        assert by["F1A"]["target"] == 0.0

    def test_a_hand_built_term_claims_nothing_and_is_not_checked(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            by = self._run([{"coils": {"F1A": 1.0}, "target": 7.0, "weight": 9.0}])
        assert by["F1A"]["target"] == 7.0 and by["F1A"]["weight"] == 9.0

    def test_the_shipped_mesh_really_carries_this_convention(self):
        """The check is only safe if the shipped mesh matches it -- read it."""
        import json
        import os
        import h5py
        mesh = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "examples", "D3D-like", "DIIID_mesh.h5")
        if not os.path.exists(mesh):
            pytest.skip("example mesh not present")
        with h5py.File(mesh, "r") as hf:
            coil_dict = json.loads(hf["mesh/coil_dict"][()])
        net = {}
        for key, v in coil_dict.items():
            net[v.get("coil_set", key)] = net.get(v.get("coil_set", key), 0.0) \
                + float(v.get("nturns", 1.0))
        assert net["F1A"] == 1.0 and net["ECOILA"] == 61.0 and net["ECOILB"] == 61.0
        for name, factor in TURNFC_D3D.items():
            assert factor != 1.0 and net[name] == 1.0


class TestSolverConfigValidation:
    """A malformed coil_reg entry used to die inside setup_solver on a bare
    KeyError/TypeError, after the mesh had been loaded."""

    def test_a_non_dict_term_is_named(self):
        from bouquet.config import SolverConfig
        with pytest.raises(TypeError, match=r"coil_reg\[1\] must be a dict"):
            SolverConfig(mesh_path="m.h5", coil_reg=[{"coils": {"F1A": 1.0}}, 3])

    def test_a_term_without_coils_is_named(self):
        from bouquet.config import SolverConfig
        with pytest.raises(ValueError, match=r"coil_reg\[0\] has no 'coils' key"):
            SolverConfig(mesh_path="m.h5", coil_reg=[{"target": 1.0}])

    def test_coils_must_be_a_non_empty_mapping(self):
        from bouquet.config import SolverConfig
        for bad in (["F1A"], {}):
            with pytest.raises(TypeError, match="non-empty"):
                SolverConfig(mesh_path="m.h5", coil_reg=[{"coils": bad}])

    def test_target_and_weight_must_be_numbers(self):
        from bouquet.config import SolverConfig
        with pytest.raises(TypeError, match="'target'"):
            SolverConfig(mesh_path="m.h5",
                         coil_reg=[{"coils": {"F1A": 1.0}, "target": "big"}])
        with pytest.raises(TypeError, match="'weight'"):
            SolverConfig(mesh_path="m.h5",
                         coil_reg=[{"coils": {"F1A": 1.0}, "weight": None}])

    def test_coil_init_must_be_a_mapping(self):
        from bouquet.config import SolverConfig
        with pytest.raises(TypeError, match="coil_init must be a"):
            SolverConfig(mesh_path="m.h5", coil_init=["F1A", 1.0])
        SolverConfig(mesh_path="m.h5", coil_init={"F1A": 1.0})      # fine

    def test_a_valid_spec_and_the_default_pass(self):
        from bouquet.config import SolverConfig
        SolverConfig(mesh_path="m.h5")
        SolverConfig(mesh_path="m.h5",
                     coil_reg=coil_reg_from_measured({"F1A": 1000.0}, **D3D))


class TestWeakExploratoryReg:
    """The weak reg the draw path explores under keeps the measured targets.

    "Weak" has to mean "the same place, held loosely": aimed at zero it pulls
    the exploratory solve along the coil null space the targets exist to remove.
    Only the WEIGHT is historical (1.0) -- deriving the targets costs ~0.2 sigma
    on the converged currents, raising the weight costs three times that.
    """

    _GS = TestApplyCoilReg._GS
    _run_gs = TestApplyCoilReg._run_gs

    SPEC = [{"coils": {"F1A": 1.0}, "target": 1234.0, "weight": 97.0}]

    def test_weak_targets_are_the_coil_reg_targets_at_weight_one(self):
        gs = self._run_gs(self.SPEC)
        weak = {list(t["coils"])[0]: t for t in gs._weak_coil_reg}
        strong = {list(t["coils"])[0]: t for t in gs.installed}
        assert weak["F1A"]["target"] == strong["F1A"]["target"] == 1234.0
        assert strong["F1A"]["weight"] == 97.0        # configured weight: unchanged
        assert weak["F1A"]["weight"] == 1.0           # exploratory magnitude: unchanged
        assert weak["F6A"]["target"] == 0.0 and weak["F6A"]["weight"] == 1.0
        assert weak["#VSC"]["target"] == 0.0 and weak["#VSC"]["weight"] == 1e-2

    def test_nothing_is_published_without_coil_reg(self):
        """No targets -> the draw path builds its own toward-zero weak reg, so
        the no-coil_reg case stays bit-identical."""
        gs = self._run_gs([])
        assert not hasattr(gs, "_weak_coil_reg")

    def test_a_stale_stash_does_not_survive_a_reset_without_targets(self):
        from bouquet.run import Bouquet
        gs = self._run_gs(self.SPEC)
        assert hasattr(gs, "_weak_coil_reg")
        obj = Bouquet.__new__(Bouquet)
        obj.config = type("C", (), {"solver": type("S", (), {"coil_reg": []})()})()
        Bouquet._apply_coil_reg(obj, gs)
        assert not hasattr(gs, "_weak_coil_reg")

    def test_weak_reg_also_skips_coils_the_mesh_does_not_model(self):
        spec = [{"coils": {"F1A": 1.0}, "target": 5.0, "weight": 1.0},
                {"coils": {"E567UP": 1.0}, "target": 7.0, "weight": 1.0}]
        with pytest.warns(UserWarning, match="E567UP"):
            gs = self._run_gs(spec)
        assert "E567UP" not in {c for t in gs._weak_coil_reg for c in t["coils"]}
        weak = {list(t["coils"])[0]: t for t in gs._weak_coil_reg}
        assert weak["F1A"]["target"] == 5.0

    def test_a_configured_vsc_term_does_not_clamp_the_exploration(self):
        """A #VSC constraint belongs to the constrained phase; at weight 1.0 it
        would clamp the vertical-stability channel the exploration needs."""
        gs = self._run_gs([{"coils": {"#VSC": 1.0}, "target": 42.0, "weight": 50.0}])
        strong = {list(t["coils"])[0]: t for t in gs.installed}
        weak = {list(t["coils"])[0]: t for t in gs._weak_coil_reg}
        assert strong["#VSC"] == {"coils": {"#VSC": 1.0}, "target": 42.0, "weight": 50.0}
        assert weak["#VSC"] == {"coils": {"#VSC": 1.0}, "target": 0.0, "weight": 1e-2}
        assert sum("#VSC" in t["coils"] for t in gs._weak_coil_reg) == 1


class TestCoilInit:
    """SolverConfig.coil_init seeds the inverse iterate; it must not constrain.

    These call the PRODUCTION path (Bouquet._seed_coil_init, which is what
    _forward_solve_imas_baseline invokes right after init_psi), so deleting the
    coil_init block fails them.
    """

    class _GS:
        def __init__(self, sets):
            self.coil_sets = list(sets)
            self._cur = {k: 0.0 for k in sets}
            self.set_calls = []
        def get_coil_currents(self): return dict(self._cur), None
        def set_coil_currents(self, cur):
            self.set_calls.append(dict(cur)); self._cur = dict(cur)

    @staticmethod
    def _obj(coil_init):
        from bouquet.run import Bouquet
        obj = Bouquet.__new__(Bouquet)
        obj.config = type("C", (), {
            "solver": type("S", (), {"coil_init": coil_init})()})()
        return obj

    def test_seeds_only_known_coils_and_keeps_the_rest(self):
        """A 24-circuit measurement against a 20-set mesh: unknown names are
        dropped, known ones are seeded, unlisted known coils keep their value."""
        gs = self._GS(["F1A", "F6A", "ECOILA"])
        gs._cur["ECOILA"] = -5.0
        out = self._obj({"F1A": 1000.0, "E567UP": 99.0})._seed_coil_init(gs)
        assert gs.set_calls[-1] == {"F1A": 1000.0, "F6A": 0.0, "ECOILA": -5.0}
        assert out == gs.set_calls[-1]
        assert "E567UP" not in gs.set_calls[-1]

    def test_unset_touches_nothing(self):
        gs = self._GS(["F1A"])
        assert self._obj(None)._seed_coil_init(gs) is None
        assert self._obj({})._seed_coil_init(gs) is None
        assert gs.set_calls == []

    def test_a_list_is_rejected_with_a_readable_error(self):
        gs = self._GS(["F1A"])
        with pytest.raises(TypeError, match="mapping"):
            self._obj([["F1A", 1.0]])._seed_coil_init(gs)

    def test_the_baseline_solve_calls_it_after_init_psi(self):
        """Ordering is load-bearing: init_psi reinitialises coil currents from
        the regularisation, so a seed installed earlier is overwritten."""
        import inspect
        from bouquet.run import Bouquet
        src = inspect.getsource(Bouquet._forward_solve_imas_baseline)
        assert src.index("init_psi(") < src.index("_seed_coil_init(")

    def test_config_field_defaults_to_none(self):
        from bouquet.config import SolverConfig
        import dataclasses
        f = {x.name: x for x in dataclasses.fields(SolverConfig)}
        assert "coil_init" in f
        assert SolverConfig(mesh_path="x").coil_init is None
