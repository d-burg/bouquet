"""Fast-pressure storage convention on the IMAS path.

``pressure_fast_parallel`` / ``pressure_fast_perpendicular`` carry two
incompatible meanings whose scalar ``p_fast`` differs by a FACTOR OF 3:

  * IMAS.jl / FUSE write the pressure PER DEGREE OF FREEDOM (``pressa/3`` in
    each field)             -> scalar = p_par + 2*p_perp   ("sum")
  * the IMAS data dictionary (and OMAS-written dds) define
    ``pressure_fast_parallel`` as the FULL parallel pressure
                            -> scalar = (p_par + 2*p_perp)/3 ("trace")

These tests pin the provenance-driven choice (``p_fast_reduction="auto"``), the
explicit override, the missing-parallel-field fallback under both rules, and an
end-to-end ``read_imas_baseline`` on a synthetic in-repo fixture.  No
proprietary data: every dd here is built in-process from small arrays.
"""

import json
import warnings

import numpy as np
import pytest

from bouquet.config import FixedComponentsConfig, ImasSource
from bouquet.io.imas import (detect_p_fast_convention, read_imas_baseline,
                             resolve_p_fast_reduction,
                             P_FAST_UNDETERMINED_FALLBACK)


# ---------------------------------------------------------------------------
# synthetic dds
# ---------------------------------------------------------------------------
def _minimal_dd(n=9, p_fast_perp=None, p_fast_par=None, with_parallel=True):
    """Smallest dd ``read_imas_baseline`` accepts, with a D + C ion mix.

    The main ion carries the fast pressure; ``equilibrium.pressure`` is built
    from the same thermal profiles so the completeness backstop has an anchor.
    """
    psi = np.linspace(0.0, 1.0, n)                 # poloidal flux (arbitrary)
    psi_N = psi.copy()
    ne = 4.0e19 * (1.0 - 0.6 * psi_N ** 2)
    te = 2.0e3 * (1.0 - 0.8 * psi_N ** 2) + 50.0
    ti = te.copy()
    nC = 0.02 * ne
    ni = ne - 6.0 * nC
    j_tor = 6.0e5 * (1.0 - psi_N ** 2)
    j_total = j_tor.copy()
    j_boot = 0.1 * j_tor
    j_ohmic = j_tor - j_boot

    if p_fast_perp is None:
        p_fast_perp = 3.0e3 * (1.0 - psi_N ** 2)
    p_fast_perp = np.asarray(p_fast_perp, dtype=float)
    if p_fast_par is None:
        p_fast_par = p_fast_perp.copy()

    EC = 1.602176634e-19
    # thermal + impurity + the "trace" reading of the fast fields
    p_eq = EC * (ne * te + ni * ti + nC * ti) + p_fast_perp

    def _sp(dens, temp, perp=None, par=None, label=None, z=None):
        d = {"density_thermal": dens.tolist(), "temperature": temp.tolist()}
        if label is not None:
            d["label"] = label
        if z is not None:
            d["element"] = [{"z_n": float(z), "a": 2.0 * float(z)}]
        if perp is not None:
            d["pressure_fast_perpendicular"] = np.asarray(perp).tolist()
            if with_parallel and par is not None:
                d["pressure_fast_parallel"] = np.asarray(par).tolist()
        return d

    return {
        "equilibrium": {
            "time": [1.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-2.0]},
            "time_slice": [{
                "time": 1.0,
                "global_quantities": {"ip": 1.0e6, "li_1": 1.0, "li_3": 0.9},
                "boundary": {"outline": {"r": [1.2, 2.2, 1.7],
                                         "z": [0.0, 0.0, 0.8]}},
                "profiles_1d": {"psi": psi.tolist(), "pressure": p_eq.tolist(),
                                "j_tor": j_tor.tolist()},
            }],
        },
        "core_profiles": {
            "time": [1.0],
            "profiles_1d": [{
                "grid": {"psi": psi.tolist()},
                "j_tor": j_tor.tolist(), "j_total": j_total.tolist(),
                "j_ohmic": j_ohmic.tolist(), "j_bootstrap": j_boot.tolist(),
                "electrons": _sp(ne, te),
                "ion": [
                    _sp(ni, ti, perp=p_fast_perp, par=p_fast_par, label="D", z=1.0),
                    _sp(nC, ti, label="C12", z=6.0),
                ],
            }],
        },
    }


def _stamp(dd, ids, group, key, text):
    dd.setdefault(ids, {}).setdefault(group, {})[key] = text
    return dd


def _write(tmp_path, dd, name="dd.json"):
    p = tmp_path / name
    p.write_text(json.dumps(dd))
    return str(p)


# ---------------------------------------------------------------------------
# provenance detection
# ---------------------------------------------------------------------------
class TestDetectConvention:
    def test_imasjl_structure_selects_sum(self):
        """FUSE/IMASdd.jl write top-level keys that are not IMAS DD IDSes."""
        for marker in ("global_time", "requirements", "build",
                       "balance_of_plant", "solid_mechanics", "costing"):
            dd = _minimal_dd()
            dd[marker] = 1.0 if marker == "global_time" else {}
            det = detect_p_fast_convention(dd)
            assert det["rule"] == "sum", marker
            assert det["basis"] == "imas.jl-structure"
            assert marker in det["evidence"]

    def test_fuse_producer_string_selects_sum(self):
        dd = _stamp(_minimal_dd(), "core_profiles", "code", "name", "FUSE")
        det = detect_p_fast_convention(dd)
        assert det["rule"] == "sum" and det["basis"] == "producer-string"

    def test_omas_producer_string_selects_trace(self):
        dd = _stamp(_minimal_dd(), "core_profiles", "code", "name",
                    "OMAS machine mapping")
        det = detect_p_fast_convention(dd)
        assert det["rule"] == "trace" and det["basis"] == "producer-string"

    @pytest.mark.parametrize("text,rule", [
        ("p_fast_reduction=trace", "trace"),
        ("p_fast_reduction = sum", "sum"),
        ("pressure_fast_convention: per_dof", "sum"),
        ("fast_pressure_convention: total", "trace"),
    ])
    def test_explicit_stamp_wins(self, text, rule):
        dd = _stamp(_minimal_dd(), "dataset_description", "ids_properties",
                    "comment", f"hand-built fixture; {text}")
        # an IMAS.jl structural marker is present and is still overruled
        dd["global_time"] = 1.0
        det = detect_p_fast_convention(dd)
        assert det["rule"] == rule and det["basis"] == "explicit-stamp"

    @pytest.mark.parametrize("stamp,value", [
        ("p_fast_reduction=tracee", "tracee"),
        ("p_fast_reduction: auto", "auto"),
        ("pressure_fast_convention = per dof", "per"),   # the space truncates it
    ])
    def test_an_unrecognised_stamp_value_is_not_discarded_silently(self, stamp,
                                                                   value):
        """A dd that states its own convention and is then misread is the one case
        the stamp mechanism was supposed to make authoritative."""
        dd = _stamp(_minimal_dd(), "dataset_description", "ids_properties",
                    "comment", f"hand-built fixture; {stamp}")
        dd["global_time"] = 1.0        # so inference has something to fall back on
        with pytest.warns(UserWarning, match="not recognised") as rec:
            det = detect_p_fast_convention(dd)
        msg = " ".join(str(w.message) for w in rec)
        assert repr(value) in msg                     # the value is named
        assert "dataset_description.ids_properties.comment" in msg
        assert "inferred" in msg
        # ... and the inference still runs, rather than the read failing
        assert det["rule"] == "sum" and det["basis"] == "imas.jl-structure"

    def test_structure_outranks_an_imported_sub_ids_producer(self):
        """A FUSE dd legitimately carries IDSes imported from other codes.

        The file's writer sets the convention, so the structural marker wins
        over an OMFIT/OMAS name on one of the scoped IDSes.
        """
        dd = _stamp(_minimal_dd(), "core_profiles", "code", "name",
                    "OMFIT beams module")
        dd["global_time"] = 1.0
        assert detect_p_fast_convention(dd)["rule"] == "sum"

    def test_unscoped_ids_does_not_vote(self):
        """Only dataset_description/core_profiles/equilibrium/summary are read."""
        dd = _stamp(_minimal_dd(), "nbi", "ids_properties", "comment",
                    "NBI data produced in OMFIT")
        assert detect_p_fast_convention(dd)["rule"] is None

    def test_undeterminable(self):
        det = detect_p_fast_convention(_minimal_dd())
        assert det["rule"] is None and det["basis"] == "undetermined"


# ---------------------------------------------------------------------------
# rule resolution: warning behaviour
# ---------------------------------------------------------------------------
class TestResolveRule:
    def test_undeterminable_warns_once_and_is_actionable(self):
        dd = _minimal_dd()
        with pytest.warns(UserWarning) as rec:
            meta = resolve_p_fast_reduction(dd, "auto")
        assert meta["rule"] == P_FAST_UNDETERMINED_FALLBACK == "sum"
        assert meta["basis"] == "undetermined-fallback" and meta["warned"]
        assert len(rec) == 1
        msg = str(rec[0].message)
        for needed in ("FACTOR OF 3", "'sum'", "'trace'", "p_fast_reduction"):
            assert needed in msg

    @pytest.mark.parametrize("rule", ["sum", "trace", "mean", "perp"])
    def test_explicit_rule_wins_and_is_silent(self, rule):
        dd = _minimal_dd()
        dd["global_time"] = 1.0          # would otherwise select "sum"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            meta = resolve_p_fast_reduction(dd, rule)
        assert meta["rule"] == rule and meta["basis"] == "explicit-argument"
        assert not meta["warned"]

    def test_detected_rule_is_silent(self):
        dd = _minimal_dd()
        dd["global_time"] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            meta = resolve_p_fast_reduction(dd, "auto")
        assert meta["rule"] == "sum" and not meta["warned"]

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown p_fast_reduction"):
            resolve_p_fast_reduction(_minimal_dd(), "rms")


# ---------------------------------------------------------------------------
# the defaults themselves -- these fail if the default is reverted
# ---------------------------------------------------------------------------
class TestDefaultsArePinned:
    def test_config_default_is_auto(self):
        assert FixedComponentsConfig().p_fast_reduction == "auto"

    def test_reader_signature_default_is_auto(self):
        import inspect
        sig = inspect.signature(read_imas_baseline)
        assert sig.parameters["p_fast_reduction"].default == "auto"

    def test_config_accepts_auto_and_rejects_nonsense(self):
        from bouquet.config import BouquetConfig, SolverConfig
        src = ImasSource(ids_path="nonexistent.json", time=1.0)
        cfg = BouquetConfig(source=src, solver=SolverConfig(mesh_path="m.h5"),
                            output_header="hdr")
        assert cfg.fixed_components.p_fast_reduction == "auto"
        with pytest.raises(ValueError, match="p_fast_reduction must be"):
            BouquetConfig(source=src, solver=SolverConfig(mesh_path="m.h5"),
                          output_header="hdr",
                          fixed_components=FixedComponentsConfig(
                              p_fast_reduction="rms"))

    def test_auto_on_a_fuse_shaped_dd_does_not_resolve_to_trace(self):
        """Regression against reverting the default to "trace".

        On an IMAS.jl-written dd the per-dof fields must be summed; a reverted
        default would return a third of the fast pressure.
        """
        perp = np.full(9, 1.0e3)
        dd = _minimal_dd(p_fast_perp=perp, p_fast_par=perp)
        dd["global_time"] = 1.0
        meta = resolve_p_fast_reduction(dd, FixedComponentsConfig().p_fast_reduction)
        assert meta["rule"] == "sum"


# ---------------------------------------------------------------------------
# missing pressure_fast_parallel
# ---------------------------------------------------------------------------
class TestMissingParallelField:
    @staticmethod
    def _perp_only_dd():
        perp = np.array([1.0e4, 5.0e3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return _minimal_dd(p_fast_perp=perp, with_parallel=False), perp

    def test_trace_gives_p_perp_and_says_what_the_closure_means(self, tmp_path):
        dd, perp = self._perp_only_dd()
        path = _write(tmp_path, dd, "perp_only_trace.json")
        with pytest.warns(UserWarning, match="pressure_fast_parallel") as rec:
            bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                                    p_fast_reduction="trace")
        # unchanged from the historical behaviour: the isotropic closure
        assert np.allclose(bl.p_fast, perp)
        msg = " ".join(str(w.message) for w in rec)
        assert "isotropic fast-ion assumption" in msg
        assert "the scalar is p_perp" in msg

    def test_sum_triples_but_never_silently(self, tmp_path):
        dd, perp = self._perp_only_dd()
        path = _write(tmp_path, dd, "perp_only_sum.json")
        with pytest.warns(UserWarning, match="pressure_fast_parallel") as rec:
            bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                                    p_fast_reduction="sum")
        assert np.allclose(bl.p_fast, 3.0 * perp)
        msg = " ".join(str(w.message) for w in rec)
        # the warning has to name the two competing readings, not just complain
        assert "2*p_perp" in msg and "p_perp" in msg and "3*p_perp" in msg

    def test_no_warning_when_the_perp_field_is_all_zero(self, tmp_path):
        dd = _minimal_dd(p_fast_perp=np.zeros(9), with_parallel=False)
        path = _write(tmp_path, dd, "zero_fast.json")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                               p_fast_reduction="sum")
        assert not [w for w in rec if "pressure_fast_parallel" in str(w.message)]


# ---------------------------------------------------------------------------
# the "auto" warning fires only where the convention moved a number
# ---------------------------------------------------------------------------
class TestUndeterminedWarningIsConditional:
    """The factor-of-3 warning is the single guard the ``auto`` design rests on.

    It has to stay credible: a user who sees it on runs where it cannot matter
    filters it, and then it does not fire on the run where it does.  These
    exercise the ``auto`` branch itself (not an explicit rule).
    """

    @staticmethod
    def _no_fast_dd(absent):
        """A provenance-less dd whose fast pressure is zero, or absent entirely."""
        dd = _minimal_dd(p_fast_perp=np.zeros(9))
        if absent:
            for sp in dd["core_profiles"]["profiles_1d"][0]["ion"]:
                sp.pop("pressure_fast_perpendicular", None)
                sp.pop("pressure_fast_parallel", None)
        return dd

    @pytest.mark.parametrize("absent", [False, True], ids=["all-zero", "absent"])
    def test_silent_when_the_dd_carries_no_fast_pressure(self, tmp_path, absent):
        dd = self._no_fast_dd(absent)
        assert detect_p_fast_convention(dd)["rule"] is None   # genuinely unknown
        path = _write(tmp_path, dd, f"no_fast_{int(absent)}.json")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0))
        assert not [w for w in rec if "FACTOR OF 3" in str(w.message)]
        assert np.allclose(bl.p_fast, 0.0)
        # the resolution is still recorded; only the warning is held back
        assert bl.p_fast_meta["basis"] == "undetermined-fallback"
        assert bl.p_fast_meta["rule"] == P_FAST_UNDETERMINED_FALLBACK
        assert not bl.p_fast_meta["warned"]

    def test_silent_when_the_user_supplies_p_fast(self, tmp_path):
        dd = _minimal_dd()              # non-zero fast fields, no provenance
        path = _write(tmp_path, dd, "user_p_fast.json")
        mine = np.linspace(2.0e3, 0.0, 9)
        fixed = FixedComponentsConfig(p_fast=mine, psi_N=np.linspace(0, 1, 9))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                                    fixed=fixed, allow_incomplete_pressure=True)
        assert not [w for w in rec if "FACTOR OF 3" in str(w.message)]
        assert np.allclose(bl.p_fast, mine)
        assert bl.p_fast_meta["basis"] == "user-override"
        assert not bl.p_fast_meta["warned"]

    def test_a_genuinely_ambiguous_dd_still_warns(self, tmp_path):
        dd = _minimal_dd()              # non-zero fast fields, no provenance
        path = _write(tmp_path, dd, "ambiguous.json")
        with pytest.warns(UserWarning, match="FACTOR OF 3") as rec:
            bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                                    allow_incomplete_pressure=True)
        assert len([w for w in rec if "FACTOR OF 3" in str(w.message)]) == 1
        assert bl.p_fast_meta["basis"] == "undetermined-fallback"
        assert bl.p_fast_meta["rule"] == P_FAST_UNDETERMINED_FALLBACK
        assert bl.p_fast_meta["warned"]


# ---------------------------------------------------------------------------
# read_imas_baseline end-to-end
# ---------------------------------------------------------------------------
class TestReadImasBaselineEndToEnd:
    def test_fuse_shaped_dd_sums_the_per_dof_fields(self, tmp_path):
        pressa = np.array([3.0e4, 1.5e4, 6.0e3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        third = pressa / 3.0
        dd = _minimal_dd(p_fast_perp=third, p_fast_par=third)
        dd["global_time"] = 1.0                       # IMAS.jl fingerprint
        path = _write(tmp_path, dd, "fuse_like.json")
        bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                                allow_incomplete_pressure=True)
        assert bl.provenance == "imas"
        assert np.allclose(bl.p_fast, pressa)
        assert bl.p_fast_meta["rule"] == "sum"
        assert bl.p_fast_meta["basis"] == "imas.jl-structure"
        assert bl.p_fast_meta["requested"] == "auto"

    def test_dictionary_convention_dd_takes_the_trace(self, tmp_path):
        full = np.array([3.0e4, 1.5e4, 6.0e3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dd = _minimal_dd(p_fast_perp=full, p_fast_par=full)
        _stamp(dd, "dataset_description", "ids_properties", "comment",
               "written by OMAS")
        path = _write(tmp_path, dd, "omas_like.json")
        bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0))
        assert np.allclose(bl.p_fast, full)
        assert bl.p_fast_meta["rule"] == "trace"
        assert bl.p_fast_meta["basis"] == "producer-string"

    def test_the_two_conventions_differ_by_exactly_three(self, tmp_path):
        full = np.array([3.0e4, 1.5e4, 6.0e3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dd = _minimal_dd(p_fast_perp=full, p_fast_par=full)
        path = _write(tmp_path, dd, "both.json")
        src = ImasSource(ids_path=path, time=1.0)
        a = read_imas_baseline(src, p_fast_reduction="sum",
                               allow_incomplete_pressure=True).p_fast
        b = read_imas_baseline(src, p_fast_reduction="trace").p_fast
        assert np.allclose(a, 3.0 * b)

    def test_user_supplied_p_fast_is_recorded_as_an_override(self, tmp_path):
        dd = _minimal_dd()
        dd["global_time"] = 1.0
        path = _write(tmp_path, dd, "override.json")
        mine = np.linspace(2.0e3, 0.0, 9)
        fixed = FixedComponentsConfig(p_fast=mine, psi_N=np.linspace(0, 1, 9))
        bl = read_imas_baseline(ImasSource(ids_path=path, time=1.0), fixed=fixed,
                                allow_incomplete_pressure=True)
        assert np.allclose(bl.p_fast, mine)
        assert bl.p_fast_meta["basis"] == "user-override"
        assert bl.p_fast_meta["rule"] is None

    def test_shipped_synthetic_example_resolves_without_warning(self):
        """The in-repo D3D-like example is dictionary-convention and says so."""
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "examples", "D3D-like",
                            "D3Dlike_baseline_omas.json")
        if not os.path.exists(path):          # pragma: no cover - source checkout only
            pytest.skip("example dd not present in this checkout")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # The example carries fast pressure but no density_fast, which the
            # reader reports on its own account; that is a statement about the
            # dilution correction, not about the pressure convention under test.
            warnings.filterwarnings(
                "always", message=".*fast-ion PRESSURE but no density_fast.*")
            bl = read_imas_baseline(ImasSource(ids_path=path, time=2.3043))
        assert bl.p_fast_meta["rule"] == "trace"
        assert bl.p_fast_meta["basis"] == "explicit-stamp"


# ---------------------------------------------------------------------------
# the completeness backstop's message
# ---------------------------------------------------------------------------
class TestCompletenessMessage:
    def _dd_with_pressure_gap(self, factor):
        """A dd whose reconstructed total misses equilibrium.pressure."""
        full = 3.0e4 * np.ones(9)
        dd = _minimal_dd(p_fast_perp=full, p_fast_par=full)
        eqp = dd["equilibrium"]["time_slice"][0]["profiles_1d"]
        eqp["pressure"] = (np.asarray(eqp["pressure"]) * factor).tolist()
        return dd

    def test_reports_the_signed_direction(self, tmp_path):
        # equilibrium.pressure halved -> the reconstruction is ABOVE it
        path = _write(tmp_path, self._dd_with_pressure_gap(0.5), "above.json")
        with pytest.warns(UserWarning, match="above") as rec:
            read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                               p_fast_reduction="trace")
        assert "below" not in str(rec[0].message)

        path = _write(tmp_path, self._dd_with_pressure_gap(2.0), "below.json")
        with pytest.warns(UserWarning, match="below"):
            read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                               p_fast_reduction="trace")

    def test_does_not_promise_an_anchor_that_is_off(self, tmp_path):
        path = _write(tmp_path, self._dd_with_pressure_gap(2.0), "noanchor.json")
        with pytest.warns(UserWarning) as rec:
            read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                               p_fast_reduction="trace",
                               anchor_pressure_to_equilibrium=False)
        msg = " ".join(str(w.message) for w in rec)
        assert "NOTHING absorbs this" in msg
        assert "anchor_pressure_to_equilibrium=True" in msg

    def test_claims_the_anchor_only_when_it_is_on(self, tmp_path):
        path = _write(tmp_path, self._dd_with_pressure_gap(2.0), "anchor.json")
        with pytest.warns(UserWarning) as rec:
            read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                               p_fast_reduction="trace",
                               anchor_pressure_to_equilibrium=True)
        msg = " ".join(str(w.message) for w in rec)
        assert "anchor_pressure_to_equilibrium is ON" in msg
        assert "NOTHING absorbs" not in msg

    def test_names_the_reduction_rule_in_force(self, tmp_path):
        path = _write(tmp_path, self._dd_with_pressure_gap(2.0), "rule.json")
        with pytest.warns(UserWarning) as rec:
            read_imas_baseline(ImasSource(ids_path=path, time=1.0),
                               p_fast_reduction="trace")
        msg = " ".join(str(w.message) for w in rec)
        assert "Fast-pressure reduction in force: 'trace'" in msg
