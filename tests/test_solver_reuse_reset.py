"""A reused solver starts every case clean (#76 B6/B7).

``Bouquet.set_case`` resets every per-case attribute -- whatever
prepare_baseline / generate set on the Bouquet, found from the instance, not
from a hand-kept list -- and the solver's device settings, while the
solver-level state (``SOLVER_ATTRS``) survives.  Runs the real
``point_solver -> _reset_solver_state -> CaseGeometry.apply`` path on the
engine's toy stand-in solver.  Synthetic example data only.
"""
import pytest

from test_engine_draws import _bq, _quiet, toy_bouquet_solver  # noqa: F401


def _source_unavailable_record():
    # what the IMAS forward solve records for swb_seed="source" on a toolkit
    # without jphi_fixed (Bouquet._resolve_swb_seed)
    return dict(requested="source", resolved="source_unavailable",
                oft_jphi_fixed=False, how="explicit")


def test_a_stale_swb_seed_record_does_not_reach_the_next_case(tmp_path, toy_bouquet_solver,
                                                              monkeypatch):
    """An IMAS case whose swb seed resolved to source_unavailable, then a
    g-file case on the same worker: the g-file case must not be refused by
    the previous case's record."""
    import bouquet.run as br
    # the non-loop draw route, where generate() consults the record
    monkeypatch.setattr(br.Bouquet, "_draws_run_jbs_loop", lambda self, m: False)
    b = _bq(tmp_path)
    b.setup_solver()
    _quiet(b.prepare_baseline)
    b._swb_seed_record = _source_unavailable_record()       # left by the IMAS case
    with pytest.raises(RuntimeError, match="swb_seed='source'"):
        _quiet(b.generate)                                   # the guard itself works
    b.set_case(source=b.config.source, header=str(tmp_path / "next"), scan_key=1)
    assert getattr(b, "_swb_seed_record", None) is None
    _quiet(b.prepare_baseline)
    _quiet(b.generate)                                       # no stale refusal


def test_set_case_resets_every_per_case_attribute(tmp_path, toy_bouquet_solver):
    import copy

    import bouquet.run as br
    b = _bq(tmp_path)
    fresh = {k: copy.copy(v) for k, v in vars(b).items()}
    b.setup_solver()
    _quiet(b.prepare_baseline)
    _quiet(b.generate)
    _quiet(b.filter)
    solver_state = {k: getattr(b, k) for k in br.Bouquet.SOLVER_ATTRS if k in vars(b)}
    case_attrs = set(vars(b)) - set(br.Bouquet.SOLVER_ATTRS)
    # what this run set per case, including the integration's records
    assert {"baseline", "_engine_run", "_geom", "_draw_bootstrap_scaling_record",
            "_experimental_features", "diagnostics", "generation_log"} <= case_attrs
    b.set_case(header=str(tmp_path / "next"), scan_key=1)
    # set_case's own record of the device restore (nothing to restore here)
    assert b._device_restored == dict(settings=[], solve_wrapper_removed=False, vsc=False, vcoils=False)
    for k in case_attrs - {"_device_restored"}:
        if k in fresh:
            assert getattr(b, k) == fresh[k], k
        else:
            assert k not in vars(b), k
    for k, v in solver_state.items():
        assert getattr(b, k) is v, k


def test_an_instance_method_override_survives_set_case(tmp_path, toy_bouquet_solver):
    """A hook or stand-in installed on the instance (b.prepare_baseline = f)
    is not per-case state: set_case keeps it."""
    b = _bq(tmp_path)
    b.setup_solver()

    def prep():
        return "stand-in"
    b.prepare_baseline = prep
    b.set_case(header=str(tmp_path / "next"), scan_key=1)
    assert b.prepare_baseline is prep


def test_the_device_settings_are_restored_between_cases(tmp_path, toy_bouquet_solver):
    """A case that dies between cap_solver and uncap_solver (or leaves an
    instance-level solve wrapper) must not hand maxits=100 or the wrapper
    to the next case."""
    b = _bq(tmp_path)
    b.setup_solver()
    gs = b.mygs
    its0 = gs.settings.maxits
    _quiet(b.prepare_baseline)
    # a case interrupted inside a capped draw solve with a timing wrapper
    gs.settings.maxits = 100
    gs.solve = lambda *a, **k: None
    b.set_case(header=str(tmp_path / "next"), scan_key=1)
    _quiet(b.prepare_baseline)
    assert gs.settings.maxits == its0
    assert "solve" not in vars(gs)
    rec = b.baseline.solver_geometry
    assert rec["device_restored"]["settings"] == ["maxits"]
    assert rec["device_restored"]["solve_wrapper_removed"] is True
    # a fresh solver's first case restored nothing
    b2 = _bq(tmp_path / "fresh")
    b2.setup_solver()
    _quiet(b2.prepare_baseline)
    assert b2.baseline.solver_geometry["device_restored"] is None


def test_the_reset_installs_the_case_geometry_after_the_clean_state(tmp_path, toy_bouquet_solver):
    """The real reset order on the stand-in solver: back to the clean
    equilibrium, coil reg, then this case's F0 and isoflux -- before the
    engine's init_psi."""
    b = _bq(tmp_path)
    b.setup_solver()
    gs = b.mygs
    seen = []
    orig_replace, orig_prof, orig_iso, orig_init = (gs.replace_eq, gs.set_profiles,
                                                    gs.set_isoflux, gs.init_psi)
    gs.replace_eq = lambda *a, **k: (seen.append("replace_eq"), orig_replace(*a, **k))[1]
    gs.set_profiles = lambda *a, **k: (seen.append("foffset" if "foffset" in k else "profiles"),
                                       orig_prof(*a, **k))[1]
    gs.set_isoflux = lambda *a, **k: (seen.append("isoflux"), orig_iso(*a, **k))[1]
    gs.init_psi = lambda *a, **k: (seen.append("init_psi"), orig_init(*a, **k))[1]
    _quiet(b.prepare_baseline)
    first = seen.index("replace_eq")
    assert seen[first:first + 3] == ["replace_eq", "foffset", "isoflux"]
    assert seen.index("init_psi") > first + 2
