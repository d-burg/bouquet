"""Config validation of the self-consistent bootstrap loop's settings.

* ``bootstrap_kwargs`` reaches SWB only: under the loop a non-empty dict
  raises a DeprecationWarning that says where it acts; a stored
  ``swb_iterations`` (retired) loads as ``bootstrap_kwargs["iterations"]``.
* ``True``/``False`` is not a tolerance or a relaxation factor; ceilings are
  integers; relaxation factors lie in (0, 1].
* ``from_dict``/``from_json`` refuse an unknown (misspelt) ``generation``
  key, naming it and the nearest valid key; fields that were removed from
  the code are dropped with a warning.
* A config dict without ``jbs_self_consistent`` still loads the LEGACY path
  (old files reproduce old results), and the warning says how to opt in.

No solver; synthetic configs only.
"""
import json
import warnings

import numpy as np
import pytest

from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                            SolverConfig)
from bouquet.jbs_loop import validate_jbs_settings
from test_jbs_loop import _GC


def _cfg(**gen):
    # the legacy paths' validation (swb_iterations is a legacy-path setting
    # the unified engine refuses outright)
    gen.setdefault("reconstruction_engine", "legacy")
    return BouquetConfig(source=ImasSource(ids_path="x.json"),
                         solver=SolverConfig(mesh_path="m.h5"),
                         output_header="t",
                         generation=GenerationConfig(**gen))


# ---------------------------------------------------------------------------
#  bootstrap_kwargs under the loop; the retired swb_iterations
# ---------------------------------------------------------------------------
def test_bootstrap_kwargs_under_the_loop_warns_where_it_acts():
    with pytest.warns(DeprecationWarning) as rec:
        _cfg(bootstrap_kwargs={"iterations": 2})
    msg = " ".join(str(w.message) for w in rec
                   if issubclass(w.category, DeprecationWarning))
    assert "bootstrap_kwargs" in msg and "jbs_init='swb'" in msg


@pytest.mark.parametrize("gen", [dict(), dict(bootstrap_kwargs={}),
                                 dict(jbs_self_consistent=False,
                                      bootstrap_kwargs={"iterations": 2})])
def test_bootstrap_kwargs_is_silent_where_it_is_honoured_or_empty(gen):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _cfg(**gen)


def test_a_stored_swb_iterations_loads_as_bootstrap_kwargs():
    d = _cfg(jbs_self_consistent=False).to_dict()
    d["generation"]["swb_iterations"] = 2
    with pytest.warns(UserWarning, match="iterations"):
        g = BouquetConfig.from_dict(d).generation
    assert g.bootstrap_kwargs == {"iterations": 2}
    assert not hasattr(g, "swb_iterations")


def test_a_stored_default_swb_iterations_is_dropped_silently():
    d = _cfg().to_dict()
    d["generation"]["swb_iterations"] = 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = BouquetConfig.from_dict(d).generation
    assert g.bootstrap_kwargs == {}


# ---------------------------------------------------------------------------
#  value validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field, bad", [
    ("jbs_rtol_j", True), ("jbs_rtol_Ip", False), ("jbs_tol_li", True),
    ("jbs_tol_q0", np.bool_(True)), ("jbs_relax", True),
    ("jbs_relax_current", True), ("jbs_relax_current", np.bool_(True)),
    ("jbs_max_passes", 8.0), ("jbs_max_passes", "8"),
    ("jbs_max_passes_draw", True), ("jbs_max_passes_draw", np.float64(12)),
    ("jbs_max_passes_post_homotopy", np.bool_(True)),
    ("jbs_relax", 0.0), ("jbs_relax", -0.5), ("jbs_relax", 1.0001),
    ("jbs_relax_current", -0.1), ("jbs_relax_current", float("inf")),
    ("jbs_relax_halve_on", np.bool_(True)),
])
def test_bools_and_non_integers_are_refused_by_name(field, bad):
    with pytest.raises(ValueError, match=field):
        validate_jbs_settings(_GC(**{field: bad}))


@pytest.mark.parametrize("field, good", [
    ("jbs_max_passes", np.int64(8)), ("jbs_relax", 1), ("jbs_relax", 0.25),
    ("jbs_relax_current", 1.0), ("jbs_rtol_j", 1), ("jbs_tol_li", 5e-4),
])
def test_valid_values_still_pass(field, good):
    validate_jbs_settings(_GC(**{field: good}))


# ---------------------------------------------------------------------------
#  from_dict / from_json: unknown and retired generation keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("typo, nearest", [
    ("jbs_self_consistant", "jbs_self_consistent"),
    ("jbs_max_pases", "jbs_max_passes"),
    ("jbs_rtol_ip", "jbs_rtol_Ip"),
    ("bootstrap_kwarg", "bootstrap_kwargs"),
])
def test_a_misspelt_generation_key_is_refused_with_the_nearest_key(typo,
                                                                   nearest):
    d = _cfg().to_dict()
    d["generation"][typo] = True
    with pytest.raises(ValueError, match=typo) as ei:
        BouquetConfig.from_dict(d)
    assert f"nearest valid key: '{nearest}'" in str(ei.value)
    with pytest.raises(ValueError, match=typo):
        BouquetConfig.from_json(json.dumps(d))


def test_a_misspelt_loop_switch_no_longer_loads_as_an_old_config():
    """The reported failure: {"jbs_self_consistant": true} without the real
    key used to load the LEGACY path with a warning that the config
    'predates' the loop."""
    d = _cfg().to_dict()
    del d["generation"]["jbs_self_consistent"]
    d["generation"]["jbs_self_consistant"] = True
    with pytest.raises(ValueError, match="jbs_self_consistant"):
        BouquetConfig.from_dict(d)


def test_retired_generation_keys_are_dropped_with_a_warning():
    d = _cfg().to_dict()
    d["generation"]["lock_coils"] = True
    d["generation"]["coil_drift_threshold_A"] = 100.0
    with pytest.warns(UserWarning, match="retired"):
        g = BouquetConfig.from_dict(d).generation
    assert not hasattr(g, "lock_coils")


def test_recorded_init_false_fields_round_trip():
    d = _cfg().to_dict()
    assert "structured_preset_in_force" in d["generation"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BouquetConfig.from_dict(d)


# ---------------------------------------------------------------------------
#  a dict without the loop field: legacy, and the message says how to opt in
# ---------------------------------------------------------------------------
def test_the_legacy_default_warning_says_how_to_opt_in():
    d = _cfg().to_dict()
    for k in list(d["generation"]):
        if k.startswith("jbs_") and k != "jbs_delta_mode":
            del d["generation"][k]
    with pytest.warns(UserWarning) as rec:
        g = BouquetConfig.from_dict(d).generation
    assert g.jbs_self_consistent is False
    msg = " ".join(str(w.message) for w in rec)
    assert "LEGACY" in msg and "reproduces its old results" in msg
    assert '"jbs_self_consistent": true' in msg
    assert "cfg.generation.jbs_self_consistent = True" in msg
