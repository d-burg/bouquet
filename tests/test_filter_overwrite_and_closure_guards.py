"""Three regressions from the 2026-09 examples/docs audit.

1. ``filter_coil_currents(apply=True)`` used to overwrite the chi2 coil
   verdict that ``Bouquet.filter()`` had just written, silently -- every
   notebook that followed the documented sequence ended up legacy-selected
   once chi2 became the default.  It now warns and relabels the scan-level
   ``coil_filter`` attr, and ``apply=False`` leaves everything alone.
2. A non-default ``closure_channel`` outside the IMAS hybrid split
   (``jBS_baseline_mode="ohmic"`` + ``recalculate_j_BS=True``) was resolved,
   announced, and then never read.  The workflow guard now refuses it.
3. ``Baseline.li_metrics`` (with the ``ip_closure`` health record and its
   ``closure_limited`` verdict) is archived on ``_baseline`` and decoded by
   ``load_baseline_profiles``; before, it existed only in memory.

All fast (no solver).
"""
import json
import warnings

import h5py
import numpy as np
import pytest

from bouquet.filtering import (filter_coil_currents, read_filter_flags,
                               _write_filter_result)
from bouquet.utils import store_baseline_profiles, load_baseline_profiles


# ---------------------------------------------------------------------------
# 1. the overwrite guard
# ---------------------------------------------------------------------------

def _archive_with_chi2_verdict(path, scan_key=0):
    """Three draws; chi2 said (pass, pass, fail); the legacy band would say
    (pass, fail, fail) -- draw 1 is the disagreement."""
    drifts = [(0.5, 0.5), (3.0, 0.5), (3.0, 3.0)]   # (F %, VSC %)
    chi2 = {0: True, 1: True, 2: False}
    with h5py.File(path, "w") as hf:
        hf.create_group(f"scan/{scan_key}/_baseline")
        for i, (F, V) in enumerate(drifts):
            d = hf.create_group(f"scan/{scan_key}/{i}")
            d.attrs["max_F_drift_pct"] = float(F)
            d.attrs["max_VSC_drift_pct"] = float(V)
            d.attrs["inspec_F_max"] = 0.02
            d.attrs["inspec_VSC_max"] = 0.02
            d.attrs["passes_boundary_filter"] = True
        hf[f"scan/{scan_key}"].attrs["coil_filter"] = "chi2"
    _write_filter_result(path, scan_key, chi2, "passes_coil_filter")
    return chi2


def test_apply_true_over_a_chi2_verdict_warns_and_relabels(tmp_path):
    h5 = str(tmp_path / "run.h5")
    chi2 = _archive_with_chi2_verdict(h5)
    before = {i: r["passes_coil_filter"]
              for i, r in read_filter_flags(h5, scan_key=0).items()}
    assert before == chi2

    with pytest.warns(UserWarning, match="overwriting the chi2"):
        summ, _ = filter_coil_currents(h5, scan_key=0, plot=False)   # apply=True default

    after = {i: r["passes_coil_filter"]
             for i, r in read_filter_flags(h5, scan_key=0).items()}
    assert after == {0: True, 1: False, 2: False}      # the legacy band, as asked
    assert summ["n_pass"] == 1
    with h5py.File(h5, "r") as hf:
        assert hf["scan/0"].attrs["coil_filter"] == "legacy"   # archive says so


def test_apply_false_reads_only(tmp_path):
    h5 = str(tmp_path / "run.h5")
    chi2 = _archive_with_chi2_verdict(h5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no warning allowed
        summ, _ = filter_coil_currents(h5, scan_key=0, apply=False, plot=False)
    assert summ["n_pass"] == 1                  # the what-if count is reported ...
    after = {i: r["passes_coil_filter"]
             for i, r in read_filter_flags(h5, scan_key=0).items()}
    assert after == chi2                        # ... but nothing was written
    with h5py.File(h5, "r") as hf:
        assert hf["scan/0"].attrs["coil_filter"] == "chi2"


def test_apply_true_without_a_prior_chi2_verdict_is_silent(tmp_path):
    """The legacy filter on a legacy archive is not an overwrite."""
    h5 = str(tmp_path / "run.h5")
    with h5py.File(h5, "w") as hf:
        hf.create_group("scan/0/_baseline")
        d = hf.create_group("scan/0/0")
        d.attrs["max_F_drift_pct"] = 0.5
        d.attrs["max_VSC_drift_pct"] = 0.5
        d.attrs["inspec_F_max"] = 0.02
        d.attrs["inspec_VSC_max"] = 0.02
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filter_coil_currents(h5, scan_key=0, plot=False)
    with h5py.File(h5, "r") as hf:
        assert hf["scan/0"].attrs["coil_filter"] == "legacy"


# ---------------------------------------------------------------------------
# 2. closure_channel is refused where it would be ignored
# ---------------------------------------------------------------------------

def _cfg(source_kind):
    from bouquet.config import (BouquetConfig, ImasSource,
                                ReconstructionSource, SolverConfig)
    if source_kind == "imas":
        src = ImasSource(ids_path="unused.json")
    else:
        src = ReconstructionSource(geqdsk_path="unused.geqdsk",
                                   profiles_path="unused.cdf")
    cfg = BouquetConfig(source=src, solver=SolverConfig(mesh_path="unused.h5"),
                        output_header="t")
    if source_kind == "imas":
        cfg.generation.perturb_jind_in_anchor = True   # the IMAS workflow lock
    return cfg


@pytest.mark.parametrize("channel", ["structured", "sawtooth_bootstrap"])
def test_channel_on_diff_mode_is_refused(channel):
    from bouquet.run import Bouquet
    cfg = _cfg("imas")
    assert cfg.generation.jBS_baseline_mode == "diff"
    cfg.generation.closure_channel = channel
    with pytest.raises(ValueError, match="silently ignored"):
        Bouquet(cfg)._validate_workflow()


def test_channel_on_a_gfile_source_is_refused():
    from bouquet.run import Bouquet
    cfg = _cfg("geqdsk")
    cfg.generation.closure_channel = "structured"
    with pytest.raises(ValueError, match="g-file source"):
        Bouquet(cfg)._validate_workflow()


def test_channel_without_recalculate_jbs_is_refused():
    from bouquet.run import Bouquet
    cfg = _cfg("imas")
    cfg.generation.jBS_baseline_mode = "ohmic"
    cfg.generation.recalculate_j_BS = False
    cfg.generation.closure_channel = "structured"
    with pytest.raises(ValueError, match="recalculate_j_BS=True"):
        Bouquet(cfg)._validate_workflow()


def test_default_channel_on_diff_mode_still_passes_the_channel_check():
    """The refusal is about a NON-default channel; 'bootstrap' on 'diff' is
    the ordinary IMAS workflow and must not trip it."""
    from bouquet.run import Bouquet
    cfg = _cfg("imas")
    try:
        Bouquet(cfg)._validate_workflow()
    except ValueError as e:                     # any other guard problem
        assert "closure_channel" not in str(e)


def test_custom_workflow_downgrades_the_refusal_to_a_warning(capsys):
    from bouquet.run import Bouquet
    cfg = _cfg("imas")
    cfg.generation.closure_channel = "structured"
    cfg.generation.workflow = "custom"
    Bouquet(cfg)._validate_workflow()          # must not raise
    assert "silently ignored" in capsys.readouterr().out


def test_baseline_time_refusal_honours_the_custom_downgrade():
    """_validate_workflow downgrades to a warning under workflow='custom';
    the baseline-solve copy of the same check must not then raise (Copilot
    review on #61).  Solve-free: the check sits before any solver call, so
    it is exercised on the source with the same predicate."""
    import inspect
    from bouquet.run import Bouquet
    src = inspect.getsource(Bouquet._forward_solve_imas_baseline)
    blk = src.split("Same refusal as _validate_workflow", 1)[1]
    blk = blk.split("if self.config.generation.recalculate_j_BS:", 1)[0]
    assert '== "custom"' in blk and "allow_unsafe_workflow" in blk
    assert blk.index('print("WARN: "') < blk.index("raise ValueError(_msg0)")


def test_baseline_meta_is_the_last_generate_bouquet_parameter():
    """generate_bouquet's trailing options are positional-capable in existing
    callers; a new optional parameter must be appended, never inserted."""
    import inspect
    from bouquet.TokaMaker_interface import generate_bouquet
    params = list(inspect.signature(generate_bouquet).parameters)
    assert params[-1] == "baseline_meta"


# ---------------------------------------------------------------------------
# 3. ip_closure travels with the archive
# ---------------------------------------------------------------------------

def _store(tmp_path, meta):
    header = str(tmp_path / "arch")
    psi = np.linspace(0.0, 1.0, 11)
    one = np.ones_like(psi)
    store_baseline_profiles(
        header, psi, one, one, one, one, one, one,
        one, one, one, one, one,
        Ip_target=1.0e6, l_i_target=0.9, scan_key=0,
        baseline_meta=meta)
    return header


def test_li_metrics_and_ip_closure_round_trip(tmp_path):
    meta = {
        "tokamaker_li_1": 0.91, "jBS_baseline_mode": "ohmic",
        "bs_scale": np.float64(0.83),                 # numpy scalar
        "ip_closure": {
            "closure_channel": "structured",
            "closure_limited": True,
            "closure_limited_reasons": ("bs_scale 0.412 < 0.5",),   # tuple
            "s_bs_minmax": np.array([0.41, 1.12]),                  # ndarray
        },
        "closure_limited": True,
    }
    header = _store(tmp_path, meta)
    bl = load_baseline_profiles(header + ".h5", scan_key=0)
    assert bl["li_metrics"]["tokamaker_li_1"] == pytest.approx(0.91)
    assert bl["li_metrics"]["bs_scale"] == pytest.approx(0.83)
    assert bl["ip_closure"]["closure_channel"] == "structured"
    assert bl["ip_closure"]["closure_limited_reasons"] == ["bs_scale 0.412 < 0.5"]
    assert bl["ip_closure"]["s_bs_minmax"] == pytest.approx([0.41, 1.12])
    assert bl["closure_limited"] is True
    # the raw attr is plain JSON, readable without bouquet
    with h5py.File(header + ".h5", "r") as hf:
        raw = hf["scan/0/_baseline"].attrs["li_metrics_json"]
        assert json.loads(raw)["ip_closure"]["closure_limited"] is True
        assert bool(hf["scan/0/_baseline"].attrs["closure_limited"]) is True


def test_no_meta_means_no_attr_and_no_keys(tmp_path):
    header = _store(tmp_path, None)
    bl = load_baseline_profiles(header + ".h5", scan_key=0)
    assert "li_metrics" not in bl and "ip_closure" not in bl
    assert "closure_limited" not in bl


def test_unserialisable_values_are_stringified_not_dropped(tmp_path):
    class Odd:
        def __repr__(self):
            return "<odd>"
    header = _store(tmp_path, {"weird": Odd(), "ok": 1})
    bl = load_baseline_profiles(header + ".h5", scan_key=0)
    assert bl["li_metrics"] == {"weird": "<odd>", "ok": 1}
