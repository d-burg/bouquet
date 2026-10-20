"""The slim LEGACY golden (``tests/golden/D3Dlike_Hmode_legacy_golden.json``).

Since 2026-10 the h5 fixture is a run of the unified engine (the default);
the legacy path keeps its numeric regression record in this small JSON, made
by the same recipe (``regenerate_golden_run.py --reconstruction-engine
legacy``, then ``make_golden_fixture.py --legacy-json``).  Its solver-side
consumer is the legacy systematics replay (``tests/test_systematics.py``).
These fast checks say what the file is: a legacy-path, self-consistent-
bootstrap, input-current run with its provenance, the replay draws the
replay needs, and no filesystem path (a public file).
"""
import json
import os
import sys

import pytest

import _harness

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
_JSON = os.path.join(_GOLDEN_DIR, "D3Dlike_Hmode_legacy_golden.json")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_JSON),
    reason="legacy golden not built; run tests/golden/make_golden_fixture.py "
           "--legacy-json")


@pytest.fixture(scope="module")
def doc():
    with open(_JSON) as fh:
        return json.load(fh)


def test_it_is_a_legacy_path_self_consistent_run(doc):
    from bouquet.config import BouquetConfig
    gen = BouquetConfig.from_json(doc["config_json"]).generation
    assert gen.reconstruction_engine == "legacy"
    assert gen.jbs_self_consistent is True
    assert doc["root_attrs"].get("golden_jphi_archival") == "input"
    loop = doc["baseline"]["jbs_loop"]
    assert loop is not None and loop["converged"], loop


def test_it_is_the_gfile_golden_on_the_legacy_path(doc):
    """The g-file golden's recipe (``tests/golden/regenerate_golden_run.py``)
    on ``reconstruction_engine="legacy"``: the synthetic g-file + IDA-layout
    profile fixtures by basename, 20 attempts at seed 12345, one thread."""
    from bouquet.config import BouquetConfig, ReconstructionSource
    sys.path.insert(0, _GOLDEN_DIR)
    import regenerate_golden_run as rgr
    cfg = BouquetConfig.from_json(doc["config_json"])
    assert isinstance(cfg.source, ReconstructionSource)
    assert cfg.source.geqdsk_path == rgr.GFILE
    assert cfg.source.profiles_path == rgr.PROFILES
    assert cfg.solver.mesh_path == rgr.MESH
    gen = cfg.generation
    assert gen.n_equils == rgr.N_DRAWS and gen.seed == rgr.SEED
    assert gen.n_inspec_target is None
    assert cfg.solver.nthreads == rgr.NTHREADS
    # the per-draw wall budget off (owner decision 2026-10-17): a
    # host-independent golden on a route whose draws take up to ~10 min
    assert gen.draw_wall_budget_s == rgr.LEGACY_DRAW_WALL_BUDGET_S is None


def test_it_says_what_built_it(doc):
    prov = doc["provenance"]
    assert prov["bouquet"].get("commit"), prov["bouquet"]
    assert prov["oft"].get("available"), prov["oft"]
    assert prov["oft"].get("library_sha256") or \
        prov["oft"].get("sources_sha256"), prov["oft"]
    assert prov["generator_args"].get("jphi_archival") == "input"
    # the replay (tests/test_systematics.py) keys its comparison on the
    # compiled library's digest; without one the fixture reads "unstamped"
    assert _harness.golden_build_check(doc, installed={}).stamped, prov["oft"]


def test_the_draw_record_is_complete(doc):
    draws = doc["draws"]
    assert len(draws) == doc["n_draws"] > 0
    assert sum(bool(d.get("in_spec")) for d in draws.values()) == \
        doc["n_in_spec"]
    sys.path.insert(0, _GOLDEN_DIR)
    import make_golden_fixture as mgf
    want = sorted((int(k) for k, d in draws.items() if d.get("in_spec")))
    want = want[:mgf.LEGACY_REPLAY_DRAWS]
    assert sorted(int(k) for k in doc["replay_draws"]) == want
    n_kin = len(doc["baseline"]["profiles"]["psi_N_kinetic"])
    n_eq = len(doc["baseline"]["profiles"]["psi_N"])
    for k, d in doc["replay_draws"].items():
        p = d["profiles"]
        for ch in ("n_e", "T_e", "n_i", "T_i"):
            assert len(p[ch]) == n_kin, (k, ch)
        chans = ["jphi_total", "jphi_ind"]
        if d.get("current_split_convention") == "pressure_separate":
            chans.append("jphi_pd")
        for ch in chans:
            assert len(p[ch]) == n_eq, (k, ch)
        assert len(d["coil_currents"]) == len(d["coil_names"])
    ref = doc["baseline"]["recon_lcfs_ref"]
    assert ref["stride"] == mgf.LEGACY_LCFS_STRIDE
    assert len(ref["points"]) == -(-ref["n_full"] // ref["stride"])


def test_it_says_where_the_pressure_driven_current_is(doc):
    """The record states its current-split convention (owner decision D2)
    and, under the separate convention, its baseline's four buckets add up:
    ``jphi_total = jphi_ind + jphi_bs + jphi_pd`` (no driven channels on the
    fixture).  The replay (``tests/test_systematics.py``) composes the legacy
    path's carried inductive from this record, so a record that mis-states
    its split would replay the wrong current."""
    import numpy as np
    from bouquet.schema import (CURRENT_SPLIT_CONVENTIONS,
                                SPLIT_PRESSURE_IN_INDUCTIVE,
                                SPLIT_PRESSURE_SEPARATE)

    def _conv(rec):
        # stated, else inferred as bouquet.schema.read_current_split_convention
        # does for an archive group (a pre-#64 record states nothing)
        v = rec.get("current_split_convention")
        if v is None:
            v = (SPLIT_PRESSURE_SEPARATE if "jphi_pd" in rec["profiles"]
                 else SPLIT_PRESSURE_IN_INDUCTIVE)
        assert v in CURRENT_SPLIT_CONVENTIONS, v
        return v
    bl = doc["baseline"]
    conv = _conv(bl)
    for k, d in doc["replay_draws"].items():
        assert _conv(d) == conv, (k, _conv(d), conv)
    p = bl["profiles"]
    if conv == SPLIT_PRESSURE_SEPARATE:
        jphi, jind, jbs, jp = (np.asarray(p[c], dtype=float) for c in
                               ("jphi_total", "jphi_ind", "jphi_bs", "jphi_pd"))
        resid = np.abs(jphi - jind - jbs - jp).max()
        assert resid <= 1e-9 * np.abs(jphi).max(), resid
        for k, d in doc["replay_draws"].items():
            assert "jphi_pd" in d["profiles"], k
    else:
        assert "jphi_pd" not in p, conv


def test_it_names_no_filesystem_path(doc):
    sys.path.insert(0, _GOLDEN_DIR)
    import make_golden_fixture as mgf
    assert mgf.find_filesystem_paths(doc) == []
