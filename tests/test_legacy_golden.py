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


def test_it_says_what_built_it(doc):
    prov = doc["provenance"]
    assert prov["bouquet"].get("commit"), prov["bouquet"]
    assert prov["oft"].get("available"), prov["oft"]
    assert prov["oft"].get("library_sha256") or \
        prov["oft"].get("sources_sha256"), prov["oft"]
    assert prov["generator_args"].get("jphi_archival") == "input"


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
        for ch in ("j_phi", "j_inductive"):
            assert len(p[ch]) == n_eq, (k, ch)
        assert len(d["coil_currents"]) == len(d["coil_names"])
    ref = doc["baseline"]["recon_lcfs_ref"]
    assert ref["stride"] == mgf.LEGACY_LCFS_STRIDE
    assert len(ref["points"]) == -(-ref["n_full"] // ref["stride"])


def test_it_names_no_filesystem_path(doc):
    sys.path.insert(0, _GOLDEN_DIR)
    import make_golden_fixture as mgf
    assert mgf.find_filesystem_paths(doc) == []
