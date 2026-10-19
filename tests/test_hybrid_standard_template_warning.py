"""The silent template fallback on the IDA-hybrid ``ni_source="standard"``
sub-route is no longer silent (1.4.1; report only).

That sub-route reads the IDA uncertainty envelopes only from
``UncertaintyConfig.ida_path`` (``Bouquet.from_imas`` sets it).  A
hand-built config that leaves it unset got the ASSUMED template for every
kinetic channel and the Z_eff width without a word, although the run's own
IDA file was in play.  It now warns; what is resolved is unchanged.
"""
import warnings

import test_ni_fast_subtraction as NF
import test_zeff_hybrid_standard_total as HS
from bouquet.baseline import resolve_uncertainty
from bouquet.config import (BouquetConfig, GenerationConfig, SolverConfig,
                            UncertaintyConfig)

_MSG = "reads the IDA uncertainty envelopes only from"


def _cfg(src, unc, kinetic_source="ida_hybrid"):
    return BouquetConfig(
        source=src, solver=SolverConfig(mesh_path="unused"),
        output_header="unused",
        generation=GenerationConfig(kinetic_source=kinetic_source),
        uncertainty=unc)


def _resolve(cfg, bl):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = resolve_uncertainty(cfg, bl)
    return env, [str(x.message) for x in w if _MSG in str(x.message)]


def test_a_hand_built_config_without_ida_path_warns(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    src = HS._src(ddp, cdf, ni_source="standard")
    bl = HS._read(ddp, cdf, ni_source="standard")
    env, msgs = _resolve(_cfg(src, UncertaintyConfig()), bl)
    assert len(msgs) == 1
    assert "'standard'" in msgs[0] and "ASSUMED template" in msgs[0]
    # the fallback it reports: the assumed template, everywhere
    assert env["zeff_sigma_tier"]["tier"] == "scalar"
    for ch in ("ne", "te", "ni", "ti"):
        assert env["sigma_source"][ch]["kind"] == "assumed", ch


def test_with_ida_path_set_there_is_no_warning(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    src = HS._src(ddp, cdf, ni_source="standard")
    bl = HS._read(ddp, cdf, ni_source="standard")
    env, msgs = _resolve(_cfg(src, UncertaintyConfig(ida_path=cdf)), bl)
    assert msgs == []
    assert env["zeff_sigma_tier"]["tier"] != "scalar"


def test_the_measured_routes_read_the_source_file_and_do_not_warn(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    src = HS._src(ddp, cdf)                       # the measured default
    bl = HS._read(ddp, cdf)
    _, msgs = _resolve(_cfg(src, UncertaintyConfig()), bl)
    assert msgs == []


def test_the_plain_dd_route_does_not_warn(tmp_path):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    ddp, *_ = NF._build(tmp_path)
    src = ImasSource(ids_path=ddp, time=1.0, impurity_Z=NF.Z_IMP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bl = read_imas_baseline(src, kinetic_source="fuse")
    _, msgs = _resolve(_cfg(src, UncertaintyConfig(), "fuse"), bl)
    assert msgs == []
