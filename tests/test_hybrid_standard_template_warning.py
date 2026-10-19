"""The silent template fallback on the IDA-hybrid ``ni_source="standard"``
sub-route is no longer silent (1.4.1; report only).

That sub-route reads the IDA uncertainty envelopes only from
``UncertaintyConfig.ida_path`` (``Bouquet.from_imas`` sets it).  A
hand-built config that leaves it unset got the ASSUMED sigma (template) for every
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
    assert "'standard'" in msgs[0] and "ASSUMED sigma" in msgs[0]
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


# ------------------------------------------------ review of PR #100, F6
def _std(tmp_path):
    ddp, cdf, *_ = NF._build(tmp_path)
    return (HS._src(ddp, cdf, ni_source="standard"),
            HS._read(ddp, cdf, ni_source="standard"))


def test_the_message_names_only_the_channels_that_fell_back(tmp_path):
    import numpy as np
    src, bl = _std(tmp_path)
    prof = {"ne": 0.05 * np.abs(np.asarray(bl.ne, dtype=float))}
    env, msgs = _resolve(_cfg(src, UncertaintyConfig(sigma_profiles=prof)),
                         bl)
    assert env["sigma_source"]["ne"]["kind"] == "explicit"
    assert len(msgs) == 1
    fell = msgs[0].split("ida_path); ")[1].split(" fall back")[0]
    assert "n_e" not in fell
    for name in ("T_e", "n_i", "T_i", "the Z_eff width"):
        assert name in fell, (name, fell)


def test_no_warning_when_nothing_falls_back(tmp_path):
    """Every kinetic channel explicit (a hand-built sigma=0 run) and the
    Z_eff width explicit: no template is used, so nothing is said."""
    import numpy as np
    src, bl = _std(tmp_path)
    z = np.zeros_like(np.asarray(bl.psi_N_kinetic, dtype=float))
    unc = UncertaintyConfig(
        sigma_profiles={c: z for c in ("ne", "te", "ni", "ti")},
        aux_sigmas={"zeff": z})
    env, msgs = _resolve(_cfg(src, unc), bl)
    assert all(env["sigma_source"][c]["kind"] != "assumed"
               for c in ("ne", "te", "ni", "ti", "zeff"))
    assert msgs == []


def test_every_run_says_it_not_only_the_first_of_a_session(tmp_path):
    """Issued against a fresh registry: a second resolution (a second
    generate() in the same kernel) warns again under the default filters."""
    src, bl = _std(tmp_path)
    cfg = _cfg(src, UncertaintyConfig())
    seen = []
    for _ in range(2):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("default")
            resolve_uncertainty(cfg, bl)
        seen.append(sum(_MSG in str(x.message) for x in w))
    assert seen == [1, 1]


def test_a_user_scalar_is_named_as_a_scalar_not_a_template(tmp_path):
    """Review of PR #100 (second pass): kind "assumed" also covers a
    user-set <chan>_scalar_sigma; the message names what was assumed."""
    src, bl = _std(tmp_path)
    env, msgs = _resolve(_cfg(src, UncertaintyConfig(te_scalar_sigma=0.1)),
                         bl)
    assert env["sigma_source"]["te"]["assumed_default"] == "scalar 0.1"
    assert len(msgs) == 1 and "T_e (scalar 0.1)" in msgs[0]
    assert "n_e (template" in msgs[0]


def test_an_internal_re_resolution_does_not_repeat_it(tmp_path):
    """The sigma=0 draw-route check re-resolves the envelope under a filter
    on the message's prefix: one warning per generate(), not two, while
    every other warning still passes."""
    import inspect
    import re
    from bouquet.baseline import UNREAD_IDA_WARNING_PREFIX
    from bouquet.run import Bouquet
    src_ = inspect.getsource(Bouquet._sigma0_draw_route)
    assert "UNREAD_IDA_WARNING_PREFIX" in src_
    src, bl = _std(tmp_path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")
        warnings.filterwarnings("ignore",
                                message=re.escape(UNREAD_IDA_WARNING_PREFIX))
        resolve_uncertainty(_cfg(src, UncertaintyConfig()), bl)
    assert not [x for x in w if _MSG in str(x.message)]
