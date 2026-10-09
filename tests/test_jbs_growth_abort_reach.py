"""Where the loop's growth abort (``JBS_GROWTH_ABORT_PASSES``) can fire.

Review B m5: at the default relaxation settings the earliest abort is pass
10, so it is inert within the post-homotopy ceiling (6) and within an
8-pass ceiling (the pre-2026-10-07 reconstruction default), and reachable
within the reconstruction and draw ceilings (both 12 now) -- or from pass 6 with the pre-2026-09-27
``jbs_relax_halve_on = 1``.  Pinned here so the constant's docstring stays
true; no value is changed.  Synthetic, solver-free.
"""
import contextlib
import io

import numpy as np
import pytest

from bouquet.config import GenerationConfig
from bouquet.jbs_loop import (JBS_GROWTH_ABORT_PASSES, JBS_RELAX_FLOOR,
                              jbs_settings, run_jbs_loop)

_X = np.linspace(0.0, 1.0, 51)


def _diverging(max_passes, **gc):
    """A loop whose Redl leaves every iterate by a growing margin, so r_j
    grows on every pass from pass 2."""
    g = GenerationConfig(reconstruction_engine="legacy",
                         jbs_self_consistent=True, jbs_loop_on_fail="flag",
                         **gc)
    s = dict(jbs_settings(g))
    s["max_passes"] = int(max_passes)
    seen = {}

    def step(jbs, k):
        seen["jbs"], seen["k"] = np.asarray(jbs, float).copy(), k
        return dict(w=np.ones_like(_X), x=_X)

    def evaluate(meas):
        return seen["jbs"] * (1.0 + 0.1 * 1.5 ** seen["k"])

    with contextlib.redirect_stdout(io.StringIO()):
        out = run_jbs_loop(np.full(_X.size, 1e5), step, evaluate, s,
                           Ip=1e6, gate_li=False)
    rec = out["record"]
    assert all(b > a for a, b in zip(rec["r_j"], rec["r_j"][1:]))
    return rec


@pytest.mark.parametrize("ceiling", [6, 8])
def test_the_growth_abort_is_inert_within_a_6_or_8_pass_ceiling(
        ceiling):
    rec = _diverging(ceiling)
    assert rec["n_passes"] == ceiling
    assert rec["stop_reason"].startswith(f"pass ceiling {ceiling} reached")


def test_the_growth_abort_fires_at_pass_10_within_the_draw_ceiling():
    assert JBS_GROWTH_ABORT_PASSES == 3 and JBS_RELAX_FLOOR == 0.25
    assert GenerationConfig().jbs_max_passes_draw == 12
    assert GenerationConfig().jbs_max_passes == 12   # reconstruction too, since 2026-10-07
    rec = _diverging(12)
    assert rec["n_passes"] == 10
    assert rec["stop_reason"].startswith("r_j grew on 3 consecutive passes")
    assert rec["omega"][-3:] == [0.25, 0.25, 0.25]


def test_with_halving_on_every_growth_it_fires_at_pass_6():
    rec = _diverging(8, jbs_relax_halve_on=1)
    assert rec["n_passes"] == 6
    assert rec["stop_reason"].startswith("r_j grew on 3 consecutive passes")
