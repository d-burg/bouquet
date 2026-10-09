"""PIN_JPHI / DIFF_BS take precedence over the self-consistent loop -- and say so.

The two diagnostic modes are the first branches of the draw's bootstrap
block, so with ``jbs_self_consistent`` on they bypass the loop.  That used to
be silent (the baseline carried a loop block, the draws none, and the run
still read as self-consistent).  Now every draw prints and records the bypass,
and ``generate_bouquet`` announces it once, loudly.

Mock solver; synthetic inputs only; no device data.
"""
import inspect

import numpy as np
import pytest

from test_draw_rejections import _FakeGS, _N, _X


def test_each_bypassed_draw_prints_and_records_why():
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    src = inspect.getsource(perturb_kinetic_equilibrium)
    # decided BEFORE the branch chain whose first two arms are the bypasses
    i_by = src.index("_jbs_bypass = None")
    i_pin = src.index("if _pin_jphi and recalculate_j_BS:")
    i_loop = src.index("elif recalculate_j_BS and _jbs_on:")
    assert i_by < i_pin < i_loop
    assert "if _jbs_on and recalculate_j_BS and (_pin_jphi or _diff_bs):" \
        in src
    assert "[jbs-loop] BYPASSED for this draw" in src
    rec = src.split("if _jbs_bypass is not None:\n", 1)[1][:400]
    for k in ("enabled=False", "bypassed=True", "bypass_reason=_jbs_bypass",
              "converged=False"):
        assert k in rec, k


@pytest.mark.parametrize("env", ["PIN_JPHI", "DIFF_BS"])
def test_generate_bouquet_announces_the_bypass(tmp_path, monkeypatch, capsys,
                                               env):
    import bouquet.TokaMaker_interface as TI
    from bouquet.jbs_loop import jbs_settings
    from test_jbs_loop_draw_guards import _GC

    monkeypatch.setenv(env, "1")

    def _perturb(*a, **k):
        raise RuntimeError("mock draw")

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _perturb)
    Jb = 1.0e5 * (1 - _X)
    ne = 5e19 * (1 - 0.8 * _X ** 2)
    te = 2e3 * (1 - 0.9 * _X ** 2) + 50.0
    jphi = 1.0e6 * (1 - _X ** 2) + Jb
    with pytest.warns(RuntimeWarning, match="BYPASSES the loop"):
        try:
            TI.generate_bouquet(
                _FakeGS(), _X, 1, str(tmp_path / "byp"), jphi, ne, te,
                0.9 * ne, te, 0.05 * ne, 0.05 * te, 0.05 * ne, 0.05 * te,
                0.05 * jphi, 0.4, 0.4, 0.25, 1.0e6, 0.8, 1.5 * np.ones(_N),
                input_jinductive=jphi - Jb, baseline_j_BS=Jb,
                diagnostic_plots=False, seed=3,
                jbs_loop=jbs_settings(_GC(), draw=True))
        except Exception:
            pass          # whatever the mock solver cannot do after the notice
    out = capsys.readouterr().out
    assert f"[jbs-loop] NOTICE: {env} is set" in out
