"""A failed state-anchor solve REJECTS a self-consistent-loop draw.

The draw's state anchor re-solves the archived total at the draw's pressure,
and the draw's bootstrap and loop start from that equilibrium.  Its failure
used to be swallowed (``except: pass``), so the draw carried on from whatever
``mygs`` held.  It is now a loud rejection with its own reason code,
``"anchor_solve_failed"``; the solver's error is chained.  A draw whose
anchor solves is untouched (the sigma=0 identity tests run that path).

Runs the real ``perturb_kinetic_equilibrium`` on the toy solver of
``tests/test_sigma0_identity_stages.py``.  Synthetic inputs only.
"""
import numpy as np
import pytest

from test_sigma0_identity_stages import (_delivered, _draw, _li,  # noqa
                                         _reconstruct, _settings, toy)


def _fail_next_solve(toy, exc):
    real = toy.solve
    state = {"armed": True, "n": 0}

    def solve():
        state["n"] += 1
        if state["armed"]:
            state["armed"] = False
            raise exc
        return real()

    toy.solve = solve
    return state


@pytest.mark.parametrize("route", ["ip_renorm", "standard"])
@pytest.mark.parametrize("exc", [Exception("GS: exceeded maxits"),
                                 RuntimeError("solver diverged"),
                                 ValueError("non-finite psi")])
def test_a_failed_anchor_solve_rejects_the_draw(toy, route, exc, capsys):
    from bouquet.TokaMaker_interface import (DRAW_REJECTION_REASONS,
                                             DrawAnchorSolveFailed,
                                             _draw_rejection_reason)
    req, jbs, fx = _reconstruct(toy, kappa_short=0.997)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    F = toy.copy_eq()
    st = _fail_next_solve(toy, exc)
    with pytest.raises(DrawAnchorSolveFailed) as ei:
        _draw(toy, route, dv["request"], dv["j_inductive"], _li(F.achieved),
              _settings(), offset=off)
    # the anchor was the draw's first solve, and nothing solved after it
    assert st["n"] == 1
    assert ei.value.__cause__ is exc
    assert "state-anchor solve failed" in str(ei.value)
    assert "[jbs-loop anchor]" in capsys.readouterr().out
    # its own reason code, never a generic perturb failure
    assert _draw_rejection_reason(ei.value, "perturb") == \
        "anchor_solve_failed"
    assert "anchor_solve_failed" in DRAW_REJECTION_REASONS


def test_a_bug_in_the_anchor_is_not_relabelled(toy):
    """Only solver-failure types are turned into the rejection; a bug
    (TypeError, ...) propagates as itself."""
    req, jbs, fx = _reconstruct(toy, kappa_short=0.997)
    dv, off, _n = _delivered(toy, req, jbs, fx)
    F = toy.copy_eq()
    _fail_next_solve(toy, TypeError("a bug"))
    with pytest.raises(TypeError):
        _draw(toy, "standard", dv["request"], dv["j_inductive"],
              _li(F.achieved), _settings(), offset=off)


def test_the_generate_loop_records_the_reason(tmp_path, monkeypatch, capsys):
    """generate_bouquet records the rejection under its own code (mocked
    draw, as tests/test_draw_rejections.py)."""
    import bouquet.TokaMaker_interface as TI
    import test_draw_rejections as R

    def _perturb(*a, **k):
        raise TI.DrawAnchorSolveFailed("state-anchor solve failed (mock)")

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _perturb)
    rej = []
    X = R._X
    ne = 5e19 * (1 - 0.8 * X ** 2)
    te = 2e3 * (1 - 0.9 * X ** 2) + 50.0
    Jb = 1.0e5 * (1 - X)
    jphi = 1.0e6 * (1 - X ** 2) + Jb
    from bouquet.jbs_loop import jbs_settings
    from test_mse_refusal_restore import _GC
    out = TI.generate_bouquet(
        R._FakeGS(), X, 2, str(tmp_path / "anc"), jphi, ne, te, 0.9 * ne,
        te, 0.05 * ne, 0.05 * te, 0.05 * ne, 0.05 * te, 0.05 * jphi,
        0.4, 0.4, 0.25, 1.0e6, 0.8, 1.5 * np.ones(len(X)),
        input_jinductive=jphi - Jb, baseline_j_BS=Jb, psi_N_kinetic=None,
        diagnostic_plots=False, seed=3,
        jbs_loop=jbs_settings(_GC(), draw=True),
        homotopy_passes=[(0.05, 0.10), (0.01, 0.01)], rejection_log=rej)
    assert out == []
    assert [r["reason"] for r in rej] == ["anchor_solve_failed"] * 2
    assert rej[0]["error_type"] == "DrawAnchorSolveFailed"
    assert "anchor_solve_failed=2" in capsys.readouterr().out
