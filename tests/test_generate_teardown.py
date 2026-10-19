"""``generate_bouquet``'s exit teardown (issue #13) -- the fast half (a toy
solver; no OFT).

The draw phase installs a strong coil regularisation (``_strong_coil_reg``)
and, with ``coil_drift_hard_factor``, hard coil bounds
(``_coil_drift_bounds``) on the solver object.  On every exit, an exception
included, ``generate_bouquet`` now puts back the reconstruction's own coil
regularisation (``_recon_coil_reg``; the toward-zero default when none is
on record), releases the bounds and removes both stashes, so a second run in
the same process does not start under them.  The live-solver half (two
sequential in-process sigma=0 runs, bit-identical archives) is
``tests/test_generate_teardown_solver.py``.
"""
import inspect

import pytest

import test_engine_draws as TED
from _engine_fake_gs import FakeTokaMaker

#: the reconstruction's term list a set-up solver carries (a sentinel)
_RECON = [(("F1A",), 1.0e5, 100.0), (("#VSC",), 0.0, 1.0e-2)]


class _SetUpFake(FakeTokaMaker):
    """A stand-in carrying the record ``Bouquet._apply_coil_reg`` publishes."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._recon_coil_reg = list(_RECON)
        self.reg = list(_RECON)


@pytest.mark.parametrize("hard_factor", [None, 2.0])
def test_the_reconstruction_reg_is_back_and_the_stashes_gone(
        tmp_path, monkeypatch, hard_factor):
    regs, bounds = [], []
    real_reg = _SetUpFake.set_coil_reg
    real_bounds = _SetUpFake.set_coil_bounds

    def set_coil_reg(self, reg_terms=None, **kw):
        regs.append(list(reg_terms or []))
        return real_reg(self, reg_terms=reg_terms, **kw)

    def set_coil_bounds(self, b):
        bounds.append(b)
        return real_bounds(self, b)
    monkeypatch.setattr(_SetUpFake, "set_coil_reg", set_coil_reg)
    monkeypatch.setattr(_SetUpFake, "set_coil_bounds", set_coil_bounds)
    diags, rej, h, G = TED._generate(
        tmp_path / "td", monkeypatch, n=2, fake_cls=_SetUpFake,
        coil_drift_hard_factor=hard_factor)
    fake = G.fake
    assert len(diags) == 2
    # the draw phase DID install the strong reg (and the hard bounds) ...
    assert any(t[2] == 1.0e4 for r in regs for t in r)
    if hard_factor is not None:
        assert any(b is not None for b in bounds)
    # ... and the exit put the reconstruction's back, bounds released
    assert regs[-1] == _RECON and fake.reg == _RECON
    assert not hasattr(fake, "_strong_coil_reg")
    assert not hasattr(fake, "_coil_drift_bounds")
    assert fake.bounds is None


def test_without_a_record_the_default_reg_is_installed(tmp_path, monkeypatch):
    """A solver ``setup_solver`` did not prepare: the toward-zero default,
    every coil at weight 1 and the #VSC at 1e-2 (what the draw phase
    replaced), never the strong one."""
    diags, rej, h, G = TED._generate(tmp_path / "td0", monkeypatch, n=1)
    fake = G.fake
    assert not hasattr(fake, "_strong_coil_reg")
    want = [((n,), 0.0, 1.0) for n in fake.coil_sets] + [(("#VSC",), 0.0,
                                                           1e-2)]
    assert fake.reg == want


def test_the_teardown_runs_when_generate_raises(tmp_path, monkeypatch):
    made = []

    class _Kept(_SetUpFake):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    def boom():
        raise OSError("ledger unavailable")
    with pytest.raises(RuntimeError, match="yield ledger"):
        TED._generate(tmp_path / "tdx", monkeypatch, n=3, n_inspec_target=1,
                      fake_cls=_Kept, on_inspec=boom)
    assert len(made) == 1
    fake = made[0]
    assert fake.reg == _RECON
    assert not hasattr(fake, "_strong_coil_reg")


def test_the_wrapper_keeps_the_function_identity():
    from bouquet.TokaMaker_interface import generate_bouquet
    assert generate_bouquet.__name__ == "generate_bouquet"
    assert "def generate_bouquet(" in inspect.getsource(generate_bouquet)
    assert "mygs" in inspect.signature(generate_bouquet).parameters
    assert "n_inspec_target" in inspect.signature(generate_bouquet).parameters
