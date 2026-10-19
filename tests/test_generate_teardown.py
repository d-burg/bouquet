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


#: a functional caller's own regularisation (measured-current targets)
_CALLER = [(("F1A",), 2.0e4, 50.0), (("F2A",), -1.5e4, 50.0),
           (("#VSC",), 0.0, 1.0e-2)]


class _RegInEqFake(FakeTokaMaker):
    """A stand-in whose equilibrium object carries the coil regularisation,
    as OpenFUSIONToolkit's does (copy_eq copies the reg matrix, targets and
    weights), with the caller's own reg installed and no reconstruction
    record (a functional caller)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reg = list(_CALLER)

    def copy_eq(self):
        return (super().copy_eq(), list(self.reg or []))

    def replace_eq(self, source_eq=None):
        if source_eq is not None:
            super().replace_eq(source_eq=source_eq[0])
            self.reg = list(source_eq[1])


def test_without_a_record_the_callers_reg_is_put_back(tmp_path, monkeypatch):
    """Review of #100, F7: a functional caller (no reconstruction record)
    gets back the regularisation IT had installed, via the entry
    equilibrium, not a guessed default."""
    import bouquet.TokaMaker_interface as TI
    seen = []
    real = TI.generate_exit_teardown

    def spy(mygs, entry_eq=None):
        seen.append(list(mygs.reg or []))           # the run left it strong
        out = real(mygs, entry_eq=entry_eq)
        seen.append(out)
        return out
    monkeypatch.setattr(TI, "generate_exit_teardown", spy)
    diags, rej, h, G = TED._generate(tmp_path / "td0", monkeypatch, n=1,
                                     fake_cls=_RegInEqFake)
    fake = G.fake
    assert any(t[2] == 1.0e4 for t in seen[0])
    assert seen[1]["coil_reg"].startswith("the caller's")
    assert fake.reg == _CALLER
    assert not hasattr(fake, "_strong_coil_reg")


def test_without_a_record_or_a_copy_the_default_is_the_last_resort(
        tmp_path, monkeypatch):
    """No record and no entry copy (a solver that cannot copy its
    equilibrium): the toward-zero default, every coil at weight 1 and the
    #VSC at 1e-2, recorded as such, never the strong one."""
    import bouquet.TokaMaker_interface as TI
    monkeypatch.setattr(TI, "generate_entry_snapshot", lambda mygs: None)
    diags, rej, h, G = TED._generate(tmp_path / "td1", monkeypatch, n=1)
    fake = G.fake
    assert not hasattr(fake, "_strong_coil_reg")
    want = [((n,), 0.0, 1.0) for n in fake.coil_sets] + [(("#VSC",), 0.0,
                                                           1e-2)]
    assert fake.reg == want


def test_with_a_record_no_entry_copy_is_taken(tmp_path, monkeypatch):
    """The class-API path (the record exists): the entry is not copied and
    the equilibrium is left as the run left it; only the reg is put back."""
    import bouquet.TokaMaker_interface as TI
    taken = []
    real = TI.generate_entry_snapshot
    monkeypatch.setattr(TI, "generate_entry_snapshot",
                        lambda mygs: taken.append(real(mygs)) or taken[-1])
    TED._generate(tmp_path / "td2", monkeypatch, n=1, fake_cls=_SetUpFake)
    assert taken == [None]


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


@pytest.mark.parametrize("fake_cls", [_SetUpFake, _RegInEqFake])
def test_a_second_functional_run_starts_where_the_first_did(
        tmp_path, monkeypatch, fake_cls):
    """Review of #100, F8: the direct functional path #13 describes --
    generate_bouquet twice on one solver, hard bounds on.  The second run
    starts under the same coil regularisation, bounds and stashes as the
    first (with the reconstruction's record, and with only the caller's
    own reg).  Without the teardown it started under the first run's strong
    reg, its hard bounds and both stashes."""
    import bouquet.TokaMaker_interface as TI
    from bouquet.utils import initialize_equilibrium_database
    real = TI.generate_bouquet
    entries = []

    def twice(mygs, *a, **k):
        out = None
        for tag in ("", "_second"):
            entries.append((list(mygs.reg or []), mygs.bounds,
                            hasattr(mygs, "_strong_coil_reg"),
                            hasattr(mygs, "_coil_drift_bounds")))
            a2 = list(a)
            a2[2] = a[2] + tag                       # the header
            if tag:
                initialize_equilibrium_database(a2[2])
            out = real(mygs, *a2, **k)
        return out
    monkeypatch.setattr(TI, "generate_bouquet", twice)
    TED._generate(tmp_path / "two", monkeypatch, n=1, fake_cls=fake_cls,
                  coil_drift_hard_factor=2.0)
    assert len(entries) == 2
    assert entries[1] == entries[0]
    assert entries[1][2:] == (False, False)
