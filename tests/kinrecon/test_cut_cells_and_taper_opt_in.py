"""The solver's cut-cell setting and the opt-in edge taper, without a solve.

* :func:`bouquet.kinrecon.forward.configure_cut_cells` on stand-ins of TokaMaker's settings object (which, like
  TokaMaker's, refuses a new attribute after construction): detection without assigning on a build that lacks the
  setting, the default (on when provided), the recorded off-switch, ``on`` refused on a build without it, the
  one-time message, and the record;
* :func:`bouquet.kinrecon.forward.solver_build` on the OpenFUSIONToolkit build that is installed (patched or not):
  the record and its refusal of a Python package and a library from different builds;
* the edge taper: off by default everywhere, applied only on request, and the request recorded with its parameters.
"""
import hashlib
import os
import sys
import warnings

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon import edge as E  # noqa: E402
from bouquet.kinrecon import forward as FW  # noqa: E402


# ------------------------------------------------------------------ stand-ins
class _Settings:
    """Like TokaMaker's settings: attributes fixed at construction, a new one refused."""

    def __init__(self, with_cut_cells):
        self.nl_tol, self.urf, self.maxits = 1.0e-6, 0.2, 40
        if with_cut_cells:
            self.lcfs_cut_cells = False
        self._initialized = True

    def __setattr__(self, name, value):
        if name not in self.__dict__ and hasattr(self, "_initialized"):
            raise AttributeError(f"Cannot add new attribute '{name}'")
        super().__setattr__(name, value)


class _GS:
    def __init__(self, with_cut_cells):
        self.settings = _Settings(with_cut_cells)
        self.sent = []

    def update_settings(self):
        self.sent.append(dict(vars(self.settings)))


def _build(provides):
    return dict(provides_lcfs_cut_cells=provides, liboftpy_path="/x/liboftpy.so", liboftpy_sha256="0" * 64)


@pytest.fixture
def fresh_notices(monkeypatch):
    monkeypatch.setattr(FW, "_NOTICES", set())


# ------------------------------------------------------------------ cut cells: detection, default, off-switch, record
def test_cut_cells_on_by_default_when_the_build_provides_them(fresh_notices):
    assert FW.ForwardSettings().lcfs_cut_cells == "if_available" == FW.LCFS_CUT_CELLS_DEFAULT
    gs = _GS(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # no message when the cut cells are in effect
        rec = FW.configure_cut_cells(gs, "if_available", _build(True))
    assert gs.settings.lcfs_cut_cells is True and len(gs.sent) == 1 and gs.sent[0]["lcfs_cut_cells"] is True
    assert rec["provided"] is True and rec["requested"] is True and rec["in_effect"] is True
    assert rec["option"] == "if_available" and rec["liboftpy_sha256"] == "0" * 64 and rec["reason"].startswith("on")
    assert FW.read_cut_cells(gs) is True
    # "on" on such a build is the same
    gs2 = _GS(True)
    assert FW.configure_cut_cells(gs2, "on", _build(True))["in_effect"] is True and len(gs2.sent) == 1


def test_build_without_the_setting_runs_as_before_and_says_so_once(fresh_notices):
    gs = _GS(False)
    with pytest.warns(RuntimeWarning, match="no 'lcfs_cut_cells' setting"):
        rec = FW.configure_cut_cells(gs, "if_available", _build(False))
    # nothing was assigned (an assignment would have raised) and nothing was sent
    assert "lcfs_cut_cells" not in vars(gs.settings) and gs.sent == []
    assert rec["provided"] is False and rec["requested"] is None and rec["in_effect"] is None
    assert "point test" in rec["reason"] and FW.read_cut_cells(gs) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # one message per process, not one per model
        FW.configure_cut_cells(_GS(False), "if_available", _build(False))
    with pytest.raises(RuntimeError, match="has no 'lcfs_cut_cells' setting"):
        FW.configure_cut_cells(_GS(False), "on", _build(False))


def test_off_switch_is_sent_and_recorded(fresh_notices):
    gs = _GS(True)
    gs.settings.lcfs_cut_cells = True                     # whatever the instance had, "off" sends False
    with pytest.warns(RuntimeWarning, match="lcfs_cut_cells='off'"):
        rec = FW.configure_cut_cells(gs, "off", _build(True))
    assert gs.settings.lcfs_cut_cells is False and gs.sent[-1]["lcfs_cut_cells"] is False
    assert rec["provided"] is True and rec["requested"] is False and rec["in_effect"] is False
    assert rec["option"] == "off" and "turned off" in rec["reason"]
    # off on a build without the setting: nothing to send
    rec2 = FW.configure_cut_cells(_GS(False), "off", _build(False))
    assert rec2["in_effect"] is None and rec2["provided"] is False


def test_cut_cell_option_and_build_record_are_checked(fresh_notices):
    with pytest.raises(ValueError, match="lcfs_cut_cells must be one of"):
        FW.configure_cut_cells(_GS(True), "auto", _build(True))
    with pytest.raises(RuntimeError, match="disagree"):
        FW.configure_cut_cells(_GS(True), "if_available", _build(False))
    with pytest.raises(RuntimeError, match="disagree"):
        FW.configure_cut_cells(_GS(False), "if_available", _build(True))
    s = FW.ForwardSettings(vsc={"A": 1.0}, lcfs_cut_cells="off")
    assert s.as_dict()["lcfs_cut_cells"] == "off"


# ------------------------------------------------------------------ the installed build
def _oft():
    cand = os.environ.get("OFT_PYTHONPATH")
    if cand and os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
    try:
        from OpenFUSIONToolkit.TokaMaker import _core  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def test_solver_build_record_of_the_installed_build(monkeypatch):
    if not _oft():
        pytest.skip("needs OpenFUSIONToolkit")
    from OpenFUSIONToolkit.TokaMaker import _core
    monkeypatch.setattr(FW, "_BUILD_CACHE", {})
    b = FW.solver_build()
    assert b["provides_lcfs_cut_cells"] == ("lcfs_cut_cells" in vars(_core.tokamaker_settings()))
    assert os.path.isfile(b["liboftpy_path"])
    with open(b["liboftpy_path"], "rb") as f:
        assert b["liboftpy_sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert set(b["tokamaker_python_sha256"]) == {"TokaMaker/_core.py", "TokaMaker/_interface.py"}
    assert FW.solver_build() == b                        # cached, and a copy
    b2 = FW.solver_build()
    b2["tokamaker_python_sha256"].clear()
    assert FW.solver_build()["tokamaker_python_sha256"]


def test_solver_build_refuses_a_python_package_and_library_of_different_builds(monkeypatch):
    if not _oft():
        pytest.skip("needs OpenFUSIONToolkit")
    from OpenFUSIONToolkit.TokaMaker import _core
    has = "lcfs_cut_cells" in vars(_core.tokamaker_settings())
    monkeypatch.setattr(FW, "_BUILD_CACHE", {})
    # pretend the library does (unpatched Python package) or does not (patched package) carry the rule
    monkeypatch.setattr(FW, "CUT_CELL_LIBRARY_SYMBOL", "oftpy_init" if not has else "no_such_symbol_in_liboftpy")
    with pytest.raises(RuntimeError, match="must come from one build"):
        FW.solver_build()
    monkeypatch.setattr(FW, "_BUILD_CACHE", {})


# ------------------------------------------------------------------ edge taper: off by default, opt-in recorded
CYCLE = list(np.tile([1.15e-5, 3.84e-5, 3.39e-5], 14)[:40])
CONV = list(np.geomspace(1e-2, 5e-7, 12))


class _Solve:
    def __init__(self, untapered, tapered):
        self.out = {False: untapered, True: tapered}
        self.calls = []

    def __call__(self, taper):
        self.calls.append(taper)
        conv, hist = self.out[taper is not None]
        return E.SolveAttempt(converged=conv, residuals=hist)


def test_edge_taper_is_off_by_default_everywhere():
    assert E.EdgeTaperSetting().mode is E.EdgeTaperMode.OFF
    assert FW.ForwardSettings().edge_taper.mode is E.EdgeTaperMode.OFF
    assert FW.ForwardSettings().as_dict()["edge_taper"]["mode"] == "off"
    assert E.EdgeTaperCase("c").setting.mode is E.EdgeTaperMode.OFF
    # a solve in a limit cycle is reported failed, NOT tapered, unless the caller asked for the taper
    f = _Solve((False, CYCLE), (True, CONV))
    o = E.solve_with_edge_taper(f)
    assert f.calls == [None] and o.record["mode"] == "off" and not o.record["taper_applied"]
    assert o.record["status"] == E.STATUS_FAILED and "mode off" in o.record["reason"]
    case = E.EdgeTaperCase("c")
    o = E.solve_with_edge_taper(_Solve((False, CYCLE), (True, CONV)), case=case, label="reference")
    assert not o.record["taper_applied"] and case.locked is None


@pytest.mark.parametrize("mode", ["on", "auto"])
def test_explicit_taper_request_is_applied_and_recorded_with_its_parameters(mode):
    spec = E.EdgeTaperSpec(variant="pressure_conserving", start=0.996, band_width=0.015)
    setting = E.EdgeTaperSetting(mode, spec)
    fs = FW.ForwardSettings(vsc={"A": 1.0}, edge_taper=setting)
    d = fs.as_dict()["edge_taper"]
    assert d["mode"] == mode and d["variant"] == "pressure_conserving" and d["start"] == 0.996
    assert d["band_width"] == 0.015
    f = _Solve((False, CYCLE), (True, CONV))
    o = E.solve_with_edge_taper(f, setting)
    assert o.record["taper_applied"] and o.record["status"] == E.STATUS_CONVERGED_TAPER
    assert o.record["setting"] == setting.as_dict() and o.record["mode"] == mode
    assert f.calls[-1] == spec and (len(f.calls) == 1 if mode == "on" else f.calls[0] is None)
