"""A legacy draw whose homotopy rollback re-solve fails is REJECTED.

Owner-approved change (2026-10-05), the legacy counterpart of the engine
draws' ``homotopy_rollback_failed`` rule: after a failed or saturated
homotopy stage the draw rolls back to the last good stage and re-solves
there; when that re-solve fails, the state left behind is not a converged
solve, so the draw is rejected with its own reason code -- never archived,
never counted toward until-N.  Before, a legacy draw printed "rollback
re-solve failed; stats may be stale" and went on from the failed solve (the
post-homotopy check measured it and the draw could be archived).

``generate_bouquet`` runs end to end on a permissive mock solver with the
draw function and the post-homotopy stage mocked (as in
``tests/test_draw_rejections.py``); only the homotopy solves fail, on the
pass and re-solve the test names.

Synthetic inputs only; no device data.
"""
import os

import h5py
import numpy as np

_N = 21
_X = np.linspace(0.0, 1.0, _N)
_COILS = {"F1A": 1.0e4, "F2A": -2.0e4, "F9A": 5.0e3, "F9B": -5.0e3}
_NAN_ERROR = "Error in solve: Non-finite value (NaN/Inf) in solution"


class _HomotopyGS:
    """Permissive stand-in; every solve made under installed coil bounds is
    a homotopy solve (pass k, then the rollback re-solve).  Those numbered
    in *fail* raise; *saturate* names the homotopy solve after which one
    coil sits ON its bound."""

    def __init__(self, fail=(), saturate=None):
        self.psi_bounds = (-0.1, 0.1)
        self.coil_sets = {n: {} for n in _COILS}
        self.o_point = np.array([1.7, 0.0])
        self._coils = dict(_COILS)
        self._bounds = None
        self.n_h = 0
        self.fail = set(fail)
        self.saturate = saturate
        self.homotopy_solves = []

    def copy_eq(self):
        return dict(coils=dict(self._coils))

    def replace_eq(self, source_eq=None):
        if source_eq:
            self._coils = dict(source_eq["coils"])

    def get_psi(self, normalized=True):
        return np.linspace(0.0, 1.0, 50)

    def set_psi(self, psi, update_bounds=False):
        pass

    def init_psi(self, *a, **k):
        pass

    def set_targets(self, **k):
        pass

    def set_profiles(self, **k):
        pass

    def solve(self, return_its=False):
        if self._bounds is not None:
            self.n_h += 1
            self.homotopy_solves.append(self.n_h)
            if self.n_h in self.fail:
                raise ValueError(_NAN_ERROR)
            if self.saturate == self.n_h:
                # pass 2 at 1 %: F1A ends exactly on its bound
                self._coils = dict(self._coils)
                self._coils["F1A"] = _COILS["F1A"] * 1.01
        return (None, 10) if return_its else None

    def get_coil_currents(self):
        return dict(self._coils), None

    def set_coil_currents(self, c):
        self._coils = dict(c)

    def set_coil_bounds(self, b=None):
        self._bounds = b

    def coil_reg_term(self, coffs, target=0.0, weight=1.0):
        return (dict(coffs), target, weight)

    def set_coil_reg(self, *a, **k):
        pass

    def set_isoflux(self, *a, **k):
        pass

    def get_stats(self, **k):
        return {"l_i": 0.8, "Ip": 1.0e6, "beta_pol": 0.5, "beta_n": 1.5}

    def get_globals(self):
        return (1.0e6, 0.0, 0.0)

    def get_q(self, *a, **k):
        q = np.linspace(1.2, 4.0, _N)
        return (_X, q, None, None, None, None)

    def get_xpoints(self):
        return None, False

    def trace_surf(self, psi):
        t = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        return np.column_stack([1.7 + 0.6 * np.cos(t), 1.0 * np.sin(t)])

    def save_eqdsk(self, *a, **k):
        # reaching the archive writer is what "archived" means here
        raise _Archiving("the mock never writes an eqdsk")


class _Archiving(RuntimeError):
    pass


def _run(tmp_path, monkeypatch, gs):
    import bouquet.TokaMaker_interface as TI
    from bouquet.jbs_loop import jbs_settings

    class _GC:
        jbs_self_consistent = True
        jbs_init = "anchor"
        jbs_rtol_j = 1e-3
        jbs_rtol_Ip = 1e-4
        jbs_tol_li = 1e-3
        jbs_tol_q0 = 2e-3
        jbs_max_passes = 8
        jbs_max_passes_draw = 12
        jbs_max_passes_post_homotopy = 6
        jbs_relax = 0.7
        jbs_loop_on_fail = "raise"

    Jb = 1.0e5 * (1 - _X)
    post = []

    def _perturb(mygs, psi_N, pressure, ne, te, ni, ti, input_j_phi, *a,
                 **k):
        diag = dict(j_BS=Jb, j_inductive=input_j_phi - Jb, j_BS_edge=None,
                    jbs_loop=dict(enabled=True, converged=True),
                    _jbs_ctx=dict(kind="standard"), r2_ip_scale=None,
                    r2_f_ind=None, aux=None, proxy_bias_observed=None)
        return ne, te, ni, ti, None, np.asarray(input_j_phi, float), diag

    def _post_homotopy(mygs, ctx, settings, psi_N, psi_pad, Ip,
                       coil_guard=None):
        post.append(1)
        return dict(accepted_without_passes=True), None, None, None

    monkeypatch.setattr(TI, "perturb_kinetic_equilibrium", _perturb)
    monkeypatch.setattr(TI, "_post_homotopy_jbs", _post_homotopy)
    rej = []
    header = str(tmp_path / "rb")
    ne = 5e19 * (1 - 0.8 * _X ** 2)
    te = 2e3 * (1 - 0.9 * _X ** 2) + 50.0
    jphi = 1.0e6 * (1 - _X ** 2) + Jb
    out = TI.generate_bouquet(
        gs, _X, 1, header, jphi, ne, te, 0.9 * ne, te,
        0.05 * ne, 0.05 * te, 0.05 * ne, 0.05 * te, 0.05 * jphi,
        0.4, 0.4, 0.25, 1.0e6, 0.8, 1.5 * np.ones(_N),
        input_jinductive=jphi - Jb, baseline_j_BS=Jb, psi_N_kinetic=None,
        diagnostic_plots=False, seed=3,
        jbs_loop=jbs_settings(_GC(), draw=True),
        homotopy_passes=[(0.05, 0.10), (0.01, 0.01)],
        rejection_log=rej, n_inspec_target=1, max_total_draws=1)
    return out, rej, header, post


def _draw_groups(header):
    path = f"{header}.h5"
    if not os.path.isfile(path):
        return []
    with h5py.File(path, "r") as hf:
        names = []
        hf.visit(names.append)
    return [n for n in names if n.split("/")[-1].isdigit()]


def _assert_rejected(out, rej, header, post, gs, txt):
    assert [r["reason"] for r in rej] == ["homotopy_rollback_failed"]
    assert rej[0]["stage"] == "homotopy rollback"
    assert "Non-finite" in rej[0]["message"]
    # pass 1, pass 2, the rollback re-solve -- and nothing after it
    assert gs.homotopy_solves == [1, 2, 3]
    assert post == []                    # the post-homotopy stage never ran
    assert out == []                     # not archived
    assert _draw_groups(header) == []
    assert "draw REJECTED (homotopy_rollback_failed)" in txt
    assert "until-N did not reach its target: 0/1" in txt   # not counted


def test_a_failed_rollback_after_a_failed_stage_rejects_a_legacy_draw(
        tmp_path, monkeypatch, capsys):
    gs = _HomotopyGS(fail={2, 3})
    out, rej, header, post = _run(tmp_path, monkeypatch, gs)
    _assert_rejected(out, rej, header, post, gs, capsys.readouterr().out)


def test_a_failed_rollback_after_saturation_rejects_a_legacy_draw(
        tmp_path, monkeypatch, capsys):
    gs = _HomotopyGS(fail={3}, saturate=2)
    out, rej, header, post = _run(tmp_path, monkeypatch, gs)
    txt = capsys.readouterr().out
    assert "saturation detected" in txt
    _assert_rejected(out, rej, header, post, gs, txt)


def test_a_good_rollback_still_archives_the_legacy_draw(tmp_path,
                                                        monkeypatch, capsys):
    """Control: the rule is about a FAILED re-solve -- a rollback whose
    re-solve converges is the pre-existing behaviour (rolled back to pass 1
    and archived)."""
    gs = _HomotopyGS(fail={2})
    # it goes on to the g-file write, where the mock's save raises (a failed
    # save is rejected as eqdsk_save_failed)
    _out, rej, _h, _p = _run(tmp_path, monkeypatch, gs)
    assert [(r["reason"], r["error_type"]) for r in rej] == [
        ("eqdsk_save_failed", "_Archiving")]
    txt = capsys.readouterr().out
    assert gs.homotopy_solves == [1, 2, 3]
    assert "rolled back to pass 1" in txt
    assert "homotopy_rollback_failed" not in txt


def test_the_reason_code_is_registered_for_both_paths():
    from bouquet.TokaMaker_interface import DRAW_REJECTION_REASONS
    msg = DRAW_REJECTION_REASONS["homotopy_rollback_failed"]
    assert "legacy draws" in msg and "Engine draws" in msg
