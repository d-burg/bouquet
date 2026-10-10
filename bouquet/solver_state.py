"""Snapshot and restore of the solver state bouquet changes temporarily.

A TokaMaker object carries its state in three places, and a restore has to
cover all of them:

* **the equilibrium object** (``copy_eq`` / ``replace_eq``): psi, the coil
  currents, the coil regularisation (matrix, targets, weights), the global
  targets (Ip, pax, ...), the flux-function profiles, and the isoflux /
  saddle / flux / Mirnov constraints;
* **the device** (NOT in the equilibrium object): the hard coil bounds
  (``set_coil_bounds``), the VSC gains (``set_coil_vsc``), the Vcoils
  (``set_vcoils``) and the settings pushed by ``update_settings`` (maxits,
  nl_tol, urf, ...);
* **Python attributes bouquet publishes on the solver object**
  (:data:`BOUQUET_SOLVER_ATTRS`): the strong / weak coil-regularisation
  stashes, the reconstruction's own term list (and its record) and the
  hard-bound stash ``generate_bouquet`` and ``Bouquet._apply_coil_reg`` leave
  there, which later draw-path code reads.

**The coil-bound mode is one-way, and bouquet enters it ONCE, at solver
setup.**  Until ``set_coil_bounds`` is first called, OpenFUSIONToolkit solves
the coil least-squares problem by the normal equations; from the first call
on (``set_coil_bounds(None)`` included, which installs +/-1e98) it solves it
by bounded least squares (BVLS), and no call returns it to the unbounded
solve.  The two agree to round-off per solve but not bit for bit, and a
converged Picard iteration carries the difference (measured on the synthetic
g-file example: 3.8e-7 in the inductive amplitude of the engine's
zero-perturbation draw).  Every ``generate()`` enters the bounded mode (its
homotopy installs and then releases bounds), so if the solver started
unbounded, the reconstruction, the first draw batch and every later solve
would not all use the same coil solver, and results would depend on call
order.  :func:`enter_bounded_coil_mode` therefore enters it once, from
:meth:`bouquet.run.Bouquet.setup_solver` (the one place), before the
reconstruction's first solve, on every path: the reconstruction, the
sigma=0 check and every draw run the same coil solve.  It installs no
constraint (+/-1e98 never binds), and :func:`coil_solve_mode` reports the
mode (recorded on the Baseline and in the engine record).  A capture
(:class:`SolverState`) no longer has to enter it; the restore still puts
the bounds on record back.

Not covered (no bouquet code changes them): the isoflux gradient-weight limit
(bouquet always uses the default), the mesh, the coil and conductor
definitions.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

#: Python attributes bouquet publishes on the TokaMaker object (stashes read
#: by later draw-path code); restored as they were (absent stays absent).
BOUQUET_SOLVER_ATTRS = ("_strong_coil_reg", "_weak_coil_reg",
                        "_coil_drift_bounds", "_recon_coil_reg",
                        "_recon_coil_reg_record")

#: The attribute :func:`enter_bounded_coil_mode` sets on the solver object.
#: Deliberately NOT in :data:`BOUQUET_SOLVER_ATTRS`: the mode it records is
#: one-way, so no restore may remove it.
COIL_SOLVE_MODE_ATTR = "_bouquet_coil_solve_mode"

#: The mode :func:`enter_bounded_coil_mode` puts the solver in.
COIL_SOLVE_BOUNDED = "bounded"
#: The mode :func:`keep_unbounded_coil_mode` records (the swb method).
COIL_SOLVE_UNBOUNDED = "unbounded"


def enter_bounded_coil_mode(mygs):
    """Put *mygs*'s coil least-squares solve in OpenFUSIONToolkit's bounded
    (BVLS) mode, once, and record it; return the mode.

    Called from ONE place, :meth:`bouquet.run.Bouquet.setup_solver`, after
    the coil set, the VSC and the coil regularisation are installed and
    before any solve -- so the reconstruction, the sigma=0 check and every
    draw run the same coil solver whatever order they are called in (see
    the module docstring for why the mode matters).  The bounds installed
    are the ones bouquet has on record (``_coil_drift_bounds``; at setup
    there are none, so +/-1e98, which never binds): entering the mode adds
    no constraint.  Idempotent: a solver already recorded as bounded is
    left alone.  A solver object without ``set_coil_bounds`` (a test
    stand-in) is left alone and reported by :func:`coil_solve_mode` as
    ``"unknown"``."""
    if getattr(mygs, COIL_SOLVE_MODE_ATTR, None) == COIL_SOLVE_BOUNDED:
        return COIL_SOLVE_BOUNDED
    if not hasattr(mygs, "set_coil_bounds"):
        return coil_solve_mode(mygs)
    mygs.set_coil_bounds(copy.deepcopy(getattr(mygs, "_coil_drift_bounds",
                                               None)))
    setattr(mygs, COIL_SOLVE_MODE_ATTR, COIL_SOLVE_BOUNDED)
    return COIL_SOLVE_BOUNDED


def keep_unbounded_coil_mode(mygs):
    """Record that *mygs* stays in the unbounded (normal-equations) coil
    solve: a method that never installs coil bounds (swb) is set up so."""
    if getattr(mygs, COIL_SOLVE_MODE_ATTR, None) != COIL_SOLVE_BOUNDED:
        setattr(mygs, COIL_SOLVE_MODE_ATTR, COIL_SOLVE_UNBOUNDED)
    return coil_solve_mode(mygs)


def coil_solve_mode(mygs):
    """``"bounded"`` once :func:`enter_bounded_coil_mode` has run on *mygs*,
    ``"unbounded"`` after :func:`keep_unbounded_coil_mode`, else
    ``"unknown"`` (OpenFUSIONToolkit does not report the mode: a solver
    bouquet did not set up may be in either)."""
    mode = getattr(mygs, COIL_SOLVE_MODE_ATTR, None)
    if mode in (COIL_SOLVE_BOUNDED, COIL_SOLVE_UNBOUNDED):
        return mode
    return "unknown"


def _settings_values(mygs):
    s = getattr(mygs, "settings", None)
    if s is None:
        return None
    out = {}
    for k in dir(s):
        if k.startswith("_"):
            continue
        v = getattr(s, k)
        if callable(v):
            continue
        out[k] = copy.deepcopy(v)
    return out


def _vsc_facs(mygs):
    vc = getattr(mygs, "_virtual_coils", None)
    if not isinstance(vc, dict):
        return None
    return copy.deepcopy((vc.get("#VSC") or {}).get("facs"))


class SolverState:
    """Everything bouquet may change on a TokaMaker object, captured so that
    :meth:`restore` puts it back.  Build with :meth:`capture`."""

    def __init__(self, mygs):
        self.mygs = mygs
        self.bounds = None
        self.eq = None
        self.settings = None
        self.attrs = {}
        self.vsc = None
        self.vcoils = None
        self.mode = None

    @classmethod
    def capture(cls, mygs):
        """Capture *mygs*'s state.  The coil-bound mode is not touched: a
        solver set up by :meth:`bouquet.run.Bouquet.setup_solver` is already
        in the bounded mode (:func:`enter_bounded_coil_mode`), which is what
        makes a capture and its restore the same state."""
        self = cls(mygs)
        self.bounds = copy.deepcopy(getattr(mygs, "_coil_drift_bounds", None))
        self.mode = coil_solve_mode(mygs)
        if hasattr(mygs, "copy_eq") and hasattr(mygs, "replace_eq"):
            self.eq = mygs.copy_eq()
        self.settings = _settings_values(mygs)
        d = getattr(mygs, "__dict__", {})
        self.attrs = {k: copy.deepcopy(d[k]) for k in BOUQUET_SOLVER_ATTRS
                      if k in d}
        self.vsc = _vsc_facs(mygs)
        self.vcoils = copy.deepcopy(getattr(mygs, "_vcoils", None))
        return self

    def restore(self):
        """Put every captured piece back: the equilibrium object, the
        settings (pushed with ``update_settings``), the VSC gains and Vcoils
        when they changed, the coil bounds on record, and the Python
        attributes (an attribute absent at capture is removed)."""
        mygs = self.mygs
        if self.eq is not None:
            mygs.replace_eq(source_eq=self.eq)
        if self.settings is not None:
            s = mygs.settings
            for k, v in self.settings.items():
                if getattr(s, k, None) != v:
                    setattr(s, k, copy.deepcopy(v))
            if hasattr(mygs, "update_settings"):
                mygs.update_settings()
        if self.vsc is not None and _vsc_facs(mygs) != self.vsc \
                and hasattr(mygs, "set_coil_vsc"):
            mygs.set_coil_vsc(copy.deepcopy(self.vsc))
        if (getattr(mygs, "_vcoils", None) != self.vcoils
                and hasattr(mygs, "set_vcoils")):
            mygs.set_vcoils(copy.deepcopy(self.vcoils or {}))
        # the coil bounds on record.  Only in the bounded mode: on a solver
        # that was not (one bouquet did not set up, with no bounds on
        # record) the call would itself switch the mode -- and the mode
        # cannot be restored anyway (it is one-way)
        if ((self.mode == COIL_SOLVE_BOUNDED or self.bounds is not None)
                and hasattr(mygs, "set_coil_bounds")):
            mygs.set_coil_bounds(copy.deepcopy(self.bounds))
        d = getattr(mygs, "__dict__", None)
        if d is not None:
            for k in BOUQUET_SOLVER_ATTRS:
                if k in self.attrs:
                    d[k] = copy.deepcopy(self.attrs[k])
                else:
                    d.pop(k, None)


@contextmanager
def preserved_solver_state(mygs):
    """``with preserved_solver_state(mygs): ...`` -- :meth:`SolverState.
    capture` on entry, :meth:`SolverState.restore` on exit (also on an
    exception)."""
    st = SolverState.capture(mygs)
    try:
        yield st
    finally:
        st.restore()
