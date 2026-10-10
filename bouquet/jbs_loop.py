"""Self-consistent bootstrap current: the outer fixed-point loop.

With ``GenerationConfig.jbs_self_consistent=True`` the bootstrap current is no
longer computed once and frozen.  Every path that builds a current profile
containing a bootstrap runs the same relaxed outer iteration::

    E_0      = anchor equilibrium (already solved by the caller)
    jBS_0    = evaluate_jBS(E_0)            (or the legacy SWB result)
    for k = 0 .. K-1:
        jc_k    = closure(E_k, jBS_k)       closure on E_k's geometry
        js_k    = (1 - beta) js_k-1 + beta jc_k     (k >= 1; js_0 = jc_0)
        E_k+1   = GS solve of js_k          (step(jBS_k) does both)
        bs_k    = (1 - beta) bs_k-1 + beta jBS_k    the bootstrap js_k
                                            carries (k >= 1; bs_0 = jBS_0)
        J       = evaluate(E_k+1)           Redl on the NEW equilibrium
        r_k     = residuals(J, bs_k, E_k+1, E_k)
        jBS_k+1 = (1 - omega) jBS_k + omega J
        converged when every active criterion holds on two consecutive passes

Two relaxations, both of the PATH only (at the fixed point ``js = jc`` and
``jBS = J``, so neither moves it):

* ``omega`` (``jbs_relax``) on the bootstrap.  Held fixed; it is halved (floor
  :data:`JBS_RELAX_FLOOR`) only on SUSTAINED growth of ``r_j`` -- growth on
  ``jbs_relax_halve_on`` consecutive passes -- because a single growth event
  is the forced response of the oscillating geometry mode below, not
  divergence.
* ``beta`` (``jbs_relax_current``) on the SOLVED current: the equilibrium of
  pass k is solved with a blend of the previous pass's solved current and the
  closure's new one.  Each pass closes on the PREVIOUS equilibrium's
  geometry, so the closure's current and the geometry it produces form an
  oscillating two-state mode (l_i swings back by a fraction g < 0 per pass)
  that ``omega`` does not act on; ``beta ~ 1/(1 - g)`` damps it.  A step opts
  in by accepting a ``relax`` keyword (:class:`CurrentRelaxer`); the record
  carries ``beta``, the per-pass gap ``||js - jc|| / ||jc||`` and, next to
  it, the UNRELAXED closure-half residual ``||jc_k - js_k-1|| / ||jc_k||``
  (``= gap / (1 - beta)`` on a blended pass; the gap understates it by
  ``1 - beta``).  By default both are recorded only, never gated.

Opt-in current gate (``GenerationConfig.jbs_gate_current_residual``, default
False; under evaluation).  With it ON a pass also has to satisfy
``current_residual = ||jc_k - js_k-1||_w / ||jc_k||_w <= rtol_j`` -- computed
directly from the two arrays: the current the previous pass SOLVED against
the current this pass's closure composes on the equilibrium that solve
produced -- on the same two consecutive passes as every other criterion.
It is measured one pass late (pass 1 cannot count).  A step without the
relaxer reports its solved current as ``meas["j_solved"]``; a step that
reports neither stops the loop at once.  It adds a criterion and changes no
tolerance or ceiling; with it OFF the kernel is bit-identical to the one
before it existed.  It bounds the closure-half RESIDUAL (the delivered
current's consistency with its own closure), not the distance to the fixed
point on a slow monotone mode.

``step`` and ``evaluate`` are supplied by the caller (the IMAS baseline, the
structured / MSE closures, a perturbation draw, the geqdsk reconstruction);
this module owns only what is common to all of them: the residual
definitions, the convergence rule, the relaxation schedule, the failure
policy and the record.

Residuals (all logged every pass)
---------------------------------
``r_j``  current-weighted L2 residual of the profile,
         ``||J - bs_k||_w / ||J||_w`` with ``||f||_w^2 = int |w| f^2 dpsi_N``
         and ``w`` the pass's own Ip weights (so the norm measures current, not
         raw density).  It is the UNRELAXED fixed-point residual -- the
         distance between the bootstrap the equilibrium was SOLVED with,
         ``bs_k``, and the Redl bootstrap of that equilibrium.  ``bs_k`` is
         the iterate ``jBS_k`` itself unless the pass solved a relaxed
         current (``beta < 1``, a blended pass): the solved current is then
         ``(1 - beta) js_k-1 + beta jc_k`` and, the composition being affine
         in the bootstrap, the bootstrap it carries is the same blend of the
         iterates, ``(1 - beta) bs_k-1 + beta jBS_k``.  So a converged loop
         delivers an equilibrium whose bootstrap is within the tolerances of
         its own Redl evaluation.  The residual against the iterate,
         ``||J - jBS_k||_w / ||J||_w`` (``1/omega`` times the relaxed step),
         is recorded beside it as ``r_j_iterate`` / ``r_I_iterate`` on a
         loop that relaxes the current -- a diagnostic, not a criterion.
         (Until 2026-10-06 the iterate residual was the criterion: on a
         blended pass it can read converged while the solved state is not --
         a toy measured 2.4e-4 against 1.09e-3 for the solved one.)  The
         ``omega`` schedule below follows the iterate residual (a path
         heuristic).
``r_I``  ``|int w (J - bs_k) dpsi_N| / Ip``: the same residual as a fraction
         of the plasma current, on the linear part of the closure's measure.
``dl_i`` ``|l_i(E_k+1) - l_i(E_k)|`` (only where the caller measures l_i).
``dq0``  ``|q0(E_k+1) - q0(E_k)|`` (only where an axis row / q0 target is
         active).
``q0 - q0_target``  the TRUE q0 residual, an ADDED criterion
         (``|.| <= q0_tol``) only with ``jbs_loop_q0_corrector=True`` on an
         axis-row channel, where the axis row is moved once per pass from the
         measured q0 (:class:`AxisRowPin`); by default the row is held and the
         residual is only read back by the record-only corrector.

The delivered equilibrium is the last solve; the bootstrap it was solved with
is ``jbs_solved`` (``bs_K``; the iterate ``jbs_used`` when the last pass did
not blend); ``J_final`` (Redl on that equilibrium) differs from it by at most
the tolerances (on a converged loop) and is returned alongside.  ``jbs_used``
is the iterate of the last pass.

Failure is never silent: the library raises :class:`JBSNotConverged` carrying
the full residual history; with ``jbs_loop_on_fail="flag"`` the last iterate is
returned with ``converged=False`` and the caller records a closure-limited
reason.  A residual that grows on ``JBS_GROWTH_ABORT_PASSES`` consecutive
passes at the relaxation floor aborts early with the same error (at the
default relaxation settings no earlier than pass 10, so only within the draw
ceiling of 12 -- see the constant), and so does a
pass that can never count (a gated ``l_i``/``q0`` the step did not return, or
an identically zero ``J`` against a non-zero iterate) -- at that pass, not at
the ceiling.  A non-finite initial guess or evaluated ``J`` raises
:class:`JBSNonFinite` at once, whatever the policy, before it can reach the
next solve.

Nothing here touches an existing solver tolerance: the GS solver's own
``nl_tol``/``maxits``, the closure tolerances and the correctors' acceptance
bands are unchanged; the numbers below define what "j_BS converged" means.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

#: Floor of the under-relaxation factor (``jbs_relax`` is halved toward it
#: on sustained growth of ``r_j``, see ``jbs_relax_halve_on``).
JBS_RELAX_FLOOR = 0.25
#: Consecutive passes that must meet every active criterion.
JBS_REQUIRED_CONSECUTIVE = 2
#: Growing-``r_j`` passes AT the relaxation floor that abort the loop.
#:
#: INERT at the default reconstruction ceilings (verified 2026-10-07, review
#: B m5; value and ceilings unchanged).  Growth is first measurable on pass
#: 2; with the defaults ``jbs_relax = 0.7`` and ``jbs_relax_halve_on = 3``
#: omega is halved after three consecutive growing passes (0.7 -> 0.35 for
#: pass 5, -> 0.25 = :data:`JBS_RELAX_FLOOR` for pass 8), so the earliest
#: abort is at the end of pass 10.  It therefore never fires within
#: ``jbs_max_passes_post_homotopy = 6`` (a diverging loop there runs to its
#: ceiling and fails, or is flagged, there -- the early stop the module
#: docstring describes does not happen), nor did it within the 8-pass
#: reconstruction ceiling that was the default before 2026-10-07.  It CAN
#: fire within ``jbs_max_passes = 12`` (the reconstruction, the engine's MSE
#: stage) and ``jbs_max_passes_draw = 12`` (a draw's loop), from pass 10,
#: and with ``jbs_relax_halve_on = 1`` (a replayed config stored 2026-09-25
#: .. 27) from pass 6.
JBS_GROWTH_ABORT_PASSES = 3
#: Default of ``GenerationConfig.jbs_max_passes_post_homotopy``: the passes
#: a draw may take at the tight coil stage after the post-perturb homotopy
#: (a ceiling, not a tolerance: the two-consecutive-pass rule applies there
#: too, so a post-homotopy stage whose first pass misses needs at least 3).
#: 6 since the owner-approved change of this ceiling from 4 (measured need
#: 5, one pass of margin; see docs/CHANGES_SUMMARY.md).  Only reached with
#: the self-consistent loop on.
JBS_POST_HOMOTOPY_PASSES = 6
#: (The MSE chord stage and the geqdsk post-corrective stage are passes of
#: the baseline loop and take ``jbs_max_passes`` as their ceiling.)
#: MSE chord iteration: the linearisation point has stopped moving when the
#: synthetic tan(gamma) changes by less than this many sigma_eff on every chord
#: between consecutive chord steps.  A NEW criterion introduced with the loop
#: (the plan leaves its value open) -- recorded with every result so it can be
#: reviewed; it gates only when the chord iteration stops, never acceptance.
MSE_CHORD_OFFSET_TOL_SIGMA = 0.1
#: Prefix of every closure_limited reason this module's callers add.
JBS_FLAG_PREFIX = "j_BS loop: "

_INIT_CHOICES = ("anchor", "swb")
_ON_FAIL_CHOICES = ("raise", "flag")


class JBSNotConverged(RuntimeError):
    """The self-consistent bootstrap loop did not converge.

    ``record`` (also ``history``) is the loop record of
    :func:`run_jbs_loop` -- every pass's residuals, relaxation factor and the
    reason it stopped -- so a failure can be diagnosed from the exception
    alone.
    """

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = dict(record or {})
        self.history = self.record


class JBSNonFinite(JBSNotConverged):
    """The loop was handed, or evaluated, a non-finite bootstrap.

    Raised at once -- before the value can be blended into the next iterate
    and handed to a GS solve -- and REGARDLESS of the ``"flag"`` policy: a
    non-finite iterate is not a result that can be flagged and kept.
    ``pass_number`` is 0 for the initial guess, ``index`` / ``psi_N`` locate
    the first non-finite node (``psi_N`` is ``None`` when the grid is not
    known yet).  A subclass of :class:`JBSNotConverged`, so every caller that
    treats a failed loop as a failed slice or draw does so here too.
    """

    def __init__(self, message, record=None, *, pass_number=None,
                 index=None, psi_N=None):
        super().__init__(message, record)
        self.pass_number = pass_number
        self.index = index
        self.psi_N = psi_N


# ---------------------------------------------------------------------------
#  settings
# ---------------------------------------------------------------------------
def validate_jbs_settings(gc) -> None:
    """Refuse malformed ``jbs_*`` values in a :class:`GenerationConfig`.

    Values only (type/range/choice); the workflow-level refusals (the loop
    with ``single_profile_jphi``, or without a bootstrap recompute) live in
    ``Bouquet._validate_workflow``.  Defaults always pass, so this is inert
    for a config that never sets the fields.
    """
    def _get(name, default):
        return getattr(gc, name, default)

    on = _get("jbs_self_consistent", False)
    if not isinstance(on, (bool, np.bool_)):
        raise ValueError(f"generation.jbs_self_consistent must be a bool, got "
                         f"{on!r}")
    pin = _get("jbs_loop_q0_corrector", False)
    if not isinstance(pin, (bool, np.bool_)):
        raise ValueError(f"generation.jbs_loop_q0_corrector must be a bool, "
                         f"got {pin!r}")
    gate = _get("jbs_gate_current_residual", False)
    if not isinstance(gate, (bool, np.bool_)):
        raise ValueError(f"generation.jbs_gate_current_residual must be a "
                         f"bool, got {gate!r}")
    init = _get("jbs_init", "anchor")
    if init not in _INIT_CHOICES:
        raise ValueError(f"generation.jbs_init must be one of {_INIT_CHOICES},"
                         f" got {init!r}")
    fail = _get("jbs_loop_on_fail", "raise")
    if fail not in _ON_FAIL_CHOICES:
        raise ValueError(f"generation.jbs_loop_on_fail must be one of "
                         f"{_ON_FAIL_CHOICES}, got {fail!r}")
    for name, default in (("jbs_rtol_j", 1e-3), ("jbs_rtol_Ip", 1e-4),
                          ("jbs_tol_li", 1e-3), ("jbs_tol_q0", 2e-3)):
        v = _get(name, default)
        if isinstance(v, (bool, np.bool_)):
            # float(True) == 1.0 would pass as a (useless) tolerance
            raise ValueError(f"generation.{name} must be a positive number, "
                             f"not a bool (got {v!r})")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"generation.{name} must be a positive number, "
                             f"got {v!r}") from None
        if not (np.isfinite(fv) and fv > 0.0):
            raise ValueError(f"generation.{name} must be a positive finite "
                             f"number, got {v!r}")
    for name, default in (("jbs_max_passes", 8), ("jbs_max_passes_draw", 12),
                          ("jbs_max_passes_post_homotopy",
                           JBS_POST_HOMOTOPY_PASSES)):
        v = _get(name, default)
        if isinstance(v, (bool, np.bool_)) \
                or not isinstance(v, (int, np.integer)):
            raise ValueError(f"generation.{name} must be an integer, got "
                             f"{v!r}")
        if int(v) < JBS_REQUIRED_CONSECUTIVE:
            raise ValueError(
                f"generation.{name}={int(v)} cannot converge: convergence "
                f"needs {JBS_REQUIRED_CONSECUTIVE} consecutive passing passes")
    w = _get("jbs_relax", 0.7)
    if isinstance(w, (bool, np.bool_)):
        raise ValueError(f"generation.jbs_relax must be a number in "
                         f"[{JBS_RELAX_FLOOR}, 1], not a bool (got {w!r})")
    try:
        fw = float(w)
    except (TypeError, ValueError):
        raise ValueError(f"generation.jbs_relax must be a number in "
                         f"[{JBS_RELAX_FLOOR}, 1], got {w!r}") from None
    if not (JBS_RELAX_FLOOR <= fw <= 1.0):
        raise ValueError(f"generation.jbs_relax must lie in "
                         f"[{JBS_RELAX_FLOOR}, 1] (the floor is "
                         f"JBS_RELAX_FLOOR), got {w!r}")
    b = _get("jbs_relax_current", 0.7)
    try:
        fb = float(b)
    except (TypeError, ValueError):
        raise ValueError(f"generation.jbs_relax_current must be a number in "
                         f"(0, 1], got {b!r}") from None
    if isinstance(b, (bool, np.bool_)) or not (np.isfinite(fb)
                                                and 0.0 < fb <= 1.0):
        raise ValueError(f"generation.jbs_relax_current must lie in (0, 1] "
                         f"(1 = no relaxation of the solved current), got "
                         f"{b!r}")
    h = _get("jbs_relax_halve_on", 3)
    if isinstance(h, (bool, np.bool_)) \
            or not isinstance(h, (int, np.integer)) or int(h) < 1:
        raise ValueError(f"generation.jbs_relax_halve_on must be an integer "
                         f">= 1 (consecutive growing passes that halve "
                         f"omega; 1 = halve on every growth), got {h!r}")


#: The retired ``GenerationConfig.swb_iterations`` default (OFT's own
#: ``solve_with_bootstrap(iterations=3)``); a stored config's other value is
#: loaded as ``bootstrap_kwargs={"iterations": n}``.
SWB_ITERATIONS_DEFAULT = 3


def deprecated_jbs_settings_warning(gc, stacklevel: int = 2) -> Optional[str]:
    """Warn (``DeprecationWarning``) about settings the loop IGNORES.

    ``bootstrap_kwargs`` configures ``solve_with_bootstrap``.  Under the
    legacy engine with ``jbs_self_consistent=True`` the loop's bootstrap is
    Redl (:func:`bouquet.physics.evaluate_jBS`) and SWB runs only for
    ``jbs_init="swb"`` and the jBS-delta / ``DIFF_BS`` caches, so a non-empty
    dict acts there alone.  (The unified engine refuses SWB-only keys
    itself.)  Returns the message (``None`` when nothing is ignored).
    """
    import warnings
    if not bool(getattr(gc, "jbs_self_consistent", False)):
        return None
    if str(getattr(gc, "reconstruction_engine", "legacy")) != "legacy":
        return None
    if str(getattr(gc, "imas_baseline", "")) == "swb":
        return None     # every swb solve is solve_with_bootstrap
    bk = dict(getattr(gc, "bootstrap_kwargs", None) or {})
    if not bk:
        return None
    msg = (f"generation.bootstrap_kwargs={bk!r} configures "
           "solve_with_bootstrap, which the self-consistent bootstrap loop "
           "(jbs_self_consistent=True) runs only for jbs_init='swb' and the "
           "jBS-delta / DIFF_BS caches; the loop's own bootstrap "
           "(evaluate_jBS) does not read it.")
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel + 1)
    return msg


def jbs_settings(gc, *, draw: bool = False) -> dict:
    """The loop settings of a :class:`GenerationConfig`, validated.

    ``draw=True`` selects ``jbs_max_passes_draw`` as the pass ceiling;
    ``post_homotopy_passes`` is ``jbs_max_passes_post_homotopy`` either way.
    ``enabled`` is ``False`` for a config without the fields (an old archive's
    provenance), so every caller can gate on it.
    """
    validate_jbs_settings(gc)
    out = dict(
        enabled=bool(getattr(gc, "jbs_self_consistent", False)),
        init=str(getattr(gc, "jbs_init", "anchor")),
        rtol_j=float(getattr(gc, "jbs_rtol_j", 1e-3)),
        rtol_Ip=float(getattr(gc, "jbs_rtol_Ip", 1e-4)),
        tol_li=float(getattr(gc, "jbs_tol_li", 1e-3)),
        tol_q0=float(getattr(gc, "jbs_tol_q0", 2e-3)),
        max_passes=int(getattr(gc, "jbs_max_passes_draw", 12) if draw
                       else getattr(gc, "jbs_max_passes", 8)),
        relax=float(getattr(gc, "jbs_relax", 0.7)),
        relax_current=float(getattr(gc, "jbs_relax_current", 0.7)),
        relax_halve_on=int(getattr(gc, "jbs_relax_halve_on", 3)),
        on_fail=str(getattr(gc, "jbs_loop_on_fail", "raise")),
        relax_floor=float(JBS_RELAX_FLOOR),
        required_consecutive=int(JBS_REQUIRED_CONSECUTIVE),
        growth_abort_passes=int(JBS_GROWTH_ABORT_PASSES),
        post_homotopy_passes=int(getattr(gc, "jbs_max_passes_post_homotopy",
                                         JBS_POST_HOMOTOPY_PASSES)),
    )
    if bool(getattr(gc, "jbs_gate_current_residual", False)):
        # only present when ON, so the default settings dict (and every
        # record built from it) is exactly what it was before the flag
        out["gate_current_residual"] = True
    from .physics import EPS_DEFINITION_DEFAULT, check_eps_definition
    eps_def = check_eps_definition(getattr(gc, "eps_definition",
                                           EPS_DEFINITION_DEFAULT))
    if eps_def != EPS_DEFINITION_DEFAULT:
        # the Redl eps / nu* R opt-in (GenerationConfig.eps_definition):
        # carried to every loop evaluation and record; only present when
        # not the default, so the default settings dict keeps its keys
        out["eps_definition"] = eps_def
    return out


def tolerances_record(settings: dict) -> dict:
    """The tolerance block every loop record carries."""
    return dict(rtol_j=settings["rtol_j"], rtol_Ip=settings["rtol_Ip"],
                tol_li=settings["tol_li"], tol_q0=settings["tol_q0"],
                max_passes=settings["max_passes"],
                required_consecutive=settings["required_consecutive"],
                relax_start=settings["relax"],
                relax_floor=settings["relax_floor"],
                relax_halve_on=int(settings.get("relax_halve_on", 1)),
                relax_current=float(settings.get("relax_current", 1.0)),
                growth_abort_passes=settings["growth_abort_passes"],
                post_homotopy_passes=int(settings.get(
                    "post_homotopy_passes", JBS_POST_HOMOTOPY_PASSES)))


# ---------------------------------------------------------------------------
#  provenance
# ---------------------------------------------------------------------------
_OFT_BUILD_CACHE = {}


#: Characters a recorded build identifier may contain: no directory
#: separator, no "~", nothing that could carry a filesystem location.
_BUILD_ID_SAFE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                           "0123456789._+-")


def _build_token(v):
    """*v* reduced to the characters of :data:`_BUILD_ID_SAFE` (``None`` when
    nothing is left): a version string or a commit id, never a path."""
    if v is None:
        return None
    t = "".join(c for c in str(v).strip() if c in _BUILD_ID_SAFE)
    return t or None


def _sha256_file(path, chunk=1 << 20):
    """Hex SHA-256 of a file's bytes."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_tree(root, suffixes=(".py",)):
    """Digest of a package's sources: sorted RELATIVE paths + contents (the
    digest ``tests/golden/make_golden_fixture.py`` stamps as
    ``sources_sha256``), so two installs of one revision in different
    directories hash identically and no path enters the result."""
    import hashlib
    import os
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if not name.endswith(suffixes):
                continue
            full = os.path.join(dirpath, name)
            h.update(os.path.relpath(full, root).encode())
            h.update(_sha256_file(full).encode())
    return h.hexdigest()


def _oft_library_path(pkg_dir):
    """The compiled ``liboftpy`` the OpenFUSIONToolkit package loads (a
    local path, used only to hash the file -- never recorded).

    The library actually loaded when ``OpenFUSIONToolkit._interface`` is
    imported (its ``ctypes`` handle's ``_name``); otherwise the package's own
    search, in its order: ``liboftpy<.so|.dylib>`` beside the package
    (``realpath`` of its directory), then ``../../bin`` from there.
    """
    import os
    import sys
    mod = sys.modules.get("OpenFUSIONToolkit._interface")
    name = getattr(getattr(mod, "oftpy_lib", None), "_name", None)
    if isinstance(name, str) and os.path.isfile(name):
        return name
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    root = os.path.realpath(pkg_dir)
    for base in (root, os.path.join(root, "..", "..", "bin")):
        cand = os.path.join(base, "liboftpy" + suffix)
        if os.path.isfile(cand):
            return cand
    return None


def oft_build_info() -> dict:
    """``{version, git_hash, library_sha256, sources_sha256, build_id}`` of
    the imported OpenFUSIONToolkit (cached).

    ``version`` is the package's ``__version__``; ``git_hash`` the SHORT
    (12-character) commit id of the checkout the package lives in, when there
    is one (an install tree without ``.git`` -- every production install --
    records ``None``); ``library_sha256`` the SHA-256 of the compiled
    ``liboftpy`` the package loads (:func:`_oft_library_path`) and
    ``sources_sha256`` the digest of the package's Python sources (sorted
    relative paths + contents) -- the two fields
    ``tests/golden/make_golden_fixture.py`` stamps, measured rather than
    stated, so a fork build and an upstream build of the same version are
    distinguishable from an archive alone; ``build_id`` all of it in one
    string (the library digest shortened to 12 characters).  **No
    filesystem path is recorded** -- loop records are written into archives
    and public fixtures, and the install location names the machine and the
    user.  Every field is reduced to ``[A-Za-z0-9._+-]``, so no directory
    component can survive.  Never raises (a field that cannot be measured is
    ``None``).
    """
    if "info" in _OFT_BUILD_CACHE:
        return dict(_OFT_BUILD_CACHE["info"])
    info = {"version": None, "git_hash": None, "library_sha256": None,
            "sources_sha256": None, "build_id": None}
    try:
        import os
        import subprocess
        import OpenFUSIONToolkit as _oft
        info["version"] = _build_token(getattr(_oft, "__version__", None))
        pkg_dir = os.path.dirname(os.path.abspath(_oft.__file__))
        try:
            out = subprocess.run(
                ["git", "-C", pkg_dir, "rev-parse", "--short=12", "HEAD"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                info["git_hash"] = _build_token(out.stdout.strip())
        except Exception:
            pass
        try:
            lib = _oft_library_path(pkg_dir)
            if lib is not None:
                info["library_sha256"] = _build_token(_sha256_file(lib))
        except Exception:
            pass
        try:
            info["sources_sha256"] = _build_token(_sha256_tree(pkg_dir))
        except Exception:
            pass
    except Exception:
        pass
    info["build_id"] = ("OpenFUSIONToolkit"
                        + ("" if info["version"] is None
                           else f" {info['version']}")
                        + ("" if info["git_hash"] is None
                           else f" git {info['git_hash']}")
                        + ("" if info["library_sha256"] is None
                           else f" lib {info['library_sha256'][:12]}"))
    _OFT_BUILD_CACHE["info"] = dict(info)
    return dict(info)


# ---------------------------------------------------------------------------
#  residuals
# ---------------------------------------------------------------------------
def _trap(y, x):
    from scipy.integrate import trapezoid
    return float(trapezoid(np.asarray(y, dtype=float),
                           np.asarray(x, dtype=float)))


def weighted_norm(f, w, x) -> float:
    """``sqrt(int |w| f^2 dx)`` -- the current-weighted L2 norm."""
    f = np.asarray(f, dtype=float)
    w = np.abs(np.asarray(w, dtype=float))
    return float(np.sqrt(max(_trap(w * f * f, x), 0.0)))


def profile_residuals(J, jbs, w, x, Ip) -> dict:
    """``r_j`` and ``r_I`` of a Redl profile ``J`` against the profile ``jbs``
    the equilibrium was solved with, plus the logged pedestal diagnostics.

    ``J`` and ``jbs`` are the field-aligned bootstrap ``kappa <j.B>`` only
    (:func:`bouquet.physics.evaluate_jBS` since ``/4``): the pressure-driven
    ``p'G`` is its own bucket (owner decision D2), so it neither enters the
    normaliser ``||J||_w`` nor the ``I_BS`` / ``jBS_peak`` diagnostics -- the
    definition of the base commit (PR #64's ``/3`` evaluator put ``p'G``
    into ``J``, which shrank ``r_j`` by ``||kappa lambda|| / ||kappa lambda +
    P||`` for the same mismatch: a silently looser ``rtol_j``)."""
    J = np.asarray(J, dtype=float)
    jbs = np.asarray(jbs, dtype=float)
    x = np.asarray(x, dtype=float)
    d = J - jbs
    nJ = weighted_norm(J, w, x)
    r_j = (weighted_norm(d, w, x) / nJ) if nJ > 0.0 else (
        0.0 if weighted_norm(d, w, x) == 0.0 else float("inf"))
    Ip = abs(float(Ip))
    I_J = _trap(np.asarray(w, dtype=float) * J, x)
    I_used = _trap(np.asarray(w, dtype=float) * jbs, x)
    r_I = abs(I_J - I_used) / Ip if Ip > 0.0 else float("inf")
    i_pk = int(np.argmax(np.abs(J))) if J.size else 0
    return dict(r_j=float(r_j), r_I=float(r_I), I_BS=float(I_J),
                I_BS_used=float(I_used),
                jBS_peak=float(J[i_pk]) if J.size else float("nan"),
                jBS_peak_psiN=float(x[i_pk]) if J.size else float("nan"))


def residual_weights(eq, psi_N, psi_pad=1e-3, coord="psi_n"):
    """``(w, x, kind)``: the per-surface Ip weights of equilibrium *eq* on
    ``psi_N``, for the residual norms.

    ``psi_N`` is the run grid; in a Φ_N run (``coord="phi_n"``) it is mapped
    to ψ_N on *eq*'s own toroidal-flux map, and ``x`` (the abscissa the
    weights integrate over) is that ψ_N.

    The linear part of the closure's ``jphi-linterp`` measure,
    ``(V'/2pi) |dpsi/dpsi_N| <1/R^2>/<1/R>``, when ``get_q`` returns ``<1/R^2>``;
    otherwise (legacy OFT layout) the ``fsa`` weights
    ``(V'/2pi) |dpsi/dpsi_N| <1/R>`` -- the two differ by the tiny
    ``<R><1/R^2>/<1/R>^2 - 1`` shaping factor, far below anything the norm is
    used to resolve.  ``kind`` names which one was used.  One ``get_q`` call,
    no trace.
    """
    from . import coords
    from .utils import fsa_current_geometry
    x = np.asarray(coords.psi_at(eq, np.asarray(psi_N, dtype=float), coord),
                   dtype=float)
    g = fsa_current_geometry(eq, x, psi_pad=psi_pad, want_pprime=False)
    base = g["dV_dpsi"] / (2.0 * np.pi) * g["dpsi_dpsiN"]
    if g["inv_R2"] is not None:
        return base * g["inv_R2"] / g["inv_R"], x, "jphi-linterp"
    return base * g["inv_R"], x, "fsa"


def _finite_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ---------------------------------------------------------------------------
#  relaxation of the solved current
# ---------------------------------------------------------------------------
class CurrentRelaxer:
    """``beta``-relaxation of the current a loop pass SOLVES.

    A step that accepts it (``step(jbs, k, relax=...)``) passes the current
    its closure assembled -- the ``jphi-linterp`` input it would otherwise
    hand the solver -- through ``relax(j_closure)`` and solves what comes
    back::

        js_0 = jc_0,    js_k = (1 - beta) js_{k-1} + beta jc_k   (k >= 1)

    ``js_{k-1}`` is the current the PREVIOUS pass solved (committed by the
    kernel after each pass), so calling ``relax`` more than once inside one
    pass (a step that re-solves) blends against the same previous current.
    ``beta = 1`` returns ``j_closure`` unchanged.  At the fixed point
    ``jc_k = js_{k-1}`` and the blend is the identity: the path changes, the
    fixed point does not.
    """

    def __init__(self, beta: float):
        self.beta = float(beta)
        self._prev = None            # js of the previous (committed) pass
        self._last = None            # (jc, js) of the current pass
        self.n_calls = 0
        #: whether the current pass's (last) call blended -- the kernel
        #: blends its bootstrap iterate the same way to know the bootstrap
        #: the pass actually SOLVED
        self.last_blended = False

    def __call__(self, j_closure):
        jc = np.asarray(j_closure, dtype=float)
        if (self.beta >= 1.0 or self._prev is None
                or self._prev.shape != jc.shape):
            js = jc.copy()
            self.last_blended = False
        else:
            js = (1.0 - self.beta) * self._prev + self.beta * jc
            self.last_blended = True
        self._last = (jc.copy(), js.copy())
        self.n_calls += 1
        return js

    @property
    def called(self) -> bool:
        """Whether the step passed a current through the relaxer on the
        current pass."""
        return self._last is not None

    def begin_pass(self):
        self._last = None
        self.last_blended = False

    def commit(self):
        """End of a pass: the current it solved becomes the blend base."""
        if self._last is not None:
            self._prev = self._last[1].copy()

    def unrelaxed_residual(self, w=None, x=None):
        """``||jc_k - js_{k-1}|| / ||jc_k||`` for the pass just taken, or
        ``None`` on a pass without a previous solved current.

        The UNRELAXED residual of the closure half of the fixed point: the
        distance between the current the closure asks for on this pass's
        geometry and the current the previous pass solved.  For a blended
        pass it equals ``gap / (1 - beta)`` exactly (the recorded ``gap`` is
        scaled down by ``1 - beta``); unlike that quotient it is also defined
        at ``beta = 1``.  RECORD ONLY -- never gated.  Current-weighted with
        the pass's ``w``/``x`` when the shapes match, plain L2 otherwise.
        """
        if self._last is None or self._prev is None:
            return None
        jc = self._last[0]
        if self._prev.shape != jc.shape:
            return None
        d = jc - self._prev
        if (w is not None and x is not None
                and np.shape(w) == jc.shape == np.shape(x)):
            n = weighted_norm(jc, w, x)
            dn = weighted_norm(d, w, x)
        else:
            n = float(np.linalg.norm(jc))
            dn = float(np.linalg.norm(d))
        return (dn / n) if n > 0.0 else (0.0 if dn == 0.0 else float("inf"))

    def gap(self, w=None, x=None):
        """``(rel, blended)`` for the pass just taken: ``||js - jc|| / ||jc||``
        (current-weighted with the pass's ``w``/``x`` when the shapes match,
        plain L2 otherwise) and whether a blend was applied at all."""
        if self._last is None:
            return None, False
        jc, js = self._last
        d = js - jc
        blended = bool(np.any(d != 0.0))
        if (w is not None and x is not None
                and np.shape(w) == jc.shape == np.shape(x)):
            n = weighted_norm(jc, w, x)
            return ((weighted_norm(d, w, x) / n) if n > 0.0 else
                    (0.0 if not blended else float("inf"))), blended
        n = float(np.linalg.norm(jc))
        return ((float(np.linalg.norm(d)) / n) if n > 0.0 else
                (0.0 if not blended else float("inf"))), blended


# ---------------------------------------------------------------------------
#  the q0 pin acting under the loop (jbs_loop_q0_corrector)
# ---------------------------------------------------------------------------
#: The update rule of :class:`AxisRowPin`, as recorded.
AXIS_ROW_UPDATE_RULE = (
    "j_ref0(k+1) = j0_solved(k) * q0(E_k+1) / q0_target: the legacy "
    "structured corrector's row update j_ref0' = j_ref0 * q0_solved/q0_target "
    "(q0 ~ 1/j_phi(0); the scalar corrector's Newton step is its first-order "
    "expansion), applied once per pass from the q0 MEASURED on that pass's "
    "solved equilibrium, with j0_solved the axis value of the current the "
    "pass actually SOLVED (= the row when the solved current is not "
    "relaxed, beta = 1)")


class AxisRowPin:
    """The on-axis safety-factor pin ACTING under the self-consistent loop.

    ``GenerationConfig.jbs_loop_q0_corrector=True`` on a channel whose axis
    row is active (``closure_channel="sawtooth_bootstrap"``, or
    ``"structured"`` with the sawtooth gate admitting the axis row).  The
    closure of every pass imposes the axis-current row :attr:`row`; after the
    pass is solved, the q0 measured on the NEW equilibrium moves the row for
    the next pass (:data:`AXIS_ROW_UPDATE_RULE`)::

        j_ref0(k+1) = j0_solved(k) * q0(E_k+1) / q0_target

    ``j0_solved`` is the axis value of the current the pass SOLVED: with the
    solved current relaxed (``jbs_relax_current = beta``) the equilibrium sees
    ``(1 - beta) js_k-1(0) + beta j_ref0(k)``, not the row, and ``q0 ~ 1/j0``
    is a statement about the current that was solved.  With ``beta = 1`` it
    IS the row and the rule is the legacy structured corrector's own.  At the
    joint fixed point the row stops moving exactly when ``q0 = q0_target``, so
    the relaxation (like ``omega`` and ``beta``) changes the path, not the
    answer.

    :meth:`within_tol` is the convergence criterion the kernel ADDS for these
    channels, ``|q0 - q0_target| <= q0_tol``, next to (never instead of) the
    ``jbs_tol_q0`` step criterion.  ``q0_tol`` is the channel's own unchanged
    acceptance band.  Nothing here relaxes a criterion: a joint iteration that
    does not reach it fails exactly as the loop fails.
    """

    def __init__(self, q0_target, q0_tol, row0, *, label: str = ""):
        t = _finite_or_none(q0_target)
        tol = _finite_or_none(q0_tol)
        r0 = _finite_or_none(row0)
        if t is None or t == 0.0:
            raise ValueError(f"AxisRowPin[{label}]: q0_target must be a finite"
                             f" non-zero number, got {q0_target!r}")
        if tol is None or tol <= 0.0:
            raise ValueError(f"AxisRowPin[{label}]: q0_tol must be a positive"
                             f" finite number, got {q0_tol!r}")
        if r0 is None or r0 == 0.0:
            raise ValueError(f"AxisRowPin[{label}]: the initial axis row must "
                             f"be a finite non-zero current, got {row0!r}")
        self.label = str(label)
        self.q0_target = t
        self.q0_tol = tol
        self.row0 = r0
        self.row = r0                 # the row the NEXT closure imposes
        self.n_updates = 0
        self._pending = None          # (q0, j0_solved) of the last observation
        self.log = dict(stage=[], axis_row=[], axis_current_solved=[], q0=[],
                        q0_residual=[], q0_residual_over_tol=[],
                        axis_row_next=[])

    def residual(self, q0):
        """``q0 - q0_target`` (``None`` when *q0* is not finite)."""
        q = _finite_or_none(q0)
        return None if q is None else q - self.q0_target

    def within_tol(self, q0) -> bool:
        """``|q0 - q0_target| <= q0_tol`` (False for a non-finite q0)."""
        r = self.residual(q0)
        return bool(r is not None and abs(r) <= self.q0_tol)

    def observe(self, q0, axis_current_solved, *, stage: str = "loop"):
        """Record one solved pass: the row it closed with, the axis current it
        solved, the measured q0 and its residual.  The row is NOT moved here
        (see :meth:`advance`)."""
        r = self.residual(q0)
        self.log["stage"].append(str(stage))
        self.log["axis_row"].append(float(self.row))
        self.log["axis_current_solved"].append(
            _finite_or_none(axis_current_solved))
        self.log["q0"].append(_finite_or_none(q0))
        self.log["q0_residual"].append(r)
        self.log["q0_residual_over_tol"].append(
            None if r is None else float(abs(r) / self.q0_tol))
        self.log["axis_row_next"].append(None)
        self._pending = (_finite_or_none(q0),
                         _finite_or_none(axis_current_solved))
        return r

    def advance(self):
        """Move the row from the last observation (a further pass follows).

        Returns the new row, or ``None`` (row kept) when the observation
        cannot define a step (non-finite q0 or axis current, or a new row that
        is not a finite current of the initial row's sign) -- the residual
        criterion then keeps failing on its own, so a kept row is never a
        silent success."""
        if self._pending is None:
            return None
        q0, j0 = self._pending
        self._pending = None
        if q0 is None or j0 is None:
            return None
        new = j0 * q0 / self.q0_target
        if not (np.isfinite(new) and new != 0.0
                and np.sign(new) == np.sign(self.row0)):
            return None
        self.row = float(new)
        self.n_updates += 1
        self.log["axis_row_next"][-1] = float(new)
        return float(new)

    def snapshot(self):
        """State a caller can hand back to :meth:`restore` (the MSE stage's
        refusal restores the pre-MSE row)."""
        return (self.row, self.n_updates)

    def restore(self, snap):
        self.row, self.n_updates = float(snap[0]), int(snap[1])
        self._pending = None

    def record(self) -> dict:
        """The JSON-safe block the loop record carries
        (``record["q0_pin"]``)."""
        last = (self.log["q0_residual"][-1] if self.log["q0_residual"]
                else None)
        return dict(
            mode="acting: per-pass axis-row update (jbs_loop_q0_corrector)",
            update_rule=AXIS_ROW_UPDATE_RULE,
            criterion=("|q0 - q0_target| <= q0_tol on the pass's solved "
                       "equilibrium, ADDED to the jbs_tol_q0 step criterion"),
            q0_target=float(self.q0_target), q0_tol=float(self.q0_tol),
            axis_row_initial=float(self.row0),
            axis_row_final=float(self.row),
            n_row_updates=int(self.n_updates),
            final_q0_residual=last,
            final_q0_residual_over_tol=(None if last is None
                                        else float(abs(last) / self.q0_tol)),
            **{k: list(v) for k, v in self.log.items()})


#: Definition string of the gated closure-half residual (recorded with it).
RESIDUAL_SOLVED_DEFINITION = (
    "r_j = ||J - bs_k||_w / ||J||_w and r_I = |int w (J - bs_k)| / Ip with "
    "J Redl on the solved equilibrium and bs_k the bootstrap that "
    "equilibrium was SOLVED with: the relaxer's blend of the iterates, "
    "(1 - beta) bs_k-1 + beta jBS_k, on a blended pass, else the iterate "
    "jBS_k; r_j_iterate / r_I_iterate (against jBS_k) recorded only")

CURRENT_GATE_DEFINITION = (
    "||jc_k - js_{k-1}||_w / ||jc_k||_w, computed directly from the two "
    "arrays: js_{k-1} is the current pass k-1 SOLVED (the jphi input of its "
    "solve) and jc_k is the current pass k's closure COMPOSED on the "
    "equilibrium that solve produced, with the bootstrap evaluated on it "
    "(relaxed by omega).  It is the closure-half residual of pass k-1's "
    "solve, measured one pass later (the closure on a solve's geometry is "
    "only composed by the next pass).  Steps without the relaxer report "
    "their solved current as meas['j_solved'] (beta not applied, so "
    "jc_k = js_k).  Pass 1 has no previous solved current and cannot count.  "
    "Norm and tolerance: the loop's current-weighted L2 norm and jbs_rtol_j.")

#: What ``run_jbs_loop(start_refresh=...)`` does, as recorded.
START_REFRESH_DEFINITION = (
    "after pass 1 (when a further pass follows) the iterate of pass 2 is "
    "start_refresh(J_1, meas_1) -- a restart from what pass 1 measured -- "
    "instead of the relaxed blend (1 - omega) jbs_0 + omega J_1; every later "
    "pass blends as usual.  r_j_before / r_I_before: pass 1's residuals (J_1 "
    "against the start it was solved with); r_j_after / r_I_after: pass 2's "
    "(Redl on the next solved state against the refreshed iterate).  The "
    "path changes, not the criteria, tolerances or ceiling; zero extra "
    "solves.")


def criteria_timeline(record: dict) -> dict:
    """For every ACTIVE criterion of a loop record, the 1-based pass from
    which it held without a break through the last pass (``None`` when it
    does not hold on the last pass), and which criterion was the LAST to be
    met (the latest such pass among the GATED criteria; a list, ties
    included; ``None`` unless every gated criterion holds on the last pass).

    Works on any record, gate on or off: the closure-half residual is read
    from ``current_residual`` (gate on) or else from the record-only
    ``current_residual_unrelaxed``, and is listed in ``gated`` only when the
    record's criteria say it was a criterion.  Pure bookkeeping.
    """
    tol = dict(record.get("tolerances") or {})
    crit = dict(record.get("criteria") or {})
    n = int(record.get("n_passes") or 0)

    def _le(v, t):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(f) and f <= float(t))

    series = {}
    for name, key, tkey in (("r_j", "r_j", "rtol_j"),
                            ("r_I", "r_I", "rtol_Ip"),
                            ("dl_i", "dl_i", "tol_li"),
                            ("dq0", "dq0", "tol_q0")):
        if crit.get(name) and tkey in tol:
            series[name] = [_le(v, tol[tkey]) for v in record.get(key, [])]
    cur = record.get("current_residual")
    if cur is None:
        cur = record.get("current_residual_unrelaxed")
    gated_cur = bool(crit.get("current_residual"))
    if cur is not None and "rtol_j" in tol and (
            gated_cur or any(v is not None for v in cur)):
        series["current_residual"] = [_le(v, tol["rtol_j"]) for v in cur]
    since = {}
    for name, oks in series.items():
        oks = list(oks)[:n]
        if not oks or not oks[-1]:
            since[name] = None
            continue
        i = len(oks) - 1
        while i > 0 and oks[i - 1]:
            i -= 1
        since[name] = i + 1
    gated = sorted(k for k in series if k != "current_residual" or gated_cur)
    last = None
    if gated and all(since[k] is not None for k in gated):
        m = max(since[k] for k in gated)
        last = sorted(k for k in gated if since[k] == m)
    return dict(met_since_pass=since, last_met=last, gated=gated)


def _relative_difference(a, b, w=None, x=None):
    """``||a - b|| / ||a||`` -- current-weighted with ``w``/``x`` when the
    shapes match (the loop's norm), plain L2 otherwise; ``None`` on a shape
    mismatch."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return None
    d = a - b
    if (w is not None and x is not None
            and np.shape(w) == a.shape == np.shape(x)):
        n, dn = weighted_norm(a, w, x), weighted_norm(d, w, x)
    else:
        n, dn = float(np.linalg.norm(a)), float(np.linalg.norm(d))
    return (dn / n) if n > 0.0 else (0.0 if dn == 0.0 else float("inf"))


def _step_takes_relax(step) -> bool:
    import inspect
    try:
        return "relax" in inspect.signature(step).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
#  the kernel
# ---------------------------------------------------------------------------
def run_jbs_loop(jbs0, step: Callable, evaluate: Callable, settings: dict, *,
                 Ip: float, meas0: Optional[dict] = None,
                 gate_li: bool = True, gate_q0: bool = False,
                 label: str = "", init: Optional[str] = None,
                 init_source: Optional[str] = None,
                 grid: str = "psi_N native",
                 on_pass: Optional[Callable] = None,
                 max_passes: Optional[int] = None,
                 raise_on_fail: Optional[bool] = None,
                 q0_pin: Optional[AxisRowPin] = None,
                 extra=None,
                 start_refresh: Optional[Callable] = None) -> dict:
    """Iterate closure <-> GS <-> Redl to the fixed point.

    Parameters
    ----------
    jbs0 : array
        Initial bootstrap profile (the one the FIRST solve is assembled with).
    step : callable ``step(jbs, k) -> meas`` or ``step(jbs, k, relax=...)``
        Closure + assembly + GS solve with bootstrap ``jbs``; returns the
        measurement dict of the NEW equilibrium: ``w`` and ``x`` (the Ip weights
        and the grid ``jbs`` lives on), optionally ``li`` and ``q0``.  Any other
        keys are carried through untouched (``meas_final``).  A step that
        accepts a ``relax`` keyword receives the pass's
        :class:`CurrentRelaxer` and solves ``relax(j_closure)`` instead of
        ``j_closure`` (the ``jbs_relax_current`` relaxation of the solved
        current); a two-argument step is called as before and the record says
        the current was not relaxed.
    evaluate : callable ``evaluate(meas) -> J``
        Redl bootstrap on the equilibrium ``step`` just produced, composed
        exactly the way ``jbs`` is (same smoothing / isolation / scale).
    settings : dict
        :func:`jbs_settings`.
    Ip : float
        Plasma current the ``r_I`` residual is normalised by.
    meas0 : dict or None
        Measurements of the starting equilibrium E_0 (``li``/``q0``), so the
        first pass has a ``dl_i``/``dq0``.
    gate_li, gate_q0 : bool
        Whether ``dl_i`` / ``dq0`` are convergence criteria here (``dq0`` only
        where an axis row / q0 target is active); they are logged either way
        whenever measured.
    on_pass : callable or None
        ``on_pass(k, meas, J, entry)`` after every pass (callers use it to
        refresh per-pass state, e.g. an axis-row target).  ``entry`` carries
        the pass's residuals, ``omega_used`` and ``omega_next`` (the
        relaxation the next iterate is built with; ``None`` when no further
        pass follows), so a caller's own per-pass update can be relaxed by the
        same factor.
    init : str or None
        The ``jbs_init`` setting recorded as ``record["init"]`` (default: the
        settings' own).
    init_source : str or None
        What ``jbs0`` actually is, in words, recorded as
        ``record["init_source"]`` (e.g. a draw's "evaluate_jBS on the draw's
        anchor equilibrium with the draw's own perturbed kinetics").  Record
        only: the fixed point does not depend on the initial iterate.
    max_passes : int or None
        Override of ``settings["max_passes"]`` (the post-homotopy stage).
    raise_on_fail : bool or None
        Override of the ``settings["on_fail"]`` policy.
    q0_pin : :class:`AxisRowPin` or None
        ``jbs_loop_q0_corrector=True`` on an axis-row channel: the step
        closes with ``q0_pin.row`` and returns ``q0`` and
        ``axis_current_solved`` (the axis value of the current it solved);
        the kernel records both, ADDS ``|q0 - q0_target| <= q0_tol`` to the
        pass criteria (next to ``dq0``, which ``gate_q0`` keeps) and moves the
        row once per pass when a further pass follows.  ``None`` (default):
        the kernel is exactly the one without the pin -- same criteria, same
        record keys.
    extra : object or None
        ADDED convergence criteria of a caller that owns rows the kernel does
        not know (the unified reconstruction engine, :mod:`bouquet.engine`:
        its l_i row and MSE chords).  An object with ``names`` (the criterion
        names recorded in ``record["criteria"]``), ``observe(k, meas) ->
        (ok, never, text)`` -- called once per pass after every built-in
        criterion; ``ok`` is ANDed into the pass verdict (an added condition,
        never a replacement), ``never`` a reason the pass can never count
        (the loop then stops at once, as for a missing gated l_i) or
        ``None``, ``text`` appended to the pass line -- and ``record()`` /
        ``history_text()`` (the block stored as ``record["extra_criteria"]``
        and the text appended to a failure message).  ``None`` (default):
        exactly the kernel without it -- same criteria, same record keys, same
        printed lines.
    start_refresh : callable or None
        A RESTART of the iterate after the first solve (the unified engine's
        draws, ``engine_draw_bootstrap_refresh``): ``start_refresh(J, meas)
        -> jbs`` is called once, after pass 1 when a further pass follows,
        with pass 1's evaluated bootstrap ``J`` and measurement ``meas``,
        and what it returns REPLACES the relaxed blend as the iterate of
        pass 2 (every later pass blends with ``omega`` as usual).  It
        changes the path only: no criterion, tolerance or ceiling is
        touched, and a pass is still judged against the bootstrap it was
        solved with.  Zero extra solves (it may only use what pass 1
        measured).  Recorded as ``record["bootstrap_refresh"]`` (the
        residuals before it -- pass 1's -- and after it -- pass 2's, the
        refreshed iterate against Redl on the next solved state); pass 2's
        ``omega`` entry is ``None`` (a restart, not a blend) and
        ``on_pass`` receives ``omega_next=1.0`` after pass 1.  ``None``
        (default): exactly the kernel without it -- same record keys, same
        printed lines.

    Returns
    -------
    dict with ``jbs_used`` (the bootstrap ITERATE of the last pass),
    ``jbs_solved`` (the bootstrap the delivered solve carries -- the
    relaxer's blend of the iterates on a blended pass, else ``jbs_used``;
    ``r_j`` / ``r_I`` are measured against it), ``J_final`` (Redl on the
    delivered equilibrium), ``meas_final``, ``converged`` and ``record`` (the
    logged block, JSON-safe; on a loop that relaxes the current at
    ``beta < 1`` it adds ``r_j_iterate`` / ``r_I_iterate`` /
    ``I_BS_used_iterate`` / ``bootstrap_blended`` per pass and
    ``residual_definition``).  On non-convergence raises
    :class:`JBSNotConverged` unless the policy is ``"flag"``.
    """
    t0 = time.perf_counter()
    K = int(settings["max_passes"] if max_passes is None else max_passes)
    need = int(settings.get("required_consecutive", JBS_REQUIRED_CONSECUTIVE))
    floor = float(settings.get("relax_floor", JBS_RELAX_FLOOR))
    n_abort = int(settings.get("growth_abort_passes", JBS_GROWTH_ABORT_PASSES))
    omega = float(settings["relax"])
    halve_on = int(settings.get("relax_halve_on", 1))
    beta = float(settings.get("relax_current", 1.0))
    takes_relax = _step_takes_relax(step)
    relaxer = CurrentRelaxer(beta) if takes_relax else None
    if raise_on_fail is None:
        raise_on_fail = (settings.get("on_fail", "raise") == "raise")

    rec = dict(
        enabled=True, label=str(label),
        init=str(init if init is not None else settings.get("init", "anchor")),
        init_source=(None if init_source is None else str(init_source)),
        grid=str(grid),
        tolerances=tolerances_record(dict(settings, max_passes=K)),
        criteria=dict(r_j=True, r_I=True, dl_i=bool(gate_li),
                      dq0=bool(gate_q0),
                      **({} if q0_pin is None else dict(q0_residual=True))),
        n_passes=0, converged=False, stop_reason=None,
        omega=[], r_j=[], r_I=[], dl_i=[], dq0=[], I_BS=[], I_BS_used=[],
        jBS_peak=[], jBS_peak_psiN=[], li=[], q0=[], pass_ok=[],
        wall_s=None,
        relax_current=(beta if takes_relax else None),
        current_relaxation=(
            ("solved current relaxed: js_k = (1-beta) js_k-1 + beta jc_k "
             "(k >= 1); path only, the fixed point is unchanged")
            if takes_relax else
            "not applied: this caller's step does not take the relaxer"),
        current_gap=[], current_blended=[],
        # RECORD ONLY, never gated: the unrelaxed residual of the closure
        # half, ||jc_k - js_{k-1}|| / ||jc_k|| (= current_gap / (1 - beta)
        # on a blended pass).  current_gap understates it by (1 - beta).
        current_residual_unrelaxed=[],
        current_residual_unrelaxed_definition=(
            "||jc_k - js_{k-1}||_w / ||jc_k||_w = current_gap / (1 - beta) "
            "on a blended pass: the unrelaxed closure-half residual; "
            "recorded only, NOT a convergence criterion"),
        relax_halve_on=int(halve_on), omega_halved_at_pass=[],
    )
    # the solved-bootstrap residual (2026-10-06): with the current relaxed
    # (beta < 1) a pass SOLVES the blend of its closure current with the
    # previous solved one, so the bootstrap the equilibrium carries is the
    # same blend of the iterates; r_j / r_I are measured against THAT, and
    # the residual against the iterate is kept as a diagnostic.  Only a
    # caller whose step takes the relaxer at beta < 1 can differ: every
    # other record is key-for-key what it was.
    track_solved = bool(takes_relax and beta < 1.0)
    if track_solved:
        rec.update(
            residual_definition=RESIDUAL_SOLVED_DEFINITION,
            r_j_iterate=[], r_I_iterate=[], I_BS_used_iterate=[],
            bootstrap_blended=[])
    if extra is not None:
        rec["criteria"].update({str(_n): True for _n in extra.names})
    if start_refresh is not None:
        rec["bootstrap_refresh"] = dict(
            requested=True, applied=False, after_pass=1,
            definition=START_REFRESH_DEFINITION)
    try:
        from .physics import evaluate_jbs_version, EPS_DEFINITION_DEFAULT
        # the version of the definition the loop evaluates with (the
        # settings carry a non-default GenerationConfig.eps_definition)
        rec["evaluate_jBS_version"] = evaluate_jbs_version(
            settings.get("eps_definition", EPS_DEFINITION_DEFAULT))
    except Exception:
        rec["evaluate_jBS_version"] = None
    rec["oft_build"] = oft_build_info()
    # ---- opt-in gate on the closure-half residual (jbs_gate_current_residual)
    # Everything below is added ONLY when the gate is on, so a default record
    # is key-for-key and value-for-value what it was before the flag existed.
    gate_cur = bool(settings.get("gate_current_residual", False))
    if gate_cur:
        rec["criteria"]["current_residual"] = True
        rec["current_gate"] = dict(
            active=True, tolerance=float(settings["rtol_j"]),
            tolerance_setting="jbs_rtol_j",
            source=("the pass's CurrentRelaxer (jc_k and js_k-1)"
                    if takes_relax else
                    "the step's meas['j_solved'] (beta not applied)"),
            definition=CURRENT_GATE_DEFINITION)
        rec["current_residual"] = []
        rec["current_residual_estimate"] = []
        rec["current_residual_estimate_definition"] = (
            "current_gap / (1 - beta) on a blended pass (None otherwise): "
            "the estimate the direct current_residual is checked against")
        rec["current_residual_direct_over_estimate"] = []
        rec["current_residual_ok"] = []
        rec["current_residual_unrelaxed_definition"] = (
            "||jc_k - js_{k-1}||_w / ||jc_k||_w = current_gap / (1 - beta) "
            "on a blended pass: the unrelaxed closure-half residual; with "
            "jbs_gate_current_residual=True the same quantity is GATED as "
            "current_residual")
    _js_prev_step = None                # j_solved of the previous pass

    def _nonfinite(k, what, arr, grid):
        """Raise :class:`JBSNonFinite` at the first non-finite node of
        *arr* (pass ``k + 1``; ``k = -1`` is the initial guess)."""
        bad = ~np.isfinite(arr)
        i = int(np.argmax(bad))
        psi = None
        if grid is not None and np.shape(grid) == np.shape(arr):
            psi = float(np.asarray(grid, dtype=float)[i])
        where = (f"index {i}" + ("" if psi is None else f", psi_N={psi:.6g}")
                 + f"; {int(bad.sum())} of {bad.size} node(s)")
        when = "before pass 1" if k < 0 else f"on pass {k + 1}/{K}"
        msg = (f"self-consistent j_BS loop"
               f"{(' [' + label + ']') if label else ''}: {what} is "
               f"non-finite {when} ({where}) -- refusing to blend it into "
               "the next iterate or hand it to a GS solve")
        rec["stop_reason"] = msg
        rec["wall_s"] = float(time.perf_counter() - t0)
        rec["jbs_converged"] = False
        rec["fail_message"] = msg
        if q0_pin is not None:
            rec["q0_pin"] = q0_pin.record()
        if extra is not None:
            rec["extra_criteria"] = extra.record()
        print("  [jbs-loop] " + msg, flush=True)
        raise JBSNonFinite(msg, rec, pass_number=max(k + 1, 0), index=i,
                           psi_N=psi)

    jbs = np.asarray(jbs0, dtype=float).copy()
    if not np.all(np.isfinite(jbs)):
        _nonfinite(-1, "the initial bootstrap guess jbs0", jbs,
                   (meas0 or {}).get("x"))
    jbs_solved_base = None            # bootstrap of the last relaxed solve
    jbs_solved = jbs
    prev = dict(meas0 or {})
    streak = 0
    growth_at_floor = 0
    grow_streak = 0                   # consecutive passes on which r_j grew
    omega_used_for_current = None     # omega that produced `jbs` (None: init)
    r_j_prev = None
    meas = None
    J = None
    jbs_used = jbs

    for k in range(K):
        jbs_used = jbs
        if relaxer is not None:
            relaxer.begin_pass()
            meas = step(jbs, k, relax=relaxer)
        else:
            meas = step(jbs, k)
        J = np.asarray(evaluate(meas), dtype=float)
        if J.shape != jbs.shape:
            raise ValueError(f"run_jbs_loop[{label}]: evaluate returned shape "
                             f"{J.shape}, the iterate has {jbs.shape}")
        if not np.all(np.isfinite(J)):
            rec["n_passes"] = k + 1
            _nonfinite(k, "the evaluated bootstrap J", J, meas.get("x"))
        # the bootstrap this pass SOLVED: the relaxer's blend applied to
        # the iterates (the composition is affine in the bootstrap), the
        # iterate itself when the pass did not blend
        _relaxed_now = bool(relaxer is not None and relaxer.called)
        _bs_blended = bool(_relaxed_now and relaxer.last_blended
                           and jbs_solved_base is not None
                           and jbs_solved_base.shape == jbs.shape)
        jbs_solved = ((1.0 - beta) * jbs_solved_base + beta * jbs
                      if _bs_blended else jbs)
        res_it = profile_residuals(J, jbs, meas["w"], meas["x"], Ip)
        # THE residual (gated, recorded as r_j / r_I): Redl on the solved
        # state against the bootstrap that state was solved with
        res = (profile_residuals(J, jbs_solved, meas["w"], meas["x"], Ip)
               if _bs_blended else res_it)
        li_new = _finite_or_none(meas.get("li"))
        q0_new = _finite_or_none(meas.get("q0"))
        li_old = _finite_or_none(prev.get("li"))
        q0_old = _finite_or_none(prev.get("q0"))
        dl_i = (abs(li_new - li_old) if (li_new is not None
                                         and li_old is not None) else None)
        dq0 = (abs(q0_new - q0_old) if (q0_new is not None
                                        and q0_old is not None) else None)
        ok = (np.isfinite(res["r_j"]) and res["r_j"] <= settings["rtol_j"]
              and np.isfinite(res["r_I"])
              and res["r_I"] <= settings["rtol_Ip"])
        if gate_li:
            ok = ok and (dl_i is not None and dl_i <= settings["tol_li"])
        if gate_q0:
            ok = ok and (dq0 is not None and dq0 <= settings["tol_q0"])
        q0_res = None
        if q0_pin is not None:
            # the TRUE residual against the target, not only the step:
            # an added condition, never a replacement
            q0_res = q0_pin.observe(q0_new, meas.get("axis_current_solved"))
            ok = ok and q0_pin.within_tol(q0_new)
        _x_never, _x_txt = None, ""
        if extra is not None:
            # the caller's own rows, on the pass's solved equilibrium: an
            # added condition, never a replacement
            _x_ok, _x_never, _x_txt = extra.observe(k, meas)
            ok = ok and bool(_x_ok)
        ok = bool(ok)
        if relaxer is not None:
            _gap, _bl = relaxer.gap(meas.get("w"), meas.get("x"))
            _unrel = relaxer.unrelaxed_residual(meas.get("w"), meas.get("x"))
            relaxer.commit()
            if _relaxed_now:
                # the blend base of the next pass, as the relaxer's own
                jbs_solved_base = np.asarray(jbs_solved, dtype=float).copy()
        else:
            _gap, _bl, _unrel = None, False, None
        _cres = None
        if gate_cur:
            if relaxer is not None:
                _cres = _unrel
                _est = ((_gap / (1.0 - beta)) if (_bl and _gap is not None
                                                  and beta < 1.0) else None)
            else:
                _js_now = meas.get("j_solved")
                if _js_now is not None:
                    _js_now = np.asarray(_js_now, dtype=float).copy()
                    if _js_prev_step is not None:
                        _cres = _relative_difference(
                            _js_now, _js_prev_step, meas.get("w"),
                            meas.get("x"))
                _js_prev_step = _js_now
                _est = None
            _ratio = (float(_cres / _est) if (_cres is not None
                                              and _est is not None
                                              and _est > 0.0) else None)
            _cok = bool(_cres is not None and np.isfinite(_cres)
                        and _cres <= settings["rtol_j"])
            ok = bool(ok and _cok)
            rec["current_residual"].append(_cres)
            rec["current_residual_estimate"].append(_est)
            rec["current_residual_direct_over_estimate"].append(_ratio)
            rec["current_residual_ok"].append(_cok)
        rec["omega"].append(None if omega_used_for_current is None
                            else float(omega_used_for_current))
        rec["r_j"].append(res["r_j"])
        rec["r_I"].append(res["r_I"])
        rec["dl_i"].append(dl_i)
        rec["dq0"].append(dq0)
        rec["I_BS"].append(res["I_BS"])
        rec["I_BS_used"].append(res["I_BS_used"])
        rec["jBS_peak"].append(res["jBS_peak"])
        rec["jBS_peak_psiN"].append(res["jBS_peak_psiN"])
        rec["li"].append(li_new)
        rec["q0"].append(q0_new)
        rec["pass_ok"].append(ok)
        rec["current_gap"].append(_gap)
        rec["current_blended"].append(bool(_bl))
        rec["current_residual_unrelaxed"].append(_unrel)
        if track_solved:
            rec["r_j_iterate"].append(res_it["r_j"])
            rec["r_I_iterate"].append(res_it["r_I"])
            rec["I_BS_used_iterate"].append(res_it["I_BS_used"])
            rec["bootstrap_blended"].append(_bs_blended)
        rec["n_passes"] = k + 1
        if (start_refresh is not None and k == 1
                and rec["bootstrap_refresh"]["applied"]):
            # the refreshed iterate judged on the next solved state
            rec["bootstrap_refresh"].update(
                r_j_after=float(res_it["r_j"]),
                r_I_after=float(res_it["r_I"]))
        entry = dict(k=k, ok=ok, dl_i=dl_i, dq0=dq0,
                     omega_used=omega_used_for_current, **res)
        print(f"  [jbs-loop{(' ' + label) if label else ''}] pass {k + 1}/{K}: "
              f"r_j={res['r_j']:.3e} (tol {settings['rtol_j']:.0e}) "
              f"r_I={res['r_I']:.3e} (tol {settings['rtol_Ip']:.0e})"
              + ("" if dl_i is None else f" dl_i={dl_i:.2e}")
              + ("" if dq0 is None else f" dq0={dq0:.2e}")
              + ("" if q0_pin is None else
                 (" q0-q0_target=n/a" if q0_res is None else
                  f" q0-q0_target={q0_res:+.2e} (q0_tol {q0_pin.q0_tol:g})"))
              + ("" if (extra is None or not _x_txt) else " " + str(_x_txt))
              + f" I_BS={res['I_BS'] / 1e3:.2f} kA"
              + ("" if omega_used_for_current is None
                 else f" omega={omega_used_for_current:.3f}")
              + ("" if not _bs_blended else
                 f" (vs iterate r_j={res_it['r_j']:.3e} r_I="
                 f"{res_it['r_I']:.3e}, record only)")
              + ("" if not _bl else f" |js-jc|/|jc|={_gap:.2e}")
              + ("" if (not _bl or _unrel is None or gate_cur)
                 else f" (unrelaxed {_unrel:.2e}, record only)")
              + ("" if not gate_cur else
                 (" cur_res=n/a (gated)" if _cres is None else
                  f" cur_res={_cres:.2e} (gated, tol "
                  f"{settings['rtol_j']:.0e})"))
              + (" ok" if ok else ""), flush=True)
        # ---- a pass that can never count: stop now, not at the ceiling ----
        _never = None
        if gate_li and li_new is None:
            _never = ("the step returned no finite l_i although l_i is a "
                      "convergence criterion here (gate_li=True)")
        elif gate_q0 and q0_new is None:
            _never = ("the step returned no finite q0 although q0 is a "
                      "convergence criterion here (gate_q0=True)")
        elif q0_pin is not None and q0_new is None:
            _never = ("the step returned no finite q0 although the q0 pin "
                      "|q0 - q0_target| <= q0_tol is a convergence "
                      "criterion here (jbs_loop_q0_corrector)")
        elif extra is not None and _x_never is not None:
            _never = str(_x_never)
        elif (not np.any(J != 0.0)) and np.any(jbs != 0.0):
            _never = ("the evaluated bootstrap J is identically zero while "
                      "the iterate is not (r_j is infinite)")
        elif gate_cur and _cres is None and (
                k >= 1 or (relaxer is None
                           and meas.get("j_solved") is None)):
            _never = ("the closure-half current residual cannot be "
                      "evaluated on this step although it is a convergence "
                      "criterion here (jbs_gate_current_residual=True): "
                      + ("the step did not pass its current through the "
                         "relaxer" if relaxer is not None else
                         "the step takes no relaxer and returned no "
                         "meas['j_solved']"))
        if _never is not None:
            rec["stop_reason"] = (f"pass {k + 1}/{K}: {_never} -- "
                                  "convergence is impossible, stopped at once "
                                  "instead of running to the pass ceiling")
            break
        streak = streak + 1 if ok else 0
        if streak >= need:
            rec["converged"] = True
            rec["stop_reason"] = (f"all active criteria met on {need} "
                                  "consecutive passes")
            if on_pass is not None:
                on_pass(k, meas, J, dict(entry, omega_next=None))
            break
        # ---- relaxation schedule: halve on SUSTAINED growth (halve_on
        # consecutive growing passes; 1 = every growth, the earlier
        # schedule), abort after n_abort growing passes at the floor --------
        # (the schedule is a PATH heuristic: it follows the iterate
        # residual, as before the solved residual was gated)
        if r_j_prev is not None and res_it["r_j"] > r_j_prev:
            if (omega_used_for_current is not None
                    and omega_used_for_current <= floor + 1e-15):
                growth_at_floor += 1
            else:
                growth_at_floor = 0
            grow_streak += 1
            if grow_streak >= halve_on:
                _om = max(0.5 * omega, floor)
                if _om < omega:
                    rec["omega_halved_at_pass"].append(k + 1)
                omega = _om
                grow_streak = 0
        else:
            growth_at_floor = 0
            grow_streak = 0
        r_j_prev = res_it["r_j"]
        stop = (growth_at_floor >= n_abort) or (k == K - 1)
        if q0_pin is not None and not stop:
            # a further pass follows: move the axis row from the q0 this
            # pass measured (once per pass; never after the last one)
            q0_pin.advance()
        _restart = bool(start_refresh is not None and k == 0 and not stop)
        if on_pass is not None:
            # omega_next: the relaxation the NEXT iterate is built with (the
            # caller relaxes any per-pass update of its own -- e.g. a moved
            # constraint row -- by the same factor); 1.0 before a restart
            on_pass(k, meas, J, dict(entry, omega_next=(
                None if stop else (1.0 if _restart else omega))))
        if growth_at_floor >= n_abort:
            rec["stop_reason"] = (f"r_j grew on {n_abort} consecutive passes "
                                  f"at the relaxation floor omega={floor:g}")
            break
        if k == K - 1:
            break                       # no further solve: keep jbs as used
        if _restart:
            _new = np.asarray(start_refresh(J, meas), dtype=float)
            if _new.shape != jbs.shape:
                raise ValueError(f"run_jbs_loop[{label}]: start_refresh "
                                 f"returned shape {_new.shape}, the iterate "
                                 f"has {jbs.shape}")
            if not np.all(np.isfinite(_new)):
                _nonfinite(k, "the refreshed bootstrap (start_refresh)",
                           _new, meas.get("x"))
            _step_rel = profile_residuals(_new, jbs, meas["w"], meas["x"], Ip)
            _blend = (1.0 - omega) * jbs + omega * J
            _vs_blend = profile_residuals(_new, _blend, meas["w"], meas["x"],
                                          Ip)
            rec["bootstrap_refresh"].update(
                applied=True, r_j_before=float(res_it["r_j"]),
                r_I_before=float(res_it["r_I"]),
                I_BS_start=float(res_it["I_BS_used"]),
                I_BS_evaluated=float(res_it["I_BS"]),
                I_BS_refreshed=float(_step_rel["I_BS"]),
                refresh_vs_start=dict(r_j=float(_step_rel["r_j"]),
                                      r_I=float(_step_rel["r_I"])),
                refresh_vs_blend=dict(r_j=float(_vs_blend["r_j"]),
                                      r_I=float(_vs_blend["r_I"]),
                                      omega=float(omega)),
                extra_solves=0)
            print(f"  [jbs-loop{(' ' + label) if label else ''}] bootstrap "
                  f"refreshed after pass 1 (restart, not a blend): I_BS "
                  f"{res_it['I_BS_used'] / 1e3:.2f} -> "
                  f"{_step_rel['I_BS'] / 1e3:.2f} kA, step r_j="
                  f"{_step_rel['r_j']:.3e}; 0 extra solves", flush=True)
            jbs = _new
            omega_used_for_current = None
        else:
            jbs = (1.0 - omega) * jbs + omega * J
            omega_used_for_current = omega
        prev = meas

    rec["wall_s"] = float(time.perf_counter() - t0)
    rec["jbs_converged"] = bool(rec["converged"])
    if not rec["converged"] and rec["stop_reason"] is None:
        rec["stop_reason"] = (f"pass ceiling {K} reached without {need} "
                              "consecutive passing passes")
    rec["final"] = dict(
        r_j=rec["r_j"][-1] if rec["r_j"] else None,
        r_I=rec["r_I"][-1] if rec["r_I"] else None,
        dl_i=rec["dl_i"][-1] if rec["dl_i"] else None,
        dq0=rec["dq0"][-1] if rec["dq0"] else None,
        I_BS=rec["I_BS"][-1] if rec["I_BS"] else None,
        current_gap=rec["current_gap"][-1] if rec["current_gap"] else None,
        current_residual_unrelaxed=(rec["current_residual_unrelaxed"][-1]
                                    if rec["current_residual_unrelaxed"]
                                    else None))
    if track_solved:
        rec["final"]["r_j_iterate"] = (rec["r_j_iterate"][-1]
                                       if rec["r_j_iterate"] else None)
        rec["final"]["r_I_iterate"] = (rec["r_I_iterate"][-1]
                                       if rec["r_I_iterate"] else None)
    if q0_pin is not None:
        rec["q0_pin"] = q0_pin.record()
        rec["final"]["q0_residual"] = rec["q0_pin"]["final_q0_residual"]
        rec["final"]["q0_residual_over_tol"] = \
            rec["q0_pin"]["final_q0_residual_over_tol"]
    if extra is not None:
        rec["extra_criteria"] = extra.record()
    if gate_cur:
        rec["final"]["current_residual"] = (rec["current_residual"][-1]
                                            if rec["current_residual"]
                                            else None)
        rec["criteria_timeline"] = criteria_timeline(rec)
        rec["last_criterion_met"] = rec["criteria_timeline"]["last_met"]
    out = dict(jbs_used=np.asarray(jbs_used, dtype=float),
               jbs_solved=np.asarray(jbs_solved, dtype=float),
               J_final=(None if J is None else np.asarray(J, dtype=float)),
               meas_final=meas, converged=bool(rec["converged"]), record=rec)
    if not rec["converged"]:
        msg = (f"self-consistent j_BS loop{(' [' + label + ']') if label else ''}"
               f" did not converge: {rec['stop_reason']}; residual history "
               f"r_j={_fmt_hist(rec['r_j'])} r_I={_fmt_hist(rec['r_I'])}"
               + ("" if not gate_li else f" dl_i={_fmt_hist(rec['dl_i'])}")
               + ("" if not gate_q0 else f" dq0={_fmt_hist(rec['dq0'])}")
               + ("" if q0_pin is None else
                  f" q0-q0_target={_fmt_hist(rec['q0_pin']['q0_residual'])}"
                  f" (q0_tol {q0_pin.q0_tol:g})")
               + ("" if not gate_cur else
                  f" current_residual={_fmt_hist(rec['current_residual'])}")
               + ("" if extra is None else " " + str(extra.history_text())))
        rec["fail_message"] = msg
        print("  [jbs-loop] " + msg, flush=True)
        if raise_on_fail:
            raise JBSNotConverged(msg, rec)
    return out


def _fmt_hist(vals):
    return "[" + ", ".join("n/a" if v is None else f"{float(v):.2e}"
                           for v in vals) + "]"


def flag_reason(record: dict) -> str:
    """The closure_limited reason a ``"flag"``-mode caller records (with the
    final q0 residual against ``q0_tol`` when the q0 pin acted)."""
    _pin = record.get("q0_pin")
    _q0 = ""
    if isinstance(_pin, dict):
        _q0 = (", q0-q0_target="
               + _fmt_one(_pin.get("final_q0_residual"))
               + " (q0_tol " + _fmt_one(_pin.get("q0_tol")) + ")")
    return (JBS_FLAG_PREFIX + "did not converge ("
            + str(record.get("stop_reason")) + "; final r_j="
            + _fmt_one((record.get("final") or {}).get("r_j")) + ", r_I="
            + _fmt_one((record.get("final") or {}).get("r_I")) + _q0 + ")")


def _fmt_one(v):
    return "n/a" if v is None else f"{float(v):.2e}"


def check_delivered(J, jbs_used, w, x, Ip, settings: dict) -> dict:
    """One residual check of an equilibrium that was moved after the loop
    converged (the draw's post-perturb homotopy): does the bootstrap it carries
    still match its own Redl bootstrap to the loop's tolerances?"""
    res = profile_residuals(J, jbs_used, w, x, Ip)
    res["ok"] = bool(np.isfinite(res["r_j"])
                     and res["r_j"] <= settings["rtol_j"]
                     and np.isfinite(res["r_I"])
                     and res["r_I"] <= settings["rtol_Ip"])
    return res


def jsonable(obj):
    """Deep-convert a loop record to JSON/HDF5-attr-safe builtins."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
