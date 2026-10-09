"""The run's radial coordinate, at the OpenFUSIONToolkit boundary.

A run holds every profile and envelope on one grid ``x`` in one coordinate:
``"psi_n"`` (normalised poloidal flux, the default) or ``"phi_n"``
(normalised toroidal flux).  Profiles cross into TokaMaker as-is, tagged
with their coordinate; ψ_N appears in bouquet only as the address at which
a readback is sampled, and in a Φ_N run that address comes from the solver's
own map (:func:`psi_of`).  A ``"psi_n"`` run never sends a ``coord`` key or
argument, so its calls are those of an OFT without toroidal-flux support.
"""

from __future__ import annotations

import numpy as np

PSI, PHI = "psi_n", "phi_n"
COORDS = (PSI, PHI)
#: Input spelling accepted at io: ρ_tor, converted exactly to Φ_N = ρ².
RHO = "rho_tor"
#: Coordinates a user-supplied grid may be given in: the run's, or ψ_N.
RUN = "run"
INPUT_COORDS = (RUN, PSI)


def check_coord(coord):
    """Return ``coord`` if it is a run coordinate, else raise."""
    if coord not in COORDS:
        raise ValueError(f"coord must be one of {COORDS}, got {coord!r}")
    return coord


def run_coord(coord):
    """The run coordinate of a source ``coord`` (``"rho_tor"`` runs as Φ_N = ρ²)."""
    if coord == RHO:
        return PHI
    if coord not in COORDS:
        raise ValueError(f"source coord must be one of {COORDS + (RHO,)}, got {coord!r}")
    return coord


def to_run_grid(x, x_coord=RUN, psi_map=None):
    """A user-supplied grid ``x`` (given in ``x_coord``) in the run coordinate.

    ``"run"`` passes ``x`` through; ``"psi_n"`` maps it through ``psi_map``
    (``(psi_N, x_run)`` pairs; ``None`` in a ψ_N run, where it is the identity).
    """
    if x_coord not in INPUT_COORDS:
        raise ValueError(f"input coord must be one of {INPUT_COORDS}, got {x_coord!r}")
    if x is None or x_coord == RUN or psi_map is None:
        return x
    return np.interp(np.asarray(x, dtype=float), *psi_map)


def phi_n_from_q(psi_N, q, bracket=False):
    """``(inside, Φ_N)``: the nodes with ψ_N ≤ 1 and their Φ_N from ``q``.

    Φ_N(ψ_N) = ∫₀^ψ_N q dψ_N / ∫₀¹ q dψ_N (trapezoid; ``q`` is interpolated
    to ψ_N = 1 when the grid does not hit it).  For a source that tabulates
    its profiles on ψ_N with its own equilibrium's q (IDA).

    ``bracket``: when the grid does not hit ψ_N = 1, also select the first
    node past it (Φ_N > 1, same trapezoid), so profiles interpolated in Φ_N
    reach Φ_N = 1 between data instead of clamping (ida_fuse's ida_phi_n).
    """
    from scipy.integrate import cumulative_trapezoid
    psi_N = np.asarray(psi_N, dtype=float)
    q = np.abs(np.asarray(q, dtype=float))
    inside = psi_N <= 1.0
    x = psi_N[inside]
    if x.size < 2 or not np.all(np.diff(x) > 0) or abs(x[0]) > 1e-9:
        raise ValueError("phi_n_from_q: psi_N must rise from 0")
    if not np.all(np.isfinite(q[inside])):
        raise ValueError("phi_n_from_q: q is not finite inside the LCFS")
    xe = x if x[-1] == 1.0 else np.append(x, 1.0)
    phi = cumulative_trapezoid(np.interp(xe, psi_N, q), xe, initial=0.0)
    out = phi[:x.size] / phi[-1]
    if bracket and x[-1] < 1.0 and not inside.all():
        b = int(np.argmin(inside))                  # first node past ψ_N = 1
        if np.isfinite(q[b]):
            q_last = q[inside][-1]
            out = np.append(out, (phi[x.size - 1] + 0.5 * (q_last + q[b])
                                  * (psi_N[b] - x[-1])) / phi[-1])
            inside = inside.copy()
            inside[b] = True
    return inside, out


def gfile_run_grids(eqdsk, kin, profiles_path, coord=PSI):
    """``(x_run, x_kin, inside, kin)``: a g-file source's nodes in ``coord``.

    In a Φ_N run the g-file's nodes are labelled ``rhovn**2`` and the kinetic
    nodes inside the LCFS by the kinetic source's own ``q`` (an IDA file),
    else through the g-file's map (a p-file); ``kin`` (``"q"`` dropped) is
    restricted to ``inside``.  In a ψ_N run: both sources' own ψ_N,
    ``inside`` is ``None``.
    """
    psi_N = np.asarray(eqdsk.psi_N, dtype=float)
    kin = dict(kin)
    q = kin.pop("q", None)
    if coord != PHI:
        return psi_N, kin["psi_N"], None, kin
    x_run = np.asarray(eqdsk.rhovn, dtype=float) ** 2
    if x_run.shape != psi_N.shape:
        raise ValueError("g-file rhovn is not on the psi_N levels")
    if q is None and str(profiles_path).endswith(".cdf"):
        raise ValueError(f"coord='phi_n': {profiles_path!r} carries no q, so "
                         "its profiles cannot be placed in Phi_N")
    psi_kin = np.asarray(kin["psi_N"], dtype=float)
    if q is not None:
        inside, x_kin = phi_n_from_q(psi_kin, q)
    else:
        inside = psi_kin <= 1.0
        x_kin = np.interp(psi_kin[inside], psi_N, x_run)
    kin = {k: (np.asarray(v)[inside] if np.shape(v) == psi_kin.shape else v)
           for k, v in kin.items()}
    return x_run, x_kin, inside, kin


def oft_prof(kind, x, y, coord=PSI):
    """A TokaMaker profile dict on the run grid.

    In a Φ_N run a ``jphi-*`` profile holds values (tag ``phi_n_relabel``)
    and any other profile holds a derivative, y = dY/dΦ_N (tag ``phi_n``).
    """
    return tag_prof({"type": kind, "y": np.array(y, dtype=float), "x": x},
                    coord)


def tag_prof(d, coord=PSI):
    """Profile dict ``d`` (on the run grid) tagged as :func:`oft_prof` tags
    it; ``d`` itself in a ψ_N run (e.g. a ``bouquet.edge_pressure`` P')."""
    if check_coord(coord) == PHI:
        d = dict(d)
        d["coord"] = "phi_n_relabel" if d["type"].startswith("jphi") else PHI
    return d


def psi_at(mygs, x, coord=PSI):
    """ψ_N of the run-grid nodes ``x`` on ``mygs``'s last solve.

    ``x`` itself in a ψ_N run; the solver's toroidal-flux map in a Φ_N run.
    This is the abscissa for anything that integrates or samples in ψ
    (``flux_integral``, the FSA current measure, ``find_optimal_scale``).
    """
    if check_coord(coord) == PSI:
        return x
    x = np.asarray(x, dtype=float)
    return np.asarray(mygs.get_torflux_map(x.copy(), inverse=True)[0], dtype=float)


def psi_of(mygs, x, coord=PSI, psi_pad=1e-3):
    """:func:`psi_at`, clipped to ``[psi_pad, 1 - psi_pad]``: where to sample
    ``get_profiles``/``get_q``/``sauter_fc`` for the run grid ``x``.
    """
    return np.clip(np.asarray(psi_at(mygs, x, coord), dtype=float),
                   psi_pad, 1.0 - psi_pad)


def readback_kw(mygs, x, coord=PSI, psi_pad=1e-3):
    """``get_profiles`` / ``get_q`` sampling keywords for the run grid ``x``.

    A ψ_N run on a uniform grid ``linspace(0, 1, n)`` (a g-file's) keeps the
    solver's own uniform sampling ``npsi=n, psi_pad``.  A KNOWN ERROR kept for
    legacy bit-identity: callers pair those padded samples with the nodes
    index for index, a misplacement of up to ``psi_pad``
    (docs/physics-notes.md, Corrective j_phi iteration).  Any other grid (a
    non-uniform IMAS grid, a Φ_N run) is sampled at its nodes' ψ_N
    (:func:`psi_of`), where the uniform sampling would misplace them.
    """
    x = np.asarray(x, dtype=float)
    if (check_coord(coord) == PSI and x.size > 1
            and np.allclose(x, np.linspace(0.0, 1.0, x.size), rtol=0.0,
                            atol=1e-12)):
        return dict(npsi=int(x.size), psi_pad=psi_pad)
    return dict(psi=psi_of(mygs, x, coord, psi_pad))


def _swb_params():
    """Argument names of the installed ``solve_with_bootstrap`` (cached)."""
    global _SWB_PARAMS
    try:
        return _SWB_PARAMS
    except NameError:
        pass
    try:
        import inspect
        from OpenFUSIONToolkit.TokaMaker.bootstrap import solve_with_bootstrap
        _SWB_PARAMS = frozenset(inspect.signature(solve_with_bootstrap).parameters)
    except Exception:
        _SWB_PARAMS = frozenset()
    return _SWB_PARAMS


def _swb_grid_arg():
    """``solve_with_bootstrap``'s grid keyword: ``"x"``, the ψ_N-only
    ``"psi_N"`` of older toolkits, or ``None`` (a uniform grid is assumed).
    """
    p = _swb_params()
    return "x" if "x" in p else ("psi_N" if "psi_N" in p else None)


def swb_grid(x):
    """The grid ``solve_with_bootstrap`` places its arrays on.

    ``x`` where the toolkit takes a grid argument; otherwise the uniform grid
    it assumes, which is also the grid of its outputs.
    """
    x = np.asarray(x, dtype=float)
    return x if _swb_grid_arg() else np.linspace(0.0, 1.0, x.size)


def swb_grid_kwargs(x, coord=PSI):
    """Grid arguments for ``solve_with_bootstrap``: the grid (:func:`_swb_grid_arg`),
    plus ``coord`` in a Φ_N run.  Empty on a toolkit without a grid argument
    (ψ_N runs only).
    """
    kw = {}
    arg = _swb_grid_arg()
    if arg:
        kw[arg] = np.asarray(x, dtype=float)
    if check_coord(coord) == PHI:
        kw["coord"] = PHI
    return kw


def swb_seed(x, psi=None):
    """Inductive seed ``(1 - s^1.5)^1.5`` (OFT's ``create_power_flux_fun(n,
    1.5, 1.5)``) at the nodes of :func:`swb_grid`, with ``s`` their ψ_N.

    ``psi`` is the ψ_N of the nodes ``x`` (:func:`psi_at`): the shape is then
    the ψ_N one in any run; ``None`` writes it in the run coordinate.
    """
    s = swb_grid(x)
    if psi is not None and _swb_grid_arg():
        s = np.asarray(psi, dtype=float)
    return np.power(1.0 - np.power(s, 1.5), 1.5)


def check_backend(coord):
    """Raise unless the installed toolkit supports ``coord``.

    A Φ_N run needs ``solve_bootstrap(coord=)``, ``get_torflux_map`` and
    ``solve_with_bootstrap(x, coord)``: an older toolkit ignores a ``coord``
    key in a profile dict, so it would solve on ψ_N without complaint.
    """
    if check_coord(coord) == PSI:
        return
    try:
        import inspect
        from OpenFUSIONToolkit.TokaMaker._core import TokaMaker
        ok = ("coord" in inspect.signature(TokaMaker.solve_bootstrap).parameters
              and hasattr(TokaMaker, "get_torflux_map")
              and _swb_grid_arg() is not None
              and "coord" in _swb_params())
    except Exception:
        ok = False
    if not ok:
        raise RuntimeError(
            "coord='phi_n' needs an OpenFUSIONToolkit with toroidal-flux profile "
            "support (solve_bootstrap(coord=), TokaMaker.get_torflux_map and "
            "solve_with_bootstrap(x, coord)).")


def check_run(config, coord=None):
    """Refuse, before any solve, a run coordinate this setup cannot run.

    ``coord`` defaults to the source's.  In a Φ_N run, checks the toolkit
    (:func:`check_backend`) and that the internal bootstrap solve is used.
    Returns the run coordinate.
    """
    if coord is None:
        coord = run_coord(getattr(config.source, "coord", PSI))
    if check_coord(coord) == PHI:
        check_backend(coord)
        if (getattr(config.generation, "bootstrap_kwargs", None)
                or {}).get("use_python_solve"):
            raise ValueError("coord='phi_n' needs the internal bootstrap "
                             "solve: drop use_python_solve from bootstrap_kwargs")
    return coord
