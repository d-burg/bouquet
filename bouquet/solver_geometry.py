"""The per-baseline solver geometry: vacuum F0, LCFS boundary, isoflux and saddle constraints.

The mesh, regions, coil mode and VSC are fixed when the solver is set up; everything here belongs to one
baseline, so one solver can run many baselines (:meth:`bouquet.run.Bouquet.point_solver`).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

ISOFLUX_WEIGHT = 500.0

#: The F0 rule of a reused solver, stamped in every :meth:`CaseGeometry.record`: since #76 every
#: baseline (each ``run_slices`` / ``set_slice`` slice, each ``set_case`` case, every swb per-solve
#: reset) is solved at ITS OWN vacuum F0; before, a reused solver kept the first baseline's F0.
F0_RULE = "per-baseline: each baseline solves at its own F0 (since #76)"


@dataclass(frozen=True)
class CaseGeometry:
    F0: float
    boundary_RZ: Optional[np.ndarray]
    iso_pts: Optional[np.ndarray]
    iso_w: Optional[np.ndarray]
    saddle_pts: Optional[np.ndarray] = None
    saddle_w: Optional[np.ndarray] = None
    # provenance (record()): where F0, the boundary and the isoflux targets came from
    F0_source: Optional[float] = None
    F0_origin: str = "source"
    boundary_origin: Optional[str] = None
    iso_origin: str = "source boundary"

    def record(self) -> dict:
        """What every reset installs for this baseline, as plain data (archived as the ``_baseline``
        attr ``solver_geometry_json``): ``solver_F0`` and where it came from, the isoflux targets and
        the saddle constraints.  The reconstruction routes may install their own isoflux afterwards
        (recorded in ``Baseline.recon``); this is the reset state every solve starts from."""
        def _w(w, n):
            if w is None:
                return None
            w = np.atleast_1d(np.asarray(w, dtype=float))
            return {"n": int(n), "min": float(w.min()), "max": float(w.max())} if w.size else None
        n_iso = 0 if self.iso_pts is None else int(len(self.iso_pts))
        n_sad = 0 if self.saddle_pts is None else int(len(self.saddle_pts))
        return {
            "solver_F0": float(self.F0),
            "F0_source": None if self.F0_source is None else float(self.F0_source),
            "F0_origin": str(self.F0_origin),
            "F0_rule": F0_RULE,
            "boundary_origin": self.boundary_origin,
            "isoflux_origin": str(self.iso_origin) if n_iso else None,
            "isoflux_n": n_iso,
            "isoflux_weights": _w(self.iso_w, n_iso),
            "saddle_origin": "SolverConfig.saddle_targets" if n_sad else None,
            "saddle_n": n_sad,
            "saddle_points": None if not n_sad else np.asarray(self.saddle_pts, float).tolist(),
            "saddle_weights": None if self.saddle_w is None else
            np.atleast_1d(np.asarray(self.saddle_w, float)).tolist(),
        }

    def apply(self, mygs):
        """Install this geometry on ``mygs`` (after a reset to the clean equilibrium)."""
        mygs.set_profiles(foffset=float(self.F0))
        if self.iso_pts is not None:
            mygs.set_isoflux(self.iso_pts, weights=self.iso_w)
        if self.saddle_pts is not None:
            mygs.set_saddle_constraints(self.saddle_pts, weights=self.saddle_w)


def case_geometry(source, solver) -> CaseGeometry:
    """The :class:`CaseGeometry` of ``source`` (g-file: R_center*B_center and its boundary; IDS: |r0*b0(t)|
    and the boundary outline or ``LCFS_geqdsk``), with ``solver.F0`` / ``isoflux_pts`` / ``saddle_targets``
    overriding."""
    from .config import ImasSource, ReconstructionSource
    if isinstance(source, ReconstructionSource):
        from .io.geqdsk import read_geqdsk
        g = read_geqdsk(source.geqdsk_path, cocos=source.cocos)
        F0_src = abs(g.R_center * g.B_center)
        boundary_RZ = np.column_stack([g.boundary_R, g.boundary_Z])
        F0_from, b_from = "source: g-file |R_center*B_center|", "source: g-file boundary"
    elif isinstance(source, ImasSource):
        from .io.imas import read_imas_geometry
        F0_src, boundary_RZ = read_imas_geometry(source)
        F0_from = "source: IDS |r0*b0(t)|"
        b_from = ("source: LCFS_geqdsk boundary" if getattr(source, "LCFS_geqdsk", None)
                  else "source: IDS boundary.outline")
    else:
        raise TypeError(f"unknown baseline source type {type(source).__name__}")
    if F0_src is None and solver.F0 is None:
        raise ValueError(f"{type(source).__name__} gives no vacuum F0: set SolverConfig.F0")
    F0 = F0_src if solver.F0 is None else float(solver.F0)
    if solver.F0 is not None:
        F0_from = "SolverConfig.F0"
    if F0_src is not None and abs(F0 - F0_src) > 1e-3 * abs(F0_src):
        warnings.warn(f"SolverConfig.F0={F0:.4f} overrides the source's F0={F0_src:.4f}; the engine's IDS "
                      "contract still uses the source's vacuum field", UserWarning, stacklevel=2)
    iso_pts, iso_w = solver.isoflux_pts, solver.isoflux_weights
    iso_from = "SolverConfig.isoflux_pts"
    if iso_pts is None:
        iso_pts, iso_w = boundary_RZ, np.full(len(boundary_RZ), ISOFLUX_WEIGHT)
        iso_from = "source boundary"
    sad = sad_w = None
    if solver.saddle_targets is not None:
        sad = np.asarray(solver.saddle_targets, dtype=np.float64).reshape(-1, 2)
        sad_w = None if solver.saddle_weights is None else np.asarray(solver.saddle_weights, dtype=np.float64)
    return CaseGeometry(F0=F0, boundary_RZ=boundary_RZ, iso_pts=iso_pts, iso_w=iso_w,
                        saddle_pts=sad, saddle_w=sad_w, F0_source=F0_src, F0_origin=F0_from,
                        boundary_origin=b_from, iso_origin=iso_from)
