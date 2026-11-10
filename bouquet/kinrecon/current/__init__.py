"""Current model: 1-D flux-surface-averaged poloidal-flux diffusion.

* :mod:`.geometry` -- :class:`FluxSurfaceGeometry`, the metric from equilibrium
  geometry (the only OpenFUSIONToolkit contact is :func:`from_tokamaker`, which
  calls methods of a TokaMaker object and imports nothing);
* :mod:`.conductivity` -- Spitzer and neoclassical (Redl 2021 / Sauter 1999)
  parallel conductivity;
* :mod:`.diffusion` -- :class:`CurrentDiffusion` (steady-state and
  time-dependent modes, Ip or loop-voltage boundary condition) and the
  serializable :class:`CurrentState`.

The formulation follows B. C. Lyons' QED.jl notes (``equations_readme.pdf``,
General Atomics, Apache-2.0) and S. C. Jardin, *Computational Methods in
Plasma Physics* (2010); see :mod:`.diffusion` for the equations and credits.
No equilibrium solver is required.  No sawtooth model is implemented here;
:meth:`CurrentDiffusion.advance` exposes the hooks for one.
"""

from .conductivity import (coulomb_log_e, f33, f33_effective_trapped_fraction, neoclassical_conductivity,
                           nu_e_star, spitzer_conductivity, trapped_fraction)
from .diffusion import (CurrentDiffusion, CurrentState, GeometrySchedule, IpBoundary, VloopBoundary,
                        as_profile_function, as_time_function)
from .geometry import MU0, FluxSurfaceGeometry, from_tokamaker

__all__ = [
    "MU0",
    "FluxSurfaceGeometry",
    "from_tokamaker",
    "CurrentDiffusion",
    "CurrentState",
    "GeometrySchedule",
    "IpBoundary",
    "VloopBoundary",
    "as_profile_function",
    "as_time_function",
    "coulomb_log_e",
    "spitzer_conductivity",
    "nu_e_star",
    "f33_effective_trapped_fraction",
    "f33",
    "neoclassical_conductivity",
    "trapped_fraction",
]
