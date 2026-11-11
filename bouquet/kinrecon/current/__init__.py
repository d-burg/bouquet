"""Current model: 1-D flux-surface-averaged poloidal-flux diffusion.

* :mod:`.geometry` -- :class:`FluxSurfaceGeometry`, the metric from equilibrium
  geometry (the only OpenFUSIONToolkit contact is :func:`from_tokamaker`, which
  calls methods of a TokaMaker object and imports nothing);
* :mod:`.conductivity` -- Spitzer and neoclassical (Redl 2021 / Sauter 1999)
  parallel conductivity;
* :mod:`.diffusion` -- :class:`CurrentDiffusion` (steady-state and
  time-dependent modes, Ip or loop-voltage boundary condition), the
  serializable :class:`CurrentState` and the lumped :class:`OuterLayer`
  between the domain edge and the last closed surface.

Domain edge (this package's rule): on a diverted boundary the diffusion domain
ends at psi_N = 0.995 (0.99 on coarse grids or when geometry routes disagree;
never beyond 0.999; :func:`choose_psi_n_edge` returns the cutoff and the
reason), the layer outside is a lumped uniform-E_par element attached to the
last cell, and Ip applies at the last closed surface.

The formulation follows B. C. Lyons' QED.jl notes (``equations_readme.pdf``,
General Atomics, Apache-2.0) and S. C. Jardin, *Computational Methods in
Plasma Physics* (2010); see :mod:`.diffusion` for the equations and credits.
No equilibrium solver is required.  The sawtooth model (helical-flux
reconnection, periodic orbit and measured crash schedules) is
:mod:`.sawtooth`; it acts through the ``post_step`` hook of
:meth:`CurrentDiffusion.advance`.
"""

from .conductivity import (coulomb_log_e, f33, f33_effective_trapped_fraction, neoclassical_conductivity,
                           nu_e_star, spitzer_conductivity, trapped_fraction)
from .diffusion import (CurrentDiffusion, CurrentState, GeometrySchedule, IpBoundary, OuterLayer, PositiveFrameError,
                        VloopBoundary, as_profile_function, as_time_function)
from .geometry import (MU0, PSI_N_EDGE_DEFAULT, PSI_N_EDGE_FALLBACK, PSI_N_EDGE_MAX, EdgeChoice, FluxSurfaceGeometry,
                       OuterLayerGeometry, check_psi_samples, choose_psi_n_edge, from_tokamaker)
from . import sawtooth  # noqa: F401  (reconnection operator, periodic orbit, crash schedules: see sawtooth.py)

__all__ = [
    "MU0",
    "FluxSurfaceGeometry",
    "OuterLayerGeometry",
    "EdgeChoice",
    "choose_psi_n_edge",
    "check_psi_samples",
    "PSI_N_EDGE_DEFAULT",
    "PSI_N_EDGE_FALLBACK",
    "PSI_N_EDGE_MAX",
    "from_tokamaker",
    "OuterLayer",
    "CurrentDiffusion",
    "CurrentState",
    "GeometrySchedule",
    "IpBoundary",
    "VloopBoundary",
    "PositiveFrameError",
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
