r"""Non-inductive sources for the kinetic reconstruction: driven current and fast ions.

Every provider returns :class:`.provider.NonInductiveSources` -- plain arrays on a stated
rho_tor_norm grid:

* ``jdotb[name]`` -- ``<J_CD . B>`` [A T/m^2], signed relative to Ip (what
  :class:`..current.diffusion.CurrentDiffusion` takes as ``j_ni``); ``j_parallel()`` gives
  ``<J.B>/B0`` and ``toroidal(geometry)`` the toroidal frames and enclosed current;
* ``p_fast[species]`` -- the isotropic-equivalent fast-ion pressure
  ``(P_par + 2 P_perp)/3 = (2/3) W_fast`` [Pa], the scalar that enters P' (true components in
  ``p_par``/``p_perp`` when known);
* ``n_fast[species]`` [m^-3] (dilution, bootstrap), ``heating`` (optional);
* ``uncertainty`` -- scale-factor and shape priors (never filled in silently;
  :func:`.priors.proposed_priors` holds the PROPOSED defaults with their evidence), and
  ``provenance`` -- producing code, settings, assumptions, limitations.

Providers:

=====================================  ==========================================================
:func:`.arrays.from_arrays`            any transport code's arrays with declared units/frames
:func:`.imas.from_imas`                IMAS ``core_sources`` / ``core_profiles`` (OMAS, FUSE, OMFIT)
:class:`.beam.SimpleBeamModel`         built-in NBI model (pencil beam or given deposition,
                                       classical slowing down, shielded fast-ion current)
:class:`.ec.SimpleECModel`             built-in ECCD estimate from a stated efficiency
=====================================  ==========================================================

Physics modules: :mod:`.slowing_down` (tau_s, E_c, distribution moments), :mod:`.nbcd`
(electron shielding), :mod:`.neutrons` (D-D rate estimates).  Conventions and every frame
conversion live in :mod:`.conventions`.  Bootstrap and ohmic current are NOT provided here
(``kinrecon`` computes its own); a provider's bootstrap is never used.
"""

from .conventions import (AmbiguousConventionError, enclosed_current, jdotb_from_jpar, jpar_from_jdotb,
                          p_fast_from_energy_density, p_fast_from_imas_components, p_fast_from_true, psi_n_to_rho,
                          rho_to_psi_n, toroidal_frames)
from .provider import SOURCES_VERSION, NonInductiveSources, Provenance, ScalePrior, ShapePrior
from .arrays import from_arrays
from .imas import from_imas, to_imas
from .slowing_down import IonSpecies, classical_moments
from .beam import NeutralBeam, SimpleBeamModel, current_fractions_from_power, power_fractions_from_current
from .ec import ECLauncher, SimpleECModel
from .priors import PRIOR_KINDS, proposed_priors

__all__ = [
    "AmbiguousConventionError", "NonInductiveSources", "Provenance", "ScalePrior", "ShapePrior", "SOURCES_VERSION",
    "from_arrays", "from_imas", "to_imas", "IonSpecies", "classical_moments", "NeutralBeam", "SimpleBeamModel",
    "power_fractions_from_current", "current_fractions_from_power", "ECLauncher", "SimpleECModel",
    "proposed_priors", "PRIOR_KINDS",
    "enclosed_current", "jdotb_from_jpar", "jpar_from_jdotb", "p_fast_from_energy_density",
    "p_fast_from_imas_components", "p_fast_from_true", "psi_n_to_rho", "rho_to_psi_n", "toroidal_frames",
]
