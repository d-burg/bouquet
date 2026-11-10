"""Shared data containers for the kinetic-reconstruction subpackage.

This subpackage is self-contained by design: it imports only the standard
library, numpy, scipy, h5py and the Open FUSION Toolkit (TokaMaker) Python
interface, and nothing from the rest of ``bouquet``, so that it can be moved
into TokaMaker's Python layer unchanged.  ``tests/test_kinrecon_isolation.py``
enforces that rule.

Three containers define the boundaries between the pieces:

* :class:`ConstraintGroup` / :class:`MeasurementSet` -- one time window of
  measurements for one device, named after the IMAS data dictionary.
* :class:`ResidualTable` -- measured vs. modeled values for any solved
  equilibrium, whether or not it was fit to the measurements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Canonical group names and the IMAS location each one is taken from.
GROUP_IDS_PATHS = {
    "flux_loop": "magnetics.flux_loop[:].flux",
    "bpol_probe": "magnetics.b_field_pol_probe[:].field",
    "ip": "magnetics.ip[0]",
    "diamagnetic_flux": "magnetics.diamagnetic_flux[0]",
    "pf_current": "pf_active.coil[:].current",
    "mse": "mse.channel[:].polarisation_angle",
    "thomson_ne": "thomson_scattering.channel[:].n_e",
    "thomson_te": "thomson_scattering.channel[:].t_e",
    "vloop": "magnetics.flux_loop[:].voltage",
    "q": "equilibrium.time_slice[:].constraints.q[:]",
}


@dataclass
class ConstraintGroup:
    """One kind of measurement, all channels, reduced to a single time window.

    ``geometry`` holds per-channel arrays whose keys depend on the group:
    ``r``, ``z`` for every local measurement; ``direction_r``, ``direction_z``
    for poloidal-field probes; ``coefficients`` (n, 9), in IMAS
    ``mse.channel[:].active_spatial_resolution[:].geometric_coefficients``
    order, for MSE; ``coil_names`` for coil currents.

    ``meta`` records how the numbers were made:

    * ``sigma_model`` -- name of the uncertainty convention (adapters'
      ``SIGMA_MODELS`` registry or an adapter-specific name), with
      ``sigma_params`` and, for the EFIT k-file route, ``VBIT``;
    * ``quantity`` -- when a group can hold more than one form of the same
      measurement (MSE: ``"polarisation_angle"`` [rad, default] or
      ``"tan_polarisation_angle"``);
    * MSE ``er_corrected`` (bool: the data already have the E_r term removed,
      so the forward model must use E_R = 0), with ``er_route`` /
      ``er_status`` / ``er_evidence`` when an adapter decided it;
    * MSE ``field_orientation`` -- ``(sign_pol, sign_tor)``, default
      ``(1, 1)``: a STATED map from the equilibrium's orientation to the
      discharge's.  ``sign_pol`` multiplies the modeled ``B_R`` and ``B_Z``,
      ``sign_tor`` the modeled ``B_phi``, before the MSE formula
      (:func:`.diagnostics.mse_from_field`).  It is a convention to be
      declared from the discharge's Ip and B_t directions, never fit;
    * flux loops ``flux_reference`` -- label of the loop the fluxes are taken
      relative to (absent or None for absolute fluxes).

    ``time`` and ``time_window`` of the enclosing :class:`MeasurementSet` are
    in seconds.
    """

    name: str
    labels: List[str]
    measured: np.ndarray
    sigma: np.ndarray
    valid: np.ndarray
    weight: np.ndarray = None
    geometry: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.measured = np.asarray(self.measured, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        self.valid = np.asarray(self.valid, dtype=bool)
        if self.weight is None:
            self.weight = np.ones_like(self.measured)
        self.weight = np.asarray(self.weight, dtype=float)
        n = self.measured.shape[0]
        for arr in (self.sigma, self.valid, self.weight):
            if arr.shape != (n,):
                raise ValueError(f"group '{self.name}': arrays must all have shape ({n},)")
        if len(self.labels) != n:
            raise ValueError(f"group '{self.name}': {len(self.labels)} labels for {n} channels")


@dataclass
class MeasurementSet:
    """All measurements for one device and one time window (times in seconds)."""

    machine: str
    time: float
    time_window: float
    groups: Dict[str, ConstraintGroup] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class ResidualTable:
    """Measured vs. modeled values, group by group, for one equilibrium.

    ``model[name]`` is aligned with ``measurements.groups[name].measured``.
    The normalized residual is ``weight * (model - measured) / sigma``,
    matching ``equilibrium.time_slice[:].constraints.*.chi_squared``.

    ``model_valid[name]`` is a per-channel mask: False where the forward model
    could not evaluate a channel (a diagnostic returned NaN with a reason:
    off the mesh, a Thomson channel outside the LCFS, non-finite).  A channel
    enters the residual and chi^2 when it is valid in the measurement set AND
    model-valid; a group without a mask counts as model-valid everywhere.
    ``excluded[name]`` lists ``(label, reason)`` for the measurement-valid
    channels the mask removed, and ``notes`` holds table-wide remarks (groups
    skipped, MSE E_r treatment).  ``measurements`` is the set that was scored,
    not a modified copy.
    """

    measurements: MeasurementSet
    model: Dict[str, np.ndarray] = field(default_factory=dict)
    model_valid: Dict[str, np.ndarray] = field(default_factory=dict)
    excluded: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def used(self, name):
        """Mask of the channels of one group that enter chi^2."""
        g = self.measurements.groups[name]
        mv = self.model_valid.get(name)
        return g.valid if mv is None else g.valid & np.asarray(mv, dtype=bool)

    def residual(self, name):
        """Normalized residuals of the used channels of one group (:meth:`used`)."""
        g = self.measurements.groups[name]
        use = self.used(name)
        r = g.weight[use] * (np.asarray(self.model[name])[use] - g.measured[use]) / g.sigma[use]
        return r

    def chi2(self, name=None):
        """Chi-squared of one group, or of all groups when ``name`` is None."""
        names = [name] if name is not None else list(self.model)
        return float(sum(np.sum(self.residual(n) ** 2) for n in names))

    def n_valid(self, name=None):
        """Number of used channels in one group, or in all modeled groups."""
        names = [name] if name is not None else list(self.model)
        return int(sum(np.count_nonzero(self.used(n)) for n in names))
