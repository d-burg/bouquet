"""
IDA-lite NetCDF profile reader
==============================

Reads kinetic profiles from IDA-lite ``.cdf`` files produced by the
Integrated Data Analysis (IDA) code at IPP Garching.

Two file layouts are supported, detected automatically from the shape of the
``n_e`` array:

- **Workflow 1** — shape ``(n_times, n_samples, n_radial)``: a single time
  point with a full posterior distribution of samples.  The central profile
  is the median (``uncertainty_method='percentile'``) or mean
  (``uncertainty_method='std'``) over the sample axis.
- **Workflow 2** — shape ``(n_times, n_radial)``: multiple time slices with
  pre-computed fitted profiles and explicit uncertainty columns
  (``n_e_err``, ``T_e_err``, ``T_12C6_err``).

Requires the ``h5py`` package.
"""

import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Module-level helpers (also used by parallel.IDALiteUncertaintyGenerator)
# ---------------------------------------------------------------------------

def _interp_to_grid(psin_src, arr, psin_tgt):
    """Linear interpolation onto *psin_tgt* with boundary fill values."""
    return interp1d(
        psin_src, arr,
        kind='linear', bounds_error=False,
        fill_value=(arr[0], arr[-1]),
    )(psin_tgt)


def _summarise_samples(samples, method='percentile'):
    """Reduce ``(n_samples, n_radial)`` posterior draws to centre and 1σ.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_radial)
    method  : ``'percentile'`` (default) or ``'std'``

    Returns
    -------
    centre : ndarray (n_radial,)
        Posterior median (percentile) or mean (std).
    sigma : ndarray (n_radial,)
        Half-width of the 16th–84th percentile band, or 1-sigma std.
    """
    if method == 'std':
        centre = samples.mean(axis=0)
        sigma  = samples.std(axis=0)
    else:  # percentile
        centre = np.median(samples, axis=0)
        lo     = np.percentile(samples, 16, axis=0)
        hi     = np.percentile(samples, 84, axis=0)
        sigma  = (hi - lo) / 2.0
    return centre, sigma


def _detect_workflow(pf_ne, profile_file):
    """Return ``1`` or ``2`` based on the dimensionality of *pf_ne*."""
    if pf_ne.ndim == 3:
        return 1  # (n_times, n_samples, n_radial) — posterior samples
    if pf_ne.ndim == 2:
        return 2  # (n_times, n_radial) — fitted profiles + error columns
    raise ValueError(
        f"Unexpected n_e shape {pf_ne.shape} in {profile_file!r}. "
        "Expected 2D (n_times, n_radial) or 3D (n_times, n_samples, n_radial)."
    )


def _select_time_index(time_arr_ms, workflow_num, time_idx, sim_time_ms):
    """Return the integer time index to use for a given CDF file."""
    if workflow_num == 1 or sim_time_ms is None:
        return time_idx
    # Workflow 2 with explicit target time
    pf_time = np.asarray(time_arr_ms) / 1e3   # ms → s
    return int(np.argmin(np.abs(pf_time - sim_time_ms / 1e3)))


# ---------------------------------------------------------------------------
# Public reader class
# ---------------------------------------------------------------------------

class IDALiteProfileReader:
    r"""Read kinetic profiles from an IDA-lite NetCDF (.cdf) file.

    Two file layouts are detected automatically from the shape of ``n_e``:

    - **Workflow 1** — ``(n_times, n_samples, n_radial)``: Bayesian posterior
      samples at a single time point.  The central profile is computed as
      the median (``uncertainty_method='percentile'``) or mean
      (``uncertainty_method='std'``) across the sample axis.
    - **Workflow 2** — ``(n_times, n_radial)``: fitted profiles for multiple
      time slices with explicit uncertainty columns.

    ``Zeff`` is read directly from the CDF variable ``Zeff``.

    By default, the main-ion (deuterium) density is computed using the
    measured carbon density via quasi-neutrality:

    .. math::

        n_i = \max\bigl(n_e - Z_C\,n_C,\; 0\bigr)

    where :math:`n_C` is read from CDF variable ``n_12C6``.  Setting
    *carbon_quasi_neutrality* to ``False`` falls back to the simpler
    approximation :math:`n_i \approx n_e`.

    Parameters
    ----------
    time_idx : int
        Time index to use for workflow 1 (typically 0 for a single-time file)
        or as a fallback for workflow 2 when *sim_time_ms* is not provided.
    sim_time_ms : float or None
        Target time in milliseconds.  When provided and the file is workflow 2,
        the nearest time slice in the CDF ``time`` variable is selected.
        Ignored for workflow 1.
    Z_C : int
        Charge number of the carbon impurity (default 6).  Only used when
        *carbon_quasi_neutrality* is ``True``.
    uncertainty_method : ``'percentile'`` | ``'std'``
        How to summarise workflow-1 posterior samples.  ``'percentile'`` uses
        the median and 16th/84th percentile band; ``'std'`` uses mean ± σ.
    carbon_quasi_neutrality : bool
        If ``True`` (default), compute :math:`n_i = \max(n_e - Z_C n_C, 0)` using the
        ``n_12C6`` CDF variable.  If ``False``, use :math:`n_i \approx n_e`.
    """

    def __init__(self, time_idx=0, sim_time_ms=None, Z_C=6,
                 uncertainty_method='percentile',
                 carbon_quasi_neutrality=True):
        self.time_idx = time_idx
        self.sim_time_ms = sim_time_ms
        self.Z_C = Z_C
        self.uncertainty_method = uncertainty_method
        self.carbon_quasi_neutrality = carbon_quasi_neutrality

    def __call__(self, geqdsk_file, profile_file, psi_N):
        """Read profiles from an IDA-lite CDF and interpolate onto *psi_N*.

        Parameters
        ----------
        geqdsk_file : str
            Path to the geqdsk (unused; present for interface consistency).
        profile_file : str
            Path to the IDA-lite ``.cdf`` file.
        psi_N : array-like
            Normalised flux grid onto which profiles are interpolated.

        Returns
        -------
        ne_SI, te_SI, ni_SI, ti_SI : ndarray
            Profiles in SI units (m^-3, eV).
        Zeff_eq : ndarray
            Effective charge (clipped to ≥ 1).
        profile_bytes : bytes
            Raw bytes of the profile file for HDF5 archival.
        """
        import h5py

        with h5py.File(profile_file, 'r') as f:
            # Read raw data, converting to working units (10^20 m^-3, keV)
            pf_ne    = f['n_e'][:]    / 1e20
            pf_te    = f['T_e'][:]    / 1e3
            pf_ti    = f['T_12C6'][:] / 1e3
            pf_zeff  = f['Zeff'][:]
            time_arr = f['time'][:]
            pf_psin  = f['psi_n'][:]
            if self.carbon_quasi_neutrality:
                pf_nc = f['n_12C6'][:] / 1e20

        workflow_num = _detect_workflow(pf_ne, profile_file)
        t_idx = _select_time_index(time_arr, workflow_num, self.time_idx,
                                   self.sim_time_ms)

        if workflow_num == 1:
            # Radial grid is the same for all samples
            psin_ida = pf_psin[t_idx, 0, :]
            ne_centre, _ = _summarise_samples(pf_ne[t_idx], self.uncertainty_method)
            te_centre, _ = _summarise_samples(pf_te[t_idx], self.uncertainty_method)
            ti_centre, _ = _summarise_samples(pf_ti[t_idx], self.uncertainty_method)
            zeff_centre  = np.clip(pf_zeff[t_idx].mean(axis=0), 1.0, None)
            if self.carbon_quasi_neutrality:
                nc_centre, _ = _summarise_samples(pf_nc[t_idx], self.uncertainty_method)
        else:  # workflow 2
            psin_ida    = pf_psin
            ne_centre   = pf_ne[t_idx].copy()
            te_centre   = pf_te[t_idx].copy()
            ti_centre   = pf_ti[t_idx].copy()
            zeff_centre = np.clip(pf_zeff[t_idx], 1.0, None)
            if self.carbon_quasi_neutrality:
                nc_centre = pf_nc[t_idx].copy()

        # Derive ni: either ne - Z_C*n_C (carbon QN) or ni ≈ ne (simple)
        if self.carbon_quasi_neutrality:
            ni_centre = np.maximum(ne_centre - self.Z_C * nc_centre, 0.0)
        else:
            ni_centre = ne_centre

        # Interpolate onto target grid
        ne_SI   = _interp_to_grid(psin_ida, ne_centre,   psi_N) * 1e20
        te_SI   = _interp_to_grid(psin_ida, te_centre,   psi_N) * 1e3
        ni_SI   = _interp_to_grid(psin_ida, ni_centre,   psi_N) * 1e20
        ti_SI   = _interp_to_grid(psin_ida, ti_centre,   psi_N) * 1e3
        Zeff_eq = np.clip(_interp_to_grid(psin_ida, zeff_centre, psi_N), 1.0, None)

        with open(profile_file, 'rb') as fh:
            profile_bytes = fh.read()

        return ne_SI, te_SI, ni_SI, ti_SI, Zeff_eq, profile_bytes

# ---------------------------------------------------------------------------
# Uncertainty generator
# ---------------------------------------------------------------------------

class IDALiteUncertaintyGenerator:
    r"""Compute 1σ kinetic-profile uncertainty envelopes from an IDA-lite CDF.

    Two file layouts are handled automatically (same detection as
    :class:`IDALiteProfileReader`):

    - **Workflow 1** — ``(n_times, n_samples, n_radial)``: Bayesian posterior
      samples at a single time point.  The 1σ half-width is derived from the
      posterior distribution:

      - ``uncertainty_method='percentile'`` (default) → half-width of the
        16th–84th percentile band.
      - ``uncertainty_method='std'`` → standard deviation across samples.

    - **Workflow 2** — ``(n_times, n_radial)``: fitted profiles with explicit
      per-variable uncertainty columns in the CDF.  The sigma arrays are read
      directly from the configurable variable names (*ne_sigma_var*, etc.).

    For ``sigma_jphi``, pass ``None`` — IDA-lite files do not carry current
    density information.  The caller is responsible for computing it from the
    reconstructed :math:`j_\phi` (see ``jphi_uncertainty_gen`` in
    :class:`~bouquet.parallel.load_IDA_file_obj`).

    Parameters
    ----------
    time_idx : int
        Time index for workflow 1 (typically 0) or fallback for workflow 2
        when *sim_time_ms* is not provided.
    sim_time_ms : float or None
        Target time in milliseconds for workflow 2 time-slice selection.
        The nearest slice in the CDF ``time`` variable is chosen.
        Ignored for workflow 1.
    Z_C : int
        Charge of the carbon impurity (default 6).  Used for error propagation
        when *carbon_quasi_neutrality* is ``True``.
    uncertainty_method : ``'percentile'`` | ``'std'``
        How to summarise workflow 1 posterior samples.
    carbon_quasi_neutrality : bool
        If ``True``, propagate the carbon-density uncertainty into
        :math:`\sigma_{n_i}` via

        .. math::

            \sigma_{n_i} = \sqrt{\sigma_{n_e}^2 + (Z_C\,\sigma_{n_C})^2}

        - **Workflow 1**: :math:`\sigma_{n_C}` is derived from the ``n_12C6``
          posterior samples using *uncertainty_method*.
        - **Workflow 2**: :math:`\sigma_{n_C}` is read from *nc_sigma_var*.

        When ``False`` the simpler :math:`\sigma_{n_i} \approx
        \sigma_{n_e}` approximation is used (or *ni_sigma_var* when set).
    ne_sigma_var : str
        CDF variable name for the electron-density 1σ [m\ :sup:`-3`]
        (workflow 2 only).
    te_sigma_var : str
        CDF variable name for the electron-temperature 1σ [eV]
        (workflow 2 only).
    ti_sigma_var : str
        CDF variable name for the ion-temperature 1σ [eV]
        (workflow 2 only).
    ni_sigma_var : str or None
        CDF variable name for an explicit main-ion density 1σ
        [m\ :sup:`-3`] (workflow 2 only).  When set this takes priority over
        both *carbon_quasi_neutrality* propagation and the *ne_sigma_var*
        fallback.
    nc_sigma_var : str
        CDF variable name for the carbon-density 1σ [m\ :sup:`-3`]
        (workflow 2 only; used only when *carbon_quasi_neutrality* is
        ``True`` and *ni_sigma_var* is ``None``).
    """

    def __init__(
        self,
        time_idx=0,
        sim_time_ms=None,
        Z_C=6,
        uncertainty_method='percentile',
        carbon_quasi_neutrality=True,
        ne_sigma_var='n_e_err',
        te_sigma_var='T_e_err',
        ti_sigma_var='T_12C6_err',
        ni_sigma_var=None,
        nc_sigma_var='n_12C6_err',
    ):
        self.time_idx                = time_idx
        self.sim_time_ms             = sim_time_ms
        self.Z_C                     = Z_C
        self.uncertainty_method      = uncertainty_method
        self.carbon_quasi_neutrality = carbon_quasi_neutrality
        self.ne_sigma_var            = ne_sigma_var
        self.te_sigma_var            = te_sigma_var
        self.ti_sigma_var            = ti_sigma_var
        self.ni_sigma_var            = ni_sigma_var
        self.nc_sigma_var            = nc_sigma_var

    def __call__(self, profile_file, profile_reader, psi_N, j_phi_fit=None):
        """Compute 1σ arrays from the CDF and interpolate onto *psi_N*.

        Parameters
        ----------
        profile_file : str
            Path to the IDA-lite ``.cdf`` file.
        profile_reader : callable
            Unused; present for interface consistency with the bouquet
            uncertainty-generator protocol.
        psi_N : ndarray
            Normalised flux grid onto which sigmas are interpolated.
        j_phi_fit : ndarray or None
            Unused; retained for interface consistency.

        Returns
        -------
        sigma_ne, sigma_te, sigma_ni, sigma_ti : ndarray
            Absolute 1σ uncertainties in SI units (m\ :sup:`-3`, eV).
        sigma_jphi : None
            Always ``None``; current-density uncertainty is not available
            from IDA-lite files and must be supplied by the caller.
        """
        import h5py

        with h5py.File(profile_file, 'r') as f:
            pf_ne    = f['n_e'][:] / 1e20   # → 10^20 m^-3
            time_arr = f['time'][:]

            workflow_num = _detect_workflow(pf_ne, profile_file)
            t_idx = _select_time_index(time_arr, workflow_num, self.time_idx,
                                       self.sim_time_ms)

            if workflow_num == 1:
                # ---- Posterior samples: derive sigma from the distribution ----
                # Radial grid is the same for all samples (shape: n_radial)
                psin_ida = f['psi_n'][t_idx, 0, :]

                _, sigma_ne_sc = _summarise_samples(
                    pf_ne[t_idx], self.uncertainty_method)                   # 10^20 m^-3
                _, sigma_te_sc = _summarise_samples(
                    f['T_e'][t_idx] / 1e3, self.uncertainty_method)          # keV
                _, sigma_ti_sc = _summarise_samples(
                    f['T_12C6'][t_idx] / 1e3, self.uncertainty_method)       # keV

                if self.carbon_quasi_neutrality:
                    # ni = ne - Z_C * n_C  →  σ(ni) = sqrt(σ(ne)^2 + (Z_C σ(n_C))^2)
                    _, sigma_nc_sc = _summarise_samples(
                        f['n_12C6'][t_idx] / 1e20, self.uncertainty_method)
                    sigma_ni_sc = np.sqrt(sigma_ne_sc**2 + (self.Z_C * sigma_nc_sc)**2)
                else:
                    sigma_ni_sc = sigma_ne_sc.copy()   # σ(ni) ≈ σ(ne)

                # Convert scaled units → SI
                sigma_ne_si = sigma_ne_sc * 1e20   # m^-3
                sigma_te_si = sigma_te_sc * 1e3    # eV
                sigma_ni_si = sigma_ni_sc * 1e20   # m^-3
                sigma_ti_si = sigma_ti_sc * 1e3    # eV

            else:
                # ---- Workflow 2: read explicit error columns (already SI) ----
                psin_ida    = f['psi_n'][:]
                sigma_ne_si = f[self.ne_sigma_var][t_idx].copy()
                sigma_te_si = f[self.te_sigma_var][t_idx].copy()
                sigma_ti_si = f[self.ti_sigma_var][t_idx].copy()
                if self.ni_sigma_var is not None:
                    # Explicit override: use the named column directly
                    sigma_ni_si = f[self.ni_sigma_var][t_idx].copy()
                elif self.carbon_quasi_neutrality:
                    # ni = ne - Z_C * n_C  →  σ(ni) = sqrt(σ(ne)^2 + (Z_C σ(n_C))^2)
                    sigma_nc_si = f[self.nc_sigma_var][t_idx].copy()
                    sigma_ni_si = np.sqrt(sigma_ne_si**2 + (self.Z_C * sigma_nc_si)**2)
                else:
                    sigma_ni_si = sigma_ne_si.copy()   # σ(ni) ≈ σ(ne) fallback

        sigma_ne   = _interp_to_grid(psin_ida, sigma_ne_si, psi_N)
        sigma_te   = _interp_to_grid(psin_ida, sigma_te_si, psi_N)
        sigma_ni   = _interp_to_grid(psin_ida, sigma_ni_si, psi_N)
        sigma_ti   = _interp_to_grid(psin_ida, sigma_ti_si, psi_N)

        return sigma_ne, sigma_te, sigma_ni, sigma_ti, None
