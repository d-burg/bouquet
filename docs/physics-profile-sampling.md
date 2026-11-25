# Physics-informed profile sampling, critical-gradient stiffness, and transport-consistent diffusivity

This page is the technical and mathematical reference for the physics-informed
kinetic, current, and transport-diffusivity sampling framework introduced on
`feat/physics-profile-sampling` (`bouquet.geometry.miller`,
`bouquet.surrogates`, `bouquet.stiffness`, `bouquet.stiff_sampler`, and the
extensions to `bouquet.sampling` and `bouquet.kinetic_sampler`). It records
every governing equation, physical assumption, explicit user prior, licensing
constraint, known limitation, and literature source for pull-request review and
publication.

For the user-facing configuration flags on `UncertaintyConfig`, see
[workflows.md](workflows.md). For the Phase-2 design proposal on generative
forward flux-matching ensembles (`sampling_mode="flux_matched"`), see
[proposals/flux-matched-surrogate-draws.md](proposals/flux-matched-surrogate-draws.md).

---

## Contents

1. [Motivation: why stationary scalar GPs and global monotonicity break down](#1-motivation-why-stationary-scalar-gps-and-global-monotonicity-break-down)
2. [Design principle: zero artificial smoothing, explicit priors](#2-design-principle-zero-artificial-smoothing-explicit-priors)
3. [Mathematical formulation](#3-mathematical-formulation)
   - [3.1 GP correlation length sign convention, coordinate mapping, and exact axis quadrature](#31-gp-correlation-length-sign-convention-coordinate-mapping-and-exact-axis-quadrature)
   - [3.2 The non-stationary Gibbs derivative paradox and Chilenski two-plateau form](#32-the-non-stationary-gibbs-derivative-paradox-and-chilenski-two-plateau-form)
   - [3.3 Miller flux-surface geometry and GACODE gyro-Bohm normalizations](#33-miller-flux-surface-geometry-and-gacode-gyro-bohm-normalizations)
   - [3.4 Pedestal window detection (outer-barrier HWHM + EPED1 width)](#34-pedestal-window-detection-outer-barrier-hwhm--eped1-width)
   - [3.5 Tier 1 analytical critical-gradient thresholds and stiffness](#35-tier-1-analytical-critical-gradient-thresholds-and-stiffness)
   - [3.6 Tier 2 standalone `TGLF-NN` and `QLKNN` flux-curve inversion](#36-tier-2-standalone-tglf-nn-and-qlknn-flux-curve-inversion)
   - [3.7 Regional positivity and baseline-normalized core gradient bounds](#37-regional-positivity-and-baseline-normalized-core-gradient-bounds)
   - [3.8 Stiff-core inward gradient integration and exact variance partitioning](#38-stiff-core-inward-gradient-integration-and-exact-variance-partitioning)
   - [3.9 Power-balance vs. surrogate thermal diffusivity (`chi_consistency`)](#39-power-balance-vs-surrogate-thermal-diffusivity-chi_consistency)
4. [Physics-derived quantities vs. explicit user priors](#4-physics-derived-quantities-vs-explicit-user-priors)
5. [Workflow selection: diagnostic UQ vs. predictive burning-plasma UQ](#5-workflow-selection-diagnostic-uq-vs-predictive-burning-plasma-uq)
6. [Assumptions, limitations, and open investigations](#6-assumptions-limitations-and-open-investigations)
7. [`TGLF-NN` and `QLKNN` licensing and distribution policy](#7-tglf-nn-and-qlknn-licensing-and-distribution-policy)
8. [Recommended pull-request structure](#8-recommended-pull-request-structure)
9. [References](#9-references)

---

## 1. Motivation: why stationary scalar GPs and global monotonicity break down

In `bouquet` v1.4.x (`length_scale_mode="manual"`, `sampling_mode="gibbs_level"`,
`gradient_policy="legacy"`, `chi_consistency="passive_gp"`), kinetic and
auxiliary profiles are perturbed using:

1. **Stationary scalar correlation lengths** (`n_ls=0.5`, `t_ls=0.4`,
   `j_ls=0.25`, `aux_length_scales={'zeff': 0.4}`) in the run radial coordinate
   $\hat{x} \in [0, 1]$ ($\hat{\psi} \equiv \psi_N$ by default, or normalized
   toroidal flux $\Phi_N = \rho_{\mathrm{tor}}^2$ when `source.coord="phi_n"`);
2. **Global monotonicity rejection** (`_draw_monotonic_perturbation`), requiring
   $\partial f / \partial\hat{x} \le 0$ at every radial node from the magnetic
   axis to the separatrix; and
3. **Uncoupled passive Gaussian-process draws** for thermal diffusivities
   $\chi_e(\hat{x})$ and $\chi_i(\hat{x})$.

While effective on monotonic, moderate-field NBI H-mode discharges where core
heat fluxes sit moderately above turbulent thresholds ($Q_{\mathrm{PB}}/Q_{\mathrm{gB,imas}} \sim 1\text{--}4$),
four structural failure modes emerge across broader tokamak regimes:

- **Cross-regime transport stiffness mismatch:** Gyro-Bohm heat-flux
  normalization scales as $Q_{\mathrm{gB}} = n_e(eT_e)c_s(\rho_s/a)^2 \propto n_e T_e^{5/2} / (B_{\mathrm{unit}}^2 a^2)$.
  In compact high-field burning plasmas ($B_0 \sim 8\text{--}12\text{ T}$,
  $n_e \sim 3\text{--}4 \times 10^{20}\text{ m}^{-3}$, $T_i \sim 15\text{--}25\text{ keV}$),
  $Q_{\mathrm{gB,imas}}$ increases by an order of magnitude relative to medium-field
  devices, driving $Q_{\mathrm{PB}} / Q_{\mathrm{gB,imas}} \ll 1$ across the
  confinement zone (Rodriguez-Fernandez et al. 2020, 2022). Core inverse
  temperature gradient scale lengths $R_0 / L_{T_s} \equiv -R_0\,\partial\ln T_s/\partial r$
  are pinned tightly to ion-temperature-gradient (ITG), trapped-electron-mode
  (TEM), and electron-temperature-gradient (ETG) critical thresholds. Applying
  a short stationary correlation length ($\ell \sim 0.25\text{--}0.40$) at
  15–20% temperature uncertainty injects local $\Delta(R_0/L_T) \sim 3\text{--}5$
  excursions that would require hundreds of megawatts of unphysical local
  heating or sinks to sustain, while simultaneously over-smoothing the narrow
  H-mode edge pedestal ($\Delta\psi_{N,\mathrm{ped}} \sim 0.03\text{--}0.07$).
- **Truncation bias on flat cores and rejection of hollow/pellet profiles:**
  Requiring $\partial f/\partial\hat{x} \le 0$ everywhere systematically
  biases sawtooth-clamped flat cores upward near the magnetic axis
  ($\mathbb{E}[f_{\mathrm{draw}}(0) - f_{\mathrm{draw}}(\hat{x}_{q=1})] > 0$)
  and rejects physically valid off-axis pellet density peaks
  ($R_0/L_{n_e} < 0$), core impurity-dilution hollows in $n_i$, and radiatively
  cooled core electron temperature profiles ($R_0/L_{T_e} < 0$ under high-$Z$
  or seeded core radiation).
- **Pedestal-core decoupling in level-space GPs:** A zero-mean GP in profile
  level $f(\hat{x})$ with $\ell_{\mathrm{ped}} \ll 1$ produces near-zero
  correlation between $T_{\mathrm{ped}}$ and $T(0)$, contradicting stiff
  profile resiliency ($T(0) / T_{\mathrm{ped}} \approx \exp[\frac{a}{R_0}\int \kappa_c\,d(r/a)]$).
- **Power-balance violation in passive $\chi_e, \chi_i$ draws:** Perturbing
  $\chi_e$ independently of $(n_e, \nabla T_e)$ allows draws that steepen
  $-\nabla T_e$ while simultaneously raising $\chi_e$, violating steady-state
  energy conservation $\chi_s = -Q_s / (n_s e \nabla T_s)$ at rational surfaces
  consumed by downstream drift-tearing and error-field-penetration solvers
  (e.g., `SLAYER`).

---

## 2. Design principle: zero artificial smoothing, explicit priors

The equations in `bouquet.stiffness`, `bouquet.stiff_sampler`, and
`bouquet.surrogates` follow three transparent design rules:

1. **Zero post-hoc spatial smoothing filters or core correlation ceilings:** No
   Gaussian or Savitzky–Golay filters are applied to resolved correlation-length
   arrays, surrogate fluxes, or sampled profiles, and core $\ell(\hat{x})$ is
   not artificially capped below `1.0` (where $\ell > 1$ physically encodes a
   nearly uniform core shift). Where surrogate training bounds require radial
   domain restrictions (such as restricting core CGM/surrogate evaluation to
   $\rho_{\mathrm{tor}} \in [0.18, 0.88]$ and bounding edge Jacobian slopes at
   interior strictly increasing minor-radius knots), those domain limits are
   documented explicitly below.
2. **Published constants and exact statistical identities:** Conversion factors
   use CODATA SI constants, multi-species Braginskii / Bosch–Hale / Stix
   transport formulas, exact Gaussian-process covariance identities, and
   7-point Gauss–Hermite quadrature.
3. **Explicit user priors where local gyrokinetics is silent:** Wherever local
   core turbulence models do not determine a fuel composition, amplitude, or
   spatial scale — specifically fuel tritium fraction (`tritium_fraction`,
   defaulting to IMAS `core_profiles` $Z=1$ inspection or `0.0` pure Deuterium),
   surface-flux uncertainty (`flux_uncertainty_rel`), gradient-residual radial
   coherence (`stiff_gradient_ls`), impurity $Z_{\mathrm{eff}}$ correlation
   length (`aux_length_scales['zeff']`), and unrelaxed inductive current
   correlation length (`j_ls` when `current_evolution_time_s is None`) — the
   parameter is exposed directly on `UncertaintyConfig`.

---

## 3. Mathematical formulation

### 3.1 GP correlation length sign convention, coordinate mapping, and exact axis quadrature

For a stationary unit-variance squared-exponential (RBF) Gaussian process
$z(r)$ with covariance $k(r, r') = \exp\!\big(-(r - r')^2 / (2\ell_r^2)\big)$,
the marginal variance of its first spatial derivative has exact prefactor `1.0`:

$$
\mathrm{Var}\!\left[\frac{dz}{dr}\right]
= \left.\frac{\partial^2 k(r, r')}{\partial r\,\partial r'}\right|_{r'=r}
= \frac{1}{\ell_r^2}.
$$

For an additive perturbation $\delta f(r) = \sigma_f(r)\,z(r)$ with relative
envelope $\sigma_{\mathrm{rel}}(r) \equiv \sigma_f(r)/|f_0(r)|$, the variance of
the linearized inverse scale-length perturbation $\delta(R_0/L_f) \approx -R_0\,\partial_r(\delta f / f_0)$ is

$$
\mathrm{Var}\!\left[\delta\!\left(\frac{R_0}{L_f}\right)\right]
= R_0^2\big(\sigma_{\mathrm{rel}}'(r)\big)^2
+ \frac{R_0^2\,\sigma_{\mathrm{rel}}(r)^2}{\ell_r(r)^2}
\;\;\simeq\;\; \frac{R_0^2\,\sigma_{\mathrm{rel}}(r)^2}{\ell_r(r)^2}
\quad\big(|\sigma_{\mathrm{rel}}'| \ll \sigma_{\mathrm{rel}}/\ell_r\big).
$$

Inverting the dominant correlation-length term for the minor-radius scale
$\ell_r(r)$ [m] that bounds $1\sigma$ logarithmic-gradient fluctuations to the
allowable transport excursion $\Delta\kappa_{\mathrm{allow}}(r) \equiv \Delta(R_0/L_f)_{\mathrm{allow}}$
gives:

$$
\ell_r(r) = \frac{R_0\,\big(\sigma_f(r) / |f_0(r)|\big)}{\Delta\kappa_{\mathrm{allow}}(r)}.
$$

- **Larger $\ell$ = smoother, more globally coherent draws** (stiff core where
  $\Delta\kappa_{\mathrm{allow}}$ is small; $\ell > 1$ indicates that a
  $1\sigma$ level shift requires the core to move up or down coherently).
- **Smaller $\ell$ = shorter-wavelength radial structure** (weakly stiff core
  or narrow H-mode pedestal barrier).

#### Run-coordinate consistency (`coord="psi_n"` vs. `coord="phi_n"`)

In `bouquet`, `MillerGeometry.psi_N` is indexed by normalized poloidal flux
$\psi_N$, whereas when `source.coord = "phi_n"` (or `"rho_tor"`),
`baseline.psi_N_kinetic` and `baseline.psi_N` store normalized **toroidal** flux
$\Phi_N = \rho_{\mathrm{tor}}^2$. Both `resolve_uncertainty` (`bouquet/baseline.py`)
and `compute_physics_length_scales` (`bouquet/stiffness.py`) map run-grid nodes
$\hat{x}$ to true poloidal flux $\psi_N$ via `baseline.psi_map` (falling back to
inverse $(\rho_{\mathrm{tor,norm}}^2, \psi_N)$ equilibrium interpolation when
`psi_map is None`) before querying `MillerGeometry` and `detect_pedestal_window`,
and then map pedestal bounds and length scales $\ell_{\hat{x}}$ back onto the
active run coordinate $\hat{x}$ ($\psi_N$ or $\Phi_N$). Likewise, when
`current_evolution_time_s > 0`, `mg_j` is resolved directly on `psi_j_pol` via
`miller_from_baseline` rather than clamping onto `psi_kin` when
$\psi_{\mathrm{kin}}[-1] < \psi_j[-1]$.

#### Mapping $\ell_r \mapsto \ell_{\hat{x}}$ without magnetic-axis singularities

Near the magnetic axis ($r \to 0$), both $\psi_N(r)$ and $\Phi_N(r)$ scale
quadratically with minor radius, $\hat{x}(r) \propto (r/a)^2$, so the local
first-order Jacobian $|d\hat{x}/dr| \to 0$. Naive chain-rule scaling
$\ell_{\hat{x}} = |d\hat{x}/dr|\,\ell_r$ artificially collapses
$\ell_{\hat{x}}(0) \to 0$ unless an ad-hoc axis floor is invented.

Instead, `stiffness._exact_ell_psi_from_ell_r` maps $\ell_r(r)$ onto the run
coordinate $\hat{x}(r)$ via the exact RMS coordinate excursion over a Gaussian
radial step $\Delta r = \ell_r(r)\,Z$ with $Z \sim \mathcal{N}(0, 1)$, reflected
across the magnetic axis ($r' = |r + \ell_r Z|$) and linearly extrapolated past
the LCFS ($r' > r_{\min,\mathrm{end}}$, using strictly increasing interior
minor-radius knots with guarded slope $s_a = \mathrm{clip}(\Delta\hat{x}/\Delta r,\, 0.2/a,\, 10/a)$
so $\ell_{\hat{x}}$ can exceed `1.0` naturally in stiff cores):

$$
\ell_{\hat{x}}(r)
= \left(\mathbb{E}_{Z \sim \mathcal{N}(0,1)}\!\left[
\Big(\hat{x}_{\mathrm{ext}}\big(|r + \ell_r(r) Z|\big) - \hat{x}(r)\Big)^2
\right]\right)^{1/2},
$$

evaluated deterministically by 7-point Gauss–Hermite quadrature. Away from the
axis ($\ell_r \ll r$), Taylor expansion recovers $|d\hat{x}/dr|\,\ell_r$; at
$r = 0$, $\hat{x}(|\ell_r Z|) \propto (\ell_r/a)^2 Z^2$ with
$\mathbb{E}[Z^4]^{1/2} = \sqrt{3}$, keeping $\ell_{\hat{x}}(0) > 0$ purely from
flux-surface geometry.

---

### 3.2 The non-stationary Gibbs derivative paradox and Chilenski two-plateau form

`GPRProfilePerturber` (`bouquet/sampling.py`) implements non-stationary
1-D Paciorek–Schervish / Gibbs kernels (Gibbs 1997; Paciorek & Schervish 2006;
Chilenski et al. 2015). For the squared-exponential Gibbs kernel:

$$
k_{\mathrm{Gibbs}}(\hat{x}, \hat{x}')
= \sqrt{\frac{2\,\ell(\hat{x})\,\ell(\hat{x}')}{\ell(\hat{x})^2 + \ell(\hat{x}')^2}}\,
\exp\!\left(-\frac{(\hat{x} - \hat{x}')^2}{\ell(\hat{x})^2 + \ell(\hat{x}')^2}\right),
$$

(and for `matern52`, scaling $|\hat{x} - \hat{x}'|$ by the quadratic mean
$\bar{\ell}_{\mathrm{RMS}} = \sqrt{(\ell(\hat{x})^2 + \ell(\hat{x}')^2)/2}$ per
Paciorek & Schervish 2006 Theorem 1 so the covariance matrix is guaranteed
positive definite across sharp pedestal length-scale drops, with strict grid
length validation `len(X) == ell.size`).

Expanding $\hat{x} = x_0 + u$ and $\hat{x}' = x_0 + v$ around $x_0$ with
$\ell(x_0 + u) = \ell_0 + \ell_0' u + \mathcal{O}(u^2)$:

$$
\sqrt{\frac{2\ell(x_0+u)\ell(x_0+v)}{\ell(x_0+u)^2 + \ell(x_0+v)^2}}
= 1 - \frac{\ell_0'^2}{4\ell_0^2}(u - v)^2 + \mathcal{O}\big(\|(u, v)\|^3\big),
$$

$$
\exp\!\left(-\frac{(u - v)^2}{\ell(x_0+u)^2 + \ell(x_0+v)^2}\right)
= 1 - \frac{(u - v)^2}{2\ell_0^2} + \mathcal{O}\big(\|(u, v)\|^3\big).
$$

Taking the mixed second derivative $\partial^2 k_{\mathrm{Gibbs}} / (\partial u\,\partial v)$
at $u = v = 0$ yields the exact sample-slope variance (Chilenski et al. 2015,
Eq. 29):

$$
\mathrm{Var}\!\left[\frac{dz}{d\hat{x}}(x_0)\right]
= \frac{1}{\ell(x_0)^2} + \frac{1}{2}\left(\frac{\ell'(x_0)}{\ell(x_0)}\right)^2.
$$

**Consequence:** Feeding a raw surface-by-surface stiffness array $\ell(\hat{x})$
(or a harmonic $1/\ell$ transition, where $|\ell'/\ell| \sim 1/\ell_{\mathrm{ped}}$
already inside the outer core!) into a level-space Gibbs GP inflates
$\frac{1}{2}(\ell'/\ell)^2$, generating artificial outer-core gradient spikes in
$\nabla p$ and $j_{\mathrm{BS}}$ (and similarly across $q = 1$ in `j_ls` unless
blended smoothly).

Following **Chilenski, Greenwald, Marzouk et al. (*Nucl. Fusion* 55, 023012,
2015, Eq. 30)** verbatim, `UncertaintyConfig.gibbs_ls_form` defaults to linear
`"chilenski_tanh"`:

$$
\ell(\hat{x})
= \big(1 - g(\hat{x})\big)\,\ell_{\mathrm{core,med}}
+ g(\hat{x})\,\ell_{\mathrm{ped}},
\qquad
g(\hat{x}) = \frac{1}{2}\left[1 + \tanh\!\left(\frac{\hat{x} - \hat{x}_{\mathrm{ped,top}}}{w_{\mathrm{trans}}}\right)\right],
$$

with $w_{\mathrm{trans}} = \frac{1}{2}\max(\ell_{\mathrm{ped}}, 0.03)$,
$\ell_{\mathrm{core,med}} = \mathrm{median}_{\rho_{\mathrm{tor}} \in [0.25, 0.75]}\big(\ell_{\mathrm{core,pt}}(\hat{x})\big)$,
and $\ell_{\mathrm{ped}} = \Delta\hat{x}_{\mathrm{ped}}$ when an H-mode pedestal
is detected (`has_pedestal=True`; constant $\ell_{\mathrm{core,med}}$ in
L-mode). Likewise, when `current_evolution_time_s > 0`, sawtooth mixing inside
$q \le 1$ is blended across $\hat{x}_{q=1}$ via a smooth `tanh` gate of width
$w_{q1} = \max(0.35\hat{x}_{q=1}, \ell_{\mathrm{skin}}(\hat{x}_{q=1}), 0.04)$
using Sauter et al. (1999) neoclassical conductivity $\sigma_{\mathrm{neo}}$. Setting `gibbs_ls_form="pointwise"` retains the raw
surface-by-surface curve $\ell_{\mathrm{core,pt}}(\hat{x})$ (note that near
$r = 0$, $\ell_{\hat{x}}(0)$ has nonzero slope $\ell_{\hat{x}}'(0)$ from the
quadratic axis Jacobian).

---

### 3.3 Miller flux-surface geometry and GACODE gyro-Bohm normalizations

`bouquet/geometry/miller.py` extracts local Miller shaping profiles
$(r, R_{\mathrm{maj}}, Z_{\mathrm{maj}}, \kappa, \delta, \zeta, q, \hat{s}, \alpha_{\mathrm{MHD}}, B_{\mathrm{unit}}, V, S, \mathrm{sign}(B_0), \mathrm{sign}(q))$
from either IMAS/OMAS data dictionaries (`miller_from_imas_dd`, computing
$\Phi_{\mathrm{tor}}$ and $\rho_{\mathrm{tor,norm}}$ prior to minor-radius
fallback so $r_{\min} = a\,\rho_{\mathrm{tor,norm}}$ avoids separatrix
$B_{\mathrm{unit}} \propto q$ divergence when `r_inboard`/`r_outboard` are
absent, and extracting LCFS shaping from `boundary.outline` when 1-D shaping
profiles are omitted) or COCOS-aware GEQDSK equilibria (`miller_from_geqdsk(..., cocos=...)`,
forwarding `ReconstructionSource.cocos` in `miller_from_baseline`), and computes
GACODE / `InputTGLF` dimensionless inputs (Candy 2009):

- **Effective flux-surface field:**
  $B_{\mathrm{unit}}(r) = \frac{|q(r)|}{r}\frac{d\psi_{\mathrm{pol}}}{dr}$, where
  $\psi_{\mathrm{pol}}$ is poloidal flux per radian $[\mathrm{Wb/rad}]$.
- **Orientation signs (`SIGN_IT` & signed $q$):** Preserves `sign_b0` and
  `sign_q` so `SIGN_IT = sign_b0 * sign_q` and `VEXB_SHEAR`, `VPAR_1`, and
  `VPAR_SHEAR_1` match `TurbulentTransport.jl` (`tglf.jl`) parity on reversed-$B_t$
  discharges.
- **Thermal species quasineutrality:** Impurity density $n_Z$ and main-ion
  fraction `AS_2`, `AS_3`, `RLNS_3` are computed smoothly with fast-ion charge
  $z_{\mathrm{fast}}$ accounted for and without discontinuous `np.where` step
  switches.
- **Deuterium reference sound speed and gyroradius:**
  $c_s = \sqrt{e T_e / m_D}$, $\Omega_{ci,D} = e B_{\mathrm{unit}} / m_D$,
  $\rho_s = c_s / \Omega_{ci,D}$ (with `XNUE` normalized consistently against
  $c_s / a$ using deuterium reference mass $\sqrt{2}$ per `GACODE`).
- **GACODE and IMAS/Miller surface-averaged gyro-Bohm flux scales:**
  $$
  Q_{\mathrm{gB}} = n_e\,(e T_e)\,c_s\left(\frac{\rho_s}{a}\right)^2\;\;[\mathrm{W/m^2}],
  \qquad
  Q_{\mathrm{gB,imas}} = Q_{\mathrm{gB}}\cdot\texttt{vprime\_miller}\;\;[\mathrm{W/m^2}],
  $$
  $$
  \Gamma_{\mathrm{gB}} = n_e\,c_s\left(\frac{\rho_s}{a}\right)^2\;\;[\mathrm{m^{-2}\,s^{-1}}],
  \qquad
  \Gamma_{\mathrm{gB,imas}} = \Gamma_{\mathrm{gB}}\cdot\texttt{vprime\_miller}\;\;[\mathrm{m^{-2}\,s^{-1}}],
  $$
  where $\texttt{vprime\_miller}$ converts TGLF's Miller-frame flux output to
  physical surface-averaged flux $P_s / S(r)$ (so $\chi_{\mathrm{turb}}$ divides
  `Qe_gB * Q_gB` without double-counting `vprime_miller`).
- **Normalization conversion for critical-gradient models:** Analytical CGM
  slopes $\chi_s^{\mathrm{Garbet}}$ in Garbet et al. (*PPCF* 46, 1351, 2004) are
  normalized to $(\rho_s / R_0)^2$, whereas GACODE normalizes $Q_{\mathrm{gB}}$
  to $(\rho_s / a)^2$, introducing the exact geometric factor $(a / R_0)^2$.

---

### 3.4 Pedestal window detection (outer-barrier HWHM + EPED1 width)

`detect_pedestal_window` locates the edge H-mode transport barrier on poloidal
flux $\psi_N \in [0.80, 0.998]$ via a finite-difference half-width at
half-maximum (HWHM) inward walk from the peak outer total thermal pressure
gradient $-dp_{\mathrm{th}}/d\psi_N$ ($p_{\mathrm{th}} = e[n_e T_e + (n_i + n_Z)T_i]$),
using the shaped LCFS poloidal perimeter $L_p = S_{\mathrm{LCFS}} / (2\pi R_{\mathrm{maj,LCFS}})$
and the **EPED1 kinetic-ballooning-mode (KBM) width scaling** (Snyder et al.
2009):

$$
\Delta\psi_{N,\mathrm{EPED1}} = \mathrm{clip}\!\big(0.076\,\beta_{p,\mathrm{ped}}^{1/2},\; 0.015,\; 0.22\big),
\qquad
\beta_{p,\mathrm{ped}} = \frac{2\mu_0\,p_{\mathrm{ped}}}{\langle B_p \rangle_{\mathrm{LCFS}}^2},
$$

with $\psi_{\mathrm{ped,top}} \in [0.72, 0.985]$ and width bounded to
$[0.6, 1.8]\,\Delta\psi_{N,\mathrm{EPED1}}$.

When the edge-to-core pressure gradient ratio exceeds `2.2` (`has_pedestal=True`),
the detected pedestal top and width (mapped to run coordinates $\hat{x}$ when
`coord="phi_n"`) set the outer plateau of $\ell(\hat{x})$ and the pedestal-top
anchor of `stiff_inward`. In L-mode (`has_pedestal=False`), `has_pedestal=False`
is propagated explicitly into `draw_bounded_perturbation` and
`sample_stiff_inward_channel`, imposing **zero** artificial edge barrier.

---

### 3.5 Tier 1 analytical critical-gradient thresholds and stiffness

When `length_scale_mode="tier1_analytical"`, `stiffness._tier1_analytical_thresholds`
evaluates closed-form linear thresholds and quasilinear stiffness (~0.2 ms per
baseline):

1. **Toroidal ETG threshold (Jenko, Dorland & Hammett 2001):**
   $$
   \left(\frac{R_0}{L_{T_e}}\right)_{\mathrm{ETG,crit}}
   = \max\!\left\{
   \left(1 + Z_{\mathrm{eff}}\frac{T_e}{T_i}\right)
   \left(1.33 + 1.91\frac{|\hat{s}|}{q}\right)(1 - 1.5\epsilon)
   \left(1 + 0.3\kappa s_\kappa\right),\;
   0.8\max\!\left(\frac{R_0}{L_{n_e}},\, 0\right)
   \right\},
   $$
   using $|\hat{s}|/q$ so reversed magnetic shear ($\hat{s} < 0$) in advanced
   tokamak / ITB cores properly raises the linear threshold.
2. **Trapped-electron-mode (TEM) threshold (Peeters et al. 2005):**
   $$
   \left(\frac{R_0}{L_{T_e}}\right)_{\mathrm{TEM,crit}}
   = \frac{0.357\sqrt{\epsilon} + 0.271}{\sqrt{\epsilon}}
   \max\!\Big(4.90 - 1.31\,\frac{R_0}{L_{n_e}} + 2.68\max(\hat{s}, 0) + \ln(1 + 20\nu_{\mathrm{eff}}),\; 1.5\Big),
   $$
   with $(R_0/L_{T_e})_{\mathrm{crit}} = \mathrm{clip}\!\big(\min[(R_0/L_{T_e})_{\mathrm{ETG,crit}}, (R_0/L_{T_e})_{\mathrm{TEM,crit}}],\, 1.2, 18.0\big)$.
3. **Toroidal ITG threshold with main-ion dilution stabilization (Romanelli
   1989; Kotschenreuther et al. 1995):**
   $$
   \left(\frac{R_0}{L_{T_i}}\right)_{\mathrm{ITG,crit}}
   = \mathrm{clip}\!\left(\max\!\left\{
   \left(1 + \frac{T_i}{f_i T_e}\right)
   \left(1.33 + 1.91\frac{|\hat{s}|}{q}\right)(1 - 1.5\epsilon)(1 + 0.3\kappa s_\kappa),\;
   0.8\max\!\left(\frac{R_0}{L_{n_i}},\, 0\right) + 0.8\left(1 + \frac{T_i}{T_e}\right)
   \right\},\; 1.2, 18.0\right),
   $$
   where $f_i = \mathrm{clip}(n_i / n_e,\, 0.1, 1.0) \le 1$ captures main-ion
   impurity dilution stabilization of ITG modes.
4. **Garbet CGM stiffness, saturating Waltz–Candy finite-$\rho_*$ relief, and
   subcritical margin (Garbet et al. 2004; Waltz, Candy & Rosenbluth 2002;
   Citrin et al. 2013):**
   With $\chi_{\mathrm{Garbet}} = q^{3/2}/\max(|\hat{s}|, 0.25)$, saturating
   finite-$\rho_*$ profile-shearing factor
   $f_{\rho_*} = \mathrm{clip}(1 - \rho_*/0.015,\, 0.25, 1.0)$ (reducing
   stiffness below gyro-Bohm at large $\rho_*$ while saturating smoothly to
   $\mathcal{O}(1)$ pure gyro-Bohm scaling as $\rho_* \to 0$ in reactor plasmas),
   and electromagnetic/fast-ion relief
   $f_{\mathrm{EM}} = 1 + (\beta_e / 0.010)(1 + 1.5\,p_{\mathrm{fast}}/p_{\mathrm{th}})$:
   $$
   \chi_{\mathrm{stiff},e} = \left(\frac{a}{R_0}\right)^2 \chi_{\mathrm{Garbet}}\,f_{\rho_*},
   \qquad
   \chi_{\mathrm{stiff},i} = \left(\frac{a}{R_0}\right)^2 \frac{1.25\,\chi_{\mathrm{Garbet}}\,f_{\rho_*}}{f_{\mathrm{EM}}}.
   $$
   Surface heat fluxes use $Q_{s,\mathrm{PB}} = -n_s e \chi_{s,0} \nabla T_{s,0}$
   when baseline $\chi_e, \chi_i$ are available, or fuel-gated Bosch–Hale D–T
   $\alpha$ power ($P_\alpha \equiv 0$ when `tritium_fraction == 0`; diluted
   $n_D n_T = f_T(1 - f_T)n_{\mathrm{fuel}}^2$ when $f_T > 0$) plus Martin et al.
   (2008) H-mode threshold scaling $P_{\mathrm{LH}}$ evaluated with the true
   chord line-averaged density $\bar{n}_e = \frac{1}{a}\int_0^a n_e(r)\,dr$.
   Allowable core excursions on $\rho_{\mathrm{tor}} \in [0.18, 0.88]$ (flatly
   extrapolated outside that core window) include the subcritical margin
   $\max(\kappa_{c,s} - \kappa_{0,s}, 0)$ whenever $\kappa_{0,s} < \kappa_{c,s}$:
   $$
   \Delta\kappa_{T_s} = \max\!\big(\kappa_{c,s} - \kappa_{0,s},\, 0\big)
   + \frac{f_{\mathrm{flux}}\,\big(Q_{s,\mathrm{PB}} / Q_{\mathrm{gB,imas}}\big)}{\chi_{\mathrm{stiff},s}}.
   $$
5. **Density peaking excursion:** Set from the collisionality scaling of H-mode
   density peaking dispersion (Angioni et al. 2007):
   $\Delta\kappa_{n_e} = \Delta\kappa_{n_i} = 0.75 + 0.35\ln(1 + 1/\max(\nu_{\mathrm{eff}}, 0.02))$.

---

### 3.6 Tier 2 standalone `TGLF-NN` and `QLKNN` flux-curve inversion

When `length_scale_mode="tier2_tglfnn"`, `bouquet/surrogates/tglfnn.py` runs a
zero-dependency pure-NumPy ONNX evaluator (`PureNumpyONNXModel` +
`TGLFNNCommittee`, supporting `Gemm`, `MatMul`, `Relu`, `Tanh`, `Sigmoid`, `Elu`,
`Gelu`, `Softplus`, and opset-compliant `Reshape`, with optional `onnxruntime`
acceleration) on the 20-member `sat2_em_d3d_azf-1_withnegD` (or `"auto"` core
`sat3_em_d3d_azf-1_withnegD`, with lazy `d3dnearedge`/`d3dedge` dispatch for
`RMIN_LOC >= 0.881`/`0.975` matching `TurbulentTransport.run_tglfnn`) ResNet
ensemble (Neiser et al. 2022, 2023), matching `FUSE` / `TurbulentTransport.jl`
to $< 7 \times 10^{-7}\;Q_{\mathrm{gB}}$ in ~5–12 ms per baseline.

#### Joint in-distribution flux-curve inversion (`solve_tglfnn_flux_excursions`)

1. **Smooth positive-driving power-balance operating point `X_pb`:**
   Scans `a/L_Te` on `[0, 10]` along the strictly positive thermal ray
   $r_{\mathrm{loc},s} = \mathrm{clip}(\max(\mathrm{RLTS}_s, 0.2)/\max(\mathrm{RLTS}_1, 0.2),\, 0.5, 2.0)$
   (zero sign-flip cliffs on radiatively hollow $T_e$ cores where $\mathrm{RLTS}_1 < 0$)
   and inverts the right-monotone lower envelope
   $q_{\mathrm{mono}}[k] = \min_{j \ge k}(Q_e + Q_i)_{\mathrm{NN}}[j]$, selecting
   the supercritical thermal branch above any low-gradient TEM trough and
   clamping continuously to the onset knee on the left and `scan_aL[-1]` on the
   right (never snapping discontinuously back to unconverged `X0`).
2. **Continuous nonlinear flux-wall & quasineutral particle inversion:**
   Evaluates thermal and particle gradient scans around `X_pb` (co-perturbing
   `RLNS_1`, `RLNS_2`, and `RLNS_3` together so `TGLF` quasineutrality
   $\sum_s Z_s\,\mathrm{AS}_s\,\mathrm{RLNS}_s = 0$ holds identically across
   $\Delta(a/L_n) \in [-4, 4]$, with smooth pinch-balance target
   $\Delta\Gamma_{\mathrm{tgt}} = f_{\mathrm{flux}}\sqrt{\Gamma_0^2 + (0.1\,\hat{Q}_{\mathrm{tot,PB}})^2} + \hat{\sigma}_{\Gamma,\mathrm{NN}}$)
   and inverts the committee-mean curve upward for
   $\Delta\hat{Q}_s = f_{\mathrm{flux}}\,\hat{Q}_{s,\mathrm{PB}} + \hat{\sigma}_{Q_s,\mathrm{NN}}$
   (using exact 20-member ensemble $\mathrm{Std}_k(Q_{e,k} + Q_{i,k})$ on the
   coupled ray).
3. **Multi-species SI Braginskii electron–ion collisional exchange coupling:**
   Local collisional heat exchange $q_{ei} = \alpha_{ei}(T_e - T_i)$ sums energy
   transfer over all thermal ion species $s$ of charge $Z_s$ and mass $A_s m_p$
   (Braginskii 1965):
   $$
   \alpha_{ei}
   = \sum_s \frac{3 m_e}{m_s}\frac{n_e e}{\tau_{es}}
   = \frac{3 m_e}{m_p}\frac{n_e e}{\tau_{e,1}}\sum_s \frac{n_s Z_s^2}{n_e A_s}
   \;\;[\mathrm{W\,m^{-3}\,eV^{-1}}],
   $$
   where $\tau_{e,1}$ is the $Z_{\mathrm{eff}} = 1$ electron collision time
   (noting that for any $A_Z = 2 Z_Z$ impurity in deuterium, quasineutrality
   gives $\sum_s n_s Z_s^2 / (n_e A_s) \equiv \frac{1}{2}$ independent of
   $Z_{\mathrm{eff}}$, avoiding a spurious factor-of-$Z_{\mathrm{eff}}$
   overestimate). `stiffness.py` blends single-species and co-moving thermal
   excursions via $\eta_s = \Delta Q_{s,\mathrm{W}} / (\Delta Q_{s,\mathrm{W}} + (V/S)\,\alpha_{ei}\,\sigma_{T_s})$.

When `length_scale_mode="tier2_qlknn"`, `bouquet/surrogates/qlknn_backend.py`
evaluates the electrostatic `QLKNN-hyper-10D` JSON committee (van de Plassche
et al. 2020) using exact CODATA electron collisionality $\nu_e = 1/\tau_e$,
summing ITG + TEM + ETG mode networks (`efeITG_GB`, `efeTEM_GB`, `efeETG_GB`,
`efiITG_GB`, `efiTEM_GB`, including `_div_` ratio committees), converting
QuaLiKiz gyro-Bohm units ($m_p, R_0, B_0, n_s T_s$) into GACODE
$Q_{\mathrm{gB,imas}}$ units, and warning when $\beta_e > 0.5\%$ enters regimes
where electromagnetic stabilization is significant.

---

### 3.7 Regional positivity and baseline-normalized core gradient bounds

`sampling.draw_bounded_perturbation` replaces global monotonicity when
`gradient_policy` is set to `"positivity_only"` or `"physics_bounds"`:

- **`"positivity_only"`:** Enforces strict positivity $f(\hat{x}) > 0$ via
  pre-factored Cholesky GP sampling (with variance-matched log-normal fallback
  and `UserWarning` if extreme relative $\sigma$ exhausts linear draws), imposing
  zero monotonicity constraints anywhere.
- **`"physics_bounds"`:** Respects `has_pedestal`:
  1. **H-mode pedestal barrier (`has_pedestal=True`, $\hat{x} \ge \hat{x}_{\mathrm{ped}}$):**
     Enforces outward decrease ($\partial f / \partial\hat{x} \le 0$) when the
     baseline itself decreases steeply across the barrier (with core bounds
     extended continuously up to the density pedestal start so no unconstrained
     shoulder gap arises). When `has_pedestal=False` (L-mode), no artificial
     edge monotonicity is imposed and core bounds span $[0, 1]$.
  2. **Antipodally symmetric baseline-normalized core gradient bounds:**
     Evaluates core gradient excursions using the **baseline-normalized** slope
     $$
     \tilde{\kappa}_{\mathrm{draw}}(x_r)
     = -\frac{R_0}{a}\frac{\partial f_{\mathrm{draw}}/\partial x_r}{\max(|f_0(x_r)|,\, f_{\mathrm{floor}})}
     \in \big[\kappa_0(x_r) - w(x_r),\; \kappa_0(x_r) + w(x_r)\big],
     $$
     with $w = \max\!\big(3.5\,\Delta\kappa_{\mathrm{allow}}, \max(\kappa_0, 0) + \Delta\kappa_{\mathrm{hollow}}\big)$.
     Because the denominator uses the deterministic baseline $|f_0(x_r)|$ rather
     than the random level $f_0 + \delta f$, the acceptance region
     $|\partial_{x_r}\delta f| \le \frac{a}{R_0} w |f_0|$ has **exact antipodal
     symmetry** under $\delta f \mapsto -\delta f$, guaranteeing **zero level
     bias AND zero slope bias** on sawtooth-flat cores while admitting pellet
     density bumps and radiatively hollow $T_e$ cores.

> [!IMPORTANT]
> **Why one-sided post-hoc `kappa_max` rejection is NOT used to enforce critical gradients:**
> Truncating the upper tail of a zero-mean GP whenever $\kappa_{\mathrm{draw}} > \kappa_c + \Delta\kappa$
> while leaving lower-gradient draws untouched creates an asymmetric truncation
> that systematically biases $\mathbb{E}[T_{\mathrm{draw}}(0)] < T_0(0)$ and
> $\mathbb{E}[P_{\mathrm{fus}}] < P_{\mathrm{fus},0}$. Critical-gradient
> stiffness must be encoded *generatively* (`sampling_mode="stiff_inward"` today,
> and [proposals/flux-matched-surrogate-draws.md](proposals/flux-matched-surrogate-draws.md)
> in Phase 2).

---

### 3.8 Stiff-core inward gradient integration and exact variance partitioning

When `sampling_mode="stiff_inward"`, `bouquet/stiff_sampler.py`
(`sample_stiff_inward_channel`) constructs each kinetic channel as a single
$C^1$-continuous logarithmic profile without mixing short-scale edge draws into
the outer core:

1. **Core inverse scale-length residual ($x \equiv r/a \le x_{\mathrm{anchor}}$):**
   Draws zero-mean GP $\delta\kappa(x)$ with correlation length
   $\ell_\kappa = \text{\texttt{stiff\_gradient\_ls}}$ in $r/a$ and marginal
   standard deviation $\sigma_\kappa(x) = \tau_{\mathrm{axis}}(x)\,\Delta\kappa_{\mathrm{allow}}(x)$,
   where $\tau_{\mathrm{axis}}(x) = x / \sqrt{x^2 + 0.04^2}$ enforces quadratic
   on-axis regularity $\delta\kappa(0) = 0$. Trapezoidal inward integration gives
   $$
   \delta J(x) = \frac{a}{R_0}\int_x^{x_{\mathrm{anchor}}} \delta\kappa(u)\,du,
   \qquad
   V_{\mathrm{stiff}}(x) \equiv \mathrm{Var}[\delta J(x)] = \| \mathbf{M}(x) \|_2^2.
   $$
2. **L-mode LCFS anchor (`has_pedestal=False`) vs. H-mode $C^1$-matched pedestal
   log-GP (`has_pedestal=True`):**
   - When `has_pedestal=False`, anchors directly at the LCFS ($x_{\mathrm{anchor}} = x_{\mathrm{LCFS}}$)
     with $W_{\mathrm{ped}} \sim \mathcal{N}(0, \sigma_{\mathrm{rel,LCFS}}^2)$ and
     zero edge barrier rejection.
   - When `has_pedestal=True`, decomposes the stationary RBF correlation on
     $\tau = \hat{\psi} - \hat{\psi}_{\mathrm{ped}} \ge 0$ ($\ell_{\mathrm{edge}} = \max(\min\ell, 10^{-3})$)
     orthogonally into its anchor value mode $\phi_0(\tau) = e^{-\tau^2/(2\ell_{\mathrm{edge}}^2)}$,
     anchor slope mode $\phi_1(\tau) = \tau\,e^{-\tau^2/(2\ell_{\mathrm{edge}}^2)}$
     ($\phi_1(0)=0, \phi_1'(0)=1$), and conditioned remainder $K_{\mathrm{rem}}(\tau, \tau')$.
     Setting the right log-slope at $\hat{\psi}_{\mathrm{ped}}^+$ equal to the
     core's left log-slope at $\hat{\psi}_{\mathrm{ped}}^-$ yields an **exact
     $C^1$ splice at $\hat{\psi}_{\mathrm{ped}}$** (`out[:i_ped + 1] = core_draw`
     directly, with zero two-draw Hermite blend zone, zero outer-core gradient
     explosion, and zero marginal variance dip). Barrier monotonicity is
     enforced strictly on steep segments where $\Delta f_0 < -10^{-3}\max|f_0|$
     (excluding flat SOL feet), and an antithetic calibration batch evaluates
     the conditioned anchor mean $\mu_{u_0}$, variance $V_{\mathrm{ped,cond}}$,
     and cumulant generating function $\psi_{\mathrm{ped}}(\beta) = \ln\mathbb{E}[e^{\beta W_{\mathrm{ped}}}\mid\mathcal{A}]$.
3. **Exact variance partitioning and cumulant-generating drift debiasing:**
   Scaling $S(x) = \beta(x)\big(W_{\mathrm{ped}} + \delta J(x)\big)$ by
   $$
   \beta(x) = \min\!\left(1,\; \frac{\sigma_{\mathrm{rel,core}}(x)}{\sqrt{V_{\mathrm{ped,cond}} + V_{\mathrm{stiff}}(x)}}\right),
   \qquad
   \Lambda_{\mathrm{drift}}(x) = \psi_{\mathrm{ped}}\big(\beta(x)\big) + \frac{1}{2}\beta(x)^2 V_{\mathrm{stiff}}(x),
   $$
   $$
   f_{\mathrm{core}}(x) = \max\!\Big(f_0(x)\,\exp\!\big(S(x) - \Lambda_{\mathrm{drift}}(x)\big),\; 0.05\max(f_0(x), 10^{-4}f_{\max})\Big),
   $$
   guarantees $\beta(x_{\mathrm{ped}}) = 1$, exact $C^1$ continuity across
   $\hat{\psi}_{\mathrm{ped}}$, $\mathrm{Var}[\ln(f_{\mathrm{core}}/f_0)] \le \sigma_{\mathrm{rel,core}}^2$,
   unbiased $\mathbb{E}[f_{\mathrm{draw}}(x)] \approx f_0(x)$ across core,
   pedestal, and SOL, deterministic single-step RNG advancement, and bit-exact
   $f_{\mathrm{draw}} \equiv f_0$ at $\sigma_f \equiv 0$.

---

### 3.9 Power-balance vs. surrogate thermal diffusivity (`chi_consistency`)

When `chi_e` / `chi_i` are present in `aux_baselines`,
`kinetic_sampler.compute_power_balance_chi` offers three physical closures:

1. **`"passive_gp"` (legacy default):** Independent GP draw on $\chi_s$,
   uncoupled from $(n_s, T_s)$.
2. **`"power_balance"` (steady-state energy conservation):** Holds conductive
   heat flux $Q_s = -n_s e \chi_s \nabla T_s$ consistent with auxiliary +
   fuel-gated DT $\alpha$ power balance:
   $$
   \chi_s^{\mathrm{draw}}(\hat{x})
   = \chi_{s,0}(\hat{x})\,M_{Q,\mathrm{aux}}(\hat{x})\,M_\alpha(\hat{x})\,
   \frac{n_{s,0}(\hat{x})}{n_s^{\mathrm{draw}}(\hat{x})}\,
   \mathcal{R}_{\nabla T}(\hat{x}),
   $$
   with smooth per-profile regularized gradient ratio
   $$
   \mathcal{R}_{\nabla T}(\hat{x})
   = \frac{\sqrt{(\partial_{\hat{x}} T_{s,0})^2 + g_{\mathrm{reg},0}^2}}
   {\sqrt{(\partial_{\hat{x}} T_s^{\mathrm{draw}})^2 + g_{\mathrm{reg},d}^2}},
   \qquad
   g_{\mathrm{reg}} = 0.10\,\big(\max T_s - \min T_s\big).
   $$
   Because $g_{\mathrm{reg}}$ scales homogeneously with each profile's span
   without clamping $\partial_{\hat{x}} T_s$ at zero, uniform scaling
   $T_s^{\mathrm{draw}} = \alpha\,T_{s,0}$ yields $\mathcal{R}_{\nabla T} \equiv 1/\alpha$
   **identically at every radial node including a parabolic/flat magnetic axis**
   ($\partial_{\hat{x}} T_s(0) = 0$), and $\mathcal{R}_{\nabla T}$ varies
   $C^\infty$-smoothly across radiatively hollow core inversions.
   When `alpha_heating_feedback=True` and `0 < tritium_fraction < 1`,
   $M_\alpha(\hat{x}) = (1 - f_{\mathrm{burn},s}) + f_{\mathrm{burn},s}(P_{\alpha,s}^{\mathrm{draw}} / P_{\alpha,s,0})$
   uses $n_D n_T = f_T(1 - f_T)n_i^2\langle\sigma v\rangle_{DT}(T_i)E_\alpha$
   (Bosch & Hale 1992), the Stix (1972) electron slowing-down fraction
   $f_{\alpha,e}(T_e) = 1 / [1 + (T_e / 33\text{ keV})^{3/2}]$, and the local
   volume-integrated burn fraction
   $f_{\mathrm{burn},s}(\hat{x}) = P_{\alpha,s,0}(\hat{x}) / \max(P_{\mathrm{cond},s,0}(\hat{x}), P_{\alpha,s,0}(\hat{x}))$
   from baseline $-n_{s,0} e \chi_{s,0} \nabla T_{s,0} S$. On pure-Deuterium
   discharges (`tritium_fraction == 0.0`, the default), $P_\alpha \equiv 0$ and
   $M_\alpha \equiv 1$.
3. **`"surrogate"` (instantaneous unrelaxed turbulent response):** Guards
   $\sigma = 0$ across all kinetic/auxiliary channels (making zero ONNX calls
   when the plasma state is unperturbed), forwards `omega_tor_rad_s` ($E \times B$
   and parallel shear) and $A_{\mathrm{main}} = 2(1 - f_T) + 3 f_T$ into
   `evaluate_tglfnn_stiffness`, and passes the regularized `TGLF-NN` surface
   heat-flux ratio $((Q_{s,\mathrm{NN}}^{\mathrm{draw}} + Q_{\mathrm{floor}}) / (Q_{s,\mathrm{NN}}^0 + Q_{\mathrm{floor}}))\,M_{Q,\mathrm{aux}}$
   directly into `compute_power_balance_chi`.

---

## 4. Physics-derived quantities vs. explicit user priors

The table below separates quantities computed from equilibrium and transport
models from the explicit user priors on `UncertaintyConfig`:

| Parameter / Profile | Source | Why |
|---|---|---|
| $\ell_{n_e}, \ell_{T_e}, \ell_{n_i}, \ell_{T_i}$ | **Physics-derived** (`stiffness.py`, `surrogates/tglfnn.py`) | Determined by ETG/TEM/ITG critical gradients or `TGLF-NN` / `QLKNN` flux-curve inversion, multi-species SI Braginskii exchange $\alpha_{ei}$, Gauss–Hermite $\ell_r \mapsto \ell_{\hat{x}}$ quadrature, and HWHM/EPED1 pedestal detection. |
| $j_{\mathrm{BS}}(\hat{x})$ | **Physics-derived** (Redl–Sauter per draw) | Computed directly from the drawn kinetic profiles $(n_e, T_e, n_i, T_i, Z_{\mathrm{eff}})$ and live flux-surface geometry; automatically inherits H-mode pedestal gradient structure. |
| `tritium_fraction` (default `None` $\to$ IMAS $Z=1$ auto or `0.0` pure D) | **Source metadata / explicit user prior** | GEQDSK + p-file / IDA reconstructions carry lumped hydrogenic $n_i$ without D/T isotope ratios; defaults safely to `0.0` (pure Deuterium, $P_\alpha \equiv 0$) unless IMAS `core_profiles` specifies $A \approx 3$ tritium or the user sets `tritium_fraction` (e.g., `0.5` for 50–50 D–T). |
| `aux_length_scales['zeff']` (default `0.4`) | **Explicit user prior** | `TGLF-NN` and `QLKNN` predict `(OUT_G_elec, OUT_P_ions, OUT_Q_elec, OUT_Q_ions)` for a lumped impurity and output **no** turbulent impurity flux $\Gamma_Z$ (note that `FACIT`, Fajardo et al. 2022, models *collisional/neoclassical* impurity transport rather than turbulent $\Gamma_Z$). |
| `j_ls` (default `0.25`) vs. `current_evolution_time_s` | **Explicit user prior** (or Sauter neoclassical resistive skin depth when $\Delta t > 0$) | `j_ls` governs perturbations to **inductive** current $j_{\mathrm{inductive}}$ (parabolic magnetic diffusion, no gyrokinetic critical-gradient threshold). Kept at `j_ls` unless `current_evolution_time_s` is supplied, which activates $\ell_r = \sqrt{\Delta t / (\mu_0 \sigma_{\mathrm{neo}})}$ (Sauter et al. 1999) with smooth `tanh` $q \le 1$ sawtooth blending. |
| `flux_uncertainty_rel` (default `0.30`) | **Explicit user prior** | Fractional uncertainty $\Delta Q / Q$ in surface heating/particle flux when inverting critical-gradient or `TGLF-NN` flux curves (`0.0` uses only the 20-member `TGLF-NN` committee epistemic spread). |
| `stiff_gradient_ls` (default `0.30`) | **Explicit user prior** | Minor-radius ($r/a$) correlation length of core inverse-scale-length residuals $\delta(R_0/L_s)$ in `sampling_mode="stiff_inward"`. |

---

## 5. Workflow selection: diagnostic UQ vs. predictive burning-plasma UQ

| Workflow | Recommended Settings | Statistical Interpretation |
|---|---|---|
| **Legacy v1.4.x reproduction** | `length_scale_mode="manual"`, `sampling_mode="gibbs_level"`, `gradient_policy="legacy"`, `chi_consistency="passive_gp"` | Seed-for-seed bit-identical with `bouquet` v1.4.1. |
| **Experimental diagnostic UQ** (e.g., Thomson + CER / IDA reconstruction on DIII-D/AUG/JET, pellet or radiative-core discharges) | `length_scale_mode="tier2_tglfnn"` (or `"tier1_analytical"`), `gibbs_ls_form="chilenski_tanh"`, `sampling_mode="gibbs_level"`, `gradient_policy="physics_bounds"`, `chi_consistency="power_balance"`, `tritium_fraction=0.0` | Propagates pointwise diagnostic error bars without forcing a pedestal-to-core transport-model lock (Chilenski et al. 2015; Ho et al. 2019), while removing flat-core truncation bias and admitting hollow/pellet cores. |
| **Predictive burning-plasma / stiff H-mode UQ** (e.g., SPARC, ARC, ITER, FUSE DT scenario sensitivity) | `length_scale_mode="tier2_tglfnn"`, `sampling_mode="stiff_inward"`, `gradient_policy="physics_bounds"`, `chi_consistency="power_balance"`, `alpha_heating_feedback=True`, `tritium_fraction=0.5` | Enforces $C^1$ pedestal-to-core stiffness propagation ($\mathrm{Corr}(T_0, T_{\mathrm{ped}}) \approx 0.7\text{--}0.8$), bounds $\Delta(R_0/L_T)$ to `TGLF-NN` flux-curve excursions without variance double-counting, and updates $\chi_e, \chi_i$ with Bosch–Hale + Stix DT $\alpha$-power feedback. |
| **Phase-2 nonlinear flux-matched scenario UQ** *(proposed)* | `sampling_mode="flux_matched"` ([proposals/flux-matched-surrogate-draws.md](proposals/flux-matched-surrogate-draws.md)) | Solves globally coupled forward 3-channel flux balance per draw across the 20-member `TGLF-NN` committee + `CGYRO`/`PORTALS` threshold-offset calibration. |

---

## 6. Assumptions, limitations, and open investigations

1. **Quasilinear surrogate accuracy vs. nonlinear gyrokinetics (`CGYRO` / `PORTALS`):**
   `TGLF-SAT2` (Staebler et al. 2021a, 2021b) and its neural-network committee
   `TGLF-NN` (Neiser et al. 2022, 2023) reproduce nonlinear `CGYRO` core
   critical gradients within ~10–15% across standard aspect-ratio tokamaks and
   agree closely with nonlinear `PORTALS-CGYRO` core temperature predictions on
   SPARC and DIII-D (Rodriguez-Fernandez et al. 2022, 2024). Known regimes
   carrying larger quasilinear epistemic error include high-$\beta_e$
   electromagnetic KBM/MTM transitions, spherical tokamaks, strong energetic-$\alpha$
   nonlinear stabilization in deep burning cores, and near-axis
   ($\rho_{\mathrm{tor}} < 0.18$) or pedestal ($\rho_{\mathrm{tor}} > 0.88$)
   zones where local flux-tube gyrokinetics is superseded by axis regularity and
   HWHM/EPED1 barrier physics.
2. **Linearized stiffness variance vs. exact per-draw nonlinear flux balance:**
   `sampling_mode="stiff_inward"` draws symmetric Gaussian gradient residuals
   $\delta\kappa(r)$ scaled to $\Delta\kappa_{\mathrm{allow}}(r)$. Exact
   nonlinear asymmetric threshold enforcement requires solving coupled forward
   flux balance per draw ([proposals/flux-matched-surrogate-draws.md](proposals/flux-matched-surrogate-draws.md)).
3. **Open investigation — `SLAYER` rational-surface drift-tearing layer closures:**
   In `SLAYER`'s 4-field two-fluid drift-magnetohydrodynamic resonant layer
   model (Cole & Fitzpatrick 2006; Fitzpatrick 2022), kinetic profile draws at
   rational surfaces ($q = 3/2, 2/1$) govern linear tearing stability $\Delta'$
   screening and error-field penetration primarily through:
   - the **electron diamagnetic frequency**
     $\omega_{*e} \propto -\frac{1}{e B n_e}\frac{dp_e}{dr}$,
   - the **parallel Spitzer resistivity / Lundquist number**
     $\eta_\parallel \propto Z_{\mathrm{eff}}\,T_e^{-3/2}$, and
   - **anomalous perpendicular layer transport closures** ($\chi_e$,
     perpendicular particle diffusivity $D_\perp$, and perpendicular ion
     viscosity $\mu_\perp$, typically $\mu_\perp \sim \chi_i$).
   Because equilibria in a `bouquet` archive represent stationary force- and
   energy-balanced states at fixed auxiliary heating power (rather than
   transient heat pulses), `chi_consistency="power_balance"` ($\chi_e$
   negatively correlated with $|\nabla T_e|$ at fixed heating power) is the
   energetically self-consistent closure, whereas `chi_consistency="surrogate"`
   ($\chi_e$ positively correlated with $|\nabla T_e|$ above critical gradient)
   implies $3\text{--}15\times$ uncompensated heating-power variations across
   draws and serves as an upper sensitivity bracket (`experimental.REGISTRY["power_balance_chi"]`).

---

## 7. `TGLF-NN` and `QLKNN` licensing and distribution policy

- **Upstream `TGLF-NN` license (`TurbulentTransport.jl` / General Atomics):**
  Released under the **Apache License 2.0** (`Copyright (c) 2024 General Atomics`),
  accompanied by a mandatory attribution `NOTICE.md` file.
- **Exact FSF compatibility with `bouquet` (`LGPL-3.0`):**
  1. **Zero bundled weights in `bouquet` git today:** `bouquet` commits **zero**
     Apache-2.0 `.onnx` weight files; `bouquet/surrogates/tglfnn.py` loads
     external model files at runtime as pure numerical data from user-specified
     paths, `BOUQUET_TGLFNN_DIR`, `~/.cache/bouquet/models/`, or local
     `TurbulentTransport.jl` installations, leaving `bouquet` cleanly under
     **`LGPL-3.0`**.
  2. **Standalone runtime data vs. single derivative work:** Under Free Software
     Foundation (FSF) licensing rules, Apache-2.0 is one-way compatible with
     **`GPL-3.0`** (and combining Apache-2.0 material into a *single linked
     derivative program* with `LGPL-3.0` code would require relicensing that
     combined derivative under `GPL-3.0` per `LGPL-3.0` §2(b), *not* pure
     `LGPL-3.0`). Therefore, any cached or companion distribution of `.onnx`
     model ensembles must keep the model files as **separate standalone runtime
     data files** (mere aggregation, never compiled into `bouquet` Python
     modules) accompanied by the verbatim Apache-2.0 `LICENSE` and `NOTICE.md`
     files and respecting the General Atomics non-endorsement trademark clause,
     preserving `bouquet` itself under `LGPL-3.0`.

---

## 8. Recommended pull-request structure

The implementation on `feat/physics-profile-sampling` decomposes into a clean
3-PR stack (with Phase 2 scoped as a follow-on PR):

1. **PR 1 — Miller geometry & standalone quasilinear surrogate backends
   (`bouquet/geometry/miller.py`, `bouquet/surrogates/`, `scripts/compute_transport_stiffness.jl`, `tests/test_surrogates.py`):**
   Pure-Python/NumPy Miller flux-surface extraction (IMAS/OMAS + COCOS-aware
   GEQDSK + signed $B_0, q$), GACODE `vprime_miller` gyro-Bohm normalizations,
   pure-NumPy ONNX `TGLF-NN` 20-member committee evaluator + smooth
   `solve_tglfnn_flux_excursions`, and multi-mode `QLKNN-10D` evaluator.
2. **PR 2 — Critical-gradient stiffness length scales & regional gradient bounds
   (`bouquet/stiffness.py`, `bouquet/sampling.py`, `tests/test_stiffness_and_hollow.py`):**
   `detect_pedestal_window` (HWHM + EPED1), `coord="phi_n"`-consistent Tier 1 &
   Tier 2 `compute_physics_length_scales` (`gibbs_ls_form="chilenski_tanh"` |
   `"pointwise"`, `tritium_fraction`), Paciorek–Schervish PSD `matern52`, and
   antipodally symmetric `draw_bounded_perturbation` (`has_pedestal`-aware).
3. **PR 3 — Stiff-core inward sampler, power-balance $\chi$, & draw-engine wiring
   (`bouquet/stiff_sampler.py`, `bouquet/kinetic_sampler.py`, `bouquet/config.py`,
   `bouquet/baseline.py`, `bouquet/engine_draws.py`, `bouquet/TokaMaker_interface.py`,
   `bouquet/swb_draws.py`, `bouquet/swb.py`, `bouquet/run.py`, `bouquet/experimental.py`,
   `bouquet/__init__.py`, `docs/`):**
   $C^1$-slope-matched `sample_stiff_inward_channel`, fuel-gated
   `compute_power_balance_chi` with Bosch–Hale + Stix DT $\alpha$ feedback,
   hardened `UncertaintyConfig` validation, experimental registry entries, and
   technical documentation.
4. **PR 4 *(Phase 2, follow-on)* — Generative flux-matched surrogate draws
   ([proposals/flux-matched-surrogate-draws.md](proposals/flux-matched-surrogate-draws.md)):**
   `sampling_mode="flux_matched"` globally coupled $42 \times 42$ forward
   `TGLF-NN` committee solver with threshold-offset `CGYRO`/`PORTALS`
   calibration.

---

## 9. References

1. **Angioni, C. et al.** (2007). Scaling of density peaking in H-mode plasmas
   based on a combined database of AUG and JET observations. *Nucl. Fusion*
   **47**, 1326–1335. `doi:10.1088/0029-5515/47/9/033`
2. **Bosch, H.-S. & Hale, G. M.** (1992). Improved formulas for fusion
   cross-sections and thermal reactivities. *Nucl. Fusion* **32**, 611–631.
   `doi:10.1088/0029-5515/32/4/I07`
3. **Candy, J.** (2009). A unified method for operator evaluation in local
   Grad–Shafranov plasma equilibria. *Plasma Phys. Control. Fusion* **51**,
   105009. `doi:10.1088/0741-3335/51/10/105009`
4. **Candy, J., Holland, C., Waltz, R. E., Fahey, M. R. & Belli, E.** (2009).
   Tokamak profile prediction using direct gyrokinetic and neoclassical
   simulation (`TGYRO`). *Phys. Plasmas* **16**, 060704.
   `doi:10.1063/1.3167820`
5. **Chilenski, M. A., Greenwald, M., Marzouk, Y., Howard, N. T., White, A. E.,
   Rice, J. E. & Walk, J. R.** (2015). Improved profile fitting and
   quantification of uncertainty in experimental measurements of impurity
   transport coefficients using Gaussian process regression. *Nucl. Fusion*
   **55**, 023012. `doi:10.1088/0029-5515/55/2/023012`
6. **Citrin, J. et al.** (2013). Nonlinear stabilization of tokamak
   microturbulence by fast ions. *Phys. Rev. Lett.* **111**, 155001.
   `doi:10.1103/PhysRevLett.111.155001`
7. **Cole, A. & Fitzpatrick, R.** (2006). Drift-magnetohydrodynamical model of
   error-field penetration in tokamak plasmas. *Phys. Plasmas* **13**, 032503.
   `doi:10.1063/1.2178167`
8. **Creely, A. J. et al.** (2020). Overview of the SPARC tokamak, *J. Plasma
   Phys.* **86**, 865860502 (`doi:10.1017/S0022377820001257`); and **Hughes,
   J. W., Howard, N. T., Rodriguez-Fernandez, P., Kuang, A. Q., Tolman, E. A.,
   Creely, A. J. & Snyder, P. B.** (2020). High confinement access and edge
   pedestal structure in the SPARC tokamak, *J. Plasma Phys.* **86**, 865860504
   (`doi:10.1017/S0022377820001300`).
9. **Fajardo, D., Angioni, C., Dux, R. & Manas, P.** (2022). Analytical model
   for collisional impurity transport in tokamaks at arbitrary collisionality
   (`FACIT`). *Plasma Phys. Control. Fusion* **64**, 044002.
   `doi:10.1088/1361-6587/ac5b4d`
10. **Fitzpatrick, R.** (2022). Influence of anomalous perpendicular transport on
    linear tearing mode dynamics in tokamak plasmas. *Phys. Plasmas* **29**,
    032507. `doi:10.1063/5.0082331`
11. **Garbet, X., Mantica, P., Ryter, F. et al.** (2004). Profile stiffness and
    global confinement. *Plasma Phys. Control. Fusion* **46**, 1351–1373.
    `doi:10.1088/0741-3335/46/9/002`
12. **Gibbs, M. N.** (1997). *Bayesian Gaussian Processes for Regression and
    Classification*. Ph.D. thesis, University of Cambridge.
13. **Ho, A., Citrin, J., Auriemma, F. et al.** (2019). Application of Gaussian
    process regression to plasma turbulent transport model validation via
    integrated modelling. *Nucl. Fusion* **59**, 056007.
    `doi:10.1088/1741-4326/ab065a`
14. **Jenko, F., Dorland, W. & Hammett, G. W.** (2001). Critical gradient
    formula for toroidal electron temperature gradient modes. *Phys. Plasmas*
    **8**, 4096–4104. `doi:10.1063/1.1391261`
15. **Kotschenreuther, M., Dorland, W., Beer, M. A. & Hammett, G. W.** (1995).
    Quantitative predictions of tokamak energy confinement from
    first-principles simulations with kinetic effects. *Phys. Plasmas* **2**,
    2381–2389. `doi:10.1063/1.871261`
16. **Martin, Y. R., Takizuka, T. & ITPA CDBM H-mode Threshold Database Working
    Group** (2008). Power requirement for accessing the H-mode in ITER. *J.
    Phys.: Conf. Ser.* **123**, 012033. `doi:10.1088/1742-6596/123/1/012033`
17. **Meneghini, O., Smith, S. P., Snyder, P. B., Staebler, G. M., Candy, J.,
    Belli, E., Lao, L., Kostuk, M., Luce, T., Luda, T., Park, J. M. & Poli, F.**
    (2017). Self-consistent core-pedestal transport simulations with neural
    network accelerated models. *Nucl. Fusion* **57**, 086034.
    `doi:10.1088/1741-4326/aa7776`
18. **Neiser, T., Meneghini, O., Smith, S. P., McClenaghan, J., Staebler, G. M.,
    Belli, E. A., Candy, J. et al.** (2022/2023). Database generation and
    deep-ensemble neural network surrogates for electromagnetic TGLF-SAT2
    (`TGLF-NN`). APS Division of Plasma Physics / General Atomics
    `TurbulentTransport.jl` model suite.
19. **Paciorek, C. J. & Schervish, M. J.** (2006). Spatial modelling using a new
    class of nonstationary covariance functions. *Environmetrics* **17**,
    483–506. `doi:10.1002/env.785`
20. **Peeters, A. G., Angioni, C., Apostoliceanu, M., Jenko, F., Ryter, F. & the
    ASDEX Upgrade Team** (2005). Linear gyrokinetic stability calculations of
    electron heat dominated plasmas in ASDEX Upgrade. *Phys. Plasmas* **12**,
    022505. `doi:10.1063/1.1848111`
21. **Redl, A., Angioni, C., Belli, E., Sauter, O. et al.** (2021). A new set of
    analytical formulae for the computation of the bootstrap current and the
    neoclassical conductivity in tokamaks. *Phys. Plasmas* **28**, 022502.
    `doi:10.1063/5.0012664`
22. **Rodriguez-Fernandez, P., Howard, N. T., Greenwald, M. J. et al.** (2020).
    Predictions of core plasma performance for the SPARC tokamak. *J. Plasma
    Phys.* **86**, 865860503. `doi:10.1017/S0022377820001075`
23. **Rodriguez-Fernandez, P., Howard, N. T. & Candy, J.** (2022). Nonlinear
    gyrokinetic predictions of SPARC burning plasma profiles enabled by
    surrogate modeling. *Nucl. Fusion* **62**, 076036.
    `doi:10.1088/1741-4326/ac64b2`
24. **Rodriguez-Fernandez, P., Howard, N. T., Saltzman, A. et al.** (2024).
    Enhancing predictive capabilities in fusion burning plasmas through
    surrogate-based optimization in core transport solvers (`PORTALS`), *Nucl.
    Fusion* **64**, 076034 (`doi:10.1088/1741-4326/ad4b3d`); and Core
    performance predictions in projected SPARC first-campaign plasmas with
    nonlinear CGYRO, *Phys. Plasmas* **31**, 062501 (`doi:10.1063/5.0209752`).
25. **Romanelli, F.** (1989). Ion temperature-gradient-driven modes and
    anomalous ion transport in tokamaks. *Phys. Fluids B* **1**, 1018–1025.
    `doi:10.1063/1.859023`
26. **Sauter, O., Angioni, C. & Lin-Liu, Y. R.** (1999). Neoclassical
    conductivity and bootstrap current formulas for general axisymmetric
    equilibria and arbitrary collisionality regime. *Phys. Plasmas* **6**,
    2834–2839. `doi:10.1063/1.873240`
27. **Snyder, P. B., Groebner, R. J., Leonard, A. W., Osborne, T. H. & Wilson,
    H. R.** (2009). Development and validation of a predictive model for the
    pedestal height, *Phys. Plasmas* **16**, 056118 (`doi:10.1063/1.3122146`);
    and **Snyder, P. B., Groebner, R. J., Hughes, J. W. et al.** (2011). A
    first-principles predictive model of the pedestal height and width:
    development, testing and ITER optimization with the EPED model, *Nucl.
    Fusion* **51**, 103016 (`doi:10.1088/0029-5515/51/10/103016`).
28. **Staebler, G. M., Candy, J., Belli, E. A., Kinsey, J. E., Bonanomi, N. &
    Patel, B.** (2021a). Geometry dependence of the fluctuation intensity in
    gyrokinetic turbulence (`TGLF-SAT2`), *Plasma Phys. Control. Fusion* **63**,
    015013 (`doi:10.1088/1361-6587/abc861`); and **Staebler, G. M., Belli, E. A.,
    Candy, J., Kinsey, J. E., Dudding, H. & Patel, B.** (2021b). Verification of
    a quasi-linear model for gyrokinetic turbulent transport, *Nucl. Fusion*
    **61**, 116007 (`doi:10.1088/1741-4326/ac243a`).
29. **Stix, T. H.** (1972). Heating of toroidal plasmas by neutral injection.
    *Plasma Phys.* **14**, 367–384. `doi:10.1088/0032-1028/14/4/002`
30. **van de Plassche, K. L., Citrin, J., Bourdelle, C., Camenen, Y., Casson,
    F. J., Dagnelie, V. I., Felici, F., Ho, A. & Van Mulders, S.** (2020). Fast
    modeling of turbulent transport in fusion plasmas using neural networks
    (`QLKNN`). *Phys. Plasmas* **27**, 022310. `doi:10.1063/1.5134126`
31. **Waltz, R. E., Candy, J. & Rosenbluth, M. N.** (2002). Gyrokinetic
    turbulence simulation of profile shear stabilization and broken gyroBohm
    scaling. *Phys. Plasmas* **9**, 1938–1946. `doi:10.1063/1.1448830`
