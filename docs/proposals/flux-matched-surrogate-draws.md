# Proposal — generative flux-matched surrogate draws (`sampling_mode="flux_matched"`)

> **Status:** scoping / design (Phase-2 follow-on to `feat/physics-profile-sampling`).
> **Target branch:** `feat/flux-matched-draws` off `feat/physics-profile-sampling` / `main`.
> **Date:** 2026-11-21.
> **Related:** [`../physics-profile-sampling.md`](../physics-profile-sampling.md), `bouquet/surrogates/tglfnn.py`, `bouquet/stiff_sampler.py`.

## 1. Problem

`feat/physics-profile-sampling` introduced two major improvements over stationary
level-space GP perturbations:

1. **Tier 1 / Tier 2 physics length scales (`bouquet/stiffness.py`):**
   Determines $\Delta(R_0/L_s)_{\mathrm{allow}}$ from analytical ETG/TEM/ITG
   thresholds or nonlinear `TGLF-NN` flux-curve excursions around power balance;
2. **Stiff-core inward integration (`sampling_mode="stiff_inward"`):**
   Integrates zero-mean Gaussian inverse scale-length perturbations
   $\delta\kappa(r)$ inward from $\hat{\psi}_{\mathrm{ped,top}}$ with exact
   pedestal + core variance partitioning (`bouquet/stiff_sampler.py`).

However, `stiff_inward` remains a **symmetric local Gaussian approximation** in
gradient space around the baseline profile $\kappa_0(r)$. In a stiff burning
plasma (e.g., compact high-field DT H-modes where $Q_{\mathrm{PB}} / Q_{\mathrm{gB,imas}} \ll 1$),
the underlying turbulent transport relation $Q(\kappa)$ is **strongly
nonlinear and asymmetric** around the critical gradient $\kappa_c$:

- **Below threshold ($\kappa < \kappa_c$):** Turbulent transport vanishes down
  to small neoclassical levels ($Q \approx Q_{\mathrm{neo}} \ll Q_{\mathrm{gB,imas}}$),
  so shallow or flat gradients (e.g., sawtooth-mixed inner cores or radiative
  core cooling) require very little sink power.
- **Above threshold ($\kappa > \kappa_c$):** Turbulent heat flux rises steeply
  as a stiff wall, $Q(\kappa) \approx Q_{\mathrm{gB,imas}}\,\chi_{\mathrm{stiff}}(\kappa - \kappa_c)_+^\alpha$.
  On an empirical baseline that was not pre-converged with a gyrokinetic solver,
  or on upper-tail Gaussian draws $+2\sigma_\kappa$, a symmetric gradient draw
  can still step past the nonlinear threshold knee into states where `TGLF-NN`
  predicts $10\times$ the available auxiliary + fusion $\alpha$ heating power.

Crucially, **post-hoc upper-tail gradient rejection (`kappa_max` filtering)
cannot be used to fix this**: clipping draws with $\kappa > \kappa_{\max}$ while
retaining draws with $\kappa < \kappa_0$ truncates one side of a symmetric
distribution and biases the ensemble mean downward
($\mathbb{E}[T_{\mathrm{draw}}(0)] < T_0(0)$ and $\mathbb{E}[P_{\mathrm{fus}}] < P_{\mathrm{fus},0}$).

At the same time, running nonlinear gyrokinetic flux-matching solvers such as
`PORTALS-CGYRO` (Rodriguez-Fernandez et al. 2022, 2024) on every draw of a
100-member equilibrium bouquet (~$10^3$ GPU-hours per profile) is intractable
for routine MHD/tearing sensitivity studies.

## 2. Goal

Provide an opt-in generative transport-solver mode,
`UncertaintyConfig.sampling_mode = "flux_matched"`, for **predictive scenario
uncertainty quantification** that:

1. **Samples the physical drivers rather than arbitrary core wiggles:**
   - Pedestal-top boundary conditions $(n_{e,\mathrm{ped}}, T_{e,\mathrm{ped}}, T_{i,\mathrm{ped}}, Z_{\mathrm{eff}})$
     from the resolved edge uncertainty envelope;
   - Auxiliary heating, radiative sink, and fueling perturbations
     $(\delta P_{\mathrm{aux}}, \delta P_{\mathrm{rad}}, \delta \hat{\Gamma}_e)$;
   - **Surrogate & gyrokinetic epistemic model uncertainty**, combining random
     member selection from the 20-member `TGLF-NN` ResNet deep ensemble with a
     calibrated quasilinear-to-nonlinear (`TGLF-SAT2` vs. `CGYRO`) critical-gradient
     threshold-offset process;
2. **Solves globally coupled forward 3-channel flux balance**
   $(Q_e, Q_i, \Gamma_e)_{\mathrm{NN+neo}} = (Q_e, Q_i, \Gamma_e)_{\mathrm{PB}}^{\mathrm{draw}}$
   with self-consistent fuel-gated Bosch–Hale DT $\alpha$ heating and
   multi-species Braginskii electron–ion exchange per draw;
3. **Executes in ~10–25 ms per draw in pure Python/NumPy** (< 0.05% of a single
   free-boundary `TokaMaker` Grad–Shafranov solve) with **zero post-hoc
   rejection bias**; and
4. Leaves experimental diagnostic UQ (`"gibbs_level"`) and variance-partitioned
   inward GP sampling (`"stiff_inward"`) untouched.

## 3. Literature verdict: how accurate is `TGLF-NN` vs. `CGYRO` / `PORTALS`?

Published multi-code verification establishes a three-tier error budget:

| Fidelity Gap | Published Magnitude | Source & How `"flux_matched"` Captures It |
|---|---|---|
| **`TGLF-NN` vs. exact `TGLF-SAT2`** | ~2–5% RMS on $\kappa_c$ and $T(0)$ in-distribution | **Neiser et al. (2022/2023); Meneghini et al. (2017):** Captured directly by drawing an ensemble member $m \sim \mathrm{Uniform}\{1, \dots, 20\}$ per bouquet draw (deep-ensemble epistemic sampling, as used in `FUSE` / `STEP` UQ). |
| **Quasilinear `TGLF-SAT2` vs. nonlinear `CGYRO`** | ~10–15% on $\kappa_c$ and $W_{\mathrm{th}}$ in standard aspect-ratio ITG/TEM/ETG cores | **Staebler et al. (*Plasma Phys. Control. Fusion* 63, 015013, 2021 & *Nucl. Fusion* 61, 116007, 2021); Rodriguez-Fernandez et al. (*Nucl. Fusion* 62, 076036, 2022 & *Nucl. Fusion* 64, 076034, 2024):** `PORTALS-TGLF` (SAT2) matches nonlinear `PORTALS-CGYRO` fusion power and core $T_e, T_i$ within ~10–15% on SPARC PRD and DIII-D H-modes. Modeled explicitly via a smooth Gaussian-process critical-gradient threshold offset $\Delta\kappa_{\mathrm{GK},s}(r)$ (`gk_epistemic_rel = 0.10` default). |
| **Out-of-regime physics** (high-$\beta_e$ KBM/MTM, spherical tokamaks, strong energetic-$\alpha$ nonlinear stabilization, $\rho_{\mathrm{tor}} < 0.15$ or $> 0.88$) | Larger (> 20%) if uncalibrated | **Citrin et al. (2013); Staebler et al. (2021):** Guarded by restricting flux-matching to $\rho_{\mathrm{tor}} \in [0.18, \rho_{\mathrm{ped,top}}]$, applying sawtooth/axis regularity inside $\rho_{\mathrm{tor}} < 0.18$, using HWHM/EPED1 barrier physics in the pedestal, and emitting an explicit out-of-distribution warning when committee relative spread or $\beta_e$ exceeds training bounds. |

**Bottom line:** While a *single deterministic* `TGLF-NN` solve can differ from
nonlinear `CGYRO` by ~10–15% in $\kappa_c$, a **20-member `TGLF-NN` committee
augmented with a 10% `CGYRO`-calibrated epistemic $\kappa_c$ threshold offset**
brackets the nonlinear gyrokinetic solution manifold at $10^{-7}\times$ the
compute cost while preventing unphysical supercritical core gradients.

## 4. Mathematical & numerical design

### 4.1 Generative per-draw random variables & threshold-offset epistemic shift

Each call to `sample_flux_matched_kinetics(kb, rng, ...)` draws four independent
physical realizations:

1. **Edge / pedestal realization ($\hat{\psi} \ge \hat{\psi}_{\mathrm{ped,top}}$):**
   Drawn on $(n_e, T_e, T_i, Z_{\mathrm{eff}})$ (with $n_i$ from quasineutrality
   when `ni_from_zeff=True`), fixing the boundary values at $\rho_{\mathrm{ped,top}}$.
2. **Surrogate committee member:**
   Draw $m \in \{0, \dots, M - 1\}$ uniformly from the loaded 20-member
   `TGLFNNCommittee` (single neural-network weight set per draw, preserving
   smooth radial flux curves within each draw).
3. **Threshold-offset quasilinear-to-nonlinear epistemic calibration field:**
   Draw smooth zero-mean GP fields $\delta_{\mathrm{GK},e}(\rho), \delta_{\mathrm{GK},i}(\rho)$
   on $\rho \in [0, 1]$ with radial correlation scale $\ell_{\mathrm{GK}} = 0.40$
   and standard deviation $\sigma_{\mathrm{GK}} = \text{\texttt{gk\_epistemic\_rel}}$
   (default `0.10`), shifting **only the thermal inverse scale lengths by an
   additive fraction of the baseline critical gradient**:
   $$
   \tilde{u}_{T_s,k} = u_{T_s,k} - \delta_{\mathrm{GK},s}(\rho_k)\,(a / L_{T_s})_{\mathrm{crit},0}(\rho_k),
   \qquad
   \tilde{u}_{n_e,k} = u_{n_e,k},
   $$
   with `RLTS_3` co-shifted identically with `RLTS_2` and `RLNS_2, RLNS_3`
   computed from `RLNS_1` and the drawn $Z_{\mathrm{eff}}(\rho)$ and $z_{\mathrm{fast}}(\rho)$
   so `TGLF` quasineutrality $\sum_s Z_s\,\mathrm{AS}_s\,\mathrm{RLNS}_s = 0$
   holds identically! Unlike multiplicative scaling $\boldsymbol{\kappa}/(1 + \delta_{\mathrm{GK}})$,
   an additive threshold offset shifts $\kappa_{c,s}$ by $\pm 10\%$ **without**
   altering post-threshold stiffness $\chi_{\mathrm{stiff}} = \partial Q / \partial\kappa$,
   distorting zero-flux convective density peaking $a/L_{n_e} = -a V/D$,
   inverting negative pellet density gradients ($a/L_{n_e} < 0$), or breaking
   species quasineutrality.
4. **Broad-scale auxiliary / sink and particle-source perturbations:**
   Draw smooth log-normal multipliers $M_{Q_e}(\rho), M_{Q_i}(\rho)$ (`flux_uncertainty_rel`,
   default `0.30`) and additive gyro-Bohm particle-flux residual
   $\delta\hat{\Gamma}_e(\rho) \sim \mathcal{GP}(0, \sigma_{\hat{\Gamma}}^2)$
   (additive in $\Gamma_{\mathrm{gB,imas}}$ units so perturbations remain
   well-defined around steady-state $\Gamma_{e,\mathrm{PB}} = 0$).

### 4.2 Collocation grid, metric-consistent normalizations & nonsingular residual

Following `TGYRO` (Candy et al. 2009, §III) and `PORTALS` (Rodriguez-Fernandez
et al. 2022, 2024), define $K = 14$ radial collocation knots on normalized
toroidal minor radius $\rho_k \in [\rho_{\mathrm{inner}}, \rho_{\mathrm{ped,top}}]$
with $\rho_{\mathrm{inner}} = \max(0.18, \rho_{q=1})$. At each knot $k$, the
unknown vector is $\mathbf{u}_k = (a/L_{n_e,k},\; a/L_{T_e,k},\; a/L_{T_i,k})^\top \in \mathbb{R}^3$
($3K = 42$ total unknowns $\mathbf{u} \in \mathbb{R}^{42}$).

Given $\mathbf{u}_{1..K}$, trapezoidal inward integration from
$(n_{e,\mathrm{ped}}, T_{e,\mathrm{ped}}, T_{i,\mathrm{ped}})$ determines
$(n_e, T_e, n_i, T_i)(\rho_k)$ for all $k \le j$, enclosed fuel-gated Bosch–Hale
fusion $\alpha$ power $P_\alpha^{\mathrm{draw}}(\rho_k)$ (`0` when
`tritium_fraction == 0`), Stix slowing-down fraction $f_{\alpha,e}(T_e)$, and
multi-species Braginskii collisional exchange
$P_{ei}^{\mathrm{draw}}(\rho_k) = \int_0^{V(\rho_k)} \alpha_{ei}(T_e - T_i)\,dV$:

$$
Q_{e,\mathrm{PB}}^{\mathrm{draw}}(\rho_k)
= \frac{M_{Q_e}(\rho_k)\,P_{e,\mathrm{aux,0}}(\rho_k) + f_{\alpha,e} P_\alpha^{\mathrm{draw}}(\rho_k) - P_{\mathrm{rad}}(\rho_k) - P_{ei}^{\mathrm{draw}}(\rho_k)}{S(\rho_k)},
$$

$$
Q_{i,\mathrm{PB}}^{\mathrm{draw}}(\rho_k)
= \frac{M_{Q_i}(\rho_k)\,P_{i,\mathrm{aux,0}}(\rho_k) + (1 - f_{\alpha,e}) P_\alpha^{\mathrm{draw}}(\rho_k) + P_{ei}^{\mathrm{draw}}(\rho_k)}{S(\rho_k)},
$$

$$
\Gamma_{e,\mathrm{PB}}^{\mathrm{draw}}(\rho_k)
= \frac{\int_0^{V(\rho_k)} S_{e,0}\,dV}{S(\rho_k)} + \Gamma_{\mathrm{gB,imas}}^{\mathrm{draw}}(\rho_k)\,\delta\hat{\Gamma}_e(\rho_k),
$$

where $S_{e,0} \equiv 0$ (zero net core particle flux $\Gamma_{e,\mathrm{PB},0} = 0$)
when no core NBI/pellet particle source is present on `Baseline`.

**Metric-consistent GACODE flux normalizations (`vprime_miller`):** Physical
surface fluxes $P_s / S(\rho_k)$ $[\mathrm{W/m^2}]$ and $\Gamma_e$
$[\mathrm{m^{-2}\,s^{-1}}]$ are normalized by the Miller flux-surface-averaged
gyro-Bohm scales (`miller.py:gyrobohm_flux_scales`, `tglfnn.py`):

$$
Q_{\mathrm{gB,imas}}^{\mathrm{draw}}(\rho_k)
= Q_{\mathrm{gB}}^{\mathrm{draw}}(\rho_k)\cdot\texttt{vprime\_miller}(\rho_k),
\qquad
\Gamma_{\mathrm{gB,imas}}^{\mathrm{draw}}(\rho_k)
= \Gamma_{\mathrm{gB}}^{\mathrm{draw}}(\rho_k)\cdot\texttt{vprime\_miller}(\rho_k),
$$

accounting for the $\sim 25\text{--}40\%$ elongated flux-surface area factor
$\texttt{vprime\_miller}$ at $\kappa \approx 1.8$.

**Neoclassical floors & `TGYRO` flux-norm conditioning:** To prevent Jacobian
singularity in subcritical or radiatively hollow zones ($Q_{e,\mathrm{PB}} \le 0$
or $\hat{\Gamma}_{e,\mathrm{NN}}^{(m)} \equiv 0$):
1. Include neoclassical thermal conductivities $\hat{Q}_{i,\mathrm{neo}}(\mathbf{u}_k)$
   (Chang–Hinton / Hinton–Hazeltine) and $\hat{Q}_{e,\mathrm{neo}}(\mathbf{u}_k)$
   plus a neoclassical particle diffusivity & Ware-pinch closure
   $\hat{\Gamma}_{e,\mathrm{neo}}(\mathbf{u}_k) = \hat{D}_{\mathrm{neo}}\,u_{n_e,k} - \hat{\Gamma}_{\mathrm{Ware},k}$
   ($\hat{D}_{\mathrm{neo}} > 0$), so $\partial\hat{\Gamma}_{e,\mathrm{model}}/\partial u_{n_e,k} \ge \hat{D}_{\mathrm{neo}} > 0$
   even when all turbulent modes are linearly stable (and clamp negative
   radiatively hollow $u_{T_e,k}$ smoothly at a physical core radiative inversion
   bound rather than diverging to $-\infty$).
2. Normalize each channel residual $c \in \{Q_e, Q_i, \Gamma_e\}$ by the
   `TGYRO` / `PORTALS` RMS flux denominator (Candy et al. 2009, Eq. 19):
   $$
   R_{c,k}(\mathbf{u})
   = \frac{\hat{F}_{c,\mathrm{model}}(\mathbf{u}, \rho_k) - \Delta\hat{F}_{c,\mathrm{anchor},k} - \hat{F}_{c,\mathrm{target}}^{\mathrm{draw}}(\mathbf{u}, \rho_k)}
   {\sqrt{\hat{F}_{c,\mathrm{model}}^2 + (\hat{F}_{c,\mathrm{target}}^{\mathrm{draw}})^2 + \hat{F}_{c,\mathrm{floor}}^2}},
   $$
   where $\hat{F}_{c,\mathrm{target}}^{\mathrm{draw}} = F_{c,\mathrm{PB}}^{\mathrm{draw}} / F_{\mathrm{gB,imas}}^{\mathrm{draw}}$,
   taming severe slope discontinuities across critical-gradient threshold kinks.

### 4.3 Two anchoring modes (`flux_match_reference`), $\sigma = 0$, and `<P>` pressure matching

1. **`flux_match_reference = "anchored"` (perturbation-around-baseline):**
   $\Delta\hat{\mathbf{F}}_{\mathrm{anchor},k} = \hat{\mathbf{F}}_{\mathrm{model},0}(\mathbf{u}_{0,k}) - \hat{\mathbf{F}}_{\mathrm{target},0,k}$.
   At $\sigma = 0$ (`gk_epistemic_rel=0`, `flux_uncertainty_rel=0`, `sigma_ped=0`),
   $\mathbf{u} \equiv \mathbf{u}_0$ is an exact root, preserving bit-level
   $\sigma = 0$ baseline recovery. When conditioning on experimental magnetics,
   `sample_kinetics` retains its standard `<P>` thermal-pressure integral
   matching (`p_thresh`).
2. **`flux_match_reference = "absolute"` (predictive transport relaxation):**
   $\Delta\hat{\mathbf{F}}_{\mathrm{anchor},k} \equiv \mathbf{0}$. Every draw is
   solved directly on the `TGLF-NN` + neoclassical transport manifold. Because an
   unconverged empirical input baseline may relax by $> 5\%$ in stored thermal
   energy $W_{\mathrm{th}}$ onto the true critical-gradient profile, `"absolute"`
   mode **bypasses `sample_kinetics`'s fixed-baseline `|Delta<P>|/<P_0> <= p_thresh`
   rejection loop** (passing each draw's self-consistent thermal pressure
   profile forward into the free-boundary Grad–Shafranov solve) and documents its
   relaxed $\sigma = 0$ reference state explicitly.

### 4.4 Globally coupled $42 \times 42$ Jacobian & runtime budget

Crucially, the residual Jacobian $\mathbf{J} = \partial\mathbf{R}/\partial\mathbf{u} \in \mathbb{R}^{3K \times 3K}$
($42 \times 42$ for $K = 14$) is **globally coupled and dense — NOT $K$
decoupled $3 \times 3$ blocks**:

- **Inward nonlocal gyro-Bohm & background coupling ($j \ge k$):** Perturbing
  $\mathbf{u}_j$ at outer knot $j$ shifts inward-integrated $(n_e, T_e, T_i)(\rho_k)$
  for all inner knots $k \le j$, strongly modifying $Q_{\mathrm{gB,imas}}(\rho_k) \propto n_e T_e^{5/2}$
  as well as local `TGLF-NN` background inputs (`TAUS_2`, `XNUE`, `BETAE`,
  `DEBYE`). As established in `TGYRO` (Candy et al. 2009, §III) and `PORTALS`
  (Rodriguez-Fernandez et al. 2022), neglecting $\partial Q_{\mathrm{gB}}/\partial T_e$
  causes severe oscillatory divergence in stiff transport Newton iterations.
- **Outward integral source/sink coupling ($j \le k$):** Perturbing inner knots
  $j \le k$ shifts volume-integrated $P_\alpha^{\mathrm{draw}}(\rho_k)$ and
  Braginskii exchange $P_{ei}^{\mathrm{draw}}(\rho_k)$ entering outer surfaces
  $k \ge j$.

Evaluating one exact forward-difference $42 \times 42$ Jacobian of a single
ONNX ResNet member $m$ requires $(3K + 1) = 43$ state evaluations across $K = 14$
knots ($43 \times 14 = 602$ batched feature rows, taking **~2.5–4.0 ms** in a
single vectorized `PureNumpyONNXModel` matrix pass). Using a damped
Gauss–Newton / Levenberg–Marquardt solver with rank-1 Broyden Jacobian updates
between full refreshes (1–2 full $42 \times 42$ Jacobians + 8–12 single-state
14-row residual steps per draw, following `TGYRO` / `PORTALS`):

- **Per-draw kinetic solve time:** ~7–15 ms (pure NumPy, single CPU thread).
- **Consistent $\chi_e, \chi_i$:** Returned directly from the converged flux
  balance, $\chi_s^{\mathrm{draw}} = -Q_{s,\mathrm{PB}}^{\mathrm{draw}} / (n_s^{\mathrm{draw}}\,e\,\nabla T_s^{\mathrm{draw}})$,
  unifying `"power_balance"` and `"surrogate"` `chi_consistency`.

### 4.5 Pedestal peeling–ballooning MHD ceiling priors & emergent downward $\beta$ skew

Core ITG/TEM critical-gradient enforcement bounds only the **logarithmic slope**
$u_{T_s} = a / L_{T_s}$, not the boundary value $T_{s,\mathrm{ped}}$: because a
stiff core acts as a multiplier on the pedestal
($T_s(0) \approx T_{s,\mathrm{ped}}\exp[\int u_{T_s}\,d\rho]$), applying a
**symmetric** Gaussian uncertainty around $T_{s,\mathrm{ped},0}$ still lifts
half the draws above baseline without violating $\kappa_c$.

In predictive burning-plasma studies (`PORTALS`/`CGYRO` and `FUSE` baselines on
SPARC and ARC; Hughes et al. 2020; Rodriguez-Fernandez et al. 2020, 2022), the
nominal pedestal height $p_{\mathrm{ped},0}$ is typically anchored at the
**ideal `EPED1` peeling–ballooning / KBM stability limit** (Snyder et al. 2009,
2011) — an **upper MHD stability ceiling** ($\alpha_{\mathrm{MHD}} = \alpha_{\mathrm{crit,PB}}$)
rather than a median operating point:

- **Upside (MHD-limited):** Exceeding $\alpha_{\mathrm{crit,PB}}$ triggers an
  ELM crash rather than higher steady-state pressure, so upward excursions above
  $p_{\mathrm{ped,EPED1}}$ are restricted to `EPED1`'s epistemic model
  uncertainty ($\sigma_{\mathrm{EPED,epist}} \approx 0.10\text{--}0.15$, Snyder
  et al. 2011).
- **Downside (regime & detachment degradation):** Radiative divertor detachment,
  high separatrix fueling/opacity, and small-ELM / ELM-suppressed regimes (EDA,
  QCE) degrade $p_{\mathrm{ped}}$ **one-sidedly downward** below the ideal
  peeling–ballooning ceiling (Hughes et al. 2020).

To capture this without hand-tuned skew parameters, `"flux_matched"` supports
`pedestal_prior_mode = "symmetric" | "eped_ceiling"`:

$$
p_{\mathrm{ped}}^{\mathrm{draw}}
= p_{\mathrm{ped},0}\,\exp\!\Big(\sigma_{\mathrm{EPED,epist}}\,Z_{\mathrm{epist}} - \Delta_{\mathrm{deg}}\Big),
\qquad
Z_{\mathrm{epist}} \sim \mathcal{N}(0, 1),
\quad
\Delta_{\mathrm{deg}} = |\sigma_{\mathrm{ped,deg}}\,Z_{\mathrm{deg}}| \ge 0,
$$

partitioned between $(n_{e,\mathrm{ped}}, T_{e,\mathrm{ped}}, T_{i,\mathrm{ped}})$
along the edge uncertainty covariance while respecting $\Delta_{\mathrm{deg}} = 0$
at $\sigma = 0$.

Together with `"absolute"` core flux-matching (`p_thresh` disabled), an
optimistic `PORTALS`/`CGYRO` burning-plasma baseline naturally sits near the top
of the resulting `bouquet` $\beta$ and $P_{\mathrm{fus}}$ distributions through
**four parameter-free physical ceilings**:

1. **Quasilinear vs. nonlinear EM/$\alpha$ stabilization gap**
   ($\kappa_{c,\mathrm{TGLF}} \lesssim \kappa_{c,\mathrm{CGYRO}}$, relaxing
   optimistic core gradients onto the `TGLF-NN` + $\delta_{\mathrm{GK}}$ manifold);
2. **Concavity of $\kappa(Q)$ across the ITG/TEM threshold knee**
   ($\kappa''(Q) < 0$: $+\delta Q$ hits the stiff wall while $-\delta Q$ from
   W/Ne core radiation drops off the knee, amplified by Bosch–Hale $P_\alpha(T_i)$);
3. **Peeling–ballooning MHD ceiling on $p_{\mathrm{ped}}$** (`"eped_ceiling"`); and
4. **Pure-fuel lower bound $Z_{\mathrm{eff}} \ge 1$** (`kinetic_clips=True`) vs.
   open upward impurity dilution.

## 5. Proposed API & schema surface

### `UncertaintyConfig` additions

```python
sampling_mode: str = "gibbs_level"         # "gibbs_level" | "stiff_inward" | "flux_matched"
flux_match_reference: str = "absolute"     # "absolute" | "anchored"
gk_epistemic_rel: float = 0.10             # 1-sigma TGLF-SAT2 vs CGYRO kappa_c threshold offset
flux_match_knots: int = 14                 # radial collocation knots on rho_tor in [0.18, rho_ped]
pedestal_prior_mode: str = "symmetric"     # "symmetric" | "eped_ceiling"
eped_epistemic_rel: float = 0.12           # 1-sigma EPED1 model uncertainty in "eped_ceiling"
```

### Per-draw provenance attributes (`scan/<key>/<draw>/`)

- `tglfnn_member_idx` (`int`): committee member $m \in [0, 19]$ used for the draw;
- `flux_match_max_res` (`float`): converged $\max_k \|\mathbf{R}_k\|_\infty$;
- `flux_match_reference` (`str`): `"absolute"` or `"anchored"`;
- `P_alpha_MW_draw` (`float`): self-consistent fuel-gated Bosch–Hale $\alpha$ power.

## 6. Phasing & verification plan

1. **Phase 2a — Core 1-D globally coupled flux-matching solver (`bouquet/flux_matcher.py`):**
   Implement batched $42 \times 42$ damped Gauss–Newton + Broyden solver against
   `TGLFNNCommittee`, `vprime_miller`-consistent `Q_gB_imas`, Chang–Hinton /
   neoclassical particle diffusivity floors, `TGYRO` flux-norm residual
   scaling, fuel-gated Bosch–Hale $P_\alpha$, and multi-species Braginskii
   $q_{ei}$; unit-test convergence, threshold-offset quasineutrality, zero
   truncation skew, and `"anchored"` $\sigma = 0$ identity.
2. **Phase 2b — Parity benchmark against `FUSE` `TransportSolver` / `PORTALS`:**
   Verify via `scripts/compute_transport_stiffness.jl` that `"absolute"` mode
   reproduces `FUSE` steady-state `TurbulentTransport.jl` core profiles within
   collocation discretization tolerance (< 2% on $T_e(0), T_i(0)$).
3. **Phase 2c — Full free-boundary `TokaMaker` ensemble integration:**
   Wire into `sample_kinetics` (bypassing fixed-`<P_0>` rejection in
   `"absolute"` mode while preserving `<P>` conditioning in `"anchored"` mode),
   register under `bouquet.experimental.REGISTRY`, and validate coil $\chi^2$
   acceptance rates and $q$-profile / `SLAYER` rational-surface distributions.
