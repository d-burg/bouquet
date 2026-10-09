# Bouquet: Architecture, Assumptions, and Caveats

This document catalogues the physics assumptions, numerical approximations,
data-format conventions, and known limitations of the bouquet perturbed-equilibrium
toolkit.  It is intended as a reference for developers and for users who need to
understand exactly what is — and is not — guaranteed by the code.

---

## Table of Contents

1. [Overview](#1-overview)
   - [1.5 Class API and Baseline Sources](#15-class-api-and-baseline-sources)
2. [Coordinate Systems and Sign Conventions](#2-coordinate-systems-and-sign-conventions)
3. [Equilibrium Perturbation Methodology](#3-equilibrium-perturbation-methodology)
4. [Quasi-Neutrality and Impurity Handling](#4-quasi-neutrality-and-impurity-handling)
5. [Current Density Decomposition](#5-current-density-decomposition)
6. [Rotation Profile Computation](#6-rotation-profile-computation)
7. [Pressure and Beta Calculations](#7-pressure-and-beta-calculations)
8. [Flux-Surface Geometry and Averaging](#8-flux-surface-geometry-and-averaging)
9. [Internal Inductance](#9-internal-inductance)
10. [Numerical Floors and Clamps](#10-numerical-floors-and-clamps)
11. [Edge Extrapolation](#11-edge-extrapolation)
12. [Data Format Assumptions](#12-data-format-assumptions)
13. [HDF5 Storage Schema](#13-hdf5-storage-schema)
14. [Unit Conventions](#14-unit-conventions)
15. [Coil Constraint Handling (DIII-D Reference)](#15-coil-constraint-handling-diii-d-reference)
16. [Known Limitations and Future Work](#16-known-limitations-and-future-work)

---

## 1. Overview

Bouquet generates families ("bouquets") of perturbed tokamak equilibria by:

1. Starting from a baseline kinetic equilibrium — either a g-file plus
   kinetic profiles (p-file or IDA netCDF), or an IMAS/OMAS data-dictionary
   JSON with pre-separated currents (see §1.5).
2. Drawing correlated perturbations of ne, Te, ni, Ti from Gaussian
   process regression (GPR) posteriors, respecting user-supplied
   uncertainty envelopes.
3. Matching the perturbed pressure profile to the baseline volume-averaged
   pressure (within a configurable tolerance).
4. Decomposing the perturbed current density into bootstrap and inductive
   components and iterating the inductive profile to match the baseline
   internal inductance li.
5. Solving the Grad-Shafranov equation via TokaMaker for each accepted
   perturbation.
6. Archiving all results (geqdsk bytes, p-file bytes, scalar diagnostics)
   to a single HDF5 database.

Each of these steps carries assumptions documented below.

### 1.5 Class API and Baseline Sources

The class-based orchestrator (`bouquet/run.py`, `bouquet/config.py`,
`bouquet/baseline.py`) wraps the functional pipeline. `Bouquet(config)` owns
the TokaMaker solver and resolves any source to a common `Baseline`
(separated currents + targets + kinetic profiles), so generation never
depends on where the baseline came from:

| Module | Role |
|---|---|
| `config.py` | Typed dataclass config: `SolverConfig`, `ReconstructionSource` / `ImasSource`, `UncertaintyConfig`, `GenerationConfig`, `FilterConfig`, `FixedComponentsConfig` |
| `baseline.py` | `Baseline` dataclass + `resolve_baseline()` dispatch + `resolve_uncertainty()` (sigma precedence: explicit profile > IDA `.cdf` > flat scalar fraction) |
| `io/ida.py` | DIII-D IDA netCDF reader (via h5py; profiles + `*_err` sigmas; ni = ne quasi-neutrality) |
| `io/imas.py` | FUSE/IMAS JSON reader (stdlib json; no OMAS install needed) |
| `physics.py` | Convention reductions shared by all sources (below) |
| `run.py` | The `Bouquet` driver: solver → baseline → generate → filter → export |

**IMAS-path physics assumptions** (all in `bouquet/physics.py` and
`io/imas.py` docstrings, summarized here because this document is the
assumption catalogue):

* **Toroidal-current authority**: the IDS `j_tor` (converted to TokaMaker
  `jphi`) is taken as the total `j_phi`; the inductive component is the
  residual `j_phi − j_BS − j_NBI − j_RF`, so the decomposition sums exactly.
* **Current conventions** (`docs/current-conventions.md`): bouquet arrays are
  TokaMaker `jphi = ⟨j_φ⟩`; IMAS `j_tor = ⟨j_φ/R⟩/⟨1/R⟩` and its parallel
  currents are `⟨J·B⟩/B0`. The reader converts exactly with the FUSE
  equilibrium's own `gm1/gm5/gm8/gm9/f/dpressure_dpsi` (averages a producer
  omits are traced from `profiles_2d.psi`): the total via (A5),
  each component's field-aligned part `F⟨1/R⟩⟨J·B⟩/⟨B²⟩`, with the pressure
  term `p′(⟨R⟩ − F²⟨1/R⟩/⟨B²⟩)` assigned to the bootstrap (as IMAS.jl does).
  `solve_with_bootstrap` output is already TokaMaker `jphi`.
* **Fast-pressure isotropization**: anisotropic fast-ion pressure
  (`pressure_fast_perpendicular` / `_parallel`) is reduced to the scalar
  GS pressure via `tr(P)/3 = (2 p_perp + p_par)/3` by default ("trace";
  also "mean", "perp"). Fast pressure and driven currents are *fixed
  components*: summed into every draw, never GPR-perturbed.
* **l_i target**: the IDS-reported `li_3` is provisional only; the
  forward solve on the TokaMaker mesh re-evaluates the baseline and sets
  `l_i_target` to the TokaMaker `li_1`, recording both for comparison
  (`Baseline.li_metrics`).

---

## 2. Coordinate Systems and Sign Conventions

### 2.1 COCOS Convention

The geqdsk reader implements the COCOS (COordinate COnventionS) framework
[O. Sauter and S.Yu. Medvedev, Computer Physics Communications **184**
(2013) 293].  Eight sign/exponent parameters fully determine the
orientation:

| Symbol | Meaning | COCOS 1 (EFIT default) |
|--------|---------|----------------------|
| `sigma_Bp` | Sign of poloidal field relative to psi gradient | +1 |
| `sigma_RpZ` | Handedness of (R, phi, Z) | +1 |
| `sigma_rhotp` | Sign of (theta_pol × phi_tor) | +1 |
| `exp_Bp` | 2pi exponent (0 for COCOS 1-8, 1 for 11-18) | 0 |

**Assumption:** The default is COCOS 1 (standard EFIT).  An incorrect
choice propagates sign errors through all flux-surface-averaged
quantities, safety factor, and current density.

**User control:** The `cocos` parameter on `GEQDSKEquilibrium`.

### 2.1b COCOS Conversion (`cocosify`)

`GEQDSKEquilibrium.cocosify(cocos_out)` converts the raw g-file data
from the current COCOS to any target COCOS, following the transformation
rules in Sauter & Medvedev (2013), Eq. 14/23.

The effective transformation parameters between `cocos_in` and
`cocos_out` are:

```
sigma_Bp_eff    = sigma_Bp_out    * sigma_Bp_in
sigma_RpZ_eff   = sigma_RpZ_out   * sigma_RpZ_in
sigma_rhotp_eff = sigma_rhotp_out * sigma_rhotp_in
exp_Bp_eff      = exp_Bp_out      - exp_Bp_in
```

The multiplicative factors applied to each g-file field:

| Field | Factor |
|-------|--------|
| PSIRZ, SIMAG, SIBRY | `sigma_RpZ_eff * sigma_Bp_eff * (2π)^exp_Bp_eff` |
| PPRIME, FFPRIM | `sigma_RpZ_eff * sigma_Bp_eff / (2π)^exp_Bp_eff` |
| FPOL, BCENTR | `sigma_RpZ_eff` |
| CURRENT | `sigma_RpZ_eff` |
| QPSI | `sigma_rhotp_eff` |

These factors have been verified field-by-field against OMFIT's
`OMFITgeqdsk.cocosify()` for COCOS 1→7, 1→11, 7→1, and 7→11
(see `examples/COCOS_Bt_Ip/omfit_cocos_comparison.ipynb`).

**Caveat:** `cocosify` transforms the raw numerical arrays but does
not re-derive any cached quantities.  The cache is cleared on every
call, so subsequent property accesses (q_profile, geometry, etc.)
will recompute using the new COCOS parameters.

### 2.1c Bt/Ip Flip (`flip_Bt_Ip`)

`GEQDSKEquilibrium.flip_Bt_Ip()` reverses the direction of both
the toroidal field and the plasma current.  This negates:

- `BCENTR`, `FPOL` (toroidal field direction)
- `CURRENT` (plasma current direction)
- `SIMAG`, `SIBRY`, `PSIRZ` (poloidal flux)
- `PPRIME`, `FFPRIM` (per-psi derivatives)

The safety factor `QPSI` is **unchanged** because q ∝ Bt/Ip and
both signs cancel.

**Use case:** Converting between experiments with opposite field
and current directions (e.g. forward vs. reversed Bt operation on
DIII-D).

### 2.1d Serialisation (`save`, `to_bytes`)

Modified equilibria can be written back to standard GEQDSK format
via `save(filename)` or serialised to bytes via `to_bytes()` for
HDF5 storage.  The writer uses the same fixed-format (5 values per
line, 16 chars each) as the parser expects, ensuring exact
round-trip fidelity.

### 2.2 Normalised Poloidal Flux (psi_N)

Defined as:

```
psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)
```

Ranges from 0 (magnetic axis) to 1 (last closed flux surface).
All 1-D profile grids in bouquet use this normalisation.

### 2.3 Toroidal Flux Coordinate (rho)

`rhovn` (normalised square-root toroidal flux) is computed by
integrating the safety factor:

```
rho(psi_N) = sqrt( integral_0^psi_N q dpsi_N  /  integral_0^1 q dpsi_N )
```

The code explicitly recomputes this from `QPSI` rather than trusting
the RHOVN array stored in many g-files, because some equilibrium solvers
write a placeholder `RHOVN = sqrt(psi_N)` which is only correct when q
is spatially constant.

**Caveat:** A ~22% discrepancy against OMFIT's rhovn has been observed
in testing and has not yet been fully resolved.  See
[Section 16, Future Work](#16-known-limitations-and-future-work).

### 2.4 Midplane Convention

The outboard midplane is defined following OMFIT:

- R_mid = geometric centre + minor radius (outboard intersection)
- Z_mid = Z of the magnetic axis
- Br, Bz interpolated from the 2-D psi grid via `RectBivariateSpline`
- Bp = signed `sqrt(Br^2 + Bz^2)`, sign from
  `-sigma_rhotp * sigma_RpZ * sign(Bz)`
- Bt = F(psi) / R_mid

This is an **exact** midplane evaluation (no flux-surface averaging).
It is used in the rotation decomposition (Er, Hahm-Burrell rate) and
has been verified against OMFIT to <0.01% in R, <0.01% in Bt, and
<0.3% in Bp (excluding the degenerate magnetic axis).

---

## 3. Equilibrium Perturbation Methodology

### 3.1 Profile Sampling

Perturbations of ne, Te, ni, Ti are drawn from a zero-mean Gaussian
process whose covariance kernel is built from user-supplied uncertainty
envelopes sigma(psi_N).  Supported kernels include RBF (squared
exponential) and Matern, with either scalar or spatially-varying
(Gibbs non-stationary) length scales.

**Covariance matrix conditioning:** The eigendecomposition uses
`np.linalg.eigh` for symmetric matrices.  Small negative eigenvalues
from floating-point round-off are clamped to zero.

**Monotonicity enforcement:** For profiles that are physically
monotonically decreasing (ne, Te, ni, Ti in a standard tokamak), the
sampler rejects draws that violate monotonicity up to a maximum of
10 000 attempts before raising an error.

**Assumption:** Profile uncertainties are Gaussian-distributed and
fully characterised by the covariance kernel.  Non-Gaussian tails
(e.g. from ELMs or sawtooth crashes) are not captured.

### 3.2 Pressure Matching

After drawing perturbed profiles, the total kinetic pressure is computed
as:

```
P = e_charge * (ne * Te + ni * Ti)    [Pa, with ne in m^-3 and Te in eV]
```

Draws whose volume-averaged `<P>` differs from the baseline by more
than `p_thresh` percent are rejected and re-sampled until acceptance
or `max_pressure_iter` attempts.

**Default `p_thresh = 5%`** (was 0.5% in early versions).  The threshold
is calibrated to DIII-D's actual pressure measurement uncertainty so
that the accepted sampling distribution spans a physically meaningful
range rather than a vanishingly narrow `<P>`-conserving subspace.

Reference values:

| Measurement | Reported uncertainty |
|---|---|
| Diamagnetic loop `W_MHD` (between ELMs) | ~10% |
| Kinetic-EFIT pressure profile | ~3-6% (Thomson + CER propagated) |
| Volume-averaged `<P>` from kinetic-EFIT | ~3-5% |

A 0.5% threshold is roughly 10-20x tighter than the data warrants and
artificially anti-correlates ne/Te perturbations in the accepted samples;
5% reflects kinetic-EFIT propagated uncertainty.

**Source:** *MHD Equilibrium Reconstruction in the DIII-D Tokamak* (Lao
et al., GA-A24687, 2004); *Equilibrium reconstruction improvement via
Kalman-filter-based vessel current estimation at DIII-D* (Ou, Walker,
Schuster, Ferron, FED 2007).

**Assumption:** The acceptance loop preserves the GPR-drawn profile
shapes; no rescaling is applied.  Only draws within the threshold are
kept.

**Iteration limit:** 100 000 attempts (effectively unlimited).

### 3.3 li Matching: Recon-Anchor + Adaptive Gate

The internal inductance is anchored to the baseline value through a
three-stage process designed to make the pipeline behave as a true
forward operator (sigma -> 0 reproduces the reconstructed equilibrium):

1. **Recon-anchor solve (NEW).**  After `solve_with_bootstrap` (SWB)
   recomputes `j_BS` and `isolated_j_BS` from the perturbed kinetics,
   we drop SWB's `matched_j_inductive` and substitute recon's converged
   `j_inductive_fit` (passed via `input_jinductive`).  SWB seeds with
   `create_power_flux_fun(1.5, 1.5)` (broad shape) and alpha-scales to
   match Ip, but the resulting inductive has a fundamentally different
   *shape* than recon's eqdsk-fit inductive (typical recon/SWB ratio
   varies 1.4-3.4x with psi on the DIII-D reference case).  Without this
   substitution the pipeline's "zero-perturbation baseline" lives at
   l_i ~0.89 vs recon's 1.10 -- a ~19% systematic offset that breaks
   sigma -> 0 reproducibility regardless of how the matching loop is
   tuned.

2. **Post-anchor l_i gate (NEW).**  Immediately after the recon-anchor
   solve, check if `|l_i_actual - l_i_target| / l_i_target` is already
   within `l_i_tolerance`.  If yes, **skip** the iterative matching
   loop entirely and accept the anchored equilibrium.  This avoids the
   `find_optimal_scale` and `jphi-correction` rescaling drift that
   otherwise nudges l_i by ~1% per draw even when no kinetic perturbation
   is present.

3. **Adaptive proxy + secant matching (legacy fallback).**  Only runs
   when the gate fails (rare; typically high-sigma draws where the
   perturbation pushes l_i outside `l_i_tolerance`):

   - **Cylindrical proxy filter:** A fast 1-D proxy for l_i is computed
     without solving Grad-Shafranov, using pre-computed geometry.
     Draws whose proxy l_i falls outside a margin of the *corrected*
     target (`PRESCREEN_MARGIN` x `l_i_tolerance`, default 3x) are
     rejected cheaply before any GS solve.
   - **Secant iteration:** Scales the inductive amplitude to match the
     true Grad-Shafranov l_i within `l_i_tolerance`.
   - **Adaptive proxy correction:** After each GS solve, the proxy
     target is updated based on the observed proxy-vs-reality offset,
     blended 70% new / 30% old.

**Empirical reproducibility at sigma=0** (DIII-D reference case, 6 draws):

| Quantity | recon | sigma=0 pipeline |
|---|---|---|
| l_i | 1.110 | 1.105-1.110 (mean offset -0.17%) |
| bnd_RMS vs eqdsk | 2.84 mm | 3.1-4.2 mm |
| X-pt drift | 0 | < 2 mm |

**Iteration limit:** 20 secant steps.  If not converged the equilibrium
is rejected.

**Step clamping:** Secant steps clamped to +/-15% to prevent runaway.

### 3.4 Safety Factor Constraint

By default (`constrain_sawteeth=True`), any equilibrium with q(0) < 1
is rejected.  This prevents the generation of sawtoothing equilibria
that would require additional physics (reconnection, island evolution)
to model self-consistently.

**Caveat:** The constraint is checked *after* the GS solve, so
rejected equilibria still cost a TokaMaker call.

### 3.5 What Is Perturbed vs. Held Fixed

| Quantity | Perturbed? | Notes |
|----------|-----------|-------|
| ne, Te, ni, Ti | Yes | Drawn from GPR posterior |
| Total pressure (ptot) | Yes | Recomputed from perturbed kinetics |
| Bootstrap current (j_BS) | Yes (optional) | Recomputed from Sauter model if `recalculate_j_BS=True` |
| Inductive current (j_ind) | Yes | Scaled to match li |
| Coil currents | Yes | Adjusted by TokaMaker to match Ip |
| nz1 (impurity density) | **No** | Kept from baseline; see [Section 4](#4-quasi-neutrality-and-impurity-handling) |
| nb (beam density) | **No** | Preserved from baseline p-file |
| pb (beam pressure) | **No** | Preserved from baseline p-file |
| Toroidal rotation (omeg) | **No** | Preserved from baseline p-file |
| Poloidal rotation (omegp) | **No** | Preserved from baseline p-file |
| kpol | **No** | Preserved from baseline p-file |
| E×B rotation (w_ExB) | **No** | Zero placeholder stored; see [Section 16](#16-known-limitations-and-future-work) |
| Diamagnetic rotations | Yes | Recomputed from perturbed ne, Te, ni, Ti + baseline nz1 |
| Er, Hahm-Burrell rate | Yes | Recomputed using exact midplane B-fields |

---

## 4. Quasi-Neutrality and Impurity Handling

### 4.1 The Problem

Quasi-neutrality requires:

```
ne = ni * Z_main + nz1 * Z_imp + nb * Z_beam
```

When bouquet perturbs ne and ni independently, this constraint can be
violated.  Recomputing `nz1 = (ne - ni - nb) / Z_imp` from
independently perturbed profiles can yield **negative impurity density**
— an unphysical result.

### 4.2 Bouquet's Approach

**`generate_bouquet()` does not recompute nz1.**  The baseline impurity
density is preserved in the perturbed p-file.  This means:

- The perturbed p-file does **not** satisfy exact quasi-neutrality.
- The impurity contribution to pressure and diamagnetic rotation uses
  the original (self-consistent) nz1 profile.
- The total pressure `ptot` is recomputed as
  `ne*Te + (ni + nz1_baseline)*Ti + pb`, which is self-consistent with
  the stored profiles.

**Rationale:** Bouquet perturbs the *thermal* species (ne, ni, Te, Ti)
within their measurement uncertainties.  The impurity content is not
being perturbed — it is a separate measurement (e.g. from charge-exchange
spectroscopy) with its own uncertainty.  Forcing quasi-neutrality by
adjusting nz1 would couple the impurity density to the thermal profile
uncertainties in a physically unmotivated way.

### 4.3 `compute_quasineutrality()` Standalone Behaviour

The `PFile.compute_quasineutrality()` method is still available for
users who want to enforce charge balance explicitly.  If the result
contains negative values, a `UserWarning` is emitted with the number
of affected grid points and the minimum value.  The negative values
are **not** clamped — the caller decides how to handle them.

### 4.4 Downstream Consequences

Because baseline nz1 is preserved:

- `omgpp` (impurity diamagnetic) reflects the original impurity
  pressure gradient, not a quasi-neutrality-derived one.
- Zeff computed from the stored profiles will differ slightly from
  `(ni + nz1*Z^2 + nb) / ne` because quasi-neutrality is not exact.
- These differences are small for typical bouquet perturbation
  magnitudes (5-10% profile variations) and are within the measurement
  uncertainty of the impurity density itself.

---

## 5. Current Density Decomposition

### 5.1 Bootstrap Current

The bootstrap current density `j_BS` is computed using the Sauter
model [O. Sauter, C. Angioni, and Y.R. Lin-Liu, Physics of Plasmas
**6** (1999) 2834], which is the standard analytic model used by most
transport codes.

**Assumption:** The Sauter model is a fit to numerical neoclassical
calculations and is accurate to ~10% for standard tokamak conditions.
It can be less accurate for:

- Very low aspect ratio (spherical tokamaks)
- Strong rotation
- Non-Maxwellian distributions
- Very steep edge pedestals

**Implementation note.** bouquet evaluates the bootstrap with the Redl et al.
(2021) fit (OFT's `redl_bootstrap`, `formula_form='jboot1'`), the successor of
the Sauter fit with the same inputs. Default
(`GenerationConfig.jbs_self_consistent=True`): `physics.evaluate_jBS` on the
delivered equilibrium and the caller's own ψ_N grid, iterated to
self-consistency with the closure and the GS solve (`bouquet/jbs_loop.py`;
joint under-relaxation of the bootstrap and the solved current, convergence on
two consecutive passes, hard failure or a flagged slice). Legacy
(`jbs_self_consistent=False`): once per baseline and per draw through OFT's
`solve_with_bootstrap` on its own auxiliary equilibrium, then frozen. The
archive records which one ran (the schema-v3 `jbs_loop` block); see
[physics-notes.md](docs/physics-notes.md#self-consistent-bootstrap-jbs_self_consistent).

**Assumption (both paths):** the kinetic profiles are held at their ψ_N labels
while the current redistributes; the Redl drive is main-ion + electron only.

### 5.2 Inductive Current

The inductive current `j_ind` is defined as the residual:

```
j_ind = j_total - j_BS
```

It is fitted with a `scipy.interpolate.UnivariateSpline` using a
smoothing factor of `len(psi) * var(residual) * 0.01`.

**Assumption:** The inductive profile shape is smooth and can be
represented by a low-order spline.  No cross-validation is performed
on the smoothing parameter.

### 5.3 Bootstrap Shelf Option

If `shelf_psi_N > 0`, the bootstrap current is flattened for
psi_N < shelf_psi_N (i.e. in the core), using the value at the shelf
location.  This separates core and edge bootstrap contributions and
can be useful when the Sauter model produces unphysical core structure.

**Caveat:** The shelf introduces a discontinuity in the j_BS derivative
at psi_N = shelf_psi_N.

---

## 6. Rotation Profile Computation

### 6.1 Diamagnetic Rotation

The diamagnetic rotation frequency for species *s* is:

```
omega_dia,s = (1 / n_s Z_s e) * d(n_s T_s) / dpsi
```

In p-file units (n in 10^20/m^3, T in keV, psi in Wb), this gives
kRad/s directly.

**Sign convention (enforced by code):**
- Impurity and main-ion diamagnetic: negative (counter-current)
- Electron diamagnetic: positive (co-current)

The code uses `np.abs()` to compute the magnitude and then applies
the sign by convention.

**Assumption:** All ion species share the same temperature unless
an explicit impurity temperature `TI` is provided.

### 6.2 ExB Decomposition

```
omega_ExB  = omega_VxB + omega_dia_impurity
omega_VxB(main)  = omega_ExB - omega_dia_main
omega_VxB(elec)  = omega_ExB - omega_dia_electron
```

If `omega_VxB` (the impurity VxB term, from measured toroidal
rotation) is not present in the p-file, it defaults to zero.

### 6.3 Radial Electric Field and Hahm-Burrell Rate

```
Er    = omega_ExB * R_mid * Bp_mid        [kV/m]
omghb = (R_mid * Bp_mid)^2 / Bt_mid * d(omega_ExB)/dpsi
```

These use exact outboard-midplane values of R, Bp, Bt (not
flux-surface averages), following the OMFIT convention.

**Hahm-Burrell derivative method:** The `d(omega_ExB)/dpsi` term is
computed using a Savitzky-Golay filter with `deriv=1` (window ≈ 3%
of the grid, minimum 7 points, polynomial order 3).  This is
mathematically equivalent to fitting a local cubic polynomial and
analytically differentiating it, which is optimal for noisy data.
A naive `np.gradient()` on `omega_ExB` amplifies grid-scale noise
because `omghb` is effectively a **second derivative** of the
kinetic profiles (`omega_ExB ~ d(nT)/dpsi`, `omghb ~ d^2(nT)/dpsi^2`).

**Baseline consistency:** When storing the baseline p-file to HDF5,
`generate_bouquet()` recomputes the baseline rotation profiles
(diamagnetic, ExB decomposition, Er, omghb) using the same midplane
method as the perturbed p-files.  This ensures the baseline and
perturbed curves in `plot_pfile_bouquet()` are directly comparable.
Without this recomputation, the baseline `omghb` (from the original
p-file creation tool) can differ from the perturbed `omghb` by an
order of magnitude purely due to methodological differences.

### 6.4 Numerical Safeguards

Near the magnetic axis, `dpsi = gradient(psi)` approaches zero and
produces division-by-zero spikes.  At the plasma edge, densities
approach zero with the same effect.  Flooring is applied:

- **dpsi floor:** `max(1e-4 * max|dpsi|, 1e-30)`
- **Density floor:** `1e-4 * max|ne|` (applied to ne, ni, nz1 in
  the denominator only)
- **Bt floor:** `sign(Bt) * 1e-6` where |Bt| < 1e-6 T

These floors are small enough to preserve the physics everywhere
except the degenerate axis/edge points.  Any remaining NaN/inf values
are replaced with zero via `np.nan_to_num`.

---

## 7. Pressure and Beta Calculations

### 7.1 Total Pressure (P-file)

```
ptot = 16.022 * (ne * Te + (ni + nz1) * Ti) + pb     [kPa]
```

where the constant 16.022 kPa/(10^20 m^-3 keV) = e_charge * 1e20.

**Assumption:** Main ions are singly charged (Z_main = 1).  The
formula is correct for arbitrary Z_main but the p-file convention
stores `ni` as the main-ion density (not `ni * Z_main`).

### 7.2 Total Pressure (Bouquet SI)

```
P = EC * (ne * Te + ni * Ti)    [Pa]
```

where `EC = 1.6022e-19` J/eV, ne in m^-3, Te in eV.

**Caveat:** The bouquet SI pressure does not include the impurity or
beam contributions.  The p-file pressure does.  When comparing the
two, the difference is `EC * nz1 * Ti + pb`.

### 7.3 Beta

```
beta_t = 2 mu0 <p> / Bt_vac^2
beta_p = 2 mu0 <p> / <Bp>^2
beta_N = beta_t / (Ip / a Bt)    [in %·m·T/MA]
```

where `Bt_vac = B0 * R0 / R_boundary`.

**Caveat:** Beta quantities are computed in the geqdsk reader but
are not yet fully validated.  See [Section 16](#16-known-limitations-and-future-work).

---

## 8. Flux-Surface Geometry and Averaging

### 8.1 Contour Tracing

Flux surfaces are traced using `contourpy` on the 2-D psi(R,Z) grid
and resampled to 257 arc-length-uniform points using periodic (interior)
or non-periodic (near-separatrix) cubic splines.

**Threshold for periodic vs. non-periodic:** psi_N = 0.99.

**Rationale:** The X-point introduces a cusp in the separatrix contour.
Periodic splines would smooth it out; non-periodic splines preserve
the cusp but may oscillate slightly.

### 8.2 X-Point Detection

The X-point is located as the sharpest bend in the separatrix contour
(minimum cosine of the angle between adjacent tangent vectors).

**Assumption:** Single lower X-point.  Double-null or upper-single-null
configurations may not be handled correctly.

### 8.3 Flux-Surface Averaging

Averages use flux-expansion weighting:

```
<Q> = (1/V') * oint Q * dl / |Bp|
V'  = oint dl / |Bp|
```

where `dl` is the arc-length element along the contour.

**Bp flooring:** `max(1e-6 * max|Bp|, 1e-14)` prevents X-point-adjacent
spikes from dominating the average.

**Caveat:** This smooths the true geometric singularity at the X-point.
Flux-surface averages on surfaces very close to the separatrix
(psi_N > 0.995) should be treated with caution.

### 8.4 Separatrix Treatment

At psi_N = 1, the code uses the stored RBBBS/ZBBBS boundary from the
g-file rather than tracing a contour.  If the boundary has fewer than
4 points, a degenerate contour at the magnetic axis is substituted
with no error raised.

### 8.5 Jt Averaging

Two definitions are computed:

- **`j_tor_averaged`** (OMFIT standard): `<Jt/R> / <1/R>`.  This is
  the current density that, when multiplied by the cross-sectional
  area, gives the total toroidal current.

- **`j_tor_averaged_direct`**: literal `<Jt>`.

These differ by a Jensen inequality correction proportional to
`p' * [<R> - 1/<1/R>]`.  The difference is small for circular
cross-sections but can be significant for highly shaped plasmas.

---

## 9. Internal Inductance

### 9.1 Definitions

Multiple definitions of li are computed:

| Name | Formula | Notes |
|------|---------|-------|
| li(1)_EFIT | `<Bp^2> / Bp_edge^2` | Standard EFIT definition |
| li(1)_TLUCE | li(1)_EFIT × shape correction | Rarely used |
| li(2) | Volume-weighted variant | |
| li(3) | Alternative magnetic energy | Used by bouquet for matching |

The shape correction for li(1)_TLUCE is `(1 + kappa^2) / (2 * kappa_a)`
where `kappa_a = V / (2 pi R0 pi a^2)`.

**Default:** `li(1)` returns the EFIT definition.

**Risk:** Different transport codes and databases use different li
definitions.  Users must verify which definition their workflow expects.

---

## 10. Numerical Floors and Clamps

| Location | Floor | Purpose |
|----------|-------|---------|
| Diamagnetic dpsi | `1e-4 * max\|dpsi\|` | Prevent axis singularity |
| Diamagnetic density | `1e-4 * max\|ne\|` | Prevent edge/negative-density singularity |
| Rotation Bt | `1e-6` T | Prevent Er/omghb blowup |
| Flux-surface Bp | `1e-6 * max\|Bp\|` | Prevent X-point weight blowup |
| Secant step | ±15% of current value | Prevent li iteration runaway |
| Ip scaling | 50% of Ip_desired | Safety floor for TokaMaker scaling |
| GPR eigenvalues | max(lambda, 0) | Numerical PSD enforcement |

These floors are empirically chosen to be small enough that they do not
affect the physics on resolved grid points.  They are not user-configurable.

---

## 11. Edge Extrapolation

Many EFIT and TokaMaker equilibria force p'(1) = FF'(1) = 0 as a
free-boundary condition.  This creates a spurious dip in the current
density at the separatrix.

When `extrapolate_edge=True` (the default in the geqdsk reader), the
last 3-4 non-zero points of p' and FF' are quadratically extrapolated
to the boundary.  A sign-protection clip prevents the extrapolation
from reversing sign.

**Caveat:** This creates artificial structure at the plasma edge and
may distort the near-edge j_phi and pressure gradient profiles.
Users working on pedestal physics should consider disabling this
feature.

---

## 12. Data Format Assumptions

### 12.1 GEQDSK (G-file)

- Standard fixed-format: 5 values per line, 16 characters per value.
- The parser handles SOLPS variants (extra whitespace) and FIESTA
  row-by-row PSIRZ output.
- **Assumption:** The grid is uniform and rectangular (guaranteed by
  the GEQDSK specification).
- Br and Bz are computed from 2-D `np.gradient()` (second-order
  finite differences), which can amplify numerical noise at grid
  edges.

### 12.2 P-file (Osborne Format)

- Header lines match the regex:
  `(\d+)\s+(\S+)\s+(\S+)\(([^)]*)\)\s+(.*?)\s*$`
- Each profile block: count, x-name, y-name(units), description.
- The `"N Z A of ION SPECIES"` block is parsed separately and encodes
  species ordering: [impurity, main ion, beam ion].
- **Assumption:** All profiles use `psinorm` as the radial coordinate.
  Non-standard p-files using other normalisations will silently produce
  incorrect results.

### 12.3 HDF5 Database

See [Section 13](#13-hdf5-storage-schema).

---

## 13. HDF5 Storage Schema

Each bouquet run produces a single HDF5 file with the structure:

```
header.h5
+-- scan/
|   +-- {scan_val_key}/
|   |   +-- _baseline/
|   |   |   +-- baseline.eqdsk          # raw geqdsk bytes
|   |   |   +-- baseline.pfile          # raw p-file bytes (optional)
|   |   |   +-- ne, te, ni, ti [...]    # baseline 1-D profiles
|   |   |   +-- sigma_ne, sigma_te [...] # uncertainty envelopes
|   |   |   +-- coil_currents [A]       # recon's converged coils
|   |   |   +-- coil_names              # JSON list of coil names
|   |   |   ^-- attrs: Ip_target, l_i_target
|   |   +-- 0/
|   |   |   +-- {header}_{sv}_{0}.eqdsk # raw geqdsk bytes
|   |   |   +-- {header}_{sv}_{0}.pfile # raw p-file bytes (optional)
|   |   |   +-- psi_N, j_phi, ne, te, ni, ti [...]
|   |   |   +-- j_BS, j_BS,edge, j_inductive, pressure, w_ExB
|   |   |   +-- coil_currents [A]       # this draw's converged coils
|   |   |   ^-- attrs: count, scan_val, l_i(1), l_i(3), coil_names,
|   |   |       homotopy_pass, homotopy_F_lim, homotopy_VSC_lim,
|   |   |       max_F_drift_pct, max_VSC_drift_pct, in_spec,
|   |   |       inspec_F_max, inspec_VSC_max
|   |   +-- 1/ ...
|   +-- {another_scan_val}/ ...
```

**Key design choices:**

- Raw geqdsk and p-file bytes are stored as opaque binary datasets.
  This enables exact round-tripping: `GEQDSKEquilibrium.from_bytes()`
  and `PFile.from_bytes()` reconstruct the original objects.
- Scalar diagnostics (l_i(1), l_i(3)) are stored as attributes.
- Coil currents are stored as a 1-D array with coil names in a
  companion string attribute.  *Both* the `_baseline` group and every
  per-draw group store `coil_currents [A]`, so per-draw drift can be
  computed as `(I_draw - I_baseline) / |I_baseline|` without
  re-running reconstruction.
- The `_baseline` group stores the unperturbed equilibrium for
  comparison.
- Homotopy / in-spec metadata (see [Section 15.7](#157-per-draw-h5-attributes))
  is stored as attributes on each per-draw group so downstream
  analysis can filter by `in_spec` or stratify by `homotopy_pass`.

---

## 14. Unit Conventions

### 14.1 Bouquet Internal (SI)

| Quantity | Unit |
|----------|------|
| ne, ni | m^-3 |
| Te, Ti | eV |
| Pressure | Pa |
| Current density | A/m^2 |
| Psi | Wb (Weber) |
| B-field | T (Tesla) |
| R, Z | m |

### 14.2 P-file

| Quantity | Unit |
|----------|------|
| ne, ni, nz1, nb | 10^20 m^-3 |
| Te, Ti | keV |
| Pressure (ptot, pb) | kPa |
| Rotation (omeg, omgeb, ...) | kRad/s |
| Er | kV/m |

### 14.3 Conversion Constants

| Conversion | Value | Used in |
|------------|-------|---------|
| EC (eV to J) | 1.6022e-19 | Bouquet pressure = EC * n * T |
| _NT_TO_KPA | 16.02176634 | P-file ptot = _NT_TO_KPA * (ne*Te + ...) |
| m^-3 to 10^20/m^3 | 1e-20 | P-file density |
| eV to keV | 1e-3 | P-file temperature |
| Pa to kPa | 1e-3 | P-file pressure |

---

## 15. Coil Constraint Handling (DIII-D Reference)

This section documents how bouquet keeps perturbed-equilibrium coil
currents close to the reconstructed baseline, what physical
tolerances motivate the default parameters, and how to interpret the
per-draw `in_spec` flag.

### 15.1 Why coils need explicit constraints

Each perturbed equilibrium is a forward Grad-Shafranov solve.  TokaMaker
adjusts coil currents to satisfy: (i) the isoflux constraint (LCFS at
specified points), (ii) the requested plasma current `Ip_target`, and
(iii) the requested current profile.  When kinetic profiles are
perturbed, the natural minimum-energy GS solution can shift coils
substantially -- particularly the F9A/F9B vertical-stability pair --
producing equilibria that are mathematically valid but engineering-
infeasible.

A "perturbed equilibrium" is physically meaningful only when its
coil currents are reachable from the reconstructed baseline within
DIII-D's actual current-control tolerance.  Bouquet enforces this by
adding explicit per-coil bounds to the GS solver's QP and rejecting
draws that cannot satisfy them.

### 15.2 Three coil classes on DIII-D

The DIII-D coil set partitions naturally:

| Class | Members | Baseline range | Role |
|---|---|---|---|
| **Non-VSC F-coils** | F1A-F8A, F1B-F8B (16 coils) | 35-180 kA | Shaping + position |
| **VSC pair** | F9A, F9B | 35-50 kA, opposed signs | Vertical-stability (antisymmetric) |
| **E-coils** | ECOILA, ECOILB | 500-1000 A | Ohmic / auxiliary |

Each class has a different natural drift scale under perturbation:
non-VSC F-coils typically need <1% to compensate kinetic perturbations,
the VSC pair physically *requires* a few percent to track the moving
plasma centroid, and E-coils sit near a noise floor in absolute terms.

### 15.3 Engineering tolerance budget

The "applied vs programmed" coil current on DIII-D is bounded by a
combination of error sources:

| Source | Typical magnitude (DIII-D F-coils) |
|---|---|
| Power-supply current ripple (12-pulse + filter) | 30-50 A (~1% of full scale) |
| Rogowski coil + integrator noise | ~1-2% |
| Steady-state regulation error (PI controller) | 10-30 A |
| Un-modelled vessel-coupled drift during shot | 10-100 A |
| EFIT "calculated - measured" coil delta | 10s to ~100 A |

Combined, an absolute floor of **~50 A** captures the realistic
engineering tolerance below which "drift" is indistinguishable from
measurement + regulation noise.  For comparison, vessel induced currents
themselves can reach 0.5-4 kA per discretized segment during plasma
flattop -- but those are *passive* vessel currents, not coil currents,
and EFIT models them as free parameters separately from PF coil fits.

**Sources:**

* *MHD Equilibrium Reconstruction in the DIII-D Tokamak* (Lao et al.,
  GA-A24687, 2004).
* *Magnetic diagnostic system of the DIII-D tokamak* (Strait et al.) --
  ~250 inductive sensors; ~1% accuracy on poloidal probes and flux
  loops.
* *Equilibrium reconstruction improvement via Kalman-filter-based
  vessel current estimation at DIII-D* (Ou, Walker, Schuster, Ferron,
  *Fusion Eng. & Design* 82 (2007) 1144) -- vessel induced currents
  0.5-4 kA per segment during flattop; PF coil currents treated as
  free fit parameters with measurement uncertainty.
* *GA-A24059: New measurements of coil-related magnetic field errors
  on DIII-D* (Anderson et al., 2002) -- coil axis alignment +/-5 mm,
  parallelism +/-0.2 deg.
* *Observation of poloidal current flow to vacuum vessel wall during
  DIII-D vertical instabilities* (Strait et al., *Nucl. Fusion* 31
  (1991) 419) -- vessel can carry hundreds of kA during VDEs (not
  applicable to steady-state shots).
* *Performance of current measurement system in poloidal field power
  supply for EAST* -- 12-pulse converter <1% ripple, 1% steady-state
  error coefficient typical.

### 15.4 Bound construction

For each coil at each homotopy pass with parameters
`(drift_F, drift_VSC)`, bouquet installs hard bounds:

```
For each coil 'name' with baseline I_base:
    delta = max(drift_F * |I_base|, coil_drift_floor_A)
    bounds[name] = [I_base - delta, I_base + delta]

Additionally on the antisymmetric VSC channel:
    vsc_delta = max(drift_VSC * min(|F9A_base|, |F9B_base|), coil_drift_floor_A)
    bounds['#VSC'] = [-vsc_delta, +vsc_delta]
```

Per-coil and channel bounds are enforced simultaneously by the GS solver
(TokaMaker's `set_coil_bounds`).  Notes:

* The bound `delta` uses the *larger* of a relative cap (`drift_F` * the
  coil baseline magnitude) and an absolute floor (`coil_drift_floor_A`,
  default 50 A).  This prevents nonsense bounds on small-baseline coils
  where a percentage bound would be smaller than measurement noise.
* The `#VSC` channel bound uses the *minimum* of `|F9A_base|` and
  `|F9B_base|` so that the bound is conservative for both coils (their
  baselines are typically asymmetric: ~45 kA vs ~37 kA on
  the reference case).
* Total drift on F9A/F9B = bare drift + VSC channel contribution.
  Caller selects passes so that `drift_F + drift_VSC <=` the desired
  total F9 tolerance.

### 15.5 Progressive homotopy

Hard bounds at the eventual target spec (e.g. +/-2% on all coils) are
often QP-infeasible from a cold start because the perturbed kinetics
naturally want a coil distribution outside the bound region.
Bouquet supports a **progressive homotopy** via the `homotopy_passes`
kwarg: a list of `(drift_F, drift_VSC)` tuples that the solver tries
in order, warm-starting each pass from the prior pass's converged psi.

A typical 3-pass schedule for the "strict ±2% global" spec:

```python
homotopy_passes = [
    (0.05, 0.10),   # Pass 1: loose, get QP near optimum
    (0.02, 0.05),   # Pass 2: intermediate tightening
    (0.01, 0.01),   # Pass 3: total F9 drift <= 2% (bare 1% + VSC 1%)
]
```

If a pass fails (`Closed flux volume lost` or `Matrix solve failed for
targets`), bouquet rolls back to the prior successful pass's psi,
re-installs that pass's bounds, re-solves once to restore mygs's
internal flux-surface state (without this re-solve `get_stats()` returns
`l_i = inf`), and stops tightening.  The draw is recorded with
`homotopy_pass` = index of the last successful pass.

If Pass 1 itself fails the draw is rejected outright.

### 15.6 The `in_spec` criterion

After homotopy, each draw is tagged `in_spec` if:

```
max_non_VSC_F_drift_pct <= inspec_F_max   AND   max_VSC_drift_pct <= inspec_VSC_max
```

For the **non-VSC F-coils** the drift is computed per-coil as
`(I_draw - I_baseline) / |I_baseline| * 100`, where
`I_draw = mygs.get_coil_currents()` (the *total* current, i.e. bare +
vcontrol·VSC contribution) and `I_baseline` is recon's coil currents
captured on entry to `generate_bouquet`.

The **VSC pair does not use that per-coil formula.** Because F9A or F9B
routinely sits near a current zero-crossing, `max_VSC_drift_pct` is scored on
the common-mode / differential channel decomposition, gated against an
error-propagated `sigma_VSC` built from the coil magnitudes
(`_vsc_channel_drift_pct`). See
[`docs/coil-constraints.md`](docs/coil-constraints.md#vsc-drift-metric-anti-series-pair)
for the derivation and its assumptions.

**E-coils are not currently part of the `in_spec` check.**  Reason:
their baselines (~500-1000 A) are small enough that the 50 A
`coil_drift_floor_A` translates to 5-10% relative drift, but in
absolute terms 50 A is the engineering noise floor (see Section 15.3).
A user who wants strict 2% on E-coils should lower
`coil_drift_floor_A` to ~10-15 A and accept that Pass 1 may need a
looser absolute floor to remain feasible.

### 15.7 Per-draw H5 attributes

After homotopy, every stored draw carries these attributes (see
[Section 13](#13-hdf5-storage-schema)):

| Attribute | Meaning |
|---|---|
| `homotopy_pass` | Index of last successful pass (0-based; -1 if no hard bounds ran) |
| `homotopy_F_lim` | `drift_F` value of the last successful pass |
| `homotopy_VSC_lim` | `drift_VSC` value of the last successful pass |
| `max_F_drift_pct` | Max non-VSC F-coil drift in percent |
| `max_VSC_drift_pct` | Max F9A/F9B drift in percent |
| `in_spec` | Boolean: did this draw satisfy the `inspec_*_max` criteria? |
| `inspec_F_max`, `inspec_VSC_max` | The thresholds that were applied |

Out-of-spec draws are still archived (with `in_spec=False`) so
downstream analysis can filter as needed.  Empirical survival rates
at the strict +/-2% global spec on the DIII-D reference case (n=8 draws per
sigma):

| `SIGMA_SCALE` | yield | `in_spec` rate |
|---|---|---|
| 0.0 | 7/8 | 2/7 (28%) |
| 0.5 | 7/8 | 1/7 (14%) |
| 1.0 | 8/8 | 4/8 (50%) |

The non-monotonic pattern is real: at sigma=0 iso-update systematically
forces ~2.5-3.5% VSC drift even with zero kinetic perturbation, while
at sigma=1.0 some draws happen to align with low-VSC equilibria and
satisfy the strict bound while others need much more.

### 15.8 The `vsc_soft_reg_weight` knob — what it controls and what it doesn't

Bouquet installs two layers of soft regularization on the F9 pair:

1. **Per-coil bare soft-reg** (weight = `soft_reg_weight`, default 1e4):
   pulls `F9A_bare` toward recon's `F9A`, and `F9B_bare` toward recon's
   `F9B`.  This is applied identically to every non-VSC F-coil.
2. **`#VSC` channel soft-reg** (weight = `vsc_soft_reg_weight`, default 1.0):
   pulls the antisymmetric channel value (the additional coordinated
   F9A↑/F9B↓ knob TokaMaker provides via `set_coil_vsc`) toward zero.

The F9 pair has two physical degrees of freedom (F9A_bare, F9B_bare).
The per-coil soft-reg already pins both at weight 1e4.  Adding
`vsc_soft_reg_weight` is a *third* pull on a 2-DoF subspace and at
sufficiently high weight makes the QP over-determined.  Empirical
findings on the DIII-D reference case (n=15 per setting):

| `vsc_soft_reg_weight` | σ=0 outcome | σ=0.5 in-spec | σ=1.0 in-spec |
|---|---|---|---|
| 1.0 (default) | 0/30 in-spec | 23% (n=30) | 23% (n=30) |
| 100 | 1/15 in-spec | **40%** | – |
| 1e3 | **QP singular, 0/15 yield** | – | – |
| 1e4 | **QP singular, 0/15 yield** | – | – |

So `vsc_soft_reg_weight` is useful only as a moderate antisymmetric
tightener at intermediate σ.  At low σ it does **not** rescue in-spec
yield, because the σ=0 in-spec gap comes from the iso-update step
shifting the LCFS, which the homotopy then must satisfy, requiring
F9 to drift ~3% on the antisymmetric mode regardless of the soft-reg
weight.

**Recommended values:**

| σ scale | `vsc_soft_reg_weight` | `homotopy_passes` |
|---|---|---|
| 0.0 | 1.0 | **`None`** (soft-reg-only mode; natural F9 drift ~0.6%, all in-spec) |
| 0.5 | 100 | `[(0.05,0.10), (0.02,0.05), (0.01,0.01)]` |
| 1.0 | 1.0 | `[(0.05,0.10), (0.02,0.05), (0.01,0.01)]` |

The σ=0 case is the asymmetric one: the **iso-update step assumes a
non-trivial boundary shift that doesn't actually happen at σ=0**, and
the resulting F9 antisymmetric drift is what blocks the strict ±2%
in-spec.  Skipping hard bounds entirely at σ=0 (`homotopy_passes=None`,
optionally `SKIP_HARD=1`) returns the pipeline to soft-reg-only mode
where natural F9 drift is 0.6% — comfortably in spec.

### 15.9 Post-perturb pipeline summary

After kinetic perturbation and the recon-anchor solve, every draw
goes through:

```
1. Ip-secant alignment
       Iterate until mygs.get_globals()[0] == _recon_Ip (within 0.1%)
       with damped retries on Picard maxits failure.

2. Iso-update  (only if hard bounds will be installed)
       Replace recon's eqdsk-derived isoflux points with the post-Ip-aligned
       LCFS so the QP has a self-consistent boundary target.

3. Progressive homotopy
       Loop over homotopy_passes, warm-starting each pass and rolling back
       on failure (re-solving to restore mygs state for downstream stats).

4. In-spec tagging
       Compute max F-coil and max VSC drift percentages, set in_spec flag,
       store all metadata on the H5 group.
```

Each step is independently configurable via env vars / kwargs:

* `SKIP_HARD=1`: skip iso-update + homotopy (use soft-reg only)
* `SKIP_ISO=1`: skip iso-update but keep homotopy (diagnostic)
* `homotopy_passes=None`: single legacy pass at uniform `coil_drift`
* `inspec_F_max`, `inspec_VSC_max`: spec thresholds (`FilterConfig`, default
  both 0.02)

---

## 16. Known Limitations and Future Work

### 16.1 E×B Rotation (w_ExB) Not Computed from Equilibrium

The `w_ExB` field stored in the HDF5 is a **zero placeholder**.
A self-consistent E×B rotation would require solving the radial
force balance for each perturbed equilibrium, which is not yet
implemented.  The diamagnetic decomposition (omgpp, ommpp, omepp)
and derived quantities (omgeb, Er, omghb) in the perturbed p-file
*are* computed, but they use the baseline VxB rotation (`omgvb`)
if present, or zero if absent.

### 16.2 rhovn Discrepancy

A ~22% discrepancy between bouquet's `rhovn` and OMFIT's `rhovn`
has been observed.  The bouquet computation integrates q over psi_N;
OMFIT may use a different integration scheme or include additional
corrections.  This affects any analysis using rho as the radial
coordinate.

### 16.3 Beta Quantities

`beta_t`, `beta_p`, `beta_N` are computed in the geqdsk reader
but have not been validated against EFIT or other reference codes.

### 16.4 Double-Null and Upper-Single-Null Equilibria

The X-point detection assumes a single lower X-point.  Double-null
or upper-single-null configurations may produce incorrect contour
cropping, flux-surface averaging, and geometric quantities (kappa,
delta, squareness).

### 16.5 Near-Axis Shaping

Elongation (kappa) and triangularity (delta) are computed from the
half-widths and extrema of each flux surface.  Near the magnetic
axis, these quantities become noisy because the contours are nearly
circular and the extrema are poorly defined.  No smoothing is
currently applied.

### 16.6 Uncertainty Envelope Model

The default uncertainty envelope is a power-law profile:

```
sigma(psi_N) = sigma_0 * (1 - psi_N)^n
```

with a flat core, smooth tanh transition, and a minimum floor.  This
is a simple parametric model.  Users with experimentally-derived
profile uncertainties should supply them directly rather than relying
on this model.

### 16.7 Non-Gaussian Profile Uncertainties

The GPR sampling assumes Gaussian-distributed profile uncertainties.
Non-Gaussian features (e.g. pedestal bifurcation, ELM-induced
transients, sawtooth mixing) are not captured.

### 16.8 Single Ion Temperature

The diamagnetic rotation calculation assumes all ion species
(main + impurity) share the same temperature `Ti` unless a separate
impurity temperature `TI` is explicitly provided.  This is a common
assumption in tokamak transport modelling but breaks down when strong
ion-impurity temperature decoupling exists (e.g. during NBI heating).

### 16.9 Beam and Impurity Perturbation

Currently, bouquet does not perturb:

- Fast-ion density (nb) or pressure (pb)
- Impurity density (nz1)
- Toroidal/poloidal rotation profiles (omeg, omegp, kpol)

Adding uncertainty-driven perturbation of these quantities is a
natural extension but would require additional uncertainty
specifications and potentially different sampling strategies (e.g.
rotation profiles are not necessarily monotonic).

### 16.10 P-file Regeneration for Example Files

The example p-files shipped with the repository have rotation profiles
computed from a specific baseline equilibrium.  If the baseline
equilibrium is changed (e.g. different bootstrap current amplitude),
the rotation profiles in the shipped p-file will be stale.
Regeneration of rotation profiles for all example files is planned.

### 16.11 Flux-Surface Average vs. Midplane for Diagnostics

Some diagnostics (e.g. Thomson scattering, charge-exchange
recombination) measure profiles at specific poloidal locations
(typically the outboard midplane), not flux-surface averages.
Bouquet perturbs flux-surface-averaged profiles.  For strongly
up-down asymmetric plasmas, this distinction matters.

### 16.12 Bootstrap Current Model Alternatives

Only the Sauter model is currently implemented.  Alternative models
(e.g. Sauter with Redl corrections, NEO, or direct drift-kinetic
solvers) may give significantly different bootstrap current profiles,
especially in the pedestal region.  Adding pluggable bootstrap
current models is planned.
