# Bouquet: Architecture, Assumptions, and Caveats

This document lists the physics assumptions, numerical approximations,
coordinate and sign conventions, tolerances and architectural limitations of
bouquet 1.4.1: what the code assumes and what it guarantees. It describes the
default route first, the unified reconstruction engine for both the
reconstruction and the draws. Where another page carries the detail, this one
states the assumption and links to it: [docs/engine.md](docs/engine.md) (the
engine, its rows, draws and boundary constraint),
[docs/workflows.md](docs/workflows.md) (the pipeline and the configuration
reference), [docs/physics-notes.md](docs/physics-notes.md),
[docs/current-conventions.md](docs/current-conventions.md) (identities
A1-A9), [docs/coil-constraints.md](docs/coil-constraints.md),
[docs/archive-schema.md](docs/archive-schema.md),
[docs/known-limitations.md](docs/known-limitations.md) and
[docs/validation-provenance.md](docs/validation-provenance.md).

## Table of Contents

1. [Overview](#1-overview)
2. [Coordinate Systems and Sign Conventions](#2-coordinate-systems-and-sign-conventions)
3. [Equilibrium Perturbation Methodology](#3-equilibrium-perturbation-methodology)
4. [Quasi-Neutrality and Impurity Handling](#4-quasi-neutrality-and-impurity-handling)
5. [Current Density Decomposition](#5-current-density-decomposition)
6. [Rotation and Radial Electric Field](#6-rotation-and-radial-electric-field)
7. [Pressure and Beta](#7-pressure-and-beta)
8. [Flux-Surface Geometry and Averaging](#8-flux-surface-geometry-and-averaging)
9. [Internal Inductance](#9-internal-inductance)
10. [Numerical Floors and Clamps](#10-numerical-floors-and-clamps)
11. [Edge Treatment](#11-edge-treatment)
12. [Data Format Assumptions](#12-data-format-assumptions)
13. [HDF5 Archive](#13-hdf5-archive)
14. [Unit Conventions](#14-unit-conventions)
15. [Coil Constraint Handling (DIII-D Reference)](#15-coil-constraint-handling-diii-d-reference)
16. [Known Limitations and Future Work](#16-known-limitations-and-future-work)

---

## 1. Overview

### 1.1 The pipeline

Bouquet turns one tokamak equilibrium into an ensemble ("bouquet") of
perturbed equilibria, each a free-boundary Grad-Shafranov solve by TokaMaker
(OpenFUSIONToolkit). On the default route:

1. **Source.** A g-file with kinetic profiles (a p-file or an IDA-layout
   profile fit), or an IMAS/OMAS data-dictionary JSON (optionally with
   IDA-layout kinetics), reduced by an adapter (`adapters.GFileAdapter`,
   `adapters.IdsAdapter`) to one contract (`adapters.EngineContract`):
   kinetics, pressure, inductive and fixed driven currents as `<j.B>`, the
   boundary and the constraint rows.
2. **Reconstruction.** `Bouquet.prepare_baseline()` runs the unified engine
   (`engine.py`): a closure on the latest solved geometry, one GS solve per
   pass, Redl re-evaluated on every solved state, until the criteria hold on
   two consecutive passes, then a delivery solve that every draw starts from
   ([§5.4](#54-the-self-consistent-loop-and-its-tolerances)).
3. **Draws.** `Bouquet.generate()` perturbs the kinetics, Z_eff and the
   inductive current, holds the reconstruction's closure coefficients, closes
   only the I_p row, runs one self-consistent bootstrap loop, then a
   progressive coil homotopy ([§3](#3-equilibrium-perturbation-methodology), [§15](#15-coil-constraint-handling-diii-d-reference)).
4. **Acceptance.** A draw is in spec on the magnetics: a chi-squared test of
   its coil currents against the device's coil uncertainties AND a cut on the
   RMS displacement of its last closed flux surface. l_i, beta_p and
   Lambda = beta_p + l_i(3)/2 are recorded, not filtered, unless an opt-in
   band is set ([§15.5](#155-the-in-spec-rule)).
5. **Archive.** Every draw that reaches the archive is stored, in spec or
   not, in one HDF5 file, schema v3 ([§13](#13-hdf5-archive)).

### 1.2 Solve routes

`GenerationConfig.solve_method` selects how the reconstruction and the draws
are made (`config.resolve_solve_method`; one draw-method object per route in
`bouquet.draw_methods`, compared hook by hook in
[docs/draw-methods.md](docs/draw-methods.md)):

| route | selected by | status |
|---|---|---|
| unified engine | `solve_method="engine"`, or `solve_method=None` with `reconstruction_engine="unified"` (the default) | default since 1.4.0 |
| legacy | `solve_method="legacy"` or `reconstruction_engine="legacy"` | deprecated since 1.4.0, removed in 1.5 (#90); one `DeprecationWarning` per run (`Bouquet._warn_legacy_route`) |
| swb | `solve_method="swb"` (IMAS sources only): OFT `solve_with_bootstrap` is the baseline and every draw | experimental |

A baseline built by one route is refused by the other's draws; legacy-path
settings the engine never reads are refused under it
(`engine.ENGINE_UNREAD_LEGACY_FIELDS`,
[docs/engine.md](docs/engine.md#settings-the-engine-does-not-read-refused)).
The legacy route appears below only where old archives or the remaining
legacy options need it.

### 1.3 Class API and baseline sources

`Bouquet(config)` (`run.py`) owns the TokaMaker solver and runs
`setup_solver → prepare_baseline → generate → filter → export`
(`Bouquet.run()`). The configuration is a set of dataclasses (`config.py`;
reference in [docs/workflows.md](docs/workflows.md#configuration-reference)):
`SolverConfig` (mesh, isoflux and saddle constraints, the VSC pair
`coil_vsc`, default `{'F9A': 1.0, 'F9B': -1.0}`, coil regularization),
`ReconstructionSource` (g-file + `profiles_path`, a p-file or an IDA `.cdf`;
`cocos` 1, `impurity_Z` 6.0, `psi_pad` 1e-3 by default), `ImasSource`
(IMAS/OMAS JSON, optional IDA file, `impurity_Z` 6.0), `UncertaintyConfig`,
`GenerationConfig`, `FilterConfig` and `FixedComponentsConfig` (user `p_fast`,
`j_NBI`, `j_RF`, `j_other`). Code anchors below name the module
(`module.function`) under `bouquet/`.

## 2. Coordinate Systems and Sign Conventions

### 2.1 COCOS (g-file reader)

`io.geqdsk.GEQDSKEquilibrium` implements the COCOS framework
[O. Sauter and S.Yu. Medvedev, Computer Physics Communications **184**
(2013) 293]. `_cocos_params` gives, for COCOS 1-8 and 11-18:

| Symbol | Meaning | COCOS 1 |
|--------|---------|---------|
| `sigma_Bp` | sign of the poloidal flux gradient | +1 |
| `sigma_RpZ` | +1 for right-handed (R, phi, Z), -1 for (R, Z, phi) | +1 |
| `sigma_rhotp` | sign of (theta_pol × phi_tor) | +1 |
| `exp_Bp` | 2π exponent: 0 for COCOS 1-8, 1 for 11-18 | 0 |

**Assumption:** the reader's default is COCOS 1 (`cocos=1`, also
`ReconstructionSource.cocos`). A wrong choice propagates sign errors into
every flux-surface average, q and the current density. The g-file adapter
refuses a file whose flux-surface current `<j_phi>` does not carry the sign
of its own I_p in the bulk (`adapters.gfile_parallel_current`), which catches
most wrong choices.

### 2.2 COCOS conversion, Bt/Ip flip, serialization

`GEQDSKEquilibrium.cocosify(cocos_out, copy=False)` converts the raw arrays
following Sauter & Medvedev (2013), Eqs. 14 and 23:

```
sigma_Bp_eff    = sigma_Bp_out    * sigma_Bp_in
sigma_RpZ_eff   = sigma_RpZ_out   * sigma_RpZ_in
sigma_rhotp_eff = sigma_rhotp_out * sigma_rhotp_in
exp_Bp_eff      = exp_Bp_out      - exp_Bp_in
```

| Field | Factor |
|-------|--------|
| PSIRZ, SIMAG, SIBRY | `sigma_RpZ_eff * sigma_Bp_eff * (2π)^exp_Bp_eff` |
| PPRIME, FFPRIM | `sigma_RpZ_eff * sigma_Bp_eff / (2π)^exp_Bp_eff` |
| FPOL, BCENTR, CURRENT | `sigma_RpZ_eff` |
| QPSI | `sigma_rhotp_eff` |

The cache of derived quantities is cleared on every call. The comparison
against OMFIT's `OMFITgeqdsk.cocosify()` (COCOS 1→7, 1→11, 7→1, 7→11) is the
notebook `examples/COCOS_Bt_Ip/omfit_cocos_comparison.ipynb`; it needs
`omfit_classes` and is not part of the test suite.

`flip_Bt_Ip()` negates `BCENTR`, `FPOL`, `CURRENT`, `SIMAG`, `SIBRY`,
`PSIRZ`, `PPRIME` and `FFPRIM` at a fixed COCOS index; `QPSI` is unchanged
(q ∝ Bt/Ip). `save(filename)` and `to_bytes()` write the fixed format the
parser reads (5 values per line, 16 characters each).

### 2.3 Normalized poloidal flux (psi_N)

```
psi_N = (psi - psi_axis) / (psi_boundary - psi_axis)
```

0 at the magnetic axis, 1 at the last closed flux surface. ψ is the poloidal
flux per radian (COCOS 1-8, TokaMaker's convention). The solver's
flux-surface quantities are sampled at `clip(psi_N, psi_pad, 1 - psi_pad)`,
`ReconstructionSource.psi_pad = 1e-3` (`physics.evaluate_jBS`,
`adapters._q0_psi`), never on the singular end surfaces.

### 2.4 Toroidal-flux coordinate and `coord="phi_n"` runs

The g-file reader's `rhovn` ignores a stored `RHOVN` and recomputes it from
`QPSI` (`GEQDSKEquilibrium.rhovn`):

```
Phi(psi) = integral_{psi_axis}^{psi} q dpsi'      (cumulative trapezoid)
rho      = sqrt( Phi / Phi_edge )
```

A source with `coord="phi_n"` puts the whole run on Φ_N = ρ² (`"rho_tor"` is
accepted and runs as Φ_N; `coords.run_coord`). Profiles, envelopes and draws
live on Φ_N; TokaMaker receives them tagged with the coordinate, and bouquet
samples every readback at the ψ_N the solver's own toroidal-flux map gives for
each node (`coords.psi_at`). Datasets named `psi_N` hold the run grid; the
baseline group records it as `profile_coord`
([docs/workflows.md](docs/workflows.md#radial-coordinate-bsourcecoord)).

### 2.5 The positive-current frame

Bouquet solves every equilibrium with I_p > 0 and `F0 = |R0 B0|`
(`utils.IFILE_FRAME`). The g-file adapter maps p', FF', F into that frame
with `sign(I_p)` and `|F|`; the IMAS reader multiplies every current it reads
by `sign(equilibrium ip)` and refuses a dictionary whose net `j_tor`
disagrees in sign (`io.imas` module docstring). The factors are archived as
`_baseline` attributes `source_current_sign`, `source_b0_sign`; the IMAS
exporter returns what it overwrites to the source's orientation. User
`j_NBI` / `j_RF` are defined in the positive frame.

### 2.6 Outboard midplane (g-file reader)

`GEQDSKEquilibrium.midplane`, following OMFIT: R_mid = geometric center +
minor radius of each traced surface; Z_mid = Z of the magnetic axis; Br, Bz
interpolated from the 2-D grid (`RectBivariateSpline`);
Bp = `-sigma_rhotp * sigma_RpZ * sign(Bz) * sqrt(Br^2 + Bz^2)`, 0 on axis;
Bt = F(psi) / R_mid. These are point values, not flux-surface averages; they
feed the E×B and Hahm-Burrell terms of the per-draw p-file
([§6.2](#62-eb-decomposition-e_r-and-the-hahm-burrell-rate)).

### 2.7 Where q0 is quoted

| quantity | radius |
|---|---|
| the engine's q0 row, the q0 sigma=0 gate, `constrain_sawteeth` | ψ_N = `psi_pad` (the clipped axis sample, `adapters._q0_psi`) |
| TokaMaker `get_stats()['q_0']` | ψ_N = 0.02 (`physics.SOLVER_Q0_PSI_N`) |
| a g-file's `QPSI[0]`, an IDS `q[0]` | the magnetic axis, ψ_N = 0 |

## 3. Equilibrium Perturbation Methodology

### 3.1 What a draw is

An engine draw (`engine_draws.GenerateEngineDraws`) starts from the
reconstruction's delivered state (`engine.EngineState`): its geometry
snapshot G*, closure coefficients x*, relaxed bootstrap λ_BS* and, when on,
the delivery correction. It

1. samples the kinetics, Z_eff, the auxiliary channels and one inductive
   candidate ([§3.2](#32-profile-sampling)), and takes the run's bootstrap
   scale, drawn uniformly from `GenerationConfig.jBS_scale_range` (default
   `(0.99, 1.01)`; 1.0 is the reconstruction's value);
2. composes the current with x* held ([§5.2](#52-the-composition-and-the-three-current-buckets)),
   the pressure-driven term from the draw's own p', and closes one row, I_p,
   as a scalar amplitude on the inductive term (zero extra solves);
3. runs one self-consistent bootstrap loop (`jbs_loop.run_jbs_loop`, ceiling
   `jbs_max_passes_draw = 12`, the closure-half current residual gated);
4. runs the coil homotopy ([§15.4](#154-progressive-homotopy)) and a
   post-homotopy bootstrap stage (ceiling `jbs_max_passes_post_homotopy = 6`);
5. records l_i(3), l_i(1), beta_N, beta_p, Lambda, q0, q95 and the axis
   position; it matches none of them.

Every perturbed quantity is formed as `base + (drawn - base)`. With every
perturbation zero and scale 1.0 the first request is bit-identical to the
reconstruction's stored request (recorded per draw as
`identity.pass1_request_bit_identical`; `EngineDrawContext` refuses to run
otherwise). `Bouquet.verify_sigma0_consistency()` runs one such draw through
the full `generate()` route and gates it ([§5.4](#54-the-self-consistent-loop-and-its-tolerances)).

Drawn kinetics outside the bootstrap model's domain (a non-finite node,
n_e, n_i, T_e, T_i ≤ 0, Z_eff < 1) are rejected before any solve
(`kinetics_nonphysical`, `engine_draws.check_draw_kinetics`). Every solve in
a draw is capped at `engine_draw_solve_maxits = 100` GS iterations; a draw
over `draw_wall_budget_s = 600` s of wall time is rejected (host-dependent).

### 3.2 Profile sampling

One `numpy.random.Generator` per run (`sampling.make_rng(seed)`) feeds every
draw in a fixed order (`engine_draws.RNG_STREAM`): n_e, T_e, then Z_eff
(deriving n_i) or n_i, then T_i, redrawn together until the pressure matches
([§3.4](#34-pressure-matching)); then each auxiliary channel; then one
inductive candidate.

- **Kernel.** `sampling.GPRProfilePerturber`, squared-exponential (the
  default of `generate_perturbed_GPR`; Matérn-5/2 available), covariance
  `C_ij = sigma_i sigma_j K_ij` with unit-variance `K`, so the marginal
  standard deviation equals the input sigma; a 1-D length-scale array gives
  the Gibbs kernel. Length scales, in the run coordinate:
  `UncertaintyConfig.n_ls = 0.5`, `t_ls = 0.4`, `j_ls = 0.25`; auxiliary
  channels and Z_eff 0.4.
- **Factorization.** A non-pivoted Cholesky of `C + jitter I`,
  `jitter = 1e-10 * n * max(diag C)` (`sampling._CHOL_JITTER`), so a seed
  gives the same draw on any LAPACK build up to rounding; each marginal sigma
  is inflated by at most a relative ~3e-8, and an all-zero sigma is exact.
- **Monotonicity.** n_e, T_e, T_i and an independently drawn n_i are redrawn
  until non-increasing in the run coordinate, at most
  `sampling._MAX_MONOTONIC_DRAWS = 10 000` attempts; the constraint is dropped
  for a channel whose baseline is itself non-monotonic. Z_eff and the
  auxiliary channels are not constrained.
- **Inductive candidate.** One sample of the parallel inductive, drawn in
  toroidal units with marginal sigma `sigma_jphi = jphi_scalar_sigma ×
  |baseline j_phi|` (`jphi_scalar_sigma = 0.1`) and length scale `j_ls`,
  redrawn (at most 500 times) only while it is negative where its mean exceeds
  its sigma; negative excursions where the mean is within sigma of zero are
  clipped to zero (`engine_draws.sample_inductive`).
- **Sigma sources** (`baseline.resolve_uncertainty`, first that applies per
  channel): an explicit `UncertaintyConfig.sigma_profiles[chan]`; an IDA file
  (`UncertaintyConfig.ida_path`, else the source's own `.cdf`); an IMAS
  source's own `*_error_upper`; else the assumed H-mode flat-top radial
  template `uncertainties.sigma_template(psi_N, chan) × |profile|` (or
  `<chan>_scalar_sigma × |profile|` when set). The winner per channel is
  logged and returned as `sigma_source`.

**Assumption:** profile uncertainties are Gaussian and fully described by the
covariance kernel. Non-Gaussian structure (ELM or sawtooth phases, pedestal
bifurcation) is not represented.

### 3.3 li Matching: Recon-Anchor + Adaptive Gate

**The default route does not match l_i.** An engine draw closes only the I_p
row; l_i(3) drifts with the draw and is recorded. An l_i band is an opt-in
acceptance filter ([§3.5](#35-opt-in-bands)), not a construction step.

**Legacy route (deprecated since 1.4.0, removed in 1.5, #90).**
`TokaMaker_interface.perturb_kinetic_equilibrium` constructs each draw so that
its l_i lands within a construction tolerance: `GenerationConfig.l_i_tolerance`
when set, else `filtering.LEGACY_L_I_CONSTRUCTION_TOLERANCE = 0.05` (fraction
of the target; `filtering.l_i_construction_tolerance`). The draw first solves
a recon-anchor state with the reconstruction's own inductive
(`input_jinductive`), then either the standard route (inductive candidates
screened with a cylindrical l_i proxy, then a secant on the inductive
amplitude, at most `sampling._MAX_LI_ITER = 20` steps) or route R2
(`perturb_jind_in_anchor=True`: the inductive is perturbed in the anchor
block and I_p sets its amplitude). The construction tolerance is recorded in
the scan's `inspec_bands_json` and is not an acceptance filter. The legacy
route is not replay-deterministic
([docs/known-limitations.md](docs/known-limitations.md)).

### 3.4 Pressure matching

The kinetic channels are redrawn together until the flux-surface integral of
the draw's thermal pressure `e (n_e T_e + n_i T_i)` on the engine grid
matches the baseline's within `p_thresh = 0.05` (a fraction: 5 %;
`kinetic_sampler.sample_kinetics`), at most `sampling._MAX_PRESSURE_ITER =
int(1e5)` attempts. The accepted draw keeps its sampled shapes; nothing is
rescaled.

### 3.5 Opt-in bands

`filtering.InspecBands`, all off by default, applied to the archived draw
(a draw outside a band is archived with `in_spec=False`, never dropped):

| band | setting | test |
|---|---|---|
| l_i(3) | `l_i_tolerance` (fraction) or `l_i_sigma` (absolute 1σ) | `\|l_i - l_i*\| <= l_i_tolerance * l_i*`, or `<= inspec_band_nsigma * l_i_sigma` |
| beta_p | `beta_p_target` with `beta_p_sigma` | `\|beta_p - beta_p_target\| <= inspec_band_nsigma * beta_p_sigma` |
| q0 | `constrain_sawteeth` (default `False`) | q0 ≥ 1 at ψ_N = `psi_pad` |

`l_i*` is the reconstruction's l_i(3); `inspec_band_nsigma = 2.0`; a missing
or non-finite banded quantity fails. With `constrain_sawteeth` set and the
reconstruction's own q(0) < 1, `generate_bouquet` switches the gate off for
the scan and records why (`inspec_bands_json["q0_gate"]`).

### 3.6 What Is Perturbed vs. Held Fixed

Default route:

| Quantity | Perturbed? | Notes |
|----------|-----------|-------|
| n_e, T_e, T_i | yes | Gaussian-process draws, pressure-matched |
| Z_eff | yes | on by default for every source; `zeff_scalar_sigma=0.0` turns it off |
| n_i, n_z | derived | quasi-neutrality with the drawn Z_eff ([§4](#4-quasi-neutrality-and-impurity-handling)) |
| solve pressure | recomputed | from the drawn kinetics; fixed `p_fast` added |
| bootstrap current | recomputed | Redl on each solved state, self-consistent ([§5.3](#53-the-bootstrap-model)) |
| inductive current | yes | one Gaussian-process candidate; its amplitude closes I_p |
| closure coefficients x*, plasma current I_p | no | the reconstruction's |
| coil currents | yes | solved within the homotopy bounds ([§15](#15-coil-constraint-handling-diii-d-reference)) |
| auxiliary channels (rotation, E_r, transport coefficients) | optional | perturbed and stored when a sigma is supplied; passive (they do not enter the solve) |
| `p_fast`, `j_NBI`, `j_RF`, `j_other`, fast-ion moments `z_fast`, `z2_fast` | no | fixed, added to every draw |
| p-file rotation profiles (`omeg`, `omegp`, `omgvb`, `kpol`) | no | kept from the baseline p-file |
| diamagnetic and E×B rotation, E_r, Hahm-Burrell rate of the per-draw p-file | recomputed | from the draw's kinetics and geometry ([§6](#6-rotation-and-radial-electric-field)) |
| `w_ExB` dataset | no | a zero placeholder ([§6.4](#64-the-w_exb-dataset)) |

## 4. Quasi-Neutrality and Impurity Handling

### 4.1 The Z_eff-primary scheme

Each draw perturbs {n_e, T_e, T_i, Z_eff} and derives the main-ion density
from quasi-neutrality with one effective impurity charge Z_imp and main-ion
charge 1 (`physics.main_ion_density_from_zeff`):

```
n_i = n_e (Z_imp - Z_eff) / (Z_imp - 1)          (no fast ions)
n_z = (n_e - n_i) / Z_imp
```

For `1 <= Z_eff <= Z_imp` this gives `0 <= n_i <= n_e` and `n_z >= 0`. The
derived n_i is an increment on the baseline n_i,
`n_i_draw = n_i_base + n_i(n_e_draw, Z_eff_draw) - n_i(n_e_base, Z_eff_base)`,
so a zero perturbation returns the baseline n_i exactly even when the baseline
is not single-impurity quasi-neutral (`kinetic_sampler.sample_kinetics`). One
Z_eff realization per draw is used in the solve, the bootstrap, the archive
and the per-draw p-file ([§4.5](#45-the-per-draw-p-file)).

On the standard ion route n_i is always derived from the drawn Z_eff. On the
measured IDA ion routes it is derived only when n_i and Z_eff come from one
IDA resolution or n_i has only the assumed sigma; otherwise n_i keeps its own
sigma channel (`baseline.resolve_uncertainty`). `UncertaintyConfig.ni_from_zeff`
overrides either. Which Z_eff definition feeds what on each route:
[docs/physics-notes.md](docs/physics-notes.md#z_eff-which-definition-feeds-what-on-every-route).

### 4.2 The impurity charge

Z_imp is the declared value where the baseline carries one (`Baseline.Z_imp`,
set on the IMAS routes); otherwise the sampler inverts it from the baseline as
the median, over surfaces with `(n_e - n_i)/n_e > 1e-3` and `Z_eff > 1`, of

```
Z_imp = 1 + n_e (Z_eff - 1) / (n_e - n_i)
```

(`physics.effective_impurity_charge`, via `kinetic_sampler.dilution_Z_imp`).
`ReconstructionSource.impurity_Z` and `ImasSource.impurity_Z` (default 6.0,
carbon) are the charge with which the IDA reader and the IDA-hybrid IMAS
route derive n_i from Z_eff (`io.ida.read_ida`,
`io.imas._merge_ida_kinetics`). Set it for the device.

### 4.3 Fast ions

With fast-ion charge moments `z_fast = Σ Z_s n_s` and `z2_fast = Σ Z_s² n_s`
the formula depends on whether Z_eff's numerator includes the fast ions
(`physics.main_ion_density_from_zeff`):

```
thermal numerator:   n_i = (Z_imp (n_e - z_fast) - Z_eff n_e) / (Z_imp - 1)
total numerator:     n_i = n_e (Z_imp - Z_eff)/(Z_imp - 1) - (Z_imp z_fast - z2_fast)/(Z_imp - 1)
```

The Z_eff window that keeps n_i ≥ 0 and n_z ≥ 0 follows the same convention
(`physics.zeff_bounds`) and reduces to `[1, Z_imp]` without fast ions. The
moments are held fixed in every draw, so the sampled Z_eff is conditional on
a fixed beam fraction ([docs/known-limitations.md](docs/known-limitations.md)).

### 4.4 Clips

`UncertaintyConfig.kinetic_clips` (default `None`, resolved on for every
route; an explicit `False` turns them off) enables in
`kinetic_sampler.sample_kinetics`: a drawn Z_eff floored at 1
(`ZEFF_DRAW_MIN`); a derived n_i held in `[0, n_e - z_fast]`; an independent
n_i capped at `n_e - z_fast` when Z_imp is declared; the same window on a
passive Z_eff auxiliary draw. With or without clips a Z_eff draw is held in
the `zeff_bounds` window. Every clip that moves a node is counted on the draw
(`KineticDraw.clips`, archived with the sampler version
`kinetic_sampler.KINETIC_SAMPLER_VERSION`).

### 4.5 The per-draw p-file

For a text p-file source, each draw's p-file is the baseline p-file with
n_e, T_e, n_i, T_i replaced, a single-species beam block from `z_fast`,
`z2_fast` when present, the impurity density recomputed from this draw's
(n_e, n_i) by `PFile.compute_quasineutrality()`
(`nz1 = (ne - ni - Z_beam nb) / Z_imp`, negatives clipped to zero here; on
its own the method only warns), then `ptot` and the rotation profiles
recomputed (`generate_bouquet`). A binary IDA `.cdf` source stores none.

`PFile` identifies ion species by charge, mass and the presence of `nb`, not
by position (`PFile.ion_roles`): the beam is the last species only with a
non-zero `nb`; the main ion is the first hydrogenic species (Z = 1,
0.5 ≤ A ≤ 3.5); the impurity is the first other species. It raises when a role
is ambiguous; a non-hydrogenic main ion is not supported.

## 5. Current Density Decomposition

### 5.1 Conventions

bouquet's toroidal current density is TokaMaker's: the plain flux-surface
average `<j_phi>` that `jphi-linterp` consumes, `J_TM = <R> p' + <1/R> FF'/mu0`
(identity A3 of [docs/current-conventions.md](docs/current-conventions.md)).
Field-aligned components are held as parallel currents `<j.B>` [T A/m²]
and converted with one factor everywhere in the package
(`physics.field_aligned_conversion`):

```
<j_phi>_component = kappa <j.B>_component,      kappa = F <1/R> / <B^2>
```

`physics.parallel_to_toroidal`, `toroidal_to_parallel` (its exact inverse),
`evaluate_jBS`, `engine.conversion_factor` and the converted SWB output all
use κ; a geometry carrying `<1/R^2>` is refused. IMAS `j_tor`
(`<j_phi/R>/<1/R>`) converts to and from `J_TM` with identity A5
(`physics.jtor_imas_to_jphi_tokamaker`, `jphi_tokamaker_to_jtor_imas`).

### 5.2 The composition and the three current buckets

The engine forms the solver's current in one place (`engine.compose`), on the
geometry of the latest solve:

```
<j_phi> = kappa [ s_ind <j.B>_ind + s_bs <j.B>_BS + <j.B>_fix ]
        + p' (<R> - F^2 <1/R> / <B^2>)                         (identity I2)
```

`s_ind`, `s_bs` are the closure's multipliers (by default four
peak-normalized Gaussians at ψ_N 0.15, 0.45, 0.75, 0.95, width 0.20,
`utils.STRUCTURED_BASIS_DEFAULT`). The last term, p'G, is the
pressure-driven current (diamagnetic plus Pfirsch-Schlüter, zero `<j.B>`),
recomputed every pass from the solved equilibrium's p'; `F`, `<B^2>` come
from `evaluate_jBS`, `<R>`, `<1/R>`, p' from `utils.fsa_current_geometry`.

The archive keeps that term as its own bucket, `j_pressure`
(`schema.SPLIT_PRESSURE_SEPARATE`, group attribute
`current_split_convention = "pressure_separate"`):

```
j_phi = j_inductive + j_BS + j_NBI + j_RF (+ j_other) + j_pressure
```

`j_BS` is the field-aligned `kappa s_bs <j.B>_BS` only, so the bootstrap
multiplier and jitter never act on p'G. All routes archive this split (the
legacy route evaluates p'G on the archived state,
`TokaMaker_interface.archived_pressure_term`). An archive written before
PR #64 has no attribute and carries p'G inside `j_inductive`;
`schema.read_current_split_convention` infers which. Engine draws also store
the parallel parts (`jB_parallel/`, identity
`j_phi = kappa (jB_inductive + jB_BS + jB_NBI + jB_RF) + j_pressure`).

The archived `j_phi` is the achieved flux-surface current of the converged
solve, not the request (`_achieved_jphi_fsa`); the engine evaluates `j_BS`
and the fixed parts on the archived equilibrium and `j_inductive` is the
unclipped residual. `jphi-linterp` delivers a current that differs from the
request near the edge; `engine_delivery_correction` (default `False`) would
add the previous pass's `request - achieved/c` to each request.

### 5.3 The bootstrap model

The bootstrap is the Redl et al. (2021) fit [A. Redl et al., Phys. Plasmas
**28**, 022502 (2021)], the successor of the Sauter fit [O. Sauter,
C. Angioni and Y.R. Lin-Liu, Phys. Plasmas **6**, 2834 (1999)], evaluated by
`physics.evaluate_jBS` on whatever equilibrium the solver holds: OFT's
`redl_bootstrap(formula_form='jboot1')` with NRL electron / Z_avg ion Coulomb
logarithms and the Koh multi-species ion collisionality
(`physics.EVALUATE_JBS_VERSION`, `evaluate_jBS/4`). It returns the
field-aligned `kappa <j.B>` and p'G separately.

- The drive is the electrons and the main ions; impurity and fast-ion
  pressure enter the GS pressure, not the bootstrap drive.
- The inverse aspect ratio is `eps = (R_max - R_min)/(R_max + R_min)` of each
  surface and `R_geo = (R_max + R_min)/2` enters ν* (`eps_definition =
  "r_over_R_geo"`, default; alternatives in `physics.EPS_DEFINITIONS`).
- Gradients are taken on the caller's own grid with the current
  equilibrium's flux range; the kinetic profiles are held at their ψ_N (or
  Φ_N) labels while the current redistributes.
- Non-physical input (n, T ≤ 0, Z_eff < 1) or a failed surface trace raises
  `JBSEvaluationError`; only on the clipped axis and separatrix end nodes is a
  non-finite value set to zero (counted in `diag["n_nonfinite_zeroed_at_ends"]`).
- Every engine Redl profile receives the shared innermost-surface repair
  (`smooth_jbs_transition`).

The Redl fit is the only bootstrap model implemented; its accuracy limits
(low aspect ratio, strong rotation, non-Maxwellian populations, steep
pedestals) are those of the published fit.

### 5.4 The self-consistent loop and its tolerances

`jbs_loop.run_jbs_loop` owns every solve that contains a bootstrap: closure on
the previous solved geometry, relax the solved current by β, one GS solve,
Redl on the new equilibrium, relax the bootstrap by ω. Both relaxations change
only the path, not the fixed point. `r_j` is the current-weighted L2 distance
between the bootstrap the equilibrium was solved with and Redl on that
equilibrium; `r_I` is the same residual as a fraction of I_p. The engine
requires `GenerationConfig.jbs_self_consistent = True`; a frozen bootstrap
(`False`) exists only on the legacy route.

Convergence (each value read from its home; `engine.convergence_table()`):

| criterion | value | source |
|---|---|---|
| r_j | 1e-3 | `GenerationConfig.jbs_rtol_j` |
| r_I | 1e-4 | `GenerationConfig.jbs_rtol_Ip` |
| \|Δl_i\| between passes | 1e-3 | `GenerationConfig.jbs_tol_li` |
| \|Δq0\| between passes (q0 row on) | 2e-3 | `GenerationConfig.jbs_tol_q0` |
| \|q0 − q0_target\| (q0 row on) | 0.01 | `GenerationConfig.q0_tol` |
| l_i, g-file hard row (absolute) | 1e-3 | `engine.gfile_li_row_tol()` (the `li_tol` default of `_rematch_li_request`) |
| l_i, IDS soft-row discrepancy | 0.005 | `GenerationConfig.structured_li_tol` |
| MSE tan γ change | 0.1 σ_eff | `jbs_loop.MSE_CHORD_OFFSET_TOL_SIGMA` |
| consecutive passes | 2 | `jbs_loop.JBS_REQUIRED_CONSECUTIVE` |
| pass ceiling: reconstruction / draw / post-homotopy | 12 / 12 / 6 | `jbs_max_passes`, `jbs_max_passes_draw`, `jbs_max_passes_post_homotopy` |
| relaxation ω / β | 0.7 / 0.7 | `jbs_relax`, `jbs_relax_current` |
| ω floor; halving after growing passes; growth abort | 0.25; 3; 3 passes at the floor | `JBS_RELAX_FLOOR`, `jbs_relax_halve_on`, `JBS_GROWTH_ABORT_PASSES` |
| closure scale bounds | 0.2 < s < 5 | `utils.close_ip_structured` default |
| I_p round trip | 0.05 % | `utils.IP_ROUNDTRIP_TOL_PCT` |

A non-converged loop raises `JBSNotConverged` (or, with
`jbs_loop_on_fail="flag"`, is delivered flagged); a non-finite iterate raises
`JBSNonFinite` at once. The reconstruction's default rows are I_p and l_i
(`engine_rows = ("Ip", "l_i")`):

| source | I_p row | l_i row |
|---|---|---|
| g-file | hard, exact | hard, target = the reader's `li(2)` key (the ITER-normalized l_i(3), [§9](#9-internal-inductance)), absolute tolerance 1e-3 |
| IDS | soft, σ = 0.5 % of I_p | soft, target = the source's `li_3` at the magnetic-axis radius, σ = 0.04 |

The IDS `li_3` radius is resolved by `GenerationConfig.imas_li3_radius =
"auto"` (`adapters.resolve_li3_radius`). The q0 and MSE rows are opt-in
([docs/engine.md](docs/engine.md)).

**sigma=0 gates** (`Bouquet.verify_sigma0_consistency`): the
zero-perturbation draw must meet r_j, r_I, |Δl_i| and |Δq0| against the
reconstruction at the loop tolerances above, at its loop stage and at its
archived stage; its maximum F-coil and VSC-coil drift must lie within the
hard coil bound the route rejects at (the tightest homotopy stage, 1 % by
default) and its LCFS RMS deviation within the in-spec boundary cut.

### 5.5 The inductive component

- **g-file.** The file carries only the total current. The adapter forms
  `<j.B>_in = F p' + F' <B^2>/mu0` from the reader's own traced surfaces
  (identity I0, `adapters.gfile_parallel_current`), subtracts the Redl
  bootstrap evaluated once on the anchor solve and the fixed parts, and
  smooths the residual (`adapters.inductive_basis`: a smoothing spline of
  degree `ReconstructionSource.n_k` (default 5) with `s = n var * 0.1` on
  ψ_N ≤ `psi_bridge` (default 0.99) with a zero anchor at ψ_N = 1, the raw
  residual blended in below ψ_N 0.30, PCHIP, floored at 0). That shape is
  held for the whole reconstruction and scaled by the closure, so the g-file
  fixed point depends on the anchor's bootstrap through that one subtraction.
- **IDS.** The inductive is the parallel residual by definition,
  `|B0| (j_total - j_bootstrap - Σ driven)`; the source's `j_ohmic` is a
  cross-check recorded in `provenance["inductive_consistency"]`, not a choice
  (`GenerationConfig.engine_ids_inductive = "residual"`; `"j_ohmic"` and
  `"auto"` are explicit options).

### 5.6 Fixed driven currents

User `j_NBI`, `j_RF`, `j_other` (toroidal, `FixedComponentsConfig`) are
converted to parallel with the anchor's κ. On an IDS source the driven
`core_sources` entries are added by their data-dictionary identifier
(`adapters.IDS_DRIVEN_SOURCE_PARTS`: nbi 2 → `nbi`; ec, lh, ic 3-5 → `rf`;
fusion 6, runaways 501, sawteeth 701 → `other`); aggregates, ohmic,
bootstrap and the neoclassical entry are never added. Every driven part is
held fixed in every draw; the engine archives `other` inside `j_RF`. The
source-time matching rules:
[docs/engine.md](docs/engine.md#the-contract-adaptersenginecontract).

### 5.7 Legacy-route notes

On the deprecated legacy route the bootstrap can be frozen
(`jbs_self_consistent=False`: one `solve_with_bootstrap` per baseline and per
draw), and then the thermal pressure uses the frozen constant
`physics.ELEMENTARY_CHARGE_LEGACY = 1.6022e-19` C
(`physics.thermal_pressure_charge`). `ReconstructionSource.shelf_psi_N`
(default 0.0, off) and `rescale_j_BS` (default `False`) apply only to the
legacy g-file reconstruction's inductive fit (`baseline.py`); the engine's
g-file adapter does not read them. SWB's bootstrap output convention is
identified from the installed toolkit and converted to `kappa <j.B>`
(`physics.swb_jbs_convention`, `_swb_jbs_to_toroidal`); an unrecognized build
is refused.

## 6. Rotation and Radial Electric Field

### 6.1 Diamagnetic rotation in the per-draw p-file

`PFile.compute_diamagnetic_rotations(psi)`, for species s:

```
omega_dia,s = (1 / (n_s Z_s e)) d(n_s T_s)/dpsi
```

In p-file units (n in 10^20 m^-3, T in keV, ψ in Wb/rad) this gives krad/s.
The code takes the magnitude and applies the sign by convention: impurity
(`omgpp`) and main-ion (`ommpp`) negative, electron (`omepp`) positive. Main
ions and electrons are taken with charge 1, the impurity charge from
`PFile.ion_roles`. All ions share `ti` unless an impurity temperature `TI` is
passed.

### 6.2 E×B decomposition, E_r and the Hahm-Burrell rate

`PFile.compute_rotation_decomposition(R, Bp, Bt, psi)`:

```
omgeb = omgvb + omgpp              (omgvb from the baseline p-file; zero if absent)
ommvb = omgeb - ommpp
omevb = omgeb - omepp
er    = omgeb * R_mid * Bp_mid                         [kV/m]
omghb = (R_mid Bp_mid)^2 / Bt_mid * d(omgeb)/dpsi
```

R, Bp, Bt are the reader's outboard-midplane values of the draw's own
archived g-file ([§2.6](#26-outboard-midplane-g-file-reader)).
`d(omgeb)/dpsi` is a Savitzky-Golay first derivative (window
`max(7, round(0.03 n) | 1)` points, polynomial order 3), because `omghb` is
effectively a second derivative of the kinetic profiles. The baseline p-file's
rotation profiles are recomputed by the same method when the baseline is
archived, so baseline and draws are comparable.

### 6.3 Radial electric field from impurity force balance

`bouquet.radial_field_from_impurity_force_balance` (public;
`radial_field_from_cer` is a deprecated alias) evaluates the steady radial
force balance of the measured impurity species, inertia and viscosity
neglected:

```
E_r = (1/(Z_imp e n_imp)) dp_imp/dR  -  v_theta B_phi  +  v_phi B_theta
p_imp = e n_imp T_imp,   v_phi = omega_tor R,   dp/dR = (dp/dpsi_N)(dpsi_N/dR)
```

Inputs are SI on a common ψ_N grid; the info dict returns the three terms
(`diamagnetic`, `toroidal` = `v_phi B_theta`, `poloidal` = `-v_theta B_phi`)
and an optional propagated 1σ. `B_theta` is the poloidal field for
right-handed (r, θ, φ), −B_Z at the outboard midplane for either I_p / B_T
direction; `v_pol` is v_θ along the same θ̂; other signs are the inputs' own,
and `B_phi`, `omega_tor`, `v_pol` must share one lab frame (`B_phi = F/R`
from the input g-file, not a bouquet-written one). Since 1.4.1
`io.ida.read_ida_cer` returns `Bpol = -Bpol_midplane` (IDA stores B_Z); an
E_r built with that reader in 1.4.0 or earlier has the wrong sign of the
`v_phi B_theta` term. Agreement with IDA's E_r was checked on
normal-orientation cases only. The library does not use E_r internally
(tests: `tests/test_physics.py::TestRadialField`, `tests/test_ida.py`).

### 6.4 The `w_ExB` dataset

The per-draw `w_ExB` dataset is a zero placeholder on every route
(`TokaMaker_interface.perturb_kinetic_equilibrium`,
`GenerateEngineDraws.draw`); a self-consistent E×B rotation would need the
radial force balance per draw, which is not implemented.

## 7. Pressure and Beta

### 7.1 The solve pressure

The adapter's pressure (`adapters.EngineContract.pressure`), in Pa:

```
p = e (n_e T_e + n_i T_i)  +  p_imp  +  p_fast
p_imp = e n_z T_i,   n_z = max((n_e,th - n_i)/Z_imp, 0)      (physics.impurity_pressure)
```

with `e = physics.ELEMENTARY_CHARGE = 1.602176634e-19` C, n in m^-3, T in eV;
the impurity is assumed thermalized at T_i. `p_fast` is fixed:
`FixedComponentsConfig.p_fast`, else a p-file's `pb` (g-file route), else the
isotropized IMAS fast pressure ([§7.5](#75-fast-pressure-from-imas-sources)).

**The impurity term is evaluated only where Z_imp is declared.** On the IMAS
routes it is (`Baseline.Z_imp`, with `n_e,th = n_e - z_fast`). On the g-file
route the adapter reads Z_imp from `FixedComponentsConfig`, which has no such
field, so `p_imp = 0` and the solve pressure is
`e (n_e T_e + n_i T_i) + p_fast` (`adapters.GFileAdapter.read`; the same holds
for the legacy g-file reconstruction, `baseline.py`).

A draw's pressure is the contract's in increment form, `base + (drawn - base)`
per part (`EngineDrawContext.pressures`). The archive stores the solve
pressure as `pressure` and `e (n_e T_e + n_i T_i)` as `pressure_thermal`.

### 7.2 Separatrix pressure and the edge P' pin

TokaMaker integrates P' inward from zero at the boundary and rescales it to an
axis target (`edge_pressure.py`):

- `GenerationConfig.separatrix_pressure = "offset"` (default): the axis target
  is `p_axis - p_sep`, so P' is the input's own; `p_sep` (the total solve
  pressure at its last node) is added back wherever pressure, beta or stored
  energy is reported or delivered, including every g-file bouquet writes.
  `"legacy"` passes the full axis pressure, which inflates P' by
  `p_axis / (p_axis - p_sep)`.
- `GenerationConfig.edge_pprime_pin = True` (default): the last P' node is
  zero. `False` keeps the profile's own edge derivative.

A pressure that is `p_sep` inside the boundary and zero outside is not
physical; only P' enters the GS equation and `p_sep` is bookkeeping
([docs/physics-notes.md](docs/physics-notes.md#the-pressure-handed-to-the-solver-separatrix-pressure-and-the-edge-p-pin)).

### 7.3 P-file total pressure

```
ptot = _NT_TO_KPA * (ne*Te + (ni + nz1)*Ti) + pb       [kPa]
_NT_TO_KPA = e * 1e20 = 16.02176634 kPa per (10^20 m^-3 keV)
```

(`PFile.compute_pressure`; `nz1`, `pb` default to zero). The p-file's `ni` is
the main-ion density.

### 7.4 Beta

Archived and engine-recorded betas are the solver's (`get_stats()`), reported
in the full-pressure frame with `p_sep` added back
(`edge_pressure.pressure_frames`). The archived `beta_p` and `Lambda` follow
`filtering.LAMBDA_DEFINITION`:

```
beta_p  = 2 mu0 <p>_V / B_pa^2,   B_pa = mu0 I_p / L_p     (L_p: LCFS poloidal circumference)
Lambda  = beta_p + l_i(3)/2                                (no "-1")
```

with `<p>_V` the volume average of the full pressure (thermal + fast, `p_sep`
added back). Lambda is reported, never filtered.

The g-file reader's betas (`GEQDSKEquilibrium.betas`, from `PRES`) describe
the input file: `beta_t = <p>_V / (Bt_vac^2/(2 mu0))`, `Bt_vac = BCENTR
RCENTR / R_geo(LCFS)`; `beta_p` with `B_pa = mu0 I_p / L` (L from the file
boundary); `beta_N = 100 beta_t / (I_p[MA] / (a Bt_vac))`.

### 7.5 Fast pressure from IMAS sources

Anisotropic `pressure_fast_perpendicular` / `_parallel` are reduced to the
scalar GS pressure by a rule that depends on the producer's storage
convention (`physics.isotropize_fast_pressure`): `"trace"`
`(2 p_perp + p_par)/3`, `"mean"` or `"perp"` for fields holding the full
directional pressures; `"sum"` `p_par + 2 p_perp` for fields holding a third
each (IMAS.jl / FUSE). `FixedComponentsConfig.p_fast_reduction = "auto"`
detects the convention from the data dictionary's provenance
(`io.imas.detect_p_fast_convention`) and otherwise falls back to `"sum"` with
a warning (`io.imas.P_FAST_UNDETERMINED_FALLBACK`). The two families differ by
a factor of 3 on the same input.

## 8. Flux-Surface Geometry and Averaging

### 8.1 Two geometry sources

- **The solver's.** Every closure, Redl evaluation, l_i and current integral
  of the engine uses TokaMaker's own flux-surface quantities on its mesh
  (`sauter_fc`, `get_q`, `utils.fsa_current_geometry`, `get_globals`), sampled
  between `psi_pad` and `1 - psi_pad`.
- **The g-file reader's** (`io.geqdsk.GEQDSKEquilibrium`). It supplies the
  g-file adapter's input current `<j.B>_in` and l_i target, the archive
  readers' and plots' quantities, and the midplane values of the per-draw
  p-file. [§8.2](#82-the-g-file-readers-geometry) describes it.

### 8.2 The g-file reader's geometry

- **Tracing.** Flux surfaces are traced with `contourpy` on the 2-D ψ(R, Z)
  grid; the main contour is the one with the largest angular-winding score
  about the axis (`_select_main_contour`). Interior surfaces (ψ_N < 0.99) are
  resampled to 257 arc-length-uniform points with a periodic cubic spline,
  near-separatrix surfaces (0.99 ≤ ψ_N < 1) with a non-periodic spline that
  keeps the X-point cusp; contours with fewer than 20 points are used raw.
- **X-point.** An open contour is closed by `_crop_at_xpoint`: the X-point is
  taken as the midpoint of the contour's gap, and the segment between the two
  closest passes that winds around the axis is kept. This assumes one X-point
  per open contour; a double-null contour is not treated separately.
- **Averages.** `<Q> = oint Q dl/|Bp| / oint dl/|Bp|` with trapezoidal
  arc-length weights. |Bp| is floored at `max(1e-6 max|Bp|, 1e-14)` on each
  surface, which smooths the X-point singularity; averages very close to the
  separatrix carry that smoothing.
- **Separatrix and axis.** At ψ_N = 1 the reader uses the file's
  `RBBBS`/`ZBBBS`, closed and unresampled; with fewer than 4 boundary points
  the surface is replaced by a degenerate contour at the axis without an
  error. At the axis, q is extrapolated quadratically from the first three
  interior surfaces, the elongations linearly from the first two, and the
  triangularities are set to zero.
- **Current densities**, analytic from the file's p' and FF':
  `j_tor_averaged` = `<Jt/R>/<1/R>` (the IMAS / OMFIT convention) and
  `j_tor_averaged_direct` = `<Jt>` = `-sigma_Bp (p' <R> + FF' <1/R>/mu0) (2π)^exp_Bp`
  (TokaMaker's, used by the engine). They are identities A4 and A3 of
  [docs/current-conventions.md](docs/current-conventions.md) and differ in
  both terms: by `p' (1/<1/R> - <R>)` and, on the FF' term, by
  `<1/R^2>/<1/R>` against `<1/R>`. A5 converts one into the other.

### 8.3 The boundary constraint

The unified engine holds one standard isoflux set for the reconstruction and
every draw (`solver_geometry.standard_isoflux`): the source boundary
resampled by arc length to `N_ISOFLUX = 256` points with the source's
X-points kept exactly, per-length weights summing to `ISOFLUX_WEIGHT_TOTAL =
2e5`, the outermost point first. The total is a constant chosen inside a
window measured on one device and the synthetic fixtures, not a per-case
calibration (#93); `SolverConfig.isoflux_weight_total` overrides it and a
`SolverConfig.isoflux_pts` set is used as given. The partition between
boundary and coil stiffness is reported (`boundary_balance`, warned outside
`[1/3, 3]`), never acted on
([docs/engine.md](docs/engine.md#the-boundary-constraint-a-constant-total-inside-a-measured-window)).

## 9. Internal Inductance

### 9.1 The estimator the chain runs on

```
l_i(3) = 2 int B_p^2 dV / ((mu0 I_p)^2 R_axis)
```

TokaMaker `get_stats(li_normalization='iter')`, `utils.li_achieved(...,
"li_3")`, the engine's l_i rows and the opt-in l_i band all use it; the
archive states it in the `_baseline` attribute `l_i_scale = "iter(li3)"`
(`utils.LI_SCALE`). The IDS `li_3` target is rescaled to the axis radius when
the source's own radius can be determined ([§5.4](#54-the-self-consistent-loop-and-its-tolerances)).

Per draw the archive stores `l_i(3)` and `l_i(1)` = TokaMaker `get_stats`
with `li_normalization='iter'` / `'std'` and `lcfs_pad=psi_pad`;
`utils.li_achieved(..., "li_1")` uses the LCFS contour's own perimeter
instead (`utils.lcfs_perimeter`).

### 9.2 g-file reader keys

`GEQDSKEquilibrium.li`, with `int B_p^2 dV = Σ_k (oint 2π |B_p| dl)_k Δψ_k`
and L, V from the file boundary (`RBBBS`/`ZBBBS`; V by Pappus):

| key | formula |
|---|---|
| `li(1)`, `li(1)_EFIT` | `(int B_p^2 dV / V) / (mu0 I_p / L)^2` |
| `li(2)` | `2 int B_p^2 dV / ((mu0 I_p)^2 R_axis)`: the ITER l_i(3) at the axis (the key name is historical; issue #20) |
| `li(3)` | `2 int B_p^2 dV / ((mu0 I_p)^2 RCENTR)` |
| `li(1)_TLUCE` | `li(3)` key × `(1 + kappa^2)/(2 kappa_a)`, `kappa` of the boundary, `kappa_a = V/(2π RCENTR π a^2)` |

The g-file adapter's l_i target is the `li(2)` key. Codes and databases use
different l_i definitions; check which one a comparison needs.

## 10. Numerical Floors and Clamps

| location | floor or clamp | purpose |
|----------|----------------|---------|
| p-file diamagnetic `dpsi` | `max(1e-4 max\|dpsi\|, 1e-30)` | axis singularity |
| p-file diamagnetic densities (denominator only) | `1e-4 max\|ne\|` | edge / zero-density singularity |
| p-file rotation `Bt` | `sign(Bt) 1e-6` T where \|Bt\| < 1e-6 T | `omghb` division |
| p-file rotations, `omghb` | non-finite → 0 | |
| p-file `Zeff` where n_e → 0 | non-finite → 1 | |
| reader flux-surface `\|Bp\|` | `max(1e-6 max\|Bp\|, 1e-14)` per surface | X-point weight |
| GP covariance | jitter `1e-10 n max(diag C)` | Cholesky well-posed |
| drawn Z_eff | `zeff_bounds` window; floor 1 with clips | n_i ≥ 0, n_z ≥ 0 |
| derived n_i (clips on) | `[0, n_e - z_fast]` | quasi-neutrality |
| inductive candidate | negative excursions → 0 where the mean is within σ of 0 | |
| adapter Z_eff; `p_imp` impurity density; per-draw p-file `nz1` | clipped below at 1; at 0; at 0 | |
| Redl end nodes (ψ_N ≤ psi_pad, ≥ 1 − psi_pad) | non-finite → 0, counted | |
| closure scales | 0.2 < s < 5, else refused | closure validity |
| coil bound half-width | `max(drift_F \|I_base\|, 50 A)` | small-baseline coils ([§15.3](#153-homotopy-bounds)) |
| `#VSC` bound scale | `max(min\|I_pair\|, 10 kA)` | near-zero VSC coil |
| homotopy saturation | drift ≥ 0.99 × installed bound | active-constraint detection |

None of these is a user setting except the clips (`kinetic_clips`) and the
coil floor (`coil_drift_floor_A`, a `generate_bouquet` keyword).

## 11. Edge Treatment

- **g-file edge extrapolation.** Many equilibrium codes write p'(1) =
  FF'(1) = 0. With `extrapolate_edge=True` (the reader's default, used by the
  g-file adapter), when the last value is exactly zero and the one before is
  not, the last node is replaced by the quadratic extrapolation
  `3 y[-2] - 3 y[-3] + y[-4]`, set to zero if that reverses sign
  (`GEQDSKEquilibrium._trace_surfaces`). Only the last node changes; on the
  g-file route it enters `<j.B>_in` there.
- **Solver edge.** The last P' node is zero by default and the separatrix
  pressure is offset ([§7.2](#72-separatrix-pressure-and-the-edge-p-pin)).
- **Bootstrap edge taper** (off by default). `bootstrap_kwargs=
  {"taper_edge_jBS": True}` multiplies every composed term by OFT's edge taper
  (1 below `taper_edge_psi0 = 0.999`, quintic smoothstep to 0 at the LCFS;
  `physics.edge_taper_weight`).
- **Kinetic grid.** Kinetics reach the run grid by PCHIP (`utils.pchip_interp`);
  an IDA grid past the separatrix is kept as `psi_N_kinetic`.

## 12. Data Format Assumptions

### 12.1 GEQDSK (g-file)

- Fixed format, 5 values per line, 16 characters each; SOLPS-style files
  (spaces between numbers, lines longer than 80 characters) and row-by-row
  `PSIRZ` blocks (FIESTA) are handled (`io.geqdsk._read_geqdsk`).
- **Assumption:** a uniform rectangular (R, Z) grid. Br, Bz and Jt come from
  `numpy.gradient` (second-order differences), noisier at the grid edges;
  the reader's current densities use p', FF' instead ([§8.2](#82-the-g-file-readers-geometry)).

### 12.2 P-file (Osborne format)

- Blocks of `count x-name y-name(units) description`; the `N Z A of ION
  SPECIES` block is parsed separately and species roles are identified as in
  [§4.5](#45-the-per-draw-p-file).
- **Assumption:** every profile uses `psinorm` (ψ_N). The x-name is not
  checked: a file on another coordinate is read as ψ_N without a warning.
- `PFile` remaps every profile onto the `ne` grid by default (`remap=True`)
  and keeps the as-read profiles for an unmodified round trip.

### 12.3 IDA-layout profile fits

`io.ida.read_ida` reads the netCDF4 file with h5py: 2-D `(n_time, n_radial)`
profiles with `*_err` 1σ companions on a `psi_n` grid that extends past the
separatrix, time in ms, SI units. There is no stored main-ion density: by
default n_i is derived from (n_e, Z_eff) with `impurity_Z`; the measured
routes (`ni_source` `"Zeff"`, `"CER"`, `"all"`) are in
[docs/physics-notes.md](docs/physics-notes.md#kinetic-assumptions-z_eff-n_i-and-the-clips-pr-56).

### 12.4 IMAS / OMAS data dictionaries

`io.imas.read_imas_baseline` reads the JSON with the standard library.
Currents are converted with the paired equilibrium slice's own `f`,
`gm1/gm5/gm8/gm9`, `dpressure_dpsi` (traced from `profiles_2d.psi` when
omitted): the total by A5, each component by its field-aligned part
([§5.1](#51-conventions)). The dictionary is COCOS 11 (ψ per full turn); the
export (`write_imas_draw`, `export_imas_drawset`) converts the archived
g-file's ψ, P', FF' with `psi_11 = -2π psi` (it treats that g-file as
COCOS 7). **Since 1.4.1** the exported `profiles_2d.psi` is written
`[i_R][i_Z]` (dim1 = R first); every earlier export is transposed (#101), and
the reader's tracer (`io.imas._fsa_from_profiles_2d`) refuses an array whose
shape is not `(len(dim1), len(dim2))`.

## 13. HDF5 Archive

Each run writes one HDF5 file, schema v3 (`schema.SCHEMA_VERSION = 3`);
[docs/archive-schema.md](docs/archive-schema.md) is the reference. Each scan
point or time slice is a group `scan/<scan_key>/` with its own `config_json`,
holding `_baseline/` (the reconstruction) and one group per archived draw
(`<count>/`). Profile datasets carry bare names
with the unit in `attrs["units"]`; the raw
g-file and p-file bytes (`eqdsk`, `pfile`) round-trip through
`GEQDSKEquilibrium.from_bytes` / `PFile.from_bytes`. Both `_baseline` and
every draw store `coil_currents`, so drifts can be recomputed without a
solve. Out-of-spec draws are archived; selection is by flags written
non-destructively by `Bouquet.filter()` (`passes_coil_filter`,
`passes_boundary_filter`, `passes_draw_band`, `selected`). v3 adds the
self-consistent bootstrap record (`jbs_loop` attributes); a group without it
carries a frozen bootstrap.

## 14. Unit Conventions

### 14.1 Bouquet internal (SI)

| Quantity | Unit |
|----------|------|
| n_e, n_i, n_z, `z_fast`, `z2_fast` | m^-3 |
| T_e, T_i | eV |
| Pressure | Pa |
| Toroidal current density `<j_phi>` | A/m² |
| Parallel current `<j.B>` | T A/m² (raw, not IMAS `<j.B>/B0`) |
| ψ | Wb/rad (COCOS 1-8; the IMAS export is COCOS 11, Wb per full turn) |
| B | T |
| R, Z | m |
| Coil currents | A as archived; the DIII-D tolerances are stated in ampere-turns of the mesh coil sets |
| Boundary deviation | mm |
| `w_ExB` | rad/s |

### 14.2 P-file

| Quantity | Unit |
|----------|------|
| ne, ni, nz1, nb | 10^20 m^-3 |
| te, ti | keV |
| ptot, pb | kPa |
| Rotation (`omeg`, `omgeb`, ...) | krad/s |
| er | kV/m |

### 14.3 Conversion constants

| Conversion | Value | Used in |
|------------|-------|---------|
| `physics.ELEMENTARY_CHARGE` (eV → J) | 1.602176634e-19 | every thermal pressure and the Redl drive on the default route |
| `physics.ELEMENTARY_CHARGE_LEGACY` | 1.6022e-19 | the frozen legacy path only ([§5.7](#57-legacy-route-notes)) |
| `io.pfile._NT_TO_KPA` | 16.02176634 | p-file `ptot` |
| m^-3 → 10^20 m^-3 | 1e-20 | p-file density |
| eV → keV | 1e-3 | p-file temperature |
| Pa → kPa | 1e-3 | p-file pressure |

## 15. Coil Constraint Handling (DIII-D Reference)

Every draw is a free-boundary solve; the coils carry the boundary. Bouquet
keeps a draw's coil currents near the reconstruction's with bounds on
TokaMaker's coil least-squares problem, applied through a progressive
homotopy, and judges the result with the in-spec rule. The DIII-D numbers
come from the device registry (`devices.DEVICES["DIII-D"]`); the user-facing
summary and the VSC metric's derivation are in
[docs/coil-constraints.md](docs/coil-constraints.md).

### 15.1 Coil solve and regularization

- Every GS solve uses OFT's bounded (BVLS) coil least squares, entered once in
  `Bouquet.setup_solver` with ±1e98 bounds that never bind
  (`solver_state.enter_bounded_coil_mode`).
- A draw's bootstrap loop solves under the reconstruction's own coil
  regularization (`GenerateEngineDraws.install_coil_reg`); the homotopy
  installs the strong one (`TokaMaker_interface.strong_coil_reg`): every coil
  set pulled toward the reconstruction's current at `soft_reg_weight = 1e4`,
  `#VSC` toward zero at `vsc_soft_reg_weight = 1.0`. `generate_bouquet`
  restores the reconstruction's regularization and releases the bounds on
  every exit.
- The engine's homotopy holds the reconstruction's own isoflux pattern
  (`GenerateEngineDraws.homotopy_isoflux`); the legacy route re-traces the
  draw's LCFS at weight 200 per point. There is no per-draw I_p
  re-alignment.

### 15.2 Coil classes on DIII-D

| class | members | treatment |
|---|---|---|
| shaping F-coils | F1A-F8A, F1B-F8B | bounded per coil; judged by the χ² test |
| vertical-control pair | F9A, F9B (`SolverConfig.coil_vsc`, anti-series) | bounded per coil and on the `#VSC` channel; judged by the χ² test |
| E-coils | ECOILA, ECOILB (or the six-circuit split on the finer mesh) | bounded per coil, with the 50 A floor; judged by the χ² test |

The device is named in `BouquetConfig.device` or detected from the exact set
of coil names in the mesh (`devices.detect_device`); there is no partial
match. The legacy drift metric's "F-coil" class is every coil whose name
starts with `F` and is not in the VSC pair (`_coil_max_drifts`), a DIII-D
naming assumption.

### 15.3 Homotopy bounds

At a stage `(drift_F, drift_VSC)` (`TokaMaker_interface._homotopy_coil_bounds`):

```
every coil set (VSC pair included):
    delta = max(drift_F * |I_base|, coil_drift_floor_A)        coil_drift_floor_A = 50 A
    bounds[name] = [I_base - delta, I_base + delta]
the #VSC channel:
    scale = max(min(|I_F9A,base|, |I_F9B,base|), 10 kA)        (_vsc_bound_scale)
    bounds['#VSC'] = ±max(drift_VSC * scale, coil_drift_floor_A)
```

`I_base` is the reconstruction's coil current. The worst-case total drift of a
VSC-pair coil is `drift_F + drift_VSC`. The 10 kA floor
(`_COIL_DRIFT_DENOM_FLOOR_A`) lifts the bound for a pair with one near-zero
coil; it changes a homotopy bound, not an acceptance gate.
`coil_drift_hard_factor` (default `None`) adds an outer bound at
`± coil_drift_hard_factor * coil_drift * |I_base|` in every solve, with the
same floored `#VSC` scale.

### 15.4 Progressive homotopy

After an engine draw's bootstrap loop, `GenerationConfig.homotopy_passes`
(default `[(0.05, 0.10), (0.02, 0.05), (0.01, 0.01)]`) is tried in order under
the strong regularization, each stage warm-started from the previous one and
capped at `engine_draw_solve_maxits`. After each stage:

- a stage whose drift reaches `COIL_SATURATION_FRACTION = 0.99` of an
  installed bound is treated as infeasible: roll back to the last good stage
  and stop; if no stage succeeded the draw is rejected (`homotopy_saturated`);
- a stage whose drift already exceeds the next stage's bound stops the
  schedule there;
- a failed solve rolls back to the last good stage (a re-solve restores the
  solver's flux-surface state) and stops; with no good stage, or a failed
  rollback, the draw is rejected.

The post-homotopy bootstrap stage (ceiling 6 passes, saturation-guarded)
follows. `homotopy_pass` is the index of the last successful stage (−1 when
no bounds ran); `engine_draw_homotopy=False` replaces the homotopy by a
measured drift. A capped stage with `#VSC` at its bound and the axis moved
vertically by more than `engine.VERTICAL_RUNAWAY_AXIS_FRAC = 0.1` a is
labeled `vertical_runaway` (not a filter). `guard_bistability=True` (default
`False`) is an opt-in experimental diagnostic (#87); the `SKIP_HARD`,
`SKIP_ISO`, `SKIP_HOMOTOPY` environment variables are diagnostic bypasses.

### 15.5 The in-spec rule

A draw is in spec when it passes the configured coil test AND the boundary
cut AND every opt-in band that is on, and was not rejected before archiving
(`filtering` module docstring; verdict formed in `generate_bouquet` through
`filtering.passes_all_filters`). The same predicate gives the archived
`in_spec` attribute (`in_spec_source = "inspec_rule"`), the until-N count
(`GenerationConfig.n_inspec_target`, default `None` = exactly `n_equils = 20`
attempts; when set, the attempt cap `max_total_draws` defaults to
`5 × n_inspec_target`, never below `n_equils`) and `Bouquet.filter()`'s
`selected` flag. The scan records the rule applied (`inspec_rule`:
`"magnetics"` by default, with `"+li"`, `"+beta_p"`, `"+q0"` for bands that
are on); `bouquet.refilter_inspec` re-judges an archive under another rule
from the stored numbers.

The magnetics constrain I_p, the boundary and a shape-weighted combination
of the current and pressure moments, which the free-boundary solve with these
two tests applies; l_i(3) and beta_p alone are inferences, so their bands are
opt-in ([§3.5](#35-opt-in-bands)).

### 15.6 The coil χ² test

`FilterConfig.coil_filter = "chi2"` (default; `coil_spec.coil_chi2`):

```
z_i     = (I_i,draw - I_i,base) / sigma_i
chi2/nu = mean_i z_i^2
pass    = chi2/nu <= chi2_max  AND  max_i |z_i| <= z_max
```

over every baseline coil with σ > 0; a draw with no coil currents fails.
σ (`coil_spec.resolve_coil_sigma`) is `FilterConfig.coil_sigma`, else the
device's model; for DIII-D `sigma_i = hypot(floor_i, 0.0035 |I_i,base|)` in
ampere-turns, era floor 325 A-t (after the coil-current acquisition upgrade)
or 825 A-t (before), refined per coil and clipped below at 100 / 250 A-t; the
era is `FilterConfig.coil_daq_era`, the tightest (latest) with a warning when
unknown. Acceptance (`coil_spec.resolve_coil_acceptance`): explicit
`chi2_max` / `z_max`, else the device's calibration of its default model
(DIII-D: `chi2/nu <= 6.1`, `|z| <= 6.3`, 95th percentiles on real flat-top
slices over the 18 F-coils), else 4.0 / 5.0 (`devices.GENERIC_ACCEPTANCE`).
The E-coils ride on the F-coil model as an extrapolation, and `max|z|` over
20 or 24 coils rejects above the nominal 5 % (`devices` module docstring).
The test judges draws relative to the reconstruction's coil set, not the
reconstruction's distance from the measured currents. An unregistered device
falls back loudly to the legacy band ([§15.8](#158-the-legacy-drift-verdict-and-per-draw-attributes)).

### 15.7 The boundary cut

`FilterConfig.rms_max_mm = "auto"` (default): the device's calibrated cut,
8.5 mm for DIII-D, else the generic 5.0 mm (`devices.boundary_cut_for`); a
number sets it; `"off"` or `None` disables it. The deviation is the RMS over
the reconstruction's LCFS points of the nearest-neighbor distance to the
draw's LCFS (`filtering.boundary_deviation_mm`; reference `recon_lcfs_ref`,
else the baseline g-file boundary). A missing contour scores NaN and fails a
cut that is on.

### 15.8 The legacy drift verdict and per-draw attributes

The flat ±% band is kept beside the rule as `coil_drift_in_spec` (never a
filter), and is the coil test with `coil_filter="legacy"` or as the χ²
fallback:

```
max_F_drift_pct   <= 100 * FilterConfig.inspec_F_max      (0.02)
max_VSC_drift_pct <= 100 * FilterConfig.inspec_VSC_max    (0.02)
```

`max_F_drift_pct` is the largest `100 (I_draw - I_base)/max(|I_base|, 1 A)`
over the non-VSC F-coils; E-coils are not part of this band.
`max_VSC_drift_pct` decomposes the pair into common-mode `(ΔI_A - ΔI_B)/2`
and differential `(ΔI_A + ΔI_B)/2` channels:

```
sigma_VSC         = hypot(I_A,base, I_B,base)/2 + 10 kA
max_VSC_drift_pct = 100 max(|ΔI_cm|, |ΔI_df|) / sigma_VSC
```

(`_vsc_channel_drift_pct`; derivation in
[docs/coil-constraints.md](docs/coil-constraints.md#vsc-drift-metric-anti-series-pair)).

Every draw group carries `homotopy_pass`, `homotopy_F_lim`,
`homotopy_VSC_lim`, both drift metrics, `coil_drift_in_spec`, `in_spec`,
`in_spec_source` and the band thresholds (table in
[docs/coil-constraints.md](docs/coil-constraints.md#per-draw-attributes)).
An archive written before 2026-10-15 has no `in_spec_source`; its `in_spec`
is the drift verdict (`filtering.in_spec_fields`).

## 16. Known Limitations and Future Work

[docs/known-limitations.md](docs/known-limitations.md) is the maintained
list, each entry with its consequence and tracking issue (bistable
near-double-null equilibria #87, MSE-stage runtime #88, the boundary-weight
total #93, the p-file beam density #97, IDA versus data-dictionary
consistency #89, the experimental swb route #81, the legacy route #90). This
section keeps only limitations that follow from architectural assumptions.

- **E×B rotation is not computed.** `w_ExB` is a zero placeholder
  ([§6.4](#64-the-w_exb-dataset)); the per-draw p-file's E×B decomposition
  uses the baseline's impurity V×B rotation (`omgvb`), or zero when absent.
- **Profiles are flux functions on a fixed label.** Kinetic profiles are
  perturbed as functions of ψ_N (or Φ_N) and held at their labels while the
  current redistributes; measurements at one poloidal location are treated as
  flux-surface values, which matters for strongly up-down asymmetric plasmas.
- **Gaussian uncertainties with a fixed kernel**; without a measured sigma,
  an assumed H-mode flat-top template ([§3.2](#32-profile-sampling)).
- **One effective impurity, one ion temperature.** Quasi-neutrality uses one
  effective impurity charge and main-ion charge 1; the impurity pressure and
  the p-file diamagnetic rotation take the impurity at T_i. Multiple
  impurities and a decoupled impurity temperature are not modeled. On the
  g-file route the impurity pressure term is zero ([§7.1](#71-the-solve-pressure)).
- **Fixed components are not perturbed.** `p_fast`, the driven currents and
  the fast-ion charge moments are held fixed (the beam fraction's own
  uncertainty is not sampled); rotation and transport channels are perturbed
  only as passive auxiliary channels.
- **One bootstrap model**, the Redl fit ([§5.3](#53-the-bootstrap-model)).
- **Scalar-pressure Grad-Shafranov.** Anisotropic fast-ion pressure is reduced
  to a scalar ([§7.5](#75-fast-pressure-from-imas-sources)).
- **The g-file reader's geometry.** The reader assumes one X-point per open
  contour and floors |Bp| on each surface ([§8.2](#82-the-g-file-readers-geometry));
  its near-separatrix averages and its geometry for double-null files carry
  those assumptions. The engine's own geometry comes from the solver
  ([§8.1](#81-two-geometry-sources)).
