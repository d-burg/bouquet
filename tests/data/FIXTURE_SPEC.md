# Specification of the IMAS-path test fixtures

The seven generated files (`README.md`) come from `make_synthetic_fixtures.py`
(owner decision D3, 2026-10-11): analytic profile families fitted to real
plasmas with every parameter rounded to two significant figures, each slice
re-solved self-consistently by TokaMaker. Every field below was found by
tracing what the code reads on the paths the two dependent tests run: `Bouquet.from_imas(..., ida_path=..., LCFS_geqdsk=...,
impurity_Z=6.0)` with `solve_method` = legacy, swb and engine, in a psi_N and a
Phi_N run (`read_imas_baseline`, the engine's `IdsAdapter.read`,
`read_imas_geometry`, `read_ida`, `read_geqdsk`), plus what the tests
themselves read. "Required" = read on those paths; "optional" = probed, used
when present; anything not listed is never read.

Conventions: the dd is IMAS COCOS 11 (psi in Wb per full turn, increasing
outward for Ip > 0); currents `j_total`, `j_bootstrap`, `j_ohmic`,
`j_parallel` are `<J.B>/B0`; `j_tor` is the IMAS `<j_phi/R>/<1/R>`
(`docs/current-conventions.md`, A4-A7). Use round numbers: B0 = -2.0 T,
R0 = 1.7 m, Ip a round value (e.g. 1.20 MA), t = 1.000 s; generic channel
names; no device metadata, no producer strings other than the
`p_fast_reduction` stamp below.

## 1. `dd_synthetic.json.gz` (gzipped OMAS/IMAS JSON)

### Time bases

| item | now | required by the tests | minimum |
|------|-----|-----------------------|---------|
| `equilibrium.time` / `time_slice[]` | 5 slices (0.7, 1.1, 1.2, 1.4, 3.0 s; Ip 0.95 MA on the first, 1.0 MA after) | the reader takes Ip, li_3 from the slice nearest the run time and pairs the current conversion with the nearest slice or the LAST slice BEFORE the core_profiles time (FUSE evaluates `j_tor` on the previous slice) | 1 (then `j_tor` must be built on that slice); 2 (one before t_cp) to keep exercising the previous-slice pairing |
| `core_profiles.time` / `profiles_1d[]` | 5 slices (one per equilibrium time) | 1 | 1 |
| `core_sources.time` | 5 (one per equilibrium time) | within half the local core_profiles step of the core_profiles time read | 1 |
| `core_sources.source[]` | 21 entries (12 beams, gas, exchange, brem, 2 x line, synch, bootstrap, ohmic, sawteeth) | at least one beam (`identifier.index = 2`) with non-zero `j_parallel` (j_NBI; swb's `jphi_fixed`) | 1 beam; add one `ohmic` (7) and one `bootstrap` (13) entry to keep testing that aggregates are never added, and one `sawteeth` (701, zero current) for the sawtooth gate |
| `equilibrium.time_slice[].profiles_2d` | one 65 x 65 rectangular psi | NOT read while `gm1/gm5/gm8/gm9` are present (only the fallback that traces them reads it) | none |

Run time used by both tests: `_TIME = 1.200` s (IDA slice `time = 1200` ms;
the pedestal-forming slice, where the dd carries ~110 kA of beam current and
so differs most from the EFIT-like g-file). Change `_TIME` in both tests
together if the pairing moves.

### Fields read

`equilibrium`
* `time`; `vacuum_toroidal_field.r0` (scalar), `.b0` (scalar or one per slice;
  F0 = |r0 b0|, the parallel normalisation, and the orientation stamp)
* `time_slice[k].global_quantities.ip` (sign = current orientation),
  `.li_3` (the l_i target; required), `.li_1` (optional)
* `time_slice[k].profiles_1d`: `psi`, `q`, `pressure`, `j_tor` (the
  equilibrium anchor, `anchor_jtor_to_equilibrium=True` by default),
  `rho_tor_norm`, `f`, `gm1` = <1/R^2>, `gm5` = <B^2>, `gm8` = <R>, `gm9` =
  <1/R>, `dpressure_dpsi`; `area` optional (orientation area weights).
  Grid: 257 points now; any monotone grid works (min ~65; the edge bootstrap
  wants >= ~129).
  Without `f`/`gm*`/`dpressure_dpsi` (and no `profiles_2d`) the reader falls
  back to the per-surface ratio j_tor/j_total with a warning -- the swb method
  then refuses (it needs `j_pressure`), so the fixture MUST carry them.

`core_profiles`
* `time`; `ids_properties.comment` containing `p_fast_reduction=sum` (else a
  loud "undetermined" warning and the 'sum' fallback)
* `profiles_1d[0].grid.psi` (COCOS 11; psi_N from it), `.grid.rho_tor_norm`
  (required: the current conversion interpolates in it, and a Phi_N run
  places nodes by it); `grid.area` optional
* `j_total`, `j_tor`, `j_bootstrap` (required); `j_ohmic`,
  `j_non_inductive` are NOT read (keep them consistent anyway: the IDS
  exporter writes them)
* `electrons.density_thermal`, `.temperature`, `.pressure_fast_parallel`,
  `.pressure_fast_perpendicular`
* `ion[]` (main ion `label = "D"`, `element[0].z_n = 1`; one impurity,
  `z_n = 6`): `label`, `element[0].z_n`, `density_thermal`, `density_fast`,
  `temperature`, `pressure_fast_parallel`, `pressure_fast_perpendicular`,
  `rotation_frequency_tor`; `zeff`, `e_field` optional
* Grid: 257 points, uniform in rho_tor_norm (so NON-uniform in psi_N: this is
  what exercises the psi_N-path node sampling, review E5). Keep it
  rho-uniform.
* Sigma envelopes (optional; read by `io.imas.read_imas_sigmas` as the
  kinetic sigmas of a run with no IDA sigma file, `sigma_mode` "auto" /
  "direct"): `electrons.density_thermal_error_upper`,
  `electrons.temperature_error_upper`, `ion[].density_thermal_error_upper`,
  `ion[].temperature_error_upper`, `zeff_error_upper`. Every slice carries
  them: the radial template of its phase (section 7) times the slice's own
  profile -- 0.7 s L-mode, 1.1 s and 1.2 s ramp, 1.4 s and 3.0 s H-mode flat
  top; main ion n_i takes the derived n_i template, carbon the n_C template,
  both ions the T_i template. Upper only (symmetric). Both dependent tests
  run `from_imas(..., ida_path=...)`, i.e. `ida_hybrid`, where the IDA file's
  envelopes are used and these are not read.

`core_sources`
* `time`; `source[].identifier.index`, `.name`; `source[].profiles_1d[].time`,
  `.j_parallel` (on the core_profiles grid)

### Consistency the code checks (and the tolerance it uses)

| relation | where | tolerance / action |
|----------|-------|--------------------|
| `j_tor` = A6(`j_total`) on the paired equilibrium slice's geometry (f, gm1, gm5, gm9, dpressure_dpsi, b0) | `_paired_current_geometry` | median relative mismatch <= 1e-3, else a warning; the generator reproduces it at round-off |
| `j_total` = `j_ohmic` + `j_bootstrap` + sum of source `j_parallel` | not checked by the reader (the inductive is the residual) | keep exact (the exporter round trip and FUSE assume it) |
| `dpressure_dpsi` = d(`pressure`)/d(`psi`) on each slice | used as p' (A5-A7, p' = -2 pi dpressure_dpsi) | exact to the spline used |
| `rho_tor_norm` = sqrt(Phi/Phi_edge), Phi = integral of q dpsi, strictly increasing | Phi_N placement (`coords.phi_n_from_q`, `_phi_n_from_rho`) | refused if non-monotone or the sqrt(psi_N) placeholder |
| gm1 >= gm9^2 (Jensen), gm5 >= f^2 gm1, gm8 gm9 >= 1 | the conversions assume them | physical |
| sign of every current (`j_tor` net, `j_total`, `j_bootstrap`, beam `j_parallel`, equilibrium `j_tor`) = sign(ip) | `_refuse_mixed_orientation` | REFUSED otherwise |
| psi_N(rho) of the dd vs of the g-file at rho = 0.2 ... 0.9 | `_psi_rho_drift` | 2e-2, else a warning (measured 0.022 at rho 0.6 against the thermal-only `g_synthetic.geqdsk`, so the warning fires: the dd slice carries the fast-ion pressure and its Shafranov shift, the g-file does not; 0.0094 against the total-pressure `g_synthetic_pfast.geqdsk`) |
| IDA n_i vs (dd total n_i - fast) at psi_N 0, 0.2, ..., 0.8 | `_subtract_fast_ni` | 1e-2 relative, else a warning (measured ~1e-5) |
| core_sources slice / each entry's own time vs the core_profiles time | `core_sources_slice`, `_source_slice_at` | half the local core_profiles step, else REFUSED (an entry carrying current) |

## 2. `diiid_profs_synthetic.cdf` (IDA layout, netCDF4/HDF5, "direct" form)

| dataset | shape | read |
|---------|-------|------|
| `time` | (1,) int or float, ms | slice selection (1200 now) |
| `psi_n` | (n,) | the fit grid; starts at 0, may extend past 1 (now 150 nodes to 1.2) |
| `n_e`, `T_e`, `T_12C6`, `Zeff`, `n_12C6`, `omega_tor_12C6` | (1, n) | kinetics (T_12C6 is T_i) |
| `n_e_err`, `T_e_err`, `T_12C6_err`, `Zeff_err`, `n_12C6_err` | (1, n) | sigma envelopes |
| `q` | (1, n) | the fit equilibrium's q: Phi_N placement of the IDA nodes |

Not read: `omega_tor_12C6_err` (a constant 10 % of |omega|; no template). Minimum n ~ 50 (the pedestal must be resolved).
Consistency: `q` must be the g-file's q at the same psi_N (the IDA file "was
fitted on" that equilibrium); n_e, T_e, T_i, Z_eff, n_C must satisfy
n_i = n_e - 6 n_C >= 0 and Z_eff = (n_i + 36 n_C)/n_e within 1e-2 (the reader
compares the two routes, report-only); the five read sigmas are explicit
2-s.f. radial templates (section 7) times the profile: this file the ramp
template (its paired dd slice, 1.2 s, is pedestal-forming), the L-mode file
(section 6) the L-mode template. Until 2026-10-14 they were constant fractions
(n_e 4 %, T_e 8 %, the rest 10 %).

## 2b. `diiid_profs_synthetic_pfast.cdf` (the same IDA fit on the total-pressure g-file)

The same file as section 2 -- the same `psi_n` grid, kinetics and every `*_err`
(the ramp template), bit for bit -- with `q` taken from `g_synthetic_pfast.geqdsk`
(section 3b) instead of `g_synthetic.geqdsk` (max difference 7 %). It is the IDA
file that belongs to the total-pressure g-file: the routes that pair the dd's
1.2 s slice with a g-file (`LCFS_geqdsk`) and an IDA file use this pair, because
the dd slice carries the fast-ion pressure and its Shafranov shift, which the
thermal-only `g_synthetic.geqdsk` does not (written by `make_synthetic_fixtures.py`,
also without a solve: `--pfast-ida-only OUT_DIR` next to an existing
`g_synthetic_pfast.geqdsk`).

## 3. `g_synthetic.geqdsk` (EFIT g-eqdsk, COCOS 1; thermal pressure only)

Read through `read_geqdsk`: `boundary_R/Z` (the LCFS: isoflux targets of every
solve), `QPSI` (rho_tor -> the psi_N(rho) drift check), `R_grid`, `Z_grid`,
`psi_N_RZ`, `R_mag`, `Z_mag` (the Phi_N test's outboard-midplane radii). Now
65 x 65 (the EFIT grid) -- use the example's TokaMaker 257 x 257 output or any
grid; F0 is taken from the dd (r0 b0), not from `BCENTR`. Must be the
equilibrium the IDA fit and its q belong to, with a current profile that
DIFFERS from the dd's (see test 1 below), and an LCFS inside the D3D-like mesh
(`examples/D3D-like/DIIID_mesh.h5`).

**Pressure (since 2026-10-15): thermal only**, the pressure the g-file +
IDA route rebuilds from the IDA file, e (n_e T_e + n_i T_i) with
n_i = n_e (6 - Z_eff)/5 (`physics.main_ion_density_from_zeff`; no impurity
term, no fast ions: the IDA layout has no fast-ion channel), p'(1) = 0
clamped as every fixture solve. Same boundary targets (16 isoflux points and
the X-point saddle), coil set and regularisation, Ip, mesh, current model
(the 1.2 s slice's ohmic shape + Redl bootstrap, no beam) and solver settings
as the total-pressure solve (section 3b), from whose converged state it is
continued. The IDA file's `q` is this file's q. Section 8 has the numbers.

## 3b. `g_synthetic_pfast.geqdsk` + `p_synthetic.peqdsk` (total pressure, fast ions included)

`g_synthetic_pfast.geqdsk` is the same 1.2 s slice and current model with the
TOTAL pressure the dd slice carries, e (n_e T_e + (n_D + n_C) T_i) + p_fast
(it was `g_synthetic.geqdsk` until 2026-10-15; bit-identical). Its kinetics
are the p-file `p_synthetic.peqdsk`, an Osborne p-file in the DIII-D
kinetic-EFIT layout `bouquet.io.pfile` reads: 201 uniform psinorm nodes on
[0, 1], each block `(psinorm, value, d/dpsiN)` written to six decimals:

| block | units | content |
|-------|-------|---------|
| `ne`, `te` | 10^20/m^3, keV | the slice's n_e, T_e (as the IDA file) |
| `ni` | 10^20/m^3 | the THERMAL deuterons n_D = n_e - 6 n_C - n_b |
| `ti` | keV | T_i (as the IDA file's T_12C6) |
| `nb` | 10^20/m^3 | the beam (fast) deuterons |
| `pb` | kPa | the beam pressure: the dd's 1.2 s `pressure_fast_parallel` = `_perpendicular` = p_fast/3 isotropised by `physics.isotropize_fast_pressure` with the dd's 'sum' rule, i.e. p_fast (58.0 kPa on axis, 0 at the separatrix) |
| `nz1` | 10^20/m^3 | carbon, n_C = n_e (Z_eff - 1)/30 |
| `omeg` | kRad/s | toroidal rotation |
| `ptot` | kPa | `PFile.compute_pressure`: n_e T_e + (n_i + n_z1) T_i + pb |
| `N Z A of ION SPECIES` | | 6 6 12 (carbon), 1 1 2 (D), 1 1 2 (beam D): impurity first, main ion, beam ion last |

The reader recomputes n_z1 from quasineutrality, (n_e - n_i - n_b)/6, and
Z_eff = (n_i + 36 n_z1 + n_b)/n_e: equal to the IDA file's Z_eff to 4.6e-5
(the six-decimal rounding). The g-file route takes `pb` as its default
`p_fast` (decision B, 2026-10-17): with `fixed_components.p_fast` unset it uses
1e3 x `pb` (Pa) on `pb`'s own psinorm (coord `"psi_n"`), stamped
`li_metrics["p_fast_source"]["p_fast_source"] = "pfile_pb"`; an explicit
`fixed_components.p_fast` still wins (a warning names the field, once per run).
The `pb` header's unit is checked: this file says `KPa`; any other label warns
and is not converted (stamped `pb_header_units`, `pb_units_ok`). Measured
on `g_synthetic_pfast.geqdsk` + this file at the defaults (engine route, one
reconstruction): beta_N 1.671 against the g-file's 1.684 (-0.7 %), report-only
`pressure_fidelity` relative RMS 0.0055, on axis -0.42 % (silent: it warns
beyond +/-10 % on axis, either sign). Before B the defaults
dropped `pb`: beta_N 0.630 (-62.6 %) with every verdict check passing
(`tests/test_pfile_pfast_default.py` holds the fast tests).

## 4. What the dependent tests assert (what a replacement must reproduce)

No stored numbers are compared: every bar is relative or statistical.

`tests/test_phi_imas_solver.py` (`-m solver`, only with `BOUQUET_RUN_DEMOS=1`;
needs an OFT with toroidal-flux support): one baseline per method (legacy,
swb, engine) x coordinate (psi_n, phi_n); for the IDA nodes with
psi_N in [0.1, 0.98]:
* psi_N-held kinetics move by `max_dr_mm > 10` mm at the outboard midplane;
* Phi_N-held move by `< min(10, 0.4 x psi-held)` mm;
* `ne_misfit` (max |n_e(run, R) - n_e(IDA, R)| / max n_e) of the Phi_N run
  `< 0.5 x` the psi_N run's.
  The test pairs the dd with `g_synthetic_pfast.geqdsk` and
  `diiid_profs_synthetic_pfast.cdf` (section 2b; since 2026-10-16): the premise
  -- the g-file is the equilibrium the dd's slice produces -- does not hold
  for the thermal-only `g_synthetic.geqdsk` (on it the Phi_N-held placement
  measured 18.7 mm against the 10 mm bound). Measured on the pfast pairing
  (2026-10-16, all three methods): psi_N-held up to 4.8-5.7 mm, Phi_N-held up
  to 0.8-1.1 mm, Phi_N-held nearer at every node; ne_misfit 0.58-0.69 %
  (psi_N-held) vs 0.14-0.17 % (Phi_N-held) (re-measured 2026-10-17: the same
  ranges, the Phi_N-held ne_misfit down to 0.14 % on the engine method). The
  bars are unchanged.

`tests/test_swb_systematics_solver.py` (`-m solver`; needs an OFT whose
solve_with_bootstrap takes x, jphi_fixed, p_fixed): one swb baseline, then
(the "measured" values below were re-measured 2026-10-16, and again 2026-10-17
with the same values, on the total-pressure
pairing, `g_synthetic_pfast.geqdsk` as the LCFS source and
`diiid_profs_synthetic_pfast.cdf` as the IDA file -- section 2b)
* sigma=0 draw after a perturbed draw = solve B bit for bit (li_3, q0, Ip,
  coils, LCFS, kinetics at psi_N 0.1/0.5/0.9, the inductive seed);
* pressure-only (10 draws): seed relative change < 1e-9 (measured 4.4e-16);
  every draw passes bouquet's OWN default draw filters as the library
  resolves them for the device (owner decision 2026-10-11: the chi2 coil
  filter, `coil_filter = "chi2"`, PR #49, and the calibrated boundary cut,
  `rms_max_mm = "auto"`, PR #63 -- 8.5 mm, chi2/nu <= 6.1, |z| <= 6.3 on
  DIII-D; measured RMS <= 2.4 mm, chi2/nu <= 3.6, |z| <= 6.1);
  |signed-mean| boundary shift < min(2 mm, 4 se); li_3, q0 means within 4 se;
* production (16 draws): every draw's |Ip/Ip_target - 1| < 1e-4; boundary
  means as above; li_3, q0, n_e, T_e, T_i, n_i, Z_eff and the seed at psi_N
  0.1/0.5/0.9 within 4 se of the baseline (measured on this set: Ip closure
  1.06e-4 on the worst draw -- red, #81 --, the largest bias 2.9 se; boundary
  |signed mean| <= 0.20 mm;
  the Ip closure is reported to the swb route's author; the bars are
  unchanged);
* seeded twins reproduce every archived dataset; another seed differs;
* NO draw rejected in any generation (`run.draw_rejections` empty).
  With the new `swb_jind_redraw_refused` rejection (review PR69 B3), the
  source's inductive current (the seed) must be >= 0 everywhere and
  seed[0] > 0.

## 5. Mutual-consistency recipe the generator follows

See `make_synthetic_fixtures_TEMPLATE.py`. In short: (1) build the g-file by a
TokaMaker solve with analytic (polynomial / Hmode_profiles-style) p' and
current, re-fit LCFS, rounded B0 / Ip; (2) the IDA file on that equilibrium's
psi_N and q from the same analytic kinetics; (3) the dd's equilibrium slices
from a SECOND TokaMaker solve with a different current profile (same
kinetics, the >10 mm surface shift), each slice's gm*/f/p'/rho_tor_norm from
`physics.capture_equilibrium_fsa` converted to COCOS 11, core_profiles on a
rho-uniform 257 grid, `j_total` from analytic ohmic + the Redl bootstrap
(`physics.evaluate_jBS(...)[1]["jdotB_bs"]`) + one analytic beam, and `j_tor`
from `j_total` by A6 on the PREVIOUS slice (`io.imas._jtor_from_jpar`), so the
reader's pairing check passes at round-off.

The g-files of the 1.2 s slice (sections 3, 3b): the total-pressure solve
first (written as `g_synthetic_pfast.geqdsk`, with `p_synthetic.peqdsk` from
the same analytic kinetics), the L-mode next, then the thermal-only solve
continued from the total-pressure state (`g_synthetic.geqdsk`; the IDA file
takes its q). The order keeps every earlier fixture bit-identical to the
generation before the thermal-only file existed.

## 6. The second regime: `g_synthetic_lmode.geqdsk` + `ida_synthetic_lmode.cdf`

A sawtoothing L-mode (q0 0.8), upper single null, Ip 1.3 MA, B0 -2.0 T,
fitted to a real L-mode plasma (kinetics from its IDA fit, no pedestal term;
the source g-file's edge current swings negative at the separatrix and is
replaced by a smooth monotone fall to 0.024 MA/m^2, stated here). Same
layouts as sections 2 and 3. Reserved for g-file + IDA workflows and the
real-data ladder's synthetic rung; no test reads it yet.

## 7. Sigma envelopes: the radial templates

Every kinetic sigma of every fixture is `bouquet.uncertainties.sigma_template`
of the fixture's phase times the synthetic profile itself:
`sigma(psi_N) = max(interp(min(psi_N, 1), GRID, TEMPLATE[q]), FLOOR[q]) * |x|`,
piecewise linear between 14 psi_N nodes (0, 0.1, ..., 0.8, 0.85, 0.9, 0.95,
0.98, 1.0), the psi_N = 1 value held on the IDA files' SOL nodes, 2 s.f. on the
template (the product is not re-rounded). The templates are the median relative
envelopes of IDA-lite deliveries on DIII-D H-mode flat-top, ramp / early
H-mode and L-mode slices, one per quantity (n_e, T_e, T_i, Z_eff and n_C each
from its own envelope, so Z_eff_err is not derived from n_C_err; n_i derived
with independent errors).

| fixture | phase | template |
|---------|-------|----------|
| dd 0.7 s | ramp-up L-mode | lmode |
| dd 1.1 s | L-H transition | ramp |
| dd 1.2 s | pedestal forming | ramp |
| dd 1.4 s | peak pedestal | hmode_flattop |
| dd 3.0 s | dense flat top | hmode_flattop |
| `diiid_profs_synthetic.cdf` | the 1.2 s profiles | ramp |
| `diiid_profs_synthetic_pfast.cdf` | the 1.2 s profiles (q of the total-pressure g-file) | ramp |
| `ida_synthetic_lmode.cdf` | L-mode | lmode |

The phase decides, not the clock: the 1.4 s slice is a developed pedestal and
takes the flat-top template. The g-files carry no kinetics. Written by
`apply_sigma_templates.py` (in place on existing fixtures, refusing any change
to a non-envelope dataset; `make_synthetic_fixtures.py` calls the same
functions); checked by `tests/test_fixture_sigma.py`.

With the carbon tier of the Z_eff ladder (`zeff_sigma_source="auto"` on a
g-file + IDA run) sigma_Z = (Z - 1) sqrt((sigma_nC/n_C)^2 + (sigma_ne/n_e)^2)
follows from these envelopes: wider than the old flat fractions, most at the
edge.

## 8. Pressure consistency of the g-file pairs

Each g-file carries the pressure its profile file gives on the g-file route
(`tests/test_fixture_pressure_consistency.py`, 2 % on axis): e (n_e T_e +
n_i T_i), n_i from Z_eff for an IDA-layout file or the p-file's own `ni`,
plus `pb` for the p-file.

| pair | g-file axis | profile file axis | axis | max over psi_N <= 0.95 |
|------|-------------|-------------------|------|------------------------|
| `g_synthetic.geqdsk` + `diiid_profs_synthetic.cdf` | 46.08 kPa | 46.13 kPa | -0.09 % | 1.47 % (the p'(1) = 0 clamp's offset, where p is small) |
| `g_synthetic_pfast.geqdsk` + `p_synthetic.peqdsk` | 90.40 kPa | 32.03 + pb 58.00 = 90.03 kPa | +0.42 % | 0.42 % |
| `g_synthetic_lmode.geqdsk` + `ida_synthetic_lmode.cdf` | 46.50 kPa | 45.96 kPa | +1.16 % | 1.16 % |

The +0.42 % and +1.16 % are the carbon pressure e n_C T_i, which the two
equilibria carry and the route's thermal pressure omits. The two H-mode
equilibria (TokaMaker on a finer machine mesh, nl_tol 1e-6, the sanity gate passed before
and after each solve):

| | `g_synthetic_pfast` (total) | `g_synthetic` (thermal) |
|---|---|---|
| axis pressure | 90.4 kPa | 46.1 kPa |
| q0 (solver, psi_N 0.02) / g-file axis | 1.307 / 1.316 | 1.405 / 1.417 |
| q95 | 4.85 | 4.77 |
| l_i(3) (solver) | 0.8326 | 0.8349 |
| beta_N / beta_p (g-file) | 1.68 / 0.921 | 0.797 / 0.436 |
| magnetic axis R | 1.762 m | 1.732 m |
| r_j max / rms | 1.0e-4 / 6.4e-5 | 1.2e-4 / 7.4e-5 |
| Newton passes, stalled solves | 8, 0 | 7, 0 |

The LCFS of the two differ by 5.7 mm rms (13 mm at most, on the lower outboard
leg); against the 16 isoflux targets they sit at 7.4 / 8.6 mm rms.
