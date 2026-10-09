# Specification of the three IMAS-path test fixtures

Replaces (owner decision D3) the quarantined `dd_synthetic.json.gz`,
`diiid_profs_synthetic.cdf` and `g_synthetic.geqdsk` (see `README.md`). Every
field below was found by tracing what the code reads on the paths the two
dependent tests run: `Bouquet.from_imas(..., ida_path=..., LCFS_geqdsk=...,
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
| `equilibrium.time` / `time_slice[]` | 3 slices (0.98, 1.00, 1.02 s) | the reader takes Ip, li_3 from the slice nearest the run time and pairs the current conversion with the nearest slice or the LAST slice BEFORE the core_profiles time (FUSE evaluates `j_tor` on the previous slice) | 1 (then `j_tor` must be built on that slice); 2 (one before t_cp) to keep exercising the previous-slice pairing |
| `core_profiles.time` / `profiles_1d[]` | 1 slice | 1 | 1 |
| `core_sources.time` | 1 | within half the local core_profiles step of the core_profiles time read | 1 |
| `core_sources.source[]` | 21 entries (12 beams, gas, exchange, brem, 2 x line, synch, bootstrap, ohmic, sawteeth) | at least one beam (`identifier.index = 2`) with non-zero `j_parallel` (j_NBI; swb's `jphi_fixed`) | 1 beam; add one `ohmic` (7) and one `bootstrap` (13) entry to keep testing that aggregates are never added, and one `sawteeth` (701, zero current) for the sawtooth gate |
| `equilibrium.time_slice[].profiles_2d` | one 65 x 65 rectangular psi | NOT read while `gm1/gm5/gm8/gm9` are present (only the fallback that traces them reads it) | none |

Run time used by both tests: `_TIME = 1.013` s (IDA slice `time = 1013` ms).
With a new fixture at t = 1.000 s, change `_TIME` in both tests together.

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

`core_sources`
* `time`; `source[].identifier.index`, `.name`; `source[].profiles_1d[].time`,
  `.j_parallel` (on the core_profiles grid)

### Consistency the code checks (and the tolerance it uses)

| relation | where | tolerance / action |
|----------|-------|--------------------|
| `j_tor` = A6(`j_total`) on the paired equilibrium slice's geometry (f, gm1, gm5, gm9, dpressure_dpsi, b0) | `_paired_current_geometry` | median relative mismatch <= 1e-3, else a warning; the quarantined file reproduces it to 2e-16 |
| `j_total` = `j_ohmic` + `j_bootstrap` + sum of source `j_parallel` | not checked by the reader (the inductive is the residual) | keep exact (the exporter round trip and FUSE assume it) |
| `dpressure_dpsi` = d(`pressure`)/d(`psi`) on each slice | used as p' (A5-A7, p' = -2 pi dpressure_dpsi) | exact to the spline used |
| `rho_tor_norm` = sqrt(Phi/Phi_edge), Phi = integral of q dpsi, strictly increasing | Phi_N placement (`coords.phi_n_from_q`, `_phi_n_from_rho`) | refused if non-monotone or the sqrt(psi_N) placeholder |
| gm1 >= gm9^2 (Jensen), gm5 >= f^2 gm1, gm8 gm9 >= 1 | the conversions assume them | physical |
| sign of every current (`j_tor` net, `j_total`, `j_bootstrap`, beam `j_parallel`, equilibrium `j_tor`) = sign(ip) | `_refuse_mixed_orientation` | REFUSED otherwise |
| psi_N(rho) of the dd vs of the g-file at rho = 0.2 ... 0.9 | `_psi_rho_drift` | 2e-2, else a warning (the quarantined set misses it: -0.038 at rho 0.4) |
| IDA n_i vs (dd total n_i - fast) at psi_N 0, 0.2, ..., 0.8 | `_subtract_fast_ni` | 1e-2 relative, else a warning (the quarantined set: 4.8e-2) |
| core_sources slice / each entry's own time vs the core_profiles time | `core_sources_slice`, `_source_slice_at` | half the local core_profiles step, else REFUSED (an entry carrying current) |

## 2. `diiid_profs_synthetic.cdf` (IDA layout, netCDF4/HDF5, "direct" form)

| dataset | shape | read |
|---------|-------|------|
| `time` | (1,) int or float, ms | slice selection (1013 now) |
| `psi_n` | (n,) | the fit grid; starts at 0, may extend past 1 (now 150 nodes to 1.2) |
| `n_e`, `T_e`, `T_12C6`, `Zeff`, `n_12C6`, `omega_tor_12C6` | (1, n) | kinetics (T_12C6 is T_i) |
| `n_e_err`, `T_e_err`, `T_12C6_err`, `Zeff_err`, `n_12C6_err` | (1, n) | sigma envelopes |
| `q` | (1, n) | the fit equilibrium's q: Phi_N placement of the IDA nodes |

Not read: `omega_tor_12C6_err`. Minimum n ~ 50 (the pedestal must be resolved).
Consistency: `q` must be the g-file's q at the same psi_N (the IDA file "was
fitted on" that equilibrium); n_e, T_e, T_i, Z_eff, n_C must satisfy
n_i = n_e - 6 n_C >= 0 and Z_eff = (n_i + 36 n_C)/n_e within 1e-2 (the reader
compares the two routes, report-only); sigmas are currently constant
fractions (n_e 4 %, T_e 8 %, the rest 10 %) -- keep explicit fractions.

## 3. `g_synthetic.geqdsk` (EFIT g-eqdsk, COCOS 1)

Read through `read_geqdsk`: `boundary_R/Z` (the LCFS: isoflux targets of every
solve), `QPSI` (rho_tor -> the psi_N(rho) drift check), `R_grid`, `Z_grid`,
`psi_N_RZ`, `R_mag`, `Z_mag` (the Phi_N test's outboard-midplane radii). Now
65 x 65 (the EFIT grid) -- use the example's TokaMaker 257 x 257 output or any
grid; F0 is taken from the dd (r0 b0), not from `BCENTR`. Must be the
equilibrium the IDA fit and its q belong to, with a current profile that
DIFFERS from the dd's (see test 1 below), and an LCFS inside the D3D-like mesh
(`examples/D3D-like/DIIID_mesh.h5`).

## 4. What the dependent tests assert (what a replacement must reproduce)

No stored numbers are compared: every bar is relative or statistical.

`tests/test_phi_imas_solver.py` (`-m solver`, only with `BOUQUET_RUN_DEMOS=1`;
needs an OFT with toroidal-flux support): one baseline per method (legacy,
swb, engine) x coordinate (psi_n, phi_n); for the IDA nodes with
psi_N in [0.1, 0.98]:
* psi_N-held kinetics move by `max_dr_mm > 10` mm at the outboard midplane
  (measured on the quarantined set: up to 25 mm, core) -- the dd's current
  profile must move the surfaces by more than 10 mm relative to the g-file;
* Phi_N-held move by `< min(10, 0.4 x psi-held)` mm (measured: <= 5 mm);
* `ne_misfit` (max |n_e(run, R) - n_e(IDA, R)| / max n_e) of the Phi_N run
  `< 0.5 x` the psi_N run's (measured 0.45 % vs 1.7 %).

`tests/test_swb_systematics_solver.py` (`-m solver`; needs an OFT whose
solve_with_bootstrap takes x, jphi_fixed, p_fixed): one swb baseline, then
* sigma=0 draw after a perturbed draw = solve B bit for bit (li_3, q0, Ip,
  coils, LCFS, kinetics at psi_N 0.1/0.5/0.9, the inductive seed);
* pressure-only (10 draws): seed relative change < 1e-9 (measured 0);
  boundary RMS < 1.5 mm (measured <= 0.64); coil change < 4 % (<= 2.1 %);
  |signed-mean| boundary shift < min(2 mm, 4 se) (<= 0.21 mm); li_3, q0
  means within 4 se;
* production (16 draws): every draw's |Ip/Ip_target - 1| < 1e-4 (measured
  <= 8e-6); boundary means as above; li_3, q0, n_e, T_e, T_i, n_i, Z_eff and
  the seed at psi_N 0.1/0.5/0.9 within 4 se of the baseline (<= 1.9 se);
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
(`physics.evaluate_jBS(...)[1]["j_dot_B"]`) + one analytic beam, and `j_tor`
from `j_total` by A6 on the PREVIOUS slice (`io.imas._jtor_from_jpar`), so the
reader's pairing check passes at round-off.
