# D3Dlike_Hmode_baseline.geqdsk — generation recipe

A **synthetic, shareable, generic** DIII-D-like H-mode equilibrium with a
self-consistent kinetic-profile / Ip / bootstrap relationship.

## Why this version (history)
The first attempt paired generic D3D-like profiles (adapted from shot **147131**,
Ip ~1.19 MA) with the 204441 boundary at **0.80 MA**.  That Ip mismatch (ratio
0.67) made the Sauter bootstrap an oversized *fraction* (j_BS peak 0.96 MA/m^2
ABOVE the total j_phi pedestal; bootstrap fraction 0.27), which evacuated the
inductive current past psi_N~0.78 and made l_i collapse ~25% under sigma-
perturbation.  Fixed by building from the **profiles' native shot 147131**, at
generic round-number targets.  (Old version archived in
`legacy/superseded_0p8MA_baseline/`.)

## Inputs (all self-consistent from one shot, 147131 @ 2300 ms)
- **Magnetic source:** `g147131.02300_DIIID_KEFIT` (GPEC kinetic-example default;
  publicly distributed).  Provides boundary shape + native Ip/Bt.
- **Profiles:** `D3Dlike_Hmode_baseline.peqdsk` (147131-derived; matches the shot's
  `.kin` ne/Te/Ti to ~1%).
- **Mesh:** `DIIID_mesh.h5`.

## Targets (generic round numbers, user choice)
- **Ip = 1.20 MA** (≈147131 native 1.187; keeps bootstrap fraction physical)
- **Bt = -2.00 T** (raised |Bt| from 1.72 -> 2.0; raising |Bt| INCREASES q, safe)
- Boundary **mildly reshaped** for a generic look: `R'=cR+0.99(R-cR)`,
  `Z'=cZ+0.97(Z-cZ)+0.02` (kappa 1.78->~1.73, nudged up) -> 21 mm RMS distinct
  from the exact 147131 LCFS, still diverted.

## Procedure (generator: /tmp/gen_d3dlike_147131.py)
0. Read the 147131 source with **nlevels=257** so the jphi-linterp profile is fit
   on a 257-pt psi_N grid (15 points across the pedestal spike vs 7 on the native
   129-pt grid) -> smooth pedestal j_phi (no piecewise-linear kinks).
1. Weak coil reg toward zero (free inverse solve), VSC weight 1e-2.
2. Override eqdsk CURRENT -> 1.20 MA; F0 = 2.0 * R_center (|Bt| target).
3. Isoflux: 16 even-angle points on the reshaped boundary, weight 200.
4. True saddle X-point constraint at reshaped min-Z vertex.
5. `reconstruct_equilibrium` (Sauter bootstrap via solve_with_bootstrap + ind fit),
   n_k=5, psi_bridge=0.99, rescale_j_BS=False.
6. save_eqdsk(nr=257, nz=257, lcfs_pad=1e-4) -> TokaMaker writes COCOS 7 (Bt>0).
7. cocosify(7->1), then Bt-only flip (negate BCENTR+FPOL; keep CURRENT/PSIRZ/
   PPRIME/FFPRIM/QPSI) -> file at COCOS 1, Bt=-2.0 (DIII-D convention).

## Result (verified)
- COCOS **1**, Ip = **+1.200 MA**, Bt = **-2.008 T**, q95 = **4.58**, q0 = 1.28.
- l_i(1) = **0.844** (file) / 0.851 (recon) -- self-consistent round-trip.
- **Bootstrap PHYSICAL:** j_BS peak 0.56 MA/m^2 (well below total j_phi 0.78);
  bootstrap fraction 0.14; inductive current present across full radius
  (min +0.09 MA/m^2, never evacuated).
- Boundary 21 mm RMS distinct from 147131, diverted lower X-point.
- Reconstruction round-trips the file (Ip, li, j_phi all match).

Generator env: `N_ISO=16 ISO_W=200 USE_SADDLE=1 SAD_W=200 IP_TARGET=1200000
BT_ABS=2.0 FLIP_BT=1 SR=0.99 SZ=0.97 DZ=0.02`.
Audit tool: /tmp/audit_recon.py (AUDIT_GEQ=<file> AUDIT_PNG=<png>).
