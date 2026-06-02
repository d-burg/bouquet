# D3Dlike_Hmode_baseline.geqdsk — generation recipe

A **synthetic, non-proprietary** DIII-D-like H-mode equilibrium, derived from
the 204441@4400 shape but built from generic profiles + a free-coil TokaMaker
solve so it is no longer the proprietary reconstruction.

## Inputs
- **Magnetic target:** 204441 g-file (`g204441.04409_719`) — boundary shape, Ip, F0.
- **Profiles:** generic D3D-like p-file `TkMkr_D3Dlike_Hmode_BSamp=1.0.peqdsk`
  (ne0≈5.5e19, Te0≈2.5 keV — distinct from 204441's IDA kinetics).
- **Mesh:** DIII-D TokaMaker mesh; `order=3`, `maxits=800`.

## Procedure
1. Weak coil regularization toward **zero** (free inverse solve), VSC weight 1e-2.
2. **Reshape** the 204441 target boundary about its center: `R' = cR + 0.99*(R-cR)`,
   `Z' = cZ + 0.98*(Z-cZ) + 0.01` (≈5% less elongated, raised ~1 cm) → distinct boundary.
3. **Isoflux:** 16 points evenly spaced in poloidal angle on the reshaped boundary, weight 200.
4. **X-point:** true saddle constraint (`set_saddle_constraints`) at the reshaped
   lower X-point (min-Z vertex) → clean diverted divertor.
5. `reconstruct_equilibrium` (Sauter bootstrap via `solve_with_bootstrap` + inductive fit),
   `n_k=5, psi_bridge=0.99, rescale_j_BS=False`.
6. `save_eqdsk(nr=257, nz=257, lcfs_pad=1e-4)` → TokaMaker writes **COCOS 7**.
7. **Convention fix:** `cocosify(7→1)` then a **Bt-only flip** (negate BCENTR + FPOL,
   keep CURRENT/PSIRZ/PPRIME/FFPRIM/QPSI) so the file matches 204441 exactly.

## Result (verified)
- COCOS **1**, Ip = **+0.802 MA**, Bt = **−2.022 T**, ⟨Jt⟩>0, q>0 — **all signs match 204441**.
- l_i(1) = **1.103**, diverted lower X-point.
- Boundary **8.8 mm RMS** distinct from 204441 (max 45 mm).
- Coil currents in the 204441 ballpark (main F-coils; ECOILB/F9B differ slightly,
  profile-driven).

Generator script: `/tmp/gen_d3dlike_eq.py`
(`N_ISO=16 ISO_W=200 USE_SADDLE=1 SR=0.99 SZ=0.98 DZ=0.01 FLIP_BT=1`).
To be ported into `bouquet_D3Dlike_example.ipynb`.
