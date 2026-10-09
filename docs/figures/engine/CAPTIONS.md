# Review figures for the unified-engine line

Every figure here is made from the synthetic examples shipped in
`examples/D3D-like/` (a D3D-like g-file + p-file and a D3D-like OMAS/IDS
file), from the test suite's toy stand-ins, or from closed-form toy models.
No experimental data. PNG at 130 dpi. Provenance is stated per figure:
**synthetic example, real solver** (OpenFUSIONToolkit build with the
`jphi_update` <1/R> fix, one thread) or **toy model / solver-free**.

## Diagrams

**`engine_review_chapters.png`** — Diagram, no data: the twelve review
chapters of the unified-engine pull request, in commit order, landing on
`main` as one merge commit.

**`engine_architecture.png`** — Diagram, no data: one pass of the unified
reconstruction engine as implemented, from the two input adapters through
composition, closure, one relaxed solve, measurement and update to the
two-consecutive-pass criteria, delivery and the draws. The numbers are the
shipped defaults; the coil solve is in the bounded mode from solver setup on.

## Reconstruction

**`engine_vs_legacy_gfile_profiles.png`** — Synthetic example (g-file +
p-file), real solver. The input against the legacy reconstruction path and
the unified engine: achieved flux-surface-averaged current and its
difference to the input, q, pressure, the engine's FF′/p′ split, and the
edge. The q-profile rms deviation over ψ_N 0.05–0.95 is 0.37 % (engine) and
0.22 % (legacy).

**`engine_pass_histories_synthetic.png`** — Synthetic examples (g-file and
IDS), real solver. Per-pass bootstrap, Ip, pass-to-pass l_i and unrelaxed
current residuals of the engine's reconstruction loop against the loop's
shipped tolerances (dashed). g-file with rows Ip + l_i: 4 passes, 8 solves;
IDS with rows Ip + l_i: 5 passes, 9 solves; IDS with the q0 row added:
6 passes, 10 solves.

**`mse_jacobian_fd_chord_vs_broyden_synthetic.png`** — Synthetic IDS example,
real solver, with eight synthetic midplane MSE chords made from the
baseline's own field × 1.03 (σ 0.004). The engine's MSE stage with a fixed
finite-difference Jacobian (the default) against a Broyden-updated one: the
fixed Jacobian converges in 5 MSE passes (23 solves), Broyden in 7
(25 solves) after a non-monotone excursion.

**`engine_ids_inductive_split.png`** — Synthetic IDS example, solver-free
(the repository's reader and IDS adapter). The engine's inductive current is
the parallel residual of the total, bootstrap and driven currents, and the
source's own ohmic current is only a stamped cross-check. The two agree to
round-off (right panel) because the example's current components are summed
consistently.

## Draws and the σ = 0 check

**`engine_sigma0_true_route.png`** — Synthetic examples (g-file and IDS),
real solver. The zero-perturbation check through the true draw route, both
sources and both reconstruction paths: (a) each residual over the loop's own
tolerance, per route and stage, and (b) the engine's archived-state change
in q0, q95 and flux range. Every route passes; the engine's first request is bit-identical to the stored one, the largest residual is 0.31 of its tolerance (IDS engine, Ip), and the archived q0, q95 and flux range move by at most 2.1e-5.

**`engine_draw_ensemble_bands.png`** — Synthetic examples, real solver.
Seeded engine-draw batches (12 draws requested) at the shipped notebooks'
inductive-shape σ of 0.05: median and 16–84 % bands of the archived draws
and of the in-spec subset (coil drift within ±2 % and l_i within the
engine's ±5 % band), with the reconstruction. Yield at σ = 0.05: g-file 12 archived of 12 attempts, 2 in spec; IDS 11 of 12, 6 in spec (at the default σ = 0.10: g-file 9 of 12, 2 in spec; IDS 7 of 12, 2 in spec; every rejection a capped first homotopy stage).

**`engine_draw_cost.png`** — Synthetic examples, real solver. Grad-Shafranov solves per
archived engine draw by stage (bars) and the post-homotopy passes against
the ceiling of 6 (diamonds), for the σ = 0.05 batches. Median wall time per archived draw 115 s (g-file) and 97 s (IDS); at most 4 post-homotopy passes.

## Pressure, boundary and orientation

**`edge_pressure_betaN_W_solver.png`** — Synthetic example (g-file + p-file),
real solver, plus one constructed case with 2 kPa added to the separatrix
pressure (not a repository example). Full-frame β_N and W_MHD against the
input, and the mid-radius P′ handed to the solver, with the old separatrix
bookkeeping (P′ inflated, separatrix pressure dropped) and the offset one, on
both paths. The offset stays within about 1.3 % of the input at both
separatrix pressures; the old bookkeeping loses 5–5.5 % once the
separatrix pressure is 2 kPa.

**`edge_pressure_separatrix_offset.png`** — Synthetic example (pressure as
handed to the solver) and a toy sweep of the separatrix pressure,
solver-free. Left: the pressure the solver ends up with under the old and the
offset bookkeeping; right: the stored-energy error of each against the
input as the separatrix pressure rises to 10 % of the axis value. Bookkeeping
only; the solved response is `edge_pressure_betaN_W_solver.png`.

**`baseline_gfile_frame_recon_vs_archive.png`** — Synthetic examples, real
solver. The reconstruction's own g-file and the archive's baseline g-file
carry the same full pressure frame (separatrix pressure added back), while a
bare solver save of the same state carries the solver frame (zero at the
edge). On the g-file with the engine: edge pressure 387.4 Pa in both written
files, 4.8 Pa in the bare save; largest difference between the two written
files 0 Pa.

**`boundary_metric_axis_selection.png`** — Toy model, solver-free: the
synthetic flux map of `tests/test_lcfs_axis_selection.py`, a plasma well with
an X-point, divertor legs and a far dip crossing the same flux level. At a
level just inside and just outside the saddle, the old rule (longest closed
segment) picks the far loop and the new rule (the curve around the magnetic
axis) picks the plasma boundary.

**`reversed_ip_identity_both_paths.png`** — Synthetic examples, real solver.
(a) Legacy path: the synthetic IDS example mirrored into the four (Ip, B0) orientations and forward-solved for each of eight closure modes, all 24 mirrored cases bit-identical to the reference orientation. (b) Engine: the synthetic g-file mirrored in its own convention and reconstructed; the toroidal-field mirror is bit-identical and the Ip mirrors differ at the contour-tracing level, at most 3.7e-5 against the stated bar of 1e-4.

## The bootstrap loop kernel

**`loop_kernel_two_state_relaxation.png`** — Toy model, solver-free: the
self-consistent bootstrap kernel on the closed-form two-state
closure↔geometry problem of `tests/test_jbs_loop.py`, started at 0.6 × the
fixed point, without and with the current relaxation (β = 1 and the default
β = 0.7). Both reach the closed-form fixed point to the loop's own
tolerances; β = 0.7 takes 8 passes instead of 13 and changes the path, not
the answer.

## Withheld

The legacy-vs-engine profile comparison on the **synthetic IDS** example is
not included. The shipped IDS example is known to be internally inconsistent
in its parallel-to-toroidal current conversion, so the comparison would
measure the example rather than the code; it returns when the example is
regenerated with consistent currents.

## How the figures are made (`scripts/`)

All scripts run from a clone of the repository; the solver runs need the
fixed OpenFUSIONToolkit build, found through `OFT_PYTHONPATH` (its `python`
directory), and one thread. Large outputs (draw archives) belong on a
machine with room for them; only small JSON is read by the plotting scripts.

* Diagrams: `dot -Tsvg scripts/<name>.dot -o <name>.svg && rsvg-convert -z
  1.8056 -b white <name>.svg -o <name>.png` for `engine_architecture` and
  `engine_review_chapters`.
* Solver-free: `PYTHONPATH=.:tests python
  docs/figures/engine/scripts/make_solver_free_figures.py . <out>` (loop
  kernel, IDS inductive split, edge-pressure bookkeeping; it also writes three
  toy figures not used here) and `make_boundary_fig.py . <out>`.
* Reconstruction, MSE, edge pressure, g-file frame:
  `BQ_REPO=<clone> OUT=<dir> PY=<python> OFT_PYTHONPATH=<build>/python bash
  scripts/run_solver_figure_measurements.sh`, then
  `PYTHONPATH=<clone> python scripts/make_engine_solver_figures.py <clone>
  <OUT>/small <out> <OFT build label>` (it also writes the withheld IDS
  comparison when its data are present).
* Draws, σ = 0 true route, reversed Ip: `BQ_REPO=<clone> OUT=<dir> PY=<python>
  OFT_PYTHONPATH=<build>/python bash scripts/run_draws_sigma0_revip.sh`, then
  `python scripts/extract_draw_data.py <OUT> <OUT>/small` and
  `python scripts/make_draw_figures.py <OUT>/small <out> <bouquet commit>
  <OFT build label>`.
