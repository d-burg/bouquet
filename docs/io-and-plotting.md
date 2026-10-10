# File I/O and plotting

The reader/writer and plotting layers are independent of TokaMaker — `import
bouquet` works, and everything on this page runs, without an OpenFUSIONToolkit
install.

## Contents

- [GEQDSK reader](#geqdsk-reader)
- [COCOS conversion and Bt/Ip flip](#cocos-conversion-and-btip-flip)
- [P-file reader/writer](#p-file-readerwriter)
- [IDA netCDF reader](#ida-netcdf-reader)
- [IMAS/OMAS reader and writer](#imasomas-reader-and-writer)
- [Plotting](#plotting)

---

## GEQDSK reader

```python
from bouquet import GEQDSKEquilibrium

eq = GEQDSKEquilibrium("g123456.01000", cocos=1)
print(f"Ip = {eq.Ip/1e6:.3f} MA, q95 = {eq.q_profile[-1]:.2f}, li1 = {eq.li['li(1)']:.3f}")
```

| Property / Method | Description |
|-------------------|-------------|
| `psi_N` | Normalised poloidal flux grid (0 → 1) |
| `psi_N_RZ` | 2-D normalised poloidal flux on the (R, Z) grid |
| `psi_axis`, `psi_boundary` | Axis and boundary flux (Wb) |
| `Ip` | Plasma current (A, sign-corrected) |
| `q_profile` | Safety factor on `psi_N` |
| `j_tor_averaged` | `<Jt/R>/<1/R>` — the standard convention (OMFIT, TRANSP) |
| `j_tor_averaged_direct` | Literal `<Jt>` from p′ and FF′ (GS equation) |
| `geometry` | Dict with R, Z, a, κ, δ, squareness per surface |
| `midplane` | Exact outboard-midplane R, Bp, Bt, Btot |
| `rhovn` | Normalised toroidal flux coordinate |
| `li` | Internal inductance dict (`li(1)`, `li(2)`, `li(3)`, …) |
| `betas` | Plasma beta values (beta_t, beta_p, beta_n) |
| `cocos` | Current COCOS convention index |
| `cocosify(out)` | Convert between COCOS conventions (in-place, or `copy=True`) |
| `flip_Bt_Ip()` | Reverse Bt and Ip signs |
| `save(path)` | Write the modified g-file to disk |
| `to_bytes()` / `from_bytes()` | Serialise for HDF5 storage / reconstruct |

All flux-surface geometry, safety factor, current density, and midplane
profiles are computed independently of any external tool. COCOS conventions
1–8 and 11–18 are fully supported. A complete reference of every derived
quantity is in
[`bouquet/io/GEQDSK_QUANTITIES.md`](../bouquet/io/GEQDSK_QUANTITIES.md).

`bouquet.io.write_geqdsk()` writes a raw g-file dict to disk (import from
`bouquet.io`, not the top level).

## COCOS conversion and Bt/Ip flip

```python
from bouquet import GEQDSKEquilibrium, read_geqdsk

eq = read_geqdsk("g123456.01000", cocos=7)
eq.cocosify(1)       # in-place; use copy=True to get a new object
eq.flip_Bt_Ip()      # in-place

eq.save("modified.geqdsk")
raw_bytes = eq.to_bytes()
eq2 = GEQDSKEquilibrium.from_bytes(raw_bytes, cocos=1)
```

Verified field-by-field against OMFIT's `OMFITgeqdsk` — see
[`examples/COCOS_Bt_Ip/omfit_cocos_comparison.ipynb`](../examples/COCOS_Bt_Ip/omfit_cocos_comparison.ipynb)
and [`examples/omfit-comparison/`](../examples/omfit-comparison/).

## P-file reader/writer

Osborne format, 24+ profile types.

```python
from bouquet import PFile, read_pfile

pf = read_pfile("p123456.01000")

ne       = pf.ne                      # profile values (attribute)
psi_grid = pf.psinorm_for("ne")       # its psi_N grid
dne      = pf.derivative_for("ne")    # its stored derivative

pf.set_profile("ne", new_psi, new_ne)

pf.compute_pressure()
pf.compute_diamagnetic_rotations(psi_Wb)
pf.compute_rotation_decomposition(R=R_mid, Bp=Bp_mid, Bt=Bt_mid, psi=psi_Wb)

raw_bytes = pf.to_bytes()
pf2 = PFile.from_bytes(raw_bytes)
```

Supported profiles: `ne`, `te`, `ni`, `ti`, `nb`, `pb`, `ptot`, `nz1`, `omeg`,
`omegp`, `omgvb`, `omgpp`, `omgeb`, `er`, `ommvb`, `ommpp`, `omevb`, `omepp`,
`kpol`, `omghb`, `vtor1`, `vpol1`, and more. Rotation handling includes
diamagnetic rotation, E×B decomposition, and the radial electric field; unit
conventions (bouquet SI vs. p-file units) are in
[architecture.md §14](../architecture.md#14-unit-conventions).

## IDA netCDF reader

`read_ida()` loads IDA `.cdf` kinetic fits as an `IDAProfiles` bundle, handling
both the direct (`*_err`) and ensemble-posterior layouts. The same file supplies
the sigma envelopes when it is used as an uncertainty source
(`UncertaintyConfig.ida_path`; `sigma_mode` and `sigma_method` control the
layout dispatch and ensemble reduction).

`read_ida_cer()` loads impurity CER channels as an `IDACERProfiles` bundle, and
`radial_field_from_impurity_force_balance()` evaluates the impurity radial force balance E_r from
them with propagated uncertainty.

## IMAS/OMAS reader and writer

`read_imas_baseline()` reads a FUSE-style IMAS data-dictionary JSON — separated
currents (`j_ohmic` / `j_bootstrap`), kinetic profiles, fast-ion pressure, and
rotation — and `read_imas_geometry()` pulls the boundary and vacuum `R·B_t` for
a given slice. `write_imas_draw()` / `export_imas_drawset()` go the other way,
reconstructing one or all perturbed IDS from an archive, each holding only
the exported time slice of the template; `fidelity` selects
where the parallel current split's geometry factor comes from (see
[workflows.md](workflows.md#ids-current-split-fidelity)).

The exported slice is the one the reader reads: every IDS is cut at the
`core_profiles` slice nearest the requested time (not at the requested time
itself), keyed on the IMAS structure -- the IDS `time`, time-tagged arrays of
structures (`time_slice`, `profiles_1d`), signals (`data` on their own `time`,
or on the IDS time when homogeneous), `vacuum_toroidal_field.b0`,
`code.output_flag` and the IDS-level `global_quantities`.  Lists of entries
(`core_sources.source`, `pf_active.coil`, `nbi.unit`), coil and limiter
outlines and radial profiles are never cut.  `core_sources` is cut with the
reader's own time rule: each entry keeps its own slices bracketing the slice
time and its first and last, and the windows of that read (the core_sources
slice window, each entry's own and core_profiles windows) are recorded under
the key `bouquet_time_window` (`bouquet.io.imas.IMAS_EXPORT_TIME_WINDOW_KEY`)
inside the schema-legal `code.parameters` string (a JSON object) of
`core_sources` and of each entry; a template's own `code.parameters` keeps
its JSON keys, and any other text it held is kept under
`template_parameters`.  The reader honours a block only when its
times are the slice it reads, so an export re-reads with the same
`source_time_match` record and the same driven currents as the source it
came from; without it the windows of a one-time file would collapse to the
10 µs single-time floor (an entry matched at an offset own time would re-read
as off, an offset `core_sources` base as a refusal).  No non-schema key is
written, so strict IMAS validators accept the export; the reader also honours
the block as a direct key of the node (exports written before 2026-10-09).

The equilibrium is the one IDS a cut keeps more than one slice of.  On a
time-dependent run FUSE converts the `core_profiles` currents on the
PREVIOUS equilibrium slice, and the reader pairs them with whichever of the
slice nearest the `core_profiles` time and the last one before it reproduces
`j_tor` from `j_total`; it reads `ip`, `l_i`, the pressure and the boundary
at the slice nearest the requested time.  A pure cut (`_slice_in_time`)
keeps every one of those slices, so it re-reads with the same currents bit
for bit, and records them and their roles under `bouquet_time_window` in
`equilibrium.code.parameters`; it also records the `core_profiles` times
next to the kept slice (the ida_hybrid time rule's local step).  An exported
DRAW holds one equilibrium slice -- the draw's own, whose geometry its
currents are written on -- and records the template slice the template's
currents were paired with (used by `fidelity="reconstruct"`).

An exported draw writes the thermal species its solve used: electrons, the
hydrogenic main ion, and one effective impurity of charge `Z_imp` at the
main-ion temperature with `n_z = (n_e - z_fast - n_i)/Z_imp` (so the export
is quasineutral, carries the drawn `Z_eff`, and passes the reader's
species-completeness check).  Any other thermal species of the template is
written with zero thermal density (an impurity of another charge is
relabelled to `Z_imp`); the fast population is the template's.  The record
sits under `bouquet_species_model` in `core_profiles.code.parameters`.

Current-convention conversions are in `bouquet.physics`
([current-conventions.md](current-conventions.md)):
`jtor_imas_to_jphi_tokamaker()` / `jphi_tokamaker_to_jtor_imas()`,
`jpar_to_jphi_tokamaker()` / `jphi_tokamaker_to_jpar()`,
`jphi_tokamaker_pressure_term()`. Also `isotropize_fast_pressure()`,
`fast_pressure_residual()`, `infer_fast_pressure()`.

`read_imas_baseline()` also has to decide which **fast-pressure storage
convention** a dd uses: IMAS.jl/FUSE write `pressure_fast_parallel` and
`pressure_fast_perpendicular` per degree of freedom, the IMAS data dictionary
defines them as the full directional pressures, and the scalar `p_fast` that
follows differs by a factor of 3. `p_fast_reduction` defaults to `"auto"`, which
reads the dd's own recorded provenance (`detect_p_fast_convention()`), warns
loudly when that is undeterminable, and records the decision on
`Baseline.p_fast_meta`. See
[workflows.md](workflows.md#fixedcomponentsconfig-bfixed_components).

## Plotting

All plotting functions return `(fig, axes)` and accept a `Bouquet`, a
`BouquetArchive`, a bare header, or a `.h5` path interchangeably.

```python
from bouquet import (
    plot_bouquet,               # full overview (dispatches on stored source kind)
    plot_bouquet_timeseries,    # across scan keys
    plot_traces,                # l_i, Ip, boundary deviation traces
    plot_boundary_point_traces, # per-boundary-point traces (uses stored X-points)
    plot_geqdsk_bouquet,        # pressure, current, q, geometry, li, flux surfaces
    plot_pfile_bouquet,         # densities, temperatures, rotations
    plot_kinetic_profiles, plot_jphi_profiles, plot_jphi,
    plot_aux_profiles,          # switchboard channels
    plot_transport_profiles,
    plot_coil_currents,         # coil-current bar chart
    plot_spec_summary,          # in-spec yield summary
    plot_tokamaker_comparison,  # TokaMaker vs source geqdsk
    plot_input_vs_recon,        # baseline-vs-reconstruction gate figure
    # draw_* variants render onto axes you already have:
    draw_kinetic_profiles, draw_pressure_profiles,
    draw_jphi_total, draw_jphi_components, draw_jphi_profiles,
    draw_flux_function,
    set_plot_style, WONG,       # colorblind-safe palette
)
```

### Selecting scans and draws

Archives are organised by `scan_key` (a user-chosen label per bouquet — a time
in ms, a beta value, …). Passing a `scan_key` that does not exist raises a
`KeyError` listing the available keys.

```python
plot_pfile_bouquet(h5path="run.h5", scan_key=0, x_coord="psi_N")   # all draws + baseline
plot_pfile_bouquet(h5path="run.h5", scan_key=0, count=3)           # one specific draw

from bouquet import discover_scan_keys
discover_scan_keys("run.h5")                    # e.g. ['0', '2000', '2200']

plot_bouquet("run.h5", scan_key=0, selection="selected")   # filter-selected subset only
```

### Styling

In HDF5 mode the **baseline** is plotted in black (background, `zorder=1`) and
**perturbed** equilibria in colour (foreground, `zorder=3`, `alpha=0.65`). In
file-list mode, where there is no baseline/perturbed distinction, all entries
use the `tab10` colormap uniformly. `set_plot_style()` applies the package
defaults; `WONG` is the colorblind-safe categorical palette.

### CLI

`plot-family` (installed as a console script) renders equilibrium families from
the terminal — see [gui-display-guide.md](gui-display-guide.md) for when to use
it versus in-notebook `plot_bouquet()`.
