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
reconstructing one or all perturbed IDS from an archive; `fidelity` selects
where the parallel current split's geometry factor comes from (see
[workflows.md](workflows.md#ids-current-split-fidelity)).

Current-convention conversions are in `bouquet.physics`:
`parallel_to_toroidal()`, `toroidal_to_parallel()` (both FSA-geometry aware),
`isotropize_fast_pressure()`, `fast_pressure_residual()`,
`infer_fast_pressure()`.

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
