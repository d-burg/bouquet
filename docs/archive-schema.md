# The bouquet HDF5 archive — schema v2

Authoritative description of the on-disk layout written by bouquet ≥ 1.0.0
(schema v2 first shipped in the 1.0.0 release).
The single source of truth in code is [`bouquet/schema.py`](../bouquet/schema.py)
(`SCHEMA_VERSION`, `PROFILE_UNITS`, fixed dataset names, `write_profile` /
`find_bytes_dataset`); this document mirrors it for human readers. Prefer
reading archives through [`bouquet.BouquetArchive`](../bouquet/archive.py) or
the functional readers (`load_equilibrium`, `load_baseline_profiles`,
`select_indices`, `load_config`) rather than raw `h5py` — see
[workflows.md](workflows.md#reading-an-archive-back) for worked examples.

## Layout

```
{header}.h5                            file attrs: schema_version (=2),
│                                      bouquet_version, created, updated
├── config_json                        JSON dump of the run BouquetConfig
│                                      (root copy = most recent write; the
│                                      per-scan copies below are authoritative)
└── scan/<scan_key>/                   one group per scan point / time slice
    ├── config_json                    this slice's exact config
    │   attrs: [coil_filter]           'chi2' | 'legacy' -- which coil filter wrote
    │          [coil_sigma_model]      passes_coil_filter last (chi2: the sigma model JSON)
    ├── _baseline/                     written once per scan point
    │   ├── eqdsk, [pfile]             raw byte-perfect g-file / p-file
    │   ├── psi_N, psi_N_kinetic
    │   ├── n_e, T_e, n_i, T_i         kinetic profiles
    │   ├── pressure[, pressure_thermal]
    │   ├── j_phi[, j_BS, j_inductive] separated toroidal currents
    │   ├── sigma_ne/te/ni/ti/jphi     the uncertainty envelope used
    │   ├── [aux_<name>, sigma_aux_<name>]   switchboard channels
    │   ├── [recon_lcfs_ref]           10k-pt LCFS reference (boundary metric)
    │   ├── [x_points], [coil_currents, coil_names]
    │   └── attrs: Ip_target, l_i_target, l_i_scale, source_kind, [diverted],
    │              [li_metrics_json, closure_limited]   baseline provenance (1.4+):
    │              Baseline.li_metrics as JSON, incl. the ip_closure health
    │              record on hybrid baselines; load_baseline_profiles() decodes
    │              it to li_metrics / ip_closure / closure_limited
    └── <count>/                       one group per accepted draw
        │                              (integer; gaps = rejected draws)
        ├── eqdsk, [pfile]             raw bytes, fixed names
        ├── psi_N[, psi_N_kinetic]
        ├── j_phi, j_BS, j_inductive[, j_BS,edge]
        ├── n_e, T_e, n_i, T_i, w_ExB[, Zeff]
        ├── [pressure, pressure_thermal]
        ├── [aux_<name>]               perturbed switchboard channels
        ├── [coil_currents, coil_names]
        ├── [perturbed_lcfs_ref], [x_points]
        ├── [eq_fsa/]                  live-equilibrium flux-surface averages
        │   ├── psi_N                  (subgroup; see below)
        │   ├── F, avg_inv_R, avg_inv_R2, avg_B2
        │   └── q, dV_dpsi, f_trap, B_avg
        └── attrs: l_i(1), l_i(3), count, homotopy_*, max_F_drift_pct,
                   max_VSC_drift_pct, in_spec, inspec_*, l_i_target_used,
                   [diverted], [passes_coil_filter, passes_boundary_filter,
                   selected]           ← filter flags, written post-hoc
```

## Conventions

- **Bare dataset names, units in attrs.** Profile datasets carry plain names
  (`j_phi`, `n_e`, …) with the unit string in `ds.attrs["units"]`
  (`PROFILE_UNITS` in `schema.py`). v1 archives embedded units in the name
  (`"j_phi [A m^-2]"`).
- **Fixed byte-blob names.** The g-file / p-file bytes are stored as `eqdsk` /
  `pfile` inside each group — the group path carries the coordinates. Bytes
  are stored opaque (`np.void`) and round-trip bit-perfect.
- **Always `scan/<key>/`.** The scan key is a user-chosen label
  (`GenerationConfig.scan_key`, default `0`) — a time in ms, a beta value, …
  Several bouquets can share one file under different keys.
- **Gap-tolerant indices.** Rejected draws leave gaps; iterate with
  `list_equilibrium_indices` / `BouquetArchive`, never `range(n)`.
- **Filtering is non-destructive.** Filters write boolean attrs
  (`passes_*`, `selected` = AND of applied flags); `export_filtered` produces
  a pruned copy, the source is never modified.
- **Provenance.** `schema_version` / `bouquet_version` / `created` are stamped
  at file creation; `config_json` is added by `write_provenance` (called from
  `Bouquet.generate`, `run_shard`, and `merge_archives`). Recover the exact
  run configuration with `bq.load_config(path, scan_key=...)`.
- **Live-equilibrium FSA (`eq_fsa/`).** Optional per-draw subgroup of
  flux-surface averages captured directly from the live TokaMaker object at
  generate time (`GenerationConfig.capture_live_eq`, on by default), on the
  `psi_N` grid of `capture_npsi` points. Keys and units are `EQ_FSA_GROUP` /
  `EQ_FSA_UNITS` in `schema.py`: `F` (T m), `avg_inv_R` (⟨1/R⟩, m⁻¹),
  `avg_inv_R2` (⟨1/R²⟩, m⁻²), `avg_B2` (⟨B²⟩, T²), `q`, `dV_dpsi`
  (m³ Wb⁻¹), `f_trap`, `B_avg` (⟨B⟩, T). `⟨1/R²⟩` is computed by exact
  FSA quadrature (`capture_exact_inv_R2`, default) with a fast path for
  `sauter_fc`'s native value when present. This is what enables the exact
  parallel↔toroidal current split in the IMAS/OMAS exporter
  (`write_imas_draw(..., fidelity="exact")`); read it back with
  `bq.load_eq_fsa`.

## Legacy (pre-v2) archives

Schema v2 was a clean break (2026-07). Files without the `schema_version`
attr are pre-v2: `BouquetArchive` opens them with a warning (byte blobs still
resolve via a suffix scan; profile keys keep their v1 bracketed names), and
`load_equilibrium` raises a clear error. Regenerate old archives with the
current package for full support.
