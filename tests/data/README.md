# tests/data

| file | status | used by |
|------|--------|---------|
| `d3dlike.geqdsk`, `d3dlike_reference.npz` | synthetic (the D3D-like example) | g-file / reference tests |
| `stored_configs/` | synthetic stored configs | stored-config compatibility tests |
| `dd_synthetic.json.gz`, `diiid_profs_synthetic.cdf`, `g_synthetic.geqdsk` | **QUARANTINED** (see below) | `test_swb_systematics_solver.py`, `test_phi_imas_solver.py` (both `-m solver`) |

## QUARANTINED: `dd_synthetic.json.gz`, `diiid_profs_synthetic.cdf`, `g_synthetic.geqdsk`

These three files came with PR #64. The owner has flagged them as **possibly
derived from real device data**: the dd's `core_profiles.j_tor` pairs with an
IMAS.jl equilibrium to machine precision, its B0 and Ip vary in time with
unrounded values, the g-file is EFIT-format on an EFIT 65x65 grid, and the
IDA-layout file carries a real-looking 150-node fit. They **must be replaced
before merge** by truly synthetic equivalents that the owner builds
(owner decision D3, 2026-10-09).

Until then they stay here, unedited, so the two solver tests keep running
locally. Importing either test module emits a `QuarantinedFixtureWarning`
(see `_fixture_quarantine.py`), so every run says so.

* What the replacement files must contain, field by field, and what the
  tests assert numerically: [`FIXTURE_SPEC.md`](FIXTURE_SPEC.md).
* A generator template to complete (not run, writes nothing as committed):
  [`make_synthetic_fixtures_TEMPLATE.py`](make_synthetic_fixtures_TEMPLATE.py).
