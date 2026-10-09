"""TEMPLATE (not run; writes nothing as committed): build truly synthetic
replacements for the quarantined IMAS-path fixtures (tests/data/README.md,
owner decision D3).

    dd_synthetic.json.gz        an OMAS/IMAS JSON (equilibrium, core_profiles,
                                core_sources) -- FIXTURE_SPEC.md section 1
    diiid_profs_synthetic.cdf   an IDA-layout fit -- section 2
    g_synthetic.geqdsk          the equilibrium the IDA fit belongs to -- section 3

The owner completes the TODO blocks (profile choices, the solves) and judges
the result; nothing here runs on import, and ``main()`` refuses until the
``TODO`` guards are removed.  It reuses the repo's own synthetic machinery
rather than starting from scratch:

* ``examples/D3D-like/D3Dlike_Hmode_baseline.geqdsk`` (+ its
  ``D3Dlike_Hmode_baseline_RECIPE.md`` / ``generate_baseline.py``): the
  shareable synthetic LCFS, mesh (``DIIID_mesh.h5``) and the TokaMaker
  reconstruction recipe (isoflux + saddle, rounded Ip / |Bt|);
* ``examples/D3D-like/D3Dlike_baseline_omas.json``: the synthetic OMAS
  structure (three equilibrium slices, core_profiles, a beam source) whose
  layout the dd copies;
* ``examples/D3D-like/add_equilibrium_geometry.py``: how the gm* / f /
  dpressure_dpsi / rho_tor_norm of a dd are made consistent with a g-file in
  COCOS 11 and ``j_total`` rebuilt so A6 holds (here the averages come from
  the live solve, ``physics.capture_equilibrium_fsa``, instead);
* ``tests/test_ida_hybrid_pipeline.py::_write_inputs``: an IDA ``.cdf`` written
  from a dd's own kinetics, with the two n_i routes agreeing;
* ``bouquet.physics`` (A5-A7 conversions, ``evaluate_jBS``) and
  ``bouquet.io.imas._jtor_from_jpar`` (FUSE's ``j_tor`` from ``j_total``).

Targets (round numbers): B0 = -2.0 T, R0 = 1.7 m, Ip = 1.20 MA, t = 1.000 s
(the IDA slice at 1000 ms; update ``_TIME`` in test_phi_imas_solver.py and
test_swb_systematics_solver.py together), generic channel names, no device
metadata.

Run (after completing it, on a machine with OFT; solver work, a few minutes):

    python tests/data/make_synthetic_fixtures_TEMPLATE.py OUT_DIR

then review OUT_DIR against FIXTURE_SPEC.md (the script's own self-checks
print every consistency number) before copying the files over the
quarantined ones.
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXAMPLE = os.path.join(REPO, "examples", "D3D-like")
sys.path.insert(0, REPO)

# ---- round-number targets ---------------------------------------------------
B0 = -2.0           # T (signed, COCOS 11 dd; the g-file is written COCOS 1)
R0 = 1.7            # m
IP = 1.20e6         # A
T_SLICES = (0.98, 1.00, 1.02)     # s: equilibrium slices (>= one before T_CP)
T_CP = 1.00         # s: the one core_profiles / core_sources slice
N_CP = 257          # core_profiles grid, UNIFORM IN rho_tor_norm (spec 1)
N_IDA = 150         # IDA nodes, psi_N in [0, 1.2]
Z_IMP = 6.0


def analytic_kinetics(psi_N):
    """Analytic H-mode-like profiles on psi_N (n in m^-3, T in eV).

    TODO(owner): choose the shapes -- e.g. OpenFUSIONToolkit's
    ``bootstrap.Hmode_profiles`` (edge / ped / core / widthp) or polynomial
    pedestal + core.  Return ne, te, ti, nC (carbon) and the beam density
    n_fast on D; Z_eff then follows (n_i = ne - Z nC - n_fast)."""
    raise NotImplementedError("TODO(owner): the analytic kinetic profiles")


def solve_equilibrium(kin, current_shape, label):
    """One TokaMaker free-boundary solve on the D3D-like mesh at the targets.

    TODO(owner): follow examples/D3D-like/generate_baseline.py: mesh
    DIIID_mesh.h5, isoflux on the D3D-like LCFS (re-fit), saddle X-point, Ip =
    IP, F0 = |B0| R0, pressure from ``kin`` (p = e(ne Te + ni Ti + nC Ti) +
    p_fast), FF' from ``current_shape`` (jphi-linterp).  Return the live
    ``mygs``.  Two solves are needed: "ida" (the equilibrium the IDA fit
    belongs to -> g_synthetic.geqdsk) and "dd" (a DIFFERENT current profile:
    the dd's equilibrium slices; the Phi_N test needs the surfaces to move by
    more than 10 mm between the two, FIXTURE_SPEC.md section 4)."""
    raise NotImplementedError("TODO(owner): the TokaMaker solves")


def write_gfile(mygs, path):
    """``mygs.save_eqdsk(path, nr=257, nz=257, lcfs_pad=1e-4)`` (COCOS 7),
    then cocosify to COCOS 1 with BCENTR = B0 (generate_baseline.py step 7)."""
    raise NotImplementedError("TODO(owner): save + COCOS 1")


def equilibrium_slice(mygs, t):
    """The IMAS ``equilibrium.time_slice`` of ``mygs`` (COCOS 11).

    ``physics.capture_equilibrium_fsa(mygs, npsi=257)`` gives F, p', <R>,
    <1/R>, <1/R^2>, <B^2>, q on a psi_N grid; convert: psi_11 = -2 pi psi_TM
    (psi_N from 0 to 1 between psi_axis and psi_bnd), dpressure_dpsi =
    p'_TM / (-2 pi), f signed with B0, gm1/gm5/gm8/gm9 as captured,
    rho_tor_norm from Phi = integral q dpsi, j_tor = A5 image of the solve's
    jphi.  global_quantities: ip, li_3 (mygs.get_stats li_normalization
    "iter"), li_1; boundary.outline from the traced LCFS."""
    from bouquet.physics import capture_equilibrium_fsa  # noqa: F401
    raise NotImplementedError("TODO(owner): the equilibrium slice")


def core_profiles_and_sources(kin_cp, slices, paired):
    """core_profiles (one slice at T_CP) + core_sources.

    On a grid uniform in rho_tor_norm (N_CP points) mapped to psi_N through
    the T_CP slice's rho: electrons / ions (D with density_fast, C12),
    pressure_fast_parallel/perpendicular (FUSE 'sum' convention; stamp
    ``ids_properties.comment = "synthetic test dd; p_fast_reduction=sum"``),
    rotation_frequency_tor.  Currents (<J.B>/B0): j_bootstrap from
    ``physics.evaluate_jBS(mygs, ...)[1]["j_dot_B"] / B0`` on the paired
    slice's equilibrium, one analytic beam j_parallel (index 2,
    "nbi_synthetic"), an analytic j_ohmic >= 0 everywhere (the swb seed must
    be non-negative), j_total = j_ohmic + j_bootstrap + beam, and
    j_tor = ``io.imas._jtor_from_jpar(j_total, geom_of(paired))`` -- the
    PREVIOUS equilibrium slice, as FUSE does, so the reader's pairing
    reproduces j_tor at round-off.  core_sources: the beam, plus "ohmic" (7)
    and "bootstrap" (13) aggregates and a zero-current "sawteeth" (701)."""
    raise NotImplementedError("TODO(owner): core_profiles / core_sources")


def write_ida(path, mygs_ida, kin):
    """IDA 'direct' layout (FIXTURE_SPEC.md section 2), as
    tests/test_ida_hybrid_pipeline.py::_write_inputs writes it: time = [1e3 *
    T_CP] ms, psi_n (N_IDA nodes to 1.2), n_e, T_e, T_12C6, Zeff, n_12C6,
    omega_tor_12C6 with *_err at explicit constant fractions, and q = the
    "ida" equilibrium's q at psi_n (beyond 1: its edge value)."""
    import h5py  # noqa: F401
    raise NotImplementedError("TODO(owner): the IDA file")


def self_checks(out_dir):
    """Print the consistency numbers of FIXTURE_SPEC.md section 1 -- read the
    new dd with ``bouquet.read_imas_baseline(ImasSource(ids_path, time=T_CP,
    ida_path, LCFS_geqdsk), kinetic_source="ida_hybrid")`` with warnings
    recorded: the j_tor pairing mismatch (<= 1e-3; expect ~1e-16), the
    psi_N(rho) drift (<= 2e-2), the IDA n_i check (<= 1e-2), no orientation
    refusal, no ratio fallback (li_metrics["imas_current_conversion"])."""
    raise NotImplementedError("TODO(owner): the self-checks")


def main(out_dir):
    raise SystemExit(
        "make_synthetic_fixtures_TEMPLATE.py is a template: complete the "
        "TODO(owner) blocks (and remove this guard) before running it")
    os.makedirs(out_dir, exist_ok=True)
    psi = np.linspace(0.0, 1.0, 257)
    kin = analytic_kinetics(psi)
    mygs_ida = solve_equilibrium(kin, current_shape="ida", label="ida")
    write_gfile(mygs_ida, os.path.join(out_dir, "g_synthetic.geqdsk"))
    write_ida(os.path.join(out_dir, "diiid_profs_synthetic.cdf"), mygs_ida, kin)
    mygs_dd = solve_equilibrium(kin, current_shape="dd", label="dd")
    slices = [equilibrium_slice(mygs_dd, t) for t in T_SLICES]
    dd = {"equilibrium": {"time": list(T_SLICES),
                          "vacuum_toroidal_field": {"r0": R0,
                                                    "b0": [B0] * len(T_SLICES)},
                          "time_slice": slices}}
    dd.update(core_profiles_and_sources(kin, slices, paired=slices[0]))
    with gzip.open(os.path.join(out_dir, "dd_synthetic.json.gz"), "wt") as fh:
        json.dump(dd, fh)
    self_checks(out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_fixtures_out")
