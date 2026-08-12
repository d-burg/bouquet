"""Live-solver halves of the reproducibility contract and the R2 Ip invariant.

Two things can only be checked with the real GS solver:

  * **The seed contract end to end.** Two seeded ensembles built from the same
    config must produce bitwise-identical archives -- every drawn profile,
    ``j_BS``, ``j_inductive`` and scalar.  ``jBS_scale_range`` and
    ``l_i_uncertainty`` are both switched on so the two streams that WERE
    seeded before this work (``np.random.uniform`` for ``scale_jBS``,
    ``np.random.normal`` for the per-draw ``l_i`` target) are exercised
    alongside the GPR and cannot regress while the GPR is fixed.

    The two ensembles run in **separate processes**.  That is how a user
    actually reproduces a run, and it is the only way to compare the *solved*
    equilibria: ``OFT_env`` is a per-process singleton, and a second
    ``generate_bouquet`` in one interpreter does not start from the same
    solver state -- the run installs a strong coil soft-reg and drift bounds
    that ``replace_eq`` does not undo (they live on the TokaMaker object, not
    the equilibrium).  Measured in-process, back-to-back seeded runs agree
    bitwise on every DRAWN quantity but land coil currents 1.4e-3 apart in
    relative terms and l_i ~0.6% apart.  So the in-process check below pins
    the draw stream (what the seed governs) and the cross-process check pins
    the whole archive (what the user gets).

  * **The R2 golden invariant.** At :math:`\\sigma=0` route R2's inductive
    :math:`I_p` renormalisation must return 1.000, because the archived split
    already IS the answer.  It returned 0.8373 while the root ran on
    ``solve_with_bootstrap``'s landed equilibrium against an uncalibrated
    ``Ip_target``; see ``TokaMaker_interface._AnchorIpRenorm``.

    ``exact`` is the sole accepted measure (the ratio calibration was
    retired 2026-08-17, issue #35 -- see the ``_S_ATOL`` retirement record
    below).  Its acceptance is stated on the Ip-space residual
    ``|s-1|*f_ind`` rather than on ``|s-1|`` -- read ``_S_FIND_ATOL_EXACT``
    before touching the number.  The probe also runs ``fsa`` once as a
    bar-less diagnostic (isolates the affine ``P'`` term) and ``legacy``
    once as the historical control.

Runs on the synthetic D3D-like example (no proprietary data).  Marked
``solver`` like ``test_systematics.py``; deselect with ``pytest -m "not
solver"``.
"""
import json
import os
import subprocess
import sys

# Must precede any `bouquet` import: puts the repo root on sys.path so a probe
# invoked directly (python tests/<this file> ...) still exercises THIS tree.
# The pytest path and the subprocess_env path do not rely on it.
import _harness

_harness.ensure_repo_on_syspath()

import numpy as np
import pytest

import h5py

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.abspath(os.path.join(_HERE, "..", "examples", "D3D-like"))
_GEQ = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EXAMPLE, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EXAMPLE, "DIIID_mesh.h5")

_files_ok = all(os.path.isfile(p) for p in (_GEQ, _PF, _MESH))


def _oft_importable():
    import sys
    for cand in (os.environ.get("OFT_PYTHONPATH"),
                 os.path.join(_HERE, "..", "..", "OpenFUSIONToolkit",
                              "build_release", "python")):
        if cand and os.path.isdir(cand):
            ap = os.path.abspath(cand)
            if ap not in (os.path.abspath(p) for p in sys.path):
                sys.path.append(ap)
    try:
        import OpenFUSIONToolkit  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.solver,
    pytest.mark.skipif(
        not (_files_ok and _oft_importable()),
        reason="needs OFT + the D3D-like mesh/baseline; skipped when "
               "unavailable"),
]

_SEED = 20260804
_N_DRAWS = 2                 # each draw is a full solve + SWB + homotopy

# Acceptance bars for the sigma=0 R2 invariant.  Measured values on this case
# are quoted in the assertions; none of these may be loosened to make a run
# pass -- a miss is a regression to report, not a tolerance to widen.
#
# PROVENANCE OF EVERY "measured" NUMBER IN THIS BLOCK: the stock R2 probe
# (`python tests/test_seeded_reproducibility.py r2 <outdir>`, _SEED below),
# the synthetic golden fixture, production defaults, single-threaded BLAS,
# on main @ 4ad4894.  Quote the mode with the number -- different measures
# do not report the same residual (issue #28).
#
# _S_ATOL RETIREMENT RECORD (was: `_S_ATOL = 1e-3` on ratio-mode |s-1|).
# Retired 2026-08-17 together with ratio mode itself, per explicit
# maintainer approval on issue #35.  Its own escape clause fired: the bar
# was defended as "a numerical-noise floor on an identity", with the caveat
# that if the ratio path ever stopped being exact at sigma=0 it would have
# to be restated.  The seven-archive decomposition (issue #35) showed the
# 8.89e-4 the fixture measured was never a noise floor: the same quantity
# spans 3.33e-5 to 3.22e-2 across real DIII-D archives (factor 965, 4/7
# over the bar), driven entirely -- termA/termB decomposition, closure
# 1e-16 -- by sigma=0 SWB bootstrap non-reproduction read through
# compute_flux_integral, while the calibration's kappa absorbed the
# archived-split amplitude bias (+1.1..+4.1 % of Ip) that `exact` correctly
# reports.  A bar on a measure that is blind to the dominant defect and
# reads the other through a discredited integral protects nothing;
# _S_FIND_ATOL_EXACT below is the invariant now.
#
# QUOTE THIS RESIDUAL WITH ITS ESTIMATOR *AND* ITS TARGET (issue #28).  Both
# have moved under this bar, and neither move was a regression -- but the
# recorded numbers were not repinned either time, which is how a 91%-of-bar
# scare got diagnosed as a drift that never happened.
#
#   * #20 put the probe AND `l_i_target` on l_i(3)/'iter'.  The pre-#20
#     records (+0.100% / +0.130%) are the same physical miss read on
#     l_i(1)/'std' -- a functional ~29% away on this fixture.  Relabelling,
#     not drift.
#   * #27 (issue #25) repointed `l_i_target` from a post-step-7 get_stats
#     read to `result['li_final']`, the step-6 secant-MATCHED value.  That
#     matters here specifically: route R2 with perturb_jind_in_anchor=True
#     takes the ACCEPT-ANCHOR 'C/perturb-anchor' branch and skips
#     find_optimal_scale + the corrective iteration entirely.  So before #27
#     this residual charged the draw for a step-7 drift the R2 path never
#     applies -- measured +0.342% on this golden at main @ 4ad4894
#     (li_final 0.653840 vs li_realized_post_corrective 0.656074).  Removing
#     that systematic is the whole reason the residual reads -0.083% here
#     and read -0.457% at ffc02d1.  Like-for-like now: neither side of the
#     comparison has the corrective iteration applied.
#
# Measured on main @ 4ad4894, golden, l_i(3)/'iter' vs the step-6 matched
# target: exact -0.117% (23% of bar); the retired ratio path read -0.083%.
#
# THE BAR ITSELF IS UNCHANGED AND ITS DERIVATION IS STILL OPEN -- 0.005 was
# set as a round 0.5% against a number that has since been re-denominated
# twice.  See #28.  For scale: `l_i_tolerance` (the per-draw acceptance band
# the generation machinery actually enforces) defaults to 0.05, so this bar
# is a 10x stricter statement than any draw is held to.
_LI_REL = 0.005              # l_i(3)/'iter' vs recon step-6 target ; measured
                             # -0.117% (exact)
_JBS_FRAC = 0.02             # sigma0 j_BS vs baseline split, fraction of peak
                             # (the sigma0-guard bar) ; measured 0.279%

# The 'exact' measure (BOUQUET_R2_IP_MODE=exact, the default and -- since the
# ratio retirement, issue #35 -- the SOLE accepted measure) is not an identity
# at sigma=0: it sees the two ways the archived split fails to be the anchor --
# the reconstruction's own j_phi residual (+0.193% of Ip at the R2 state
# anchor, i.e. -0.25% of inductive amplitude) and the sigma=0 SWB residual
# (-0.085%).  That sensitivity is exactly why it survived #35's decomposition
# and ratio did not.  This bar may not be loosened: a miss is a regression to
# report.
#
# WHAT IS BOUNDED, AND WHY IT IS NOT |s-1| (issue #23; APPROVED CHANGE).
#
# The budget above is quoted in Ip space -- percentages OF Ip.  `s` is not:
# route R2 charges the whole Ip miss to the inductive amplitude alone (j_BS is
# held fixed, by design), so for an affine measure
#
#     s - 1 = -Delta / int(w * j_ind)          Delta = Ip[split] - Ip_demand
#  => |s - 1| * f_ind = |Delta / Ip_demand|,   f_ind = int(w*j_ind)/Ip_demand
#
# exactly (verified to 2.2e-16 on the archived-total probe in the #23
# investigation).  `f_ind` is the inductive share: 0.772 at the baseline-recon
# geometry of this golden (the #23-derived value -- see the two-measurement
# note below), but 0.45-0.95 across a single-machine beta ramp.  So a CONSTANT
# bar on |s-1| is
# a bar whose stringency silently scales with 1/f_ind -- it reads ~2x tighter
# on a high-bootstrap archive than on the case it was calibrated on, and #23
# measured exactly that: archives as self-consistent as this golden
# (residual -0.166% vs -0.162%) tripped 5e-3 purely on the denominator.
#
# So the bound is rebased onto the residual it was derived in:
#
#     3.86e-3 = 5e-3 * 0.772
#             = (the old |s-1| bar) * (the golden's f_ind, as derived in #23)
#
# i.e. the SAME BAR AT THE GOLDEN by construction -- what changes is only that
# it no longer tightens itself on archives at other inductive fractions.
#
# TWO f_ind NUMBERS APPEAR BELOW, AT TWO DIFFERENT MEASUREMENT POINTS.  They
# are not a contradiction; label them and the apparent conflict dissolves:
#
#   0.772  -- PROVENANCE OF THE CONSTANT.  The issue-#23-derived value, taken
#             at the BASELINE-RECON geometry with an independent integrator
#             (#23 quotes 0.772/0.760 there).  This is the number the approved
#             3.86e-3 was built from, and it is frozen as such.
#   0.7976 -- IN-FIXTURE MEASUREMENT.  What f_ind actually measures in THIS
#             fixture at the sigma=0 R2 anchor, in exact mode -- a different
#             geometry and a different integrator from the line above.
#
# The few-% spread between them IS the baseline->anchor step that #23 flags as
# not decomposed; it is named here rather than averaged away.  Carrying the
# approved 3.86e-3 anyway makes this bar 3.2 % TIGHTER at the golden than
# 5e-3 was (3.86e-3 vs 5e-3*0.7976 = 3.99e-3), not looser -- so nothing is
# relaxed by the discrepancy, and the constant stays the reviewed number
# rather than one this branch re-derived for itself.  The 3.86e-3 below is
# unchanged by this labelling.
#
# Known real drift, named rather than absorbed: #23 also measured a genuine
# +0.077 +/- 0.009 %/beta_p rise in this residual (a P'-bookkeeping lead, NOT
# the FF' back-solve of #25 -- opposite sign, 13x smaller slope).  Over the
# beta_p = 0.32 -> 1.65 ramp that is ~+0.10 percentage points of Ip, which sits
# inside this margin (beta-scan measured max 0.267% against the 0.386% bar,
# ~31% spare).  It is inside the bar, not absent from the physics; if that
# slope grows, this is the bound that should catch it.
#
# MUST NOT BE LOOSENED.  It is also not a licence to restate the bound on
# |s-1| again: reporting |s-1| without f_ind is not interpretable across
# operating points, which is the finding that produced this constant.
_S_FIND_ATOL_EXACT = 3.86e-3     # |s-1| * f_ind in exact mode ; golden
                                 # measured 2.42e-3 (|s-1| 3.04e-3,
                                 # f_ind 0.7976) -- 37% margin


def _draw_groups(path):
    """{draw index: {dataset name: ndarray}} plus the scalar attrs."""
    out = {}
    with h5py.File(path, "r") as hf:
        for sk in sorted(hf["scan"].keys()):
            g = hf[f"scan/{sk}"]
            for k in sorted(g.keys()):
                if not str(k).lstrip("-").isdigit():
                    continue
                grp = g[k]
                rec = {n: np.asarray(grp[n][()]) for n in sorted(grp.keys())
                       if isinstance(grp[n], h5py.Dataset)}
                rec["_attrs"] = {a: grp.attrs[a] for a in sorted(grp.attrs)}
                out[f"{sk}/{k}"] = rec
    return out


# ---------------------------------------------------------------------------
#  the seed contract, end to end (two INDEPENDENT processes)
# ---------------------------------------------------------------------------
def _generate_one_ensemble(header):
    """Baseline + one seeded ensemble, in THIS process.

    The subprocess entry point (see ``__main__`` at the bottom) and therefore
    also the definition of "a run" for the twin comparison.

    Driven through ``generate_bouquet`` rather than ``Bouquet.generate()``
    because ``l_i_uncertainty`` -- one of the two streams that WAS seeded
    before this work -- has no ``GenerationConfig`` field, so the class API
    cannot switch it on.  Everything else mirrors what ``Bouquet.generate()``
    passes, in particular the production coil-drift / homotopy schedule: the
    bare ``generate_bouquet`` defaults are a single +/-1 % pass, which this
    case cannot meet under a 5 % j_phi envelope, and every draw is then
    discarded as infeasible.

    Writes ``<header>.h5`` and ``<header>_diag.json`` (the per-draw sampler
    inputs, which are not archived).
    """
    import numpy as np
    import bouquet as bq
    from bouquet import generate_bouquet
    from bouquet.baseline import resolve_uncertainty
    from bouquet.utils import initialize_equilibrium_database, pchip_interp

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=header, n_draws=_N_DRAWS)
    b.setup_solver()
    bl = b.prepare_baseline()
    gc, fc = b.config.generation, b.config.filtering
    b.config.uncertainty.jphi_scalar_sigma = 0.05
    env = resolve_uncertainty(b.config, bl)
    psi_N = np.asarray(bl.psi_N, dtype=float)
    Zeff_eq = np.clip(
        pchip_interp(np.asarray(bl.psi_N_kinetic, dtype=float),
                     np.asarray(bl.Zeff, dtype=float), psi_N), 1.0, None)

    if os.path.exists(header + ".h5"):
        os.remove(header + ".h5")
    initialize_equilibrium_database(header)
    diags = generate_bouquet(
        b.mygs, psi_N, _N_DRAWS, header,
        np.asarray(bl.j_phi, dtype=float),
        bl.ne, bl.te, bl.ni, bl.ti,
        env["sigma_ne"], env["sigma_te"], env["sigma_ni"], env["sigma_ti"],
        env["sigma_jphi"], env["n_ls"], env["t_ls"], env["j_ls"],
        float(bl.Ip_target), float(bl.l_i_target), Zeff_eq,
        input_jinductive=np.asarray(bl.j_inductive, dtype=float),
        l_i_tolerance=gc.l_i_tolerance,
        psi_pad=float(getattr(b.config.source, "psi_pad", 1e-3)),
        constrain_sawteeth=gc.constrain_sawteeth, recalculate_j_BS=True,
        isolate_edge_jBS=gc.isolate_edge_jBS, floor_j_BS=gc.floor_j_BS,
        scan_key=0,
        psi_N_kinetic=np.asarray(bl.psi_N_kinetic, dtype=float),
        # production coil schedule (what Bouquet.generate() passes)
        coil_drift=gc.coil_drift,
        coil_drift_hard_factor=gc.coil_drift_hard_factor,
        homotopy_passes=gc.homotopy_passes,
        inspec_F_max=fc.inspec_F_max, inspec_VSC_max=fc.inspec_VSC_max,
        # both previously-seeded streams ON, alongside the GPR
        jBS_scale_range=(0.97, 1.03), l_i_uncertainty=0.05,
        diagnostic_plots=False, seed=_SEED,
    )
    with open(header + "_diag.json", "w") as fh:
        json.dump([{k: float(d[k]) for k in ("scale_jBS", "l_i_target_used")}
                   for d in diags], fh)


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory):
    """The same seeded ensemble generated twice, in two fresh interpreters."""
    work = tmp_path_factory.mktemp("twin")
    # subprocess_env, not dict(os.environ, ...): the child is launched as a
    # SCRIPT, so its sys.path[0] is <repo>/tests and `import bouquet` would
    # otherwise fall through to an editable install pointing at another
    # checkout -- silently solving against the wrong revision.  See
    # tests/_harness.py for the two incidents this prevents.
    env = _harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    out = []
    for tag in ("a", "b"):
        header = str(work / f"twin_{tag}")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                               "ensemble", header], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"seeded ensemble {tag} failed "
                        f"(rc={proc.returncode}):\n{proc.stderr[-4000:]}")
        with open(header + "_diag.json") as fh:
            out.append((_draw_groups(header + ".h5"), json.load(fh)))
    return out


def test_same_seed_reproduces_every_draw_dataset_bitwise(twin_runs):
    """Same seed -> bitwise-identical archives.  This is the whole contract."""
    (a, _), (b, _) = twin_runs
    assert a, "the seeded run produced no draws"
    assert sorted(a) == sorted(b), "the two runs kept different draws"
    for key in sorted(a):
        for name in sorted(a[key]):
            if name == "_attrs":
                continue
            np.testing.assert_array_equal(
                a[key][name], b[key][name],
                err_msg=f"draw {key} dataset {name!r} is not reproducible")


def test_same_seed_reproduces_the_drawn_kinetics_and_currents(twin_runs):
    """Name the datasets that matter, so a silent rename cannot hollow out the
    test above into a no-op."""
    (a, _), (b, _) = twin_runs
    required = ("n_e", "T_e", "n_i", "T_i", "j_phi", "j_inductive", "j_BS")
    for key in sorted(a):
        missing = sorted(set(required) - set(a[key]))
        assert not missing, f"draw {key}: missing {missing}"
        for name in required:
            np.testing.assert_array_equal(a[key][name], b[key][name],
                                          err_msg=f"{key}/{name}")


def test_same_seed_reproduces_the_previously_seeded_streams(twin_runs):
    """scale_jBS and the per-draw l_i target were the ONLY seeded pieces before
    the GPR was fixed; migrating them onto the Generator must not regress them.
    """
    (a, da), (b, db) = twin_runs
    assert len(da) == len(db) and da, "no per-draw diagnostics returned"
    for i, (x, y) in enumerate(zip(da, db)):
        for k in ("scale_jBS", "l_i_target_used"):
            assert x[k] == y[k], \
                f"draw {i}: {k} is not reproducible ({x[k]} vs {y[k]})"
    for key in sorted(a):
        for attr in ("l_i(1)", "l_i(3)", "Ip", "l_i_target_used"):
            if attr in a[key]["_attrs"]:
                assert a[key]["_attrs"][attr] == b[key]["_attrs"][attr], \
                    f"{key}/{attr} not reproducible"


def test_the_previously_seeded_streams_actually_varied(twin_runs):
    """Guard the guard: if jBS_scale_range/l_i_uncertainty had collapsed to a
    constant, the reproducibility assertions above would be vacuous."""
    (_a, da), _b = twin_runs
    if len(da) < 2:
        pytest.skip("only one draw completed; nothing to compare")
    assert len({d["scale_jBS"] for d in da}) > 1, \
        "scale_jBS is constant across draws -- the uniform draw is not firing"
    assert len({d["l_i_target_used"] for d in da}) > 1, \
        "l_i target is constant across draws -- the normal draw is not firing"


def test_seeded_draws_are_not_all_identical(twin_runs):
    """A run whose sigmas collapsed to zero would reproduce trivially."""
    (a, _), _ = twin_runs
    keys = sorted(a)
    if len(keys) < 2:
        pytest.skip("only one draw survived; nothing to compare")
    assert not np.array_equal(a[keys[0]]["n_e"], a[keys[1]]["n_e"]), \
        "the two draws are identical -- the sampler is not perturbing"


# ---------------------------------------------------------------------------
#  the R2 golden invariant: sigma=0 -> the archived split, s == 1.000
# ---------------------------------------------------------------------------
def _run_r2_probe(outdir):
    """sigma=0 route-R2 draws through ``perturb_kinetic_equilibrium``.

    Runs the default ('exact') measure TWICE -- so the pair also tests that
    the anchor snapshot/restore injects no state drift -- the 'fsa'
    diagnostic once (bar-less; isolates the affine ``P'`` term, issue #35),
    and the 'legacy' path once for the before/after contrast.  ('ratio' was
    retired 2026-08-17, issue #35.)
    Subprocess entry point; results land in ``<outdir>/r2.npz``.
    """
    import numpy as np
    import bouquet as bq
    from bouquet import perturb_kinetic_equilibrium
    from bouquet.utils import pchip_interp

    b = bq.Bouquet.from_geqdsk(_GEQ, profiles=_PF, mesh=_MESH, nthreads=1,
                               header=os.path.join(outdir, "r2"), n_draws=1)
    b.setup_solver()
    bl = b.prepare_baseline()
    gc = b.config.generation
    psi_N = np.asarray(bl.psi_N, dtype=float)
    psi_kin = np.asarray(bl.psi_N_kinetic, dtype=float)
    psi_pad = float(getattr(b.config.source, "psi_pad", 1e-3))
    EC = 1.6022e-19

    def _k2e(a):
        return pchip_interp(psi_kin, np.asarray(a, dtype=float), psi_N)

    pressure = EC * (_k2e(bl.ne) * _k2e(bl.te) + _k2e(bl.ni) * _k2e(bl.ti))
    Zeff_eq = np.clip(_k2e(bl.Zeff), 1.0, None)
    zk, zj = np.zeros_like(psi_kin), np.zeros_like(psi_N)
    snapshot = b.mygs.copy_eq()

    def _once(mode):
        os.environ["BOUQUET_R2_IP_MODE"] = mode
        b.mygs.replace_eq(source_eq=snapshot)
        out = perturb_kinetic_equilibrium(
            b.mygs, psi_N, pressure,
            bl.ne, bl.te, bl.ni, bl.ti, np.asarray(bl.j_phi, dtype=float),
            zk, zk, zk, zk, zj,                     # sigma = 0 everywhere
            0.5, 0.4, 0.25,
            float(bl.Ip_target), float(bl.l_i_target), Zeff_eq, len(psi_N),
            input_jinductive=np.asarray(bl.j_inductive, dtype=float),
            l_i_tolerance=gc.l_i_tolerance, psi_pad=psi_pad,
            constrain_sawteeth=False, recalculate_j_BS=True,
            isolate_edge_jBS=gc.isolate_edge_jBS, floor_j_BS=gc.floor_j_BS,
            scale_jBS=float(getattr(bl, "bs_scale", 1.0)),
            perturb_jind_in_anchor=True, accept_anchor_inband=False,
            psi_N_kinetic=psi_kin, p_thresh=0.05, rng=_SEED,
        )
        d = out[6]
        # f_ind travels WITH s, always: the acceptance is on their product and
        # |s-1| alone is not interpretable across operating points (issue #23).
        # nan, not a guessed 1.0, when the mode has no frozen anchor to read it
        # off ('legacy') -- a wrong denominator would silently rescale the bar.
        _f = d.get("r2_f_ind")
        return (float(d["r2_ip_scale"]),
                float("nan") if _f is None else float(_f),
                # Measure l_i on the SAME estimator bl.l_i_target lives on
                # ('iter' == li(3)); get_stats' default is 'std' == li(1),
                # which is a different functional (~25% apart) and would make
                # the recon-li assertions below compare apples to oranges.
                # Issue #20 -- estimator consistency, not a tolerance change.
                float(b.mygs.get_stats(lcfs_pad=psi_pad,
                                       li_normalization="iter")["l_i"]),
                np.asarray(d["j_BS"], dtype=float),
                np.asarray(d["j_inductive"], dtype=float))

    s_ex1, f_ex1, li_ex1, jbs_ex1, jind_ex1 = _once("exact")
    s_ex2, f_ex2, li_ex2, jbs_ex2, jind_ex2 = _once("exact")
    s_fsa, f_fsa, li_fsa, jbs_fsa, _ = _once("fsa")
    s_leg, f_leg, li_leg, _, _ = _once("legacy")
    np.savez(
        os.path.join(outdir, "r2.npz"),
        s_exact=np.array([s_ex1, s_ex2]), li_exact=np.array([li_ex1, li_ex2]),
        f_ind_exact=np.array([f_ex1, f_ex2]),
        jbs_exact1=jbs_ex1, jbs_exact2=jbs_ex2,
        jind_exact1=jind_ex1, jind_exact2=jind_ex2,
        s_fsa=np.array([s_fsa]), li_fsa=np.array([li_fsa]),
        f_ind_fsa=np.array([f_fsa]), jbs_fsa=jbs_fsa,
        s_legacy=np.array([s_leg]), li_legacy=np.array([li_leg]),
        f_ind_legacy=np.array([f_leg]),
        jbs_baseline=np.asarray(bl.j_BS, dtype=float),
        l_i_target=np.array([float(bl.l_i_target)]),
    )


@pytest.fixture(scope="module")
def sigma0_anchor(tmp_path_factory):
    """The R2 probe, in its own interpreter.

    A subprocess for the same reason the twins are: ``OFT_env`` is a
    per-process singleton, so a module that builds a solver in the pytest
    process makes ``pytest -m solver`` unrunnable alongside
    ``test_systematics.py``.  Keeping every solver call behind a subprocess
    means this module never instantiates one.
    """
    work = tmp_path_factory.mktemp("r2")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "r2", str(work)],
        env=_harness.subprocess_env(OMP_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"R2 probe failed (rc={proc.returncode}):\n"
                    f"{proc.stderr[-4000:]}")
    return np.load(str(work / "r2.npz"))


# (test_sigma0_r2_ip_scale_is_unity, _recovers_the_recon_li and
#  _is_bit_reproducible ran on the retired ratio calibration and are gone
#  with it -- issue #35.  Their exact-mode counterparts below carry every one
#  of those claims on the surviving measure; the pre-fix history they
#  documented -- s = 0.8373 from rooting on the SWB-landed geometry -- lives
#  on in test_legacy_mode_still_shows_the_defect.)


def test_sigma0_r2_reproduces_the_baseline_jbs(sigma0_anchor):
    """The premise of the invariant: at sigma=0 SWB must return the baseline
    bootstrap, to the same bar the sigma0 guard uses.

    NOTE the norm: max deviation as a fraction of peak.  Issue #35 measured
    that this norm can read 0.04 % of peak while the INTEGRATED discrepancy
    reaches 0.88 % of Ip on a real archive (201586) -- whether that is a
    state difference or a norm insensitivity is the open question split out
    of #35.  This bar is kept as-is until that is settled.
    """
    jbs_bl = np.asarray(sigma0_anchor["jbs_baseline"], dtype=float)
    peak = float(np.max(np.abs(jbs_bl)))
    dev = float(np.max(np.abs(
        np.asarray(sigma0_anchor["jbs_exact1"], dtype=float) - jbs_bl)))
    assert dev / peak <= _JBS_FRAC, (
        f"sigma=0 j_BS is {100 * dev / peak:.3f}% of peak from the baseline "
        f"split (bar {100 * _JBS_FRAC:.1f}%)")


def test_sigma0_r2_exact_measure_lands_in_its_own_budget(sigma0_anchor):
    """``BOUQUET_R2_IP_MODE=exact``: the FSA current integral
    (``utils.Ip_fsa_integral``), the sole sigma=0 invariant since the ratio
    retirement (issue #35).

    It does not land AT 1.000 -- 3.04e-3 here -- and that is the expected,
    understood result: the measure charges the draw for the reconstruction's
    own j_phi residual (+0.193% of Ip at the R2 state anchor) on top of the
    sigma=0 SWB residual (-0.085%).  -0.335% of inductive amplitude
    predicted, -0.325% measured.  That sensitivity is the reason this is the
    measure that survived #35: on real archives it correctly reports the
    +1.1..+4.1 % archived-split amplitude bias the retired calibration
    absorbed into its demand.

    The acceptance is on ``|s-1| * f_ind``, the residual IN Ip SPACE that the
    -0.335% budget is quoted in -- see ``_S_FIND_ATOL_EXACT`` for the algebra
    and for why the equivalent |s-1| statement is operating-point dependent
    (issue #23).  At this golden the two are essentially the same bar by
    construction, because 3.86e-3 = 5e-3 * 0.772, where 0.772 is the
    #23-derived f_ind at the baseline-recon geometry.  (This fixture's own
    f_ind, measured at the sigma=0 R2 anchor, is 0.7976 -- the two numbers and
    why they differ are labelled at ``_S_FIND_ATOL_EXACT``.)
    """
    s = float(sigma0_anchor["s_exact"][0])
    f_ind = float(sigma0_anchor["f_ind_exact"][0])
    assert np.isfinite(f_ind) and f_ind > 0.0, (
        f"exact mode reported no usable inductive share (f_ind={f_ind}); the "
        f"bound cannot be evaluated, and |s-1| alone is not a substitute")
    resid = abs(s - 1.0) * f_ind
    assert resid <= _S_FIND_ATOL_EXACT, (
        f"exact-mode sigma=0 Ip-space residual is {resid:.3e} "
        f"(bar {_S_FIND_ATOL_EXACT:.2e}) -- larger than the residual budget "
        f"accounts for.  Scale s={s:.6f}, |s-1|={abs(s - 1.0):.3e}, "
        f"f_ind={f_ind:.4f}")


def test_sigma0_r2_exact_measure_reports_a_plausible_inductive_share(
        sigma0_anchor):
    """Guard the denominator.  ``_S_FIND_ATOL_EXACT`` is a bar on a PRODUCT, so
    an ``f_ind`` that silently collapsed to something tiny would turn the test
    above into a no-op that passes on any archive.

    The window is deliberately wide -- it is a sanity range on a physical
    quantity (the inductive share of Ip), not a second acceptance criterion.
    Anything outside it means the measure, not the equilibrium, has moved.
    """
    f_ind = float(sigma0_anchor["f_ind_exact"][0])
    assert 0.2 <= f_ind <= 1.2, (
        f"exact-mode f_ind = {f_ind:.4f} is outside the physical window for an "
        f"inductive share; the Ip-space bound's denominator is wrong, so its "
        f"pass above means nothing")
    assert (float(sigma0_anchor["f_ind_exact"][1])
            == pytest.approx(f_ind, rel=0, abs=0)), \
        "f_ind is not bit-reproducible between two identical exact-mode calls"


def test_sigma0_r2_exact_measure_still_recovers_the_recon_li(sigma0_anchor):
    """The physics acceptance is unchanged by the change of measure: same 0.5%
    bar as the default path.

    Measured on main @ 4ad4894: **-0.117%** on l_i(3)/'iter' against the
    step-6 matched `l_i_target` (the retired ratio path read -0.083%).  The records
    this docstring used to carry, +0.130% / +0.100%, were the same miss read
    on l_i(1)/'std' before #20 and against the post-step-7 target before #27
    -- see `_LI_REL` for both re-denominations and issue #28.
    """
    target = float(sigma0_anchor["l_i_target"][0])
    got = float(sigma0_anchor["li_exact"][0])
    assert abs(got - target) / target <= _LI_REL, (
        f"exact-mode sigma=0 l_i = {got:.5f} vs recon {target:.5f} "
        f"({100 * (got / target - 1):+.3f}%, bar {100 * _LI_REL:.1f}%)")


def test_sigma0_r2_exact_measure_is_bit_reproducible(sigma0_anchor):
    """The FSA weights are captured off a ``copy_eq`` snapshot; two identical
    calls must agree to the bit, exactly as the default path does."""
    d = sigma0_anchor
    assert float(d["s_exact"][0]) == float(d["s_exact"][1])
    assert float(d["li_exact"][0]) == float(d["li_exact"][1])
    np.testing.assert_array_equal(d["jbs_exact1"], d["jbs_exact2"])
    np.testing.assert_array_equal(d["jind_exact1"], d["jind_exact2"])


def test_sigma0_r2_exact_measure_leaves_the_bootstrap_alone(sigma0_anchor):
    """Changing the measure must not move j_BS: route R2 holds the bootstrap
    fixed and moves only the ohmic drive.  (Compared exact-vs-ratio until the
    ratio retirement, issue #35; exact-vs-fsa makes the same claim -- the two
    surviving measures differ only in the Ip weights, which must not touch
    the bootstrap.)"""
    np.testing.assert_array_equal(
        sigma0_anchor["jbs_exact1"], sigma0_anchor["jbs_fsa"],
        err_msg="the Ip measure changed the sigma=0 bootstrap")


def test_legacy_mode_still_shows_the_defect(sigma0_anchor):
    """Documents what was fixed: BOUQUET_R2_IP_MODE=legacy reproduces the old
    geometry error, so the acceptance above is not vacuous."""
    s_legacy = float(sigma0_anchor["s_legacy"][0])
    # 1e-2 absolute: an order below the measured legacy defect (|s-1| =
    # 1.7e-1 on this fixture) and an order above every live mode.  Was
    # `10 * _S_ATOL` until that constant retired with ratio mode (issue #35);
    # same protective role, now self-contained.
    assert abs(s_legacy - 1.0) > 1e-2, (
        f"legacy mode returned {s_legacy:.6f}, which is already at unity -- "
        f"the fix may no longer be doing anything on this case")


# ---------------------------------------------------------------------------
#  subprocess entry point for the twin ensembles
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Every solver call in this module runs here, in a fresh interpreter:
    #   python tests/test_seeded_reproducibility.py ensemble <header>
    #   python tests/test_seeded_reproducibility.py r2       <outdir>
    # OFT_env is a per-process singleton, so keeping the pytest process free
    # of solvers is what lets this module coexist with test_systematics.py in
    # one `pytest -m solver` run -- and, for the twins, it is also the only
    # honest way to compare two runs' solved equilibria.  pytest never
    # executes this block.
    # Prove we are running the tree we were launched from BEFORE any solver
    # work -- a wrong-tree import otherwise surfaces as a downstream nan or a
    # missing diagnostics key, far from its cause.
    _harness.ensure_repo_on_syspath()
    _harness.assert_bouquet_is_repo_local()
    _oft_importable()
    _WHAT, _ARG = sys.argv[1], sys.argv[2]
    if _WHAT == "ensemble":
        _generate_one_ensemble(_ARG)
    elif _WHAT == "r2":
        _run_r2_probe(_ARG)
    else:
        raise SystemExit(f"unknown subcommand {_WHAT!r}")
