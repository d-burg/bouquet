"""bouquet.stats.bands_summary: the error-bar section of every example
notebook. Fast, on the committed golden archive. The golden is regenerated
from time to time, so no count is pinned: the expected counts and values are
read from its manifest (golden_manifest.json) and from the archive itself."""

import json
import os
import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq
from bouquet.stats import UnfilteredArchiveError

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
_MANIFEST = os.path.join(_HERE, "golden", "golden_manifest.json")


def _manifest_scan():
    m = json.load(open(_MANIFEST))
    key = sorted(m["scans"])[0]
    return key, m["scans"][key]


@pytest.fixture(scope="module")
def stamped(tmp_path_factory):
    """A copy of the golden whose selection is the manifest's in-spec verdict
    (the slim golden carries no filter stamp of its own)."""
    if not (os.path.isfile(_GOLDEN) and os.path.isfile(_MANIFEST)):
        pytest.skip("golden fixture absent")
    key, ms = _manifest_scan()
    p = str(tmp_path_factory.mktemp("bs") / "golden_stamped.h5")
    shutil.copy(_GOLDEN, p)
    with h5py.File(p, "a") as hf:
        sg = hf[f"scan/{key}"]
        sg.attrs["coil_filter"] = "legacy"
        for d, rec in ms["draws"].items():
            ok = bool(rec["in_spec"])
            g = sg[str(d)]
            g.attrs["passes_coil_filter"] = ok
            g.attrs["passes_boundary_filter"] = True
            g.attrs["selected"] = ok
    inspec = sorted(int(d) for d, r in ms["draws"].items() if r["in_spec"])
    if len(inspec) < 2:
        pytest.skip("the golden has fewer than two in-spec draws")
    return p, key, ms, inspec


def _q(p, key, d):
    with h5py.File(p, "r") as hf:
        g = hf[f"scan/{key}/{d}/eq_fsa"]
        return np.asarray(g["psi_N"][()]), np.abs(np.asarray(g["q"][()]))


def _outermost(x, y, t):
    dy = y - t
    idx = [i for i in range(len(y) - 1) if dy[i] * dy[i + 1] <= 0
           and not (dy[i] == 0 and dy[i + 1] == 0)]
    if not idx:
        return None
    i = idx[-1]
    w = 0.0 if dy[i] == dy[i + 1] else dy[i] / (dy[i] - dy[i + 1])
    return x[i] + w * (x[i + 1] - x[i])


def _grad_at(p, key, d, ds, psi_s):
    with h5py.File(p, "r") as hf:
        g = hf[f"scan/{key}/{d}"]
        x, y = np.asarray(g["psi_N"][()]), np.asarray(g[ds][()])
    return float(np.interp(psi_s, x, np.gradient(y, x)))


def test_counts_and_scalar_bands_follow_the_manifest(stamped):
    p, key, ms, inspec = stamped
    s = bq.bands_summary(p, key)
    assert s.n_selected == len(inspec) == ms["n_in_spec"]
    assert s.population == "in-spec draws"
    li = s["l_i_3"]
    x = np.array([ms["draws"][str(d)]["l_i(3)"] for d in inspec])
    assert li.n_used == len(inspec)
    assert sorted(li.values) == inspec
    np.testing.assert_allclose(
        [li.median, li.p16, li.p84, li.min, li.max],
        [np.median(x), *np.percentile(x, (16, 84)), x.min(), x.max()], rtol=1e-12)
    for q in ("beta_N", "Lambda"):
        r = s[q]
        assert r.status == "ok" and r.n_used == len(inspec), q
        assert r.min <= r.p16 <= r.median <= r.p84 <= r.max, q
    # the baseline overlay of l_i(3): the recorded TokaMaker value, else the
    # baseline's l_i_target on the estimator its l_i_scale names
    bl = bq.load_baseline_profiles(p, scan_key=key)
    want = (bl.get("li_metrics") or {}).get("tokamaker_li_3")
    if want is None and str(bl.get("l_i_scale", "")).startswith("iter"):
        want = bl.get("l_i_target")
    if want is not None:
        assert li.baseline_value == pytest.approx(float(want), rel=1e-12)


def test_gradients_at_each_draws_own_q2_surface(stamped):
    p, key, ms, inspec = stamped
    s = bq.bands_summary(p, key, at={"q": 2.0})
    own = {d: _outermost(*_q(p, key, d), 2.0) for d in inspec}
    has = {d: x for d, x in own.items() if x is not None}
    if not has:
        pytest.skip("no in-spec golden draw has a q = 2 surface")
    assert s.surface["mode"] == "own_surface"
    assert s.surface["n_located"] == len(has)
    assert s.surface["n_no_surface"] == len(inspec) - len(has)
    for d, x in has.items():
        assert s.surface["psi_N_by_draw"][d] == pytest.approx(x, rel=1e-12)
        psi, q = _q(p, key, d)
        assert np.interp(x, psi, q) == pytest.approx(2.0, rel=1e-6)
    for name, ds in (("dp_dpsi", "pressure"), ("djphi_dpsi", "jphi_total")):
        r = s[name]
        assert r.n_used == len(has)
        for d, x in has.items():
            assert r.values[d] == pytest.approx(_grad_at(p, key, d, ds, x), rel=1e-10)
        assert r.provenance["surface"]["mode"] == "own_surface"
    assert s["dp_dpsi"].median < 0            # pressure falls outward at q = 2
    txt = s.table()
    assert f"n = {len(inspec)}" in txt and "own q = 2 surface" in txt


def test_draws_without_the_surface_are_counted_and_excluded(stamped):
    p, key, ms, inspec = stamped
    # a target some in-spec draws reach and some do not, read off the archive
    q0 = sorted(float(_q(p, key, d)[1].min()) for d in inspec)
    cut = [0.5 * (a + b) for a, b in zip(q0, q0[1:]) if b > a]
    if not cut:
        pytest.skip("every in-spec golden draw has the same q0")
    t = cut[len(cut) // 2]
    s = bq.bands_summary(p, key, at={"q": t})
    miss = [d for d in inspec if _outermost(*_q(p, key, d), t) is None]
    assert 0 < len(miss) < len(inspec)
    assert s.surface["no_surface_draws"] == sorted(miss)
    r = s["dp_dpsi"]
    assert r.n_irregular == len(miss)
    assert r.n_used == len(inspec) - len(miss)
    assert sorted(d for d, why in r.dropped if why.startswith("irregular:")) == sorted(miss)
    assert all(why == f"irregular:no_q={t:g}_surface"
               for d, why in r.dropped if d in miss)
    assert any("excluded from the dp_dpsi band" in l for l in r.provenance["limitations"])
    # the scalars are not touched by the surface
    assert s["l_i_3"].n_used == len(inspec)


def test_falls_back_to_the_named_psi_n_and_says_so(stamped):
    p, key, ms, inspec = stamped
    s = bq.bands_summary(p, key, at={"q": 1.0e3}, fallback_psi_N=0.5)
    fixed = bq.bands_summary(p, key, at={"psi_N": 0.5})
    assert s.surface["mode"] == "fallback_psi_N"
    assert s.surface["n_located"] == 0 and s.surface["n_no_surface"] == len(inspec)
    for g in ("dp_dpsi", "djphi_dpsi"):
        assert s[g].values == fixed[g].values
        assert s[g].n_used == len(inspec)
        assert any("FALLBACK psi_N = 0.5" in l for l in s[g].provenance["limitations"])
    assert "FALLBACK" in s.table() and "FALLBACK" in s.where
    # without a fallback the gradients come back empty, never silently moved
    n = bq.bands_summary(p, key, at={"q": 1.0e3}, fallback_psi_N=None)
    assert n["dp_dpsi"].status == "empty"
    assert n["dp_dpsi"].empty_reason == "no_regular_draws"


def test_the_fallback_is_decided_on_the_bands_own_population(stamped, tmp_path):
    """rescued="exclude": when the only in-spec draws with the surface are
    rescued ones, the band (which drops them) has no draw on the surface, so
    the mode is the fallback and n_located counts only the band's draws."""
    p0, key, ms, inspec = stamped
    q0 = sorted(float(_q(p0, key, d)[1].min()) for d in inspec)
    cut = [0.5 * (a + b) for a, b in zip(q0, q0[1:]) if b > a]
    if not cut:
        pytest.skip("every in-spec golden draw has the same q0")
    t = cut[len(cut) // 2]
    reach = [d for d in inspec if _outermost(*_q(p0, key, d), t) is not None]
    assert 0 < len(reach) < len(inspec)
    p = str(tmp_path / "rescued.h5")
    shutil.copy(p0, p)
    with h5py.File(p, "a") as hf:
        for d in reach:
            hf[f"scan/{key}/{d}"].attrs["solve_recovered"] = True
            hf[f"scan/{key}/{d}"].attrs["solve_recovered_by"] = "test"
    inc = bq.bands_summary(p, key, at={"q": t}, fallback_psi_N=0.5)
    assert inc.surface["mode"] == "own_surface"
    assert inc.surface["n_located"] == len(reach)
    exc = bq.bands_summary(p, key, at={"q": t}, fallback_psi_N=0.5,
                           rescued="exclude")
    assert exc.surface["mode"] == "fallback_psi_N"
    assert exc.surface["n_located"] == 0
    assert exc.surface["n_considered"] == len(inspec) - len(reach)
    assert exc["dp_dpsi"].n_used == len(inspec) - len(reach)


def test_phi_n_profiles_are_mapped_through_the_draws_own_q(stamped, tmp_path):
    """A Phi_N-coordinate archive: d/dpsi_N comes back through the draw's own
    q map. Analytic profiles of psi_N, written on a Phi_N-uniform grid, must
    give their analytic psi_N derivative at psi_N = 0.5."""
    p, key, ms, inspec = stamped
    P = lambda x: 6.0e4 * (1.0 - x ** 2) ** 2                 # Pa
    dP = lambda x: -2.4e5 * x * (1.0 - x ** 2)
    J = lambda x: 1.5e6 * (1.0 - x ** 1.5) ** 1.2              # A m^-2
    dJ = lambda x: -2.7e6 * x ** 0.5 * (1.0 - x ** 1.5) ** 0.2
    q2 = str(tmp_path / "phi.h5")
    shutil.copy(p, q2)
    from scipy.integrate import cumulative_trapezoid
    with h5py.File(q2, "a") as hf:
        for d in inspec:
            g = hf[f"scan/{key}/{d}"]
            psi_q, q = _q(q2, key, d)
            xe = np.unique(np.concatenate([[0.0], psi_q[(psi_q > 0) & (psi_q < 1)], [1.0]]))
            phi = cumulative_trapezoid(np.interp(xe, psi_q, q), xe, initial=0.0)
            phi /= phi[-1]
            phi_grid = np.linspace(0.0, 1.0, 513)           # a Phi_N-uniform run grid
            psi_at = np.interp(phi_grid, phi, xe)            # independent of the helper
            for ds, f in (("pressure", P), ("jphi_total", J)):
                del g[ds]
                g[ds] = f(psi_at)
            del g["psi_N"]
            g["psi_N"] = phi_grid
            g.attrs["profile_coord"] = "phi_n"
    s = bq.bands_summary(q2, key, scalars=(), at={"psi_N": 0.5})
    for gname, exact in (("dp_dpsi", dP(0.5)), ("djphi_dpsi", dJ(0.5))):
        assert s[gname].n_used == len(inspec)
        for d in inspec:
            # measured <= 1.3e-5 (central differences on the 513-node grid)
            assert s[gname].values[d] == pytest.approx(exact, rel=1e-4), (gname, d)


def test_refuses_unknown_quantities_and_an_unfiltered_archive(stamped):
    p, key, ms, inspec = stamped
    with pytest.raises(ValueError, match="unknown quantities"):
        bq.bands_summary(p, key, scalars=("beta_X",))
    with pytest.raises(ValueError, match="at must be"):
        bq.bands_summary(p, key, at={"rho": 0.5})
    with pytest.raises(UnfilteredArchiveError):
        bq.bands_summary(_GOLDEN, key)
    u = bq.bands_summary(_GOLDEN, key, require_filter=False)
    assert u.population.startswith("stored draws of an UNFILTERED")
    assert "UNFILTERED" in u.table()


def test_one_band_figure_states_n(stamped):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, key, ms, inspec = stamped
    s = bq.bands_summary(p, key)
    ax = s.plot()
    assert f"n = {len(inspec)}" in ax.get_title()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == len(s.records)
    assert all(f"(n={s[q].n_used})" in lab for q, lab in zip(s.records, labels))
    plt.close(ax.figure)
