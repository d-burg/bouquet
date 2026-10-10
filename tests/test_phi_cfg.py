"""phi_n config checks, input coordinates, and the stubbed reconstruction
baseline (pure: no OpenFUSIONToolkit solve)."""
import os
import sys
import types

import numpy as np
import pytest

from bouquet import coords
from bouquet.config import (BouquetConfig, FixedComponentsConfig,
                            GenerationConfig, ImasSource, ReconstructionSource,
                            SolverConfig)

HERE = os.path.dirname(__file__)
GFILE = os.path.join(HERE, "data", "d3dlike.geqdsk")
PFILE = os.path.join(HERE, "..", "examples", "D3D-like",
                     "D3Dlike_Hmode_baseline.peqdsk")


def _legacy(**kw):
    return GenerationConfig(reconstruction_engine="legacy", **kw)


def _cfg(source=None, **kw):
    # the legacy frozen-SWB path (the stubbed reconstruction runs no loop)
    kw.setdefault("generation", _legacy(jbs_self_consistent=False))
    return BouquetConfig(
        source=source or ReconstructionSource(geqdsk_path=GFILE,
                                              profiles_path="p.cdf",
                                              coord="phi_n"),
        solver=SolverConfig(mesh_path="unused"), output_header="t", **kw)


# ---------------------------------------------------------------------------
# config values, reserved keys, JSON round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["coord", "x", "psi_N"])
def test_reserved_grid_keys(key):
    with pytest.raises(ValueError, match="passed explicitly"):
        GenerationConfig(bootstrap_kwargs={key: 1})


@pytest.mark.parametrize("make", [
    lambda: ReconstructionSource(geqdsk_path="g", profiles_path="p", coord="psi"),
    lambda: ImasSource(ids_path="d", coord="rho"),
    lambda: FixedComponentsConfig(coord="phi_n"),
])
def test_bad_coord_values_raise_at_construction(make):
    with pytest.raises(ValueError):
        make()


def test_rho_tor_is_accepted():
    assert ImasSource(ids_path="d", coord="rho_tor").coord == "rho_tor"


def test_phi_run_refuses_the_python_solve_at_construction(monkeypatch):
    # a toolkit WITH the Python solve option (the internal-solve build): the
    # refusal under test is the Phi_N one, not the toolkit-capability one
    from _swb import FORK_BOOTSTRAP_NAMES
    import bouquet.config as _config
    monkeypatch.setattr(_config, "_bootstrap_kwarg_names",
                        lambda: FORK_BOOTSTRAP_NAMES)
    with pytest.raises(ValueError, match="use_python_solve"):
        _cfg(generation=_legacy(bootstrap_kwargs={"use_python_solve": True}))
    _cfg(source=ImasSource(ids_path="d"), generation=_legacy(
        bootstrap_kwargs={"use_python_solve": True}))       # psi_n: allowed


def test_json_round_trip_keeps_the_coords():
    fc = FixedComponentsConfig(j_NBI=np.ones(3), psi_N=np.linspace(0, 1, 3),
                               coord="psi_n")
    cfg = _cfg(generation=_legacy(jbs_self_consistent=False),
               fixed_components=fc)
    for back in (BouquetConfig.from_json(cfg.to_json()),
                 BouquetConfig.from_dict(cfg.to_dict())):
        assert back.source.coord == "phi_n"
        assert back.fixed_components.coord == "psi_n"
        np.testing.assert_array_equal(back.fixed_components.psi_N, fc.psi_N)


# ---------------------------------------------------------------------------
# check_backend / check_run against a fake toolkit
# ---------------------------------------------------------------------------
def _fake_oft(monkeypatch, sb_coord=True, torflux=True, swb=("x", "coord")):
    # its bootstrap options are the internal-solve toolkit's
    from _swb import FORK_BOOTSTRAP_NAMES
    import bouquet.config as _config
    monkeypatch.setattr(_config, "_bootstrap_kwarg_names",
                        lambda: FORK_BOOTSTRAP_NAMES)
    ns = {}
    exec("def solve_bootstrap(self, %s): pass" % ("coord=None" if sb_coord else "x=None"), ns)
    TM = type("TokaMaker", (), {"solve_bootstrap": ns["solve_bootstrap"]})
    if torflux:
        TM.get_torflux_map = lambda self, x, inverse=False: (x, x)
    exec("def solve_with_bootstrap(mygs, %s): pass"
         % ", ".join(f"{a}=None" for a in ("ne",) + tuple(swb)), ns)
    root = types.ModuleType("OpenFUSIONToolkit")
    tm = types.ModuleType("OpenFUSIONToolkit.TokaMaker")
    core = types.ModuleType("OpenFUSIONToolkit.TokaMaker._core")
    boot = types.ModuleType("OpenFUSIONToolkit.TokaMaker.bootstrap")
    core.TokaMaker = TM
    boot.solve_with_bootstrap = ns["solve_with_bootstrap"]
    root.TokaMaker, tm._core, tm.bootstrap = tm, core, boot
    for m in [k for k in sys.modules if k.split(".")[0] == "OpenFUSIONToolkit"]:
        monkeypatch.delitem(sys.modules, m)
    for m in (root, tm, core, boot):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    monkeypatch.delattr(coords, "_SWB_PARAMS", raising=False)


def test_check_backend_with_a_full_toolkit(monkeypatch):
    _fake_oft(monkeypatch)
    coords.check_backend(coords.PHI)


@pytest.mark.parametrize("missing", [
    dict(sb_coord=False), dict(torflux=False), dict(swb=("x",)),
    dict(swb=("coord",))])
def test_check_backend_refuses_a_missing_piece(monkeypatch, missing):
    _fake_oft(monkeypatch, **missing)
    with pytest.raises(RuntimeError, match="toroidal-flux"):
        coords.check_backend(coords.PHI)


def test_check_run_rechecks_a_mutated_config(monkeypatch):
    _fake_oft(monkeypatch)
    cfg = _cfg()
    assert coords.check_run(cfg) == coords.PHI
    assert coords.check_run(cfg, coords.PSI) == coords.PSI
    cfg.generation.bootstrap_kwargs = {"use_python_solve": True}
    with pytest.raises(ValueError, match="use_python_solve"):
        coords.check_run(cfg)


def test_check_run_accepts_phi_n_with_the_loop_and_the_engine(monkeypatch):
    """The self-consistent loop and the unified engine run on Phi_N (each
    solve's nodes mapped to psi_N on its own toroidal-flux map)."""
    _fake_oft(monkeypatch)
    cfg = _cfg()
    cfg.generation.jbs_self_consistent = True
    assert coords.check_run(cfg) == coords.PHI
    assert coords.check_run(cfg, coords.PSI) == coords.PSI
    cfg.generation.reconstruction_engine = "unified"
    assert coords.check_run(cfg) == coords.PHI


def test_to_run_grid():
    g = np.linspace(0, 1, 5)
    pm = (np.linspace(0, 1, 11), np.linspace(0, 1, 11) ** 1.25)
    assert coords.to_run_grid(g, "run", pm) is g
    assert coords.to_run_grid(g, "psi_n", None) is g
    assert coords.to_run_grid(None, "psi_n", pm) is None
    np.testing.assert_allclose(coords.to_run_grid(g, "psi_n", pm),
                               np.interp(g, *pm))
    with pytest.raises(ValueError):
        coords.to_run_grid(g, "phi_n", pm)


# ---------------------------------------------------------------------------
# stubbed reconstruction baseline
# ---------------------------------------------------------------------------
PSI_IDA = np.linspace(0.0, 1.1, 45)        # extends past the LCFS


def _cdf(path, q=None, ensemble_q=None, sig_te=None):
    h5py = pytest.importorskip("h5py")
    psi = PSI_IDA
    ne = 5.0e19 * (1.2 - 0.7 * psi ** 2)
    nc = 0.015 * ne
    zeff = 1.0 + 30.0 * nc / ne
    te = 3.0e3 * (1.2 - 0.9 * psi ** 2)
    with h5py.File(path, "w") as f:
        f["time"] = np.array([1000.0])
        f["psi_n"] = psi
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", 0.9 * te),
                     ("Zeff", zeff), ("n_12C6", nc)]:
            f[k] = np.asarray(v)[None, :]
        for k, v in [("n_e_err", 0.05 * ne),
                     ("T_e_err", 0.04 * te if sig_te is None else sig_te),
                     ("T_12C6_err", 0.06 * te), ("Zeff_err", 0.03 * zeff),
                     ("n_12C6_err", 0.10 * nc)]:
            f[k] = np.asarray(v)[None, :]
        if q is not None:
            f["q"] = np.asarray(q)[None, :]
    return path


class _GS:
    def set_isoflux(self, pts, weights=None):
        self.iso = pts


@pytest.fixture
def recon(monkeypatch):
    import bouquet.TokaMaker_interface as tmi
    import bouquet.baseline as bmod
    monkeypatch.setattr(coords, "_SWB_PARAMS", frozenset({"x", "coord"}),
                        raising=False)
    monkeypatch.setattr(bmod, "_reconstruction_metrics", lambda *a, **k: {})

    def run(profiles, coord="phi_n", **cfgkw):
        seen = {}

        def fake(mygs, eqdsk, ne, te, ni, ti, Zeff, iso_pts, iso_w, psi_pad, **kw):
            seen.update(kw, ne=ne, eqdsk=eqdsk)
            n = len(eqdsk.psi_N)
            return dict(j_phi_fit=np.linspace(2.0e6, 2.0e5, n),
                        j_BS_used=np.full(n, 1.0e5), li_final=0.9,
                        li_realized_post_corrective=0.9)
        monkeypatch.setattr(tmi, "reconstruct_equilibrium", fake)
        cfg = _cfg(source=ReconstructionSource(
            geqdsk_path=GFILE, profiles_path=profiles, time=1.0, coord=coord),
            **cfgkw)
        return bmod._resolve_reconstruction(cfg.source, cfg, _GS()), seen, cfg
    return run


Q_SRC = 1.0 + 2.0 * PSI_IDA ** 2


def test_recon_phi_grids(recon, tmp_path):
    cdf = _cdf(str(tmp_path / "ida.cdf"), q=Q_SRC)
    bl, seen, _ = recon(cdf)
    eq = seen["eqdsk"]
    np.testing.assert_array_equal(bl.psi_N, np.asarray(eq.rhovn) ** 2)
    assert seen["coord"] == "phi_n" and seen["x"] is bl.psi_N
    inside, phi = coords.phi_n_from_q(PSI_IDA, Q_SRC)
    assert bl.psi_N_kinetic.size == inside.sum() < PSI_IDA.size
    np.testing.assert_array_equal(bl.psi_N_kinetic, phi)
    np.testing.assert_array_equal(bl.psi_map[0], PSI_IDA[inside])
    np.testing.assert_array_equal(bl.psi_map[1], phi)
    assert bl.ne.shape == phi.shape and bl.coord == "phi_n"
    # the seed is the psi_N shape at the g-file nodes
    psi = np.asarray(eq.psi_N, dtype=float)
    np.testing.assert_allclose(seen["guess_jinductive"], (1 - psi ** 1.5) ** 1.5)


def test_recon_psi_run_is_untouched(recon, tmp_path):
    cdf = _cdf(str(tmp_path / "ida.cdf"), q=Q_SRC)
    bl, seen, _ = recon(cdf, coord="psi_n")
    np.testing.assert_array_equal(bl.psi_N, seen["eqdsk"].psi_N)
    np.testing.assert_array_equal(bl.psi_N_kinetic, PSI_IDA)
    assert bl.psi_map is None and "x" not in seen and "coord" not in seen


def test_recon_ida_without_q_raises(recon, tmp_path):
    with pytest.raises(ValueError, match="no q"):
        recon(_cdf(str(tmp_path / "ida.cdf")))


def test_recon_pfile_uses_the_gfile_map(recon):
    from bouquet.baseline import _load_kinetic_profiles
    if not os.path.exists(PFILE):
        pytest.skip("example p-file not present")
    bl, seen, cfg = recon(PFILE)
    psi_pf = _load_kinetic_profiles(cfg.source)["psi_N"]
    inside = psi_pf <= 1.0
    np.testing.assert_allclose(
        bl.psi_N_kinetic,
        np.interp(psi_pf[inside], seen["eqdsk"].psi_N, bl.psi_N))


@pytest.mark.parametrize("fc_coord", ["run", "psi_n"])
def test_recon_fixed_components_on_phi(recon, tmp_path, fc_coord):
    cdf = _cdf(str(tmp_path / "ida.cdf"), q=Q_SRC)
    g = np.linspace(0.0, 1.0, 21)
    f = 1.0e5 * (1.0 - g)
    fc = FixedComponentsConfig(j_NBI=f, p_fast=1.0e3 * (1.0 - g), psi_N=g,
                               coord=fc_coord)
    bl, seen, _ = recon(cdf, fixed_components=fc)
    gx = g if fc_coord == "run" else np.interp(
        g, seen["eqdsk"].psi_N, bl.psi_N)
    np.testing.assert_allclose(bl.j_NBI, np.interp(bl.psi_N, gx, f))
    np.testing.assert_allclose(bl.p_fast,
                               np.interp(bl.psi_N_kinetic, gx, 1.0e3 * (1.0 - g)))
    if fc_coord == "psi_n":     # not the same as reading g as Phi_N
        assert np.max(np.abs(bl.j_NBI - np.interp(bl.psi_N, g, f))) > 1e2


# ---------------------------------------------------------------------------
# an envelope file other than the source is placed by its own q
# ---------------------------------------------------------------------------
def test_other_ida_envelope_by_its_own_q(recon, tmp_path):
    from bouquet.baseline import resolve_uncertainty
    src = _cdf(str(tmp_path / "src.cdf"), q=Q_SRC)
    # q = 1 + 3 psi^2  ->  Phi_N = (psi + psi^3) / 2; sigma_te = 100 Phi_N
    phi_exact = (PSI_IDA + PSI_IDA ** 3) / 2.0
    other = _cdf(str(tmp_path / "other.cdf"), q=1.0 + 3.0 * PSI_IDA ** 2,
                 sig_te=100.0 * phi_exact)
    bl, _, cfg = recon(src)
    cfg.uncertainty.ida_path = other
    cfg.uncertainty.log_sigma_sources = False
    env = resolve_uncertainty(cfg, bl)
    np.testing.assert_allclose(env["sigma_te"], 100.0 * bl.psi_N_kinetic, atol=0.5)
    # the source's own map would misplace it
    moved = np.interp(bl.psi_N_kinetic, bl.psi_map[1],
                      100.0 * phi_exact[PSI_IDA <= 1.0])
    assert np.max(np.abs(moved - 100.0 * bl.psi_N_kinetic)) > 2.0


def _multi_cdf(path, times_ms):
    """A multi-slice copy of _cdf's file (the same profiles at each time)."""
    import h5py
    one = _cdf(path + ".one", q=1.0 + 3.0 * PSI_IDA ** 2)
    with h5py.File(one, "r") as f1, h5py.File(path, "w") as f:
        f["time"] = np.asarray(times_ms, dtype=float)
        for k in f1:
            if k == "time":
                continue
            v = np.asarray(f1[k][()])
            f[k] = (np.repeat(v, len(times_ms), axis=0) if v.ndim == 2
                    else v)
    return path


def test_a_separate_sigma_file_is_read_by_the_ida_time_rule(recon,
                                                            tmp_path):
    """#73 B2/B6 residual (D.md item): UncertaintyConfig.ida_path, a file
    other than the source's, is read at its nearest slice to the source's
    time WITHIN half its local time-step (never interpolated), the match
    recorded on the envelope and in the baseline record; outside the window
    it is refused."""
    from bouquet.baseline import resolve_uncertainty
    src = _cdf(str(tmp_path / "src.cdf"), q=Q_SRC)
    other = _multi_cdf(str(tmp_path / "other.cdf"), [990.0, 1002.0, 1010.0])
    bl, _, cfg = recon(src)
    cfg.uncertainty.ida_path = other
    cfg.uncertainty.log_sigma_sources = False
    env = resolve_uncertainty(cfg, bl)      # 1.000 s -> 1.002 s (|dt| 2 ms)
    m = env["ida_sigma_time_match"]
    assert m["ida_time_used"] == pytest.approx(1.002)
    # the window: half the local step on the requested time's side (12 ms)
    assert m["dt"] == pytest.approx(0.002) and m["half_window"] == \
        pytest.approx(0.006)
    assert bl.li_metrics["ida_sigma_time_match"] == m
    far = _multi_cdf(str(tmp_path / "far.cdf"), [1020.0, 1030.0, 1040.0])
    cfg.uncertainty.ida_path = far          # nearest 1.020 s, window 5 ms
    with pytest.raises(ValueError, match="no IDA slice within"):
        resolve_uncertainty(cfg, bl)


def test_the_sources_own_file_is_read_at_its_kinetics_slice(recon,
                                                           tmp_path):
    """... while the source's own kinetics file keeps the slice its
    kinetics were read at (no separate match recorded)."""
    from bouquet.baseline import resolve_uncertainty
    src = _cdf(str(tmp_path / "src.cdf"), q=Q_SRC)
    bl, _, cfg = recon(src)
    cfg.uncertainty.log_sigma_sources = False
    assert "ida_sigma_time_match" not in resolve_uncertainty(cfg, bl)


def test_source_ida_envelope_keeps_the_source_map(recon, tmp_path):
    from bouquet.baseline import resolve_uncertainty
    src = _cdf(str(tmp_path / "src.cdf"), q=Q_SRC)
    bl, _, cfg = recon(src)
    cfg.uncertainty.log_sigma_sources = False
    env = resolve_uncertainty(cfg, bl)
    np.testing.assert_allclose(env["sigma_te"], 0.04 * bl.te, rtol=1e-12)


def test_ensemble_q_is_reduced_over_samples(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bouquet.io.ida import read_ida
    ns, psi = 5, np.linspace(0.0, 1.0, 17)
    rng = np.random.default_rng(0)
    q = (1.0 + 2.0 * psi ** 2)[None, :] * (1.0 + 0.1 * rng.random((ns, 1)))
    ne = 5.0e19 * (1.2 - 0.7 * psi ** 2) * np.ones((ns, 1))
    te = 3.0e3 * (1.2 - 0.9 * psi ** 2) * np.ones((ns, 1))
    p = str(tmp_path / "ens.cdf")
    with h5py.File(p, "w") as f:
        f["time"] = np.array([1000.0])
        f["psi_n"] = np.broadcast_to(psi, (ns, psi.size))[None]
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", 0.9 * te),
                     ("Zeff", 1.5 + 0 * ne), ("q", q)]:
            f[k] = np.asarray(v)[None]
    np.testing.assert_allclose(read_ida(p, ni_source="Zeff").q, q.mean(axis=0))
    np.testing.assert_allclose(
        read_ida(p, ni_source="Zeff", ensemble_median=True).q,
        np.median(q, axis=0))
