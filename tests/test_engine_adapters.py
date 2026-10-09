"""The unified engine's source adapters (:mod:`bouquet.adapters`) -- fast.

On the repository's synthetic inputs (the D3D-like example g-file + p-file
and the synthetic OMAS file; no solver):

* each adapter produces the contract (kinetics, fixed pressure, parallel
  components, boundary, rows, signs);
* the g-file's ``<j.B>`` (identity I0 on the reader's own surfaces) composes
  back to the g-file's own ``<j_phi>`` (identity I2) to rounding, and a file
  read with the wrong COCOS is refused;
* the g-file inductive is smoothed with EXACTLY the legacy
  ``fit_inductive_profile`` basis, without its amplitude search;
* the IDS inductive is, by definition (the default), the parallel residual
  ``|B0| (j_total - j_bootstrap - sum(driven))`` in the positive frame, with
  the source's ``j_ohmic`` a stamped cross-check ("auto" / "j_ohmic" use it
  explicitly); the rows are the source's own (Ip soft, ``li_3`` soft, q0 at
  the measurement radius);
* a raw-E_r MSE request is refused, E_r-corrected chords are accepted;
* identity (I2) on STORED geometry of the synthetic golden fixture (read-only
  h5py): the draw's own g-file composes back exactly, and the conversion
  factor ``F<1/R>/<B^2>`` from its surfaces agrees with TokaMaker's archived
  flux-surface averages to the level the verification report measured.

Synthetic inputs only.
"""
import os

import numpy as np
import pytest

from bouquet.adapters import (IDS_INDUCTIVE_DEFAULT, EngineInputRefused,
                              GFileAdapter, IdsAdapter,
                              gfile_parallel_current, inductive_basis,
                              mse_rows, validate_contract)
from bouquet.engine import compose, conversion_factor

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")
_GOLD = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
_TIME = 2.3043


def _gcfg(**gen):
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=_GEQ, profiles_path=_PF),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


def _icfg(**gen):
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    return BouquetConfig(
        source=ImasSource(ids_path=_OMAS, time=_TIME),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


@pytest.fixture(scope="module")
def gfile():
    ad = GFileAdapter(_gcfg().source, _gcfg())
    c = ad.read()
    return ad, c


@pytest.fixture(scope="module")
def ids():
    import warnings
    from bouquet.baseline import resolve_baseline
    cfg = _icfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
    ad = IdsAdapter(cfg.source, cfg, bl)
    return ad, ad.read(), bl


def _geom(parts):
    return dict(F=parts["F"], R_avg=parts["R_avg"], inv_R=parts["inv_R"],
                B2=parts["B2"], pprime=parts["pprime"])


# ---------------------------------------------------------------------------
#  g-file
# ---------------------------------------------------------------------------
def test_the_gfile_adapter_reads_the_contract(gfile):
    ad, c = gfile
    assert c.kind == "gfile" and c.jB_ind is None      # before the anchor
    n = c.psi_N.size
    for k in ("ne", "te", "ni", "ti", "zeff"):
        assert c.kinetics[k].shape == (n,) and np.all(c.kinetics[k] > 0)
    assert np.all(c.pressure >= 0) and c.pressure[0] > c.pressure[-1]
    np.testing.assert_allclose(
        c.pressure, sum(c.pressure_parts[k] for k in
                        ("thermal", "impurity", "fast")), rtol=0, atol=0)
    assert c.Ip == pytest.approx(abs(float(ad.eqdsk.Ip)))
    li = c.rows["l_i"]
    assert li["hard"] and li["kind"] == "li_3" and li["tol"] == 1e-3
    assert li["target"] == float(ad.eqdsk.li["li(2)"])
    # q0 at the measurement radius (like radii), with the sawtooth gate
    q = c.rows["q0"]
    assert q["psi"] == pytest.approx(1e-3)
    assert q["target"] == pytest.approx(float(np.interp(
        1e-3, ad.eqdsk.psi_N, np.abs(ad.eqdsk.qpsi))))
    assert q["admitted"] is (abs(q["q0_source_axis"]) <= 1.1)
    assert c.rows["mse"] is None
    assert c.signs["current_sign"] == 1.0
    assert c.boundary.shape[1] == 2 and c.boundary.shape[0] > 10


def test_the_gfile_parallel_current_composes_back_to_its_own_jphi(gfile):
    """Identity (I2) on the g-file's own surfaces, with any split."""
    ad, c = gfile
    jB, parts = gfile_parallel_current(ad.eqdsk)
    assert parts["frame_identity_max_rel"] < 1e-9
    J, p = compose(_geom(parts), 0.6 * jB, 0.3 * jB, 0.1 * jB)
    scale = float(np.max(np.abs(parts["jphi_in"])))
    assert float(np.max(np.abs(J - parts["jphi_in"]))) / scale < 1e-9
    # the anchor request is the legacy reconstruction's total
    np.testing.assert_array_equal(
        c.anchor_request, np.abs(ad.eqdsk.j_tor_averaged_direct))
    # the pressure-driven term is a measurable piece of the edge current
    frac = p["pressure"] / parts["jphi_in"]
    assert 0.0 < float(np.median(frac[c.psi_N > 0.9])) < 1.0


def test_the_gfile_adapter_finalizes_with_the_anchor_bootstrap(gfile):
    ad, c = gfile
    jB, parts = gfile_parallel_current(ad.eqdsk)
    psi = c.psi_N
    redl = 0.2 * float(np.max(jB)) * np.exp(-0.5 * ((psi - 0.95) / 0.03) ** 2)
    ad2 = GFileAdapter(ad.source, ad.config)
    ad2.read()
    c2 = ad2.finalize(redl, _geom(parts))
    validate_contract(c2)
    assert np.all(c2.jB_ind >= 0.0)
    np.testing.assert_array_equal(c2.jB_bs_anchor, redl)
    expect = inductive_basis(psi, jB - redl - c2.jB_fix,
                             k=int(ad.source.n_k),
                             psi_bridge=float(ad.source.psi_bridge))
    np.testing.assert_array_equal(c2.jB_ind, expect)
    # no fixed parts configured: the fixed <j.B> is zero
    assert not np.any(c2.jB_fix)


def test_a_gfile_read_with_the_wrong_cocos_is_refused():
    import h5py
    from bouquet.io.geqdsk import GEQDSKEquilibrium
    with h5py.File(_GOLD, "r") as f:
        raw = bytes(f["scan/0/0/eqdsk"][()])
    g = GEQDSKEquilibrium.from_bytes(raw, cocos=1)   # it is written COCOS 7
    with pytest.raises(EngineInputRefused, match="sign"):
        gfile_parallel_current(g)


def test_the_inductive_basis_is_the_legacy_basis_without_the_amplitude(
        gfile, monkeypatch):
    """fit_inductive_profile returns ind_scale * basis; the adapter's copy
    returns basis -- the same to rounding."""
    import bouquet.TokaMaker_interface as ti
    ad, c = gfile
    jB, _ = gfile_parallel_current(ad.eqdsk)
    psi = c.psi_N
    resid = jB - 0.3 * jB * np.exp(-0.5 * ((psi - 0.95) / 0.03) ** 2)
    # the amplitude search's proxy, replaced by a linear stand-in
    monkeypatch.setattr(ti, "calc_cylindrical_li_proxy",
                        lambda mygs, j, pad, *a: float(np.sum(j)) * 1e-9)
    fit = ti.fit_inductive_profile(None, resid, np.zeros_like(psi), psi,
                                   1e-3, float(np.sum(resid)) * 0.9e-9,
                                   k=5, psi_bridge=0.99)
    assert fit["ind_scale"] != 1.0
    np.testing.assert_allclose(fit["j_inductive_fit"] / fit["ind_scale"],
                               inductive_basis(psi, resid, k=5,
                                               psi_bridge=0.99),
                               rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
#  IDS
# ---------------------------------------------------------------------------
def test_the_ids_adapter_reads_the_contract(ids):
    import json
    ad, c, bl = ids
    assert c.kind == "ids"
    validate_contract(c)
    with open(_OMAS) as fh:
        dd = json.load(fh)
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    # the default inductive is the parallel residual, by definition
    assert ad.inductive == IDS_INDUCTIVE_DEFAULT == "residual"
    np.testing.assert_allclose(
        c.jB_ind, B0 * (np.asarray(cp["j_total"])
                        - np.asarray(cp["j_bootstrap"]) - c.jB_fix / B0),
        rtol=1e-12)
    assert c.provenance["inductive"].startswith("residual")
    assert c.rows["Ip"]["hard"] is False
    assert c.rows["Ip"]["sigma"] == pytest.approx(0.005 * bl.Ip_target)
    li = c.rows["l_i"]
    gq = dd["equilibrium"]["time_slice"][2]["global_quantities"]
    assert li["kind"] == "li_3" and li["target"] == float(gq["li_3"])
    assert li["hard"] is False and li["sigma"] == 0.04
    q = c.rows["q0"]
    assert q["psi"] == pytest.approx(1e-3) and q["target"] > 0
    assert c.signs["b0_sign"] == -1.0 and c.signs["current_sign"] == 1.0
    # pressure: thermal + impurity + fast, NO p_diff
    assert "no p_diff" in c.provenance["pressure"]
    np.testing.assert_array_equal(c.anchor_request, bl.j_phi)


def test_the_ids_residual_inductive(ids):
    """Explicit "residual" is the default contract bit for bit; explicit
    "j_ohmic" takes the source's j_ohmic exactly."""
    import json
    ad, c, bl = ids
    ad2 = IdsAdapter(ad.source, ad.config, bl, inductive="residual")
    c2 = ad2.read()
    with open(_OMAS) as fh:
        cp = json.load(fh)["core_profiles"]["profiles_1d"][2]
    B0 = 1.8
    nbi = c2.jB_fix / B0
    np.testing.assert_allclose(
        c2.jB_ind, B0 * (np.asarray(cp["j_total"])
                         - np.asarray(cp["j_bootstrap"]) - nbi), rtol=1e-12)
    assert c2.provenance["inductive"].startswith("residual")
    np.testing.assert_array_equal(c2.jB_ind, c.jB_ind)
    assert c2.provenance["inductive"] == c.provenance["inductive"]
    c3 = IdsAdapter(ad.source, ad.config, bl, inductive="j_ohmic").read()
    np.testing.assert_allclose(c3.jB_ind, B0 * np.asarray(cp["j_ohmic"]),
                               rtol=1e-15)
    assert c3.provenance["inductive"].startswith("j_ohmic")


def test_the_ids_adapter_refuses_without_b0(ids, tmp_path):
    import json
    ad, c, bl = ids
    with open(_OMAS) as fh:
        dd = json.load(fh)
    del dd["equilibrium"]["vacuum_toroidal_field"]["b0"]
    p = tmp_path / "no_b0.json"
    p.write_text(json.dumps(dd))
    from bouquet.config import ImasSource
    src = ImasSource(ids_path=str(p), time=_TIME)
    with pytest.raises(EngineInputRefused, match="b0"):
        IdsAdapter(src, ad.config, bl).read()


def _mse_block(**over):
    n = 6
    md = dict(R=np.linspace(1.7, 2.2, n), Z=np.zeros(n),
              tgamma=np.linspace(0.02, 0.1, n), sigma=np.full(n, 0.002),
              weight=np.ones(n), A1=np.ones(n), A2=np.ones(n),
              A3=np.zeros(n), A4=np.zeros(n))
    md.update(over)
    return md


_SIGNS = dict(ip_sign=1.0, bt_sign=-1.0, basis="test source")


def _gen(md):
    from bouquet.config import GenerationConfig
    return GenerationConfig(reconstruction_engine="unified",
                            engine_rows=["Ip", "l_i", "mse"], mse_data=md)


def test_raw_er_mse_is_refused_and_corrected_mse_is_accepted():
    g = _gen(_mse_block(er_corrected=True, ip_sign=1, bt_sign=-1))
    r = mse_rows(g, _SIGNS)
    assert r["chords"]["n_active"] == 6 and r["chords"]["er_corrected"]
    for bad in (dict(), dict(A5=np.ones(6), Er=np.full(6, 1e3))):
        g.mse_data = _mse_block(ip_sign=1, bt_sign=-1, **bad)
        with pytest.raises(EngineInputRefused, match="E_r"):
            mse_rows(g, _SIGNS)


def test_the_mse_orientation_is_stated_by_the_block_or_the_source():
    # a block that states nothing is completed from the source
    r = mse_rows(_gen(_mse_block(er_corrected=True)), _SIGNS)
    assert (r["chords"]["ip_sign"], r["chords"]["bt_sign"]) == (1.0, -1.0)
    assert r["orientation"]["filled_from_source"] == ["ip_sign", "bt_sign"]
    # a block that agrees keeps its statement
    r = mse_rows(_gen(_mse_block(er_corrected=True, ip_sign=1, bt_sign=-1)),
                 _SIGNS)
    assert r["orientation"]["filled_from_source"] == []
    # a block the source contradicts is refused, never resolved silently
    with pytest.raises(EngineInputRefused, match="disagree"):
        mse_rows(_gen(_mse_block(er_corrected=True, ip_sign=1, bt_sign=1)),
                 _SIGNS)
    # neither states it: refused
    with pytest.raises(EngineInputRefused, match="bt_sign"):
        mse_rows(_gen(_mse_block(er_corrected=True)),
                 dict(ip_sign=1.0, bt_sign=None, basis="no b0"))


def test_both_adapters_declare_the_same_orientation_for_the_same_discharge(
        gfile, ids):
    """The synthetic g-file (COCOS 1, Ip > 0, B_t < 0) and the OMAS file
    (COCOS 11, ip > 0, b0 < 0) describe one generic discharge: both state
    ip_sign = +1, bt_sign = -1 in the (R, phi, Z) frame."""
    _ad, cg = gfile
    _ai, ci, _bl = ids
    for c in (cg, ci):
        assert (c.signs["ip_sign_RphiZ"], c.signs["bt_sign_RphiZ"]) \
            == (1.0, -1.0)
        assert c.signs["orientation_basis"]


# ---------------------------------------------------------------------------
#  IDS orientation: ONE normalisation, the reader's
# ---------------------------------------------------------------------------
def _ids_contract(path, fixed=None, **src):
    """``(adapter, contract, baseline)`` of the IDS at *path* through the
    reader and the adapter (no solver)."""
    import warnings
    from bouquet.baseline import resolve_baseline
    from bouquet.config import (BouquetConfig, FixedComponentsConfig,
                                GenerationConfig, ImasSource, SolverConfig)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=str(path), time=_TIME, **src),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"),
        fixed_components=(fixed or FixedComponentsConfig()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
    ad = IdsAdapter(cfg.source, cfg, bl)
    return ad, ad.read(), bl


def _example_dd():
    import json
    with open(_OMAS) as fh:
        return json.load(fh)


def _write_dd(tmp_path, dd, name):
    import json
    p = tmp_path / name
    with open(p, "w") as fh:
        json.dump(dd, fh)
    return p


def test_a_reversed_source_gives_the_same_contract_in_the_positive_frame(
        tmp_path, ids):
    """Every orientation of one discharge reads to the SAME parallel
    components, bit for bit: the adapter applies the reader's factor."""
    import _mirror_dd as mdd
    _ad, ref, bl_ref = ids
    assert ref.signs["current_sign"] == bl_ref.source_current_sign == 1.0
    for s_ip, s_b0 in mdd.ORIENTATIONS:
        p = _write_dd(tmp_path, mdd.mirror_dd(_example_dd(), s_ip, s_b0),
                      f"dd_{mdd.tag(s_ip, s_b0)}.json")
        _a, c, bl = _ids_contract(p)
        assert c.signs["current_sign"] == bl.source_current_sign == s_ip
        assert c.signs["current_sign_origin"] == \
            bl.source_current_sign_origin
        for name in ("jB_ind", "jB_fix"):
            np.testing.assert_array_equal(getattr(c, name),
                                          getattr(ref, name), err_msg=name)
        np.testing.assert_array_equal(c.jB_fix_parts["nbi"],
                                      ref.jB_fix_parts["nbi"])
        assert np.median(c.jB_ind) > 0.0                # co-current positive
        # the orientation the MSE rows take: the source's own directions
        assert c.signs["ip_sign_RphiZ"] == s_ip
        assert c.signs["bt_sign_RphiZ"] == -1.0 * s_b0   # the example: b0 < 0


def test_the_adapter_takes_the_readers_factor_not_a_second_one(tmp_path,
                                                               ids):
    """A source whose currents are stored reversed against its ip, read with
    ImasSource.current_orientation: the adapter follows the reader's factor
    (the contract is the consistent file's), and -- the Ip direction being
    stated twice and inconsistently -- states no ip_sign for the MSE rows."""
    _ad, ref, _bl = ids
    dd = _example_dd()
    for c in dd["core_profiles"]["profiles_1d"]:
        for k in ("j_tor", "j_total", "j_ohmic", "j_bootstrap",
                  "j_non_inductive"):
            if k in c:
                c[k] = [-v for v in c[k]]
    for s in dd["core_sources"]["source"]:
        for pr in s["profiles_1d"]:
            pr["j_parallel"] = [-v for v in pr["j_parallel"]]
    for ts in dd["equilibrium"]["time_slice"]:
        ts["profiles_1d"]["j_tor"] = [-v for v in ts["profiles_1d"]["j_tor"]]
    p = _write_dd(tmp_path, dd, "cur_rev.json")
    with pytest.raises(ValueError, match="disagree in sign"):
        _ids_contract(p)                                  # the reader refuses
    _a, c, bl = _ids_contract(p, current_orientation=-1)
    assert bl.source_current_sign == -1.0
    assert c.signs["current_sign"] == -1.0
    assert "override" in c.signs["current_sign_origin"]
    np.testing.assert_array_equal(c.jB_ind, ref.jB_ind)
    np.testing.assert_array_equal(c.jB_fix, ref.jB_fix)
    # ip > 0 in the file, currents read with -1: no Ip direction is stated
    assert c.signs["ip_sign_RphiZ"] is None
    assert "NOT stated" in c.signs["orientation_basis"]
    assert c.signs["bt_sign_RphiZ"] == -1.0
    # ... so an MSE row needs the block's own ip_sign
    g = _gen(_mse_block(er_corrected=True))
    with pytest.raises(EngineInputRefused, match="ip_sign"):
        mse_rows(g, dict(ip_sign=c.signs["ip_sign_RphiZ"],
                         bt_sign=c.signs["bt_sign_RphiZ"],
                         basis=c.signs["orientation_basis"]))
    r = mse_rows(_gen(_mse_block(er_corrected=True, ip_sign=1)),
                 dict(ip_sign=None, bt_sign=-1.0, basis="test"))
    assert r["chords"]["ip_sign"] == 1.0


def test_user_driven_currents_are_positive_frame_for_any_orientation(
        tmp_path):
    """The same FixedComponentsConfig j_NBI / j_RF (co-current positive)
    gives the same toroidal fixed parts whichever way the source is
    oriented -- as the reader takes them; only the dd's own currents are
    re-signed."""
    import _mirror_dd as mdd
    from bouquet.config import FixedComponentsConfig
    x = np.linspace(0.0, 1.0, 33)
    j_nbi = 4.0e4 * (1.0 - x ** 2)
    j_rf = 1.5e4 * np.exp(-0.5 * ((x - 0.4) / 0.1) ** 2)
    got = {}
    for s_ip, s_b0 in ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)):
        p = _write_dd(tmp_path, mdd.mirror_dd(_example_dd(), s_ip, s_b0),
                      f"dd_{mdd.tag(s_ip, s_b0)}.json")
        ad, c, bl = _ids_contract(p, fixed=FixedComponentsConfig(
            psi_N=x, j_NBI=j_nbi, j_RF=j_rf))
        assert bl.source_current_sign == s_ip
        got[(s_ip, s_b0)] = ad._user_fix_tor
        np.testing.assert_array_equal(
            ad._user_fix_tor["nbi"], np.interp(c.psi_N, x, j_nbi))
        np.testing.assert_array_equal(
            ad._user_fix_tor["rf"], np.interp(c.psi_N, x, j_rf))
        assert np.all(ad._user_fix_tor["nbi"] >= 0.0)
        # the reader resampled the same arrays, in the same frame
        assert np.all(np.asarray(bl.j_NBI) >= 0.0)
    for k in ("nbi", "rf"):
        np.testing.assert_array_equal(got[(1.0, 1.0)][k],
                                      got[(-1.0, 1.0)][k])
        np.testing.assert_array_equal(got[(1.0, 1.0)][k],
                                      got[(-1.0, -1.0)][k])


# ---------------------------------------------------------------------------
#  identity (I2) on the golden fixture's stored geometry (read-only)
# ---------------------------------------------------------------------------
def test_identity_I2_on_the_golden_fixtures_stored_geometry():
    import h5py
    from bouquet.io.geqdsk import GEQDSKEquilibrium
    with h5py.File(_GOLD, "r") as f:
        raw = bytes(f["scan/0/0/eqdsk"][()])
        fsa = {k: f["scan/0/0/eq_fsa/" + k][()]
               for k in ("psi_N", "F", "avg_inv_R", "avg_B2")}
    g = GEQDSKEquilibrium.from_bytes(raw, cocos=7)
    jB, parts = gfile_parallel_current(g)
    J, _ = compose(_geom(parts), 0.6 * jB, 0.3 * jB, 0.1 * jB)
    scale = float(np.max(np.abs(parts["jphi_in"])))
    assert float(np.max(np.abs(J - parts["jphi_in"]))) / scale < 1e-9
    # the conversion factor from the g-file's own surfaces against
    # TokaMaker's archived averages of the same draw: the verification
    # report measured the archived averages against independent contour
    # averages at 7e-6 to 1.3e-3 (worst at psi_N 0.97); the same level
    # here, over psi_N 0.01-0.99
    kap_g = np.interp(fsa["psi_N"], g.psi_N, conversion_factor(_geom(parts)))
    kap_t = np.abs(fsa["F"]) * fsa["avg_inv_R"] / fsa["avg_B2"]
    m = (fsa["psi_N"] >= 0.01) & (fsa["psi_N"] <= 0.99)
    assert float(np.max(np.abs(kap_g[m] / kap_t[m] - 1.0))) <= 1.3e-3


# ---------------------------------------------------------------------------
#  the IDS source's current split: every driven entry held fixed, and
#  j_ohmic checked against the parallel residual (owner-set 2 % tolerance)
# ---------------------------------------------------------------------------
def _ids_from(tmp_path, dd, name, **kw):
    """Adapter + contract of a modified copy of the example dd."""
    import warnings
    from bouquet.baseline import resolve_baseline
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    p = _write_dd(tmp_path, dd, name)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=str(p), time=_TIME),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
    return IdsAdapter(cfg.source, cfg, bl, **kw)


def _add_source(dd, name, index, j_parallel_of):
    """Append a core_sources entry shaped like the beam one, with
    ``j_parallel = j_parallel_of(profiles_1d entry)`` at every time."""
    import copy
    base = next(s for s in dd["core_sources"]["source"]
                if s["identifier"]["index"] == 2)
    s = copy.deepcopy(base)
    s["identifier"] = dict(name=name, index=index)
    for i, q in enumerate(s["profiles_1d"]):
        q["j_parallel"] = [float(v) for v in
                           j_parallel_of(dd["core_profiles"]["profiles_1d"][i])]
    dd["core_sources"]["source"].append(s)
    return s


def test_the_ids_split_is_checked_and_stamped_on_a_consistent_source(ids):
    """The default (residual by definition) stamps the j_ohmic cross-check
    with no threshold; explicit "auto" on the same consistent source keeps
    j_ohmic, with the same numbers stamped."""
    ad, c, bl = ids
    k = c.provenance["inductive_consistency"]
    assert k["checked"] is True and k["action"] == "residual_by_definition"
    assert k["tol"] is None and abs(k["net_frac"]) <= 0.02
    assert np.isfinite(k["rms_frac"])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ca = IdsAdapter(ad.source, ad.config, bl, inductive="auto").read()
    ka = ca.provenance["inductive_consistency"]
    assert ka["action"] == "kept_j_ohmic" and ka["tol"] == 0.02
    assert (ka["net_frac"], ka["rms_frac"]) == (k["net_frac"], k["rms_frac"])
    assert ca.provenance["inductive"].startswith("j_ohmic")
    assert sorted(c.jB_fix_parts) == ["nbi", "other", "rf"]
    np.testing.assert_array_equal(c.jB_fix_parts["rf"], 0.0)
    np.testing.assert_array_equal(c.jB_fix_parts["other"], 0.0)
    np.testing.assert_array_equal(
        c.jB_fix, c.jB_fix_parts["nbi"] + c.jB_fix_parts["rf"]
        + c.jB_fix_parts["other"])
    assert [d["part"] for d in c.provenance["driven_sources"]] == ["nbi"]


def test_an_inconsistent_split_falls_back_to_the_residual_loudly(tmp_path):
    """j_ohmic scaled by 1.06 with j_total untouched: the split misses by
    ~5 % of the total current, above the 2 % tolerance.  Explicit "auto"
    takes the residual and says so; inductive="j_ohmic" keeps j_ohmic and
    says so; "auto" with a looser tolerance keeps j_ohmic silently.  The
    source is never altered: the stamped numbers are the same in all
    three."""
    dd = _example_dd()
    for q in dd["core_profiles"]["profiles_1d"]:
        q["j_ohmic"] = [1.06 * float(v) for v in q["j_ohmic"]]
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    with pytest.warns(UserWarning, match="does not add up"):
        c = _ids_from(tmp_path, dd, "ohm_off.json", inductive="auto").read()
    k = c.provenance["inductive_consistency"]
    assert k["action"] == "fallback_to_residual" and k["net_frac"] > 0.02
    assert c.provenance["inductive"].startswith("residual")
    np.testing.assert_allclose(
        c.jB_ind, B0 * (np.asarray(cp["j_total"])
                        - np.asarray(cp["j_bootstrap"]) - c.jB_fix / B0),
        rtol=1e-12)
    with pytest.warns(UserWarning, match="kept although"):
        c2 = _ids_from(tmp_path, dd, "ohm_off.json",
                       inductive="j_ohmic").read()
    assert c2.provenance["inductive_consistency"]["action"] == \
        "kept_j_ohmic_over_tol"
    np.testing.assert_allclose(c2.jB_ind, B0 * np.asarray(cp["j_ohmic"]),
                               rtol=1e-15)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c3 = _ids_from(tmp_path, dd, "ohm_off.json", inductive="auto",
                       inductive_tol=0.10).read()
    assert c3.provenance["inductive_consistency"]["action"] == "kept_j_ohmic"
    np.testing.assert_allclose(c3.jB_ind, c2.jB_ind, rtol=0.0)
    for cc in (c2, c3):
        assert cc.provenance["inductive_consistency"]["net_frac"] == \
            k["net_frac"]


def test_every_driven_source_entry_is_held_fixed_and_in_the_residual(
        tmp_path):
    """An ec entry (rf), a model's sawteeth entry (other), plus ohmic and
    bootstrap entries that must be IGNORED: the fixed current is the sum
    of the driven ones by part, the residual subtracts them all, and a
    source whose j_ohmic accounts for them stays consistent."""
    dd = _example_dd()
    ec = lambda q: 0.03 * np.asarray(q["j_total"])          # noqa: E731
    saw = lambda q: 0.002 * np.asarray(q["j_total"])        # noqa: E731
    _add_source(dd, "ec", 3, ec)
    _add_source(dd, "sawteeth", 701, saw)
    _add_source(dd, "ohmic", 7, lambda q: np.asarray(q["j_ohmic"]))
    _add_source(dd, "bootstrap", 13, lambda q: np.asarray(q["j_bootstrap"]))
    for q in dd["core_profiles"]["profiles_1d"]:
        q["j_ohmic"] = [float(v) for v in
                        np.asarray(q["j_ohmic"]) - ec(q) - saw(q)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = _ids_from(tmp_path, dd, "driven.json", inductive="auto").read()
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    np.testing.assert_allclose(c.jB_fix_parts["rf"], B0 * ec(cp), rtol=1e-12)
    np.testing.assert_allclose(c.jB_fix_parts["other"], B0 * saw(cp),
                               rtol=1e-12)
    np.testing.assert_allclose(
        c.jB_fix, c.jB_fix_parts["nbi"] + B0 * (ec(cp) + saw(cp)),
        rtol=1e-12)
    used = {(d["name"], d["part"]) for d in c.provenance["driven_sources"]}
    assert used == {("nbi_synthetic", "nbi"), ("ec", "rf"),
                    ("sawteeth", "other")}
    k = c.provenance["inductive_consistency"]
    assert k["action"] == "kept_j_ohmic" and abs(k["net_frac"]) <= 0.02
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c2 = _ids_from(tmp_path, dd, "driven.json").read()     # the default
    k2 = c2.provenance["inductive_consistency"]
    assert k2["action"] == "residual_by_definition"
    assert k2["net_frac"] == k["net_frac"]
    np.testing.assert_array_equal(c2.jB_fix, c.jB_fix)
    np.testing.assert_allclose(
        c2.jB_ind, B0 * (np.asarray(cp["j_total"])
                         - np.asarray(cp["j_bootstrap"])) - c.jB_fix,
        rtol=1e-12)


def _driven_dd():
    """The example dd with an ec entry and a sawteeth entry whose j_ohmic
    accounts for them (consistent split)."""
    dd = _example_dd()
    ec = lambda q: 0.03 * np.asarray(q["j_total"])          # noqa: E731
    saw = lambda q: 0.002 * np.asarray(q["j_total"])        # noqa: E731
    _add_source(dd, "ec", 3, ec)
    _add_source(dd, "sawteeth", 701, saw)
    for q in dd["core_profiles"]["profiles_1d"]:
        q["j_ohmic"] = [float(v) for v in
                        np.asarray(q["j_ohmic"]) - ec(q) - saw(q)]
    return dd


@pytest.mark.parametrize("index, name", [
    (1, "total"), (100, "auxiliary"), (101, "ic_nbi"), (104, "ec_lh"),
    (107, "ec_lh_ic"), (203, "impurity_radiation"), (401, "neoclassical")])
def test_an_aggregate_or_bootstrap_like_entry_is_never_added(tmp_path,
                                                             index, name):
    """A "total" entry (the sum of every source -- here nbi + ec + sawteeth
    + ohmic + bootstrap, as an aggregate carries) or a combination entry,
    or a bootstrap published as "neoclassical": NOT added to the driven
    current (it would double-count its constituents), stamped in
    provenance["ignored_sources"] and warned about.  The contract is the
    one without that entry, bit for bit."""
    import warnings
    dd0 = _driven_dd()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c0 = _ids_from(tmp_path, dd0, "base.json").read()
    dd = _driven_dd()
    _add_source(dd, name, index, lambda q: np.asarray(q["j_total"]))
    with pytest.warns(UserWarning, match="NOT added to the driven current"):
        c = _ids_from(tmp_path, dd, f"agg{index}.json").read()
    np.testing.assert_array_equal(c.jB_fix, c0.jB_fix)
    np.testing.assert_array_equal(c.jB_ind, c0.jB_ind)
    for k in ("nbi", "rf", "other"):
        np.testing.assert_array_equal(c.jB_fix_parts[k], c0.jB_fix_parts[k])
    ig = c.provenance["ignored_sources"]
    assert [(d["name"], d["index"]) for d in ig] == [(name, index)]
    assert ig[0]["j_parallel_max_abs"] > 0.0
    assert index not in [d["index"] for d in c.provenance["driven_sources"]]
    assert c0.provenance["ignored_sources"] == []
    # the j_ohmic-vs-residual cross-check and its stamp stay
    assert c.provenance["inductive_consistency"] == \
        c0.provenance["inductive_consistency"]
    assert c.provenance["inductive_consistency"]["checked"] is True


def test_the_pre_fix_rule_would_have_double_counted_a_total(tmp_path):
    """Witness: held fixed (the old "everything except ohmic and bootstrap"
    rule), a total entry drives the residual inductive current NEGATIVE
    over the bulk -- what the classification prevents."""
    dd = _driven_dd()
    cp = dd["core_profiles"]["profiles_1d"][2]
    jt = np.asarray(cp["j_total"], float)
    resid_old = (jt - np.asarray(cp["j_bootstrap"], float)
                 - 0.03 * jt - 0.002 * jt - jt)
    assert np.median(resid_old) < 0.0


def test_an_unknown_index_is_held_fixed_as_other_with_a_warning(tmp_path):
    import warnings
    dd0 = _driven_dd()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c0 = _ids_from(tmp_path, dd0, "base.json").read()
    dd = _driven_dd()
    extra = lambda q: 0.004 * np.asarray(q["j_total"])      # noqa: E731
    _add_source(dd, "custom_1", 901, extra)
    with pytest.warns(UserWarning, match="not a known driven source"):
        c = _ids_from(tmp_path, dd, "unknown.json").read()
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    np.testing.assert_allclose(c.jB_fix_parts["other"] - c0.jB_fix_parts[
        "other"], B0 * extra(cp), rtol=1e-12, atol=1e-9)
    u = [d for d in c.provenance["driven_sources"] if d["index"] == 901]
    assert u and u[0]["part"] == "other" and u[0]["unclassified"] is True
    # the known ones are classified without a warning
    assert {(d["index"], d["part"]) for d in c0.provenance["driven_sources"]} \
        == {(2, "nbi"), (3, "rf"), (701, "other")}


def test_an_entry_is_read_at_its_own_time_not_its_list_index(tmp_path):
    """A model's entry that starts later than the IDS time base (one slice
    fewer, each with its own time): the slice read is the one AT the
    core_sources time, not the list index (which would be the NEXT time),
    and at a time the entry does not cover it is not added (stamped).  An
    entry with a different slice count and no times cannot be aligned and
    is refused -- never its first slice in place of the missing one."""
    import warnings
    dd = _driven_dd()
    t = list(dd["core_sources"]["time"])
    saw = next(s for s in dd["core_sources"]["source"]
               if s["identifier"]["index"] == 701)
    for q, tq in zip(saw["profiles_1d"], t):
        q["time"] = tq
    for s in dd["core_sources"]["source"]:
        for q, tq in zip(s["profiles_1d"], t):
            q["time"] = tq
    marks = [np.full(len(saw["profiles_1d"][0]["j_parallel"]), 1.0e3 * (k + 1))
             for k in range(len(t))]
    for q, m in zip(saw["profiles_1d"], marks):
        q["j_parallel"] = m.tolist()
    # the entry starts one slice late: drop its first slice
    saw["profiles_1d"] = saw["profiles_1d"][1:]
    isrc = len(t) - 1                    # the example's time is the last
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = _ids_from(tmp_path, dd, "late.json").read()
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    np.testing.assert_allclose(c.jB_fix_parts["other"], B0 * marks[isrc],
                               rtol=1e-12)
    d = [x for x in c.provenance["driven_sources"] if x["index"] == 701]
    assert d[0]["slice"] == "matched by time"
    # no per-slice time and a different count: refused.  Since the legacy
    # reader reads its sawtooth gate by the same rule (2026-10-05), the
    # reader -- which runs first -- refuses such a sawteeth entry before the
    # adapter sees it; the adapter's own refusal is checked directly.
    from bouquet.adapters import _ids_source_slice
    for q in saw["profiles_1d"]:
        q.pop("time")
    with pytest.raises(ValueError, match="IMAS reader: .*cannot be aligned"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ids_from(tmp_path, dd, "late_notime.json").read()
    with pytest.raises(EngineInputRefused, match="IDS adapter: .*cannot be "
                                                 "aligned"):
        _ids_source_slice(saw, isrc, float(t[isrc]), len(t))


def test_a_driven_entry_without_a_slice_at_this_time_is_refused(tmp_path):
    """A driven entry whose nearest own slice is more than half its local
    time-step from the slice time is REFUSED (owner-approved 2026-10-06; it
    was dropped to zero and stamped under the 1e-6 s match).  An aggregate
    entry in the same position is never added anyway: stamped, not
    refused.  An all-zero driven entry has nothing to drop: skipped.
    (Owner decision 2026-10-06: a slice BEFORE the entry's first own time is
    OFF -- off_before_record, tests/test_imas_time_rule.py -- so the refusal
    is checked past the entry's end, where it is unchanged.)"""
    from bouquet.adapters import _ids_driven_currents
    n = 5
    srcs = dict(time=[1.0, 2.0, 3.0], source=[dict(
        identifier=dict(name="sawteeth", index=701),
        profiles_1d=[dict(time=2.0, j_parallel=[1.0] * n),
                     dict(time=3.0, j_parallel=[2.0] * n)])])
    late = dict(srcs, time=[2.0, 3.0, 4.0])
    with pytest.raises(EngineInputRefused,
                       match=r"IDS adapter: core_sources 'sawteeth' \(index "
                             r"701\) carries a non-zero j_parallel but has no "
                             r"profiles_1d slice within half a time-step of "
                             r"t = 4 s.*Refusing"):
        _ids_driven_currents(late, 2, n, 1.0)
    off = []
    with pytest.warns(UserWarning, match="off_before_record"):
        parts, used, ignored = _ids_driven_currents(srcs, 0, n, 1.0, off=off,
                                                    announce_key="t-refused")
    assert used == [] and np.all(parts["other"] == 0.0)
    assert off == [dict(name="sawteeth", index=701,
                        reason="off_before_record", first_own_time=2.0)]
    parts, used, ignored = _ids_driven_currents(srcs, 2, n, -1.0)
    np.testing.assert_array_equal(parts["other"], -2.0)
    assert ignored == []
    # nearest own slice within half a step: matched, not dropped
    parts, used, ignored = _ids_driven_currents(
        dict(srcs, time=[1.6, 2.0, 3.0]), 0, n, 1.0)
    np.testing.assert_array_equal(parts["other"], 1.0)
    assert used[0]["slice"] == "matched by time"
    # an aggregate entry out of window: stamped, never added, not refused
    agg = dict(time=[1.0, 2.0, 3.0], source=[dict(
        identifier=dict(name="total", index=1),
        profiles_1d=[dict(time=2.0, j_parallel=[1.0] * n),
                     dict(time=3.0, j_parallel=[2.0] * n)])])
    with pytest.warns(UserWarning, match="NOT added"):
        parts, used, ignored = _ids_driven_currents(agg, 0, n, 1.0)
    assert used == [] and ignored[0]["index"] == 1
    assert ignored[0]["reason"].startswith("no profiles_1d slice within half")
    # an all-zero driven entry out of window: nothing to drop
    zero = dict(time=[1.0, 2.0, 3.0], source=[dict(
        identifier=dict(name="nbi", index=2),
        profiles_1d=[dict(time=2.0, j_parallel=[0.0] * n),
                     dict(time=3.0, j_parallel=[0.0] * n)])])
    parts, used, ignored = _ids_driven_currents(zero, 0, n, 1.0)
    assert used == [] and ignored == [] and np.all(parts["nbi"] == 0.0)


def test_a_two_microsecond_offset_entry_is_read_at_its_nearest_slice():
    """The 2026-10-06 review's case on the engine path: own times 2 us off
    the base -- each slice reads its nearest own slice (it was dropped to
    zero and stamped under the 1e-6 s match)."""
    from bouquet.adapters import _ids_driven_currents
    n = 4
    tb = [2.1, 2.2, 2.3043]
    srcs = dict(time=tb, source=[dict(
        identifier=dict(name="nbi", index=2),
        profiles_1d=[dict(time=tk + 2e-6, j_parallel=[1.0e3 * (k + 1)] * n)
                     for k, tk in enumerate(tb)])])
    for k in range(3):
        parts, used, ignored = _ids_driven_currents(srcs, k, n, 1.0, tb)
        np.testing.assert_array_equal(parts["nbi"], 1.0e3 * (k + 1))
        assert ignored == [] and used[0]["slice"] == "matched by time"
    # a single-time entry uses the core_profiles step (0.1 s here)
    one = dict(time=tb, source=[dict(
        identifier=dict(name="nbi", index=2),
        profiles_1d=[dict(time=2.2 + 2e-6, j_parallel=[5.0] * n)])])
    parts, _, _ = _ids_driven_currents(one, 1, n, 1.0, tb)
    np.testing.assert_array_equal(parts["nbi"], 5.0)
    with pytest.raises(EngineInputRefused, match="within half a time-step"):
        _ids_driven_currents(one, 2, n, 1.0, tb)        # 0.1 s past its end
    # 0.1 s BEFORE its only own time: off_before_record (owner decision
    # 2026-10-06; it was refused), announced
    off = []
    with pytest.warns(UserWarning, match="off_before_record"):
        parts, used, _ = _ids_driven_currents(one, 0, n, 1.0, tb, off=off,
                                              announce_key="t-2us")
    assert used == [] and np.all(parts["nbi"] == 0.0)
    assert off[0]["reason"] == "off_before_record"


# ---------------------------------------------------------------------------
#  refinement of the half-step rule (2026-10-06): an entry carrying no
#  current on its own slices BRACKETING the slice time is off there, not
#  missing -- it contributes zero and is stamped (provenance "off_sources"),
#  never refused; one carrying current there is still refused
# ---------------------------------------------------------------------------
def _saw_srcs(own_times, own_j, n, base=(1.0, 2.0, 3.0)):
    return dict(time=list(base), source=[dict(
        identifier=dict(name="sawteeth", index=701),
        profiles_1d=[dict(time=t, j_parallel=[float(j)] * n)
                     for t, j in zip(own_times, own_j)])])


def test_an_entry_starting_a_step_late_and_idle_there_is_off_not_refused():
    """The measured case's geometry: the entry's own grid starts one step
    after the slice time and its first own slice carries no current -- it
    is off at that time: zero contribution, an "off" stamp naming the
    bracketing slice, nothing ignored, no warning, no refusal."""
    import warnings
    from bouquet.adapters import _ids_driven_currents
    n = 5
    srcs = _saw_srcs([2.0, 3.0], [0.0, 2.0], n)
    off = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parts, used, ignored = _ids_driven_currents(srcs, 0, n, 1.0, off=off)
    assert all(np.all(parts[k] == 0.0) for k in parts)
    assert used == [] and ignored == []
    assert [(d["name"], d["index"]) for d in off] == [("sawteeth", 701)]
    assert off[0]["reason"].startswith("off near the slice: no current on "
                                       "its bracketing slices at 2 s")
    # without a collecting list the result is the same (zero, no refusal)
    parts, used, ignored = _ids_driven_currents(srcs, 0, n, 1.0)
    assert used == [] and ignored == [] and np.all(parts["other"] == 0.0)
    # where the entry matches, its current is held as before
    off = []
    parts, used, _ = _ids_driven_currents(srcs, 2, n, 1.0, off=off)
    np.testing.assert_array_equal(parts["other"], 2.0)
    assert off == [] and used[0]["slice"] == "matched by time"


def test_an_entry_starting_a_step_late_with_current_there_is_off_before_record():
    """Same geometry, but the entry's first own slice carries current.
    Owner decision 2026-10-06: the entry has no record before its first own
    time, so it is OFF at the earlier slice -- zero, stamped
    off_before_record with that first own time, announced (it was
    REFUSED).  The mirror case -- the slice a step past the entry's LAST
    own time, which carries current -- is still REFUSED, naming the entry
    and the window; an unknown index alike."""
    from bouquet.adapters import _ids_driven_currents
    n = 5
    srcs = _saw_srcs([2.0, 3.0], [1.0, 0.0], n)
    off = []
    with pytest.warns(UserWarning, match=r"'sawteeth' \(index 701\) has no "
                                         r"record before its first own time "
                                         r"2 s"):
        parts, used, ignored = _ids_driven_currents(
            srcs, 0, n, 1.0, off=off, announce_key="t-late-current")
    assert used == [] and ignored == [] and np.all(parts["other"] == 0.0)
    assert off == [dict(name="sawteeth", index=701,
                        reason="off_before_record", first_own_time=2.0)]
    past = _saw_srcs([0.0, 1.0], [0.0, 1.0], n, base=(1.0, 2.0, 3.0))
    with pytest.raises(EngineInputRefused,
                       match=r"IDS adapter: core_sources 'sawteeth' \(index "
                             r"701\) carries a non-zero j_parallel but has no "
                             r"profiles_1d slice within half a time-step of "
                             r"t = 2 s \(nearest own time 1 s, \|dt\| = 1 s > "
                             r"0\.5 s.*bracketing.*Refusing"):
        _ids_driven_currents(past, 1, n, 1.0)
    past["source"][0]["identifier"] = dict(name="custom_1", index=901)
    with pytest.raises(EngineInputRefused, match="'custom_1'.*Refusing"):
        _ids_driven_currents(past, 1, n, 1.0)


def test_the_bracketing_slices_of_a_time_inside_a_coarse_own_grid():
    """A slice time INSIDE the entry's range, between two own slices of a
    coarser grid: both bracketing slices are judged.  Both zero -> off;
    either non-zero -> carries current (refused by the caller).  Through
    the adapter an interior time always lies within half its own interval
    of one end, so it is matched; this checks the helper the adapter and
    the reader share, and the entry-level verdict, directly."""
    from bouquet.io.imas import _entry_bracketing_slices, _entry_off_near
    times = [0.0, 1.0, 3.0, 4.0]
    assert _entry_bracketing_slices(times, 2.2) == [1, 2]
    assert _entry_bracketing_slices(times, 1.0) == [1]       # on a node
    assert _entry_bracketing_slices(times, -0.5) == [0]      # before range
    assert _entry_bracketing_slices(times, 4.5) == [3]       # past range
    assert _entry_bracketing_slices([3.0, 0.0, 1.0], 2.0) == [0, 2]
    n = 3

    def ent(js):
        return dict(identifier=dict(name="ec", index=3), profiles_1d=[
            dict(time=t, j_parallel=[float(j)] * n)
            for t, j in zip(times, js)])
    why = _entry_off_near(ent([5.0, 0.0, 0.0, 5.0]), 2.2)
    assert why is not None and "bracketing slices at 1, 3 s" in why
    assert _entry_off_near(ent([0.0, 5.0, 0.0, 0.0]), 2.2) is None
    assert _entry_off_near(ent([0.0, 0.0, 5.0, 0.0]), 2.2) is None
    # an absent j_parallel counts as no current; no per-slice time: no
    # verdict (an entry matched by index is not judged here)
    e = ent([0.0, 0.0, 0.0, 0.0])
    e["profiles_1d"][1].pop("j_parallel")
    assert _entry_off_near(e, 2.2) is not None
    e["profiles_1d"][0].pop("time")
    assert _entry_off_near(e, 2.2) is None


def test_an_entry_with_current_only_far_from_the_slice_is_off_not_refused():
    """The measured real-file case: the entry's current is identically zero
    near the slice and non-zero only far in the future (or the past) on its
    own grid.  Before the refinement the whole-history scan refused it; it
    is off at that time."""
    from bouquet.adapters import _ids_driven_currents
    n = 4
    future = _saw_srcs([2.0, 3.0, 4.0, 5.0, 6.0], [0, 0, 0, 0, 7.0], n)
    off = []
    parts, used, ignored = _ids_driven_currents(future, 0, n, 1.0, off=off)
    assert used == [] and ignored == [] and np.all(parts["other"] == 0.0)
    assert len(off) == 1 and "at 2 s" in off[0]["reason"]
    past = _saw_srcs([1.0, 2.0, 3.0, 4.0], [7.0, 0, 0, 0], n,
                     base=(5.0, 6.0, 7.0))
    off = []
    parts, used, ignored = _ids_driven_currents(past, 0, n, 1.0, off=off)
    assert used == [] and ignored == [] and np.all(parts["other"] == 0.0)
    assert len(off) == 1 and "at 4 s" in off[0]["reason"]
    # an aggregate entry in the same position keeps its old stamp
    future["source"][0]["identifier"] = dict(name="total", index=1)
    off = []
    with pytest.warns(UserWarning, match="NOT added"):
        _, used, ignored = _ids_driven_currents(future, 0, n, 1.0, off=off)
    assert off == [] and used == [] and ignored[0]["index"] == 1


def test_an_idle_late_entry_is_stamped_off_in_the_contract(tmp_path):
    """End to end on the example dd: a sawteeth entry whose own grid starts
    one step after the slice time, idle on its first own slice.  The
    contract is the one without it, bit for bit, nothing is warned, and
    provenance["off_sources"] names it."""
    import copy
    import warnings
    dd0 = _example_dd()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c0 = _ids_from(tmp_path, dd0, "base.json").read()
    dd = _example_dd()
    t = [float(x) for x in dd["core_sources"]["time"]]
    assert _TIME == t[-1]
    step = t[-1] - t[-2]
    s = _add_source(dd, "sawteeth", 701,
                    lambda q: 0.002 * np.asarray(q["j_total"]))
    s["profiles_1d"] = copy.deepcopy(s["profiles_1d"][:2])
    s["profiles_1d"][0]["time"] = t[-1] + step
    s["profiles_1d"][1]["time"] = t[-1] + 2.0 * step
    s["profiles_1d"][0]["j_parallel"] = [0.0] * len(
        s["profiles_1d"][0]["j_parallel"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = _ids_from(tmp_path, dd, "off.json").read()
    np.testing.assert_array_equal(c.jB_fix, c0.jB_fix)
    np.testing.assert_array_equal(c.jB_ind, c0.jB_ind)
    assert c.provenance["ignored_sources"] == []
    assert [(d["name"], d["index"]) for d in c.provenance["off_sources"]] \
        == [("sawteeth", 701)]
    assert c0.provenance["off_sources"] == []
    # carrying current on that first own slice: OFF before its record
    # (owner decision 2026-10-06; it was refused) -- the same contract,
    # stamped off_before_record with its first own time
    s["profiles_1d"][0]["j_parallel"] = [1.0e3] * len(
        s["profiles_1d"][0]["j_parallel"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = _ids_from(tmp_path, dd, "on.json").read()
    np.testing.assert_array_equal(c2.jB_fix, c0.jB_fix)
    assert c2.provenance["off_sources"] == [dict(
        name="sawteeth", index=701, reason="off_before_record",
        first_own_time=t[-1] + step)]


def test_a_malformed_driven_source_is_refused(tmp_path):
    dd = _example_dd()
    s = _add_source(dd, "ec", 3, lambda q: 0.01 * np.asarray(q["j_total"]))
    s["profiles_1d"][2]["j_parallel"][5] = float("nan")
    with pytest.raises(EngineInputRefused, match="j_parallel is malformed"):
        _ids_from(tmp_path, dd, "bad_ec.json").read()


def test_the_inductive_tolerance_is_validated(ids):
    ad, c, bl = ids
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="inductive_tol"):
            IdsAdapter(ad.source, ad.config, bl, inductive_tol=bad)


# ---------------------------------------------------------------------------
#  the default: the parallel residual by definition (owner decision
#  2026-10-02); j_ohmic a stamped cross-check
# ---------------------------------------------------------------------------
def test_the_default_is_the_residual_everywhere():
    import inspect
    from bouquet.config import GenerationConfig
    from bouquet.engine import ENGINE_FIELD_DEFAULTS
    assert IDS_INDUCTIVE_DEFAULT == "residual"
    assert inspect.signature(IdsAdapter).parameters["inductive"].default \
        == IDS_INDUCTIVE_DEFAULT
    assert GenerationConfig().engine_ids_inductive == IDS_INDUCTIVE_DEFAULT
    assert ENGINE_FIELD_DEFAULTS["engine_ids_inductive"] == \
        IDS_INDUCTIVE_DEFAULT


def test_residual_stamps_the_j_ohmic_check_without_warning(tmp_path, ids):
    """Under the default, the j_ohmic-vs-residual numbers are computed and
    stamped on a consistent source AND on one whose j_ohmic is scaled by
    1.06 (~5 % net miss), with no warning either way; the inductive is the
    residual, the same in both (j_total, j_bootstrap and the driven parts
    are untouched)."""
    import warnings
    _ad, c_ok, _bl = ids
    dd = _example_dd()
    for q in dd["core_profiles"]["profiles_1d"]:
        q["j_ohmic"] = [1.06 * float(v) for v in q["j_ohmic"]]
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c_ok2 = _ids_from(tmp_path, _example_dd(), "ok.json").read()
        c_off = _ids_from(tmp_path, dd, "ohm_off.json").read()
    k_ok = c_ok2.provenance["inductive_consistency"]
    k_off = c_off.provenance["inductive_consistency"]
    for k in (k_ok, k_off):
        assert k["checked"] is True
        assert k["action"] == "residual_by_definition"
        assert k["tol"] is None
        assert np.isfinite(k["net_frac"]) and np.isfinite(k["rms_frac"])
    assert abs(k_ok["net_frac"]) <= 0.02
    assert k_off["net_frac"] > 0.02 and k_off["rms_frac"] > k_ok["rms_frac"]
    assert k_ok == c_ok.provenance["inductive_consistency"]
    for cc in (c_ok2, c_off):
        assert cc.provenance["inductive"].startswith("residual")
    np.testing.assert_allclose(
        c_off.jB_ind, B0 * (np.asarray(cp["j_total"])
                            - np.asarray(cp["j_bootstrap"])) - c_off.jB_fix,
        rtol=1e-12)
    np.testing.assert_array_equal(c_off.jB_ind, c_ok2.jB_ind)
    np.testing.assert_array_equal(c_ok2.jB_ind, c_ok.jB_ind)


def _adapter_on_full_baseline(ids, tmp_path, dd, name, **kw):
    """The adapter reading a modified dd with the baseline of the full
    example (the reader itself needs every core_profiles current; only the
    adapter's own re-read is under test)."""
    from bouquet.config import ImasSource
    ad, _c, bl = ids
    src = ImasSource(ids_path=str(_write_dd(tmp_path, dd, name)), time=_TIME)
    return IdsAdapter(src, ad.config, bl, **kw)


def test_residual_without_j_ohmic_is_stamped_unchecked(tmp_path, ids):
    import warnings
    dd = _example_dd()
    for q in dd["core_profiles"]["profiles_1d"]:
        del q["j_ohmic"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = _adapter_on_full_baseline(ids, tmp_path, dd, "no_ohm.json").read()
    k = c.provenance["inductive_consistency"]
    assert k["checked"] is False and k["action"] == "unchecked"
    assert k["net_frac"] is None and k["rms_frac"] is None
    assert c.provenance["inductive"].startswith("residual")


@pytest.mark.parametrize("drop", ["j_total", "j_bootstrap"])
def test_residual_refuses_a_source_without_the_residual_and_names_j_ohmic(
        tmp_path, ids, drop):
    """No j_total (or j_bootstrap): the default refuses and names
    inductive='j_ohmic' as the explicit way -- it never falls back to
    j_ohmic silently; explicit "j_ohmic" then reads the source's."""
    dd = _example_dd()
    for q in dd["core_profiles"]["profiles_1d"]:
        del q[drop]
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    ad = _adapter_on_full_baseline(ids, tmp_path, dd, f"no_{drop}.json")
    with pytest.raises(EngineInputRefused) as ei:
        ad.read()
    msg = str(ei.value)
    assert drop in msg and "inductive='j_ohmic'" in msg
    assert "by definition" in msg
    c = _adapter_on_full_baseline(ids, tmp_path, dd, f"no_{drop}.json",
                                  inductive="j_ohmic").read()
    np.testing.assert_allclose(c.jB_ind, B0 * np.asarray(cp["j_ohmic"]),
                               rtol=1e-15)
    assert c.provenance["inductive_consistency"]["action"] == "unchecked"


def test_explicit_auto_and_j_ohmic_keep_their_semantics_on_a_split_miss(
        tmp_path):
    """The 1.06-scaled source: explicit "auto" falls back to the residual
    with its warning; explicit "j_ohmic" forces the source's j_ohmic with
    its warning -- unchanged by the default change."""
    dd = _example_dd()
    for q in dd["core_profiles"]["profiles_1d"]:
        q["j_ohmic"] = [1.06 * float(v) for v in q["j_ohmic"]]
    cp = dd["core_profiles"]["profiles_1d"][2]
    B0 = abs(float(dd["equilibrium"]["vacuum_toroidal_field"]["b0"][2]))
    with pytest.warns(UserWarning, match="does not add up"):
        ca = _ids_from(tmp_path, dd, "ohm_off.json", inductive="auto").read()
    assert ca.provenance["inductive_consistency"]["action"] == \
        "fallback_to_residual"
    assert ca.provenance["inductive"].startswith("residual")
    with pytest.warns(UserWarning, match="kept although"):
        cj = _ids_from(tmp_path, dd, "ohm_off.json",
                       inductive="j_ohmic").read()
    assert cj.provenance["inductive_consistency"]["action"] == \
        "kept_j_ohmic_over_tol"
    assert cj.provenance["inductive"].startswith("j_ohmic")
    np.testing.assert_allclose(cj.jB_ind, B0 * np.asarray(cp["j_ohmic"]),
                               rtol=1e-15)
