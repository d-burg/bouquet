"""The IDS l_i row's target radius (``GenerationConfig.imas_li3_radius``).

The IMAS data dictionary gives ``equilibrium...global_quantities.li_3`` no
normalisation radius; IMAS.jl (and so FUSE) writes it with the boundary's
geometric radius ``R_geo = (R_out + R_in)/2``, while the engine's
measurement (``utils.li_achieved``) normalises by the magnetic axis.  On a
shifted axis the two differ by a few per cent -- 3 % here, a D3D-like
``R_geo / R_axis`` -- so the row would pull toward the wrong l_i.  "auto"
recomputes li_3 from the source's own equilibrium with each radius, takes
the one that reproduces the stored value within 0.5 %, and rescales the
target to the measurement's radius by ``R_src / R_axis``, recording the
choice, the ratios and the factor; neither -> refused (never silently
rescaled).

Synthetic equilibria only; no solver, no device data.
"""
import copy
import json
import os
import sys
import warnings

import numpy as np
import pytest

from bouquet.adapters import (LI3_RADIUS_MATCH_TOL, Li3RadiusRefused,
                              resolve_li3_radius)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

MU0 = 4.0e-7 * np.pi
IP = 1.2e6
R_AXIS = 1.75                     # the magnetic axis, shifted outboard
R_OUT, R_IN = 2.30, 1.0925        # R_geo = 1.69625: R_geo / R_axis = 0.969


def _slice(li3_radius, psi=None):
    """A synthetic IDS equilibrium time slice whose flux-surface averages
    give a known int B_p^2 dV, with li_3 written at *li3_radius* (or None
    for no li_3); on the grid *psi* when given."""
    psi = (np.linspace(0.0, 1.6, 41) if psi is None       # Wb (COCOS 11)
           else np.asarray(psi, dtype=float))
    n = psi.size
    rho = np.sqrt(np.linspace(0.0, 1.0, n)) * 0.8
    dpdr = 4.0 * (rho + 0.05)
    gm2 = 0.30 + 0.10 * rho
    dv = 12.0 + 20.0 * psi
    from scipy.integrate import trapezoid
    bp2v = abs(float(trapezoid(dpdr ** 2 * gm2 / (2 * np.pi) ** 2 * dv,
                               psi)))
    gq = dict(ip=IP, magnetic_axis=dict(r=R_AXIS, z=0.0))
    if li3_radius is not None:
        gq["li_3"] = 2.0 * bp2v / ((MU0 * IP) ** 2 * li3_radius)
    ts = dict(global_quantities=gq, profiles_1d=dict(
        psi=psi.tolist(), gm2=gm2.tolist(), dpsi_drho_tor=dpdr.tolist(),
        dvolume_dpsi=dv.tolist(),
        r_outboard=np.linspace(R_AXIS, R_OUT, n).tolist(),
        r_inboard=np.linspace(R_AXIS, R_IN, n).tolist()))
    return ts, bp2v


def _li3(bp2v, R):
    return 2.0 * bp2v / ((MU0 * IP) ** 2 * R)


def test_a_geometric_li3_is_detected_rescaled_and_recorded():
    R_geo = 0.5 * (R_OUT + R_IN)
    ts, bp2v = _slice(R_geo)
    r = resolve_li3_radius(ts, "auto")
    assert R_geo / R_AXIS == pytest.approx(0.969, abs=1e-3)   # ~3 % bias
    assert r["choice"] == "geometric" and r["status"] == "matched"
    assert r["ratio_geometric"] == pytest.approx(1.0, rel=1e-12)
    assert r["ratio_axis"] == pytest.approx(R_AXIS / R_geo, rel=1e-12)
    assert abs(r["ratio_axis"] - 1.0) > LI3_RADIUS_MATCH_TOL
    assert r["factor"] == pytest.approx(R_geo / R_AXIS, rel=1e-12)
    # the target is li_3 at the MEASUREMENT's radius
    assert r["target"] == pytest.approx(_li3(bp2v, R_AXIS), rel=1e-12)
    assert r["target"] < r["li3_source"]
    assert r["R_geo_source"].startswith("profiles_1d r_outboard")
    json.dumps(r)


def test_an_axis_li3_keeps_factor_one():
    ts, bp2v = _slice(R_AXIS)
    r = resolve_li3_radius(ts, "auto")
    assert r["choice"] == "axis" and r["status"] == "matched"
    assert r["factor"] == 1.0
    assert r["target"] == r["li3_source"]


def test_a_li3_matching_neither_radius_is_refused_naming_both():
    ts, bp2v = _slice(R_AXIS)
    ts["global_quantities"]["li_3"] *= 1.02          # neither within 0.5 %
    with pytest.raises(Li3RadiusRefused) as ei:
        resolve_li3_radius(ts, "auto")
    msg = str(ei.value)
    assert "R_axis" in msg and "R_geo" in msg and "never rescaled" in msg


def test_a_stated_radius_is_applied_and_geometric_needs_both_radii():
    R_geo = 0.5 * (R_OUT + R_IN)
    ts, _ = _slice(R_AXIS)
    r = resolve_li3_radius(ts, "geometric")
    assert r["status"] == "stated"
    assert r["factor"] == pytest.approx(R_geo / R_AXIS)
    assert resolve_li3_radius(ts, "axis")["factor"] == 1.0
    del ts["global_quantities"]["magnetic_axis"]
    with pytest.raises(Li3RadiusRefused):
        resolve_li3_radius(ts, "geometric")


def test_an_unrecomputable_source_is_undetermined_and_recorded(capsys):
    """Without the flux-surface averages li_3 cannot be recomputed: the
    target is the stored li_3 UNRESCALED (what the row used before the
    setting existed), printed and recorded 'undetermined' with what is
    missing."""
    ts, _ = _slice(R_AXIS)
    del ts["profiles_1d"]["gm2"]
    r = resolve_li3_radius(ts, "auto")
    assert r["status"] == "undetermined" and r["factor"] == 1.0
    assert r["target"] == r["li3_source"]
    assert r["missing"] == ["profiles_1d gm2"]
    assert "cannot tell the radius" in capsys.readouterr().out


def test_a_stored_imas_unified_config_replays_with_axis():
    """An IMAS unified config stored before the setting ran its row on the
    unrescaled li_3: it loads with 'axis' (warned); a g-file one is not
    concerned (default, silent -- a non-default would be refused there)."""
    import test_engine_adapters as TA
    from bouquet.config import BouquetConfig
    for cfg, imas in ((TA._icfg(), True), (TA._gcfg(), False)):
        d = cfg.to_dict()
        assert d["generation"]["imas_li3_radius"] == "auto"
        del d["generation"]["imas_li3_radius"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = BouquetConfig.from_dict(d).generation
        hit = [x for x in w if "imas_li3_radius" in str(x.message)]
        assert g.imas_li3_radius == ("axis" if imas else "auto")
        assert bool(hit) is imas


def _ids_adapter(tmp_path, dd, setting):
    from bouquet.adapters import IdsAdapter
    from bouquet.baseline import resolve_baseline
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    import test_engine_adapters as TA
    p = TA._write_dd(tmp_path, dd, f"dd_{setting}.json")
    cfg = BouquetConfig(
        source=ImasSource(ids_path=str(p), time=TA._TIME),
        solver=SolverConfig(mesh_path=TA._MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified",
                                    imas_li3_radius=setting))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
        return IdsAdapter(cfg.source, cfg, bl).read()


def test_the_adapter_rescales_the_row_target_and_records_it(tmp_path):
    """Through IdsAdapter.read on the example IDS with a synthetic,
    R_geo-written li_3 equilibrium slice: the l_i row's target is li_3 at
    the axis, the factor and the choice are on the row and in the
    contract's provenance (the engine record)."""
    import test_engine_adapters as TA
    with open(TA._OMAS) as fh:
        dd = json.load(fh)
    R_geo = 0.5 * (R_OUT + R_IN)
    for ts in dd["equilibrium"]["time_slice"]:
        # the averages tabulated on the example's own psi grid
        syn, _ = _slice(R_geo, psi=ts["profiles_1d"]["psi"])
        ts["global_quantities"].update(syn["global_quantities"])
        ts["profiles_1d"].update({k: v for k, v in syn["profiles_1d"].items()
                                  if k != "psi"})
    c = _ids_adapter(tmp_path, copy.deepcopy(dd), "auto")
    row = c.rows["l_i"]
    rec = c.provenance["li3_radius"]
    assert rec["choice"] == "geometric" and rec["status"] == "matched"
    assert row["radius_factor"] == pytest.approx(R_geo / R_AXIS)
    assert row["target"] == pytest.approx(row["li3_source"] * R_geo / R_AXIS)
    c_ax = _ids_adapter(tmp_path, copy.deepcopy(dd), "axis")
    assert c_ax.rows["l_i"]["target"] == pytest.approx(
        c_ax.rows["l_i"]["li3_source"])
    assert c_ax.provenance["li3_radius"]["status"] == "stated"


def test_the_shipped_example_is_undetermined_and_unchanged(tmp_path):
    """The shipped example IDS carries no flux-surface averages: 'auto'
    leaves its l_i target exactly as before (factor 1), recorded."""
    import test_engine_adapters as TA
    with open(TA._OMAS) as fh:
        dd = json.load(fh)
    c = _ids_adapter(tmp_path, dd, "auto")
    rec = c.provenance["li3_radius"]
    assert rec["status"] == "undetermined" and rec["factor"] == 1.0
    assert c.rows["l_i"]["target"] == rec["li3_source"]


def test_the_setting_is_validated_and_refused_on_a_gfile_source():
    from bouquet.config import GenerationConfig
    from bouquet.engine import ENGINE_FIELD_DEFAULTS, validate_engine_settings
    assert ENGINE_FIELD_DEFAULTS["imas_li3_radius"] == "auto"
    assert GenerationConfig().imas_li3_radius == "auto"
    gc = GenerationConfig(reconstruction_engine="unified",
                          imas_li3_radius="bogus")
    with pytest.raises(ValueError, match="imas_li3_radius"):
        validate_engine_settings(gc)
    import bouquet as bq
    from bouquet.engine import prepare_engine_baseline
    ex = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
    b = bq.Bouquet.from_geqdsk(
        os.path.join(ex, "D3Dlike_Hmode_baseline.geqdsk"),
        profiles=os.path.join(ex, "D3Dlike_Hmode_baseline.peqdsk"),
        mesh=os.path.join(ex, "DIIID_mesh.h5"), n_draws=1,
        reconstruction_engine="unified")
    b.config.generation.imas_li3_radius = "geometric"
    b.mygs = object()
    with pytest.raises(ValueError, match="imas_li3_radius.*g-file"):
        prepare_engine_baseline(b)
