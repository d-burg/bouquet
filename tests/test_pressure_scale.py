"""The stored-energy pressure scale ``s_W`` (:mod:`bouquet.pressure_scale`)
and the g-file route's impurity pressure -- fast (no GS solver; the toy
stand-in of ``tests/_engine_fake_gs.py`` / ``tests/_engine_toy.py`` where a
run is needed).

* ``s_W = W_input / W_kinetic`` from the INPUT's own volumes, recovered on a
  synthetic g-file whose ``PRES`` is the kinetic pressure x 1.15, and 1 when
  ``PRES`` IS the kinetic pressure;
* ``"match_W"`` / ``"none"`` / a float; the legacy route and the swb solve
  method refuse an explicit non-default value and accept the default;
* the warning fires on both sides of the band (with the cause), never
  inside it;
* the g-file adapter carries the thermal impurity term at the source's
  ``impurity_Z`` and draws add the same term;
* a draw's pressure is ``s`` times its kinetic assembly; the sigma = 0 draw
  reproduces the reconstruction bit for bit with ``s != 1``;
* the archive stamps ``s`` on the baseline and every draw, and an archive
  written before the feature reads as ``s = 1``.

Synthetic inputs only; no device data.
"""
import os
import sys
import warnings

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_engine_draws as TD  # noqa: E402

from bouquet.adapters import (GFileAdapter, gfile_impurity_charge,  # noqa: E402
                              ids_input_volume)
from bouquet.pressure_scale import (CAUSE_HIGH, CAUSE_LOW,  # noqa: E402
                                    CAUSE_NO_FAST, PRESSURE_SCALE_WARN_BAND,
                                    check_pressure_scale_setting,
                                    pressure_scale_record, shape_residual,
                                    stored_energy, warn_pressure_scale)

toy_bouquet_solver = TD.toy_bouquet_solver
_quiet = TD._quiet

_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")

#: The g-file text format writes each PRES value with 9 significant digits
#: (``%16.9E``): a relative rounding of at most 5e-10 per value.  W is a
#: positive-weighted sum of those values with the SAME quadrature (the
#: reader's volume_integral on the same traced surfaces) for both pressures,
#: so the quadrature error cancels in the ratio and s_W carries at most that
#: rounding.  1e-8 is that bound with a 20x margin.
S_RTOL = 1e-8


def _cfg(geq=_GEQ, **gen):
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=geq, profiles_path=_PF),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


def _read(geq=_GEQ, **gen):
    cfg = _cfg(geq, **gen)
    ad = GFileAdapter(cfg.source, cfg)
    return ad, ad.read()


def _gfile_with_pres(tmp_path, factor, name):
    """A copy of the fixture g-file whose PRES is ``factor`` x the g-file
    adapter's own kinetic pressure (thermal + impurity + fast) on its nodes."""
    from bouquet.io.geqdsk import _read_geqdsk, _write_geqdsk
    _, c = _read(pressure_scale="none")
    g = _read_geqdsk(_GEQ)
    p_kin = np.asarray(c.pressure_parts["kinetic"], dtype=float)
    assert np.shape(g["PRES"]) == p_kin.shape
    g["PRES"] = factor * p_kin
    path = str(tmp_path / name)
    _write_geqdsk(g, path)
    return path


# ---------------------------------------------------------------------------
#  the record
# ---------------------------------------------------------------------------
def _rec(W_in, W_kin, setting="match_W", fast=False):
    return pressure_scale_record(
        setting, W_input=W_in, W_parts=dict(thermal=W_kin, impurity=0.0,
                                            fast=0.0),
        p_kin_on_input=np.array([2.0, 1.0, 0.5]),
        p_input=(W_in / W_kin) * np.array([2.0, 1.0, 0.5]),
        p_fast_supplied=fast)


def test_the_record_modes():
    r = _rec(1.3, 1.0)
    assert r["s_W"] == pytest.approx(1.3) and r["s"] == r["s_W"]
    assert r["mode"] == "match_W" and r["applied"]
    assert r["shape_residual"] == pytest.approx(0.0, abs=1e-15)
    assert r["shape_residual_unscaled"] > 0.2
    n = _rec(1.3, 1.0, "none")
    assert n["s"] == 1.0 and n["s_W"] == pytest.approx(1.3)
    assert not n["applied"] and n["mode"] == "none"
    f = _rec(1.3, 1.0, 1.05)
    assert f["s"] == 1.05 and f["mode"] == "fixed"
    assert f["s_W"] == pytest.approx(1.3)
    # unset reads as the unified default
    assert _rec(1.3, 1.0, None)["mode"] == "match_W"


def test_an_unavailable_s_W_leaves_s_one_and_warns():
    r = pressure_scale_record("match_W", W_input=float("nan"), W_parts={},
                              p_kin_on_input=None, p_input=None,
                              p_fast_supplied=False,
                              unavailable="no volume on the IDS")
    assert r["status"] == "unavailable" and r["s"] == 1.0
    assert not np.isfinite(r["s_W"])
    with pytest.warns(UserWarning, match="could not be computed"):
        warn_pressure_scale(r)
    # a fixed s still applies
    assert pressure_scale_record(
        1.1, W_input=float("nan"), W_parts={}, p_kin_on_input=None,
        p_input=None, p_fast_supplied=False, unavailable="x")["s"] == 1.1


@pytest.mark.parametrize("s_W,fast,cause", [
    (1.25, False, CAUSE_NO_FAST), (1.25, True, CAUSE_HIGH),
    (0.80, False, CAUSE_LOW), (0.80, True, CAUSE_LOW)])
def test_the_warning_fires_outside_the_band_with_its_cause(s_W, fast, cause):
    r = _rec(s_W, 1.0, fast=fast)
    assert r["warn_flag"] and r["cause"] == cause
    with pytest.warns(UserWarning) as w:
        warn_pressure_scale(r)
    assert len(w) == 1 and cause in str(w[0].message)
    if cause == CAUSE_NO_FAST:
        assert "fixed_components.p_fast" in str(w[0].message)


@pytest.mark.parametrize("s_W", [1.0, 1.19, 1.0 / 1.19, 1.2, 1.0 / 1.2])
def test_no_warning_inside_the_band(s_W):
    r = _rec(s_W, 1.0)
    assert not r["warn_flag"] and r["cause"] is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_pressure_scale(r)


def test_the_band_is_the_owners_constant():
    assert PRESSURE_SCALE_WARN_BAND == 1.2


def test_stored_energy_and_shape_residual():
    V = np.linspace(0.0, 2.0, 5)
    assert stored_energy(np.full(5, 4.0), V) == pytest.approx(1.5 * 8.0)
    p = np.array([3.0, 2.0, 1.0])
    assert shape_residual(1.0, p, p) == 0.0
    assert shape_residual(2.0, p, 2.0 * p) == 0.0


@pytest.mark.parametrize("v", ["match_W", "none", 0.9, 1, 1.5, None])
def test_valid_settings(v):
    check_pressure_scale_setting(v)


@pytest.mark.parametrize("v", ["matchW", "", 0.0, -1.0, float("nan"),
                               float("inf"), True, [1.0]])
def test_invalid_settings_are_refused(v):
    with pytest.raises(ValueError, match="pressure_scale"):
        check_pressure_scale_setting(v)


# ---------------------------------------------------------------------------
#  the g-file adapter
# ---------------------------------------------------------------------------
def test_s_is_recovered_on_a_gfile_whose_pressure_is_1p15_x_kinetic(
        tmp_path):
    geq = _gfile_with_pres(tmp_path, 1.15, "g_115.geqdsk")
    ad, c = _read(geq)
    r = c.provenance["pressure_scale"]
    assert r["status"] == "computed" and r["mode"] == "match_W"
    assert r["s_W"] == pytest.approx(1.15, rel=S_RTOL)
    assert r["s"] == r["s_W"]
    kin = c.pressure_parts["kinetic"]
    np.testing.assert_array_equal(c.pressure, r["s"] * kin)
    # a scalar fixes the level and the shape here is the kinetic one
    assert r["shape_residual"] < S_RTOL
    assert r["shape_residual_unscaled"] == pytest.approx(0.15 / 1.15,
                                                         rel=1e-6)
    # the parts stay unscaled, and W is the full pressure's
    assert set(r["W_kinetic_parts_J"]) == {"thermal", "impurity", "fast"}
    assert r["W_kinetic_J"] == pytest.approx(
        sum(r["W_kinetic_parts_J"].values()), rel=1e-14)
    assert r["W_input_J"] == pytest.approx(1.15 * r["W_kinetic_J"],
                                           rel=S_RTOL)
    # 1.15 is inside the band
    assert not r["warn_flag"]


def test_s_is_one_when_the_input_pressure_is_the_kinetic_one(tmp_path):
    geq = _gfile_with_pres(tmp_path, 1.0, "g_100.geqdsk")
    _, c = _read(geq)
    r = c.provenance["pressure_scale"]
    assert r["s_W"] == pytest.approx(1.0, rel=S_RTOL)
    np.testing.assert_allclose(c.pressure, c.pressure_parts["kinetic"],
                               rtol=S_RTOL, atol=0.0)


def test_none_and_float_modes_on_the_adapter():
    _, cn = _read(pressure_scale="none")
    rn = cn.provenance["pressure_scale"]
    assert rn["s"] == 1.0 and np.isfinite(rn["s_W"])
    np.testing.assert_array_equal(cn.pressure, cn.pressure_parts["kinetic"])
    _, cf = _read(pressure_scale=1.3)
    rf = cf.provenance["pressure_scale"]
    assert rf["s"] == 1.3 and rf["s_W"] == rn["s_W"]
    np.testing.assert_array_equal(cf.pressure,
                                  1.3 * cf.pressure_parts["kinetic"])
    # the thermal part (the draws' pressure match) is never scaled
    np.testing.assert_array_equal(cf.pressure_parts["thermal"],
                                  cn.pressure_parts["thermal"])


def test_the_gfile_route_carries_the_impurity_term():
    from bouquet.physics import impurity_pressure
    _, c = _read()
    k = c.kinetics
    assert c.pressure_parts["Z_imp"] == 6.0
    np.testing.assert_array_equal(
        c.pressure_parts["impurity"],
        impurity_pressure(k["ne"], k["ni"], k["ti"], 6.0))
    assert np.all(c.pressure_parts["impurity"] >= 0.0)
    assert c.pressure_parts["impurity"][0] > 0.0
    assert c.pressure_parts["impurity"][0] / c.pressure_parts["thermal"][0] \
        > 1e-3


def test_the_impurity_charge_is_the_sources():
    from types import SimpleNamespace
    assert gfile_impurity_charge(SimpleNamespace(impurity_Z=6.0)) == 6.0
    assert gfile_impurity_charge(SimpleNamespace(impurity_Z=0.0)) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gfile_impurity_charge(SimpleNamespace(impurity_Z=6.0),
                              {"Z_imp_file": 6.0})
    with pytest.warns(UserWarning, match="footer"):
        gfile_impurity_charge(SimpleNamespace(impurity_Z=6.0),
                              {"Z_imp_file": 10.0})


def test_ids_volume_ladder():
    psi = np.linspace(0.0, -1.0, 9)
    q = np.linspace(1.0, 4.0, 9)
    f = np.full(9, 3.0)
    gm1 = np.full(9, 0.4)
    p = np.linspace(5e4, 1e3, 9)
    V = np.linspace(0.0, 20.0, 9)
    x, pp, vv, src, why = ids_input_volume(dict(psi=psi, pressure=p,
                                                volume=V))
    assert why is None and src.endswith(".volume")
    np.testing.assert_array_equal(vv, V)
    x, pp, vv, src, why = ids_input_volume(dict(psi=psi, pressure=p, q=q,
                                                f=f, gm1=gm1))
    assert why is None and "q, f, gm1" in src
    vp = 2.0 * np.pi * q / (3.0 * 0.4)
    assert vv[-1] == pytest.approx(float(np.sum(0.5 * (vp[1:] + vp[:-1])
                                                * 1.0 / 8.0)))
    x, pp, vv, src, why = ids_input_volume(dict(psi=psi, pressure=p))
    assert vv is None and why is not None


# ---------------------------------------------------------------------------
#  the legacy route and swb refuse an explicit value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("v", ["match_W", 1.1])
@pytest.mark.parametrize("route", [dict(reconstruction_engine="legacy"),
                                   dict(solve_method="swb")])
def test_legacy_and_swb_refuse_an_explicit_value(v, route):
    from bouquet.config import GenerationConfig, _EffectiveSolveFields
    from bouquet.engine import validate_engine_settings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            g = GenerationConfig(**route)
        except ValueError:
            pytest.skip("route not constructible standalone")
    g.pressure_scale = v
    with pytest.raises(ValueError, match="pressure_scale"):
        validate_engine_settings(_EffectiveSolveFields(g))


@pytest.mark.parametrize("v", [None, "none"])
def test_legacy_accepts_the_default(v):
    from bouquet.config import GenerationConfig
    from bouquet.engine import validate_engine_settings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = GenerationConfig(reconstruction_engine="legacy")
    g.pressure_scale = v
    validate_engine_settings(g)


def test_legacy_resolves_to_none_and_unified_to_match_W():
    from bouquet.engine import resolve_engine_defaults
    for eng, want in (("legacy", "none"), ("unified", "match_W")):
        cfg = _cfg()
        cfg.generation.reconstruction_engine = eng
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = resolve_engine_defaults(cfg)
        assert cfg.generation.pressure_scale == want
        assert rec["pressure_scale"]["value"] == want


# ---------------------------------------------------------------------------
#  a toy run: draws, sigma = 0, the archive
# ---------------------------------------------------------------------------
def _toy(tmp_path, setting, n=2):
    b = TD._bq(tmp_path, n=n)
    b.config.generation.pressure_scale = setting
    b.setup_solver()
    _quiet(b.prepare_baseline)
    return b


def test_the_baseline_carries_the_record(tmp_path, toy_bouquet_solver):
    b = _toy(tmp_path, 1.3)
    ps = b.baseline.pressure_scale
    assert ps["s"] == 1.3 and ps["mode"] == "fixed"
    assert b.baseline.engine["pressure_scale"]["s"] == 1.3
    c = b._engine_run["engine"].c
    np.testing.assert_array_equal(c.pressure,
                                  1.3 * c.pressure_parts["kinetic"])
    # the report-only fidelity check is on the scaled pressure and keeps
    # the unscaled numbers beside it
    pf = b.baseline.reconstruction_metrics.get("pressure_fidelity") or {}
    assert "rel_rms" in pf, pf
    assert pf["pressure_scale"] == 1.3 and "rel_rms" in pf["unscaled"]


def test_a_draws_pressure_is_s_times_its_kinetic_assembly(
        tmp_path, toy_bouquet_solver):
    from bouquet.engine_draws import context_from_run
    from bouquet.physics import ELEMENTARY_CHARGE as EC, impurity_pressure
    b = _toy(tmp_path, 1.3)
    ctx = context_from_run(b._engine_run, b.config.generation, b.baseline)
    assert ctx.pressure_scale == 1.3
    k0 = {kk: np.asarray(v, dtype=float).copy()
          for kk, v in ctx.c.kinetics.items()}
    k1 = dict(k0)
    k1["te"] = 1.1 * k0["te"]
    k1["ni"] = 0.97 * k0["ni"]
    p1, th1 = ctx.pressures(k1)
    pp = ctx.c.pressure_parts
    want_kin = (EC * (k1["ne"] * k1["te"] + k1["ni"] * k1["ti"])
                + impurity_pressure(k1["ne"], k1["ni"], k1["ti"],
                                    pp["Z_imp"])
                + pp["fast"])
    np.testing.assert_allclose(p1, 1.3 * want_kin, rtol=1e-12, atol=0.0)
    # the pressure match's thermal is unscaled
    np.testing.assert_allclose(
        th1, EC * (k1["ne"] * k1["te"] + k1["ni"] * k1["ti"]), rtol=1e-12)
    # zero perturbation: exactly the contract's pressure
    p0, _ = ctx.pressures(k0)
    np.testing.assert_array_equal(p0, ctx.c.pressure)


@pytest.mark.parametrize("setting", [1.3, "match_W"])
def test_sigma0_passes_with_s_not_one(tmp_path, toy_bouquet_solver, setting):
    b = _toy(tmp_path, setting)
    s = b.baseline.pressure_scale["s"]
    assert s != 1.0
    v = _quiet(b.verify_sigma0_consistency)
    assert v["passed"] and v["request_bit_identical"]


def test_the_archive_round_trip(tmp_path, toy_bouquet_solver):
    import h5py
    from bouquet.archive import BouquetArchive
    b = _toy(tmp_path, 1.3)
    _quiet(b.generate)
    ar = BouquetArchive(b.config.output_header)
    sc = ar.scan(b.config.generation.scan_key) if hasattr(ar, "scan") \
        else ar[b.config.generation.scan_key]
    bp = sc.baseline_pressure_scale
    assert bp["recorded"] and bp["s"] == 1.3 and bp["mode"] == "fixed"
    assert bp["record"]["s"] == 1.3 and np.isfinite(bp["record"]["s_W"])
    draws = list(sc)
    assert draws
    for d in draws:
        ps = d.pressure_scale
        assert ps["recorded"] and ps["s"] == 1.3
    # the archived baseline pressure is the solved (scaled) one
    with h5py.File(b.config.output_header + ".h5", "r") as hf:
        gp = [k for k in hf["scan"]][0]
        p_bl = np.asarray(hf[f"scan/{gp}/_baseline/pressure"])
    np.testing.assert_allclose(
        p_bl, np.asarray(b._engine_run["engine"].c.pressure), rtol=1e-12)


def test_an_archive_without_the_record_reads_as_one(tmp_path):
    import h5py
    from bouquet.schema import read_pressure_scale
    p = tmp_path / "old.h5"
    with h5py.File(p, "w") as hf:
        g = hf.create_group("scan/0/_baseline")
        g.attrs["Ip_target"] = 1.0
    with h5py.File(p, "r") as hf:
        r = read_pressure_scale(hf["scan/0/_baseline"].attrs)
    assert r == dict(s=1.0, mode=None, record=None, recorded=False)


def _golden_scan(path):
    from bouquet.archive import BouquetArchive
    ar = BouquetArchive(path)
    keys = ar.scan_keys if hasattr(ar, "scan_keys") else None
    return ar.scan(keys[0]) if keys else ar[0]


@pytest.mark.parametrize("name", ["D3Dlike_Hmode_golden_slim.h5",
                                  "D3Dlike_ids_golden_slim.h5"])
def test_the_golden_archive_carries_its_scale_and_reads_as_one_without_it(
        name, tmp_path):
    """The h5 goldens are runs of the default (regenerated under the
    feature): the baseline records its match_W scale (s = s_W) and every
    draw carries that same s.  The same archive with the pressure-scale
    attrs stripped (an archive written before the feature) reads as
    s = 1, unrecorded."""
    import shutil

    import h5py
    from bouquet.schema import (PRESSURE_SCALE_ATTR, PRESSURE_SCALE_JSON_ATTR,
                                PRESSURE_SCALE_MODE_ATTR)
    gold = os.path.join(_HERE, "golden", name)
    if not os.path.isfile(gold):
        pytest.skip("no golden archive")
    sc = _golden_scan(gold)
    b = sc.baseline_pressure_scale
    assert b["recorded"] and b["mode"] == "match_W"
    assert b["s"] == b["record"]["s"] == b["record"]["s_W"]
    for d in list(sc)[:2]:
        assert d.pressure_scale["recorded"]
        assert d.pressure_scale["s"] == b["s"]
    old = tmp_path / name
    shutil.copyfile(gold, old)
    with h5py.File(old, "r+") as hf:
        def _strip(_n, obj):
            for k in (PRESSURE_SCALE_ATTR, PRESSURE_SCALE_MODE_ATTR,
                      PRESSURE_SCALE_JSON_ATTR):
                if k in obj.attrs:
                    del obj.attrs[k]
        hf.visititems(_strip)
    sc = _golden_scan(str(old))
    assert sc.baseline_pressure_scale["s"] == 1.0
    assert not sc.baseline_pressure_scale["recorded"]
    for d in list(sc)[:2]:
        assert d.pressure_scale["s"] == 1.0
        assert not d.pressure_scale["recorded"]
