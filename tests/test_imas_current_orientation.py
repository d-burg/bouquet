"""The IMAS reader refuses a mixed-orientation source; the override is explicit.

A dd whose current profiles and plasma current were written in DIFFERENT
orientations (e.g. an IDS conversion that mixed COCOS between the equilibrium
and core_profiles IDSs) cannot be put into bouquet's positive-Ip frame by
``sign(ip)``: multiplying by it would flip currents that were already co-Ip,
and the recomputed (positive) bootstrap would be added against them.  The
reader therefore

  * REFUSES (``ValueError``, naming each quantity and its sign) when the net,
    area-weighted ``core_profiles.j_tor`` -- or the equilibrium ``j_tor`` the
    jphi anchor uses -- disagrees in sign with the orientation factor;
  * lets a user who knows the file's convention name the factor with
    ``ImasSource.current_orientation = +1 / -1`` (default ``"auto"`` =
    ``sign(ip)``), and records the factor and its origin;
  * leaves every consistent source exactly as it was, bit for bit.

Synthetic dds only (the shipped D3D-like example and dds built in-process).
"""
import json

import numpy as np
import pytest
from scipy.integrate import trapezoid

import _mirror_dd as mdd
from bouquet.config import BouquetConfig, ImasSource, SolverConfig
from bouquet.io.imas import (ORIENTATION_ORIGIN_AUTO,
                             ORIENTATION_ORIGIN_OVERRIDE,
                             parse_current_orientation, read_imas_baseline)
from test_imas_p_fast_convention import _minimal_dd
from test_reversed_ip_current_sign import (_assert_same_baseline, _example,
                                           _read, _write)


def _neg_eq_ip(dd, also_eq_jtor=False):
    """Reverse ONLY the equilibrium IDS's ip (and optionally its j_tor): the
    core_profiles currents stay as stored -- a mixed-orientation dd."""
    for ts in dd["equilibrium"]["time_slice"]:
        ts["global_quantities"]["ip"] = -float(ts["global_quantities"]["ip"])
        if also_eq_jtor:
            ts["profiles_1d"]["j_tor"] = [-v for v in ts["profiles_1d"]["j_tor"]]
    return dd


def _src(path, time=mdd.EXAMPLE_TIME, **kw):
    return ImasSource(ids_path=path, time=time, **kw)


# ---------------------------------------------------------------------------
#  1. a mixed-sign source is refused, and the refusal says why
# ---------------------------------------------------------------------------
class TestMixedSignSourceIsRefused:
    def test_ip_reversed_but_currents_not(self, tmp_path):
        """The review's scenario: equilibrium ip < 0 while every current profile
        is positive.  sign(ip) = -1 would flip co-Ip currents counter-Ip."""
        p = _write(tmp_path, _neg_eq_ip(_example()), "mixed.json")
        with pytest.raises(ValueError) as ei:
            read_imas_baseline(_src(p))
        msg = str(ei.value)
        assert "disagree in sign with its plasma current" in msg
        assert "equilibrium ip = -1.23459e+06 A (sign -)" in msg
        # both toroidal currents in use are named, with their stored sign
        assert "core_profiles.j_tor: net +" in msg and "(sign +;" in msg
        assert "equilibrium.profiles_1d.j_tor: net +" in msg
        assert "current_orientation" in msg          # the way out is named

    def test_equilibrium_ids_reversed_core_profiles_not(self, tmp_path):
        """ip AND equilibrium j_tor reversed (a self-consistent equilibrium
        IDS), core_profiles not: only core_profiles.j_tor is named."""
        p = _write(tmp_path, _neg_eq_ip(_example(), also_eq_jtor=True), "m.json")
        with pytest.raises(ValueError) as ei:
            read_imas_baseline(_src(p))
        msg = str(ei.value)
        assert "core_profiles.j_tor: net +" in msg
        assert "equilibrium.profiles_1d.j_tor" not in msg

    def test_unused_equilibrium_jtor_is_not_judged(self, tmp_path):
        """With anchor_jtor_to_equilibrium=False the equilibrium j_tor is not
        used, so only core_profiles.j_tor must agree with the factor."""
        dd = _example()
        for ts in dd["equilibrium"]["time_slice"]:
            ts["profiles_1d"]["j_tor"] = [-v for v in ts["profiles_1d"]["j_tor"]]
        p = _write(tmp_path, dd, "eqonly.json")
        with pytest.raises(ValueError, match="equilibrium.profiles_1d.j_tor"):
            read_imas_baseline(_src(p))
        ref = _read(_write(tmp_path, _example(), "ref.json"),
                    anchor_jtor_to_equilibrium=False)
        bl = _read(p, anchor_jtor_to_equilibrium=False)
        _assert_same_baseline(ref, bl, "eq j_tor unused")


# ---------------------------------------------------------------------------
#  2. the override: explicit, validated, recorded
# ---------------------------------------------------------------------------
class TestOrientationOverride:
    def test_plus_one_keeps_the_stored_currents(self, tmp_path):
        """ip < 0 but currents co-Ip: +1 reads exactly what the un-reversed
        file reads, and says it was an override."""
        ref = _read(_write(tmp_path, _example(), "ref.json"))
        p = _write(tmp_path, _neg_eq_ip(_example()), "mixed.json")
        bl = read_imas_baseline(_src(p, current_orientation=+1))
        _assert_same_baseline(ref, bl, "override +1")
        assert bl.source_current_sign == 1.0
        assert bl.source_current_sign_origin == ORIENTATION_ORIGIN_OVERRIDE
        assert ref.source_current_sign_origin == ORIENTATION_ORIGIN_AUTO

    def test_minus_one_on_a_reversed_source_equals_auto(self, tmp_path):
        p = _write(tmp_path, mdd.mirror_dd(_example(), -1.0, 1.0), "rev.json")
        auto = _read(p)
        forced = read_imas_baseline(_src(p, current_orientation=-1))
        _assert_same_baseline(auto, forced, "override -1")
        assert forced.source_current_sign == auto.source_current_sign == -1.0
        assert forced.source_current_sign_origin == ORIENTATION_ORIGIN_OVERRIDE
        assert auto.source_current_sign_origin == ORIENTATION_ORIGIN_AUTO

    def test_minus_one_reverses_the_stored_currents(self, tmp_path):
        """-1 on a file whose ip is positive but whose currents are all
        stored reversed: the read equals the consistent file's."""
        dd = _example()
        for c in dd["core_profiles"]["profiles_1d"]:
            for k in ("j_tor", "j_total", "j_ohmic", "j_bootstrap",
                      "j_non_inductive"):
                c[k] = [-v for v in c[k]]
        for s in dd["core_sources"]["source"]:
            for pr in s["profiles_1d"]:
                pr["j_parallel"] = [-v for v in pr["j_parallel"]]
        for ts in dd["equilibrium"]["time_slice"]:
            ts["profiles_1d"]["j_tor"] = [-v for v in ts["profiles_1d"]["j_tor"]]
        p = _write(tmp_path, dd, "cur_rev.json")
        with pytest.raises(ValueError, match="disagree in sign"):
            read_imas_baseline(_src(p))                   # auto refuses
        ref = _read(_write(tmp_path, _example(), "ref.json"))
        bl = read_imas_baseline(_src(p, current_orientation=-1))
        _assert_same_baseline(ref, bl, "override -1 on reversed-current file")
        assert bl.source_current_sign == -1.0

    def test_an_override_that_leaves_currents_counter_ip_is_refused(self,
                                                                     tmp_path):
        p = _write(tmp_path, _example(), "ref.json")
        with pytest.raises(ValueError) as ei:
            read_imas_baseline(_src(p, current_orientation=-1))
        assert "set by ImasSource.current_orientation" in str(ei.value)

    def test_override_is_logged(self, tmp_path, capsys):
        p = _write(tmp_path, _neg_eq_ip(_example()), "mixed.json")
        read_imas_baseline(_src(p, current_orientation=1))
        assert "current_orientation = +1 (override)" in capsys.readouterr().out

    @pytest.mark.parametrize("good,val", [("auto", "auto"), ("AUTO", "auto"),
                                          (1, 1.0), (-1, -1.0), (1.0, 1.0),
                                          (-1.0, -1.0), ("+1", 1.0),
                                          ("-1", -1.0)])
    def test_accepted_spellings(self, good, val):
        assert parse_current_orientation(good) == val

    @pytest.mark.parametrize("bad", [0, 2, -0.5, True, False, None, "rev",
                                     float("nan"), ""])
    def test_rejected_values(self, bad):
        with pytest.raises(ValueError, match="current_orientation"):
            parse_current_orientation(bad)

    def test_config_validates_early_and_round_trips(self, tmp_path):
        p = _write(tmp_path, _example(), "ref.json")
        with pytest.raises(ValueError, match="current_orientation"):
            BouquetConfig(source=ImasSource(ids_path=p, current_orientation=0),
                          solver=SolverConfig(mesh_path="m.h5"),
                          output_header="x")
        cfg = BouquetConfig(source=ImasSource(ids_path=p, current_orientation=-1),
                            solver=SolverConfig(mesh_path="m.h5"),
                            output_header="x")
        back = BouquetConfig.from_json(cfg.to_json())
        assert back.source.current_orientation == -1
        assert ImasSource(ids_path=p).current_orientation == "auto"

    def test_archive_stamp_records_the_origin(self, tmp_path):
        import h5py
        from bouquet.utils import stamp_source_orientation

        p = str(tmp_path / "a.h5")
        with h5py.File(p, "w") as hf:
            hf.create_group("scan/0/_baseline")
        stamp_source_orientation(p, scan_key=0, current_sign=1.0, b0_sign=-1.0,
                                 current_sign_origin=ORIENTATION_ORIGIN_OVERRIDE)
        with h5py.File(p, "r") as hf:
            a = hf["scan/0/_baseline"].attrs
            assert a["source_current_sign"] == 1.0
            assert a["source_current_sign_origin"] == ORIENTATION_ORIGIN_OVERRIDE


# ---------------------------------------------------------------------------
#  3. the check weighs the NET current by area when the file has it
# ---------------------------------------------------------------------------
def _hollow_dd(with_area):
    """ip > 0; j_tor NEGATIVE inside psi_N < 0.4 and positive outside.

    Unweighted in psi_N the profile integrates positive (-0.4 + 0.6); by
    area -- here area ~ sqrt(psi_N), i.e. most of the cross-section at small
    psi_N -- it integrates NEGATIVE: the net current opposes ip.
    """
    dd = _minimal_dd(n=41)
    cp = dd["core_profiles"]["profiles_1d"][0]
    psi = np.asarray(cp["grid"]["psi"], float)
    j = np.where(psi < 0.4, -1.0e5, 1.0e5)
    for k in ("j_tor", "j_total"):
        cp[k] = j.tolist()
    cp["j_bootstrap"] = np.zeros_like(j).tolist()
    cp["j_ohmic"] = j.tolist()
    dd["equilibrium"]["time_slice"][0]["profiles_1d"]["j_tor"] = [1.0e5] * psi.size
    if with_area:
        cp["grid"]["area"] = (0.8 * np.sqrt(psi)).tolist()
    return dd


class TestAreaWeighting:
    def test_psi_N_and_area_weighting_disagree_on_this_profile(self):
        dd = _hollow_dd(True)
        cp = dd["core_profiles"]["profiles_1d"][0]
        psi = np.asarray(cp["grid"]["psi"], float)
        j = np.asarray(cp["j_tor"], float)
        assert trapezoid(j, psi) > 0.0
        assert trapezoid(j, np.asarray(cp["grid"]["area"], float)) < 0.0

    def test_area_on_file_is_used(self, tmp_path):
        p = _write(tmp_path, _hollow_dd(True), "hollow_area.json")
        with pytest.raises(ValueError) as ei:
            read_imas_baseline(ImasSource(ids_path=p, time=1.0),
                               p_fast_reduction="trace")
        assert "core_profiles.j_tor: net -" in str(ei.value)
        assert "area-weighted (own grid area)" in str(ei.value)

    def test_equilibrium_area_is_used_when_the_grid_has_none(self, tmp_path):
        dd = _hollow_dd(False)
        p1 = dd["equilibrium"]["time_slice"][0]["profiles_1d"]
        p1["area"] = (0.8 * np.sqrt(np.asarray(p1["psi"], float))).tolist()
        p = _write(tmp_path, dd, "hollow_eqarea.json")
        with pytest.raises(ValueError,
                           match=r"area-weighted \(equilibrium profiles_1d.area\)"):
            read_imas_baseline(ImasSource(ids_path=p, time=1.0),
                               p_fast_reduction="trace")

    def test_without_geometry_the_psi_N_fallback_is_used(self, tmp_path):
        """No area and no rho_tor_norm on file: the check falls back to psi_N,
        in which this profile's net current is co-Ip -- it is read."""
        p = _write(tmp_path, _hollow_dd(False), "hollow.json")
        bl = _read(p, time=1.0, p_fast_reduction="trace")
        assert bl.source_current_sign == 1.0

    def test_rho_tor_norm_proxy(self, tmp_path):
        """rho_tor_norm^2 is the area proxy when no area is on file."""
        dd = _hollow_dd(False)
        cp = dd["core_profiles"]["profiles_1d"][0]
        psi = np.asarray(cp["grid"]["psi"], float)
        cp["grid"]["rho_tor_norm"] = (psi ** 0.25).tolist()   # rho^2 = sqrt(psi)
        p = _write(tmp_path, dd, "hollow_rho.json")
        with pytest.raises(ValueError, match=r"rho_tor_norm\^2-weighted"):
            read_imas_baseline(ImasSource(ids_path=p, time=1.0),
                               p_fast_reduction="trace")

    def test_boundary_first_storage_does_not_flip_the_sign(self):
        from bouquet.io.imas import _net_current
        x = np.linspace(0.0, 1.0, 11)
        j = 1.0 + x
        assert _net_current(j, x) == pytest.approx(_net_current(j[::-1], x[::-1]))
        assert _net_current(j[::-1], x[::-1]) > 0.0


# ---------------------------------------------------------------------------
#  4. B0 = 0 carries no orientation; the sign comes from the currents' slice
# ---------------------------------------------------------------------------
class TestRecordedSigns:
    def test_zero_b0_records_none(self, tmp_path):
        dd = _example()
        dd["equilibrium"]["vacuum_toroidal_field"]["b0"] = [0.0, 0.0, 0.0]
        bl = _read(_write(tmp_path, dd, "b0zero.json"))
        assert bl.source_b0_sign is None

    def test_sign_is_taken_at_the_slice_of_the_currents(self, tmp_path):
        """A single-slice core_profiles at t=2.2 against a three-slice
        equilibrium, requested at t=2.3043: the currents are the t=2.2 ones,
        so the orientation must be read at the t=2.2 equilibrium slice (here
        reversed only there) -- not at the requested one."""
        dd = mdd.mirror_dd(_example(), -1.0, 1.0)
        cpi = dd["core_profiles"]
        cpi["time"] = [2.2]
        cpi["profiles_1d"] = [cpi["profiles_1d"][1]]
        # the requested slice (index 2) says ip > 0; the currents' slice says < 0
        dd["equilibrium"]["time_slice"][2]["global_quantities"]["ip"] = abs(
            float(dd["equilibrium"]["time_slice"][2]["global_quantities"]["ip"]))
        dd["equilibrium"]["time_slice"][2]["profiles_1d"]["j_tor"] = [
            abs(v) for v in dd["equilibrium"]["time_slice"][2]["profiles_1d"]["j_tor"]]
        bl = _read(_write(tmp_path, dd, "slices.json"),
                   anchor_jtor_to_equilibrium=False)
        assert bl.source_current_sign == -1.0
        assert trapezoid(bl.j_phi, bl.psi_N) > 0.0
