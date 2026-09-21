"""Thermal ni for the bootstrap on the ida_hybrid path.

IDA's ni is a TOTAL deuteron density -- neither the VB Z_eff nor the CER
carbon sees the beam population, so ``ne(Z-Zeff)/(Z-1)`` counts fast ions with
thermal ones.  FUSE's bootstrap (``IMAS.Sauter_neo2021_bootstrap``) is driven
by ``cp1d.pressure_thermal``, i.e. ``density_thermal`` only.  Subtracting the
dd's main-ion ``density_fast`` is what puts the sigma=0 draw on FUSE's footing.

The subtraction is GATED on the IDA ni already reproducing the dd TOTAL ni:
that agreement is the evidence the IDA ni really is the total.  These tests pin
the gate in both directions, the sigma scaling, and the end-to-end wiring.
"""

import json

import h5py
import numpy as np
import pytest

from bouquet.config import ImasSource
from bouquet.io.imas import (NI_FAST_GATE_PSI_N, NI_FAST_RTOL,
                             _subtract_fast_ni, read_imas_baseline)


# ---------------------------------------------------------------------------
# the helper in isolation
# ---------------------------------------------------------------------------
def _profiles(n=32, fast_frac=0.2):
    psi = np.linspace(0.0, 1.0, n)
    ni_total = 4.0e19 * (1.0 - 0.7 * psi ** 2)
    ni_fast = fast_frac * ni_total * (1.0 - psi ** 2)      # core-peaked beam
    return psi, ni_total, ni_fast, ni_total - ni_fast


class TestGate:
    def test_matching_total_fires_and_subtracts(self):
        psi, ni_total, ni_fast, ni_th = _profiles()
        ni, sig, meta = _subtract_fast_ni(psi, ni_total, 0.1 * ni_total,
                                          ni_th, ni_fast)
        assert meta["applied"]
        np.testing.assert_allclose(ni, ni_th, rtol=1e-12)

    def test_a_mismatched_total_refuses_and_says_why(self):
        psi, ni_total, ni_fast, ni_th = _profiles()
        # IDA ni 1% above the dd total: the two are not the same quantity, so
        # subtracting density_fast would correct one disagreement with another.
        ni, sig, meta = _subtract_fast_ni(psi, 1.01 * ni_total, 0.1 * ni_total,
                                          ni_th, ni_fast)
        assert not meta["applied"]
        assert meta["mismatch"] > NI_FAST_RTOL
        assert "not the dd's total" in meta["evidence"]
        np.testing.assert_allclose(ni, 1.01 * ni_total)   # untouched

    def test_the_gate_sits_exactly_at_the_documented_tolerance(self):
        """A uniform relative offset straddling NI_FAST_RTOL flips the gate."""
        psi, ni_total, ni_fast, ni_th = _profiles()
        for mult, applied in ((0.5, True), (2.0, False)):
            scaled = ni_total * (1.0 + mult * NI_FAST_RTOL)
            _, _, meta = _subtract_fast_ni(psi, scaled, None, ni_th, ni_fast)
            assert meta["applied"] is applied, mult
            assert meta["mismatch"] == pytest.approx(mult * NI_FAST_RTOL,
                                                     rel=1e-6)

    def test_the_gate_is_evaluated_at_the_five_declared_points(self):
        psi, ni_total, ni_fast, ni_th = _profiles()
        _, _, meta = _subtract_fast_ni(psi, ni_total, None, ni_th, ni_fast)
        assert tuple(meta["gate"]) == NI_FAST_GATE_PSI_N == (0.0, 0.2, 0.4,
                                                             0.6, 0.8)

    def test_a_disagreement_outside_the_gate_points_is_not_consulted(self):
        """Past 0.8 both profiles roll off; a ratio there is the edge model.

        The gate deliberately does not look there, so an edge-only deviation
        must not veto a subtraction the interior fully supports.
        """
        psi, ni_total, ni_fast, ni_th = _profiles(n=128)
        bad = ni_total.copy()
        bad[psi > 0.9] *= 1.5                     # gross, and entirely outside
        _, _, meta = _subtract_fast_ni(psi, bad, None, ni_th, ni_fast)
        assert meta["applied"]

    def test_a_disagreement_on_axis_alone_is_enough_to_veto(self):
        """psi_N=0 is in the gate: that is where a beam is most peaked."""
        psi, ni_total, ni_fast, ni_th = _profiles(n=128)
        bad = ni_total.copy()
        bad[psi < 0.05] *= 1.0 + 10.0 * NI_FAST_RTOL
        _, _, meta = _subtract_fast_ni(psi, bad, None, ni_th, ni_fast)
        assert not meta["applied"]
        assert meta["gate"][0.0] > NI_FAST_RTOL

    def test_no_density_fast_is_inert(self):
        psi, ni_total, _, _ = _profiles(fast_frac=0.0)
        sig = 0.1 * ni_total
        ni, s, meta = _subtract_fast_ni(psi, ni_total, sig, ni_total,
                                        np.zeros_like(ni_total))
        assert not meta["applied"]
        np.testing.assert_array_equal(ni, ni_total)
        np.testing.assert_array_equal(s, sig)
        assert "no main-ion density_fast" in meta["evidence"]


class TestSigmaScaling:
    def test_the_fractional_error_is_preserved(self):
        psi, ni_total, ni_fast, ni_th = _profiles()
        sig = 0.13 * ni_total
        ni, s, meta = _subtract_fast_ni(psi, ni_total, sig, ni_th, ni_fast)
        assert meta["applied"]
        # The envelope is a measurement error on the deuteron inventory; the
        # fast density removed from the mean carries no IDA error of its own.
        np.testing.assert_allclose(s / ni, sig / ni_total, rtol=1e-12)

    def test_the_envelope_shrinks_with_the_mean_never_grows(self):
        psi, ni_total, ni_fast, ni_th = _profiles()
        sig = 0.13 * ni_total
        _, s, _ = _subtract_fast_ni(psi, ni_total, sig, ni_th, ni_fast)
        assert np.all(s <= sig + 1e-30)
        assert float(np.max(sig - s)) > 0.0          # it actually moved

    def test_a_none_envelope_survives(self):
        psi, ni_total, ni_fast, ni_th = _profiles()
        ni, s, meta = _subtract_fast_ni(psi, ni_total, None, ni_th, ni_fast)
        assert meta["applied"] and s is None


# ---------------------------------------------------------------------------
# end to end through read_imas_baseline
# ---------------------------------------------------------------------------
Z_IMP = 6.0
N = 33


def _ida_cdf(path, fast_frac=0.2):
    """IDA file whose two ni routes agree exactly, on the dd's own psi_N grid.

    Returns the ni it implies, which is the TOTAL deuteron density the dd must
    reproduce as ``density_thermal + density_fast`` for the gate to open.
    """
    psi = np.linspace(0.0, 1.0, N)
    ne = 5.0e19 * (1.0 - 0.7 * psi ** 2)
    nc = 0.015 * ne
    zeff = 1.0 + Z_IMP * (Z_IMP - 1.0) * nc / ne        # exactly consistent
    te = 3.0e3 * (1.0 - 0.9 * psi ** 2) + 50.0
    with h5py.File(path, "w") as f:
        f["time"] = np.array([1000.0])                  # ms; dd slice is 1.0 s
        f["psi_n"] = psi
        for k, v in [("n_e", ne), ("T_e", te), ("T_12C6", 0.9 * te),
                     ("Zeff", zeff), ("n_12C6", nc)]:
            f[k] = np.asarray(v)[None, :]
        for k, v in [("n_e_err", 0.05 * ne), ("T_e_err", 0.04 * te),
                     ("T_12C6_err", 0.06 * te), ("Zeff_err", 0.03 * zeff),
                     ("n_12C6_err", 0.10 * nc)]:
            f[k] = np.asarray(v)[None, :]
    return ne, nc, ne - Z_IMP * nc


def _dd(ni_total, ne, nc, fast_frac=0.2, with_fast_density=True):
    psi = np.linspace(0.0, 1.0, N)
    ni_fast = fast_frac * ni_total * (1.0 - psi ** 2)
    ni_th = ni_total - ni_fast
    ti = 2.5e3 * (1.0 - 0.9 * psi ** 2) + 50.0
    te = ti.copy()
    j_tor = 6.0e5 * (1.0 - psi ** 2)
    p_fast = 3.0e3 * (1.0 - psi ** 2)
    EC = 1.602176634e-19
    p_eq = EC * (ne * te + ni_th * ti + nc * ti) + p_fast

    d_ion = {"density_thermal": ni_th.tolist(), "temperature": ti.tolist(),
             "label": "D", "element": [{"z_n": 1.0, "a": 2.0}],
             "pressure_fast_perpendicular": (p_fast / 3.0).tolist(),
             "pressure_fast_parallel": (p_fast / 3.0).tolist()}
    if with_fast_density:
        d_ion["density_fast"] = ni_fast.tolist()
    return {
        "equilibrium": {
            "time": [1.0],
            "vacuum_toroidal_field": {"r0": 1.7, "b0": [-2.0]},
            "time_slice": [{
                "time": 1.0,
                "global_quantities": {"ip": 1.0e6, "li_1": 1.0, "li_3": 0.9},
                "boundary": {"outline": {"r": [1.2, 2.2, 1.7],
                                         "z": [0.0, 0.0, 0.8]}},
                "profiles_1d": {"psi": psi.tolist(), "pressure": p_eq.tolist(),
                                "j_tor": j_tor.tolist()},
            }],
        },
        "core_profiles": {
            "time": [1.0],
            "profiles_1d": [{
                "grid": {"psi": psi.tolist()},
                "j_tor": j_tor.tolist(), "j_total": j_tor.tolist(),
                "j_ohmic": (0.9 * j_tor).tolist(),
                "j_bootstrap": (0.1 * j_tor).tolist(),
                "electrons": {"density_thermal": ne.tolist(),
                              "temperature": te.tolist()},
                "ion": [d_ion,
                        {"density_thermal": nc.tolist(),
                         "temperature": ti.tolist(), "label": "C12",
                         "element": [{"z_n": 6.0, "a": 12.0}]}],
            }],
        },
    }, ni_fast, ni_th


def _build(tmp_path, fast_frac=0.2, with_fast_density=True, dd_mutate=None):
    cdf = tmp_path / "ida.cdf"
    ne, nc, ni_total = _ida_cdf(str(cdf), fast_frac)
    dd, ni_fast, ni_th = _dd(ni_total, ne, nc, fast_frac, with_fast_density)
    if dd_mutate is not None:
        dd_mutate(dd)
    ddp = tmp_path / "dd.json"
    ddp.write_text(json.dumps(dd))
    return str(ddp), str(cdf), ni_total, ni_fast, ni_th


def _read(ddp, cdf, **kw):
    return read_imas_baseline(
        ImasSource(ids_path=ddp, time=1.0, ida_path=cdf, impurity_Z=Z_IMP, **kw),
        kinetic_source="ida_hybrid")


class TestEndToEnd:
    def test_the_baseline_ni_is_thermal_by_default(self, tmp_path):
        ddp, cdf, ni_total, ni_fast, ni_th = _build(tmp_path)
        bl = _read(ddp, cdf)
        assert bl.aux["ni_fast_meta"]["applied"]
        np.testing.assert_allclose(bl.ni, ni_th, rtol=2e-6)
        # and decisively not the total it started as
        assert float(np.max(np.abs(bl.ni - ni_total))) > 0.1 * float(np.max(ni_fast))

    def test_opting_out_keeps_the_total(self, tmp_path):
        ddp, cdf, ni_total, ni_fast, ni_th = _build(tmp_path)
        bl = _read(ddp, cdf, ni_subtract_fast=False)
        assert not bl.aux["ni_fast_meta"]["applied"]
        np.testing.assert_allclose(bl.ni, ni_total, rtol=2e-6)

    def test_the_ida_envelope_is_scaled_with_it(self, tmp_path):
        ddp, cdf, *_ = _build(tmp_path)
        on = _read(ddp, cdf)
        off = _read(ddp, cdf, ni_subtract_fast=False)
        np.testing.assert_allclose(on.aux["sigma_ni_ida"] / on.ni,
                                   off.aux["sigma_ni_ida"] / off.ni, rtol=1e-9)
        assert float(np.max(off.aux["sigma_ni_ida"]
                            - on.aux["sigma_ni_ida"])) > 0.0

    def test_a_dd_without_density_fast_is_untouched(self, tmp_path):
        ddp, cdf, ni_total, _, _ = _build(tmp_path, with_fast_density=False)
        bl = _read(ddp, cdf)
        assert not bl.aux["ni_fast_meta"]["applied"]
        np.testing.assert_allclose(bl.ni, ni_total, rtol=2e-6)

    def test_a_dd_whose_total_disagrees_is_left_alone(self, tmp_path):
        def bump(dd):
            ion = dd["core_profiles"]["profiles_1d"][0]["ion"][0]
            ion["density_thermal"] = (np.asarray(ion["density_thermal"])
                                      * (1.0 + 10.0 * NI_FAST_RTOL)).tolist()
        ddp, cdf, ni_total, _, _ = _build(tmp_path, dd_mutate=bump)
        bl = _read(ddp, cdf)
        assert not bl.aux["ni_fast_meta"]["applied"]
        assert bl.aux["ni_fast_meta"]["mismatch"] > NI_FAST_RTOL
        np.testing.assert_allclose(bl.ni, ni_total, rtol=2e-6)

    def test_zero_fast_fraction_reproduces_the_opted_out_baseline(self, tmp_path):
        """The no-beam case must not depend on the flag at all."""
        ddp, cdf, *_ = _build(tmp_path, fast_frac=0.0)
        on, off = _read(ddp, cdf), _read(ddp, cdf, ni_subtract_fast=False)
        np.testing.assert_array_equal(on.ni, off.ni)
        np.testing.assert_array_equal(on.aux["sigma_ni_ida"],
                                      off.aux["sigma_ni_ida"])


class TestQuasineutralityIsPreserved:
    def test_the_impurity_density_the_pressure_uses_stays_the_measured_carbon(
            self, tmp_path):
        """nz = (ne - z_fast - ni)/Z must recover the CER carbon.

        With ni TOTAL this is short by z_fast/Z; the subtraction is what makes
        the archived (ne, ni, z_fast, Z_imp) set mutually quasineutral, which
        is the invariant p_imp and the p-file export assume.
        """
        ddp, cdf, ni_total, ni_fast, ni_th = _build(tmp_path)
        bl = _read(ddp, cdf)
        ne = np.asarray(bl.ne, dtype=float)
        z_fast = np.asarray(bl.z_fast, dtype=float)
        nz = (ne - z_fast - np.asarray(bl.ni, dtype=float)) / Z_IMP
        nc_expected = (ne - ni_total) / Z_IMP          # the CER carbon
        np.testing.assert_allclose(nz, nc_expected, rtol=1e-5)

    def test_leaving_ni_total_breaks_that_invariant(self, tmp_path):
        ddp, cdf, ni_total, ni_fast, _ = _build(tmp_path)
        bl = _read(ddp, cdf, ni_subtract_fast=False)
        ne = np.asarray(bl.ne, dtype=float)
        nz = (ne - np.asarray(bl.z_fast, dtype=float)
              - np.asarray(bl.ni, dtype=float)) / Z_IMP
        nc_expected = (ne - ni_total) / Z_IMP
        assert not np.allclose(nz, nc_expected, rtol=1e-3)
