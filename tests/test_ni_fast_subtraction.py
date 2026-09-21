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
def _profiles(n=32, fast_frac=0.2, Z_beam=1.0, Z_imp=6.0):
    """Returns (psi, ni_total, ni_fast_equivalent, ni_thermal, z_fast, z2_fast).

    ``ni_fast_equivalent`` is what a MEASURED ni carries of the beam --
    n_fast Z_b(Z_imp - Z_b)/(Z_imp - 1) -- not the bare particle density.
    """
    psi = np.linspace(0.0, 1.0, n)
    ni_total = 4.0e19 * (1.0 - 0.7 * psi ** 2)
    n_fast = fast_frac * ni_total * (1.0 - psi ** 2)       # core-peaked beam
    z_fast = Z_beam * n_fast
    z2_fast = Z_beam ** 2 * n_fast
    ni_fast = n_fast * Z_beam * (Z_imp - Z_beam) / (Z_imp - 1.0)
    return psi, ni_total, ni_fast, ni_total - ni_fast, z_fast, z2_fast


class TestGate:
    def test_matching_total_fires_and_subtracts(self):
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        ni, sig, meta = _subtract_fast_ni(psi, ni_total, 0.1 * ni_total,
                                          ni_th, zf, z2, 6.0)
        assert meta["applied"]
        np.testing.assert_allclose(ni, ni_th, rtol=1e-12)

    def test_a_mismatched_total_refuses_and_says_why(self):
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        # IDA ni 1% above the dd total: the two are not the same quantity, so
        # subtracting density_fast would correct one disagreement with another.
        ni, sig, meta = _subtract_fast_ni(psi, 1.01 * ni_total, 0.1 * ni_total,
                                          ni_th, zf, z2, 6.0)
        assert not meta["applied"]
        assert meta["mismatch"] > NI_FAST_RTOL
        assert "not the dd's total" in meta["evidence"]
        np.testing.assert_allclose(ni, 1.01 * ni_total)   # untouched

    def test_the_gate_sits_exactly_at_the_documented_tolerance(self):
        """A uniform relative offset straddling NI_FAST_RTOL flips the gate."""
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        for mult, applied in ((0.5, True), (2.0, False)):
            scaled = ni_total * (1.0 + mult * NI_FAST_RTOL)
            _, _, meta = _subtract_fast_ni(psi, scaled, None, ni_th, zf, z2, 6.0)
            assert meta["applied"] is applied, mult
            assert meta["mismatch"] == pytest.approx(mult * NI_FAST_RTOL,
                                                     rel=1e-6)

    def test_the_gate_is_evaluated_at_the_five_declared_points(self):
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        _, _, meta = _subtract_fast_ni(psi, ni_total, None, ni_th, zf, z2, 6.0)
        assert tuple(meta["gate"]) == NI_FAST_GATE_PSI_N == (0.0, 0.2, 0.4,
                                                             0.6, 0.8)

    def test_a_disagreement_outside_the_gate_points_is_not_consulted(self):
        """Past 0.8 both profiles roll off; a ratio there is the edge model.

        The gate deliberately does not look there, so an edge-only deviation
        must not veto a subtraction the interior fully supports.
        """
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles(n=128)
        bad = ni_total.copy()
        bad[psi > 0.9] *= 1.5                     # gross, and entirely outside
        _, _, meta = _subtract_fast_ni(psi, bad, None, ni_th, zf, z2, 6.0)
        assert meta["applied"]

    def test_a_disagreement_on_axis_alone_is_enough_to_veto(self):
        """psi_N=0 is in the gate: that is where a beam is most peaked."""
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles(n=128)
        bad = ni_total.copy()
        bad[psi < 0.05] *= 1.0 + 10.0 * NI_FAST_RTOL
        _, _, meta = _subtract_fast_ni(psi, bad, None, ni_th, zf, z2, 6.0)
        assert not meta["applied"]
        assert meta["gate"][0.0] > NI_FAST_RTOL

    def test_no_density_fast_is_inert(self):
        psi, ni_total, _, _, zf, z2 = _profiles(fast_frac=0.0)
        sig = 0.1 * ni_total
        ni, s, meta = _subtract_fast_ni(psi, ni_total, sig, ni_total,
                                        zf, z2, 6.0)
        assert not meta["applied"]
        np.testing.assert_array_equal(ni, ni_total)
        np.testing.assert_array_equal(s, sig)
        assert "no fast-ion population" in meta["evidence"]


class TestSigmaScaling:
    def test_the_fractional_error_is_preserved(self):
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        sig = 0.13 * ni_total
        ni, s, meta = _subtract_fast_ni(psi, ni_total, sig, ni_th, zf, z2, 6.0)
        assert meta["applied"]
        # The envelope is a measurement error on the deuteron inventory; the
        # fast density removed from the mean carries no IDA error of its own.
        np.testing.assert_allclose(s / ni, sig / ni_total, rtol=1e-12)

    def test_the_envelope_shrinks_with_the_mean_never_grows(self):
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        sig = 0.13 * ni_total
        _, s, _ = _subtract_fast_ni(psi, ni_total, sig, ni_th, zf, z2, 6.0)
        assert np.all(s <= sig + 1e-30)
        assert float(np.max(sig - s)) > 0.0          # it actually moved

    def test_a_none_envelope_survives(self):
        psi, ni_total, ni_fast, ni_th, zf, z2 = _profiles()
        ni, s, meta = _subtract_fast_ni(psi, ni_total, None, ni_th, zf, z2, 6.0)
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

    @pytest.mark.parametrize("subtract", [True, False])
    def test_the_draws_get_the_readers_envelope(self, tmp_path, subtract):
        """resolve_uncertainty must not re-derive sigma_ni from the raw file."""
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import BouquetConfig, SolverConfig
        ddp, cdf, *_ = _build(tmp_path)
        bl = _read(ddp, cdf, ni_subtract_fast=subtract)
        cfg = BouquetConfig(
            source=ImasSource(ids_path=ddp, time=1.0, ida_path=cdf,
                              impurity_Z=Z_IMP, ni_subtract_fast=subtract),
            solver=SolverConfig(mesh_path="unused"), output_header="unused")
        env = resolve_uncertainty(cfg, bl)
        np.testing.assert_allclose(env["sigma_ni"], bl.aux["sigma_ni_ida"],
                                   rtol=1e-12)

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


# ---------------------------------------------------------------------------
# the consumers that re-derive the impurity from (ne, ni)
# ---------------------------------------------------------------------------
class TestArchiveConsumers:
    """A thermal n_i must be paired with a thermal n_e everywhere.

    Pairing the archive's full ne with a thermal n_i makes every (ne, ni)
    inversion charge the whole beam to the impurity.
    """

    def _case(self, fast_frac=0.20):
        Z, ne, ti = 6.0, 5.0e19, 2.0e3
        nC = 0.005 * ne
        zeff = 1.0 + Z * (Z - 1.0) * nC / ne
        ni_total = ne - Z * nC
        z_fast = fast_frac * ne
        return Z, ne, nC, zeff, ni_total, z_fast, ti

    def test_pairing_thermal_ni_with_full_ne_is_the_trap(self):
        from bouquet.physics import (effective_impurity_charge,
                                     impurity_pressure)
        Z, ne, nC, zeff, ni_total, z_fast, ti = self._case()
        ni_th = ni_total - z_fast
        a, b = np.full(4, ne), np.full(4, ni_th)
        bad = impurity_pressure(a, b, np.full(4, ti),
                                effective_impurity_charge(a, b,
                                                          np.full(4, zeff)))
        true = 1.602176634e-19 * nC * ti
        assert bad[0] > 10.0 * true          # the regression this guards

    def test_the_archived_Z_imp_makes_the_pair_exact(self):
        from bouquet.physics import impurity_pressure
        Z, ne, nC, zeff, ni_total, z_fast, ti = self._case()
        ni_th = ni_total - z_fast
        ne_th = ne - z_fast
        good = impurity_pressure(np.full(4, ne_th), np.full(4, ni_th),
                                 np.full(4, ti), Z)
        np.testing.assert_allclose(good, 1.602176634e-19 * nC * ti, rtol=1e-12)

    def test_the_baseline_carries_what_the_consumers_need(self, tmp_path):
        ddp, cdf, *_ = _build(tmp_path)
        bl = _read(ddp, cdf)
        assert bl.z_fast is not None and np.any(bl.z_fast)
        assert bl.Z_imp == Z_IMP
        # and the convention flag the per-draw Z_eff maths needs
        assert bl.zeff_includes_fast is True

    def test_zeff_from_fuse_flips_the_convention(self, tmp_path):
        ddp, cdf, *_ = _build(tmp_path)
        assert _read(ddp, cdf, zeff_from_fuse=True).zeff_includes_fast is False


class TestZeffBounds:
    """The window both Z_eff draw paths clip to (physics.zeff_bounds)."""

    NE, ZI = np.full(4, 5e19), 6.0

    def _moments(self, frac=0.2, Z_beam=1.0):
        n_fast = frac * self.NE
        return Z_beam * n_fast, Z_beam ** 2 * n_fast

    def test_no_fast_ions_is_the_familiar_window(self):
        from bouquet.physics import zeff_bounds
        assert zeff_bounds(self.NE, self.ZI) == (1.0, 6.0)

    def test_both_conventions_reduce_to_it_at_zero_fast(self):
        from bouquet.physics import zeff_bounds
        zf, z2 = np.zeros(4), np.zeros(4)
        for flag in (False, True):
            lo, hi = zeff_bounds(self.NE, self.ZI, zf, z2,
                                 zeff_includes_fast=flag)
            np.testing.assert_allclose(lo, 1.0)
            np.testing.assert_allclose(hi, 6.0)

    def test_the_two_conventions_differ_once_a_beam_is_present(self):
        from bouquet.physics import zeff_bounds
        zf, z2 = self._moments(0.2)                      # hydrogenic
        lo_t, hi_t = zeff_bounds(self.NE, self.ZI, zf, z2,
                                 zeff_includes_fast=False)
        lo_m, hi_m = zeff_bounds(self.NE, self.ZI, zf, z2,
                                 zeff_includes_fast=True)
        np.testing.assert_allclose(lo_t, 0.8)            # thermal numerator
        np.testing.assert_allclose(hi_t, 4.8)
        np.testing.assert_allclose(lo_m, 1.0)            # measured (all ions)
        np.testing.assert_allclose(hi_m, 5.0)

    def test_the_measured_window_moves_with_the_beam_charge(self):
        """A He beam is not a D beam at the same charge density."""
        from bouquet.physics import zeff_bounds
        zf_d, z2_d = self._moments(0.2, Z_beam=1.0)
        zf_he, z2_he = self._moments(0.1, Z_beam=2.0)    # same z_fast
        np.testing.assert_allclose(zf_d, zf_he)          # identical charge
        lo_d, hi_d = zeff_bounds(self.NE, self.ZI, zf_d, z2_d,
                                 zeff_includes_fast=True)
        lo_h, hi_h = zeff_bounds(self.NE, self.ZI, zf_he, z2_he,
                                 zeff_includes_fast=True)
        assert not np.allclose(lo_d, lo_h)               # z2 tells them apart
        assert not np.allclose(hi_d, hi_h)

    @pytest.mark.parametrize("Z_beam", [1.0, 2.0, 6.0])
    def test_the_window_is_exactly_where_ni_and_nz_stay_positive(self, Z_beam):
        """Each bound is the edge case it claims, in its own convention."""
        from bouquet.physics import main_ion_density_from_zeff, zeff_bounds
        zf, z2 = self._moments(0.1, Z_beam)
        for flag in (False, True):
            lo, hi = zeff_bounds(self.NE, self.ZI, zf, z2,
                                 zeff_includes_fast=flag)
            ni_hi = main_ion_density_from_zeff(self.NE, hi, self.ZI, z_fast=zf,
                                               z2_fast=z2,
                                               zeff_includes_fast=flag)
            np.testing.assert_allclose(ni_hi, 0.0, atol=1e6)   # ni -> 0 at hi
            ni_lo = main_ion_density_from_zeff(self.NE, lo, self.ZI, z_fast=zf,
                                               z2_fast=z2,
                                               zeff_includes_fast=flag)
            nz_lo = (self.NE - zf - ni_lo) / self.ZI
            np.testing.assert_allclose(nz_lo, 0.0, atol=1e6)   # nz -> 0 at lo

    def test_the_measured_branch_refuses_to_guess_the_beam_charge(self):
        from bouquet.physics import zeff_bounds
        zf, _ = self._moments()
        with pytest.raises(ValueError, match="z2_fast"):
            zeff_bounds(self.NE, self.ZI, zf, None, zeff_includes_fast=True)


class TestFastIonDensityEquivalent:
    """What a MEASURED ni carries of the fast population."""

    def test_a_hydrogenic_beam_is_its_own_density(self):
        from bouquet.physics import fast_ion_density_equivalent
        n = np.full(4, 1e19)
        np.testing.assert_allclose(
            fast_ion_density_equivalent(n, n, 6.0), n, rtol=1e-12)

    def test_a_beam_at_the_impurity_charge_contributes_nothing(self):
        """It is indistinguishable from thermal impurity, in both channels."""
        from bouquet.physics import fast_ion_density_equivalent
        n, Z = np.full(4, 1e19), 6.0
        np.testing.assert_allclose(
            fast_ion_density_equivalent(Z * n, Z * Z * n, Z), 0.0, atol=1e3)

    @pytest.mark.parametrize("Z_beam", [1.0, 2.0, 3.0, 6.0])
    def test_it_matches_the_per_species_sum(self, Z_beam):
        from bouquet.physics import fast_ion_density_equivalent
        n, Z = np.full(4, 1e19), 6.0
        np.testing.assert_allclose(
            fast_ion_density_equivalent(Z_beam * n, Z_beam ** 2 * n, Z),
            n * Z_beam * (Z - Z_beam) / (Z - 1.0), rtol=1e-12)

    def test_two_species_add(self):
        from bouquet.physics import fast_ion_density_equivalent as f
        nD, nHe, Z = np.full(4, 1e19), np.full(4, 4e18), 6.0
        zf = 1.0 * nD + 2.0 * nHe
        z2 = 1.0 * nD + 4.0 * nHe
        np.testing.assert_allclose(f(zf, z2, Z),
                                   f(nD, nD, Z) + f(2 * nHe, 4 * nHe, Z),
                                   rtol=1e-12)


class TestMainIonConvention:
    NE, ZI, ZEFF = np.full(4, 5e19), 6.0, np.full(4, 1.15)

    def test_the_measured_convention_subtracts_the_equivalent_not_the_charge(self):
        from bouquet.physics import (fast_ion_density_equivalent,
                                     main_ion_density_from_zeff)
        n = np.full(4, 1e19)
        zf, z2 = 2.0 * n, 4.0 * n                    # He beam: z_fast != n
        total = main_ion_density_from_zeff(self.NE, self.ZEFF, self.ZI)
        got = main_ion_density_from_zeff(self.NE, self.ZEFF, self.ZI,
                                         z_fast=zf, z2_fast=z2,
                                         zeff_includes_fast=True)
        np.testing.assert_allclose(
            got, total - fast_ion_density_equivalent(zf, z2, self.ZI),
            rtol=1e-12)
        assert not np.allclose(got, total - zf)      # subtracting charge is wrong

    def test_using_the_wrong_convention_biases_ni(self):
        from bouquet.physics import main_ion_density_from_zeff
        n = np.full(4, 1e19)
        for Z_beam in (1.0, 2.0):
            zf, z2 = Z_beam * n, Z_beam ** 2 * n
            a = main_ion_density_from_zeff(self.NE, self.ZEFF, self.ZI,
                                           z_fast=zf, z2_fast=z2,
                                           zeff_includes_fast=True)
            b = main_ion_density_from_zeff(self.NE, self.ZEFF, self.ZI,
                                           z_fast=zf, z2_fast=z2,
                                           zeff_includes_fast=False)
            # measured minus thermal-numerator == z2_fast/(Z_imp - 1),
            # which for a hydrogenic beam is z_fast/(Z_imp - 1) -- 20 % of the
            # beam density at Z_imp = 6.
            np.testing.assert_allclose(a - b, z2 / (self.ZI - 1.0),
                                       rtol=1e-12)

    def test_the_measured_branch_refuses_to_guess_the_beam_charge(self):
        from bouquet.physics import main_ion_density_from_zeff
        with pytest.raises(ValueError, match="z2_fast"):
            main_ion_density_from_zeff(self.NE, self.ZEFF, self.ZI,
                                       z_fast=np.full(4, 1e19),
                                       zeff_includes_fast=True)

    def test_z_fast_none_is_bit_identical_under_either_flag(self):
        from bouquet.physics import main_ion_density_from_zeff
        ne, Z, zeff = np.full(8, 5e19), 6.0, np.linspace(1.0, 5.0, 8)
        base = main_ion_density_from_zeff(ne, zeff, Z)
        for flag in (False, True):
            np.testing.assert_array_equal(
                main_ion_density_from_zeff(ne, zeff, Z,
                                           zeff_includes_fast=flag), base)


class TestPFileFastIonBlock:
    """The p-file model is nz1 = (ne - ni - nb)/Z_imp.

    It has always expected a THERMAL ni plus a separate nb, so the subtraction
    puts bouquet's ni on the right footing -- but only if nb travels with it.
    """

    def _pf(self, npts=32, with_nb=True, Z_beam=1.0):
        from bouquet.io.pfile import PFile
        Z, psi = 6.0, np.linspace(0.0, 1.0, npts)
        ne = 5.0e19 * (1.0 - 0.7 * psi ** 2)
        nC = 0.005 * ne
        nb = 0.20 * ne / Z_beam          # same fast CHARGE at any beam charge
        ni_th = ne - Z * nC - Z_beam * nb
        pf = PFile.__new__(PFile)
        pf._raw, pf._modified = {}, set()
        pf.set_profile("ne", psi, ne * 1e-20)
        pf.set_profile("ni", psi, ni_th * 1e-20)
        if with_nb:
            pf.set_profile("nb", psi, nb * 1e-20)
        pf.set_ion_species([1, 1, 1], [Z, 1.0, Z_beam], [12.0, 2.0, 2.0])
        return pf, psi, nC

    @pytest.mark.parametrize("Z_beam", [1.0, 2.0])
    def test_nb_recovers_the_carbon(self, Z_beam):
        """nb is a PARTICLE density, so quasineutrality must weight it Z_beam.

        Charging it at Z=1 pushes (Z_beam - 1) nb into nz1 -- invisible for the
        hydrogenic beams these files usually carry, wrong for a He beam.
        """
        pf, psi, nC = self._pf(Z_beam=Z_beam)
        pf.compute_quasineutrality()
        np.testing.assert_allclose(np.asarray(pf["nz1"]["data"]) * 1e20,
                                   nC, rtol=1e-9)

    def test_without_nb_the_beam_is_charged_to_the_impurity(self):
        """nz1 inflates by exactly nb/(Z nC) -- 7.67x for this 20 % beam.

        Smaller than the plotting failure (~28x) because Z_imp comes from the
        species block here rather than an (ne, ni) inversion that is itself
        thrown off; only the numerator is wrong.
        """
        import warnings
        pf, psi, nC = self._pf(with_nb=False)
        nb = 0.20 * 5.0e19 * (1.0 - 0.7 * psi ** 2)   # Z_beam = 1 here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pf.compute_quasineutrality()
        nz1 = np.asarray(pf["nz1"]["data"]) * 1e20
        np.testing.assert_allclose(nz1, nC + nb / 6.0, rtol=1e-9)
        assert np.median(nz1 / nC) == pytest.approx(7.667, abs=0.01)
