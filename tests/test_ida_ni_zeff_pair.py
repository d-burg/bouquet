"""ni and Z_eff from one IDA resolution are drawn as one channel.

read_ida derives ni from its resolved Z_eff (ni = ne (Z - Zeff)/(Z - 1)), so
drawing the two apart handed each draw a Z_eff its own densities contradict
(median 0.65 in the core on shot 174956).  The draw now derives ni from the
drawn (ne, Zeff); IDAProfiles.zeff_dne carries the CER route's ne dependence
so the derived ni keeps the reader's sigma_ni.
"""
import sys
import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet.io.ida import read_ida
from bouquet.physics import main_ion_density_from_zeff

Z = 6.0


def _write(path, zeff_err=True, carbon=True, nr=48):
    psi = np.linspace(0.0, 1.2, nr)
    ne = 5e19 * (1 - 0.8 * (psi / 1.2) ** 2) + 1e18
    nc = 0.03 * ne
    zeff = 1.0 + Z * (Z - 1.0) * nc / ne            # both routes agree
    te = 3000.0 * (1 - 0.9 * (psi / 1.2) ** 2) + 50.0
    with h5py.File(path, "w") as f:
        f["time"] = np.array([3000.0])
        f["psi_n"] = psi
        for k, v, e in (("n_e", ne, 0.04), ("T_e", te, 0.05), ("T_12C6", 0.9 * te, 0.05)):
            f[k] = v[None, :]
            f[k + "_err"] = (e * v)[None, :]
        f["Zeff"] = zeff[None, :]
        if zeff_err:
            f["Zeff_err"] = (0.08 * zeff)[None, :]
        if carbon:
            f["n_12C6"] = nc[None, :]
            f["n_12C6_err"] = (0.15 * nc)[None, :]
    return str(path)


def _derived_sigma_ni(r):
    """First-order sigma of ni(ne, Zeff) with Zeff = indep + zeff_dne * dne."""
    s_ind2 = np.maximum(r.sigma_Zeff ** 2 - (r.zeff_dne * r.sigma_ne) ** 2, 0.0)
    d_ne = (Z - r.Zeff) / (Z - 1.0) - r.ne / (Z - 1.0) * r.zeff_dne
    return np.sqrt((d_ne * r.sigma_ne) ** 2 + (r.ne / (Z - 1.0)) ** 2 * s_ind2)


class TestReader:
    def test_vb_only_zeff_does_not_move_with_ne(self, tmp_path):
        r = read_ida(_write(tmp_path / "a.cdf", carbon=False))
        np.testing.assert_array_equal(r.zeff_dne, 0.0)

    def test_cer_only_zeff_is_one_over_ne(self, tmp_path):
        r = read_ida(_write(tmp_path / "a.cdf", zeff_err=False), ni_source="CER")
        np.testing.assert_allclose(r.zeff_dne, -(r.Zeff - 1.0) / r.ne, rtol=1e-12)

    @pytest.mark.parametrize("ni_source", ["all", "CER", "Zeff"])
    def test_ni_derived_from_the_drawn_zeff_carries_sigma_ni(self, tmp_path, ni_source):
        r = read_ida(_write(tmp_path / "a.cdf"), ni_source=ni_source)
        np.testing.assert_allclose(_derived_sigma_ni(r), r.sigma_ni, rtol=1e-10)

    def test_without_the_coupling_the_derived_sigma_is_wrong(self, tmp_path):
        r = read_ida(_write(tmp_path / "a.cdf"))
        r.zeff_dne = np.zeros_like(r.ne)
        core = r.psi_N < 0.9
        assert np.max(np.abs(_derived_sigma_ni(r) / r.sigma_ni - 1.0)[core]) > 0.05


def _bl(psi, ne, zeff):
    from bouquet.baseline import Baseline
    psi_N = np.linspace(0.0, 1.0, 33)
    j = 8e5 * (1 - psi_N ** 2)
    return Baseline(psi_N=psi_N, j_phi=j, j_inductive=0.85 * j, j_BS=0.15 * j,
                    psi_N_kinetic=psi, ne=ne, te=ne * 0 + 1e3,
                    ni=main_ion_density_from_zeff(ne, zeff, Z), ti=ne * 0 + 1e3,
                    Zeff=zeff, Ip_target=1.2e6, l_i_target=0.85,
                    provenance="reconstruction")


def _cfg(tmp_path, source, unc=None):
    from bouquet.config import BouquetConfig, SolverConfig, UncertaintyConfig
    return BouquetConfig(source=source, solver=SolverConfig(mesh_path=str(tmp_path / "m.h5")),
                         output_header=str(tmp_path / "out"),
                         uncertainty=unc or UncertaintyConfig())


class TestResolve:
    def _recon(self, tmp_path, unc=None):
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import ReconstructionSource
        cdf = _write(tmp_path / "own.cdf")
        r = read_ida(cdf)
        src = ReconstructionSource(geqdsk_path=str(tmp_path / "g"), profiles_path=cdf,
                                   time=3.0)
        return resolve_uncertainty(_cfg(tmp_path, src, unc), _bl(r.psi_N, r.ne, r.Zeff)), r

    def test_an_ida_pair_derives_ni_from_zeff(self, tmp_path):
        env, r = self._recon(tmp_path)
        assert env["ni_from_zeff"]
        np.testing.assert_allclose(env["zeff_dne"], r.zeff_dne)
        np.testing.assert_allclose(env["aux_sigmas"]["zeff"], r.sigma_Zeff)

    def test_an_explicit_ni_envelope_stays_independent(self, tmp_path):
        from bouquet.config import UncertaintyConfig
        r = read_ida(_write(tmp_path / "own.cdf"))
        env, _ = self._recon(tmp_path, UncertaintyConfig(
            sigma_profiles={"ni": 0.1 * r.ne}))
        assert not env["ni_from_zeff"] and env["zeff_dne"] is None

    def test_explicit_false_still_wins(self, tmp_path):
        from bouquet.config import UncertaintyConfig
        env, _ = self._recon(tmp_path, UncertaintyConfig(ni_from_zeff=False))
        assert not env["ni_from_zeff"] and env["zeff_dne"] is None

    @pytest.mark.parametrize("zeff_from_fuse, pair", [(False, True), (True, False)])
    def test_ida_hybrid(self, tmp_path, zeff_from_fuse, pair):
        """zeff_from_fuse pairs IDA's ni with FUSE's Z_eff: not one resolution."""
        import warnings
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import ImasSource
        cdf = _write(tmp_path / "ida.cdf")
        r = read_ida(cdf)
        bl = _bl(r.psi_N, r.ne, r.Zeff)
        bl.aux = {"ida_profiles": (cdf, r), "zeff": r.Zeff}
        src = ImasSource(ids_path=str(tmp_path / "dd.json"), time=3.0, ida_path=cdf,
                         zeff_from_fuse=zeff_from_fuse)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = resolve_uncertainty(_cfg(tmp_path, src), bl)
        assert env["ni_from_zeff"] is pair
        assert (env["zeff_dne"] is not None) is pair


# --- the draw itself: perturb_kinetic_equilibrium up to the solve ---------------

class _Stop(Exception):
    pass


class _GS:
    def flux_integral(self, psi, f):
        return np.trapezoid(f, psi)

    def set_targets(self, **kw):
        raise _Stop


@pytest.fixture
def no_oft(monkeypatch):
    names = ("OpenFUSIONToolkit", "OpenFUSIONToolkit.TokaMaker",
             "OpenFUSIONToolkit.TokaMaker.util", "OpenFUSIONToolkit.TokaMaker.bootstrap")
    for n in names:
        m = types.ModuleType(n)
        m.get_jphi_from_GS = m.solve_with_bootstrap = m.find_optimal_scale = None
        monkeypatch.setitem(sys.modules, n, m)


def _draws(r, zeff_dne, n=600, seed=3):
    import bouquet.physics as phys
    from bouquet.TokaMaker_interface import perturb_kinetic_equilibrium
    got = []
    orig = phys.main_ion_density_from_zeff

    def spy(ne, zeff, *a, **k):
        ni = orig(ne, zeff, *a, **k)
        got.append((np.array(ne), np.array(zeff), ni))
        return ni
    phys.main_ion_density_from_zeff = spy
    try:
        psi = r.psi_N
        conf = psi <= 1.0
        sl = lambda a: np.asarray(a)[conf]   # noqa: E731
        rng = np.random.default_rng(seed)
        for _ in range(n):
            with pytest.raises(_Stop):
                perturb_kinetic_equilibrium(
                    _GS(), sl(psi), sl(r.ne) * 1e3 * 1.602e-19, sl(r.ne), sl(r.te),
                    sl(r.ni), sl(r.ti), np.ones(conf.sum()),
                    sl(r.sigma_ne), sl(r.sigma_te), sl(r.sigma_ni), sl(r.sigma_ti),
                    np.zeros(conf.sum()), 0.5, 0.4, 0.25, 1e6, 0.9, sl(r.Zeff), 65,
                    p_thresh=1e9, Z_imp=Z, input_jinductive=np.zeros(conf.sum()),
                    aux_sigmas={"zeff": sl(r.sigma_Zeff)},
                    aux_baselines={"zeff": sl(r.Zeff)},
                    aux_length_scales={"zeff": 0.5},
                    ni_from_zeff=True, zeff_dne=None if zeff_dne is None else sl(zeff_dne),
                    rng=rng)
    finally:
        phys.main_ion_density_from_zeff = orig
    ne, zf, ni = (np.array(x) for x in zip(*got))
    return ne, zf, ni, conf


class TestDraw:
    def test_draws_keep_both_reader_envelopes(self, tmp_path, no_oft):
        r = read_ida(_write(tmp_path / "a.cdf"))
        ne, zf, ni, conf = _draws(r, r.zeff_dne)
        core = r.psi_N[conf] < 0.8
        # sampling + monotonic-rejection slack; without zeff_dne ni falls ~12 % short
        np.testing.assert_allclose(np.std(zf, 0)[core], r.sigma_Zeff[conf][core], rtol=0.1)
        np.testing.assert_allclose(np.std(ni, 0)[core], r.sigma_ni[conf][core], rtol=0.08)
        # and Z_eff is the draw densities' own
        nz = (ne - ni) / Z
        np.testing.assert_allclose(zf, (ni + Z * Z * nz) / ne, rtol=1e-9)

    def test_the_coupling_is_the_ne_zeff_correlation(self, tmp_path, no_oft):
        r = read_ida(_write(tmp_path / "a.cdf"))
        ne, zf, _, conf = _draws(r, r.zeff_dne, n=400)
        _, zf0, _, _ = _draws(r, None, n=400)
        i = int(np.argmin(np.abs(r.psi_N[conf] - 0.3)))
        c = np.corrcoef(ne[:, i], zf[:, i])[0, 1]
        expect = (r.zeff_dne[conf][i] * r.sigma_ne[conf][i]) / r.sigma_Zeff[conf][i]
        assert expect < -0.1 and abs(c - expect) < 0.12
        assert abs(np.corrcoef(ne[:, i], zf0[:, i])[0, 1]) < 0.12
