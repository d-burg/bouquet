"""IMAS read/write in a toroidal-flux (phi_n) run.

The reader relabels the core_profiles nodes by their own rho_tor_norm² and
places equilibrium.profiles_1d by the equilibrium's rho_tor_norm²; the draw
writer lands a phi_n archive on the template's Φ_N nodes and samples the
draw's geometry at the draw's ψ_N of those nodes.
"""
import json
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bouquet import coords
from bouquet.config import ImasSource
from bouquet.io.imas import read_imas_baseline, write_imas_draw
from bouquet.io.imas import _fuse_current_geometry
from bouquet.physics import (jphi_tokamaker_to_jpar, jphi_tokamaker_to_jtor_imas,
                             jtor_imas_to_jphi_tokamaker)
from test_ni_fast_subtraction import N, _dd
from test_imas_export import (_GEQ, _J_BS, _J_IND, _J_PHI, _P, _PEQ,
                              _fsa_geom_on, _make_archive, _make_template)

PSI = np.linspace(0.0, 1.0, N)
RHO = PSI ** 0.55                     # core_profiles' real rho_tor_norm
NEQ = 41
PSI_EQ = np.linspace(0.0, 1.0, NEQ)
RHO_EQ = PSI_EQ ** 0.45               # the equilibrium's own (different) map


def _norm(phi):
    return (phi - phi[0]) / (phi[-1] - phi[0])


def _write_dd(tmp_path, rho=RHO, rho_eq=RHO_EQ, eq_q=True):
    ne = 5.0e19 * (1.0 - 0.8 * PSI ** 2) + 1e18
    nc = 1.0e18 * (1.0 - 0.5 * PSI ** 2)
    dd, _, _ = _dd(4.0e19 * (1.0 - 0.7 * PSI ** 2), ne, nc)
    cp = dd["core_profiles"]["profiles_1d"][0]
    if rho is not None:
        cp["grid"]["rho_tor_norm"] = np.asarray(rho).tolist()
    # equilibrium on its own grid, with its own rho_tor_norm
    eqp = dd["equilibrium"]["time_slice"][0]["profiles_1d"]
    for k in ("pressure", "j_tor", "f", "gm1", "gm5", "gm8", "gm9", "dpressure_dpsi"):
        eqp[k] = np.interp(PSI_EQ, PSI, eqp[k]).tolist()
    eqp["psi"] = PSI_EQ.tolist()
    eqp.pop("rho_tor_norm", None)
    if rho_eq is not None:
        eqp["rho_tor_norm"] = np.asarray(rho_eq).tolist()
    eqp.pop("q", None)
    if eq_q:
        eqp["q"] = (1.0 + 3.0 * PSI_EQ ** 2).tolist()
    p = tmp_path / "dd.json"
    p.write_text(json.dumps(dd))
    return str(p), eqp


def _read(ddp, coord):
    return read_imas_baseline(ImasSource(ids_path=ddp, time=1.0, coord=coord),
                              allow_incomplete_pressure=True)


class TestReadPhi:
    def test_phi_n_is_a_relabel_of_psi_n(self, tmp_path):
        ddp, _ = _write_dd(tmp_path)
        bp, bf = _read(ddp, "psi_n"), _read(ddp, "phi_n")
        np.testing.assert_array_equal(bp.psi_N, PSI)
        np.testing.assert_allclose(bf.psi_N, _norm(RHO ** 2), rtol=0, atol=1e-15)
        for k in ("ne", "te", "ni", "ti", "j_phi", "j_BS"):
            np.testing.assert_array_equal(getattr(bf, k), getattr(bp, k))

    def test_the_equilibrium_is_placed_by_its_own_rho(self, tmp_path):
        ddp, eqp = _write_dd(tmp_path)
        bp, bf = _read(ddp, "psi_n"), _read(ddp, "phi_n")
        phi_eq = _norm(RHO_EQ ** 2)
        exp_p = np.interp(bf.psi_N, phi_eq, eqp["pressure"])
        eq_jphi = jtor_imas_to_jphi_tokamaker(            # on the slice's own geometry
            np.asarray(eqp["j_tor"]),
            _fuse_current_geometry(json.load(open(ddp))["equilibrium"], 0))
        exp_j = np.interp(bf.psi_N, phi_eq, eq_jphi)
        np.testing.assert_allclose(bf.p_equilibrium, exp_p, rtol=1e-12)
        np.testing.assert_allclose(bf.jphi_diff + bf.j_phi, exp_j, rtol=1e-12)
        # psi_n keeps the psi_N placement, which differs here
        np.testing.assert_allclose(
            bp.p_equilibrium, np.interp(PSI, PSI_EQ, eqp["pressure"]), rtol=1e-12)
        assert np.max(np.abs(bf.p_equilibrium - bp.p_equilibrium)) > 1e-3 * np.max(exp_p)

    def test_rho_tor_is_phi_n(self, tmp_path):
        ddp, _ = _write_dd(tmp_path)
        bf, br = _read(ddp, "phi_n"), _read(ddp, "rho_tor")
        assert br.coord == "phi_n"
        for k in ("psi_N", "ne", "j_phi", "p_equilibrium", "jphi_diff"):
            np.testing.assert_array_equal(getattr(br, k), getattr(bf, k))

    @pytest.mark.parametrize("which", ["cp", "eq"])
    @pytest.mark.parametrize("bad", ["placeholder", "missing"])
    def test_an_unusable_rho_is_refused(self, tmp_path, which, bad):
        rho = {"placeholder": np.sqrt, "missing": lambda p: None}[bad]
        kw = {"rho": rho(PSI)} if which == "cp" else {"rho_eq": rho(PSI_EQ), "eq_q": False}
        ddp, _ = _write_dd(tmp_path, **kw)
        with pytest.raises(ValueError, match="rho_tor_norm"):
            _read(ddp, "phi_n")
        if which == "eq" and bad == "missing":   # nor q: the exact current
            # conversion (it interpolates in rho) is unavailable -- a psi_n
            # read falls back to the ratio method, loudly and stamped
            # (review PR64 B7; a refusal was an undeclared input change)
            with pytest.warns(UserWarning, match="FALLING BACK"):
                bp = _read(ddp, "psi_n")
            conv = bp.li_metrics["imas_current_conversion"]
            assert conv["method"].startswith("ratio")
            assert "rho_tor_norm" in conv["reason"]
        else:
            _read(ddp, "psi_n")              # psi_n placement never looks at rho

    def test_an_equilibrium_without_rho_is_placed_by_its_q(self, tmp_path):
        from bouquet.coords import phi_n_from_q
        ddp, eqp = _write_dd(tmp_path, rho_eq=None)
        bf = _read(ddp, "phi_n")
        phi_eq = phi_n_from_q(PSI_EQ, eqp["q"])[1]
        np.testing.assert_allclose(
            bf.p_equilibrium, np.interp(bf.psi_N, phi_eq, eqp["pressure"]), rtol=1e-12)

    def test_fixed_components_on_psi_n_go_through_the_dd_map(self, tmp_path):
        from bouquet.config import FixedComponentsConfig
        ddp, _ = _write_dd(tmp_path)
        g = np.linspace(0.0, 1.0, 17)
        j = 1e5 * (1.0 - g ** 2)
        src = ImasSource(ids_path=ddp, time=1.0, coord="phi_n")
        run_ = read_imas_baseline(src, fixed=FixedComponentsConfig(j_NBI=j, psi_N=g),
                                  allow_incomplete_pressure=True)
        psi_ = read_imas_baseline(src, fixed=FixedComponentsConfig(
            j_NBI=j, psi_N=g, coord="psi_n"), allow_incomplete_pressure=True)
        x = run_.psi_N
        np.testing.assert_allclose(run_.j_NBI, np.interp(x, g, j), rtol=1e-12)
        np.testing.assert_allclose(
            psi_.j_NBI, np.interp(x, np.interp(g, PSI, x), j), rtol=1e-12)

    def test_a_rho_grid_off_0_to_1_is_renormalised(self, tmp_path):
        rho, rho_eq = 0.02 + 0.97 * RHO, 0.01 + 0.98 * RHO_EQ
        ddp, eqp = _write_dd(tmp_path, rho=rho, rho_eq=rho_eq)
        bf = _read(ddp, "phi_n")
        np.testing.assert_allclose(bf.psi_N, _norm(rho ** 2), rtol=0, atol=1e-15)
        assert bf.psi_N[0] == 0.0 and bf.psi_N[-1] == 1.0
        np.testing.assert_allclose(
            bf.p_equilibrium,
            np.interp(bf.psi_N, _norm(rho_eq ** 2), eqp["pressure"]), rtol=1e-12)


@pytest.mark.skipif(not os.path.isfile(_GEQ), reason="d3dlike.geqdsk absent")
class TestWriteDrawPhi:
    def _setup(self, tmp_path, coord):
        arc = str(tmp_path / "run.h5"); _make_archive(arc, with_fsa=True)
        self.zeff = np.linspace(2.0, 1.5, 20)
        self.kin = {"n_e": 1.0, "T_e": 2.0, "n_i": 3.0, "T_i": 4.0}
        with h5py.File(arc, "a") as hf:
            g = hf["scan/0/0"]
            for k, s in self.kin.items():      # distinct per channel
                del g[k]
                g.create_dataset(k, data=s * np.linspace(1.0, 0.1, 20))
            g.create_dataset("aux_zeff", data=self.zeff)
            if coord == "phi_n":
                hf.require_group("scan/0/_baseline").attrs["profile_coord"] = "phi_n"
        tmpl = str(tmp_path / "tmpl.json"); psi, _ = _make_template(tmpl)
        psiN_t = _norm(psi)
        rho = psiN_t ** 0.4
        t = json.load(open(tmpl))
        t["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = rho.tolist()
        json.dump(t, open(tmpl, "w"))
        out = str(tmp_path / "draw.json")
        write_imas_draw(arc, 0, tmpl, out, scan_key=0, fidelity="exact")
        cp = json.load(open(out))["core_profiles"]["profiles_1d"][0]
        return cp, psi, psiN_t, rho

    def _geq(self):
        from bouquet.io.geqdsk import read_geqdsk
        return read_geqdsk(_GEQ)

    def _check_channels(self, cp, x):
        pk = np.linspace(0, 1, 20)
        s = self.kin
        np.testing.assert_allclose(cp["electrons"]["density_thermal"],
                                   np.interp(x, pk, s["n_e"] * np.linspace(1, .1, 20)), rtol=1e-12)
        np.testing.assert_allclose(cp["electrons"]["temperature"],
                                   np.interp(x, pk, s["T_e"] * np.linspace(1, .1, 20)), rtol=1e-12)
        ion = cp["ion"][0]
        np.testing.assert_allclose(ion["density_thermal"],
                                   np.interp(x, pk, s["n_i"] * np.linspace(1, .1, 20)), rtol=1e-12)
        np.testing.assert_allclose(ion["temperature"],
                                   np.interp(x, pk, s["T_i"] * np.linspace(1, .1, 20)), rtol=1e-12)
        np.testing.assert_allclose(cp["zeff"], np.interp(x, pk, self.zeff), rtol=1e-12)

    def _geom(self, psiN):
        return _fsa_geom_on(psiN)

    def _check_currents(self, cp, x, geom, psiN):
        pt = _P(psiN)
        at = lambda j: np.interp(x, _PEQ, j)            # noqa: E731
        np.testing.assert_allclose(cp["j_tor"], jphi_tokamaker_to_jtor_imas(at(_J_PHI), geom),
                                   rtol=1e-10)
        for k, j, p in (("j_total", _J_PHI, pt), ("j_ohmic", _J_IND, 0.0),
                        ("j_bootstrap", _J_BS, pt)):
            np.testing.assert_allclose(cp[k], jphi_tokamaker_to_jpar(at(j) - p, geom), rtol=1e-10)

    def test_a_phi_archive_lands_on_rho2_and_uses_the_draw_psi(self, tmp_path):
        cp, _, psiN_t, rho = self._setup(tmp_path, "phi_n")
        x = rho ** 2
        self._check_channels(cp, x)
        geq = self._geq()
        psiN_d = np.interp(x, _norm(np.asarray(geq.rhovn) ** 2), geq.psi_N)
        assert np.max(np.abs(psiN_d - psiN_t)) > 1e-3   # the maps differ
        self._check_currents(cp, x, self._geom(psiN_d), psiN_d)
        # the draw's psi at the nodes, in COCOS 11 (psi_11 = -2 pi psi_7 of the
        # archived eqdsk; review PR64 B4)
        np.testing.assert_allclose(
            cp["grid"]["psi"], -2.0 * np.pi * (
                geq.psi_axis + psiN_d * (geq.psi_boundary - geq.psi_axis)),
            rtol=1e-12)
        np.testing.assert_allclose(cp["grid"]["rho_tor_norm"], rho, rtol=0, atol=0)

    def test_a_psi_archive_is_unchanged(self, tmp_path):
        cp, psi, psiN_t, _ = self._setup(tmp_path, "psi_n")
        self._check_channels(cp, psiN_t)
        np.testing.assert_array_equal(cp["grid"]["psi"], psi)
        self._check_currents(cp, psiN_t, self._geom(psiN_t), psiN_t)


class TestDdPhiN:
    def _cp(self, rho):
        return {"grid": {"rho_tor_norm": list(rho)}}

    def test_relabels_the_nodes(self):
        from bouquet.io.imas import _dd_phi_n
        pn = np.linspace(0, 1, 11)
        rho = pn ** 0.4
        np.testing.assert_allclose(_dd_phi_n(self._cp(rho), pn), rho ** 2)

    @pytest.mark.parametrize("rho", [None, "placeholder", "short", "decreasing"])
    def test_refuses_a_grid_that_does_not_place_the_nodes(self, rho):
        from bouquet.io.imas import _dd_phi_n
        pn = np.linspace(0, 1, 11)
        cp = {"grid": {}}
        if rho == "placeholder":
            cp = self._cp(np.sqrt(pn))
        elif rho == "short":
            cp = self._cp(pn[:-1] ** 0.4)
        elif rho == "decreasing":
            cp = self._cp((pn ** 0.4)[::-1])
        with pytest.raises(ValueError):
            _dd_phi_n(cp, pn)


class TestIdaHybridPhi:
    """ida_hybrid in a phi_n run places the IDA fits by the file's own q."""

    def _build(self, tmp_path, ida_q=True):
        h5py = pytest.importorskip("h5py")
        import json
        from test_ni_fast_subtraction import _build
        ddp, cdf, *_ = _build(tmp_path)
        psi = np.linspace(0.0, 1.0, 33)
        dd = json.loads(open(ddp).read())
        # the dd's own map (q = 1 + 2 psi^2) differs from the IDA's (1 + 4 psi^2)
        dd["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = \
            np.sqrt((3 * psi + 2 * psi ** 3) / 5).tolist()
        dd["equilibrium"]["time_slice"][0]["profiles_1d"]["rho_tor_norm"] = \
            np.sqrt((3 * psi + 2 * psi ** 3) / 5).tolist()
        open(ddp, "w").write(json.dumps(dd))
        if ida_q:
            with h5py.File(cdf, "a") as f:
                f["q"] = (1.0 + 4.0 * psi ** 2)[None, :]
        return ddp, cdf, psi

    def _read(self, ddp, cdf, coord):
        from bouquet.io.imas import read_imas_baseline
        from bouquet.config import ImasSource
        return read_imas_baseline(
            ImasSource(ids_path=ddp, time=1.0, ida_path=cdf, impurity_Z=6.0,
                       coord=coord), kinetic_source="ida_hybrid")

    def test_te_is_placed_by_the_ida_phi_n(self, tmp_path):
        ddp, cdf, psi = self._build(tmp_path)
        bl = self._read(ddp, cdf, "phi_n")
        ida = bl.aux["ida_profiles"][1]
        _, phi_ida = coords.phi_n_from_q(ida.psi_N, ida.q)
        np.testing.assert_allclose(bl.te, np.interp(bl.psi_N, phi_ida, ida.te), rtol=1e-12)
        np.testing.assert_array_equal(bl.psi_map[1], phi_ida)
        # not the psi_N placement the dd's own map would give
        blp = self._read(ddp, cdf, "psi_n")
        assert np.max(np.abs(bl.te - blp.te)) > 0.01 * np.max(blp.te)

    def test_the_envelope_follows_the_same_map(self, tmp_path):
        from bouquet.baseline import resolve_uncertainty
        from bouquet.config import (BouquetConfig, ImasSource, SolverConfig,
                                    UncertaintyConfig)
        ddp, cdf, _ = self._build(tmp_path)
        bl = self._read(ddp, cdf, "phi_n")
        # the IDA sigma file wired as from_imas wires it (under the default
        # ni_source="standard" only uncertainty.ida_path supplies IDA sigmas)
        cfg = BouquetConfig(
            source=ImasSource(ids_path=ddp, time=1.0, ida_path=cdf,
                              impurity_Z=6.0, coord="phi_n"),
            solver=SolverConfig(mesh_path="unused"), output_header="unused",
            uncertainty=UncertaintyConfig(ida_path=cdf))
        env = resolve_uncertainty(cfg, bl)
        np.testing.assert_allclose(env["sigma_te"], bl.aux["sigma_te_ida"], rtol=1e-12)

    def test_an_ida_without_q_is_refused(self, tmp_path):
        ddp, cdf, _ = self._build(tmp_path, ida_q=False)
        with pytest.raises(ValueError, match="no q"):
            self._read(ddp, cdf, "phi_n")
