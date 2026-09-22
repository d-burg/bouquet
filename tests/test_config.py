"""Config serialization + h5 provenance (Phase 1: F8/F9/F10, F25)."""

import json
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq
from bouquet.config import (BouquetConfig, SolverConfig, ReconstructionSource,
                            ImasSource, UncertaintyConfig, GenerationConfig,
                            FilterConfig, FixedComponentsConfig)


def _full_recon_cfg(header="RUN"):
    psi = np.linspace(0, 1, 8)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf", time=4.4,
                                    impurity_Z=6.0, profile_overrides={"ti": psi * 1e3}),
        solver=SolverConfig(mesh_path="m.h5", isoflux_pts=np.c_[psi, psi],
                            isoflux_weights=psi),
        output_header=header,
        uncertainty=UncertaintyConfig(jphi_scalar_sigma=0.05,
                                      sigma_profiles={"ne": psi * 1e19},
                                      aux_baselines={"e_r": psi * 1e3},
                                      aux_sigmas={"e_r": psi * 1e2}),
        generation=GenerationConfig(seed=42, scan_key=4400, jBS_scale_range=(0.99, 1.01),
                                    homotopy_passes=[(0.05, 0.10), (0.02, 0.05)]),
        fixed_components=FixedComponentsConfig(p_fast=psi * 1e3, psi_N=psi),
        verbose=True)


class TestSerialization:
    def test_roundtrip_reconstruction(self):
        cfg = _full_recon_cfg()
        cfg2 = BouquetConfig.from_json(cfg.to_json())
        # exact re-encoded equality (avoids ndarray __eq__ ambiguity)
        assert json.dumps(cfg.to_dict(), sort_keys=True) == \
               json.dumps(cfg2.to_dict(), sort_keys=True)

    def test_types_restored(self):
        cfg2 = BouquetConfig.from_json(_full_recon_cfg().to_json())
        assert isinstance(cfg2.source, ReconstructionSource)
        assert isinstance(cfg2.solver.isoflux_pts, np.ndarray)
        assert cfg2.solver.isoflux_pts.shape == (8, 2)
        assert isinstance(cfg2.uncertainty.sigma_profiles["ne"], np.ndarray)
        assert isinstance(cfg2.fixed_components.p_fast, np.ndarray)
        assert cfg2.generation.seed == 42 and cfg2.verbose is True

    def test_imas_source_discriminator(self):
        cfg = BouquetConfig(source=ImasSource(ids_path="dd.json", time=2.3, ida_path="x.cdf"),
                            solver=SolverConfig(mesh_path="m.h5"), output_header="I")
        cfg2 = BouquetConfig.from_json(cfg.to_json())
        assert isinstance(cfg2.source, ImasSource)
        assert cfg2.source.ida_path == "x.cdf"

    def test_to_dict_is_json_serializable(self):
        json.dumps(_full_recon_cfg().to_dict())   # must not raise


class TestProvenance:
    def test_write_and_load(self, tmp_path):
        hdr = str(tmp_path / "prov")
        bq.initialize_equilibrium_database(hdr)
        cfg = _full_recon_cfg(header=hdr)
        bq.write_provenance(hdr, config=cfg, scan_key=4400)
        with h5py.File(hdr + ".h5", "r") as hf:
            assert hf.attrs["schema_version"] == 2
            assert hf.attrs["bouquet_version"] == bq.__version__
            assert "created" in hf.attrs and "updated" in hf.attrs
            assert "config_json" in hf
            assert "scan/4400/config_json" in hf
        c2 = bq.load_config(hdr)
        c3 = bq.load_config(hdr, scan_key=4400)
        assert isinstance(c2.source, ReconstructionSource)
        assert c2.generation.seed == 42
        assert c2.to_json() == c3.to_json()

    def test_pre_provenance_raises(self, tmp_path):
        hdr = str(tmp_path / "bare")
        bq.initialize_equilibrium_database(hdr)
        with pytest.raises(KeyError, match="no stored config"):
            bq.load_config(hdr)

    def test_created_is_stable_across_updates(self, tmp_path):
        hdr = str(tmp_path / "twice")
        bq.initialize_equilibrium_database(hdr)
        cfg = _full_recon_cfg(header=hdr)
        bq.write_provenance(hdr, config=cfg)
        with h5py.File(hdr + ".h5", "r") as hf:
            created1 = hf.attrs["created"]
        bq.write_provenance(hdr, config=cfg)
        with h5py.File(hdr + ".h5", "r") as hf:
            assert hf.attrs["created"] == created1     # set once, never overwritten


class TestBootstrapKwargs:
    """``GenerationConfig.bootstrap_kwargs`` -- the one surface for the
    ``solve_with_bootstrap`` options bouquet does not set itself."""

    def test_default_is_an_empty_per_instance_dict(self):
        a, b = GenerationConfig(), GenerationConfig()
        assert a.bootstrap_kwargs == {}
        a.bootstrap_kwargs["djBS_tol"] = 1e-5
        assert b.bootstrap_kwargs == {}      # default_factory, not a shared dict

    def test_backend_options_pass_validation(self):
        gc = GenerationConfig(bootstrap_kwargs={"djBS_tol": 1e-5,
                                                "taper_edge_jBS": True,
                                                "iterations": 2})
        assert gc.bootstrap_kwargs["taper_edge_jBS"] is True

    @pytest.mark.parametrize("key", ["scale_jBS", "isolate_edge_jBS", "verbose",
                                     "diagnostic_plots", "mygs", "Ip_target"])
    def test_an_argument_the_call_sites_set_is_refused(self, key):
        # These arrive as explicit keywords at every solve_with_bootstrap call,
        # so **bootstrap_kwargs would either duplicate them (TypeError deep in
        # a draw) or silently override a per-draw value.
        with pytest.raises(ValueError, match=key):
            GenerationConfig(bootstrap_kwargs={key: 1})

    def test_it_survives_a_config_roundtrip(self):
        cfg = _full_recon_cfg()
        cfg.generation.bootstrap_kwargs = {"djBS_tol": 1e-5}
        cfg2 = BouquetConfig.from_json(cfg.to_json())
        assert cfg2.generation.bootstrap_kwargs == {"djBS_tol": 1e-5}
