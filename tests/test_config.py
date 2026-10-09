"""Config serialization + h5 provenance (Phase 1: F8/F9/F10, F25)."""

import json
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import bouquet as bq
from bouquet.config import (BouquetConfig, SolverConfig, ReconstructionSource,
                            ImasSource, UncertaintyConfig, GenerationConfig,
                            FilterConfig, FixedComponentsConfig,
                            validate_bootstrap_kwargs, _bootstrap_kwarg_names,
                            _BOOTSTRAP_RESERVED)


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
            assert hf.attrs["schema_version"] == bq.schema.SCHEMA_VERSION == 3
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

    def test_backend_options_pass_validation(self):
        # Under the (default) unified engine the taper keys are the engine's
        # own and need no toolkit capability; djBS_tol needs the explicit
        # convergence opt-in (and is refused by the engine at BouquetConfig
        # level -- it is a solve_with_bootstrap option).
        with pytest.warns(UserWarning, match="CONVERGENCE"):
            gc = GenerationConfig(bootstrap_kwargs={"djBS_tol": 1e-5,
                                                    "taper_edge_jBS": True,
                                                    "iterations": 2},
                                  bootstrap_convergence_override=True)
        assert gc.bootstrap_kwargs["taper_edge_jBS"] is True

    def test_a_convergence_option_needs_the_explicit_override(self):
        with pytest.raises(ValueError, match="bootstrap_convergence_override"):
            GenerationConfig(bootstrap_kwargs={"djBS_tol": 1e-2})
        with pytest.raises(ValueError, match="saw_relax"):
            GenerationConfig(bootstrap_kwargs={"saw_relax": 0.5},
                             reconstruction_engine="legacy")

    def test_the_engine_taper_needs_no_toolkit_capability_under_unified(
            self, monkeypatch):
        """OpenFUSIONToolkit main has no taper_edge_* option: under the
        unified engine (which implements the taper itself) that is not a
        reason to refuse the engine's own setting."""
        import bouquet.config as _cfg
        main = frozenset({"Zis", "iterations", "parameterize_jBS",
                          "use_OMFIT_sauter"})
        monkeypatch.setattr(_cfg, "_bootstrap_kwarg_names", lambda: main)
        gc = GenerationConfig(reconstruction_engine="unified",
                              bootstrap_kwargs={"taper_edge_jBS": True,
                                                "taper_edge_psi0": 0.995,
                                                "taper_edge_shape": 1})
        assert gc.bootstrap_kwargs["taper_edge_psi0"] == 0.995
        # ... while the legacy path, which forwards it to that toolkit's
        # solve_with_bootstrap, refuses it by capability, naming it
        with pytest.raises(ValueError, match="installed OpenFUSIONToolkit"):
            GenerationConfig(reconstruction_engine="legacy",
                             bootstrap_kwargs={"taper_edge_jBS": True})

    def test_the_installed_toolkit_decides_a_legacy_option(self):
        """No skip: where the toolkit has the internal-solve option it is
        accepted, where it does not it is refused by capability."""
        known = _bootstrap_kwarg_names()
        if known is None:
            pytest.skip("OpenFUSIONToolkit is not importable here")
        kw = dict(reconstruction_engine="legacy",
                  bootstrap_kwargs={"use_sauter_eps": True})
        if "use_sauter_eps" in known:
            assert GenerationConfig(**kw).bootstrap_kwargs == {
                "use_sauter_eps": True}
        else:
            with pytest.raises(ValueError, match="use_sauter_eps"):
                GenerationConfig(**kw)

    @pytest.mark.parametrize("key", ["scale_jBS", "isolate_edge_jBS", "verbose",
                                     "diagnostic_plots", "mygs", "Ip_target",
                                     "ffp_prof", "te_prof", "p_fixed_prof",
                                     "jphi_fixed_prof", "jphi_saw_prof"])
    def test_an_argument_the_call_sites_set_is_refused(self, key):
        # These arrive as explicit keywords at every solve_with_bootstrap call,
        # so **bootstrap_kwargs would either duplicate them (TypeError deep in
        # a draw) or silently override a per-draw value.
        with pytest.raises(ValueError, match=key):
            GenerationConfig(bootstrap_kwargs={key: 1})

    def test_it_survives_a_config_roundtrip(self):
        cfg = _full_recon_cfg()
        cfg.generation.reconstruction_engine = "legacy"     # an SWB option
        cfg.generation.bootstrap_kwargs = {"iterations": 2}  # every toolkit
        cfg2 = BouquetConfig.from_json(cfg.to_json())
        assert cfg2.generation.bootstrap_kwargs == {"iterations": 2}

    def test_a_stored_internal_solve_option_reloads_on_any_toolkit(self):
        """An archive written on a toolkit with the internal bootstrap solve
        (djBS_tol, with the convergence opt-in) must reload everywhere: on a
        toolkit without the option it loads with a warning instead of being
        refused (a legacy run of it would then fail at the call)."""
        cfg = _full_recon_cfg()
        cfg.generation.reconstruction_engine = "legacy"
        d = json.loads(cfg.to_json())
        d["generation"]["bootstrap_kwargs"] = {"djBS_tol": 1e-5}
        d["generation"]["bootstrap_convergence_override"] = True
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg2 = BouquetConfig.from_dict(d)
        assert cfg2.generation.bootstrap_kwargs == {"djBS_tol": 1e-5}
        assert cfg2.generation.bootstrap_convergence_override is True
        known = _bootstrap_kwarg_names()
        if known is not None and "djBS_tol" not in known:
            assert any("installed OpenFUSIONToolkit" in str(x.message)
                       for x in w)
            # a NEW config with it is still refused on this toolkit
            with pytest.raises(ValueError, match="installed OpenFUSIONToolkit"):
                cfg2.generation.bootstrap_kwargs = {"djBS_tol": 2e-5}
            assert cfg2.generation.bootstrap_kwargs == {"djBS_tol": 1e-5}

    def test_a_stored_convergence_option_predating_the_override_loads_with_it(
            self):
        cfg = _full_recon_cfg()
        cfg.generation.reconstruction_engine = "legacy"
        d = json.loads(cfg.to_json())
        d["generation"]["bootstrap_kwargs"] = {"saw_relax": 0.5}
        d["generation"].pop("bootstrap_convergence_override")
        with pytest.warns(UserWarning, match="predates"):
            cfg2 = BouquetConfig.from_dict(d)
        assert cfg2.generation.bootstrap_convergence_override is True

    def test_a_stored_unified_config_with_swb_only_keys_reloads(self):
        """The D3D-like notebooks set bootstrap_kwargs={'iterations': 3}
        after construction under the unified engine (2026-10-04..09);
        generate() ignored it and wrote it into config_json, which then
        could not be reloaded.  It reloads now, the key dropped, said so."""
        cfg = _full_recon_cfg()
        d = json.loads(cfg.to_json())
        assert d["generation"]["reconstruction_engine"] == "unified"
        d["generation"]["bootstrap_kwargs"] = {"iterations": 3,
                                               "taper_edge_jBS": False}
        with pytest.warns(UserWarning, match="never ran"):
            cfg2 = BouquetConfig.from_dict(d)
        assert cfg2.generation.bootstrap_kwargs == {"taper_edge_jBS": False}

    def test_reassigning_the_attribute_is_validated_and_a_refusal_keeps_it(
            self):
        gc = GenerationConfig(reconstruction_engine="legacy")
        with pytest.raises(ValueError, match="iteratons"):
            gc.bootstrap_kwargs = {"iteratons": 3}       # typo
        assert gc.bootstrap_kwargs == {}
        with pytest.raises(ValueError, match="must be a dict"):
            gc.bootstrap_kwargs = None
        gc.bootstrap_kwargs = {"iterations": 2}
        assert gc.bootstrap_kwargs == {"iterations": 2}


class TestBootstrapKwargValidation:
    """The unknown-key guard (the call sites end in ``**kwargs``, so a wrong
    key would otherwise fail every draw).  These run without
    OpenFUSIONToolkit, so they pass ``known`` explicitly."""

    KNOWN = {"iterations", "djBS_tol", "taper_edge_jBS", "taper_edge_psi0",
             "use_python_solve", "scale_jBS", "verbose"}

    def _check(self, d):
        validate_bootstrap_kwargs(d, known=self.KNOWN)

    def test_a_known_option_passes(self):
        self._check({"djBS_tol": 1e-5, "taper_edge_jBS": True})

    def test_a_misspelled_option_is_refused_and_the_message_lists_the_real_ones(self):
        with pytest.raises(ValueError) as e:
            self._check({"djBS_tolerance": 1e-5})
        assert "djBS_tolerance" in str(e.value)
        assert "djBS_tol" in str(e.value)          # the accepted spelling
        assert "scale_jBS" not in str(e.value)     # reserved, not offered

    def test_the_old_swb_iterations_spelling_names_its_replacement(self):
        with pytest.raises(ValueError, match="iterations"):
            self._check({"swb_iterations": 2})

    def test_a_reserved_name_is_reported_as_reserved_not_as_unknown(self):
        with pytest.raises(ValueError, match="passed explicitly at call sites"):
            self._check({"scale_jBS": 1.0})

    def test_without_the_toolkit_the_allow_list_still_refuses_a_typo(
            self, monkeypatch):
        # known=None means "cannot introspect" the installed toolkit: only
        # its capability check is skipped.  The explicit allow-list
        # (BOOTSTRAP_KWARGS_ALLOWED) still refuses a key bouquet does not
        # know, so a typo never survives to the draw loop (it used to).
        import bouquet.config as _cfg
        monkeypatch.setattr(_cfg, "_bootstrap_kwarg_names", lambda: None)
        with pytest.raises(ValueError, match="anything_at_all"):
            validate_bootstrap_kwargs({"anything_at_all": 1}, known=None)
        validate_bootstrap_kwargs({"djBS_tol": 1e-5}, known=None)
        with pytest.raises(ValueError, match="passed explicitly"):
            validate_bootstrap_kwargs({"verbose": False}, known=None)

    def test_a_failed_introspection_is_not_cached(self, monkeypatch):
        """A config built before OpenFUSIONToolkit is on the path must not
        switch the capability check off for the whole process."""
        import builtins
        import bouquet.config as _cfg
        monkeypatch.setattr(_cfg, "_BOOTSTRAP_NAMES_CACHE", {})
        real_import = builtins.__import__

        def _no_oft(name, *a, **k):
            if name.startswith("OpenFUSIONToolkit"):
                raise ImportError("not on the path yet")
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", _no_oft)
        assert _cfg._bootstrap_kwarg_names() is None
        assert _cfg._BOOTSTRAP_NAMES_CACHE == {}

    def test_the_introspected_name_set_is_cached(self):
        assert _bootstrap_kwarg_names() is _bootstrap_kwarg_names()

    def test_every_reserved_name_is_a_real_argument(self):
        # A typo in _BOOTSTRAP_RESERVED would silently stop guarding it.
        # Some reserved names exist only on a toolkit with the internal
        # bootstrap solve (djBS_tol is its marker).
        known = _bootstrap_kwarg_names()
        if known is None or "djBS_tol" not in known:
            pytest.skip("needs an OpenFUSIONToolkit with the internal "
                        "Fortran bootstrap solve")
        grid = {"x", "psi_N"}           # the grid keyword, new or old spelling
        assert _BOOTSTRAP_RESERVED - grid <= known and grid & known
