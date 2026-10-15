"""The assumed default envelope of a sigma-less run is the H-mode flat-top template.

``UncertaintyConfig.{ne,te,ni,ti,zeff}_scalar_sigma`` default to None: the
channel's envelope is ``sigma_template(psi_N, chan) * |baseline|``.  A set
scalar is an explicit flat fraction and wins; a stored config with the old
explicit 0.05 / 0.05 / 0.10 / 0.10 / 0.05 keeps them.  Which default applied
is stamped in resolve_uncertainty()'s "sigma_source".
"""
import json
import os
import warnings

import numpy as np
import pytest

from bouquet.baseline import Baseline, resolve_uncertainty
from bouquet.config import (BouquetConfig, ReconstructionSource, SolverConfig,
                            UncertaintyConfig)
from bouquet.uncertainties import sigma_template

_CH = ("ne", "te", "ni", "ti")
_STORED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stored_configs")


def _bl(psi_kin, psi_map=None):
    psi_N = np.linspace(0.0, 1.0, 41)
    ne = 5e19 * (1 - 0.8 * psi_kin ** 2) + 1e18
    te = 3000.0 * (1 - 0.9 * psi_kin ** 2) + 50.0
    j = 8.0e5 * (1.0 - psi_N ** 2)
    return Baseline(psi_N=psi_N, jphi_total=j, jphi_ind=0.85 * j, jphi_bs=0.15 * j,
                    psi_N_kinetic=psi_kin, ne=ne, te=te, ni=0.85 * ne, ti=0.9 * te,
                    Zeff=1.5 + 0.5 * psi_kin, Ip_target=1.2e6, l_i_target=0.85,
                    provenance="reconstruction", psi_map=psi_map)


def _cfg(tmp_path, unc=None):
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=str(tmp_path / "g.geqdsk"),
                                    profiles_path=str(tmp_path / "p.peqdsk"), time=3.0),
        solver=SolverConfig(mesh_path=str(tmp_path / "mesh.h5")),
        output_header=str(tmp_path / "out"), uncertainty=unc or UncertaintyConfig())


def _prof(bl, ch):
    return np.abs(np.asarray(getattr(bl, ch), float))


def test_the_defaults_are_none():
    u = UncertaintyConfig()
    for ch in _CH + ("zeff",):
        assert getattr(u, f"{ch}_scalar_sigma") is None, ch
    assert u.jphi_scalar_sigma == 0.10          # unchanged


def test_a_sigma_less_run_draws_the_flattop_template(tmp_path):
    psi = np.linspace(0.0, 1.0, 51)
    bl = _bl(psi)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no fallback warning: there is no ladder
        env = resolve_uncertainty(_cfg(tmp_path), bl)
    for ch in _CH:
        np.testing.assert_allclose(env[f"sigma_{ch}"],
                                   sigma_template(psi, ch, "hmode_flattop") * _prof(bl, ch),
                                   rtol=1e-12)
        rec = env["sigma_source"][ch]
        assert rec["kind"] == "assumed" and rec["assumed_default"] == "template hmode_flattop"
    np.testing.assert_allclose(env["aux_sigmas"]["zeff"],
                               sigma_template(psi, "zeff") * _prof(bl, "Zeff"), rtol=1e-12)
    z = env["zeff_sigma_tier"]
    assert z["tier"] == "scalar" and z["assumed_default"] == "template hmode_flattop"
    assert env["sigma_source"]["zeff"]["assumed_default"] == "template hmode_flattop"
    assert env["ni_from_zeff"] is True          # the channel is on by default, as before


def test_how_much_the_default_widths_moved(tmp_path):
    """Uniform-in-psi_N means of the new default relative envelope vs the old
    flat 5 / 5 / 10 / 10 / 5 % (docs/CHANGES_SUMMARY.md)."""
    psi = np.linspace(0.0, 1.0, 2001)
    bl = _bl(psi)
    env = resolve_uncertainty(_cfg(tmp_path, UncertaintyConfig(log_sigma_sources=False)), bl)
    rel = {ch: env[f"sigma_{ch}"] / _prof(bl, ch) for ch in _CH}
    rel["zeff"] = env["aux_sigmas"]["zeff"] / _prof(bl, "Zeff")
    mean = {k: float(f"{np.trapezoid(v, psi):.2g}") for k, v in rel.items()}
    assert mean == {"ne": 0.040, "te": 0.084, "ni": 0.096, "ti": 0.097, "zeff": 0.11}


def test_a_set_scalar_wins_per_channel(tmp_path):
    psi = np.linspace(0.0, 1.2, 61)
    bl = _bl(psi)
    unc = UncertaintyConfig(ne_scalar_sigma=0.05, ti_scalar_sigma=0.0, zeff_scalar_sigma=0.07)
    env = resolve_uncertainty(_cfg(tmp_path, unc), bl)
    np.testing.assert_allclose(env["sigma_ne"], 0.05 * _prof(bl, "ne"))
    assert float(np.max(env["sigma_ti"])) == 0.0
    np.testing.assert_allclose(env["sigma_te"], sigma_template(psi, "te") * _prof(bl, "te"))
    assert env["sigma_source"]["ne"]["assumed_default"] == "scalar 0.05"
    assert env["sigma_source"]["te"]["assumed_default"] == "template hmode_flattop"
    np.testing.assert_allclose(env["aux_sigmas"]["zeff"], 0.07 * _prof(bl, "Zeff"))
    assert env["zeff_sigma_tier"]["assumed_default"] == "scalar 0.07"
    env0 = resolve_uncertainty(_cfg(tmp_path, UncertaintyConfig(zeff_scalar_sigma=0.0)), bl)
    assert "zeff" not in env0["aux_sigmas"] and env0["sigma_source"]["zeff"]["kind"] == "off"


def test_a_toroidal_flux_run_evaluates_the_template_at_psi_N(tmp_path):
    psi = np.linspace(0.0, 1.0, 51)
    phi = psi ** 1.3                                 # the run grid (Phi_N) at the nodes
    bl = _bl(phi, psi_map=(psi, phi))
    env = resolve_uncertainty(_cfg(tmp_path), bl)
    np.testing.assert_allclose(env["sigma_te"], sigma_template(psi, "te") * _prof(bl, "te"),
                               rtol=1e-10)
    assert not np.allclose(env["sigma_te"], sigma_template(phi, "te") * _prof(bl, "te"))


@pytest.mark.parametrize("sha", ["554edfb", "d874822"])
def test_a_stored_config_keeps_its_explicit_scalars(sha):
    with open(os.path.join(_STORED, f"config_{sha}.json")) as fh:
        d = json.load(fh)["legacy"]
    stored = d["uncertainty"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = BouquetConfig.from_dict(d).uncertainty
    for ch in _CH + ("zeff",):
        assert getattr(u, f"{ch}_scalar_sigma") == stored[f"{ch}_scalar_sigma"], ch
    assert (u.ne_scalar_sigma, u.te_scalar_sigma, u.ni_scalar_sigma, u.ti_scalar_sigma,
            u.zeff_scalar_sigma) == (0.05, 0.05, 0.10, 0.10, 0.05)
    # and the defaults round-trip as None
    c = BouquetConfig.from_dict(BouquetConfig(
        source=ReconstructionSource(geqdsk_path="g", profiles_path="p.cdf", time=1.0),
        solver=SolverConfig(mesh_path="m"), output_header="o").to_dict())
    assert c.uncertainty.te_scalar_sigma is None and c.uncertainty.zeff_scalar_sigma is None


@pytest.mark.parametrize("bad", [-0.1, float("nan"), "0.05", True])
def test_a_malformed_scalar_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="te_scalar_sigma"):
        _cfg(tmp_path, UncertaintyConfig(te_scalar_sigma=bad)).validate()
