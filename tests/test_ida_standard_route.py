"""The default IDA route (``ni_source="standard"``) is the reader before PR #56.

Owner decision 2026-10-09: the PR #56 Z_eff / n_i combination is
EXPERIMENTAL and opt-in.  With defaults, :func:`bouquet.io.ida.read_ida`
must return what the reader returned before PR #56 (main afe1a99), bit for
bit, on the synthetic IDA fixture and on both synthetic layouts.

Two references, both judged on behaviour (the returned arrays):

* an independent recomputation of the pre-#56 formulas (always runs);
* the afe1a99 reader itself, loaded from git (skipped where git or the
  commit is unavailable).
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from bouquet.io.ida import read_ida

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "data", "diiid_profs_synthetic.cdf")
MAIN_REF = "afe1a99"

#: the IDAProfiles fields the pre-#56 reader returned
PRE56_FIELDS = ("psi_N", "ne", "te", "ni", "ti", "Zeff", "sigma_ne",
                "sigma_te", "sigma_ni", "sigma_ti", "time", "sigma_Zeff",
                "sigma_Zeff_source", "sigma_Zeff_carbon",
                "sigma_Zeff_carbon_source", "zeff_carbon_dev", "raw_bytes")


def _write_ensemble(path, nr=24, ns=128, seed=3):
    import h5py
    rng = np.random.default_rng(seed)
    psi = np.linspace(0.0, 1.1, nr)

    def s(mu, frac):
        return mu[None, :] * (1.0 + frac * rng.standard_normal((ns, nr)))
    ne = 5e19 * (1 - 0.6 * psi ** 2) + 1e18
    zf = 1.8 + 0.4 * psi
    with h5py.File(path, "w") as f:
        f["time"] = np.array([2500.0])
        f["psi_n"] = np.broadcast_to(psi, (1, ns, nr)).copy()
        f["n_e"] = s(ne, 0.05)[None]
        f["T_e"] = s(2e3 * (1 - 0.8 * psi ** 2) + 50, 0.05)[None]
        f["T_12C6"] = s(1.8e3 * (1 - 0.8 * psi ** 2) + 50, 0.05)[None]
        # a few nodes outside [1, Z] so the ni window is exercised
        zs = s(zf, 0.08)
        zs[:, -2:] = 0.9
        f["Zeff"] = zs[None]
        f["n_12C6"] = s(ne * (zf - 1) / 30.0, 0.06)[None]
    return path


def _load_main_reader():
    """afe1a99's bouquet/io/ida.py as a module of this package (its one
    import, physics.main_ion_density_from_zeff, is unchanged for the
    no-fast-ion call it makes)."""
    try:
        src = subprocess.run(
            ["git", "-C", REPO, "show", f"{MAIN_REF}:bouquet/io/ida.py"],
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {MAIN_REF} unavailable")
    name = "bouquet.io._ida_main_reference"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "bouquet.io"
    sys.modules[name] = mod
    exec(compile(src, f"{MAIN_REF}:bouquet/io/ida.py", "exec"), mod.__dict__)
    return mod


def _same(a, b, field):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert a is not None and b is not None, field
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                      err_msg=field)
        assert np.asarray(a).dtype == np.asarray(b).dtype, field
    elif isinstance(a, dict):
        assert set(a) == set(b), field
        for k in a:
            _same(a[k], b[k], f"{field}[{k}]")
    else:
        assert a == b, (field, a, b)


@pytest.fixture(params=["fixture", "ensemble"])
def ida_file(request, tmp_path):
    if request.param == "fixture":
        return FIXTURE
    return _write_ensemble(str(tmp_path / "ens.cdf"))


def test_default_is_the_pre56_formulas(ida_file):
    """Independent of git: the default reader's Z_eff, n_i and sigmas are
    the pre-#56 formulas, exactly."""
    import h5py
    p = read_ida(ida_file)
    assert p.zeff_provenance["ni_source"] == "standard"
    with h5py.File(ida_file, "r") as f:
        ens = np.asarray(f["n_e"].shape).size == 3
        if ens:
            zf = np.asarray(f["Zeff"][0], dtype=float).mean(0)
            ne = np.asarray(f["n_e"][0], dtype=float).mean(0)
        else:
            zf = np.asarray(f["Zeff"][0], dtype=float)
            ne = np.asarray(f["n_e"][0], dtype=float)
            np.testing.assert_array_equal(
                p.sigma_Zeff, np.asarray(f["Zeff_err"][0], dtype=float))
    # Z_eff is the stored VB value, NOT clipped and NOT a VB/CER mean
    np.testing.assert_array_equal(p.Zeff, zf)
    np.testing.assert_array_equal(p.ne, ne)
    Z = 6.0
    ni = ne * (Z - np.clip(zf, 1.0, Z)) / (Z - 1.0)
    np.testing.assert_array_equal(p.ni, ni)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(ne > 0, p.sigma_ne / ne, 0.0)
    np.testing.assert_array_equal(p.sigma_ni, np.abs(ni) * frac)
    assert p.sigma_Zeff_source == ("ensemble-samples" if ens else "Zeff_err")
    assert p.zeff_dne is None and p.ni_route_chi is None


def test_default_reader_is_bit_identical_to_main(ida_file):
    """Against the afe1a99 reader itself: every field it returned."""
    ref = _load_main_reader().read_ida(ida_file)
    new = read_ida(ida_file)
    for f in PRE56_FIELDS:
        _same(getattr(new, f), getattr(ref, f), f)


def test_experimental_route_still_reachable(ida_file):
    """The PR #56 route is an explicit option and differs from the default
    where VB and CER disagree."""
    std = read_ida(ida_file)
    allr = read_ida(ida_file, ni_source="all")
    assert allr.zeff_provenance["ni_source"] == "all"
    assert allr.sigma_Zeff_source.startswith("VB+CER")
    assert not np.array_equal(std.Zeff, allr.Zeff)
