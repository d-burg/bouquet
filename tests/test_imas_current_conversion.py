"""read_imas_baseline converts FUSE currents exactly to TokaMaker jphi.

Synthetic FUSE-shaped dd: two equilibrium slices, core_profiles.j_tor built
from j_total with the EARLIER slice's geometry (A6, as FUSE does on a
time-dependent run), a bootstrap and a beam source.  Checks the slice pairing,
the exact conversions (docs/current-conventions.md A5-A7), that the components
sum to j_phi, the jphi_diff anchor, and the averages a non-FUSE dd omits being
traced from profiles_2d (else an error).
"""
import json

import numpy as np
import pytest

from _imas_geometry import current_geom
from bouquet.config import ImasSource
from bouquet.io.imas import NBI_SOURCE_INDEX, read_imas_baseline
from bouquet.physics import (jpar_to_jphi_tokamaker, jphi_tokamaker_pressure_term,
                             jphi_tokamaker_to_jtor_imas, jtor_imas_to_jphi_tokamaker)

_EC = 1.602176634e-19
_N = 41
_RHO = np.linspace(0.0, 1.0, _N)            # shared by eq and cp: knots, no interp
_PSI = np.linspace(-2.4, -0.5, _N)          # COCOS 11, increasing outward
_B0 = [-1.92, -1.95]                        # per equilibrium slice


def _eq_geometry(shift):
    """FUSE-named equilibrium profiles; ``shift`` distinguishes the slices."""
    x = _RHO
    return {
        "rho_tor_norm": x.tolist(),
        "f": (-3.31 + 0.07 * x**2 + shift).tolist(),
        "gm8": (1.74 - 0.30 * x**2 + shift).tolist(),           # <R>
        "gm9": (0.573 + 0.13 * x**2 - shift / 10).tolist(),     # <1/R>
        "gm1": ((0.573 + 0.13 * x**2) ** 2 * (1 + 0.06 * x**2)).tolist(),
        "gm5": (3.60 + 1.9 * x**3 + shift).tolist(),            # <B^2>
        "dpressure_dpsi": (-3.0e4 * x * (1 - 0.5 * x)
                           - 2.0e4 * np.exp(-((x - 0.95) / 0.03) ** 2)).tolist(),
    }


def _dd(with_geometry=True):
    x = _RHO
    ne = 5e19 * (1 - 0.7 * x**2) + 1e18
    te = 3e3 * (1 - 0.9 * x**2) + 50
    ni, ti = 0.9 * ne, te.copy()
    nC = (ne - ni) / 6.0
    p_eq = _EC * (ne * te + ni * ti + nC * ti)
    j_ohm = 1.4e6 * (1 - x**2) ** 1.5 + 1e4                  # <J.B>/B0
    j_bs = 6e5 * np.exp(-((x - 0.93) / 0.05) ** 2) + 1e3
    j_nbi = 8e4 * np.exp(-(x / 0.4) ** 2)
    j_total = j_ohm + j_bs + j_nbi
    eq_p1 = [_eq_geometry(0.0), _eq_geometry(0.01)]
    paired = current_geom(eq_p1[0], _B0[0])                          # the EARLIER slice
    j_tor = jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(j_total, paired) + jphi_tokamaker_pressure_term(paired),
        paired)                                               # A6, as FUSE writes it
    slices = []
    for k, t in enumerate((0.98, 1.0)):
        p1 = {"psi": _PSI.tolist(), "pressure": p_eq.tolist(),
              "j_tor": (j_tor * (1.0 + 0.02 * k * x)).tolist()}
        if with_geometry:
            p1.update(eq_p1[k])
        slices.append({"time": t, "profiles_1d": p1,
                       "global_quantities": {"ip": 1.3e6, "li_3": 0.9}})
    sp = lambda n_, t_, z: {"density_thermal": n_.tolist(), "temperature": t_.tolist(),
                            "element": [{"z_n": z}]}
    grid = {"psi": _PSI.tolist(), "rho_tor_norm": x.tolist()}
    return {
        "equilibrium": {"time": [0.98, 1.0],
                        "vacuum_toroidal_field": {"r0": 1.69, "b0": _B0},
                        "time_slice": slices},
        "core_profiles": {"time": [1.0], "profiles_1d": [{
            "time": 1.0, "grid": grid,
            "j_total": j_total.tolist(), "j_tor": j_tor.tolist(),
            "j_ohmic": j_ohm.tolist(), "j_bootstrap": j_bs.tolist(),
            "electrons": {"density_thermal": ne.tolist(), "temperature": te.tolist()},
            "ion": [sp(ni, ti, 1.0), sp(nC, ti, 6.0)]}]},
        "core_sources": {"time": [1.0], "source": [
            {"identifier": {"index": NBI_SOURCE_INDEX, "name": "beam"},
             "profiles_1d": [{"time": 1.0, "j_parallel": j_nbi.tolist()}]}]},
    }, dict(j_total=j_total, j_tor=j_tor, j_bs=j_bs, j_nbi=j_nbi)


def _read(tmp_path, dd):
    path = tmp_path / "dd.json"
    path.write_text(json.dumps(dd))
    return read_imas_baseline(ImasSource(ids_path=str(path), time=1.0),
                              p_fast_reduction="sum")


def test_reader_converts_exactly_on_the_paired_slice(tmp_path, capsys):
    dd, raw = _dd()
    bl = _read(tmp_path, dd)
    out = capsys.readouterr().out
    assert "equilibrium t=0.9800" in out                  # the earlier slice won
    g = current_geom(dd["equilibrium"]["time_slice"][0]["profiles_1d"], _B0[0])
    pt = jphi_tokamaker_pressure_term(g)
    assert np.allclose(bl.j_phi, jtor_imas_to_jphi_tokamaker(raw["j_tor"], g), rtol=1e-12)
    # the bootstrap is field-aligned only; p'G is the third bucket (D2)
    assert np.allclose(bl.j_BS, jpar_to_jphi_tokamaker(raw["j_bs"], g), rtol=1e-12)
    assert np.allclose(bl.j_pressure, pt, rtol=1e-12)
    assert np.max(np.abs(pt)) > 1e-2 * np.max(np.abs(bl.j_BS))
    assert np.allclose(bl.j_NBI, jpar_to_jphi_tokamaker(raw["j_nbi"], g), rtol=1e-12)
    # the total's own parallel current closes on j_phi (j_tor came from it)
    assert np.allclose(jpar_to_jphi_tokamaker(raw["j_total"], g) + pt, bl.j_phi,
                       rtol=1e-12)
    # J_TM != J_IMAS: the conversion is not vacuous on this geometry
    assert np.max(np.abs(bl.j_phi / raw["j_tor"] - 1)) > 1e-2


def test_reader_components_sum_to_j_phi(tmp_path):
    dd, _ = _dd()
    bl = _read(tmp_path, dd)
    total = bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF
    assert np.allclose(total, bl.j_phi, rtol=0, atol=1e-9 * np.max(np.abs(bl.j_phi)))
    # the solve split's inductive residual is the ohmic <J.B>'s field-aligned
    # image plus the pressure-driven p'G it carries for the solve; minus
    # j_pressure it is the ohmic alone (the archived third-bucket split)
    g = current_geom(dd["equilibrium"]["time_slice"][0]["profiles_1d"], _B0[0])
    j_ohm = np.asarray(dd["core_profiles"]["profiles_1d"][0]["j_ohmic"])
    assert np.allclose(bl.j_inductive - bl.j_pressure,
                       jpar_to_jphi_tokamaker(j_ohm, g), rtol=1e-10)
    assert (bl.li_metrics["imas_current_conversion"]["current_split_convention"]
            == "pressure_in_inductive")


def test_jphi_diff_uses_the_anchor_slice_own_geometry(tmp_path):
    dd, _ = _dd()
    bl = _read(tmp_path, dd)
    p1 = dd["equilibrium"]["time_slice"][1]["profiles_1d"]   # nearest to T
    eq_jphi = jtor_imas_to_jphi_tokamaker(np.asarray(p1["j_tor"]), current_geom(p1, _B0[1]))
    assert np.allclose(bl.jphi_diff, eq_jphi - bl.j_phi, rtol=1e-12, atol=1e-6)


def test_dd_without_geometry_or_profiles_2d_falls_back_to_the_ratio(tmp_path):
    """A dd without the averages (and no profiles_2d to trace them) is read
    with the pre-PR #64 per-surface ratio c = j_tor/j_total, with a warning
    and a stamp -- not refused (review PR64 B7)."""
    dd, raw = _dd(with_geometry=False)
    with pytest.warns(UserWarning, match="FALLING BACK"):
        bl = _read(tmp_path, dd)
    conv = bl.li_metrics["imas_current_conversion"]
    assert conv["method"].startswith("ratio")
    assert "profiles_2d" in conv["reason"]
    c = raw["j_tor"] / raw["j_total"]
    np.testing.assert_allclose(bl.j_phi, raw["j_tor"], rtol=1e-15)
    np.testing.assert_allclose(bl.j_BS, c * raw["j_bs"], rtol=1e-14)
    np.testing.assert_allclose(bl.j_NBI, c * raw["j_nbi"], rtol=1e-14)
    assert bl.j_pressure is None
    total = bl.j_inductive + bl.j_BS + bl.j_NBI + bl.j_RF
    np.testing.assert_allclose(total, bl.j_phi, rtol=0,
                               atol=1e-9 * np.max(np.abs(bl.j_phi)))


# Concentric circular surfaces, psi ~ r^2: weight dl/Bp ~ R dtheta, so
# <1/R> = 1/R0, <R> = R0 + r^2/(2R0), <1/R^2> = 1/(R0 sqrt(R0^2 - r^2)),
# <B^2> = (F^2 + c^2 r^2)<1/R^2> with Bp = c r/R (COCOS 11), c = dpsi/(pi a^2).
_R0, _A, _F = 1.7, 0.6, -3.4
_SQRT_PSIN = np.sqrt((_PSI - _PSI[0]) / (_PSI[-1] - _PSI[0]))   # r/a; rho_tor_norm at constant q


def _circular_slice(n2=65):
    R = np.linspace(0.9, 2.5, n2)
    Z = np.linspace(-0.8, 0.8, n2)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    dpsi = _PSI[-1] - _PSI[0]
    th = np.linspace(0, 2 * np.pi, 201)[:-1]
    r = _A * _SQRT_PSIN
    c = dpsi / (np.pi * _A**2)
    inv_R2 = 1 / (_R0 * np.sqrt(_R0**2 - r**2))
    exact = {"gm9": np.full(_N, 1 / _R0), "gm8": _R0 + r**2 / (2 * _R0),
             "gm1": inv_R2, "gm5": (_F**2 + c**2 * r**2) * inv_R2}
    ts = {"profiles_1d": {"psi": _PSI.tolist(), "f": [_F] * _N,
                          "q": [2.0] * _N},       # constant q: rho = sqrt(psi_N)
          "profiles_2d": [{"grid_type": {"index": 1},
                           "grid": {"dim1": R.tolist(), "dim2": Z.tolist()},
                           "psi": (_PSI[0] + dpsi * ((RR - _R0)**2 + ZZ**2)
                                   / _A**2).tolist()}],
          "global_quantities": {"magnetic_axis": {"r": _R0, "z": 0.0}},
          "boundary": {"outline": {"r": (_R0 + _A * np.cos(th)).tolist(),
                                   "z": (_A * np.sin(th)).tolist()}}}
    return ts, exact


def test_averages_from_profiles_2d_match_the_analytic_circle():
    from bouquet.io.imas import _fsa_from_profiles_2d
    ts, exact = _circular_slice()
    got = _fsa_from_profiles_2d(ts)
    for k, v in exact.items():
        np.testing.assert_allclose(got[k], v, rtol=1e-4, err_msg=k)


def test_reader_computes_missing_averages_from_profiles_2d(tmp_path):
    from scipy.interpolate import CubicSpline
    ts, exact = _circular_slice()
    dd, raw = _dd()
    for s in dd["equilibrium"]["time_slice"]:          # the FUSE-complete dd
        p1 = s["profiles_1d"]
        p1.update({k: v.tolist() for k, v in exact.items()},
                  f=[_F] * _N, rho_tor_norm=_SQRT_PSIN.tolist(),
                  dpressure_dpsi=CubicSpline(_PSI, p1["pressure"]).derivative()(_PSI).tolist())
    dd["core_profiles"]["profiles_1d"][0]["grid"]["rho_tor_norm"] = _SQRT_PSIN.tolist()
    g = current_geom(dd["equilibrium"]["time_slice"][0]["profiles_1d"], _B0[0])
    dd["core_profiles"]["profiles_1d"][0]["j_tor"] = jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(raw["j_total"], g) + jphi_tokamaker_pressure_term(g),
        g).tolist()
    ref = _read(tmp_path, dd)
    for s in dd["equilibrium"]["time_slice"]:          # what a non-FUSE dd has
        for k in ("gm1", "gm5", "gm8", "gm9", "rho_tor_norm", "dpressure_dpsi"):
            s["profiles_1d"].pop(k)
        s["profiles_1d"]["q"] = ts["profiles_1d"]["q"]
        s.update({k: ts[k] for k in ("profiles_2d", "boundary")})
        s["global_quantities"].update(ts["global_quantities"])
    bl = _read(tmp_path, dd)
    pk = np.max(np.abs(ref.j_phi))
    for name in ("j_phi", "j_BS", "j_NBI", "j_inductive", "jphi_diff"):
        assert np.max(np.abs(getattr(bl, name) - getattr(ref, name))) < 2e-4 * pk, name


def test_pairing_prefers_the_slice_that_reproduces_j_tor(tmp_path, capsys):
    # if j_tor came from the NEAREST slice instead, that slice is chosen
    dd, raw = _dd()
    g1 = current_geom(dd["equilibrium"]["time_slice"][1]["profiles_1d"], _B0[1])
    jt = jphi_tokamaker_to_jtor_imas(
        jpar_to_jphi_tokamaker(raw["j_total"], g1) + jphi_tokamaker_pressure_term(g1), g1)
    dd["core_profiles"]["profiles_1d"][0]["j_tor"] = jt.tolist()
    bl = _read(tmp_path, dd)
    assert "equilibrium t=1.0000" in capsys.readouterr().out
    assert np.allclose(bl.j_phi, jtor_imas_to_jphi_tokamaker(jt, g1), rtol=1e-12)


if __name__ == "__main__":                                   # pragma: no cover
    pytest.main([__file__, "-q"])
