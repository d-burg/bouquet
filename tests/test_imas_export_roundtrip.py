"""IDS export round trip: ``write_imas_draw`` -> ``IdsAdapter.read``.

The exporter writes IMAS parallel currents (``<j.B>/B0``).  The engine's
IDS adapter reads them back as ``<j.B>`` parts and composes
``<j_phi> = kappa <j.B> + P`` with the pressure-driven term
``P = p'(<R> - F^2<1/R>/<B^2>)`` recovered from the pressure.  An archived
toroidal ``j_BS`` CARRIES ``P``; the exporter takes it off before
converting, so no exported parallel current counts it.  These tests build a fake draw archive whose
``<j.B>`` parts are known, export it into the synthetic IMAS example, read
the export with the reader + adapter, and require the parts and the
composed ``<j_phi>`` back to 1e-9 relative:

* an ENGINE archive (the draw's ``jB_parallel/`` subgroup, schema) --
  written from the stored parts, no conversion;
* a LEGACY-style archive (no ``jB_parallel/``; ``P`` in ``j_BS``) --
  ``P`` from the archived eqdsk subtracted before the ``eq_fsa``
  conversion.

Solver-free.
"""
import json
import os
import sys
import warnings

import h5py
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from test_engine_adapters import _MESH, _OMAS, _TIME, _ids_contract  # noqa: E402

_GOLD = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
_GP = "scan/0/0"
RTOL = 1e-9

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_OMAS) and os.path.isfile(_GOLD)),
    reason="synthetic IMAS example or golden archive absent")


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(scope="module")
def source():
    """The example's contract (grid, |B0|, driven parts) and a draw's
    archived eqdsk + captured eq_fsa from the synthetic golden archive."""
    _ad, c, bl = _ids_contract(_OMAS)
    with h5py.File(_GOLD, "r") as hf:
        eqb = bytes(hf[_GP]["eqdsk"][()])
        fsa = {k: np.asarray(hf[_GP]["eq_fsa"][k][()], dtype=float)
               for k in hf[_GP]["eq_fsa"]}
    from bouquet.physics import field_aligned_conversion
    psi = np.asarray(c.psi_N, dtype=float)
    kap = field_aligned_conversion(
        np.interp(psi, fsa["psi_N"], fsa["F"]),
        np.interp(psi, fsa["psi_N"], fsa["avg_inv_R"]),
        np.interp(psi, fsa["psi_N"], fsa["avg_B2"]))
    jB = dict(
        ind=np.asarray(c.jB_ind, dtype=float) * 1.03,      # the draw's own
        bs=1.5e5 * np.exp(-((psi - 0.93) / 0.05) ** 2) + 2.0e4 * psi,
        nbi=np.asarray(c.jB_fix_parts["nbi"], dtype=float),
        rf=np.asarray(c.jB_fix_parts["rf"], dtype=float)
        + np.asarray(c.jB_fix_parts["other"], dtype=float))
    return dict(c=c, bl=bl, psi=psi, eqb=eqb, fsa=fsa, kap=kap, jB=jB)


def _write_archive(path, s, j_phi, j_ind, j_bs, parallel=None):
    from bouquet.schema import write_jB_parallel
    bl = s["bl"]
    with h5py.File(path, "w") as hf:
        g = hf.require_group(_GP)
        g.create_dataset("psi_N", data=s["psi"])
        g.create_dataset("psi_N_kinetic",
                         data=np.asarray(bl.psi_N_kinetic, dtype=float))
        for k, a in (("n_e", bl.ne), ("T_e", bl.te), ("n_i", bl.ni),
                     ("T_i", bl.ti)):
            g.create_dataset(k, data=np.asarray(a, dtype=float))
        g.create_dataset("j_phi", data=j_phi)
        g.create_dataset("j_inductive", data=j_ind)
        g.create_dataset("j_BS", data=j_bs)
        g.create_dataset("eqdsk", data=np.void(s["eqb"]))
        fg = g.create_group("eq_fsa")
        for k, v in s["fsa"].items():
            fg.create_dataset(k, data=v)
        if parallel is not None:
            write_jB_parallel(g, parallel)


def _export_and_read(tmp_path, arc):
    """Export draw 0 into the example at its slice; read it back with the
    reader + the engine's IDS adapter (the default residual inductive)."""
    from bouquet.adapters import IdsAdapter
    from bouquet.baseline import resolve_baseline
    from bouquet.config import (BouquetConfig, GenerationConfig, ImasSource,
                                SolverConfig)
    from bouquet.io.imas import write_imas_draw
    out = str(tmp_path / "draw0.json")
    write_imas_draw(arc, 0, _OMAS, out, scan_key=0, time=_TIME)
    # the exported equilibrium carries no j_tor (write_imas_draw docstring)
    cfg = BouquetConfig(
        source=ImasSource(ids_path=out, time=_TIME),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified",
                                    anchor_jtor_to_equilibrium=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bl = resolve_baseline(cfg, None)
        c = IdsAdapter(cfg.source, cfg, bl).read()
    with open(out) as fh:
        cp = json.load(fh)["core_profiles"]["profiles_1d"]
    ic = int(np.argmin(np.abs(np.asarray(
        json.load(open(out))["core_profiles"]["time"]) - _TIME)))
    sgn = float(c.signs["current_sign"])
    jB_bs_read = sgn * c.provenance["B0"] * np.asarray(
        cp[ic]["j_bootstrap"], dtype=float)
    return c, jB_bs_read, cp[ic]


def _check(s, c, jB_bs_read, j_phi, P):
    jB = s["jB"]
    assert _rel(c.jB_ind, jB["ind"]) <= RTOL
    assert _rel(jB_bs_read, jB["bs"]) <= RTOL
    assert _rel(c.jB_fix_parts["nbi"], jB["nbi"]) <= RTOL
    assert _rel(c.jB_fix, jB["nbi"] + jB["rf"]) <= RTOL
    # the composed <j_phi> of what was read IS the archived <j_phi>
    composed = s["kap"] * (c.jB_ind + jB_bs_read + c.jB_fix) + P
    assert _rel(composed, j_phi) <= RTOL


def test_engine_archive_round_trips_through_the_ids_adapter(source,
                                                            tmp_path):
    s = source
    kap, jB, psi = s["kap"], s["jB"], s["psi"]
    P = 3.0e4 * psi * (1.0 - psi) + 1.0e3     # the archived state's P
    fix = jB["nbi"] + jB["rf"]
    j_phi = kap * (jB["ind"] + jB["bs"] + fix) + P
    j_bs = kap * jB["bs"] + P                 # the split's j_BS carries P
    j_ind = j_phi - j_bs - kap * fix
    arc = str(tmp_path / "engine.h5")
    _write_archive(arc, s, j_phi, j_ind, j_bs, parallel=dict(
        psi_N=psi, jB_inductive=jB["ind"], jB_BS=jB["bs"], jB_NBI=jB["nbi"],
        jB_RF=jB["rf"], kappa=kap, j_pressure=P))
    c, jbs, _ = _export_and_read(tmp_path, arc)
    _check(s, c, jbs, j_phi, P)


def test_legacy_archive_export_subtracts_the_pressure_driven_term(source,
                                                                  tmp_path):
    from bouquet.io.imas import archived_pressure_term
    s = source
    kap, jB, psi = s["kap"], s["jB"], s["psi"]
    # P of the archived eqdsk -- what the exporter subtracts; the draw's
    # j_BS carries it
    P = archived_pressure_term(s["eqb"], psi)
    assert np.max(np.abs(P)) > 1e-2 * np.max(np.abs(kap * jB["ind"]))
    fix = jB["nbi"] + jB["rf"]
    j_ind = kap * jB["ind"]
    j_bs = kap * jB["bs"] + P
    j_phi = j_ind + j_bs + kap * fix
    arc = str(tmp_path / "legacy.h5")
    _write_archive(arc, s, j_phi, j_ind, j_bs)
    c, jbs, _ = _export_and_read(tmp_path, arc)
    _check(s, c, jbs, j_phi, P)


def test_jB_parallel_identity_is_what_the_schema_states(tmp_path):
    """write/read of the subgroup: every key, its unit, the identity attr;
    a missing part is refused."""
    import bouquet.schema as sc
    n = 5
    parts = {k: np.linspace(1.0, 2.0, n) for k in sc.JB_PARALLEL_UNITS}
    with h5py.File(tmp_path / "a.h5", "w") as hf:
        g = hf.create_group("d")
        sc.write_jB_parallel(g, parts)
        back = sc.read_jB_parallel(g)
        assert set(back) == set(sc.JB_PARALLEL_UNITS)
        assert g[sc.JB_PARALLEL_GROUP]["jB_BS"].attrs["units"] == "T A m^-2"
        assert "j_pressure" in g[sc.JB_PARALLEL_GROUP].attrs["identity"]
        bad = dict(parts)
        bad.pop("j_pressure")
        with pytest.raises(ValueError, match="j_pressure"):
            sc.write_jB_parallel(g, bad)
        assert sc.read_jB_parallel(hf.create_group("legacy")) is None
