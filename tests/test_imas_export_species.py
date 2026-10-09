"""An exported draw writes every thermal species its solve used.

bouquet's solves (engine, legacy, swb) carry ONE species model: electrons,
a hydrogenic main ion, one effective impurity of charge ``Z_imp`` at the
main-ion ``T_i`` with ``n_z = (n_e - z_fast - n_i) / Z_imp``, and the fast
population held fixed.  ``write_imas_draw`` wrote the draw's electrons and
main ion but kept the TEMPLATE's impurity, so whenever a draw's ``n_i`` /
``Z_eff`` / ``T_i`` differ from the template's -- every ida_hybrid draw, whose
kinetics are IDA's, not the dd's -- the export was not quasineutral, its
``Z_eff`` was not the drawn one, and the default reader refused it ("thermal
species gap ... > 2 %").  The writer now writes the solve's impurity
(:func:`bouquet.io.imas._write_draw_ion_species`).

Synthetic inputs only: the shipped D3D-like OMAS example with a beam density
on D, and a synthetic single-slice IDA file of a DIFFERENT plasma (its carbon
x1.4, its T_i x1.1), as the IDA and dd plasmas differ on real data.  No
solver: the draw archive is written directly (a Z_eff-primary draw of the
ida_hybrid baseline, the sampler's density rule).
"""
import json
import os
import warnings

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(os.path.dirname(_HERE), "examples", "D3D-like")
_OMAS = os.path.join(_EX, "D3Dlike_baseline_omas.json")
_GOLD = os.path.join(_HERE, "golden", "D3Dlike_Hmode_golden_slim.h5")
_GP = "scan/0/0"
T = 2.2
Z = 6.0
_EC = 1.602176634e-19

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_OMAS) and os.path.isfile(_GOLD)),
    reason="synthetic IMAS example or golden archive absent")


def _inputs(work, carbon=1.4, ti=1.1, fast=0.15):
    """dd with a beam on D (FUSE's carving: total D held) + an IDA file of
    another plasma (carbon and T_i scaled)."""
    with open(_OMAS) as fh:
        dd = json.load(fh)
    for p in dd["core_profiles"]["profiles_1d"]:
        psi = np.asarray(p["grid"]["psi"], float)
        pn = (psi - psi[0]) / (psi[-1] - psi[0])
        d = next(i for i in p["ion"] if i["label"] == "D")
        n_tot = np.asarray(d["density_thermal"], float)
        nf = fast * n_tot * (1.0 - pn ** 2) ** 2
        d["density_thermal"], d["density_fast"] = (n_tot - nf).tolist(), \
            nf.tolist()
    ddp = os.path.join(work, "dd.json")
    with open(ddp, "w") as fh:
        json.dump(dd, fh)
    ic = int(np.argmin(np.abs(np.asarray(dd["core_profiles"]["time"]) - T)))
    p = dd["core_profiles"]["profiles_1d"][ic]
    psi = np.asarray(p["grid"]["psi"], float)
    pn = (psi - psi[0]) / (psi[-1] - psi[0])
    ne = np.asarray(p["electrons"]["density_thermal"], float)
    te = np.asarray(p["electrons"]["temperature"], float)
    d = next(i for i in p["ion"] if i["label"] == "D")
    c = next(i for i in p["ion"] if i["label"] != "D")
    ni_tot = (np.asarray(d["density_thermal"], float)
              + np.asarray(d["density_fast"], float))
    nc = carbon * np.asarray(c["density_thermal"], float)
    tc = ti * np.asarray(c["temperature"], float)
    zeff = (ni_tot + Z ** 2 * nc) / ne
    cdf = os.path.join(work, "ida.cdf")
    with h5py.File(cdf, "w") as f:
        f["time"] = np.array([1e3 * T])
        f["psi_n"] = pn
        for k, v, e in (("n_e", ne, 0.04), ("T_e", te, 0.05),
                        ("T_12C6", tc, 0.06), ("Zeff", zeff, 0.08),
                        ("n_12C6", nc, 0.15)):
            f[k] = v[None, :]
            f[k + "_err"] = (e * v)[None, :]
    return ddp, cdf


def _read(path, **kw):
    from bouquet.config import ImasSource
    from bouquet.io.imas import read_imas_baseline
    src = {k: kw.pop(k) for k in ("ida_path", "impurity_Z") if k in kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return read_imas_baseline(ImasSource(ids_path=str(path), time=T,
                                             **src),
                                  anchor_jtor_to_equilibrium=False, **kw)


def _draw(bl):
    """A Z_eff-primary draw of the ida_hybrid baseline: n_e and T_i moved,
    Z_eff drawn, n_i DERIVED (the sampler's rule), n_z implied."""
    from bouquet.physics import main_ion_density_from_zeff
    x = np.asarray(bl.psi_N_kinetic, float)
    ne = np.asarray(bl.ne, float) * (1.0 + 0.03 * np.exp(-(x / 0.4) ** 2))
    ti = np.asarray(bl.ti, float) * 0.97
    zeff = np.asarray(bl.Zeff, float) * (1.0 + 0.05 * (1.0 - x ** 2))
    ni = main_ion_density_from_zeff(ne, zeff, bl.Z_imp, bl.z_fast,
                                    bl.z2_fast, zeff_includes_fast=True)
    return dict(ne=ne, te=np.asarray(bl.te, float), ni=np.asarray(ni, float),
                ti=ti, zeff=zeff)


def _archive(path, bl, d):
    with h5py.File(_GOLD, "r") as hf:
        eqb = bytes(hf[_GP]["eqdsk"][()])
        fsa = {k: np.asarray(hf[_GP]["eq_fsa"][k][()], dtype=float)
               for k in hf[_GP]["eq_fsa"]}
    with h5py.File(path, "w") as hf:
        b = hf.require_group("scan/0/_baseline")
        b.attrs["Z_imp"] = float(bl.Z_imp)
        g = hf.require_group(_GP)
        g.create_dataset("psi_N", data=np.asarray(bl.psi_N, float))
        g.create_dataset("psi_N_kinetic",
                         data=np.asarray(bl.psi_N_kinetic, float))
        for k, a in (("n_e", d["ne"]), ("T_e", d["te"]), ("n_i", d["ni"]),
                     ("T_i", d["ti"]), ("aux_zeff", d["zeff"]),
                     ("z_fast", bl.z_fast), ("z2_fast", bl.z2_fast)):
            g.create_dataset(k, data=np.asarray(a, float))
        g.attrs["Z_imp"] = float(bl.Z_imp)
        for k in ("j_phi", "j_inductive", "j_BS"):
            g.create_dataset(k, data=np.asarray(getattr(bl, k), float))
        g.create_dataset("eqdsk", data=np.void(eqb))
        fg = g.create_group("eq_fsa")
        for k, v in fsa.items():
            fg.create_dataset(k, data=v)


def _species_gap(cp, bl):
    """The reader's thermal species gap (_validate_pressure_completeness)
    of the exported slice *cp* as re-read into *bl*."""
    from bouquet.physics import impurity_pressure
    ne = np.asarray(cp["electrons"]["density_thermal"], float)
    te = np.asarray(cp["electrons"]["temperature"], float)
    full = _EC * ne * te
    for ion in cp["ion"]:
        full = full + _EC * (np.asarray(ion["density_thermal"], float)
                             * np.asarray(ion["temperature"], float))
    ne_th = np.maximum(ne - np.asarray(bl.z_fast, float), 0.0)
    recon = _EC * (ne * te + bl.ni * bl.ti) + impurity_pressure(
        ne_th, bl.ni, bl.ti, bl.Z_imp)
    return float(np.mean(np.abs(full - recon)) / np.mean(np.abs(full)))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from bouquet.io.imas import write_imas_draw
    work = str(tmp_path_factory.mktemp("ida_export"))
    ddp, cdf = _inputs(work)
    bl = _read(ddp, ida_path=cdf, impurity_Z=Z, kinetic_source="ida_hybrid")
    assert bl.Z_imp == Z and bl.zeff_includes_fast
    d = _draw(bl)
    arc = os.path.join(work, "draw.h5")
    _archive(arc, bl, d)
    out = os.path.join(work, "draw0.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_imas_draw(arc, 0, ddp, out, scan_key=0, time=T)
    with open(out) as fh:
        ex = json.load(fh)
    return dict(bl=bl, d=d, out=out, ex=ex, ddp=ddp, work=work)


def test_the_ida_hybrid_draw_re_reads_with_the_default_reader(exported):
    """No refusal; the draw's kinetics come back exactly (the archive is on
    the template's own nodes), its Z_eff to rounding, Z_imp is the solve's,
    and the Z_eff convention is recognised as the draw's (measured: fast
    ions in the numerator)."""
    bl, d = exported["bl"], exported["d"]
    rr = _read(exported["out"])                       # default: dd kinetics
    for k, a in (("ne", d["ne"]), ("te", d["te"]), ("ni", d["ni"]),
                 ("ti", d["ti"])):
        np.testing.assert_array_equal(getattr(rr, k), a, k)
    np.testing.assert_allclose(rr.Zeff, d["zeff"], rtol=1e-12, atol=0)
    assert rr.Z_imp == pytest.approx(Z, rel=1e-12)
    assert rr.zeff_includes_fast
    prov = rr.li_metrics["zeff_dd_provenance"]
    assert prov["convention"] == "thermal+fast"
    assert not prov["matches_neither"]
    np.testing.assert_array_equal(rr.z_fast, bl.z_fast)
    np.testing.assert_array_equal(rr.p_fast, bl.p_fast)


def test_the_export_is_quasineutral_with_no_species_gap(exported):
    from bouquet.io.imas import IMAS_EXPORT_SPECIES_KEY
    cp = exported["ex"]["core_profiles"]["profiles_1d"][0]
    d = cp["ion"]
    D = next(i for i in d if float(i["element"][0]["z_n"]) == 1.0)
    C = next(i for i in d if float(i["element"][0]["z_n"]) == Z)
    ne = np.asarray(cp["electrons"]["density_thermal"], float)
    res = (ne - np.asarray(D["density_fast"], float)
           - np.asarray(D["density_thermal"], float)
           - Z * np.asarray(C["density_thermal"], float))
    assert np.max(np.abs(res)) <= 1e-12 * np.max(ne)
    np.testing.assert_array_equal(C["temperature"], D["temperature"])
    rr = _read(exported["out"])
    assert _species_gap(cp, rr) <= 1e-12
    rec = json.loads(exported["ex"]["core_profiles"]["code"]["parameters"])[
        IMAS_EXPORT_SPECIES_KEY]
    assert rec["Z_imp"] == Z and rec["impurity"] == C["label"]
    assert rec["zeroed"] == [] and rec["impurity_relabelled_from"] is None


def test_the_templates_impurity_is_what_the_reader_refused(exported,
                                                           tmp_path):
    """Put the template's carbon back (what the writer did before): the
    default reader refuses the file, as it refused the real export."""
    with open(exported["ddp"]) as fh:
        tpl = json.load(fh)
    ic = int(np.argmin(np.abs(np.asarray(tpl["core_profiles"]["time"]) - T)))
    c0 = next(i for i in tpl["core_profiles"]["profiles_1d"][ic]["ion"]
              if float(i["element"][0]["z_n"]) == Z)
    ex = json.loads(json.dumps(exported["ex"]))
    c1 = next(i for i in ex["core_profiles"]["profiles_1d"][0]["ion"]
              if float(i["element"][0]["z_n"]) == Z)
    c1["density_thermal"] = c0["density_thermal"]
    c1["temperature"] = c0["temperature"]
    p = tmp_path / "old.json"
    p.write_text(json.dumps(ex))
    with pytest.raises(ValueError, match="thermal species gap"):
        _read(str(p))


# ---------------------------------------------------------------------------
#  the species rule on templates with other species lists
# ---------------------------------------------------------------------------
def _cp(ions, n=4):
    return {"ion": [dict(label=lb, element=[dict(z_n=z)],
                         density_thermal=[dn] * n, temperature=[1.0] * n)
                    for lb, z, dn in ions]}


def test_further_thermal_species_are_zeroed_and_named():
    from bouquet.io.imas import _write_draw_ion_species
    cp = _cp([("D", 1.0, 5.0), ("C", 6.0, 0.1), ("Ne", 10.0, 0.01),
              ("H", 1.0, 0.2)])
    ne, ni, zf = np.full(4, 7.0), np.full(4, 4.0), np.full(4, 0.4)
    rec = _write_draw_ion_species(cp, ni, np.full(4, 2.0), ne, zf, 6.0)
    by = {i["label"]: i for i in cp["ion"]}
    np.testing.assert_allclose(by["C"]["density_thermal"], (7 - 0.4 - 4) / 6)
    assert by["C"]["temperature"] == [2.0] * 4
    assert by["Ne"]["density_thermal"] == [0.0] * 4
    assert by["H"]["density_thermal"] == [0.0] * 4
    assert rec["zeroed"] == ["Ne", "H"]


def test_an_impurity_of_another_charge_is_relabelled_to_z_imp():
    from bouquet.io.imas import _write_draw_ion_species
    cp = _cp([("D", 1.0, 5.0), ("Ne", 10.0, 0.1)])
    rec = _write_draw_ion_species(cp, np.full(4, 4.0), np.full(4, 2.0),
                                  np.full(4, 7.0), None, 6.5)
    imp = cp["ion"][1]
    assert imp["element"][0]["z_n"] == 6.5
    np.testing.assert_allclose(imp["density_thermal"], 3.0 / 6.5)
    assert rec["impurity_relabelled_from"] == {"label": "Ne", "z_n": 10.0}
    # none at all: one is added
    cp = _cp([("D", 1.0, 5.0)])
    _write_draw_ion_species(cp, np.full(4, 4.0), np.full(4, 2.0),
                            np.full(4, 7.0), None, 6.0)
    assert cp["ion"][1]["element"][0]["z_n"] == 6.0


def test_an_archive_without_z_imp_keeps_the_templates_impurity():
    from bouquet.io.imas import _write_draw_ion_species
    cp = _cp([("D", 1.0, 5.0), ("C", 6.0, 0.1)])
    rec = _write_draw_ion_species(cp, np.full(4, 4.0), np.full(4, 2.0),
                                  np.full(4, 7.0), None, None)
    assert cp["ion"][1]["density_thermal"] == [0.1] * 4
    assert cp["ion"][0]["density_thermal"] == [4.0] * 4
    assert rec["Z_imp"] is None
