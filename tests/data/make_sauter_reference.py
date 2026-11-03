"""Freeze OMFIT's Sauter bootstrap on synthetic profiles: the independent
reference of ``tests/test_sauter_bootstrap.py``.

    python tests/data/make_sauter_reference.py      # needs omfit_classes

Writes ``tests/data/sauter_omfit_reference.json``: the inputs (profiles,
geometry, gradients, and the collisionalities and Zeff OMFIT used) and
OMFIT's ``sauter_bootstrap(version="jboot1")`` output, its three bracket
terms and ``alpha``.  The profiles span Zeff 1.2-3.5 and nu_e* from the
banana regime to ~10 (a cold dense edge), so every coefficient fit is
exercised.  Synthetic numbers only; no device data.
"""
import json
import os

import numpy as np


def profiles(n=41):
    x = np.linspace(0.02, 0.98, n)
    ped = 0.5 * (1.0 - np.tanh((x - 0.93) / 0.025))
    te = 30.0 + 6000.0 * (1.0 - x ** 2) ** 1.5 * (0.3 + 0.7 * ped)
    ti = 30.0 + 5000.0 * (1.0 - x ** 2) ** 1.2 * (0.3 + 0.7 * ped)
    ne = 1.0e19 * (2.0 + 6.0 * (1.0 - 0.5 * x ** 2) * ped)
    zeff = 1.2 + 2.3 * x ** 3
    z_imp = 10.0
    ni = ne * (z_imp - zeff) / (z_imp - 1.0)
    q = 1.0 + 3.0 * x ** 2
    eps = 0.32 * np.sqrt(x)
    ft = 1.0 - (1.0 - eps) ** 2 / (np.sqrt(1.0 - eps ** 2)
                                   * (1.0 + 1.46 * np.sqrt(eps)))
    R, F = 1.7, 3.4
    psi_range = 0.8                           # Wb/rad, axis to edge
    d = lambda y: np.gradient(y, x, edge_order=2) / psi_range
    return dict(psi_N=x, Te=te, Ti=ti, ne=ne, ni=ni, Zeff=zeff, q=q, eps=eps,
                fT=ft, R=np.full_like(x, R), I_psi=np.full_like(x, F),
                psiraw=x * psi_range, dT_e_dpsi=d(te), dT_i_dpsi=d(ti),
                dn_e_dpsi=d(ne), dn_i_dpsi=d(ni))


def build():
    """The reference dict (needs omfit_classes)."""
    from omfit_classes.utils_fusion import (nclass_conductivity,
                                            sauter_bootstrap)
    import scipy.constants as sc
    P = profiles()
    one = lambda a: np.asarray(a, float)[None]
    pe, pi = sc.e * P["ne"] * P["Te"], sc.e * P["ni"] * P["Ti"]
    common = dict(psi_N=P["psi_N"], Te=one(P["Te"]), Ti=one(P["Ti"]),
                  ne=one(P["ne"]), q=one(P["q"]), eps=one(P["eps"]),
                  R=one(P["R"]), fT=one(P["fT"]), nis=[one(P["ni"])],
                  Zis=[1.0], Zeff=one(P["Zeff"]),
                  charge_number_to_use_in_ion_collisionality="Koh",
                  charge_number_to_use_in_ion_lnLambda="Zavg")
    info = nclass_conductivity(return_info_pack=True, **common)
    pkg = sauter_bootstrap(p=one(pe + pi), gEQDSKs=[None],
                           psiraw=one(P["psiraw"]), I_psi=one(P["I_psi"]),
                           nt=1, version="jboot1", return_units=False,
                           return_package=True,
                           dT_e_dpsi=one(P["dT_e_dpsi"]),
                           dT_i_dpsi=one(P["dT_i_dpsi"]),
                           dn_e_dpsi=one(P["dn_e_dpsi"]),
                           dnis_dpsi=[one(P["dn_i_dpsi"])], **common)
    flat = lambda a: np.asarray(getattr(a, "magnitude", a), float).ravel()
    out = dict(
        source="omfit_classes.utils_fusion.sauter_bootstrap(version='jboot1'),"
               " Koh nu_i* charge, Zavg ion lnLambda",
        inputs={k: flat(v).tolist() for k, v in P.items()},
        pe=flat(pe).tolist(), pi=flat(pi).tolist(),
        Zeff_used=flat(info["Zeff"]).tolist(),
        nu_e_star=flat(info["electron_collisionality"]).tolist(),
        nu_i_star=flat(info["ion_collisionality"]).tolist(),
        j_dot_B=flat(pkg["bootstrap_current_result"]).tolist(),
        term1=flat(pkg["term1"]).tolist(), term2=flat(pkg["term2"]).tolist(),
        term3=flat(pkg["term3"]).tolist(), alpha=flat(pkg["alpha"]).tolist(),
        alpha0=flat(pkg["alpha0"]).tolist())
    return out


def main():
    out = build()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sauter_omfit_reference.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, "nu_e* range",
          f"{min(out['nu_e_star']):.3g}..{max(out['nu_e_star']):.3g}")


if __name__ == "__main__":
    main()
