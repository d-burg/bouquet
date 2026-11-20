"""The engine's bootstrap through the solver's ``eval_bootstrap`` equals its earlier Python transcription.

On the public D3D-like equilibrium (the one of ``test_score_solver.py``), on a physics grid of the engine's form
(points uniform in sqrt(psi_N), clipped to [1e-3, 1 - 5e-5], plus a fine layer at the edge) and with the engine's
geometry route (``get_fsa`` / ``sauter_fc``), :func:`bouquet.kinrecon.current.bootstrap.solver_jdotb` is compared
with the transcription the engine carried before (Redl 2021 jboot1, kept below as the reference): with an
impurity in the ion pressure (the engine's call), without one, and with the NRL logarithms.  Bound: 1e-10 of
the peak (double-precision round-off through the gradient stencils on psi_N spacings down to 1e-5; the two
routes differ in the order of operations only: temperatures sent in keV, gradients on the absolute flux).

``OFT_env`` is a per-process singleton, so the live work runs in ONE child interpreter.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import test_score_solver as _tss  # noqa: E402  (equilibrium builder, OFT path and markers)

pytestmark = _tss.pytestmark
BOUND = 1.0e-10
EC = 1.602176634e-19


# ---------------------------------------------------------------- reference: the earlier engine transcription
def _ref_jdotb(x, te, ti, ne, ni, zeff, q, R, eps, ft, F, dpsi, nimp=None, ln_lambda_model="sauter"):
    from bouquet.kinrecon.current.conductivity import coulomb_log_e, coulomb_log_ii
    z = np.clip(ne / ni, 1.0, None)
    if ln_lambda_model == "sauter":
        ln_e, ln_ii = coulomb_log_e(ne, te, model="sauter"), coulomb_log_ii(ni, ti, z)
    else:
        ln_e = np.maximum(coulomb_log_e(ne, te, model="nrl"), 10.0)
        ln_ii = np.maximum(coulomb_log_ii(ni, ti, z), 10.0)
    nue = 6.921e-18 * np.abs(q) * R * ne * zeff * ln_e / (te ** 2 * eps ** 1.5)
    nui = 4.90e-18 * np.abs(q) * R * ni * ((ne / ni) * zeff) * ln_ii / (ti ** 2 * eps ** 1.5)
    fT, Zeff = ft, zeff
    d1 = (0.67 * (1 - 0.7 * fT) * np.sqrt(nue)) / (0.56 + 0.44 * Zeff)
    d2 = ((0.52 + 0.086 * np.sqrt(nue)) * (1 + 0.87 * fT) * nue) / (1 + 1.13 * np.sqrt(np.maximum(Zeff - 1, 0.0)))
    X31 = fT / (1 + d1 + d2)
    dZ = Zeff ** 1.2 - 0.71
    L31 = (1 + 0.15 / dZ) * X31 - (0.22 / dZ) * X31 ** 2 + (0.01 / dZ) * X31 ** 3 + (0.06 / dZ) * X31 ** 4
    dee_2 = (0.23 * (1 - 0.96 * fT) * np.sqrt(nue)) / np.sqrt(Zeff)
    dee_3 = ((0.13 * (1 - 0.38 * fT) * nue / Zeff ** 2)
             * (np.sqrt(1 + 2 * np.sqrt(np.maximum(Zeff - 1, 0.0)))
                + fT ** 2 * np.sqrt((0.075 + 0.25 * (Zeff - 1) ** 2) * nue)))
    X32e = fT / (1 + dee_2 + dee_3)
    F32_ee = ((0.1 + 0.6 * Zeff) / (Zeff * (0.77 + 0.63 * (1 + np.maximum(Zeff - 1, 0) ** 1.1)))
              * (X32e - X32e ** 4)
              + 0.7 / (1 + 0.2 * Zeff) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4))
              + 1.3 / (1 + 0.5 * Zeff) * X32e ** 4)
    dei_2 = (0.87 * (1 + 0.39 * fT) * np.sqrt(nue)) / (1 + 2.95 * (Zeff - 1) ** 2)
    dei_3 = 1.53 * (1 - 0.37 * fT) * nue * (2 + 0.375 * (Zeff - 1))
    X32i = fT / (1 + dei_2 + dei_3)
    F32_ei = (-(0.4 + 1.93 * Zeff) / (Zeff * (0.8 + 0.6 * Zeff)) * (X32i - X32i ** 4)
              + 5.5 / (1.5 + 2 * Zeff) * (X32i ** 2 - X32i ** 4 - 0.8 * (X32i ** 3 - X32i ** 4))
              - 1.3 / (1 + 0.5 * Zeff) * X32i ** 4)
    L32 = F32_ee + F32_ei
    alpha0 = (-(0.62 + 0.055 * (Zeff - 1)) / (0.53 + 0.17 * (Zeff - 1)) * (1 - fT)
              / (1 - (0.31 - 0.065 * (Zeff - 1)) * fT - 0.25 * fT ** 2))
    alpha = (((alpha0 + 0.7 * Zeff * np.sqrt(fT) * np.sqrt(nui)) / (1 + 0.18 * np.sqrt(nui))
              - 0.002 * nui ** 2 * fT ** 6) / (1 + 0.004 * nui ** 2 * fT ** 6))
    nit = ni + (0.0 if nimp is None else nimp)
    pe, pi = EC * ne * te, EC * nit * ti
    rpe = pe / (pe + pi)

    def d(y):
        return np.gradient(y, x, edge_order=2) / float(dpsi)

    dte, dti, dne, dnit = d(te), d(ti), d(ne), d(nit)
    dp = (ne * dte + te * dne + nit * dti + ti * dnit) * EC
    return -np.abs(F) * pe * (L31 * dp / pe + L32 * dte / te + L31 * alpha * (1 - rpe) / rpe * dti / ti)


# ---------------------------------------------------------------- child
def child(out):
    from bouquet.kinrecon.current.bootstrap import solver_jdotb
    gs, _ = _tss._build_equilibrium()
    rho = np.linspace(0.0, 1.0, 201)
    grid = np.unique(np.concatenate((rho ** 2, np.linspace(0.995, 1.0, 41))))
    ge = np.unique(np.clip(grid, 1.0e-3, 1.0 - 5.0e-5))
    ge = ge[np.concatenate(([True], np.diff(ge) > 1e-9))]
    fsa = gs.get_fsa(psi=ge)
    _, fc, _, _, eps = gs.sauter_fc(psi=ge, return_eps=True)
    q = np.abs(np.asarray(fsa["q"], float))
    R = 0.5 * (np.asarray(fsa["R_min"], float) + np.asarray(fsa["R_max"], float))
    F = np.asarray(fsa["F"], float)
    dpsi = abs(float(fsa["psi_boundary"]) - float(fsa["psi_axis"]))
    te = 3.0e3 * (1.0 - ge ** 1.5) ** 1.5 + 100.0
    ti = 2.5e3 * (1.0 - ge ** 1.4) ** 1.4 + 120.0
    ne = 5.0e19 * (1.0 - 0.6 * ge ** 2) + 0.5e19
    zeff = 1.8 + 0.2 * ge
    ni = ne * (6.0 - zeff) / 5.0
    nimp = ne * (zeff - 1.0) / 30.0
    rec = dict(n=int(ge.size), psi_bounds=[float(v) for v in gs.psi_bounds], F_sign=float(np.sign(F[0])))
    for case, kw in (("impurity_sauter", dict(nimp=nimp)), ("no_impurity_sauter", dict()),
                     ("impurity_nrl", dict(nimp=nimp, ln_lambda_model="nrl"))):
        ref = _ref_jdotb(ge, te, ti, ne, ni, zeff, q, R, np.asarray(eps, float), 1.0 - np.asarray(fc, float), F,
                         dpsi, **kw)
        new, settings = solver_jdotb(gs, ge, te, ti, ne, ni, zeff, F, **kw)
        dif = np.abs(new - ref)
        rec[case] = dict(max_abs_over_peak=float(dif.max() / np.abs(ref).max()),
                         argmax_psi_n=float(ge[int(np.argmax(dif))]), peak=float(np.abs(ref).max()),
                         pointwise_max_rel=float(np.max(dif / np.maximum(np.abs(ref), 1e-300))),
                         solver_options=settings["solver_options"])
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("bootstrap_solver")
    out = str(d / "result.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_tss._REPO, _tss._OFT or "", env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), out], cwd=str(d), env=env,
                          capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    with open(out) as fh:
        return json.load(fh)


@pytest.mark.parametrize("case", ["impurity_sauter", "no_impurity_sauter", "impurity_nrl"])
def test_solver_bootstrap_equals_the_engine_transcription(result, case):
    assert result[case]["max_abs_over_peak"] < BOUND, result[case]


if __name__ == "__main__":
    child(sys.argv[1])
