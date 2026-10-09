"""The MSE term is OPT-IN: without ``mse_data`` the structured closure is the
closure as it was before the term existed -- checked against that code, not
against itself.

``tests/golden/structured_closure_pre_mse.json`` holds the outputs of the
solver-free scenarios below as computed by the commit BEFORE the MSE term
(the structured closure with l_i as a second measurement, a one-sided
inductive prior and the one-switch preset).  It was written by running THIS
module on that commit::

    PYTHONPATH=<checkout of that commit> python tests/test_structured_mse_optin.py --write tests/golden/structured_closure_pre_mse.json

and the test re-runs the same scenarios on the current code with no MSE block
and compares every output: the same record keys (so no MSE key leaks into a
record that never asked for one), the same strings and flags, and every
number to ``rtol = 1e-10`` (portable across BLAS builds; on the machine that
wrote the golden file the outputs were bit-identical).

The scenarios are the two structured solvers (hard KKT with and without the
one-sided prior and an axis row, soft posterior mode with and without it)
and the run.py predictor -> common-tail solve -> q0/l_i corrector pipeline on
an analytic cylinder (default preset, the opted-out hard channel, an l_i
target on both, and a two-step l_i corrector).  No GS solver, no machine
data.
"""
import json
import os
import sys
import types
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.join(_HERE, "golden", "structured_closure_pre_mse.json")
_RTOL = 1e-10
#: Absolute floor for the float comparison (owner-approved 2026-10-06):
#: a residual that is zero in effect (1e-15) rounds differently per BLAS
#: (the golden was recorded on one platform; CI runs on another), and a
#: relative tolerance alone cannot accept that.  Real changes of these
#: records are many orders larger.
_ATOL = 1e-12

# Keys the self-consistent-loop lineage added to the NON-MSE closure records
# (the soft solver's noise-floor stop test and its logged single retry,
# "jbs loop: joint relaxation ... soft closure: noise-floor stop test + logged
# single retry").  The golden file was written on the commit before the MSE
# term, which predates that lineage, so on the merged tree a record may carry
# EXACTLY one of these named sets in addition to the golden keys -- never a
# subset, never any other key, and never an MSE key -- and every key the
# golden file has must still be there with the same value.  (On the merge the
# no-MSE outputs were also checked bit-identical to the loop lineage's own
# pre-merge tip.)
_LOOP_LINEAGE_SOLVER_KEYS = frozenset(
    ("gn_stop", "gn_stop_reason", "n_noise_floor_accepts",
     "started_from_x0"))
_LOOP_LINEAGE_RECORD_KEYS = frozenset(
    ("structured_closure_retry", "structured_closure_retry_first_error",
     "structured_gn_stop", "structured_gn_stop_reason",
     "structured_n_noise_floor_accepts"))
_LOOP_LINEAGE_ADDED = (frozenset(), _LOOP_LINEAGE_SOLVER_KEYS,
                       _LOOP_LINEAGE_RECORD_KEYS)

_N = 201
R0, AMINOR = 1.7, 0.6


def _geom():
    psi = np.linspace(0.01, 0.99, _N)
    r = AMINOR * np.sqrt(psi)
    R_avg = R0 + 0.1 * r ** 2 / AMINOR
    inv_R = (1.0 / R0) * (1.0 + 0.05 * (r / AMINOR) ** 2)
    inv_R2 = inv_R ** 2 * (1.0 + 0.02 * (r / AMINOR) ** 2)
    dV = 4.0 * np.pi ** 2 * R0 * r * (AMINOR / (2.0 * np.sqrt(psi)))
    return {"psi_N": psi, "psi_q": psi, "R_avg": R_avg, "inv_R": inv_R,
            "inv_R2": inv_R2, "dV_dpsi": dV, "dpsi_dpsiN": 0.9,
            "pprime": -8.0e3 * (1.0 - psi)}


def _parts():
    from bouquet.utils import Ip_fsa_weights
    g = _geom()
    w, c = Ip_fsa_weights(g, convention="jphi-linterp")
    psi = g["psi_N"]
    j_ind = 8.0e5 * (1.0 - psi) ** 1.5
    j_bs = 3.0e5 * np.exp(-((psi - 0.9) / 0.06) ** 2) + 2.0e4 * (1.0 - psi)
    j_fix = 1.0e5 * (1.0 - psi) ** 3
    lin = lambda j: float(trapezoid(w * np.asarray(j, float), psi))  # noqa
    Ip_s = 1.04 * (lin(j_ind) + lin(j_bs) + lin(j_fix) + c)
    return g, psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s


class _Cyl:
    """Analytic stand-in for a solve: enclosed current renormalised to Ip."""

    def __init__(self, psi, Ip, li_scale):
        self.psi, self.Ip, self.I, self.n = psi, Ip, None, 0
        self.li_scale = li_scale

    def solve(self, j):
        I = np.pi * AMINOR ** 2 * cumulative_trapezoid(
            np.asarray(j, float), self.psi, initial=0.0)
        I = I + np.pi * AMINOR ** 2 * float(j[0]) * self.psi[0]
        self.I = I * (self.Ip / I[-1])
        self.n += 1
        return 7

    def li(self):
        # a smooth l_i-like functional of the solved current, placed near the
        # closure's own model value so the corrector takes real steps
        q = float(trapezoid(self.I / self.I[-1], self.psi))
        return float(self.li_scale[0] * (q / self.li_scale[1]) ** 2)


class _Snap:
    def __init__(self, cyl):
        self.cyl = cyl

    def get_q(self, psi=None, compute_geo=False):
        return (np.asarray(psi, float), np.full(np.size(psi), 1.2), None,
                None)

    def get_stats(self, lcfs_pad=None, li_normalization=None):
        return {"l_i": self.cyl.li() * 1.008}


class _GS:
    def __init__(self, cyl):
        self.cyl = cyl

    def copy_eq(self):
        return _Snap(self.cyl)


def _li_geometry(eq, geom, convention=None, pprime_sign=1.0, psi_pad=1e-3):
    from bouquet.utils import Ip_fsa_affine_profile
    g = _geom()
    psi = g["psi_N"]
    return dict(psi_N=psi, dpsi_dpsiN=g["dpsi_dpsiN"],
                vol=float(trapezoid(g["dV_dpsi"], psi) * g["dpsi_dpsiN"]),
                perimeter=2 * np.pi * AMINOR, R_axis=R0,
                affine_cum=Ip_fsa_affine_profile(g, convention="jphi-linterp"),
                psi_pad=1e-3, perimeter_source="synthetic circle",
                perimeter_get_stats_dl=2 * np.pi * AMINOR * 1.01,
                perimeter_ratio_dl_over_L=1.01)


def _li_achieved(eq, li_kind="li_1", psi_pad=1e-3, perimeter=None):
    return eq.cyl.li(), {"perimeter": perimeter}


def _scenarios(patch):
    """Every output, as plain Python; *patch(obj, name, value)* stubs l_i."""
    from bouquet import utils
    from bouquet.config import GenerationConfig
    from bouquet.run import Bouquet
    from bouquet.utils import (close_ip_structured, close_ip_structured_soft,
                               structured_basis_eval)

    patch(utils, "li_achieved", _li_achieved)
    patch(utils, "li_closure_geometry", _li_geometry)

    g, psi, w, c, j_ind, j_bs, j_fix, lin, Ip_s = _parts()
    Phi = structured_basis_eval(None, psi)
    model = utils.structured_li_model(psi, w, Phi, j_ind, j_bs, j_fix,
                                      _li_geometry(None, None), Ip_s, "li_1")
    li0 = float(utils.structured_li_of(model, np.zeros(2 * Phi.shape[0]))[0])
    cy = _Cyl(psi, Ip_s, (1.0, 1.0))
    cy.solve(j_ind + j_bs + j_fix)
    li_scale = (0.985 * li0, float(trapezoid(cy.I / cy.I[-1], psi)))

    out = {}
    sig = dict(sigma_ind=[0.1, 0.4, 0.4, 0.4], sigma_bs=[0.5, 0.3, 0.15, 0.1])
    ax = dict(psi=0.0, j_ind0=float(j_ind[0]), j_bs0=float(j_bs[0]),
              j_fix0=float(j_fix[0]),
              j_ref0=1.02 * float(j_ind[0] + j_bs[0] + j_fix[0]))
    out["hard"] = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix)
    out["hard_up"] = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs, j_fix,
                                         sigma_ind_up=[0.1, 0.1, 0.1, 0.4])
    out["hard_axis"] = close_ip_structured(psi, w, c, Ip_s, j_ind, j_bs,
                                           j_fix, axis=ax)
    out["soft"] = close_ip_structured_soft(psi, w, c, Ip_s, 0.005 * Ip_s,
                                           j_ind, j_bs, j_fix, **sig)
    out["soft_up"] = close_ip_structured_soft(
        psi, w, c, Ip_s, 0.005 * Ip_s, j_ind, j_bs, j_fix,
        sigma_ind_up=[0.1, 0.1, 0.1, 0.4], **sig)

    def pipeline(**kw):
        gc = GenerationConfig(closure_channel="structured",
                              jBS_baseline_mode="ohmic", **kw)
        cyl = _Cyl(psi, Ip_s, li_scale)
        probe = j_ind + j_bs + j_fix
        bl = types.SimpleNamespace(sawtooth=None, ip_closure=None)
        s_ind, s_bs, ohm, bs, extra, state = \
            Bouquet._close_ip_structured_predictor(
                gc, bl, _Snap(cyl), dict(g), probe, psi, j_ind, j_bs, j_fix,
                probe, 1.0, abs(Ip_s), w, c, lin(j_ind), lin(j_bs),
                lin(j_fix))
        bl.j_inductive, bl.j_BS = s_ind * j_ind, s_bs * j_bs
        bl.j_phi = bl.j_inductive + bl.j_BS + j_fix
        bl.ohm_scale, bl.bs_scale = ohm, bs
        bl.ip_closure = dict(closure_limited=False,
                             closure_limited_reasons=(), **extra)
        cyl.solve(bl.j_phi)
        if state is not None:
            Bouquet._close_ip_structured_corrector(
                state, bl, _GS(cyl), cyl.solve,
                ip_of=lambda j: float(lin(j) + c),
                roundtrip_gate=Bouquet._structured_roundtrip_gate(abs(Ip_s)))
        return dict(record=bl.ip_closure, j_phi=bl.j_phi,
                    j_inductive=bl.j_inductive, j_BS=bl.j_BS,
                    ohm_scale=bl.ohm_scale, bs_scale=bl.bs_scale,
                    n_solves=cyl.n, state=(None if state is None
                                           else sorted(state)))

    out["p_default"] = pipeline()
    out["p_none"] = pipeline(structured_preset="none")
    out["p_li"] = pipeline(structured_li_target=1.01 * li0)
    out["p_li_none"] = pipeline(structured_preset="none",
                                structured_li_target=1.01 * li0)
    out["p_li2"] = pipeline(structured_li_target=1.01 * li0,
                            structured_li_max_corrector_steps=2)
    return out


_W = np.random.default_rng(7).standard_normal(4096)


def _encode(v):
    """JSON form: numbers exact (float.hex), long arrays as summaries."""
    if isinstance(v, dict):
        return {"__dict__": {str(k): _encode(x) for k, x in v.items()}}
    if isinstance(v, (str, bool)) or v is None:
        return v
    if isinstance(v, (int, np.integer)):
        return {"__int__": int(v)}
    if isinstance(v, (float, np.floating)):
        return {"__f__": float(v).hex()}
    if isinstance(v, (list, tuple)) and not all(
            isinstance(x, (int, float, np.integer, np.floating))
            and not isinstance(x, bool) for x in v):
        return {"__seq__": [_encode(x) for x in v],
                "tuple": isinstance(v, tuple)}
    a = np.asarray(v)
    if a.dtype.kind in "biuf":
        a = a.astype(float).ravel()
        if a.size <= 16:
            return {"__arr__": [float(x).hex() for x in a],
                    "shape": list(np.shape(v))}
        return {"__sum__": {"n": int(a.size), "shape": list(np.shape(v)),
                            "stats": [float(x).hex() for x in (
                                a.sum(), a @ _W[:a.size], a[0], a[-1],
                                a.min(), a.max(), float(np.sqrt(a @ a)))]}}
    return {"__repr__": repr(v)}


def _compare(exp, got, path, errs):
    if isinstance(exp, dict) and "__dict__" in exp:
        if not (isinstance(got, dict) and "__dict__" in got):
            errs.append(f"{path}: expected a dict")
            return
        e, g = exp["__dict__"], got["__dict__"]
        if (set(e) - set(g)
                or frozenset(set(g) - set(e)) not in _LOOP_LINEAGE_ADDED):
            errs.append(f"{path}: keys differ; only before: "
                        f"{sorted(set(e) - set(g))}, only now: "
                        f"{sorted(set(g) - set(e))}")
        for k in set(e) & set(g):
            _compare(e[k], g[k], f"{path}.{k}", errs)
        return
    if isinstance(exp, dict) and ("__f__" in exp or "__arr__" in exp
                                  or "__sum__" in exp):
        def _vals(d):
            if "__f__" in d:
                return [float.fromhex(d["__f__"])], None
            if "__arr__" in d:
                return [float.fromhex(x) for x in d["__arr__"]], d["shape"]
            return ([float.fromhex(x) for x in d["__sum__"]["stats"]],
                    (d["__sum__"]["n"], d["__sum__"]["shape"]))
        if not isinstance(got, dict) or set(got) != set(exp):
            errs.append(f"{path}: kind differs ({exp!r} vs {got!r})"[:300])
            return
        ev, es = _vals(exp)
        gv, gs = _vals(got)
        if es != gs or len(ev) != len(gv) or not np.allclose(
                gv, ev, rtol=_RTOL, atol=_ATOL, equal_nan=True):
            errs.append(f"{path}: {gv} != {ev}"[:300])
        return
    if isinstance(exp, dict) and "__seq__" in exp:
        if not (isinstance(got, dict) and "__seq__" in got
                and len(got["__seq__"]) == len(exp["__seq__"])
                and got["tuple"] == exp["tuple"]):
            errs.append(f"{path}: sequence differs")
            return
        for i, (a, b) in enumerate(zip(exp["__seq__"], got["__seq__"])):
            _compare(a, b, f"{path}[{i}]", errs)
        return
    if exp != got:
        errs.append(f"{path}: {got!r} != {exp!r}"[:300])


def test_no_mse_block_reproduces_the_pre_mse_closure(monkeypatch):
    with open(_GOLDEN) as f:
        golden = json.load(f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        now = _encode(_scenarios(monkeypatch.setattr))
    now = json.loads(json.dumps(now))
    errs = []
    _compare(golden["outputs"], now, "", errs)
    assert not errs, "\n".join(errs[:20])


def test_the_golden_file_is_not_trivially_empty():
    with open(_GOLDEN) as f:
        golden = json.load(f)
    out = golden["outputs"]["__dict__"]
    assert {"hard", "soft", "p_default", "p_li", "p_li2"} <= set(out)
    rec = out["p_li"]["__dict__"]["record"]["__dict__"]
    # the l_i corrector really took a step in the golden run
    assert rec["n_extra_solves"] == {"__int__": 1}
    assert not any(k.startswith("structured_mse") or k.startswith("mse_")
                   for k in rec)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--write":
        sys.exit("usage: python tests/test_structured_mse_optin.py --write "
                 "<golden.json>")
    import bouquet

    def _patch(obj, name, value):
        setattr(obj, name, value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _encode(_scenarios(_patch))
    with open(sys.argv[2], "w") as f:
        json.dump({"about": "structured closure outputs WITHOUT the MSE term, "
                            "written by tests/test_structured_mse_optin.py "
                            "on the commit before the MSE term; see its "
                            "docstring",
                   "bouquet_version": str(getattr(bouquet, "__version__",
                                                  "?")),
                   "outputs": outputs}, f, indent=0, sort_keys=True)
    print("wrote", sys.argv[2])
