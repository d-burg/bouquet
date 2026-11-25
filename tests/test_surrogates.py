"""Standalone PR-1 tests for Miller flux-surface geometry and pure-NumPy ONNX/QLKNN surrogates."""

from __future__ import annotations

import json
import os
import struct
import time
import warnings

import numpy as np
import pytest
from scipy.special import erf

from bouquet.baseline import Baseline
from bouquet.config import ReconstructionSource
from bouquet.geometry import (
    MillerGeometry,
    build_tglf_inputs,
    gyrobohm_flux_scales,
    miller_from_baseline,
    miller_from_geqdsk,
    miller_from_imas_dd,
)
from bouquet.io.geqdsk import GEQDSKEquilibrium
from bouquet.surrogates import (
    DEFAULT_TGLFNN_MODEL,
    PureNumpyONNXModel,
    QLKNNJSONModel,
    TGLFNNCommittee,
    evaluate_qlknn_stiffness,
    evaluate_tglfnn_stiffness,
    is_lfs_pointer_file,
    resolve_tglfnn_model_dir,
    solve_tglfnn_flux_excursions,
)
from tests.test_geqdsk import generate_solovev_gfile


def _has_local_tglfnn() -> bool:
    try:
        resolve_tglfnn_model_dir(DEFAULT_TGLFNN_MODEL)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Minimal pure-stdlib protobuf encoder for synthetic ONNX unit tests
# ---------------------------------------------------------------------------

def _pb_varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_field(fnum: int, wtype: int, payload: bytes) -> bytes:
    if wtype == 2:
        return _pb_varint((fnum << 3) | 2) + _pb_varint(len(payload)) + payload
    return _pb_varint((fnum << 3) | wtype) + payload


def _pb_str(fnum: int, text: str) -> bytes:
    return _pb_field(fnum, 2, text.encode("utf-8"))


def _pb_tensor(name: str, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    dtype_code = 7 if arr.dtype.kind == "i" else 1
    cast = arr.astype("<i8" if dtype_code == 7 else "<f4")
    msg = b"".join(_pb_field(1, 0, _pb_varint(int(d))) for d in cast.shape)
    msg += _pb_field(2, 0, _pb_varint(dtype_code))
    if name:
        msg += _pb_str(8, name)
    msg += _pb_field(9, 2, cast.tobytes())
    return msg


def _pb_attr(name: str, *, f: float | None = None, i: int | None = None, s: str | None = None,
             ints: list[int] | None = None, t: np.ndarray | None = None) -> bytes:
    msg = _pb_str(1, name)
    if f is not None:
        msg += _pb_field(2, 5, struct.pack("<f", f))
    elif i is not None:
        msg += _pb_field(3, 0, _pb_varint(i))
    elif s is not None:
        msg += _pb_str(4, s)
    elif t is not None:
        msg += _pb_field(5, 2, _pb_tensor("", t))
    elif ints is not None:
        for iv in ints:
            msg += _pb_field(8, 0, _pb_varint(iv))
    return msg


def _pb_node(op_type: str, inputs: list[str], outputs: list[str], attrs: list[bytes] | None = None) -> bytes:
    msg = b"".join(_pb_str(1, x) for x in inputs)
    msg += b"".join(_pb_str(2, y) for y in outputs)
    msg += _pb_str(4, op_type)
    for a in attrs or []:
        msg += _pb_field(5, 2, a)
    return msg


def _pb_model(inputs: list[str], outputs: list[str], nodes: list[bytes],
              initializers: list[tuple[str, np.ndarray]]) -> bytes:
    graph = b"".join(_pb_field(1, 2, n) for n in nodes)
    graph += b"".join(_pb_field(5, 2, _pb_tensor(k, v)) for k, v in initializers)
    graph += b"".join(_pb_field(11, 2, _pb_str(1, x)) for x in inputs)
    graph += b"".join(_pb_field(12, 2, _pb_str(1, y)) for y in outputs)
    return _pb_field(7, 2, graph)


# ---------------------------------------------------------------------------
# PureNumpyONNXModel operator & graph tests
# ---------------------------------------------------------------------------

def test_pure_numpy_onnx_gemm_constant_bias_gelu_and_reshape_zero_dim(tmp_path):
    """Verify non-initializer Gemm bias (from Constant), Gelu (erf & tanh), and Reshape 0-dims."""
    rng = np.random.default_rng(42)
    W = rng.normal(size=(6, 4)).astype(np.float32)  # transB=1 -> (4, 6) -> (batch, 6)
    b_const = np.array([0.5, -0.25, 1.2, -0.8, 0.3, 0.9], dtype=np.float32)
    W_out = rng.normal(size=(6, 2)).astype(np.float32)

    nodes = [
        _pb_node("Constant", [], ["b_c"], [_pb_attr("value", t=b_const)]),
        # Gemm with W in initializers and bias coming from Constant node (NOT initializers!)
        _pb_node("Gemm", ["x", "W", "b_c"], ["h1"], [
            _pb_attr("transB", i=1),
            _pb_attr("alpha", f=1.25),
            _pb_attr("beta", f=0.8),
        ]),
        _pb_node("Gelu", ["h1"], ["g_exact"]),
        _pb_node("Gelu", ["h1"], ["g_tanh"], [_pb_attr("approximate", s="tanh")]),
        _pb_node("Add", ["g_exact", "g_tanh"], ["g_sum"]),
        # Reshape (batch, 6) -> (0, 2, 3) -> (0, -1) preserving batch dimension via 0-dim spec
        _pb_node("Reshape", ["g_sum", "sh_3d"], ["r_3d"]),
        _pb_node("Reshape", ["r_3d", "sh_2d"], ["r_2d"]),
        # General Gemm where A is dynamic and W_out comes from Constant (neither in initializers)
        _pb_node("Constant", [], ["W2_c"], [_pb_attr("value", t=W_out)]),
        _pb_node("Gemm", ["r_2d", "W2_c"], ["y"]),
    ]
    raw_pb = _pb_model(
        inputs=["x"],
        outputs=["y"],
        nodes=nodes,
        initializers=[
            ("W", W),
            ("sh_3d", np.array([0, 2, 3], dtype=np.int64)),
            ("sh_2d", np.array([0, -1], dtype=np.int64)),
        ],
    )
    onnx_path = tmp_path / "ops_test.onnx"
    onnx_path.write_bytes(raw_pb)

    mdl = PureNumpyONNXModel(str(onnx_path))
    X = rng.normal(size=(7, 4)).astype(np.float32)
    got = mdl(X)

    h1_ref = 1.25 * (X @ W.T) + 0.8 * b_const
    g_exact_ref = 0.5 * h1_ref * (1.0 + erf(h1_ref / np.sqrt(2.0)))
    g_tanh_ref = 0.5 * h1_ref * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h1_ref + 0.044715 * (h1_ref ** 3))))
    y_ref = (g_exact_ref + g_tanh_ref) @ W_out

    assert got.shape == (7, 2)
    np.testing.assert_allclose(got, y_ref, rtol=1e-5, atol=1e-5)
    # 1-D single-sample call squeezes leading batch axis
    got_1d = mdl(X[0])
    assert got_1d.shape == (2,)
    np.testing.assert_allclose(got_1d, y_ref[0], rtol=1e-5, atol=1e-5)

    # LFS stub detection
    lfs_stub = tmp_path / "stub.onnx"
    lfs_stub.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 1234\n")
    assert is_lfs_pointer_file(str(lfs_stub))
    with pytest.raises(RuntimeError, match="LFS pointer"):
        PureNumpyONNXModel(str(lfs_stub))


# ---------------------------------------------------------------------------
# Miller geometry: GEQDSK COCOS 1 vs 11, IMAS D3D-like OMAS, & TGLF inputs
# ---------------------------------------------------------------------------

def test_miller_from_geqdsk_path_bytes_and_instance(tmp_path):
    gpath = tmp_path / "solovev.geqdsk"
    generate_solovev_gfile(str(gpath), R0=1.85, a=0.57, B0=12.0, Ip=8.7e6, kappa=1.0, NW=65, NH=65)
    raw = gpath.read_bytes()
    psi_out = np.linspace(0.0, 1.0, 51)

    mg_path = miller_from_geqdsk(gpath, psi_N_out=psi_out)
    mg_bytes = miller_from_geqdsk(raw, psi_N_out=psi_out)
    mg_inst = miller_from_geqdsk(GEQDSKEquilibrium(str(gpath)), psi_N_out=psi_out)

    assert np.isclose(mg_path.R0_m, 1.85, atol=0.05)
    assert np.isclose(mg_path.a_m, 0.57, atol=0.03)
    assert np.isclose(mg_path.B0_T, 12.0, atol=0.2)
    assert np.allclose(mg_path.r_min_m, mg_bytes.r_min_m)
    assert np.allclose(mg_path.B_unit_T, mg_inst.B_unit_T)
    assert np.all(mg_path.volume_m3 >= 0.0)
    assert mg_path.volume_m3[-1] > 5.0


def test_miller_geqdsk_and_baseline_cocos_1_vs_11_parity(tmp_path):
    """COCOS 11 (Wb, exp_Bp=1) g-files must match COCOS 1 (Wb/rad, exp_Bp=0) without 2*pi errors."""
    g1_path = tmp_path / "sol_cocos1.geqdsk"
    g11_path = tmp_path / "sol_cocos11.geqdsk"
    generate_solovev_gfile(str(g1_path), R0=1.75, a=0.58, B0=5.2, Ip=2.4e6, kappa=1.4, NW=65, NH=65)

    geq1 = GEQDSKEquilibrium(str(g1_path), cocos=1)
    geq11 = geq1.cocosify(11, copy=True)
    geq11.save(str(g11_path))
    raw11 = g11_path.read_bytes()

    psi = np.linspace(0.0, 1.0, 41)
    mg_c1 = miller_from_geqdsk(g1_path, psi_N_out=psi, cocos=1)
    mg_c11_path = miller_from_geqdsk(g11_path, psi_N_out=psi, cocos=11)
    mg_c11_bytes = miller_from_geqdsk(raw11, psi_N_out=psi, cocos=11)

    np.testing.assert_allclose(mg_c11_path.q, mg_c1.q, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mg_c11_bytes.q, mg_c1.q, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mg_c11_path.B_unit_T, mg_c1.B_unit_T, rtol=2e-3, atol=5e-3)

    # Also verify miller_from_baseline forwards ReconstructionSource.cocos on both eqdsk_bytes and geqdsk_path
    ne = 6.0e19 * (1.0 - 0.6 * psi)
    te = 3.0e3 * (1.0 - 0.8 * psi) + 100.0
    j = 1.5e6 * (1.0 - psi)
    src11 = ReconstructionSource(geqdsk_path=str(g11_path), profiles_path="unused.h5", cocos=11)

    bl_path = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=0.9 * ne, ti=te,
        Zeff=np.full_like(psi, 1.5), Ip_target=2.4e6, l_i_target=0.9,
        provenance="reconstruction",
    )
    bl_bytes = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=0.9 * ne, ti=te,
        Zeff=np.full_like(psi, 1.5), Ip_target=2.4e6, l_i_target=0.9,
        provenance="reconstruction", eqdsk_bytes=raw11,
    )
    mg_bl_path = miller_from_baseline(bl_path, src11, psi)
    mg_bl_bytes = miller_from_baseline(bl_bytes, src11, psi)
    np.testing.assert_allclose(mg_bl_path.q, mg_c1.q, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mg_bl_bytes.q, mg_c1.q, rtol=1e-4, atol=1e-4)

    inp1 = build_tglf_inputs(mg_c1, psi, ne, te, 0.9 * ne, te, np.full_like(psi, 1.5))
    inp11 = build_tglf_inputs(mg_bl_bytes, psi, ne, te, 0.9 * ne, te, np.full_like(psi, 1.5))
    np.testing.assert_allclose(inp11["Q_PRIME_LOC"][2:], inp1["Q_PRIME_LOC"][2:], rtol=2e-3, atol=1e-2)
    np.testing.assert_allclose(inp11["P_PRIME_LOC"][2:], inp1["P_PRIME_LOC"][2:], rtol=5e-3, atol=1e-3)


def test_miller_from_imas_dd_d3dlike_omas_and_none_array_guards():
    """Tracked D3Dlike_baseline_omas.json lacks r_inboard/r_outboard and stores a
    sqrt(psi_N) placeholder in rho_tor_norm; B_unit must stay smooth (no q-proportional
    separatrix blow-up) and boundary.outline shaping must be extracted."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    omas_path = os.path.join(repo_root, "examples", "D3D-like", "D3Dlike_baseline_omas.json")
    mg = miller_from_imas_dd(omas_path)

    assert np.all(np.isfinite(mg.B_unit_T))
    # B0 = 1.8 T, kappa_edge ~ 1.74 -> B_unit ~ 2.6 T smoothly across [0, 1] (never 9.5 T!)
    assert 2.0 <= float(np.min(mg.B_unit_T)) <= float(np.max(mg.B_unit_T)) <= 3.5
    assert float(np.ptp(mg.B_unit_T)) < 0.15
    assert mg.kappa[-1] > 1.65
    assert mg.delta[-1] > 0.35
    assert mg.zeta[-1] < -0.05
    assert mg.Z_maj_m[-1] < -0.04
    assert mg.sign_b0 == -1.0
    assert mg.sign_q == 1.0

    # Guard None / empty / wrong-length optional 1-D arrays cleanly
    with open(omas_path, "r") as fh:
        dd_mod = json.load(fh)
    eq1d = dd_mod["equilibrium"]["time_slice"][0]["profiles_1d"]
    eq1d["elongation"] = None
    eq1d["triangularity_upper"] = []
    eq1d["volume"] = [1.0, 2.0]  # mismatched length -> ignored safely
    mg_guarded = miller_from_imas_dd(dd_mod, psi_N_out=np.linspace(0.0, 1.0, 33))
    assert mg_guarded.psi_N.size == 33
    assert np.all(np.isfinite(mg_guarded.B_unit_T))
    assert np.isclose(mg_guarded.delta[-1], mg.delta[-1], atol=1e-6)


def test_build_tglf_inputs_smooth_quasineutrality_xnue_and_signed_rotation():
    """Verify C^inf quasineutrality across ni == ne, alpha z_fast accounting,
    fixed-deuteron sqrt(2) XNUE, and signed (B0, q) ExB/parallel rotation parity."""
    psi = np.linspace(0.0, 1.0, 51)
    ne = 1.0e20 * (1.0 - 0.5 * (psi ** 1.3))
    te = 5.0e3 * (1.0 - 0.8 * psi) + 200.0
    ti = 4.8e3 * (1.0 - 0.8 * psi) + 200.0
    zeff = np.full_like(psi, 1.8)
    j = 2.0e6 * (1.0 - psi)
    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=0.84 * ne, ti=ti,
        Zeff=zeff, Ip_target=2.0e6, l_i_target=0.9, provenance="imas",
    )
    mg_pos = miller_from_baseline(bl, R0_fallback=1.70, a_fallback=0.60, B0_fallback=2.0)

    # Independent ni draw crossing ne at mid-radius: RLNS_3 must remain smooth (zero -246 spike!)
    ni_cross = ne * (0.95 + 0.10 * psi)
    inp_cross = build_tglf_inputs(mg_pos, psi, ne, te, ni_cross, ti, zeff, Z_imp=6.0)
    assert np.all(np.abs(inp_cross["RLNS_3"]) < 10.0)
    np.testing.assert_allclose(inp_cross["RLNS_3"], inp_cross["RLNS_1"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(inp_cross["AS_2"] + 6.0 * inp_cross["AS_3"], 1.0, atol=1e-5)

    # Alpha-particle z_fast = 2 * n_alpha reduces thermal nz_zeff and AS_2 consistently
    z_fast = 0.06 * ne  # n_alpha = 0.03 * ne, Z_fast = 2 -> contributes 0.12 to Zeff
    inp_alpha = build_tglf_inputs(mg_pos, psi, ne, te, 0.82 * ne, ti, zeff, Z_imp=6.0, z_fast_m3=z_fast)
    expected_nz_frac = (1.8 - 1.0 - 0.06) / 30.0
    np.testing.assert_allclose(inp_alpha["AS_3"], expected_nz_frac, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(inp_alpha["AS_2"] + 6.0 * inp_alpha["AS_3"] + z_fast / ne, 1.0, atol=1e-5)

    # XNUE uses sqrt(2) consistently with c_s = sqrt(Te / m_D) regardless of A_main
    inp_dt = build_tglf_inputs(mg_pos, psi, ne, te, 0.84 * ne, ti, zeff, A_main=2.5)
    np.testing.assert_allclose(inp_dt["XNUE"], inp_cross["XNUE"], rtol=1e-12, atol=0.0)

    # Standard DIII-D orientation (Ip > 0, Bt < 0 => sign_b0 = -1, sign_q = -1, SIGN_IT = +1):
    # VEXB_SHEAR flips sign with sign_q while VPAR_1 preserves SIGN_IT = +1
    omega = 6.0e4 * (1.0 - 0.75 * psi)
    mg_d3d = miller_from_baseline(bl, R0_fallback=1.70, a_fallback=0.60, B0_fallback=-2.0)
    mg_d3d.sign_q = -1.0
    inp_rot_pos = build_tglf_inputs(mg_pos, psi, ne, te, 0.84 * ne, ti, zeff, omega_tor_rad_s=omega)
    inp_rot_d3d = build_tglf_inputs(mg_d3d, psi, ne, te, 0.84 * ne, ti, zeff, omega_tor_rad_s=omega)
    np.testing.assert_allclose(inp_rot_d3d["VEXB_SHEAR"][1:], -inp_rot_pos["VEXB_SHEAR"][1:], rtol=1e-12)
    np.testing.assert_allclose(inp_rot_d3d["VPAR_1"], inp_rot_pos["VPAR_1"], rtol=1e-12)
    np.testing.assert_allclose(inp_rot_d3d["VPAR_SHEAR_1"], inp_rot_pos["VPAR_SHEAR_1"], rtol=1e-12)


# ---------------------------------------------------------------------------
# QLKNN tests: CODATA collisionality, ITG+TEM+ETG + _div_ ratios, GACODE units
# ---------------------------------------------------------------------------

def _make_qlknn_json_spec(feats: list[str], weights_1: np.ndarray, bias_1: np.ndarray,
                          weights_2: np.ndarray, bias_2: np.ndarray, hidden_act: str = "relu") -> dict:
    return {
        "feature_names": feats,
        "target_names": ["flux"],
        "feature_min": {k: -10.0 for k in feats},
        "feature_max": {k: 30.0 for k in feats},
        "hidden_activation": [hidden_act],
        "output_activation": "none",
        "settings": {
            "feature_prescale_factor": {k: 1.0 for k in feats},
            "feature_prescale_bias": {k: 0.0 for k in feats},
            "target_prescale_factor": {"flux": 1.0},
            "target_prescale_bias": {"flux": 0.0},
        },
        "layer1/weights:0": weights_1.tolist(),
        "layer1/biases:0": bias_1.tolist(),
        "output/weights:0": weights_2.tolist(),
        "output/biases:0": bias_2.tolist(),
    }


def test_qlknn_json_evaluator_and_high_beta_warning(tmp_path):
    feats = ["Zeff", "Ati", "Ate", "An", "q", "smag", "x", "Ti_Te", "logNustar"]
    for fname, idx_drive in (("efeETG_GB.json", 2), ("efiITG_GB.json", 1)):
        W1 = np.zeros((len(feats), 4))
        W1[idx_drive, :] = [0.25, 0.35, 0.15, 0.20]
        b1 = np.array([-1.0, -1.5, -0.8, -1.2])
        W2 = np.array([[4.0], [5.0], [2.0], [3.0]])
        b2 = np.array([0.0])
        spec = _make_qlknn_json_spec(feats, W1, b1, W2, b2, hidden_act="relu")
        (tmp_path / fname).write_text(json.dumps(spec))

    psi = np.linspace(0.0, 1.0, 25)
    ne = 3.0e20 * (1.0 - 0.5 * psi)
    te = 15.0e3 * (1.0 - 0.8 * psi) + 200.0
    j = 3.0e6 * (1.0 - psi)
    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=0.9 * ne, ti=te,
        Zeff=np.full_like(psi, 1.5), Ip_target=5.0e6, l_i_target=0.9,
        provenance="imas",
    )
    mg = miller_from_baseline(bl, R0_fallback=1.85, a_fallback=0.57, B0_fallback=6.0)
    with pytest.warns(UserWarning, match="Electromagnetic stabilization"):
        res = evaluate_qlknn_stiffness(mg, psi, ne, te, 0.9 * ne, te, np.full_like(psi, 1.5), model_dir=str(tmp_path))
    assert np.all(res["dQe_dRLTe_gB"] > 0.0)
    assert np.all(res["dQi_dRLTi_gB"] > 0.0)


def test_qlknn_collisionality_multimode_div_ratios_gacode_units_and_activations(tmp_path):
    """Verify CODATA nu_star (not 1e-6 floor), ITG+TEM+ETG + _div_ summing,
    QuaLiKiz->GACODE Q_gB_imas conversion, subcritical threshold scan, and activation validation."""
    feats = ["Zeff", "Ati", "Ate", "An", "q", "smag", "x", "Ti_Te", "logNustar", "gammaE"]
    idx_ati = feats.index("Ati")
    idx_ate = feats.index("Ate")
    idx_nu = feats.index("logNustar")

    # efiITG_GB: ReLU(Ati - 4.0) * 2.0 + 0.1 * (logNustar + 6.0) to probe logNustar directly
    W_itg = np.zeros((len(feats), 2))
    W_itg[idx_ati, 0] = 1.0
    W_itg[idx_nu, 1] = 1.0
    spec_efi_itg = _make_qlknn_json_spec(
        feats, W_itg, np.array([-4.0, 6.0]), np.array([[2.0], [0.1]]), np.array([0.0]), "relu"
    )
    # efeITG_GB_div_efiITG_GB: constant ratio 0.5 via sigmoid(0) = 0.5
    spec_efe_div = _make_qlknn_json_spec(
        feats, np.zeros((len(feats), 1)), np.array([0.0]), np.array([[1.0]]), np.array([0.0]), "sigmoid"
    )
    # efeTEM_GB: 1.5 * ReLU(Ate - 3.5)
    W_tem = np.zeros((len(feats), 1))
    W_tem[idx_ate, 0] = 1.0
    spec_efe_tem = _make_qlknn_json_spec(
        feats, W_tem, np.array([-3.5]), np.array([[1.5]]), np.array([0.0]), "relu"
    )
    # efiTEM_GB_div_efeTEM_GB: constant ratio 0.4 via softplus(0)/ln(2) * 0.4
    spec_efi_div = _make_qlknn_json_spec(
        feats, np.zeros((len(feats), 1)), np.array([0.0]), np.array([[0.4 / np.log(2.0)]]), np.array([0.0]), "softplus"
    )
    # efeETG_GB: 1.0 * ReLU(Ate - 5.0)
    spec_efe_etg = _make_qlknn_json_spec(
        feats, W_tem, np.array([-5.0]), np.array([[1.0]]), np.array([0.0]), "relu"
    )

    (tmp_path / "efiITG_GB.json").write_text(json.dumps(spec_efi_itg))
    (tmp_path / "efeITG_GB_div_efiITG_GB.json").write_text(json.dumps(spec_efe_div))
    (tmp_path / "efeTEM_GB.json").write_text(json.dumps(spec_efe_tem))
    (tmp_path / "efiTEM_GB_div_efeTEM_GB.json").write_text(json.dumps(spec_efi_div))
    (tmp_path / "efeETG_GB.json").write_text(json.dumps(spec_efe_etg))

    # Hollow inner core (Ate < 0, Ati < 0 for psi < 0.25) transitioning to peaked outer core
    psi = np.linspace(0.0, 1.0, 31)
    ne = 5.0e19 * (1.0 - 0.4 * psi)
    te = 2.2e3 * (1.0 + 0.25 * psi - 0.95 * (psi ** 2)) + 150.0
    ni = 0.85 * ne
    ti = 0.90 * te
    zeff = np.full_like(psi, 1.6)
    j = 1.2e6 * (1.0 - psi)
    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=ni, ti=ti,
        Zeff=zeff, Ip_target=1.4e6, l_i_target=0.9, provenance="imas",
    )
    mg = miller_from_baseline(bl, R0_fallback=1.70, a_fallback=0.58, B0_fallback=3.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res = evaluate_qlknn_stiffness(mg, psi, ne, te, ni, ti, zeff, model_dir=str(tmp_path))

    gb = gyrobohm_flux_scales(mg, ne, te)
    # 1. Exact GACODE Q_gB_imas_W_m2 normalization consistency
    np.testing.assert_allclose(res["Qe_W_m2"], res["Qe_gB"] * gb["Q_gB_imas_W_m2"], rtol=1e-12)
    np.testing.assert_allclose(res["Qi_W_m2"], res["Qi_gB"] * gb["Q_gB_imas_W_m2"], rtol=1e-12)

    # 2. At psi[2] (hollow core where Ati < 0), ReLU(Ati - 4.0) == 0, so Qi_QLK comes solely
    #    from 0.1 * (logNustar + 6.0): with CODATA nu_e, logNustar ~ -1.2 > -6.0 -> Qi_gB > 0!
    assert res["Qi_gB"][2] > 0.05, f"Expected CODATA logNustar > -6, got Qi_gB[2]={res['Qi_gB'][2]}"

    # 3. Subcritical/hollow core still finds true TEM threshold Ate_crit ~ 3.5 (NOT Ate_base!)
    assert 3.4 <= res["rlte_crit"][2] <= 4.0, f"Expected rlte_crit ~ 3.6 on hollow core, got {res['rlte_crit'][2]}"

    # 4. Unknown activation raises ValueError explicitly (never silently treated as linear)
    bad_spec = _make_qlknn_json_spec(
        feats, np.zeros((len(feats), 1)), np.array([0.0]), np.array([[1.0]]), np.array([0.0]), "bogus_act"
    )
    bad_path = tmp_path / "bad_act.json"
    bad_path.write_text(json.dumps(bad_spec))
    bad_mdl = QLKNNJSONModel(str(bad_path))
    with pytest.raises(ValueError, match="Unsupported QLKNN activation"):
        bad_mdl(np.zeros((2, len(feats))))


# ---------------------------------------------------------------------------
# Synthetic ONNX TGLFNNCommittee tests (always run even without ~/.julia)
# ---------------------------------------------------------------------------

def _write_synthetic_tglfnn_committee(model_dir, n_members: int = 3) -> None:
    os.makedirs(model_dir, exist_ok=True)
    xnames = [
        "AS_2", "AS_3", "BETAE_log10", "DEBYE_log10", "DELTA_LOC", "DRMAJDX_LOC",
        "DZMAJDX_LOC", "KAPPA_LOC", "P_PRIME_LOC", "Q_LOC", "Q_PRIME_LOC",
        "RLNS_1", "RLNS_2", "RLNS_3", "RLTS_1", "RLTS_2", "RLTS_3",
        "RMAJ_LOC", "RMIN_LOC", "S_DELTA_LOC", "S_KAPPA_LOC", "S_ZETA_LOC",
        "TAUS_2", "TAUS_3", "VEXB_SHEAR", "VPAR_1", "VPAR_SHEAR_1",
        "XNUE_log10", "ZEFF", "ZETA_LOC", "ZMAJ_LOC",
    ]
    ynames = ["OUT_G_elec", "OUT_P_ions", "OUT_Q_elec", "OUT_Q_ions"]
    (model_dir / "xnames.txt").write_text(" ".join(xnames) + "\n")
    (model_dir / "ynames.txt").write_text(" ".join(ynames) + "\n")
    np.savetxt(model_dir / "xm.txt", np.zeros(len(xnames)))
    np.savetxt(model_dir / "xsigma.txt", np.ones(len(xnames)))
    np.savetxt(model_dir / "ym.txt", np.zeros(len(ynames)))
    np.savetxt(model_dir / "ysigma.txt", np.ones(len(ynames)))

    i_rn1 = xnames.index("RLNS_1")
    i_rn3 = xnames.index("RLNS_3")
    i_rt1 = xnames.index("RLTS_1")
    i_rt2 = xnames.index("RLTS_2")

    # 3-member committee where Qe = c_k * ReLU(RLTS_1 - 1.2), Qi = c_k * ReLU(RLTS_2 - 1.0),
    # Ge = 0.4 * (RLNS_1 + 0.2 * RLNS_3 - 1.0)
    scales = np.linspace(0.85, 1.15, n_members)
    for k, sc in enumerate(scales):
        W1 = np.zeros((3, len(xnames)), dtype=np.float32)
        b1 = np.zeros(3, dtype=np.float32)
        W1[0, i_rt1] = 1.0
        b1[0] = -1.2
        W1[1, i_rt2] = 1.0
        b1[1] = -1.0
        W1[2, i_rn1] = 1.0
        W1[2, i_rn3] = 0.2
        b1[2] = -1.0

        W2 = np.zeros((4, 3), dtype=np.float32)
        W2[0, 2] = 0.4 * sc     # OUT_G_elec (linear via bypass below)
        W2[2, 0] = 2.5 * sc     # OUT_Q_elec
        W2[3, 1] = 3.0 * sc     # OUT_Q_ions

        # Build graph: h_relu = Relu(Gemm(x, W1, b1)); g_lin = Gemm(x, Wg, bg); y = Add(...)
        Wg = np.zeros((4, len(xnames)), dtype=np.float32)
        Wg[0, i_rn1] = 0.4 * sc
        Wg[0, i_rn3] = 0.08 * sc
        bg = np.array([-0.48 * sc, 0.0, 0.0, 0.0], dtype=np.float32)
        W2_q = W2.copy()
        W2_q[0, :] = 0.0

        nodes = [
            _pb_node("Gemm", ["x", "W1", "b1"], ["h"], [_pb_attr("transB", i=1)]),
            _pb_node("Relu", ["h"], ["hr"]),
            _pb_node("Gemm", ["hr", "W2", "b2"], ["yq"], [_pb_attr("transB", i=1)]),
            _pb_node("Gemm", ["x", "Wg", "bg"], ["yg"], [_pb_attr("transB", i=1)]),
            _pb_node("Add", ["yq", "yg"], ["y"]),
        ]
        raw = _pb_model(
            inputs=["x"], outputs=["y"], nodes=nodes,
            initializers=[("W1", W1), ("b1", b1), ("W2", W2_q), ("b2", np.zeros(4, dtype=np.float32)), ("Wg", Wg), ("bg", bg)],
        )
        (model_dir / f"member_{k}.onnx").write_bytes(raw)


def test_synthetic_committee_excursions_hollow_continuity_and_coupled_std(tmp_path):
    """Unconditional standalone test of evaluate_tglfnn_stiffness and solve_tglfnn_flux_excursions."""
    mdir = tmp_path / "syn_tglfnn"
    _write_synthetic_tglfnn_committee(mdir, n_members=3)

    psi = np.linspace(0.0, 1.0, 49)
    ne = 8.0e19 * (1.0 - 0.5 * (psi ** 1.2))
    # Hollow Te inside psi < 0.30, peaked outside; peaked Ti everywhere
    te = 3.2e3 * (1.0 + 0.35 * np.exp(-((psi - 0.26) / 0.14) ** 2) - 0.82 * (psi ** 1.4)) + 150.0
    ti = 3.5e3 * ((1.0 - 0.80 * psi) ** 1.3) + 150.0
    ni = 0.88 * ne
    zeff = np.full_like(psi, 1.6)
    j = 1.6e6 * (1.0 - psi)
    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=j, j_BS=np.zeros_like(j),
        psi_N_kinetic=psi, ne=ne, te=te, ni=ni, ti=ti,
        Zeff=zeff, Ip_target=1.8e6, l_i_target=0.9, provenance="imas",
    )
    mg = miller_from_baseline(bl, R0_fallback=1.74, a_fallback=0.58, B0_fallback=3.5)

    st = evaluate_tglfnn_stiffness(mg, psi, ne, te, ni, ti, zeff, model_spec=str(mdir))
    R_over_a = mg.R0_m / mg.a_m
    # Even on hollow inner nodes, scan [0.1, max(4.0, 1.3*aL)] finds true threshold a/L_Te ~ 1.23 -> R/L_Te ~ 3.69
    assert np.allclose(st.rlte_crit[:8], 1.232 * R_over_a, atol=0.25)

    # Verify Miller-frame chi_e_turb divides by Q_gB_W_m2 (without vprime_miller inflation)
    gb = gyrobohm_flux_scales(mg, ne, te)
    inp = build_tglf_inputs(mg, psi, ne, te, ni, ti, zeff)
    k_mid = 30
    dte_dr_mid = abs(inp["RLTS_1"][k_mid]) * te[k_mid] / mg.a_m
    expected_chi_e = (st.Qe_gB[k_mid] * gb["Q_gB_W_m2"][k_mid]) / (ne[k_mid] * 1.602176634e-19 * dte_dr_mid)
    assert np.isclose(st.chi_e_turb_m2_s[k_mid], np.clip(expected_chi_e, 1e-3, 100.0), rtol=1e-6)

    # Verify smooth C^0 excursions across hollow Te, scan boundaries, and flux_uncertainty_rel=0.0
    exc = solve_tglfnn_flux_excursions(
        mg, psi, ne, te, ni, ti, zeff,
        np.full_like(psi, 2.0), np.full_like(psi, 2.0),
        flux_uncertainty_rel=0.30, model_spec=str(mdir),
    )
    core = (mg.rho_tor_norm >= 0.20) & (mg.rho_tor_norm <= 0.82)
    dk_ti = exc["dkappa_ti_free"][core]
    step_ratios = np.maximum(dk_ti[1:] / dk_ti[:-1], dk_ti[:-1] / dk_ti[1:])
    assert np.max(step_ratios) < 1.25, f"Unexpected cliff in dkappa_ti_free: max step ratio {np.max(step_ratios)}"

    exc_epistemic = solve_tglfnn_flux_excursions(
        mg, psi, ne, te, ni, ti, zeff,
        np.full_like(psi, 2.0), np.full_like(psi, 2.0),
        flux_uncertainty_rel=0.0, model_spec=str(mdir),
    )
    assert np.all(exc_epistemic["dkappa_te_free"] > 0.0)
    assert np.all(exc_epistemic["dkappa_te_free"] < exc["dkappa_te_free"])

    # In this synthetic committee Qe_k and Qi_k are perfectly positively correlated (both scaled by sc_k),
    # so std_k(Qe_k + Qi_k) == std_k(Qe_k) + std_k(Qi_k) > sqrt(std_Qe^2 + std_Qi^2)!
    comm = TGLFNNCommittee(str(mdir), backend="numpy", use_edge_models=False)
    X_test = comm.pack_features(inp)[25:26].copy()
    X_test[0, comm.xnames.index("RLTS_1")] = 2.5
    X_test[0, comm.xnames.index("RLTS_2")] = 2.5
    ens = comm.predict_ensemble(X_test)
    std_sum = float(np.std(ens[:, 0, 2] + ens[:, 0, 3], ddof=1))
    std_quad = float(np.hypot(np.std(ens[:, 0, 2], ddof=1), np.std(ens[:, 0, 3], ddof=1)))
    assert std_sum > 1.35 * std_quad


# ---------------------------------------------------------------------------
# Real 20-member TGLF-NN ONNX committee tests (when materialized locally)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_local_tglfnn(), reason="Local TGLF-NN ONNX models not present")
def test_pure_numpy_tglfnn_stiffness_speed_and_monotonic_flux():
    # Verify resolve_tglfnn_model_dir("auto") never selects an edge-only committee
    auto_dir = resolve_tglfnn_model_dir("auto")
    assert "edge" not in os.path.basename(os.path.normpath(auto_dir))

    psi = np.linspace(0.0, 1.0, 41)
    ne = 1.2e20 * (1.0 - 0.65 * (psi ** 1.4))
    te = 8.0e3 * ((1.0 - 0.88 * psi) ** 1.3) + 150.0
    ni = 0.88 * ne
    ti = 7.5e3 * ((1.0 - 0.88 * psi) ** 1.3) + 150.0
    zeff = np.full_like(psi, 1.6)
    j = 2.0e6 * (1.0 - psi)

    bl = Baseline(
        psi_N=psi, j_phi=j, j_inductive=0.7 * j, j_BS=0.3 * j,
        psi_N_kinetic=psi, ne=ne, te=te, ni=ni, ti=ti, Zeff=zeff,
        Ip_target=2.5e6, l_i_target=0.95, provenance="imas",
    )
    mg = miller_from_baseline(bl, R0_fallback=1.75, a_fallback=0.58, B0_fallback=5.0)

    # Warmup load + timed evaluation
    _ = evaluate_tglfnn_stiffness(mg, psi, ne, te, ni, ti, zeff)
    t0 = time.perf_counter()
    stiff = evaluate_tglfnn_stiffness(mg, psi, ne, te, ni, ti, zeff)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert dt_ms < 250.0, f"Expected fast pure-NumPy TGLF-NN evaluation, took {dt_ms:.1f} ms"
    core = (stiff.rho_tor_norm >= 0.30) & (stiff.rho_tor_norm <= 0.80)
    assert np.all(np.isfinite(stiff.Qe_gB))
    assert np.all(np.isfinite(stiff.Qi_gB))
    assert np.median(stiff.dQe_dRLTe_gB[core]) > 0.01
    assert np.median(stiff.dQi_dRLTi_gB[core]) > 0.01
    assert np.all(stiff.rlte_crit[core] >= 0.5)
    assert np.all(stiff.rlti_crit[core] >= 0.5)
    assert np.median(stiff.rlte_crit[core]) > 1.5
    assert np.median(stiff.rlti_crit[core]) > 1.5

    # Standalone solve_tglfnn_flux_excursions continuity across scan boundaries & hollow Te
    te_hollow = 4.0e3 * (1.0 + 0.30 * np.exp(-((psi - 0.25) / 0.14) ** 2) - 0.85 * (psi ** 1.4)) + 150.0
    e_hi_a = solve_tglfnn_flux_excursions(mg, psi, ne, te_hollow, ni, ti, zeff, np.full_like(psi, 400.0), np.full_like(psi, 400.0))
    e_hi_b = solve_tglfnn_flux_excursions(mg, psi, ne, te_hollow, ni, ti, zeff, np.full_like(psi, 400.4), np.full_like(psi, 400.4))
    np.testing.assert_allclose(e_hi_a["dkappa_ti_free"], e_hi_b["dkappa_ti_free"], atol=1e-2)
    e_lo_a = solve_tglfnn_flux_excursions(mg, psi, ne, te_hollow, ni, ti, zeff, np.zeros_like(psi), np.zeros_like(psi))
    e_lo_b = solve_tglfnn_flux_excursions(mg, psi, ne, te_hollow, ni, ti, zeff, np.full_like(psi, 1.0e-4), np.full_like(psi, 1.0e-4))
    np.testing.assert_allclose(e_lo_a["dkappa_ti_free"], e_lo_b["dkappa_ti_free"], atol=0.15)


@pytest.mark.skipif(not _has_local_tglfnn(), reason="Local TGLF-NN ONNX models not present")
def test_pure_numpy_tglfnn_matches_julia_turbulent_transport_to_1e6():
    """Exact numerical parity against FUSE.TurbulentTransport.run_tglfnn."""
    comm = TGLFNNCommittee(DEFAULT_TGLFNN_MODEL, backend="numpy")
    vals = {
        "AS_2": [0.85], "AS_3": [0.025],
        "TAUS_2": [0.95], "TAUS_3": [0.95],
        "RLNS_1": [1.2], "RLNS_2": [1.1], "RLNS_3": [1.4],
        "RLTS_1": [2.4], "RLTS_2": [2.2], "RLTS_3": [2.2],
        "BETAE": [0.006], "DEBYE": [0.03], "XNUE": [0.08], "ZEFF": [1.75],
        "RMIN_LOC": [0.55], "RMAJ_LOC": [3.1], "ZMAJ_LOC": [0.0],
        "DRMAJDX_LOC": [-0.14], "DZMAJDX_LOC": [0.0],
        "KAPPA_LOC": [1.55], "DELTA_LOC": [0.22], "ZETA_LOC": [-0.02],
        "S_KAPPA_LOC": [0.18], "S_DELTA_LOC": [0.25], "S_ZETA_LOC": [-0.03],
        "Q_LOC": [1.85], "Q_PRIME_LOC": [14.5], "P_PRIME_LOC": [-0.09],
        "VEXB_SHEAR": [0.0], "VPAR_1": [0.0], "VPAR_SHEAR_1": [0.0],
    }
    pred = comm.predict(vals)
    # Reference output from Julia FUSE.TurbulentTransport.run_tglfnn(vals; model_filename="sat2_em_d3d_azf-1_withnegD")
    julia_ref = {
        "OUT_P_ions": 0.03146062842857424,
        "OUT_Q_elec": 0.8243894307383142,
        "OUT_Q_ions": 0.16187674812897007,
        "OUT_G_elec": -0.012505941408929471,
    }
    for key, ref_val in julia_ref.items():
        assert np.isclose(float(pred[key][0]), ref_val, atol=2e-6), (
            f"{key}: got {pred[key][0]} vs Julia {ref_val}"
        )


@pytest.mark.skipif(not _has_local_tglfnn(), reason="Local TGLF-NN ONNX models not present")
def test_clean_room_tglfnn_excursions_and_explicit_zeff_prior():
    """Verify parameter-free Gauss-Hermite axis regularity, unclamped ell > 1 in
    stiff burning cores, flux_uncertainty_rel=0.0 committee-only mode, and
    preservation of explicit Zeff scalar prior."""
    from bouquet.stiffness import _exact_ell_psi_from_ell_r, compute_physics_length_scales

    psi = np.linspace(0.0, 1.0, 65)
    r_m = 0.57 * np.sqrt(psi)
    # At r=0, psi(r) = (r/a)^2 -> exact E[psi(|ell_r Z|)^2]^(1/2) = sqrt(3) * (ell_r/a)^2 > 0
    ell_ax = _exact_ell_psi_from_ell_r(psi, r_m, 0.57, np.full_like(psi, 0.57))
    assert ell_ax[0] > 1.0, f"Unclamped stiff core ell_psi(0) should exceed 1.0, got {ell_ax[0]}"

    from tests.test_stiffness_and_hollow import _hmode_profile

    ne_d = _hmode_profile(psi, 4.5e19, 3.2e19, 0.8e19, psi_ped=0.93)
    te_d = _hmode_profile(psi, 2.8e3, 0.8e3, 80.0, psi_ped=0.93)
    j_d = (1.1e6 / (np.pi * 0.60 ** 2 * 1.80)) * (1.0 - psi)
    bl_d3d = Baseline(
        psi_N=psi, j_phi=j_d, j_inductive=0.8 * j_d, j_BS=0.2 * j_d,
        psi_N_kinetic=psi, ne=ne_d, te=te_d, ni=0.85 * ne_d, ti=1.15 * te_d,
        Zeff=np.full_like(psi, 2.0), Ip_target=1.1e6, l_i_target=0.9, provenance="imas",
    )
    mg_d = miller_from_baseline(
        bl_d3d, R0_fallback=1.67, a_fallback=0.60, B0_fallback=2.0,
        kappa_edge_fallback=1.80, delta_edge_fallback=0.45,
    )
    chi_d = (5.0e6 * (1.0 - np.exp(-((mg_d.rho_tor_norm / 0.35) ** 2))) / np.maximum(mg_d.surface_m2, 1.0)) / (
        ne_d * 1.602176634e-19 * np.maximum(np.abs(np.gradient(te_d, mg_d.r_min_m)), 50.0)
    )
    bl_d3d.aux = {"miller_geometry": mg_d, "chi_e": chi_d, "chi_i": chi_d, "omega_tor": 8.0e4 * (1.0 - 0.8 * psi)}
    u_d3d = {
        "sigma_ne": 0.10 * ne_d, "sigma_te": 0.15 * te_d,
        "sigma_ni": 0.10 * 0.85 * ne_d, "sigma_ti": 0.15 * 1.15 * te_d,
        "sigma_jphi": 0.10 * j_d,
    }

    ne_s = _hmode_profile(psi, 3.2e20, 2.4e20, 0.5e20, psi_ped=0.93)
    te_s = _hmode_profile(psi, 20.0e3, 4.4e3, 100.0, psi_ped=0.93)
    j_s = (8.7e6 / (np.pi * 0.57 ** 2 * 1.97)) * (1.0 - psi)
    bl_sparc = Baseline(
        psi_N=psi, j_phi=j_s, j_inductive=0.8 * j_s, j_BS=0.2 * j_s,
        psi_N_kinetic=psi, ne=ne_s, te=te_s, ni=0.9 * ne_s, ti=0.95 * te_s,
        Zeff=np.full_like(psi, 1.5), Ip_target=8.7e6, l_i_target=0.9, provenance="imas",
    )
    mg_s = miller_from_baseline(
        bl_sparc, R0_fallback=1.85, a_fallback=0.57, B0_fallback=12.2,
        kappa_edge_fallback=1.97, delta_edge_fallback=0.54,
    )
    bl_sparc.aux = {"miller_geometry": mg_s}
    u_sparc = {
        "sigma_ne": 0.10 * ne_s, "sigma_te": 0.15 * te_s,
        "sigma_ni": 0.10 * 0.9 * ne_s, "sigma_ti": 0.15 * 0.95 * te_s,
        "sigma_jphi": 0.10 * j_s,
        "j_ls": 0.29,
        "aux_length_scales": {"zeff": 0.37},
    }

    s_d3d = compute_physics_length_scales(
        bl_d3d, u_d3d, mode="tier2_tglfnn", tritium_fraction=0.0
    )
    s_sparc = compute_physics_length_scales(
        bl_sparc, u_sparc, mode="tier2_tglfnn", tritium_fraction=0.5
    )
    s_epistemic = compute_physics_length_scales(
        bl_sparc, u_sparc, mode="tier2_tglfnn", flux_uncertainty_rel=0.0, tritium_fraction=0.5
    )
    s_skin = compute_physics_length_scales(
        bl_sparc, u_sparc, mode="tier2_tglfnn", current_evolution_time_s=0.5, tritium_fraction=0.5
    )

    core = (s_d3d.rho_kin >= 0.30) & (s_d3d.rho_kin <= 0.75)
    assert np.median(s_sparc.ti_ls[core]) > np.median(s_d3d.ti_ls[core])
    assert np.all(np.isfinite(s_epistemic.ti_ls)) and np.all(s_epistemic.ti_ls > 0.0)
    # Explicit user Zeff and j_phi length-scale priors are preserved verbatim unless skin time set
    np.testing.assert_allclose(s_sparc.zeff_ls, 0.37)
    np.testing.assert_allclose(s_sparc.j_ls, 0.29)
    assert not np.allclose(s_skin.j_ls, 0.29)
