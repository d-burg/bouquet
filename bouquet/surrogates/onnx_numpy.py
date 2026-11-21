"""Zero-dependency pure-NumPy ONNX computational-graph loader and evaluator.

Parses standard ONNX ``ModelProto`` protobuf files directly using Python's
built-in ``struct`` module and executes MLP / ResNet operator graphs (``Gemm``,
``MatMul``, ``Add``, ``Sub``, ``Mul``, ``Div``, ``Elu``, ``Relu``,
``LeakyRelu``, ``Tanh``, ``Sigmoid``, ``Selu``, ``Softplus``, ``Gelu``,
``Identity``, ``Reshape``, ``Transpose``, ``Concat``) in vectorized NumPy.

This allows ``bouquet`` to evaluate pre-trained ``TGLF-NN`` ONNX committees
out of the box with zero external ML runtime dependencies, while optionally
supporting ``onnxruntime`` when installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import erf

_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# ONNX TensorProto.DataType enum values
_DTYPE_MAP = {
    1: np.dtype("<f4"),   # FLOAT
    2: np.dtype("u1"),    # UINT8
    3: np.dtype("i1"),    # INT8
    5: np.dtype("<i2"),   # INT16
    6: np.dtype("<i4"),   # INT32
    7: np.dtype("<i8"),   # INT64
    10: np.dtype("<f2"),  # FLOAT16
    11: np.dtype("<f8"),  # DOUBLE
}


def is_lfs_pointer_file(path: str) -> bool:
    """Return True if ``path`` is an unmaterialized Git LFS pointer stub."""
    if not os.path.isfile(path):
        return False
    sz = os.path.getsize(path)
    if sz > 512 or sz < len(_LFS_PREFIX):
        return False
    with open(path, "rb") as fh:
        return fh.read(len(_LFS_PREFIX)) == _LFS_PREFIX


def _read_varint(buf: memoryview, pos: int) -> Tuple[int, int]:
    val = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def _to_signed64(u: int) -> int:
    return u - (1 << 64) if (u & (1 << 63)) else u


def _iter_proto(buf: memoryview, start: int = 0, end: Optional[int] = None):
    if end is None:
        end = len(buf)
    pos = start
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:
            val, pos = _read_varint(buf, pos)
        elif wire_type == 1:
            val = buf[pos:pos + 8]
            pos += 8
        elif wire_type == 2:
            ln, pos = _read_varint(buf, pos)
            val = (pos, pos + ln)
            pos += ln
        elif wire_type == 5:
            val = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire_type={wire_type}")
        yield field_num, wire_type, val


def _parse_tensor(buf: memoryview, start: int, end: int) -> Tuple[str, np.ndarray]:
    dims: List[int] = []
    data_type = 1
    name = ""
    raw_span: Optional[Tuple[int, int]] = None
    float_vals: List[float] = []
    int64_vals: List[int] = []
    int32_vals: List[int] = []
    double_vals: List[float] = []

    for fnum, wtype, val in _iter_proto(buf, start, end):
        if fnum == 1:  # dims
            if wtype == 0:
                dims.append(_to_signed64(val))
            elif wtype == 2:
                p = val[0]
                while p < val[1]:
                    d, p = _read_varint(buf, p)
                    dims.append(_to_signed64(d))
        elif fnum == 2 and wtype == 0:  # data_type
            data_type = int(val)
        elif fnum == 4:  # float_data
            if wtype == 2:
                arr = np.frombuffer(buf[val[0]:val[1]], dtype="<f4")
                float_vals.extend(arr.tolist())
            elif wtype == 5:
                float_vals.append(struct.unpack("<f", val)[0])
        elif fnum == 5:  # int32_data
            if wtype == 2:
                p = val[0]
                while p < val[1]:
                    v, p = _read_varint(buf, p)
                    int32_vals.append(_to_signed64(v))
            elif wtype == 0:
                int32_vals.append(_to_signed64(val))
        elif fnum == 7:  # int64_data
            if wtype == 2:
                p = val[0]
                while p < val[1]:
                    v, p = _read_varint(buf, p)
                    int64_vals.append(_to_signed64(v))
            elif wtype == 0:
                int64_vals.append(_to_signed64(val))
        elif fnum == 8 and wtype == 2:  # name
            name = bytes(buf[val[0]:val[1]]).decode("utf-8")
        elif fnum == 9 and wtype == 2:  # raw_data
            raw_span = val
        elif fnum == 10:  # double_data
            if wtype == 2:
                arr = np.frombuffer(buf[val[0]:val[1]], dtype="<f8")
                double_vals.extend(arr.tolist())
            elif wtype == 1:
                double_vals.append(struct.unpack("<d", val)[0])

    dt = _DTYPE_MAP.get(data_type, np.dtype("<f4"))
    if raw_span is not None:
        arr = np.frombuffer(buf[raw_span[0]:raw_span[1]], dtype=dt).copy()
    elif float_vals:
        arr = np.asarray(float_vals, dtype=np.float32)
    elif double_vals:
        arr = np.asarray(double_vals, dtype=np.float64)
    elif int64_vals:
        arr = np.asarray(int64_vals, dtype=np.int64)
    elif int32_vals:
        arr = np.asarray(int32_vals, dtype=np.int32)
    else:
        arr = np.empty(0, dtype=dt)

    if dims:
        arr = arr.reshape(tuple(dims))
    return name, arr


def _parse_attribute(buf: memoryview, start: int, end: int):
    name = ""
    val = None
    ints: List[int] = []
    floats: List[float] = []
    for fnum, wtype, v in _iter_proto(buf, start, end):
        if fnum == 1 and wtype == 2:  # name
            name = bytes(buf[v[0]:v[1]]).decode("utf-8")
        elif fnum == 2 and wtype == 5:  # f
            val = float(struct.unpack("<f", v)[0])
        elif fnum == 3 and wtype == 0:  # i
            val = _to_signed64(v)
        elif fnum == 4 and wtype == 2:  # s
            val = bytes(buf[v[0]:v[1]]).decode("utf-8", errors="replace")
        elif fnum == 5 and wtype == 2:  # t (TensorProto)
            _, val = _parse_tensor(buf, v[0], v[1])
        elif fnum == 7:  # floats
            if wtype == 5:
                floats.append(float(struct.unpack("<f", v)[0]))
            elif wtype == 2:
                floats.extend(np.frombuffer(buf[v[0]:v[1]], dtype="<f4").tolist())
        elif fnum == 8:  # ints
            if wtype == 0:
                ints.append(_to_signed64(v))
            elif wtype == 2:
                p = v[0]
                while p < v[1]:
                    iv, p = _read_varint(buf, p)
                    ints.append(_to_signed64(iv))
    if val is None:
        if ints:
            val = ints
        elif floats:
            val = floats
    return name, val


def _parse_value_info_name(buf: memoryview, start: int, end: int) -> str:
    for fnum, wtype, val in _iter_proto(buf, start, end):
        if fnum == 1 and wtype == 2:
            return bytes(buf[val[0]:val[1]]).decode("utf-8")
    return ""


@dataclass
class _ONNXNode:
    op_type: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attrs: Dict[str, object]


class PureNumpyONNXModel:
    """Lightweight pure-NumPy evaluator for feed-forward & ResNet ONNX graphs."""

    def __init__(self, path: str):
        if is_lfs_pointer_file(path):
            raise RuntimeError(
                f"ONNX model at '{path}' is an unmaterialized Git LFS pointer stub."
            )
        with open(path, "rb") as fh:
            raw = fh.read()
        mv = memoryview(raw)

        self.initializers: Dict[str, np.ndarray] = {}
        self.nodes: List[_ONNXNode] = []
        graph_inputs: List[str] = []
        graph_outputs: List[str] = []

        for fnum, wtype, val in _iter_proto(mv):
            if fnum == 7 and wtype == 2:  # ModelProto.graph
                gs, ge = val
                for gf, gw, gv in _iter_proto(mv, gs, ge):
                    if gf == 1 and gw == 2:  # node
                        ns, ne = gv
                        op_type = ""
                        inps: List[str] = []
                        outs: List[str] = []
                        attrs: Dict[str, object] = {}
                        for nf, nw, nv in _iter_proto(mv, ns, ne):
                            if nf == 1 and nw == 2:
                                inps.append(bytes(mv[nv[0]:nv[1]]).decode("utf-8"))
                            elif nf == 2 and nw == 2:
                                outs.append(bytes(mv[nv[0]:nv[1]]).decode("utf-8"))
                            elif nf == 4 and nw == 2:
                                op_type = bytes(mv[nv[0]:nv[1]]).decode("utf-8")
                            elif nf == 5 and nw == 2:
                                ak, av = _parse_attribute(mv, nv[0], nv[1])
                                attrs[ak] = av
                        self.nodes.append(_ONNXNode(op_type, tuple(inps), tuple(outs), attrs))
                    elif gf == 5 and gw == 2:  # initializer
                        tname, tarr = _parse_tensor(mv, gv[0], gv[1])
                        self.initializers[tname] = tarr
                    elif gf == 11 and gw == 2:  # input
                        graph_inputs.append(_parse_value_info_name(mv, gv[0], gv[1]))
                    elif gf == 12 and gw == 2:  # output
                        graph_outputs.append(_parse_value_info_name(mv, gv[0], gv[1]))

        init_set = set(self.initializers)
        self.input_names = [n for n in graph_inputs if n and n not in init_set]
        self.output_names = [n for n in graph_outputs if n]
        if not self.input_names or not self.output_names:
            raise ValueError(f"Malformed ONNX graph in '{path}': missing inputs/outputs")

        # Pre-transpose Gemm weights where both W and optional bias b live in initializers
        self._gemm_cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], float, float, int]] = {}
        for idx, node in enumerate(self.nodes):
            if (
                node.op_type == "Gemm"
                and len(node.inputs) >= 2
                and node.inputs[1] in self.initializers
                and (len(node.inputs) < 3 or not node.inputs[2] or node.inputs[2] in self.initializers)
            ):
                W = self.initializers[node.inputs[1]]
                if int(node.attrs.get("transB", 0)):
                    W = np.ascontiguousarray(W.T)
                b = (
                    self.initializers[node.inputs[2]]
                    if len(node.inputs) >= 3 and node.inputs[2] in self.initializers
                    else None
                )
                alpha = float(node.attrs.get("alpha", 1.0))
                beta = float(node.attrs.get("beta", 1.0))
                transA = int(node.attrs.get("transA", 0))
                self._gemm_cache[idx] = (W, b, alpha, beta, transA)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate graph on ``x`` of shape ``(batch, n_features)``."""
        x_arr = np.asarray(x, dtype=np.float32)
        squeeze_out = False
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
            squeeze_out = True

        env: Dict[str, np.ndarray] = dict(self.initializers)
        env[self.input_names[0]] = x_arr

        for idx, node in enumerate(self.nodes):
            op = node.op_type
            if idx in self._gemm_cache:
                W, b, alpha, beta, transA = self._gemm_cache[idx]
                A = env[node.inputs[0]]
                if transA:
                    A = A.T
                y = (alpha * (A @ W)) if alpha != 1.0 else (A @ W)
                if b is not None:
                    y = y + (beta * b if beta != 1.0 else b)
                env[node.outputs[0]] = y
            elif op == "Gemm":
                A = env[node.inputs[0]]
                B = env[node.inputs[1]]
                if int(node.attrs.get("transA", 0)):
                    A = A.T
                if int(node.attrs.get("transB", 0)):
                    B = B.T
                alpha = float(node.attrs.get("alpha", 1.0))
                beta = float(node.attrs.get("beta", 1.0))
                y = (alpha * (A @ B)) if alpha != 1.0 else (A @ B)
                if len(node.inputs) >= 3 and node.inputs[2]:
                    C = env[node.inputs[2]]
                    y = y + (beta * C if beta != 1.0 else C)
                env[node.outputs[0]] = y
            elif op == "Elu":
                a = env[node.inputs[0]]
                alpha = float(node.attrs.get("alpha", 1.0))
                # Safe clip on negative branch to prevent exp overflow
                neg = np.minimum(a, 0.0)
                env[node.outputs[0]] = np.where(a > 0.0, a, alpha * np.expm1(np.maximum(neg, -80.0)))
            elif op == "Relu":
                env[node.outputs[0]] = np.maximum(env[node.inputs[0]], 0.0)
            elif op == "LeakyRelu":
                a = env[node.inputs[0]]
                alpha = float(node.attrs.get("alpha", 0.01))
                env[node.outputs[0]] = np.where(a >= 0.0, a, alpha * a)
            elif op == "Gelu":
                a = env[node.inputs[0]]
                approx = str(node.attrs.get("approximate", "none")).lower()
                if approx == "tanh":
                    env[node.outputs[0]] = 0.5 * a * (
                        1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (a + 0.044715 * (a ** 3)))
                    )
                else:
                    env[node.outputs[0]] = 0.5 * a * (1.0 + erf(a / np.sqrt(2.0)))
            elif op == "Tanh":
                env[node.outputs[0]] = np.tanh(env[node.inputs[0]])
            elif op == "Sigmoid":
                a = np.clip(env[node.inputs[0]], -80.0, 80.0)
                env[node.outputs[0]] = 1.0 / (1.0 + np.exp(-a))
            elif op == "Softplus":
                a = env[node.inputs[0]]
                env[node.outputs[0]] = np.log1p(np.exp(-np.abs(a))) + np.maximum(a, 0.0)
            elif op == "Selu":
                a = env[node.inputs[0]]
                alpha = float(node.attrs.get("alpha", 1.6732631921768188))
                gamma = float(node.attrs.get("gamma", 1.0507009873554805))
                neg = np.minimum(a, 0.0)
                env[node.outputs[0]] = gamma * np.where(a > 0.0, a, alpha * np.expm1(np.maximum(neg, -80.0)))
            elif op == "Add":
                env[node.outputs[0]] = env[node.inputs[0]] + env[node.inputs[1]]
            elif op == "Sub":
                env[node.outputs[0]] = env[node.inputs[0]] - env[node.inputs[1]]
            elif op == "Mul":
                env[node.outputs[0]] = env[node.inputs[0]] * env[node.inputs[1]]
            elif op == "Div":
                env[node.outputs[0]] = env[node.inputs[0]] / env[node.inputs[1]]
            elif op == "MatMul":
                env[node.outputs[0]] = env[node.inputs[0]] @ env[node.inputs[1]]
            elif op == "Identity":
                env[node.outputs[0]] = env[node.inputs[0]]
            elif op == "Constant":
                if "value" in node.attrs:
                    env[node.outputs[0]] = np.asarray(node.attrs["value"])
                elif "value_float" in node.attrs:
                    env[node.outputs[0]] = np.asarray(node.attrs["value_float"], dtype=np.float32)
                elif "value_int" in node.attrs:
                    env[node.outputs[0]] = np.asarray(node.attrs["value_int"], dtype=np.int64)
                elif "value_floats" in node.attrs:
                    env[node.outputs[0]] = np.asarray(node.attrs["value_floats"], dtype=np.float32)
                elif "value_ints" in node.attrs:
                    env[node.outputs[0]] = np.asarray(node.attrs["value_ints"], dtype=np.int64)
                else:
                    raise ValueError("PureNumpyONNXModel: Constant node missing value attribute")
            elif op == "Reshape":
                data = env[node.inputs[0]]
                raw_shape = [int(d) for d in np.ravel(env[node.inputs[1]])]
                allowzero = int(node.attrs.get("allowzero", 0))
                resolved: List[int] = []
                for k_dim, dim_val in enumerate(raw_shape):
                    if dim_val == 0 and not allowzero:
                        if k_dim >= data.ndim:
                            raise ValueError(
                                f"PureNumpyONNXModel: Reshape 0-dim at axis {k_dim} "
                                f"exceeds input rank {data.ndim}"
                            )
                        resolved.append(int(data.shape[k_dim]))
                    else:
                        resolved.append(dim_val)
                env[node.outputs[0]] = data.reshape(tuple(resolved))
            elif op == "Transpose":
                perm = node.attrs.get("perm")
                env[node.outputs[0]] = np.transpose(env[node.inputs[0]], axes=perm)
            elif op == "Concat":
                axis = int(node.attrs.get("axis", 0))
                env[node.outputs[0]] = np.concatenate([env[i] for i in node.inputs], axis=axis)
            else:
                raise NotImplementedError(f"PureNumpyONNXModel: unsupported ONNX op '{op}'")

        out = np.asarray(env[self.output_names[0]], dtype=np.float64)
        return out[0] if squeeze_out else out
