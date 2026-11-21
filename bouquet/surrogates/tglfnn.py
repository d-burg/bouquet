"""Standalone Python ``TGLF-NN`` committee evaluator and stiffness calculator.

Supports pre-trained ``TurbulentTransport.jl`` / GACODE ONNX neural-network
committees (e.g. ``sat2_em_d3d_azf-1_withnegD``, ``sat3_em_d3d_azf-1_withnegD``,
``sat3_em_mastu_azf-1``) in pure NumPy via
:class:`~bouquet.surrogates.onnx_numpy.PureNumpyONNXModel` (zero ML framework
dependencies) or ``onnxruntime`` when installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .onnx_numpy import PureNumpyONNXModel, is_lfs_pointer_file

DEFAULT_TGLFNN_MODEL = "sat2_em_d3d_azf-1_withnegD"

_RADIAL_D3D_MODELS = frozenset({
    "sat0quench_em_d3d_azf+1_withnegD",
    "sat1_em_d3d_azf-1_withnegD",
    "sat2_em_d3d_azf-1_withnegD",
    "sat3_em_d3d_azf-1_withnegD",
})


@dataclass
class TGLFNNStiffnessProfile:
    """Radial transport fluxes, stiffness Jacobians, and critical gradients."""

    psi_N: np.ndarray
    rho_tor_norm: np.ndarray
    model_name: str
    # Gyro-Bohm normalized baseline fluxes
    Qe_gB: np.ndarray
    Qi_gB: np.ndarray
    Ge_gB: np.ndarray
    Pi_gB: np.ndarray
    # Epistemic committee standard deviations (gyro-Bohm units)
    Qe_gB_std: np.ndarray
    Qi_gB_std: np.ndarray
    # Physical surface fluxes [W/m^2]
    Qe_W_m2: np.ndarray
    Qi_W_m2: np.ndarray
    # Local stiffness derivatives d(Q_gB) / d(R/L_T) and d(Gamma_gB) / d(R/L_n)
    dQe_dRLTe_gB: np.ndarray
    dQi_dRLTi_gB: np.ndarray
    dGe_dRLne_gB: np.ndarray
    # Effective critical inverse scale lengths (R_0 / L_s)_crit
    rlte_base: np.ndarray
    rlti_base: np.ndarray
    rlne_base: np.ndarray
    rlte_crit: np.ndarray
    rlti_crit: np.ndarray
    # Effective turbulent thermal diffusivities [m^2/s]
    chi_e_turb_m2_s: np.ndarray
    chi_i_turb_m2_s: np.ndarray
    Ge_gB_std: Optional[np.ndarray] = None


def _is_materialized_model_dir(cand: str) -> bool:
    if not os.path.isdir(cand):
        return False
    onnx_files = sorted(glob.glob(os.path.join(cand, "*.onnx")))
    xnames_f = os.path.join(cand, "xnames.txt")
    return bool(onnx_files and os.path.isfile(xnames_f) and not is_lfs_pointer_file(onnx_files[0]))


def _candidate_model_roots() -> List[str]:
    roots: List[str] = []
    env_dir = os.environ.get("BOUQUET_TGLFNN_DIR")
    if env_dir and os.path.isdir(os.path.expanduser(env_dir)):
        roots.append(os.path.expanduser(env_dir))
    cache_dir = os.path.expanduser("~/.cache/bouquet/models")
    if os.path.isdir(cache_dir):
        roots.append(cache_dir)
    julia_glob = sorted(
        glob.glob(os.path.expanduser("~/.julia/packages/TurbulentTransport/*/models")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0,
        reverse=True,
    )
    roots.extend(julia_glob)
    return roots


def resolve_tglfnn_model_dir(model_spec: str = DEFAULT_TGLFNN_MODEL) -> str:
    """Locate a materialized ``TGLF-NN`` ONNX committee directory."""
    expanded = os.path.expanduser(str(model_spec))
    if _is_materialized_model_dir(expanded):
        return expanded
    if os.path.isdir(expanded) and model_spec != "auto":
        return expanded

    candidates = (
        [DEFAULT_TGLFNN_MODEL, "sat3_em_d3d_azf-1_withnegD", "sat3_em_mastu_azf-1"]
        if model_spec == "auto"
        else [str(model_spec)]
    )

    for root in _candidate_model_roots():
        for name in candidates:
            cand = os.path.join(root, name)
            if _is_materialized_model_dir(cand):
                return cand
            if os.path.basename(os.path.normpath(root)) == name and _is_materialized_model_dir(root):
                return root

    raise FileNotFoundError(
        f"Could not find materialized TGLF-NN ONNX committee '{model_spec}'. "
        "Set BOUQUET_TGLFNN_DIR, populate ~/.cache/bouquet/models, or ensure "
        "TurbulentTransport.jl ONNX models are present."
    )


class TGLFNNCommittee:
    """20-member ``TGLF-NN`` neural-network committee evaluator."""

    def __init__(
        self,
        model_spec: str = DEFAULT_TGLFNN_MODEL,
        *,
        backend: str = "auto",
        use_edge_models: bool = True,
    ):
        self.model_dir = resolve_tglfnn_model_dir(model_spec)
        self.name = os.path.basename(os.path.normpath(self.model_dir))
        self.use_edge_models = bool(use_edge_models)

        with open(os.path.join(self.model_dir, "xnames.txt"), "r") as fh:
            self.xnames: Tuple[str, ...] = tuple(fh.read().strip().split())
        with open(os.path.join(self.model_dir, "ynames.txt"), "r") as fh:
            self.ynames: Tuple[str, ...] = tuple(fh.read().strip().split())

        # Reject residual GKNN models that require TGLF outputs as inputs unless parent chained
        if any(xn.startswith("OUT_") for xn in self.xnames):
            raise ValueError(
                f"Model '{self.name}' is a residual GKNN model requiring TGLF outputs "
                f"({[x for x in self.xnames if x.startswith('OUT_')]}). "
                "Select a direct plasma-input TGLF-NN committee such as "
                f"'{DEFAULT_TGLFNN_MODEL}' or 'sat3_em_mastu_azf-1'."
            )

        self.xm = np.loadtxt(os.path.join(self.model_dir, "xm.txt"), dtype=np.float64)
        self.xsigma = np.loadtxt(os.path.join(self.model_dir, "xsigma.txt"), dtype=np.float64)
        self.ym = np.loadtxt(os.path.join(self.model_dir, "ym.txt"), dtype=np.float64)
        self.ysigma = np.loadtxt(os.path.join(self.model_dir, "ysigma.txt"), dtype=np.float64)

        self._base_fields: List[str] = []
        self._is_log10 = np.zeros(len(self.xnames), dtype=bool)
        for i, xn in enumerate(self.xnames):
            if xn.endswith("_log10"):
                self._base_fields.append(xn[:-6])
                self._is_log10[i] = True
            else:
                self._base_fields.append(xn)
        self._idx_rmin: Optional[int] = (
            self.xnames.index("RMIN_LOC") if "RMIN_LOC" in self.xnames else None
        )

        onnx_paths = sorted(
            glob.glob(os.path.join(self.model_dir, "*.onnx")),
            key=lambda p: (len(os.path.basename(p)), os.path.basename(p)),
        )
        if not onnx_paths:
            raise FileNotFoundError(f"No .onnx files found in '{self.model_dir}'")

        use_ort = False
        if backend in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort  # type: ignore
                self._ort_sessions = [
                    ort.InferenceSession(p, providers=["CPUExecutionProvider"])
                    for p in onnx_paths
                ]
                self._ort_in_name = self._ort_sessions[0].get_inputs()[0].name
                use_ort = True
            except ImportError:
                if backend == "onnxruntime":
                    raise
        self.backend = "onnxruntime" if use_ort else "numpy"
        if not use_ort:
            self._np_models = [PureNumpyONNXModel(p) for p in onnx_paths]
        self.n_models = len(onnx_paths)

        self._edge_loaded = False
        self._nearedge_comm: Optional[TGLFNNCommittee] = None
        self._edge_comm: Optional[TGLFNNCommittee] = None

    def _ensure_edge_committees(self) -> None:
        if self._edge_loaded:
            return
        self._edge_loaded = True
        if not self.use_edge_models or self.name not in _RADIAL_D3D_MODELS:
            return
        parent = os.path.dirname(os.path.normpath(self.model_dir))
        near_dir = os.path.join(parent, self.name.replace("d3d", "d3dnearedge"))
        edge_dir = os.path.join(parent, self.name.replace("d3d", "d3dedge"))
        if _is_materialized_model_dir(near_dir):
            cand = TGLFNNCommittee(near_dir, backend=self.backend, use_edge_models=False)
            if (
                cand.n_models == self.n_models
                and cand.xnames == self.xnames
                and cand.ynames == self.ynames
            ):
                self._nearedge_comm = cand
        if _is_materialized_model_dir(edge_dir):
            cand = TGLFNNCommittee(edge_dir, backend=self.backend, use_edge_models=False)
            if (
                cand.n_models == self.n_models
                and cand.xnames == self.xnames
                and cand.ynames == self.ynames
            ):
                self._edge_comm = cand

    def pack_features(self, tglf_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Assemble ``(M, N_features)`` matrix from :func:`build_tglf_inputs` dict."""
        first = np.atleast_1d(np.asarray(tglf_inputs["Q_LOC"], dtype=float))
        m_pts = first.size
        mat = np.empty((m_pts, len(self.xnames)), dtype=np.float64)
        for j, (field_name, is_log) in enumerate(zip(self._base_fields, self._is_log10)):
            if field_name not in tglf_inputs:
                raise KeyError(f"TGLFNNCommittee '{self.name}' missing input '{field_name}'")
            col = np.broadcast_to(np.asarray(tglf_inputs[field_name], dtype=float), (m_pts,))
            if is_log:
                mat[:, j] = np.log10(np.maximum(col, 1e-12))
            else:
                mat[:, j] = col
        return mat

    def _eval_raw_ensemble(self, X_arr: np.ndarray) -> np.ndarray:
        X_norm = ((X_arr - self.xm[None, :]) / self.xsigma[None, :]).astype(np.float32)
        preds = np.empty((self.n_models, X_norm.shape[0], len(self.ynames)), dtype=np.float64)
        if self.backend == "onnxruntime":
            for k, sess in enumerate(self._ort_sessions):
                preds[k] = sess.run(None, {self._ort_in_name: X_norm})[0]
        else:
            for k, mdl in enumerate(self._np_models):
                preds[k] = mdl(X_norm)
        return preds * self.ysigma[None, None, :] + self.ym[None, None, :]

    def predict_ensemble(self, X_transformed: np.ndarray) -> np.ndarray:
        """Run all ``K`` committee members on ``(M, N_features)`` (already log10-transformed).

        Returns ``preds_gb`` of shape ``(K, M, N_outputs)`` in denormalized
        gyro-Bohm units, honoring ``TurbulentTransport.run_tglfnn`` radial
        ``d3dnearedge`` (``0.881 <= RMIN_LOC < 0.975``) and ``d3dedge``
        (``RMIN_LOC >= 0.975``) model dispatch when sibling directories exist.
        """
        X_arr = np.asarray(X_transformed, dtype=np.float64)
        preds_gb = self._eval_raw_ensemble(X_arr)
        if (
            self.use_edge_models
            and self._idx_rmin is not None
            and self.name in _RADIAL_D3D_MODELS
            and X_arr.shape[0] > 0
        ):
            rmin_vals = X_arr[:, self._idx_rmin]
            if np.any(rmin_vals >= 0.881):
                self._ensure_edge_committees()
                if self._nearedge_comm is not None:
                    mask_near = (rmin_vals >= 0.881) & (rmin_vals < 0.975)
                    if np.any(mask_near):
                        preds_gb[:, mask_near, :] = self._nearedge_comm._eval_raw_ensemble(X_arr[mask_near])
                if self._edge_comm is not None:
                    mask_edge = rmin_vals >= 0.975
                    if np.any(mask_edge):
                        preds_gb[:, mask_edge, :] = self._edge_comm._eval_raw_ensemble(X_arr[mask_edge])
        return preds_gb

    def predict_matrix(self, X_transformed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run committee on ``(M, N_features)`` (already log10-transformed).

        Returns ``(mean_flux, std_flux)`` each of shape ``(M, N_outputs)`` in
        denormalized gyro-Bohm units matching ``ynames``.
        """
        preds_gb = self.predict_ensemble(X_transformed)
        mean_gb = np.mean(preds_gb, axis=0)
        std_gb = np.std(preds_gb, axis=0, ddof=1 if self.n_models > 1 else 0)
        return mean_gb, std_gb

    def predict(self, tglf_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Evaluate gyro-Bohm fluxes from a :func:`build_tglf_inputs` dict."""
        X = self.pack_features(tglf_inputs)
        mean_gb, std_gb = self.predict_matrix(X)
        out: Dict[str, np.ndarray] = {}
        for j, yn in enumerate(self.ynames):
            out[yn] = mean_gb[:, j]
            out[f"{yn}_std"] = std_gb[:, j]
        return out


@functools.lru_cache(maxsize=8)
def load_tglfnn_committee(
    model_spec: str = DEFAULT_TGLFNN_MODEL,
    backend: str = "auto",
    use_edge_models: bool = True,
) -> TGLFNNCommittee:
    """Cached loader for :class:`TGLFNNCommittee`."""
    return TGLFNNCommittee(
        model_spec=model_spec,
        backend=backend,
        use_edge_models=use_edge_models,
    )


def evaluate_tglfnn_stiffness(
    miller,
    psi_N: np.ndarray,
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    zeff: np.ndarray,
    *,
    Z_imp: float = 6.0,
    A_main: float = 2.0,
    z_fast_m3: Optional[np.ndarray] = None,
    omega_tor_rad_s: Optional[np.ndarray] = None,
    p_thermal_Pa: Optional[np.ndarray] = None,
    model_spec: str = DEFAULT_TGLFNN_MODEL,
    q_crit_gb_threshold: float = 0.08,
) -> TGLFNNStiffnessProfile:
    """Compute radial ``TGLF-NN`` fluxes, stiffness Jacobians, and critical gradients.

    All radial grid points AND finite-difference / threshold-scan evaluations
    are packed into vectorized matrix passes (~3-10 ms total on CPU).
    """
    from ..geometry.miller import build_tglf_inputs, gyrobohm_flux_scales
    from ..physics import ELEMENTARY_CHARGE

    mg = miller.interpolate(psi_N)
    inp = build_tglf_inputs(
        mg, psi_N, ne_m3, te_eV, ni_m3, ti_eV, zeff,
        Z_imp=Z_imp, A_main=A_main, z_fast_m3=z_fast_m3,
        omega_tor_rad_s=omega_tor_rad_s, p_thermal_Pa=p_thermal_Pa,
    )

    comm = load_tglfnn_committee(model_spec)
    X0 = comm.pack_features(inp)
    m_pts = X0.shape[0]

    idx_rlte = comm.xnames.index("RLTS_1")
    idx_rlti = comm.xnames.index("RLTS_2")
    idx_rlts3 = comm.xnames.index("RLTS_3") if "RLTS_3" in comm.xnames else None
    idx_rlne = comm.xnames.index("RLNS_1")
    idx_rlni = comm.xnames.index("RLNS_2")
    idx_rlns3 = comm.xnames.index("RLNS_3") if "RLNS_3" in comm.xnames else None

    idx_qe = comm.ynames.index("OUT_Q_elec")
    idx_qi = comm.ynames.index("OUT_Q_ions")
    idx_ge = comm.ynames.index("OUT_G_elec")
    idx_pi = comm.ynames.index("OUT_P_ions")

    # Standard centered finite-difference step for local Jacobian dQ / d(a/L_s)
    # Co-perturb RLNS_3 with RLNS_1, RLNS_2 to preserve quasineutrality sum_s Z_s AS_s RLNS_s = 0.
    h_aL = 1.0e-2
    batch = np.tile(X0, (7, 1))
    batch[1 * m_pts:2 * m_pts, idx_rlte] += h_aL
    batch[2 * m_pts:3 * m_pts, idx_rlte] -= h_aL
    batch[3 * m_pts:4 * m_pts, idx_rlti] += h_aL
    batch[4 * m_pts:5 * m_pts, idx_rlti] -= h_aL
    if idx_rlts3 is not None:
        batch[3 * m_pts:4 * m_pts, idx_rlts3] += h_aL
        batch[4 * m_pts:5 * m_pts, idx_rlts3] -= h_aL
    batch[5 * m_pts:6 * m_pts, idx_rlne] += h_aL
    batch[5 * m_pts:6 * m_pts, idx_rlni] += h_aL
    batch[6 * m_pts:7 * m_pts, idx_rlne] -= h_aL
    batch[6 * m_pts:7 * m_pts, idx_rlni] -= h_aL
    if idx_rlns3 is not None:
        batch[5 * m_pts:6 * m_pts, idx_rlns3] += h_aL
        batch[6 * m_pts:7 * m_pts, idx_rlns3] -= h_aL

    mean_all, std_all = comm.predict_matrix(batch)
    y0 = mean_all[0:m_pts]
    s0 = std_all[0:m_pts]

    Qe_gB = np.maximum(y0[:, idx_qe], 0.0)
    Qi_gB = np.maximum(y0[:, idx_qi], 0.0)
    Ge_gB = y0[:, idx_ge]
    Pi_gB = y0[:, idx_pi]

    # Convert derivatives from d/d(a/L_s) to d/d(R0/L_s): d/d(R0/L) = (a/R0) * d/d(a/L)
    R_over_a = float(mg.R0_m / max(mg.a_m, 1e-4))
    a_over_R = 1.0 / R_over_a

    dQe_daLTe = (mean_all[1 * m_pts:2 * m_pts, idx_qe] - mean_all[2 * m_pts:3 * m_pts, idx_qe]) / (2.0 * h_aL)
    dQi_daLTi = (mean_all[3 * m_pts:4 * m_pts, idx_qi] - mean_all[4 * m_pts:5 * m_pts, idx_qi]) / (2.0 * h_aL)
    dGe_daLne = (mean_all[5 * m_pts:6 * m_pts, idx_ge] - mean_all[6 * m_pts:7 * m_pts, idx_ge]) / (2.0 * h_aL)

    dQe_dRLTe = np.maximum(dQe_daLTe * a_over_R, 1e-6)
    dQi_dRLTi = np.maximum(dQi_daLTi * a_over_R, 1e-6)
    dGe_dRLne = np.maximum(np.abs(dGe_daLne) * a_over_R, 1e-6)

    # Vectorized scan along a/L_Te and a/L_Ti on [0.1, max(4.0, 1.3*aL_base)] so flat or
    # radiatively hollow cores (aL_base <= 0) still bracket the true critical gradient.
    n_scan = 16
    u_scan = np.linspace(0.0, 1.0, n_scan)[None, :]
    hi_te = np.maximum(4.0, 1.3 * X0[:, idx_rlte])[:, None]
    hi_ti = np.maximum(4.0, 1.3 * X0[:, idx_rlti])[:, None]
    grid_aLTe = 0.1 + (hi_te - 0.1) * u_scan
    grid_aLTi = 0.1 + (hi_ti - 0.1) * u_scan

    scan_te = np.repeat(X0, n_scan, axis=0)
    scan_ti = np.repeat(X0, n_scan, axis=0)
    scan_te[:, idx_rlte] = grid_aLTe.ravel()
    scan_ti[:, idx_rlti] = grid_aLTi.ravel()
    if idx_rlts3 is not None:
        scan_ti[:, idx_rlts3] = grid_aLTi.ravel()

    y_scan_te, _ = comm.predict_matrix(scan_te)
    y_scan_ti, _ = comm.predict_matrix(scan_ti)
    qe_curve = y_scan_te[:, idx_qe].reshape(m_pts, n_scan)
    qi_curve = y_scan_ti[:, idx_qi].reshape(m_pts, n_scan)

    def _find_crit(
        grid_aL: np.ndarray,
        q_curve: np.ndarray,
        q0_val: np.ndarray,
        dq_dRL: np.ndarray,
        rl_base_arr: np.ndarray,
    ) -> np.ndarray:
        rl_crit = np.empty(m_pts, dtype=float)
        for i in range(m_pts):
            above = np.where(q_curve[i] >= q_crit_gb_threshold)[0]
            if above.size > 0 and above[0] > 0:
                k = int(above[0])
                q_lo, q_hi = float(q_curve[i, k - 1]), float(q_curve[i, k])
                x_lo, x_hi = float(grid_aL[i, k - 1]), float(grid_aL[i, k])
                w = np.clip((q_crit_gb_threshold - q_lo) / max(q_hi - q_lo, 1e-9), 0.0, 1.0)
                rl_crit[i] = (x_lo + w * (x_hi - x_lo)) * R_over_a
            elif above.size == 0:
                # Subcritical across the entire [0.1, max(4.0, 1.3*aL_base)] scan
                rl_crit[i] = float(grid_aL[i, -1]) * R_over_a
            else:
                # Background mode already drives flux at grid_aL[i, 0] == 0.1:
                # use supercritical linear tangent intercept
                if float(rl_base_arr[i]) > 0.5 and float(q0_val[i]) >= q_crit_gb_threshold:
                    rl_ref = float(rl_base_arr[i])
                    q_ref = float(q0_val[i])
                    dq_ref = max(float(dq_dRL[i]), 0.1)
                else:
                    rl_ref = float(grid_aL[i, -1]) * R_over_a
                    q_ref = float(q_curve[i, -1])
                    dq_end = (float(q_curve[i, -1] - q_curve[i, -2])
                              / max(float(grid_aL[i, -1] - grid_aL[i, -2]) * R_over_a, 1e-9))
                    dq_ref = max(dq_end, 0.1)
                rl_crit[i] = max(0.5, rl_ref - q_ref / dq_ref)
        return np.clip(rl_crit, 0.5, 25.0)

    rlte_base = inp["RLTS_1"] * R_over_a
    rlti_base = inp["RLTS_2"] * R_over_a
    rlne_base = inp["RLNS_1"] * R_over_a

    rlte_crit = _find_crit(grid_aLTe, qe_curve, Qe_gB, dQe_dRLTe, rlte_base)
    rlti_crit = _find_crit(grid_aLTi, qi_curve, Qi_gB, dQi_dRLTi, rlti_base)

    gb = gyrobohm_flux_scales(mg, ne_m3, te_eV)
    Qe_W_m2 = Qe_gB * gb["Q_gB_imas_W_m2"]
    Qi_W_m2 = Qi_gB * gb["Q_gB_imas_W_m2"]

    # Miller-frame turbulent diffusivity chi_s = Q_s_miller / (n_s * e * |dT_s/dr_min|) [m^2/s],
    # where Q_s_miller = Q_s_gB * Q_gB_W_m2 = Q_s_imas / vprime_miller and OUT_Q_ions sums all ions.
    Qe_miller_W_m2 = Qe_gB * gb["Q_gB_W_m2"]
    Qi_miller_W_m2 = Qi_gB * gb["Q_gB_W_m2"]
    ne_safe = np.maximum(np.asarray(ne_m3, dtype=float), 1e16)
    n_ions_m3 = np.maximum(np.asarray(ni_m3, dtype=float), 1e15) + inp["AS_3"] * ne_safe
    dte_dr = np.maximum(np.abs(inp["RLTS_1"]) * np.maximum(te_eV, 1.0) / max(mg.a_m, 1e-4), 1.0)
    dti_dr = np.maximum(np.abs(inp["RLTS_2"]) * np.maximum(ti_eV, 1.0) / max(mg.a_m, 1e-4), 1.0)
    chi_e_turb = Qe_miller_W_m2 / (ne_safe * ELEMENTARY_CHARGE * dte_dr)
    chi_i_turb = Qi_miller_W_m2 / (n_ions_m3 * ELEMENTARY_CHARGE * dti_dr)

    return TGLFNNStiffnessProfile(
        psi_N=np.asarray(psi_N, dtype=float),
        rho_tor_norm=mg.rho_tor_norm.copy(),
        model_name=comm.name,
        Qe_gB=Qe_gB,
        Qi_gB=Qi_gB,
        Ge_gB=Ge_gB,
        Pi_gB=Pi_gB,
        Qe_gB_std=s0[:, idx_qe],
        Qi_gB_std=s0[:, idx_qi],
        Qe_W_m2=Qe_W_m2,
        Qi_W_m2=Qi_W_m2,
        dQe_dRLTe_gB=dQe_dRLTe,
        dQi_dRLTi_gB=dQi_dRLTi,
        dGe_dRLne_gB=dGe_dRLne,
        rlte_base=rlte_base,
        rlti_base=rlti_base,
        rlne_base=rlne_base,
        rlte_crit=rlte_crit,
        rlti_crit=rlti_crit,
        chi_e_turb_m2_s=np.clip(chi_e_turb, 1e-3, 100.0),
        chi_i_turb_m2_s=np.clip(chi_i_turb, 1e-3, 100.0),
        Ge_gB_std=s0[:, idx_ge],
    )


def solve_tglfnn_flux_excursions(
    miller,
    psi_N: np.ndarray,
    ne_m3: np.ndarray,
    te_eV: np.ndarray,
    ni_m3: np.ndarray,
    ti_eV: np.ndarray,
    zeff: np.ndarray,
    Qe_pb_gB: np.ndarray,
    Qi_pb_gB: np.ndarray,
    *,
    flux_uncertainty_rel: float = 0.30,
    Z_imp: float = 6.0,
    A_main: float = 2.0,
    z_fast_m3: Optional[np.ndarray] = None,
    omega_tor_rad_s: Optional[np.ndarray] = None,
    p_thermal_Pa: Optional[np.ndarray] = None,
    model_spec: str = DEFAULT_TGLFNN_MODEL,
    rho_flux_tube_bounds: Tuple[float, float] = (0.18, 0.85),
) -> Dict[str, np.ndarray]:
    """Invert 1-D nonlinear ``TGLF-NN`` flux curves for finite excursions ``Delta(R0/L_s)``.

    At each radial node inside the valid flux-tube domain ``rho_flux_tube_bounds``,
    finds the power-balance operating point ``q_NN(kappa_PB) == Q_pb_gB`` along a
    strictly positive, sign-flip-free thermal gradient ray and inverts the finite
    excursion ``Delta_kappa`` satisfying

        |q_NN(kappa_PB +- Delta_kappa) - q_NN(kappa_PB)| == flux_uncertainty_rel * Q_pb_gB + sigma_NN,

    where ``sigma_NN`` is the exact 20-member committee epistemic standard deviation.
    """
    from ..geometry.miller import build_tglf_inputs

    mg = miller.interpolate(psi_N)
    psi = np.asarray(psi_N, dtype=float)
    inp = build_tglf_inputs(
        mg, psi, ne_m3, te_eV, ni_m3, ti_eV, zeff,
        Z_imp=Z_imp, A_main=A_main, z_fast_m3=z_fast_m3,
        omega_tor_rad_s=omega_tor_rad_s, p_thermal_Pa=p_thermal_Pa,
    )
    comm = load_tglfnn_committee(model_spec)
    X0 = comm.pack_features(inp)
    m_pts = X0.shape[0]
    R_over_a = float(mg.R0_m / max(mg.a_m, 1e-4))
    f_rel = max(float(flux_uncertainty_rel), 0.0)

    idx_rlte = comm.xnames.index("RLTS_1")
    idx_rlti = comm.xnames.index("RLTS_2")
    idx_rlts3 = comm.xnames.index("RLTS_3") if "RLTS_3" in comm.xnames else None
    idx_rlne = comm.xnames.index("RLNS_1")
    idx_rlni = comm.xnames.index("RLNS_2")
    idx_rlns3 = comm.xnames.index("RLNS_3") if "RLNS_3" in comm.xnames else None
    idx_qe = comm.ynames.index("OUT_Q_elec")
    idx_qi = comm.ynames.index("OUT_Q_ions")
    idx_ge = comm.ynames.index("OUT_G_elec")

    scan_aL = np.linspace(0.0, 10.0, 65)
    n_s = scan_aL.size
    rho_lo, rho_hi = float(rho_flux_tube_bounds[0]), float(rho_flux_tube_bounds[1])
    in_domain = (mg.rho_tor_norm >= rho_lo) & (mg.rho_tor_norm <= rho_hi)

    ti_cols = [idx_rlti] + ([idx_rlts3] if idx_rlts3 is not None else [])
    tot_cols = [idx_rlte, idx_rlti] + ([idx_rlts3] if idx_rlts3 is not None else [])
    ne_cols = [idx_rlne, idx_rlni] + ([idx_rlns3] if idx_rlns3 is not None else [])

    # 1. Locate the joint in-distribution power-balance operating state X_pb along
    #    positive driving coordinates (RLTS_1, r_loc * RLTS_1) where (Qe + Qi)_NN == Qe_pb + Qi_pb.
    #    Using r_loc = clip(max(RLTS_s, 0.2) / max(RLTS_1, 0.2), 0.5, 2.0) and clamping aL_pb
    #    continuously to scan_aL[0] / scan_aL[-1] guarantees strict C^0 continuity everywhere
    #    (including hollow Te/Ti cores and steep-TEM / high-flux scan boundaries).
    q_tot_pb = np.maximum(np.asarray(Qe_pb_gB, dtype=float) + np.asarray(Qi_pb_gB, dtype=float), 0.0)
    den_pos = np.maximum(X0[:, idx_rlte], 0.2)
    rat_cols: Dict[int, np.ndarray] = {}
    batch_tot = np.repeat(X0, n_s, axis=0)
    batch_tot[:, idx_rlte] = np.tile(scan_aL, m_pts)
    for col in tot_cols[1:]:
        rat = np.clip(np.maximum(X0[:, col], 0.2) / den_pos, 0.5, 2.0)
        rat_cols[col] = rat
        batch_tot[:, col] = np.tile(scan_aL, m_pts) * np.repeat(rat, n_s)

    ym_tot, _ = comm.predict_matrix(batch_tot)
    qtot_c = (ym_tot[:, idx_qe] + ym_tot[:, idx_qi]).reshape(m_pts, n_s)
    # Right-monotone lower envelope q_mono[k] = min_{j >= k} q[j] selects the
    # supercritical thermal-driven branch above any low-gradient TEM trough or
    # zero-flux subcritical plateau, clamping continuously to the onset knee
    # on the left and scan_aL[-1] on the right with zero branch cliffs.
    qtot_mono = np.minimum.accumulate(qtot_c[:, ::-1], axis=1)[:, ::-1]

    X_pb = X0.copy()
    for i in range(m_pts):
        q_row = qtot_mono[i]
        eff_q = max(float(q_tot_pb[i]), float(q_row[0]))
        k_lo = int(np.where(q_row <= eff_q)[0][-1])
        if k_lo >= n_s - 1:
            aL_pb = float(scan_aL[-1])
        else:
            w0 = np.clip(
                (eff_q - float(q_row[k_lo]))
                / max(float(q_row[k_lo + 1] - q_row[k_lo]), 1e-12),
                0.0,
                1.0,
            )
            aL_pb = float(scan_aL[k_lo] + w0 * (scan_aL[k_lo + 1] - scan_aL[k_lo]))
        X_pb[i, idx_rlte] = aL_pb
        for col in tot_cols[1:]:
            X_pb[i, col] = aL_pb * float(rat_cols[col][i])

    # 2. Nonlinear particle-flux scan around X_pb co-shifting (RLNS_1, RLNS_2, RLNS_3)
    #    so TGLF quasineutrality sum_s Z_s AS_s RLNS_s = 0 is preserved identically.
    deltas_n = np.linspace(-4.0, 4.0, 65)
    i0_n = int(np.argmin(np.abs(deltas_n)))
    b_ne = np.repeat(X_pb, deltas_n.size, axis=0)
    for col in ne_cols:
        b_ne[:, col] = np.repeat(X_pb[:, col], deltas_n.size) + np.tile(deltas_n, m_pts)
    ym_n, ys_n = comm.predict_matrix(b_ne)
    g_curve = ym_n[:, idx_ge].reshape(m_pts, deltas_n.size)
    gs_curve = ys_n[:, idx_ge].reshape(m_pts, deltas_n.size)

    dk_ne = np.full(m_pts, np.nan, dtype=float)
    max_dn = float(deltas_n[-1])
    for i in range(m_pts):
        g0 = float(g_curve[i, i0_n])
        # Smooth pinch-balance particle flux scale sqrt(g0^2 + (0.1 * Q_tot_pb)^2)
        # keeps flux_uncertainty_rel active at Gamma_e == 0 while vanishing at f_rel == 0.
        g_scale = float(np.hypot(g0, 0.1 * float(q_tot_pb[i])))
        dg_tgt = max(f_rel * g_scale + float(gs_curve[i, i0_n]), 0.0)
        dq_u = np.maximum.accumulate(np.abs(g_curve[i, i0_n:] - g0))
        k_u = int(np.where(dq_u <= dg_tgt)[0][-1])
        if k_u >= dq_u.size - 1:
            da_u = max_dn
        else:
            w = np.clip((dg_tgt - float(dq_u[k_u])) / max(float(dq_u[k_u + 1] - dq_u[k_u]), 1e-12), 0.0, 1.0)
            da_u = max(float(deltas_n[i0_n + k_u] + w * (deltas_n[i0_n + k_u + 1] - deltas_n[i0_n + k_u])), 1e-6)

        dq_d = np.maximum.accumulate(np.abs(g_curve[i, : i0_n + 1][::-1] - g0))
        dg_grid = -deltas_n[: i0_n + 1][::-1]
        k_d = int(np.where(dq_d <= dg_tgt)[0][-1])
        if k_d >= dq_d.size - 1:
            da_d = max_dn
        else:
            w = np.clip((dg_tgt - float(dq_d[k_d])) / max(float(dq_d[k_d + 1] - dq_d[k_d]), 1e-12), 0.0, 1.0)
            da_d = max(float(dg_grid[k_d] + w * (dg_grid[k_d + 1] - dg_grid[k_d])), 1e-6)

        dk_ne[i] = (2.0 / (1.0 / da_u + 1.0 / da_d)) * R_over_a

    valid_n = np.where(np.isfinite(dk_ne) & in_domain)[0]
    if valid_n.size > 0:
        dk_ne = np.interp(psi, psi[valid_n], dk_ne[valid_n])
    else:
        dk_ne = np.ones(m_pts)

    # 3. Invert nonlinear upward turbulent-wall excursion around X_pb using exact
    #    20-member ensemble standard deviation std_k(sum_j y_k,j) across summed channels.
    deltas_t = np.linspace(0.0, 6.0, 49)
    ddof_ens = 1 if comm.n_models > 1 else 0

    def _invert_from_xpb(cols: List[int], y_indices: List[int], q_pb_scale: np.ndarray) -> np.ndarray:
        b = np.repeat(X_pb, deltas_t.size, axis=0)
        for col in cols:
            if col is not None:
                b[:, col] = np.repeat(X_pb[:, col], deltas_t.size) + np.tile(deltas_t, m_pts)
        preds_gb = comm.predict_ensemble(b)
        tot_k = sum(preds_gb[:, :, yi] for yi in y_indices)
        qc = np.mean(tot_k, axis=0).reshape(m_pts, deltas_t.size)
        sc = np.std(tot_k, axis=0, ddof=ddof_ens).reshape(m_pts, deltas_t.size)
        dk = np.full(m_pts, np.nan, dtype=float)
        max_dt = float(deltas_t[-1])
        for i in range(m_pts):
            q_ref = float(qc[i, 0])
            d_flux = max(f_rel * max(float(q_pb_scale[i]), 0.0) + float(sc[i, 0]), 0.0)
            dq_u = np.maximum.accumulate(np.maximum(qc[i, :] - q_ref, 0.0))
            k_lo = int(np.where(dq_u <= d_flux)[0][-1])
            if k_lo >= deltas_t.size - 1:
                da_up = max_dt
            else:
                wu = np.clip((d_flux - float(dq_u[k_lo])) / max(float(dq_u[k_lo + 1] - dq_u[k_lo]), 1e-12), 0.0, 1.0)
                da_up = max(float(deltas_t[k_lo] + wu * (deltas_t[k_lo + 1] - deltas_t[k_lo])), 1e-6)
            dk[i] = da_up * R_over_a

        valid = np.where(np.isfinite(dk) & in_domain)[0]
        if valid.size > 0:
            return np.interp(psi, psi[valid], dk[valid])
        valid_any = np.where(np.isfinite(dk))[0]
        return np.interp(psi, psi[valid_any], dk[valid_any]) if valid_any.size > 0 else np.ones(m_pts)

    dk_te_free = _invert_from_xpb([idx_rlte], [idx_qe], np.asarray(Qe_pb_gB, dtype=float))
    dk_ti_free = _invert_from_xpb(ti_cols, [idx_qi], np.asarray(Qi_pb_gB, dtype=float))
    dk_t_coupled = _invert_from_xpb(tot_cols, [idx_qe, idx_qi], q_tot_pb)

    return {
        "dkappa_te_free": dk_te_free,
        "dkappa_ti_free": dk_ti_free,
        "dkappa_t_coupled": dk_t_coupled,
        "dkappa_ne": dk_ne,
    }
