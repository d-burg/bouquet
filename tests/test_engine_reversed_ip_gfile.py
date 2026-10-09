"""The unified engine on a reversed-Ip / reversed-B_t g-file -- fast half.

The synthetic example g-file (COCOS 1, Ip > 0, B_t < 0) is mirrored into all
four (Ip, B_t) orientations in the same COCOS -- an Ip reversal flips every
Ip-odd raw field (CURRENT, SIMAG, SIBRY, PSIRZ, PPRIME, FFPRIM, QPSI), a B_t
reversal BCENTR, FPOL and QPSI -- and written as real g-files.  Through the
g-file adapter and the engine (on the toy Grad-Shafranov stand-in, on the
g-file's own grid):

* every orientation gives the SAME contract in the positive frame (the
  B_t mirror bit for bit; the Ip mirror to the reader's contour-tracing
  level, ~5e-6 relative: it traces the negated psi_RZ);
* the orientation stamps say what the file is: ``current_sign``,
  ``b0_sign``, the (R, phi, Z) ``ip_sign`` / ``bt_sign`` -- and an MSE block
  that states no orientation is completed with exactly those;
* the engine's composition and the delivered request are the same for all
  four, positive, and the engine record carries the source's stamps.

The live-solver counterpart is ``test_engine_reversed_ip_gfile_solver.py``.
Synthetic inputs only; no solver.
"""
import contextlib
import copy
import io
import os

import numpy as np
import pytest

import _engine_toy as T
from bouquet.adapters import GFileAdapter, gfile_parallel_current, mse_rows
from bouquet.engine import compose, reconstruct

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.join(_HERE, os.pardir, "examples", "D3D-like")
_GEQ = os.path.join(_EX, "D3Dlike_Hmode_baseline.geqdsk")
_PF = os.path.join(_EX, "D3Dlike_Hmode_baseline.peqdsk")
_MESH = os.path.join(_EX, "DIIID_mesh.h5")

ORIENTATIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]   # (Ip, B_t) factors
_IDS = ["normal", "Bt-reversed", "Ip-reversed", "both-reversed"]
#: the reader traces the negated psi_RZ to surfaces that differ at this
#: relative level (a property of the contour tracing, measured 3.4e-6 on
#: <j.B> and 5.0e-6 on |<j_phi>|); every Ip-even quantity is exact
_TRACE = 1e-5


def mirror_raw(raw, ip, bt):
    """The raw g-file dict of the (Ip, B_t) = (ip, bt) mirror, same COCOS."""
    raw = copy.deepcopy(raw)
    if ip < 0:
        for k in ("CURRENT", "SIMAG", "SIBRY"):
            raw[k] = -raw[k]
        for k in ("PSIRZ", "PPRIME", "FFPRIM", "QPSI"):
            raw[k] = -np.asarray(raw[k])
    if bt < 0:
        raw["BCENTR"] = -raw["BCENTR"]
        for k in ("FPOL", "QPSI"):
            raw[k] = -np.asarray(raw[k])
    return raw


def write_mirrors(outdir, geqdsk=_GEQ):
    """Write the four mirrors of *geqdsk*; ``{(ip, bt): path}``."""
    from bouquet.io.geqdsk import _write_geqdsk, read_geqdsk
    ref = read_geqdsk(geqdsk)
    out = {}
    for ip, bt in ORIENTATIONS:
        p = os.path.join(str(outdir), f"mirror_ip{ip:+d}_bt{bt:+d}.geqdsk")
        _write_geqdsk(mirror_raw(ref._raw, ip, bt), p)
        out[(ip, bt)] = p
    return out


def _cfg(path, **gen):
    from bouquet.config import (BouquetConfig, GenerationConfig,
                                ReconstructionSource, SolverConfig)
    return BouquetConfig(
        source=ReconstructionSource(geqdsk_path=path, profiles_path=_PF),
        solver=SolverConfig(mesh_path=_MESH), output_header="t",
        generation=GenerationConfig(reconstruction_engine="unified", **gen))


@pytest.fixture(scope="module")
def mirrors(tmp_path_factory):
    paths = write_mirrors(tmp_path_factory.mktemp("mirrors"))
    out = {}
    for o, p in paths.items():
        ad = GFileAdapter(_cfg(p).source, _cfg(p))
        out[o] = (ad, ad.read())
    return out


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) or 1.0))


@pytest.mark.parametrize("o", ORIENTATIONS, ids=_IDS)
def test_the_mirror_is_what_it_says(mirrors, o):
    ad, c = mirrors[o]
    ip, bt = o
    ref_ad, _ = mirrors[(1, 1)]
    assert np.sign(ad.eqdsk.Ip) == ip * np.sign(ref_ad.eqdsk.Ip)
    assert np.sign(ad.eqdsk.B_center) == bt * np.sign(ref_ad.eqdsk.B_center)


@pytest.mark.parametrize("o", ORIENTATIONS, ids=_IDS)
def test_every_orientation_reads_the_same_contract(mirrors, o):
    ad, c = mirrors[o]
    ref_ad, ref = mirrors[(1, 1)]
    tol = 0.0 if o[0] > 0 else _TRACE
    jB, parts = gfile_parallel_current(ad.eqdsk)
    jB0, parts0 = gfile_parallel_current(ref_ad.eqdsk)
    assert np.all(jB[c.psi_N < 0.95] > 0.0)       # positive frame
    assert _rel(jB, jB0) <= tol
    assert _rel(parts["jphi_in"], parts0["jphi_in"]) <= tol
    for k in ("F", "pprime", "ffprime"):
        assert _rel(parts[k], parts0[k]) <= tol, k
    assert parts["frame_identity_max_rel"] < 1e-9
    # Ip-even inputs are exact in every orientation
    np.testing.assert_array_equal(c.pressure, ref.pressure)
    np.testing.assert_array_equal(c.boundary, ref.boundary)
    assert c.Ip == ref.Ip > 0.0
    assert c.rows["q0"]["target"] == ref.rows["q0"]["target"] > 0.0
    assert abs(c.rows["l_i"]["target"] - ref.rows["l_i"]["target"]) \
        <= tol * ref.rows["l_i"]["target"]
    assert np.all(c.anchor_request >= 0.0)
    assert _rel(c.anchor_request, ref.anchor_request) <= tol
    # the composition on the file's own surfaces is its own |<j_phi>|
    g = dict(F=parts["F"], R_avg=parts["R_avg"], inv_R=parts["inv_R"],
             B2=parts["B2"], pprime=parts["pprime"])
    J, _ = compose(g, 0.7 * jB, 0.3 * jB, 0.0 * jB)
    assert _rel(J, parts["jphi_in"]) < 1e-9


@pytest.mark.parametrize("o", ORIENTATIONS, ids=_IDS)
def test_the_orientation_stamps_and_the_mse_block(mirrors, o):
    ad, c = mirrors[o]
    ip, bt = o
    _ref_ad, ref = mirrors[(1, 1)]
    s, s0 = c.signs, ref.signs
    assert s["current_sign"] == ip * s0["current_sign"]
    assert s["b0_sign"] == bt * s0["b0_sign"]
    assert s["ip_sign_RphiZ"] == ip * s0["ip_sign_RphiZ"]
    assert s["bt_sign_RphiZ"] == bt * s0["bt_sign_RphiZ"]
    assert ad.orientation["ip_sign"] == s["ip_sign_RphiZ"]
    # an MSE block stating no orientation is completed with the FILE's
    n = 6
    md = dict(R=np.linspace(1.7, 2.2, n), Z=np.zeros(n),
              tgamma=np.linspace(0.02, 0.1, n), sigma=np.full(n, 0.002),
              weight=np.ones(n), A1=np.ones(n), A2=np.ones(n),
              A3=np.zeros(n), A4=np.zeros(n), er_corrected=True)
    from bouquet.config import GenerationConfig
    r = mse_rows(GenerationConfig(reconstruction_engine="unified",
                                  engine_rows=["Ip", "l_i", "mse"],
                                  mse_data=md), ad.orientation)
    assert r["chords"]["ip_sign"] == s["ip_sign_RphiZ"]
    assert r["chords"]["bt_sign"] == s["bt_sign_RphiZ"]


@pytest.fixture(scope="module")
def engine_runs(mirrors):
    """The engine on the toy stand-in over each mirror's own grid.  (The
    toy is calibrated on its own grid; on the g-file grid its loop needs more
    passes than the package ceiling: 20 is this stand-in run's setting.)"""
    out = {}
    for o, (ad, c) in mirrors.items():
        ad2 = GFileAdapter(ad.source, ad.config)
        c2 = ad2.read()
        b = T.ToyGS(psi=c2.psi_N, Ip=c2.Ip)
        with contextlib.redirect_stdout(io.StringIO()):
            eng, res, rec = reconstruct(ad2, b, T.settings(jbs_max_passes=20),
                                        label=f"gfile{o}")
        out[o] = (eng, res, rec)
    return out


@pytest.mark.parametrize("o", ORIENTATIONS, ids=_IDS)
def test_the_engine_delivers_the_same_positive_state(engine_runs, o):
    eng, res, rec = engine_runs[o]
    eng0, res0, rec0 = engine_runs[(1, 1)]
    assert res["converged"] and res0["converged"]
    tol = 0.0 if o[0] > 0 else _TRACE
    R, R0 = res["state"].request, res0["state"].request
    assert np.all(R > 0.0)
    assert _rel(R, R0) <= tol
    assert _rel(eng.c.jB_ind, eng0.c.jB_ind) <= tol
    d, d0 = rec["delivered"]["checks"], rec0["delivered"]["checks"]
    assert abs(d["l_i"]["delivered"] - d0["l_i"]["delivered"]) \
        <= tol * d0["l_i"]["delivered"]
    # the record stamps the source's orientation; the frame is positive
    sg = rec["contract"]["signs"]
    assert sg["current_sign"] == float(o[0]) * rec0["contract"]["signs"][
        "current_sign"]
    assert sg["b0_sign"] == float(o[1]) * rec0["contract"]["signs"]["b0_sign"]
    assert sg["frame"].startswith("positive")
