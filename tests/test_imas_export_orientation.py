"""The IMAS export is self-consistent in the SOURCE's orientation.

bouquet's archive is in its positive-Ip frame; the export template is the
source dd, in the source's own frame, and the writer keeps the template fields
it does not overwrite (core_sources beam current, pf_active, b0, ...).  Every
quantity it does overwrite must therefore be taken back to the source frame,
or a re-read of the export mixes frames (e.g. a positive ip with the
template's negative beam current: j_inductive off by 2|j_NBI|).

Checked on the synthetic D3D-like example and its four orientation mirrors
(tests/_mirror_dd.py): the export of a mirrored source IS the mirror of the
normal export, field for field, and re-reads to a bit-identical Baseline.
"""
import json
import os
import warnings

import h5py
import numpy as np
import pytest

import _mirror_dd as mdd
from bouquet.io.imas import (ORIENTATION_ORIGIN_AUTO,
                             ORIENTATION_ORIGIN_OVERRIDE, write_imas_draw)
from bouquet.utils import stamp_source_orientation
from test_reversed_ip_current_sign import (_assert_same_baseline, _example,
                                           _read, _write)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEQ = os.path.join(_HERE, "data", "d3dlike.geqdsk")   # TokaMaker, CURRENT > 0

pytestmark = pytest.mark.skipif(not os.path.isfile(_GEQ),
                                reason="d3dlike.geqdsk absent")


def _archive(path, bl, stamp=None):
    """A one-draw archive in the positive frame, the draw = the baseline read
    of the (un-mirrored) example; optionally stamped as generate() would."""
    with h5py.File(path, "w") as hf:
        hf.require_group("scan/0/_baseline")
        g = hf.require_group("scan/0/0")
        g.create_dataset("psi_N", data=np.asarray(bl.psi_N))
        g.create_dataset("psi_N_kinetic", data=np.asarray(bl.psi_N_kinetic))
        for k, v in (("n_e", bl.ne), ("T_e", bl.te), ("n_i", bl.ni),
                     ("T_i", bl.ti), ("j_phi", bl.j_phi),
                     ("j_inductive", bl.j_inductive), ("j_BS", bl.j_BS)):
            g.create_dataset(k, data=np.asarray(v, float))
        with open(_GEQ, "rb") as fh:
            g.create_dataset("eqdsk", data=np.void(fh.read()))
    if stamp is not None:
        stamp_source_orientation(path, scan_key=0, **stamp)
    return path


def _export(tmp_path, tag, template_dd, bl, stamp):
    tmpl = _write(tmp_path, template_dd, f"tmpl_{tag}.json")
    arc = _archive(str(tmp_path / f"run_{tag}.h5"), bl, stamp)
    out = str(tmp_path / f"draw_{tag}.json")
    write_imas_draw(arc, 0, tmpl, out, scan_key=0, time=mdd.EXAMPLE_TIME)
    with open(out) as fh:
        return out, json.load(fh)


def _stamp(s_ip, s_b0):
    # the example is Ip > 0, B0 < 0
    return dict(current_sign=s_ip, b0_sign=-s_b0,
                current_sign_origin=ORIENTATION_ORIGIN_AUTO)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    tp = tmp_path_factory.mktemp("base")
    dd = _example()
    bl = _read(_write(tp, dd, "ref.json"))
    return dd, bl


class TestExportIsInTheSourceFrame:
    @pytest.mark.parametrize("orient", mdd.ORIENTATIONS,
                             ids=[mdd.tag(*o) for o in mdd.ORIENTATIONS])
    def test_mirrored_export_is_the_mirror_of_the_normal_export(
            self, tmp_path, base, orient):
        dd, bl = base
        _, ref = _export(tmp_path, "n", dd, bl, _stamp(1.0, 1.0))
        _, got = _export(tmp_path, "m", mdd.mirror_dd(dd, *orient), bl,
                         _stamp(*orient))
        assert got == mdd.mirror_dd(ref, *orient)

    @pytest.mark.parametrize("orient", mdd.ORIENTATIONS,
                             ids=[mdd.tag(*o) for o in mdd.ORIENTATIONS])
    def test_reread_round_trips(self, tmp_path, base, orient):
        """Re-reading the export of a mirrored source gives the SAME Baseline
        as re-reading the normal export (the written equilibrium carries no
        j_tor, so the re-read runs without the jphi anchor)."""
        dd, bl = base
        p_ref, _ = _export(tmp_path, "n", dd, bl, _stamp(1.0, 1.0))
        p_got, got = _export(tmp_path, "m", mdd.mirror_dd(dd, *orient), bl,
                             _stamp(*orient))
        a = _read(p_ref, anchor_jtor_to_equilibrium=False)
        b = _read(p_got, anchor_jtor_to_equilibrium=False)
        _assert_same_baseline(a, b, mdd.tag(*orient))
        assert b.source_current_sign == orient[0]
        # one frame inside the file: ip, the written currents and the kept
        # template beam current all carry the source's Ip sign
        s = orient[0]
        ts = got["equilibrium"]["time_slice"][2]
        assert np.sign(ts["global_quantities"]["ip"]) == s
        cp = got["core_profiles"]["profiles_1d"][2]
        assert np.sign(np.sum(cp["j_tor"])) == s
        assert np.sign(np.sum(cp["j_total"])) == s
        nbi = got["core_sources"]["source"][0]["profiles_1d"][2]["j_parallel"]
        assert np.sign(np.sum(nbi)) == s
        # and F agrees with the kept b0
        b0 = got["equilibrium"]["vacuum_toroidal_field"]["b0"][2]
        assert np.all(np.sign(ts["profiles_1d"]["f"]) == np.sign(b0))

    def test_positive_ip_values_are_the_archive_values(self, tmp_path, base):
        """ip > 0: currents, psi, P', FF', ip are the archive's (positive-frame)
        values unchanged; q keeps the template's own (positive) q sign; only f
        takes the template b0's sign."""
        from bouquet.io.geqdsk import read_geqdsk

        dd, bl = base
        _, out = _export(tmp_path, "n", dd, bl, _stamp(1.0, 1.0))
        g = read_geqdsk(_GEQ)
        p1 = out["equilibrium"]["time_slice"][2]["profiles_1d"]
        assert p1["dpressure_dpsi"] == np.asarray(g.pprime, float).tolist()
        assert p1["f_df_dpsi"] == np.asarray(g.ffprim, float).tolist()
        assert p1["q"] == np.asarray(g.qpsi, float).tolist()
        assert p1["f"] == (-np.asarray(g.fpol, float)).tolist()     # b0 < 0
        gq = out["equilibrium"]["time_slice"][2]["global_quantities"]
        assert gq["ip"] == float(g.Ip) > 0.0

    def test_unstamped_archive_restores_the_template_orientation_and_warns(
            self, tmp_path, base):
        dd, bl = base
        _, ref = _export(tmp_path, "n", dd, bl, None)
        m = mdd.mirror_dd(dd, -1.0, 1.0)
        with pytest.warns(UserWarning, match="no current-orientation stamp"):
            _, got = _export(tmp_path, "m", m, bl, None)
        assert got == mdd.mirror_dd(ref, -1.0, 1.0)

    def test_a_template_that_is_not_the_source_is_refused(self, tmp_path, base):
        dd, bl = base
        with pytest.raises(ValueError, match="not the source dd"):
            _export(tmp_path, "x", mdd.mirror_dd(dd, -1.0, 1.0), bl,
                    _stamp(1.0, 1.0))
        with pytest.raises(ValueError, match="source_b0_sign"):
            _export(tmp_path, "y", mdd.mirror_dd(dd, 1.0, -1.0), bl,
                    _stamp(1.0, 1.0))

    def test_an_override_stamp_is_honoured(self, tmp_path, base):
        """With ImasSource.current_orientation set, the archive's factor need
        not equal sign(template ip); the stamp's factor is what is restored."""
        dd, bl = base
        st = dict(current_sign=1.0, b0_sign=-1.0,
                  current_sign_origin=ORIENTATION_ORIGIN_OVERRIDE)
        m = json.loads(json.dumps(dd))
        for ts in m["equilibrium"]["time_slice"]:
            ts["global_quantities"]["ip"] = -ts["global_quantities"]["ip"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, got = _export(tmp_path, "o", m, bl, st)
        cp = got["core_profiles"]["profiles_1d"][2]
        assert np.sign(np.sum(cp["j_tor"])) == 1.0
        assert got["equilibrium"]["time_slice"][2]["global_quantities"]["ip"] > 0
