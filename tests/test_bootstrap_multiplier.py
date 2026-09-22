"""The draws apply the baseline's bootstrap multiplier after SWB, as it did.

In ohmic mode the baseline is ``bl.j_BS = m * SWB(scale 1)``, with ``m`` the
closure's ``bs_scale`` or the structured ``s_bs(psi)``.  The draws (and the
sigma=0 check) used to pass ``m`` INTO SWB as ``scale_jBS`` instead; OFT
applies that inside SWB's self-consistent iteration, so the sigma=0 draw
missed ``bl.j_BS`` by 8.9 % of peak at the pedestal at ``bs_scale`` 0.77 (shot
174956 t = 2.0 s), and the structured profile never reached the draws at all.
"""
import inspect
import types

import numpy as np

import bouquet.TokaMaker_interface as tmi
from bouquet.run import Bouquet


def _mult(bs=1.0, prof=None, n=5):
    bl = types.SimpleNamespace(bs_scale=bs, bs_scale_profile=prof,
                               psi_N=np.linspace(0, 1, n))
    return Bouquet._bootstrap_multiplier(types.SimpleNamespace(baseline=bl))


class TestMultiplier:
    def test_unit_scale_is_none(self):
        """diff mode and the geqdsk path: nothing changes."""
        assert _mult(1.0) is None

    def test_scalar_scale_is_a_flat_profile(self):
        np.testing.assert_array_equal(_mult(0.77), np.full(5, 0.77))

    def test_the_structured_profile_wins(self):
        prof = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        np.testing.assert_array_equal(_mult(0.95, prof), prof)


class TestWiring:
    def test_every_spike_path_applies_it_after_swb(self):
        src = inspect.getsource(tmi.perturb_kinetic_equilibrium)
        assert "spike_profile = _bs_mult * smooth_jbs_transition(" in src
        assert "spike_profile = _delta_bl + _bs_mult * (_spike_raw - _delta_ref)" in src
        assert "delta_spike = _bs_mult * (_spike_perturbed - spike_profile_recon_cached)" in src

    def test_generate_passes_the_multiplier_and_the_bare_jitter(self):
        src = inspect.getsource(Bouquet.generate)
        assert "jBS_scale_profile=_bs_mult," in src
        assert "gc.jBS_scale_range[0] * _bs" not in src
        assert "jBS_scale_profile=jBS_scale_profile," in inspect.getsource(tmi.generate_bouquet)

    def test_the_sigma0_check_mirrors_the_draw(self):
        src = inspect.getsource(Bouquet.verify_sigma0_consistency)
        assert "scale_jBS=1.0," in src
        assert "_mult = self._bootstrap_multiplier()" in src

    def test_every_closure_site_records_the_profile(self):
        src = inspect.getsource(Bouquet)
        assert src.count("bl.bs_scale_profile = ") == 3
