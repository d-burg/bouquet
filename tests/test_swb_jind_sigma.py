"""The swb inductive redraw's sigma is the jphi envelope's relative width on the
inductive current, not its absolute width (which is the total's)."""
import numpy as np

from bouquet.swb import swb_jind_sigma


def test_the_relative_width_is_carried_onto_the_inductive_seed():
    x = np.linspace(0.0, 1.0, 101)
    jt = 1.7e6 * (1 - x ** 2) + 9e5 * np.exp(-((x - 0.92) / 0.03) ** 2)   # bootstrap pedestal peak
    seed = 1.6e6 * (1 - x ** 2) ** 1.5 + 1e3                              # the inductive part
    sig = swb_jind_sigma(0.1 * np.abs(jt), jt, seed)
    np.testing.assert_allclose(sig, 0.1 * seed)
    ped = np.argmin(abs(x - 0.92))
    assert 0.1 * jt[ped] > seed[ped] > sig[ped]          # the absolute width would exceed the seed there
    np.testing.assert_array_equal(swb_jind_sigma(np.zeros_like(x), jt, seed), 0.0)   # sigma=0 check
    assert swb_jind_sigma([1.0], [0.0], [5.0])[0] == 0.0

