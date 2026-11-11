"""Spitzer conductivity against an independent number (added after an independent review: the earlier
"hand values" re-typed the code's own constants, so a wrong shared constant would have passed).

NRL Plasma Formulary (2019), "Transport coefficients": eta_perp = 1.03e-4 Z lnL T^-3/2 Ohm m (T in eV) and, for
Z = 1, eta_par = eta_perp / 1.96 (recalled).  At Z = 1 the module's Sauter 18a form gives
1.9012e4 / N(1) with N(1) = 0.58 + 0.74/1.76 = 1.0005, i.e. eta_par = 5.262e-5 lnL T^-3/2 against NRL's 5.255e-5.
Bound 2 %: NRL's three-digit constants and the 1.96 factor (vs the N(Z) fit) carry ~0.5 % each; a wrong shared
constant would show up as a larger offset.
"""
import numpy as np
import pytest

from bouquet.kinrecon.current.conductivity import spitzer_conductivity


@pytest.mark.parametrize("te", [100.0, 1000.0, 5000.0])
def test_spitzer_against_nrl_formulary(te):
    ln_l = 17.0
    sigma = spitzer_conductivity(1e19, te, 1.0, ln_lambda=ln_l)
    eta_nrl = 1.03e-4 * ln_l * te ** -1.5 / 1.96
    assert sigma * eta_nrl == pytest.approx(1.0, abs=0.02)
