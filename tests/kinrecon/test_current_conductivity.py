"""Conductivity formulas of ``bouquet.kinrecon.current.conductivity``.

The Redl-2021 and Sauter-1999 fits are checked against the values published
by Redl et al. with their paper: Zenodo dataset doi:10.5281/zenodo.4110606
(CC BY 4.0), file ``Coefficient_arbitrary_collisionality_Zeff_1.8.txt``,
columns ``nue_star_*``, ``ftrap_*``, ``sigma_SA_*/sigma_Spitzer`` (Sauter
1999 fit) and ``sigma_new_*/sigma_Spitzer`` (Redl 2021 fit).  The fit values
are deterministic functions of (f_t, nu_e*, Z), so agreement is expected to
round-off.  (Their ``sigma_neo`` column, the NEO code result itself, is
listed for reference: the fits approximate it to a few percent.)
Synthetic / public numbers only.
"""
import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bouquet.kinrecon.current import conductivity as cnd  # noqa: E402

ZEFF = 1.8
# (nue_star, ftrap, sigma_NEO, sigma_Sauter1999_fit, sigma_Redl2021_fit), all sigma / sigma_Spitzer
REDL_DATASET_ZEFF_1P8 = [
    (0.0018857671203296868, 0.23959988, 0.7966250392955582, 0.7353102834593953, 0.7503422191952418),
    (0.0002612896639919625, 0.4522813, 0.5697487017593741, 0.516098408521499, 0.5410716183591787),
    (8.30533423867525e-05, 0.63549, 0.39486839363835396, 0.33962673363871726, 0.3655672397877398),
    (0.18859538299340886, 0.23959988, 0.8109090427023142, 0.7819864908023462, 0.7882245720161215),
    (0.02613155342541345, 0.4522813, 0.575871370561282, 0.5472006356827072, 0.5593862580255272),
    (0.008306156548945384, 0.63549, 0.39658254223495537, 0.3626757694346245, 0.3767074271099317),
    (1.8859555273062212, 0.23959988, 0.8787127360566016, 0.8600215203404922, 0.8803289169591147),
    (0.2613157694400116, 0.4522813, 0.6249761699184383, 0.6125884558767782, 0.6177009764643823),
    (0.08306164024546553, 0.63549, 0.41933779753741085, 0.41349493196664305, 0.41153865290488284),
    (18.85955697039935, 0.23959988, 0.9722031570031664, 0.9522538317323168, 0.9734137342369222),
    (2.6131579295811442, 0.4522813, 0.7839073442467266, 0.7535535189195156, 0.7953210610050103),
    (0.8306164772091257, 0.63549, 0.5475437820514584, 0.5406564665532009, 0.5547118607661159),
    (188.5955714018557, 0.23959988, 1.0035568226732188, 0.9917879688934745, 0.9967528564756876),
    (26.13157953106522, 0.4522813, 0.9649141869268738, 0.9147948756932874, 0.9579923980117376),
    (8.306164846868853, 0.63549, 0.8524389426238217, 0.7566918833405059, 0.8428216237613243),
    (1885.9557157164184, 0.23959988, 1.006952988977705, 0.9990124184279343, 0.9996600270578488),
    (261.31579554590587, 0.4522813, 1.0072716556829053, 0.9849044037043366, 0.9950375891019196),
    (83.06164854346609, 0.63549, 1.0044044323350783, 0.9321994223852819, 0.9770116272093199),
]


@pytest.mark.parametrize("model,col", [("redl2021", 4), ("sauter1999", 3)])
def test_f33_matches_published_fit_values(model, col):
    data = np.array(REDL_DATASET_ZEFF_1P8)
    nu, ft = data[:, 0], data[:, 1]
    x = cnd.f33_effective_trapped_fraction(ft, nu, ZEFF, model=model)
    ratio = cnd.f33(x, ZEFF, model=model)
    # The dataset prints 16-17 significant digits; f_t is printed to 7-8, which bounds
    # the attainable agreement at ~1e-8 relative.  Bound fixed before running.
    np.testing.assert_allclose(ratio, data[:, col], rtol=1e-7, atol=0)


def test_redl_fit_tracks_neo_within_published_accuracy():
    """Redl 2021 approximates NEO to a few percent; Sauter 1999 is worse at high nu*."""
    data = np.array(REDL_DATASET_ZEFF_1P8)
    x = cnd.f33_effective_trapped_fraction(data[:, 1], data[:, 0], ZEFF)
    redl = cnd.f33(x, ZEFF)
    assert np.max(np.abs(redl / data[:, 2] - 1.0)) < 0.08
    assert np.max(np.abs(data[:, 3] / data[:, 2] - 1.0)) > np.max(np.abs(redl / data[:, 2] - 1.0))


def test_collisionless_and_collisional_limits():
    z = np.array([1.0, 2.0, 4.0])
    # nu* -> 0: X -> f_t; F33(0) = 1 for any Z; nu* -> inf: X -> 0, sigma -> Spitzer
    for model in ("redl2021", "sauter1999"):
        np.testing.assert_allclose(cnd.f33_effective_trapped_fraction(0.4, 0.0, z, model=model), 0.4, rtol=1e-15)
        np.testing.assert_allclose(cnd.f33(0.0, z, model=model), 1.0, rtol=1e-15)
        assert np.all(cnd.f33_effective_trapped_fraction(0.4, 1e12, z, model=model) < 1e-5)
    # Redl F33(X=1) = 1 - 1 - 0.21/Z + 0.54/Z - 0.33/Z = 0 exactly (sigma_neo -> 0 for f_t -> 1)
    np.testing.assert_allclose(cnd.f33(1.0, z), 0.0, atol=1e-15)


def test_spitzer_and_coulomb_log_hand_values():
    ne, te, z = 5.0e19, 2000.0, 2.0
    lnl = 31.3 - np.log(np.sqrt(ne) / te)
    np.testing.assert_allclose(cnd.coulomb_log_e(ne, te), lnl, rtol=1e-15)
    nz = 0.58 + 0.74 / (0.76 + z)
    np.testing.assert_allclose(cnd.spitzer_conductivity(ne, te, z), 1.9012e4 * te ** 1.5 / (z * nz * lnl), rtol=1e-15)
    # NRL e-e form, as written in IMAS.jl lnLambda_e / OFT calculate_ln_lambda('NRL')
    nrl = 23.5 - np.log(np.sqrt(ne / 1e6) * te ** -1.25) - np.sqrt(1e-5 + (np.log(te) - 2) ** 2 / 16)
    np.testing.assert_allclose(cnd.coulomb_log_e(ne, te, model="nrl"), nrl, rtol=1e-15)
    nus = cnd.nu_e_star(2.0, 1.7, ne, te, z, 0.2)
    np.testing.assert_allclose(nus, 6.921e-18 * 2.0 * 1.7 * ne * z * lnl / (te ** 2 * 0.2 ** 1.5), rtol=1e-15)


def test_neoclassical_conductivity_composition():
    ne, te, z, ft, q, R, eps = 4e19, 1500.0, 1.6, 0.45, 1.8, 1.75, 0.25
    sig, parts = cnd.neoclassical_conductivity(ne, te, z, ft, q, R, eps)
    nus = cnd.nu_e_star(q, R, ne, te, z, eps)
    x = cnd.f33_effective_trapped_fraction(ft, nus, z)
    np.testing.assert_allclose(sig, cnd.spitzer_conductivity(ne, te, z) * cnd.f33(x, z), rtol=1e-15)
    np.testing.assert_allclose(parts["nu_e_star"], nus, rtol=1e-15)


def _ft_reference(eps):
    """Sauter 1999 eq. 12 by adaptive quadrature for the surface R = R0(1 + eps cos t), B ~ 1/R,
    flux-surface weight dl/B_p ~ R dt (circular surface, B_p ~ 1/R)."""
    def avg(fn):
        num = quad(lambda t: fn(t) * (1 + eps * np.cos(t)), 0, np.pi, epsabs=0, epsrel=1e-12, limit=400)[0]
        den = quad(lambda t: (1 + eps * np.cos(t)), 0, np.pi, epsabs=0, epsrel=1e-12)[0]
        return num / den
    b = lambda t: 1.0 / (1 + eps * np.cos(t))  # noqa: E731
    bmax = 1.0 / (1 - eps)
    b2 = avg(lambda t: b(t) ** 2)
    inner = quad(lambda lam: lam / avg(lambda t: np.sqrt(max(1 - lam * b(t), 0.0))), 0, 1 / bmax,
                 epsabs=0, epsrel=1e-9, limit=200)[0]
    return 1 - 0.75 * b2 * inner


def test_trapped_fraction_quadrature():
    eps = 0.3
    t = np.linspace(0, 2 * np.pi, 4001)[:-1]
    b = 1.0 / (1 + eps * np.cos(t))
    w = 1 + eps * np.cos(t)
    ft = cnd.trapped_fraction(b, w)
    ref = _ft_reference(eps)
    assert abs(ft / ref - 1) < 1e-4          # documented accuracy at the default nlambda


def test_trapped_fraction_small_eps_limit():
    """Leading-order large-aspect-ratio limit f_t -> 1.46 sqrt(eps) (Hirshman-Sigmar; recalled)."""
    eps = 1e-4
    ref = _ft_reference(eps)
    assert abs(ref / (1.46 * np.sqrt(eps)) - 1) < 0.01
