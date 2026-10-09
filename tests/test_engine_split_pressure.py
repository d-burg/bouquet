"""The engine's archived split keeps the pressure-driven current
p'(<R> - F^2<1/R>/<B^2>) as its own bucket, j_pressure (owner decision D2,
2026-10-09; PR #64 had put it on j_BS).  Solver-free."""
from types import SimpleNamespace

import numpy as np

from bouquet.engine import _split, composed_factor, pressure_term


def test_the_split_keeps_pG_as_its_own_bucket():
    n = 21
    psi = np.linspace(0.0, 1.0, n)
    g = dict(psi_N=psi, F=np.full(n, 3.4), inv_R=np.full(n, 0.6),
             B2=np.full(n, 4.0), R_avg=np.full(n, 1.7),
             pprime=-1e4 * (1.0 - psi))
    st = SimpleNamespace(geom=g, request=np.linspace(2e6, 1e5, n),
                         lambda_bs=np.full(n, 0.3))
    c = SimpleNamespace(jB_fix_parts=dict(nbi=np.full(n, 0.1),
                                          rf=np.full(n, 0.05)))
    eng = SimpleNamespace(c=c,
                          delivered_closure=dict(out=dict(s_bs=1.2)))
    R, j_ind, j_bs, j_nbi, j_rf, j_p = _split(eng, dict(state=st))
    pg = pressure_term(g)
    assert np.max(np.abs(pg)) > 0.0
    np.testing.assert_array_equal(j_p, pg)
    # j_BS is the field-aligned bootstrap alone
    np.testing.assert_allclose(
        j_bs, 1.2 * composed_factor(g) * st.lambda_bs, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(j_ind, R - j_bs - j_nbi - j_rf - j_p)
