"""The ion-temperature coverage check of the kinetic inputs (``assemble_kinetic_inputs(..., ti_coverage=...)``), on
synthetic profiles (no solver).

A slice whose ion temperature has no valid measurement point inside the plasma boundary is refused by default, in
the form of the other input refusals (a ``ValueError`` subclass carrying the check's record); a slice covered only
at the edge passes unless the caller applies a core criterion, whose threshold has no default; the explicit
assumption ``T_i = T_e`` (with a stated uncertainty) is applied only to a slice that is not covered, and is then
recorded (assumed, rule, radial range, uncertainty), named in the provenance and warned about; a covered slice's
profiles are untouched bit for bit.
"""
import inspect
import json
import warnings

import numpy as np
import pytest

from bouquet.kinrecon import kinetics as KI

X = np.linspace(0.0, 1.0, 51)                                  # psi_N table of the producing equilibrium
NE = 4e19 * (1.0 - 0.8 * X ** 2) + 1e18
TE = 2e3 * (1.0 - 0.9 * X ** 2) + 50.0
TI = 0.8 * TE
CORE = np.linspace(0.05, 0.85, 9)                              # points across the plasma
EDGE = np.array([0.90, 0.94, 0.98])                            # points in the edge only
OUTSIDE = np.array([1.02, 1.05])                               # points beyond the boundary only


def _pts(psi, sigma=True, **kw):
    v = np.interp(psi, X, TI, right=TI[-1])
    d = dict(value=v, psi_N=np.asarray(psi, float), label="synthetic T_i points", **kw)
    if sigma:
        d["sigma"] = 0.1 * v
    return d


def _build(ti_coverage, **kw):
    return KI.assemble_kinetic_inputs(X, NE, TE, TI, zeff_candidates={"vb": ("zeff", np.full_like(X, 1.8))},
                                      zeff_source="vb", z_imp=6.0, n_fast="none", coordinate="psi_n",
                                      ti_coverage=ti_coverage, **kw)


def _no_ti_warning():
    return warnings.catch_warnings(record=True)


def test_a_covered_slice_is_fitted_as_given_and_recorded():
    with _no_ti_warning() as w:
        warnings.simplefilter("always")
        k = _build(dict(points=_pts(CORE), core_psi_n_max=None))
    assert not [m for m in w if issubclass(m.category, KI.TiCoverageWarning)]
    r = k.record["ti_coverage"]
    assert r["verdict"] == KI.TI_COVERED and r["ti"] == "measured" and r["reason"] == "covered"
    assert r["n_points"] == 9 and r["n_valid"] == 9 and r["n_valid_inside"] == 9
    assert r["psi_n_inner"] == pytest.approx(0.05) and r["psi_n_outer"] == pytest.approx(0.85)
    assert r["core_criterion"].startswith("not applied") and r["core_psi_n_max"] is None
    assert np.array_equal(k.ti, TI) and k.provenance["ti"] == "caller"
    json.dumps(r)                                              # the record goes into the fit record as JSON


def test_no_point_inside_the_boundary_is_refused_with_the_record():
    for pts in (_pts(OUTSIDE), _pts(np.array([])), dict(value=[np.nan, 1e37], psi_N=[0.3, 0.5])):
        with pytest.raises(KI.TiCoverageRefused, match="ion temperature not measured inside the plasma boundary") as e:
            _build(dict(points=pts, core_psi_n_max=None))
        assert isinstance(e.value, ValueError)                 # the form of the other input refusals
        assert e.value.record["verdict"] == KI.TI_NOT_COVERED and e.value.record["n_valid_inside"] == 0
        assert "no valid point inside the boundary" in e.value.record["reason"]


def test_invalid_points_do_not_count():
    # value <= 0, non-finite position, unusable uncertainty: none of them is a measurement inside the boundary
    pts = dict(value=[-5.0, 800.0, 900.0, 700.0], psi_N=[0.2, np.nan, 0.4, 0.5], sigma=[50.0, 50.0, np.nan, 0.0])
    with pytest.raises(KI.TiCoverageRefused) as e:
        _build(dict(points=pts, core_psi_n_max=None))
    assert e.value.record["n_points"] == 3 and e.value.record["n_valid"] == 0
    pts["sigma"][2] = 40.0
    k = _build(dict(points=pts, core_psi_n_max=None))
    assert k.record["ti_coverage"]["n_valid_inside"] == 1 and k.record["ti_coverage"]["psi_n_inner"] == 0.4


def test_points_with_times_count_only_inside_the_slice_window():
    pts = _pts(CORE, time=np.where(CORE < 0.5, 1.000, 1.100))
    with pytest.raises(ValueError, match="slice_window"):
        _build(dict(points=pts, core_psi_n_max=None))
    k = _build(dict(points=pts, core_psi_n_max=None, slice_window=(0.990, 1.010)))
    assert k.record["ti_coverage"]["n_valid_inside"] == int(np.sum(CORE < 0.5))
    with pytest.raises(KI.TiCoverageRefused):
        _build(dict(points=pts, core_psi_n_max=None, slice_window=(1.200, 1.300)))


def test_edge_only_coverage_passes_without_and_fails_with_a_core_criterion():
    k = _build(dict(points=_pts(EDGE), core_psi_n_max=None))
    assert k.record["ti_coverage"]["verdict"] == KI.TI_COVERED
    with pytest.raises(KI.TiCoverageRefused, match="innermost valid point at psi_N 0.900") as e:
        _build(dict(points=_pts(EDGE), core_psi_n_max=0.5))
    assert e.value.record["core_criterion"] == "applied" and e.value.record["core_psi_n_max"] == 0.5
    assert _build(dict(points=_pts(CORE), core_psi_n_max=0.5)).record["ti_coverage"]["verdict"] == KI.TI_COVERED


def test_the_core_threshold_has_no_default():
    p = inspect.signature(KI.ti_coverage_check).parameters["core_psi_n_max"]
    assert p.default is inspect.Parameter.empty and p.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match="core_psi_n_max"):
        _build(dict(points=_pts(CORE)))
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="core_psi_n_max"):
            _build(dict(points=_pts(CORE), core_psi_n_max=bad))


def test_the_check_is_required_and_its_options_are_checked():
    with pytest.raises(ValueError, match="ti_coverage is required"):
        _build(None)
    bad = [(dict(points=_pts(CORE), core_psi_n_max=None, bogus=1), "may carry only"),
           (dict(points=dict(value=[1e3], rho=[0.3]), core_psi_n_max=None), "psi_N"),
           (dict(points=_pts(OUTSIDE), core_psi_n_max=None, assumption="flat"), "assumption must be one of"),
           (dict(points=_pts(OUTSIDE), core_psi_n_max=None, assumption="equal_to_electron"), "assumption_sigma_rel"),
           (dict(points=_pts(OUTSIDE), core_psi_n_max=None, assumption="equal_to_electron", assumption_sigma_rel=0.0),
            "assumption_sigma_rel"),
           (dict(points=_pts(CORE), core_psi_n_max=None, assumption_sigma_rel=0.3), "without an assumption")]
    for opts, msg in bad:
        with pytest.raises(ValueError, match=msg):
            _build(opts)


def test_the_override_on_an_uncovered_slice_assumes_ti_equal_te_and_records_it():
    opts = dict(points=_pts(OUTSIDE), core_psi_n_max=None, assumption=KI.TI_EQUAL_TO_ELECTRON,
                assumption_sigma_rel=0.3)
    with pytest.warns(KI.TiCoverageWarning, match="ion temperature ASSUMED"):
        k = _build(opts, provenance=dict(ti="profile fit of the T_i points"))
    r = k.record["ti_coverage"]
    assert r["verdict"] == KI.TI_NOT_COVERED and r["ti"] == "assumed" and r["assumption_named"] == "equal_to_electron"
    a = r["assumption"]
    assert a["rule"] == "equal_to_electron" and a["statement"] == "T_i = T_e" and a["sigma_rel"] == 0.3
    assert a["coordinate"] == "psi_n" and a["range"] == [0.0, 1.0] and a["extent"] == "the whole table"
    assert np.array_equal(k.ti, TE)
    assert k.provenance["ti"].startswith("ASSUMED: T_i = T_e (rule 'equal_to_electron'")
    assert "profile fit of the T_i points" in k.provenance["ti"] and "not used" in k.provenance["ti"]
    p = k.thermal_pressure_at(psi_n=X)
    np.testing.assert_allclose(p, KI.EC * (NE * TE + (k.ni + k.nimp) * TE), rtol=1e-12)
    json.dumps(r)


def test_the_override_off_refuses_and_on_a_covered_slice_changes_nothing():
    with pytest.raises(KI.TiCoverageRefused, match="equal_to_electron"):
        _build(dict(points=_pts(OUTSIDE), core_psi_n_max=None))
    ref = _build(dict(points=_pts(CORE), core_psi_n_max=None))
    with _no_ti_warning() as w:
        warnings.simplefilter("always")
        k = _build(dict(points=_pts(CORE), core_psi_n_max=None, assumption=KI.TI_EQUAL_TO_ELECTRON,
                        assumption_sigma_rel=0.3))
    assert not [m for m in w if issubclass(m.category, KI.TiCoverageWarning)]
    r = k.record["ti_coverage"]
    assert r["ti"] == "measured" and "assumption" not in r and "not applied" in r["note"]
    for f in ("ti", "te", "ne", "ni", "nimp", "zeff"):
        assert np.array_equal(getattr(k, f), getattr(ref, f)), f
    assert k.provenance == ref.provenance


def test_the_check_applies_to_a_rho_table_too():
    rho = np.linspace(0.0, 1.0, 41)
    k = KI.assemble_kinetic_inputs(rho, np.interp(rho, X, NE), np.interp(rho, X, TE), np.interp(rho, X, TI),
                                   zeff_candidates={"vb": ("zeff", np.full_like(rho, 1.8))}, zeff_source="vb",
                                   z_imp=6.0, n_fast="none", ti_coverage=dict(points=_pts(EDGE), core_psi_n_max=None))
    assert k.coordinate == "rho_tor_norm" and k.record["ti_coverage"]["verdict"] == KI.TI_COVERED
