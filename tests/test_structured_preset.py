"""``structured_preset``: the named one-switch configuration of the closure.

A preset is a PRIOR plus a claim about the data.  These tests pin exactly that
and nothing more:

* the ladders it fills, in SIGMA terms, are the documented ones, and they
  survive the ``sigma -> W -> sigma`` round trip the config does internally
  (the config stores trust weights so the record reads on the usual scale, but
  the quantity anyone reasons about is the width);
* **explicit settings always win**: the preset fills only fields still at their
  dataclass default, checked field by field;
* a preset with NO l_i target omits ``structured_li_sigma`` entirely, so no
  sigma is recorded for a measurement that was never supplied;
* it changes no acceptance criterion, no tolerance and not ``closure_channel``
  -- the structured channel stays opt-in;
* an unknown name is REFUSED at construction, never silently ignored;
* the resolved settings round-trip through the config serialisation.

Solve-free throughout: this is configuration resolution, not closure algebra.
"""
import numpy as np
import pytest

from bouquet.config import GenerationConfig
from bouquet.utils import (STRUCTURED_PRESETS, sigma_from_weights,
                           structured_preset_settings)

PRESET = "li_soft_onesided"

#: The documented ladders, repeated here as an INDEPENDENT statement of intent:
#: if someone edits STRUCTURED_PRESETS, this test is what says so out loud.
SIGMA_BS = (0.50, 0.30, 0.15, 0.10)
SIGMA_IND_DOWN = (0.10, 0.40, 0.40, 0.40)
SIGMA_IND_UP = (0.10, 0.10, 0.10, 0.40)


def _cfg(**kw):
    return GenerationConfig(structured_preset=PRESET, **kw)


class TestPresetFills:
    def test_the_sigma_ladders_are_the_documented_ones(self):
        g = _cfg()
        sig = sigma_from_weights(g.structured_weights, K=4)
        assert np.allclose(sig["bs"], SIGMA_BS, rtol=1e-12, atol=0.0)
        assert np.allclose(sig["ind"], SIGMA_IND_DOWN, rtol=1e-12, atol=0.0)
        assert np.allclose(g.structured_sigma_ind_up, SIGMA_IND_UP,
                           rtol=1e-12, atol=0.0)

    def test_the_weights_are_sigma_to_the_minus_two(self):
        """The config stores W; W = sigma^-2 is the only relation allowed."""
        g = _cfg()
        W = g.structured_weights
        assert np.allclose(W["bs"], 1.0 / np.asarray(SIGMA_BS) ** 2)
        assert np.allclose(W["ind"], 1.0 / np.asarray(SIGMA_IND_DOWN) ** 2)
        assert W["name"] == PRESET          # provenance in the record

    def test_the_up_ladder_pins_and_frees_where_the_down_ladder_does(self):
        """The one-sided solvers refuse a ladder whose 0/inf pattern differs
        from the symmetric one, because the SET OF UNKNOWNS must not flip with
        a sign.  The shipped preset must therefore satisfy that itself."""
        g = _cfg()
        down = np.asarray(sigma_from_weights(g.structured_weights, K=4)["ind"])
        up = np.asarray(g.structured_sigma_ind_up, dtype=float)
        assert up.shape == down.shape
        assert np.array_equal(down <= 0.0, up <= 0.0)
        assert np.array_equal(np.isinf(down), np.isinf(up))

    def test_the_up_side_is_never_looser_than_the_down_side(self):
        """One-sided means "resist a RISE": every up sigma is <= its down
        sigma, and at least one is strictly tighter or the prior is symmetric
        and the field is pointless."""
        g = _cfg()
        down = np.asarray(sigma_from_weights(g.structured_weights, K=4)["ind"])
        up = np.asarray(g.structured_sigma_ind_up, dtype=float)
        assert np.all(up <= down + 1e-15)
        assert np.any(up < down - 1e-15)

    def test_the_bootstrap_ladder_tightens_towards_the_pedestal(self):
        """Redl/Sauter is validated against drift-kinetic calculations in the
        steep-gradient pedestal and is at its worst on axis, so sigma_bs must
        DECREASE outward."""
        assert np.all(np.diff(SIGMA_BS) < 0)

    def test_the_inductive_core_is_the_tightest_entry(self):
        """The axis row already determines the inductive core wherever the
        sawtooth gate admits one, so the core is pinned and the outer three
        are left looser -- that is where the closure is allowed to work."""
        s = np.asarray(SIGMA_IND_DOWN)
        assert s[0] == pytest.approx(0.10)
        assert np.all(s[1:] > s[0])

    def test_data_statement_switches(self):
        g = _cfg()
        assert g.structured_soft is True
        assert g.structured_ip_sigma_frac == pytest.approx(0.005)
        assert g.structured_ip_sigma is None        # frac and absolute exclude

    def test_li_sigma_only_when_a_target_was_supplied(self):
        assert _cfg().structured_li_sigma is None
        assert _cfg(structured_li_target=0.92).structured_li_sigma == \
            pytest.approx(0.04)

    def test_no_preset_changes_nothing(self):
        g = GenerationConfig()
        assert g.structured_preset is None
        assert g.structured_weights is None
        assert g.structured_sigma_ind_up is None
        assert g.structured_soft is False
        assert g.structured_ip_sigma_frac is None
        assert g.structured_li_sigma is None


class TestPresetTouchesNoCriterion:
    """A preset is a prior.  It may not move a tolerance, a bound or a gate."""

    def test_channel_stays_opt_in(self):
        assert _cfg().closure_channel == "bootstrap"
        assert GenerationConfig().closure_channel == "bootstrap"

    def test_tolerances_and_gates_are_untouched(self):
        base, g = GenerationConfig(), _cfg()
        for name in ("structured_li_tol", "q0_tol", "q0_gate",
                     "structured_li_max_corrector_steps",
                     "structured_li_kind", "structured_basis",
                     "structured_li_target"):
            assert getattr(g, name) == getattr(base, name), name


class TestOverridePrecedence:
    """Every field the preset fills must yield to an explicit setting."""

    @pytest.mark.parametrize("name,value", [
        ("structured_weights", dict(name="uniform", ind=(1.0, 1.0, 1.0, 1.0),
                                    bs=(1.0, 1.0, 1.0, 1.0))),
        ("structured_sigma_ind_up", [0.25, 0.25, 0.25, 0.25]),
        ("structured_ip_sigma_frac", 0.01),
    ])
    def test_explicit_setting_wins(self, name, value):
        g = GenerationConfig(structured_preset=PRESET, **{name: value})
        assert getattr(g, name) == value

    def test_explicit_li_sigma_wins_over_the_preset_default(self):
        g = GenerationConfig(structured_preset=PRESET,
                             structured_li_target=0.92,
                             structured_li_sigma=0.10)
        assert g.structured_li_sigma == pytest.approx(0.10)

    def test_an_override_does_not_suppress_the_other_fills(self):
        g = GenerationConfig(structured_preset=PRESET,
                             structured_ip_sigma_frac=0.01)
        assert g.structured_soft is True
        assert g.structured_weights["name"] == PRESET

    def test_an_absolute_ip_sigma_survives_and_frac_is_still_filled(self):
        """The two are mutually exclusive downstream; the preset must not be
        the thing that decides which one the caller meant, so the clash is
        left to the closure's own refusal rather than silently resolved."""
        g = GenerationConfig(structured_preset=PRESET,
                             structured_ip_sigma=1.0e5)
        assert g.structured_ip_sigma == pytest.approx(1.0e5)
        assert g.structured_ip_sigma_frac == pytest.approx(0.005)


class TestRefusals:
    def test_unknown_preset_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="unknown structured_preset"):
            GenerationConfig(structured_preset="li_soft_twosided")

    def test_the_refusal_names_the_known_presets(self):
        with pytest.raises(ValueError, match=PRESET):
            GenerationConfig(structured_preset="nope")

    def test_the_helper_refuses_the_same_way(self):
        with pytest.raises(ValueError, match="unknown structured_preset"):
            structured_preset_settings("nope")

    def test_the_registry_is_not_mutated_by_a_caller(self):
        """``structured_preset_settings`` must hand back a fresh dict, or one
        campaign runner's edit silently reconfigures the next."""
        first = structured_preset_settings(PRESET)
        first["structured_soft"] = False
        first["structured_weights"]["ind"] = (0.0, 0.0, 0.0, 0.0)
        assert structured_preset_settings(PRESET)["structured_soft"] is True
        assert _cfg().structured_weights["ind"][0] == pytest.approx(100.0)
        assert STRUCTURED_PRESETS[PRESET]["sigma_ind"] == SIGMA_IND_DOWN


class TestSerialisation:
    def test_resolved_preset_round_trips(self):
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t",
                            generation=GenerationConfig(
                                structured_preset=PRESET,
                                closure_channel="structured",
                                structured_li_target=0.92))
        back = BouquetConfig.from_json(cfg.to_json()).generation
        assert back.structured_preset == PRESET
        assert back.structured_soft is True
        assert back.structured_li_sigma == pytest.approx(0.04)
        assert np.allclose(back.structured_sigma_ind_up, SIGMA_IND_UP)
        assert np.allclose(
            sigma_from_weights(back.structured_weights, K=4)["bs"], SIGMA_BS)
