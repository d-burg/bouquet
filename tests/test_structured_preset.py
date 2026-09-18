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
* the resolved settings round-trip through the config serialisation;
* the structured channel's DEFAULT preset (``TestDefaultPreset`` onwards): a
  bare ``closure_channel="structured"`` resolves to ``li_soft_onesided`` --
  same settings AND same closure answer as naming it -- ``structured_preset=
  "none"`` declines it and reproduces the raw shipped fields exactly, explicit
  fields still win, every other channel is untouched, and the record says which
  preset was in force and whether it was chosen by default or explicitly.

Solve-free throughout: this is configuration resolution, not closure algebra --
except ``TestDefaultIsTheValidatedConfiguration``, which runs the shipped
algebra on a small synthetic closure problem (still no GS solve) to show the
two routes give the SAME numbers, not merely the same fields.
"""
import numpy as np
import pytest

from bouquet.config import GenerationConfig, resolve_structured_preset
from bouquet.utils import (STRUCTURED_PRESET_DEFAULT, STRUCTURED_PRESET_NONE,
                           STRUCTURED_PRESETS, close_ip_structured,
                           close_ip_structured_soft, sigma_from_weights,
                           structured_basis_eval, structured_preset_settings)

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
    """Every field the preset fills must yield to an explicit setting that
    DIFFERS from the field's own default -- which is as far as a plain
    dataclass can see (see the limitation pinned at the end of this class).
    """

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

    def test_a_field_set_to_its_own_default_is_overridden_and_warns(self):
        """The limitation the docstrings used to deny: a plain dataclass
        cannot tell "left at the default" from "explicitly set to the default
        value", so the preset overrides the second one too.  Pinned here
        rather than redesigned -- what is required is that the override is
        NAMED, so it is visible instead of silent."""
        default_soft = GenerationConfig.__dataclass_fields__[
            "structured_soft"].default
        assert default_soft is False
        with pytest.warns(UserWarning, match="structured_soft"):
            g = GenerationConfig(structured_preset=PRESET,
                                 structured_soft=default_soft)
        assert g.structured_soft is True          # the preset won

    def test_the_warning_names_every_field_the_preset_filled(self):
        with pytest.warns(UserWarning) as rec:
            GenerationConfig(structured_preset=PRESET)
        msg = str(rec[0].message)
        for name in ("structured_weights", "structured_sigma_ind_up",
                     "structured_soft", "structured_ip_sigma_frac"):
            assert name in msg


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


# ── the structured channel's DEFAULT preset ─────────────────────────────────
#
# Before this, a bare closure_channel="structured" ran on the raw shipped
# fields -- the symmetric physics ladder on the hard solver -- which is the
# configuration the l_i study SUPERSEDED, and the validated one was reachable
# only by naming it.  The default now IS the validated one; "none" is the
# opt-out that reproduces the old behaviour exactly.

#: Every config field a preset can fill.  Kept here (not imported) so that a
#: preset growing a new field has to be acknowledged in this test file too.
PRESET_FIELDS = ("structured_weights", "structured_sigma_ind_up",
                 "structured_soft", "structured_ip_sigma_frac",
                 "structured_li_sigma")

#: Fields the structured channel reads that NO preset may touch.
UNTOUCHED_FIELDS = ("closure_channel", "structured_basis",
                    "structured_li_target", "structured_li_kind",
                    "structured_li_tol", "structured_li_max_corrector_steps",
                    "structured_ip_sigma", "q0_tol", "q0_gate")


def _defaults(*names):
    d = GenerationConfig.__dataclass_fields__
    return {n: d[n].default for n in names}


def _structured_fields(g):
    return {n: getattr(g, n) for n in PRESET_FIELDS + UNTOUCHED_FIELDS
            if n != "closure_channel"}


def _synthetic_problem():
    """A small, closable synthetic Ip problem: (psi, w, c, j_ind, j_bs, j_fix,
    Ip_target).  No geometry, no solver -- the algebra only needs a positive
    linear measure and three component profiles that MISS the target.
    """
    psi = np.linspace(0.01, 0.99, 201)
    w = 2.0e0 * psi + 0.5                      # any positive linear measure
    c = 3.0e4                                  # the affine p' term
    j_ind = 8.0e5 * (1.0 - psi) ** 1.5
    j_bs = 3.0e5 * np.exp(-((psi - 0.9) / 0.06) ** 2) + 2.0e4 * (1.0 - psi)
    j_fix = 1.0e5 * (1.0 - psi) ** 3
    lin = lambda j: float(np.trapezoid(w * np.asarray(j, float), psi)) \
        if hasattr(np, "trapezoid") else float(np.trapz(w * np.asarray(j, float), psi))
    raw = lin(j_ind) + lin(j_bs) + lin(j_fix) + c
    return psi, w, c, j_ind, j_bs, j_fix, 1.04 * raw


def _close_from_config(g):
    """Drive the SHIPPED algebra exactly as ``run.py`` does, from a config.

    Mirrors ``Bouquet._close_ip_structured_predictor``'s dispatch (hard vs
    soft, sigma = W^-1/2, sigma_Ip resolved from the fraction) and nothing
    else: both sides of every comparison below go through this same driver, so
    it can only ever cancel.
    """
    psi, w, c, j_ind, j_bs, j_fix, Ip = _synthetic_problem()
    if g.structured_soft:
        K = structured_basis_eval(g.structured_basis, psi).shape[0]
        sig = sigma_from_weights(g.structured_weights, K)
        ip_sigma = (g.structured_ip_sigma if g.structured_ip_sigma is not None
                    else (None if g.structured_ip_sigma_frac is None
                          else float(g.structured_ip_sigma_frac) * abs(Ip)))
        return close_ip_structured_soft(
            psi, w, c, Ip, ip_sigma, j_ind, j_bs, j_fix,
            basis=g.structured_basis, sigma_ind=sig["ind"], sigma_bs=sig["bs"],
            sigma_ind_up=g.structured_sigma_ind_up, axis=None)
    return close_ip_structured(
        psi, w, c, Ip, j_ind, j_bs, j_fix, basis=g.structured_basis,
        weights=g.structured_weights,
        sigma_ind_up=g.structured_sigma_ind_up, axis=None)


class TestDefaultPreset:
    """A bare structured channel gets the validated preset."""

    def test_bare_structured_channel_resolves_to_the_default_preset(self):
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured")
        assert STRUCTURED_PRESET_DEFAULT == PRESET
        assert g.structured_preset_in_force == PRESET
        assert g.structured_preset_source == "default"
        assert g.structured_preset is None          # the caller named nothing
        assert g.structured_soft is True
        assert g.structured_weights["name"] == PRESET

    def test_default_and_explicit_give_the_same_settings(self):
        with pytest.warns(UserWarning):
            by_default = GenerationConfig(closure_channel="structured")
        with pytest.warns(UserWarning):
            explicit = GenerationConfig(closure_channel="structured",
                                        structured_preset=PRESET)
        assert _structured_fields(by_default) == _structured_fields(explicit)

    def test_default_and_explicit_agree_with_an_li_target(self):
        kw = dict(closure_channel="structured", structured_li_target=0.92)
        with pytest.warns(UserWarning):
            by_default = GenerationConfig(**kw)
        with pytest.warns(UserWarning):
            explicit = GenerationConfig(structured_preset=PRESET, **kw)
        assert _structured_fields(by_default) == _structured_fields(explicit)
        assert by_default.structured_li_sigma == pytest.approx(0.04)

    def test_without_an_li_target_it_degrades_exactly_as_when_named(self):
        """No l_i target -> soft Ip + the one-sided prior, and NO l_i row:
        the same degradation the named preset has always had."""
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured")
        assert g.structured_li_target is None
        assert g.structured_li_sigma is None        # no sigma for no target
        assert g.structured_soft is True
        assert g.structured_ip_sigma_frac == pytest.approx(0.005)
        assert np.allclose(g.structured_sigma_ind_up, SIGMA_IND_UP)

    def test_the_warning_says_the_preset_came_by_default_and_how_to_decline(self):
        with pytest.warns(UserWarning) as rec:
            GenerationConfig(closure_channel="structured")
        msg = str(rec[0].message)
        assert "default" in msg.lower()
        assert "BY DEFAULT" in msg
        assert f"structured_preset={STRUCTURED_PRESET_NONE!r}" in msg
        for name in ("structured_weights", "structured_sigma_ind_up",
                     "structured_soft", "structured_ip_sigma_frac"):
            assert name in msg

    def test_the_default_changes_no_acceptance_criterion(self):
        base = GenerationConfig()
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured")
        for name in UNTOUCHED_FIELDS:
            if name == "closure_channel":
                continue
            assert getattr(g, name) == getattr(base, name), name


class TestOptOut:
    """``structured_preset="none"`` reproduces the pre-change default."""

    def test_opt_out_leaves_every_structured_field_as_shipped(self):
        g = GenerationConfig(closure_channel="structured",
                             structured_preset=STRUCTURED_PRESET_NONE)
        assert _structured_fields(g) == _structured_fields(GenerationConfig())
        assert g.structured_preset_in_force is None
        assert g.structured_preset_source == "opt-out"

    def test_opt_out_does_not_warn(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            GenerationConfig(closure_channel="structured",
                             structured_preset=STRUCTURED_PRESET_NONE)

    def test_opt_out_round_trips(self):
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t",
                            generation=GenerationConfig(
                                closure_channel="structured",
                                structured_preset=STRUCTURED_PRESET_NONE))
        back = BouquetConfig.from_json(cfg.to_json()).generation
        assert back.structured_preset == STRUCTURED_PRESET_NONE
        assert back.structured_preset_source == "opt-out"
        assert _structured_fields(back) == _structured_fields(
            GenerationConfig())

    def test_the_helper_returns_nothing_for_the_opt_out(self):
        assert structured_preset_settings(STRUCTURED_PRESET_NONE) == {}


class TestOtherChannelsUntouched:
    """Nothing changes for anyone who does not select the structured channel."""

    @pytest.mark.parametrize("channel", ["bootstrap", "ohmic",
                                         "sawtooth_bootstrap"])
    def test_no_preset_is_applied_off_the_structured_channel(self, channel):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")       # not even a warning fires
            g = GenerationConfig(closure_channel=channel)
        assert g.structured_preset_in_force is None
        assert g.structured_preset_source == "unset"
        assert _structured_fields(g) == _structured_fields(GenerationConfig())

    def test_the_shipped_channel_default_is_still_bootstrap(self):
        assert GenerationConfig().closure_channel == "bootstrap"
        assert GenerationConfig().structured_preset is None

    def test_every_field_of_a_default_config_is_its_dataclass_default(self):
        """Bit-identical for a user who selects nothing: the whole config, not
        just the structured block."""
        import dataclasses as dc
        g = GenerationConfig()
        for f in dc.fields(g):
            if not f.init or f.default is dc.MISSING:
                continue
            assert getattr(g, f.name) == f.default, f.name


class TestDefaultYieldsToExplicitSettings:
    """Explicit non-default fields win over the DEFAULT preset exactly as they
    win over a named one."""

    @pytest.mark.parametrize("name,value", [
        ("structured_weights", dict(name="uniform", ind=(1.0, 1.0, 1.0, 1.0),
                                    bs=(1.0, 1.0, 1.0, 1.0))),
        ("structured_sigma_ind_up", [0.25, 0.25, 0.25, 0.25]),
        ("structured_ip_sigma_frac", 0.01),
    ])
    def test_explicit_setting_wins(self, name, value):
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured",
                                 **{name: value})
        assert getattr(g, name) == value
        assert g.structured_preset_source == "default"
        assert name not in g.structured_preset_fields

    def test_an_explicit_li_sigma_wins(self):
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured",
                                 structured_li_target=0.92,
                                 structured_li_sigma=0.10)
        assert g.structured_li_sigma == pytest.approx(0.10)

    def test_an_absolute_ip_sigma_is_not_turned_into_a_refusal(self):
        """A NAMED preset fills the fraction alongside an absolute sigma and
        lets the closure refuse the (mutually exclusive) pair -- the caller
        asked for the preset, so the ambiguity is theirs.  A DEFAULT may not
        do that: it would turn a configuration that ran yesterday into a
        refusal.  So the fraction is left unfilled here, and only here."""
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured",
                                 structured_ip_sigma=1.0e5)
        assert g.structured_ip_sigma == pytest.approx(1.0e5)
        assert g.structured_ip_sigma_frac is None
        assert g.structured_soft is True         # the rest of the preset holds
        with pytest.warns(UserWarning):
            named = GenerationConfig(closure_channel="structured",
                                     structured_preset=PRESET,
                                     structured_ip_sigma=1.0e5)
        assert named.structured_ip_sigma_frac == pytest.approx(0.005)

    def test_a_custom_basis_declines_the_default_rather_than_mispairing(self):
        """The preset's ladders are widths at the SHIPPED basis's radii; on
        another basis they mean nothing (and a different K is refused outright
        downstream).  So the default steps aside, loudly, and every field keeps
        its own default."""
        with pytest.warns(UserWarning, match="structured_basis"):
            g = GenerationConfig(closure_channel="structured",
                                 structured_basis=dict(kind="constant"))
        assert g.structured_preset_in_force is None
        assert g.structured_preset_source == "default-declined-custom-basis"
        assert g.structured_weights is None
        assert g.structured_soft is False
        assert g.structured_sigma_ind_up is None

    def test_a_named_preset_still_applies_on_a_custom_basis(self):
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured",
                                 structured_preset=PRESET,
                                 structured_basis=dict(kind="gaussian",
                                                       centres=[0.2, 0.8],
                                                       widths=[0.15, 0.15]))
        assert g.structured_weights["name"] == PRESET
        assert g.structured_preset_source == "explicit"


class TestDefaultIsTheValidatedConfiguration:
    """The two routes must agree on the ANSWER, not just on the fields."""

    def test_same_closure_as_naming_the_preset(self):
        with pytest.warns(UserWarning):
            by_default = GenerationConfig(closure_channel="structured")
        with pytest.warns(UserWarning):
            explicit = GenerationConfig(closure_channel="structured",
                                        structured_preset=PRESET)
        a, b = _close_from_config(by_default), _close_from_config(explicit)
        for key in ("a", "b", "s_ind", "s_bs"):
            assert np.array_equal(np.asarray(a[key], float),
                                  np.asarray(b[key], float)), key
        assert a["solver"] == b["solver"]

    def test_the_opt_out_reproduces_the_pre_change_closure_exactly(self):
        """The raw shipped fields, driven straight from a default config, are
        bit-identical to the opt-out's -- that is what "reproduces the old
        default" means."""
        raw = GenerationConfig()                    # channel irrelevant here
        opted_out = GenerationConfig(
            closure_channel="structured",
            structured_preset=STRUCTURED_PRESET_NONE)
        a, b = _close_from_config(raw), _close_from_config(opted_out)
        for key in ("a", "b", "s_ind", "s_bs"):
            assert np.array_equal(np.asarray(a[key], float),
                                  np.asarray(b[key], float)), key
        assert b["solver"] == a["solver"]
        assert b["weights_name"] == "physics-prior"

    def test_the_default_actually_differs_from_the_opt_out(self):
        """If these agreed, the change would be cosmetic and the preset
        pointless."""
        with pytest.warns(UserWarning):
            by_default = GenerationConfig(closure_channel="structured")
        opted_out = GenerationConfig(
            closure_channel="structured",
            structured_preset=STRUCTURED_PRESET_NONE)
        a, b = _close_from_config(by_default), _close_from_config(opted_out)
        assert a["solver"] != b["solver"]           # posterior mode vs hard KKT
        assert not np.allclose(np.asarray(a["s_ind"], float),
                               np.asarray(b["s_ind"], float))


class TestResolutionIsRecordedAndIdempotent:
    def test_resolving_twice_changes_nothing_and_warns_once(self):
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured")
        before = _structured_fields(g)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # no second warning
            rec = resolve_structured_preset(g)
        assert _structured_fields(g) == before
        assert rec["source"] == "default"
        assert rec["name"] == PRESET
        assert set(rec["fields"]) == set(g.structured_preset_fields)

    def test_a_channel_set_after_construction_is_still_resolved(self):
        """The docs' own one-liner sets the channel on an existing config; the
        closure resolves again at its entry point so that route is not left on
        the superseded raw fields."""
        g = GenerationConfig()
        assert g.structured_preset_source == "unset"
        g.closure_channel = "structured"
        with pytest.warns(UserWarning, match="BY DEFAULT"):
            rec = resolve_structured_preset(g)
        assert rec["source"] == "default" and rec["name"] == PRESET
        assert g.structured_soft is True

    def test_the_record_names_the_preset_and_who_chose_it(self):
        with pytest.warns(UserWarning):
            g = GenerationConfig(closure_channel="structured")
        assert g.structured_preset_in_force == PRESET
        assert g.structured_preset_source == "default"
        assert "structured_weights" in g.structured_preset_fields

    def test_provenance_round_trips_through_serialisation(self):
        from bouquet.config import BouquetConfig, ImasSource, SolverConfig
        with pytest.warns(UserWarning):
            gen = GenerationConfig(closure_channel="structured")
        cfg = BouquetConfig(source=ImasSource(ids_path="unused.json"),
                            solver=SolverConfig(mesh_path="unused.h5"),
                            output_header="t", generation=gen)
        d = cfg.to_dict()["generation"]
        assert d["structured_preset_in_force"] == PRESET
        assert d["structured_preset_source"] == "default"
        back = BouquetConfig.from_dict(cfg.to_dict()).generation
        assert back.structured_preset_in_force == PRESET
        assert back.structured_preset_source == "default"
        # (the weight tuples come back as lists -- JSON has one sequence type;
        # compare them on the scale anyone reasons about instead)
        for key in ("ind", "bs"):
            assert np.allclose(sigma_from_weights(back.structured_weights,
                                                  K=4)[key],
                               sigma_from_weights(gen.structured_weights,
                                                  K=4)[key])
        assert {k: v for k, v in _structured_fields(back).items()
                if k != "structured_weights"} == \
            {k: v for k, v in _structured_fields(gen).items()
             if k != "structured_weights"}

    def test_the_closure_resolves_and_announces_the_default(self):
        """The run path's own hook, pinned by source: the predictor resolves
        the preset and says in its printed line that it came BY DEFAULT, with
        the scope caveat and the opt-out."""
        import inspect
        from bouquet.run import Bouquet
        src = inspect.getsource(Bouquet._close_ip_structured_predictor)
        assert "resolve_structured_preset(gc)" in src
        assert "BY DEFAULT" in src
        assert "structured_preset='none'" in src
        assert "structured_preset_source=preset_rec[\"source\"]" in src


class TestOptOutRefusals:
    def test_the_opt_out_is_the_only_extra_spelling_accepted(self):
        with pytest.raises(ValueError, match="unknown structured_preset"):
            GenerationConfig(structured_preset="off")

    def test_the_refusal_mentions_the_opt_out(self):
        with pytest.raises(ValueError, match=STRUCTURED_PRESET_NONE):
            GenerationConfig(structured_preset="nope")
