"""The stimulus axis: naming, aliases, and which rows are actually trained.

Three things are pinned here.

**`none` is a catch-all and says so.** It covers both "no treatment recorded"
and "condition not recorded at all". The slot used to be called `basal`, which
asserted a resting biological state; we rarely have evidence for that claim,
only the absence of a recorded treatment. The conflation is accepted, but it
must stay visible rather than being dressed up as a measurement.

**Two rows are dead.** `ifn_type1` and `tnf_alpha` match zero rows in the
corpus, so their embedding rows never receive gradient. That is the same shape
as gap 2, and it is recorded here so it stays a known gap.

**An unmapped condition is not the same as a missing one.** If hitlist grows a
treatment category this table does not know, folding it silently into `none`
would score genuinely stimulated samples as unstimulated.
"""

import pytest

from presto.data.vocab import (  # noqa: E402
    CONDITION_TO_STIMULUS,
    LEGACY_STIMULUS_ALIASES,
    PROCESSING_STIMULI,
    PROCESSING_STIMULUS_TO_IDX,
    is_unmapped_condition,
    processing_stimulus_index,
    stimulus_for_condition,
)

#: Tokens with real support, measured over all 4,260,527 elution rows:
#:
#:   none             4,095,917   ifn_gamma   71,040
#:   cell_activation     37,929   tlr         34,247
#:   ifn_type1           21,394   tnf_alpha        0
#:
#: `ifn_type1` had zero support until infection categories were mapped on the
#: evidence that viral infection induces endogenous type I interferon.
STIMULI_WITH_DATA = ("none", "ifn_gamma", "ifn_type1", "tlr", "cell_activation")

#: Declared but unsupported. Kept as headroom; pinned so it stays visible.
STIMULI_WITHOUT_DATA = ("tnf_alpha",)


class TestVocabularyShape:
    def test_default_is_none_not_basal(self):
        """`basal` overclaimed: it asserted a resting state we did not measure."""
        assert PROCESSING_STIMULI[0] == "none"
        assert "basal" not in PROCESSING_STIMULI

    def test_type1_interferon_is_spelled_out(self):
        """`ifn_ab` reads as "antibody" in an immunology codebase."""
        assert "ifn_type1" in PROCESSING_STIMULI
        assert "ifn_ab" not in PROCESSING_STIMULI

    def test_alpha_and_beta_share_one_token(self):
        """Both bind IFNAR1/2 and drive the same ISGF3 program."""
        assert CONDITION_TO_STIMULUS["IFN_alpha_treatment"] == "ifn_type1"
        assert CONDITION_TO_STIMULUS["IFN_beta_treatment"] == "ifn_type1"

    def test_gamma_is_kept_separate_from_type_1(self):
        """Type II, different receptor (IFNGR1/2), different program."""
        assert CONDITION_TO_STIMULUS["IFN_gamma_treatment"] == "ifn_gamma"
        assert CONDITION_TO_STIMULUS["IFN_gamma_treatment"] != "ifn_type1"


class TestLegacySpellings:
    @pytest.mark.parametrize("legacy,current", sorted(LEGACY_STIMULUS_ALIASES.items()))
    def test_legacy_name_resolves_to_the_same_row(self, legacy, current):
        """A saved record must not drift onto a different embedding row.

        Without the alias, `ifn_ab` would fall through to `none` and a
        stimulated sample would be scored as unstimulated.
        """
        assert processing_stimulus_index(legacy) == PROCESSING_STIMULUS_TO_IDX[current]

    def test_indices_are_unchanged_by_the_rename(self):
        """Positions are the checkpoint contract; only the spellings moved."""
        assert PROCESSING_STIMULI.index("none") == 0
        assert PROCESSING_STIMULI.index("ifn_gamma") == 1
        assert PROCESSING_STIMULI.index("ifn_type1") == 2
        assert PROCESSING_STIMULI.index("tnf_alpha") == 3
        assert PROCESSING_STIMULI.index("tlr") == 4


class TestNoneIsACatchAll:
    @pytest.mark.parametrize("condition", ["untreated", "", None, "anything_unmapped"])
    def test_unrecognized_conditions_fall_back_to_none(self, condition):
        assert stimulus_for_condition(condition) == "none"

    def test_missing_and_unmapped_are_distinguishable(self):
        """Both become `none`, but only one is a maintenance signal.

        An unmapped non-empty category means hitlist grew a treatment this
        table does not know about -- real stimulated samples scored as
        unstimulated. That must be detectable even though the token is shared.
        """
        assert is_unmapped_condition("some_new_hitlist_category") is True
        assert is_unmapped_condition("") is False
        assert is_unmapped_condition(None) is False
        assert is_unmapped_condition("IFN_gamma_treatment") is False


class TestKnownDeadRows:
    """Pins which embedding rows the corpus cannot train.

    If a future corpus supplies these, this test fails and should be updated
    -- that failure is the notification that the headroom became real.
    """

    @pytest.mark.parametrize("token", STIMULI_WITHOUT_DATA)
    def test_row_is_declared_but_unsupported(self, token):
        assert token in PROCESSING_STIMULI
        assert token not in STIMULI_WITH_DATA

    @pytest.mark.parametrize("token", STIMULI_WITH_DATA)
    def test_supported_rows_are_in_the_vocabulary(self, token):
        assert token in PROCESSING_STIMULI

    def test_every_token_is_accounted_for(self):
        """No third category: a token either has data or is declared dead."""
        assert set(PROCESSING_STIMULI) == (
            set(STIMULI_WITH_DATA) | set(STIMULI_WITHOUT_DATA)
        )


class TestInfectionMapping:
    """Infection categories carry a stimulus; they must not read as resting.

    Viral infection is sensed by RIG-I/MDA5 and cGAS-STING and drives
    autocrine type I interferon -- the immunoproteasome swap, TAP1/2 and MHC-I
    upregulation that this axis exists to represent. Leaving it in `none`
    scored ~21k genuinely stimulated samples as unstimulated and left the
    `ifn_type1` embedding row with no data at all.
    """

    def test_viral_infection_is_type_1_interferon(self):
        assert stimulus_for_condition("infection_viral") == "ifn_type1"

    def test_bacterial_and_parasitic_infection_is_tlr(self):
        assert stimulus_for_condition("infection_bacterial_or_parasite") == "tlr"

    def test_inactivated_virus_control_stays_unstimulated(self):
        """The paired comparator for infection_viral; pairing gives contrast."""
        assert stimulus_for_condition("virus_inactivated_control") == "none"

    def test_gene_delivery_is_not_an_infection(self):
        """Vector-derived, but not an immunological infection."""
        for category in ("transduction", "transfection", "CIITA_transduction"):
            assert stimulus_for_condition(category) == "none"

    def test_apm_categories_are_reviewed_not_unmapped(self):
        """They ride the apm_perturbation axis; listing them keeps the
        unmapped-category signal meaningful rather than noisy."""
        for category in (
            "ERAP1_perturbation",
            "HLA-DM_perturbation",
            "TAP_perturbation",
            "MHC-I_loss_B2M",
        ):
            assert not is_unmapped_condition(category)
