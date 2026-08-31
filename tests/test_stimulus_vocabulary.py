"""The stimulus axis: naming, aliases, and which rows are actually trained.

Three things are pinned here.

**`none` is a catch-all and says so.** It covers both "no treatment recorded"
and "condition not recorded at all". The slot used to be called `basal`, which
asserted a resting biological state; we rarely have evidence for that claim,
only the absence of a recorded treatment. The conflation is accepted, but it
must stay visible rather than being dressed up as a measurement.

**One row is dead.** `tnf_alpha` matches zero rows in the corpus, so its
embedding never receives gradient. That is the same shape as gap 2, and it is
recorded here so it stays a known gap. (`ifn_type1` was dead too until viral
infection was mapped onto it; the header used to still say so after the table
below had been updated, which is the drift these tests exist to prevent.)

**An unmapped condition is not the same as a missing one.** If hitlist grows a
treatment category this table does not know, folding it silently into `none`
would score genuinely stimulated samples as unstimulated.
"""

import re
from pathlib import Path

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
STIMULI_WITH_DATA = (
    "none",
    "ifn_gamma",
    "ifn_type1",
    "tlr",
    "cell_activation",
    "cytokine_unspecified",
)

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


class TestEveryCorpusCategoryIsMapped:
    """The corpus's own `condition_category` values, counted from hitlist.

    `SPPL3_perturbation` (14,906 rows) and `IRF2_perturbation` (5,314) were
    absent from CONDITION_TO_STIMULUS while carrying real volume. The predicate
    that detects this already existed and the count was already in the ingest
    stats -- nothing read the key, so both were scored as unstimulated in
    silence. This pins the observed category set so the next addition fails
    here instead.
    """

    #: Every distinct `condition_category` in the MS evidence table, with row
    #: counts, as of hitlist 1.45.0. Empty string (1,435,250 rows) omitted: it
    #: is "not recorded", which is a missing condition rather than an unmapped
    #: one, and is covered by TestNoneIsACatchAll.
    OBSERVED = {
        "unperturbed": 2362271,
        "virus_inactivated_control": 82182,
        "IFN_gamma_treatment": 71910,
        "CIITA_transduction": 53465,
        "PLC_chaperone_perturbation": 52269,
        "HLA-DM_perturbation": 42405,
        "ERAP1_perturbation": 41081,
        "cell_activation": 37929,
        "drug_exposure": 35344,
        "transduction": 32574,
        "TAP_perturbation": 31067,
        "infection_bacterial_or_parasite": 24534,
        "ERAP2_perturbation": 22266,
        "infection_viral": 21481,
        "transfection": 19344,
        "SPPL3_perturbation": 14906,
        "transplant": 14842,
        "labeling_control": 11147,
        "TLR_stimulation": 9900,
        "tapasin_perturbation": 7662,
        "cytokine_treatment_generic": 6159,
        "IRF2_perturbation": 5314,
        "other_perturbation": 3499,
        "MHC-I_loss_B2M": 520,
    }

    @pytest.mark.parametrize("category", sorted(OBSERVED))
    def test_observed_category_is_mapped(self, category):
        assert not is_unmapped_condition(category), (
            f"{category!r} ({self.OBSERVED[category]} rows) is not in "
            "CONDITION_TO_STIMULUS and is being scored as unstimulated"
        )

    def test_generic_cytokine_is_not_called_unstimulated(self):
        """The one case where `none` would state something known to be false.

        These cells were treated with a cytokine; the deposit just does not
        name which. `none` means "not known to be stimulated" -- here we know.
        """
        assert (
            CONDITION_TO_STIMULUS["cytokine_treatment_generic"]
            == "cytokine_unspecified"
        )
        assert stimulus_for_condition("cytokine_treatment_generic") != "none"

    def test_apm_lesions_are_not_stimuli(self):
        """A standing lesion in the machinery is not a treatment applied.

        These ride the separate `apm_perturbation` axis; listing them here only
        keeps them from registering as unreviewed categories.
        """
        for category in ("SPPL3_perturbation", "IRF2_perturbation"):
            assert stimulus_for_condition(category) == "none"


class TestDocsMatchTheVocabulary:
    """`docs/model_io_contract.md` must list the tokens the code actually has.

    The contract doc is normative, so a stale row there is a wrong spec rather
    than an out-of-date comment. When `inducer` was renamed to `stimulus` the
    row *label* was updated but its token set was not, leaving the doc
    advertising `{basal, ifn_gamma, ifn_ab, tnf_alpha, tlr}` -- two names the
    code had dropped, one it had added -- plus the "unperturbed cells carry
    basal interferon tone" claim the rename existed to remove. Nothing caught
    it, because no test compared the two.
    """

    CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "model_io_contract.md"

    def _stimulus_row(self) -> str:
        for line in self.CONTRACT.read_text().splitlines():
            if line.startswith("| `stimulus`"):
                return line
        raise AssertionError("no `stimulus` row in docs/model_io_contract.md")

    def test_documented_token_set_matches_code(self):
        row = self._stimulus_row()
        braced = re.search(r"\{([^}]*)\}", row)
        assert braced, f"no token set in the stimulus row: {row}"
        documented = {tok.strip() for tok in braced.group(1).split(",") if tok.strip()}
        assert documented == set(PROCESSING_STIMULI), (
            "docs/model_io_contract.md lists a different stimulus vocabulary "
            f"than data/vocab.py.\n  only in docs: {sorted(documented - set(PROCESSING_STIMULI))}"
            f"\n  only in code: {sorted(set(PROCESSING_STIMULI) - documented)}"
        )

    @pytest.mark.parametrize("retired", ["basal", "ifn_ab", "inducer"])
    def test_retired_names_do_not_reappear_in_the_contract(self, retired):
        text = self.CONTRACT.read_text()
        assert not re.search(rf"\b{re.escape(retired)}\b", text), (
            f"{retired!r} is a retired name and must not appear in the contract doc"
        )
