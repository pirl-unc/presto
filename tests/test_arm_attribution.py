"""An unresolved arm is not a control (presto#15).

hitlist emits `apm_perturbed=False` and an empty `apm_genes_perturbed` for two
different situations: a genuine control, and a row whose experimental arm it
could not attribute (pirl-unc/hitlist#392). Folding the second into the control
class asserts something unknown to be false.

Inside the 24 APM-perturbed studies that is 221,930 observations. Among the
*attributed* rows in those same studies 32.5% are perturbed, so a comparable
share of that block is on the wrong side of exactly the KO/WT contrast the APM
axis exists to learn -- not merely noisier, partially cancelled.

`sample_attribution` distinguishes them, and doubles as an evidence tier.
"""

import pytest

from presto.data.vocab import (  # noqa: E402
    APM_PERTURBATIONS,
    ATTRIBUTION_TIERS,
    apm_group_for_genes,
    apm_group_for_row,
    attribution_is_per_peptide,
)


class TestUnknownIsDistinctFromNone:
    def test_unknown_is_in_the_vocabulary(self):
        assert "unknown" in APM_PERTURBATIONS

    def test_unknown_was_appended_so_existing_indices_are_stable(self):
        """Position is the contract; only the tail may grow."""
        assert APM_PERTURBATIONS[-1] == "unknown"
        assert APM_PERTURBATIONS[:7] == [
            "none",
            "peptide_supply",
            "n_term_trimming",
            "loading_complex",
            "mhc_null",
            "class_ii_loading",
            "other",
        ]

    def test_unattributed_row_is_unknown_not_none(self):
        """The whole point: an empty gene list with no arm is not a control."""
        assert apm_group_for_row("", "") == "unknown"
        assert apm_group_for_row(None, "") == "unknown"

    def test_attributed_control_is_still_none(self):
        """A resolved arm with no perturbation genuinely is a control."""
        assert apm_group_for_row("", "allele_exact") == "none"

    @pytest.mark.parametrize(
        "genes,expected",
        [("TAP1", "peptide_supply"), ("ERAP1", "n_term_trimming"), ("B2M", "mhc_null")],
    )
    def test_attributed_perturbation_is_unchanged(self, genes, expected):
        assert apm_group_for_row(genes, "allele_exact") == expected

    def test_a_source_without_arm_columns_keeps_the_old_behaviour(self):
        """`None` means "this source has no arm information at all", which is
        different from "hitlist looked and could not tell"."""
        assert apm_group_for_row("", None) == apm_group_for_genes("")
        assert apm_group_for_row("", None) == "none"

    def test_a_perturbed_row_with_no_attribution_still_reports_its_genes(self):
        """Genes present means the arm *was* identified, whatever the column
        says -- do not throw away a positive label."""
        assert apm_group_for_row("TAP1", "") == "unknown"


class TestAttributionTiers:
    def test_tiers_are_ordered_strongest_first(self):
        assert ATTRIBUTION_TIERS[0] == "allele_exact"
        assert ATTRIBUTION_TIERS[-1] == "pmid_ambiguous"

    @pytest.mark.parametrize("tier", ["allele_exact", "elution_conditions"])
    def test_per_peptide_evidence(self, tier):
        assert attribution_is_per_peptide(tier)

    @pytest.mark.parametrize("tier", ["class_pool", "pmid_ambiguous", "", None])
    def test_pool_level_evidence_is_not_per_peptide(self, tier):
        """These resolve to a pool or a deposit, so the arm label may not be
        this peptide's."""
        assert not attribution_is_per_peptide(tier)

    def test_case_and_whitespace_are_tolerated(self):
        assert attribution_is_per_peptide("  Allele_Exact  ")


class TestTheColumnsAreRequested:
    """A feature built from a column nobody asked for is a constant."""

    @pytest.mark.parametrize("column", ["sample_attribution", "is_control_arm", "sample_label"])
    def test_arm_column_is_in_the_contract(self, column):
        from presto.data.hitlist_source import MS_COLUMNS

        assert column in MS_COLUMNS
