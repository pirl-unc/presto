"""Tests for peptide-grouped train/val splitting.

Row-level splitting leaked 41.7% of validation peptides into training on a
representative corpus, and 82.7% of excision validation rows -- the latter by
construction, since excision negatives are the same peptide relabeled with a
mismatched enzyme. A metric computed over that split cannot distinguish
generalization from recall.
"""

from presto.data.bulk_ms import BulkMSRecord
from presto.data.loaders import (
    PrestoDataset,
    peptide_grouped_split_indices,
    peptide_grouped_three_way_split_indices,
)


def _dataset(n_peptides=40):
    records = []
    for i in range(n_peptides):
        peptide = f"SIINFEKL{chr(65 + i % 20)}{chr(65 + i // 20)}"
        # Same peptide under two enzymes: the exact pairing that leaked.
        records.append(
            BulkMSRecord(
                peptide=peptide,
                machinery="trypsin",
                protein_id=f"P{i % 5}",
                detectability_label=1.0,
                excision_label=1.0,
            )
        )
        records.append(
            BulkMSRecord(
                peptide=peptide,
                machinery="gluc",
                protein_id=f"P{i % 5}",
                detectability_label=None,
                excision_label=0.0,
            )
        )
    return PrestoDataset(bulk_ms_records=records, strict_mhc_resolution=False)


class TestPeptideGroupedSplit:
    def test_no_peptide_appears_on_both_sides(self):
        dataset = _dataset()
        train, val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
        train_peptides = {dataset[i].peptide for i in train}
        val_peptides = {dataset[i].peptide for i in val}
        assert train_peptides & val_peptides == set()

    def test_both_sides_are_non_empty(self):
        dataset = _dataset()
        train, val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
        assert train and val

    def test_every_row_lands_exactly_once(self):
        dataset = _dataset()
        train, val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
        assert sorted(train + val) == list(range(len(dataset)))

    def test_enzyme_pairs_stay_together(self):
        """The trypsin/GluC rows for one peptide must not be split apart."""
        dataset = _dataset()
        train, val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
        val_set = set(val)
        by_peptide = {}
        for index in range(len(dataset)):
            by_peptide.setdefault(dataset[index].peptide, []).append(index)
        for indices in by_peptide.values():
            sides = {index in val_set for index in indices}
            assert len(sides) == 1, "a peptide's rows were split across sides"

    def test_split_is_deterministic(self):
        dataset = _dataset()
        first = peptide_grouped_split_indices(dataset, 0.2, seed=7)
        second = peptide_grouped_split_indices(dataset, 0.2, seed=7)
        assert first == second

    def test_different_seeds_give_different_splits(self):
        dataset = _dataset()
        assert peptide_grouped_split_indices(dataset, 0.2, seed=1) != peptide_grouped_split_indices(
            dataset, 0.2, seed=2
        )

    def test_val_fraction_is_approximately_honored(self):
        dataset = _dataset(n_peptides=100)
        _, val = peptide_grouped_split_indices(dataset, 0.25, seed=42)
        assert 0.15 <= len(val) / len(dataset) <= 0.35

    def test_single_peptide_dataset_falls_back_without_crashing(self):
        records = [
            BulkMSRecord(
                peptide="SIINFEKLA",
                machinery="trypsin",
                detectability_label=1.0,
                excision_label=1.0,
            ),
            BulkMSRecord(
                peptide="SIINFEKLA", machinery="gluc", detectability_label=None, excision_label=0.0
            ),
        ]
        dataset = PrestoDataset(bulk_ms_records=records, strict_mhc_resolution=False)
        train, val = peptide_grouped_split_indices(dataset, 0.5, seed=42)
        assert train and val


class TestPeptideGroupedThreeWaySplit:
    def test_partitions_are_complete_and_peptide_disjoint(self):
        dataset = _dataset(n_peptides=100)
        train, val, test = peptide_grouped_three_way_split_indices(
            dataset, val_fraction=0.1, test_fraction=0.1, seed=42
        )

        assert train and val and test
        assert sorted(train + val + test) == list(range(len(dataset)))
        peptide_sets = [
            {dataset[index].peptide for index in indices} for indices in (train, val, test)
        ]
        assert not (peptide_sets[0] & peptide_sets[1])
        assert not (peptide_sets[0] & peptide_sets[2])
        assert not (peptide_sets[1] & peptide_sets[2])

    def test_is_deterministic_and_approximately_honors_fractions(self):
        dataset = _dataset(n_peptides=100)
        first = peptide_grouped_three_way_split_indices(dataset, 0.1, 0.2, seed=7)
        second = peptide_grouped_three_way_split_indices(dataset, 0.1, 0.2, seed=7)
        assert first == second
        _, val, test = first
        assert 0.05 <= len(val) / len(dataset) <= 0.15
        assert 0.15 <= len(test) / len(dataset) <= 0.25

    def test_zero_test_fraction_preserves_two_way_split(self):
        dataset = _dataset()
        expected_train, expected_val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
        train, val, test = peptide_grouped_three_way_split_indices(dataset, 0.2, 0.0, seed=42)
        assert (train, val, test) == (expected_train, expected_val, [])


def test_row_split_leaks_where_grouped_split_does_not():
    """Pins the defect being fixed, so a regression is visible."""
    import random

    dataset = _dataset()
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    cut = int(len(indices) * 0.2)
    row_val, row_train = indices[:cut], indices[cut:]
    row_leak = {dataset[i].peptide for i in row_val} & {dataset[i].peptide for i in row_train}
    assert row_leak, "expected the row split to leak; fixture may be too small"

    train, val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
    grouped_leak = {dataset[i].peptide for i in val} & {dataset[i].peptide for i in train}
    assert grouped_leak == set()


class TestTheTestSplitIsIndependentOfTuning:
    """A held-out test set drawn from peptides tuning already saw is not held out.

    Both helpers shuffle the same sorted peptide list. They used the same
    `seed + 53` stream, and the two-way helper fills *validation* from the
    front while the three-way fills *test* from the front -- so at a given
    seed the three-way test peptides were exactly the peptides an earlier
    `--test-frac 0` run had selected models against. Every test metric would
    have read as clean while being measured on tuning data.
    """

    def test_test_peptides_are_not_the_same_seeds_validation_peptides(self):
        dataset = _dataset()
        _, tuning_val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
        _, _, test = peptide_grouped_three_way_split_indices(dataset, 0.2, 0.1, seed=42)
        tuning_val_peptides = {dataset[i].peptide for i in tuning_val}
        test_peptides = {dataset[i].peptide for i in test}
        assert test_peptides, "sanity: a test split was produced"
        assert not test_peptides.issubset(tuning_val_peptides), (
            "every held-out test peptide had already been used for model selection at this seed"
        )

    def test_the_split_is_still_deterministic(self):
        dataset = _dataset()
        first = peptide_grouped_three_way_split_indices(dataset, 0.2, 0.1, seed=42)
        second = peptide_grouped_three_way_split_indices(dataset, 0.2, 0.1, seed=42)
        assert first == second


class TestDegenerateFractionsStillSplit:
    """The three-way helper must not reject what the two-way helper accepts."""

    def test_zero_validation_delegates_rather_than_raising(self):
        """`--val-frac 0` yielded a one-row validation split for years.

        The two-way helper clamps with `max(1, ...)`. Validating the fractions
        before delegating turned that working command line into a crash -- and
        one raised after the entire corpus had been loaded.
        """
        dataset = _dataset()
        train, val, test = peptide_grouped_three_way_split_indices(dataset, 0.0, 0.0, seed=42)
        assert test == []
        assert len(val) >= 1
        assert len(train) + len(val) == len(dataset)

    def test_a_real_three_way_split_still_validates_its_fractions(self):
        import pytest

        dataset = _dataset()
        with pytest.raises(ValueError, match="val_fraction must be > 0"):
            peptide_grouped_three_way_split_indices(dataset, 0.0, 0.1, seed=42)
        with pytest.raises(ValueError, match="must be < 1"):
            peptide_grouped_three_way_split_indices(dataset, 0.8, 0.2, seed=42)
