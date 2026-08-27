"""Tests for peptide-grouped train/val splitting.

Row-level splitting leaked 41.7% of validation peptides into training on a
representative corpus, and 82.7% of excision validation rows -- the latter by
construction, since excision negatives are the same peptide relabeled with a
mismatched enzyme. A metric computed over that split cannot distinguish
generalization from recall.
"""

import pytest

from presto.data.bulk_ms import BulkMSRecord
from presto.data.loaders import PrestoDataset, peptide_grouped_split_indices


def _dataset(n_peptides=40):
    records = []
    for i in range(n_peptides):
        peptide = f"SIINFEKL{chr(65 + i % 20)}{chr(65 + i // 20)}"
        # Same peptide under two enzymes: the exact pairing that leaked.
        records.append(
            BulkMSRecord(peptide=peptide, machinery="trypsin", protein_id=f"P{i%5}",
                         detectability_label=1.0, excision_label=1.0)
        )
        records.append(
            BulkMSRecord(peptide=peptide, machinery="gluc", protein_id=f"P{i%5}",
                         detectability_label=None, excision_label=0.0)
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
        assert (
            peptide_grouped_split_indices(dataset, 0.2, seed=1)
            != peptide_grouped_split_indices(dataset, 0.2, seed=2)
        )

    def test_val_fraction_is_approximately_honored(self):
        dataset = _dataset(n_peptides=100)
        _, val = peptide_grouped_split_indices(dataset, 0.25, seed=42)
        assert 0.15 <= len(val) / len(dataset) <= 0.35

    def test_single_peptide_dataset_falls_back_without_crashing(self):
        records = [
            BulkMSRecord(peptide="SIINFEKLA", machinery="trypsin",
                         detectability_label=1.0, excision_label=1.0),
            BulkMSRecord(peptide="SIINFEKLA", machinery="gluc",
                         detectability_label=None, excision_label=0.0),
        ]
        dataset = PrestoDataset(bulk_ms_records=records, strict_mhc_resolution=False)
        train, val = peptide_grouped_split_indices(dataset, 0.5, seed=42)
        assert train and val


def test_row_split_leaks_where_grouped_split_does_not():
    """Pins the defect being fixed, so a regression is visible."""
    import random

    dataset = _dataset()
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    cut = int(len(indices) * 0.2)
    row_val, row_train = indices[:cut], indices[cut:]
    row_leak = {dataset[i].peptide for i in row_val} & {
        dataset[i].peptide for i in row_train
    }
    assert row_leak, "expected the row split to leak; fixture may be too small"

    train, val = peptide_grouped_split_indices(dataset, 0.2, seed=42)
    grouped_leak = {dataset[i].peptide for i in val} & {
        dataset[i].peptide for i in train
    }
    assert grouped_leak == set()
