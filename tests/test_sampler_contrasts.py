"""Tests for contrast-group co-batching in the balanced sampler.

The shotgun branch identifies its terms by *comparison*: same peptide across
enzymes isolates excision, same protein and enzyme across outcomes isolates
detectability. Random sampling confounds both with per-protein abundance, so
the sampler has to place complete groups in a batch rather than hope they land
together.
"""

import random

import pytest

from presto.data.bulk_ms import BulkMSRecord
from presto.data.loaders import BalancedMiniBatchSampler, PrestoDataset

BATCH_SIZE = 16


def _paired_records(n_peptides=60, seed=0):
    rng = random.Random(seed)
    records = []
    for i in range(n_peptides):
        peptide = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(9)) + "K"
        protein = f"P{i % 7:04d}"
        records.append(
            BulkMSRecord(
                peptide=peptide, machinery="trypsin", protein_id=protein,
                detectability_label=1.0, excision_label=1.0, flank_c="AAAA",
            )
        )
        records.append(
            BulkMSRecord(
                peptide=peptide, machinery="gluc", protein_id=protein,
                detectability_label=None, excision_label=0.0, flank_c="AAAA",
            )
        )
    return records


@pytest.fixture(scope="module")
def bulk_dataset():
    return PrestoDataset(bulk_ms_records=_paired_records(), strict_mhc_resolution=False)


def _protease_pairs(dataset, batch):
    machineries = {}
    for idx in batch:
        sample = dataset[idx]
        machineries.setdefault(sample.peptide, set()).add(sample.machinery)
    return sum(1 for values in machineries.values() if len(values) > 1)


class TestContrastGroups:
    def test_groups_are_discovered(self, bulk_dataset):
        sampler = BalancedMiniBatchSampler(bulk_dataset, batch_size=BATCH_SIZE, seed=0)
        assert sampler._contrast_groups, "no contrast groups built for a paired corpus"

    def test_every_batch_contains_a_protease_contrast(self, bulk_dataset):
        sampler = BalancedMiniBatchSampler(bulk_dataset, batch_size=BATCH_SIZE, seed=0)
        batches = list(sampler)
        assert batches
        pairs = [_protease_pairs(bulk_dataset, batch) for batch in batches]
        assert all(count >= 1 for count in pairs), (
            f"batches without a same-peptide/different-enzyme pair: {pairs}"
        )

    def test_machinery_is_a_sampling_stratum(self, bulk_dataset):
        sampler = BalancedMiniBatchSampler(bulk_dataset, batch_size=BATCH_SIZE, seed=0)
        machineries = set(sampler._machinery_by_index.values())
        assert {"trypsin", "gluc"} <= machineries

    def test_branch_is_a_sampling_stratum(self, bulk_dataset):
        sampler = BalancedMiniBatchSampler(bulk_dataset, batch_size=BATCH_SIZE, seed=0)
        assert set(sampler._branch_by_index.values()) == {"shotgun"}

    def test_mhc_only_corpus_builds_no_contrast_groups(self):
        """The MHC-only path must behave exactly as it did before."""
        from presto.data.loaders import BindingRecord

        mhc_seq = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPR"
        records = [
            BindingRecord(
                peptide=f"SIINFEKL{aa}", mhc_allele="HLA-A*02:01", value=25.0,
                measurement_type="IC50", mhc_sequence=mhc_seq, mhc_class="I",
            )
            for aa in "ACDEFGHIKLMNPQRSTVWY"
        ]
        dataset = PrestoDataset(binding_records=records, strict_mhc_resolution=False)
        sampler = BalancedMiniBatchSampler(dataset, batch_size=8, seed=0)
        assert sampler._contrast_groups == []
        assert sampler._contrast_quota() == 0
