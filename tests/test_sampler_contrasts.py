"""Co-batch genuine comparisons without claiming biological identifiability."""

import random

import pytest

from presto.data.bulk_ms import BulkMSRecord
from presto.data.collate import PrestoSample
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
                peptide=peptide,
                machinery="trypsin",
                protein_id=protein,
                detectability_label=1.0,
                excision_label=1.0,
                flank_c="AAAA",
            )
        )
        records.append(
            BulkMSRecord(
                peptide=peptide,
                machinery="gluc",
                protein_id=protein,
                detectability_label=None,
                excision_label=0.0,
                flank_c="AAAA",
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
                peptide=f"SIINFEKL{aa}",
                mhc_allele="HLA-A*02:01",
                value=25.0,
                measurement_type="IC50",
                mhc_sequence=mhc_seq,
                mhc_class="I",
            )
            for aa in "ACDEFGHIKLMNPQRSTVWY"
        ]
        dataset = PrestoDataset(binding_records=records, strict_mhc_resolution=False)
        sampler = BalancedMiniBatchSampler(dataset, batch_size=8, seed=0)
        assert sampler._contrast_groups == []
        assert sampler._contrast_quota() == 0


def scoped_sample(label, **overrides):
    fields = dict(
        peptide="ACDEFGHIK",
        machinery="trypsin",
        source_protein="P00001",
        assay_group="bulk_ms",
        ms_detectability_label=label,
        label_bucket="positive",  # Generic/excision labels must not select this comparison.
        source_bundle_id="bundle-a",
        source_search_space_id="search-a",
        source_protocol_id="protocol-a",
        source_reference_sha256="reference-a",
        source_label_evidence="By MS/MS" if label else "not observed in scope",
    )
    fields.update(overrides)
    return PrestoSample(**fields)


def test_capped_group_and_actual_batches_keep_both_detectability_outcomes():
    samples = [scoped_sample(1.0) for _ in range(8)] + [scoped_sample(0.0)]
    sampler = BalancedMiniBatchSampler(samples, batch_size=2, seed=3)
    assert sampler._contrast_groups
    for group in sampler._contrast_groups:
        assert len(group) <= sampler.CONTRAST_MAX_GROUP
        assert {samples[i].ms_detectability_label for i in group} == {0.0, 1.0}
    for batch in sampler:
        assert len(batch) == len(set(batch)) == 2
        assert {samples[i].ms_detectability_label for i in batch} == {0.0, 1.0}


@pytest.mark.parametrize(
    "field",
    ["source_bundle_id", "source_search_space_id", "source_protocol_id", "source_reference_sha256"],
)
def test_detectability_groups_do_not_cross_captured_scopes(field):
    samples = [
        scoped_sample(1.0),
        scoped_sample(0.0, label_bucket="negative", **{field: "different"}),
    ]
    assert BalancedMiniBatchSampler(samples, batch_size=2)._contrast_groups == []


@pytest.mark.parametrize("bad_label", [None, float("nan"), float("inf"), -1, 0.5, 2])
def test_detectability_groups_require_actual_binary_targets(bad_label):
    samples = [scoped_sample(1.0), scoped_sample(bad_label, label_bucket="negative")]
    assert BalancedMiniBatchSampler(samples, batch_size=2)._contrast_groups == []


@pytest.mark.parametrize("field", ["source_protein", "source_protocol_id"])
def test_unknown_parent_or_partial_scope_does_not_create_comparison(field):
    samples = [scoped_sample(1.0), scoped_sample(0.0, label_bucket="negative", **{field: ""})]
    assert BalancedMiniBatchSampler(samples, batch_size=2)._contrast_groups == []


def test_excision_only_rows_do_not_create_detectability_groups():
    samples = [
        scoped_sample(None, excision_label=1.0, label_bucket="positive"),
        scoped_sample(None, excision_label=0.0, label_bucket="negative"),
    ]
    assert BalancedMiniBatchSampler(samples, batch_size=2)._contrast_groups == []


def test_protease_variation_survives_both_group_and_batch_caps():
    records = [BulkMSRecord("ACDEFGHIK", machinery="trypsin", excision_label=1.0) for _ in range(8)]
    records.append(BulkMSRecord("ACDEFGHIK", machinery="gluc", excision_label=0.0))
    dataset = PrestoDataset(bulk_ms_records=records)
    first = list(BalancedMiniBatchSampler(dataset, batch_size=2, seed=7))
    assert first == list(BalancedMiniBatchSampler(dataset, batch_size=2, seed=7))
    assert all({dataset[i].machinery for i in batch} == {"trypsin", "gluc"} for batch in first)


def test_detectability_only_rows_do_not_create_protease_comparisons():
    samples = [scoped_sample(1.0), scoped_sample(0.0, machinery="gluc")]
    assert BalancedMiniBatchSampler(samples, batch_size=2)._contrast_groups == []


@pytest.mark.parametrize("batch_size", [1, 2, 3, 8, 16])
def test_contrast_seeds_consume_task_quota_and_never_overfill_batches(batch_size):
    samples = [scoped_sample(float(i % 2)) for i in range(20)]
    samples += [
        PrestoSample(peptide="ACDEFGHIK", bind_value=100, assay_group="binding_affinity")
        for _ in range(20)
    ]
    sampler = BalancedMiniBatchSampler(samples, batch_size=batch_size, seed=11)
    for _ in range(3):
        for batch in sampler:
            assert len(batch) == len(set(batch)) == batch_size
            if batch_size >= 2:
                assert {samples[i].assay_group for i in batch} == {"bulk_ms", "binding_affinity"}


@pytest.mark.parametrize("separate_source", [False, True])
def test_legacy_comparisons_are_explicitly_separated_by_source(separate_source):
    records = [
        BulkMSRecord("ACDEFGHIK", detectability_label=1.0, protein_id="P00001"),
        BulkMSRecord(
            "ACDEFGHIR",
            detectability_label=0.0,
            protein_id="P00001",
            observed=False,
            source="other_legacy" if separate_source else "bulk_proteomics",
        ),
    ]
    dataset = PrestoDataset(bulk_ms_records=records)
    sampler = BalancedMiniBatchSampler(dataset, batch_size=2)
    assert bool(sampler._contrast_groups) is not separate_source
