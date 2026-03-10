"""Tests for focused binding probe utilities."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from presto.data import BindingRecord, PrestoDataset
from presto.data.collate import PrestoCollator, PrestoSample
from presto.models.presto import Presto
from presto.scripts.focused_binding_probe import (
    CombinedSampleDataset,
    DatasetContract,
    PreparedBindingState,
    StrictAlleleBalancedBatchSampler,
    _affinity_only_loss,
    _augment_train_records_only,
    _balance_alleles,
    _contract_cache_key,
    _collect_binding_contrastive_pairs,
    _collect_binding_peptide_ranking_pairs,
    _create_focused_train_loader,
    _filter_compatible_state_dict,
    _filter_shared_peptides_only,
    _keep_binding_qualifier,
    _load_binding_records_from_merged_tsv,
    _parse_synthetic_modes,
    _require_target_allele_coverage,
    _resolve_batch_synthetic_fraction,
    _split_records_by_peptide,
    _summarize_binding_records,
    _load_prepared_binding_state_from_cache,
    _write_prepared_binding_state_to_cache,
)


class _TinyDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _rec(
    allele: str,
    peptide: str,
    *,
    measurement_type: str = "IC50",
    value: float = 100.0,
    mhc_class: str = "I",
    qualifier: int = 0,
    mhc_sequence: str | None = None,
    source: str = "iedb",
):
    return SimpleNamespace(
        mhc_allele=allele,
        peptide=peptide,
        measurement_type=measurement_type,
        value=value,
        mhc_class=mhc_class,
        qualifier=qualifier,
        mhc_sequence=mhc_sequence,
        source=source,
    )


def test_balance_alleles_auto_balance_preserves_shared_peptides():
    alleles = ["A*02:01", "A*24:02"]
    records = [
        _rec("A*02:01", "SHARED1", value=10.0),
        _rec("A*24:02", "SHARED1", value=1000.0),
        _rec("A*02:01", "SHARED2", value=20.0),
        _rec("A*24:02", "SHARED2", value=2000.0),
    ]
    records.extend(_rec("A*02:01", f"A02ONLY{i}", value=50.0 + i) for i in range(8))
    records.extend(_rec("A*24:02", f"A24ONLY{i}", value=50.0 + i) for i in range(2))

    balanced, stats = _balance_alleles(records, alleles, max_per_allele=0, rng_seed=42)

    assert stats["cap"] == 4
    assert stats["counts_after"] == {"A*02:01": 4, "A*24:02": 4}
    assert stats["shared_peptides_before"] == 2
    assert stats["shared_peptides_selected"] == 2
    shared_balanced = {
        (rec.mhc_allele, rec.peptide)
        for rec in balanced
        if rec.peptide.startswith("SHARED")
    }
    assert ("A*02:01", "SHARED1") in shared_balanced
    assert ("A*24:02", "SHARED1") in shared_balanced
    assert ("A*02:01", "SHARED2") in shared_balanced
    assert ("A*24:02", "SHARED2") in shared_balanced


def test_balance_alleles_preserves_non_target():
    alleles = ["A*02:01"]
    records = [_rec("A*02:01", f"P{i}") for i in range(10)] + [
        _rec("B*07:02", f"Q{i}") for i in range(3)
    ]

    balanced, stats = _balance_alleles(records, alleles, max_per_allele=5, rng_seed=7)

    assert stats["counts_after"]["A*02:01"] == 5
    non_target = [r for r in balanced if r.mhc_allele == "B*07:02"]
    assert len(non_target) == 3


def test_split_records_by_peptide_keeps_families_together():
    records = [
        _rec("A*02:01", "PEP1"),
        _rec("A*24:02", "PEP1"),
        _rec("A*02:01", "PEP2"),
        _rec("A*24:02", "PEP2"),
        _rec("A*02:01", "PEP3"),
        _rec("A*24:02", "PEP4"),
    ]

    train, val, stats = _split_records_by_peptide(
        records,
        val_fraction=0.4,
        seed=13,
        alleles=["A*02:01", "A*24:02"],
    )

    train_peptides = {rec.peptide for rec in train}
    val_peptides = {rec.peptide for rec in val}
    assert train_peptides.isdisjoint(val_peptides)
    assert train_peptides | val_peptides == {"PEP1", "PEP2", "PEP3", "PEP4"}
    assert stats["shared_peptides_total"] == 2
    assert stats["shared_peptides_train"] + stats["shared_peptides_val"] == 2


def test_split_records_by_peptide_has_non_empty_train_and_val():
    records = [_rec("A*02:01", f"P{i}") for i in range(4)]
    train, val, stats = _split_records_by_peptide(records, val_fraction=0.25, seed=3)

    assert train
    assert val
    assert stats["train_rows"] + stats["val_rows"] == len(records)


def test_split_records_by_peptide_uses_all_alleles_when_no_panel_supplied():
    records = [
        _rec("A*02:01", "PEP1"),
        _rec("A*24:02", "PEP1"),
        _rec("B*07:02", "PEP2"),
        _rec("B*07:02", "PEP2B"),
        _rec("B*44:02", "PEP3"),
        _rec("B*44:02", "PEP3B"),
        _rec("A*03:01", "PEP4"),
        _rec("A*03:01", "PEP4B"),
        _rec("A*11:01", "PEP5"),
        _rec("A*11:01", "PEP5B"),
    ]

    train, val, stats = _split_records_by_peptide(records, val_fraction=0.4, seed=11)

    assert train
    assert val
    assert stats["train_rows"] + stats["val_rows"] == len(records)
    assert stats["val_peptides"] >= 4
    assert stats["shared_peptides_total"] == 1
    assert stats["shared_peptides_train"] + stats["shared_peptides_val"] == 1


def test_require_target_allele_coverage_raises_on_missing_target():
    records = [_rec("A*02:01", "P1")]
    with pytest.raises(RuntimeError, match="A\\*24:02"):
        _require_target_allele_coverage(records, ["A*02:01", "A*24:02"])


def test_augment_train_records_only_leaves_validation_real_only():
    train_records = [_rec("A*02:01", "SLLQHLIGL", value=20.0)]
    val_records = [_rec("A*24:02", "TYRPEPTID", value=2000.0)]

    train_aug, val_out, stats = _augment_train_records_only(
        train_records=train_records,
        val_records=val_records,
        mhc_sequences={"A*02:01": "A" * 181, "A*24:02": "B" * 181},
        negative_ratio=1.0,
        seed=5,
        class_i_anchor_strategy="property_opposite",
    )

    assert len(train_aug) > len(train_records)
    assert val_out == val_records
    assert stats["train"]["added"] > 0
    assert stats["val"] == {"added": 0, "reason": "validation_real_only"}


def test_augment_train_records_only_changes_synthetics_when_seed_changes():
    train_records = [_rec("A*02:01", "SLLQHLIGL", value=20.0)]
    val_records = [_rec("A*24:02", "TYRPEPTID", value=2000.0)]

    train_aug_a, _, _ = _augment_train_records_only(
        train_records=train_records,
        val_records=val_records,
        mhc_sequences={"A*02:01": "A" * 181, "A*24:02": "C" * 181},
        negative_ratio=1.0,
        seed=5,
        class_i_anchor_strategy="property_opposite",
        modes=("peptide_scramble",),
    )
    train_aug_b, _, _ = _augment_train_records_only(
        train_records=train_records,
        val_records=val_records,
        mhc_sequences={"A*02:01": "A" * 181, "A*24:02": "C" * 181},
        negative_ratio=1.0,
        seed=6,
        class_i_anchor_strategy="property_opposite",
        modes=("peptide_scramble",),
    )

    synth_a = [rec.peptide for rec in train_aug_a[1:]]
    synth_b = [rec.peptide for rec in train_aug_b[1:]]
    assert synth_a != synth_b


def test_summarize_binding_records_handles_missing_values():
    records = [
        SimpleNamespace(
            mhc_allele="A*02:01",
            peptide="P1",
            measurement_type="IC50",
            value=None,
            qualifier=None,
        ),
        SimpleNamespace(
            mhc_allele="A*02:01",
            peptide="P2",
            measurement_type="IC50",
            value=40.0,
            qualifier=0,
        ),
    ]

    summary = _summarize_binding_records(records)

    assert summary["rows"] == 2
    assert summary["rows_by_allele"]["A*02:01"] == 2
    assert summary["fraction_le_500_by_allele"]["A*02:01"] == 0.5


def test_collect_binding_contrastive_pairs_finds_rankable_same_peptide_pairs():
    batch = SimpleNamespace(
        bind_mask=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        bind_target_log10=torch.log10(torch.tensor([10.0, 5000.0, 100.0], dtype=torch.float32)),
        bind_qual=torch.tensor([0, 0, 0], dtype=torch.float32),
        same_peptide_diff_allele_pairs=torch.tensor([[0, 1]], dtype=torch.long),
    )

    pairs, metrics = _collect_binding_contrastive_pairs(
        batch,
        target_gap_min=0.3,
        max_pairs=8,
    )

    assert len(pairs) == 1
    gap, stronger_idx, weaker_idx = pairs[0]
    assert gap > 0.3
    assert (stronger_idx, weaker_idx) == (0, 1)
    assert metrics["out_binding_same_peptide_pairs"] == 1.0
    assert metrics["out_binding_same_peptide_rankable_pairs"] == 1.0


def test_create_focused_train_loader_passes_runtime_options_to_global_balance_path():
    dataset = _TinyDataset(
        [
            PrestoSample(peptide="AAAAAAA", mhc_a="A" * 80, mhc_b="B" * 80, mhc_class="I", primary_allele="HLA-A*02:01"),
            PrestoSample(peptide="BBBBBBB", mhc_a="A" * 80, mhc_b="B" * 80, mhc_class="I", primary_allele="HLA-A*24:02"),
        ]
    )
    loader = _create_focused_train_loader(
        dataset,
        batch_size=2,
        collator=PrestoCollator(),
        balanced=True,
        seed=1,
        alleles=["HLA-A*02:01", "HLA-A*24:02"],
        force_global_balance=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.prefetch_factor == 4


def test_create_focused_train_loader_passes_runtime_options_to_strict_sampler_path():
    dataset = _TinyDataset(
        [
            PrestoSample(peptide="AAAAAAA", mhc_a="A" * 80, mhc_b="B" * 80, mhc_class="I", primary_allele="HLA-A*02:01"),
            PrestoSample(peptide="CCCCCCC", mhc_a="A" * 80, mhc_b="B" * 80, mhc_class="I", primary_allele="HLA-A*02:01"),
            PrestoSample(peptide="BBBBBBB", mhc_a="A" * 80, mhc_b="B" * 80, mhc_class="I", primary_allele="HLA-A*24:02"),
            PrestoSample(peptide="DDDDDDD", mhc_a="A" * 80, mhc_b="B" * 80, mhc_class="I", primary_allele="HLA-A*24:02"),
        ]
    )
    loader = _create_focused_train_loader(
        dataset,
        batch_size=2,
        collator=PrestoCollator(),
        balanced=True,
        seed=1,
        alleles=["HLA-A*02:01", "HLA-A*24:02"],
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.prefetch_factor == 2


def test_collect_binding_peptide_ranking_pairs_finds_rankable_same_allele_pairs():
    batch = SimpleNamespace(
        bind_mask=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        bind_target_log10=torch.log10(torch.tensor([15.0, 5000.0, 40.0], dtype=torch.float32)),
        bind_qual=torch.tensor([0, 0, 0], dtype=torch.float32),
        same_allele_diff_peptide_pairs=torch.tensor([[0, 1]], dtype=torch.long),
    )

    pairs, metrics = _collect_binding_peptide_ranking_pairs(
        batch,
        target_gap_min=0.5,
        max_pairs=8,
    )

    assert len(pairs) == 1
    gap, stronger_idx, weaker_idx = pairs[0]
    assert gap > 0.5
    assert (stronger_idx, weaker_idx) == (0, 1)
    assert metrics["out_binding_same_allele_pairs"] == 1.0
    assert metrics["out_binding_same_allele_rankable_pairs"] == 1.0


def test_collator_emits_fixed_metadata_and_pair_indices():
    collator = PrestoCollator()
    batch = collator(
        [
            PrestoSample(
                peptide="SLLQHLIGL",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                bind_value=10.0,
                bind_qual=0,
                primary_allele="HLA-A*02:01",
                dataset_index=11,
                peptide_id=3,
                allele_id=7,
                bind_target_log10=1.0,
            ),
            PrestoSample(
                peptide="SLLQHLIGL",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                bind_value=5000.0,
                bind_qual=0,
                primary_allele="HLA-A*24:02",
                dataset_index=12,
                peptide_id=3,
                allele_id=8,
                bind_target_log10=torch.log10(torch.tensor(5000.0)).item(),
            ),
            PrestoSample(
                peptide="FLRYLLFGI",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                bind_value=40.0,
                bind_qual=0,
                primary_allele="HLA-A*02:01",
                dataset_index=13,
                peptide_id=4,
                allele_id=7,
                bind_target_log10=torch.log10(torch.tensor(40.0)).item(),
            ),
        ]
    )

    assert batch.dataset_index.tolist() == [11, 12, 13]
    assert batch.peptide_id.tolist() == [3, 3, 4]
    assert batch.allele_id.tolist() == [7, 8, 7]
    assert batch.same_peptide_diff_allele_pairs.tolist() == [[0, 1]]
    assert batch.same_allele_diff_peptide_pairs.tolist() == [[0, 2]]


def test_combined_dataset_preserves_real_and_synthetic_dataset_index_offsets():
    mhc_sequences = {
        "HLA-A*02:01": "A" * 120,
        "HLA-A*24:02": "C" * 120,
    }
    real_dataset = PrestoDataset(
        binding_records=[
            BindingRecord(peptide="SLLQHLIGL", mhc_allele="HLA-A*02:01", value=10.0, qualifier=0),
            BindingRecord(peptide="FLRYLLFGI", mhc_allele="HLA-A*24:02", value=1000.0, qualifier=0),
        ],
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    synth_dataset = PrestoDataset(
        binding_records=[
            BindingRecord(
                peptide="XXXXXXXXX",
                mhc_allele="HLA-A*02:01",
                value=50000.0,
                qualifier=0,
                source="synthetic_negative_peptide_random",
            ),
        ],
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
        dataset_index_offset=len(real_dataset),
    )
    combined = CombinedSampleDataset(real_dataset, synth_dataset)

    assert [sample.dataset_index for sample in combined.samples] == [0, 1, 2]


def test_affinity_only_loss_assay_heads_only_uses_headwise_targets():
    collator = PrestoCollator()
    batch = collator(
        [
            PrestoSample(
                peptide="SLLQHLIGL",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                bind_value=10.0,
                bind_measurement_type="dissociation constant KD",
                bind_qual=0,
                primary_allele="HLA-A*02:01",
            ),
            PrestoSample(
                peptide="FLRYLLFGI",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                bind_value=50.0,
                bind_measurement_type="half maximal inhibitory concentration (IC50)",
                bind_qual=0,
                primary_allele="HLA-A*24:02",
            ),
            PrestoSample(
                peptide="NFLIKFLLI",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                bind_value=100.0,
                bind_measurement_type="half maximal effective concentration (EC50)",
                bind_qual=1,
                primary_allele="HLA-A*03:01",
            ),
        ]
    )
    model = Presto(d_model=32, n_layers=1, n_heads=2)

    loss, metrics = _affinity_only_loss(
        model,
        batch,
        device="cpu",
        regularization={
            "binding_contrastive_weight": 0.0,
            "binding_peptide_contrastive_weight": 0.0,
        },
        loss_mode="assay_heads_only",
    )

    assert torch.isfinite(loss)
    assert metrics["support_binding_kd"] == 1.0
    assert metrics["support_binding_ic50"] == 1.0
    assert metrics["support_binding_ec50"] == 1.0
    assert "support_binding" not in metrics


def test_filter_shared_peptides_only_keeps_only_cross_allele_families():
    records = [
        _rec("A*02:01", "SHARED1"),
        _rec("A*24:02", "SHARED1"),
        _rec("A*02:01", "A02ONLY"),
        _rec("A*24:02", "A24ONLY"),
    ]

    filtered, stats = _filter_shared_peptides_only(records, ["A*02:01", "A*24:02"])

    assert stats["shared_peptides"] == 1
    assert stats["rows_after"] == 2
    assert {(r.mhc_allele, r.peptide) for r in filtered} == {
        ("A*02:01", "SHARED1"),
        ("A*24:02", "SHARED1"),
    }


def test_keep_binding_qualifier_exact_only_keeps_qualifier_zero():
    assert _keep_binding_qualifier(0, "exact") is True
    assert _keep_binding_qualifier(1, "exact") is False
    assert _keep_binding_qualifier(-1, "exact") is False
    assert _keep_binding_qualifier(None, "exact") is True


def test_collate_binding_context_maps_known_assay_fields():
    collator = PrestoCollator()
    batch = collator(
        [
            PrestoSample(
                peptide="SLLQHLIGL",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                binding_assay_type="half maximal inhibitory concentration (IC50)",
                binding_assay_method="purified MHC/competitive/radioactivity",
            ),
            PrestoSample(
                peptide="FLRYLLFGI",
                mhc_a="A" * 91,
                mhc_b="C" * 93,
                mhc_class="I",
                binding_assay_type="dissociation constant KD (~IC50)",
                binding_assay_method="purified MHC/direct/fluorescence",
            ),
        ]
    )

    assert batch.binding_context["assay_type_idx"].tolist() == [4, 2]
    assert batch.binding_context["assay_method_idx"].tolist() == [1, 2]


def test_load_binding_records_from_merged_tsv_retains_binding_metadata(tmp_path):
    merged = tmp_path / "merged.tsv"
    merged.write_text(
        "\t".join(
            [
                "peptide",
                "mhc_allele",
                "mhc_class",
                "source",
                "record_type",
                "value",
                "value_type",
                "qualifier",
                "response",
                "assay_type",
                "assay_method",
                "apc_name",
                "effector_culture_condition",
                "apc_culture_condition",
                "in_vitro_process_type",
                "in_vitro_responder_cell",
                "in_vitro_stimulator_cell",
                "species",
                "antigen_species",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "SLLQHLIGL",
                "HLA-A*02:01",
                "I",
                "iedb",
                "binding",
                "25",
                "half maximal inhibitory concentration (IC50)",
                "0",
                "",
                "half maximal inhibitory concentration (IC50)",
                "purified MHC/competitive/radioactivity",
                "",
                "",
                "",
                "",
                "",
                "",
                "Homo sapiens",
                "SARS-CoV-2",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "PEPTIDEII",
                "HLA-DRB1*04:01",
                "II",
                "iedb",
                "binding",
                "50",
                "dissociation constant KD",
                "0",
                "",
                "dissociation constant KD",
                "purified MHC/direct/fluorescence",
                "",
                "",
                "",
                "",
                "",
                "",
                "Homo sapiens",
                "SARS-CoV-2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = _load_binding_records_from_merged_tsv(
        merged,
        mhc_class_filter="I",
    )

    assert len(records) == 1
    assert records[0].assay_type == "half maximal inhibitory concentration (IC50)"
    assert records[0].assay_method == "purified MHC/competitive/radioactivity"
    assert stats["rows_selected"] == 1


def test_load_binding_records_from_merged_tsv_skips_modified_peptides(tmp_path):
    merged = tmp_path / "merged.tsv"
    merged.write_text(
        "\t".join(
            [
                "peptide",
                "mhc_allele",
                "mhc_class",
                "source",
                "record_type",
                "value",
                "value_type",
                "qualifier",
                "response",
                "assay_type",
                "assay_method",
                "apc_name",
                "effector_culture_condition",
                "apc_culture_condition",
                "in_vitro_process_type",
                "in_vitro_responder_cell",
                "in_vitro_stimulator_cell",
                "species",
                "antigen_species",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "ALEGSLQKR + CITR(R9)",
                "HLA-A*02:01",
                "I",
                "iedb",
                "binding",
                "25",
                "half maximal inhibitory concentration (IC50)",
                "0",
                "",
                "half maximal inhibitory concentration (IC50)",
                "purified MHC/competitive/radioactivity",
                "",
                "",
                "",
                "",
                "",
                "",
                "Homo sapiens",
                "SARS-CoV-2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records, stats = _load_binding_records_from_merged_tsv(
        merged,
        mhc_class_filter="I",
    )

    assert records == []
    assert stats["rows_selected"] == 0


def test_filter_compatible_state_dict_drops_shape_mismatches():
    model = Presto(d_model=64, n_layers=1, n_heads=4)
    state = model.state_dict()
    mutated = dict(state)
    mutated["affinity_predictor.kd_assay_bias.0.weight"] = torch.zeros(64, 999)

    compatible, stats = _filter_compatible_state_dict(model, mutated)

    assert "affinity_predictor.kd_assay_bias.0.weight" not in compatible
    assert "affinity_predictor.kd_assay_bias.0.weight" in stats["skipped_shape"]


class _MiniDataset:
    def __init__(self, alleles):
        self.samples = []
        for item in alleles:
            if isinstance(item, tuple):
                allele, synthetic_kind = item
            else:
                allele, synthetic_kind = item, None
            self.samples.append(
                SimpleNamespace(primary_allele=allele, synthetic_kind=synthetic_kind)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_strict_allele_balanced_batch_sampler_balances_each_batch():
    dataset = _MiniDataset(
        ["A*02:01"] * 6 + ["A*24:02"] * 2
    )
    sampler = StrictAlleleBalancedBatchSampler(
        dataset,
        ["A*02:01", "A*24:02"],
        batch_size=4,
        seed=7,
    )

    batches = list(iter(sampler))

    assert len(batches) == 3
    for batch in batches:
        alleles = [dataset[idx].primary_allele for idx in batch]
        assert alleles.count("A*02:01") == 2
        assert alleles.count("A*24:02") == 2


def test_create_focused_train_loader_uses_strict_sampler_for_multi_allele_panel():
    dataset = _MiniDataset(
        ["A*02:01"] * 6 + ["A*24:02"] * 2
    )
    loader = _create_focused_train_loader(
        dataset,
        batch_size=4,
        collator=lambda items: items,
        balanced=True,
        seed=7,
        alleles=["A*02:01", "A*24:02"],
    )

    batch = next(iter(loader))
    alleles = [sample.primary_allele for sample in batch]
    assert alleles.count("A*02:01") == 2
    assert alleles.count("A*24:02") == 2


def test_strict_allele_balanced_batch_sampler_balances_real_and_synthetic_when_requested():
    dataset = _MiniDataset(
        [("A*02:01", None)] * 4
        + [("A*02:01", "synthetic_negative_peptide_random")] * 4
        + [("A*24:02", None)] * 2
        + [("A*24:02", "synthetic_negative_peptide_random")] * 2
    )
    sampler = StrictAlleleBalancedBatchSampler(
        dataset,
        ["A*02:01", "A*24:02"],
        batch_size=4,
        synthetic_fraction=0.5,
        seed=7,
    )

    batch = next(iter(sampler))
    picked = [dataset[idx] for idx in batch]
    assert sum(1 for sample in picked if sample.primary_allele == "A*02:01") == 2
    assert sum(1 for sample in picked if sample.primary_allele == "A*24:02") == 2
    assert sum(1 for sample in picked if not sample.synthetic_kind) == 2
    assert sum(1 for sample in picked if sample.synthetic_kind) == 2


def test_parse_synthetic_modes_validates_and_normalizes():
    assert _parse_synthetic_modes("") is None
    assert _parse_synthetic_modes("all") is None
    assert _parse_synthetic_modes("peptide_scramble,mhc_random") == (
        "peptide_scramble",
        "mhc_random",
    )
    with pytest.raises(ValueError):
        _parse_synthetic_modes("bad_mode")


def test_resolve_batch_synthetic_fraction_derives_from_negative_ratio():
    assert _resolve_batch_synthetic_fraction(
        synthetic_negatives=False,
        negative_ratio=1.0,
        explicit_fraction=-1.0,
    ) == pytest.approx(0.0)
    assert _resolve_batch_synthetic_fraction(
        synthetic_negatives=True,
        negative_ratio=1.0,
        explicit_fraction=-1.0,
    ) == pytest.approx(0.5)
    assert _resolve_batch_synthetic_fraction(
        synthetic_negatives=True,
        negative_ratio=1.0,
        explicit_fraction=0.25,
    ) == pytest.approx(0.25)


def test_prepared_binding_state_cache_roundtrip(tmp_path):
    merged_tsv = tmp_path / "merged.tsv"
    merged_tsv.write_text("mhc_allele\tpeptide\tvalue\tvalue_type\tqualifier\tsource\n", encoding="utf-8")
    index_csv = tmp_path / "mhc_index.csv"
    index_csv.write_text("allele,sequence\nHLA-A*02:01,AAAA\n", encoding="utf-8")
    contract = DatasetContract(
        source="iedb",
        probe_alleles=("HLA-A*02:01", "HLA-A*24:02"),
        training_alleles=("HLA-A*02:01", "HLA-A*24:02"),
        train_all_alleles=False,
        train_mhc_class_filter="I",
        max_records=0,
        sampling_seed=59,
        measurement_profile="numeric",
        measurement_type_filter="ic50",
        qualifier_filter="all",
        shared_peptides_only=False,
        max_per_allele=-1,
        split_seed=42,
        explicit_probe_peptides=("SLLQHLIGL",),
    )
    cache_key = _contract_cache_key(contract=contract, merged_tsv=merged_tsv, index_csv=index_csv)
    state = PreparedBindingState(
        real_records=[BindingRecord(peptide="PEP", mhc_allele="HLA-A*02:01", value=10.0)],
        real_train_records=[BindingRecord(peptide="PEP", mhc_allele="HLA-A*02:01", value=10.0)],
        real_val_records=[BindingRecord(peptide="PEQ", mhc_allele="HLA-A*24:02", value=20.0)],
        subset_stats={"rows_selected": 2},
        shared_peptide_stats={"shared_peptides": 0},
        probe_allele_counts_after_filter={"HLA-A*02:01": 1, "HLA-A*24:02": 1},
        balance_stats={"skipped": True},
        split_stats={"train_rows": 1, "val_rows": 1},
        mhc_sequences={"HLA-A*02:01": "AAAA"},
        mhc_stats={"resolved": 1},
        probe_support={"SLLQHLIGL": {"total_rows": 0}},
    )
    cache_dir = tmp_path / "cache"

    _write_prepared_binding_state_to_cache(
        cache_dir=cache_dir,
        cache_key=cache_key,
        contract=contract,
        merged_tsv=merged_tsv,
        index_csv=index_csv,
        state=state,
    )
    loaded = _load_prepared_binding_state_from_cache(cache_dir=cache_dir, cache_key=cache_key)

    assert loaded is not None
    assert loaded.cache_hit is True
    assert loaded.cache_key == cache_key
    assert len(loaded.real_records) == 1
    assert loaded.probe_support["SLLQHLIGL"]["total_rows"] == 0
