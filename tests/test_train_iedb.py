"""Tests for IEDB training utilities."""

import argparse
from pathlib import Path

import pytest
import torch

import presto.scripts.train_iedb as train_iedb_module
from presto.data import PrestoDataset
from presto.data.collate import PrestoBatch
from presto.data.cross_source_dedup import UnifiedRecord
from presto.data.loaders import (
    BindingRecord,
    ElutionRecord,
    MIN_MHC_CHAIN_LENGTH,
    ProcessingRecord,
    TCellRecord,
)
from presto.scripts.train_iedb import (
    _audit_mhc_sequence_coverage,
    _effective_mhc_augmentation_sample_limit,
    _filter_records_to_resolved_mhc,
    _resolve_run_args,
    _write_mhc_sequence_coverage_report,
    audit_loaded_mhc_sequence_quality,
    _evaluate_pmhc_information_flow,
    _merge_records_with_limit,
    augment_binding_records_with_synthetic_negatives,
    bootstrap_missing_modalities_for_canary,
    cascade_binding_negatives_to_downstream,
    augment_elution_records_with_synthetic_negatives,
    augment_processing_records_with_synthetic_negatives,
    find_iedb_export_file,
    load_iedb_binding_and_elution_records,
    load_records_from_merged_tsv,
    load_iedb_tcell_records,
    resolve_mhc_sequences_from_index,
    run,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _PeptideOnlyFlowModel(torch.nn.Module):
    def forward(
        self,
        pep_tok,
        mhc_a_tok,
        mhc_b_tok,
        mhc_class=None,
        species=None,
        tcr_a_tok=None,
        tcr_b_tok=None,
        flank_n_tok=None,
        flank_c_tok=None,
        tcell_context=None,
    ):
        pep_signal = pep_tok.float().sum(dim=1)
        binding = pep_signal
        return {
            "binding_logit": binding,
            "binding_prob": torch.sigmoid(binding),
            "presentation_logit": (0.5 * binding).unsqueeze(-1),
            "presentation_prob": torch.sigmoid(0.5 * binding).unsqueeze(-1),
            "processing_logit": (0.1 * binding).unsqueeze(-1),
            "assays": {"KD_nM": (3.0 - 0.05 * binding).unsqueeze(-1)},
        }


class _InteractionFlowModel(torch.nn.Module):
    def forward(
        self,
        pep_tok,
        mhc_a_tok,
        mhc_b_tok,
        mhc_class=None,
        species=None,
        tcr_a_tok=None,
        tcr_b_tok=None,
        flank_n_tok=None,
        flank_c_tok=None,
        tcell_context=None,
    ):
        pep_signal = pep_tok.float().sum(dim=1)
        mhc_signal = mhc_a_tok.float().sum(dim=1) + mhc_b_tok.float().sum(dim=1)
        binding = 0.05 * pep_signal * mhc_signal
        return {
            "binding_logit": binding,
            "binding_prob": torch.sigmoid(binding),
            "presentation_logit": (0.25 * binding).unsqueeze(-1),
            "presentation_prob": torch.sigmoid(0.25 * binding).unsqueeze(-1),
            "processing_logit": (0.1 * pep_signal).unsqueeze(-1),
            "assays": {"KD_nM": (4.0 - 0.1 * binding).unsqueeze(-1)},
        }


def _make_flow_probe_batch() -> PrestoBatch:
    return PrestoBatch(
        pep_tok=torch.tensor(
            [
                [1, 2, 0],
                [3, 1, 0],
                [2, 2, 0],
                [4, 1, 0],
            ],
            dtype=torch.long,
        ),
        mhc_a_tok=torch.tensor(
            [
                [2, 2, 0],
                [1, 1, 0],
                [3, 3, 0],
                [1, 2, 0],
            ],
            dtype=torch.long,
        ),
        mhc_b_tok=torch.tensor(
            [
                [1, 0],
                [2, 0],
                [3, 0],
                [4, 0],
            ],
            dtype=torch.long,
        ),
        mhc_class=["I", "I", "I", "I"],
        processing_species=["human", "human", "human", "human"],
    )


def test_resolve_run_args_canary_profile_applies_fast_caps():
    args = argparse.Namespace(profile="canary", config=None)
    resolved = _resolve_run_args(args)
    assert resolved.profile == "canary"
    assert resolved.epochs == 1
    assert resolved.max_binding == 512
    assert resolved.max_elution == 512
    assert resolved.max_tcell == 512
    assert resolved.max_vdjdb == 256
    assert resolved.max_10x == 256
    assert resolved.supervised_loss_aggregation == "sample_weighted"
    assert resolved.track_pmhc_flow is True
    assert resolved.pmhc_flow_batches == 2
    assert resolved.pmhc_flow_max_samples == 512


def test_resolve_run_args_diagnostic_profile_enables_richer_diagnostics():
    args = argparse.Namespace(profile="diagnostic", config=None)
    resolved = _resolve_run_args(args)
    assert resolved.profile == "diagnostic"
    assert resolved.track_probe_affinity is True
    assert resolved.track_pmhc_flow is True
    assert resolved.track_output_latent_stats is True
    assert resolved.pmhc_flow_batches == 8
    assert resolved.pmhc_flow_max_samples == 2048
    assert resolved.output_latent_stats_batches == 8
    assert resolved.output_latent_stats_max_samples == 2048
    assert resolved.filter_unresolved_mhc is True
    assert resolved.strict_mhc_resolution is True


def test_call_train_epoch_compat_forwards_scheduler(monkeypatch):
    seen = {}

    def _fake_train_epoch(model, train_loader, optimizer, device, scheduler=None, **kwargs):
        seen["scheduler"] = scheduler
        return 0.25, {"loss_binding": 0.1}

    scheduler = object()
    monkeypatch.setattr(train_iedb_module, "train_epoch", _fake_train_epoch)

    loss, metrics = train_iedb_module._call_train_epoch_compat(
        model=None,
        train_loader=[],
        optimizer=None,
        device="cpu",
        scheduler=scheduler,
        uncertainty_weighting=None,
        pcgrad=None,
    )

    assert loss == pytest.approx(0.25)
    assert metrics == {"loss_binding": 0.1}
    assert seen["scheduler"] is scheduler


def test_pmhc_information_flow_detects_peptide_dominant_behavior():
    metrics = _evaluate_pmhc_information_flow(
        model=_PeptideOnlyFlowModel(),
        val_loader=[_make_flow_probe_batch()],
        device="cpu",
        n_batches=1,
        max_samples=64,
        non_blocking_transfer=False,
    )
    assert metrics["pmhc_flow_samples"] == 4.0
    assert metrics["pmhc_flow_binding_logit_delta_mhc_abs"] < 1e-7
    assert metrics["pmhc_flow_binding_logit_delta_peptide_abs"] > 0.1
    assert metrics["pmhc_flow_status_code"] == 1.0


def test_pmhc_information_flow_detects_joint_interaction_signal():
    metrics = _evaluate_pmhc_information_flow(
        model=_InteractionFlowModel(),
        val_loader=[_make_flow_probe_batch()],
        device="cpu",
        n_batches=1,
        max_samples=64,
        non_blocking_transfer=False,
    )
    assert metrics["pmhc_flow_samples"] == 4.0
    assert metrics["pmhc_flow_binding_logit_delta_mhc_abs"] > 0.05
    assert metrics["pmhc_flow_binding_logit_delta_peptide_abs"] > 0.05
    assert metrics["pmhc_flow_binding_logit_interaction_abs"] > 0.01
    assert metrics["pmhc_flow_status_code"] in {2.0, 3.0}


def test_audit_loaded_mhc_sequence_quality_flags_noncanonical_x_and_short():
    quality = audit_loaded_mhc_sequence_quality(
        {
            "HLA-A*02:01": "A" * 181,
            "HLA-A*24:02": ("A" * 170) + "X",
            "HLA-B*07:02": ("A" * 110) + "J",
            "HLA-C*04:01": "C" * (MIN_MHC_CHAIN_LENGTH - 1),
        }
    )

    assert quality["total_sequences"] == 4
    assert quality["x_sequence_count"] == 1
    assert quality["x_residue_total"] == 1
    assert quality["noncanonical_count"] == 1
    assert quality["short_count"] == 1
    assert any(example[0] == "HLA-B*07:02" for example in quality["noncanonical_examples"])
    assert any(example[0] == "HLA-C*04:01" for example in quality["short_examples"])


def test_audit_loaded_mhc_sequence_quality_accepts_groove_length_fragments():
    quality = audit_loaded_mhc_sequence_quality(
        {
            "HLA-A*02:01": "A" * MIN_MHC_CHAIN_LENGTH,
            "HLA-A*24:02": "C" * (MIN_MHC_CHAIN_LENGTH + 12),
        }
    )

    assert quality["noncanonical_count"] == 0
    assert quality["short_count"] == 0


def test_resolve_run_args_canary_keeps_explicit_caps():
    args = argparse.Namespace(profile="canary", config=None, max_binding=33, epochs=2)
    resolved = _resolve_run_args(args)
    assert resolved.epochs == 2
    assert resolved.max_binding == 33
    assert resolved.max_tcell == 512


def test_resolve_run_args_prefers_train_unified_config(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "train:\n"
        "  unified:\n"
        "    epochs: 3\n"
        "    max_binding: 777\n"
        "  iedb:\n"
        "    epochs: 9\n"
        "    max_binding: 42\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(config=str(cfg), profile="full")
    resolved = _resolve_run_args(args)
    assert resolved.epochs == 3
    assert resolved.max_binding == 777


def test_resolve_run_args_falls_back_to_train_iedb_config(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "train:\n"
        "  iedb:\n"
        "    epochs: 6\n"
        "    max_binding: 123\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(config=str(cfg), profile="full")
    resolved = _resolve_run_args(args)
    assert resolved.epochs == 6
    assert resolved.max_binding == 123


def test_canary_bootstrap_backfills_missing_modalities():
    binding = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
        )
    ]
    kinetics, stability, processing, stats = bootstrap_missing_modalities_for_canary(
        binding_records=binding,
        kinetics_records=[],
        stability_records=[],
        processing_records=[],
        seed=13,
        max_per_modality=4,
    )
    assert len(kinetics) == 4
    assert len(stability) == 4
    assert len(processing) == 4
    assert stats == {"kinetics": 4, "stability": 4, "processing": 4}
    assert all(rec.source == "canary_bootstrap" for rec in kinetics)
    assert all(rec.source == "canary_bootstrap" for rec in stability)
    assert all(rec.source == "canary_bootstrap" for rec in processing)


def test_canary_bootstrap_keeps_existing_modalities():
    binding = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
        )
    ]
    existing_processing = [
        ProcessingRecord(
            peptide="SIINFEKL",
            flank_n="AAAAAAAAAA",
            flank_c="CCCCCCCCCC",
            label=1.0,
            mhc_allele="HLA-A*02:01",
            mhc_class="I",
            source="iedb",
        )
    ]
    _, _, processing, stats = bootstrap_missing_modalities_for_canary(
        binding_records=binding,
        kinetics_records=[],
        stability_records=[],
        processing_records=existing_processing,
        seed=7,
        max_per_modality=3,
    )
    assert len(processing) == 1
    assert processing[0].source == "iedb"
    assert stats["processing"] == 0


def test_find_iedb_export_file_prefers_keyword_match_and_size(tmp_path):
    root = tmp_path / "iedb"
    _write_text(root / "other.csv", "a,b\n1,2\n")
    _write_text(root / "sub" / "tcell_small.tsv", "x\ty\n1\t2\n")
    largest = root / "nested" / "tcell_full_v3.csv"
    _write_text(largest, "h1,h2\n" + ("1,2\n" * 40))

    selected = find_iedb_export_file(root, keywords=("tcell",))
    assert selected == largest


def test_find_iedb_export_file_requires_keywords_for_cedar_selection(tmp_path):
    root = tmp_path / "iedb"
    non_cedar = root / "mhc_ligand_full.csv"
    cedar = root / "cedar_mhc_ligand_full.csv"
    _write_text(non_cedar, "h1,h2\n" + ("1,2\n" * 100))
    _write_text(cedar, "h1,h2\n1,2\n")

    selected = find_iedb_export_file(
        root,
        keywords=("mhc", "ligand"),
        required_keywords=("cedar",),
    )
    assert selected == cedar


def test_find_iedb_export_file_required_keywords_missing_raises(tmp_path):
    root = tmp_path / "iedb"
    _write_text(root / "mhc_ligand_full.csv", "h1,h2\n1,2\n")

    try:
        find_iedb_export_file(
            root,
            keywords=("mhc", "ligand"),
            required_keywords=("cedar",),
        )
    except FileNotFoundError as exc:
        assert "required keywords" in str(exc).lower()
    else:
        raise AssertionError("Expected missing required keyword selection to fail")


def test_find_iedb_export_file_matches_any_keyword_for_10x_inputs(tmp_path):
    root = tmp_path / "10x"
    sc10x = root / "10x_pbmc_10k_tcr.csv"
    _write_text(sc10x, "barcode,chain\nA,TRA\n")

    selected = find_iedb_export_file(root, keywords=("contig", "10x", "vdj", "tcr"))
    assert selected == sc10x


def test_merge_records_with_limit_caps_total_across_groups():
    merged = _merge_records_with_limit(
        record_lists=[
            ["a1", "a2", "a3"],
            ["b1", "b2", "b3"],
        ],
        max_records=4,
        seed=7,
    )
    assert len(merged) == 4
    assert set(merged).issubset({"a1", "a2", "a3", "b1", "b2", "b3"})


def test_resolve_mhc_sequences_from_index_maps_input_alleles(tmp_path, monkeypatch):
    index_csv = tmp_path / "mhc_index.csv"
    index_csv.write_text(
        (
            "allele_raw,normalized,gene,mhc_class,species,source,seq_len,sequence\n"
            "HLA-A*02:01,HLA-A*02:01,A,I,human,imgt,4,AAAA\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "presto.scripts.train_iedb.resolve_alleles",
        lambda index_csv, alleles, include_sequence: [
            {
                "input": "HLA-A*02:01",
                "found": True,
                "sequence": "AAAA",
            },
            {
                "input": "HLA-B*07:02",
                "found": False,
                "sequence": "",
            },
        ],
    )

    mapping, stats = resolve_mhc_sequences_from_index(
        index_csv=str(index_csv),
        alleles=["HLA-A*02:01", "HLA-B*07:02"],
    )

    assert mapping["HLA-A*02:01"] == "AAAA"
    assert "HLA-B*07:02" not in mapping
    assert stats["resolved"] == 1
    assert stats["total"] == 2


def test_audit_mhc_sequence_coverage_reports_resolved_missing_and_species_buckets():
    binding = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        ),
        BindingRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*99:99",
            value=5000.0,
            mhc_class="I",
            species="human",
            source="iedb",
        ),
    ]
    processing = [
        ProcessingRecord(
            peptide="SSYRRPVGI",
            flank_n="AAAAAAAAAA",
            flank_c="CCCCCCCCCC",
            label=1.0,
            mhc_allele="H2-K*b",
            mhc_class="I",
            species="mouse",
            source="iedb",
        )
    ]
    elution = [
        ElutionRecord(
            peptide="SSYRRPVGI",
            alleles=["H2-K*b", "H2-K*zzz"],
            detected=True,
            mhc_class="I",
            species="mouse",
            source="iedb",
        )
    ]
    tcell = [
        TCellRecord(
            peptide="NLVPMVATV",
            mhc_allele="MAMU-A1*001",
            response=1.0,
            mhc_class="I",
            species="macaque",
            source="iedb",
        )
    ]
    mhc_sequences = {
        "HLA-A*02:01": "AAAA",
        "H2-K*b": "BBBB",
    }
    coverage = _audit_mhc_sequence_coverage(
        binding_records=binding,
        kinetics_records=[],
        stability_records=[],
        processing_records=processing,
        elution_records=elution,
        tcell_records=tcell,
        vdjdb_records=[],
        mhc_sequences=mhc_sequences,
    )

    overall = coverage["overall"]
    assert overall["rows_considered"] == 6
    assert overall["resolved_rows"] == 3
    assert overall["missing_rows"] == 3
    assert coverage["species_by_state"]["resolved"] == {
        "human": 1,
        "murine": 2,
        "nhp": 0,
        "other": 0,
    }
    assert coverage["species_by_state"]["missing"] == {
        "human": 1,
        "murine": 1,
        "nhp": 1,
        "other": 0,
    }
    assert coverage["by_modality"]["binding"]["resolved_rows"] == 1
    assert coverage["by_modality"]["binding"]["missing_rows"] == 1


def test_write_mhc_sequence_coverage_report_persists_json_and_csv(tmp_path):
    coverage = {
        "overall": {
            "rows_considered": 10,
            "resolved_rows": 7,
            "missing_rows": 3,
            "resolved_fraction": 0.7,
            "missing_fraction": 0.3,
        },
        "species_by_state": {
            "resolved": {"human": 5, "murine": 2, "nhp": 0, "other": 0},
            "missing": {"human": 1, "murine": 1, "nhp": 1, "other": 0},
        },
        "by_modality": {},
        "by_modality_species": {},
    }
    paths = _write_mhc_sequence_coverage_report(run_dir=tmp_path, coverage=coverage)
    json_path = paths["json"]
    csv_path = paths["csv"]
    assert json_path is not None and json_path.exists()
    assert csv_path is not None and csv_path.exists()
    assert "\"rows_considered\": 10" in json_path.read_text(encoding="utf-8")
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "scope,modality,state,species,count,fraction_of_total_rows,fraction_within_scope"


def test_filter_records_to_resolved_mhc_drops_unresolved_and_trims_elution_alleles():
    binding = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            source="iedb",
        ),
        BindingRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*99:99",
            value=5000.0,
            mhc_class="I",
            source="iedb",
        ),
    ]
    elution = [
        ElutionRecord(
            peptide="SSYRRPVGI",
            alleles=["H2-K*b", "H2-K*zzz"],
            detected=True,
            mhc_class="I",
            species="mouse",
            source="iedb",
        )
    ]
    tcell = [
        TCellRecord(
            peptide="NLVPMVATV",
            mhc_allele="MAMU-A1*001",
            response=1.0,
            mhc_class="I",
            source="iedb",
        )
    ]
    mhc_sequences = {"HLA-A*02:01": "AAAA", "H2-K*b": "BBBB"}

    (
        binding_out,
        kinetics_out,
        stability_out,
        processing_out,
        elution_out,
        tcell_out,
        vdjdb_out,
        stats,
    ) = _filter_records_to_resolved_mhc(
        binding_records=binding,
        kinetics_records=[],
        stability_records=[],
        processing_records=[],
        elution_records=elution,
        tcell_records=tcell,
        vdjdb_records=[],
        mhc_sequences=mhc_sequences,
    )

    assert len(binding_out) == 1
    assert len(kinetics_out) == 0
    assert len(stability_out) == 0
    assert len(processing_out) == 0
    assert len(elution_out) == 1
    assert elution_out[0].alleles == ["H2-K*b"]
    assert len(tcell_out) == 0
    assert len(vdjdb_out) == 0
    assert stats["binding_dropped"] == 1
    assert stats["elution_alleles_dropped"] == 1
    assert stats["elution_rows_dropped"] == 0
    assert stats["tcell_dropped"] == 1


def test_load_iedb_binding_and_elution_records_fallback_parser(monkeypatch, tmp_path):
    data_file = tmp_path / "mhc.csv"
    data_file.write_text("unused\n", encoding="utf-8")

    monkeypatch.setattr("presto.scripts.train_iedb.load_iedb_binding", lambda _: iter([]))
    monkeypatch.setattr("presto.scripts.train_iedb.load_iedb_elution", lambda _: iter([]))
    monkeypatch.setattr(
        "presto.scripts.train_iedb.parse_iedb_binding",
        lambda _: iter(
            [
                UnifiedRecord(
                    peptide="SIINFEKL",
                    mhc_allele="HLA-A*02:01",
                    mhc_class="I",
                    source="iedb",
                    record_type="binding",
                    value=50.0,
                    qualifier=0,
                    value_type="IC50",
                ),
                UnifiedRecord(
                    peptide="GILGFVFTL",
                    mhc_allele="HLA-B*07:02",
                    mhc_class="I",
                    source="iedb",
                    record_type="binding",
                    value=None,
                    qualifier=0,
                ),
            ]
        ),
    )

    binding, elution = load_iedb_binding_and_elution_records(
        data_file,
        max_binding=10,
        max_elution=10,
    )
    assert len(binding) == 1
    assert len(elution) == 1
    assert binding[0].mhc_allele == "HLA-A*02:01"
    assert elution[0].alleles == ["HLA-B*07:02"]


def test_load_iedb_binding_and_elution_records_zero_limit_means_unlimited(monkeypatch, tmp_path):
    data_file = tmp_path / "mhc.csv"
    data_file.write_text("unused\n", encoding="utf-8")

    monkeypatch.setattr("presto.scripts.train_iedb.load_iedb_binding", lambda _: iter([]))
    monkeypatch.setattr("presto.scripts.train_iedb.load_iedb_elution", lambda _: iter([]))
    monkeypatch.setattr(
        "presto.scripts.train_iedb.parse_iedb_binding",
        lambda _: iter(
            [
                UnifiedRecord(
                    peptide="SIINFEKL",
                    mhc_allele="HLA-A*02:01",
                    mhc_class="I",
                    source="iedb",
                    record_type="binding",
                    value=50.0,
                    qualifier=0,
                    value_type="IC50",
                ),
                UnifiedRecord(
                    peptide="GILGFVFTL",
                    mhc_allele="HLA-B*07:02",
                    mhc_class="I",
                    source="iedb",
                    record_type="binding",
                    value=None,
                    qualifier=0,
                ),
            ]
        ),
    )

    binding, elution = load_iedb_binding_and_elution_records(
        data_file,
        max_binding=0,
        max_elution=0,
    )
    assert len(binding) == 1
    assert len(elution) == 1


def test_load_iedb_tcell_records_fallback_parser(monkeypatch, tmp_path):
    data_file = tmp_path / "tcell.csv"
    data_file.write_text("unused\n", encoding="utf-8")

    monkeypatch.setattr("presto.scripts.train_iedb.load_iedb_tcell", lambda _: iter([]))
    monkeypatch.setattr(
        "presto.scripts.train_iedb.parse_iedb_tcell",
        lambda _: iter(
            [
                UnifiedRecord(
                    peptide="SIINFEKL",
                    mhc_allele="HLA-A*02:01",
                    mhc_class="I",
                    source="iedb",
                    record_type="tcell",
                    response="positive-high",
                ),
                UnifiedRecord(
                    peptide="GILGFVFTL",
                    mhc_allele="HLA-B*07:02",
                    mhc_class="I",
                    source="iedb",
                    record_type="tcell",
                    response="0",
                ),
                UnifiedRecord(
                    peptide="QLSPFPFDL",
                    mhc_allele="HLA-A*03:01",
                    mhc_class="I",
                    source="iedb",
                    record_type="tcell",
                    response="unknown",
                ),
            ]
        ),
    )

    records = load_iedb_tcell_records(data_file, max_tcell=10)
    assert len(records) == 2
    assert records[0].response == 1.0
    assert records[1].response == 0.0


def test_load_iedb_tcell_records_zero_limit_means_unlimited(monkeypatch, tmp_path):
    data_file = tmp_path / "tcell.csv"
    data_file.write_text("unused\n", encoding="utf-8")

    monkeypatch.setattr("presto.scripts.train_iedb.load_iedb_tcell", lambda _: iter([]))
    monkeypatch.setattr(
        "presto.scripts.train_iedb.parse_iedb_tcell",
        lambda _: iter(
            [
                UnifiedRecord(
                    peptide="SIINFEKL",
                    mhc_allele="HLA-A*02:01",
                    mhc_class="I",
                    source="iedb",
                    record_type="tcell",
                    response="positive",
                ),
                UnifiedRecord(
                    peptide="GILGFVFTL",
                    mhc_allele="HLA-B*07:02",
                    mhc_class="I",
                    source="iedb",
                    record_type="tcell",
                    response="negative",
                ),
            ]
        ),
    )

    records = load_iedb_tcell_records(data_file, max_tcell=0)
    assert len(records) == 2


def test_load_records_from_merged_tsv_maps_assays_and_tcell_context(tmp_path):
    merged = tmp_path / "merged_deduped.tsv"
    merged.write_text(
        (
            "peptide\tmhc_allele\tmhc_class\tpmid\tdoi\treference_text\tsource\trecord_type\t"
            "value\tvalue_type\tqualifier\tresponse\tassay_type\tassay_method\tapc_name\t"
            "effector_culture_condition\tapc_culture_condition\tin_vitro_process_type\t"
            "in_vitro_responder_cell\tin_vitro_stimulator_cell\tcdr3_alpha\tcdr3_beta\ttrav\ttrbv\tspecies\n"
            "SIINFEKL\tHLA-A*02:01\tI\t\t\t\tiedb\tbinding\t42\tIC50\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\thuman\n"
            "GILGFVFTL\tHLA-A*02:01\tI\t\t\t\tiedb\tbinding\t\tligand presentation\t0\tpositive\t\t\t\t\t\t\t\t\t\t\t\t\thuman\n"
            "NLVPMVATV\tHLA-A*02:01\tI\t\t\t\tiedb\ttcell\t\t\t0\tpositive\tIFNg release\tELISPOT\tPBMC\t"
            "Direct ex vivo\tNA\tNA\tNA\tNA\t\t\t\t\thuman\n"
            "GLCTLVAML\tHLA-A*02:01\tI\t\t\t\tvdjdb\ttcr\t\t\t0\t\t\t\t\t\t\t\t\t\tCAVRDSNYQLIW\tCASSIRSSYEQYF\tTRAV12-2\tTRBV7-9\thuman\n"
        ),
        encoding="utf-8",
    )

    (
        binding,
        kinetics,
        stability,
        processing,
        elution,
        tcell,
        tcr,
        stats,
    ) = load_records_from_merged_tsv(
        merged,
        max_binding=0,
        max_kinetics=0,
        max_stability=0,
        max_processing=0,
        max_elution=0,
        max_tcell=0,
        max_vdjdb=0,
    )

    assert len(binding) == 1
    assert len(kinetics) == 0
    assert len(stability) == 0
    assert len(processing) == 0
    assert len(elution) == 1
    assert len(tcell) == 1
    assert len(tcr) == 1
    assert tcell[0].assay_method == "ELISPOT"
    assert tcell[0].assay_type == "IFNg release"
    assert tcell[0].apc_name == "PBMC"
    assert "binding_affinity" in stats["rows_by_assay"]
    assert "elution_ms" in stats["rows_by_assay"]
    assert "tcell_response" in stats["rows_by_assay"]


def test_load_records_from_merged_tsv_drops_invalid_peptides_and_sanitizes_optional_sequences(tmp_path):
    merged = tmp_path / "merged_deduped.tsv"
    merged.write_text(
        (
            "peptide\tmhc_allele\tmhc_class\tpmid\tdoi\treference_text\tsource\trecord_type\t"
            "value\tvalue_type\tqualifier\tresponse\tassay_type\tassay_method\tapc_name\t"
            "effector_culture_condition\tapc_culture_condition\tin_vitro_process_type\t"
            "in_vitro_responder_cell\tin_vitro_stimulator_cell\tcdr3_alpha\tcdr3_beta\ttrav\ttrbv\tspecies\n"
            "SIINFEKL\tHLA-A*02:01\tI\t\t\t\tiedb\tbinding\t42\tIC50\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\thuman\n"
            "PKYVKQNTLKLAT + BIOT(P1)\tHLA-A*02:01\tI\t\t\t\tiedb\tbinding\t120\tIC50\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\thuman\n"
            "GILGFVFTL\tHLA-A*02:01\tI\t\t\t\tvdjdb\ttcr\t\t\t0\t\t\t\t\t\t\t\t\t\tNA\tCASSIRSSYEQYF\tTRAV12-2\tTRBV7-9\thuman\n"
        ),
        encoding="utf-8",
    )

    (
        binding,
        kinetics,
        stability,
        processing,
        elution,
        tcell,
        tcr,
        stats,
    ) = load_records_from_merged_tsv(
        merged,
        max_binding=0,
        max_kinetics=0,
        max_stability=0,
        max_processing=0,
        max_elution=0,
        max_tcell=0,
        max_vdjdb=0,
    )

    assert len(binding) == 1
    assert len(kinetics) == 0
    assert len(stability) == 0
    assert len(processing) == 0
    assert len(elution) == 0
    assert len(tcell) == 0
    assert len(tcr) == 1
    assert tcr[0].cdr3_alpha is None
    assert stats["rows_dropped_invalid_peptide"] == 1
    assert stats["rows_sanitized_optional_sequences"] >= 1


def test_load_records_from_merged_tsv_cap_sampling_modes(tmp_path):
    merged = tmp_path / "merged_deduped.tsv"
    header = (
        "peptide\tmhc_allele\tmhc_class\tpmid\tdoi\treference_text\tsource\trecord_type\t"
        "value\tvalue_type\tqualifier\tresponse\tassay_type\tassay_method\tapc_name\t"
        "effector_culture_condition\tapc_culture_condition\tin_vitro_process_type\t"
        "in_vitro_responder_cell\tin_vitro_stimulator_cell\tcdr3_alpha\tcdr3_beta\ttrav\ttrbv\tspecies\n"
    )
    rows = []
    # First 10 rows are one allele; remaining rows are another allele.
    for i in range(10):
        rows.append(
            "SIINFEKL\tHLA-A*02:01\tI\t\t\t\tiedb\tbinding\t"
            f"{50 + i}\tIC50\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\thuman\n"
        )
    for i in range(90):
        rows.append(
            "SIINFEKL\tHLA-B*07:02\tI\t\t\t\tiedb\tbinding\t"
            f"{500 + i}\tIC50\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\thuman\n"
        )
    merged.write_text(header + "".join(rows), encoding="utf-8")

    # Head mode preserves first-N bias.
    (
        binding_head,
        *_,
        stats_head,
    ) = load_records_from_merged_tsv(
        merged,
        max_binding=10,
        max_kinetics=0,
        max_stability=0,
        max_processing=0,
        max_elution=0,
        max_tcell=0,
        max_vdjdb=0,
        cap_sampling="head",
        sampling_seed=7,
    )
    assert len(binding_head) == 10
    assert all(rec.mhc_allele == "HLA-A*02:01" for rec in binding_head)
    assert stats_head["cap_sampling"] == "head"

    # Reservoir mode should pull a representative capped sample.
    (
        binding_reservoir,
        *_,
        stats_reservoir,
    ) = load_records_from_merged_tsv(
        merged,
        max_binding=10,
        max_kinetics=0,
        max_stability=0,
        max_processing=0,
        max_elution=0,
        max_tcell=0,
        max_vdjdb=0,
        cap_sampling="reservoir",
        sampling_seed=7,
    )
    assert len(binding_reservoir) == 10
    # With deterministic seed and 90% B*07:02 prevalence, capped sample includes B*07:02.
    assert any(rec.mhc_allele == "HLA-B*07:02" for rec in binding_reservoir)
    assert stats_reservoir["cap_sampling"] == "reservoir"


def test_augment_binding_records_with_synthetic_negatives_range_and_modes():
    base = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            source="iedb",
        ),
        BindingRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-B*07:02",
            value=100.0,
            mhc_class="I",
            source="iedb",
        ),
    ]
    mhc_sequences = {"HLA-A*02:01": "AAAA", "HLA-B*07:02": "BBBB"}

    augmented, stats = augment_binding_records_with_synthetic_negatives(
        binding_records=base,
        mhc_sequences=mhc_sequences,
        negative_ratio=1.5,
        weak_value_min_nM=50000.0,
        weak_value_max_nM=100000.0,
        seed=13,
    )

    assert len(augmented) == len(base) + 3
    assert stats["added"] == 3
    assert stats["added_general"] == 3
    assert stats["no_mhc_beta"] == 0
    assert (
        stats["peptide_scramble"]
        + stats["peptide_random"]
        + stats["mhc_scramble"]
        + stats["mhc_random"]
        + stats["no_mhc_alpha"]
        + stats["no_mhc_beta"]
    ) == 3

    synthetic = [rec for rec in augmented if rec.source.startswith("synthetic_negative_")]
    assert len(synthetic) == 3
    for rec in synthetic:
        assert 50000.0 <= rec.value <= 100000.0
        assert rec.measurement_type == "IC50"
        assert rec.assay_type == rec.source
        assert rec.unit == "nM"
        assert rec.mhc_allele in mhc_sequences
        assert rec.peptide


def test_augment_binding_records_adds_class_i_no_mhc_beta_negatives():
    base = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            mhc_class="I",
            species="human",
            source="iedb",
        ),
        BindingRecord(
            peptide="KPVSKMRMATPLLMQALPM",
            mhc_allele="HLA-DRB1*04:01",
            value=500.0,
            mhc_class="II",
            species="human",
            source="iedb",
        ),
    ]
    mhc_sequences = {"HLA-A*02:01": "AAAA", "HLA-DRB1*04:01": "DDDD"}

    augmented, stats = augment_binding_records_with_synthetic_negatives(
        binding_records=base,
        mhc_sequences=mhc_sequences,
        negative_ratio=0.0,
        weak_value_min_nM=50000.0,
        weak_value_max_nM=100000.0,
        seed=5,
        class_i_no_mhc_beta_ratio=1.0,
    )

    assert stats["added_general"] == 0
    assert stats["no_mhc_beta"] == 1
    assert stats["added"] == 1

    synthetic = [rec for rec in augmented if rec.source == "synthetic_negative_no_mhc_beta"]
    assert len(synthetic) == 1
    rec = synthetic[0]
    assert rec.mhc_class == "I"
    assert rec.mhc_allele == "HLA-A*02:01"
    assert rec.measurement_type == "IC50"
    assert rec.assay_type == "synthetic_negative_no_mhc_beta"


def test_synthetic_binding_negatives_stay_in_parent_binding_task_group():
    base = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            measurement_type="IC50",
            mhc_class="I",
            source="iedb",
        ),
    ]
    mhc_sequences = {"HLA-A*02:01": "A" * 181}

    augmented, _ = augment_binding_records_with_synthetic_negatives(
        binding_records=base,
        mhc_sequences=mhc_sequences,
        negative_ratio=1.0,
        weak_value_min_nM=50000.0,
        weak_value_max_nM=100000.0,
        seed=11,
    )
    dataset = PrestoDataset(
        binding_records=augmented,
        mhc_sequences=mhc_sequences,
    )

    synthetic_samples = [
        sample for sample in dataset.samples
        if (sample.sample_source or "").startswith("synthetic_negative_")
    ]
    assert synthetic_samples
    assert all(sample.assay_group == "binding_ic50" for sample in synthetic_samples)
    assert all((sample.synthetic_kind or "").startswith("synthetic_negative_") for sample in synthetic_samples)


def test_effective_mhc_augmentation_sample_limit_caps_fixed_request():
    assert _effective_mhc_augmentation_sample_limit(60000, 125055, 0.05) == 6253
    assert _effective_mhc_augmentation_sample_limit(1000, 125055, 0.05) == 1000
    assert _effective_mhc_augmentation_sample_limit(1000, 125055, 0.0) == 1000
    assert _effective_mhc_augmentation_sample_limit(1000, 0, 0.05) == 0


def test_augment_elution_records_with_synthetic_negatives():
    base = [
        ElutionRecord(
            peptide="SIINFEKL",
            alleles=["HLA-A*02:01"],
            detected=True,
            mhc_class="I",
            source="iedb",
        )
    ]

    augmented, stats = augment_elution_records_with_synthetic_negatives(
        elution_records=base,
        negative_ratio=1.0,
        seed=7,
    )

    assert len(augmented) == 2
    assert stats["added"] == 1
    assert stats["hard_pair"] == 0
    assert stats["peptide_random_mhc_real"] == 1
    assert stats["peptide_real_mhc_random"] == 0
    assert stats["peptide_random_mhc_random"] == 0
    negatives = [rec for rec in augmented if not rec.detected]
    assert len(negatives) == 1
    assert negatives[0].source == "synthetic_negative"
    assert negatives[0].alleles == ["HLA-A*02:01"]


def test_augment_elution_records_with_allele_mismatch_mode():
    base = [
        ElutionRecord(
            peptide="SIINFEKL",
            alleles=["HLA-A*02:01"],
            detected=True,
            mhc_class="I",
            source="iedb",
        ),
        ElutionRecord(
            peptide="GILGFVFTL",
            alleles=["HLA-B*07:02"],
            detected=True,
            mhc_class="I",
            source="iedb",
        ),
    ]
    source_allele_by_peptide = {rec.peptide: rec.alleles[0] for rec in base}

    augmented, stats = augment_elution_records_with_synthetic_negatives(
        elution_records=base,
        negative_ratio=1.5,  # 3 synthetic samples => random/real, real/random, random/random
        seed=11,
    )

    assert len(augmented) == 5
    assert stats["added"] == 3
    assert stats["hard_pair"] == 0
    assert stats["peptide_random_mhc_real"] == 1
    assert stats["peptide_real_mhc_random"] == 1
    assert stats["peptide_random_mhc_random"] == 1

    negatives = [rec for rec in augmented if rec.source == "synthetic_negative"]
    real_peptide_random_mhc_negatives = [
        rec
        for rec in negatives
        if rec.peptide in source_allele_by_peptide
        and rec.alleles
        and rec.alleles[0] != source_allele_by_peptide[rec.peptide]
    ]
    assert real_peptide_random_mhc_negatives, "expected at least one real-peptide/random-MHC synthetic negative"


def test_augment_elution_records_adds_hard_pair_negatives_from_known_non_presenters():
    base = [
        ElutionRecord(
            peptide="SLLQHLIGL",
            alleles=["HLA-A*02:01"],
            detected=True,
            mhc_class="I",
            source="iedb",
        ),
        ElutionRecord(
            peptide="SLLQHLIGL",
            alleles=["HLA-C*07:01"],
            detected=False,
            mhc_class="I",
            source="iedb",
        ),
    ]

    augmented, stats = augment_elution_records_with_synthetic_negatives(
        elution_records=base,
        negative_ratio=0.0,
        seed=3,
    )

    hard = [rec for rec in augmented if rec.source == "synthetic_negative_hard_pair"]
    assert stats["hard_pair"] >= 1
    assert hard
    assert any(rec.peptide == "SLLQHLIGL" and rec.alleles == ["HLA-C*07:01"] for rec in hard)


def test_augment_processing_records_with_synthetic_negatives():
    base = [
        ProcessingRecord(
            peptide="SIINFEKL",
            flank_n="AAAAAA",
            flank_c="GGGGGG",
            label=1.0,
            mhc_allele="HLA-A*02:01",
            mhc_class="I",
            source="iedb",
        )
    ]

    augmented, stats = augment_processing_records_with_synthetic_negatives(
        processing_records=base,
        negative_ratio=2.0,
        seed=7,
    )

    assert len(augmented) == 3
    assert stats["added"] == 2
    assert stats["flank_shuffle"] == 1
    assert stats["peptide_scramble"] == 1
    negatives = [rec for rec in augmented if rec.source.startswith("synthetic_negative_processing_")]
    assert len(negatives) == 2
    assert all(rec.label == 0.0 for rec in negatives)
    assert all(rec.flank_n and rec.flank_c for rec in negatives)


def test_cascade_binding_negatives_to_downstream_adds_elution_and_tcell():
    binding = [
        BindingRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=75000.0,
            mhc_class="I",
            species="human",
            source="synthetic_negative",
        ),
        BindingRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            value=500.0,
            mhc_class="I",
            species="human",
            source="iedb",
        ),
    ]
    elution = [
        ElutionRecord(
            peptide="GILGFVFTL",
            alleles=["HLA-A*02:01"],
            detected=True,
            mhc_class="I",
            source="iedb",
        )
    ]
    tcell = [
        TCellRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            response=1.0,
            mhc_class="I",
            source="iedb",
        )
    ]

    out_elution, out_tcell, stats = cascade_binding_negatives_to_downstream(
        binding_records=binding,
        elution_records=elution,
        tcell_records=tcell,
        elution_ratio=1.0,
        tcell_ratio=1.0,
        seed=7,
    )

    assert stats["elution_added"] == 1
    assert stats["tcell_added"] == 1

    synthetic_elution = [rec for rec in out_elution if rec.source == "synthetic_negative_from_binding"]
    synthetic_tcell = [rec for rec in out_tcell if rec.source == "synthetic_negative_from_binding"]
    assert len(synthetic_elution) == 1
    assert len(synthetic_tcell) == 1
    assert synthetic_elution[0].detected is False
    assert synthetic_tcell[0].response == 0.0


def test_run_fails_fast_when_strict_mhc_resolution_finds_unresolved(tmp_path, monkeypatch):
    merged_tsv = tmp_path / "merged_deduped.tsv"
    merged_tsv.write_text("placeholder\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    def _fake_load_records_from_merged_tsv(**kwargs):
        binding = [
            BindingRecord(
                peptide="SIINFEKL",
                mhc_allele="HLA-A*99:99",
                value=50.0,
                mhc_class="I",
                source="iedb",
            )
        ]
        return (
            binding,
            [],
            [],
            [],
            [],
            [],
            [],
            {
                "rows_scanned": 1,
                "rows_by_assay": {"binding": 1},
                "rows_by_source": {"iedb": 1},
            },
        )

    monkeypatch.setattr(
        "presto.scripts.train_iedb.load_records_from_merged_tsv",
        _fake_load_records_from_merged_tsv,
    )

    args = argparse.Namespace(
        data_dir=str(tmp_path),
        merged_tsv=str(merged_tsv),
        run_dir=str(run_dir),
        strict_mhc_resolution=True,
        index_csv=None,
    )
    with pytest.raises(RuntimeError, match="Unresolved MHC alleles are present"):
        run(args)

    unresolved_alleles = run_dir / "unresolved_mhc_alleles.csv"
    unresolved_detail = run_dir / "unresolved_mhc_detail.csv"
    assert unresolved_alleles.exists()
    assert unresolved_detail.exists()
    allele_rows = unresolved_alleles.read_text(encoding="utf-8")
    assert "HLA-A*99:99" in allele_rows
    assert "category" in allele_rows.splitlines()[0]
    detail_rows = unresolved_detail.read_text(encoding="utf-8")
    assert "modality,source,allele,category,count" in detail_rows.splitlines()[0]
