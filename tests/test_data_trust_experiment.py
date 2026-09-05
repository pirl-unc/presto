"""Regression tests for the frozen PR #45 audit and aggregation scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "2026-09-04_1121_claude_groove-corrected-baseline"
)


def _load_script(name: str, relative_path: str):
    path = EXPERIMENT_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prediction_row(sample_id: str, y_true: float, y_pred: float) -> dict[str, str]:
    return {
        "task": "binding",
        "sample_id": sample_id,
        "peptide": "SIINFEKL",
        "source_mhc_alleles": "HLA-A*02:01;HLA-B*99:99",
        "resolved_mhc_alleles": "HLA-A*02:01",
        "source": "iedb",
        "qualifier": "0",
        "y_true": str(y_true),
        "y_pred": str(y_pred),
        "y_prob": "",
        "evidence_row_id": f"binding:{sample_id}",
        "assay_iri": "assay:1",
        "reference_iri": "reference:1",
        "pmid": "1",
        "source_sample_label": "",
        "source_sample_attribution": "",
        "source_mapping_category": "single",
        "source_mapping_n_candidates": "1",
        "source_mapping_n_genes": "1",
        "source_mapping_n_flank_pairs": "1",
        "mapping_gene_name": "GENE",
        "mapping_gene_id": "ENSG1",
        "mapping_protein_id": "ENSP1",
        "mapping_transcript_id": "ENST1",
        "mapping_position": "10",
        "mapping_proteome": "UP000005640",
        "mapping_proteome_source": "uniprot",
        "mapping_is_canonical_transcript": "True",
    }


def test_aggregator_pairs_and_emits_metrics_from_canonical_prediction_schema(tmp_path):
    aggregate = _load_script("presto_pr45_aggregate", "code/aggregate.py")
    legacy = [
        _prediction_row("strong", 50.0, 60.0),
        _prediction_row("weak", 5_000.0, 4_000.0),
    ]
    masked = [
        _prediction_row("strong", 50.0, 55.0),
        _prediction_row("weak", 5_000.0, 4_500.0),
    ]

    assert "mhc_alleles" not in aggregate.PAIRING_FIELDS
    assert str(EXPERIMENT_DIR.parents[1]) in sys.path
    assert str(EXPERIMENT_DIR.parents[2]) not in sys.path[:1]
    assert {"source_mhc_alleles", "resolved_mhc_alleles"}.issubset(aggregate.PAIRING_FIELDS)
    assert aggregate._pairing_signature(legacy) == aggregate._pairing_signature(masked)

    metrics = aggregate._metrics(
        masked,
        split="test",
        policy="mask_unresolved",
        seed=42,
        scope="overall",
    )
    output = tmp_path / "condition_metrics.csv"
    aggregate._write_csv(output, [metrics])

    assert metrics["n_all"] == 2
    assert output.is_file()
    assert "exact_rmse" in output.read_text(encoding="utf-8")


def test_data_audit_selects_validated_snapshot_before_loading(tmp_path, monkeypatch):
    audit = _load_script("presto_pr45_data_audit", "analysis/data_audit.py")
    artifact_names = ("observations.parquet", "binding.parquet")
    for name in artifact_names:
        (tmp_path / name).write_bytes(name.encode())
    configured = []
    monkeypatch.setattr(audit, "set_data_dir", lambda path: configured.append(Path(path)))
    monkeypatch.setattr(audit, "configured_hitlist_data_dir", lambda: configured[-1])

    selected = audit._configure_hitlist_snapshot(tmp_path, artifact_names)

    assert selected == tmp_path.resolve()
    assert configured == [tmp_path.resolve()]
