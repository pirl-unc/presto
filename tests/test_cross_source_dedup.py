"""Tests for cross-source deduplication helpers."""

from pathlib import Path

from presto.data.cross_source_dedup import (
    CrossSourceDeduplicator,
    UnifiedRecord,
    classify_assay_type,
    write_assay_csvs,
)


def test_fuzzy_reference_dedup_collapses_cross_source_duplicates():
    records = [
        UnifiedRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            source="iedb",
            record_type="binding",
            value_type="IC50",
            qualifier=0,
            reference_text="Smith et al 2020 Journal of Immunology 12(3):123-130",
        ),
        UnifiedRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            source="cedar",
            record_type="binding",
            value_type="IC50",
            qualifier=0,
            reference_text="Smith et al., 2020 J Immunol 12 3 123 130",
        ),
    ]

    deduper = CrossSourceDeduplicator(reference_similarity_threshold=0.8)
    deduped = deduper.deduplicate(records)
    stats = deduper.get_stats()

    assert len(deduped) == 1
    assert stats["cross_source_duplicates"] == 1
    assert stats["fuzzy_reference_duplicates"] == 1


def test_tcell_dedup_key_keeps_positive_and_negative_rows_separate():
    records = [
        UnifiedRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            source="iedb",
            record_type="tcell",
            response="positive",
            pmid="12345678",
        ),
        UnifiedRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            source="cedar",
            record_type="tcell",
            response="negative",
            pmid="12345678",
        ),
    ]
    deduped = CrossSourceDeduplicator().deduplicate(records)
    assert len(deduped) == 2


def test_write_assay_csvs_emits_one_file_per_assay_type(tmp_path: Path):
    records = [
        UnifiedRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            source="iedb",
            record_type="binding",
            value_type="IC50",
            value=42.0,
        ),
        UnifiedRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            source="iedb",
            record_type="tcell",
            response="positive",
        ),
        UnifiedRecord(
            peptide="NLVPMVATV",
            mhc_allele="HLA-A*02:01",
            source="vdjdb",
            record_type="tcr",
            cdr3_beta="CASSIRSSYEQYF",
        ),
    ]
    file_map = write_assay_csvs(records, tmp_path)

    assert "binding_affinity" in file_map
    assert "tcell_response" in file_map
    assert "tcr_pmhc" in file_map
    for _, path in file_map.items():
        assert Path(path).exists()

    binding_text = Path(file_map["binding_affinity"]).read_text(encoding="utf-8")
    assert "peptide,mhc_allele,mhc_class,species,antigen_species,source,pmid,doi,reference_text" in binding_text
    assert "SIINFEKL" in binding_text


def test_classify_assay_type_routes_binding_and_elution_rows():
    binding = UnifiedRecord(
        peptide="SIINFEKL",
        mhc_allele="HLA-A*02:01",
        source="iedb",
        record_type="binding",
        value=123.0,
        value_type="IC50",
    )
    elution = UnifiedRecord(
        peptide="SIINFEKL",
        mhc_allele="HLA-A*02:01",
        source="iedb",
        record_type="binding",
        value=None,
        value_type="ligand presentation",
    )
    assert classify_assay_type(binding) == "binding_affinity"
    assert classify_assay_type(elution) == "elution_ms"
