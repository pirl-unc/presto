"""Tests for cross-source deduplication helpers."""

from copy import deepcopy
from pathlib import Path

from presto.data import cross_source_dedup as dedup_mod
from presto.data.cross_source_dedup import (
    CrossSourceDeduplicator,
    UnifiedRecord,
    classify_assay_type,
    parse_iedb_binding,
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
    assert "tcr_evidence" in file_map
    for _, path in file_map.items():
        assert Path(path).exists()

    binding_text = Path(file_map["binding_affinity"]).read_text(encoding="utf-8")
    assert (
        "peptide,mhc_allele,mhc_allele_set,mhc_allele_provenance,"
        "mhc_allele_bag_size,mhc_class,species,antigen_species,source,pmid,doi,reference_text"
    ) in binding_text
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


def test_parse_iedb_binding_populates_apc_name_from_cell_or_tissue(tmp_path: Path):
    path = tmp_path / "iedb_ligand.csv"
    path.write_text(
        "\n".join(
            [
                "Reference,Epitope,MHC Restriction,Antigen Presenting Cell Source,Antigen Presenting Cell Source,Assay,Assay,Assay",
                "PMID,Name,Name,Name,Tissue,Response measured,Method,Qualitative measurement",
                "12345,SIINFEKL,HLA-A*02:01,,PBMC,ligand presentation,mass spectrometry,Positive",
            ]
        ),
        encoding="utf-8",
    )
    recs = list(parse_iedb_binding(path))
    assert len(recs) == 1
    assert recs[0].apc_name == "PBMC"


def test_parse_iedb_binding_preserves_serotype_bag_fields(tmp_path: Path):
    path = tmp_path / "iedb_ligand_serotype.csv"
    path.write_text(
        "\n".join(
            [
                "Reference,Epitope,MHC Restriction,Assay,Assay,Assay",
                "PMID,Name,Name,Response measured,Method,Qualitative measurement",
                "12345,SIINFEKL,HLA-A2,ligand presentation,mass spectrometry,Positive",
            ]
        ),
        encoding="utf-8",
    )
    recs = list(parse_iedb_binding(path))
    assert len(recs) == 1
    rec = recs[0]
    assert rec.mhc_allele == "HLA-A2"
    assert rec.mhc_allele_provenance == "serotype_expanded"
    assert rec.mhc_allele_bag_size is not None and rec.mhc_allele_bag_size >= 1
    assert rec.mhc_allele_set is not None
    assert "HLA-A*02:01" in rec.mhc_allele_set.split(";")


def test_cell_hla_lookup_annotation_and_elution_filter():
    records = [
        UnifiedRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01/HLA-B*07:02",
            pmid="12345",
            source="iedb",
            record_type="binding",
            value=None,
            value_type="ligand presentation",
            assay_method="mass spectrometry",
            response="positive",
            apc_name="PBMC",
        ),
        UnifiedRecord(
            peptide="NLVPMVATV",
            mhc_allele="",
            pmid="12345",
            source="iedb",
            record_type="binding",
            value=None,
            value_type="ligand presentation",
            assay_method="mass spectrometry",
            response="positive",
            apc_name="unknown cell",
        ),
        UnifiedRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            pmid="12345",
            source="iedb",
            record_type="tcell",
            response="positive",
            apc_name="PBMC",
        ),
    ]

    lookup, lookup_stats = dedup_mod._build_elution_cell_hla_lookup(records)
    assert ("12345", "pbmc") in lookup
    assert "HLA-A*02:01" in lookup[("12345", "pbmc")]
    assert lookup_stats["elution_rows_total"] == 2

    ann_stats = dedup_mod._annotate_cell_hla_sets(records, lookup)
    assert ann_stats["cellular_records_with_allele_set"] >= 2
    assert records[2].cell_hla_allele_set is not None

    filtered, filter_stats = dedup_mod._filter_elution_without_cell_hla(records)
    assert len(filtered) == 2
    assert filter_stats["elution_rows_total"] == 2
    assert filter_stats["elution_rows_kept_with_allele_set"] == 1
    assert filter_stats["elution_rows_dropped_missing_allele_set"] == 1


def test_write_merge_funnel_artifacts_writes_tsv(tmp_path: Path):
    out = tmp_path / "merged_deduped.tsv"
    stage_counts = [
        ("loaded", {"elution_ms": 100, "binding_affinity": 20}),
        ("deduped", {"elution_ms": 80, "binding_affinity": 18}),
        ("final", {"elution_ms": 60, "binding_affinity": 18}),
    ]
    files = dedup_mod._write_merge_funnel_artifacts(out, stage_counts)
    assert "tsv" in files
    tsv = Path(files["tsv"])
    assert tsv.exists()
    text = tsv.read_text(encoding="utf-8")
    assert "stage\tassay\tcount" in text
    assert "final\telution_ms\t60" in text


def test_single_pass_cell_hla_annotation_filter_matches_two_pass():
    records = [
        UnifiedRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01/HLA-B*07:02",
            pmid="12345",
            source="iedb",
            record_type="binding",
            value=None,
            value_type="ligand presentation",
            assay_method="mass spectrometry",
            response="positive",
            apc_name="PBMC",
        ),
        UnifiedRecord(
            peptide="LLFGYPVYV",
            mhc_allele="",
            pmid="12345",
            source="iedb",
            record_type="binding",
            value=None,
            value_type="ligand presentation",
            assay_method="mass spectrometry",
            response="positive",
            apc_name="unknown cell",
        ),
        UnifiedRecord(
            peptide="GILGFVFTL",
            mhc_allele="HLA-A*02:01",
            pmid="12345",
            source="iedb",
            record_type="tcell",
            response="positive",
            apc_name="PBMC",
        ),
    ]
    lookup, _ = dedup_mod._build_elution_cell_hla_lookup(records)

    two_pass_records = deepcopy(records)
    two_pass_ann = dedup_mod._annotate_cell_hla_sets(two_pass_records, lookup)
    two_pass_filtered, two_pass_filter = dedup_mod._filter_elution_without_cell_hla(two_pass_records)

    one_pass_records = deepcopy(records)
    one_pass_filtered, one_pass_ann, one_pass_filter = dedup_mod._annotate_and_filter_cell_hla(
        one_pass_records,
        lookup,
    )

    assert len(one_pass_filtered) == len(two_pass_filtered)
    assert [rec.peptide for rec in one_pass_filtered] == [rec.peptide for rec in two_pass_filtered]
    assert one_pass_ann["cellular_records_with_allele_set"] == two_pass_ann["cellular_records_with_allele_set"]
    assert one_pass_filter["elution_rows_total"] == two_pass_filter["elution_rows_total"]
    assert (
        one_pass_filter["elution_rows_kept_with_allele_set"]
        == two_pass_filter["elution_rows_kept_with_allele_set"]
    )


def test_informative_allele_token_filters_generic_class_labels():
    assert dedup_mod._is_informative_allele_token("HLA-A*02:01")
    assert dedup_mod._is_informative_allele_token("DPB1*04:01")
    assert dedup_mod._is_informative_allele_token("H2-Db")
    assert dedup_mod._is_informative_allele_token("H2-IAg7")

    assert not dedup_mod._is_informative_allele_token("HLA class I")
    assert not dedup_mod._is_informative_allele_token("H2 class II")
    assert not dedup_mod._is_informative_allele_token("MHC class I")
    assert not dedup_mod._is_informative_allele_token("HLA-B*27:05 C67S mutant")
    assert not dedup_mod._is_informative_allele_token("HLA-A2")


def test_lookup_drops_generic_class_tokens_from_mixed_allele_string():
    records = [
        UnifiedRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA class I/HLA-A*02:01/H2 class II/H2-Db",
            pmid="12345",
            source="iedb",
            record_type="binding",
            value=None,
            value_type="ligand presentation",
            assay_method="mass spectrometry",
            response="positive",
            apc_name="PBMC",
        ),
    ]
    lookup, _ = dedup_mod._build_elution_cell_hla_lookup(records)
    key = ("12345", "pbmc")
    assert key in lookup
    assert "HLA class I" not in lookup[key]
    assert "H2 class II" not in lookup[key]
    assert "HLA-A*02:01" in lookup[key]
    assert "H2-D*b" in lookup[key]


def test_single_pass_drops_non_elution_cellular_rows_without_known_alleles():
    records = [
        UnifiedRecord(
            peptide="AAAAA",
            mhc_allele="",
            pmid="111",
            source="iedb",
            record_type="tcell",
            response="positive",
            apc_name="PBMC",
        ),
        UnifiedRecord(
            peptide="BBBBB",
            mhc_allele="HLA-A*02:01",
            pmid="111",
            source="iedb",
            record_type="tcell",
            response="positive",
            apc_name="PBMC",
        ),
    ]
    filtered, ann, filt = dedup_mod._annotate_and_filter_cell_hla(records, lookup={})
    assert [rec.peptide for rec in filtered] == ["BBBBB"]
    assert ann["cellular_records_total"] == 2
    assert ann["cellular_records_with_allele_set"] == 1
    assert filt["cellular_rows_dropped_missing_allele_set"] == 1
