"""Tests for data downloaders and deduplication module."""

import csv
import tempfile
import zipfile
from pathlib import Path

import pytest

import presto.data.downloaders as downloaders
from presto.data.downloaders import (
    DATASETS,
    DatasetInfo,
    DownloadManifest,
    DownloadState,
    ReferenceInfo,
    AssayRecord,
    AssayDeduplicator,
    list_datasets,
    get_all_sources,
    get_all_categories,
    get_datasets_by_source,
    get_datasets_by_category,
    deduplicate_binding_file,
    deduplicate_tcell_file,
)


class TestDatasetRegistry:
    """Tests for the dataset registry."""

    def test_datasets_not_empty(self):
        """Registry should contain datasets."""
        assert len(DATASETS) > 0

    def test_all_datasets_have_required_fields(self):
        """All datasets should have required fields."""
        for name, info in DATASETS.items():
            assert info.name == name
            assert info.description
            assert info.url
            assert info.filename
            assert info.source
            assert info.category
            assert info.file_format

    def test_iedb_datasets_require_agreement(self):
        """IEDB datasets (including CEDAR) should require agreement."""
        for name, info in DATASETS.items():
            if info.source == "iedb":
                assert info.requires_agreement, f"{name} should require agreement"

    def test_list_datasets_no_filter(self):
        """list_datasets with no filter returns all datasets."""
        all_datasets = list_datasets()
        assert len(all_datasets) == len(DATASETS)

    def test_list_datasets_by_source(self):
        """list_datasets can filter by source."""
        iedb_datasets = list_datasets(sources=["iedb"])
        assert all(d.source == "iedb" for d in iedb_datasets)
        assert len(iedb_datasets) > 0

    def test_list_datasets_by_category(self):
        """list_datasets can filter by category."""
        tcr_datasets = list_datasets(categories=["tcr"])
        assert all(d.category == "tcr" for d in tcr_datasets)
        assert len(tcr_datasets) > 0

    def test_get_all_sources(self):
        """get_all_sources returns all unique sources."""
        sources = get_all_sources()
        assert "iedb" in sources
        assert "vdjdb" in sources
        assert "imgt" in sources
        assert "ipd_mhc" in sources
        assert "10x" in sources

    def test_get_all_categories(self):
        """get_all_categories returns all unique categories."""
        categories = get_all_categories()
        assert "binding" in categories
        assert "tcell" in categories
        assert "tcr" in categories
        assert "mhc_sequence" in categories
        # New categories
        assert "vdj_genes" in categories

    def test_get_datasets_by_source(self):
        """get_datasets_by_source returns correct datasets."""
        imgt = get_datasets_by_source("imgt")
        assert len(imgt) > 0
        assert all(d.source == "imgt" for d in imgt)

    def test_get_datasets_by_category(self):
        """get_datasets_by_category returns correct datasets."""
        binding = get_datasets_by_category("binding")
        assert len(binding) > 0
        assert all(d.category == "binding" for d in binding)


class TestDownloadValidation:
    """Tests for download-time content validation."""

    def test_stcrdab_download_fails_on_html_content(self, tmp_path, monkeypatch):
        """STCRDab download should fail fast when server returns HTML."""

        def fake_download(url, dest, progress_callback=None, max_redirects=5):
            dest.write_text("<!DOCTYPE html><html><body>not data</body></html>", encoding="utf-8")
            return dest.stat().st_size

        monkeypatch.setattr(downloaders, "_download_with_progress", fake_download)
        state = downloaders.download_dataset(
            "stcrdab",
            data_dir=tmp_path,
            force=True,
            agree_terms=False,
            verbose=False,
        )

        assert state.status == "failed"
        assert state.error is not None
        assert "html" in state.error.lower()

    def test_iedb_cedar_extract_writes_source_aliases(self, tmp_path, monkeypatch):
        """CEDAR zip extraction should emit a cedar-prefixed stable CSV alias."""

        def fake_download(url, dest, progress_callback=None, max_redirects=5):
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("tcell_full_v3.csv", "peptide,response\nAAA,positive\n")
            return dest.stat().st_size

        monkeypatch.setattr(downloaders, "_download_with_progress", fake_download)

        state = downloaders.download_dataset(
            "iedb_cedar_tcell",
            data_dir=tmp_path,
            force=True,
            agree_terms=True,
            verbose=False,
        )
        assert state.status == "completed"
        assert (tmp_path / "iedb" / "tcell_full_v3.csv").exists()
        alias = tmp_path / "iedb" / "cedar_tcell_full_v3.csv"
        assert alias.exists()

        alias.unlink()
        state2 = downloaders.download_dataset(
            "iedb_cedar_tcell",
            data_dir=tmp_path,
            force=False,
            agree_terms=True,
            verbose=False,
        )
        assert state2.status == "completed"
        assert alias.exists()


class TestDownloadManifest:
    """Tests for DownloadManifest."""

    def test_create_empty_manifest(self):
        """Create an empty manifest."""
        manifest = DownloadManifest()
        assert manifest.version == "1.0"
        assert manifest.downloads == {}

    def test_manifest_save_load(self):
        """Manifest can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            # Create and save
            manifest = DownloadManifest(data_dir=tmpdir)
            manifest.downloads["test"] = DownloadState(
                dataset="test",
                status="completed",
                url="http://example.com",
                local_path="/path/to/file",
            )
            manifest.save(manifest_path)

            # Load
            loaded = DownloadManifest.load(manifest_path)
            assert loaded.data_dir == tmpdir
            assert "test" in loaded.downloads
            assert loaded.downloads["test"].status == "completed"

    def test_manifest_load_nonexistent(self):
        """Loading nonexistent manifest returns empty manifest."""
        manifest = DownloadManifest.load(Path("/nonexistent/path"))
        assert manifest.downloads == {}


class TestReferenceInfo:
    """Tests for ReferenceInfo."""

    def test_key_with_pubmed(self):
        """Key prioritizes PubMed ID."""
        ref = ReferenceInfo(pubmed_id="12345678", doi="10.1000/test")
        assert ref.key() == "pmid:12345678"

    def test_key_with_doi_only(self):
        """Key falls back to DOI."""
        ref = ReferenceInfo(doi="10.1000/test")
        assert ref.key() == "doi:10.1000/test"

    def test_key_with_title_only(self):
        """Key falls back to title hash."""
        ref = ReferenceInfo(title="Test Paper Title")
        key = ref.key()
        assert key.startswith("title:")

    def test_key_unknown(self):
        """Key returns unknown when no identifiers."""
        ref = ReferenceInfo()
        assert ref.key() == "unknown"


class TestAssayRecord:
    """Tests for AssayRecord."""

    def test_dedup_key(self):
        """Deduplication key combines peptide, allele, value_type."""
        rec = AssayRecord(
            peptide="SIINFEKL",
            mhc_allele="HLA-A*02:01",
            value=50.0,
            value_type="IC50",
        )
        assert rec.dedup_key() == "SIINFEKL|HLA-A*02:01|IC50"


class TestAssayDeduplicator:
    """Tests for AssayDeduplicator."""

    def test_no_duplicates(self):
        """Records with no duplicates pass through unchanged."""
        records = [
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=10.0, value_type="IC50"),
            AssayRecord(peptide="BBB", mhc_allele="HLA-A", value=20.0, value_type="IC50"),
            AssayRecord(peptide="AAA", mhc_allele="HLA-B", value=30.0, value_type="IC50"),
        ]
        deduplicator = AssayDeduplicator()
        result = deduplicator.deduplicate(records)
        assert len(result) == 3

    def test_same_reference_aggregation_median(self):
        """Multiple values from same reference are aggregated by median."""
        ref = ReferenceInfo(pubmed_id="12345")
        records = [
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=10.0, value_type="IC50", reference=ref),
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=20.0, value_type="IC50", reference=ref),
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=30.0, value_type="IC50", reference=ref),
        ]
        deduplicator = AssayDeduplicator(aggregate_same_ref="median")
        result = deduplicator.deduplicate(records)
        assert len(result) == 1
        assert result[0].value == 20.0  # median of 10, 20, 30

    def test_same_reference_aggregation_mean(self):
        """Multiple values from same reference can be aggregated by mean."""
        ref = ReferenceInfo(pubmed_id="12345")
        records = [
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=10.0, value_type="IC50", reference=ref),
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=20.0, value_type="IC50", reference=ref),
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=30.0, value_type="IC50", reference=ref),
        ]
        deduplicator = AssayDeduplicator(aggregate_same_ref="mean")
        result = deduplicator.deduplicate(records)
        assert len(result) == 1
        assert result[0].value == 20.0  # mean of 10, 20, 30

    def test_different_references_prefer_recent(self):
        """When multiple references, prefer more recent publication."""
        ref_old = ReferenceInfo(pubmed_id="11111", year=2010)
        ref_new = ReferenceInfo(pubmed_id="22222", year=2020)
        records = [
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=10.0, value_type="IC50", reference=ref_old),
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=20.0, value_type="IC50", reference=ref_new),
        ]
        deduplicator = AssayDeduplicator(prefer_recent=True)
        result = deduplicator.deduplicate(records)
        assert len(result) == 1
        assert result[0].value == 20.0  # from 2020

    def test_different_references_prefer_exact(self):
        """When multiple references, prefer exact measurements over bounds."""
        ref1 = ReferenceInfo(pubmed_id="11111")
        ref2 = ReferenceInfo(pubmed_id="22222")
        records = [
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=10.0, value_type="IC50",
                       qualifier=-1, reference=ref1),  # <10
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=20.0, value_type="IC50",
                       qualifier=0, reference=ref2),  # =20
        ]
        deduplicator = AssayDeduplicator(prefer_exact=True, prefer_recent=False)
        result = deduplicator.deduplicate(records)
        assert len(result) == 1
        assert result[0].qualifier == 0  # exact measurement

    def test_stats(self):
        """Deduplicator tracks statistics."""
        ref = ReferenceInfo(pubmed_id="12345")
        records = [
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=10.0, value_type="IC50", reference=ref),
            AssayRecord(peptide="AAA", mhc_allele="HLA-A", value=20.0, value_type="IC50", reference=ref),
        ]
        deduplicator = AssayDeduplicator()
        deduplicator.deduplicate(records)
        stats = deduplicator.get_stats()
        assert stats['total_input'] == 2
        assert stats['total_output'] == 1
        assert stats['duplicates_removed'] == 1
        assert stats['by_same_reference'] == 1


class TestDeduplicateBindingFile:
    """Tests for deduplicate_binding_file function."""

    def test_deduplicate_csv_file(self):
        """Can deduplicate a CSV binding file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.tsv"

            # Create input file with duplicates
            with open(input_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['peptide', 'mhc_allele', 'value', 'measurement_type', 'pubmed_id'])
                writer.writerow(['SIINFEKL', 'HLA-A*02:01', '50', 'IC50', '12345'])
                writer.writerow(['SIINFEKL', 'HLA-A*02:01', '60', 'IC50', '12345'])  # same ref
                writer.writerow(['GILGFVFTL', 'HLA-A*02:01', '100', 'IC50', '67890'])

            stats = deduplicate_binding_file(input_path, output_path, verbose=False)

            assert output_path.exists()
            assert stats['total_input'] == 3
            assert stats['total_output'] == 2

            # Verify output
            with open(output_path) as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
                assert len(rows) == 2
                peptides = {row['peptide'] for row in rows}
                assert peptides == {'SIINFEKL', 'GILGFVFTL'}


class TestDeduplicateTcellFile:
    """Tests for deduplicate_tcell_file function."""

    def test_deduplicate_tcell_majority_vote(self):
        """T-cell data uses majority vote for conflicting responses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            output_path = Path(tmpdir) / "output.tsv"

            # Create input file with conflicting responses from same reference
            with open(input_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['peptide', 'mhc_allele', 'response', 'pubmed_id'])
                writer.writerow(['SIINFEKL', 'HLA-A*02:01', 'positive', '12345'])
                writer.writerow(['SIINFEKL', 'HLA-A*02:01', 'positive', '12345'])
                writer.writerow(['SIINFEKL', 'HLA-A*02:01', 'negative', '12345'])

            stats = deduplicate_tcell_file(input_path, output_path, verbose=False)

            assert output_path.exists()

            # Verify output - majority vote should be positive
            with open(output_path) as f:
                reader = csv.DictReader(f, delimiter='\t')
                rows = list(reader)
                assert len(rows) == 1
                assert rows[0]['response'] == '1'  # positive (2 pos vs 1 neg)


class TestNewDatasets:
    """Tests for newly added datasets."""

    def test_cedar_datasets_exist(self):
        """CEDAR datasets should be registered under IEDB source."""
        cedar_names = [name for name in DATASETS if "cedar" in name]
        assert len(cedar_names) >= 3  # tcell, bcell, mhc_ligand
        for name in cedar_names:
            assert DATASETS[name].source == "iedb"  # CEDAR is part of IEDB

    def test_ipd_mhc_datasets_exist(self):
        """IPD-MHC datasets should be registered."""
        ipd_names = [name for name in DATASETS if name.startswith("ipd_mhc")]
        assert len(ipd_names) >= 1
        for name in ipd_names:
            assert DATASETS[name].source == "ipd_mhc"

    def test_10x_datasets_exist(self):
        """10x Genomics datasets should be registered."""
        sc10x_names = [name for name in DATASETS if name.startswith("10x_")]
        assert len(sc10x_names) >= 1
        for name in sc10x_names:
            assert DATASETS[name].source == "10x"

    def test_pird_datasets_exist(self):
        """PIRD datasets should be registered."""
        pird_names = [name for name in DATASETS if name.startswith("pird_")]
        assert len(pird_names) >= 1

    def test_imgt_gene_datasets_exist(self):
        """IMGT V/D/J gene datasets should be registered."""
        vdj_names = [name for name in DATASETS if DATASETS[name].category == "vdj_genes"]
        assert len(vdj_names) >= 1

    def test_stcrdab_exists(self):
        """STCRDab dataset should be registered."""
        assert "stcrdab" in DATASETS
        assert DATASETS["stcrdab"].source == "stcrdab"
