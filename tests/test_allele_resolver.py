"""Tests for allele resolver module."""

import pytest
import tempfile
import os

from presto.data.allele_resolver import (
    AlleleResolver,
    AlleleRecord,
    normalize_allele_name,
    infer_mhc_class,
    infer_gene,
    normalize_processing_species_label,
    infer_processing_species_from_allele,
)


class TestNormalizeAlleleName:
    """Tests for allele name normalization."""

    def test_already_normalized(self):
        """Standard format stays unchanged."""
        assert normalize_allele_name("HLA-A*02:01") == "HLA-A*02:01"

    def test_missing_hla_prefix(self):
        """HLA- prefix is added for human alleles."""
        assert normalize_allele_name("A*02:01") == "HLA-A*02:01"

    def test_compact_format(self):
        """Compact format (A0201) is expanded."""
        assert normalize_allele_name("A0201") == "HLA-A*02:01"
        assert normalize_allele_name("B0702") == "HLA-B*07:02"

    def test_short_format(self):
        """Short format (A2) is expanded."""
        assert normalize_allele_name("A2") == "HLA-A*02"
        assert normalize_allele_name("B7") == "HLA-B*07"

    def test_uppercase_conversion(self):
        """Lowercase is converted to uppercase."""
        assert normalize_allele_name("hla-a*02:01") == "HLA-A*02:01"
        assert normalize_allele_name("a0201") == "HLA-A*02:01"

    def test_whitespace_stripped(self):
        """Whitespace is stripped."""
        assert normalize_allele_name("  HLA-A*02:01  ") == "HLA-A*02:01"

    def test_mouse_alleles_preserved(self):
        """Mouse H2- alleles don't get HLA- prefix."""
        assert normalize_allele_name("H2-Kb") == "H2-KB"
        assert normalize_allele_name("H2-Db") == "H2-DB"

    def test_macaque_alleles_preserved(self):
        """Macaque MAMU- alleles don't get HLA- prefix."""
        assert normalize_allele_name("Mamu-A*01") == "MAMU-A*01"

    def test_non_human_prefix_preserved(self):
        """Non-human species prefixes are preserved."""
        assert normalize_allele_name("Aona-DQA1*27:01") == "AONA-DQA1*27:01"

    def test_class_ii_drb1(self):
        """Class II DRB1 alleles normalized correctly."""
        assert normalize_allele_name("DRB1*01:01") == "HLA-DRB1*01:01"
        assert normalize_allele_name("HLA-DRB1*01:01") == "HLA-DRB1*01:01"


class TestInferMHCClass:
    """Tests for MHC class inference."""

    def test_class_i_alleles(self):
        """Class I alleles (A, B, C) return 'I'."""
        assert infer_mhc_class("HLA-A*02:01") == "I"
        assert infer_mhc_class("HLA-B*07:02") == "I"
        assert infer_mhc_class("HLA-C*04:01") == "I"

    def test_class_ii_dr(self):
        """DR alleles return 'II'."""
        assert infer_mhc_class("HLA-DRA*01:01") == "II"
        assert infer_mhc_class("HLA-DRB1*01:01") == "II"

    def test_class_ii_dq(self):
        """DQ alleles return 'II'."""
        assert infer_mhc_class("HLA-DQA1*01:01") == "II"
        assert infer_mhc_class("HLA-DQB1*02:01") == "II"

    def test_class_ii_dp(self):
        """DP alleles return 'II'."""
        assert infer_mhc_class("HLA-DPA1*01:01") == "II"
        assert infer_mhc_class("HLA-DPB1*01:01") == "II"

    def test_mouse_class_i(self):
        """Mouse Class I (H2-K, H2-D) return 'I'."""
        assert infer_mhc_class("H2-Kb") == "I"
        assert infer_mhc_class("H2-Db") == "I"

    def test_mouse_class_ii(self):
        """Mouse Class II (H2-Ab, H2-Eb) return 'II'."""
        assert infer_mhc_class("H2-Ab") == "II"
        assert infer_mhc_class("H2-Eb") == "II"

    def test_case_insensitive(self):
        """Class inference is case insensitive."""
        assert infer_mhc_class("hla-drb1*01:01") == "II"
        assert infer_mhc_class("HLA-A*02:01") == "I"


class TestInferGene:
    """Tests for gene extraction."""

    def test_class_i_genes(self):
        """Extract gene from Class I alleles."""
        assert infer_gene("HLA-A*02:01") == "A"
        assert infer_gene("HLA-B*07:02") == "B"
        assert infer_gene("HLA-C*04:01") == "C"

    def test_class_ii_genes(self):
        """Extract gene from Class II alleles."""
        assert infer_gene("HLA-DRB1*01:01") == "DRB1"
        assert infer_gene("HLA-DQA1*01:01") == "DQA1"
        assert infer_gene("HLA-DPB1*02:01") == "DPB1"

    def test_mouse_genes(self):
        """Extract gene from mouse alleles."""
        assert infer_gene("H2-Kb") == "KB"

    def test_normalizes_first(self):
        """Gene extraction normalizes the allele first."""
        assert infer_gene("a0201") == "A"
        assert infer_gene("drb1*01:01") == "DRB1"

    def test_non_human_prefix_gene(self):
        """Extract gene from non-human prefix alleles."""
        assert infer_gene("Aona-DQA1*27:01") == "DQA1"


class TestProcessingSpeciesNormalization:
    """Tests for coarse processing-species bucket normalization."""

    def test_normalize_processing_species_label(self):
        assert normalize_processing_species_label("Homo sapiens (human)") == "human"
        assert normalize_processing_species_label("Mus musculus C57BL/6") == "murine"
        assert normalize_processing_species_label("Rattus norvegicus") == "murine"
        assert normalize_processing_species_label("Macaca mulatta (rhesus macaque)") == "nhp"
        assert normalize_processing_species_label("Pan troglodytes (chimpanzee)") == "nhp"
        assert normalize_processing_species_label("Gallus gallus") == "bird"
        assert normalize_processing_species_label("Bos taurus") == "other_mammal"
        assert normalize_processing_species_label("Salmo salar") == "fish"

    def test_infer_processing_species_from_allele(self):
        assert infer_processing_species_from_allele("HLA-A*02:01") == "human"
        assert infer_processing_species_from_allele("H2-Kd") == "murine"
        assert infer_processing_species_from_allele("Mamu-A*01:01") == "nhp"
        assert infer_processing_species_from_allele("Aona-DQA1*27:01") == "nhp"
        assert infer_processing_species_from_allele("BoLA-2*01:01") == "other_mammal"
        assert infer_processing_species_from_allele("Gaga-BF1*01:01") == "bird"


class TestAlleleResolver:
    """Tests for AlleleResolver class."""

    def test_empty_resolver(self):
        """Empty resolver returns None for unknown alleles."""
        resolver = AlleleResolver()
        assert resolver.resolve("HLA-A*02:01") is None
        assert resolver.get_sequence("HLA-A*02:01") is None

    def test_has_beta2m(self):
        """Resolver has beta2m sequence by default."""
        resolver = AlleleResolver()
        assert resolver.beta2m is not None
        assert len(resolver.beta2m) > 50

    def test_get_mhc_class_without_sequence(self):
        """Can infer MHC class even without loaded sequences."""
        resolver = AlleleResolver()
        assert resolver.get_mhc_class("HLA-A*02:01") == "I"
        assert resolver.get_mhc_class("HLA-DRB1*01:01") == "II"

    def test_load_imgt_fasta(self):
        """Load sequences from IMGT-formatted FASTA."""
        # Create a temporary FASTA file
        fasta_content = """>HLA:HLA00001 A*01:01:01:01 365 bp
MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQ
>HLA:HLA00002 A*02:01:01:01 365 bp
MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFSTSVSRPGRGEPRFISVGYVDDTQFVRFDSDAASQKMEPRAPWIEQ
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            fasta_path = f.name

        try:
            resolver = AlleleResolver(imgt_fasta=fasta_path)
            assert len(resolver.records) == 2

            # Check resolution
            record = resolver.resolve("HLA-A*01:01:01:01")
            assert record is not None
            assert record.gene == "A"
            assert record.mhc_class == "I"

            # Check alias resolution (truncated version)
            record = resolver.resolve("HLA-A*01:01")
            assert record is not None

            # Check sequence retrieval
            seq = resolver.get_sequence("HLA-A*02:01")
            assert seq is not None
            assert "MAV" in seq
        finally:
            os.unlink(fasta_path)

    def test_load_ipd_fasta(self):
        """Load sequences from IPD-MHC-formatted FASTA."""
        fasta_content = """>IPD-MHC:NHP00001 Aona-DQA1*27:01 73 bp
DHVAAYGINLYQSYGLSGQYTHEFDGDEEFYVDLGRKETVWRLPVFSKFAGFDPQGALTN
>IPD-MHC:NHP00002 Mamu-A1*001:01 90 bp
MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRF
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "ipd_mhc_prot.fasta")
            with open(fasta_path, "w") as f:
                f.write(fasta_content)

            resolver = AlleleResolver(ipd_mhc_dir=tmpdir)
            assert len(resolver.records) == 2

            record = resolver.resolve("Aona-DQA1*27:01")
            assert record is not None
            assert record.gene == "DQA1"
            assert record.mhc_class == "II"

    def test_list_alleles(self):
        """List available alleles with filters."""
        fasta_content = """>HLA:HLA00001 A*01:01:01:01 365 bp
MAVMAPRTLLLLLSGALALTQTWAG
>HLA:HLA00002 A*02:01:01:01 365 bp
MAVMAPRTLLLLLSGALALTQTWAG
>HLA:HLA00003 DRB1*01:01:01 365 bp
MGSGWVPWVVALLVNLTRLDSSMTQ
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            fasta_path = f.name

        try:
            resolver = AlleleResolver(imgt_fasta=fasta_path)

            # List all
            all_alleles = resolver.list_alleles()
            assert len(all_alleles) == 3

            # Filter by gene
            a_alleles = resolver.list_alleles(gene="A")
            assert len(a_alleles) == 2

            # Filter by class
            class_ii = resolver.list_alleles(mhc_class="II")
            assert len(class_ii) == 1
        finally:
            os.unlink(fasta_path)

    def test_sequence_similarity(self):
        """Test sequence similarity calculation."""
        resolver = AlleleResolver()

        # Identical sequences
        sim = resolver._sequence_similarity("ACDEFG", "ACDEFG")
        assert sim == 1.0

        # Completely different
        sim = resolver._sequence_similarity("AAAAAA", "FFFFFF")
        assert sim == 0.0

        # Partial match
        sim = resolver._sequence_similarity("ACDEFG", "ACDXXX")
        assert 0.4 < sim < 0.6

    def test_nearest_alleles(self):
        """Find nearest alleles by sequence similarity."""
        fasta_content = """>HLA:HLA00001 A*01:01:01:01 365 bp
MAVMAPRTL
>HLA:HLA00002 A*02:01:01:01 365 bp
MAVMAPRTX
>HLA:HLA00003 B*07:02:01 365 bp
XXXXXXXXX
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            fasta_path = f.name

        try:
            resolver = AlleleResolver(imgt_fasta=fasta_path)

            # Query similar to A*01:01
            results = resolver.nearest("MAVMAPRTL", top_k=2)
            assert len(results) == 2
            # First result should be exact match
            assert results[0][0] == "HLA-A*01:01:01:01"
            assert results[0][1] == 1.0
        finally:
            os.unlink(fasta_path)


class TestAlleleRecord:
    """Tests for AlleleRecord dataclass."""

    def test_record_creation(self):
        """Create an AlleleRecord."""
        record = AlleleRecord(
            name="HLA-A*02:01",
            sequence="MAVMAPRTL",
            gene="A",
            mhc_class="I",
        )
        assert record.name == "HLA-A*02:01"
        assert record.species == "human"  # default

    def test_record_with_species(self):
        """Create record with non-human species."""
        record = AlleleRecord(
            name="H2-Kb",
            sequence="MAVMAPRTL",
            gene="K",
            mhc_class="I",
            species="mouse",
        )
        assert record.species == "mouse"
