"""Tests for vocab module - pins down vocabulary constants and compatibility matrices."""

import pytest


class TestAAVocab:
    """Test amino acid vocabulary."""

    def test_aa_vocab_has_standard_amino_acids(self):
        from presto.data.vocab import AA_VOCAB
        standard = set("ACDEFGHIKLMNPQRSTVWY")
        assert standard.issubset(set(AA_VOCAB))

    def test_aa_vocab_has_special_tokens(self):
        from presto.data.vocab import AA_VOCAB
        # PAD=0, UNK, BOS, EOS should be present
        assert AA_VOCAB[0] == "<PAD>"
        assert "<UNK>" in AA_VOCAB
        assert "<BOS>" in AA_VOCAB
        assert "<EOS>" in AA_VOCAB

    def test_aa_to_idx_mapping(self):
        from presto.data.vocab import AA_TO_IDX, AA_VOCAB
        for i, aa in enumerate(AA_VOCAB):
            assert AA_TO_IDX[aa] == i

    def test_idx_to_aa_mapping(self):
        from presto.data.vocab import IDX_TO_AA, AA_VOCAB
        for i, aa in enumerate(AA_VOCAB):
            assert IDX_TO_AA[i] == aa


class TestChainTypes:
    """Test chain type vocabulary."""

    def test_chain_types_full_complete(self):
        from presto.data.vocab import CHAIN_TYPES_FULL
        expected = {"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"}
        assert set(CHAIN_TYPES_FULL) == expected

    def test_chain_types_cdr3_complete(self):
        from presto.data.vocab import CHAIN_TYPES_CDR3
        expected = {"TRA_CDR3", "TRB_CDR3", "TRG_CDR3", "TRD_CDR3", "IGH_CDR3", "IGK_CDR3", "IGL_CDR3"}
        assert set(CHAIN_TYPES_CDR3) == expected

    def test_chain_types_combined(self):
        from presto.data.vocab import CHAIN_TYPES, CHAIN_TYPES_FULL, CHAIN_TYPES_CDR3
        assert len(CHAIN_TYPES) == len(CHAIN_TYPES_FULL) + len(CHAIN_TYPES_CDR3)

    def test_chain_to_idx(self):
        from presto.data.vocab import CHAIN_TO_IDX, CHAIN_TYPES
        for i, ct in enumerate(CHAIN_TYPES):
            assert CHAIN_TO_IDX[ct] == i

    def test_cdr3_to_full_mapping(self):
        from presto.data.vocab import CDR3_TO_FULL, is_cdr3_only, get_base_chain_type
        assert CDR3_TO_FULL["TRA_CDR3"] == "TRA"
        assert CDR3_TO_FULL["IGH_CDR3"] == "IGH"
        assert is_cdr3_only("TRB_CDR3") is True
        assert is_cdr3_only("TRB") is False
        assert get_base_chain_type("TRA_CDR3") == "TRA"
        assert get_base_chain_type("TRA") == "TRA"


class TestCellTypes:
    """Test cell type vocabulary."""

    def test_cell_types_complete(self):
        from presto.data.vocab import CELL_TYPES
        expected = {"CD4_T", "CD8_T", "ab_T", "gd_T", "B_cell"}
        assert set(CELL_TYPES) == expected

    def test_cell_to_idx(self):
        from presto.data.vocab import CELL_TO_IDX, CELL_TYPES
        for i, ct in enumerate(CELL_TYPES):
            assert CELL_TO_IDX[ct] == i


class TestMHCTypes:
    """Test MHC type vocabulary."""

    def test_mhc_types_complete(self):
        from presto.data.vocab import MHC_TYPES
        expected = {"MHC_I", "MHC_II", "HLA_E", "HLA_F", "HLA_G"}
        assert set(MHC_TYPES) == expected

    def test_mhc_to_idx(self):
        from presto.data.vocab import MHC_TO_IDX, MHC_TYPES
        for i, mt in enumerate(MHC_TYPES):
            assert MHC_TO_IDX[mt] == i


class TestChainSpecies:
    """Test 6-class chain species vocabulary."""

    def test_chain_species_complete(self):
        from presto.data.vocab import CHAIN_SPECIES_CATEGORIES
        expected = {"human", "nhp", "murine", "other_mammal", "bird", "other_vertebrate"}
        assert set(CHAIN_SPECIES_CATEGORIES) == expected

    def test_chain_species_to_idx(self):
        from presto.data.vocab import CHAIN_SPECIES_TO_IDX, CHAIN_SPECIES_CATEGORIES
        for i, sp in enumerate(CHAIN_SPECIES_CATEGORIES):
            assert CHAIN_SPECIES_TO_IDX[sp] == i

    def test_idx_to_chain_species(self):
        from presto.data.vocab import IDX_TO_CHAIN_SPECIES, CHAIN_SPECIES_CATEGORIES
        for i, sp in enumerate(CHAIN_SPECIES_CATEGORIES):
            assert IDX_TO_CHAIN_SPECIES[i] == sp


class TestValidChainCell:
    """Test chain-cell validity matrix."""

    def test_alpha_chains_with_t_cells(self):
        from presto.data.vocab import VALID_CHAIN_CELL
        # TRA can appear in CD4_T, CD8_T, ab_T
        assert VALID_CHAIN_CELL["TRA"] == {"CD4_T", "CD8_T", "ab_T"}
        assert VALID_CHAIN_CELL["TRB"] == {"CD4_T", "CD8_T", "ab_T"}

    def test_gamma_delta_chains(self):
        from presto.data.vocab import VALID_CHAIN_CELL
        assert VALID_CHAIN_CELL["TRG"] == {"gd_T"}
        assert VALID_CHAIN_CELL["TRD"] == {"gd_T"}

    def test_ig_chains_with_b_cells(self):
        from presto.data.vocab import VALID_CHAIN_CELL
        assert VALID_CHAIN_CELL["IGH"] == {"B_cell"}
        assert VALID_CHAIN_CELL["IGK"] == {"B_cell"}
        assert VALID_CHAIN_CELL["IGL"] == {"B_cell"}

    def test_cdr3_chains_same_as_full(self):
        from presto.data.vocab import VALID_CHAIN_CELL
        # CDR3 variants should have same cell type mappings
        assert VALID_CHAIN_CELL["TRA_CDR3"] == VALID_CHAIN_CELL["TRA"]
        assert VALID_CHAIN_CELL["TRB_CDR3"] == VALID_CHAIN_CELL["TRB"]
        assert VALID_CHAIN_CELL["IGH_CDR3"] == VALID_CHAIN_CELL["IGH"]


class TestCellMHCCompatibility:
    """Test cell-MHC compatibility matrix (biological constraints)."""

    def test_cd4_binds_mhc_ii(self):
        from presto.data.vocab import CELL_MHC_COMPATIBILITY
        assert CELL_MHC_COMPATIBILITY["CD4_T"] == {"MHC_II"}

    def test_cd8_binds_mhc_i_and_hla_e(self):
        from presto.data.vocab import CELL_MHC_COMPATIBILITY
        assert CELL_MHC_COMPATIBILITY["CD8_T"] == {"MHC_I", "HLA_E"}

    def test_ab_t_binds_all_classical(self):
        from presto.data.vocab import CELL_MHC_COMPATIBILITY
        # ab_T with unknown restriction can bind any
        assert CELL_MHC_COMPATIBILITY["ab_T"] == {"MHC_I", "MHC_II", "HLA_E", "HLA_F", "HLA_G"}

    def test_gd_t_does_not_bind_classical_pmhc(self):
        from presto.data.vocab import CELL_MHC_COMPATIBILITY
        assert CELL_MHC_COMPATIBILITY["gd_T"] == set()

    def test_b_cell_does_not_bind_pmhc(self):
        from presto.data.vocab import CELL_MHC_COMPATIBILITY
        assert CELL_MHC_COMPATIBILITY["B_cell"] == set()


class TestCompatibilityHelpers:
    """Test helper functions for checking compatibility."""

    def test_is_valid_chain_cell_true(self):
        from presto.data.vocab import is_valid_chain_cell
        assert is_valid_chain_cell("TRA", "CD4_T") is True
        assert is_valid_chain_cell("TRB", "CD8_T") is True
        assert is_valid_chain_cell("IGH", "B_cell") is True

    def test_is_valid_chain_cell_false(self):
        from presto.data.vocab import is_valid_chain_cell
        assert is_valid_chain_cell("TRA", "B_cell") is False
        assert is_valid_chain_cell("IGH", "CD4_T") is False
        assert is_valid_chain_cell("TRG", "CD4_T") is False

    def test_is_compatible_cell_mhc_true(self):
        from presto.data.vocab import is_compatible_cell_mhc
        assert is_compatible_cell_mhc("CD4_T", "MHC_II") is True
        assert is_compatible_cell_mhc("CD8_T", "MHC_I") is True
        assert is_compatible_cell_mhc("CD8_T", "HLA_E") is True

    def test_is_compatible_cell_mhc_false(self):
        from presto.data.vocab import is_compatible_cell_mhc
        assert is_compatible_cell_mhc("CD4_T", "MHC_I") is False
        assert is_compatible_cell_mhc("gd_T", "MHC_I") is False
        assert is_compatible_cell_mhc("B_cell", "MHC_II") is False


class TestNormalizeSpecies:
    """Test unified fine-grained species normalizer and roll-ups."""

    def test_none_and_empty(self):
        from presto.data.vocab import normalize_species
        assert normalize_species(None) is None
        assert normalize_species("") is None
        assert normalize_species("  ") is None

    def test_direct_fine_labels(self):
        from presto.data.vocab import normalize_species, FINE_SPECIES
        for label in FINE_SPECIES:
            assert normalize_species(label) == label, f"Direct label {label!r} should match itself"

    @pytest.mark.parametrize("raw,expected", [
        ("Homo sapiens", "human"),
        ("HUMAN", "human"),
        ("homo sapiens (human)", "human"),
    ])
    def test_human(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Mus musculus", "mouse"),
        ("BALB/c mouse", "mouse"),
        ("C57BL/6", "mouse"),
        ("murine", "mouse"),
        ("H2-Kb", "mouse"),
    ])
    def test_mouse(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Rattus norvegicus", "rat"),
        ("rat ", "rat"),
    ])
    def test_rat(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Macaca mulatta", "macaque"),
        ("rhesus macaque", "macaque"),
        ("Mamu-A1*001:01", "macaque"),
    ])
    def test_macaque(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Pan troglodytes", "chimpanzee"),
        ("chimpanzee", "chimpanzee"),
    ])
    def test_chimpanzee(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Gorilla gorilla", "gorilla"),
    ])
    def test_gorilla(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Pongo pygmaeus", "orangutan"),
    ])
    def test_orangutan(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Papio anubis", "baboon"),
        ("baboon", "baboon"),
    ])
    def test_baboon(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Aotus nancymaae", "other_nhp"),
        ("Saguinus oedipus", "other_nhp"),
    ])
    def test_other_nhp(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Bos taurus", "cattle"),
        ("bovine", "cattle"),
        ("cattle", "cattle"),
        ("BoLA-A*01:01", "cattle"),
    ])
    def test_cattle(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Sus scrofa", "pig"),
        ("porcine", "pig"),
        ("SLA-1*04:01", "pig"),
    ])
    def test_pig(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Equus caballus", "horse"),
        ("ELA-A1", "horse"),
    ])
    def test_horse(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Ovis aries", "sheep"),
        ("OLA-DRB1", "sheep"),
    ])
    def test_sheep(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Capra hircus", "goat"),
        ("goat", "goat"),
    ])
    def test_goat(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Canis lupus familiaris", "dog"),
        ("DLA-88*001:01", "dog"),
    ])
    def test_dog(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Felis catus", "cat"),
    ])
    def test_cat(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Oryctolagus cuniculus", "rabbit"),
        ("rabbit", "rabbit"),
    ])
    def test_rabbit(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Gallus gallus", "chicken"),
        ("chicken", "chicken"),
        ("GaGa-BF1", "chicken"),
    ])
    def test_chicken(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("duck", "other_bird"),
        ("turkey", "other_bird"),
        ("avian influenza host", "other_bird"),
    ])
    def test_other_bird(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Salmo salar", "salmon"),
        ("Atlantic salmon", "salmon"),
        ("Oncorhynchus mykiss", "salmon"),
    ])
    def test_salmon(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Danio rerio", "zebrafish"),
        ("zebrafish", "zebrafish"),
    ])
    def test_zebrafish(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    def test_salmonella_is_bacteria_not_fish(self):
        """Bug fix: 'salmonella' must NOT match fish."""
        from presto.data.vocab import normalize_species, normalize_organism
        assert normalize_species("salmonella") == "bacteria"
        assert normalize_species("Salmonella enterica") == "bacteria"
        assert normalize_organism("salmonella") == "bacteria"

    @pytest.mark.parametrize("raw,expected", [
        ("HIV-1", "viruses"),
        ("Influenza A virus", "viruses"),
        ("SARS-CoV-2", "viruses"),
        ("Mycobacterium tuberculosis", "bacteria"),
        ("Escherichia coli", "bacteria"),
        ("Candida albicans", "fungi"),
        ("Saccharomyces cerevisiae", "fungi"),
        ("Sulfolobus solfataricus", "archaea"),
    ])
    def test_pathogens(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    @pytest.mark.parametrize("raw,expected", [
        ("Xenopus laevis", "other_vertebrate"),
        ("snake", "other_vertebrate"),
        ("Drosophila melanogaster", "invertebrate"),
        ("Plasmodium falciparum", "invertebrate"),
    ])
    def test_non_mammal(self, raw, expected):
        from presto.data.vocab import normalize_species
        assert normalize_species(raw) == expected

    def test_unrecognizable_returns_none(self):
        from presto.data.vocab import normalize_species
        assert normalize_species("xyzzy_unknown_123") is None


class TestSpeciesRollUps:
    """Test that roll-up dicts cover all fine species and produce correct values."""

    def test_fine_to_organism_complete(self):
        from presto.data.vocab import FINE_SPECIES, FINE_TO_ORGANISM, ORGANISM_CATEGORIES
        for fs in FINE_SPECIES:
            assert fs in FINE_TO_ORGANISM, f"{fs} missing from FINE_TO_ORGANISM"
            assert FINE_TO_ORGANISM[fs] in ORGANISM_CATEGORIES

    def test_fine_to_chain_species_complete(self):
        from presto.data.vocab import FINE_SPECIES, FINE_TO_CHAIN_SPECIES, CHAIN_SPECIES_CATEGORIES
        for fs in FINE_SPECIES:
            assert fs in FINE_TO_CHAIN_SPECIES, f"{fs} missing from FINE_TO_CHAIN_SPECIES"
            val = FINE_TO_CHAIN_SPECIES[fs]
            assert val is None or val in CHAIN_SPECIES_CATEGORIES

    def test_fine_to_b2m_key_complete(self):
        from presto.data.vocab import FINE_SPECIES, FINE_TO_B2M_KEY
        for fs in FINE_SPECIES:
            assert fs in FINE_TO_B2M_KEY, f"{fs} missing from FINE_TO_B2M_KEY"

    def test_fine_to_is_foreign_complete(self):
        from presto.data.vocab import FINE_SPECIES, FINE_TO_IS_FOREIGN
        for fs in FINE_SPECIES:
            assert fs in FINE_TO_IS_FOREIGN, f"{fs} missing from FINE_TO_IS_FOREIGN"

    def test_foreign_labels_correct(self):
        from presto.data.vocab import FINE_TO_IS_FOREIGN
        assert FINE_TO_IS_FOREIGN["viruses"] is True
        assert FINE_TO_IS_FOREIGN["bacteria"] is True
        assert FINE_TO_IS_FOREIGN["fungi"] is True
        assert FINE_TO_IS_FOREIGN["archaea"] is True
        assert FINE_TO_IS_FOREIGN["human"] is False
        assert FINE_TO_IS_FOREIGN["mouse"] is False
        assert FINE_TO_IS_FOREIGN["chicken"] is False

    def test_organism_rollup_values(self):
        from presto.data.vocab import FINE_TO_ORGANISM
        assert FINE_TO_ORGANISM["human"] == "human"
        assert FINE_TO_ORGANISM["macaque"] == "nhp"
        assert FINE_TO_ORGANISM["chimpanzee"] == "nhp"
        assert FINE_TO_ORGANISM["mouse"] == "murine"
        assert FINE_TO_ORGANISM["rat"] == "murine"
        assert FINE_TO_ORGANISM["cattle"] == "other_mammal"
        assert FINE_TO_ORGANISM["chicken"] == "bird"
        assert FINE_TO_ORGANISM["salmon"] == "other_vertebrate"

    def test_chain_species_rollup_values(self):
        from presto.data.vocab import FINE_TO_CHAIN_SPECIES
        assert FINE_TO_CHAIN_SPECIES["human"] == "human"
        assert FINE_TO_CHAIN_SPECIES["mouse"] == "murine"
        assert FINE_TO_CHAIN_SPECIES["rat"] == "murine"
        assert FINE_TO_CHAIN_SPECIES["macaque"] == "nhp"
        assert FINE_TO_CHAIN_SPECIES["chimpanzee"] == "nhp"
        assert FINE_TO_CHAIN_SPECIES["cattle"] == "other_mammal"
        assert FINE_TO_CHAIN_SPECIES["other_vertebrate"] == "other_vertebrate"
        assert FINE_TO_CHAIN_SPECIES["viruses"] is None


class TestNormalizeOrganismBackwardCompat:
    """Verify normalize_organism returns identical values after delegation."""

    @pytest.mark.parametrize("raw,expected", [
        ("Homo sapiens", "human"),
        ("human", "human"),
        ("Mus musculus", "murine"),
        ("mouse", "murine"),
        ("murine", "murine"),
        ("Rattus norvegicus", "murine"),
        ("Macaca mulatta", "nhp"),
        ("chimpanzee", "nhp"),
        ("rhesus", "nhp"),
        ("Bos taurus", "other_mammal"),
        ("pig", "other_mammal"),
        ("dog", "other_mammal"),
        ("chicken", "bird"),
        ("Salmo salar", "other_vertebrate"),
        ("zebrafish", "other_vertebrate"),
        ("HIV-1", "viruses"),
        ("Mycobacterium tuberculosis", "bacteria"),
        ("Candida albicans", "fungi"),
        ("archaea", "archaea"),
        ("Xenopus laevis", "other_vertebrate"),
        ("Drosophila melanogaster", "invertebrate"),
        ("salmonella", "bacteria"),
        (None, None),
        ("", None),
    ])
    def test_backward_compat(self, raw, expected):
        from presto.data.vocab import normalize_organism
        assert normalize_organism(raw) == expected


class TestB2MExternalized:
    """Test that B2M sequences are correctly loaded from CSV."""

    def test_b2m_csv_loads(self):
        from presto.data.allele_resolver import _B2M_SEQUENCES
        assert len(_B2M_SEQUENCES) >= 10
        assert "human" in _B2M_SEQUENCES
        assert "mouse" in _B2M_SEQUENCES
        assert "chicken" in _B2M_SEQUENCES

    def test_backward_compat_aliases(self):
        from presto.data.allele_resolver import (
            HUMAN_B2M_SEQUENCE, MOUSE_B2M_SEQUENCE,
            MACAQUE_B2M_SEQUENCE, _B2M_SEQUENCES,
        )
        assert HUMAN_B2M_SEQUENCE == _B2M_SEQUENCES["human"]
        assert MOUSE_B2M_SEQUENCE == _B2M_SEQUENCES["mouse"]
        assert MACAQUE_B2M_SEQUENCE == _B2M_SEQUENCES["human"]

    def test_class_i_beta2m_sequence_human(self):
        from presto.data.allele_resolver import class_i_beta2m_sequence, _B2M_SEQUENCES
        assert class_i_beta2m_sequence("Homo sapiens") == _B2M_SEQUENCES["human"]
        assert class_i_beta2m_sequence("human") == _B2M_SEQUENCES["human"]

    def test_class_i_beta2m_sequence_mouse(self):
        from presto.data.allele_resolver import class_i_beta2m_sequence, _B2M_SEQUENCES
        assert class_i_beta2m_sequence("Mus musculus") == _B2M_SEQUENCES["mouse"]

    def test_class_i_beta2m_sequence_nhp_uses_human(self):
        from presto.data.allele_resolver import class_i_beta2m_sequence, _B2M_SEQUENCES
        assert class_i_beta2m_sequence("Macaca mulatta") == _B2M_SEQUENCES["human"]
        assert class_i_beta2m_sequence("chimpanzee") == _B2M_SEQUENCES["human"]

    def test_class_i_beta2m_sequence_cattle(self):
        from presto.data.allele_resolver import class_i_beta2m_sequence, _B2M_SEQUENCES
        assert class_i_beta2m_sequence("Bos taurus") == _B2M_SEQUENCES["cattle"]

    def test_class_i_beta2m_sequence_none(self):
        from presto.data.allele_resolver import class_i_beta2m_sequence
        assert class_i_beta2m_sequence(None) is None
        assert class_i_beta2m_sequence("xyzzy_unknown") is None

    def test_class_i_beta2m_fish_returns_salmon(self):
        from presto.data.allele_resolver import class_i_beta2m_sequence, _B2M_SEQUENCES
        assert class_i_beta2m_sequence("zebrafish") == _B2M_SEQUENCES["salmon"]
        assert class_i_beta2m_sequence("Salmo salar") == _B2M_SEQUENCES["salmon"]
