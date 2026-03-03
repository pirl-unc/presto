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


class TestSpecies:
    """Test species vocabulary."""

    def test_species_complete(self):
        from presto.data.vocab import SPECIES
        expected = {"human", "mouse", "macaque", "other"}
        assert set(SPECIES) == expected

    def test_species_to_idx(self):
        from presto.data.vocab import SPECIES_TO_IDX, SPECIES
        for i, sp in enumerate(SPECIES):
            assert SPECIES_TO_IDX[sp] == i


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
