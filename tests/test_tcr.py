"""Tests for TCR modules - pins down TCR encoding and matching API.

Key biological constraints:
1. Chain types: TRA, TRB, TRG, TRD, IGH, IGK, IGL
2. Cell types: CD4_T, CD8_T, ab_T, gd_T, B_cell
3. Compatibility: CD4↔MHC_II, CD8↔MHC_I/HLA_E
4. γδ TCR and BCR do NOT bind classical pMHC (used as negatives)
"""

import pytest
import torch


# --------------------------------------------------------------------------
# TCR Encoder Tests
# --------------------------------------------------------------------------

class TestTCREncoder:
    """Test TCR sequence encoder."""

    def test_tcr_encoder_single_chain(self):
        from presto.models.tcr import TCREncoder
        enc = TCREncoder(d_model=64)
        chain_tok = torch.randint(4, 24, (2, 30))
        z = enc.encode_chain(chain_tok)
        assert z.shape == (2, 64)

    def test_tcr_encoder_paired_chains(self):
        from presto.models.tcr import TCREncoder
        enc = TCREncoder(d_model=64)
        alpha_tok = torch.randint(4, 24, (2, 30))
        beta_tok = torch.randint(4, 24, (2, 30))
        z = enc(alpha_tok, beta_tok)
        assert z.shape == (2, 64)

    def test_tcr_encoder_with_none_chain(self):
        """Should handle single-chain TCR data."""
        from presto.models.tcr import TCREncoder
        enc = TCREncoder(d_model=64)
        alpha_tok = torch.randint(4, 24, (2, 30))
        z = enc(alpha_tok, None)  # Beta chain missing
        assert z.shape == (2, 64)


# --------------------------------------------------------------------------
# Chain Classifier Tests
# --------------------------------------------------------------------------

class TestChainClassifier:
    """Classify chain type: TRA, TRB, TRG, TRD, IGH, IGK, IGL."""

    def test_chain_classifier_output_shape(self):
        from presto.models.tcr import ChainClassifier
        from presto.data.vocab import CHAIN_TYPES
        clf = ChainClassifier(d_model=64)
        chain_tok = torch.randint(4, 24, (3, 30))
        logits = clf(chain_tok)
        assert logits.shape == (3, len(CHAIN_TYPES))

    def test_chain_classifier_probabilities(self):
        from presto.models.tcr import ChainClassifier
        clf = ChainClassifier(d_model=64)
        chain_tok = torch.randint(4, 24, (2, 30))
        logits = clf(chain_tok)
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))


class TestChainAttributeClassifier:
    """Per-chain classifier for species × chain type × phenotype."""

    def test_chain_attribute_classifier_output_shape(self):
        from presto.models.tcr import ChainAttributeClassifier
        from presto.data.vocab import CHAIN_TYPES, CELL_TYPES, CHAIN_SPECIES_CATEGORIES
        clf = ChainAttributeClassifier(d_model=64)
        chain_tok = torch.randint(4, 24, (3, 30))
        outputs = clf(chain_tok)
        assert outputs["species_logits"].shape == (3, len(CHAIN_SPECIES_CATEGORIES))
        assert outputs["chain_logits"].shape == (3, len(CHAIN_TYPES))
        assert outputs["phenotype_logits"].shape == (3, len(CELL_TYPES))

    def test_chain_attribute_classifier_predict(self):
        from presto.models.tcr import ChainAttributeClassifier
        clf = ChainAttributeClassifier(d_model=64)
        chain_tok = torch.randint(4, 24, (2, 30))
        preds = clf.predict(chain_tok)
        assert preds["species"].shape == (2,)
        assert preds["chain_type"].shape == (2,)
        assert preds["phenotype"].shape == (2,)
        # Probs should sum to 1
        assert torch.allclose(preds["species_probs"].sum(dim=-1), torch.ones(2))
        assert torch.allclose(preds["chain_probs"].sum(dim=-1), torch.ones(2))
        assert torch.allclose(preds["phenotype_probs"].sum(dim=-1), torch.ones(2))


# --------------------------------------------------------------------------
# Cell Type Classifier Tests
# --------------------------------------------------------------------------

class TestCellTypeClassifier:
    """Classify cell type: CD4_T, CD8_T, ab_T, gd_T, B_cell."""

    def test_cell_classifier_from_paired_chains(self):
        from presto.models.tcr import CellTypeClassifier
        from presto.data.vocab import CELL_TYPES
        clf = CellTypeClassifier(d_model=64)
        tcr_vec = torch.randn(2, 64)
        logits = clf(tcr_vec)
        assert logits.shape == (2, len(CELL_TYPES))


# --------------------------------------------------------------------------
# TCR-pMHC Compatibility Tests
# --------------------------------------------------------------------------

class TestTCRpMHCCompatibility:
    """Test biological compatibility constraints."""

    def test_compatibility_mask_cd4_mhc_ii(self):
        """CD4 T cells should only match MHC-II."""
        from presto.models.tcr import get_compatibility_mask
        from presto.data.vocab import CELL_TO_IDX, MHC_TO_IDX
        mask = get_compatibility_mask()
        cd4_idx = CELL_TO_IDX["CD4_T"]
        mhc_ii_idx = MHC_TO_IDX["MHC_II"]
        mhc_i_idx = MHC_TO_IDX["MHC_I"]
        assert mask[cd4_idx, mhc_ii_idx] == 1.0
        assert mask[cd4_idx, mhc_i_idx] == 0.0

    def test_compatibility_mask_cd8_mhc_i(self):
        """CD8 T cells should match MHC-I and HLA-E."""
        from presto.models.tcr import get_compatibility_mask
        from presto.data.vocab import CELL_TO_IDX, MHC_TO_IDX
        mask = get_compatibility_mask()
        cd8_idx = CELL_TO_IDX["CD8_T"]
        assert mask[cd8_idx, MHC_TO_IDX["MHC_I"]] == 1.0
        assert mask[cd8_idx, MHC_TO_IDX["HLA_E"]] == 1.0
        assert mask[cd8_idx, MHC_TO_IDX["MHC_II"]] == 0.0

    def test_compatibility_mask_gd_t_no_pmhc(self):
        """γδ T cells do NOT bind classical pMHC."""
        from presto.models.tcr import get_compatibility_mask
        from presto.data.vocab import CELL_TO_IDX, MHC_TO_IDX
        mask = get_compatibility_mask()
        gd_idx = CELL_TO_IDX["gd_T"]
        # All zeros for gd_T
        assert mask[gd_idx].sum() == 0.0

    def test_compatibility_mask_b_cell_no_pmhc(self):
        """B cells do NOT bind pMHC (they use BCR)."""
        from presto.models.tcr import get_compatibility_mask
        from presto.data.vocab import CELL_TO_IDX
        mask = get_compatibility_mask()
        b_idx = CELL_TO_IDX["B_cell"]
        assert mask[b_idx].sum() == 0.0


# --------------------------------------------------------------------------
# TCR-pMHC Matching Head Tests
# --------------------------------------------------------------------------

class TestTCRpMHCMatcher:
    """Test TCR-pMHC recognition prediction."""

    def test_matcher_basic(self):
        from presto.models.tcr import TCRpMHCMatcher
        matcher = TCRpMHCMatcher(d_model=64)
        tcr_vec = torch.randn(2, 64)
        pmhc_vec = torch.randn(2, 64)
        logit = matcher(tcr_vec, pmhc_vec)
        assert logit.shape == (2,)

    def test_matcher_with_multiple_core_windows(self):
        """Should aggregate over pMHC core windows."""
        from presto.models.tcr import TCRpMHCMatcher
        matcher = TCRpMHCMatcher(d_model=64)
        tcr_vec = torch.randn(2, 64)
        pmhc_core_windows_vec = torch.randn(2, 5, 64)
        logit = matcher(tcr_vec, pmhc_core_windows_vec, reduce="max")
        assert logit.shape == (2,)

    def test_matcher_reduce_modes(self):
        from presto.models.tcr import TCRpMHCMatcher
        matcher = TCRpMHCMatcher(d_model=64)
        tcr_vec = torch.randn(2, 64)
        pmhc_core_windows_vec = torch.randn(2, 3, 64)
        logit_max = matcher(tcr_vec, pmhc_core_windows_vec, reduce="max")
        logit_mean = matcher(tcr_vec, pmhc_core_windows_vec, reduce="mean")
        assert logit_max.shape == logit_mean.shape == (2,)


# --------------------------------------------------------------------------
# Repertoire Recognition Head Tests
# --------------------------------------------------------------------------

class TestRepertoireHead:
    """Amortized prediction of P(some TCR in repertoire recognizes pMHC)."""

    def test_repertoire_head_basic(self):
        """When no specific TCR given, predict repertoire recognition."""
        from presto.models.tcr import RepertoireHead
        head = RepertoireHead(d_model=64)
        pmhc_vec = torch.randn(2, 64)
        logit = head(pmhc_vec)
        assert logit.shape == (2, 1)

    def test_repertoire_head_with_species(self):
        """Can condition on species (human vs mouse repertoire)."""
        from presto.models.tcr import RepertoireHead
        from presto.data.vocab import CHAIN_SPECIES_CATEGORIES
        head = RepertoireHead(d_model=64, n_species=len(CHAIN_SPECIES_CATEGORIES))
        pmhc_vec = torch.randn(2, 64)
        species_idx = torch.tensor([0, 1])  # human, mouse
        logit = head(pmhc_vec, species_idx=species_idx)
        assert logit.shape == (2, 1)

    def test_repertoire_head_gradient_flow(self):
        from presto.models.tcr import RepertoireHead
        head = RepertoireHead(d_model=64)
        pmhc_vec = torch.randn(2, 64, requires_grad=True)
        logit = head(pmhc_vec)
        logit.sum().backward()
        assert pmhc_vec.grad is not None


# --------------------------------------------------------------------------
# Contrastive Learning Tests
# --------------------------------------------------------------------------

class TestContrastiveTCR:
    """Test contrastive learning utilities for TCR."""

    def test_info_nce_loss(self):
        from presto.models.tcr import info_nce_loss
        tcr_vec = torch.randn(4, 64)
        pmhc_vec = torch.randn(4, 64)
        tcr_vec = tcr_vec / tcr_vec.norm(dim=-1, keepdim=True)
        pmhc_vec = pmhc_vec / pmhc_vec.norm(dim=-1, keepdim=True)
        loss = info_nce_loss(tcr_vec, pmhc_vec, temperature=0.07)
        assert loss.shape == ()
        assert loss >= 0

    def test_info_nce_with_negatives(self):
        from presto.models.tcr import info_nce_loss
        tcr_vec = torch.randn(4, 64)
        pmhc_vec = torch.randn(4, 64)
        negative_vec = torch.randn(10, 64)
        tcr_vec = tcr_vec / tcr_vec.norm(dim=-1, keepdim=True)
        pmhc_vec = pmhc_vec / pmhc_vec.norm(dim=-1, keepdim=True)
        negative_vec = negative_vec / negative_vec.norm(dim=-1, keepdim=True)
        loss = info_nce_loss(tcr_vec, pmhc_vec, negatives=negative_vec, temperature=0.07)
        assert loss.shape == ()
