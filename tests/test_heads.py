"""Tests for prediction heads - pins down IEDB measurement prediction API.

Heads for all IEDB measurement types:
- Binding: KD, IC50, EC50 (in log10 nM)
- Kinetics: kon, koff (in log10 units)
- Stability: t1/2 (log10 min), Tm (Celsius, normalized)
- T-cell: functional assay outcomes
- Elution/MS: detection probability
"""

import pytest
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Binding Heads Tests
# --------------------------------------------------------------------------

class TestBindingHeads:
    """Test binding affinity prediction heads."""

    def test_kd_head(self):
        from presto.models.heads import KDHead
        head = KDHead(d_model=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)

    def test_ic50_head(self):
        from presto.models.heads import IC50Head
        head = IC50Head(d_model=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)

    def test_ec50_head(self):
        from presto.models.heads import EC50Head
        head = EC50Head(d_model=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)


# --------------------------------------------------------------------------
# Kinetics Heads Tests
# --------------------------------------------------------------------------

class TestKineticsHeads:
    """Test kinetics prediction heads."""

    def test_kon_head(self):
        from presto.models.heads import KonHead
        head = KonHead(d_model=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)

    def test_koff_head(self):
        from presto.models.heads import KoffHead
        head = KoffHead(d_model=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)


# --------------------------------------------------------------------------
# Stability Heads Tests
# --------------------------------------------------------------------------

class TestStabilityHeads:
    """Test stability prediction heads."""

    def test_half_life_head(self):
        from presto.models.heads import HalfLifeHead
        head = HalfLifeHead(input_dim=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)

    def test_tm_head(self):
        from presto.models.heads import TmHead
        head = TmHead(input_dim=64)
        z = torch.randn(2, 64)
        pred = head(z)
        assert pred.shape == (2, 1)


# --------------------------------------------------------------------------
# Combined Assay Heads Tests
# --------------------------------------------------------------------------

class TestAssayHeads:
    """Test combined assay heads module."""

    def test_assay_heads_all_outputs(self):
        from presto.models.heads import AssayHeads
        heads = AssayHeads(d_model=64)
        ba_vec = torch.randn(2, 64)
        bs_vec = torch.randn(2, 64)
        outputs = heads(ba_vec, bs_vec)
        # Should have all assay types
        expected_keys = ["KD_nM", "IC50_nM", "EC50_nM", "kon", "koff", "t_half", "Tm"]
        for key in expected_keys:
            assert key in outputs
            assert outputs[key].shape == (2, 1)

    def test_assay_heads_gradient_flow(self):
        from presto.models.heads import AssayHeads
        heads = AssayHeads(d_model=64)
        ba_vec = torch.randn(2, 64, requires_grad=True)
        bs_vec = torch.randn(2, 64, requires_grad=True)
        outputs = heads(ba_vec, bs_vec)
        loss = sum(v.sum() for v in outputs.values())
        loss.backward()
        assert ba_vec.grad is not None
        assert bs_vec.grad is not None

    def test_assay_heads_clamp_weak_affinity_to_100k_nm(self):
        from presto.models.heads import AssayHeads

        heads = AssayHeads(d_model=64, max_log10_nM=5.0)
        ba_vec = torch.randn(2, 64)
        bs_vec = torch.randn(2, 64)
        latents = {
            "log_koff": torch.full((2, 1), 3.0),  # very fast off-rate
            "log_kon_intrinsic": torch.full((2, 1), -3.0),
            "log_kon_chaperone": torch.full((2, 1), -3.0),
        }
        outputs = heads(ba_vec, bs_vec, binding_latents=latents)

        assert torch.all(outputs["KD_nM"] <= 5.0 + 1e-6)
        assert torch.all(outputs["IC50_nM"] <= 5.0 + 1e-6)
        assert torch.all(outputs["EC50_nM"] <= 5.0 + 1e-6)

    def test_assay_heads_keep_gradients_near_weak_affinity_cap(self):
        from presto.models.heads import AssayHeads

        heads = AssayHeads(d_model=16, max_log10_nM=5.0)
        ba_vec = torch.randn(2, 16)
        bs_vec = torch.randn(2, 16)
        latents = {
            "log_koff": torch.full((2, 1), 2.0, requires_grad=True),
            "log_kon_intrinsic": torch.full((2, 1), 5.0, requires_grad=True),
            "log_kon_chaperone": torch.full((2, 1), 5.0, requires_grad=True),
        }
        outputs = heads(ba_vec, bs_vec, binding_latents=latents)
        outputs["KD_nM"].sum().backward()

        assert latents["log_koff"].grad is not None
        assert latents["log_kon_intrinsic"].grad is not None
        assert latents["log_kon_chaperone"].grad is not None
        assert latents["log_koff"].grad.abs().max().item() > 0.0

    def test_assay_heads_extreme_latents_stay_finite(self):
        from presto.models.heads import AssayHeads

        heads = AssayHeads(d_model=64, max_log10_nM=5.0)
        ba_vec = torch.randn(2, 64)
        bs_vec = torch.randn(2, 64)
        latents = {
            "log_koff": torch.full((2, 1), 1000.0),
            "log_kon_intrinsic": torch.full((2, 1), -1000.0),
            "log_kon_chaperone": torch.full((2, 1), 1000.0),
        }
        outputs = heads(ba_vec, bs_vec, binding_latents=latents)
        for value in outputs.values():
            assert torch.isfinite(value).all()

    def test_mhcflurry_residuals_are_added_in_logit_space(self):
        from presto.models.affinity import (
            affinity_log10_to_target_logit,
            affinity_target_logit_to_log10,
        )
        from presto.models.heads import AssayHeads, smooth_lower_bound, smooth_upper_bound

        heads = AssayHeads(
            d_model=8,
            affinity_target_encoding="mhcflurry",
            max_affinity_nM=50000.0,
            affinity_assay_residual_mode="shared_base_segment_residual",
            sequence_summary_dim=8,
        )
        heads.ic50_residual = nn.Identity()

        kd_base = torch.tensor([[torch.log10(torch.tensor(500.0)).item()]], dtype=torch.float32)
        kd_base_target_logit = affinity_log10_to_target_logit(
            kd_base,
            encoding="mhcflurry",
            max_affinity_nM=50000.0,
        )
        residual = torch.tensor([[1.0]], dtype=torch.float32)

        observed = heads._affinity_residual_output(
            kd_base=kd_base,
            kd_base_target_logit=kd_base_target_logit,
            residual=residual,
        )
        expected = affinity_target_logit_to_log10(
            kd_base_target_logit + residual,
            encoding="mhcflurry",
            max_affinity_nM=50000.0,
        )
        expected = smooth_lower_bound(expected, -3.0)
        expected = smooth_upper_bound(expected, heads.max_log10_nM)

        assert torch.allclose(observed, expected, atol=1e-6)


# --------------------------------------------------------------------------
# T-cell Head Tests
# --------------------------------------------------------------------------

class TestTCellHead:
    """Test T-cell functional assay head."""

    def test_tcell_head_basic(self):
        from presto.models.heads import TCellHead
        head = TCellHead(d_model=64)
        pmhc_vec = torch.randn(2, 64)
        tcr_vec = torch.randn(2, 64)
        logit = head(pmhc_vec, tcr_vec)
        assert logit.shape == (2, 1)

    def test_tcell_head_without_tcr(self):
        """Can predict with repertoire-level features instead of specific TCR."""
        from presto.models.heads import TCellHead
        head = TCellHead(d_model=64)
        pmhc_vec = torch.randn(2, 64)
        logit = head(pmhc_vec, tcr_vec=None)
        assert logit.shape == (2, 1)

    def test_tcell_assay_head_uses_immunogenicity_and_context(self):
        from presto.models.heads import TCellAssayHead

        head = TCellAssayHead(d_model=64)
        ig_cd8_vec = torch.randn(2, 64)
        ig_cd4_vec = torch.randn(2, 64)
        pres1 = torch.randn(2, 1)
        pres2 = torch.randn(2, 1)
        bind1 = torch.randn(2, 1)
        bind2 = torch.randn(2, 1)
        class_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
        context = {
            "assay_method_idx": torch.tensor([1, 0], dtype=torch.long),
            "assay_readout_idx": torch.tensor([1, 0], dtype=torch.long),
            "apc_type_idx": torch.tensor([1, 0], dtype=torch.long),
            "culture_context_idx": torch.tensor([1, 0], dtype=torch.long),
            "stim_context_idx": torch.tensor([1, 0], dtype=torch.long),
        }

        logit = head(
            immunogenicity_cd8_vec=ig_cd8_vec,
            immunogenicity_cd4_vec=ig_cd4_vec,
            presentation_class1_logit=pres1,
            presentation_class2_logit=pres2,
            binding_class1_logit=bind1,
            binding_class2_logit=bind2,
            class_probs=class_probs,
            assay_method_idx=context["assay_method_idx"],
            assay_readout_idx=context["assay_readout_idx"],
            apc_type_idx=context["apc_type_idx"],
            culture_context_idx=context["culture_context_idx"],
            stim_context_idx=context["stim_context_idx"],
        )

        assert logit.shape == (2, 1)

    def test_tcell_assay_head_output_shape_no_context(self):
        from presto.models.heads import TCellAssayHead

        head = TCellAssayHead(d_model=64)
        ig_cd8_vec = torch.randn(2, 64)
        ig_cd4_vec = torch.randn(2, 64)
        pres1 = torch.randn(2, 1)
        pres2 = torch.randn(2, 1)
        bind1 = torch.randn(2, 1)
        bind2 = torch.randn(2, 1)
        class_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)

        logit = head(
            immunogenicity_cd8_vec=ig_cd8_vec,
            immunogenicity_cd4_vec=ig_cd4_vec,
            presentation_class1_logit=pres1,
            presentation_class2_logit=pres2,
            binding_class1_logit=bind1,
            binding_class2_logit=bind2,
            class_probs=class_probs,
        )

        assert logit.shape == (2, 1)


# --------------------------------------------------------------------------
# Elution/MS Head Tests
# --------------------------------------------------------------------------

class TestElutionHead:
    """Test elution/MS detection head."""

    def test_elution_head(self):
        from presto.models.heads import ElutionHead
        head = ElutionHead()
        pres = torch.randn(2, 1)
        ms_detect = torch.randn(2, 1)
        logit = head(pres, ms_detect)
        assert logit.shape == (2, 1)

    def test_elution_head_uses_presentation_signal(self):
        from presto.models.heads import ElutionHead

        head = ElutionHead()
        head.eval()
        ms_detect = torch.zeros(2, 1)
        pres_low = torch.full((2, 1), -5.0)
        pres_high = torch.full((2, 1), 5.0)

        with torch.no_grad():
            low = head(pres_low, ms_detect)
            high = head(pres_high, ms_detect)

        assert torch.all(high > low)


# --------------------------------------------------------------------------
# Multi-Head Module Tests
# --------------------------------------------------------------------------

class TestMultiTaskHeads:
    """Test multi-task head configuration."""

    def test_all_heads_share_encoder(self):
        """All heads should work with same encoder output."""
        from presto.models.heads import AssayHeads, ElutionHead, TCellHead
        d_model = 64
        assay = AssayHeads(d_model=d_model)
        elution = ElutionHead()
        tcell = TCellHead(d_model=d_model)

        ba_vec = torch.randn(2, d_model)
        bs_vec = torch.randn(2, d_model)
        pmhc_vec = torch.randn(2, d_model)
        tcr_vec = torch.randn(2, d_model)
        pres = torch.randn(2, 1)
        ms_detect = torch.randn(2, 1)

        _ = assay(ba_vec, bs_vec)
        _ = elution(pres, ms_detect)
        _ = tcell(pmhc_vec, tcr_vec)

    def test_heads_independent_gradients(self):
        """Different heads should have independent gradients."""
        from presto.models.heads import KDHead, IC50Head
        kd = KDHead(d_model=64)
        ic50 = IC50Head(d_model=64)

        z = torch.randn(2, 64, requires_grad=True)
        pred_kd = kd(z)
        pred_ic50 = ic50(z)

        # Backprop through KD only
        pred_kd.sum().backward(retain_graph=True)
        kd_grad = z.grad.clone()
        z.grad.zero_()

        # Backprop through IC50 only
        pred_ic50.sum().backward()
        ic50_grad = z.grad.clone()

        # Gradients should be different (different heads)
        assert not torch.allclose(kd_grad, ic50_grad)


# --------------------------------------------------------------------------
# Unit Conversion Tests
# --------------------------------------------------------------------------

class TestUnitConversions:
    """Test unit conversion utilities."""

    def test_to_log10_nm(self):
        from presto.models.heads import to_log10_nM
        # 1 nM -> log10(1) = 0
        assert torch.allclose(to_log10_nM(torch.tensor(1.0)), torch.tensor(0.0))
        # 1000 nM -> log10(1000) = 3
        assert torch.allclose(to_log10_nM(torch.tensor(1000.0)), torch.tensor(3.0))

    def test_from_log10_nm(self):
        from presto.models.heads import from_log10_nM
        # log10(1) = 0 -> 1 nM
        assert torch.allclose(from_log10_nM(torch.tensor(0.0)), torch.tensor(1.0))
        # log10(1000) = 3 -> 1000 nM
        assert torch.allclose(from_log10_nM(torch.tensor(3.0)), torch.tensor(1000.0))

    def test_normalize_tm(self):
        from presto.models.heads import normalize_tm, denormalize_tm
        # Room temp ~25C normalized
        tm_raw = torch.tensor(25.0)
        tm_norm = normalize_tm(tm_raw)
        tm_back = denormalize_tm(tm_norm)
        assert torch.allclose(tm_raw, tm_back, atol=0.1)
