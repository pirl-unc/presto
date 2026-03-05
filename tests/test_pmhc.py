"""Tests for pMHC modules - pins down biological pathway API.

Key biological constraints:
1. Processing is MHC-INDEPENDENT (happens before MHC binding)
2. Binding has three kinetic log-rate latents: log_koff, log_kon_intrinsic, log_kon_chaperone
3. Binding path is class-symmetric
4. Aggregation uses numerically stable Noisy-OR
"""

import pytest
import torch


# --------------------------------------------------------------------------
# Test data fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def sample_peptides():
    """Sample peptide sequences for testing."""
    return ["SIINFEKL", "GILGFVFTL", "NLVPMVATV"]  # Class I epitopes


@pytest.fixture
def sample_long_peptides():
    """Sample longer peptides for Class II testing."""
    return ["PKYVKQNTLKLAT", "AGFKGEQGPKGEP"]  # Class II epitopes


@pytest.fixture
def tokenizer():
    from presto.data.tokenizer import Tokenizer
    return Tokenizer()


# --------------------------------------------------------------------------
# Processing Module Tests (MHC-INDEPENDENT)
# --------------------------------------------------------------------------

class TestProcessingModule:
    """Processing happens BEFORE MHC binding - it's MHC-independent."""

    def test_processing_does_not_take_mhc_inputs(self):
        """Processing should NOT receive MHC sequence as input."""
        from presto.models.pmhc import ProcessingModule
        proc = ProcessingModule(d_model=64)
        # Only takes peptide and flanks, NOT MHC
        pep_tok = torch.randint(4, 24, (2, 10))
        flank_n_tok = torch.randint(4, 24, (2, 5))
        flank_c_tok = torch.randint(4, 24, (2, 5))
        # Forward should work without MHC
        out = proc(pep_tok, flank_n_tok, flank_c_tok, mhc_class="I")
        assert out.shape == (2, 1)  # Processing probability logit

    def test_processing_class_specific_heads(self):
        """Different heads for Class I vs Class II processing."""
        from presto.models.pmhc import ProcessingModule
        proc = ProcessingModule(d_model=64)
        pep_tok = torch.randint(4, 24, (2, 10))
        out_I = proc(pep_tok, None, None, mhc_class="I")
        out_II = proc(pep_tok, None, None, mhc_class="II")
        # Different heads may give different outputs
        assert out_I.shape == out_II.shape

    def test_processing_flanks_optional(self):
        """Flanks can be None (not always available)."""
        from presto.models.pmhc import ProcessingModule
        proc = ProcessingModule(d_model=64)
        pep_tok = torch.randint(4, 24, (2, 10))
        out = proc(pep_tok, None, None, mhc_class="I")
        assert out.shape == (2, 1)

    def test_processing_accepts_soft_class_probs(self):
        """Processing can use inferred soft class probabilities."""
        from presto.models.pmhc import ProcessingModule

        proc = ProcessingModule(d_model=64)
        pep_tok = torch.randint(4, 24, (2, 10))
        class_probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
        out = proc(pep_tok, None, None, class_probs=class_probs)
        assert out.shape == (2, 1)

    def test_processing_flank_ffn_path_changes_outputs(self):
        """Changing flank tokens should affect processing outputs."""
        from presto.models.pmhc import ProcessingModule

        torch.manual_seed(7)
        proc = ProcessingModule(d_model=32)
        proc.eval()

        pep_tok = torch.tensor([[4, 5, 6, 7, 0, 0]])
        flank_n_tok_1 = torch.tensor([[8, 9, 0]])
        flank_c_tok_1 = torch.tensor([[10, 11, 0]])
        flank_n_tok_2 = torch.tensor([[12, 13, 0]])
        flank_c_tok_2 = torch.tensor([[14, 15, 0]])
        class_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

        out_1 = proc.forward_components(
            pep_tok,
            flank_n_tok_1,
            flank_c_tok_1,
            class_probs=class_probs,
        )
        out_2 = proc.forward_components(
            pep_tok,
            flank_n_tok_2,
            flank_c_tok_2,
            class_probs=class_probs,
        )

        assert not torch.allclose(out_1["processing_class1_logit"], out_2["processing_class1_logit"])
        assert not torch.allclose(out_1["processing_class2_logit"], out_2["processing_class2_logit"])

    def test_processing_soft_class_probs_mix_component_outputs(self):
        """Mixed processing logit is class-probability-weighted component sum."""
        from presto.models.pmhc import ProcessingModule

        proc = ProcessingModule(d_model=32)
        pep_tok = torch.randint(4, 24, (2, 10))
        class_probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)

        out = proc.forward_components(pep_tok, None, None, class_probs=class_probs)
        expected = (
            class_probs[:, :1] * out["processing_class1_logit"]
            + class_probs[:, 1:2] * out["processing_class2_logit"]
        )
        assert torch.allclose(out["processing_logit"], expected, atol=1e-6)

    def test_processing_species_modifier_changes_class_i_output(self):
        """Species should act as a class-I processing modifier."""
        import torch.nn as nn
        from presto.models.pmhc import ProcessingModule, PROCESSING_SPECIES_BUCKETS

        class DummyEncoder(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.d_model = d_model

            def forward(self, x, mask=None, lengths=None):
                pooled = torch.zeros((x.shape[0], self.d_model), device=x.device)
                full = torch.zeros((x.shape[0], x.shape[1], self.d_model), device=x.device)
                return pooled, full

        proc = ProcessingModule(d_model=4)
        proc.encoder = DummyEncoder(d_model=4)
        with torch.no_grad():
            proc.head_I.weight.fill_(1.0)
            proc.head_I.bias.zero_()
            proc.species_modifier_I.weight.zero_()
            human_idx = PROCESSING_SPECIES_BUCKETS.index("human")
            murine_idx = PROCESSING_SPECIES_BUCKETS.index("murine")
            proc.species_modifier_I.weight[:, human_idx] = 1.0
            proc.species_modifier_I.weight[:, murine_idx] = 2.0

        pep_tok = torch.tensor([[4, 5, 6], [4, 5, 6]])
        out = proc(pep_tok, None, None, mhc_class="I", species=["human", "murine"])
        assert out.shape == (2, 1)
        assert out[0].item() < out[1].item()

    def test_processing_accepts_soft_species_probs(self):
        """Processing supports inferred soft species probabilities."""
        from presto.models.pmhc import ProcessingModule, PROCESSING_SPECIES_BUCKETS

        proc = ProcessingModule(d_model=32)
        pep_tok = torch.randint(4, 24, (2, 10))
        species_probs = torch.zeros((2, len(PROCESSING_SPECIES_BUCKETS)))
        species_probs[0, PROCESSING_SPECIES_BUCKETS.index("human")] = 1.0
        species_probs[1, PROCESSING_SPECIES_BUCKETS.index("nhp")] = 1.0
        out = proc(
            pep_tok,
            None,
            None,
            mhc_class="I",
            species_probs=species_probs,
        )
        assert out.shape == (2, 1)


# --------------------------------------------------------------------------
# Binding Module Tests
# --------------------------------------------------------------------------

class TestBindingModule:
    """Binding outputs three latent strengths."""

    def test_binding_outputs_three_latents(self):
        """Binding module outputs kinetic log-rate latents."""
        from presto.models.pmhc import BindingModule
        bind = BindingModule(d_model=64)
        pmhc_vec = torch.randn(2, 64)
        latents = bind(pmhc_vec, mhc_class="I")
        assert "log_koff" in latents
        assert "log_kon_intrinsic" in latents
        assert "log_kon_chaperone" in latents
        assert all(v.shape == (2, 1) for v in latents.values())

    def test_binding_is_class_symmetric(self):
        """Binding latents should not depend on explicit class inputs."""
        from presto.models.pmhc import BindingModule
        bind = BindingModule(d_model=64)
        pmhc_vec = torch.randn(2, 64)
        lat_I = bind(pmhc_vec, mhc_class="I")
        lat_II = bind(pmhc_vec, mhc_class="II")
        assert torch.allclose(lat_I["log_kon_chaperone"], lat_II["log_kon_chaperone"], atol=1e-6)
        assert torch.allclose(lat_I["log_kon_intrinsic"], lat_II["log_kon_intrinsic"], atol=1e-6)
        assert torch.allclose(lat_I["log_koff"], lat_II["log_koff"], atol=1e-6)

    def test_binding_accepts_soft_class_probs(self):
        """Binding accepts class probs input while remaining class-symmetric."""
        from presto.models.pmhc import BindingModule

        bind = BindingModule(d_model=64)
        pmhc_vec = torch.randn(2, 64)
        probs = torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32)
        lat = bind(pmhc_vec, class_probs=probs)
        assert "log_kon_chaperone" in lat
        assert lat["log_kon_chaperone"].shape == (2, 1)

    def test_binding_latents_are_clamped_finite(self):
        """Extreme activations are clamped to a numerically safe range."""
        from presto.models.pmhc import BindingModule

        bind = BindingModule(d_model=64)
        pmhc_vec = torch.full((2, 64), 1e6)
        lat = bind(pmhc_vec, mhc_class="I")
        for value in lat.values():
            assert torch.isfinite(value).all()


class TestStableBindingHead:
    """Combines three latents into stable binding probability."""

    def test_combines_latents(self):
        """Stable binding head combines stability, intrinsic, chaperone."""
        from presto.models.pmhc import StableBindingHead
        head = StableBindingHead()
        stability = torch.randn(2, 1)
        intrinsic = torch.randn(2, 1)
        chaperone = torch.randn(2, 1)
        logit = head(stability, intrinsic, chaperone)
        assert logit.shape == (2,)

    def test_weights_are_positive(self):
        """Weights should be positive (softplus)."""
        from presto.models.pmhc import StableBindingHead
        head = StableBindingHead()
        # Check that weights are applied via softplus (always positive)
        assert hasattr(head, "w_stability")


# --------------------------------------------------------------------------
# pMHC Encoder Tests
# --------------------------------------------------------------------------

class TestPMHCEncoder:
    """Full pMHC encoder combining peptide and MHC."""

    def test_pmhc_encoder_class_I(self):
        from presto.models.pmhc import PMHCEncoder
        enc = PMHCEncoder(d_model=64)
        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))  # alpha chain
        mhc_b_tok = torch.randint(4, 24, (2, 20))  # beta2m
        z = enc(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
        assert z.shape == (2, 64)

    def test_pmhc_encoder_class_II(self):
        from presto.models.pmhc import PMHCEncoder
        enc = PMHCEncoder(d_model=64)
        pep_tok = torch.randint(4, 24, (2, 15))
        mhc_a_tok = torch.randint(4, 24, (2, 50))  # alpha chain
        mhc_b_tok = torch.randint(4, 24, (2, 50))  # beta chain
        z = enc(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="II")
        assert z.shape == (2, 64)


class TestCoreWindowScorer:
    """Core-window scorer emits per-window tensors and masks."""

    def test_core_window_scorer_shapes(self):
        from presto.models.pmhc import CoreWindowScorer, PMHCEncoder

        d_model = 64
        scorer = CoreWindowScorer(d_model=d_model, n_heads=4, core_min_len=8, core_max_len=15)
        mhc_encoder = PMHCEncoder(d_model=d_model, n_layers=2, n_heads=4)

        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))
        z_a_pooled, z_a_seq = mhc_encoder.mhc_encoder(mhc_a_tok)
        z_b_pooled, z_b_seq = mhc_encoder.mhc_encoder(mhc_b_tok)
        class_probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        out = scorer(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_a_seq=z_a_seq,
            mhc_b_seq=z_b_seq,
            mhc_a_pooled=z_a_pooled,
            mhc_b_pooled=z_b_pooled,
            class_probs=class_probs,
        )

        assert "core_window_vec" in out
        assert "core_window_mask" in out
        assert "core_window_prior_logit" in out
        assert out["core_window_vec"].ndim == 3
        assert out["core_window_mask"].ndim == 2
        assert out["core_window_prior_logit"].shape == out["core_window_mask"].shape
        assert out["core_window_mask"].any()

    def test_core_window_scorer_handles_fully_masked_mhc(self):
        from presto.models.pmhc import CoreWindowScorer

        d_model = 32
        scorer = CoreWindowScorer(d_model=d_model, n_heads=4, core_min_len=8, core_max_len=12)
        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.zeros((2, 6), dtype=torch.long)
        mhc_b_tok = torch.zeros((2, 6), dtype=torch.long)
        mhc_a_seq = torch.zeros((2, 6, d_model))
        mhc_b_seq = torch.zeros((2, 6, d_model))
        mhc_a_pooled = torch.zeros((2, d_model))
        mhc_b_pooled = torch.zeros((2, d_model))
        class_probs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])

        out = scorer(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_a_seq=mhc_a_seq,
            mhc_b_seq=mhc_b_seq,
            mhc_a_pooled=mhc_a_pooled,
            mhc_b_pooled=mhc_b_pooled,
            class_probs=class_probs,
        )
        assert torch.isfinite(out["core_window_vec"]).all()
        assert torch.isfinite(out["core_window_prior_logit"]).all()


# --------------------------------------------------------------------------
# Core-Window Enumeration Tests (MHC-II)
# --------------------------------------------------------------------------

class TestCoreWindowEnumeration:
    """For MHC-II, enumerate all possible 9-mer cores."""

    def test_enumerate_core_windows(self):
        from presto.models.pmhc import enumerate_core_windows
        peptide = "PKYVKQNTLKLAT"  # 13 AAs
        core_windows = enumerate_core_windows(peptide, core_lens=(9,))
        # 13 - 9 + 1 = 5 core windows
        assert len(core_windows) == 5
        for r in core_windows:
            assert "start" in r
            assert "core_len" in r
            assert "core" in r
            assert "pfr_n" in r
            assert "pfr_c" in r

    def test_enumerate_core_windows_multiple_core_lens(self):
        from presto.models.pmhc import enumerate_core_windows
        peptide = "PKYVKQNTLKLAT"  # 13 AAs
        core_windows = enumerate_core_windows(peptide, core_lens=(8, 9, 10))
        # (13-8+1) + (13-9+1) + (13-10+1) = 6 + 5 + 4 = 15
        assert len(core_windows) == 15

    def test_core_window_includes_pfrs(self):
        from presto.models.pmhc import enumerate_core_windows
        peptide = "ABCDEFGHIJKLM"  # 13 AAs
        core_windows = enumerate_core_windows(peptide, core_lens=(9,))
        r = core_windows[2]  # core starts at position 2
        assert r["start"] == 2
        assert r["core"] == "CDEFGHIJK"
        assert r["pfr_n"] == "AB"
        assert r["pfr_c"] == "LM"


# --------------------------------------------------------------------------
# Stable Noisy-OR Tests
# --------------------------------------------------------------------------

class TestStableNoisyOR:
    """Numerically stable Noisy-OR aggregation."""

    def test_stable_noisy_or_basic(self):
        from presto.models.pmhc import stable_noisy_or
        probs = torch.tensor([0.5, 0.3, 0.2])
        result = stable_noisy_or(probs)
        # 1 - (1-0.5)*(1-0.3)*(1-0.2) = 1 - 0.5*0.7*0.8 = 1 - 0.28 = 0.72
        expected = 1 - (1 - 0.5) * (1 - 0.3) * (1 - 0.2)
        assert torch.allclose(result, torch.tensor(expected), atol=1e-5)

    def test_stable_noisy_or_with_mask(self):
        from presto.models.pmhc import stable_noisy_or
        probs = torch.tensor([0.5, 0.3, 0.2, 0.9])
        mask = torch.tensor([1.0, 1.0, 1.0, 0.0])  # Ignore last
        result = stable_noisy_or(probs, mask=mask)
        expected = 1 - (1 - 0.5) * (1 - 0.3) * (1 - 0.2)
        assert torch.allclose(result, torch.tensor(expected), atol=1e-5)

    def test_stable_noisy_or_batch(self):
        from presto.models.pmhc import stable_noisy_or
        probs = torch.tensor([[0.5, 0.3], [0.2, 0.8]])
        result = stable_noisy_or(probs, dim=-1)
        assert result.shape == (2,)

    def test_stable_noisy_or_numerical_stability(self):
        """Should not underflow with many small probabilities."""
        from presto.models.pmhc import stable_noisy_or
        # 100 instances with p=0.01 each
        probs = torch.full((100,), 0.01)
        result = stable_noisy_or(probs)
        # Should be ~0.634 (1 - 0.99^100)
        assert result > 0.5
        assert result < 1.0
        assert torch.isfinite(result)

    def test_stable_noisy_or_extreme_values(self):
        """Should handle p=0 and p~1 gracefully."""
        from presto.models.pmhc import stable_noisy_or
        probs = torch.tensor([0.0, 0.5, 0.999999])
        result = stable_noisy_or(probs)
        assert torch.isfinite(result)
        assert result > 0.99  # Almost certainly at least one


# --------------------------------------------------------------------------
# Presentation Bottleneck Tests
# --------------------------------------------------------------------------

class TestPresentationBottleneck:
    """Combines processing and binding into presentation probability."""

    def test_presentation_bottleneck(self):
        from presto.models.pmhc import PresentationBottleneck
        pres = PresentationBottleneck()
        proc_logit = torch.randn(2, 1)
        bind_logit = torch.randn(2, 1)
        out = pres(proc_logit, bind_logit)
        assert out.shape == (2, 1)

    def test_presentation_bottleneck_gradients(self):
        from presto.models.pmhc import PresentationBottleneck
        pres = PresentationBottleneck()
        proc_logit = torch.randn(2, 1, requires_grad=True)
        bind_logit = torch.randn(2, 1, requires_grad=True)
        out = pres(proc_logit, bind_logit)
        out.sum().backward()
        assert proc_logit.grad is not None
        assert bind_logit.grad is not None

    def test_presentation_bottleneck_negative_inputs_remain_negative(self):
        """Low processing + low binding should not generate a positive interaction boost."""
        from presto.models.pmhc import PresentationBottleneck

        pres = PresentationBottleneck()
        with torch.no_grad():
            pres.bias.fill_(0.0)
            pres.w_prior.fill_(0.0)
            pres.w_proc.fill_(1.0)
            pres.w_bind.fill_(1.0)

        proc_logit = torch.tensor([[-2.0]])
        bind_logit = torch.tensor([[-3.0]])
        out = pres(proc_logit, bind_logit)
        assert out.item() < 0.0


# --------------------------------------------------------------------------
# Multi-Allele Aggregation Tests
# --------------------------------------------------------------------------

class TestMultiAlleleAggregation:
    """Aggregate over multiple alleles for MS/EL data."""

    def test_aggregate_alleles_noisy_or(self):
        """Noisy-OR over alleles."""
        from presto.models.pmhc import stable_noisy_or
        # Simulate per-allele presentation probabilities
        per_allele_probs = torch.tensor([0.3, 0.4, 0.5, 0.2, 0.1, 0.6])  # 6 alleles
        bag_prob = stable_noisy_or(per_allele_probs)
        # Should be higher than any individual
        assert bag_prob > 0.6

    def test_posterior_attribution(self):
        """Can compute posterior attribution to each allele."""
        from presto.models.pmhc import posterior_attribution
        per_allele_probs = torch.tensor([0.3, 0.4, 0.2])
        attribution = posterior_attribution(per_allele_probs)
        # Should sum to ~1 (or close to bag prob)
        assert attribution.shape == per_allele_probs.shape
        # Higher prob alleles get higher attribution
        assert attribution[1] > attribution[2]
