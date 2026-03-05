"""Integration tests for full Presto model.

Tests that all components work together correctly.
"""

import pytest
import torch
import types


# --------------------------------------------------------------------------
# Full Model Integration Tests
# --------------------------------------------------------------------------

class TestPrestoModel:
    """Test full Presto model."""

    def test_model_init(self):
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert model is not None

    def test_model_forward_pmhc_only(self):
        """Forward pass with just pMHC (no TCR)."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class="I",
        )

        # Should have binding/assay predictions
        assert "pmhc_vec" in outputs
        assert "assays" in outputs
        assert "processing_logit" in outputs
        assert "processing_class1_logit" in outputs
        assert "processing_class2_logit" in outputs
        assert "presentation_logit" in outputs
        assert "presentation_class1_logit" in outputs
        assert "presentation_class2_logit" in outputs
        assert "binding_base_logit" in outputs
        assert "binding_class1_logit" in outputs
        assert "binding_class2_logit" in outputs
        assert "mhc_is_class1_prob" in outputs
        assert "mhc_is_class2_prob" in outputs
        assert "species_probs" in outputs
        assert "recognition_cd8_logit" in outputs
        assert "recognition_cd4_logit" in outputs
        assert "immunogenicity_cd8_logit" in outputs
        assert "immunogenicity_cd4_logit" in outputs
        assert "immunogenicity_mixture_logit" in outputs
        assert "immunogenicity_mixed_logit" in outputs
        assert "immunogenicity_logit" in outputs
        assert "tcell_logit" in outputs
        assert "tcell_context_logits" in outputs
        assert "tcell_panel_logits" in outputs
        for panel_key in (
            "assay_method",
            "assay_readout",
            "apc_type",
            "culture_context",
            "stim_context",
            "peptide_format",
        ):
            assert panel_key in outputs["tcell_context_logits"]
            assert panel_key in outputs["tcell_panel_logits"]
            assert outputs["tcell_context_logits"][panel_key].shape[0] == pep_tok.shape[0]
        assert "mhc_class_logits" in outputs
        assert "mhc_species_logits" in outputs
        assert "core_start_logit" in outputs
        assert "core_start_prob" in outputs
        assert "processing_mixed_logit" in outputs
        assert "binding_mixed_logit" in outputs
        assert "presentation_mixed_logit" in outputs
        assert "recognition_mixed_logit" in outputs
        assert "latent_vecs" in outputs
        for key in (
            "processing_class1",
            "processing_class2",
            "ms_detectability",
            "binding_affinity",
            "binding_stability",
            "presentation_class1",
            "presentation_class2",
            "recognition_cd8",
            "recognition_cd4",
            "recognition_mixed",
            "immunogenicity_cd8",
            "immunogenicity_cd4",
            "immunogenicity_mixed",
        ):
            assert key in outputs["latent_vecs"]

    def test_model_forward_with_tcr(self):
        """Forward pass with pMHC and TCR inputs (TCR path currently disabled)."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))
        tcr_a_tok = torch.randint(4, 24, (2, 30))
        tcr_b_tok = torch.randint(4, 24, (2, 30))

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class="I",
            tcr_a_tok=tcr_a_tok,
            tcr_b_tok=tcr_b_tok,
        )

        # TCR path is currently disabled in canonical forward.
        assert "tcr_vec" not in outputs
        assert "match_logit" not in outputs
        assert "immunogenicity_logit" in outputs
        assert "immunogenicity_cd8_logit" in outputs
        assert "immunogenicity_cd4_logit" in outputs
        assert "tcell_context_logits" in outputs
        assert "chain_species_logits" not in outputs
        assert "chain_type_logits" not in outputs
        assert "chain_phenotype_logits" not in outputs

    def test_model_forward_class_ii(self):
        """Forward pass for MHC Class II."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 15))  # Longer peptide
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 50))  # Full beta chain

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class="II",
        )

        assert "pmhc_vec" in outputs
        assert "presentation_logit" in outputs

    def test_model_forward_without_class_label(self):
        """Model can infer class directly from MHC sequences."""
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
        )

        assert "mhc_class_logits" in outputs
        assert outputs["mhc_class_logits"].shape == (2, 2)
        assert "mhc_species_logits" in outputs
        assert outputs["mhc_species_logits"].shape[0] == 2
        assert "presentation_logit" in outputs


class TestPrestoGradients:
    """Test gradient flow through full model."""

    def test_full_model_is_differentiable(self):
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))
        tcr_a_tok = torch.randint(4, 24, (2, 30))
        tcr_b_tok = torch.randint(4, 24, (2, 30))

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class="I",
            tcr_a_tok=tcr_a_tok,
            tcr_b_tok=tcr_b_tok,
        )

        # Backprop through immunogenicity
        loss = outputs["immunogenicity_logit"].sum()
        loss.backward()

        # Check that at least some gradients exist (not all params are in ig path)
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grads


class TestPrestoWithProcessingFlanks:
    """Test model with processing context (flanks)."""

    def test_with_flanks(self):
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))
        flank_n_tok = torch.randint(4, 24, (2, 5))
        flank_c_tok = torch.randint(4, 24, (2, 5))

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class="I",
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
        )

        # Processing should use flanks
        assert "processing_logit" in outputs


class TestPrestoOutputConsistency:
    """Test output consistency."""

    def test_deterministic_eval(self):
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out1 = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            out2 = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")

        assert torch.allclose(out1["pmhc_vec"], out2["pmhc_vec"])
        assert torch.allclose(out1["presentation_logit"], out2["presentation_logit"])

    def test_binding_logit_matches_kd_calibration(self):
        from presto.models.presto import Presto
        from presto.models.affinity import binding_logit_from_kd_log10
        import torch.nn.functional as F

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")

        expected = binding_logit_from_kd_log10(
            out["assays"]["KD_nM"].squeeze(-1),
            midpoint_nM=model.binding_midpoint_nM,
            log10_scale=model.binding_log10_scale,
        ).clamp(min=-20.0, max=20.0)
        assert torch.allclose(out["binding_base_logit"], expected, atol=1e-5)

        class_margin = out["mhc_class_probs"][:, :1] - out["mhc_class_probs"][:, 1:2]
        expected_class1 = expected.unsqueeze(-1) + F.softplus(model.w_binding_class1_calibration) * class_margin
        expected_class2 = expected.unsqueeze(-1) - F.softplus(model.w_binding_class2_calibration) * class_margin
        expected_mixed = (
            out["mhc_class_probs"][:, :1] * expected_class1
            + out["mhc_class_probs"][:, 1:2] * expected_class2
        ).squeeze(-1)

        assert torch.allclose(out["binding_class1_logit"], expected_class1, atol=1e-5)
        assert torch.allclose(out["binding_class2_logit"], expected_class2, atol=1e-5)
        assert torch.allclose(out["binding_logit"], expected_mixed, atol=1e-5)

    def test_kd_soft_cap_not_hard_clamped_in_forward(self):
        from presto.models.presto import Presto

        class _ZeroBias(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.new_zeros((x.shape[0], 1))

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        model.kd_assay_bias = _ZeroBias()

        def _derive_kd_override(_self, binding_latents):
            batch_size = binding_latents["log_koff"].shape[0]
            return torch.full(
                (batch_size, 1),
                20.0,  # force extremely weak affinity branch
                device=binding_latents["log_koff"].device,
            )

        model.binding.derive_kd = types.MethodType(_derive_kd_override, model.binding)

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")

        kd = out["assays"]["KD_nM"]
        # Hard pre-clamping to max would collapse to exactly 5-softplus(0) ~ 4.3069.
        assert torch.all(kd > 4.8)
        assert torch.all(kd < model.max_log10_nM + 1e-6)

    def test_presentation_uses_class_specific_additive_logit_path(self):
        from presto.models.presto import Presto
        import torch.nn.functional as F

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            class1_base = model.presentation(
                out["processing_class1_logit"],
                out["binding_class1_logit"],
            )
            class1_base = class1_base + model.w_presentation_class1_latent * model.presentation_class1_latent_head(
                out["latent_vecs"]["presentation_class1"]
            )
            class2_base = model.presentation(
                out["processing_class2_logit"],
                out["binding_class2_logit"],
            )
            class2_base = class2_base + model.w_presentation_class2_latent * model.presentation_class2_latent_head(
                out["latent_vecs"]["presentation_class2"]
            )
            class1_prob = out["mhc_class_probs"][:, :1]
            class2_prob = out["mhc_class_probs"][:, 1:2]
            stability = -out["binding_latents"]["log_koff"]
            class1_prob_logit = torch.logit(class1_prob.clamp(min=1e-4, max=1.0 - 1e-4))
            class2_prob_logit = torch.logit(class2_prob.clamp(min=1e-4, max=1.0 - 1e-4))
            expected_class1 = (
                class1_base
                + F.softplus(model.w_class1_presentation_stability) * stability
                + F.softplus(model.w_class1_presentation_class) * class1_prob_logit
            )
            expected_class2 = (
                class2_base
                + F.softplus(model.w_class2_presentation_stability) * stability
                + F.softplus(model.w_class2_presentation_class) * class2_prob_logit
            )
            expected = class1_prob * expected_class1 + class2_prob * expected_class2

        assert torch.allclose(out["presentation_class1_logit"], expected_class1, atol=1e-5)
        assert torch.allclose(out["presentation_class2_logit"], expected_class2, atol=1e-5)
        assert torch.allclose(out["presentation_logit"], expected, atol=1e-5)

    def test_ic50_ec50_share_kd_latent_calibration(self):
        from presto.models.presto import Presto
        from presto.models.heads import smooth_upper_bound

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            ba_vec = out["latent_vecs"]["binding_affinity"]
            expected_ic50 = smooth_upper_bound(
                torch.clamp(
                    out["assays"]["KD_nM"] + model.assay_heads.ic50_residual(ba_vec),
                    min=-3.0,
                ),
                model.max_log10_nM,
            )
            expected_ec50 = smooth_upper_bound(
                torch.clamp(
                    out["assays"]["KD_nM"] + model.assay_heads.ec50_residual(ba_vec),
                    min=-3.0,
                ),
                model.max_log10_nM,
            )

        assert torch.allclose(out["assays"]["IC50_nM"], expected_ic50, atol=1e-5)
        assert torch.allclose(out["assays"]["EC50_nM"], expected_ec50, atol=1e-5)

    def test_elution_head_receives_presentation_and_ms_detectability(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            expected = model.elution_head(
                out["presentation_logit"],
                out["ms_detectability_logit"],
            )

        assert torch.allclose(out["elution_logit"], expected, atol=1e-5)
        # ms_logit == elution_logit (same tensor per S9.3)
        assert torch.allclose(out["ms_logit"], out["elution_logit"])

    def test_binding_latent_does_not_see_flank_tokens(self):
        """Binding latent segments exclude flanks (design S7.5)."""
        from presto.models.presto import Presto
        assert "nflank" not in Presto.LATENT_SEGMENTS["binding_affinity"]
        assert "cflank" not in Presto.LATENT_SEGMENTS["binding_affinity"]
        assert "nflank" not in Presto.LATENT_SEGMENTS["binding_stability"]
        assert "cflank" not in Presto.LATENT_SEGMENTS["binding_stability"]


# --------------------------------------------------------------------------
# Multi-Allele Aggregation Tests
# --------------------------------------------------------------------------

class TestMultiAlleleForward:
    """Test forward pass with multiple alleles (MS/EL scenario)."""

    def test_multi_allele_aggregation(self):
        from presto.models.presto import Presto
        from presto.models.pmhc import stable_noisy_or
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (1, 10))
        # Simulate 6 alleles for one sample
        allele_toks_a = [torch.randint(4, 24, (1, 50)) for _ in range(6)]
        allele_toks_b = [torch.randint(4, 24, (1, 20)) for _ in range(6)]

        with torch.no_grad():
            probs = []
            for a_tok, b_tok in zip(allele_toks_a, allele_toks_b):
                out = model(pep_tok, a_tok, b_tok, mhc_class="I")
                prob = torch.sigmoid(out["presentation_logit"]).flatten()
                probs.append(prob)

            probs_tensor = torch.cat(probs)
            bag_prob = stable_noisy_or(probs_tensor)

        # Bag probability should be higher than individual (scalar comparison)
        assert bag_prob.item() >= probs_tensor.max().item() - 0.01


# --------------------------------------------------------------------------
# Model Configuration Tests
# --------------------------------------------------------------------------

class TestPrestoConfig:
    """Test model configuration options."""

    def test_different_model_sizes(self):
        from presto.models.presto import Presto
        small = Presto(d_model=64, n_layers=2, n_heads=2)
        large = Presto(d_model=256, n_layers=6, n_heads=8)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params

    def test_model_save_load(self, tmp_path):
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)

        # Save
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        # Load into new model
        model2 = Presto(d_model=64, n_layers=2, n_heads=4)
        model2.load_state_dict(torch.load(path))

        # Should give same outputs
        model.eval()
        model2.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out1 = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            out2 = model2(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")

        assert torch.allclose(out1["pmhc_vec"], out2["pmhc_vec"])


# --------------------------------------------------------------------------
# Design Alignment Tests
# --------------------------------------------------------------------------

class TestDesignAlignment:
    """Tests verifying code matches design.md specification."""

    def test_latent_order_has_12_latents(self):
        """Design S7.1: 12 latent query tokens (11 original + species_of_origin)."""
        from presto.models.presto import Presto
        assert len(Presto.LATENT_ORDER) == 12

    def test_ms_detectability_in_latent_order(self):
        """Design S7.1: ms_detectability is the 11th latent."""
        from presto.models.presto import Presto
        assert "ms_detectability" in Presto.LATENT_ORDER

    def test_presentation_latents_have_no_token_access(self):
        """Design S7.5: presentation latents have empty segment lists."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["presentation_class1"] == []
        assert Presto.LATENT_SEGMENTS["presentation_class2"] == []

    def test_recognition_latents_see_peptide_only(self):
        """Design S7.5: recognition latents see only peptide tokens."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["recognition_cd8"] == ["peptide"]
        assert Presto.LATENT_SEGMENTS["recognition_cd4"] == ["peptide"]

    def test_immunogenicity_latents_have_no_token_access(self):
        """Design S7.5: immunogenicity latents are MLP only."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["immunogenicity_cd8"] == []
        assert Presto.LATENT_SEGMENTS["immunogenicity_cd4"] == []

    def test_ms_detectability_is_peptide_only(self):
        """Design S7.5: ms_detectability sees only peptide."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["ms_detectability"] == ["peptide"]

    def test_recognition_depends_on_foreignness(self):
        """Recognition latents depend on peptide-attended foreignness signal."""
        from presto.models.presto import Presto
        assert Presto.LATENT_DEPS["recognition_cd8"] == ["foreignness"]
        assert Presto.LATENT_DEPS["recognition_cd4"] == ["foreignness"]

    def test_immunogenicity_deps_match_design(self):
        """Design S7.2: immunogenicity depends on binding + recognition."""
        from presto.models.presto import Presto
        assert set(Presto.LATENT_DEPS["immunogenicity_cd8"]) == {
            "binding_affinity", "binding_stability", "recognition_cd8"
        }
        assert set(Presto.LATENT_DEPS["immunogenicity_cd4"]) == {
            "binding_affinity", "binding_stability", "recognition_cd4"
        }

    def test_immunogenicity_uses_mlp_not_cross_attention(self):
        """Design S7.4: immunogenicity is MLP, not cross-attention."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert "immunogenicity_cd8" not in model.latent_layers
        assert "immunogenicity_cd4" not in model.latent_layers
        assert hasattr(model, "immunogenicity_cd8_mlp")
        assert hasattr(model, "immunogenicity_cd4_mlp")

    def test_two_layer_cross_attention_per_latent(self):
        """Design S7.3: N_latent = 2 cross-attention layers per latent."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        for name in Presto.CROSS_ATTN_LATENTS:
            assert len(model.latent_layers[name]) == 2

    def test_per_chain_mhc_inference(self):
        """Design S5.1-S5.2: per-chain type and species classification."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok)

        assert "mhc_a_type_logits" in out
        assert "mhc_b_type_logits" in out
        from presto.data.vocab import N_MHC_CHAIN_FINE_TYPES
        assert out["mhc_a_type_logits"].shape == (2, N_MHC_CHAIN_FINE_TYPES)
        assert out["mhc_b_type_logits"].shape == (2, N_MHC_CHAIN_FINE_TYPES)
        assert "mhc_a_species_logits" in out
        assert "mhc_b_species_logits" in out
        assert "chain_compat_logit" in out
        assert "chain_compat_prob" in out

    def test_ms_detectability_output(self):
        """Design S7.4: ms_detectability produces a readout logit."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")

        assert "ms_detectability_logit" in out
        assert out["ms_detectability_logit"].shape == (2, 1)
        assert "ms_detectability" in out["latent_vecs"]

    def test_pmhc_vec_from_latent_vectors(self):
        """Design S9.7: pmhc_vec from latent vectors, not segment pools."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            lvecs = out["latent_vecs"]
            expected = model.pmhc_vec_proj(torch.cat([
                lvecs["binding_affinity"],
                lvecs["presentation_class1"],
                lvecs["presentation_class2"],
            ], dim=-1))

        assert torch.allclose(out["pmhc_vec"], expected, atol=1e-5)

    def test_segment_specific_positional_encoding(self):
        """Design S3.2.3: segment-specific positional encoding tables exist."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert hasattr(model, "pep_nterm_pos")
        assert hasattr(model, "pep_cterm_pos")
        assert hasattr(model, "pep_frac_mlp")
        assert hasattr(model, "nflank_dist_pos")
        assert hasattr(model, "cflank_dist_pos")
        assert hasattr(model, "mhc_a_pos")
        assert hasattr(model, "mhc_b_pos")
        assert not hasattr(model, "position_embedding")

    def test_global_conditioning_embedding(self):
        """Design S3.2.4: global conditioning embedding tables exist."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert hasattr(model, "species_cond_embed")
        assert model.species_cond_embed.num_embeddings == 7
        assert not hasattr(model, "mhc_class_cond_embed")
        assert hasattr(model, "chain_completeness_embed")
        assert model.chain_completeness_embed.num_embeddings == 64

    def test_missing_segment_uses_dedicated_missing_token(self):
        """Missing optional segments should map to <MISSING>, not <UNK>."""
        from presto.models.presto import Presto
        from presto.data.vocab import AA_TO_IDX

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        seg = model._ensure_optional_segment(None, batch_size=2, device=torch.device("cpu"))
        assert seg.shape == (2, 1)
        assert torch.all(seg == AA_TO_IDX["<MISSING>"])

    def test_x_embedding_row_is_fixed_zero(self):
        from presto.models.presto import Presto
        from presto.data.vocab import AA_TO_IDX

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        x_idx = AA_TO_IDX["X"]
        with torch.no_grad():
            x_row = model.aa_embedding.weight[x_idx]
        assert torch.allclose(x_row, torch.zeros_like(x_row))

        tok = torch.tensor([[x_idx, x_idx]], dtype=torch.long)
        out = model.aa_embedding(tok).sum()
        out.backward()
        grad = model.aa_embedding.weight.grad
        assert grad is not None
        assert torch.allclose(grad[x_idx], torch.zeros_like(grad[x_idx]))

    def test_stream_builder_has_no_class_embedding_input(self):
        """Class is no longer a token-stream conditioning input."""
        import inspect
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        sig = inspect.signature(model._build_single_stream)
        assert "mhc_class_id" not in sig.parameters

    def test_forward_can_emit_binding_attention_support_stats(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                return_binding_attention=True,
            )

        assert "binding_mhc_attention_effective_residues" in out
        assert "binding_mhc_attention_mass" in out
        assert out["binding_mhc_attention_effective_residues"].shape[0] == pep_tok.shape[0]
        assert torch.all(out["binding_mhc_attention_effective_residues"] >= 0)

    def test_presentation_latent_branch_has_step1_gradients(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=1, n_heads=4)
        model.train()

        pep_tok = torch.randint(4, 24, (4, 12))
        mhc_a_tok = torch.randint(4, 24, (4, 100))
        mhc_b_tok = torch.randint(4, 24, (4, 40))
        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
        )
        loss = outputs["presentation_logit"].mean()
        loss.backward()

        q1_grad = model.latent_queries["presentation_class1"].grad
        q2_grad = model.latent_queries["presentation_class2"].grad
        h1_grad = model.presentation_class1_latent_head.weight.grad
        h2_grad = model.presentation_class2_latent_head.weight.grad

        assert q1_grad is not None
        assert q2_grad is not None
        assert h1_grad is not None
        assert h2_grad is not None
        assert float(q1_grad.abs().max().item()) > 0.0
        assert float(q2_grad.abs().max().item()) > 0.0
        assert float(h1_grad.abs().max().item()) > 0.0
        assert float(h2_grad.abs().max().item()) > 0.0

    def test_recognition_only_uses_peptide_and_foreignness(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4, enable_tcr=True)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok_1 = torch.randint(4, 24, (2, 80))
        mhc_b_tok_1 = torch.randint(4, 24, (2, 40))
        mhc_a_tok_2 = torch.randint(4, 24, (2, 80))
        mhc_b_tok_2 = torch.randint(4, 24, (2, 40))
        flank_n_tok_1 = torch.randint(4, 24, (2, 6))
        flank_c_tok_1 = torch.randint(4, 24, (2, 6))
        flank_n_tok_2 = torch.zeros((2, 6), dtype=torch.long)
        flank_c_tok_2 = torch.zeros((2, 6), dtype=torch.long)
        tcr_a_tok_1 = torch.randint(4, 24, (2, 16))
        tcr_b_tok_1 = torch.randint(4, 24, (2, 16))
        tcr_a_tok_2 = torch.randint(4, 24, (2, 16))
        tcr_b_tok_2 = torch.randint(4, 24, (2, 16))

        with torch.no_grad():
            out_ref = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok_1,
                mhc_b_tok=mhc_b_tok_1,
                flank_n_tok=flank_n_tok_1,
                flank_c_tok=flank_c_tok_1,
                tcr_a_tok=tcr_a_tok_1,
                tcr_b_tok=tcr_b_tok_1,
            )
            out_changed_context = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok_2,
                mhc_b_tok=mhc_b_tok_2,
                flank_n_tok=flank_n_tok_2,
                flank_c_tok=flank_c_tok_2,
                tcr_a_tok=tcr_a_tok_2,
                tcr_b_tok=tcr_b_tok_2,
            )

        assert torch.allclose(
            out_ref["recognition_cd8_logit"],
            out_changed_context["recognition_cd8_logit"],
            atol=1e-6,
        )
        assert torch.allclose(
            out_ref["recognition_cd4_logit"],
            out_changed_context["recognition_cd4_logit"],
            atol=1e-6,
        )

    def test_species_of_origin_override_controls_foreignness_and_recognition(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 80))
        mhc_b_tok = torch.randint(4, 24, (2, 40))

        with torch.no_grad():
            out_default = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
            )
            out_foreign = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                species_of_origin=["viruses", "bacteria"],
            )

        assert not torch.allclose(
            out_default["foreignness_logit"],
            out_foreign["foreignness_logit"],
            atol=1e-6,
        )
        assert not torch.allclose(
            out_default["recognition_cd8_logit"],
            out_foreign["recognition_cd8_logit"],
            atol=1e-6,
        )

    def test_split_mixed_outputs_match_class_weighted_mixtures(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 80))
        mhc_b_tok = torch.randint(4, 24, (2, 40))
        with torch.no_grad():
            out = model(pep_tok=pep_tok, mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok)

        p1 = out["mhc_class_probs"][:, :1]
        p2 = out["mhc_class_probs"][:, 1:2]
        expected_processing = p1 * out["processing_class1_logit"] + p2 * out["processing_class2_logit"]
        expected_binding = p1 * out["binding_class1_logit"] + p2 * out["binding_class2_logit"]
        expected_presentation = p1 * out["presentation_class1_logit"] + p2 * out["presentation_class2_logit"]
        expected_recognition = p1 * out["recognition_cd8_logit"] + p2 * out["recognition_cd4_logit"]

        assert torch.allclose(out["processing_mixed_logit"], expected_processing, atol=1e-6)
        assert torch.allclose(out["binding_mixed_logit"], expected_binding, atol=1e-6)
        assert torch.allclose(out["presentation_mixed_logit"], expected_presentation, atol=1e-6)
        assert torch.allclose(out["recognition_mixed_logit"], expected_recognition, atol=1e-6)
        assert torch.allclose(out["immunogenicity_mixed_logit"], out["immunogenicity_mixture_logit"], atol=1e-6)
