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

    def test_model_init_scales_latent_queries_and_embeddings(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        query_norms = []
        for param in model.latent_queries.values():
            rows = param.detach().reshape(-1, param.shape[-1])
            query_norms.extend(rows.norm(dim=-1).tolist())

        assert query_norms
        assert min(query_norms) > 0.4
        assert max(query_norms) < 2.0
        assert torch.allclose(
            model.aa_embedding.weight[0],
            torch.zeros_like(model.aa_embedding.weight[0]),
        )
        assert 0.05 < float(model.segment_embedding.weight.std().item()) < 0.25

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
        assert "binding_affinity_score" in outputs
        assert "binding_stability_score" in outputs
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
        assert "core_window_mask" in outputs
        assert "core_window_start" in outputs
        assert "core_window_posterior_prob" in outputs
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

    def test_model_forward_receptor_free(self):
        """Forward pass stays receptor-free under canonical contract."""
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

        assert "immunogenicity_logit" in outputs
        assert "immunogenicity_cd8_logit" in outputs
        assert "immunogenicity_cd4_logit" in outputs
        assert "tcell_context_logits" in outputs
        assert "tcr_evidence_logit" in outputs
        assert "tcr_evidence_method_logits" in outputs

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

    def test_model_forward_supports_mhcflurry_affinity_target_encoding(self):
        from presto.models.presto import Presto

        model = Presto(
            d_model=64,
            n_layers=2,
            n_heads=4,
            affinity_target_encoding="mhcflurry",
            max_affinity_nM=100000.0,
        )
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

        assert outputs["assays"]["IC50_nM"].shape == (2, 1)
        assert torch.isfinite(outputs["assays"]["IC50_nM"]).all()

    def test_model_accepts_binding_context_input(self):
        """Verify factorized assay context embeddings are wired through forward()."""
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))
        binding_context = {
            "assay_type_idx": torch.tensor([0, 1], dtype=torch.long),
            "assay_method_idx": torch.tensor([1, 2], dtype=torch.long),
            "assay_prep_idx": torch.tensor([1, 2], dtype=torch.long),
            "assay_geometry_idx": torch.tensor([1, 2], dtype=torch.long),
            "assay_readout_idx": torch.tensor([1, 2], dtype=torch.long),
        }

        with torch.no_grad():
            outputs = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class="I",
                binding_context=binding_context,
            )
        # factorized context vec should be non-zero when binding_context is provided
        fac_vec = outputs.get("binding_factorized_assay_context_vec")
        assert fac_vec is not None
        assert fac_vec.abs().mean() > 0, "Factorized assay context should be non-zero"

    @pytest.mark.parametrize(
        "peptide_pos_mode",
        ["abs_only", "triple_plus_abs", "start_only", "mlp_start_end_frac"],
    )
    @pytest.mark.parametrize(
        "groove_pos_mode",
        ["abs_only", "triple_plus_abs", "start_plus_end", "concat_start_end_frac"],
    )
    def test_model_forward_with_extended_position_modes(
        self,
        peptide_pos_mode: str,
        groove_pos_mode: str,
    ):
        from presto.models.presto import Presto

        model = Presto(
            d_model=64,
            n_layers=2,
            n_heads=4,
            peptide_pos_mode=peptide_pos_mode,
            groove_pos_mode=groove_pos_mode,
        )
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

        assert outputs["pmhc_vec"].shape[0] == 2
        assert outputs["assays"]["IC50_nM"].shape[0] == 2

    def test_forward_affinity_only_matches_full_forward(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            full = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class="I",
            )
            affinity = model.forward_affinity_only(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class="I",
            )

        for key in (
            "binding_logit",
            "binding_affinity_score",
            "binding_affinity_probe_kd",
            "binding_stability_score",
            "binding_mixed_kd_log10",
            "binding_class1_logit",
            "binding_class2_logit",
            "pmhc_vec",
            "groove_vec",
        ):
            assert key in affinity
            assert torch.allclose(affinity[key], full[key], atol=1e-6)
        assert torch.allclose(
            affinity["assays"]["KD_nM"],
            full["assays"]["KD_nM"],
            atol=1e-6,
        )

    def test_model_rejects_noncanonical_affinity_assay_mode(self):
        from presto.models.presto import Presto

        with pytest.raises(TypeError):
            Presto(d_model=64, n_layers=2, n_heads=4, affinity_assay_mode="legacy")

    def test_triple_plus_abs_positions_and_segment_residual_forward(self):
        from presto.models.presto import Presto

        model = Presto(
            d_model=64,
            n_layers=2,
            n_heads=4,
            peptide_pos_mode="triple_plus_abs",
            groove_pos_mode="triple_plus_abs",
            affinity_assay_residual_mode="shared_base_segment_residual",
            core_window_lengths=(8, 9, 10, 11),
            core_refinement_mode="shared",
        )
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 11))
        mhc_a_tok = torch.randint(4, 24, (2, 91))
        mhc_b_tok = torch.randint(4, 24, (2, 93))

        with torch.no_grad():
            out = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=["I", "I"],
            )

        assert out["binding_sequence_summary_vec"].shape == (2, 64)
        assert out["binding_affinity_score"].shape == (2, 1)
        assert out["assays"]["IC50_nM"].shape == (2, 1)

    def test_split_kd_forward_without_assay_selector_inputs(self):
        from presto.models.presto import Presto

        model = Presto(
            d_model=64,
            n_layers=2,
            n_heads=4,
            affinity_assay_residual_mode="shared_base_factorized_context_plus_segment_residual",
            kd_grouping_mode="split_kd_proxy",
        )
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 91))
        mhc_b_tok = torch.randint(4, 24, (2, 93))

        with torch.no_grad():
            out = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class="I",
            )

        assert torch.count_nonzero(out["binding_factorized_assay_context_vec"]) == 0
        assert out["assays"]["KD_nM"].shape == (2, 1)
        assert out["assays"]["KD_proxy_ic50_nM"].shape == (2, 1)
        assert out["assays"]["KD_proxy_ec50_nM"].shape == (2, 1)

    def test_forward_mhc_only_returns_aux_logits(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        mhc_a_tok = torch.randint(4, 24, (3, 91))
        mhc_b_tok = torch.randint(4, 24, (3, 93))

        outputs = model.forward_mhc_only(mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok)

        assert outputs["mhc_a_vec"].shape == (3, 64)
        assert outputs["mhc_b_vec"].shape == (3, 64)
        assert outputs["mhc_a_type_logits"].shape[0] == 3
        assert outputs["mhc_b_type_logits"].shape[0] == 3
        assert outputs["mhc_a_species_logits"].shape[0] == 3
        assert outputs["mhc_b_species_logits"].shape[0] == 3

    def test_model_forward_under_cpu_bf16_autocast(self):
        """Autocast forward should not hit mixed-dtype indexed writes."""
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=1, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 180))
        mhc_b_tok = torch.randint(4, 24, (2, 90))
        flank_n_tok = torch.randint(4, 24, (2, 5))
        flank_c_tok = torch.randint(4, 24, (2, 5))

        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=["I", "II"],
                species=["human", "mouse"],
                flank_n_tok=flank_n_tok,
                flank_c_tok=flank_c_tok,
            )

        assert torch.isfinite(outputs["binding_logit"]).all()
        assert torch.isfinite(outputs["presentation_logit"]).all()

    def test_model_forward_tolerates_bf16_positional_segment_source(self):
        """Segment positional assembly should not rely on fp32 slice writes."""
        from presto.models.presto import Presto

        class _BFloat16Wrap(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x).to(torch.bfloat16)

        model = Presto(d_model=64, n_layers=1, n_heads=4)
        model.pep_frac_mlp = _BFloat16Wrap(model.pep_frac_mlp)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 180))
        mhc_b_tok = torch.randint(4, 24, (2, 90))
        flank_n_tok = torch.randint(4, 24, (2, 5))
        flank_c_tok = torch.randint(4, 24, (2, 5))

        with torch.no_grad():
            outputs = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=["I", "II"],
                species=["human", "mouse"],
                flank_n_tok=flank_n_tok,
                flank_c_tok=flank_c_tok,
            )

        assert torch.isfinite(outputs["binding_logit"]).all()


class TestPrestoGradients:
    """Test gradient flow through full model."""

    def test_full_model_is_differentiable(self):
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        outputs = model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class="I",
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
        model.binding_probe_mix_logit.data.fill_(-12.0)

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
        # Hard pre-clamping to the 50k ceiling would collapse to exactly
        # log10(50000)-softplus(0) ~ 4.0058. Soft bounding should stay
        # noticeably above that.
        assert torch.all(kd > 4.55)
        assert torch.all(kd < model.max_log10_nM + 1e-6)

    def test_presentation_logits_are_readouts_of_class_specific_vectors(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            # Class-specific presentation vecs computed from distinct projections
            pres_class1_vec = out["latent_vecs"]["presentation_class1"]
            pres_class2_vec = out["latent_vecs"]["presentation_class2"]
            class1_base = model.presentation_class1_latent_head(pres_class1_vec)
            class2_base = model.presentation_class2_latent_head(pres_class2_vec)
            class1_prob = out["mhc_class_probs"][:, :1]
            class2_prob = out["mhc_class_probs"][:, 1:2]
            expected = class1_prob * class1_base + class2_prob * class2_base

        assert torch.allclose(out["presentation_class1_logit"], class1_base, atol=1e-5)
        assert torch.allclose(out["presentation_class2_logit"], class2_base, atol=1e-5)
        assert torch.allclose(out["presentation_logit"], expected, atol=1e-5)

    def test_ic50_ec50_share_kd_latent_calibration(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            ba_vec = out["latent_vecs"]["binding_affinity"]
            derived = model.assay_heads.derive_affinity_observables(
                ba_vec,
                out["binding_mixed_kd_log10"],
                assay_context_vec=out["binding_assay_context_vec"],
                binding_affinity_score=out["binding_affinity_score"],
            )

        assert torch.allclose(out["assays"]["IC50_nM"], derived["IC50_nM"], atol=1e-5)
        assert torch.allclose(out["assays"]["EC50_nM"], derived["EC50_nM"], atol=1e-5)

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
        assert "nflank" not in Presto.LATENT_SEGMENTS["pmhc_interaction"]
        assert "cflank" not in Presto.LATENT_SEGMENTS["pmhc_interaction"]


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

    def test_latent_order_has_refactored_core_latents(self):
        """Phase 1 refactor: cross-attention DAG is reduced to shared core concepts."""
        from presto.models.presto import Presto
        assert Presto.LATENT_ORDER == [
            "processing",
            "ms_detectability",
            "species_of_origin",
            "pmhc_interaction",
            "recognition",
        ]

    def test_ms_detectability_in_latent_order(self):
        """Design S7.1: ms_detectability is the 11th latent."""
        from presto.models.presto import Presto
        assert "ms_detectability" in Presto.LATENT_ORDER

    def test_presentation_uses_mlp_not_cross_attention(self):
        """Presentation is derived from class-specific MLPs, not a latent query."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert "presentation" not in Presto.CROSS_ATTN_LATENTS
        assert hasattr(model, "presentation_class1_mlp")
        assert hasattr(model, "presentation_class2_mlp")

    def test_recognition_latents_see_peptide_only(self):
        """Design S7.5: recognition latents see only peptide tokens."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["recognition"] == ["peptide"]

    def test_immunogenicity_uses_mlp_not_cross_attention(self):
        """Immunogenicity uses lineage-specific MLPs over interaction + recognition."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert "immunogenicity" not in Presto.CROSS_ATTN_LATENTS
        assert hasattr(model, "immunogenicity_cd8_mlp")
        assert hasattr(model, "immunogenicity_cd4_mlp")

    def test_ms_detectability_is_peptide_only(self):
        """Design S7.5: ms_detectability sees only peptide."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["ms_detectability"] == ["peptide"]

    def test_recognition_depends_on_foreignness(self):
        """Recognition latents depend on peptide-attended foreignness signal."""
        from presto.models.presto import Presto
        assert Presto.LATENT_DEPS["recognition"] == ["foreignness"]

    def test_pmhc_interaction_is_the_only_mhc_aware_cross_attention_latent(self):
        """Phase 1 refactor: MHC sequence enters the DAG through pmhc_interaction."""
        from presto.models.presto import Presto
        assert Presto.LATENT_SEGMENTS["pmhc_interaction"] == ["peptide", "mhc_a", "mhc_b"]
        assert "nflank" not in Presto.LATENT_SEGMENTS["pmhc_interaction"]
        assert "cflank" not in Presto.LATENT_SEGMENTS["pmhc_interaction"]

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

    def test_pmhc_vec_is_interaction_vec(self):
        """pmhc_vec is now a direct alias for interaction_vec (no projection)."""
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        with torch.no_grad():
            out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")

        assert torch.allclose(out["pmhc_vec"], out["latent_vecs"]["pmhc_interaction"], atol=1e-5)

    def test_binding_kinetic_input_modes_forward(self):
        from presto.models.presto import Presto

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        for mode in ("affinity_vec", "interaction_vec", "fused"):
            model = Presto(
                d_model=64,
                n_layers=2,
                n_heads=4,
                binding_kinetic_input_mode=mode,
            )
            model.eval()
            with torch.no_grad():
                out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            assert out["binding_logit"].shape == (2,)
            assert out["binding_latents"]["log_koff"].shape == (2, 1)
            assert out["binding_kinetic_input_mode"] == mode

    def test_binding_direct_segment_modes_forward(self):
        from presto.models.presto import Presto

        pep_tok = torch.randint(4, 24, (2, 10))
        mhc_a_tok = torch.randint(4, 24, (2, 50))
        mhc_b_tok = torch.randint(4, 24, (2, 20))

        for mode in ("off", "affinity_residual", "affinity_stability_residual", "gated_affinity"):
            model = Presto(
                d_model=64,
                n_layers=2,
                n_heads=4,
                binding_direct_segment_mode=mode,
            )
            model.eval()
            with torch.no_grad():
                out = model(pep_tok, mhc_a_tok, mhc_b_tok, mhc_class="I")
            assert out["binding_logit"].shape == (2,)
            assert out["binding_direct_segment_mode"] == mode
            assert out["binding_direct_affinity_vec"].shape == (2, 64)
            assert out["binding_direct_stability_vec"].shape == (2, 64)

    def test_apc_cell_type_context_alias_exists(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        assert hasattr(model, "apc_cell_type_context_proj")

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

    def test_groove_vec_is_invariant_to_class_override(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 80))
        mhc_b_tok = torch.randint(4, 24, (2, 40))

        with torch.no_grad():
            out_class1 = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class="I",
            )
            out_class2 = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class="II",
            )

        assert out_class1["groove_vec"].shape == (2, 64)
        assert torch.allclose(
            out_class1["groove_vec"],
            out_class2["groove_vec"],
            atol=1e-6,
        )

    def test_core_window_enumeration_respects_short_and_long_peptides(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()

        pep_tok = torch.randint(4, 24, (3, 15))
        pep_tok[0, 8:] = 0   # 8-mer -> 1 candidate, width 8
        pep_tok[1, 11:] = 0  # 11-mer -> 3 candidates, width 9
        mhc_a_tok = torch.randint(4, 24, (3, 64))
        mhc_b_tok = torch.randint(4, 24, (3, 32))

        with torch.no_grad():
            out = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=["I", "I", "II"],
            )

        assert out["core_window_mask"].sum(dim=1).tolist() == [1, 3, 7]
        row0_lengths = out["core_window_length"][0][out["core_window_mask"][0]].tolist()
        row1_lengths = out["core_window_length"][1][out["core_window_mask"][1]].tolist()
        row2_lengths = out["core_window_length"][2][out["core_window_mask"][2]].tolist()
        assert row0_lengths == [8]
        assert row1_lengths == [9, 9, 9]
        assert row2_lengths == [9, 9, 9, 9, 9, 9, 9]
        assert torch.allclose(
            out["core_window_posterior_prob"].sum(dim=1),
            torch.ones(3),
            atol=1e-6,
        )
        assert torch.all(out["core_start_prob"].sum(dim=1) > 0.999)

    def test_triple_groove_positions_and_variable_core_lengths_forward(self):
        from presto.models.presto import Presto

        model = Presto(
            d_model=64,
            n_layers=2,
            n_heads=4,
            groove_pos_mode="triple",
            core_window_lengths=(8, 9, 10, 11),
            core_refinement_mode="class_specific",
        )
        model.eval()

        pep_tok = torch.randint(4, 24, (3, 15))
        pep_tok[0, 8:] = 0   # 8-mer
        pep_tok[1, 11:] = 0  # 11-mer
        mhc_a_tok = torch.randint(4, 24, (3, 91))
        mhc_b_tok = torch.randint(4, 24, (3, 93))

        with torch.no_grad():
            out = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=["I", "I", "II"],
            )

        assert out["groove_vec"].shape == (3, 64)
        assert out["core_window_mask"].sum(dim=1).tolist() == [1, 10, 26]
        observed_lengths = sorted(
            {
                int(length)
                for length, mask in zip(
                    out["core_window_length"][2].tolist(),
                    out["core_window_mask"][2].tolist(),
                )
                if mask
            }
        )
        assert observed_lengths == [8, 9, 10, 11]
        assert torch.allclose(
            out["core_window_posterior_prob"].sum(dim=1),
            torch.ones(3),
            atol=1e-6,
        )

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

        interaction_grad = model.latent_queries["pmhc_interaction"].grad
        mlp1_grad = model.presentation_class1_mlp[0].weight.grad
        mlp2_grad = model.presentation_class2_mlp[0].weight.grad
        h1_grad = model.presentation_class1_latent_head.weight.grad
        h2_grad = model.presentation_class2_latent_head.weight.grad

        assert interaction_grad is not None
        assert mlp1_grad is not None
        assert mlp2_grad is not None
        assert h1_grad is not None
        assert h2_grad is not None
        assert float(interaction_grad.abs().max().item()) > 0.0
        assert float(mlp1_grad.abs().max().item()) > 0.0
        assert float(mlp2_grad.abs().max().item()) > 0.0
        assert float(h1_grad.abs().max().item()) > 0.0
        assert float(h2_grad.abs().max().item()) > 0.0

    def test_recognition_only_uses_peptide_and_foreignness(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
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
        with torch.no_grad():
            out_ref = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok_1,
                mhc_b_tok=mhc_b_tok_1,
                flank_n_tok=flank_n_tok_1,
                flank_c_tok=flank_c_tok_1,
            )
            out_changed_context = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok_2,
                mhc_b_tok=mhc_b_tok_2,
                flank_n_tok=flank_n_tok_2,
                flank_c_tok=flank_c_tok_2,
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

    def test_mhc_species_override_conditions_processing_path(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 80))
        mhc_b_tok = torch.randint(4, 24, (2, 40))

        with torch.no_grad():
            out_human = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_species="human",
            )
            out_fish = model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_species="other_vertebrate",
            )

        assert not torch.allclose(
            out_human["processing_logit"],
            out_fish["processing_logit"],
            atol=1e-6,
        )

    def test_core_position_embedding_does_not_leak_into_processing_or_ms(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 80))
        mhc_b_tok = torch.randint(4, 24, (2, 40))

        with torch.no_grad():
            out_before = model(pep_tok=pep_tok, mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok)
            model.core_position_embed.weight.uniform_(-10.0, 10.0)
            out_after = model(pep_tok=pep_tok, mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok)

        assert torch.allclose(
            out_before["processing_logit"],
            out_after["processing_logit"],
            atol=1e-6,
        )
        assert torch.allclose(
            out_before["ms_detectability_logit"],
            out_after["ms_detectability_logit"],
            atol=1e-6,
        )
        assert not torch.allclose(
            out_before["binding_logit"],
            out_after["binding_logit"],
            atol=1e-6,
        )

    def test_groove_context_does_not_leak_into_processing(self):
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        model.eval()
        pep_tok = torch.randint(4, 24, (2, 12))
        mhc_a_tok = torch.randint(4, 24, (2, 80))
        mhc_b_tok = torch.randint(4, 24, (2, 40))

        with torch.no_grad():
            out_before = model(pep_tok=pep_tok, mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok)
            model.groove_query.uniform_(-10.0, 10.0)
            out_after = model(pep_tok=pep_tok, mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok)

        assert torch.allclose(
            out_before["processing_logit"],
            out_after["processing_logit"],
            atol=1e-6,
        )
        assert not torch.allclose(
            out_before["groove_vec"],
            out_after["groove_vec"],
            atol=1e-6,
        )
