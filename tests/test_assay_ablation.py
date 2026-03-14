"""Smoke tests for assay ablation probe variants.

Verifies that each of the 8 ablation variants (a1-a8) can:
1. Produce correct output shapes on a forward pass
2. Backpropagate a loss through the full model
"""

import pytest
import torch

from presto.scripts.assay_ablation_probe import (
    build_ablation_model,
    VARIANT_NAMES,
    TYPED_VARIANTS,
    CONTEXT_VARIANTS,
)

# Small dimensions for fast tests
EMBED_DIM = 16
HIDDEN_DIM = 16
N_HEADS = 2
N_LAYERS = 1
BATCH = 4
PEP_LEN = 9
MHC_LEN = 20


def _make_model(variant):
    return build_ablation_model(
        variant,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )


def _make_inputs():
    """Random token tensors for peptide and MHC chains."""
    pep_tok = torch.randint(1, 20, (BATCH, PEP_LEN))
    mhc_a_tok = torch.randint(1, 20, (BATCH, MHC_LEN))
    mhc_b_tok = torch.randint(1, 20, (BATCH, MHC_LEN))
    return pep_tok, mhc_a_tok, mhc_b_tok


def _forward_with_context(model, variant, pep_tok, mhc_a_tok, mhc_b_tok):
    """Forward pass, injecting context kwargs for variants that need them."""
    kwargs = {}
    if variant in CONTEXT_VARIANTS:
        kwargs["assay_type_idx"] = torch.randint(0, 3, (BATCH,))
        if variant == "a5":
            kwargs["assay_method_idx"] = torch.randint(0, 3, (BATCH,))
    return model(pep_tok, mhc_a_tok, mhc_b_tok, **kwargs)


# --------------------------------------------------------------------------
# Forward pass tests
# --------------------------------------------------------------------------


class TestForwardPass:
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_all_variants_forward(self, variant):
        """Each variant produces a dict with 'ic50' of shape (B, 1)."""
        model = _make_model(variant)
        model.eval()
        pep_tok, mhc_a_tok, mhc_b_tok = _make_inputs()

        with torch.no_grad():
            out = _forward_with_context(
                model, variant, pep_tok, mhc_a_tok, mhc_b_tok
            )

        assert isinstance(out, dict), f"Expected dict, got {type(out)}"
        assert "ic50" in out, f"Missing 'ic50' key in output for {variant}"
        assert out["ic50"].shape == (BATCH, 1), (
            f"Expected (4,1), got {out['ic50'].shape} for {variant}"
        )

    def test_typed_variants_multi_output(self):
        """Typed variants (a2, a3, a7) also produce 'kd' and 'ec50'."""
        pep_tok, mhc_a_tok, mhc_b_tok = _make_inputs()

        for variant in sorted(TYPED_VARIANTS):
            model = _make_model(variant)
            model.eval()

            with torch.no_grad():
                out = _forward_with_context(
                    model, variant, pep_tok, mhc_a_tok, mhc_b_tok
                )

            for key in ("ic50", "kd", "ec50"):
                assert key in out, f"{variant} missing '{key}' key"
                assert out[key].shape == (BATCH, 1), (
                    f"{variant} '{key}' shape {out[key].shape} != (4,1)"
                )

    def test_context_variants_with_context(self):
        """Context variants (a5, a8) work when given explicit context tensors."""
        pep_tok, mhc_a_tok, mhc_b_tok = _make_inputs()

        for variant in sorted(CONTEXT_VARIANTS):
            model = _make_model(variant)
            model.eval()

            kwargs = {"assay_type_idx": torch.randint(0, 3, (BATCH,))}
            if variant == "a5":
                kwargs["assay_method_idx"] = torch.randint(0, 3, (BATCH,))

            with torch.no_grad():
                out = model(pep_tok, mhc_a_tok, mhc_b_tok, **kwargs)

            assert "ic50" in out
            assert out["ic50"].shape == (BATCH, 1)


# --------------------------------------------------------------------------
# Backward pass tests
# --------------------------------------------------------------------------


class TestBackwardPass:
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_all_variants_backward(self, variant):
        """Loss backpropagates through each variant without error."""
        model = _make_model(variant)
        model.train()
        pep_tok, mhc_a_tok, mhc_b_tok = _make_inputs()

        out = _forward_with_context(
            model, variant, pep_tok, mhc_a_tok, mhc_b_tok
        )

        target = torch.randn(BATCH, 1)
        loss = torch.nn.functional.mse_loss(out["ic50"], target)
        loss.backward()

        # At least some parameters should have gradients
        grads = [
            p.grad
            for p in model.parameters()
            if p.grad is not None
        ]
        assert len(grads) > 0, f"No gradients computed for {variant}"


# --------------------------------------------------------------------------
# predict_ic50 method
# --------------------------------------------------------------------------


class TestPredictIC50:
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_predict_ic50(self, variant):
        """predict_ic50 returns tensor of shape (B, 1)."""
        model = _make_model(variant)
        model.eval()
        pep_tok, mhc_a_tok, mhc_b_tok = _make_inputs()

        with torch.no_grad():
            ic50 = model.predict_ic50(pep_tok, mhc_a_tok, mhc_b_tok)

        assert ic50.shape == (BATCH, 1), (
            f"predict_ic50 shape {ic50.shape} != (4,1) for {variant}"
        )
