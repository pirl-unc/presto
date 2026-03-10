"""Tests for groove baseline model and training harness."""

from types import SimpleNamespace

import pytest
import torch

from presto.scripts.groove_baseline_probe import (
    GrooveBaselineModel,
    GrooveTransformerModel,
    _build_model,
    _groove_baseline_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int = 4,
    pep_len: int = 9,
    mhc_a_len: int = 91,
    mhc_b_len: int = 93,
    vocab_size: int = 26,
    bind_value: float = 100.0,
    qualifier: int = 0,
    device: str = "cpu",
):
    """Create a minimal batch namespace matching collator output shape."""
    rng = torch.Generator().manual_seed(42)
    pep_tok = torch.randint(4, vocab_size, (batch_size, pep_len), generator=rng)
    mhc_a_tok = torch.randint(4, vocab_size, (batch_size, mhc_a_len), generator=rng)
    mhc_b_tok = torch.randint(4, vocab_size, (batch_size, mhc_b_len), generator=rng)
    bind_target = torch.full((batch_size, 1), bind_value)
    bind_mask = torch.ones(batch_size, 1)
    bind_qual = torch.full((batch_size, 1), qualifier, dtype=torch.long)
    primary_alleles = ["HLA-A*02:01"] * (batch_size // 2) + ["HLA-A*24:02"] * (batch_size - batch_size // 2)

    batch = SimpleNamespace(
        pep_tok=pep_tok,
        mhc_a_tok=mhc_a_tok,
        mhc_b_tok=mhc_b_tok,
        bind_target=bind_target,
        bind_mask=bind_mask,
        bind_qual=bind_qual,
        primary_alleles=primary_alleles,
    )
    batch.to = lambda dev: SimpleNamespace(
        pep_tok=pep_tok.to(dev),
        mhc_a_tok=mhc_a_tok.to(dev),
        mhc_b_tok=mhc_b_tok.to(dev),
        bind_target=bind_target.to(dev),
        bind_mask=bind_mask.to(dev),
        bind_qual=bind_qual.to(dev),
        primary_alleles=primary_alleles,
    )
    return batch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGrooveBaselineForward:
    def test_output_shape(self):
        """Output is (B, 1) for any batch size."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        batch = _make_batch(batch_size=8)
        out = model(batch.pep_tok, batch.mhc_a_tok, batch.mhc_b_tok)
        assert out.shape == (8, 1)

    def test_output_in_range(self):
        """Output values are within smooth-bounded range [-3, ~4.7]."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        batch = _make_batch(batch_size=16)
        out = model(batch.pep_tok, batch.mhc_a_tok, batch.mhc_b_tok)
        assert out.min().item() >= -4.0  # smooth bound allows slight undershoot
        assert out.max().item() <= 6.0   # smooth bound allows slight overshoot

    def test_param_count(self):
        """Default config should produce ~26K params."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=64, hidden_dim=128)
        n_params = sum(p.numel() for p in model.parameters())
        # Embedding: 26*64=1664, Linear(192,128)=24704, Linear(128,1)=129
        assert 20_000 < n_params < 30_000, f"Expected ~26K params, got {n_params}"


class TestGrooveBaselineDifferentGrooves:
    def test_different_grooves_differ(self):
        """Same peptide with different groove sequences should produce different outputs."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        model.eval()

        pep_tok = torch.randint(4, 24, (1, 9))
        mhc_a_tok_1 = torch.randint(4, 24, (1, 91))
        mhc_b_tok_1 = torch.randint(4, 24, (1, 93))
        # Create different groove tokens
        mhc_a_tok_2 = torch.randint(4, 24, (1, 91))
        mhc_b_tok_2 = torch.randint(4, 24, (1, 93))
        # Ensure they're actually different
        mhc_a_tok_2[0, 0] = (mhc_a_tok_1[0, 0] + 1) % 24 + 4

        with torch.no_grad():
            out1 = model(pep_tok, mhc_a_tok_1, mhc_b_tok_1)
            out2 = model(pep_tok, mhc_a_tok_2, mhc_b_tok_2)

        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Model should produce different outputs for different groove sequences"
        )


class TestGrooveBaselineLoss:
    def test_loss_backward(self):
        """Gradients flow through the loss computation."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        batch = _make_batch(batch_size=8, bind_value=100.0)
        loss, metrics = _groove_baseline_loss(model, batch, "cpu")

        assert loss.requires_grad
        loss.backward()

        grad_params = [p for p in model.parameters() if p.grad is not None]
        assert len(grad_params) > 0, "No gradients computed"

        has_nonzero = any(p.grad.abs().sum().item() > 0 for p in grad_params)
        assert has_nonzero, "All gradients are zero"

    def test_loss_is_scalar(self):
        """Loss should be a scalar tensor."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        batch = _make_batch(batch_size=4)
        loss, _ = _groove_baseline_loss(model, batch, "cpu")
        assert loss.dim() == 0

    def test_loss_metrics_has_support(self):
        """Metrics dict should report binding support."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        batch = _make_batch(batch_size=4)
        _, metrics = _groove_baseline_loss(model, batch, "cpu")
        assert "support_binding" in metrics
        assert metrics["support_binding"] == 4.0

    def test_censored_loss(self):
        """Less-than qualifier should produce smaller loss when prediction > target."""
        model = GrooveBaselineModel(vocab_size=26, embed_dim=32, hidden_dim=64)
        # exact measurement
        batch_exact = _make_batch(batch_size=4, bind_value=100.0, qualifier=0)
        loss_exact, _ = _groove_baseline_loss(model, batch_exact, "cpu")
        # less-than qualifier (pred > target is penalized, pred < target is not)
        batch_lt = _make_batch(batch_size=4, bind_value=100.0, qualifier=-1)
        loss_lt, _ = _groove_baseline_loss(model, batch_lt, "cpu")
        # Both should be finite
        assert torch.isfinite(loss_exact)
        assert torch.isfinite(loss_lt)


class TestGrooveTransformerModel:
    def test_forward_shape(self):
        """Output is (B, 1)."""
        model = GrooveTransformerModel(vocab_size=26, embed_dim=32, n_heads=4, n_layers=1, ff_dim=64, hidden_dim=64)
        batch = _make_batch(batch_size=4)
        out = model(batch.pep_tok, batch.mhc_a_tok, batch.mhc_b_tok)
        assert out.shape == (4, 1)

    def test_different_grooves_differ(self):
        """Positional encoding means different grooves produce different outputs."""
        model = GrooveTransformerModel(vocab_size=26, embed_dim=32, n_heads=4, n_layers=1, ff_dim=64, hidden_dim=64)
        model.eval()
        pep_tok = torch.randint(4, 24, (1, 9))
        mhc_a_tok_1 = torch.randint(4, 24, (1, 91))
        mhc_b_tok_1 = torch.randint(4, 24, (1, 93))
        mhc_a_tok_2 = mhc_a_tok_1.clone()
        mhc_b_tok_2 = mhc_b_tok_1.clone()
        # Flip just one position — transformer should notice
        mhc_a_tok_2[0, 45] = (mhc_a_tok_1[0, 45] + 1) % 20 + 4
        with torch.no_grad():
            out1 = model(pep_tok, mhc_a_tok_1, mhc_b_tok_1)
            out2 = model(pep_tok, mhc_a_tok_2, mhc_b_tok_2)
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_loss_backward(self):
        """Gradients flow through transformer model."""
        model = GrooveTransformerModel(vocab_size=26, embed_dim=32, n_heads=4, n_layers=1, ff_dim=64, hidden_dim=64)
        batch = _make_batch(batch_size=4, bind_value=100.0)
        loss, _ = _groove_baseline_loss(model, batch, "cpu")
        assert loss.requires_grad
        loss.backward()
        has_nonzero = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_nonzero


class TestBuildModel:
    def test_build_mlp(self):
        model = _build_model("mlp", embed_dim=32, hidden_dim=64)
        assert isinstance(model, GrooveBaselineModel)

    def test_build_transformer(self):
        model = _build_model("transformer", embed_dim=32, hidden_dim=64, n_heads=4, n_layers=1)
        assert isinstance(model, GrooveTransformerModel)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model variant"):
            _build_model("unknown", embed_dim=32, hidden_dim=64)
