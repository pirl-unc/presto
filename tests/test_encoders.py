"""Tests for sequence encoders - pins down encoder API."""

import pytest
import torch


class TestSequenceEncoder:
    """Test base SequenceEncoder."""

    def test_encoder_init(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=128, n_layers=2, n_heads=4)
        assert enc.d_model == 128

    def test_encoder_forward_shape(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=128, n_layers=2, n_heads=4)
        # Input: (batch, seq_len)
        x = torch.randint(0, 25, (4, 20))
        # Output: (batch, d_model) for pooled, (batch, seq_len, d_model) for full
        pooled, full = enc(x)
        assert pooled.shape == (4, 128)
        assert full.shape == (4, 20, 128)

    def test_encoder_with_mask(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=128, n_layers=2, n_heads=4)
        x = torch.randint(0, 25, (4, 20))
        mask = torch.ones(4, 20)
        mask[:, 15:] = 0  # Mask out last 5 positions
        pooled, full = enc(x, mask=mask)
        assert pooled.shape == (4, 128)

    def test_encoder_with_lengths(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=128, n_layers=2, n_heads=4)
        x = torch.randint(0, 25, (4, 20))
        lengths = torch.tensor([10, 15, 20, 8])
        pooled, full = enc(x, lengths=lengths)
        assert pooled.shape == (4, 128)

    def test_encoder_pooling_modes(self):
        from presto.models.encoders import SequenceEncoder
        enc_mean = SequenceEncoder(d_model=64, n_layers=1, n_heads=2, pool="mean")
        enc_cls = SequenceEncoder(d_model=64, n_layers=1, n_heads=2, pool="cls")
        x = torch.randint(0, 25, (2, 10))
        p1, _ = enc_mean(x)
        p2, _ = enc_cls(x)
        assert p1.shape == p2.shape == (2, 64)


class TestProjectionHead:
    """Test projection/L2 normalization."""

    def test_l2_normalize(self):
        from presto.models.encoders import l2_normalize
        x = torch.randn(4, 128)
        x_norm = l2_normalize(x)
        norms = x_norm.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_projection_head(self):
        from presto.models.encoders import ProjectionHead
        proj = ProjectionHead(d_in=128, d_out=64, hidden=256)
        x = torch.randn(4, 128)
        y = proj(x)
        assert y.shape == (4, 64)
        # Output should be L2 normalized
        norms = y.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_sinusoidal_pe(self):
        from presto.models.encoders import SinusoidalPE
        pe = SinusoidalPE(d_model=64, max_len=100)
        x = torch.randn(4, 50, 64)
        y = pe(x)
        assert y.shape == x.shape
        # Position encoding should add to input
        assert not torch.allclose(x, y)

    def test_learnable_pe(self):
        from presto.models.encoders import LearnablePE
        pe = LearnablePE(d_model=64, max_len=100)
        x = torch.randn(4, 50, 64)
        y = pe(x)
        assert y.shape == x.shape


class TestEncoderGradients:
    """Test gradient flow."""

    def test_encoder_is_differentiable(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=64, n_layers=1, n_heads=2)
        x = torch.randint(0, 25, (2, 10))
        pooled, _ = enc(x)
        loss = pooled.sum()
        loss.backward()
        # Check gradients exist for embedding
        assert enc.embedding.weight.grad is not None


class TestEncoderBatching:
    """Test batching behavior."""

    def test_single_sample(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=64, n_layers=1, n_heads=2)
        x = torch.randint(0, 25, (1, 10))
        pooled, full = enc(x)
        assert pooled.shape == (1, 64)
        assert full.shape == (1, 10, 64)

    def test_variable_length_batch(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=64, n_layers=1, n_heads=2)
        # Simulate padded batch
        x = torch.randint(0, 25, (3, 15))
        x[0, 8:] = 0  # Pad positions
        x[1, 12:] = 0
        lengths = torch.tensor([8, 12, 15])
        pooled, full = enc(x, lengths=lengths)
        assert pooled.shape == (3, 64)


class TestEncoderDeterminism:
    """Test deterministic behavior."""

    def test_eval_mode_deterministic(self):
        from presto.models.encoders import SequenceEncoder
        enc = SequenceEncoder(d_model=64, n_layers=1, n_heads=2)
        enc.eval()
        x = torch.randint(0, 25, (2, 10))
        with torch.no_grad():
            p1, _ = enc(x)
            p2, _ = enc(x)
        assert torch.allclose(p1, p2)
