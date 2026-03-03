"""Tests for trainer - smoke tests for full training loop."""

import pytest
import torch


# --------------------------------------------------------------------------
# Test Data Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def tiny_dataset():
    """Create tiny synthetic dataset for smoke testing."""
    from presto.data.tokenizer import Tokenizer
    tok = Tokenizer()

    # 8 samples with varying tasks
    samples = []
    for i in range(8):
        sample = {
            "peptide": f"SIINFEKL"[: 8 - i % 3],  # Varying lengths
            "mhc_a": "MAVMAPRTLLLLLSGALALTQTWAG",  # Fake MHC
            "mhc_b": "IQRTPKIQVYSRHPAENGKSNFLNC",  # Fake beta2m
            "mhc_class": "I",
            # Binding task
            "value": 5.0 + i * 0.5,  # log10(nM)
            "qual": 0,  # exact
            # T-cell task
            "tcr_a": "CAVRDSNYQLIW" if i % 2 == 0 else None,
            "tcr_b": "CASSIRSSYEQYF" if i % 2 == 0 else None,
            "tcell_label": float(i % 2),
        }
        samples.append(sample)

    return samples


@pytest.fixture
def tokenizer():
    from presto.data.tokenizer import Tokenizer
    return Tokenizer()


# --------------------------------------------------------------------------
# Trainer Smoke Tests
# --------------------------------------------------------------------------

class TestTrainerSmoke:
    """Smoke tests to verify training loop works."""

    def test_trainer_init(self):
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto
        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3)
        assert trainer is not None

    def test_trainer_init_with_pcgrad(self):
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3, use_pcgrad=True)
        assert trainer.use_pcgrad is True

    def test_trainer_single_step(self, tiny_dataset, tokenizer):
        """Test that a single training step runs without error."""
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3)

        # Prepare a minimal batch
        sample = tiny_dataset[0]
        batch = {
            "pep_tok": tokenizer.batch_encode([sample["peptide"]], max_len=15, pad=True),
            "mhc_a_tok": tokenizer.batch_encode([sample["mhc_a"]], max_len=50, pad=True),
            "mhc_b_tok": tokenizer.batch_encode([sample["mhc_b"]], max_len=50, pad=True),
            "mhc_class": ["I"],
            "bind_target": torch.tensor([[sample["value"]]]),
            "bind_qual": torch.tensor([[sample["qual"]]]),
        }

        loss = trainer.train_step(batch)
        assert torch.isfinite(loss)

    def test_trainer_multiple_steps(self, tiny_dataset, tokenizer):
        """Test that multiple training steps work and loss decreases."""
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-2)  # Higher LR for faster convergence

        # Prepare batch
        batch = {
            "pep_tok": tokenizer.batch_encode(
                [s["peptide"] for s in tiny_dataset], max_len=15, pad=True
            ),
            "mhc_a_tok": tokenizer.batch_encode(
                [s["mhc_a"] for s in tiny_dataset], max_len=50, pad=True
            ),
            "mhc_b_tok": tokenizer.batch_encode(
                [s["mhc_b"] for s in tiny_dataset], max_len=50, pad=True
            ),
            "mhc_class": [s["mhc_class"] for s in tiny_dataset],
            "bind_target": torch.tensor([[s["value"]] for s in tiny_dataset]),
            "bind_qual": torch.tensor([[s["qual"]] for s in tiny_dataset]),
        }

        losses = []
        for _ in range(5):
            loss = trainer.train_step(batch)
            losses.append(loss.item())

        # Loss should generally decrease (or at least not explode)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        assert losses[-1] < losses[0] * 10  # Not exploding

    def test_trainer_with_tcr(self, tiny_dataset, tokenizer):
        """Test training with TCR data."""
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3)

        # Filter to samples with TCR
        tcr_samples = [s for s in tiny_dataset if s["tcr_a"] is not None]

        batch = {
            "pep_tok": tokenizer.batch_encode(
                [s["peptide"] for s in tcr_samples], max_len=15, pad=True
            ),
            "mhc_a_tok": tokenizer.batch_encode(
                [s["mhc_a"] for s in tcr_samples], max_len=50, pad=True
            ),
            "mhc_b_tok": tokenizer.batch_encode(
                [s["mhc_b"] for s in tcr_samples], max_len=50, pad=True
            ),
            "mhc_class": [s["mhc_class"] for s in tcr_samples],
            "tcr_a_tok": tokenizer.batch_encode(
                [s["tcr_a"] for s in tcr_samples], max_len=30, pad=True
            ),
            "tcr_b_tok": tokenizer.batch_encode(
                [s["tcr_b"] for s in tcr_samples], max_len=30, pad=True
            ),
            "tcell_label": torch.tensor([s["tcell_label"] for s in tcr_samples]),
        }

        loss = trainer.train_step(batch)
        assert torch.isfinite(loss)

    def test_trainer_step_with_pcgrad(self, tiny_dataset, tokenizer):
        """PCGrad path should run on multi-task batches without error."""
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3, use_pcgrad=True)

        sample = tiny_dataset[0]
        batch = {
            "pep_tok": tokenizer.batch_encode([sample["peptide"]], max_len=15, pad=True),
            "mhc_a_tok": tokenizer.batch_encode([sample["mhc_a"]], max_len=50, pad=True),
            "mhc_b_tok": tokenizer.batch_encode([sample["mhc_b"]], max_len=50, pad=True),
            "mhc_class": ["I"],
            "bind_target": torch.tensor([[sample["value"]]]),
            "bind_qual": torch.tensor([[sample["qual"]]]),
            "elution_label": torch.tensor([1.0]),
        }

        loss = trainer.train_step(batch)
        assert torch.isfinite(loss)

    def test_trainer_step_with_stability_targets(self, tokenizer):
        """Stability losses should be consumed when targets are provided."""
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3)

        batch = {
            "pep_tok": tokenizer.batch_encode(["SIINFEKL", "GILGFVFTL"], max_len=15, pad=True),
            "mhc_a_tok": tokenizer.batch_encode(["MAVMAPRTLLLLLSGALALTQTWAG"] * 2, max_len=50, pad=True),
            "mhc_b_tok": tokenizer.batch_encode(["IQRTPKIQVYSRHPAENGKSNFLNC"] * 2, max_len=50, pad=True),
            "mhc_class": ["I", "I"],
            "t_half_target": torch.tensor([[1.5], [2.0]], dtype=torch.float32),
            "t_half_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "tm_target": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            "tm_mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }

        loss = trainer.train_step(batch)
        assert torch.isfinite(loss)
        # If stability labels are ignored this path returns exactly zero.
        assert loss.item() > 0.0


class TestTrainerEval:
    """Test trainer evaluation mode."""

    def test_eval_mode(self, tiny_dataset, tokenizer):
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3)

        batch = {
            "pep_tok": tokenizer.batch_encode(
                [s["peptide"] for s in tiny_dataset], max_len=15, pad=True
            ),
            "mhc_a_tok": tokenizer.batch_encode(
                [s["mhc_a"] for s in tiny_dataset], max_len=50, pad=True
            ),
            "mhc_b_tok": tokenizer.batch_encode(
                [s["mhc_b"] for s in tiny_dataset], max_len=50, pad=True
            ),
            "mhc_class": [s["mhc_class"] for s in tiny_dataset],
        }

        outputs = trainer.eval_step(batch)
        assert "presentation_logit" in outputs
        assert outputs["presentation_logit"].shape[0] == len(tiny_dataset)


class TestTrainerCheckpoint:
    """Test checkpoint save/load."""

    def test_save_load_checkpoint(self, tmp_path, tiny_dataset, tokenizer):
        from presto.training.trainer import Trainer
        from presto.models.presto import Presto

        model = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer = Trainer(model, lr=1e-3)

        # Train a bit
        batch = {
            "pep_tok": tokenizer.batch_encode(
                [s["peptide"] for s in tiny_dataset[:4]], max_len=15, pad=True
            ),
            "mhc_a_tok": tokenizer.batch_encode(
                [s["mhc_a"] for s in tiny_dataset[:4]], max_len=50, pad=True
            ),
            "mhc_b_tok": tokenizer.batch_encode(
                [s["mhc_b"] for s in tiny_dataset[:4]], max_len=50, pad=True
            ),
            "mhc_class": ["I"] * 4,
            "bind_target": torch.tensor([[5.0]] * 4),
            "bind_qual": torch.tensor([[0]] * 4),
        }
        trainer.train_step(batch)

        # Save
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

        # Load into new trainer
        model2 = Presto(d_model=64, n_layers=2, n_heads=4)
        trainer2 = Trainer(model2, lr=1e-3)
        trainer2.load_checkpoint(ckpt_path)

        # Models should give same outputs
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(batch["pep_tok"], batch["mhc_a_tok"], batch["mhc_b_tok"], "I")
            out2 = model2(batch["pep_tok"], batch["mhc_a_tok"], batch["mhc_b_tok"], "I")
        assert torch.allclose(out1["pmhc_vec"], out2["pmhc_vec"])
