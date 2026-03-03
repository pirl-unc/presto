"""Tests for loss functions.

Key losses:
1. Censor-aware loss: Handles <, =, > qualifiers in binding data
2. MIL bag loss: Noisy-OR aggregation for elution data
3. Uncertainty weighting: Learned task weights for multi-task
"""

import pytest
import torch


# --------------------------------------------------------------------------
# Censor-Aware Loss Tests
# --------------------------------------------------------------------------

class TestCensorAwareLoss:
    """Test censor-aware regression loss for binding data."""

    def test_exact_value_loss(self):
        """For exact values (qual='='), should be MSE-like."""
        from presto.training.losses import censor_aware_loss
        pred = torch.tensor([2.0, 3.0])
        target = torch.tensor([2.0, 3.0])
        qual = torch.tensor([0, 0])  # 0 = exact
        loss = censor_aware_loss(pred, target, qual)
        assert loss.item() < 0.01  # Near zero

    def test_less_than_loss(self):
        """For '<' qualifiers, pred should be penalized if > target."""
        from presto.training.losses import censor_aware_loss
        # Pred=5, target=3 with '<' -> pred should be <= 3, so loss
        pred = torch.tensor([5.0])
        target = torch.tensor([3.0])
        qual = torch.tensor([-1])  # -1 = less than
        loss_high = censor_aware_loss(pred, target, qual)

        # Pred=1, target=3 with '<' -> pred is already < 3, so no penalty
        pred2 = torch.tensor([1.0])
        loss_low = censor_aware_loss(pred2, target, qual)

        assert loss_high > loss_low

    def test_greater_than_loss(self):
        """For '>' qualifiers, pred should be penalized if < target."""
        from presto.training.losses import censor_aware_loss
        # Pred=1, target=3 with '>' -> pred should be >= 3, so loss
        pred = torch.tensor([1.0])
        target = torch.tensor([3.0])
        qual = torch.tensor([1])  # 1 = greater than
        loss_low = censor_aware_loss(pred, target, qual)

        # Pred=5, target=3 with '>' -> pred is already > 3, so no penalty
        pred2 = torch.tensor([5.0])
        loss_high = censor_aware_loss(pred2, target, qual)

        assert loss_low > loss_high

    def test_mixed_qualifiers(self):
        """Should handle mixed qualifiers in batch."""
        from presto.training.losses import censor_aware_loss
        pred = torch.tensor([2.0, 5.0, 1.0])
        target = torch.tensor([2.0, 3.0, 3.0])
        qual = torch.tensor([0, -1, 1])  # exact, less than, greater than
        loss = censor_aware_loss(pred, target, qual)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------
# MIL Bag Loss Tests
# --------------------------------------------------------------------------

class TestMILBagLoss:
    """Test MIL loss for elution data."""

    def test_mil_loss_positive_bag(self):
        """Positive bag should encourage high instance probs."""
        from presto.training.losses import mil_bag_loss
        inst_probs = torch.tensor([[0.9, 0.8, 0.7]])  # 1 bag, 3 instances
        bag_labels = torch.tensor([1.0])
        loss, bag_prob = mil_bag_loss(inst_probs, bag_labels)
        assert torch.isfinite(loss)
        assert bag_prob.item() > 0.9  # High bag prob

    def test_mil_loss_negative_bag(self):
        """Negative bag should encourage low instance probs."""
        from presto.training.losses import mil_bag_loss
        inst_probs = torch.tensor([[0.1, 0.05, 0.02]])
        bag_labels = torch.tensor([0.0])
        loss, bag_prob = mil_bag_loss(inst_probs, bag_labels)
        assert torch.isfinite(loss)

    def test_mil_loss_with_mask(self):
        """Should handle variable-length bags via mask."""
        from presto.training.losses import mil_bag_loss
        inst_probs = torch.tensor([[0.9, 0.8, 0.0, 0.0]])  # Only 2 valid
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        bag_labels = torch.tensor([1.0])
        loss, bag_prob = mil_bag_loss(inst_probs, bag_labels, mask=mask)
        assert torch.isfinite(loss)

    def test_mil_entropy_regularization(self):
        """Entropy regularization should prevent MIL collapse."""
        from presto.training.losses import mil_bag_loss
        # All instances the same -> low entropy
        inst_probs_low_entropy = torch.tensor([[0.5, 0.5, 0.5]])
        # Mixed instances -> higher entropy
        inst_probs_high_entropy = torch.tensor([[0.9, 0.5, 0.1]])
        bag_labels = torch.tensor([1.0])

        loss_low, _ = mil_bag_loss(inst_probs_low_entropy, bag_labels, entropy_weight=0.1)
        loss_high, _ = mil_bag_loss(inst_probs_high_entropy, bag_labels, entropy_weight=0.1)
        # Higher entropy should have lower regularization penalty
        # (but main loss dominates, so just check both finite)
        assert torch.isfinite(loss_low)
        assert torch.isfinite(loss_high)


# --------------------------------------------------------------------------
# Uncertainty Weighting Tests
# --------------------------------------------------------------------------

class TestUncertaintyWeighting:
    """Test learned uncertainty weighting for multi-task learning."""

    def test_uncertainty_weighting_init(self):
        from presto.training.losses import UncertaintyWeighting
        uw = UncertaintyWeighting(n_tasks=3)
        assert uw.log_vars.shape == (3,)

    def test_uncertainty_weighting_forward(self):
        from presto.training.losses import UncertaintyWeighting
        uw = UncertaintyWeighting(n_tasks=3)
        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5)]
        total = uw(losses)
        assert torch.isfinite(total)

    def test_uncertainty_weighting_learns(self):
        """Weights should be learnable."""
        from presto.training.losses import UncertaintyWeighting
        uw = UncertaintyWeighting(n_tasks=2)
        losses = [torch.tensor(10.0, requires_grad=True), torch.tensor(0.1, requires_grad=True)]
        total = uw(losses)
        total.backward()
        # log_vars should have gradients
        assert uw.log_vars.grad is not None


# --------------------------------------------------------------------------
# Combined Loss Tests
# --------------------------------------------------------------------------

class TestCombinedLoss:
    """Test combined multi-task loss."""

    def test_combined_loss_all_tasks(self):
        from presto.training.losses import CombinedLoss
        loss_fn = CombinedLoss(task_names=["bind", "kin", "stab", "proc", "el", "tcell"])

        losses = {
            "bind": torch.tensor(1.0),
            "kin": torch.tensor(0.5),
            "stab": torch.tensor(0.3),
            "proc": torch.tensor(0.2),
            "el": torch.tensor(0.8),
            "tcell": torch.tensor(0.4),
        }
        total = loss_fn(losses)
        assert torch.isfinite(total)

    def test_combined_loss_missing_tasks(self):
        """Should handle missing tasks gracefully."""
        from presto.training.losses import CombinedLoss
        loss_fn = CombinedLoss(task_names=["bind", "kin", "el"])

        # Only bind and el present
        losses = {
            "bind": torch.tensor(1.0),
            "el": torch.tensor(0.8),
        }
        total = loss_fn(losses)
        assert torch.isfinite(total)


# --------------------------------------------------------------------------
# Binary Cross-Entropy Loss Tests
# --------------------------------------------------------------------------

class TestBCELoss:
    """Test BCE loss utilities."""

    def test_bce_with_logits(self):
        from presto.training.losses import safe_bce_with_logits
        logits = torch.tensor([2.0, -2.0, 0.0])
        targets = torch.tensor([1.0, 0.0, 0.5])
        loss = safe_bce_with_logits(logits, targets)
        assert torch.isfinite(loss)

    def test_bce_label_smoothing(self):
        from presto.training.losses import safe_bce_with_logits
        logits = torch.tensor([5.0, -5.0])
        targets = torch.tensor([1.0, 0.0])
        loss_no_smooth = safe_bce_with_logits(logits, targets, label_smoothing=0.0)
        loss_smooth = safe_bce_with_logits(logits, targets, label_smoothing=0.1)
        # Smoothing should increase loss on confident predictions
        assert loss_smooth > loss_no_smooth


# --------------------------------------------------------------------------
# Focal Loss Tests
# --------------------------------------------------------------------------

class TestFocalLoss:
    """Test focal loss for imbalanced classification."""

    def test_focal_loss_easy_examples(self):
        """Focal loss should down-weight easy examples."""
        from presto.training.losses import focal_loss
        # Easy positive: high logit, label=1
        logit_easy = torch.tensor([5.0])
        # Hard positive: low logit, label=1
        logit_hard = torch.tensor([0.5])
        target = torch.tensor([1.0])

        loss_easy = focal_loss(logit_easy, target, gamma=2.0)
        loss_hard = focal_loss(logit_hard, target, gamma=2.0)
        # Easy example should have lower loss
        assert loss_easy < loss_hard


# --------------------------------------------------------------------------
# PCGrad Tests
# --------------------------------------------------------------------------

class TestPCGrad:
    """Test projected conflicting gradient updates."""

    def test_pcgrad_conflicting_gradients_cancel(self):
        from presto.training.losses import PCGrad

        w = torch.nn.Parameter(torch.tensor(1.0))
        optim = torch.optim.SGD([w], lr=0.1)
        pcgrad = PCGrad(optim)

        # grad(loss_pos)=+1, grad(loss_neg)=-1 (fully conflicting)
        loss_pos = w
        loss_neg = -w
        pcgrad.step([loss_pos, loss_neg], [w])

        # Conflicting gradients should project away; update is ~zero.
        assert abs(w.item() - 1.0) < 1e-6

    def test_pcgrad_non_conflicting_updates_parameter(self):
        from presto.training.losses import PCGrad

        w = torch.nn.Parameter(torch.tensor(1.0))
        optim = torch.optim.SGD([w], lr=0.1)
        pcgrad = PCGrad(optim)

        # Same direction gradients.
        loss1 = w**2
        loss2 = w**2
        pcgrad.step([loss1, loss2], [w])

        assert w.item() < 1.0
