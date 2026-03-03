"""Loss functions for Presto training.

Includes:
- Censor-aware loss for binding data (handles <, =, > qualifiers)
- MIL bag loss with noisy-OR aggregation for elution data
- Uncertainty weighting for multi-task learning
- Focal loss for imbalanced classification
"""

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.pmhc import stable_noisy_or


def censor_aware_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    qual: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Censor-aware regression loss for binding data.

    Handles censored measurements:
    - qual = 0: exact value (standard MSE)
    - qual = -1: less than (<), only penalize if pred > target
    - qual = 1: greater than (>), only penalize if pred < target

    Args:
        pred: Predicted values
        target: Target values
        qual: Qualifier codes (-1, 0, 1)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    # Standard MSE for exact values
    mse = (pred - target) ** 2

    # For < qualifiers: only penalize if pred > target
    # Loss = max(0, pred - target)^2
    less_than_loss = F.relu(pred - target) ** 2

    # For > qualifiers: only penalize if pred < target
    # Loss = max(0, target - pred)^2
    greater_than_loss = F.relu(target - pred) ** 2

    # Select loss based on qualifier
    is_exact = qual == 0
    is_less = qual == -1
    is_greater = qual == 1

    loss = (
        is_exact.float() * mse
        + is_less.float() * less_than_loss
        + is_greater.float() * greater_than_loss
    )

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def mil_bag_loss(
    inst_probs: torch.Tensor,
    bag_labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    entropy_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MIL bag loss using noisy-OR aggregation.

    Args:
        inst_probs: Instance probabilities (batch, n_instances)
        bag_labels: Bag labels (batch,)
        mask: Optional mask for variable-length bags (batch, n_instances)
        entropy_weight: Weight for entropy regularization (prevents MIL collapse)

    Returns:
        loss: Bag-level BCE loss
        bag_probs: Aggregated bag probabilities
    """
    # Aggregate instances to bag probability via noisy-OR
    if mask is not None:
        bag_probs = stable_noisy_or(inst_probs, mask=mask, dim=-1)
    else:
        bag_probs = stable_noisy_or(inst_probs, dim=-1)

    # BCE loss on bag probability
    eps = 1e-7
    bag_probs_clamped = torch.clamp(bag_probs, eps, 1 - eps)
    bce = -bag_labels * torch.log(bag_probs_clamped) - (1 - bag_labels) * torch.log(
        1 - bag_probs_clamped
    )
    loss = bce.mean()

    # Entropy regularization to prevent MIL collapse
    if entropy_weight > 0:
        # Binary entropy of instance probs
        inst_probs_clamped = torch.clamp(inst_probs, eps, 1 - eps)
        if mask is not None:
            entropy = -(
                inst_probs_clamped * torch.log(inst_probs_clamped)
                + (1 - inst_probs_clamped) * torch.log(1 - inst_probs_clamped)
            ) * mask
            entropy = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        else:
            entropy = -(
                inst_probs_clamped * torch.log(inst_probs_clamped)
                + (1 - inst_probs_clamped) * torch.log(1 - inst_probs_clamped)
            ).mean(dim=-1)
        # Maximize entropy (negative sign because we minimize)
        loss = loss - entropy_weight * entropy.mean()

    return loss, bag_probs


class UncertaintyWeighting(nn.Module):
    """Learned uncertainty weighting for multi-task learning.

    Based on Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics".

    Each task has a learned log-variance parameter that scales its loss.
    """

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses.

        Args:
            losses: List of task losses

        Returns:
            Weighted total loss
        """
        total = 0.0
        for i, loss in enumerate(losses):
            # Weight = 1 / (2 * variance) = 1 / (2 * exp(log_var))
            # Plus log(variance) = log_var as regularization
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
        return total


class CombinedLoss(nn.Module):
    """Combined multi-task loss with uncertainty weighting."""

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        self.uw = UncertaintyWeighting(n_tasks=len(task_names))

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined loss.

        Args:
            losses: Dict mapping task name to loss value

        Returns:
            Weighted total loss
        """
        loss_list = []
        for name in self.task_names:
            if name in losses:
                loss_list.append(losses[name])
            else:
                # Missing task: use zero loss
                loss_list.append(torch.tensor(0.0, device=next(iter(losses.values())).device))
        return self.uw(loss_list)


class PCGrad:
    """Projected Conflicting Gradient optimizer wrapper.

    Implements the PCGrad idea: project away gradient components that
    conflict (negative cosine / dot product) across task losses.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer

    def step(
        self,
        losses: List[torch.Tensor],
        parameters: Iterable[torch.nn.Parameter],
    ) -> Optional[float]:
        """Apply one PCGrad update.

        Args:
            losses: Task losses to combine.
            parameters: Model parameters to update.

        Returns:
            Optional scalar average loss value for logging.
        """
        valid_losses = [loss for loss in losses if loss is not None]
        if not valid_losses:
            return None

        params = [p for p in parameters if p.requires_grad]
        if not params:
            return None

        # Collect per-task gradients.
        task_grads: List[List[torch.Tensor]] = []
        for idx, loss in enumerate(valid_losses):
            self._optimizer.zero_grad(set_to_none=True)
            retain_graph = idx < len(valid_losses) - 1
            loss.backward(retain_graph=retain_graph)
            grads = [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in params
            ]
            task_grads.append(grads)

        # Project conflicting components.
        projected: List[List[torch.Tensor]] = []
        eps = 1e-12
        for i, gi in enumerate(task_grads):
            gi_proj = [g.clone() for g in gi]
            for j, gj in enumerate(task_grads):
                if i == j:
                    continue
                dot = sum(torch.sum(gp * gq) for gp, gq in zip(gi_proj, gj))
                if dot < 0:
                    norm2 = sum(torch.sum(gq * gq) for gq in gj) + eps
                    coeff = dot / norm2
                    gi_proj = [gp - coeff * gq for gp, gq in zip(gi_proj, gj)]
            projected.append(gi_proj)

        # Average projected gradients and step.
        self._optimizer.zero_grad(set_to_none=True)
        for p_idx, param in enumerate(params):
            merged = sum(g[p_idx] for g in projected) / len(projected)
            param.grad = merged
        self._optimizer.step()

        return float(torch.stack([loss.detach() for loss in valid_losses]).mean().item())


def safe_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE with logits with optional label smoothing.

    Args:
        logits: Predicted logits
        targets: Target labels (0 or 1)
        label_smoothing: Smoothing factor (0 = no smoothing)
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for imbalanced classification.

    Down-weights easy examples to focus on hard ones.

    Args:
        logits: Predicted logits
        targets: Target labels (0 or 1)
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing parameter for positive class
        reduction: "mean", "sum", or "none"

    Returns:
        Loss value
    """
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Focal weight: (1 - p_t)^gamma
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * ce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
