"""Historical optimizer wrapper; not part of manifest training."""

from typing import Iterable, List, Optional

import torch


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
