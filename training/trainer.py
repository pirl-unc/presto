"""Trainer utilities for Presto model.

Canonical production training is unified mixed-source (`scripts/train_iedb.py`).
"""

from pathlib import Path
from typing import Dict, Any

import torch

from ..models.presto import Presto
from ..models.affinity import (
    DEFAULT_MAX_AFFINITY_NM,
    normalize_binding_target_log10,
)
from .checkpointing import save_model_checkpoint
from .losses import PCGrad, censor_aware_loss, CombinedLoss, safe_bce_with_logits


class Trainer:
    """Simple trainer for Presto model.

    Handles multi-task training with binding, T-cell, and elution losses.
    """

    def __init__(
        self,
        model: Presto,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        use_pcgrad: bool = False,
        device: str = None,
    ):
        """Initialize trainer.

        Args:
            model: Presto model
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            use_pcgrad: Use projected conflicting gradients for multi-task updates
            device: Device to train on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_pcgrad = use_pcgrad

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.pcgrad = PCGrad(self.optimizer) if use_pcgrad else None

        self.loss_fn = CombinedLoss(
            task_names=[
                "bind",
                "tcell",
                "elution",
                "processing",
                "kon",
                "koff",
                "t_half",
                "tm",
                "contrastive",
            ]
        )
        self.step_count = 0

    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Single training step.

        Args:
            batch: Dict with:
                - pep_tok, mhc_a_tok, mhc_b_tok, mhc_class (required)
                - bind_target, bind_qual (optional, for binding task)
                - tcr_a_tok, tcr_b_tok, tcell_label (optional, for T-cell task)
                - elution_label (optional, for elution task)

        Returns:
            Total loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move tensors to device
        pep_tok = batch["pep_tok"].to(self.device)
        mhc_a_tok = batch["mhc_a_tok"].to(self.device)
        mhc_b_tok = batch["mhc_b_tok"].to(self.device)
        mhc_class = batch["mhc_class"]

        # Optional TCR
        tcr_a_tok = batch.get("tcr_a_tok")
        tcr_b_tok = batch.get("tcr_b_tok")
        flank_n_tok = batch.get("flank_n_tok")
        flank_c_tok = batch.get("flank_c_tok")
        tcell_context = batch.get("tcell_context")
        if tcr_a_tok is not None:
            tcr_a_tok = tcr_a_tok.to(self.device)
        if tcr_b_tok is not None:
            tcr_b_tok = tcr_b_tok.to(self.device)
        if flank_n_tok is not None:
            flank_n_tok = flank_n_tok.to(self.device)
        if flank_c_tok is not None:
            flank_c_tok = flank_c_tok.to(self.device)
        if isinstance(tcell_context, dict):
            tcell_context = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in tcell_context.items()
            }

        # Forward pass
        outputs = self.model(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            mhc_class=mhc_class,
            tcr_a_tok=tcr_a_tok,
            tcr_b_tok=tcr_b_tok,
            flank_n_tok=flank_n_tok,
            flank_c_tok=flank_c_tok,
            tcell_context=tcell_context,
        )

        losses = {}

        targets = batch.get("targets")
        target_masks = batch.get("target_masks")
        target_quals = batch.get("target_quals")

        def _target(task_name: str, fallback_key: str):
            if isinstance(targets, dict) and task_name in targets:
                return targets[task_name].to(self.device)
            value = batch.get(fallback_key)
            return value.to(self.device) if value is not None else None

        def _mask(task_name: str, fallback_key: str):
            if isinstance(target_masks, dict) and task_name in target_masks:
                return target_masks[task_name].to(self.device)
            value = batch.get(fallback_key)
            return value.to(self.device) if value is not None else None

        def _qual(task_name: str, fallback_key: str):
            if isinstance(target_quals, dict) and task_name in target_quals:
                return target_quals[task_name].to(self.device)
            value = batch.get(fallback_key)
            return value.to(self.device) if value is not None else None

        # Binding loss
        bind_target = _target("binding", "bind_target")
        bind_qual = _qual("binding", "bind_qual")
        if bind_target is not None and bind_qual is not None:
            # Use KD prediction from assays
            bind_pred = outputs["assays"]["KD_nM"]
            bind_target_log10 = normalize_binding_target_log10(
                bind_target.squeeze(-1),
                max_affinity_nM=DEFAULT_MAX_AFFINITY_NM,
            )
            bind_loss = censor_aware_loss(
                bind_pred.squeeze(-1),
                bind_target_log10,
                bind_qual.squeeze(-1),
            )
            bind_mask = _mask("binding", "bind_mask")
            if bind_mask is not None:
                bind_loss_vec = censor_aware_loss(
                    bind_pred.squeeze(-1),
                    bind_target_log10,
                    bind_qual.squeeze(-1),
                    reduction="none",
                )
                if bind_mask.sum() > 0:
                    bind_loss = (bind_loss_vec * bind_mask).sum() / (bind_mask.sum() + 1e-8)
            losses["bind"] = bind_loss

        # T-cell loss
        tcell_label = _target("tcell", "tcell_label")
        if tcell_label is not None and ("tcell_logit" in outputs or "recognition_repertoire_logit" in outputs):
            tcell_logit = (
                outputs["tcell_logit"].squeeze(-1)
                if "tcell_logit" in outputs
                else outputs["recognition_repertoire_logit"].squeeze(-1)
            )
            tcell_mask = _mask("tcell", "tcell_mask")
            if tcell_mask is not None:
                tcell_loss = safe_bce_with_logits(
                    tcell_logit, tcell_label, reduction="none"
                )
                if tcell_mask.sum() > 0:
                    losses["tcell"] = (tcell_loss * tcell_mask).sum() / (tcell_mask.sum() + 1e-8)
            else:
                losses["tcell"] = safe_bce_with_logits(tcell_logit, tcell_label)

        # Elution loss
        elution_label = _target("elution", "elution_label")
        if elution_label is not None:
            elution_logit = outputs["elution_logit"].squeeze(-1)
            elution_mask = _mask("elution", "elution_mask")
            if elution_mask is not None:
                elution_loss = safe_bce_with_logits(
                    elution_logit, elution_label, reduction="none"
                )
                if elution_mask.sum() > 0:
                    losses["elution"] = (elution_loss * elution_mask).sum() / (elution_mask.sum() + 1e-8)
            else:
                losses["elution"] = safe_bce_with_logits(elution_logit, elution_label)

        # Processing loss
        processing_label = _target("processing", "processing_label")
        if processing_label is not None:
            processing_logit = outputs["processing_logit"].squeeze(-1)
            processing_mask = _mask("processing", "processing_mask")
            if processing_mask is not None:
                processing_loss = safe_bce_with_logits(
                    processing_logit, processing_label, reduction="none"
                )
                if processing_mask.sum() > 0:
                    losses["processing"] = (
                        processing_loss * processing_mask
                    ).sum() / (processing_mask.sum() + 1e-8)
            else:
                losses["processing"] = safe_bce_with_logits(
                    processing_logit, processing_label
                )

        # Kinetics/stability losses from assay heads.
        if "assays" in outputs:
            assay_map = {
                "kon": ("kon", "kon_target", "kon_mask"),
                "koff": ("koff", "koff_target", "koff_mask"),
                "t_half": ("t_half", "t_half_target", "t_half_mask"),
                "tm": ("tm", "tm_target", "tm_mask"),
            }
            for loss_name, (task_name, target_key, mask_key) in assay_map.items():
                assay_key = "Tm" if loss_name == "tm" else loss_name
                target = _target(task_name, target_key)
                if target is None or assay_key not in outputs["assays"]:
                    continue
                pred = outputs["assays"][assay_key].squeeze(-1)
                mse = (pred - target.squeeze(-1)) ** 2
                mask = _mask(task_name, mask_key)
                if mask is not None:
                    if mask.sum() <= 0:
                        continue
                    losses[loss_name] = (mse * mask).sum() / (mask.sum() + 1e-8)
                else:
                    losses[loss_name] = mse.mean()

        # Combine losses / update
        if losses:
            if self.use_pcgrad and len(losses) > 1:
                # PCGrad handles backward + optimizer step internally.
                mean_loss = self.pcgrad.step(list(losses.values()), self.model.parameters())
                total_loss = torch.tensor(
                    0.0 if mean_loss is None else mean_loss,
                    device=self.device,
                )
            else:
                total_loss = self.loss_fn(losses)
                total_loss.backward()
                self.optimizer.step()
        else:
            # Fallback: use presentation as regularization
            total_loss = outputs["presentation_logit"].mean() * 0.0 + 0.0

        self.step_count += 1

        return total_loss.detach()

    def eval_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Evaluation step (no gradient computation).

        Args:
            batch: Input batch

        Returns:
            Model outputs
        """
        self.model.eval()

        pep_tok = batch["pep_tok"].to(self.device)
        mhc_a_tok = batch["mhc_a_tok"].to(self.device)
        mhc_b_tok = batch["mhc_b_tok"].to(self.device)
        mhc_class = batch["mhc_class"]

        tcr_a_tok = batch.get("tcr_a_tok")
        tcr_b_tok = batch.get("tcr_b_tok")
        flank_n_tok = batch.get("flank_n_tok")
        flank_c_tok = batch.get("flank_c_tok")
        tcell_context = batch.get("tcell_context")
        if tcr_a_tok is not None:
            tcr_a_tok = tcr_a_tok.to(self.device)
        if tcr_b_tok is not None:
            tcr_b_tok = tcr_b_tok.to(self.device)
        if flank_n_tok is not None:
            flank_n_tok = flank_n_tok.to(self.device)
        if flank_c_tok is not None:
            flank_c_tok = flank_c_tok.to(self.device)
        if isinstance(tcell_context, dict):
            tcell_context = {
                key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in tcell_context.items()
            }

        with torch.no_grad():
            outputs = self.model(
                pep_tok=pep_tok,
                mhc_a_tok=mhc_a_tok,
                mhc_b_tok=mhc_b_tok,
                mhc_class=mhc_class,
                tcr_a_tok=tcr_a_tok,
                tcr_b_tok=tcr_b_tok,
                flank_n_tok=flank_n_tok,
                flank_c_tok=flank_c_tok,
                tcell_context=tcell_context,
            )

        # Move outputs to CPU
        result = {}
        for k, v in outputs.items():
            result[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        return result

    def save_checkpoint(self, path: Path) -> None:
        """Save checkpoint.

        Args:
            path: Path to save checkpoint
        """
        save_model_checkpoint(
            path,
            model=self.model,
            optimizer_state_dict=self.optimizer.state_dict(),
            step=self.step_count,
            extra={"loss_fn_state_dict": self.loss_fn.state_dict()},
        )

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step_count", checkpoint.get("step", 0))
        loss_fn_state = checkpoint.get("loss_fn_state_dict")
        if loss_fn_state is None:
            loss_fn_state = checkpoint.get("extra", {}).get("loss_fn_state_dict")
        if loss_fn_state is not None:
            self.loss_fn.load_state_dict(loss_fn_state)
