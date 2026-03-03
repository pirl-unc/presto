"""Evaluation CLI commands."""

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from ..data import PrestoCollator, PrestoDataset, create_dataloader
from ..scripts import train_synthetic
from ..training.checkpointing import load_model_from_checkpoint


def _binary_auc(labels: Iterable[float], scores: Iterable[float]) -> Optional[float]:
    ys = list(labels)
    ss = list(scores)
    n = len(ys)
    if n == 0:
        return None
    n_pos = sum(1 for y in ys if y >= 0.5)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: ss[i])
    ranks = [0] * n
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    sum_ranks_pos = sum(ranks[i] for i, y in enumerate(ys) if y >= 0.5)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _compute_recall_at_k(
    tcr_vec_list: List[torch.Tensor],
    pmhc_vec_list: List[torch.Tensor],
    k_values: List[int],
) -> Dict[str, Optional[float]]:
    if not tcr_vec_list:
        return {f"recall_at_{k}": None for k in k_values}

    tcr_vec = torch.stack(tcr_vec_list, dim=0)
    pmhc_vec = torch.stack(pmhc_vec_list, dim=0)
    tcr_vec = F.normalize(tcr_vec, dim=-1)
    pmhc_vec = F.normalize(pmhc_vec, dim=-1)

    sim = tcr_vec @ pmhc_vec.T
    n = sim.size(0)
    results: Dict[str, Optional[float]] = {}
    for k in k_values:
        k_eff = min(k, n)
        topk = torch.topk(sim, k=k_eff, dim=1).indices
        correct = 0
        for i in range(n):
            if (topk[i] == i).any():
                correct += 1
        results[f"recall_at_{k}"] = float(correct / n)
    return results


def _collect_aux_metrics(model, val_loader, device: str) -> Dict[str, Any]:
    model.eval()
    tcell_labels: List[float] = []
    tcell_scores: List[float] = []
    elution_labels: List[float] = []
    elution_scores: List[float] = []
    tcr_vec_pos: List[torch.Tensor] = []
    pmhc_vec_pos: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(
                pep_tok=batch.pep_tok,
                mhc_a_tok=batch.mhc_a_tok,
                mhc_b_tok=batch.mhc_b_tok,
                mhc_class=batch.mhc_class[0] if batch.mhc_class else "I",
                tcr_a_tok=batch.tcr_a_tok,
                tcr_b_tok=batch.tcr_b_tok,
                flank_n_tok=batch.flank_n_tok,
                flank_c_tok=batch.flank_c_tok,
            )

            if batch.tcell_label is not None and "tcell_logit" in outputs:
                scores = torch.sigmoid(outputs["tcell_logit"].squeeze(-1)).detach().cpu()
                labels = batch.tcell_label.detach().cpu()
                if batch.tcell_mask is not None:
                    mask = batch.tcell_mask.detach().cpu() > 0.5
                else:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                for y, s, keep in zip(labels.tolist(), scores.tolist(), mask.tolist()):
                    if keep:
                        tcell_labels.append(float(y))
                        tcell_scores.append(float(s))

            if batch.elution_label is not None and "elution_logit" in outputs:
                scores = torch.sigmoid(outputs["elution_logit"].squeeze(-1)).detach().cpu()
                labels = batch.elution_label.detach().cpu()
                if batch.elution_mask is not None:
                    mask = batch.elution_mask.detach().cpu() > 0.5
                else:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                for y, s, keep in zip(labels.tolist(), scores.tolist(), mask.tolist()):
                    if keep:
                        elution_labels.append(float(y))
                        elution_scores.append(float(s))

            # Retrieval pool: positive labeled T-cell rows with non-empty TCR tokens.
            if "tcr_vec" in outputs and batch.tcell_label is not None and batch.tcell_mask is not None:
                tcell_mask = batch.tcell_mask.detach().cpu() > 0.5
                tcell_pos = batch.tcell_label.detach().cpu() > 0.5
                if batch.tcr_a_tok is not None:
                    has_tcr_a = (batch.tcr_a_tok.detach().cpu() != 0).any(dim=1)
                else:
                    has_tcr_a = torch.zeros_like(tcell_pos, dtype=torch.bool)
                if batch.tcr_b_tok is not None:
                    has_tcr_b = (batch.tcr_b_tok.detach().cpu() != 0).any(dim=1)
                else:
                    has_tcr_b = torch.zeros_like(tcell_pos, dtype=torch.bool)
                has_tcr = has_tcr_a | has_tcr_b

                keep = tcell_mask & tcell_pos & has_tcr
                tcr_vec_batch = outputs["tcr_vec"].detach().cpu()
                pmhc_vec_batch = outputs["pmhc_vec"].detach().cpu()
                for idx, use in enumerate(keep.tolist()):
                    if use:
                        tcr_vec_pos.append(tcr_vec_batch[idx])
                        pmhc_vec_pos.append(pmhc_vec_batch[idx])

    metrics: Dict[str, Any] = {
        "tcell_auroc": _binary_auc(tcell_labels, tcell_scores),
        "elution_auroc": _binary_auc(elution_labels, elution_scores),
        "retrieval_n": len(tcr_vec_pos),
    }
    metrics.update(_compute_recall_at_k(tcr_vec_pos, pmhc_vec_pos, [1, 5, 10]))
    return metrics


def cmd_evaluate_synthetic(args: Any) -> int:
    """Evaluate a checkpoint on synthetic data."""
    if not args.checkpoint:
        raise ValueError("Missing --checkpoint")

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(tempfile.mkdtemp()) / "presto_eval_data"

    if args.json:
        # Keep JSON output machine-readable.
        with contextlib.redirect_stdout(io.StringIO()):
            binding_data, elution_data, tcr_data, mhc_sequences = train_synthetic.create_synthetic_data(
                data_dir, args.n_binding, args.n_elution, args.n_tcr
            )
    else:
        binding_data, elution_data, tcr_data, mhc_sequences = train_synthetic.create_synthetic_data(
            data_dir, args.n_binding, args.n_elution, args.n_tcr
        )

    dataset = PrestoDataset(
        binding_records=binding_data,
        elution_records=elution_data,
        tcr_records=tcr_data,
        mhc_sequences=mhc_sequences,
    )

    val_size = max(int(0.2 * len(dataset)), 1)
    train_size = len(dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    collator = PrestoCollator()
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, collator=collator)

    model, _ = load_model_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        strict=True,
    )
    model.to(device)

    val_loss = train_synthetic.evaluate(model, val_loader, device)
    aux_metrics = _collect_aux_metrics(model, val_loader, device)

    payload = {"val_loss": val_loss}
    payload.update(aux_metrics)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"val_loss: {val_loss:.4f}")
        for key in ("tcell_auroc", "elution_auroc", "retrieval_n", "recall_at_1", "recall_at_5", "recall_at_10"):
            print(f"{key}: {payload.get(key)}")

    return 0
