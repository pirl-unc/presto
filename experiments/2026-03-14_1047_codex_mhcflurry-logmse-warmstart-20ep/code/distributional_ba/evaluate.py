"""Evaluation helpers for the clean distributional BA experiment."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from presto.data.groove import prepare_mhc_input
from presto.data.tokenizer import Tokenizer
from presto.scripts.groove_baseline_probe import _find_allele_sequence

from .config import DistributionalModel
from .metrics import calibration_metrics, point_metrics


def evaluate_probe_panel(
    model: DistributionalModel,
    tokenizer: Tokenizer,
    allele_sequences: Mapping[str, str],
    peptides: Sequence[str],
    alleles: Sequence[str],
    device: str,
) -> List[Dict[str, Any]]:
    """Predict the probe panel for each peptide x allele pair."""

    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for peptide in peptides:
            pep = str(peptide or "").strip().upper()
            if not pep:
                continue
            pep_tok = torch.tensor(tokenizer.encode(pep, max_len=50)).unsqueeze(0).to(device)
            for allele in alleles:
                mhc_seq = _find_allele_sequence(allele_sequences, allele)
                if not mhc_seq:
                    continue
                prepared = prepare_mhc_input(mhc_a=mhc_seq, mhc_class="I")
                mhc_a_tok = torch.tensor(
                    tokenizer.encode(prepared.groove_half_1, max_len=120)
                ).unsqueeze(0).to(device)
                mhc_b_tok = torch.tensor(
                    tokenizer.encode(prepared.groove_half_2, max_len=120)
                ).unsqueeze(0).to(device)

                out = model(pep_tok, mhc_a_tok, mhc_b_tok)
                h = model.encoder(pep_tok, mhc_a_tok, mhc_b_tok)
                assay_emb = model._assay_embedding(batch_size=1, device=device)
                dist = model.head.predict_distribution(h, assay_emb)

                row: Dict[str, Any] = {
                    "peptide": pep,
                    "allele": str(allele),
                    "ic50_nM": float(out["pred_ic50_nM"][0].item()),
                    "ic50_log10": float(torch.log10(out["pred_ic50_nM"][0].clamp(min=1e-3)).item()),
                }
                if dist is not None:
                    if "entropy" in dist:
                        row["entropy"] = float(dist["entropy"][0].item())
                    if "sigma" in dist:
                        row["sigma"] = float(dist["sigma"][0].item())
                    if "iqr" in dist:
                        row["iqr"] = float(dist["iqr"][0].item())
                rows.append(row)
    model.train()
    return rows


def evaluate_held_out(
    model: DistributionalModel,
    loader: DataLoader,
    device: str,
    *,
    split_name: str,
    threshold_nM: float = 500.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one held-out split and return metrics plus per-example rows.

    Point metrics and calibration are computed on exact rows only (`qualifier == 0`).
    Overall censor-aware loss is still computed on all supervised rows.
    """

    model.eval()
    dataset = getattr(loader, "dataset", None)
    dataset_samples = list(getattr(dataset, "samples", []))

    all_pred_nM: List[torch.Tensor] = []
    all_true_nM: List[torch.Tensor] = []
    all_qual: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    all_exact_mask: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_edges: List[torch.Tensor] = []
    all_true_y: List[torch.Tensor] = []
    prediction_rows: List[Dict[str, Any]] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch_cpu = batch
            batch = batch.to(device)

            bind_target = getattr(batch, "bind_target", None)
            bind_mask = getattr(batch, "bind_mask", None)
            bind_qual = getattr(batch, "bind_qual", None)
            if bind_target is None or bind_mask is None:
                continue

            ic50_nM = bind_target.float().reshape(-1).to(device)
            mask = bind_mask.float().reshape(-1).to(device)
            qual = (
                bind_qual.long().reshape(-1).to(device)
                if bind_qual is not None
                else torch.zeros_like(mask, dtype=torch.long)
            )

            h = model.encoder(batch.pep_tok, batch.mhc_a_tok, batch.mhc_b_tok)
            binding_ctx = getattr(batch, "binding_context", {})
            assay_emb = model._assay_embedding(
                batch_size=h.shape[0],
                device=h.device,
                assay_type_idx=binding_ctx.get("assay_type_idx"),
                assay_prep_idx=binding_ctx.get("assay_prep_idx"),
                assay_geometry_idx=binding_ctx.get("assay_geometry_idx"),
                assay_readout_idx=binding_ctx.get("assay_readout_idx"),
            )

            out = model.head(h, assay_emb)
            dist = model.head.predict_distribution(h, assay_emb)
            loss, _ = model.head.compute_loss(h, assay_emb, ic50_nM, qual, mask)
            total_loss += float(loss.detach())
            n_batches += 1

            pred_nM = out["pred_ic50_nM"].detach().cpu().reshape(-1)
            true_nM = ic50_nM.detach().cpu().reshape(-1)
            qual_cpu = qual.detach().cpu().reshape(-1)
            mask_cpu = mask.detach().cpu().reshape(-1)
            exact_mask_cpu = (mask_cpu > 0.0) & (qual_cpu == 0)

            all_pred_nM.append(pred_nM)
            all_true_nM.append(true_nM)
            all_qual.append(qual_cpu)
            all_mask.append(mask_cpu)
            all_exact_mask.append(exact_mask_cpu)

            probs_cpu = None
            edges_cpu = None
            entropy_cpu = None
            sigma_cpu = None
            iqr_cpu = None
            if dist is not None:
                if "probs" in dist and "bin_edges" in dist:
                    probs_cpu = dist["probs"].detach().cpu()
                    edges_cpu = dist["bin_edges"].detach().cpu()
                    if edges_cpu.dim() == 1:
                        edges_cpu = edges_cpu.unsqueeze(0).expand(pred_nM.shape[0], -1)
                    if exact_mask_cpu.any():
                        all_probs.append(probs_cpu[exact_mask_cpu])
                        all_edges.append(edges_cpu[exact_mask_cpu])
                        all_true_y.append(torch.log1p(true_nM[exact_mask_cpu].clamp(min=0.0)))
                if "entropy" in dist:
                    entropy_cpu = dist["entropy"].detach().cpu().reshape(-1)
                if "sigma" in dist:
                    sigma_cpu = dist["sigma"].detach().cpu().reshape(-1)
                if "iqr" in dist:
                    iqr_cpu = dist["iqr"].detach().cpu().reshape(-1)

            dataset_indices = (
                batch_cpu.dataset_index.detach().cpu().reshape(-1).tolist()
                if getattr(batch_cpu, "dataset_index", None) is not None
                else [-1] * pred_nM.shape[0]
            )
            peptide_ids = (
                batch_cpu.peptide_id.detach().cpu().reshape(-1).tolist()
                if getattr(batch_cpu, "peptide_id", None) is not None
                else [-1] * pred_nM.shape[0]
            )
            sample_ids = list(getattr(batch_cpu, "sample_ids", []))
            primary_alleles = list(getattr(batch_cpu, "primary_alleles", []))

            for idx in range(pred_nM.shape[0]):
                dataset_index = int(dataset_indices[idx]) if idx < len(dataset_indices) else -1
                sample = dataset_samples[dataset_index] if 0 <= dataset_index < len(dataset_samples) else None
                row: Dict[str, Any] = {
                    "split": split_name,
                    "sample_id": sample_ids[idx] if idx < len(sample_ids) else "",
                    "dataset_index": dataset_index,
                    "peptide_id": int(peptide_ids[idx]) if idx < len(peptide_ids) else -1,
                    "peptide": str(getattr(sample, "peptide", "") or ""),
                    "allele": primary_alleles[idx] if idx < len(primary_alleles) else str(getattr(sample, "primary_allele", "") or ""),
                    "measurement_type": str(getattr(sample, "bind_measurement_type", "") or ""),
                    "qualifier": int(qual_cpu[idx].item()),
                    "is_exact": bool(exact_mask_cpu[idx].item()),
                    "masked": bool(mask_cpu[idx].item() > 0),
                    "true_ic50_nM": float(true_nM[idx].item()),
                    "true_ic50_log10": float(torch.log10(true_nM[idx].clamp(min=1e-3)).item()),
                    "pred_ic50_nM": float(pred_nM[idx].item()),
                    "pred_ic50_log10": float(torch.log10(pred_nM[idx].clamp(min=1e-3)).item()),
                }
                if entropy_cpu is not None:
                    row["entropy"] = float(entropy_cpu[idx].item())
                if sigma_cpu is not None:
                    row["sigma"] = float(sigma_cpu[idx].item())
                if iqr_cpu is not None:
                    row["iqr"] = float(iqr_cpu[idx].item())
                prediction_rows.append(row)

    model.train()

    if not all_pred_nM:
        return {"loss": 0.0, "split": split_name}, []

    pred_all = torch.cat(all_pred_nM)
    true_all = torch.cat(all_true_nM)
    qual_all = torch.cat(all_qual)
    mask_all = torch.cat(all_mask)
    exact_mask_all = torch.cat(all_exact_mask)

    metrics: Dict[str, Any] = {
        "split": split_name,
        "loss": total_loss / max(n_batches, 1),
        "n_supervised": int(mask_all.bool().sum().item()),
        "n_exact": int(exact_mask_all.bool().sum().item()),
        "n_censored": int(((mask_all > 0.0) & (qual_all != 0)).sum().item()),
    }
    metrics.update(point_metrics(pred_all, true_all, exact_mask_all.float(), threshold_nM=threshold_nM))

    if all_probs:
        probs = torch.cat(all_probs)
        edges = torch.cat(all_edges)
        true_y = torch.cat(all_true_y)
        metrics.update(
            calibration_metrics(
                probs=probs,
                edges=edges,
                true_y=true_y,
                mask=torch.ones(probs.shape[0]),
            )
        )

    return metrics, prediction_rows
