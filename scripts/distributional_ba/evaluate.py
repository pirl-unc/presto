"""Evaluation routines: probe panel and held-out set evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    """Evaluate probe panel — predict IC50 for each peptide×allele pair."""
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
                ic50_nM = float(out["pred_ic50_nM"][0].item())

                row = {
                    "peptide": pep,
                    "allele": str(allele),
                    "ic50_nM": ic50_nM,
                    "ic50_log10": float(torch.log10(torch.tensor(max(ic50_nM, 1e-3)))),
                }

                # Distributional extras
                h_probe = model.encode_input(pep_tok, mhc_a_tok, mhc_b_tok)
                assay_emb_probe = model._compute_assay_emb(
                    h_probe,
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.zeros(1, dtype=torch.long, device=device),
                )
                dist = model.head.predict_distribution(h_probe, assay_emb_probe)
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
    threshold_nM: float = 500.0,
) -> Dict[str, Any]:
    """Evaluate on held-out (val or test) data loader.

    Returns point metrics plus calibration metrics (for distributional heads).
    """
    model.eval()
    all_pred_nM: List[torch.Tensor] = []
    all_true_nM: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_edges: List[torch.Tensor] = []
    all_true_y: List[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pep_tok = batch.pep_tok
            mhc_a_tok = batch.mhc_a_tok
            mhc_b_tok = batch.mhc_b_tok

            bind_target = getattr(batch, "bind_target", None)
            bind_mask = getattr(batch, "bind_mask", None)
            bind_qual = getattr(batch, "bind_qual", None)
            if bind_target is None or bind_mask is None:
                continue

            ic50_nM = bind_target.float().reshape(-1).to(device)
            mask = bind_mask.float().reshape(-1).to(device)
            qual = (bind_qual.long().reshape(-1).to(device) if bind_qual is not None
                    else torch.zeros_like(mask, dtype=torch.long))

            h = model.encode_input(pep_tok, mhc_a_tok, mhc_b_tok)
            binding_ctx = getattr(batch, "binding_context", {})
            assay_emb = model._compute_assay_emb(
                h,
                binding_ctx.get("assay_type_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
                binding_ctx.get("assay_prep_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
                binding_ctx.get("assay_geometry_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
                binding_ctx.get("assay_readout_idx", torch.zeros(h.shape[0], dtype=torch.long, device=device)),
            )

            out = model.head(h, assay_emb)
            loss, _ = model.head.compute_loss(h, assay_emb, ic50_nM, qual, mask)
            total_loss += float(loss.detach())
            n_batches += 1

            all_pred_nM.append(out["pred_ic50_nM"].detach().cpu())
            all_true_nM.append(ic50_nM.detach().cpu())
            all_mask.append(mask.detach().cpu())

            # Calibration data (only for bin-based distributional heads)
            dist = model.head.predict_distribution(h, assay_emb)
            if dist is not None and "probs" in dist and "bin_edges" in dist:
                all_probs.append(dist["probs"].detach().cpu())
                edges = dist["bin_edges"]
                if edges.dim() == 1:
                    edges = edges.unsqueeze(0).expand(h.shape[0], -1)
                all_edges.append(edges.detach().cpu())
                all_true_y.append(torch.log(1.0 + ic50_nM.clamp(min=0)).detach().cpu())

    model.train()

    if not all_pred_nM:
        return {"loss": 0.0}

    pred_nM = torch.cat(all_pred_nM)
    true_nM = torch.cat(all_true_nM)
    mask_all = torch.cat(all_mask)

    result: Dict[str, Any] = {"loss": total_loss / max(n_batches, 1)}
    result.update(point_metrics(pred_nM, true_nM, mask_all, threshold_nM=threshold_nM))

    if all_probs:
        probs = torch.cat(all_probs)
        edges = torch.cat(all_edges)
        true_y = torch.cat(all_true_y)
        result.update(calibration_metrics(probs, edges, true_y, mask_all))

    return result
