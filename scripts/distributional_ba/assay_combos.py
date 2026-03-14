"""Enumerate observed assay condition combinations and generate predictions.

After training, this module produces predictions for ALL observed assay combos
so the model can be used without assay metadata at inference.  For distributional
heads the per-combo predicted distributions are also returned and can be
mixture-averaged into a single marginal distribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from presto.data.vocab import (
    BINDING_ASSAY_TYPES,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_GEOMETRY,
    BINDING_ASSAY_READOUT,
)

from .config import DistributionalModel


def enumerate_observed_combos(
    records: list,
) -> List[Dict[str, Any]]:
    """Extract all unique (type, prep, geometry, readout) tuples from data.

    Returns list of dicts with index values for each combo, sorted by
    frequency (most common first).
    """
    from collections import Counter

    combo_counts: Counter = Counter()
    for r in records:
        key = (
            getattr(r, "assay_type_idx", 0),
            getattr(r, "assay_prep_idx", 0),
            getattr(r, "assay_geometry_idx", 0),
            getattr(r, "assay_readout_idx", 0),
        )
        combo_counts[key] += 1

    combos = []
    for (t, p, g, rd), count in combo_counts.most_common():
        combos.append({
            "assay_type_idx": int(t),
            "assay_prep_idx": int(p),
            "assay_geometry_idx": int(g),
            "assay_readout_idx": int(rd),
            "assay_type": BINDING_ASSAY_TYPES[int(t)] if int(t) < len(BINDING_ASSAY_TYPES) else f"idx_{t}",
            "assay_prep": BINDING_ASSAY_PREP[int(p)] if int(p) < len(BINDING_ASSAY_PREP) else f"idx_{p}",
            "assay_geometry": BINDING_ASSAY_GEOMETRY[int(g)] if int(g) < len(BINDING_ASSAY_GEOMETRY) else f"idx_{g}",
            "assay_readout": BINDING_ASSAY_READOUT[int(rd)] if int(rd) < len(BINDING_ASSAY_READOUT) else f"idx_{rd}",
            "count": count,
        })
    return combos


def _assay_tensors(combo: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    """Build single-sample assay index tensors from a combo dict."""
    return {
        "assay_type_idx": torch.tensor([combo["assay_type_idx"]], device=device),
        "assay_prep_idx": torch.tensor([combo["assay_prep_idx"]], device=device),
        "assay_geometry_idx": torch.tensor([combo["assay_geometry_idx"]], device=device),
        "assay_readout_idx": torch.tensor([combo["assay_readout_idx"]], device=device),
    }


def predict_all_combos(
    model: DistributionalModel,
    pep_tok: torch.Tensor,
    mhc_a_tok: torch.Tensor,
    mhc_b_tok: torch.Tensor,
    combos: List[Dict[str, Any]],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run predictions for one input across all assay combos.

    Returns one dict per combo with:
      - ic50_nM, ic50_log10  (always)
      - pred_bounded          (MHCflurry heads)
      - probs, bin_edges, bin_centers, entropy  (distributional heads)
      - mu, sigma             (Gaussian head)
      - quantiles, iqr        (Quantile head)
    """
    model.eval()
    results = []

    with torch.no_grad():
        # Encode once — shared across combos
        pep = pep_tok.to(device)
        mhc_a = mhc_a_tok.to(device)
        mhc_b = mhc_b_tok.to(device)
        h = model.encoder(pep, mhc_a, mhc_b)

        for combo in combos:
            atensors = _assay_tensors(combo, device)
            assay_emb = model._compute_assay_emb(
                h,
                atensors["assay_type_idx"],
                atensors["assay_prep_idx"],
                atensors["assay_geometry_idx"],
                atensors["assay_readout_idx"],
            )

            # Point prediction
            out = model.head(h, assay_emb)
            ic50 = float(out["pred_ic50_nM"][0].item())
            row: Dict[str, Any] = {
                "ic50_nM": ic50,
                "ic50_log10": float(torch.log10(torch.tensor(max(ic50, 1e-3)))),
                **{k: combo[k] for k in (
                    "assay_type", "assay_prep", "assay_geometry", "assay_readout", "count",
                )},
            }
            if "pred_bounded" in out:
                row["pred_bounded"] = float(out["pred_bounded"][0].item())

            # Distributional prediction
            dist = model.head.predict_distribution(h, assay_emb)
            if dist is not None:
                if "probs" in dist:
                    row["probs"] = dist["probs"][0].cpu()           # (K,)
                if "bin_edges" in dist:
                    edges = dist["bin_edges"]
                    row["bin_edges"] = (edges[0] if edges.dim() > 1 else edges).cpu()
                if "bin_centers" in dist:
                    centers = dist["bin_centers"]
                    row["bin_centers"] = (centers[0] if centers.dim() > 1 else centers).cpu()
                if "entropy" in dist:
                    row["entropy"] = float(dist["entropy"][0].item())
                if "mu" in dist:
                    row["mu"] = float(dist["mu"][0].item())
                if "sigma" in dist:
                    row["sigma"] = float(dist["sigma"][0].item())
                if "quantiles" in dist:
                    row["quantiles"] = dist["quantiles"][0].cpu()    # (5,)
                if "iqr" in dist:
                    row["iqr"] = float(dist["iqr"][0].item())

            results.append(row)

    model.train()
    return results


def predict_marginal(
    model: DistributionalModel,
    pep_tok: torch.Tensor,
    mhc_a_tok: torch.Tensor,
    mhc_b_tok: torch.Tensor,
    combos: List[Dict[str, Any]],
    device: str = "cpu",
    weighting: str = "uniform",
) -> Dict[str, Any]:
    """Produce a single prediction by marginalizing over assay combos.

    For point predictions: weighted geometric mean in log10 space.
    For distributional heads: mixture of per-combo distributions weighted
    the same way, giving a proper marginal p(y | peptide, MHC).

    Args:
        weighting: "uniform" weights all combos equally; "frequency" weights
            by training set frequency.

    Returns:
        Dict with marginal ic50_nM, ic50_log10, assay_spread_log10,
        and optionally marginal_probs / marginal_entropy for distributional heads.
    """
    per_combo = predict_all_combos(model, pep_tok, mhc_a_tok, mhc_b_tok, combos, device)
    if not per_combo:
        return {"ic50_nM": float("nan"), "ic50_log10": float("nan")}

    n = len(per_combo)
    if weighting == "frequency":
        total = sum(c["count"] for c in per_combo)
        weights = [c["count"] / total for c in per_combo]
    else:
        weights = [1.0 / n] * n

    # --- Point prediction: weighted geometric mean ---
    log10_vals = [c["ic50_log10"] for c in per_combo]
    marginal_log10 = sum(w * v for w, v in zip(weights, log10_vals))
    marginal_nM = 10.0 ** marginal_log10
    spread = max(log10_vals) - min(log10_vals) if n > 1 else 0.0

    result: Dict[str, Any] = {
        "ic50_nM": marginal_nM,
        "ic50_log10": marginal_log10,
        "assay_spread_log10": spread,
        "n_combos": n,
        "per_combo": per_combo,
    }

    # --- Distributional marginal: mixture of per-combo distributions ---
    if "probs" in per_combo[0]:
        prob_stack = torch.stack([c["probs"] for c in per_combo])  # (C, K)
        w_tensor = torch.tensor(weights, dtype=prob_stack.dtype).unsqueeze(-1)  # (C, 1)
        marginal_probs = (w_tensor * prob_stack).sum(dim=0)  # (K,)
        # Re-normalize (should already sum to ~1, but be safe)
        marginal_probs = marginal_probs / marginal_probs.sum().clamp(min=1e-8)
        result["marginal_probs"] = marginal_probs
        result["marginal_entropy"] = float(
            -(marginal_probs * torch.log(marginal_probs.clamp(min=1e-10))).sum()
        )
        # Expected value from marginal distribution
        if "bin_centers" in per_combo[0]:
            centers = per_combo[0]["bin_centers"]  # (K,) — same for all if d2_logit
            ev_log1p = (marginal_probs * centers).sum()
            ev_nM = float((torch.exp(ev_log1p) - 1.0).clamp(min=0.0))
            result["marginal_ev_nM"] = ev_nM
            result["marginal_ev_log10"] = float(torch.log10(torch.tensor(max(ev_nM, 1e-3))))

    # --- Gaussian marginal: mixture of Gaussians ---
    if "mu" in per_combo[0] and "sigma" in per_combo[0]:
        mus = torch.tensor([c["mu"] for c in per_combo])
        sigmas = torch.tensor([c["sigma"] for c in per_combo])
        w_t = torch.tensor(weights)
        # Mixture mean
        mix_mu = (w_t * mus).sum()
        # Mixture variance = E[sigma^2] + E[mu^2] - E[mu]^2
        mix_var = (w_t * (sigmas ** 2 + mus ** 2)).sum() - mix_mu ** 2
        result["marginal_mu"] = float(mix_mu)
        result["marginal_sigma"] = float(mix_var.clamp(min=1e-8).sqrt())

    # --- Quantile marginal: weighted average of quantiles ---
    if "quantiles" in per_combo[0]:
        q_stack = torch.stack([c["quantiles"] for c in per_combo])  # (C, 5)
        w_t = torch.tensor(weights, dtype=q_stack.dtype).unsqueeze(-1)
        result["marginal_quantiles"] = (w_t * q_stack).sum(dim=0)  # (5,)

    return result
