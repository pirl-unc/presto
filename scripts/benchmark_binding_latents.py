#!/usr/bin/env python
"""Benchmark binding latent architecture variants.

Generates structured synthetic data where KD correlates with peptide-MHC
compatibility, then trains 8 model configurations to compare learning
capacity for binding signal.

Usage:
    python -m presto.scripts.benchmark_binding_latents
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from presto.models.presto import Presto
from presto.models.affinity import normalize_binding_target_log10
from presto.data.vocab import AA_VOCAB, AA_TO_IDX

# ---------------------------------------------------------------------------
# Amino acid property groups for scoring peptide-MHC compatibility
# ---------------------------------------------------------------------------

# Amino acid property classification
HYDROPHOBIC = set("AILMFWVP")
POLAR = set("STNQCY")
CHARGED_POS = set("RKH")
CHARGED_NEG = set("DE")
AROMATIC = set("FWY")

# Standard amino acids for random sequence generation
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


def aa_property_class(aa: str) -> int:
    """Map amino acid to property class (0-3)."""
    if aa in HYDROPHOBIC:
        return 0
    if aa in POLAR:
        return 1
    if aa in CHARGED_POS:
        return 2
    if aa in CHARGED_NEG:
        return 3
    return 1  # default to polar


# HLA Class I contact residue positions in the α chain (0-indexed)
# These residues form the peptide-binding groove (α1+α2 domains)
# Pocket positions: A (7), B (9,24,45), C (73), D (99,114), E (147), F (77,80,81,84)
MHC_POCKET_POSITIONS = [7, 9, 24, 45, 59, 63, 66, 70, 73, 77, 80, 81, 84, 97, 99, 114, 116, 147, 152, 156]

# Peptide anchor positions for 9-mers (0-indexed): P2 and P9 (PΩ)
PEPTIDE_ANCHOR_P2 = 1   # P2 (0-indexed)
PEPTIDE_ANCHOR_POMEGA = -1  # PΩ (last residue)


def compute_binding_score(
    peptide: str,
    mhc_a_seq: str,
    noise_std: float = 0.5,
) -> float:
    """Compute synthetic binding score based on peptide-MHC compatibility.

    Score is based on amino acid property matches at anchor-pocket pairs.
    Returns log10(KD) in nM, where lower = tighter binding.

    Rules (simplified from crystallographic knowledge):
    - P2 anchor prefers hydrophobic match with B pocket (pos 9, 45)
    - PΩ anchor prefers hydrophobic match with F pocket (pos 77, 80, 81, 84)
    - Charge complementarity at D pocket (pos 99, 114) with P5/P6
    """
    score = 0.0

    # P2 anchor - B pocket interaction
    p2 = aa_property_class(peptide[PEPTIDE_ANCHOR_P2]) if len(peptide) > 1 else 1
    b_pocket_props = []
    for pos in [9, 45]:
        if pos < len(mhc_a_seq):
            b_pocket_props.append(aa_property_class(mhc_a_seq[pos]))
    if b_pocket_props:
        # Hydrophobic-hydrophobic match = strong binding
        b_avg = sum(b_pocket_props) / len(b_pocket_props)
        if p2 == 0 and b_avg < 0.5:  # both hydrophobic
            score += 2.0
        elif p2 == b_avg:  # same property class
            score += 1.0
        else:
            score -= 0.5

    # PΩ anchor - F pocket interaction
    p_omega = aa_property_class(peptide[PEPTIDE_ANCHOR_POMEGA])
    f_pocket_props = []
    for pos in [77, 80, 81, 84]:
        if pos < len(mhc_a_seq):
            f_pocket_props.append(aa_property_class(mhc_a_seq[pos]))
    if f_pocket_props:
        f_avg = sum(f_pocket_props) / len(f_pocket_props)
        if p_omega == 0 and f_avg < 0.5:
            score += 2.0
        elif p_omega == f_avg:
            score += 1.0
        else:
            score -= 0.5

    # Charge complementarity at D pocket with mid-peptide
    if len(peptide) >= 6:
        mid_prop = aa_property_class(peptide[4])  # P5
        d_pocket_props = []
        for pos in [99, 114]:
            if pos < len(mhc_a_seq):
                d_pocket_props.append(aa_property_class(mhc_a_seq[pos]))
        if d_pocket_props:
            d_avg = sum(d_pocket_props) / len(d_pocket_props)
            # Charge complementarity
            if (mid_prop == 2 and d_avg > 2.5) or (mid_prop == 3 and d_avg < 2.5 and d_avg > 1.5):
                score += 1.5
            elif mid_prop == d_avg:
                score += 0.5

    # Peptide length preference (9-mers bind best for Class I)
    if len(peptide) == 9:
        score += 0.5
    elif len(peptide) in (8, 10):
        score += 0.25

    # Map score to log10(KD) in nM
    # High score -> tight binding (low KD) -> low log10(KD)
    # Score range ~[-1.5, 6.5], map to KD range ~[1 nM, 100,000 nM]
    # log10(KD) = 5.0 - score * 0.6 + noise
    log10_kd = 4.0 - score * 0.5 + random.gauss(0, noise_std)
    log10_kd = max(-1.0, min(6.0, log10_kd))  # clamp to reasonable range

    return log10_kd


# ---------------------------------------------------------------------------
# Structured synthetic data generation
# ---------------------------------------------------------------------------

def random_peptide(length: int = 9) -> str:
    """Generate random peptide sequence."""
    return "".join(random.choice(STANDARD_AA) for _ in range(length))


def random_mhc_sequence(length: int = 275) -> str:
    """Generate random MHC sequence."""
    return "".join(random.choice(STANDARD_AA) for _ in range(length))


def tokenize_sequence(seq: str, max_len: int) -> torch.Tensor:
    """Tokenize amino acid sequence to tensor."""
    ids = []
    for aa in seq[:max_len]:
        idx = AA_TO_IDX.get(aa, AA_TO_IDX["<UNK>"])
        ids.append(idx)
    # Pad to max_len
    while len(ids) < max_len:
        ids.append(0)  # PAD
    return torch.tensor(ids, dtype=torch.long)


@dataclass
class BindingSample:
    """A single binding sample with tokenized sequences."""
    pep_tok: torch.Tensor
    mhc_a_tok: torch.Tensor
    mhc_b_tok: torch.Tensor
    log10_kd: float


class BindingDataset(Dataset):
    """Dataset of binding samples."""

    def __init__(self, samples: List[BindingSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "pep_tok": s.pep_tok,
            "mhc_a_tok": s.mhc_a_tok,
            "mhc_b_tok": s.mhc_b_tok,
            "log10_kd": torch.tensor(s.log10_kd, dtype=torch.float32),
        }


def generate_structured_data(
    n_samples: int = 500,
    n_alleles: int = 10,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Tuple[List[BindingSample], List[BindingSample]]:
    """Generate structured synthetic binding data with learnable signal.

    Returns (train_samples, val_samples) split 80/20.
    """
    random.seed(seed)

    # Generate fixed MHC alleles
    mhc_a_seqs = [random_mhc_sequence(180) for _ in range(n_alleles)]
    # Beta2m is constant for Class I
    b2m_seq = random_mhc_sequence(99)

    samples: List[BindingSample] = []
    for _ in range(n_samples):
        # Random peptide length 8-11
        pep_len = random.randint(8, 11)
        peptide = random_peptide(pep_len)
        allele_idx = random.randint(0, n_alleles - 1)
        mhc_a_seq = mhc_a_seqs[allele_idx]

        log10_kd = compute_binding_score(peptide, mhc_a_seq, noise_std=noise_std)

        pep_tok = tokenize_sequence(peptide, max_len=15)
        mhc_a_tok = tokenize_sequence(mhc_a_seq, max_len=180)
        mhc_b_tok = tokenize_sequence(b2m_seq, max_len=99)

        samples.append(BindingSample(
            pep_tok=pep_tok,
            mhc_a_tok=mhc_a_tok,
            mhc_b_tok=mhc_b_tok,
            log10_kd=log10_kd,
        ))

    # Shuffle and split 80/20
    random.shuffle(samples)
    split = int(0.8 * len(samples))
    return samples[:split], samples[split:]


def collate_binding(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate binding samples into a batch."""
    return {
        "pep_tok": torch.stack([b["pep_tok"] for b in batch]),
        "mhc_a_tok": torch.stack([b["mhc_a_tok"] for b in batch]),
        "mhc_b_tok": torch.stack([b["mhc_b_tok"] for b in batch]),
        "log10_kd": torch.stack([b["log10_kd"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(
    config_name: str,
    model_kwargs: Dict,
    train_samples: List[BindingSample],
    val_samples: List[BindingSample],
    d_model: int = 64,
    n_layers: int = 1,
    n_heads: int = 4,
    n_steps: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
) -> Dict:
    """Train one configuration and return metrics."""
    torch.manual_seed(seed)

    model = Presto(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        **model_kwargs,
    )
    model.train()

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    train_loader = DataLoader(
        BindingDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_binding,
    )

    start_time = time.time()
    step = 0
    losses: List[float] = []

    while step < n_steps:
        for batch in train_loader:
            if step >= n_steps:
                break

            outputs = model(
                pep_tok=batch["pep_tok"],
                mhc_a_tok=batch["mhc_a_tok"],
                mhc_b_tok=batch["mhc_b_tok"],
                mhc_class="I",
            )

            # Get predicted KD from assay head
            pred_kd_log10 = outputs["assays"]["KD_nM"].squeeze(-1)
            target_kd = batch["log10_kd"]

            # Simple MSE loss on log10(KD)
            loss = F.mse_loss(pred_kd_log10, target_kd)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1

    wall_time = time.time() - start_time

    # Evaluate on validation set
    model.eval()
    all_pred = []
    all_true = []
    val_losses = []

    with torch.no_grad():
        val_loader = DataLoader(
            BindingDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_binding,
        )
        for batch in val_loader:
            outputs = model(
                pep_tok=batch["pep_tok"],
                mhc_a_tok=batch["mhc_a_tok"],
                mhc_b_tok=batch["mhc_b_tok"],
                mhc_class="I",
            )
            pred_kd_log10 = outputs["assays"]["KD_nM"].squeeze(-1)
            target_kd = batch["log10_kd"]

            val_loss = F.mse_loss(pred_kd_log10, target_kd)
            val_losses.append(val_loss.item())

            all_pred.append(pred_kd_log10)
            all_true.append(target_kd)

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)

    # Pearson correlation
    pred_mean = all_pred.mean()
    true_mean = all_true.mean()
    pred_centered = all_pred - pred_mean
    true_centered = all_true - true_mean
    numer = (pred_centered * true_centered).sum()
    denom = (pred_centered.pow(2).sum().sqrt() * true_centered.pow(2).sum().sqrt()).clamp(min=1e-8)
    pearson_r = (numer / denom).item()

    final_train_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1)
    final_val_loss = sum(val_losses) / max(len(val_losses), 1)

    return {
        "config": config_name,
        "n_params": n_params,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "pearson_r": pearson_r,
        "wall_time": wall_time,
        "pred_std": all_pred.std().item(),
        "true_std": all_true.std().item(),
    }


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline": {},
    "A": {"binding_n_latent_layers": 6},
    "B8": {"binding_n_queries": 8, "binding_use_decoder_layers": True},
    "C": {"use_pmhc_interaction_block": True},
    "D": {"use_groove_prior": True},
    "A+C": {
        "binding_n_latent_layers": 6,
        "use_pmhc_interaction_block": True,
    },
    "A+D": {"binding_n_latent_layers": 6, "use_groove_prior": True},
    "A+C+D": {
        "binding_n_latent_layers": 6,
        "use_pmhc_interaction_block": True,
        "use_groove_prior": True,
    },
    "C+D": {"use_pmhc_interaction_block": True, "use_groove_prior": True},
    "B4+C": {
        "binding_n_queries": 4,
        "binding_use_decoder_layers": True,
        "use_pmhc_interaction_block": True,
    },
    "B8+C": {
        "binding_n_queries": 8,
        "binding_use_decoder_layers": True,
        "use_pmhc_interaction_block": True,
    },
    "A+B4+C": {
        "binding_n_latent_layers": 4,
        "binding_n_queries": 4,
        "binding_use_decoder_layers": True,
        "use_pmhc_interaction_block": True,
    },
}


def main():
    """Run benchmark across all configurations."""
    print("=" * 80)
    print("Binding Latent Architecture Benchmark")
    print("=" * 80)
    print()

    # Generate data once, shared across all configs
    print("Generating structured synthetic data...")
    train_samples, val_samples = generate_structured_data(
        n_samples=500,
        n_alleles=10,
        noise_std=0.5,
        seed=42,
    )
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    print()

    # Verify signal in data
    kds = [s.log10_kd for s in train_samples + val_samples]
    print(f"  log10(KD) range: [{min(kds):.2f}, {max(kds):.2f}]")
    print(f"  log10(KD) mean: {sum(kds)/len(kds):.2f}, std: {(sum((k - sum(kds)/len(kds))**2 for k in kds) / len(kds))**0.5:.2f}")
    print()

    # Run all configurations
    results = []
    for config_name, kwargs in CONFIGS.items():
        print(f"Training {config_name}...")
        result = train_and_evaluate(
            config_name=config_name,
            model_kwargs=kwargs,
            train_samples=train_samples,
            val_samples=val_samples,
            d_model=64,
            n_layers=1,
            n_heads=4,
            n_steps=500,
            batch_size=32,
            lr=1e-3,
            seed=42,
        )
        print(f"  Pearson r={result['pearson_r']:.4f}, val_loss={result['final_val_loss']:.4f}, "
              f"params={result['n_params']:,}, time={result['wall_time']:.1f}s")
        results.append(result)

    # Print comparison table
    print()
    print("=" * 100)
    print(f"{'Config':<12} {'Params':>10} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'Pearson r':>10} {'Pred Std':>10} {'Time (s)':>10}")
    print("-" * 100)
    for r in results:
        print(f"{r['config']:<12} {r['n_params']:>10,} {r['final_train_loss']:>12.4f} "
              f"{r['final_val_loss']:>12.4f} {r['pearson_r']:>10.4f} "
              f"{r['pred_std']:>10.4f} {r['wall_time']:>10.1f}")
    print("=" * 100)

    # Rank by Pearson r
    ranked = sorted(results, key=lambda r: r["pearson_r"], reverse=True)
    print()
    print("Ranking by Pearson r:")
    for i, r in enumerate(ranked):
        efficiency = r["pearson_r"] / (r["n_params"] / 1e6) if r["n_params"] > 0 else 0
        print(f"  {i+1}. {r['config']:<12} r={r['pearson_r']:.4f}  "
              f"(params={r['n_params']:,}, efficiency={efficiency:.2f} r/M-params)")


if __name__ == "__main__":
    main()
