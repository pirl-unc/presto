#!/usr/bin/env python
"""End-to-end training script with synthetic data.

This script demonstrates the full PRESTO training pipeline:
1. Generate synthetic training data
2. Create data loaders
3. Train the model
4. Evaluate on held-out data

Usage:
    python -m presto.scripts.train_synthetic --epochs 5 --batch_size 16
"""

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split

from presto.models.presto import Presto
from presto.data import (
    PrestoDataset,
    PrestoCollator,
    create_dataloader,
    generate_synthetic_binding_data,
    generate_synthetic_elution_data,
    generate_synthetic_tcr_data,
    generate_synthetic_mhc_sequences,
    write_binding_csv,
    write_elution_csv,
    write_tcr_csv,
    write_mhc_fasta,
)
from presto.training.losses import censor_aware_loss, UncertaintyWeighting


def create_synthetic_data(data_dir: Path, n_binding: int = 200, n_elution: int = 100, n_tcr: int = 100):
    """Generate and save synthetic training data."""
    print("Generating synthetic data...")

    alleles = ["HLA-A*02:01", "HLA-A*03:01", "HLA-B*07:02", "HLA-B*08:01"]

    # Generate data
    binding_data = generate_synthetic_binding_data(n_binding, alleles)
    elution_data = generate_synthetic_elution_data(n_elution, alleles)
    tcr_data = generate_synthetic_tcr_data(n_tcr, alleles[:2])  # Fewer alleles for TCR
    mhc_sequences = generate_synthetic_mhc_sequences(alleles)

    # Save to files
    data_dir.mkdir(parents=True, exist_ok=True)
    write_binding_csv(binding_data, data_dir / "binding.csv")
    write_elution_csv(elution_data, data_dir / "elution.csv")
    write_tcr_csv(tcr_data, data_dir / "tcr.csv")
    write_mhc_fasta(mhc_sequences, data_dir / "mhc.fasta")

    print(f"  Binding samples: {len(binding_data)}")
    print(f"  Elution samples: {len(elution_data)}")
    print(f"  TCR samples: {len(tcr_data)}")
    print(f"  MHC alleles: {len(mhc_sequences)}")

    return binding_data, elution_data, tcr_data, mhc_sequences


def compute_loss(model, batch, device, uncertainty_weighting=None):
    """Compute multi-task loss for a batch."""
    # Move batch to device
    batch = batch.to(device)

    # Forward pass
    outputs = model(
        pep_tok=batch.pep_tok,
        mhc_a_tok=batch.mhc_a_tok,
        mhc_b_tok=batch.mhc_b_tok,
        mhc_class=batch.mhc_class[0],  # Assume uniform class in batch
        tcr_a_tok=batch.tcr_a_tok,
        tcr_b_tok=batch.tcr_b_tok,
        flank_n_tok=batch.flank_n_tok,
        flank_c_tok=batch.flank_c_tok,
    )

    losses = {}

    # Binding loss (censor-aware)
    if batch.bind_target is not None and batch.bind_mask is not None:
        if batch.bind_mask.sum() > 0:
            # Use KD prediction from assays (already in log10 nM)
            kd_pred = outputs["assays"]["KD_nM"]
            # Convert target from nM to log10(nM)
            # Clamp to avoid log(0)
            target_log10 = torch.log10(batch.bind_target.clamp(min=1e-3))
            bind_loss = censor_aware_loss(
                kd_pred.squeeze(-1),
                target_log10.squeeze(-1),
                batch.bind_qual.squeeze(-1),
                reduction='none',  # Get per-sample losses for masking
            )
            # Mask to valid samples
            bind_loss = (bind_loss * batch.bind_mask).sum() / (batch.bind_mask.sum() + 1e-8)
            losses["binding"] = bind_loss

    # Elution loss (BCE)
    if batch.elution_label is not None and batch.elution_mask is not None:
        if batch.elution_mask.sum() > 0:
            elution_pred = outputs["elution_logit"].squeeze(-1)
            elution_loss = nn.functional.binary_cross_entropy_with_logits(
                elution_pred, batch.elution_label, reduction='none'
            )
            elution_loss = (elution_loss * batch.elution_mask).sum() / (batch.elution_mask.sum() + 1e-8)
            losses["elution"] = elution_loss

    # T-cell loss (BCE)
    if batch.tcell_label is not None and batch.tcell_mask is not None:
        if batch.tcell_mask.sum() > 0:
            if "tcell_logit" in outputs:
                tcell_pred = outputs["tcell_logit"].squeeze(-1)
                tcell_loss = nn.functional.binary_cross_entropy_with_logits(
                    tcell_pred, batch.tcell_label, reduction='none'
                )
                tcell_loss = (tcell_loss * batch.tcell_mask).sum() / (batch.tcell_mask.sum() + 1e-8)
                losses["tcell"] = tcell_loss

    # Combine losses
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if uncertainty_weighting is not None:
        total_loss = uncertainty_weighting(list(losses.values()))
    else:
        total_loss = sum(losses.values()) / len(losses)

    return total_loss, losses


def train_epoch(model, train_loader, optimizer, device, uncertainty_weighting=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss, loss_dict = compute_loss(model, batch, device, uncertainty_weighting)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            loss, _ = compute_loss(model, batch, device)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def run(args: argparse.Namespace) -> None:
    """Run synthetic training with parsed arguments."""

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(tempfile.mkdtemp()) / "presto_data"

    # Generate synthetic data
    binding_data, elution_data, tcr_data, mhc_sequences = create_synthetic_data(
        data_dir, args.n_binding, args.n_elution, args.n_tcr
    )

    # Create dataset
    dataset = PrestoDataset(
        binding_records=binding_data,
        elution_records=elution_data,
        tcr_records=tcr_data,
        mhc_sequences=mhc_sequences,
    )
    print(f"Total samples: {len(dataset)}")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    collator = PrestoCollator()
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, collator=collator)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, collator=collator)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = Presto(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and uncertainty weighting
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    uncertainty_weighting = UncertaintyWeighting(n_tasks=3).to(device)
    optimizer.add_param_group({"params": uncertainty_weighting.parameters()})

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, uncertainty_weighting)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.checkpoint:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "model_config": {
                        "d_model": args.d_model,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                    },
                }, args.checkpoint)
                print(f"  Saved checkpoint to {args.checkpoint}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")

    # Quick inference test
    print("\nRunning inference test...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        batch = batch.to(device)
        outputs = model(
            pep_tok=batch.pep_tok,
            mhc_a_tok=batch.mhc_a_tok,
            mhc_b_tok=batch.mhc_b_tok,
            mhc_class="I",
        )
        pres_prob = torch.sigmoid(outputs["presentation_logit"])
        print(f"Sample presentation probabilities: {pres_prob[:5].cpu().numpy().flatten()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train PRESTO on synthetic data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_binding", type=int, default=200, help="Number of binding samples")
    parser.add_argument("--n_elution", type=int, default=100, help="Number of elution samples")
    parser.add_argument("--n_tcr", type=int, default=100, help="Number of TCR samples")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (temp if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    run(args)


if __name__ == "__main__":
    main()
