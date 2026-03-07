#!/usr/bin/env python
"""Train on real IEDB data for 500 minibatches, probing SLLQHLIGL predictions.

Loads real binding/elution/tcell/processing data from the merged TSV,
trains with the full multi-task loss, and tracks per-allele predictions
for SLLQHLIGL every 50 minibatches.

Usage:
    python scripts/probe_training.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from presto.models.presto import Presto
from presto.data import PrestoDataset, PrestoCollator, create_dataloader
from presto.data.tokenizer import Tokenizer
from presto.data.groove import prepare_mhc_input
from presto.data.mhc_index import build_mhc_sequence_lookup, load_mhc_index
from presto.training.losses import UncertaintyWeighting
from presto.scripts.train_synthetic import compute_loss, LOSS_TASK_NAMES
from presto.scripts.train_iedb import (
    load_records_from_merged_tsv,
    augment_binding_records_with_synthetic_negatives,
    augment_elution_records_with_synthetic_negatives,
    augment_processing_records_with_synthetic_negatives,
    cascade_binding_negatives_to_downstream,
    bootstrap_missing_modalities_for_canary,
    resolve_mhc_sequences_from_index,
    SYNTHETIC_ELUTION_NEGATIVE_SCALE,
    SYNTHETIC_CASCADE_ELUTION_NEGATIVE_SCALE,
    SYNTHETIC_CASCADE_TCELL_NEGATIVE_SCALE,
)
from presto.models.affinity import DEFAULT_MAX_AFFINITY_NM

# --- Config ---
TOTAL_BATCHES = 500
PROBE_EVERY = 50
PEPTIDE = "SLLQHLIGL"
ALLELES = ["HLA-A*02:01", "HLA-A*24:02"]
BATCH_SIZE = 128
D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
LR = 2.8e-4
SEED = 42
DATA_DIR = Path("./data")  # overridden by --data-dir
OUT_DIR = Path("artifacts/probe_training_iedb")  # overridden by --out-dir

# Data caps — enough real signal for allele differentiation
MAX_BINDING = 2000
MAX_ELUTION = 2000
MAX_TCELL = 1000
MAX_PROCESSING = 500
MAX_KINETICS = 500
MAX_STABILITY = 500
MAX_VDJDB = 500


def load_allele_sequences():
    """Load allele sequences from the MHC index."""
    index_csv = DATA_DIR / "mhc_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"MHC index not found: {index_csv}")
    records = load_mhc_index(str(index_csv))
    return build_mhc_sequence_lookup(records)


def probe_peptide(model, tokenizer, allele_sequences, device):
    """Run SLLQHLIGL through the model for each allele, return detailed outputs."""
    model.eval()
    results = {}
    for allele in ALLELES:
        mhc_a_seq = None
        for key in [allele, allele.replace("HLA-", "")]:
            if key in allele_sequences:
                mhc_a_seq = allele_sequences[key]
                break
        if mhc_a_seq is None:
            for k, v in allele_sequences.items():
                if allele in k or k in allele:
                    mhc_a_seq = v
                    break
        if mhc_a_seq is None:
            raise ValueError(f"Cannot resolve allele: {allele}")

        prepared = prepare_mhc_input(mhc_a=mhc_a_seq, mhc_class="I")
        pep_tok = torch.tensor(tokenizer.encode(PEPTIDE, max_len=50)).unsqueeze(0).to(device)
        mhc_a_tok = torch.tensor(
            tokenizer.encode(prepared.groove_half_1, max_len=120)
        ).unsqueeze(0).to(device)
        mhc_b_tok = torch.tensor(
            tokenizer.encode(prepared.groove_half_2, max_len=120)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(
                pep_tok=pep_tok, mhc_a_tok=mhc_a_tok, mhc_b_tok=mhc_b_tok,
                mhc_class="I", species="human",
            )

        entry = {}
        for key in [
            "processing_logit", "binding_logit", "presentation_logit",
            "elution_logit", "ms_detectability_logit",
            "recognition_cd8_logit", "recognition_cd4_logit",
            "immunogenicity_cd8_logit", "immunogenicity_cd4_logit",
            "immunogenicity_logit", "foreignness_logit",
        ]:
            if key in outputs:
                val = outputs[key]
                if isinstance(val, torch.Tensor):
                    entry[key] = float(val.squeeze().item())
                    entry[key.replace("_logit", "_prob")] = float(
                        torch.sigmoid(val).squeeze().item()
                    )

        if "binding_latents" in outputs:
            for k, v in outputs["binding_latents"].items():
                if isinstance(v, torch.Tensor):
                    entry[f"binding_latent_{k}"] = float(v.squeeze().item())

        if "assays" in outputs and isinstance(outputs["assays"], dict):
            for k, v in outputs["assays"].items():
                if isinstance(v, torch.Tensor):
                    entry[f"assay_{k}"] = float(v.squeeze().item())

        if "mhc_class_probs" in outputs:
            cp = outputs["mhc_class_probs"].squeeze()
            if cp.numel() >= 2:
                entry["mhc_class1_prob"] = float(cp[0].item())
                entry["mhc_class2_prob"] = float(cp[1].item())

        if "latent_vecs" in outputs:
            for k, v in outputs["latent_vecs"].items():
                if isinstance(v, torch.Tensor):
                    entry[f"latent_norm_{k}"] = float(v.norm().item())

        if "core_start_probs" in outputs:
            csp = outputs["core_start_probs"].squeeze()
            entry["core_expected_start"] = float(
                (torch.arange(csp.numel(), device=csp.device).float() * csp).sum().item()
            )

        results[allele] = entry
    model.train()
    return results


def main():
    global DATA_DIR, OUT_DIR, TOTAL_BATCHES, BATCH_SIZE
    parser = argparse.ArgumentParser(description="Probe training on real IEDB data")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--batches", type=int, default=TOTAL_BATCHES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    TOTAL_BATCHES = args.batches
    BATCH_SIZE = args.batch_size

    torch.manual_seed(SEED)
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load real data from merged TSV ---
    merged_tsv = DATA_DIR / "merged_deduped.tsv"
    if not merged_tsv.exists():
        raise FileNotFoundError(
            f"Merged TSV not found: {merged_tsv}. "
            "Run `python -m presto data merge --datadir ./data` first."
        )

    print(f"Loading data from {merged_tsv}...")
    (
        binding_records, kinetics_records, stability_records,
        processing_records, elution_records, tcell_records,
        vdjdb_records, merged_stats,
    ) = load_records_from_merged_tsv(
        merged_tsv=merged_tsv,
        max_binding=MAX_BINDING,
        max_kinetics=MAX_KINETICS,
        max_stability=MAX_STABILITY,
        max_processing=MAX_PROCESSING,
        max_elution=MAX_ELUTION,
        max_tcell=MAX_TCELL,
        max_vdjdb=MAX_VDJDB,
        cap_sampling="reservoir",
        sampling_seed=SEED + 17,
    )
    print(f"  Rows scanned: {merged_stats['rows_scanned']}")
    for assay, count in merged_stats["rows_by_assay"].items():
        print(f"    {assay}: {count}")

    # Collect unique alleles from all record types, then resolve sequences
    index_csv = str(DATA_DIR / "mhc_index.csv")
    all_alleles: list[str] = []
    for rec in binding_records + kinetics_records + stability_records + processing_records + tcell_records:
        a = getattr(rec, "mhc_allele", None)
        if a:
            all_alleles.append(a.strip())
    for rec in elution_records:
        for a in getattr(rec, "alleles", []) or []:
            if a:
                all_alleles.append(a.strip())
    for rec in vdjdb_records:
        a = getattr(rec, "mhc_a", None)
        if a:
            all_alleles.append(a.strip())
    unique_alleles = sorted({a for a in all_alleles if a})
    mhc_sequences, mhc_stats = resolve_mhc_sequences_from_index(
        index_csv=index_csv,
        alleles=unique_alleles,
    )
    print(f"  Resolved {mhc_stats['resolved']}/{mhc_stats['total']} MHC alleles")

    # Bootstrap sparse modalities
    extra_kin, extra_stab, extra_proc, boot_stats = bootstrap_missing_modalities_for_canary(
        binding_records=binding_records,
        kinetics_records=kinetics_records,
        stability_records=stability_records,
        processing_records=processing_records,
        seed=SEED,
    )
    kinetics_records.extend(extra_kin)
    stability_records.extend(extra_stab)
    processing_records.extend(extra_proc)
    print(f"  Bootstrapped: kin={len(extra_kin)}, stab={len(extra_stab)}, proc={len(extra_proc)}")

    # Synthetic negatives
    synth_ratio = 0.5
    binding_records, _ = augment_binding_records_with_synthetic_negatives(
        binding_records=binding_records,
        mhc_sequences=mhc_sequences,
        negative_ratio=synth_ratio,
        weak_value_min_nM=DEFAULT_MAX_AFFINITY_NM * 0.5,
        weak_value_max_nM=DEFAULT_MAX_AFFINITY_NM,
        seed=SEED,
        class_i_no_mhc_beta_ratio=0.0,
    )
    elution_records, _ = augment_elution_records_with_synthetic_negatives(
        elution_records=elution_records,
        negative_ratio=synth_ratio * SYNTHETIC_ELUTION_NEGATIVE_SCALE,
        seed=SEED,
    )
    processing_records, _ = augment_processing_records_with_synthetic_negatives(
        processing_records=processing_records,
        negative_ratio=0.25,
        seed=SEED,
    )
    elution_records, tcell_records, _ = cascade_binding_negatives_to_downstream(
        binding_records=binding_records,
        elution_records=elution_records,
        tcell_records=tcell_records,
        elution_ratio=synth_ratio * SYNTHETIC_CASCADE_ELUTION_NEGATIVE_SCALE,
        tcell_ratio=synth_ratio * SYNTHETIC_CASCADE_TCELL_NEGATIVE_SCALE,
        seed=SEED,
    )

    # Build dataset
    dataset = PrestoDataset(
        binding_records=binding_records,
        kinetics_records=kinetics_records,
        stability_records=stability_records,
        processing_records=processing_records,
        elution_records=elution_records,
        tcell_records=tcell_records,
        tcr_evidence_records=vdjdb_records,
        mhc_sequences=mhc_sequences,
        strict_mhc_resolution=False,
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])

    collator = PrestoCollator()
    loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collator=collator)
    print(f"Dataset: {len(dataset)} total, {train_size} train, {len(loader)} batches/epoch")

    # --- Model ---
    model = Presto(
        d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
        use_pmhc_interaction_block=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    uw = UncertaintyWeighting(n_tasks=len(LOSS_TASK_NAMES)).to(device)
    optimizer.add_param_group({"params": uw.parameters()})

    regularization_cfg = {
        "consistency_cascade_weight": 0.2,
        "consistency_assay_affinity_weight": 0.1,
        "consistency_no_b2m_weight": 0.5,
        "binding_orthogonality_weight": 0.01,
        "mhc_attention_sparsity_weight": 0.5,
        "mhc_attention_sparsity_min_residues": 20.0,
        "mhc_attention_sparsity_max_residues": 50.0,
    }

    # --- Allele sequences for probing ---
    tokenizer = Tokenizer()
    allele_sequences = load_allele_sequences()
    print(f"Loaded {len(allele_sequences)} allele sequences for probing")

    # --- Training loop ---
    history = []
    batch_idx = 0
    model.train()

    probes = probe_peptide(model, tokenizer, allele_sequences, device)
    history.append({"batch_idx": 0, "losses": {}, "probes": probes})
    print(f"[batch 0] Initial probe done")

    loader_iter = iter(loader)
    pbar = tqdm(total=TOTAL_BATCHES, desc="Training")

    while batch_idx < TOTAL_BATCHES:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch_idx += 1
        total_loss, task_losses, _ = compute_loss(
            model, batch, device,
            uncertainty_weighting=uw,
            regularization=regularization_cfg,
        )
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_scalars = {
            k: float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in task_losses.items()
        }
        loss_scalars["total"] = float(total_loss.detach().item())

        pbar.set_postfix(loss=f"{loss_scalars['total']:.4f}")
        pbar.update(1)

        if batch_idx % PROBE_EVERY == 0:
            probes = probe_peptide(model, tokenizer, allele_sequences, device)
            history.append({"batch_idx": batch_idx, "losses": loss_scalars, "probes": probes})
            a02 = probes.get("HLA-A*02:01", {})
            a24 = probes.get("HLA-A*24:02", {})
            print(
                f"\n[batch {batch_idx}] loss={loss_scalars['total']:.4f} | "
                f"A0201 bind={a02.get('binding_prob', '?'):.3f} pres={a02.get('presentation_prob', '?'):.3f} | "
                f"A2402 bind={a24.get('binding_prob', '?'):.3f} pres={a24.get('presentation_prob', '?'):.3f}"
            )

    pbar.close()

    history_path = OUT_DIR / "probe_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to {history_path}")

    try:
        make_plots(history, OUT_DIR)
        print(f"Plots saved to {OUT_DIR}")
    except ImportError as e:
        print(f"Skipping plots (matplotlib not available): {e}")


def make_plots(history, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe_entries = [h for h in history if h["probes"]]
    steps = [h["batch_idx"] for h in probe_entries]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for allele, color in zip(ALLELES, ["tab:blue", "tab:orange"]):
        bind_probs = [h["probes"].get(allele, {}).get("binding_prob", None) for h in probe_entries]
        pres_probs = [h["probes"].get(allele, {}).get("presentation_prob", None) for h in probe_entries]
        proc_probs = [h["probes"].get(allele, {}).get("processing_prob", None) for h in probe_entries]
        axes[0].plot(steps, bind_probs, "-o", color=color, label=f"{allele} binding", markersize=4)
        axes[0].plot(steps, pres_probs, "--s", color=color, label=f"{allele} presentation", markersize=4)
        if proc_probs[0] is not None:
            axes[0].plot(steps, proc_probs, ":^", color=color, label=f"{allele} processing", markersize=4, alpha=0.6)
    axes[0].set_xlabel("Minibatch")
    axes[0].set_ylabel("Probability")
    axes[0].set_title(f"SLLQHLIGL: Binding & Presentation (IEDB)")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    for allele, color in zip(ALLELES, ["tab:blue", "tab:orange"]):
        bind_logits = [h["probes"].get(allele, {}).get("binding_logit", None) for h in probe_entries]
        axes[1].plot(steps, bind_logits, "-o", color=color, label=allele, markersize=4)
    axes[1].set_xlabel("Minibatch")
    axes[1].set_ylabel("Logit")
    axes[1].set_title(f"SLLQHLIGL: Binding Logit (raw)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "binding_presentation.png", dpi=150)
    plt.close(fig)

    loss_entries = [h for h in history if h["losses"]]
    if loss_entries:
        loss_steps = [h["batch_idx"] for h in loss_entries]
        all_loss_keys = sorted(set(k for h in loss_entries for k in h["losses"] if k != "total"))
        n_loss_keys = len(all_loss_keys)
        ncols = 3
        nrows = max((n_loss_keys + ncols - 1) // ncols, 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes_flat = axes.flatten() if n_loss_keys > 1 else [axes]
        for i, key in enumerate(all_loss_keys):
            vals = [h["losses"].get(key, None) for h in loss_entries]
            axes_flat[i].plot(loss_steps, vals, "-", linewidth=1.2)
            axes_flat[i].set_title(key, fontsize=10)
            axes_flat[i].set_xlabel("Minibatch")
            axes_flat[i].set_ylabel("Loss")
            axes_flat[i].grid(True, alpha=0.3)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Individual Task Losses (IEDB)", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(out_dir / "individual_losses.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_steps, [h["losses"]["total"] for h in loss_entries], "-o", markersize=3)
        ax.set_xlabel("Minibatch")
        ax.set_ylabel("Total Loss")
        ax.set_title("Total Training Loss (IEDB)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "total_loss.png", dpi=150)
        plt.close(fig)

    latent_keys = sorted(set(
        k for h in probe_entries for allele in ALLELES
        for k in h["probes"].get(allele, {}) if k.startswith("latent_norm_")
    ))
    if latent_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax_idx, allele in enumerate(ALLELES):
            for lk in latent_keys:
                short = lk.replace("latent_norm_", "")
                vals = [h["probes"].get(allele, {}).get(lk, None) for h in probe_entries]
                axes[ax_idx].plot(steps, vals, "-", label=short, linewidth=1.2)
            axes[ax_idx].set_xlabel("Minibatch")
            axes[ax_idx].set_ylabel("L2 Norm")
            axes[ax_idx].set_title(f"Latent Norms: {allele}")
            axes[ax_idx].legend(fontsize=7, ncol=2)
            axes[ax_idx].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "latent_norms.png", dpi=150)
        plt.close(fig)

    assay_keys = sorted(set(
        k for h in probe_entries for allele in ALLELES
        for k in h["probes"].get(allele, {}) if k.startswith("assay_")
    ))
    if assay_keys:
        fig, axes = plt.subplots(1, len(assay_keys), figsize=(5 * len(assay_keys), 4))
        if len(assay_keys) == 1:
            axes = [axes]
        for i, ak in enumerate(assay_keys):
            for allele, color in zip(ALLELES, ["tab:blue", "tab:orange"]):
                vals = [h["probes"].get(allele, {}).get(ak, None) for h in probe_entries]
                axes[i].plot(steps, vals, "-o", color=color, label=allele, markersize=4)
            axes[i].set_title(ak.replace("assay_", ""), fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
        fig.suptitle("Assay Outputs (log10 nM scale)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "assay_outputs.png", dpi=150)
        plt.close(fig)

    rec_keys = [
        ("recognition_cd8_prob", "Recog CD8"), ("recognition_cd4_prob", "Recog CD4"),
        ("immunogenicity_cd8_prob", "Immuno CD8"), ("immunogenicity_cd4_prob", "Immuno CD4"),
        ("foreignness_prob", "Foreignness"),
    ]
    available_rec = [
        (k, l) for k, l in rec_keys
        if any(h["probes"].get(ALLELES[0], {}).get(k) is not None for h in probe_entries)
    ]
    if available_rec:
        fig, ax = plt.subplots(figsize=(10, 5))
        for allele, ls in zip(ALLELES, ["-", "--"]):
            for k, label in available_rec:
                vals = [h["probes"].get(allele, {}).get(k, None) for h in probe_entries]
                ax.plot(steps, vals, ls, label=f"{allele} {label}", linewidth=1.2)
        ax.set_xlabel("Minibatch")
        ax.set_ylabel("Probability")
        ax.set_title("Recognition & Immunogenicity")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "recognition_immunogenicity.png", dpi=150)
        plt.close(fig)

    print(f"Generated plots in {out_dir}/")


if __name__ == "__main__":
    main()
