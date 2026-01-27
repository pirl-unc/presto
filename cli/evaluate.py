"""Evaluation CLI commands."""

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from ..data import PrestoCollator, PrestoDataset, create_dataloader
from ..models.presto import Presto
from ..scripts import train_synthetic


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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = None
    if isinstance(checkpoint, dict):
        config = checkpoint.get("model_config") or checkpoint.get("config")

    d_model = args.d_model or (config.get("d_model") if config else 128)
    n_layers = args.n_layers or (config.get("n_layers") if config else 2)
    n_heads = args.n_heads or (config.get("n_heads") if config else 4)

    model = Presto(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    val_loss = train_synthetic.evaluate(model, val_loader, device)

    payload = {"val_loss": val_loss}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"val_loss: {val_loss:.4f}")

    return 0
