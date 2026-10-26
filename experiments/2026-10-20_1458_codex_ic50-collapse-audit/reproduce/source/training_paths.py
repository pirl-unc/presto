"""Read-only checkpoint/support inventory; no forward, backward or optimizer step."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from presto.training.checkpointing import load_model_from_checkpoint


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_name(name):
    parts = name.split(".")
    if parts[0] in {"latent_queries", "latent_layers"}:
        return ".".join(parts[:2])
    if parts[0] == "affinity_predictor":
        return ".".join(parts[:3] if parts[1] == "assay_heads" else parts[:2])
    return parts[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    initial = torch.load(args.run / "initial_checkpoint.pt", map_location="cpu", weights_only=True)
    model, selected = load_model_from_checkpoint(args.run / "checkpoint.pt")
    params = list(model.named_parameters())
    state0, state1 = initial["model_state_dict"], selected["model_state_dict"]
    assert set(state0) == set(state1)
    assert all(state0[k].shape == state1[k].shape for k in state0)
    optimizer = selected["optimizer_state_dict"]
    # Canonical trainer uses AdamW(model.parameters()) in exactly this order;
    # uncertainty parameters are added in a separate group. Validate every
    # available slot's shape rather than mapping a differently ordered optimizer.
    param_ids = optimizer["param_groups"][0]["params"]
    assert len(param_ids) == len(params)
    rows = []
    for (name, parameter), param_id in zip(params, param_ids, strict=True):
        before, after = state0[name].double(), state1[name].double()
        assert before.shape == parameter.shape
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        delta = after - before
        slot = optimizer["state"].get(param_id, {})
        moment = slot.get("exp_avg_sq")
        if moment is not None:
            assert moment.shape == parameter.shape
            assert torch.isfinite(moment).all() and (moment >= 0).all()
        rows.append(
            {
                "parameter": name,
                "group": group_name(name),
                "shape": json.dumps(list(parameter.shape)),
                "elements": parameter.numel(),
                "changed_elements": int(torch.count_nonzero(delta)),
                "delta_l2": float(delta.norm()),
                "delta_max_abs": float(delta.abs().max()),
                "initial_l2": float(before.norm()),
                "selected_l2": float(after.norm()),
                "optimizer_state_present": bool(slot),
                "optimizer_step": int(slot["step"]) if slot else 0,
                "nonzero_second_moment_elements": int(torch.count_nonzero(moment)) if slot else 0,
                "second_moment_sum": float(moment.double().sum()) if slot else 0.0,
            }
        )
    groups = defaultdict(lambda: Counter())
    for row in rows:
        group = groups[row["group"]]
        group["tensors"] += 1
        group["elements"] += row["elements"]
        group["changed_tensors"] += row["changed_elements"] > 0
        group["changed_elements"] += row["changed_elements"]
        group["tensors_with_optimizer_state"] += row["optimizer_state_present"]
        group["tensors_with_nonzero_moment"] += row["nonzero_second_moment_elements"] > 0
        group["nonzero_second_moment_elements"] += row["nonzero_second_moment_elements"]
        group["delta_squared_sum"] += row["delta_l2"] ** 2
    grouped = []
    for name, counts in sorted(groups.items()):
        result = dict(counts)
        result["delta_l2"] = result.pop("delta_squared_sum") ** 0.5
        grouped.append({"group": name, **result})
    write_csv(args.output / "parameter-inventory.csv", rows)
    write_csv(args.output / "module-inventory.csv", grouped)
    updates = json.loads((args.run / "output_updates.json").read_text())
    update_columns = [
        "endpoint",
        "column",
        "batches_with_direct_labels",
        "batches_with_labels",
        "label_observations_seen",
        "positive_labels_seen",
        "batches_with_output_gradient",
        "output_gradient_abs_sum",
        "nonfinite_output_gradient_elements",
    ]
    write_csv(
        args.output / "historical-output-gradients.csv",
        [{key: row[key] for key in update_columns} for row in updates["outputs"]],
    )
    support_tables = {}
    for filename in ["row_split_support.csv", "mil_split_support.csv"]:
        with (args.run / filename).open() as handle:
            support_tables[filename] = [
                row for row in csv.DictReader(handle) if row["split"] == "train"
            ]
        write_csv(args.output / f"train-{filename.replace('_', '-')}", support_tables[filename])
    training_metrics = [
        row
        for line in (args.run / "metrics.jsonl").read_text().splitlines()
        if (row := json.loads(line)).get("split") == "train"
    ]
    objectives = []
    for row in training_metrics:
        for key, loss in row.items():
            if key.startswith("loss_"):
                task = key.removeprefix("loss_")
                objectives.append(
                    {
                        "epoch": row["step"],
                        "task": task,
                        "mean_task_loss": loss,
                        "mean_batch_support_when_present": row.get(f"batch_support_{task}"),
                        "mean_normalized_task_weight_when_present": row.get(
                            f"batch_supervised_weight_{task}"
                        ),
                        "uncertainty_weight_at_epoch_end": row.get(f"uw_weight_{task}"),
                    }
                )
    write_csv(args.output / "historical-objectives.csv", objectives)
    with (args.run / "membership.tsv").open() as handle:
        members = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["source_partition"] == "train"
        ]
    endpoints = Counter(
        (row["assay_group"], row["source_endpoint"], row["source_units"]) for row in members
    )
    write_csv(
        args.output / "train-source-endpoints.csv",
        [
            {
                "assay_group": key[0],
                "source_endpoint": key[1],
                "source_units": key[2],
                "count": count,
            }
            for key, count in sorted(endpoints.items())
        ],
    )
    totals = {
        "parameter_tensors": len(rows),
        "parameter_elements": sum(row["elements"] for row in rows),
        "unchanged_tensors": sum(row["changed_elements"] == 0 for row in rows),
        "elements_in_wholly_unchanged_tensors": sum(
            row["elements"] for row in rows if row["changed_elements"] == 0
        ),
        "tensors_without_optimizer_state": sum(not row["optimizer_state_present"] for row in rows),
        "tensors_with_nonzero_second_moment": sum(
            row["nonzero_second_moment_elements"] > 0 for row in rows
        ),
    }
    assert totals["parameter_elements"] == 409850
    assert len(members) == 7241
    repo = Path(__file__).resolve().parents[3]
    summary = {
        "scope": "Read-only training-path inventory; no new predictions or optimization",
        "agent": "Codex / GPT-6",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_config": selected["model_config"],
        "model_dimensions": {
            "d_model": model.d_model,
            "stream_layers": len(model.stream_encoder.layers),
            "binding_queries": model.binding_n_queries,
            "binding_layers": model.binding_n_latent_layers,
            "pmhc_interaction_dim": model.pmhc_interaction_vec_dim,
        },
        "initial_epoch": initial["epoch"],
        "selected_epoch": selected["epoch"],
        "historical_ledger_epochs": updates["epochs_observed"],
        "historical_ledger_optimizer_steps": updates["optimizer_steps"],
        "totals": totals,
        "evidence_limits": [
            "Parameter deltas and Adam second moments cover only the selected epoch-1 checkpoint.",
            "Output gradient ledger and objective summaries cover both historical epochs.",
            "Nonzero second moments show past nonzero gradients, not useful learned biology.",
            "Weight decay can change parameters with zero task gradient; "
            "unchanged subrows can coexist with changed tensors.",
            "No held-out inference or quality evaluation was performed.",
            "Optimizer parameter order is the canonical trainer's model.parameters() order; "
            "count and moment shapes verified.",
        ],
        "source_files": {
            name: {"path": str(args.run / name), "sha256": sha256(args.run / name)}
            for name in [
                "initial_checkpoint.pt",
                "checkpoint.pt",
                "output_updates.json",
                "row_split_support.csv",
                "mil_split_support.csv",
                "metrics.jsonl",
                "membership.tsv",
            ]
        },
        "script_sha256": sha256(Path(__file__)),
        "runtime_seconds": time.monotonic() - started,
    }
    (args.output / "training-path-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"totals": totals, "dimensions": summary["model_dimensions"]}, indent=2))


if __name__ == "__main__":
    main()
