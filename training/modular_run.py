"""Manifest-driven implementation behind the existing unified training command."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time

import torch

from ..data.modular import ENDPOINTS, fit_target_scales, identity_head
from ..data.modular_sources import ObservationIndex, file_sha256
from ..models.modular import ModularConfig, ModularPresto
from ..models.modular_support import feature_report
from .modular import (
    identity_targets,
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from .modular_evaluation import evaluate

DEFAULT_TRAINING = {
    "seed": 42,
    "batch_size": 64,
    "max_epochs": 100,
    "max_steps": None,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "identity_weight": 0.1,
    "patience": 15,
    "min_epochs": 10,
    "identity_batch_size": 64,
    "test_policy": "evaluate_selected_once",
    "no_test_reason": None,
    "coverage_policy": "required",
    "coverage_exception_reason": None,
}


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def preflight(index: ObservationIndex):
    config = ModularConfig.from_dict(index.manifest["model"])
    supplied = index.manifest.get("training", {})
    if set(supplied) - DEFAULT_TRAINING.keys():
        raise ValueError("unknown modular training option")
    training = {**DEFAULT_TRAINING, **supplied}
    for key in ("batch_size", "max_epochs", "patience", "min_epochs", "identity_batch_size"):
        if not isinstance(training[key], int) or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if training["max_steps"] is not None and (
        not isinstance(training["max_steps"], int) or training["max_steps"] <= 0
    ):
        raise ValueError("max_steps must be a positive integer")
    if (
        not all(
            math.isfinite(training[key])
            for key in ("learning_rate", "weight_decay", "identity_weight")
        )
        or training["learning_rate"] <= 0
        or training["weight_decay"] < 0
        or training["identity_weight"] < 0
    ):
        raise ValueError("invalid optimizer or identity replay weights")
    if training["test_policy"] not in ("evaluate_selected_once", "no_test_training_diagnostic"):
        raise ValueError("explicit supported test-use policy required")
    if training["coverage_policy"] not in ("required", "diagnostic"):
        raise ValueError("unsupported coverage gate policy")
    if training["coverage_policy"] == "diagnostic" and (
        training["test_policy"] != "no_test_training_diagnostic"
        or not training["coverage_exception_reason"]
    ):
        raise ValueError("coverage exception requires a justified, declared no-test diagnostic")
    if training["test_policy"] == "no_test_training_diagnostic":
        if not training["no_test_reason"] or any(
            key.startswith("test:") for key in index.partitions
        ):
            raise ValueError("no-test diagnostic requires a reason and no test split")
    else:
        if not any(key.startswith("validation:") for key in index.partitions) or not any(
            key.startswith("test:") for key in index.partitions
        ):
            raise ValueError(
                "selected-checkpoint evaluation requires validation and test partitions"
            )
    scales = fit_target_scales(index)
    counts, label_values, feature_states = Counter(), defaultdict(set), defaultdict(set)
    source_label_values = defaultdict(set)
    observed_features = {"acquisition": set(), "observation": set()}
    assay_classes, has_multi_register = set(), False
    heldout_multi_register = False
    # Label and capability inspection is sequence-only; no training or random model scores.
    validator = ModularPresto(config)
    for row in index:
        if row.endpoint == "identity":
            targets = identity_targets(row, validator)
            for name in targets:
                if name.startswith("mhc") and not row.mhc:
                    raise ValueError("MHC identity label without grooves")
                if name.startswith("tcr") and not (row.tcr_a and row.tcr_b):
                    raise ValueError("TCR identity label without both chains")
                if name.startswith("peptide_source:") and not row.peptide:
                    raise ValueError("peptide source label without a peptide")
                if row.split == "train":
                    counts[identity_head(name)] += 1
                    label_values[identity_head(name)].add(row.identity_labels[name])
                    if name.startswith("peptide_source:"):
                        source_label_values[name.split(":", 1)[1]].add(row.identity_labels[name])
        else:
            if row.endpoint not in config.endpoints:
                raise ValueError(f"admitted row targets an unallocated endpoint: {row.endpoint}")
            route = ENDPOINTS[row.endpoint].route
            if route in ("bulk", "detector") or row.endpoint == "elution_observation":
                validator._covariates(
                    config.acquisition_features,
                    row.context.acquisition,
                    row.context.acquisition_units,
                    config.acquisition_units,
                )
                if row.split == "train":
                    observed_features["acquisition"].update(row.context.acquisition)
            if row.endpoint in ("bulk_observation", "elution_observation"):
                validator._covariates(
                    config.observation_features,
                    row.context.observation,
                    row.context.observation_units,
                    config.observation_units,
                )
                if row.split == "train":
                    observed_features["observation"].update(row.context.observation)
            if route in ("bulk", "digest") and row.context.enzyme is not None:
                if row.context.enzyme not in config.enzymes:
                    raise ValueError("enzyme outside checkpoint vocabulary")
            for mhc in row.mhc:
                known = mhc.class_override or mhc.mhc_class
                if known is None and "mhc_class" not in config.identity_tasks:
                    raise ValueError("inferred class requires an allocated identity task")
                if known is not None and known not in config.mhc_classes:
                    raise ValueError("row MHC class outside configured branches")
                if known != "I" and len(row.peptide) < 9:
                    raise ValueError("class-II/uncertain class requires complete 9-residue cores")
                if row.split == "train":
                    assay_classes.update((known,) if known else config.mhc_classes)
                    has_multi_register |= known != "I" and len(row.peptide) > 9
                else:
                    heldout_multi_register |= known != "I" and len(row.peptide) > 9
            if row.split == "train":
                counts[row.endpoint] += 1
                if ENDPOINTS[row.endpoint].route in ("cellular", "tcr_cellular"):
                    for name in config.components:
                        component = row.context.component_overrides.get(
                            name
                        ) or row.context.components.get(name)
                        feature_states[name].add(component.state if component else "unknown")
    for endpoint in config.endpoints:
        if not counts[endpoint]:
            raise ValueError(f"allocated endpoint has no training labels: {endpoint}")
    for task in config.identity_tasks:
        if counts[task] == 0 and not index.manifest.get("pretrained"):
            raise ValueError(f"identity task has no labels or inherited weights: {task}")
        if counts[task] and len(label_values[task]) < 2:
            raise ValueError(f"identity task needs varied training labels: {task}")
    if "peptide_source" in config.identity_tasks:
        for category in config.source_categories:
            if source_label_values[category] != {"present", "absent"}:
                raise ValueError(
                    "peptide source category needs positive and negative reference labels: "
                    f"{category}"
                )
    if config.endpoints and config.identity_tasks:
        if not index.partitions.get("train:identity") or training["identity_weight"] <= 0:
            raise ValueError("fine-tuning the shared encoder requires explicit identity replay")
    for name, states in feature_states.items():
        if len(states - {"unknown", "reference"}) < 2:
            raise ValueError(f"component effect lacks contrasted training states: {name}")
    for family, configured in (
        ("acquisition", config.acquisition_features),
        ("observation", config.observation_features),
    ):
        missing = set(configured) - observed_features[family]
        if missing:
            raise ValueError(
                f"{family} features have no observed training values: {sorted(missing)}"
            )
    pruning = []
    if validator.has_pmhc:
        if set(config.mhc_classes) - assay_classes:
            raise ValueError("declared MHC branch has no admitted assay training support")
        if has_multi_register and not config.register_pooling:
            raise ValueError("multi-register training observations require register pooling")
        if "II" in config.mhc_classes and not has_multi_register:
            if heldout_multi_register:
                raise ValueError("held-out multi-register peptides lack training-path support")
            config = replace(config, register_pooling=False)
            pruning.append("register pooling: all admitted class-II peptides have one core")
    return (
        config,
        training,
        scales,
        {
            "census": dict(index.census),
            "training_labels": dict(counts),
            "component_states": {key: sorted(value) for key, value in feature_states.items()},
            "manifest_sha256": index.manifest_sha256,
            "module_pruning": pruning,
            "source_label_values": {
                key: sorted(value) for key, value in source_label_values.items()
            },
            "observed_numeric_features": {
                key: sorted(value) for key, value in observed_features.items()
            },
        },
    )


def _selection_score(metrics):
    if "ic50" in metrics and "exact" in metrics["ic50"]:
        return metrics["ic50"]["exact"]["rmse"]
    assays = {key: value for key, value in metrics.items() if key in ENDPOINTS}
    metrics = assays or metrics
    count = sum(value["n"] for value in metrics.values())
    return sum(value["loss"] * value["n"] for value in metrics.values()) / count if count else None


def _accumulate_coverage(total, step):
    for name, evidence in step["coverage"].items():
        record = total.setdefault(
            name,
            {
                "gradient_steps": 0,
                "update_steps": 0,
                "gradient_and_update_steps": 0,
                "max_gradient_norm": 0.0,
            },
        )
        record["gradient_steps"] += evidence["gradient_norm"] > 0
        record["update_steps"] += evidence["update_norm"] > 0
        record["gradient_and_update_steps"] += evidence["gradient_and_update"]
        record["max_gradient_norm"] = max(record["max_gradient_norm"], evidence["gradient_norm"])


def fit_index(index, output_dir, *, device="cpu"):
    """Library training core; launchers must register experiments before calling."""
    config, training, scales, admission = preflight(index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "selected.pt").exists() or (output_dir / "history.json").exists():
        raise ValueError("refusing to overwrite a run; exact resume is not yet supported")
    _json(output_dir / "admission.json", admission)
    _json(
        output_dir / "resolved-config.json",
        {
            "model": asdict(config),
            "training": training,
            "scales": {key: asdict(value) for key, value in scales.items()},
        },
    )
    torch.manual_seed(training["seed"])
    rng = random.Random(training["seed"])
    identity_rng = random.Random(training["seed"] + 1)
    model = ModularPresto(config).to(device)
    inherited = None
    if index.manifest.get("pretrained"):
        reference = index.manifest["pretrained"]
        path = (index.manifest_path.parent / reference["path"]).resolve()
        if file_sha256(path) != reference["sha256"]:
            raise ValueError("pretrained checkpoint hash mismatch")
        inherited = initialize_from_checkpoint(model, path)
        for endpoint, prior_scale in inherited["scales"].items():
            if scales.get(endpoint) != prior_scale:
                raise ValueError("inherited target scalers require an explicit migration")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    base_indices = index.partitions.get("train:assay") or index.partitions.get("train:identity", [])
    replay_indices = (
        list(index.partitions.get("train:identity", []))
        if index.partitions.get("train:assay")
        else []
    )
    if not base_indices:
        raise ValueError("no training observations")
    replay_order, replay_position = [], 0
    validation = index.partitions.get("validation:assay", []) + index.partitions.get(
        "validation:identity", []
    )
    coverage, history, exposures = {}, [], Counter()
    steps, best, best_epoch, stale = 0, float("inf"), 0, 0
    start_time = time.monotonic()
    initial_metadata = {
        "manifest_sha256": index.manifest_sha256,
        "module_pruning": admission["module_pruning"],
        "training": training,
        "source_contract": index.manifest["sources"],
        "inherited": bool(inherited),
        "pretrained_contract": index.manifest.get("pretrained"),
        "inherited_parameters": inherited["inherited_parameters"] if inherited else [],
    }
    save_checkpoint(
        output_dir / "initial.pt", model, scales, optimizer=optimizer, metadata=initial_metadata
    )
    for epoch in range(1, training["max_epochs"] + 1):
        order = list(base_indices)
        rng.shuffle(order)
        epoch_losses = []
        epoch_exposures = Counter()
        processed = 0
        for start in range(0, len(order), training["batch_size"]):
            selected = order[start : start + training["batch_size"]]
            rows = [index[i] for i in selected]
            identity_rows = []
            if replay_indices:
                for _ in range(training["identity_batch_size"]):
                    if replay_position == len(replay_order):
                        replay_order = list(replay_indices)
                        identity_rng.shuffle(replay_order)
                        replay_position = 0
                    identity_rows.append(index[replay_order[replay_position]])
                    replay_position += 1
            step = train_step(
                model,
                optimizer,
                rows,
                scales,
                identity_rows=identity_rows,
                identity_weight=training["identity_weight"],
            )
            _accumulate_coverage(coverage, step)
            exposures.update(step["counts"])
            epoch_exposures.update(step["counts"])
            processed += len(selected)
            epoch_losses.append(step["loss"])
            steps += 1
            if training["max_steps"] is not None and steps >= training["max_steps"]:
                break
        validation_metrics, _ = evaluate(
            model, index, validation, scales, batch_size=training["batch_size"]
        )
        score = _selection_score(validation_metrics)
        if score is None:
            # A declared training-only diagnostic selects its last executed update.
            improved = True
        else:
            improved = score < best
        if improved:
            best = score if score is not None else best
            best_epoch, stale = epoch, 0
            save_checkpoint(
                output_dir / "selected.pt",
                model,
                scales,
                optimizer=optimizer,
                metadata={
                    **initial_metadata,
                    "epoch": epoch,
                    "steps": steps,
                    "coverage": coverage,
                    "selection_metric": score,
                },
            )
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "steps": steps,
                "training_loss": sum(epoch_losses) / len(epoch_losses),
                "validation": validation_metrics,
                "selected": improved,
                "complete_epoch": processed == len(order),
                "label_exposures": dict(epoch_exposures),
            }
        )
        _json(output_dir / "history.json", history)
        print(
            f"modular epoch={epoch} steps={steps} "
            f"loss={history[-1]['training_loss']:.6f} validation={score}",
            flush=True,
        )
        if (training["max_steps"] is not None and steps >= training["max_steps"]) or (
            epoch >= training["min_epochs"] and stale >= training["patience"]
        ):
            break
    save_checkpoint(
        output_dir / "last.pt",
        model,
        scales,
        optimizer=optimizer,
        metadata={**initial_metadata, "epoch": epoch, "steps": steps, "coverage": coverage},
    )
    selected, scales, payload = load_checkpoint(output_dir / "selected.pt", device=device)
    metrics, overall_losses = {}, {}
    for split in ("train", "validation", "test"):
        indices = index.partitions.get(f"{split}:assay", []) + index.partitions.get(
            f"{split}:identity", []
        )
        metrics[split], records = evaluate(
            selected, index, indices, scales, batch_size=training["batch_size"]
        )
        _jsonl(output_dir / f"{split}-predictions.jsonl", records)
        assay = [row["loss"] for row in records if "identity_task" not in row]
        identity = [row["loss"] for row in records if "identity_task" in row]
        assay_mean = sum(assay) / len(assay) if assay else None
        identity_mean = sum(identity) / len(identity) if identity else None
        overall_losses[split] = {
            "assay_observation_mean": assay_mean,
            "identity_label_mean": identity_mean,
            "combined": (
                assay_mean + training["identity_weight"] * (identity_mean or 0.0)
                if assay_mean is not None
                else identity_mean
            ),
        }
    # Qualification requires scientific gates beyond these software/training reports.
    selected_coverage = payload["metadata"]["coverage"]
    unsupported = [
        name
        for name, evidence in selected_coverage.items()
        if not evidence["gradient_and_update_steps"]
    ]
    features = feature_report(selected)
    missing_features = {
        name: value["exposed_without_update"]
        for name, value in features.items()
        if value["exposed_without_update"]
    }
    missing_auxiliaries = [
        f"peptide_source:{category}"
        for category, values in selected.source_label_support.items()
        if "peptide_source" in config.identity_tasks and not values
    ]
    coverage_gate = {
        "status": "failed" if unsupported or missing_features or missing_auxiliaries else "passed",
        "untrained_parameters": unsupported,
        "exposed_features_without_updates": missing_features,
        "policy": training["coverage_policy"],
        "exception_reason": training["coverage_exception_reason"],
        "untrained_auxiliary_outputs": missing_auxiliaries,
    }
    summary = {
        "metrics": metrics,
        "overall_losses": overall_losses,
        "elapsed_seconds": time.monotonic() - start_time,
        "steps": steps,
        "selected_epoch": best_epoch,
        "training_exposures": dict(exposures),
        "unique_training_labels": admission["training_labels"],
        "support": selected.support,
        "untrained_selected_parameters": unsupported,
        "coverage_gate": coverage_gate,
        "qualification": "not_qualified",
        "test_policy": training["test_policy"],
        "device": device,
        "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
    }
    _json(output_dir / "summary.json", summary)
    _json(
        output_dir / "coverage.json",
        {
            "all_steps": coverage,
            "selected": selected_coverage,
            "input_features": features,
        },
    )
    if coverage_gate["status"] == "failed" and training["coverage_policy"] == "required":
        raise ValueError("training coverage gate failed; inspect summary.json and coverage.json")
    return summary


def _metric_table(summary):
    lines = [
        "| Split | Endpoint | N | Loss | Exact RMSE | Spearman | Pearson | AUROC | AUPRC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, endpoints in summary["metrics"].items():
        for endpoint, metrics in endpoints.items():
            exact = metrics.get("exact", {})
            binary = exact.get("binding_at_500_nM", metrics)
            values = (
                metrics["n"],
                metrics["loss"],
                exact.get("rmse"),
                exact.get("spearman"),
                exact.get("pearson"),
                binary.get("auroc"),
                binary.get("auprc"),
            )
            cells = ["—" if value is None else f"{value:.6g}" for value in values]
            lines.append(f"| {split} | {endpoint} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_manifest(args):
    """Dispatch from both canonical CLIs; modular settings live in the frozen manifest."""
    allowed = {"model_kind", "observations_manifest", "modular_preflight", "run_dir", "device"}
    explicit = set(getattr(args, "_explicit_cli_dests", ()) or ())
    if explicit - allowed - {"func", "command", "train_command"}:
        raise ValueError(
            "modular training settings belong in the manifest, not legacy CLI flags: "
            + ", ".join(sorted(explicit - allowed))
        )
    if not getattr(args, "observations_manifest", None):
        raise ValueError("modular model requires --observations-manifest")
    index = ObservationIndex(args.observations_manifest)
    _, _, _, admission = preflight(index)
    if getattr(args, "modular_preflight", False):
        print(json.dumps(admission, indent=2, sort_keys=True))
        return admission
    from ..scripts import experiment_registry as registry

    output = (
        Path(args.run_dir).resolve()
        if getattr(args, "run_dir", None)
        else registry.default_experiment_dir(slug="modular-model-fit", agent_label="codex")
    )
    root = registry.EXPERIMENTS_ROOT.resolve()
    if output.parent != root or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}_\d{4}_[a-z0-9_]+_[a-z0-9-]+", output.name
    ):
        raise ValueError("modular fits must use a timestamped top-level experiments directory")
    if output.exists():
        raise ValueError("experiment directory already exists; choose a fresh registered run")
    output = registry.initialize_experiment_dir(
        out_dir=str(output),
        slug="modular-model-fit",
        title="Modular directly supervised fit",
        source_script="training/modular_run.py",
        agent_label="codex",
        metadata={
            "dataset_contract": index.manifest["sources"],
            "training": index.manifest.get("training", {}),
            "tested": ["one declared modular source/subset contract"],
        },
    )
    # Freeze an executable manifest with absolute input references and exact hashes.
    # A replay creates a fresh experiment directory rather than overwriting evidence.
    frozen = json.loads(json.dumps(index.manifest))
    for source, source_path in zip(frozen["sources"], index.source_paths, strict=True):
        source["path"] = str(source_path)
    if frozen.get("pretrained"):
        frozen["pretrained"]["path"] = str(
            (index.manifest_path.parent / frozen["pretrained"]["path"]).resolve()
        )
    frozen_path = output / "reproduce" / "source-manifest.json"
    _json(frozen_path, frozen)
    _json(
        output / "reproduce" / "manifest-origin.json",
        {
            "original_path": str(index.manifest_path),
            "original_sha256": index.manifest_sha256,
            "frozen_sha256": file_sha256(frozen_path),
            "change": (
                "make source and pretrained paths absolute; model/data/training contracts unchanged"
            ),
        },
    )
    index = ObservationIndex(frozen_path)
    registry.write_reproducibility_bundle(
        out_dir=output,
        source_script="training/modular_run.py",
        argv=[
            "-m",
            "presto.cli.main",
            "train",
            "unified",
            "--model-kind",
            "modular",
            "--observations-manifest",
            str(frozen_path),
            "--device",
            getattr(args, "device", None) or "cpu",
        ],
    )
    source_root = Path(__file__).resolve().parent.parent
    for path in (
        "data/modular.py",
        "data/modular_context.py",
        "data/modular_sources.py",
        "data/modular_lineage.py",
        "data/modular_preparation.py",
        "data/modular_auxiliary.py",
        "models/modular.py",
        "models/modular_support.py",
        "models/encoders.py",
        "training/modular.py",
        "training/modular_run.py",
        "training/modular_evaluation.py",
        "cli/training_sources.py",
        "scripts/train_iedb.py",
    ):
        target = output / "reproduce" / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / path, target)
    device = getattr(args, "device", None) or "cpu"
    try:
        summary = fit_index(index, output, device=device)
    except Exception as error:
        _json(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise
    narrative = (
        "# Modular directly supervised fit\n\n"
        f"Agent: Codex. Python: `{sys.version.split()[0]}`. Device: `{device}`.\n\n"
        "## Frozen contract\n\n"
        "See `reproduce/launch.json`, `reproduce/source-manifest.json`, `resolved-config.json` "
        "and `admission.json` for source hashes, splits, units, admitted labels "
        "and training settings. "
        "One exact/censored loss per assay observation; one binary loss per observation bag; "
        "identity CE has separately declared replay. "
        "No generated labels are added by the trainer.\n\n"
        "## Results and decision\n\n"
        "`summary.json` contains train/validation/test metrics, endpoint support, "
        "runtime and coverage. "
        "All three prediction dumps retain the targets needed to recompute metrics. "
        "This run does not automatically qualify a biological endpoint. Compare the prespecified "
        "baseline, memorization and noninferiority gates before promotion.\n\n"
        f"Test-use policy: `{summary['test_policy']}`. "
        f"No-test reason: {index.manifest.get('training', {}).get('no_test_reason')}.\n\n"
        "Selected/initial/last checkpoints and optimizer states are local raw artifacts "
        "in this directory. "
        "`coverage.json` separates genuine gradients from parameter movement.\n"
    )
    (output / "README.md").write_text(narrative + "\n" + _metric_table(summary))
    with registry.REGISTRY_MD.open("a") as stream:
        stream.write(
            f"\n#### Closure: {output.name}\n\n"
            f"Agent: Codex. Results: [{output.name}/summary.json]({output.name}/summary.json). "
            f"Selected epoch {summary['selected_epoch']}; {summary['steps']} updates; "
            f"{summary['elapsed_seconds']:.2f} seconds on {device}. "
            "One tested condition; no comparative winner or biological qualification. "
            f"Test policy `{summary['test_policy']}`; reason: "
            f"{index.manifest.get('training', {}).get('no_test_reason')}.\n\n"
            + _metric_table(summary)
        )
    return summary
