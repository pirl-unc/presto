"""Manifest-driven implementation behind the existing unified training command."""

from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

import torch

from ..data.modular import (
    endpoint_spec,
    is_endpoint,
    TargetScaleFitter,
    identity_head,
    supervised_targets,
)
from ..data.modular_quantitative import quantity_domain
from ..data.modular_context import resolve_cell_context
from ..data.modular_sources import ObservationIndex, file_sha256
from ..models.modular import ModularConfig, ModularPresto
from ..models.modular_support import feature_report, parameter_report
from ..models.modular_dependencies import freeze_dependencies, frozen_output
from .modular import (
    identity_targets,
    initialize_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
    train_step,
)
from .modular_controls import (
    CONTROL_DEFAULTS,
    EMA,
    cell_blocks,
    cell_key,
    control,
    learning_rate,
    segment_for,
    spearman_score,
    study_of,
    validate_controls,
)
from .modular_evaluation import evaluate
from .modular_resume import (
    SOURCE_FILES,
    atomic_bytes,
    atomic_json,
    cached_evaluation,
    check_contract,
    complete_run,
    completed_result,
    digest,
    execution_contract,
    host_drift,
    host_profile,
    optimizer_names,
    restore_rng,
    run_lock,
    save_resume,
)

from .modular_sampling import (
    epoch_order,
    index_stratum,
    validate_balance_by,
    validate_stratum_weights,
    weighted_quotas,
)


DEFAULT_TRAINING = {
    "balance_by": [],
    "stratum_weights": [],
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
    "checkpoint_every_steps": 100,
    "validation_every_steps": None,
    "selection_endpoint": None,
    "selection_reduction": "observation_mean",
    "train_evaluation_max_rows": None,
}


def _json(path, value):
    atomic_json(path, value)


def _jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def initialized_model(config, index, *, device="cpu"):
    model = ModularPresto(config).to(device)
    inherited = None
    reference = index.manifest.get("pretrained")
    if reference:
        if set(reference) - {"path", "sha256", "allow_task_removal", "freeze"}:
            raise ValueError("unknown pretrained transfer option")
        if type(reference.get("allow_task_removal", False)) is not bool:
            raise ValueError("allow_task_removal must be boolean")
        targets = reference.get("freeze", [])
        if not isinstance(targets, list) or any(not isinstance(value, str) for value in targets):
            raise ValueError("pretrained freeze must be a list of dependency targets")
        path = (index.manifest_path.parent / reference["path"]).resolve()
        if file_sha256(path) != reference["sha256"]:
            raise ValueError("pretrained checkpoint hash mismatch")
        inherited = initialize_from_checkpoint(
            model, path, allow_task_removal=reference.get("allow_task_removal", False)
        )
        freeze_dependencies(model, targets)
    return model, inherited


def check_training_strata(index, training, indices):
    """Every sampling regime (base weights and each schedule segment) must be realizable."""
    regimes = [training["stratum_weights"]] + [
        segment["stratum_weights"] for segment in control(training, "schedule")
    ]
    sizes = None
    for declared in regimes:
        weights = validate_stratum_weights(training["balance_by"], declared)
        if weights is None:
            continue
        if sizes is None:
            sizes = Counter(index_stratum(index, i, training["balance_by"]) for i in indices)
        validate_stratum_weights(training["balance_by"], declared, sizes)
        weighted_quotas(sizes, weights)


def preflight(index: ObservationIndex):
    config = ModularConfig.from_dict(index.manifest["model"])
    supplied = index.manifest.get("training", {})
    if set(supplied) - DEFAULT_TRAINING.keys() - CONTROL_DEFAULTS.keys():
        raise ValueError("unknown modular training option")
    training = {**DEFAULT_TRAINING, **supplied}
    validate_balance_by(training["balance_by"])
    validate_stratum_weights(training["balance_by"], training["stratum_weights"])
    for key in (
        "batch_size",
        "max_epochs",
        "patience",
        "min_epochs",
        "identity_batch_size",
        "checkpoint_every_steps",
    ):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    interval = training["validation_every_steps"]
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("validation_every_steps must be a positive integer or null")
    if training["max_steps"] is not None and (
        type(training["max_steps"]) is not int or training["max_steps"] <= 0
    ):
        raise ValueError("max_steps must be a positive integer")
    limit = training["train_evaluation_max_rows"]
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("train_evaluation_max_rows must be a positive integer or null")
    if training["selection_reduction"] not in ("observation_mean", "endpoint_mean"):
        raise ValueError("unsupported selection_reduction")
    if (
        training["selection_endpoint"] is not None
        and training["selection_reduction"] != "observation_mean"
    ):
        raise ValueError(
            "selection_endpoint and selection_reduction cannot both override selection"
        )
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
    validate_controls(training, config, training["balance_by"])
    # Freezing removes rows from the trained strata, and the frozen set needs the
    # built model; `_fit_index` repeats this check on the rows it actually trains.
    if training["balance_by"] and not (index.manifest.get("pretrained") or {}).get("freeze"):
        check_training_strata(
            index,
            training,
            index.partitions.get("train:assay", []) or index.partitions.get("train:identity", []),
        )
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
    validator, inherited = initialized_model(config, index)
    primary = training["selection_endpoint"]
    if primary is not None:
        if not isinstance(primary, str) or primary not in config.endpoints:
            raise ValueError("selection_endpoint must name an active assay endpoint or be null")
        if frozen_output(validator, primary):
            raise ValueError("selection_endpoint cannot name a frozen outcome")
    inherited_scales = inherited["scales"] if inherited else {}
    scale_fitter = TargetScaleFitter()
    trainable_targets = {
        target
        for target in (*config.endpoints, *config.identity_tasks)
        if not frozen_output(validator, target)
    }
    validation_primary = validation_trainable = False
    spearman_enabled = control(training, "selection_metric") == "within_study_spearman"
    spearman_rows = defaultdict(int)
    counts, label_values, feature_states = Counter(), defaultdict(set), defaultdict(set)
    source_label_values = defaultdict(set)
    observed_features = {"acquisition": set(), "observation": set(), "preparation": set()}
    preparation_values = defaultdict(set)
    cell_types, cytokine_states, cytokine_numeric = set(), defaultdict(set), defaultdict(set)
    assertion_states = {"pathway": defaultdict(set), "intervention": defaultdict(set)}
    ordinal_levels = defaultdict(set)
    assay_classes = set()
    # Label and capability inspection is sequence-only; no training or random model scores.
    assay_track_counts = Counter()
    for row in index:
        if row.endpoint not in inherited_scales:
            scale_fitter.add(row)
        if row.split == "validation":
            validation_primary |= row.endpoint == primary
            validation_trainable |= bool(supervised_targets(row) & trainable_targets)
            if (
                spearman_enabled
                and row.endpoint in control(training, "selection_spearman_endpoints")
                and row.qualifier == "="
                and row.value is not None
                and study_of(row.provenance) is not None
            ):
                spearman_rows[(row.endpoint, str(study_of(row.provenance)))] += 1
        if row.endpoint == "identity":
            targets = identity_targets(row, validator)
            for name in targets:
                if row.split == "train":
                    counts[identity_head(name)] += 1
                    label_values[identity_head(name)].add(row.identity_labels[name])
                    if name.startswith("peptide_source:"):
                        source_label_values[name.split(":", 1)[1]].add(row.identity_labels[name])
        else:
            if row.endpoint not in config.endpoints:
                raise ValueError(f"admitted row targets an unallocated endpoint: {row.endpoint}")
            selected_track = validator.assay_selection(row)
            if selected_track is not None and row.split == "train" and row.value is not None:
                assay_track_counts[selected_track.key] += 1
            spec = endpoint_spec(row.endpoint)
            route = spec.route
            if (
                spec.quantitative
                and row.quantity_contract != config.quantity_domains[quantity_domain(row.endpoint)]
            ):
                raise ValueError("quantity measurement contract differs from configured domain")
            if route in ("cellular", "tcr_cellular"):
                cell = resolve_cell_context(
                    row.context,
                    config.cell_types,
                    config.cytokines,
                    config.cytokine_dose_units,
                    config.cytokine_duration_features,
                    processing_pathways=config.processing_pathways,
                    interventions=config.interventions,
                    ordinal_scales=config.ordinal_scales,
                )
                if row.split == "train":
                    for name, value in cell.ordinal_values.items():
                        ordinal_levels[name].add(value.level or "unknown")
                    for family, assertions in (
                        ("pathway", cell.processing_pathways),
                        ("intervention", cell.interventions),
                    ):
                        for name, assertion in assertions.items():
                            assertion_states[family][name].add(assertion.state)
                    if cell.cell_type.value is not None:
                        cell_types.add(cell.cell_type.value)
                    for name, exposure in cell.cytokines.items():
                        cytokine_states[name].add(exposure.state)
                        if exposure.dose is not None:
                            cytokine_numeric[f"{name}:dose"].add(exposure.dose)
                        if exposure.duration_hours is not None:
                            cytokine_numeric[f"{name}:duration_hours"].add(exposure.duration_hours)
            if "ms_response" in spec.parts:
                validator._covariates(
                    config.acquisition_features,
                    row.context.acquisition,
                    row.context.acquisition_units,
                    config.acquisition_units,
                )
                if row.split == "train":
                    observed_features["acquisition"].update(row.context.acquisition)
            if spec.observation_covariates:
                validator._covariates(
                    config.observation_features,
                    row.context.observation,
                    row.context.observation_units,
                    config.observation_units,
                )
                if row.split == "train":
                    observed_features["observation"].update(row.context.observation)
            if route in ("bulk", "digest"):
                if row.context.enzyme is not None and row.context.enzyme not in config.enzymes:
                    raise ValueError("enzyme outside checkpoint vocabulary")
                validator._covariates(
                    config.preparation_features,
                    row.context.preparation,
                    row.context.preparation_units,
                    config.preparation_units,
                )
                if row.split == "train":
                    observed_features["preparation"].update(row.context.preparation)
                    for name, value in row.context.preparation.items():
                        preparation_values[name].add(value)
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
            if row.split == "train":
                counts[row.endpoint] += 1
                if endpoint_spec(row.endpoint).route in ("cellular", "tcr_cellular"):
                    for name in config.components:
                        component = row.context.component_overrides.get(
                            name
                        ) or row.context.components.get(name)
                        feature_states[name].add(component.state if component else "unknown")
    if spearman_enabled:
        for name in control(training, "selection_spearman_endpoints"):
            if frozen_output(validator, name):
                raise ValueError("selection_spearman_endpoints cannot name a frozen outcome")
        minimum = control(training, "selection_min_cell_rows")
        if not any(count >= minimum for count in spearman_rows.values()):
            raise ValueError("no validation study cell meets selection_min_cell_rows")
    if primary is not None and not validation_primary:
        raise ValueError("selection_endpoint has no validation outcomes")
    scales = {**inherited_scales, **scale_fitter.finish()}
    for track in config.assay_tracks:
        if not assay_track_counts[track.key] and not frozen_output(validator, track.endpoint):
            raise ValueError(f"declared assay track has no training observations: {track.key}")
    for endpoint in config.endpoints:
        if not counts[endpoint] and not frozen_output(validator, endpoint):
            raise ValueError(f"allocated endpoint has no training labels: {endpoint}")
    for task in config.identity_tasks:
        if counts[task] == 0 and not frozen_output(validator, task):
            raise ValueError(f"identity task has no labels or inherited weights: {task}")
        if counts[task] and len(label_values[task]) < 2 and not frozen_output(validator, task):
            raise ValueError(f"identity task needs varied training labels: {task}")
    if "peptide_source" in config.identity_tasks and not frozen_output(validator, "peptide_source"):
        for category in config.source_categories:
            if source_label_values[category] != {"present", "absent"}:
                raise ValueError(
                    "peptide source category needs positive and negative reference labels: "
                    f"{category}"
                )
    if any(not frozen_output(validator, name) for name in config.endpoints) and any(
        not frozen_output(validator, task) for task in config.identity_tasks
    ):
        if not index.partitions.get("train:identity") or training["identity_weight"] <= 0:
            raise ValueError("fine-tuning the shared encoder requires explicit identity replay")
    for name, states in feature_states.items():
        if (
            len(states - {"unknown", "reference"}) < 2
            and "context_fusion.0.weight" not in validator.frozen_parameters
        ):
            raise ValueError(f"component effect lacks contrasted training states: {name}")
    missing_cells = set(config.cell_types) - cell_types
    if missing_cells and "context_fusion.0.weight" not in validator.frozen_parameters:
        raise ValueError(f"cell types have no observed training labels: {sorted(missing_cells)}")
    for name, scale in config.ordinal_scales.items():
        missing = set(scale.levels) - ordinal_levels[name]
        if missing and "context_fusion.0.weight" not in validator.frozen_parameters:
            raise ValueError(f"ordinal levels lack training support: {name}: {sorted(missing)}")
    for name in config.cytokines:
        if (
            not {"present", "absent"} <= cytokine_states[name]
            and "context_fusion.0.weight" not in validator.frozen_parameters
        ):
            raise ValueError(f"cytokine effect lacks present/absent training contrast: {name}")
    for family, configured in (
        ("pathway", config.processing_pathways),
        ("intervention", config.interventions),
    ):
        for name in configured:
            if (
                not {"present", "absent"} <= assertion_states[family][name]
                and "context_fusion.0.weight" not in validator.frozen_parameters
            ):
                raise ValueError(f"{family} lacks present/absent training contrast: {name}")
    for suffix, names in (
        ("dose", config.cytokine_dose_units),
        ("duration_hours", config.cytokine_duration_features),
    ):
        for name in names:
            if (
                len(cytokine_numeric[f"{name}:{suffix}"]) < 2
                and "context_fusion.0.weight" not in validator.frozen_parameters
            ):
                raise ValueError(
                    f"cytokine numeric feature lacks varied training values: {name}:{suffix}"
                )
    for family, configured in (
        ("acquisition", config.acquisition_features),
        ("observation", config.observation_features),
        ("preparation", config.preparation_features),
    ):
        dependencies = {
            "acquisition": ["ms_fusion.0.weight"],
            "preparation": ["digest_fusion.0.weight"],
            "observation": [
                name for name in validator.feature_layout if name.startswith("offsets.")
            ],
        }
        missing = set(configured) - observed_features[family]
        if all(name in validator.frozen_parameters for name in dependencies[family]):
            missing = set()
        if missing:
            raise ValueError(
                f"{family} features have no observed training values: {sorted(missing)}"
            )
    for name in config.preparation_features:
        if (
            len(preparation_values[name]) < 2
            and "digest_fusion.0.weight" not in validator.frozen_parameters
        ):
            raise ValueError(f"preparation feature lacks varied training values: {name}")
    pruning = []
    if validator.has_pmhc and "pmhc_fusion.0.weight" not in validator.frozen_parameters:
        if set(config.mhc_classes) - assay_classes:
            raise ValueError("declared MHC branch has no admitted assay training support")
    if training["test_policy"] == "evaluate_selected_once" and not validation_trainable:
        raise ValueError("validation has no outcomes for a trainable task")
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
            "cell_types": sorted(cell_types),
            "ordinal_levels": {key: sorted(value) for key, value in ordinal_levels.items()},
            "cytokine_states": {key: sorted(value) for key, value in cytokine_states.items()},
            "cytokine_numeric": {key: sorted(value) for key, value in cytokine_numeric.items()},
            "context_assertion_states": {
                family: {name: sorted(states) for name, states in values.items()}
                for family, values in assertion_states.items()
            },
            "observed_numeric_features": {
                key: sorted(value) for key, value in observed_features.items()
            },
            "frozen_targets": list(validator.frozen_targets),
            "frozen_parameters": sorted(validator.frozen_parameters),
            "inherited_scalers": sorted(inherited_scales),
            **(
                {
                    "selection_cells": [
                        {"endpoint": key[0], "study": key[1], "n": count}
                        for key, count in sorted(spearman_rows.items())
                        if count >= control(training, "selection_min_cell_rows")
                    ],
                    "selection_cells_skipped": [
                        {"endpoint": key[0], "study": key[1], "n": count}
                        for key, count in sorted(spearman_rows.items())
                        if count < control(training, "selection_min_cell_rows")
                    ],
                }
                if spearman_enabled
                else {}
            ),
        },
    )


def _selection_score(metrics, model=None, primary_endpoint=None, reduction="observation_mean"):
    if model is not None:
        metrics = {
            key: value
            for key, value in metrics.items()
            if not frozen_output(model, key if is_endpoint(key) else identity_head(key))
        }
    if primary_endpoint is not None:
        if primary_endpoint not in metrics:
            raise ValueError("selection_endpoint has no evaluated validation outcomes")
        primary = metrics[primary_endpoint]
        return primary["exact"]["rmse"] if "exact" in primary else primary["loss"]
    assays = {key: value for key, value in metrics.items() if is_endpoint(key)}
    metrics = assays or metrics
    if reduction == "endpoint_mean":
        return sum(value["loss"] for value in metrics.values()) / len(metrics) if metrics else None
    if reduction != "observation_mean":
        raise ValueError("unsupported selection reduction")
    count = sum(value["n"] for value in metrics.values())
    return sum(value["loss"] * value["n"] for value in metrics.values()) / count if count else None


def evaluation_indices(index, split, training):
    """Optionally sample TRAIN diagnostics; every held-out outcome stays included."""
    indices = index.partitions.get(f"{split}:assay", []) + index.partitions.get(
        f"{split}:identity", []
    )
    available = len(indices)
    limit = training.get("train_evaluation_max_rows") if split == "train" else None
    if limit is not None and available > limit:
        indices = sorted(random.Random(training["seed"] + 0xA114).sample(indices, limit))
    return indices, {
        "available_rows": available,
        "evaluated_rows": len(indices),
        "policy": "seeded_uniform_train_diagnostic" if len(indices) < available else "all_rows",
    }


def _accumulate_coverage(total, step):
    for name, evidence in step["coverage"].items():
        record = total.setdefault(
            name,
            {
                "gradient_steps": 0,
                "update_steps": 0,
                "gradient_and_update_steps": 0,
                "max_gradient_norm": 0.0,
                "frozen_steps": 0,
            },
        )
        record["gradient_steps"] += evidence["gradient_norm"] > 0
        record["frozen_steps"] += evidence.get("frozen", False)
        record["update_steps"] += evidence["update_norm"] > 0
        record["gradient_and_update_steps"] += evidence["gradient_and_update"]
        record["max_gradient_norm"] = max(record["max_gradient_norm"], evidence["gradient_norm"])


def fit_index(index, output_dir, *, device="cpu", resume=False):
    """Library training core; launchers must register experiments before calling."""
    with run_lock(output_dir):
        return _fit_index(index, output_dir, device=device, resume=resume)


def trainable_indices(index, model):
    """Resolve fixed model dependencies once, then filter admitted task metadata."""
    targets = {
        target
        for target in (*model.config.endpoints, *model.config.identity_tasks)
        if not frozen_output(model, target)
    }
    return tuple(
        [
            i
            for i in index.partitions.get(f"train:{family}", [])
            if index.metadata(i).targets & targets
        ]
        for family in ("assay", "identity")
    )


def _record_host(output_dir, host, drift):
    """Append where each segment of a run executed; resume stays allowed across hosts."""
    entry = {"time_ns": time.time_ns(), "drift": drift, "host": host}
    with (output_dir / "host-history.jsonl").open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")


def _fit_index(index, output_dir, *, device, resume):
    config, training, scales, admission = preflight(index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = execution_contract(index, config, training, scales, device)
    host = host_profile(device)
    snapshot_path = output_dir / "resume.pt"
    saved = None
    if resume:
        if not snapshot_path.exists():
            raise ValueError("resume requires an existing durable resume.pt snapshot")
        saved_payload = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        saved = check_contract(saved_payload, contract)
        drift = host_drift(saved.get("host"), host)
        if drift:
            logging.warning("resuming on a different host; changed: %s", ", ".join(drift))
        _record_host(output_dir, host, drift)
        closed = completed_result(output_dir, contract)
        if closed is not None:
            _check_coverage_result(closed)
            return closed
        for name, expected in saved["state"]["initial_artifacts"].items():
            if file_sha256(output_dir / name) != expected:
                raise ValueError(f"initial run artifact changed: {name}")
    else:
        if any(
            (output_dir / name).exists()
            for name in (
                "initial.pt",
                "resume.pt",
                "selected.pt",
                "history.json",
                "completed.json",
            )
        ):
            raise ValueError("refusing to overwrite a run; use explicit resume")
        _json(output_dir / "admission.json", admission)
        _record_host(output_dir, host, [])
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
    if saved is None:
        model, inherited = initialized_model(config, index, device=device)
    else:
        model, restored_scales, saved_payload = load_checkpoint(snapshot_path, device=device)
        if asdict(model.config) != asdict(config) or restored_scales != scales:
            raise ValueError("resume model/scalers differ from the frozen execution contract")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError(
            "no trainable parameters; use checkpoint inference for a fully frozen model"
        )
    optimizer = torch.optim.AdamW(
        trainable, lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    if saved is not None:
        if saved["optimizer_names"] != optimizer_names(model, optimizer):
            raise ValueError("resume optimizer parameter order or membership changed")
        optimizer.load_state_dict(saved_payload["optimizer"])
    trainable_assays, trainable_identity = trainable_indices(index, model)
    base_indices = trainable_assays or trainable_identity
    replay_indices = trainable_identity if trainable_assays else []
    if not base_indices:
        raise ValueError("no observations with a trainable supervised loss")
    if training["balance_by"]:
        check_training_strata(index, training, base_indices)
    validation = index.partitions.get("validation:assay", []) + index.partitions.get(
        "validation:identity", []
    )
    start_time = time.monotonic()
    if saved is None:
        initial_metadata = {
            "manifest_sha256": index.manifest_sha256,
            "module_pruning": admission["module_pruning"],
            "training": training,
            "source_contract": index.manifest["sources"],
            "inherited": bool(inherited),
            "pretrained_contract": index.manifest.get("pretrained"),
            "inherited_parameters": inherited["inherited_parameters"] if inherited else [],
            "frozen_targets": list(model.frozen_targets),
            "frozen_parameters": sorted(model.frozen_parameters),
        }
        save_checkpoint(
            output_dir / "initial.pt", model, scales, optimizer=optimizer, metadata=initial_metadata
        )
        state = {
            "phase": "training",
            "epoch": 1,
            "last_epoch": 0,
            "order": [],
            "position": 0,
            "replay_order": [],
            "replay_position": 0,
            "epoch_losses": [],
            "epoch_exposures": {},
            "coverage": {},
            "history": [],
            "exposures": {},
            "steps": 0,
            "best": float("inf"),
            "best_epoch": 0,
            "stale": 0,
            "elapsed_seconds": 0,
            "selected_bytes": None,
            "initial_artifacts": {
                name: file_sha256(output_dir / name)
                for name in ("initial.pt", "admission.json", "resolved-config.json")
            },
        }
    else:
        state = saved["state"]
        initial_metadata = {
            key: value for key, value in saved_payload["metadata"].items() if key != "resume"
        }
        # Construction/loading can consume RNG. Restore only after both are complete.
        restore_rng(saved["rng"], rng, identity_rng)
        _json(output_dir / "history.json", state["history"])
    elapsed_before = state["elapsed_seconds"]
    schedule = control(training, "schedule")
    lr_schedule = control(training, "lr_schedule")
    batching = control(training, "cell_batching")
    ranking = control(training, "within_group_ranking_loss")
    ema = (
        EMA(model, control(training, "ema_decay"), saved=state.get("ema"))
        if control(training, "ema_decay") is not None
        else None
    )
    # Deterministic row -> comparison-cell map; reads training rows only, never RNG.
    cell_keys = {i: cell_key(index[i]) for i in base_indices} if batching else {}
    extra_step_options = {}
    if control(training, "loss_normalization") != "observation_mean":
        extra_step_options["normalization"] = control(training, "loss_normalization")
        extra_step_options["endpoint_weights"] = control(training, "endpoint_loss_weights")
    if ranking is not None:
        extra_step_options["ranking"] = ranking
    if control(training, "grad_clip_norm") is not None:
        extra_step_options["grad_clip_norm"] = control(training, "grad_clip_norm")

    def snapshot():
        state["elapsed_seconds"] = elapsed_before + time.monotonic() - start_time
        if ema is not None:
            state["ema"] = ema.state()
        save_resume(
            snapshot_path,
            model,
            scales,
            optimizer,
            state,
            contract,
            rng,
            identity_rng,
            initial_metadata,
            host,
        )

    if saved is None:
        snapshot()
    coverage, history = state["coverage"], state["history"]
    exposures = Counter(state["exposures"])
    validation_interval = training["validation_every_steps"]
    while state["phase"] == "training":
        epoch = state["epoch"]
        if not state["order"]:
            weights = training["stratum_weights"]
            if schedule:
                state["segment"] = segment_for(schedule, state["steps"])
                if state["segment"] < len(schedule):
                    weights = schedule[state["segment"]]["stratum_weights"]
            state["order"] = epoch_order(index, base_indices, training["balance_by"], rng, weights)
            if batching:
                state["order"] = cell_blocks(
                    state["order"], cell_keys, batching["rows_per_cell"], rng
                )
        order = state["order"]
        epoch_exposures = Counter(state["epoch_exposures"])
        while (
            not state.get("validation_pending", False)
            and state["position"] < len(order)
            and (training["max_steps"] is None or state["steps"] < training["max_steps"])
            and segment_for(schedule, state["steps"]) == state.get("segment", 0)
        ):
            selected = order[state["position"] : state["position"] + training["batch_size"]]
            rows = [index[i] for i in selected]
            identity_rows = []
            if replay_indices:
                for _ in range(training["identity_batch_size"]):
                    if state["replay_position"] == len(state["replay_order"]):
                        state["replay_order"] = list(replay_indices)
                        identity_rng.shuffle(state["replay_order"])
                        state["replay_position"] = 0
                    identity_rows.append(index[state["replay_order"][state["replay_position"]]])
                    state["replay_position"] += 1
            if lr_schedule is not None:
                rate = learning_rate(
                    training["learning_rate"], lr_schedule, state["steps"], training["max_steps"]
                )
                for group in optimizer.param_groups:
                    group["lr"] = rate
            step = train_step(
                model,
                optimizer,
                rows,
                scales,
                identity_rows=identity_rows,
                identity_weight=training["identity_weight"],
                **extra_step_options,
            )
            if ema is not None:
                ema.update(model)
            _accumulate_coverage(coverage, step)
            exposures.update(step["counts"])
            epoch_exposures.update(step["counts"])
            state["exposures"] = dict(exposures)
            state["epoch_exposures"] = dict(epoch_exposures)
            state["position"] += len(selected)
            state["epoch_losses"].append(step["loss"])
            state["steps"] += 1
            if validation_interval and state["steps"] % validation_interval == 0:
                state["validation_pending"] = True
            if state["steps"] % training["checkpoint_every_steps"] == 0:
                snapshot()
            if validation_interval and state["steps"] % validation_interval == 0:
                break
        if (
            schedule
            and not state.get("validation_pending", False)
            and state["position"] < len(order)
            and (training["max_steps"] is None or state["steps"] < training["max_steps"])
            and segment_for(schedule, state["steps"]) != state["segment"]
        ):
            # A schedule boundary discards the unconsumed order; it is rebuilt with the
            # new segment's weights from the restored sampler RNG (not a validation event).
            state.update(order=[], position=0)
            continue
        complete_epoch = state["position"] == len(order)
        budget_done = (complete_epoch and epoch == training["max_epochs"]) or (
            training["max_steps"] is not None and state["steps"] >= training["max_steps"]
        )
        if validation_interval and not budget_done and state["steps"] % validation_interval:
            # Reaching a data epoch is not a validation event in step-cadence mode.
            state.update(epoch=epoch + 1, order=[], position=0)
            continue
        with ema.swap(model) if ema is not None else contextlib.nullcontext():
            validation_metrics, validation_records = evaluate(
                model, index, validation, scales, batch_size=training["batch_size"]
            )
        selection_report = None
        if control(training, "selection_metric") == "within_study_spearman":
            score, selection_report = spearman_score(
                validation_records,
                control(training, "selection_spearman_endpoints"),
                control(training, "selection_min_cell_rows"),
            )
        else:
            score = _selection_score(
                validation_metrics,
                model,
                training["selection_endpoint"],
                training["selection_reduction"],
            )
        if score is None:
            # A declared training-only diagnostic selects its last executed update.
            improved = True
        else:
            improved = score < state["best"]
        if improved:
            state["best"] = score if score is not None else state["best"]
            state["best_epoch"], state["stale"] = epoch, 0
            with ema.swap(model) if ema is not None else contextlib.nullcontext():
                save_checkpoint(
                    output_dir / "selected.pt",
                    model,
                    scales,
                    optimizer=optimizer,
                    metadata={
                        **initial_metadata,
                        "epoch": epoch,
                        "steps": state["steps"],
                        "coverage": coverage,
                        "selection_metric": score,
                        **({"weights": "ema"} if ema is not None else {}),
                        **({"selection_cells": selection_report} if selection_report else {}),
                    },
                )
            state["selected_bytes"] = (output_dir / "selected.pt").read_bytes()
        else:
            state["stale"] += 1
        history.append(
            {
                "epoch": epoch,
                "steps": state["steps"],
                "training_loss": sum(state["epoch_losses"]) / len(state["epoch_losses"]),
                "validation": validation_metrics,
                "selected": improved,
                "complete_epoch": complete_epoch,
                "label_exposures": dict(epoch_exposures),
                **(
                    {"learning_rate": optimizer.param_groups[0]["lr"]}
                    if lr_schedule is not None
                    else {}
                ),
                **({"selection": selection_report} if selection_report else {}),
            }
        )
        _json(output_dir / "history.json", history)
        print(
            f"modular epoch={epoch} steps={state['steps']} "
            f"loss={history[-1]['training_loss']:.6f} validation={score}",
            flush=True,
        )
        state["last_epoch"] = epoch
        if validation_interval:
            state["validation_pending"] = False
        if budget_done or (
            epoch >= training["min_epochs"] and state["stale"] >= training["patience"]
        ):
            state["phase"] = "evaluation"
        else:
            state.update(epoch_losses=[], epoch_exposures={})
            if complete_epoch:
                state.update(epoch=epoch + 1, order=[], position=0)
        snapshot()
    with ema.swap(model) if ema is not None else contextlib.nullcontext():
        save_checkpoint(
            output_dir / "last.pt",
            model,
            scales,
            optimizer=optimizer,
            metadata={
                **initial_metadata,
                "epoch": state["last_epoch"],
                "steps": state["steps"],
                "coverage": coverage,
                **({"weights": "ema"} if ema is not None else {}),
            },
        )
    # Recover the selection committed with this snapshot, even if a crash left a newer file.
    if state["selected_bytes"] is None:
        raise ValueError("evaluation phase has no selected checkpoint")
    atomic_bytes(output_dir / "selected.pt", state["selected_bytes"])
    selected, scales, payload = load_checkpoint(output_dir / "selected.pt", device=device)
    metrics, overall_losses, evaluation_populations = {}, {}, {}
    for split in ("train", "validation", "test"):
        indices, evaluation_populations[split] = evaluation_indices(index, split, training)
        metrics[split], records = cached_evaluation(
            output_dir,
            split,
            {
                "contract_sha256": digest(contract),
                "selected_sha256": file_sha256(output_dir / "selected.pt"),
                "split": split,
            },
            lambda indices=indices: evaluate(
                selected, index, indices, scales, batch_size=training["batch_size"]
            ),
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
        if not evidence["gradient_and_update_steps"] and name not in selected.frozen_parameters
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
    assay_support = selected.assay_capabilities()
    missing_assays = [
        key for key, value in assay_support.items() if value["status"] == "random/untrained"
    ]
    coverage_gate = {
        "status": "failed"
        if unsupported or missing_features or missing_auxiliaries or missing_assays
        else "passed",
        "untrained_assay_tracks": missing_assays,
        "untrained_parameters": unsupported,
        "exposed_features_without_updates": missing_features,
        "policy": training["coverage_policy"],
        "exception_reason": training["coverage_exception_reason"],
        "untrained_auxiliary_outputs": missing_auxiliaries,
        "frozen_dependencies_with_inherited_support": sorted(selected.frozen_parameters),
    }
    summary = {
        "metrics": metrics,
        "overall_losses": overall_losses,
        "evaluation_populations": evaluation_populations,
        "elapsed_seconds": elapsed_before + time.monotonic() - start_time,
        "steps": state["steps"],
        "selected_epoch": state["best_epoch"],
        "training_exposures": dict(exposures),
        "unique_training_labels": admission["training_labels"],
        "support": selected.support,
        "assay_tracks": assay_support,
        "untrained_selected_parameters": unsupported,
        "coverage_gate": coverage_gate,
        "qualification": "not_qualified",
        "test_policy": training["test_policy"],
        "device": device,
        "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
    }
    _json(output_dir / "summary.json", summary)
    _json(
        output_dir / "coverage.json",
        {
            "all_steps": coverage,
            "selected": selected_coverage,
            "input_features": features,
            "parameter_elements": parameter_report(selected),
        },
    )
    complete_run(output_dir, contract)
    _check_coverage_result(summary)
    return summary


class CoverageGateError(ValueError):
    """A completed run whose required training-coverage gate failed."""


def _check_coverage_result(summary):
    gate = summary["coverage_gate"]
    if gate["status"] == "failed" and gate["policy"] == "required":
        raise CoverageGateError(
            "training coverage gate failed; inspect summary.json and coverage.json"
        )


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


def _agent(args):
    return getattr(args, "agent", None) or os.environ.get("PRESTO_AGENT") or "unspecified"


def _register_run(index, output, args, registry):
    if output.exists():
        raise ValueError("experiment directory already exists; choose a fresh registered run")
    output = registry.initialize_experiment_dir(
        out_dir=str(output),
        slug="modular-model-fit",
        title="Modular directly supervised fit",
        source_script="training/modular_run.py",
        agent_label=_agent(args),
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
    for path in SOURCE_FILES:
        target = output / "reproduce" / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / path, target)
    return index


def run_manifest(args):
    """Dispatch from both canonical CLIs; modular settings live in the frozen manifest."""
    allowed = {
        "model_kind",
        "observations_manifest",
        "modular_preflight",
        "modular_resume",
        "run_dir",
        "device",
    }
    explicit = set(getattr(args, "_explicit_cli_dests", ()) or ())
    if explicit - allowed - {"func", "command", "train_command"}:
        raise ValueError(
            "modular training settings belong in the manifest, not legacy CLI flags: "
            + ", ".join(sorted(explicit - allowed))
        )
    if not getattr(args, "observations_manifest", None):
        raise ValueError("modular model requires --observations-manifest")
    resume = getattr(args, "modular_resume", False)
    if resume and (not getattr(args, "run_dir", None) or getattr(args, "modular_preflight", False)):
        raise ValueError("modular resume requires --run-dir and cannot be combined with preflight")
    index = ObservationIndex(args.observations_manifest)
    _, _, _, admission = preflight(index)
    if getattr(args, "modular_preflight", False):
        print(json.dumps(admission, indent=2, sort_keys=True))
        return admission
    from ..scripts import experiment_registry as registry

    agent = _agent(args)
    output = (
        Path(args.run_dir).resolve()
        if getattr(args, "run_dir", None)
        else registry.default_experiment_dir(slug="modular-model-fit", agent_label=agent)
    )
    root = registry.EXPERIMENTS_ROOT.resolve()
    if output.parent != root or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}_\d{4}_[a-z0-9_]+_[a-z0-9-]+", output.name
    ):
        raise ValueError("modular fits must use a timestamped top-level experiments directory")
    if resume:
        origin_path = output / "reproduce" / "manifest-origin.json"
        if not origin_path.exists():
            raise ValueError("resume requires an existing registered modular experiment")
        origin = json.loads(origin_path.read_text())
        if index.manifest_sha256 not in (origin["original_sha256"], origin["frozen_sha256"]):
            raise ValueError("resume manifest differs from the registered original/frozen manifest")
        frozen_path = output / "reproduce" / "source-manifest.json"
        if file_sha256(frozen_path) != origin["frozen_sha256"]:
            raise ValueError("frozen registered manifest changed")
        source_root = Path(__file__).resolve().parent.parent
        for name in SOURCE_FILES:
            if file_sha256(source_root / name) != file_sha256(
                output / "reproduce" / "source" / name
            ):
                raise ValueError(f"registered resume implementation changed: {name}")
        index = ObservationIndex(frozen_path)
        _json(
            output / "reproduce" / f"resume-attempt-{time.time_ns()}.json",
            {
                "manifest": str(frozen_path),
                "resume_snapshot_sha256": file_sha256(output / "resume.pt"),
                "device": getattr(args, "device", None) or "cpu",
            },
        )
    else:
        index = _register_run(index, output, args, registry)
    device = getattr(args, "device", None) or "cpu"
    gate_error = None
    try:
        summary = fit_index(index, output, device=device, resume=resume)
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        _json(output / "failures" / f"{time.time_ns()}.json", failure)
        if not (output / "failure.json").exists():
            _json(output / "failure.json", failure)
        if not isinstance(error, CoverageGateError):
            raise
        # The fit finished and wrote its summary; close it out, then surface the gate.
        gate_error = error
        summary = json.loads((output / "summary.json").read_text())
    narrative = (
        "# Modular directly supervised fit\n\n"
        f"Agent: {agent}. Python: `{sys.version.split()[0]}`. Device: `{device}`.\n\n"
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
        f"Coverage gate: `{summary['coverage_gate']['status']}` "
        f"(policy `{summary['coverage_gate']['policy']}`). "
        f"Test-use policy: `{summary['test_policy']}`. "
        f"No-test reason: {index.manifest.get('training', {}).get('no_test_reason')}.\n\n"
        "Selected/initial/last checkpoints and optimizer states are local raw artifacts "
        "in this directory. "
        "`coverage.json` separates genuine gradients from parameter movement.\n"
    )
    (output / "README.md").write_text(narrative + "\n" + _metric_table(summary))
    heading = f"\n#### Closure: {output.name}\n"
    existing = registry.REGISTRY_MD.read_text() if registry.REGISTRY_MD.exists() else ""
    if heading not in existing:
        closure = (
            f"\n#### Closure: {output.name}\n\n"
            f"Agent: {agent}. Results: [{output.name}/summary.json]({output.name}/summary.json). "
            f"Selected epoch {summary['selected_epoch']}; {summary['steps']} updates; "
            f"{summary['elapsed_seconds']:.2f} seconds on {device}. "
            "One tested condition; no comparative winner or biological qualification. "
            f"Test policy `{summary['test_policy']}`; reason: "
            f"{index.manifest.get('training', {}).get('no_test_reason')}.\n\n"
            + _metric_table(summary)
        )
        atomic_bytes(registry.REGISTRY_MD, (existing + closure).encode())
    if gate_error is not None:
        raise gate_error
    return summary
