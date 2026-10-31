"""Direct observation losses, training coverage and strict modular checkpoints."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from ..data.modular import (
    endpoint_spec,
    Observation,
    TargetScale,
    observation_target,
    identity_head,
    identity_vocabulary,
    has_identity_input,
)
from ..models.modular import MODEL_VERSION, ModularConfig, ModularPresto, Prediction
from ..models.modular_support import feature_inputs, record_feature_updates
from ..models.modular_dependencies import (
    freeze_dependencies,
    parameter_dependencies,
    validate_freezing,
)


def identity_targets(row: Observation, model: ModularPresto) -> dict[str, int]:
    targets = {}
    for name, label in row.identity_labels.items():
        head = identity_head(name)
        if head is None or head not in model.config.identity_tasks:
            raise ValueError(f"unsupported identity label: {name}")
        if not has_identity_input(row, name):
            raise ValueError(f"identity input missing for {name}")
        vocabulary = identity_vocabulary(name, model.config.species, model.config.source_categories)
        if label not in vocabulary:
            raise ValueError(f"identity label {name}={label} outside declared vocabulary")
        targets[name] = vocabulary.index(label)
    return targets


def observation_loss(
    output: Prediction, row: Observation, scales: dict[str, TargetScale]
) -> torch.Tensor | None:
    if row.value is None:
        return None
    if output.value is None:
        raise ValueError("missing assay prediction")
    value = observation_target(row)
    spec = endpoint_spec(row.endpoint)
    if spec.binary:
        return F.binary_cross_entropy_with_logits(output.value, output.value.new_tensor(value))
    if row.endpoint not in scales:
        raise ValueError(f"missing training-only scaler for {row.endpoint}")
    residual = (
        (output.value - value) / scales[row.endpoint].scale
        if spec.quantitative
        else output.value - scales[row.endpoint].encode(value)
    )
    if row.qualifier in ("<", "<="):
        residual = residual.clamp_min(0)
    elif row.qualifier in (">", ">="):
        residual = residual.clamp_max(0)
    return residual.square()


def batch_loss(
    model: ModularPresto,
    rows: Sequence[Observation],
    outputs: Sequence[Prediction],
    scales: dict[str, TargetScale],
) -> tuple[torch.Tensor, dict[str, int]]:
    if len(rows) != len(outputs):
        raise ValueError("prediction/observation alignment mismatch")
    losses, counts = [], Counter()
    for row, output in zip(rows, outputs, strict=True):
        if row.observation_id != output.observation_id or row.endpoint != output.endpoint:
            raise ValueError("prediction identity mismatch")
        if row.endpoint == "identity":
            for name, target in identity_targets(row, model).items():
                if name not in output.identity:
                    raise ValueError(f"identity input missing for {name}")
                logits = output.identity[name]
                losses.append(
                    F.cross_entropy(logits[None], torch.tensor([target], device=logits.device))
                )
                counts[identity_head(name)] += 1
        else:
            loss = observation_loss(output, row, scales)
            if loss is not None:
                losses.append(loss)
                counts[row.endpoint] += 1
    if not losses:
        raise ValueError("batch has no supported measured labels")
    return torch.stack(losses).mean(), dict(counts)


def optimizer_membership(model: ModularPresto, optimizer: torch.optim.Optimizer) -> None:
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("optimizer must contain each trainable parameter exactly once")


def train_step(
    model: ModularPresto,
    optimizer: torch.optim.Optimizer,
    rows: Sequence[Observation],
    scales: dict[str, TargetScale],
    *,
    identity_rows: Sequence[Observation] = (),
    identity_weight: float = 0.1,
) -> dict[str, Any]:
    """One measured vote per row/bag; identity replay is a separate mean CE."""
    if type(model) is not ModularPresto:
        raise ValueError(
            "new training requires the shared Presto model; historical models are inference-only"
        )
    if any(row.split != "train" for row in (*rows, *identity_rows)):
        raise ValueError("training may not consume held-out observations")
    if identity_weight < 0:
        raise ValueError("negative identity weight")
    if any(row.endpoint == "identity" for row in rows) and any(
        row.endpoint != "identity" for row in rows
    ):
        raise ValueError("assay and identity batches must be sampled separately")
    if any(row.endpoint != "identity" for row in identity_rows):
        raise ValueError("identity replay contains assay rows")
    optimizer_membership(model, optimizer)
    validate_freezing(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with feature_inputs(model):
        loss, counts = batch_loss(model, rows, model(rows), scales)
    loss_parts = {"primary": float(loss.detach())}
    if identity_rows and identity_weight > 0:
        with feature_inputs(model):
            identity_loss, identity_counts = batch_loss(
                model, identity_rows, model(identity_rows), scales
            )
        loss = loss + identity_weight * identity_loss
        loss_parts["identity"] = float(identity_loss.detach())
        if identity_weight > 0:
            counts = dict(Counter(counts) + Counter(identity_counts))
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    if not loss.requires_grad:
        raise ValueError("batch contains no trainable loss; frozen rows belong in evaluation")
    loss.backward()
    before, coverage = {}, {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            optimizer.zero_grad(set_to_none=True)
            raise ValueError(f"nonfinite gradient: {name}")
        before[name] = parameter.detach().clone()
        coverage[name] = {
            "gradient_norm": float(gradient.norm()) if gradient is not None else 0.0,
            "gradient_present": gradient is not None,
            "frozen": name in model.frozen_parameters,
        }
    optimizer.step()
    record_feature_updates(model, before)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"nonfinite parameter after optimizer update: {name}")
        change = (parameter.detach() - before[name]).norm().item()
        coverage[name]["update_norm"] = change
        evidence = (
            (parameter.grad.ne(0) & parameter.detach().ne(before[name]))
            if parameter.grad is not None
            else torch.zeros_like(parameter, dtype=torch.bool)
        )
        model.trained_elements[name] = model.trained_elements[name].to(parameter.device) | evidence
        count = int(evidence.sum())
        coverage[name]["gradient_and_update"] = count > 0
        coverage[name]["gradient_and_update_elements"] = count
    model.trained_parameters.update(
        name for name, evidence in coverage.items() if evidence["gradient_and_update"]
    )
    for row in rows:
        if row.value is not None and row.endpoint != "identity":
            track = model.assay_selection(row)
            if track is not None:
                model.assay_support[track.key] += 1
    updated = {name for name, evidence in coverage.items() if evidence["gradient_and_update"]}
    for target, support in model.support.items():
        if parameter_dependencies(model, target) & updated:
            support["validation"] = "not_evaluated"
    for row in (*rows, *(identity_rows if identity_weight > 0 else ())):
        if row.endpoint != "identity":
            continue
        for name, label in row.identity_labels.items():
            if name.startswith("peptide_source:"):
                category = name.split(":", 1)[1]
                index = model.config.source_categories.index(category)
                parameter_name = "identity_heads.peptide_source.weight"
                parameter = model.get_parameter(parameter_name)
                if (
                    parameter.grad is not None
                    and (
                        parameter.grad[index].ne(0)
                        & (parameter.detach()[index] - before[parameter_name][index]).ne(0)
                    ).any()
                ):
                    model.source_label_support[category].add(label)
    for name, count in counts.items():
        model.support[name]["observations"] += count
        prefix = (
            f"identity_heads.{name}."
            if name in model.config.identity_tasks
            else f"assay_layer.factors.endpoint.{name}"
            if name in model.affinity_endpoints
            else f"heads.{name}."
            if name in model.heads
            else f"offsets.{name}."
        )
        if any(
            evidence["gradient_and_update"]
            for parameter, evidence in coverage.items()
            if (
                parameter == prefix
                if name in model.affinity_endpoints
                else parameter.startswith(prefix)
            )
        ):
            model.support[name]["status"] = "directly_supervised"
        if model.support[name]["status"] != "pretrained_frozen":
            model.support[name]["validation"] = "not_evaluated"
    return {
        "loss": float(loss.detach()),
        "loss_parts": loss_parts,
        "counts": counts,
        "coverage": coverage,
    }


def coverage_summary(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count gradient-driven updates, never count weight-decay movement as training."""
    parameters: dict[str, dict[str, Any]] = {}
    for step in steps:
        for name, evidence in step["coverage"].items():
            record = parameters.setdefault(
                name,
                {
                    "gradient_steps": 0,
                    "update_steps": 0,
                    "gradient_and_update_steps": 0,
                    "max_gradient_norm": 0.0,
                    "trainable_steps": 0,
                },
            )
            record["gradient_steps"] += evidence["gradient_norm"] > 0
            record["trainable_steps"] += not evidence.get("frozen", False)
            record["update_steps"] += evidence["update_norm"] > 0
            record["gradient_and_update_steps"] += evidence["gradient_and_update"]
            record["max_gradient_norm"] = max(
                record["max_gradient_norm"], evidence["gradient_norm"]
            )
    return {
        "parameters": parameters,
        "untrained_parameters": [
            name
            for name, value in parameters.items()
            if value["trainable_steps"] and not value["gradient_and_update_steps"]
        ],
    }


def save_checkpoint(
    path: str | Path,
    model: ModularPresto,
    scales: dict[str, TargetScale],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if type(model) is not ModularPresto:
        raise ValueError("cannot save a historical model as a current training checkpoint")
    validate_freezing(model)
    payload = {
        "format": MODEL_VERSION,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "scales": {key: asdict(value) for key, value in scales.items()},
        "support": model.support,
        "assay_support": model.assay_support,
        "trained_parameters": sorted(model.trained_parameters),
        "element_coverage": {
            "masks": {name: value.cpu() for name, value in model.trained_elements.items()},
            "complete_history": sorted(model.element_history_complete),
        },
        "frozen_targets": list(model.frozen_targets),
        "frozen_parameters": sorted(model.frozen_parameters),
        "trained_features": {key: sorted(value) for key, value in model.trained_features.items()},
        "exposed_features": {key: sorted(value) for key, value in model.exposed_features.items()},
        "source_label_support": {
            key: sorted(value) for key, value in model.source_label_support.items()
        },
        "metadata": metadata or {},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[ModularPresto, dict[str, TargetScale], dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") == "presto.modular.v2":
        from ..legacy.modular_v2 import ModularConfig as HistoricalConfig
        from ..legacy.modular_v2 import ModularPresto as HistoricalPresto

        model = HistoricalPresto(HistoricalConfig.from_dict(payload["config"])).to(device)
    elif payload.get("format") == "presto.shared.v3":
        from ..legacy.shared_v3 import PrestoConfig as HistoricalConfig
        from ..legacy.shared_v3 import Presto as HistoricalPresto

        model = HistoricalPresto(HistoricalConfig.from_dict(payload["config"])).to(device)
    elif payload.get("format") == "presto.shared.v4":
        from ..legacy.shared_v4 import PrestoConfig as HistoricalConfig
        from ..legacy.shared_v4 import Presto as HistoricalPresto

        model = HistoricalPresto(HistoricalConfig.from_dict(payload["config"])).to(device)
    elif payload.get("format") == "presto.shared.v5":
        from ..legacy.shared_v5 import PrestoConfig as HistoricalConfig
        from ..legacy.shared_v5 import Presto as HistoricalPresto

        model = HistoricalPresto(HistoricalConfig.from_dict(payload["config"])).to(device)
    elif payload.get("format") == "presto.shared.v6":
        from ..legacy.shared_v6 import PrestoConfig as HistoricalConfig
        from ..legacy.shared_v6 import Presto as HistoricalPresto

        model = HistoricalPresto(HistoricalConfig.from_dict(payload["config"])).to(device)
    elif payload.get("format") == MODEL_VERSION:
        model = ModularPresto(ModularConfig.from_dict(payload["config"])).to(device)
    else:
        raise ValueError("incompatible checkpoint format")
    model.load_state_dict(payload["state_dict"], strict=True)
    if set(payload["support"]) != set(model.support):
        raise ValueError("checkpoint support/module mismatch")
    allowed = {"random/untrained", "indirectly_trained", "directly_supervised", "pretrained_frozen"}
    for name, value in payload["support"].items():
        if value.get("status") not in allowed or value.get("observations", -1) < 0:
            raise ValueError(f"invalid checkpoint support: {name}")
    model.support = payload["support"]
    trained = set(payload["trained_parameters"])
    if trained - dict(model.named_parameters()).keys():
        raise ValueError("checkpoint training coverage refers to absent parameters")
    model.trained_parameters = trained
    if hasattr(model, "assay_support"):
        support = payload.get("assay_support")
        if (
            not isinstance(support, dict)
            or set(support) != set(model.assay_support)
            or any(type(value) is not int or value < 0 for value in support.values())
        ):
            raise ValueError("invalid checkpoint assay support")
        model.assay_support = support
    # v5 already saved coordinate evidence. Changing its reader to historical
    # inference must not erase that evidence or fabricate complete zero masks.
    if hasattr(model, "trained_elements"):
        evidence = payload.get("element_coverage")
        if evidence is None:
            model.element_history_complete = set()
        else:
            masks, complete = evidence.get("masks"), evidence.get("complete_history")
            parameters = dict(model.named_parameters())
            if (
                not isinstance(masks, dict)
                or set(masks) != set(parameters)
                or not isinstance(complete, list)
                or set(complete) - set(parameters)
            ):
                raise ValueError("invalid checkpoint element coverage names")
            for name, mask in masks.items():
                if (
                    not isinstance(mask, torch.Tensor)
                    or mask.dtype != torch.bool
                    or mask.shape != parameters[name].shape
                ):
                    raise ValueError("invalid checkpoint element coverage mask")
            model.trained_elements = masks
            model.element_history_complete = set(complete)
    source_support = payload.get("source_label_support", {})
    if set(source_support) != set(model.config.source_categories) or any(
        not isinstance(values, list) or set(values) - {"present", "absent"}
        for values in source_support.values()
    ):
        raise ValueError("invalid checkpoint peptide source output support")
    model.source_label_support = {key: set(value) for key, value in source_support.items()}
    for key in ("trained_features", "exposed_features"):
        values = payload.get(key)
        if not isinstance(values, dict) or set(values) != set(model.feature_layout):
            raise ValueError("checkpoint input-feature support/module mismatch")
        for name, indices in values.items():
            size = len(model.feature_layout[name][1])
            if not isinstance(indices, list) or any(
                type(i) is not int or not 0 <= i < size for i in indices
            ):
                raise ValueError("invalid checkpoint input-feature evidence")
            getattr(model, key)[name] = set(indices)
    if any(
        model.trained_features[name] - model.exposed_features[name] for name in model.feature_layout
    ):
        raise ValueError("trained feature lacks exposure evidence")
    scales = {key: TargetScale(**value) for key, value in payload["scales"].items()}
    expected = {name for name in model.config.endpoints if not endpoint_spec(name).binary}
    if set(scales) != expected:
        raise ValueError("checkpoint scaler/endpoint mismatch")
    declared = payload.get("frozen_targets", [])
    frozen = payload.get("frozen_parameters", [])
    if not isinstance(declared, list) or not isinstance(frozen, list):
        raise ValueError("invalid checkpoint freeze contract")
    restored_freeze = freeze_dependencies(model, declared)
    if restored_freeze["parameters"] != sorted(frozen):
        raise ValueError("checkpoint frozen dependencies disagree with target declarations")
    return model, scales, payload


def initialize_from_checkpoint(
    model: ModularPresto, path: str | Path, *, allow_task_removal: bool = False
) -> dict[str, Any]:
    """Transfer compatible tasks; retained scalers keep learned output meanings."""
    previous, scales, payload = load_checkpoint(path, device=str(model.device))
    if type(model) is not ModularPresto or type(previous) is not ModularPresto:
        raise ValueError(
            "historical checkpoint feature contracts require frozen-code reproduction, "
            "not warm-start"
        )
    old, new = asdict(previous.config), asdict(model.config)
    # Endpoint/identity unions may grow; feature slots may not silently change meaning.
    shared_contract = {
        "assay_tracks",
        "assay_mode",
        "max_length",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "peptide_context",
        "species_defaults",
        "identity_threshold",
    }
    if previous.has_pmhc and model.has_pmhc:
        shared_contract.update({"mhc_classes", "register_pooling"})
    if previous.has_context and model.has_context:
        shared_contract.add("auxiliary_context")
    if (
        (
            "peptide_source" in previous.config.identity_tasks
            and "peptide_source" in model.config.identity_tasks
        )
        or any(name.startswith("peptide_source:") for name in previous.config.auxiliary_context)
        and model.has_context
    ):
        shared_contract.add("source_categories")
    if (
        previous.has_pmhc
        and model.has_pmhc
        or previous.config.context_species
        and model.config.context_species
        or any(
            task.endswith("species")
            for task in set(previous.config.identity_tasks) & set(model.config.identity_tasks)
        )
    ):
        shared_contract.add("species")
    if previous.has_context and model.has_context:
        shared_contract.add("context_species")
    if previous.has_cellular and model.has_cellular:
        shared_contract.update({"components", "component_activity_units", "machinery_profiles"})
        shared_contract.update(
            {
                "machinery_references",
                "cell_types",
                "cytokines",
                "cytokine_dose_units",
                "cytokine_duration_features",
                "processing_pathways",
                "interventions",
                "ordinal_scales",
            }
        )
    if previous.has_digest and model.has_digest:
        shared_contract.update({"enzymes", "preparation_features", "preparation_units"})
    if previous.has_cellular and model.has_cellular or previous.has_digest and model.has_digest:
        shared_contract.add("source_context")
    if previous.has_ms and model.has_ms:
        shared_contract.update({"acquisition_features", "acquisition_units"})
    if any(
        endpoint_spec(name).observation_covariates
        for name in set(previous.config.endpoints) & set(model.config.endpoints)
    ):
        shared_contract.update({"observation_features", "observation_units"})
    for domain, contract in previous.config.quantity_domains.items():
        if (
            domain in model.config.quantity_domains
            and model.config.quantity_domains[domain] != contract
        ):
            raise ValueError(f"incompatible pretrained measurement domain: {domain}")
    if getattr(previous, "staged_elution", False) or model.staged_elution:
        shared_contract.update(
            {"elution_staging", "elution_affinity_endpoint", "elution_detach_affinity"}
        )
        old, new = ({"elution_staging": "legacy_joint", **side} for side in (old, new))
    for key in shared_contract:
        if old[key] != new[key]:
            raise ValueError(f"incompatible pretrained feature contract: {key}")
    removed_tasks = (set(previous.config.endpoints) | set(previous.config.identity_tasks)) - (
        set(model.config.endpoints) | set(model.config.identity_tasks)
    )
    if removed_tasks and not allow_task_removal:
        raise ValueError("pretrained task removal requires an explicit migration")
    current = {name: value.clone() for name, value in model.state_dict().items()}
    inherited, projected = [], {}
    for name, value in previous.state_dict().items():
        if name not in current:
            if allow_task_removal:
                continue
            raise ValueError(f"incompatible pretrained module: {name}")
        old_layout, new_layout = previous.feature_layout.get(name), model.feature_layout.get(name)
        if old_layout != new_layout and old_layout is not None and new_layout is not None:
            axis, old_labels = old_layout
            new_axis, new_labels = new_layout
            if axis != new_axis or current[name].shape[1 - axis] != value.shape[1 - axis]:
                raise ValueError(f"incompatible pretrained input feature layout: {name}")
            removed = set(old_labels) - set(new_labels)
            removable_prefixes = []
            if name in ("context_fusion.0.weight", "elution_processing.0.weight"):
                if previous.has_tcr and not model.has_tcr:
                    removable_prefixes += ["full_peptide:", "tcr_a:", "tcr_b:"]
                if previous.has_cellular and not model.has_cellular:
                    removable_prefixes += [
                        "source_",
                        "n_terminus:",
                        "c_terminus:",
                        "cell_type:",
                        "cytokine:",
                        "component:",
                        "activity:",
                        "delivery:",
                    ]
                if previous.has_coreceptor and not model.has_coreceptor:
                    removable_prefixes += ["coreceptor:"]
            if any(not label.startswith(tuple(removable_prefixes)) for label in removed):
                raise ValueError(f"retained function loses active input features: {name}")
            old_positions = {label: index for index, label in enumerate(old_labels)}
            copied = []
            for j, label in enumerate(new_labels):
                if label not in old_positions:
                    continue
                i = old_positions[label]
                if axis == 1:
                    current[name][:, j] = value[:, i]
                else:
                    current[name][j] = value[i]
                copied.append(label)
            projected[name] = {
                "copied": copied,
                "new": sorted(set(new_labels) - set(old_labels)),
                "removed": sorted(set(old_labels) - set(new_labels)),
            }
        elif current[name].shape != value.shape:
            raise ValueError(f"incompatible pretrained module: {name}")
        else:
            current[name] = value
        inherited.append(name)
    model.load_state_dict(current, strict=True)
    model.assay_support = dict(previous.assay_support)
    model.trained_parameters.update(previous.trained_parameters & set(inherited))
    for name in inherited:
        if name not in model.trained_elements:
            continue
        if name not in previous.element_history_complete:
            model.element_history_complete.discard(name)
        old_mask = previous.trained_elements[name].to(model.device)
        if name in projected:
            model.trained_elements[name] = model.trained_elements[name].to(model.device)
            axis, old_labels = previous.feature_layout[name]
            new_positions = {label: i for i, label in enumerate(model.feature_layout[name][1])}
            for i, label in enumerate(old_labels):
                if label not in new_positions:
                    continue
                j = new_positions[label]
                if axis == 1:
                    model.trained_elements[name][:, j] = old_mask[:, i]
                else:
                    model.trained_elements[name][j] = old_mask[i]
        else:
            model.trained_elements[name] = old_mask.clone()
    for category, values in previous.source_label_support.items():
        if category in model.source_label_support:
            model.source_label_support[category] = set(values)
    for name in previous.feature_layout:
        if name not in model.feature_layout:
            continue
        old_labels = previous.feature_layout[name][1]
        new_positions = {label: index for index, label in enumerate(model.feature_layout[name][1])}
        for attribute in ("trained_features", "exposed_features"):
            getattr(model, attribute)[name] = {
                new_positions[old_labels[i]]
                for i in getattr(previous, attribute)[name]
                if old_labels[i] in new_positions
            }
    for name, support in previous.support.items():
        if name in model.support:
            model.support[name] = {
                **support,
                "status": "directly_supervised"
                if support["status"] == "pretrained_frozen"
                else support["status"],
                "validation": "requires_requalification_after_finetuning",
            }
    return {
        "inherited_parameters": inherited,
        "removed_tasks": sorted(removed_tasks),
        "feature_projection": projected,
        "scales": {name: scale for name, scale in scales.items() if name in model.config.endpoints},
        "metadata": payload["metadata"],
    }
