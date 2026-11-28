"""Direct observation losses, training coverage and strict modular checkpoints."""

from __future__ import annotations

from collections import Counter
import contextlib
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
from .allele_diff import allele_diff_loss
from .allele_swap import allele_swap_loss
from .apc_contrasts import apc_contrast_loss
from .modular_controls import ranking_loss, state_contrast_loss, tcell_ranking_loss
from .processing_pairs import pair_loss, validate_pair
from ..models.imgt_groove import (
    NUM_LEGACY_POCKET_SLOTS,
    POCKET_SLOT_SCHEME_LEGACY58,
    POCKET_SLOT_SCHEME_UNION46,
    pocket_slot_overlap_mapping,
    pocket_slot_specs,
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
    output: Prediction,
    row: Observation,
    scales: dict[str, TargetScale],
    *,
    tcell_ordinal_weight: float = 1.0,
) -> torch.Tensor | None:
    parts = observation_loss_parts(output, row, scales)
    if parts is None:
        return None
    if tcell_ordinal_weight == 0 or "conditional_ordinal" not in parts:
        return parts["primary"]
    return parts["primary"] + tcell_ordinal_weight * parts["conditional_ordinal"]


def observation_loss_parts(
    output: Prediction, row: Observation, scales: dict[str, TargetScale]
) -> dict[str, torch.Tensor] | None:
    if row.value is None:
        return None
    if output.value is None:
        raise ValueError("missing assay prediction")
    value = observation_target(row)
    spec = endpoint_spec(row.endpoint)
    if spec.binary:
        parts = {
            "primary": F.binary_cross_entropy_with_logits(
                output.value, output.value.new_tensor(value)
            )
        }
        if row.tcell_intensity is not None:
            logits = output.tcell_intensity_value
            if logits is None or logits.shape != (2,):
                raise ValueError("missing conditional T-cell intensity prediction")
            ordinal_targets = logits.new_tensor(
                [row.tcell_intensity > boundary for boundary in range(2)]
            )
            parts["conditional_ordinal"] = F.binary_cross_entropy_with_logits(
                logits, ordinal_targets
            )
        return parts
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
    return {"primary": residual.square()}


def batch_loss(
    model: ModularPresto,
    rows: Sequence[Observation],
    outputs: Sequence[Prediction],
    scales: dict[str, TargetScale],
    *,
    normalization: str = "observation_mean",
    endpoint_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Mean loss per observation, or (per_endpoint) a weighted mean of endpoint means."""
    if len(rows) != len(outputs):
        raise ValueError("prediction/observation alignment mismatch")
    losses, keys, counts = [], [], Counter()
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
                keys.append(identity_head(name))
        else:
            loss = observation_loss(
                output,
                row,
                scales,
                tcell_ordinal_weight=getattr(model.config, "tcell_ordinal_loss_weight", 1.0),
            )
            if loss is not None:
                losses.append(loss)
                counts[row.endpoint] += 1
                keys.append(row.endpoint)
    if not losses:
        raise ValueError("batch has no supported measured labels")
    if normalization == "per_endpoint":
        weights = endpoint_weights or {}
        members = {key: [] for key in dict.fromkeys(keys)}
        for key, loss in zip(keys, losses, strict=True):
            members[key].append(loss)
        total = sum(weights.get(key, 1.0) for key in members)
        return (
            sum(
                weights.get(key, 1.0) * torch.stack(values).mean()
                for key, values in members.items()
            )
            / total,
            dict(counts),
        )
    if normalization != "observation_mean":
        raise ValueError("unsupported loss normalization")
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
    normalization: str = "observation_mean",
    endpoint_weights: dict[str, float] | None = None,
    ranking: dict[str, Any] | None = None,
    state_contrast: dict[str, Any] | None = None,
    processing_pairs: Sequence[tuple[Observation, Observation]] = (),
    processing_pair_weight: float = 1.0,
    processing_pair_logit_weight: float = 0.0,
    processing_pair_selection_weight: float = 0.0,
    processing_pair_margin: float = 0.0,
    apc_contrasts: tuple[Sequence[Any], Sequence[Any]] | None = None,
    apc_contrast_options: dict[str, Any] | None = None,
    allele_swap_groups: Sequence[Any] = (),
    allele_swap_weight: float = 0.0,
    allele_swap_temperature: float = 0.5,
    allele_diff_pairs: Sequence[Any] = (),
    allele_diff_weight: float = 0.0,
    allele_diff_beta: float = 0.5,
    tcell_ranking_weight: float = 0.0,
    grad_clip_norm: float | None = None,
    amp_bfloat16: bool = False,
    elution_pmhc_residual_l2_weight: float = 0.0,
    ms_response_elution_l2_weight: float = 0.0,
) -> dict[str, Any]:
    """One measured vote per row/bag; identity replay is a separate mean CE."""
    if type(model) is not ModularPresto:
        raise ValueError(
            "new training requires the shared Presto model; historical models are inference-only"
        )
    if any(row.split != "train" for row in (*rows, *identity_rows)):
        raise ValueError("training may not consume held-out observations")
    if processing_pairs:
        if not model.staged_elution or model.frozen_parameters:
            raise ValueError("processing pair training requires unfrozen staged presentation")
        if type(processing_pair_weight) not in (
            int,
            float,
        ) or not 0 < processing_pair_weight < float("inf"):
            raise ValueError("processing pair weight must be positive and finite")
        for positive, background in processing_pairs:
            validate_pair(positive, background)
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
    use_amp = bool(amp_bfloat16) and model.device.type == "cuda"
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    )
    with feature_inputs(model), amp_ctx:
        outputs = model(rows)
    loss, counts = batch_loss(
        model,
        rows,
        outputs,
        scales,
        normalization=normalization,
        endpoint_weights=endpoint_weights,
    )
    loss_parts = {"primary": float(loss.detach())}
    if processing_pairs:
        with feature_inputs(model), amp_ctx:
            pair_outputs = model([r for pair in processing_pairs for r in pair])
        paired_loss = pair_loss(
            pair_outputs,
            logit_weight=processing_pair_logit_weight,
            selection_weight=processing_pair_selection_weight,
            margin=processing_pair_margin,
        )
        loss = loss + processing_pair_weight * paired_loss
        loss_parts["processing_pairs"] = len(processing_pairs)
        loss_parts["processing_pair_loss"] = float(paired_loss.detach())
        loss_parts["processing_pair_weight"] = processing_pair_weight
        if processing_pair_logit_weight:
            loss_parts["processing_pair_logit_weight"] = processing_pair_logit_weight
        if processing_pair_selection_weight:
            loss_parts["processing_pair_selection_weight"] = processing_pair_selection_weight
        if processing_pair_margin:
            loss_parts["processing_pair_margin"] = processing_pair_margin
    if apc_contrasts is not None and apc_contrast_options is not None:
        if len(apc_contrasts) == 3:
            sel_draws, yld_draws, quant_draws = apc_contrasts
        else:
            sel_draws, yld_draws = apc_contrasts
            quant_draws = ()
        if sel_draws or yld_draws or quant_draws:
            with feature_inputs(model), amp_ctx:
                apc_term, apc_parts = apc_contrast_loss(
                    model,
                    sel_draws,
                    yld_draws,
                    quant_draws=quant_draws,
                    yield_weight=apc_contrast_options.get("yield_weight", 0.5),
                    margin=apc_contrast_options.get("margin", 0.25),
                    quant_weight=apc_contrast_options.get("quant_weight", 0.5),
                    quant_rank_weight=apc_contrast_options.get("quant_rank_weight", 0.5),
                    quant_scale=apc_contrast_options.get("quant_scale", 0.40),
                )
            if apc_term is not None:
                loss = loss + apc_contrast_options["weight"] * apc_term.float()
                loss_parts["apc_contrast_loss"] = float(apc_term.detach())
                loss_parts.update(apc_parts)
    if allele_swap_groups and allele_swap_weight > 0:
        with feature_inputs(model), amp_ctx:
            swap_term, swap_parts = allele_swap_loss(
                model, allele_swap_groups, temperature=allele_swap_temperature
            )
        if swap_term is not None:
            loss = loss + float(allele_swap_weight) * swap_term.float()
            loss_parts.update(swap_parts)
    if allele_diff_pairs and allele_diff_weight > 0:
        with feature_inputs(model), amp_ctx:
            diff_term, diff_parts = allele_diff_loss(
                model, allele_diff_pairs, scales, beta=allele_diff_beta
            )
        if diff_term is not None:
            loss = loss + float(allele_diff_weight) * diff_term.float()
            loss_parts.update(diff_parts)
    if tcell_ranking_weight > 0:
        tc_rank, tc_pairs = tcell_ranking_loss(rows, outputs)
        if tc_rank is not None:
            loss = loss + float(tcell_ranking_weight) * tc_rank.float()
            loss_parts["tcell_ranking_loss"] = float(tc_rank.detach())
            loss_parts["tcell_ranking_pairs"] = tc_pairs
    tcell_binary, tcell_ordinal = [], []
    for row, output in zip(rows, outputs, strict=True):
        if row.endpoint != "tcell_response" or row.value is None:
            continue
        parts = observation_loss_parts(output, row, scales)
        tcell_binary.append(parts["primary"])
        if "conditional_ordinal" in parts:
            tcell_ordinal.append(parts["conditional_ordinal"])
    if tcell_binary:
        loss_parts["tcell_binary"] = float(torch.stack(tcell_binary).mean().detach())
    if tcell_ordinal:
        loss_parts["tcell_conditional_ordinal"] = float(torch.stack(tcell_ordinal).mean().detach())
        loss_parts["tcell_ordinal_weight"] = model.config.tcell_ordinal_loss_weight
    if ranking is not None:
        pairwise, pairs = ranking_loss(
            rows,
            outputs,
            ranking["endpoints"],
            grouping=ranking.get("grouping", "study"),
            include_censored=bool(ranking.get("include_censored", False)),
        )
        if pairwise is not None:
            loss = loss + ranking["weight"] * pairwise
            loss_parts["ranking"] = float(pairwise.detach())
        loss_parts["ranking_pairs"] = pairs
    if state_contrast is not None:
        with feature_inputs(model):
            contrast, contrast_rows = state_contrast_loss(model, rows)
        if contrast is not None:
            loss = loss + state_contrast["weight"] * contrast
            loss_parts["state_contrast"] = float(contrast.detach())
        loss_parts["state_contrast_rows"] = contrast_rows
    if elution_pmhc_residual_l2_weight > 0:
        res_terms = []
        for row, output in zip(rows, outputs, strict=True):
            if row.endpoint != "elution_observation":
                continue
            for cand in output.diagnostics.get("candidates", ()):
                for reg in cand.get("registers", ()):
                    et = reg.get("elution_terms")
                    if et and "pmhc_residual" in et:
                        res_terms.append(et["pmhc_residual"].float().pow(2))
        if res_terms:
            res_l2 = torch.stack(res_terms).mean()
            loss = loss + float(elution_pmhc_residual_l2_weight) * res_l2
            loss_parts["pmhc_residual_l2"] = float(res_l2.detach())
    if ms_response_elution_l2_weight > 0:
        ms_terms = [
            output.components["ms_response"].float().pow(2)
            for row, output in zip(rows, outputs, strict=True)
            if row.endpoint == "elution_observation" and "ms_response" in output.components
        ]
        if ms_terms:
            ms_l2 = torch.stack(ms_terms).mean()
            loss = loss + float(ms_response_elution_l2_weight) * ms_l2
            loss_parts["ms_response_elution_l2"] = float(ms_l2.detach())
    if identity_rows and identity_weight > 0:
        with feature_inputs(model), amp_ctx:
            id_outputs = model(identity_rows)
        identity_loss, identity_counts = batch_loss(model, identity_rows, id_outputs, scales)
        loss = loss + identity_weight * identity_loss
        loss_parts["identity"] = float(identity_loss.detach())
        if identity_weight > 0:
            counts = dict(Counter(counts) + Counter(identity_counts))
    loss = loss.float()
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
    grad_norm = None
    if grad_clip_norm is not None:
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.grad is not None],
                grad_clip_norm,
            )
        )
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
            tcell_track = model.tcell_selection(row)
            if tcell_track is not None:
                model.tcell_support[tcell_track.key]["binary"] += 1
                if row.tcell_intensity is not None:
                    model.tcell_support[tcell_track.key]["conditional_ordinal"] += 1
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
            if name in model.assay_endpoints
            else f"heads.{name}."
            if name in model.heads
            else f"offsets.{name}."
        )
        trained = any(
            evidence["gradient_and_update"]
            for parameter, evidence in coverage.items()
            if (
                parameter == prefix
                if name in model.assay_endpoints
                else parameter.startswith(prefix)
            )
        )
        if name == "tcell_response" and model.config.tcell_tracks:
            trained = any(
                coverage[parameter]["gradient_and_update"]
                for parameter in model.tcell_assay_layer.dependencies(ordinal=False)
            )
        if trained:
            model.support[name]["status"] = "directly_supervised"
        if model.support[name]["status"] != "pretrained_frozen":
            model.support[name]["validation"] = "not_evaluated"
    return {
        "loss": float(loss.detach()),
        "loss_parts": loss_parts,
        **({"grad_norm": grad_norm} if grad_clip_norm is not None else {}),
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
        "tcell_support": model.tcell_support,
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
        cfg_data = dict(payload["config"])
        if (
            cfg_data.get("imgt_pocket_cross") or cfg_data.get("direct_pocket_bilinear")
        ) and "imgt_pocket_slot_scheme" not in cfg_data:
            sd = payload.get("state_dict") or {}
            dpb_v = sd.get("direct_pocket_bilinear.by_class.I.mhc_v")
            if dpb_v is None:
                dpb_v = sd.get("direct_pocket_bilinear.by_class.II.mhc_v")
            ipc_ln = sd.get("imgt_pocket_cross.direct_mlp.0.weight")
            if (dpb_v is not None and dpb_v.shape[0] == NUM_LEGACY_POCKET_SLOTS) or (
                ipc_ln is not None and ipc_ln.shape[0] == (9 + NUM_LEGACY_POCKET_SLOTS) * 16 + 2
            ):
                cfg_data["imgt_pocket_slot_scheme"] = POCKET_SLOT_SCHEME_LEGACY58
        model = ModularPresto(ModularConfig.from_dict(cfg_data)).to(device)
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
    if hasattr(model, "tcell_support"):
        support = payload.get("tcell_support", {})
        if (
            not isinstance(support, dict)
            or set(support) != set(model.tcell_support)
            or any(
                not isinstance(value, dict)
                or set(value) != {"binary", "conditional_ordinal"}
                or any(type(count) is not int or count < 0 for count in value.values())
                for value in support.values()
            )
        ):
            raise ValueError("invalid checkpoint T-cell support")
        model.tcell_support = support
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
    model: ModularPresto,
    path: str | Path,
    *,
    allow_task_removal: bool = False,
    allow_apc_expansion: bool = False,
    allow_apc_vocab_expansion: bool = False,
    allow_pathway_expansion: bool = False,
    allow_junction_expansion: bool = False,
    allow_shared_processing_expansion: bool = False,
    allow_interaction_expansion: bool = False,
    allow_tcell_staging_expansion: bool = False,
) -> dict[str, Any]:
    """Transfer compatible tasks; retained scalers keep learned output meanings."""
    previous, scales, payload = load_checkpoint(path, device=str(model.device))
    if type(model) is not ModularPresto or type(previous) is not ModularPresto:
        raise ValueError(
            "historical checkpoint feature contracts require frozen-code reproduction, "
            "not warm-start"
        )
    old, new = asdict(previous.config), asdict(model.config)
    if type(allow_interaction_expansion) is not bool:
        raise ValueError("allow_interaction_expansion must be boolean")
    old_im = old.get("interaction_mode", "mean")
    new_im = new.get("interaction_mode", "mean")
    old_gse = bool(old.get("groove_segment_embedding", False))
    new_gse = bool(new.get("groove_segment_embedding", False))
    old_imgt = bool(old.get("imgt_groove_embedding", False))
    new_imgt = bool(new.get("imgt_groove_embedding", False))
    old_ipc = bool(old.get("imgt_pocket_cross", False))
    new_ipc = bool(new.get("imgt_pocket_cross", False))
    old_pss = str(old.get("imgt_pocket_slot_scheme", POCKET_SLOT_SCHEME_UNION46))
    new_pss = str(new.get("imgt_pocket_slot_scheme", POCKET_SLOT_SCHEME_UNION46))
    pocket_scheme_upgraded = (
        (old_ipc or bool(old.get("direct_pocket_bilinear", False)))
        and old_pss == POCKET_SLOT_SCHEME_LEGACY58
        and new_pss == POCKET_SLOT_SCHEME_UNION46
    )
    old_ema = bool(old.get("elution_mhc_abundance", False))
    new_ema = bool(new.get("elution_mhc_abundance", False))
    old_c1o = bool(old.get("class1_overhang_registers", False))
    new_c1o = bool(new.get("class1_overhang_registers", False))
    old_tcst = bool(old.get("tcell_class_specific_tcr", False))
    new_tcst = bool(new.get("tcell_class_specific_tcr", False))
    old_tras = bool(old.get("tcell_register_aligned_supply", False))
    new_tras = bool(new.get("tcell_register_aligned_supply", False))
    old_ccao = bool(old.get("class_conditioned_assay_offset", False))
    new_ccao = bool(new.get("class_conditioned_assay_offset", False))
    old_dpb = bool(old.get("direct_pocket_bilinear", False))
    new_dpb = bool(new.get("direct_pocket_bilinear", False))
    old_gcc = bool(old.get("groove_cross_chain", False))
    new_gcc = bool(new.get("groove_cross_chain", False))
    old_qae = bool(old.get("query_anchor_embedding", False))
    new_qae = bool(new.get("query_anchor_embedding", False))
    old_gpp = bool(old.get("groove_pocket_pool", False))
    new_gpp = bool(new.get("groove_pocket_pool", False))
    old_coop = bool(old.get("interaction_cooperativity", False))
    new_coop = bool(new.get("interaction_cooperativity", False))
    old_pfr = bool(old.get("pmhc_pfr_conditioning", False))
    new_pfr = bool(new.get("pmhc_pfr_conditioning", False))
    old_ref = bool(old.get("pmhc_refine", False))
    new_ref = bool(new.get("pmhc_refine", False))
    old_rpm = old.get("register_pooling_mode", "normalized")
    new_rpm = new.get("register_pooling_mode", "normalized")
    interaction_expanded = (
        (old_im == "anchor_v2" and new_im == "anchor_v3")
        or (not old_gse and new_gse)
        or (not old_imgt and new_imgt)
        or (not old_ipc and new_ipc)
        or pocket_scheme_upgraded
        or (not old_ema and new_ema)
        or (not old_c1o and new_c1o)
        or (not old_tcst and new_tcst)
        or (not old_tras and new_tras)
        or (not old_ccao and new_ccao)
        or (not old_dpb and new_dpb)
        or (not old_gcc and new_gcc)
        or (not old_qae and new_qae)
        or (not old_gpp and new_gpp)
        or (not old_coop and new_coop)
        or (not old_pfr and new_pfr)
        or (not old_ref and new_ref)
        or (old_rpm == "normalized" and new_rpm == "thermodynamic")
    )
    if old_im != new_im and not (old_im == "anchor_v2" and new_im == "anchor_v3"):
        raise ValueError(f"unsupported interaction_mode migration: {old_im} -> {new_im}")
    if old_gse and not new_gse:
        raise ValueError("cannot remove pretrained groove_segment_embedding")
    if old_imgt and not new_imgt:
        raise ValueError("cannot remove pretrained imgt_groove_embedding")
    if old_ipc and not new_ipc:
        raise ValueError("cannot remove pretrained imgt_pocket_cross")
    if (old_ipc or old_dpb) and old_pss != new_pss and not pocket_scheme_upgraded:
        raise ValueError(f"unsupported imgt_pocket_slot_scheme migration: {old_pss} -> {new_pss}")
    if old_ema and not new_ema:
        raise ValueError("cannot remove pretrained elution_mhc_abundance")
    if old_c1o and not new_c1o:
        raise ValueError("cannot remove pretrained class1_overhang_registers")
    if old_tcst and not new_tcst:
        raise ValueError("cannot remove pretrained tcell_class_specific_tcr")
    if old_tras and not new_tras:
        raise ValueError("cannot remove pretrained tcell_register_aligned_supply")
    if old_ccao and not new_ccao:
        raise ValueError("cannot remove pretrained class_conditioned_assay_offset")
    if old_dpb and not new_dpb:
        raise ValueError("cannot remove pretrained direct_pocket_bilinear")
    if old_gcc and not new_gcc:
        raise ValueError("cannot remove pretrained groove_cross_chain")
    if old_qae and not new_qae:
        raise ValueError("cannot remove pretrained query_anchor_embedding")
    if old_gpp and not new_gpp:
        raise ValueError("cannot remove pretrained groove_pocket_pool")
    if old_coop and not new_coop:
        raise ValueError("cannot remove pretrained interaction_cooperativity")
    if old_pfr and not new_pfr:
        raise ValueError("cannot remove pretrained pmhc_pfr_conditioning")
    if old_ref and not new_ref:
        raise ValueError("cannot remove pretrained pmhc_refine")
    if old_rpm != new_rpm and not (old_rpm == "normalized" and new_rpm == "thermodynamic"):
        raise ValueError(f"unsupported register_pooling_mode migration: {old_rpm} -> {new_rpm}")
    if allow_interaction_expansion != interaction_expanded:
        raise ValueError(
            "expanding pMHC interaction/readout architecture requires explicit "
            "allow_interaction_expansion"
        )
    if type(allow_tcell_staging_expansion) is not bool:
        raise ValueError("allow_tcell_staging_expansion must be boolean")
    old_tcs = bool(old.get("tcell_staged", False))
    new_tcs = bool(new.get("tcell_staged", False))
    tcell_staging_added = not old_tcs and new_tcs
    if old_tcs and not new_tcs:
        raise ValueError("cannot remove pretrained tcell_staged head")
    if allow_tcell_staging_expansion != tcell_staging_added:
        raise ValueError("enabling tcell_staged requires explicit allow_tcell_staging_expansion")
    if type(allow_shared_processing_expansion) is not bool:
        raise ValueError("allow_shared_processing_expansion must be boolean")
    shared_added = not old["elution_shared_processing"] and new["elution_shared_processing"]
    if allow_shared_processing_expansion != shared_added:
        raise ValueError(
            "adding shared processing requires explicit allow_shared_processing_expansion"
        )
    if old["elution_shared_processing"] and not new["elution_shared_processing"]:
        raise ValueError("cannot remove pretrained shared processing")
    if type(allow_junction_expansion) is not bool:
        raise ValueError("allow_junction_expansion must be boolean")
    junction_added = not old["elution_junction_processing"] and new["elution_junction_processing"]
    if allow_junction_expansion != junction_added:
        raise ValueError("adding junction processing requires explicit allow_junction_expansion")
    if old["elution_junction_processing"] and not new["elution_junction_processing"]:
        raise ValueError("cannot remove pretrained junction processing")
    if type(allow_pathway_expansion) is not bool:
        raise ValueError("allow_pathway_expansion must be boolean")
    pathway_added = not old["elution_pathway_conditioning"] and new["elution_pathway_conditioning"]
    if allow_pathway_expansion != pathway_added:
        raise ValueError("adding pathway conditioning requires explicit allow_pathway_expansion")
    if old["elution_pathway_conditioning"] and not new["elution_pathway_conditioning"]:
        raise ValueError("cannot remove pretrained pathway conditioning")
    apc_keys = {
        "components",
        "component_activity_units",
        "machinery_profiles",
        "machinery_references",
        "cell_types",
        "cytokines",
        "cytokine_dose_units",
        "cytokine_duration_features",
        "processing_pathways",
        "interventions",
    }
    if type(allow_apc_vocab_expansion) is not bool:
        raise ValueError("allow_apc_vocab_expansion must be boolean")
    if allow_apc_expansion and allow_apc_vocab_expansion:
        raise ValueError("allow_apc_expansion and allow_apc_vocab_expansion are mutually exclusive")
    if allow_apc_expansion and (
        not getattr(model, "harmonized_apc", False)
        or previous.config.state_bottleneck
        or any(old[k] for k in apc_keys)
    ):
        raise ValueError(
            "APC expansion requires a context-unallocated parent and harmonized target"
        )
    if allow_apc_vocab_expansion:
        if not getattr(model, "harmonized_apc", False):
            raise ValueError("APC vocabulary expansion requires harmonized APC target")
        for k in apc_keys:
            ov, nv = old[k], new[k]
            if isinstance(ov, dict):
                if not isinstance(nv, dict) or any(nv.get(dk) != dv for dk, dv in ov.items()):
                    raise ValueError(f"APC vocabulary expansion must preserve existing {k}")
            elif not (set(ov) <= set(nv)):
                raise ValueError(f"APC vocabulary expansion must be a superset for {k}")
    # Endpoint/identity unions may grow; feature slots may not silently change meaning.
    shared_contract = {
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
    # A newly added elution output has no learned staging contract to preserve.
    # Existing presentation outputs still require the identical dependency path.
    if "elution_observation" in previous.config.endpoints and (
        getattr(previous, "staged_elution", False) or model.staged_elution
    ):
        shared_contract.update(
            {"elution_staging", "elution_affinity_endpoint", "elution_detach_affinity"}
        )
        old, new = ({"elution_staging": "legacy_joint", **side} for side in (old, new))
    for key in shared_contract:
        if (allow_apc_expansion or allow_apc_vocab_expansion) and key in apc_keys:
            continue
        if old[key] != new[key]:
            raise ValueError(f"incompatible pretrained feature contract: {key}")
    old_tracks = set(previous.config.assay_tracks)
    new_tracks = set(model.config.assay_tracks)
    removed_tracks = old_tracks - new_tracks
    if removed_tracks and not allow_task_removal:
        raise ValueError("pretrained assay-track removal requires an explicit migration")
    old_tcell_tracks = set(previous.config.tcell_tracks)
    new_tcell_tracks = set(model.config.tcell_tracks)
    removed_tcell_tracks = old_tcell_tracks - new_tcell_tracks
    if removed_tcell_tracks and not allow_task_removal:
        raise ValueError("pretrained T-cell-track removal requires an explicit migration")
    removed_tasks = (set(previous.config.endpoints) | set(previous.config.identity_tasks)) - (
        set(model.config.endpoints) | set(model.config.identity_tasks)
    )
    if removed_tasks and not allow_task_removal:
        raise ValueError("pretrained task removal requires an explicit migration")
    current = {name: value.clone() for name, value in model.state_dict().items()}
    pocket_overlap = pocket_slot_overlap_mapping(old_pss, new_pss) if pocket_scheme_upgraded else []
    old_n_pockets = len(pocket_slot_specs(old_pss)) if pocket_scheme_upgraded else 0
    new_n_pockets = len(pocket_slot_specs(new_pss)) if pocket_scheme_upgraded else 0
    inherited, projected = [], {}
    for name, value in previous.state_dict().items():
        if pocket_scheme_upgraded and name.startswith("imgt_pocket_cross.direct_mlp."):
            # Re-init direct_mlp cleanly (final layer is zero-init -> exact 0.0 step-0 shift).
            continue
        if name not in current:
            if (
                allow_task_removal
                or (
                    allow_tcell_staging_expansion
                    and name.startswith("tcell_assay_layer.positivity_decoder.")
                )
                or (
                    allow_interaction_expansion
                    and name == "tcell_staged_head.stab_weight_raw.II"
                    and new_tcst
                )
            ):
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
            if allow_apc_vocab_expansion and name in (
                "apm_state.encoder.weight",
                "elution_shared.apc_encoder.weight",
            ):
                current[name].zero_()
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
            if allow_apc_vocab_expansion and name in (
                "apm_state.mask_i",
                "apm_state.mask_ii",
                "elution_shared.apc_mask_i",
                "elution_shared.apc_mask_ii",
                "elution_shared.pan_yield_caps",
            ):
                pass
            elif (
                allow_apc_vocab_expansion
                and name == "elution_shared.yield_theta"
                and hasattr(previous, "elution_shared")
                and hasattr(model, "elution_shared")
            ):
                old_pan = {
                    col[0]: idx for idx, col in enumerate(previous.elution_shared.pan_columns)
                }
                current[name].zero_()
                for j, col in enumerate(model.elution_shared.pan_columns):
                    if col[0] in old_pan:
                        current[name][j] = value[old_pan[col[0]]]
            elif (
                allow_interaction_expansion
                and name == "interaction_anchor_proj.weight"
                and old_im == "anchor_v2"
                and new_im == "anchor_v3"
            ):
                d = model.config.d_model
                current[name].zero_()
                for old_slot, new_slot in ((0, 0), (1, 1), (2, 4), (3, 7), (4, 8)):
                    current[name][:, new_slot * d : (new_slot + 1) * d] = value[
                        :, old_slot * d : (old_slot + 1) * d
                    ]
            elif (
                allow_interaction_expansion
                and pocket_scheme_upgraded
                and name.startswith("direct_pocket_bilinear.by_class.")
                and (name.endswith(".mhc_v") or name.endswith(".mhc_bias"))
            ):
                for old_s, new_s in pocket_overlap:
                    current[name][new_s] = value[old_s]
            elif (
                allow_interaction_expansion
                and pocket_scheme_upgraded
                and name.startswith("direct_pocket_bilinear.by_class.")
                and (name.endswith(".linear_energy.weight") or name.endswith(".mlp.0.weight"))
            ):
                current[name].zero_()
                for p in range(9):
                    for old_s, new_s in pocket_overlap:
                        current[name][:, p * new_n_pockets + new_s] = value[
                            :, p * old_n_pockets + old_s
                        ]
                current[name][:, -3:] = value[:, -3:]
            else:
                raise ValueError(f"incompatible pretrained module: {name}")
        else:
            current[name] = value
        inherited.append(name)
    if (
        "elution_stability_weight" in current
        and "elution_stability_weight" not in previous.state_dict()
    ):
        current["elution_stability_weight"].zero_()
    model.load_state_dict(current, strict=True)
    model.assay_support = {key: previous.assay_support.get(key, 0) for key in model.assay_support}
    model.tcell_support = {
        key: previous.tcell_support.get(key, {"binary": 0, "conditional_ordinal": 0})
        for key in model.tcell_support
    }
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
        elif (
            allow_apc_vocab_expansion
            and name == "elution_shared.yield_theta"
            and old_mask.shape != model.trained_elements[name].shape
        ):
            model.trained_elements[name] = model.trained_elements[name].to(model.device)
            old_pan = {col[0]: idx for idx, col in enumerate(previous.elution_shared.pan_columns)}
            for j, col in enumerate(model.elution_shared.pan_columns):
                if col[0] in old_pan:
                    model.trained_elements[name][j] = old_mask[old_pan[col[0]]]
        elif (
            allow_interaction_expansion
            and name == "interaction_anchor_proj.weight"
            and old_mask.shape != model.trained_elements[name].shape
        ):
            d = model.config.d_model
            model.trained_elements[name] = model.trained_elements[name].to(model.device)
            model.trained_elements[name].zero_()
            for old_slot, new_slot in ((0, 0), (1, 1), (2, 4), (3, 7), (4, 8)):
                model.trained_elements[name][:, new_slot * d : (new_slot + 1) * d] = old_mask[
                    :, old_slot * d : (old_slot + 1) * d
                ]
        elif (
            allow_interaction_expansion
            and pocket_scheme_upgraded
            and name.startswith("direct_pocket_bilinear.by_class.")
            and (name.endswith(".mhc_v") or name.endswith(".mhc_bias"))
            and old_mask.shape != model.trained_elements[name].shape
        ):
            model.trained_elements[name] = model.trained_elements[name].to(model.device)
            model.trained_elements[name].zero_()
            for old_s, new_s in pocket_overlap:
                model.trained_elements[name][new_s] = old_mask[old_s]
        elif (
            allow_interaction_expansion
            and pocket_scheme_upgraded
            and name.startswith("direct_pocket_bilinear.by_class.")
            and (name.endswith(".linear_energy.weight") or name.endswith(".mlp.0.weight"))
            and old_mask.shape != model.trained_elements[name].shape
        ):
            model.trained_elements[name] = model.trained_elements[name].to(model.device)
            model.trained_elements[name].zero_()
            for p in range(9):
                for old_s, new_s in pocket_overlap:
                    model.trained_elements[name][:, p * new_n_pockets + new_s] = old_mask[
                        :, p * old_n_pockets + old_s
                    ]
            model.trained_elements[name][:, -3:] = old_mask[:, -3:]
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
        "added_assay_tracks": sorted(track.key for track in new_tracks - old_tracks),
        "removed_assay_tracks": sorted(track.key for track in removed_tracks),
        "added_tcell_tracks": sorted(track.key for track in new_tcell_tracks - old_tcell_tracks),
        "removed_tcell_tracks": sorted(track.key for track in removed_tcell_tracks),
        "feature_projection": projected,
        "scales": {name: scale for name, scale in scales.items() if name in model.config.endpoints},
        "metadata": payload["metadata"],
    }
