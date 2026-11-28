"""Supported modular checkpoint inference in declared target units."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Sequence

import torch

from ..data.modular import endpoint_spec, Observation, identity_vocabulary
from ..data.modular_context import project_apc_context
from ..training.modular import load_checkpoint


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class ModularPredictor:
    """Unqualified scores carry support metadata; untrained requests fail."""

    def __init__(self, checkpoint, *, device="cpu"):
        self.model, self.scales, self.checkpoint = load_checkpoint(checkpoint, device=device)

    def predict(
        self, rows: Sequence[Observation], *, unsupported_apc_policy: str = "error"
    ) -> list[dict]:
        if unsupported_apc_policy not in ("error", "unknown"):
            raise ValueError("unsupported_apc_policy must be error or unknown")
        if unsupported_apc_policy == "unknown":
            config = self.model.config
            rows = [
                replace(
                    row,
                    context=project_apc_context(
                        row.context,
                        cell_types=config.cell_types,
                        cytokines=config.cytokines,
                        processing_pathways=config.processing_pathways,
                        interventions=config.interventions,
                        shared_gene_perturbations=getattr(
                            config, "elution_shared_processing", False
                        ),
                    ),
                )
                for row in rows
            ]
        outputs = self.model.predict(rows)
        results = []
        capabilities = self.model.capabilities()
        panel_support = capabilities.get("assay_tracks", {})
        tcell_support = capabilities.get("tcell_tracks", {})
        for row, output in zip(rows, outputs, strict=True):
            result = {
                "observation_id": row.observation_id,
                "endpoint": row.endpoint,
                "identity_probabilities": {
                    key: value.softmax(-1).cpu().tolist() for key, value in output.identity.items()
                },
                "identity_vocabularies": {
                    "mhc_class": ["I", "II"],
                    "species": list(self.model.config.species),
                    "tcr_lineage": ["CD4", "CD8"],
                    "source_categories": list(self.model.config.source_categories),
                    "per_output": {
                        name: list(
                            identity_vocabulary(
                                name,
                                self.model.config.species,
                                self.model.config.source_categories,
                            )
                        )
                        for name in output.identity
                    },
                },
                "diagnostics": _plain(output.diagnostics),
                "component_scores": _plain(output.components),
                "component_interpretation": (
                    "indirect diagnostics unless separately anchored and validated"
                ),
            }
            omitted = row.context.provenance.get("unrepresented_apc_context")
            if omitted:
                result["unrepresented_apc_context"] = omitted
            if output.tcell_values:
                result["tcell_panel"] = {}
                for key, logit in output.tcell_values.items():
                    support = tcell_support[key]
                    ordinal_logits = output.tcell_intensity_values[key]
                    ordinal_supported = (
                        support["conditional_ordinal_status"] == "directly_supervised"
                    )
                    conditional_intensity = None
                    if ordinal_supported:
                        greater = ordinal_logits.sigmoid()
                        probabilities = torch.stack(
                            (1 - greater[0], greater[0] - greater[1], greater[1])
                        )
                        conditional_intensity = {
                            "logits": _plain(ordinal_logits),
                            "probabilities": _plain(probabilities),
                            "levels": ["low", "intermediate", "high"],
                            "expected_level": float(
                                (probabilities * probabilities.new_tensor([0, 1, 2])).sum()
                            ),
                        }
                    result["tcell_panel"][key] = {
                        **support,
                        "logit": float(logit),
                        "score": float(logit.sigmoid()),
                        "score_interpretation": "uncalibrated measured-outcome score",
                        "conditional_intensity": conditional_intensity,
                    }
                result["selected_tcell_track"] = (
                    row.tcell_track.key if row.tcell_track is not None else None
                )
            if output.value is not None:
                spec = endpoint_spec(row.endpoint)
                result["assay_panel"] = {}
                for key, item in getattr(output, "assay_values", {}).items():
                    status = panel_support[key]
                    value = None if status["status"] == "random/untrained" else float(item)
                    if value is not None and not spec.binary and not spec.quantitative:
                        value = self.scales[row.endpoint].decode(value)
                    result["assay_panel"][key] = {
                        **status,
                        "transformed_value": value,
                        "target_space": "logit"
                        if spec.binary
                        else (f"log10({spec.unit})" if spec.log10 else spec.unit),
                    }
                if spec.binary:
                    result.update(
                        logit=float(output.value),
                        score=float(output.value.sigmoid()),
                        score_interpretation="uncalibrated measured-outcome score",
                    )
                elif spec.quantitative:
                    transformed = float(output.value)
                    contract = row.quantity_contract
                    result.update(
                        transformed_value=transformed,
                        target_space=contract.target_space,
                        unit=contract.unit,
                        value=contract.inverse(transformed),
                        quantity_contract=asdict(contract),
                    )
                    if result["value"] is None:
                        result["physical_value_status"] = "outside_floating_point_range"
                else:
                    transformed = self.scales[row.endpoint].decode(float(output.value))
                    result.update(
                        transformed_value=transformed,
                        target_space=f"log10({spec.unit})" if spec.log10 else spec.unit,
                        unit=spec.unit,
                    )
                    try:
                        physical = 10.0**transformed if spec.log10 else transformed
                    except OverflowError:
                        physical = None
                    result["value"] = (
                        physical if physical is not None and math.isfinite(physical) else None
                    )
                    if result["value"] is None:
                        result["physical_value_status"] = "outside_floating_point_range"
            results.append(result)
        return results

    def rank_pmhc(
        self,
        *,
        tcr_a: str,
        tcr_b: str,
        candidates: Sequence[Observation],
        endpoint: str = "tcr_binding",
    ) -> list[dict]:
        """Score an explicit candidate library; this does not generate a peptide or allele."""
        if endpoint not in ("tcr_binding", "tcr_response"):
            raise ValueError("ranking requires a paired-receptor endpoint")
        if len({row.observation_id for row in candidates}) != len(candidates):
            raise ValueError("candidate library IDs must be unique")
        requests = [
            replace(
                row,
                endpoint=endpoint,
                tcr_a=tcr_a,
                tcr_b=tcr_b,
                value=None,
                unit=None,
                split="predict",
                identity_labels={},
            )
            for row in candidates
        ]
        results = self.predict(requests)
        return sorted(results, key=lambda result: (-result["score"], result["observation_id"]))


class ModularEnsemblePredictor:
    """Multi-checkpoint ensemble predictor averaging calibrated logits and regression values."""

    def __init__(self, checkpoints: Sequence, *, device="cpu"):
        if not checkpoints:
            raise ValueError("ModularEnsemblePredictor requires at least one checkpoint")
        self.predictors = [ModularPredictor(ckpt, device=device) for ckpt in checkpoints]

    def predict(
        self, rows: Sequence[Observation], *, unsupported_apc_policy: str = "error"
    ) -> list[dict]:
        member_outputs = [
            p.predict(rows, unsupported_apc_policy=unsupported_apc_policy) for p in self.predictors
        ]
        n = len(member_outputs)
        merged = []
        for idx, row in enumerate(rows):
            items = [out[idx] for out in member_outputs]
            base = dict(items[0])
            base["ensemble_size"] = n
            if "logit" in base:
                mean_logit = sum(float(it["logit"]) for it in items) / n
                base["logit"] = mean_logit
                base["score"] = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, mean_logit))))
                base["member_logits"] = [float(it["logit"]) for it in items]
            elif "transformed_value" in base:
                mean_tv = sum(float(it["transformed_value"]) for it in items) / n
                base["transformed_value"] = mean_tv
                base["member_transformed_values"] = [float(it["transformed_value"]) for it in items]
                spec = endpoint_spec(row.endpoint)
                if not spec.quantitative:
                    try:
                        phys = 10.0**mean_tv if spec.log10 else mean_tv
                    except OverflowError:
                        phys = None
                    base["value"] = phys if phys is not None and math.isfinite(phys) else None
            if base.get("component_scores"):
                comp = {}
                for k in base["component_scores"]:
                    vals = [
                        float(it["component_scores"][k])
                        for it in items
                        if k in it.get("component_scores", {})
                    ]
                    if vals:
                        comp[k] = sum(vals) / len(vals)
                base["component_scores"] = comp
            merged.append(base)
        return merged
