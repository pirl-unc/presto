"""Supported modular checkpoint inference in declared target units."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Sequence

import torch

from ..data.modular import endpoint_spec, Observation, identity_vocabulary
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

    def predict(self, rows: Sequence[Observation]) -> list[dict]:
        outputs = self.model.predict(rows)
        results = []
        panel_support = self.model.capabilities().get("assay_tracks", {})
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
