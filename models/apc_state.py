"""One APC representation for measured state, cell identity and interventions.

Categorical interventions never become invented expression fold changes.
Namespaced factors preserve concurrent knockout backgrounds and rescue constructs.
The shared state feeds supply and MHC-dependent selection, never intrinsic affinity.
"""

from __future__ import annotations

import math
import torch
from torch import nn

from ..data.modular import STATES, EXPOSURE_STATES
from . import apm_state as biology


APC_PREFIXES = ("component:", "activity:", "cell_type:", "cytokine:", "pathway:", "intervention:")


def factor_classes(name):
    """Direct pathway scope; unclassified exposures retain both class routes."""
    target = name.split(":", 1)[-1].split(":", 1)[0].upper().replace("-", "_")
    target = {"TAPBPR": "TAPBPL", "ERAAP": "ERAP1", "GILT": "IFI30"}.get(target, target)
    if target in {"HLA_A", "HLA_B", "HLA_C", "HLA_E", "HLA_F", "HLA_G"}:
        return ("I",)
    if target.startswith(
        ("HLA_DRA", "HLA_DRB", "HLA_DPA", "HLA_DPB", "HLA_DQA", "HLA_DQB", "HLA_DM", "HLA_DO")
    ):
        return ("II",)
    if target == "TAX1BP1":
        return ("I", "II")
    if target not in biology.APM_GENE_STEPS:
        return ("I", "II")
    steps = set(biology.APM_GENE_STEPS[target])
    result = tuple(
        c
        for c, p in (("I", "class_i"), ("II", "class_ii"))
        if steps.intersection(biology.PATHWAY_STEPS[p])
    )
    return result or ("I", "II")


def apc_columns(config):
    """Ordered external feature contract, also used by checkpoint support."""
    columns = []

    def add(key, family, name, part, classes=("I", "II")):
        columns.append((key, family, name, part, classes))

    for name in config.components:
        classes = factor_classes(name)
        for state in STATES:
            if state != "unknown":
                add(f"component:{name}:{state}", "component", name, state, classes)
        if name in config.component_activity_units:
            for part in ("value", "known", "reference"):
                add(f"activity:{name}:{part}", "activity", name, part, classes)
    for name in config.cell_types:
        add(f"cell_type:{name}", "cell", name, "probability")
    for name in config.cytokines:
        for state in EXPOSURE_STATES:
            if state != "unknown":
                add(f"cytokine:{name}:{state}", "cytokine", name, state)
        if name in config.cytokine_dose_units:
            for part in ("dose", "dose_known"):
                add(f"cytokine:{name}:{part}", "cytokine", name, part)
        if name in config.cytokine_duration_features:
            for part in ("duration", "duration_known"):
                add(f"cytokine:{name}:{part}", "cytokine", name, part)
    for family, names in (
        ("intervention", config.interventions),
        ("pathway", config.processing_pathways),
    ):
        for name in names:
            classes = factor_classes(name) if family == "intervention" else ("I", "II")
            for state in EXPOSURE_STATES:
                if state != "unknown":
                    add(f"{family}:{name}:{state}", family, name, state, classes)
    return tuple(columns)


class HarmonizedAPCState(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.columns = apc_columns(config)
        if not self.columns:
            raise ValueError("harmonized APC state requires allocated context inputs")
        self.encoder = nn.Linear(len(self.columns), config.state_bottleneck, bias=False)
        for c, key in (("I", "mask_i"), ("II", "mask_ii")):
            self.register_buffer(key, torch.tensor([float(c in col[4]) for col in self.columns]))
        width = config.d_model + (2 * config.d_model + 6 if config.source_context else 0)
        self.supply = nn.Linear(width, config.state_bottleneck)
        self.selection = nn.Linear(config.d_model, config.state_bottleneck, bias=False)
        nn.init.zeros_(self.supply.weight)
        nn.init.zeros_(self.supply.bias)
        nn.init.zeros_(self.selection.weight)

    def residual_penalty(self):
        return self.encoder.weight.square().sum()

    def features(self, components, cell):
        values = []
        for _, family, name, part, _ in self.columns:
            if family == "component":
                value = float(components[name].state == part)
            elif family == "activity":
                claim = components[name]
                if (
                    claim.activity is not None
                    and claim.unit != self.config.component_activity_units[name]
                ):
                    raise ValueError(f"APC activity unit mismatch for {name}")
                if part == "known":
                    value = float(claim.activity is not None)
                elif part == "reference":
                    value = float(
                        claim.activity is not None
                        and (claim.state == "reference" or bool(claim.evidence.get("assumption")))
                    )
                else:
                    value = claim.activity or 0.0
                    if claim.unit == "TPM":
                        if value < 0:
                            raise ValueError("APC RNA abundance cannot be negative")
                        value = math.log1p(value)
            elif family == "cell":
                value = cell.cell_type.probabilities[self.config.cell_types.index(name)]
            elif family == "cytokine":
                exposure = cell.cytokines[name]
                if part in EXPOSURE_STATES:
                    value = float(exposure.state == part)
                else:
                    numeric = exposure.dose if part.startswith("dose") else exposure.duration_hours
                    value = (
                        float(numeric is not None)
                        if part.endswith("known")
                        else math.log1p(numeric or 0)
                    )
            else:
                claims = (
                    cell.interventions if family == "intervention" else cell.processing_pathways
                )
                value = float(claims[name].state == part)
            values.append(value)
        return self.encoder.weight.new_tensor(values)

    def forward(self, components, cell, mhc_class, source_vector, mhc_hidden):
        features = self.features(components, cell)
        mask = self.mask_i if mhc_class == "I" else self.mask_ii
        state = torch.tanh(self.encoder(features * mask))
        supply = (self.supply(source_vector) * state).sum(-1)
        selection = (self.selection(mhc_hidden) * state).sum(-1)
        return (
            supply,
            selection,
            {
                "feature_names": [c[0] for c in self.columns],
                "features": features,
                "pathway_mask": mask,
                "state": state,
            },
        )
