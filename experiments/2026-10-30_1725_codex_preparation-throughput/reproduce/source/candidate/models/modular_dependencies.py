"""Declared function dependencies for transfer, freezing and training admission."""

from ..data.modular import endpoint_spec, supervised_targets
from ..data.modular_quantitative import quantity_domain


def parameter_dependencies(model, target):
    """Conservative full input-domain dependency set, not only a scalar head."""
    parameters = dict(model.named_parameters())
    prefixes = {"encoder."}
    exact = set()

    def pmhc(contextual=False, receptor=False):
        prefixes.update(("pmhc_fusion.", "residue_interaction.", "interaction_norm."))
        if hasattr(model, "register_score"):
            prefixes.add("register_score.")
        if contextual:
            prefixes.add("context_fusion.")
        for name in model.config.identity_tasks:
            if name.startswith("mhc") or (receptor and name.startswith("tcr")):
                prefixes.add(f"identity_heads.{name}.")
            if contextual and name == "peptide_source":
                prefixes.add(f"identity_heads.{name}.")

    def component(part, domain=None):
        if domain is None:
            name = "ms_score" if part == "ms_response" else f"{part}_score"
            if not hasattr(model, name):
                raise ValueError(f"component is not allocated: {part}")
            prefixes.add(f"{name}.")
        else:
            if domain not in model.quantity_scores or part not in model.quantity_scores[domain]:
                raise ValueError(f"quantity component is not allocated: {part}:{domain}")
            prefixes.add(f"quantity_scores.{domain}.{part}.")
        if part == "ms_response":
            prefixes.add("ms_fusion.")
        elif part == "digest":
            prefixes.add("digest_fusion.")
        elif part == "cellular":
            pmhc(contextual=True)
        else:
            raise ValueError(f"unsupported component: {part}")

    if target in model.config.identity_tasks:
        prefixes.add(f"identity_heads.{target}.")
    elif target in model.config.endpoints:
        spec = endpoint_spec(target)
        if spec.route in ("pmhc", "cellular", "tcr", "tcr_cellular"):
            pmhc(spec.route != "pmhc", spec.route in ("tcr", "tcr_cellular"))
        for part in spec.parts:
            component(part, quantity_domain(target))
        prefixes.add(f"offsets.{target}." if spec.parts else f"heads.{target}.")
        if target in getattr(model, "affinity_endpoints", ()):
            exact.update(model.assay_layer.dependencies(target))
        elif any(track.endpoint == target for track in getattr(model.config, "assay_tracks", ())):
            prefixes.add("assay_layer.")
    elif target.startswith("component:"):
        parts = target.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("component target must name a component and optional domain")
        component(parts[1], parts[2] if len(parts) == 3 else None)
    else:
        raise ValueError(f"unknown dependency target: {target}")
    return exact | {
        name for name in parameters if any(name.startswith(prefix) for prefix in prefixes)
    }


def frozen_output(model, target):
    return parameter_dependencies(model, target) <= model.frozen_parameters


def freeze_dependencies(model, targets):
    """Freeze previously trained functions; no validation claim is created."""
    if any(not isinstance(target, str) for target in targets):
        raise ValueError("freeze targets must be strings")
    targets = tuple(sorted(set(targets)))
    frozen = set().union(*(parameter_dependencies(model, target) for target in targets))
    missing = frozen - model.trained_parameters
    if missing:
        raise ValueError(
            "cannot freeze dependencies without trained support: " + ", ".join(sorted(missing))
        )
    model.frozen_targets = targets
    model.frozen_parameters = frozen
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in frozen)
        parameter.grad = None
    for name, support in model.support.items():
        if frozen_output(model, name) and support["status"] != "random/untrained":
            support["status"] = "pretrained_frozen"
        elif support["status"] == "pretrained_frozen":
            support["status"] = "directly_supervised"
            support["validation"] = "requires_requalification_after_finetuning"
    model.train(model.training)
    return {"targets": list(targets), "parameters": sorted(frozen)}


def validate_freezing(model):
    expected = set().union(
        *(parameter_dependencies(model, target) for target in model.frozen_targets)
    )
    actual = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if actual != expected or model.frozen_parameters != expected:
        raise ValueError("parameter freeze flags disagree with declared dependency targets")
    if expected - model.trained_parameters:
        raise ValueError("frozen dependency lacks inherited training support")


def row_has_trainable_loss(model, row):
    return any(not frozen_output(model, target) for target in supervised_targets(row))
