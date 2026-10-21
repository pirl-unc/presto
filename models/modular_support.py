"""Evidence for input columns/categories, distinct from tensor-level training."""

from contextlib import contextmanager


from ..data.modular import AA, CORECEPTORS, DELIVERY, EXPOSURE_STATES, STATES, identity_vocabulary


def feature_layout(model):
    """Name the externally selectable columns and residue embedding rows."""
    config = model.config

    def vector(name):
        return [f"{name}:representation:{i}" for i in range(128)]

    def numeric(prefix, names):
        return [f"{prefix}:{name}:{part}" for name in names for part in ("value", "known")]

    source = (
        vector("source_n")
        + vector("source_c")
        + [
            "source_n:known",
            "source_c:known",
            "n_terminus:value",
            "n_terminus:known",
            "c_terminus:value",
            "c_terminus:known",
        ]
    )
    result = {"encoder.embedding.weight": (0, ["padding", *AA])}
    if model.has_pmhc:
        names = vector("peptide") + vector("mhc1") + vector("mhc2")
        names += [f"pfr_{side}:{aa}" for side in ("n", "c") for aa in AA]
        names += ["pfr_n:length", "pfr_c:length", "peptide:length", "class:I", "class:II"]
        names += [
            f"{role}:species:{species}" for role in ("mhc1", "mhc2") for species in config.species
        ]
        if "mhc_pairing" in config.auxiliary_context:
            names += ["auxiliary:mhc_pairing:matched", "auxiliary:mhc_pairing:mismatched"]
        result["pmhc_fusion.0.weight"] = (1, names)
    if model.has_context:
        names = vector("pmhc")
        if model.has_tcr:
            names += vector("full_peptide") + vector("tcr_a") + vector("tcr_b")
        if config.source_context and model.has_cellular:
            names += source
        names += [
            f"{role}:species:{species}"
            for role in config.context_species
            for species in config.species
        ]
        for component in config.components:
            names += [f"component:{component}:{state}" for state in STATES]
            if component in config.component_activity_units:
                names += numeric("activity", [component])
        if config.cell_types:
            names += [f"cell_type:{name}" for name in (*config.cell_types, "unknown")]
        for cytokine in config.cytokines:
            names += [f"cytokine:{cytokine}:{state}" for state in EXPOSURE_STATES]
            if cytokine in config.cytokine_dose_units:
                names += [f"cytokine:{cytokine}:log1p_dose", f"cytokine:{cytokine}:dose_known"]
            if cytokine in config.cytokine_duration_features:
                names += [
                    f"cytokine:{cytokine}:log1p_duration_hours",
                    f"cytokine:{cytokine}:duration_known",
                ]
        if model.has_cellular:
            names += [f"delivery:{value}" for value in DELIVERY]
        if model.has_coreceptor:
            names += [f"coreceptor:{value}" for value in CORECEPTORS]
        for name in config.auxiliary_context:
            if name != "mhc_pairing":
                names += [
                    f"auxiliary:{name}:{value}"
                    for value in identity_vocabulary(
                        name,
                        config.species,
                        config.source_categories,
                    )
                ]
        result["context_fusion.0.weight"] = (1, names)
    if model.has_digest:
        names = vector("peptide") + (source if config.source_context else [])
        names += [f"enzyme:{value}" for value in (*config.enzymes, "unknown")]
        names += numeric("preparation", config.preparation_features)
        result["digest_fusion.0.weight"] = (1, names)
    if model.has_ms:
        names = vector("peptide") + numeric("acquisition", config.acquisition_features)
        result["ms_fusion.0.weight"] = (1, names)
    for endpoint, offset in model.offsets.items():
        if offset.covariates is not None:
            result[f"offsets.{endpoint}.covariates.weight"] = (
                1,
                numeric("observation", config.observation_features),
            )
    for name, (axis, labels) in result.items():
        if len(labels) != model.get_parameter(name).shape[axis]:
            raise ValueError(f"input feature layout disagrees with allocated weights: {name}")
    return result


@contextmanager
def feature_inputs(model, *, enforce=False):
    """Observe actual endpoint-masked inputs, or reject untrained active columns."""
    handles = []
    try:
        for name, (axis, labels) in model.feature_layout.items():
            module = model.get_submodule(name.rsplit(".", 1)[0])

            def inspect(_module, args, name=name, axis=axis, labels=labels):
                values = args[0].detach()
                if axis == 0:
                    active = set(values.unique().cpu().tolist()) - {0}
                else:
                    active = set(
                        values.reshape(-1, len(labels))
                        .ne(0)
                        .any(0)
                        .nonzero()
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                if enforce or not model.get_parameter(name).requires_grad:
                    missing = active - model.trained_features[name]
                    if missing:
                        raise ValueError(
                            "untrained input features: "
                            + ", ".join(labels[i] for i in sorted(missing))
                        )
                else:
                    model.exposed_features[name].update(active)

            handles.append(module.register_forward_pre_hook(inspect))
        yield
    finally:
        for handle in handles:
            handle.remove()


def record_feature_updates(model, before):
    for name, (axis, _) in model.feature_layout.items():
        parameter = model.get_parameter(name)
        if parameter.grad is None:
            continue
        # Require a gradient and actual movement on the same weight element.
        evidence = parameter.grad.ne(0) & (parameter.detach() - before[name]).ne(0)
        trained = evidence.any(dim=1 - axis).nonzero().flatten().cpu().tolist()
        model.trained_features[name].update(trained)


def feature_report(model):
    report = {}
    for name, (axis, labels) in model.feature_layout.items():
        fixed = {0} if name == "encoder.embedding.weight" else set()
        exposed, trained = model.exposed_features[name], model.trained_features[name]
        report[name] = {
            "axis": axis,
            "trained": [labels[i] for i in sorted(trained)],
            "exposed_without_update": [labels[i] for i in sorted(exposed - trained - fixed)],
            "unobserved": [labels[i] for i in sorted(set(range(len(labels))) - exposed - fixed)],
            "fixed": [labels[i] for i in sorted(fixed)],
        }
    return report
