"""Behavioral contracts for the versioned small model; no biological claims."""

from dataclasses import asdict, replace
import math

import pytest
import torch

from presto.data.modular import (
    Component,
    Context,
    MHCInput,
    Observation,
    TargetScale,
    fit_target_scales,
    transformed_target,
)
from presto.data.modular_context import resolve_species, resolve_components
from presto.models.modular import ModularConfig, ModularPresto, Prediction, candidates
from presto.training.modular import (
    batch_loss,
    coverage_summary,
    initialize_from_checkpoint,
    load_checkpoint,
    observation_loss,
    save_checkpoint,
    train_step,
)


def row(endpoint="ic50", *, mhc_class="II", **kwargs):
    defaults = dict(
        observation_id="r1",
        source="software-fixture",
        endpoint=endpoint,
        peptide="ACDEFGHIKLMNPQR",
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", mhc_class),),
        value=100.0,
        unit="nM",
        group_id="g1",
    )
    defaults.update(kwargs)
    return Observation(**defaults)


def model(**kwargs):
    return ModularPresto(ModularConfig(dropout=0, **kwargs)).eval()


def scale():
    return {"ic50": TargetScale(2.0, 1.0, 2)}


def test_target_arithmetic_preserves_units_and_uncapped_values():
    assert transformed_target("ic50", 1000000, "nM") == 6
    assert transformed_target("kd", 0.1, "uM") == 2
    assert transformed_target("half_life", 2, "h") == pytest.approx(math.log10(120))
    assert transformed_target("koff", 60, "1/min") == 0
    assert transformed_target("tm", -10, "C") == -10
    for endpoint, value, unit in [
        ("ic50", 0, "nM"),
        ("tm", 20, "K"),
        ("bulk_observation", 0.2, "binary"),
    ]:
        with pytest.raises(ValueError, match="positive|unsupported|binary outcome"):
            transformed_target(endpoint, value, unit)


def test_scaler_uses_only_exact_training_values():
    rows = [
        row(value=10),
        row(value=1000, observation_id="r2"),
        row(value=1e12, qualifier=">"),
        row(value=1e15, split="validation"),
    ]
    scales = fit_target_scales(rows)
    assert scales["ic50"] == TargetScale(2, 1, 2)
    with pytest.raises(ValueError, match="distinct exact"):
        fit_target_scales([row(), row(value=1e8, qualifier="<")])


@pytest.mark.parametrize(
    ("qualifier", "prediction", "expected"),
    [
        ("=", 2.0, 4.0),
        ("<", -2.0, 0.0),
        ("<", 2.0, 4.0),
        (">", 2.0, 0.0),
        (">", -2.0, 4.0),
    ],
)
def test_censored_loss(qualifier, prediction, expected):
    observation = row(qualifier=qualifier)
    output = Prediction("r1", "ic50", torch.tensor(prediction))
    assert observation_loss(output, observation, scale()).item() == expected


def test_candidates_preserve_register_positions_and_pfrs():
    whole = candidates("ACDEFGHIKLMNPQR", "I")
    assert len(whole) == 1 and whole[0][1] == "ACDEFGHIKLMNPQR"
    assert whole[0][2][:2] == [0, 0]
    cores = candidates("ACDEFGHIKLMNPQR", "II")
    assert len(cores) == 7 and [x[0] for x in cores] == list(range(7))
    assert cores[0][1] == "ACDEFGHIK" and cores[-1][1] == "HIKLMNPQR"
    assert len(cores[3][2]) == 3
    assert len(candidates("ACDEFGHIK", "II")) == 1
    with pytest.raises(ValueError, match="at least 9"):
        candidates("ACDE", "II")


def test_sequence_validation_rejects_invalid_short_classii_and_missing_partner():
    network = model()
    for observation in (
        row(peptide="AX*"),
        row(peptide="ACDE"),
        row(mhc=(MHCInput("ACD", "", "II"),)),
    ):
        with pytest.raises(ValueError, match="unsupported|required|at least 9"):
            network([observation])


def test_padding_batch_and_irrelevant_metadata_invariance():
    network = model()
    original = row()
    expected = network([original])[0].value
    other = row(observation_id="other", peptide="ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQ")
    actual = network([original, other])[0].value
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    changed = replace(
        original,
        provenance={"study": "new", "allele": "different"},
        tcr_a="VWYACDEF",
        tcr_b="LMNPQRST",
        context=Context(
            components={"component_a": Component("lost")},
            species={"apc": "mouse"},
            enzyme="arbitrary",
        ),
    )
    torch.testing.assert_close(network([changed])[0].value, expected, rtol=1e-5, atol=1e-6)
    changed_partner = replace(original, mhc=(replace(original.mhc[0], groove2="AAAAAAAAAAAA"),))
    assert not torch.allclose(network([changed_partner])[0].value, expected)


def test_class_weights_normalize_before_mixing_and_override_keeps_prediction():
    network = model(identity_tasks=("mhc_class",))
    with torch.no_grad():
        network.identity_heads["mhc_class"].weight.zero_()
        network.identity_heads["mhc_class"].bias.zero_()
    inferred = row(mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC"),))
    output = network([inferred])[0]
    candidate = output.diagnostics["candidates"][0]
    assert [item["class_probability"] for item in candidate["registers"]] == [0.5, 0.5]
    assert all(
        torch.isclose(item["weights"].sum(), torch.tensor(1.0)) for item in candidate["registers"]
    )
    override = replace(inferred, mhc=(replace(inferred.mhc[0], class_override="I"),))
    known = network([override])[0].diagnostics["candidates"][0]
    assert known["class"]["predicted"] == (0.5, 0.5)
    assert known["class"]["probabilities"] == (1.0, 0.0)
    assert len(known["registers"]) == 1


def test_assay_loss_does_not_train_identity_classifiers():
    network = model(identity_tasks=("mhc_class", "mhc_species"), species=("human", "mouse"))
    observation = row(mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC"),))
    loss, _ = batch_loss(network, [observation], network([observation]), scale())
    loss.backward()
    assert all(p.grad is None for p in network.identity_heads.parameters())
    assert network.encoder.embedding.weight.grad.norm() > 0
    assert network.register_score.weight.grad.norm() > 0


def test_defaults_overrides_conflicting_species_and_multiple_losses():
    context = Context(
        species_overrides={"apc": "mouse"},
        components={"a": Component("lost"), "b": Component("reduced")},
        component_overrides={"b": Component("intact")},
    )
    mhc = MHCInput("ACD", "EFG", "II", species1="human", species2="human")
    resolved = resolve_species(("human", "mouse"), context, mhc)
    assert resolved["immune"].value == "human"
    assert resolved["immune"].source.startswith("default:")
    assert resolved["apc"].value == "mouse" and resolved["apc"].source == "override"
    assert resolved["antigen"].value is None
    components = resolve_components(
        context, resolved["apc"], ("a", "b", "c", "d"), {"mouse": ("a", "b", "c")}
    )
    assert [components[key].state for key in ("a", "b", "c", "d")] == [
        "lost",
        "intact",
        "reference",
        "unknown",
    ]
    mixed = resolve_species(("human", "mouse"), Context(), replace(mhc, species2="mouse"))
    assert mixed["immune"].value is None
    strict = resolve_species(("human", "mouse"), Context(), mhc, defaults=False)
    assert strict["apc"].value is None


def test_bulk_subset_shares_detector_with_elution_and_ignores_enzyme_in_detector():
    network = model(
        endpoints=("bulk_observation", "elution_observation"),
        enzymes=("trypsin", "other"),
        source_context=True,
        components=("a",),
        acquisition_features=("energy",),
        acquisition_units={"energy": "eV"},
    )
    bulk = row(
        "bulk_observation", mhc=(), value=1, unit="binary", context=Context(enzyme="trypsin")
    )
    elution = row("elution_observation", value=1, unit="binary")
    outputs = network([bulk, elution])
    torch.testing.assert_close(
        outputs[0].components["ms_response"], outputs[1].components["ms_response"]
    )
    changed = replace(bulk, context=Context(enzyme="other", components={"a": Component("lost")}))
    alternative = network([changed])[0]
    torch.testing.assert_close(
        alternative.components["ms_response"], outputs[0].components["ms_response"]
    )
    assert not torch.allclose(alternative.components["digest"], outputs[0].components["digest"])
    outputs[0].value.backward()
    assert network.ms_score.weight.grad.norm() > 0 and network.digest_score.weight.grad.norm() > 0
    assert network.cellular_score.weight.grad is None


def test_subset_module_allocation():
    bulk = model(endpoints=("bulk_observation",))
    assert not hasattr(bulk, "pmhc_fusion") and not hasattr(bulk, "context_fusion")
    assert not bulk.identity_heads and not bulk.heads
    assert bulk([row("bulk_observation", mhc=(), value=0, unit="binary")])[0].value.ndim == 0
    identity = model(endpoints=(), identity_tasks=("mhc_class",))
    assert not hasattr(identity, "pmhc_fusion") and not hasattr(identity, "ms_fusion")
    record = row("identity", peptide="", value=None, unit=None, identity_labels={"mhc_class": "II"})
    assert identity([record])[0].value is None
    binding = model(mhc_classes=("I",))
    assert not hasattr(binding, "register_score") and not hasattr(binding, "ms_score")
    with pytest.raises(ValueError, match="not allocated"):
        binding([row("elution_observation", value=1, unit="binary")])


def test_candidate_bag_is_one_loss_and_duplicate_candidates_do_not_change_score():
    network = model(endpoints=("elution_observation",))
    original = row("elution_observation", value=1, unit="binary")
    duplicated = replace(original, mhc=original.mhc * 2)
    torch.testing.assert_close(network([original])[0].value, network([duplicated])[0].value)
    multi = replace(original, mhc=(*original.mhc, MHCInput("VVVVVVVV", "YYYYYYYY", "I")))
    outputs = network([multi])
    loss, counts = batch_loss(network, [multi], outputs, {})
    assert counts == {"elution_observation": 1}
    assert loss.item() == pytest.approx(torch.nn.functional.softplus(-outputs[0].value).item())


def test_tcr_conditions_registers_but_does_not_change_pmhc_prediction():
    network = model(endpoints=("ic50", "tcr_binding", "tcr_response"), components=("a",))
    binding = row()
    receptor = row("tcr_binding", tcr_a="ACDEFGHIKLMN", tcr_b="PQRSTVWYACD", value=1, unit="binary")
    first = network([binding, receptor])
    altered = replace(receptor, tcr_b="AAAAAAAAAAAA")
    second = network([binding, altered])
    torch.testing.assert_close(first[0].value, second[0].value)
    assert not torch.allclose(first[1].value, second[1].value)
    a = first[1].diagnostics["candidates"][0]["registers"][0]["weights"]
    b = second[1].diagnostics["candidates"][0]["registers"][0]["weights"]
    assert not torch.allclose(a, b)
    changed_apc = replace(receptor, context=Context(components={"a": Component("lost")}))
    torch.testing.assert_close(network([changed_apc])[0].value, first[1].value)


def test_identity_replay_training_coverage_and_checkpoint_round_trip(tmp_path):
    network = model(identity_tasks=("mhc_class",))
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
    identity = row(
        "identity", peptide="", value=None, unit=None, identity_labels={"mhc_class": "II"}
    )
    result = train_step(network, optimizer, [row()], scale(), identity_rows=[identity])
    summary = coverage_summary([result])
    assert not summary["untrained_parameters"]
    assert result["counts"] == {"ic50": 1, "mhc_class": 1}
    output = network.predict([row()])[0].value
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, network, scale(), optimizer=optimizer, metadata={"software_test": True})
    restored, scales, payload = load_checkpoint(path)
    torch.testing.assert_close(restored.predict([row()])[0].value, output)
    assert scales == scale() and payload["optimizer"] is not None
    incompatible = dict(payload)
    incompatible["format"] = "presto.v2"
    torch.save(incompatible, path)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)


def test_untrained_serving_and_wrong_optimizer_are_rejected():
    network = model()
    with pytest.raises(ValueError, match="no trained"):
        network.predict([row()])
    optimizer = torch.optim.AdamW(network.heads.parameters())
    with pytest.raises(ValueError, match="each trainable"):
        train_step(network, optimizer, [row()], scale())
    with pytest.raises(ValueError, match="held-out"):
        train_step(network, optimizer, [row(split="test")], scale())


def test_identity_pretraining_can_initialize_new_assay_subset(tmp_path):
    pretrained = model(endpoints=(), identity_tasks=("mhc_class",))
    optimizer = torch.optim.AdamW(pretrained.parameters(), lr=3e-4)
    identity = row(
        "identity", peptide="", value=None, unit=None, identity_labels={"mhc_class": "II"}
    )
    train_step(pretrained, optimizer, [identity], {})
    path = tmp_path / "identity.pt"
    save_checkpoint(path, pretrained, {})
    extended = model(identity_tasks=("mhc_class",))
    inherited = initialize_from_checkpoint(extended, path)
    assert inherited["inherited_parameters"]
    torch.testing.assert_close(
        pretrained.encoder.embedding.weight, extended.encoder.embedding.weight
    )
    assert extended.support["ic50"]["status"] == "random/untrained"
    assert extended.support["mhc_class"]["status"] == "directly_supervised"


def test_serialization_preserves_context_and_provenance():
    original = row(
        context=Context(components={"a": Component("lost", evidence={"sample": "x"})}),
        provenance={"unknown_field": "retained"},
    )
    assert Observation.from_dict(original.to_dict()) == original
    assert ModularConfig.from_dict(asdict(ModularConfig())) == ModularConfig()


def test_auxiliary_tcr_training_needs_no_pmhc_and_keeps_actual_coreceptor_separate():
    network = model(
        endpoints=(), identity_tasks=("tcr_species", "tcr_lineage"), species=("human", "mouse")
    )
    receptor = Observation(
        "identity-tcr",
        "fixture",
        "identity",
        tcr_a="ACDEFGHIKLMN",
        tcr_b="PQRSTVWYACDE",
        identity_labels={"tcr_a": "human", "tcr_b": "human", "tcr_lineage": "CD4"},
        context=Context(coreceptor_override="CD8"),
    )
    outputs = network([receptor])
    loss, counts = batch_loss(network, [receptor], outputs, {})
    loss.backward()
    assert counts == {"tcr_species": 2, "tcr_lineage": 1}
    assert all(parameter.grad is not None for parameter in network.parameters())
    assert outputs[0].diagnostics["class"] is None


def test_cellular_component_losses_are_used_and_unknown_is_not_intact():
    network = model(
        endpoints=("ic50", "elution_observation"), components=("a", "b"), source_context=True
    )
    elution = row("elution_observation", value=1, unit="binary")
    context = Context(
        components={"a": Component("lost"), "b": Component("reduced")}, source_n="AAA"
    )
    altered = replace(elution, context=context)
    assert not torch.allclose(network([elution])[0].value, network([altered])[0].value)
    original_binding = network([row()])[0].value
    torch.testing.assert_close(original_binding, network([row(context=context)])[0].value)
    output = network([altered])[0]
    output.value.backward()
    assert network.context_fusion[0].weight.grad.norm() > 0
    assert network.ms_score.weight.grad.norm() > 0


def test_component_status_and_zero_update_are_not_mislabeled_as_trained():
    network = model(endpoints=("bulk_observation",))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0)
    observation = row("bulk_observation", mhc=(), value=1, unit="binary")
    train_step(network, optimizer, [observation], {})
    assert network.support["bulk_observation"]["status"] == "random/untrained"
    optimizer.param_groups[0]["lr"] = 3e-4
    train_step(network, optimizer, [observation], {})
    capabilities = network.capabilities()
    assert capabilities["outputs"]["ic50"]["status"] == "not_allocated"
    assert capabilities["components"]["ms_response"]["status"] == "indirectly_trained"
    assert capabilities["components"]["digest"]["status"] == "indirectly_trained"


def test_detector_only_and_digest_only_use_no_mhc():
    for endpoint in ("ms_response", "digest_observation"):
        network = model(endpoints=(endpoint,))
        observation = row(endpoint, mhc=(), value=1, unit="binary")
        optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
        step = train_step(network, optimizer, [observation], {})
        assert not coverage_summary([step])["untrained_parameters"]
        assert not hasattr(network, "pmhc_fusion")


def test_inference_units_and_candidate_library_ranking(tmp_path):
    from presto.inference.modular import ModularPredictor

    network = model(endpoints=("ic50", "tcr_binding"))
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
    receptor = row(
        "tcr_binding", tcr_a="ACDEFGHIKLMN", tcr_b="PQRSTVWYACDE", value=1, unit="binary"
    )
    other = replace(receptor, observation_id="r2", peptide="LMNPQRSTVWYACDE", value=0)
    train_step(network, optimizer, [row(), receptor, other], scale())
    path = tmp_path / "model.pt"
    save_checkpoint(path, network, scale())
    predictor = ModularPredictor(path)
    prediction = predictor.predict([row()])[0]
    assert prediction["value"] == pytest.approx(10 ** prediction["transformed_value"])
    assert prediction["unit"] == "nM"
    library = [row(), row(observation_id="r2", peptide="LMNPQRSTVWYACDE")]
    ranked = predictor.rank_pmhc(tcr_a=receptor.tcr_a, tcr_b=receptor.tcr_b, candidates=library)
    assert {item["observation_id"] for item in ranked} == {"r1", "r2"}
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_untrained_register_pooling_cannot_hide_behind_a_trained_affinity_head():
    network = model()
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)
    one_core = row(peptide="ACDEFGHIK")
    train_step(network, optimizer, [one_core], scale())
    network.predict([one_core])
    with pytest.raises(ValueError, match="register_score"):
        network.predict([row()])


def test_numeric_acquisition_features_require_declared_units_and_known_masks():
    network = model(
        endpoints=("ms_response",),
        acquisition_features=("energy",),
        acquisition_units={"energy": "eV"},
    )
    original = row("ms_response", mhc=(), value=1, unit="binary")
    measured = replace(
        original, context=Context(acquisition={"energy": 0}, acquisition_units={"energy": "eV"})
    )
    assert not torch.allclose(network([original])[0].value, network([measured])[0].value)
    wrong_units = replace(
        original,
        context=Context(acquisition={"energy": 30}, acquisition_units={"energy": "percent"}),
    )
    with pytest.raises(ValueError, match="unit/scale mismatch"):
        network([wrong_units])


def test_detector_anchor_ignores_observation_covariates_and_mhc_bags_keep_identity_scoped():
    network = model(
        endpoints=("bulk_observation", "ms_response", "elution_observation"),
        observation_features=("amount",),
        observation_units={"amount": "log10_fmol"},
        identity_tasks=("mhc_class",),
    )
    anchor = row("ms_response", mhc=(), value=1, unit="binary")
    unrelated = replace(
        anchor, context=Context(observation={"amount": 10}, observation_units={"amount": "other"})
    )
    torch.testing.assert_close(network([anchor])[0].value, network([unrelated])[0].value)
    bag = row(
        "elution_observation",
        value=1,
        unit="binary",
        mhc=(
            MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I"),
            MHCInput("NPQRSTVWYAC", "ACDEFGHIKLM", "II"),
        ),
    )
    output = network([bag])[0]
    assert output.identity == {}
    assert [candidate["mhc"]["mhc_class"] for candidate in output.diagnostics["candidates"]] == [
        "I",
        "II",
    ]


def test_bulk_pretraining_extends_to_pmhc_without_inventing_bulk_species_inputs(tmp_path):
    pretrained = model(endpoints=("bulk_observation",))
    optimizer = torch.optim.AdamW(pretrained.parameters(), lr=3e-4)
    train_step(pretrained, optimizer, [row("bulk_observation", mhc=(), value=1, unit="binary")], {})
    path = tmp_path / "bulk.pt"
    save_checkpoint(path, pretrained, {})
    extended = model(
        endpoints=("bulk_observation", "elution_observation"), species=("human", "mouse")
    )
    initialize_from_checkpoint(extended, path)
    torch.testing.assert_close(extended.ms_fusion[0].weight, pretrained.ms_fusion[0].weight)
    assert extended.capabilities()["components"]["ms_response"]["status"] == "indirectly_trained"
    assert extended.support["elution_observation"]["status"] == "random/untrained"


def test_unobserved_feature_columns_cannot_be_activated_at_inference(tmp_path):
    network = model(
        endpoints=("ms_response",),
        acquisition_features=("energy",),
        acquisition_units={"energy": "eV"},
    )
    original = row("ms_response", mhc=(), value=1, unit="binary")
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=0)
    train_step(network, optimizer, [original], {})
    assert torch.count_nonzero(network.ms_fusion[0].weight.grad[:, -2:]) == 0
    assert network.predict([original])[0].value is not None
    unseen = replace(
        original,
        context=Context(
            acquisition={"energy": 30},
            acquisition_units={"energy": "eV"},
        ),
    )
    with pytest.raises(ValueError, match="untrained input features.*energy"):
        network.predict([unseen])
    path = tmp_path / "feature-support.pt"
    save_checkpoint(path, network, {})
    restored, _, _ = load_checkpoint(path)
    with pytest.raises(ValueError, match="untrained input features.*energy"):
        restored.predict([unseen])


def test_unseen_component_state_is_not_supported_by_another_state():
    network = model(endpoints=("tcell_response",), components=("component_a",))
    original = row(
        "tcell_response",
        value=1,
        unit="binary",
        context=Context(components={"component_a": Component("lost")}),
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=0)
    train_step(network, optimizer, [original], {})
    assert network.predict([original])[0].value is not None
    unseen = replace(original, context=Context(components={"component_a": Component("intact")}))
    with pytest.raises(ValueError, match="untrained input features.*component_a.*intact"):
        network.predict([unseen])


def test_peptide_source_pretraining_is_multilabel_and_needs_no_mhc(tmp_path):
    network = model(
        endpoints=(), identity_tasks=("peptide_source",), source_categories=("human", "rodents")
    )
    sample = row(
        "identity",
        mhc=(),
        value=None,
        unit=None,
        identity_labels={
            "peptide_source:human": "present",
            "peptide_source:rodents": "present",
        },
    )
    assert not hasattr(network, "pmhc_fusion")
    with torch.no_grad():
        network.identity_heads["peptide_source"].bias.fill_(2)
    output = network([sample])[0]
    assert sum(value.softmax(-1)[1] for value in output.identity.values()) > 1
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=0)
    step = train_step(network, optimizer, [sample], {})
    assert step["counts"] == {"peptide_source": 2}
    assert not coverage_summary([step])["untrained_parameters"]
    assert network.source_label_support == {"human": {"present"}, "rodents": {"present"}}
    path = tmp_path / "source.pt"
    save_checkpoint(path, network, {})
    restored, _, _ = load_checkpoint(path)
    assert restored.predict([sample])[0].identity.keys() == output.identity.keys()


def test_source_category_without_a_label_stays_unsupported():
    network = model(
        endpoints=(), identity_tasks=("peptide_source",), source_categories=("human", "rodents")
    )
    sample = row(
        "identity",
        mhc=(),
        value=None,
        unit=None,
        identity_labels={"peptide_source:human": "present"},
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=0)
    step = train_step(network, optimizer, [sample], {})
    assert step["counts"] == {"peptide_source": 1}
    assert torch.count_nonzero(network.identity_heads["peptide_source"].weight.grad[1]) == 0
    with pytest.raises(ValueError, match="untrained peptide source outputs.*rodents"):
        network.predict([sample])


def test_chain_pairing_auxiliaries_share_encoder_without_assay_heads():
    network = model(endpoints=(), identity_tasks=("mhc_pairing", "tcr_pairing"))
    sample = row(
        "identity",
        peptide="",
        value=None,
        unit=None,
        tcr_a="ACDEFGHIKLMN",
        tcr_b="PQRSTVWYACDE",
        identity_labels={"mhc_pairing": "matched", "tcr_pairing": "matched"},
    )
    original = network([sample])[0]
    altered = network([replace(sample, tcr_a="LMNPQRSTVWYA")])[0]
    torch.testing.assert_close(original.identity["mhc_pairing"], altered.identity["mhc_pairing"])
    assert not torch.allclose(original.identity["tcr_pairing"], altered.identity["tcr_pairing"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=0)
    step = train_step(network, optimizer, [sample], {})
    assert step["counts"] == {"mhc_pairing": 1, "tcr_pairing": 1}
    assert not coverage_summary([step])["untrained_parameters"]


def test_auxiliary_overrides_keep_raw_predictions_and_respect_endpoint_masks():
    network = model(
        endpoints=("ic50", "tcr_binding", "tcr_response", "elution_observation"),
        identity_tasks=("peptide_source", "mhc_pairing", "tcr_pairing", "tcr_lineage"),
        source_categories=("human",),
        auxiliary_context=("mhc_pairing", "tcr_pairing", "tcr_lineage", "peptide_source:human"),
    )
    original = row(
        context=Context(auxiliary_values={"peptide_source:human": "present", "tcr_lineage": "CD4"})
    )
    overridden = replace(
        original,
        context=replace(
            original.context,
            auxiliary_overrides={
                "peptide_source:human": "absent",
                "tcr_lineage": "CD8",
            },
        ),
    )
    a, b = network([original, overridden])
    torch.testing.assert_close(a.value, b.value)
    auxiliary = b.diagnostics["candidates"][0]["auxiliary"]["peptide_source:human"]
    assert auxiliary["value"] == "absent" and auxiliary["supplied"] == "present"
    assert auxiliary["source"] == "override" and auxiliary["predicted"]
    for endpoint in ("tcr_binding", "tcr_response", "elution_observation"):
        inputs = [
            replace(
                item,
                endpoint=endpoint,
                value=1,
                unit="binary",
                tcr_a="ACDEFGHIKLMN",
                tcr_b="PQRSTVWYACDE",
            )
            for item in (original, overridden)
        ]
        a, b = network(inputs)
        if endpoint == "tcr_binding":
            torch.testing.assert_close(a.value, b.value)
        else:
            assert not torch.allclose(a.value, b.value)
        if endpoint == "elution_observation":
            torch.testing.assert_close(a.components["ms_response"], b.components["ms_response"])
    response = replace(
        original,
        endpoint="tcr_response",
        value=1,
        unit="binary",
        tcr_a="ACDEFGHIKLMN",
        tcr_b="PQRSTVWYACDE",
    )
    observation_loss(network([response])[0], response, {}).backward()
    assert all(parameter.grad is None for parameter in network.identity_heads.parameters())
    assert network.encoder.embedding.weight.grad.norm() > 0
