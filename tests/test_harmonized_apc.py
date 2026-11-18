"""Shared condition routing, frozen affinity and genuine parameter-update checks."""

from dataclasses import replace
import pytest
import torch

from presto.data.modular import Context, ContextAssertion, Component, Exposure, Observation
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import initialize_from_checkpoint, save_checkpoint
from presto.models.modular_dependencies import parameter_dependencies
from test_apm_state_bottleneck import elution, scales, STAGED, ENDPOINTS


def build(seed=0, **kwargs):
    torch.manual_seed(seed)
    config = dict(
        dropout=0,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        endpoints=ENDPOINTS,
        state_bottleneck=4,
        state_mode="harmonized",
        interventions=("knockout:B2M", "knockout:TAP1", "transduction:CIITA", "drug:decitabine"),
        **STAGED,
    )
    config.update(kwargs)
    return ModularPresto(ModularConfig(**config)).eval()


def randomize(net):
    with torch.no_grad():
        for p in net.apm_state.parameters():
            p.normal_(std=0.3)


def test_central_adapter_namespaced_factors_have_model_effects():
    from presto.data.modular_hitlist_ms import source_condition, study_factor_vocabulary

    raw = dict(
        pmid="fixture",
        condition_id="ko",
        condition_status="annotated",
        condition_evidence="curated_text",
        sample_attribution="discriminated",
        condition_knockout_genes="B2M",
        condition_drugs="decitabine",
    )
    context = source_condition(raw, study_factor_vocabulary([raw])["fixture"])
    assert context.interventions["knockout:B2M"].state == "present"
    net = build()
    randomize(net)
    a, b = net([elution(), replace(elution(), context=context)])
    assert not torch.allclose(a.value, b.value)
    roundtrip = Observation.from_dict(replace(elution(), context=context).to_dict())
    assert roundtrip.context == context


def test_class_masks_cannot_be_bypassed_by_flat_context():
    net = build()
    randomize(net)
    for mhc_class, factor in (
        ("I", "transduction:CIITA"),
        ("II", "knockout:B2M"),
        ("II", "knockout:TAP1"),
    ):
        a, b = net([elution(mhc_class), elution(mhc_class, {factor: "present"})])
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)
    a, b = net([elution("II"), elution("II", {"transduction:CIITA": "present"})])
    assert not torch.allclose(a.value, b.value)
    assert not any(
        x.startswith(("intervention:", "cytokine:", "cell_type:", "component:"))
        for x in net.feature_layout["elution_processing.0.weight"][1]
    )


def test_mhc_dependent_state_selection_and_registers_preserve_affinity():
    net = build()
    randomize(net)
    row = elution("II")
    other = replace(row, mhc=(replace(row.mhc[0], groove1="YYYYYYYYYYY", groove2="AAAAAAAAAAA"),))

    def treated(r):
        return replace(
            r, context=Context(interventions={"transduction:CIITA": ContextAssertion("present")})
        )

    pred = net([row, treated(row), other, treated(other)])
    assert (
        abs(float(((pred[1].value - pred[0].value) - (pred[3].value - pred[2].value)).detach()))
        > 1e-7
    )
    # Use the actual diagnostic structure exposed by Prediction.
    affinity_rows = [
        replace(r, endpoint="ic50", unit="nM", value=100.0) for r in [row, treated(row)]
    ]
    a, b = net(affinity_rows)
    torch.testing.assert_close(a.value, b.value, rtol=0, atol=0)


def test_sources_share_encoder_and_measured_expression_stays_distinct():
    net = build(
        components=("B2M",),
        component_activity_units={"B2M": "TPM"},
        cell_types=("dendritic",),
        cytokines=("IFNG",),
        source_context=True,
    )
    randomize(net)
    ctx = Context(
        cell_type="dendritic",
        components={"B2M": Component("reference", 10.0, "TPM", {"source": "reference RNA"})},
        cytokines={"IFNG": Exposure("present")},
        interventions={"knockout:B2M": ContextAssertion("present")},
        source_n="ACDEF",
        source_c="GHIKL",
    )
    row = replace(elution(), context=ctx)
    before = net([row])[0].value
    after = net([replace(row, context=replace(ctx, source_n="YYYYY"))])[0].value
    assert not torch.allclose(before, after)
    assert "activity:B2M:reference" in net.feature_layout["apm_state.encoder.weight"][1]


def test_warmstart_roundtrip_dependency_and_training(tmp_path):
    old = build(state_bottleneck=0, interventions=())
    path = tmp_path / "base.pt"
    save_checkpoint(path, old, scales())
    net = build(seed=3)
    with pytest.raises(ValueError, match="feature contract"):
        initialize_from_checkpoint(net, path)
    initialize_from_checkpoint(net, path, allow_apc_expansion=True)
    rows = [
        elution("I", {"knockout:B2M": "present"}),
        elution("II", {"transduction:CIITA": "present"}),
    ]
    for a, b in zip(old(rows), net(rows)):
        torch.testing.assert_close(a.value, b.value, rtol=0, atol=1e-6)
    presentation = parameter_dependencies(net, "elution_observation")
    affinity = parameter_dependencies(net, "ic50")
    names = {n for n, _ in net.named_parameters() if n.startswith("apm_state.")}
    assert names <= presentation and not names.intersection(affinity)
    before = {n: p.detach().clone() for n, p in net.apm_state.named_parameters()}
    opt = torch.optim.Adam(net.apm_state.parameters(), lr=0.01)
    for _ in range(4):
        opt.zero_grad()
        logits = torch.stack([p.value for p in net(rows)])
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.0, 1.0])
        ).backward()
        opt.step()
    assert all(not torch.equal(before[n], p) for n, p in net.apm_state.named_parameters())
    restored = build()
    restored.load_state_dict(net.state_dict())
    for a, b in zip(net(rows), restored(rows)):
        torch.testing.assert_close(a.value, b.value)


def test_ko_and_rescue_are_separate_concurrent_factors():
    from presto.data.modular_hitlist_ms import source_condition

    raw = dict(
        condition_status="annotated",
        condition_evidence="curated_text",
        condition_id="rescue",
        sample_attribution="elution_conditions",
        condition_knockout_genes="TAPBPR",
        condition_transduction="TAPBPR",
        condition_genetic_variants="TAPBPR:M29",
    )
    ctx = source_condition(raw, {})
    assert set(ctx.interventions) == {
        "knockout:TAPBPR",
        "transduction:TAPBPR",
        "genetic_variant:TAPBPR:M29",
    }


def test_harmonized_requires_inputs():
    with pytest.raises(ValueError, match="allocated context"):
        build(interventions=())


def test_saved_checkpoint_reloads_shared_state(tmp_path):
    from presto.training.modular import load_checkpoint

    net = build()
    randomize(net)
    path = tmp_path / 'harmonized.pt'
    save_checkpoint(path, net, scales())
    restored, _, _ = load_checkpoint(path)
    restored.eval()
    rows = [elution('II', {'transduction:CIITA': 'present'})]
    torch.testing.assert_close(net(rows)[0].value, restored(rows)[0].value, rtol=0, atol=0)


def test_expansion_cannot_discard_existing_context_weights(tmp_path):
    path = tmp_path / 'contextual-parent.pt'
    save_checkpoint(path, build(state_bottleneck=0), scales())
    with pytest.raises(ValueError, match='context-unallocated parent'):
        initialize_from_checkpoint(build(), path, allow_apc_expansion=True)
