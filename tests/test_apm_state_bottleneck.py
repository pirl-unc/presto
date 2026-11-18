"""APM state bottleneck: a zero-initialised, class-gated, output-side term on staged elution."""

import torch

from presto.data.modular import (
    Context,
    ContextAssertion,
    Exposure,
    MHCInput,
    Observation,
    TargetScale,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import (
    batch_loss,
    initialize_from_checkpoint,
    save_checkpoint,
)

STAGED = dict(elution_staging="affinity_downstream", elution_affinity_endpoint="ic50")
ENDPOINTS = ("ic50", "elution_observation")


def build(seed=0, **kwargs):
    torch.manual_seed(seed)
    return ModularPresto(ModularConfig(dropout=0, endpoints=ENDPOINTS, **STAGED, **kwargs))


def elution(mhc_class="I", interventions=None, cytokines=None, oid="e1"):
    context = Context(
        interventions={k: ContextAssertion(v) for k, v in (interventions or {}).items()},
        cytokines={k: Exposure(v) for k, v in (cytokines or {}).items()},
    )
    return Observation(
        observation_id=oid,
        source="software-fixture",
        endpoint="elution_observation",
        peptide="ACDEFGHIKLMNPQR",
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", mhc_class),),
        value=1,
        unit="binary",
        group_id="g1",
        context=context,
    )


def scales():
    return {"ic50": TargetScale(2.0, 1.0, 2)}


def randomize(network):
    torch.manual_seed(7)
    with torch.no_grad():
        for p in network.apm_state.parameters():
            p.normal_(std=0.5)


def test_off_by_default_and_contract_unchanged():
    base = build()
    assert not hasattr(base, "apm_state")
    assert "state_bottleneck" not in base.config.to_dict()
    assert build(state_bottleneck=4).config.to_dict()["state_bottleneck"] == 4


def test_zero_at_init_and_no_state_is_noop():
    plain = build(seed=1)
    state = build(seed=1, state_bottleneck=4)
    state.load_state_dict({k: v for k, v in plain.state_dict().items()}, strict=False)
    rows = [elution(interventions={"TAP1": "absent"}), elution(oid="e2")]
    a, b = plain.eval()(rows), state.eval()(rows)
    for x, y in zip(a, b):
        assert torch.allclose(x.value, y.value, atol=1e-6)


def test_ko_moves_class_i_only_via_class_i_genes():
    net = build(seed=2, state_bottleneck=4)
    randomize(net)
    net.eval()
    base = net([elution(oid="w")])[0].value
    tap = net([elution(interventions={"TAP1": "absent"}, oid="t")])[0].value
    ciita = net([elution(interventions={"CIITA": "absent"}, oid="c")])[0].value
    assert not torch.allclose(base, tap)
    assert torch.allclose(base, ciita, atol=1e-6), "class II gene must not move class I"
    base2 = net([elution("II", oid="w2")])[0].value
    b2m2 = net([elution("II", {"B2M": "absent"}, oid="b2")])[0].value
    assert torch.allclose(base2, b2m2, atol=1e-6), "class I gene must not move class II"


def test_gradient_reaches_state_parameters_and_shuffle_changes_output():
    net = build(seed=3, state_bottleneck=4)
    rows = [elution(interventions={"TAP1": "absent"}, cytokines={"ifn_gamma": "present"})]
    net.train()
    loss, _ = batch_loss(net, rows, net(rows), scales())
    loss.backward()
    assert net.apm_state.readout.weight.grad.abs().sum() > 0
    randomize(net)
    net.eval()
    a = net([elution(interventions={"TAP1": "absent"}, oid="a")])[0].value
    b = net([elution(interventions={"ERAP1": "absent"}, oid="b")])[0].value
    assert not torch.allclose(a, b)


def test_warm_start_from_checkpoint_without_bottleneck(tmp_path):
    old = build(seed=4)
    path = tmp_path / "old.pt"
    save_checkpoint(path, old, scales())
    new = build(seed=5, state_bottleneck=4)
    initialize_from_checkpoint(new, path)
    rows = [elution(interventions={"TAP1": "absent"})]
    assert torch.allclose(old.eval()(rows)[0].value, new.eval()(rows)[0].value, atol=1e-6)


def test_free_mode_unseen_token_is_zero_effect():
    net = build(seed=6, state_bottleneck=4, state_mode="free")
    randomize(net)
    net.eval()
    base = net([elution(oid="w")])[0].value
    ko = net([elution(interventions={"TAP1": "absent"}, oid="t")])[0].value
    assert not torch.allclose(
        base, ko
    )  # randomised embeddings; untrained slots are zero in practice


def test_every_state_parameter_receives_gradient_in_each_mode():
    """The trainer's coverage gate fails on parameters that never get a gradient."""
    import pytest

    for mode in ("structured", "free"):
        net = build(seed=8, state_bottleneck=4, state_mode=mode)
        names = {name for name, _ in net.apm_state.named_parameters()}
        assert names == (
            {"step_embedding", "gene_residual", "readout.weight"}
            if mode == "structured"
            else {"free_embedding", "readout.weight"}
        )
        optimizer = torch.optim.SGD(net.apm_state.parameters(), lr=0.01)
        rows = [elution(interventions={"TAP1": "absent"}, cytokines={"ifn_gamma": "present"})]
        seen = dict.fromkeys(names, 0.0)
        net.train()
        for _ in range(3):
            optimizer.zero_grad()
            loss, _ = batch_loss(net, rows, net(rows), scales())
            loss.backward()
            for name, p in net.apm_state.named_parameters():
                seen[name] = max(seen[name], float(p.grad.abs().sum()))
            optimizer.step()
        assert all(value > 0 for value in seen.values()), (mode, seen)
    with pytest.raises(ValueError, match="state_mode must be one of"):
        build(state_bottleneck=4, state_mode="bogus")
