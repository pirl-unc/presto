"""Paired counterfactual state-contrast loss."""

import torch

from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import train_step
from presto.training.modular_controls import state_contrast_loss

STAGED = dict(elution_staging="affinity_downstream", elution_affinity_endpoint="ic50")


def elution(oid, peptide, label, state=None):
    prov = {}
    if state is not None:
        prov = {"contrast_state": state, "contrast_label": label}
    return Observation(
        observation_id=oid,
        source="software-fixture",
        endpoint="elution_observation",
        peptide=peptide,
        mhc=(MHCInput("ACDEFGHIKLM", "NPQRSTVWYAC", "I"),),
        value=1.0,
        unit="binary",
        group_id="g",
        context=Context(),
        provenance=prov,
    )


def build(seed=0, **kw):
    torch.manual_seed(seed)
    return ModularPresto(
        ModularConfig(dropout=0, endpoints=("ic50", "elution_observation"), **STAGED, **kw)
    )


def test_ignored_without_bottleneck_or_without_contrast_fields():
    state = {"interventions": {"TAP1": "absent"}}
    rows = [elution("a", "ACDEFGHIK", 1, state)]
    assert state_contrast_loss(build(), rows) == (None, 0)
    assert state_contrast_loss(build(state_bottleneck=4), [elution("b", "ACDEFGHIK", 1)]) == (
        None,
        0,
    )


def test_contrast_loss_trains_the_state_shift():
    net = build(seed=1, state_bottleneck=4)
    state = {"interventions": {"TAP1": "absent"}}
    ko = [elution(f"k{i}", p, 1, state) for i, p in enumerate(("ACDEFGHIK", "ACDEFGHIL"))]
    wt = [elution(f"w{i}", p, 0, state) for i, p in enumerate(("KLMNPQRST", "KLMNPQRSV"))]
    rows = ko + wt
    for r in rows:
        object.__setattr__(r, "split", "train")
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    scales = {"ic50": TargetScale(2.0, 1.0, 2)}
    first = None
    for _ in range(30):
        out = train_step(net, opt, rows, scales, state_contrast={"weight": 1.0})
        first = first if first is not None else out["loss_parts"]["state_contrast"]
    assert out["loss_parts"]["state_contrast_rows"] == 4
    assert out["loss_parts"]["state_contrast"] < first
