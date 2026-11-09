"""Check position marginalization and conditioning against full span enumeration."""

import json
from pathlib import Path

import torch
from learning import estimator_check, observed_log_probs
from presto.models.peptide_availability import PeptideAvailabilityPolicy

FAMILY = Path(__file__).resolve().parents[1]
torch.manual_seed(928)
model = PeptideAvailabilityPolicy(context_size=1).double()
sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
example = {
    "context": [1.0],
    "candidates": [
        {"sequence": "ACDEF", "start_positions_0based": [0, 20]},
        {"sequence": "HIKLMN", "start_positions_0based": [6, 26]},
        {"sequence": "ACD", "start_positions_0based": [0, 20]},
    ],
}
actual, mass, _ = observed_log_probs(model, sequence, example)
state = model.prepare(sequence, [1.0])
spans = list(model.enumerate(state))
starts = torch.cat([x[0] for x in spans])
lengths = torch.cat([x[1] for x in spans])
full = torch.cat([x[2] for x in spans])
grouped = []
selected = torch.zeros_like(full, dtype=torch.bool)
for candidate in example["candidates"]:
    keep = torch.isin(starts, torch.tensor(candidate["start_positions_0based"])) & (
        lengths == len(candidate["sequence"])
    )
    selected |= keep
    grouped.append(torch.logsumexp(full[keep], 0))
expected = torch.stack(grouped).log_softmax(0)
assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
assert torch.allclose(mass, full[selected].logsumexp(0), atol=1e-12, rtol=1e-12)
weights = torch.tensor([0.4, -0.7, 0.9], dtype=torch.float64)
parameters = list(model.parameters())
a = torch.cat(
    [g.flatten() for g in torch.autograd.grad((actual * weights).sum(), parameters)]
)
b = torch.cat(
    [g.flatten() for g in torch.autograd.grad((expected * weights).sum(), parameters)]
)
error = float((a - b).abs().max())
assert error < 1e-11
q = full.detach().exp()
changed = q.clone()
changed[selected] *= 0.25
changed[~selected] *= (1 - changed[selected].sum()) / q[~selected].sum()
assert torch.allclose(changed.sum(), torch.tensor(1.0, dtype=torch.float64))
changed_group = []
for candidate in example["candidates"]:
    keep = torch.isin(starts, torch.tensor(candidate["start_positions_0based"])) & (
        lengths == len(candidate["sequence"])
    )
    changed_group.append(changed[keep].sum())
observed = torch.stack(changed_group)
observed /= observed.sum()
assert torch.allclose(observed, actual.detach().exp(), atol=1e-12, rtol=1e-12)
report = {
    "status": "passed",
    "position_marginalization_max_abs_probability_difference": float(
        (actual.detach().exp() - expected.detach().exp()).abs().max()
    ),
    "parameter_gradient_max_abs_difference": error,
    "unobserved_mass_nonidentifiability": {
        "original_selected_mass": float(q[selected].sum()),
        "changed_selected_mass": float(changed[selected].sum()),
        "observable_probabilities_unchanged": True,
    },
    "score_estimator": estimator_check(),
}
(FAMILY / "results/observation-check.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
