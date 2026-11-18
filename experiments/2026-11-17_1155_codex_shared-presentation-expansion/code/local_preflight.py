"""Real-input software smoke check, not a predictive training experiment."""

import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import torch
from prepare import rows, sha
from presto.data.modular import Observation
from presto.data.presentation import (
    RelatedSequences,
    background_candidate,
    read_gencode_proteins,
    source_junction,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import (
    initialize_from_checkpoint,
    load_checkpoint,
    train_step,
)

root = Path(__file__).resolve().parents[3]
e = Path(__file__).resolve().parents[1]
parent = (
    root
    / "experiments/2026-11-16_1925_codex_harmonized-apc/results/parent-reference.pt"
)
assert sha(parent) == "1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a"
torch.manual_seed(42)
torch.set_num_threads(4)
start = time.monotonic()
proteins = read_gencode_proteins(
    root / "data/shared-presentation-20261117/proteins.fa.gz"
)
selected = defaultdict(list)
seen = set()
for r in rows(e / "results/local-admission/admitted.jsonl.gz"):
    cls = r["mhc"][0]["mhc_class"]
    row = Observation.from_dict(r)
    if (
        len(selected[cls]) >= 4
        or row.peptide in seen
        or len(row.peptide) > 30
        or not row.context.cell_type
    ):
        continue
    mapping = pq.read_table(
        e / "results/local-reference-mappings.parquet",
        filters=[("peptide", "=", row.peptide)],
    ).to_pylist()
    if not mapping:
        continue
    occurrences = [(x["protein_id"], x["start"]) for x in mapping]
    selected[cls].append((row, occurrences))
    seen.add(row.peptide)
    if len(selected["I"]) == len(selected["II"]) == 4:
        break
assert all(len(selected[c]) == 4 for c in ["I", "II"])
observed = set(
    pq.read_table(
        root / "data/shared-presentation-20261117/observed-identities.parquet"
    )["peptide"].to_pylist()
)
ctx = {"cell_types": set(), "cytokines": set(), "interventions": set()}
batches = {}
for cls, items in selected.items():
    batch = []
    for row, occurrences in items:
        positive_context, policy = source_junction(row.peptide, occurrences, proteins)
        negative = background_candidate(
            row.peptide,
            occurrences,
            proteins,
            observed=observed,
            protected=RelatedSequences(seen),
            key="local-software:" + row.peptide,
        )
        assert negative is not None
        p, protein, position, negative_context = negative
        if policy == "ambiguous_reference_junction":
            negative_context = positive_context
        positive = replace(
            row,
            split="train",
            group_id=occurrences[0][0],
            context=replace(row.context, **positive_context),
        )
        background = replace(
            row,
            split="train",
            group_id=occurrences[0][0],
            observation_id=row.observation_id + ":software-background",
            peptide=p,
            value=0.0,
            context=replace(row.context, **negative_context),
        )
        batch.extend([positive, background])
        ctx["cell_types"].add(row.context.cell_type)
        ctx["cytokines"].update(row.context.cytokines)
        ctx["interventions"].update(row.context.interventions)
    batches[cls] = batch
parent_model, scales, _ = load_checkpoint(parent, device="cpu")
config = parent_model.config.to_dict()
del parent_model
config.update(
    {k: sorted(v) for k, v in ctx.items()}, state_bottleneck=64, state_mode="harmonized"
)
model = ModularPresto(ModularConfig.from_dict(config))
transfer = initialize_from_checkpoint(model, parent, allow_apc_expansion=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
updates = {}
for cls, batch in batches.items():
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(3):
        result = train_step(model, optimizer, batch, scales)
    changes = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert any(n.startswith("apm_state.") for n in changes)
    assert any(n.startswith(("encoder.", "sequence_encoder.")) for n in changes), (
        changes
    )
    gradients = {
        n: v["gradient_norm"]
        for n, v in result["coverage"].items()
        if v["gradient_norm"] > 0
    }
    assert any(n.startswith("apm_state.encoder.") for n in gradients)
    assert any(n.startswith("encoder.") for n in gradients)
    updates[cls] = {
        "nonzero_gradients": gradients,
        "rows": len(batch),
        "changed_parameters": changes,
        "last_loss": result["loss"],
    }
model.eval()
with torch.no_grad():
    outputs = model(
        [replace(batches[c][0], value=None, split="predict") for c in ["I", "II"]]
    )
    assert all(torch.isfinite(o.value) for o in outputs)
assert not model.frozen_parameters
report = {
    "status": "passed",
    "kind": "real-input software smoke only; six CPU optimizer steps; checkpoint not saved or eligible for model selection",
    "class_updates": updates,
    "model_context": {k: sorted(v) for k, v in ctx.items()},
    "label_free_inference_classes": ["I", "II"],
    "frozen_parameters": list(model.frozen_parameters),
    "seconds": time.monotonic() - start,
    "test_accessed": False,
}
(e / "results/local-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
