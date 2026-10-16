"""Verify complete original outputs and recompute held-out probability metrics."""

import csv
import hashlib
import io
import json
import math
import sys
import tarfile
from pathlib import Path

family = Path(__file__).resolve().parents[1]
receipt = json.loads((family / "results/sampling.json").read_text())
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(receipt["local_paths"]["run.tar.gz"]).parent
for name, pin in receipt["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == pin["size_bytes"]
    assert hashlib.sha256(data).hexdigest() == pin["sha256"]
inventory = json.loads((root / "inventory.json").read_text())
with tarfile.open(root / "run.tar.gz") as archive:
    files = {}
    for member in archive.getmembers():
        assert member.isfile() and member.name in inventory and member.name not in files
        data = archive.extractfile(member).read()
        assert len(data) == inventory[member.name]["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == inventory[member.name]["sha256"]
        files[member.name] = data
    assert set(files) == set(inventory)

selection = json.loads(files["selection.json"])
metrics = []
exposure = []
seen = {}
for condition in ("balanced", "shuffled"):
    for name in ("comparison.json", "comparison.csv", "uncertainty.json", "regularization.json"):
        (family / "results" / (condition + "-" + name)).write_bytes(files[condition + "/" + name])
    updates = json.loads(files[condition + "/output_updates.json"])
    row = next(x for x in updates["outputs"] if x["endpoint"] == "ms_detectability_logit")
    exposure.append({"condition": condition, "optimizer_steps": updates["optimizer_steps"], **row})
    assert updates["optimizer_steps"] == 200
    assert row["batches_with_output_gradient"] == 200
    assert row["nonfinite_output_gradient_elements"] == 0
    assert row["batches_with_direct_labels"] == 0
    regularization = json.loads(files[condition + "/regularization.json"])
    assert all(
        value == 0 for key, value in regularization["base"].items() if key.endswith("_weight")
    )
    comparison = json.loads(files[condition + "/comparison.json"])
    for split in ("val", "test"):
        predictions = list(
            csv.DictReader(
                io.StringIO(files[f"{condition}/{split}_comparison.tsv"].decode()), delimiter="\t"
            )
        )
        assert len(predictions) == 1000
        keys = {
            (x["sample_id"], x["detectability_label"], x["source_partition"]) for x in predictions
        }
        assert len(keys) == 1000
        assert {x["source_partition"] for x in predictions} == {split}
        if split in seen:
            assert keys == seen[split]
        seen[split] = keys
        for predictor in ("presto", "train_prevalence", "length_composition_logistic"):
            expected = next(
                x for x in comparison if x["split"] == split and x["predictor"] == predictor
            )
            pairs = [
                (float(x["detectability_label"]), float(x[predictor + "_probability"]))
                for x in predictions
            ]
            loss = sum(
                -y * math.log(max(p, 1e-15)) - (1 - y) * math.log(max(1 - p, 1e-15))
                for y, p in pairs
            ) / len(pairs)
            brier = sum((p - y) ** 2 for y, p in pairs) / len(pairs)
            assert math.isclose(loss, expected["log_loss"], rel_tol=1e-10, abs_tol=1e-12)
            assert math.isclose(brier, expected["brier"], rel_tol=1e-10, abs_tol=1e-12)
            metrics.append({"condition": condition, **expected})
    for name in (
        "assignments.parquet",
        "membership.parquet",
        "membership.tsv",
        "parent_graph.parquet",
        "policy.json",
        "source_registry.json",
        "leakage_audit.json",
    ):
        assert files[f"{condition}/membership/{name}"] == files[f"preflight/membership/{name}"]
assert (
    selection["selected_condition"]
    == min(
        (x for x in metrics if x["predictor"] == "presto" and x["split"] == "val"),
        key=lambda x: x["log_loss"],
    )["condition"]
)
summary = {
    "schema_version": 1,
    "n_verified_run_files": len(files),
    "selection": selection,
    "probability_metrics_recomputed_from_predictions": True,
    "n_held_out_predictions_per_condition": 2000,
    "identical_held_out_ids_and_labels_across_conditions": True,
    "identical_frozen_membership_across_conditions": True,
    "metrics": metrics,
    "exposure": exposure,
    "decision": "Shuffled sampling improves probability quality over balanced sampling and prevalence; length/composition logistic remains preferred on validation. No readiness claim or global default change.",
    "limitations": [
        "One search, overlapping proteins, donor/specimen independence unknown.",
        "One seed, short CPU fits, proxy targets only.",
        "Repeated test outcomes are exploratory; selection uses validation only.",
        "Bootstrap sampling unit is peptide, not independent biological material.",
        "Original shared-volume projection attempt stopped before fitting; preserved separately.",
    ],
}
(family / "results/qualification-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
(family / "results/selection.json").write_bytes(files["selection.json"])
with (family / "results/comparison.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
print(
    json.dumps(
        {
            "n_verified_files": len(files),
            "selected": selection["selected_condition"],
            "positive_exposure": {
                r["condition"]: [r["positive_labels_seen"], r["label_observations_seen"]]
                for r in exposure
            },
        }
    )
)
