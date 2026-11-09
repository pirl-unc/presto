"""Once-only held-out evaluation after checkpoint and covariate-prediction sealing."""

import csv
import datetime
import gzip
import json

import numpy as np
from core import ROOT, Data, center, groups, metrics, sha, write


def main():
    if (ROOT / "results/test-opened.json").exists():
        raise FileExistsError("Test already opened; no evaluation retry")
    seal = json.loads((ROOT / "results/prediction-seal.json").read_text())
    for relative, digest in seal.items():
        assert sha(ROOT / relative) == digest
    data = Data()
    saved = np.load(ROOT / "results/covariate-predictions.npz")
    fitting = json.loads((ROOT / "results/models.json").read_text())
    choice = fitting["validation_choices"]
    labels_path = ROOT / "results/test-labels.json"
    inputs = json.loads((ROOT / "results/input-seal.json").read_text())
    assert sha(labels_path) == inputs["test-labels.json"]
    write(
        ROOT / "results/test-opened.json",
        {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prediction_seal_sha256": sha(ROOT / "results/prediction-seal.json"),
            "test_labels_sha256": inputs["test-labels.json"],
        },
    )
    labels = json.loads(labels_path.read_text())
    data.labels["test"] = {
        key: np.array([labels[data.rows[i]["measurement_id"]][key] for i in data.indices["test"]])
        for key in ("y", "p")
    }
    records = []
    strata = []
    component_losses = {}
    predictions = saved.files
    fields = [
        "split",
        "model",
        "measurement_id",
        "peptide_sequence",
        "group_id",
        "component_id",
        "acquisition_id",
        "weight",
        "group_weight",
        "target_centered_log10",
        "target_fraction",
        "prediction_centered_log10",
        "predicted_fraction",
    ]
    with gzip.open(ROOT / "results/predictions.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for split in ("train", "val", "test"):
            indices = data.indices[split]
            rows = [data.rows[i] for i in indices]
            y, p = data.labels[split]["y"], data.labels[split]["p"]
            _, inverse, counts = groups(rows)
            for name in predictions:
                raw = saved[name][indices]
                result, q = metrics(rows, y, p, raw)
                prediction = center(raw, inverse, counts)
                records.append({"split": split, "model": name, **result})
                for i, row in enumerate(rows):
                    writer.writerow(
                        {
                            "split": split,
                            "model": name,
                            **{key: row[key] for key in fields[2:9]},
                            "target_centered_log10": y[i],
                            "target_fraction": p[i],
                            "prediction_centered_log10": prediction[i],
                            "predicted_fraction": q[i],
                        }
                    )
                for field in ("acquisition_id", "component_id"):
                    identifiers = np.array([r[field] for r in rows])
                    for identifier in np.unique(identifiers):
                        selected = np.flatnonzero(identifiers == identifier)
                        local = [rows[i] for i in selected]
                        score, _ = metrics(local, y[selected], p[selected], raw[selected])
                        strata.append(
                            {
                                "split": split,
                                "model": name,
                                "axis": field,
                                "value": identifier,
                                "n_observations": len(selected),
                                **score,
                            }
                        )
                if split == "test":
                    component_ids = np.array([r["component_id"] for r in rows])
                    losses = []
                    for component in np.unique(component_ids):
                        ix = np.flatnonzero(component_ids == component)
                        w = data.weights[indices[ix]]
                        w = w / w.sum()
                        losses.append(float(w @ ((prediction[ix] - y[ix]) ** 2)))
                    component_losses[name] = np.array(losses)
    lookup = {(r["split"], r["model"]): r for r in records}
    gains = {
        s: 1 - lookup[s, choice["sequence"]]["rmse"] / lookup[s, choice["length"]]["rmse"]
        for s in ("val", "test")
    }
    passed = (
        all(g >= 0.05 for g in gains.values())
        and lookup["test", choice["sequence"]]["r2"] > 0
        and lookup["test", choice["sequence"]]["prediction_sd"] > 1e-12
    )
    generator = np.random.default_rng(20261024)
    samples = generator.integers(
        0, len(component_losses["uniform"]), size=(2000, len(component_losses["uniform"]))
    )
    intervals = {}
    comparisons = [(choice["sequence"], choice["length"])] + [
        (name, "uniform") for name in predictions if name == "exact" or name.startswith("reinforce")
    ]
    for candidate, control in comparisons:
        values = 1 - np.sqrt(component_losses[candidate][samples].mean(1)) / np.sqrt(
            component_losses[control][samples].mean(1)
        )
        intervals[candidate + "_vs_" + control] = {
            "rmse_relative_gain": 1
            - lookup["test", candidate]["rmse"] / lookup["test", control]["rmse"],
            "component_bootstrap_95": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    decision = {
        "primary_sequence_gate_passed": passed,
        "primary_rmse_relative_gains": gains,
        "validation_choices": choice,
        "distribution_gate": {
            name: all(
                lookup[s, name]["reverse_kl"] < lookup[s, "uniform"]["reverse_kl"]
                for s in ("val", "test")
            )
            for name in predictions
            if name == "exact" or name.startswith("reinforce")
        },
        "paired_component_bootstrap": intervals,
    }
    write(ROOT / "results/metrics.json", records)
    write(ROOT / "results/strata.json", strata)
    write(ROOT / "results/decision.json", decision)
    with (ROOT / "results/metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(decision, indent=2), flush=True)
    print(json.dumps([r for r in records if r["split"] == "test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
