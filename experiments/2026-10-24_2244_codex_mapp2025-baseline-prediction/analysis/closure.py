"""Independent decision/bootstrap replay, declared accession-token provenance, environment."""

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def token(value):
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:sp|tr)\|([^|]+)\|[^|]+", value)
    if match:
        return match[1]
    return value if re.fullmatch(r"[A-Z][A-Z0-9]{5,9}(?:-\d+)?", value) else None


def main():
    frame = pd.read_csv(ROOT / "results/predictions.csv.gz", float_precision="round_trip")
    frame = frame[frame.split == "test"]
    declared = json.loads((ROOT / "results/decision.json").read_text())
    per_component = {}
    for name, rows in frame.groupby("model", sort=False):
        losses = []
        for component, group in rows.groupby("component_id", sort=True):
            losses.append(
                float(
                    np.average(
                        (group.prediction_centered_log10 - group.target_centered_log10) ** 2,
                        weights=group.weight,
                    )
                )
            )
        per_component[name] = np.array(losses)
    generator = np.random.default_rng(20261024)
    n = len(per_component["uniform"])
    resampled = generator.integers(0, n, size=(2000, n))
    errors = []
    for name, record in declared["paired_component_bootstrap"].items():
        model, reference = name.split("_vs_")
        gains = 1 - np.sqrt(per_component[model][resampled].mean(1)) / np.sqrt(
            per_component[reference][resampled].mean(1)
        )
        interval = np.percentile(gains, [2.5, 97.5])
        errors.extend(abs(interval - np.array(record["component_bootstrap_95"])))
    assert max(errors) < 1e-10
    metric = {
        (r["split"], r["model"]): r for r in json.loads((ROOT / "results/metrics.json").read_text())
    }
    choices = declared["validation_choices"]
    gains = {
        split: 1
        - metric[split, choices["sequence"]]["rmse"] / metric[split, choices["length"]]["rmse"]
        for split in ("val", "test")
    }
    primary = (
        all(g >= 0.05 for g in gains.values())
        and metric["test", choices["sequence"]]["r2"] > 0
        and metric["test", choices["sequence"]]["prediction_sd"] > 1e-12
    )
    assert primary == declared["primary_sequence_gate_passed"]
    for name, passed in declared["distribution_gate"].items():
        assert passed == all(
            metric[split, name]["reverse_kl"] < metric[split, "uniform"]["reverse_kl"]
            for split in ("val", "test")
        )
    current = json.loads((ROOT / "results/covariates.json").read_text())
    overlap = {}
    paths = {
        "HAP1": "data/hap1_response_recipes/2026-10-24_0730_v1/source/references.parquet",
        "lymph_DC": "data/lymph_composition_recipes/2026-10-24_1938_v1/source_tables/references.parquet",
    }
    for name, relative in paths.items():
        path = REPO / relative
        raw = pq.read_table(path, columns=["accession_raw"]).column(0).to_pylist()
        ids = {token(x) for x in raw} - {None}
        overlap[name] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_source_identifiers": len(ids),
            "by_split": {
                split: len(
                    {token(r["parent_accession"]) for r in current if r["split"] == split} & ids
                )
                for split in ("train", "val", "test")
            },
        }
    report = {
        "decision_and_bootstrap_passed": True,
        "bootstrap_maxabs": float(max(errors)),
        "accession_rule": "exact sp|accession|entry or tr|accession|entry field extraction; bare accession retained; no isoform collapse or aliases",
        "accession_inventory_overlap": overlap,
        "timing": "postfit provenance only; no source split, model or eligibility changes",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "torch", "pandas", "pyarrow", "networkx", "psutil"]
            },
        },
    }
    (ROOT / "results/closure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
