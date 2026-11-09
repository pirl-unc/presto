"""Post-evaluation covariate-only provenance audit; does not change any fit."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def features(seq):
    return (
        [len(seq), len(seq) ** 2]
        + [seq.count(a) / len(seq) for a in AA]
        + [int(seq[0] == a) for a in AA]
        + [int(seq[-1] == a) for a in AA]
    )


def main():
    source = REPO / "data/measured_ms1_grouped_observer/2026-10-24_1702_canonical"
    admission = json.loads((ROOT / "results/independent-admission.json").read_text())
    rows = json.loads(
        (
            Path(admission["prepared_directory"]) / "human_native/covariates.json"
        ).read_text()
    )
    old = {}
    hashes = {}
    for split, name in [
        ("train", "train.jsonl"),
        ("validation", "validation.jsonl"),
        ("test", "test-covariates.jsonl"),
    ]:
        path = source / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        old[split] = {
            json.loads(line)["sequence"] for line in path.read_text().splitlines()
        }
    raw = np.array([features(s) for s in sorted(old["train"])])
    mask = np.load(
        REPO
        / "experiments/2026-10-24_1702_codex_ms1-grouped-observer/results/fit/sequence_0.1.npz"
    )["mask"]
    results = {}
    for split in ("train", "val", "test"):
        new = [r["sequence"] for r in rows if r["split"] == split]
        f = np.array([features(s) for s in new])
        outside = ((f < raw.min(0)) | (f > raw.max(0)))[:, mask].any(1)
        results[split] = {
            "n_rows": len(new),
            "n_exact_sequence_overlap": {
                s: len(set(new) & seqs) for s, seqs in old.items()
            },
            "n_outside_any_active_observer_training_feature_range": int(outside.sum()),
            "length_range": [min(map(len, new)), max(map(len, new))],
        }
    result = {
        "scope": "Post-evaluation covariate-only diagnostic; no new fit or exclusion",
        "observer_train_length_range": [int(raw[:, 0].min()), int(raw[:, 0].max())],
        "source_hashes": hashes,
        "cohorts": results,
    }
    output = ROOT / "results/observer-exposure.json"
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
