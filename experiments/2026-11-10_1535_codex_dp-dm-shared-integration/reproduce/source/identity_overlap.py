"""Identity-only DP/DM overlap against pinned presentation and Atlas indexes."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PRESENTATION_SHA = "0d95780f2c918ed73fb6d0a8b1a0d6dafcaf45f32ab7b49340bc7ad08b530df0"
ATLAS_SHA = "8332c0a8181416f3d51007708723e48fe12f4f8c13b03eef17b9788200ac3e27"
EXPORT_SHA = "b7ac777432942bab9f56f72c840876d125a26acba8048936ddcf2616a4b91d00"


def core8(peptide):
    return {peptide[i : i + 8] for i in range(len(peptide) - 7)}


def check(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Input identity changed: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presentation", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    for path, expected in (
        (args.presentation, PRESENTATION_SHA),
        (args.atlas, ATLAS_SHA),
        (args.export, EXPORT_SHA),
    ):
        check(path, expected)
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = pd.read_parquet(
        args.export, columns=["peptide", "sample_label", "evidence_row_id"]
    )
    peptides = set(candidates.peptide)
    candidate_cores = {p: core8(p) for p in peptides}
    wanted_cores = set().union(*candidate_cores.values())
    exact = defaultdict(set)
    cores = defaultdict(set)
    counts = Counter()
    source_rows = Counter()
    for name, path, columns in (
        (
            "presentation",
            args.presentation,
            [
                "partition",
                "peptide",
                "study_id",
                "group_id",
                "sample_id",
                "original_measurement_id",
            ],
        ),
        ("atlas", args.atlas, ["fold", "peptide"]),
    ):
        for batch in pq.ParquetFile(path).iter_batches(
            columns=columns, batch_size=50000
        ):
            for row in batch.to_pylist():
                partition = row["partition"] if name == "presentation" else row["fold"]
                group = f"{name}:{partition}"
                peptide = row["peptide"]
                counts[group] += 1
                if peptide in peptides:
                    exact[group].add(peptide)
                cores[group].update(core8(peptide) & wanted_cores)
                if name == "presentation" and any(
                    token in str(row[field])
                    for token in ("PXD017154", "33025005")
                    for field in columns
                    if field not in {"peptide", "partition"}
                ):
                    source_rows[group] += 1
    membership = candidates.copy()
    summary = {}
    protected = []
    for group in sorted(counts):
        exact_mask = candidates.peptide.isin(exact[group])
        overlapping = {p for p in peptides if candidate_cores[p] & cores[group]}
        core_mask = candidates.peptide.isin(overlapping)
        membership[group + ":exact"] = exact_mask
        membership[group + ":core8"] = core_mask
        if group in {
            "presentation:consumed_development",
            "atlas:validation",
            "atlas:test",
        }:
            protected.append(core_mask | exact_mask)
        summary[group] = {
            "n_index_rows": counts[group],
            "n_exact_overlap_candidate_rows": int(exact_mask.sum()),
            "n_exact_overlap_candidate_peptides": len(exact[group]),
            "n_core8_overlap_candidate_rows": int(core_mask.sum()),
            "n_core8_overlap_candidate_peptides": len(overlapping),
            "n_source_token_matches": source_rows[group]
            if group.startswith("presentation:")
            else None,
        }
    assert set(counts) == {
        "presentation:train",
        "presentation:consumed_development",
        "atlas:train",
        "atlas:validation",
        "atlas:test",
    }
    embargo = protected[0]
    for mask in protected[1:]:
        embargo |= mask
    membership["protected_presentation_or_atlas_overlap"] = embargo
    membership["training_eligible"] = False
    membership.to_parquet(
        args.output / "candidate-identity-overlaps.parquet", index=False
    )
    result = {
        "status": "presentation_and_atlas_identity_audit_complete",
        "inputs": {
            str(args.presentation): PRESENTATION_SHA,
            str(args.atlas): ATLAS_SHA,
            str(args.export): EXPORT_SHA,
        },
        "cohorts": summary,
        "n_candidate_rows": len(candidates),
        "n_candidate_peptides": len(peptides),
        "n_candidate_rows_overlapping_protected_identities": int(embargo.sum()),
        "n_candidate_rows_clear_of_these_protected_identities": int((~embargo).sum()),
        "n_candidate_peptides_clear_of_these_protected_identities": candidates.loc[
            ~embargo, "peptide"
        ].nunique(),
        "n_training_eligible_rows": 0,
        "outcomes_or_predictions_accessed": False,
        "scope_limit": "Presentation and Atlas identities only; quantitative-affinity and other reserved cohort identities still require reconciliation before training. TRAIN identity overlap does not establish independent new measurement or label conflict.",
        "seconds": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
