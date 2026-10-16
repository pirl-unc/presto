"""Verify the frozen run and bind fixed IC50 requirements for a future fit."""

import csv
import hashlib
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

from datacache import materialize

FAMILY = Path(__file__).resolve().parents[1]


def pin(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    receipt_path = FAMILY / "results/preflight.json"
    receipt = json.loads(receipt_path.read_text())
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(receipt["local_paths"]["run.tar.gz"]).parent
    )
    assert receipt["summary"]["status"] == "success"
    for name, expected in receipt["files"].items():
        assert pin(root / name) == {k: expected[k] for k in ("sha256", "size_bytes")}
    inventory = json.loads((root / "inventory.json").read_text())
    files = {}
    with tarfile.open(root / "run.tar.gz") as archive:
        for member in archive.getmembers():
            assert member.isfile() and member.name in inventory and member.name not in files
            data = archive.extractfile(member).read()
            expected = inventory[member.name]
            assert len(data) == expected["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == expected["sha256"]
            files[member.name] = data
    assert set(files) == set(inventory)
    candidate = json.loads(files["candidate-support.json"])
    assert candidate == json.loads((FAMILY / "reproduce/candidate-support.json").read_text())
    invocation = json.loads(files["invocation.json"])
    assert invocation["dirty"] is False
    assert (
        hashlib.sha256(files["candidate-support.json"]).hexdigest()
        == invocation["candidate_support_sha256"]
    )
    assert invocation["installed_producer"] == json.loads(files["producer.json"])
    coverage = json.loads(files["output_coverage.json"])
    canonical = json.loads(files["supported_outputs.json"])
    assert canonical["mode"] == "exploratory"
    assert all(row["status"] == "undeclared" for row in canonical["outputs"])
    minima = {
        "unique_observations": candidate["minimum_unique_observations"],
        "distinct_peptides": candidate["minimum_unique_peptides"],
        "unique_exact": candidate["minimum_unique_exact_observations"],
    }
    claim = {
        "endpoint": candidate["output"],
        "columns": candidate["columns"],
        "required_splits": candidate["required_splits"],
        "evidence_roles": [candidate["evidence_role"]],
        "source_families": [candidate["evidence_family"]],
        "raw_sources": candidate["raw_sources"],
        "minimums": minima,
    }
    prospective = {
        "schema_version": 1,
        "model_contract": coverage["contract"]["configuration"],
        "source_contract": coverage["source_contract"],
        "claims": [claim],
    }
    support_rows, reasons = [], []
    for split in candidate["required_splits"]:
        entry = coverage["splits"][split]["outputs"][candidate["output"]]
        # Aggregate distinct counts are valid only if every evidence group is admitted.
        # Summing distinct-peptide counts across raw sources could double count peptides.
        admitted = all(
            e["role"] == candidate["evidence_role"]
            and e["family"] == candidate["evidence_family"]
            and e["source"] in candidate["raw_sources"]
            for e in entry["evidence"]
        )
        if not admitted:
            reasons.append(
                f"{split}: aggregate includes unadmitted evidence; canonical census needed"
            )
            support_rows.append({"split": split, "counts_available": False})
            continue
        counts = {k: entry[k] for k in minima}
        support_rows.append({"split": split, "counts_available": True, **counts})
        reasons.extend(
            f"{split}:{key}={counts[key]} requires >= {minimum}"
            for key, minimum in minima.items()
            if counts[key] < minimum
        )
    candidate_report = {
        "candidate": candidate,
        "counts": support_rows,
        "fixed_candidate_minima_met": not reasons,
        "reasons": reasons,
        "canonical_manifest_gate_executed": False,
        "interpretation": "Count evidence for the next prospective fit; no trained-output claim.",
    }
    funnel = json.loads(files["data_funnel.json"])
    audit = json.loads(files["frozen_assignments/lineage_audit.json"])
    rows = list(csv.DictReader(io.StringIO(files["split_support.csv"].decode())))
    summary = {
        "schema_version": 1,
        "status": "data preflight passed; output performance not established",
        "n_verified_run_files": len(files),
        "n_undeclared_outputs": len(canonical["outputs"]),
        "source_counts_before_cap": funnel["source_loader"]["counts_before_cap"],
        "source_counts_after_cap": funnel["source_loader"]["counts"],
        "source_loader": funnel["source_loader"],
        "drop_reasons": funnel["drop_reasons"],
        "partition_coverage": json.loads(files["frozen_membership/coverage.json"]),
        "supervision_counts": {
            split: {
                row["target"]: {
                    k: int(row[k])
                    for k in (
                        "count",
                        "exact",
                        "censored_lower",
                        "censored_upper",
                        "positive",
                        "negative",
                    )
                    if row[k]
                }
                for row in rows
                if row["split"] == split
            }
            for split in ("train", "val", "test")
        },
        "overlaps": [
            {
                "left": pair["left"],
                "right": pair["right"],
                "n_shared_values": {k: v["n_shared_values"] for k, v in pair["overlaps"].items()},
            }
            for pair in audit["pairs"]
        ],
        "lineage_coverage": audit["partitions"],
        "candidate_support": candidate_report,
        "limitations": [
            "No model constructed; no predictions, optimization or performance metrics exist.",
            "Every canonical output remains undeclared; the future manifest gate has not run.",
            "Positive-only elution and human-source-only foreignness do not qualify binary discrimination.",
            "Unknown origin/donor/specimen/acquisition identities cannot establish independence.",
            "Shared studies are reported; peptide-disjoint is not study or biological independence.",
            "Source export dates remain unverified; exact original bytes and receipts are preserved.",
            "Source generation, reservoir selection and allocation all change from the original baseline; this is not a causal ablation.",
        ],
    }
    report_names = [
        "data_funnel.json",
        "data_funnel.csv",
        "split_support.csv",
        "mhc_sequence_coverage.json",
        "mhc_sequence_coverage_post_filter.json",
        "frozen_assignments/coverage.json",
        "frozen_assignments/lineage_audit.json",
        "frozen_membership/coverage.json",
        "frozen_membership/model_peptide_audit.json",
        "frozen_membership/root_lineage_audit.json",
    ]
    output_names = [name.replace("/", "-") for name in report_names] + [
        "qualification-summary.json",
        "candidate-support-counts.json",
        "prospective-supported-outputs.json",
    ]

    def builder(inputs, outputs):
        for name in report_names:
            Path(outputs[name.replace("/", "-")]).write_bytes(files[name])
        write(outputs["qualification-summary.json"], summary)
        write(outputs["candidate-support-counts.json"], candidate_report)
        write(outputs["prospective-supported-outputs.json"], prospective)

    paths = materialize(
        Path.home() / ".hitlist/presto-corrected-human-qualification/analysis",
        {
            name: {"path": str(path), "sha256": pin(path)["sha256"]}
            for name, path in {
                "run.tar.gz": root / "run.tar.gz",
                "inventory.json": root / "inventory.json",
                "receipt.json": receipt_path,
                "recipe.py": Path(__file__),
                "candidate-support.json": FAMILY / "reproduce/candidate-support.json",
            }.items()
        },
        transform={"version": "corrected-human-preflight-analysis-v1"},
        outputs={name: {} for name in output_names},
        builder=builder,
    )
    for name, path in paths.items():
        shutil.copyfile(path, FAMILY / "results" / name)
    write(
        FAMILY / "results/analysis-receipt.json",
        {name: {"path": str(path), **pin(path)} for name, path in paths.items()},
    )
    print(
        json.dumps(
            {"n_verified_run_files": len(files), "candidate_support": candidate_report}, indent=2
        )
    )


if __name__ == "__main__":
    main()
