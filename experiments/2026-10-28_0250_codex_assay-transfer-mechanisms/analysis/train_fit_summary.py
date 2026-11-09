"""Independently verify saved training fits to distinguish weak fitting from overfit."""

from collections import defaultdict
import gzip
import hashlib
import json

from verify import (
    F,
    PINS,
    UPLOAD,
    assert_metric,
    csv_write,
    metrics,
    physical,
    read,
    sha,
    write,
)


def identity(row):
    mhc = row["mhc"][0]
    values = (
        row["observation_id"],
        row["endpoint"],
        row["peptide"],
        mhc["mhc_class"],
        mhc["groove1"],
        mhc["groove2"],
        row["qualifier"],
        row["value"],
        row["unit"],
        row["split"],
        row["provenance"]["original_measurement_id"],
    )
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    table = []
    for name in PINS["conditions"]:
        directory = F / "results" / name
        if not (directory / "report.json").exists():
            continue
        path = directory / "train-predictions.jsonl.gz"
        certificate = F / "results" / f"{name}-training-verification.json"
        digest = sha(path)
        if certificate.exists():
            previous = read(certificate)
            assert previous["predictions_sha256"] == digest
            table.extend(previous["rows"])
            continue
        expected = {}
        for spec in PINS["manifests"][name]["sources"]:
            # Frozen files are split-homogeneous. Leave heldout files unopened.
            if "-train-" not in spec["path"]:
                continue
            source = UPLOAD / "inputs" / spec["path"]
            assert sha(source) == spec["sha256"]
            with source.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row["split"] == "train"
                    if row["split"] == "train":
                        assert row["observation_id"] not in expected
                        expected[row["observation_id"]] = identity(row)
        groups = defaultdict(list)
        seen = set()
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                key = row["observation_id"]
                assert key not in seen and row["split"] == "train"
                assert identity(row["observation"]) == expected[key]
                assert abs(physical(row["observation"]) - row["target"]) < 1e-12
                seen.add(key)
                compact = {
                    k: row[k] for k in ("target", "prediction", "qualifier", "loss")
                }
                groups[(row["endpoint"], row["mhc_classes"][0])].append(compact)
        assert seen == set(expected)
        resolved = read(directory / "resolved-config.json")
        report = read(directory / "report.json")
        rows = []
        for (endpoint, cls), values in groups.items():
            result = metrics(
                values, resolved["scales"][endpoint]["scale"], endpoint != "half_life"
            )
            if result["n"]:
                assert_metric(
                    result,
                    report["summary"]["metrics"]["train"][endpoint]["by_class"][cls],
                )
            rows.append(
                dict(
                    condition=name,
                    endpoint=endpoint,
                    mhc_class=cls,
                    **{k: v for k, v in result.items() if not isinstance(v, dict)},
                )
            )
        write(
            certificate,
            dict(
                status="passed",
                predictions_sha256=digest,
                checkpoint_sha256=report["checkpoint_sha256"],
                unique_training_observations=len(seen),
                scope="Saved training predictions and frozen source identity/units/loss/metrics only; no new fitting or held-out inference.",
                rows=rows,
            ),
        )
        table.extend(rows)
        print(name, len(seen), "training observations verified", flush=True)
    csv_write(F / "results/training-metrics-independent.csv", table)


if __name__ == "__main__":
    main()
