"""Upgrade independently reported cell identities without changing cohort membership."""

import hashlib
import json
import time
import uuid
from collections import Counter, defaultdict

from prepare import rows, sha
from presto.data.modular_hitlist_ms import source_cell_type

POLICY = "reported_or_curated_row_identity_independent_of_condition_arm"


def invariant(row):
    copy = {**row, "context": {**row["context"]}}
    copy["context"].pop("cell_type", None)
    provenance = dict(copy["context"].get("provenance", {}))
    provenance.pop("cell_type_policy", None)
    copy["context"]["provenance"] = provenance
    return json.dumps(copy, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def refresh(root):
    started = time.monotonic()
    report_path = root / "preparation.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "prepared":
        raise ValueError("completed cohort preparation required")
    receipt_path = root / "context-refresh.json"
    if receipt_path.exists():
        raise ValueError(
            "context refresh already recorded; do not overwrite its evidence"
        )
    (root / "preparation-before-context-refresh.json").write_text(
        report_path.read_text()
    )
    receipt = {"policy": POLICY, "files": {}, "test_accessed": False}
    context_counts = defaultdict(Counter)
    census = Counter()
    for name in [
        "expanded-train.jsonl",
        "expanded-validation.jsonl",
        "expanded-II-train.jsonl",
    ]:
        path = root / "inputs" / name
        original_sha = sha(path)
        if original_sha != report["files"][name]["sha256"]:
            raise ValueError("prepared source changed: " + name)
        before, after = hashlib.sha256(), hashlib.sha256()
        written = hashlib.sha256()
        counts = Counter()
        legacy_temp = path.with_suffix(".jsonl.context-refresh")
        if legacy_temp.exists():
            counts["discarded_incomplete_legacy_bytes"] = legacy_temp.stat().st_size
            legacy_temp.unlink()
        temp = path.with_suffix(f".jsonl.context-refresh-{uuid.uuid4().hex}")
        with temp.open("w", buffering=1024 * 1024) as f:
            for row in rows(path):
                before.update(invariant(row))
                context = row["context"]
                old = context.get("cell_type")
                context["cell_type"] = source_cell_type(row["provenance"]["raw_record"])
                context.setdefault("provenance", {})["cell_type_policy"] = POLICY
                after.update(invariant(row))
                counts["rows"] += 1
                counts["changed_cell_type"] += old != context["cell_type"]
                counts["known_cell_type"] += context["cell_type"] is not None
                if name == "expanded-train.jsonl" and row["value"] == 1.0:
                    cls = row["mhc"][0]["mhc_class"]
                    census[(cls, "positive_rows")] += 1
                    census[(cls, "known_cell_type")] += context["cell_type"] is not None
                    census[(cls, "resolved_arm")] += bool(
                        context["provenance"].get("condition_admitted")
                    )
                    if context["cell_type"]:
                        context_counts["cell_types"][context["cell_type"]] += 1
                    for kind in ("interventions", "cytokines"):
                        for factor, assertion in context[kind].items():
                            context_counts[kind][factor] += 1
                            census[(cls, f"{kind}:{factor}:{assertion['state']}")] += 1
                line = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                f.write(line)
                written.update(line.encode())
        if before.hexdigest() != after.hexdigest():
            raise ValueError("context refresh changed protected cohort fields")
        if sha(temp) != written.hexdigest():
            raise ValueError(
                "written context output differs from verified transformed rows"
            )
        temp.replace(path)
        report["files"][name] = {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "rows": counts["rows"],
        }
        receipt["files"][name] = {
            **dict(counts),
            "before_sha256": original_sha,
            "after_sha256": report["files"][name]["sha256"],
            "all_other_fields_sha256": before.hexdigest(),
            "all_other_fields_unchanged": True,
            "written_bytes_verified": True,
        }
        print(name, receipt["files"][name], flush=True)
    report["context_counts"] = {k: dict(v) for k, v in context_counts.items()}
    report["model_context"] = {k: sorted(v) for k, v in context_counts.items()}
    report["context_policy"] = POLICY
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    receipt["class_context_census"] = [
        {"class": cls, "feature": feature, "positive_rows": n}
        for (cls, feature), n in sorted(census.items())
    ]
    receipt["seconds"] = time.monotonic() - started
    receipt["status"] = "passed"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
