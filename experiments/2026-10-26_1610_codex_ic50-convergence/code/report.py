"""Descriptive strata and closure from saved predictions; never rescore a model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math

import numpy as np

from verify import FAMILY, metrics, read, records, sha

CONDITIONS = ("seed42", "seed43", "seed44", "permuted42")
BEGIN = "<!-- ic50-convergence-status:begin -->"
END = "<!-- ic50-convergence-status:end -->"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stratum(row, axis):
    observation = row["observation"]
    if axis == "allele_pair":
        pairs = [
            tuple(m["provenance"].get("catalog_alleles") or ["unknown"]) for m in observation["mhc"]
        ]
        return json.dumps(pairs, separators=(",", ":"))
    raw = observation["provenance"].get("raw_record", {})
    if raw.get("reference_iri"):
        return str(raw["reference_iri"])
    if raw.get("pmid"):
        return "PMID:" + str(raw["pmid"])
    return "unknown"


def describe(rows):
    result = metrics(rows)
    p = np.asarray([r["prediction"] for r in rows])
    result["prediction_range"] = [float(p.min()), float(p.max())]
    result["prediction_quantiles"] = np.quantile(p, [0, 0.25, 0.5, 0.75, 1]).tolist()
    exact = [r for r in rows if r["qualifier"] == "="]
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for r in exact:
        positive = r["target"] <= math.log10(500)
        called = r["prediction"] <= math.log10(500)
        counts[("t" if positive == called else "f") + ("p" if called else "n")] += 1
    result["exact_binding_confusion"] = counts
    result["unique_peptides"] = len({r["peptide"] for r in rows})
    return result


def stratify(rows):
    result = {"overall": describe(rows)}
    result["class"] = {
        c: describe(subset)
        for c in ("I", "II")
        if (subset := [r for r in rows if r["mhc_classes"] == [c]])
    }
    for axis in ("allele_pair", "study"):
        groups = defaultdict(list)
        for row in rows:
            groups[stratum(row, axis)].append(row)
        result[axis] = {key: describe(value) for key, value in sorted(groups.items())}
        for value in result[axis].values():
            exact = value["exact"]
            supported = bool(exact and exact["n"] >= 20 and exact["target_sd"] > 0)
            value["exact_regression_supported"] = supported
            value["support_policy"] = (
                "at least 20 exact measurements with nonzero target spread"
                if supported
                else "unsupported; descriptive estimates only, insufficient exact count or spread"
            )
        assert sum(v["n"] for v in result[axis].values()) == len(rows)
    return result


def report(condition):
    verification = FAMILY / "results" / f"{condition}-independent-verification.json"
    assert read(verification)["status"] == "passed"
    folder = FAMILY / "results" / condition
    paths = sorted(folder.glob("*-predictions.jsonl.gz"))
    assert len(paths) == (6 if condition == "baselines" else 3)
    inputs = {str(p.relative_to(FAMILY)): sha(p) for p in [verification, *paths]}
    destination = FAMILY / "results" / f"{condition}-strata.json"
    if destination.exists():
        assert read(destination)["inputs"] == inputs
        assert read(destination)["report_source_sha256"] == sha(__file__)
        return
    result = {
        "condition": condition,
        "inputs": inputs,
        "report_source_sha256": sha(__file__),
        "policy": "All strata retained; descriptive only; studies are not independent splits.",
        "strata": {},
    }
    for path in paths:
        key = path.name.removesuffix("-predictions.jsonl.gz")
        result["strata"][key] = stratify(records(path))
    write(destination, result)


def replace_block(path, body):
    text = path.read_text()
    block = BEGIN + "\n" + body.rstrip() + "\n" + END
    if BEGIN in text:
        before, rest = text.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)


def refresh():
    results = FAMILY / "results"
    rows = []
    baseline = read(results / "baselines-independent-verification.json")
    for method, splits in baseline["metrics"].items():
        rows.append({"condition": method, "status": "verified", "metrics": splits})
    for condition in CONDITIONS:
        path = results / f"{condition}-independent-verification.json"
        if path.exists():
            v = read(path)
            rows.append(
                {
                    "condition": condition,
                    "status": "verified",
                    "metrics": v["metrics"],
                    "selected_epoch": v["selected_epoch"],
                    "completed_epochs": v["completed_epochs"],
                    "scientific_screen_passed": v["scientific_screen_passed"],
                    "contraction_strata": v["contraction_strata"],
                }
            )
        else:
            receipts = sorted((FAMILY / "reproduce").glob(f"{condition}-[0-9]*.json"))
            status = read(receipts[-1])["status"] if receipts else "not_started"
            rows.append({"condition": condition, "status": status})
    complete = all(r["status"] == "verified" for r in rows)
    passed = (
        complete
        and all(r["scientific_screen_passed"] for r in rows[2:5])
        and not rows[5]["scientific_screen_passed"]
    )
    snapshot = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "ic50_comparison_screen_passed": passed,
        "full_model_qualified": False,
        "conditions": rows,
    }
    write(results / "family-status.json", snapshot)
    table = [
        "| Condition | State | Selected / completed epochs | Validation RMSE | Test RMSE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if "metrics" in row:
            v = row["metrics"]["validation"]["exact"]["rmse"]
            t = row["metrics"]["test"]["exact"]["rmse"]
            epochs = (
                f"{row['selected_epoch']} / {row['completed_epochs']}"
                if "selected_epoch" in row
                else "n/a"
            )
            table.append(f"| {row['condition']} | {row['status']} | {epochs} | {v:.6f} | {t:.6f} |")
        else:
            table.append(f"| {row['condition']} | {row['status']} | pending | pending | pending |")
    narrative = (
        "## Current saved-output status\n\nAs of "
        + snapshot["utc"]
        + ".\n\n"
        + "\n".join(table)
        + f"\n\nAll fixed conditions verified: **{complete}**. "
        + f"IC50 comparison screen passes: **{passed}**."
        + "\nNo best-seed selection, test-based retuning, or full-model promotion. "
        + "All class-specific, 500 nM, censoring, allele/study and paired-bootstrap "
        + "details remain in `results/`.\n"
        + "TCRvdb is not yet available (confirmed by the user October 26). "
        + "Auxiliary, censor-training, "
        + "APC/MS transfer and serving gates remain open. PR 131 stays draft.\n\n"
        + "Status: "
        + ("fixed comparison closed" if complete else "fixed comparison active")
        + ". Next Step: "
        + (
            "review all gates and register the next independent condition"
            if complete
            else "finish and verify the remaining declared arms"
        )
        + ". Open Questions: authorized TCRvdb input and the separately scoped "
        + "biological milestones.\n"
    )
    replace_block(FAMILY / "README.md", narrative)
    replace_block(
        FAMILY.parent / "experiment_log.md",
        narrative.replace(
            "## Current saved-output status", "### IC50 convergence saved-output status"
        ),
    )
    print(json.dumps({k: v for k, v in snapshot.items() if k != "conditions"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=("baselines", *CONDITIONS, "refresh"))
    args = parser.parse_args()
    if args.condition != "refresh":
        report(args.condition)
    refresh()


if __name__ == "__main__":
    main()
