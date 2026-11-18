"""Describe explicit intervention support in the exact admitted populations."""

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import modal
from launch import NAME, E, image, volume

app = modal.App("presto-presentation-context-support")


def signature(context):
    result = []
    for family, overrides in (
        ("cytokines", "cytokine_overrides"),
        ("interventions", "intervention_overrides"),
        ("processing_pathways", "pathway_overrides"),
    ):
        claims = {**context.get(family, {}), **context.get(overrides, {})}
        for name, claim in sorted(claims.items()):
            state = claim["state"]
            if state not in {"present", "absent", "unknown"}:
                raise ValueError("invalid exposure state")
            if state != "unknown":
                result.append((family, name, state))
    return tuple(result)


def summarize(records):
    totals = Counter()
    state_counts = defaultdict(Counter)
    conditions = defaultdict(set)
    signatures = Counter()
    cells = Counter()
    for row in records:
        if row["endpoint"] != "elution_observation":
            raise ValueError("presentation source required")
        if row["split"] == "test":
            raise ValueError("TEST is excluded")
        if row["value"] != 1:
            continue
        classes = {m.get("class_override") or m["mhc_class"] for m in row["mhc"]}
        if len(classes) != 1:
            raise ValueError("one class required")
        cls = classes.pop()
        p, c = row["provenance"], row["context"]
        study = str(p.get("study_id") or p.get("pmid") or "unknown")
        raw = p.get("raw_record", {})
        line = str(raw.get("cell_line_name") or "").strip()
        if line.lower() in {"unknown", "none", "nan", "n/a"}:
            line = ""
        cell = c.get("cell_type_override") or c.get("cell_type") or "unknown"
        condition = str(
            p.get("condition_id")
            or c.get("provenance", {}).get("condition_id")
            or "unknown"
        )
        scopes = [("study_cell_type", study, cell, cls)]
        if line:
            scopes.append(("study_reported_cell_line", study, line, cls))
        sig = signature(c)
        cells[(cls, cell)] += 1
        signatures[(cls, study, line, cell, sig)] += 1
        for scope in scopes:
            totals[scope] += 1
            for family, name, state in sig:
                key = (*scope, family, name)
                state_counts[key][state] += 1
                conditions[(*key, state)].add(condition)
    output = []
    for key, counts in sorted(state_counts.items()):
        scope, study, cell, cls, family, name = key
        output.append(
            {
                "scope": scope,
                "study": study,
                "cell": cell,
                "class": cls,
                "family": family,
                "factor": name,
                "present_positive_rows": counts["present"],
                "absent_positive_rows": counts["absent"],
                "unknown_positive_rows": totals[key[:4]] - sum(counts.values()),
                "both_explicit_states": bool(counts["present"] and counts["absent"]),
                "present_conditions": sorted(conditions[(*key, "present")]),
                "absent_conditions": sorted(conditions[(*key, "absent")]),
            }
        )
    return {
        "positive_rows": sum(cells.values()),
        "cell_counts": [
            {"class": c, "cell_type": cell, "positive_rows": n}
            for (c, cell), n in sorted(cells.items())
        ],
        "factor_support": output,
        "concurrent_signatures": [
            {
                "class": cls,
                "study": study,
                "reported_cell_line": line or None,
                "cell_type": cell,
                "signature": list(sig),
                "positive_rows": n,
            }
            for (cls, study, line, cell, sig), n in sorted(signatures.items())
        ],
    }


@app.function(
    image=image, cpu=4, memory=16384, timeout=1800, volumes={"/volume": volume}
)
def audit():
    root = Path("/volume") / NAME
    report = json.loads((root / "preparation.json").read_text())
    start = time.monotonic()
    summaries = {}
    for name in (
        "expanded-train.jsonl",
        "expanded-validation.jsonl",
        "apc-validation.jsonl",
    ):
        digest = hashlib.sha256()

        def records(name=name, digest=digest):
            with (root / "inputs" / name).open("rb", buffering=1024 * 1024) as f:
                for line in f:
                    digest.update(line)
                    yield json.loads(line)

        summary = summarize(records())
        if digest.hexdigest() != report["files"][name]["sha256"]:
            raise ValueError("source hash mismatch: " + name)
        summaries[name] = {"sha256": digest.hexdigest(), **summary}
    return {
        "status": "passed",
        "seconds": time.monotonic() - start,
        "populations": summaries,
        "test_accessed": False,
        "interpretation": "Attribution support only. Both explicit states within a study/line are necessary but not sufficient for an isolated intervention comparison; concurrent exposures and MHC differences remain relevant. Detection counts are not calibrated abundances.",
    }


@app.local_entrypoint()
def main():
    result = audit.remote()
    path = E / "results/context-conditional-support.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(path),
                "seconds": result["seconds"],
                "positive_rows": {
                    k: v["positive_rows"] for k, v in result["populations"].items()
                },
            }
        )
    )
