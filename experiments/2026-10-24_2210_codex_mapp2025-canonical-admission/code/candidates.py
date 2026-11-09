"""Build the registered review-candidate recipe, then independently check cells."""

import csv
import io
import json
import math
import re
import resource
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
PACKAGE = REPOSITORY / "data/mapp2025_source/2026-10-24_2210_canonical"
OUTPUT = REPOSITORY / "data/mapp2025_source/2026-10-24_2210_baseline_candidates"


def rows(path):
    for batch in pq.ParquetFile(path).iter_batches():
        yield from batch.to_pylist()


def main():
    from hitlist.evidence_recipe import materialize_recipe
    from hitlist.evidence_store import EvidenceStore
    from hitlist.goldberg_candidates import RECIPE_ID

    start = time.monotonic()
    store = EvidenceStore.open(PACKAGE)
    recipe = {
        "schema_version": 1,
        "name": "baseline_candidate_review",
        "registered_recipe": RECIPE_ID,
    }
    materialize_recipe(store, recipe, OUTPUT)
    build_seconds = time.monotonic() - start
    audit_start = time.monotonic()
    root = PACKAGE / "goldberg_mapp"
    contract = json.loads((root / "contract.json").read_text())
    csv.field_size_limit(16 * 1024**2)
    original, taxonomy, experiments = {}, defaultdict(set), {}
    with zipfile.ZipFile(root / "source_assets" / contract["archive"]) as archive:
        for member in ("peptides.txt", "proteinGroups.txt", "summary.txt"):
            with archive.open(member) as handle:
                for row in csv.DictReader(io.TextIOWrapper(handle), delimiter="\t"):
                    if member == "peptides.txt":
                        original[row["id"]] = row
                    elif member == "summary.txt" and row["Experiment"]:
                        experiments[row["Experiment"]] = row["Raw file"]
                    elif member == "proteinGroups.txt":
                        for header in row["Fasta headers"].split(";"):
                            parts = header.split("|")
                            tokens = re.findall(r"\bOX=(\d+)(?=\s)", header)
                            if len(parts) >= 3 and len(tokens) == 1:
                                taxonomy[parts[1]].add(tokens[0])
    deposition = openpyxl.load_workbook(
        root / "source_assets" / contract["deposition"], read_only=True
    )
    sheet = deposition.active.values
    header = next(sheet)
    run_context = {r["Raw File"]: r for r in (dict(zip(header, row, strict=True)) for row in sheet)}
    deposition.close()
    native_ids = {r["peptide_id"]: r["native_id"] for r in rows(root / "typed/peptides.parquet")}
    candidates = rows(OUTPUT / "typed/candidates.parquet")
    counts, confidence = Counter(), Counter()
    seen = set()
    for decision in rows(OUTPUT / "typed/decisions.parquet"):
        native = native_ids[decision["peptide_id"]]
        row = original[native]
        field = decision["source_field"]
        key = (native, field)
        assert key not in seen
        seen.add(key)
        assert field == "Intensity" or field.startswith("Intensity ")
        raw_file = None if field == "Intensity" else experiments[field[len("Intensity ") :]]
        context = run_context.get(raw_file)
        baseline = (
            context is not None
            and context["Sample Name"]
            in {"Ctrl_1", "Ctrl_2", "Ctrl_3", "Ctrl4_1", "Ctrl4_2", "Ctrl4_3"}
            and context["Experiment"] == "EC infection Degradome MAPP Peptides"
        )
        value = float(row[field])
        parent_taxa = [taxonomy[p] for p in row["Proteins"].split(";")]
        human_only = all(t == {"9606"} for t in parent_taxa)
        reasons = []
        if not baseline:
            reasons.append("outside_baseline_acquisition_scope")
        if not math.isfinite(value) or value <= 0:
            reasons.append("not_positive_finite_source_intensity")
        if not row["Sequence"] or set(row["Sequence"]) - set("ACDEFGHIKLMNPQRSTVWY"):
            reasons.append("noncanonical_or_missing_sequence")
        if row["Reverse"] != "" or row["Potential contaminant"] != "":
            reasons.append("source_flagged_or_flag_unresolved")
        if not human_only:
            reasons.append("source_human_only_annotation_not_established")
        expected_status = (
            "outside_scope" if not baseline else "excluded" if reasons else "candidate"
        )
        assert decision["status"] == expected_status
        assert json.loads(decision["reasons_json"]) == reasons
        assert decision["quantitative_raw"] == row[field]
        assert decision["quantitative_value"] == value
        assert decision["log10_intensity"] == (math.log10(value) if value > 0 else None)
        assert decision["raw_file"] == raw_file
        assert decision["peptide_sequence"] == row["Sequence"]
        assert decision["parent_label_raw"] == row["Proteins"]
        assert decision["confidence_raw"] == row["PEP"]
        assert decision["confidence_policy_status"] == "retained_not_yet_selected"
        assert (
            decision["training_eligibility_status"] == "not_fully_qualified_processing_supervision"
        )
        counts[expected_status] += 1
        if expected_status == "candidate":
            assert next(candidates) == decision
            pep = float(row["PEP"])
            confidence[
                (decision["identification_type"], "PEP_lt_0.01" if pep < 0.01 else "PEP_ge_0.01")
            ] += 1
    assert next(candidates, None) is None
    assert len(seen) == sum(
        sum(k == "Intensity" or k.startswith("Intensity ") for k in row)
        for row in original.values()
    )
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert counts["candidate"] == summary["n_candidate_cells"]
    assert counts["excluded"] == summary["n_excluded_baseline_cells"]
    assert counts["outside_scope"] == summary["n_outside_scope_cells"]
    size = sum(
        p.stat().st_size
        for location in (PACKAGE, OUTPUT, EXPERIMENT)
        for p in location.rglob("*")
        if p.is_file()
    )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    elapsed = time.monotonic() - start
    assert size <= 10 * 1024**3 and peak <= 6 * 1024**3 and elapsed <= 1200
    result = {
        "recipe_summary": summary,
        "independent_n_decisions": len(seen),
        "status_counts": dict(counts),
        "candidate_confidence_axes": [
            {"identification_type": key[0], "PEP_stratum": key[1], "n_cells": count}
            for key, count in sorted(confidence.items())
        ],
        "build_with_source_and_recipe_replay_seconds": build_seconds,
        "independent_audit_seconds": time.monotonic() - audit_start,
        "total_seconds": elapsed,
        "peak_rss_bytes": peak,
        "total_package_recipe_audit_bytes": size,
    }
    (EXPERIMENT / "results/candidate-admission.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (EXPERIMENT / "results/recipe-manifest.json").write_bytes(
        (OUTPUT / "manifest.json").read_bytes()
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
