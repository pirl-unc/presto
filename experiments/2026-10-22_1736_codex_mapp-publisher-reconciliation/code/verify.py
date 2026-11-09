"""Independent scalar/CSV reconstruction of source filtering and publisher cells."""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import openpyxl

FAMILY = Path(__file__).resolve().parents[1]
RESULTS = FAMILY / "results"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path):
    with open(path, newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def close(a, b):
    # Match NumPy's recorded asymmetric rounding tolerance without using NumPy
    # arithmetic to reconstruct any quantity or filter rule.
    return abs(a-b) <= 1 + 1e-4 * abs(b)


def main():
    started = time.monotonic()
    hashes = json.loads((RESULTS / "output_hashes.json").read_text())
    for name, expected in hashes.items():
        assert sha(RESULTS / name) == expected
    paths = json.loads((RESULTS / "input_paths.json").read_text())
    members = paths["archive_members"]
    old = FAMILY.parent / "2026-10-22_1630_codex_mapp-source-admission"
    member_receipts = json.loads((old / "reproduce/archive_members.json").read_text())
    publisher_receipts = json.loads((FAMILY / "reproduce/publisher_assets.json").read_text())
    for name, path in members.items():
        assert sha(path) == member_receipts[name]["sha256"]
    for name, path in paths["publisher"].items():
        assert sha(path) == publisher_receipts["assets"][name]["sha256"]
    peptides = {r["id"]: r for r in records(members["txt/peptides.txt"])}
    assert len(peptides) == 10673
    totals = defaultdict(float)
    for r in records(members["txt/evidence.txt"]):
        totals[(r["Peptide ID"], r["Experiment"])] += float(r["Intensity"] or 0)
    with np.load(RESULTS / "filter_comparisons.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    ids = arrays["peptide_ids"].tolist()
    experiments = arrays["experiments"].tolist()
    beads = arrays["bead_experiments"].tolist()
    names = arrays["rule_names"].tolist()
    columns = {c.removeprefix("Intensity ").replace("-MAPPbeadFilt", ""): c
               for c in next(iter(peptides.values())) if c.startswith("Intensity ")}
    conditions = {c: [j for j, e in enumerate(experiments) if f"_{c}_" in e]
                  for c in ("UT", "TNF+IFN", "CHX", "NO-DSP", "NO-INH")}
    counts = {name: Counter() for name in names}
    source_medians = {}
    membership = {c: set() for c in ("UT", "TNF+IFN")}
    source_flags = {}
    for i, pid in enumerate(ids):
        row = peptides[pid]
        raw = [totals[(pid, e)] for e in experiments]
        observed = [float(row[columns[e]]) for e in experiments]
        background = [totals[(pid, e)] for e in beads]
        assert all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-6)
                   for x, y in zip(raw, arrays["raw"][i], strict=True))
        assert observed == arrays["measured"][i].tolist()
        assert all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-6)
                   for x, y in zip(background, arrays["beads"][i], strict=True))
        for c, indices in conditions.items():
            vals = [observed[j] for j in indices]
            positive = [v for v in vals if v > 0]
            source_medians[(row["Sequence"], c, "include_zero")] = statistics.median(vals)
            source_medians[(row["Sequence"], c, "positive_only")] = statistics.median(positive) if positive else None
            valid = row["Reverse"] != "+" and row["Potential contaminant"] != "+"
            groups = row["Protein group IDs"].split(";") if row["Protein group IDs"] else []
            valid = valid and len(groups) == 1 and groups[0].isdigit()
            if c in membership and valid and len(positive) >= 2:
                membership[c].add(row["Sequence"])
        source_flags[row["Sequence"]] = {
            "source_id": pid, "source_group_ids": row["Protein group IDs"],
            "reverse": row["Reverse"], "contaminant": row["Potential contaminant"],
            "unique_groups": row.get("Unique (Groups)"),
        }
        for k, name in enumerate(names):
            numerator_name, denominator_name = name.split("_over_beads_")
            background_value = {"mean": statistics.mean, "median": statistics.median, "max": max}[denominator_name](background)
            for j, e in enumerate(experiments):
                if numerator_name == "run":
                    numerator = raw[j]
                else:
                    condition = next(c for c, indices in conditions.items() if j in indices)
                    values = [raw[t] for t in conditions[condition]]
                    numerator = (statistics.mean if numerator_name == "condition_mean" else statistics.median)(values)
                keep = numerator >= 2 * background_value
                assert keep == bool(arrays["masks"][k, i, j]), (name, pid, e)
                prediction = raw[j] if keep else 0
                counts[name]["matching_cells"] += close(prediction, observed[j])
                counts[name]["mismatching_cells"] += not close(prediction, observed[j])
                counts[name]["reported_zero_predicted_positive"] += observed[j] == 0 and prediction > 0
                counts[name]["reported_positive_predicted_zero"] += observed[j] > 0 and prediction == 0
                counts[name]["positive_numerator_exact_boundary"] += numerator > 0 and numerator == 2*background_value
    rules = json.loads((RESULTS / "background_rules.json").read_text())
    for r in rules:
        assert r["cells"] == len(ids)*len(experiments)
        for key, value in counts[r["rule"]].items():
            assert r[key] == value, (r["rule"], key, r[key], value)
    workbook = openpyxl.load_workbook(paths["publisher"]["suppdata2.xlsx"], read_only=True, data_only=True)
    sheet = workbook["Sheet1"]
    cells = {cell.coordinate: cell.value for row in sheet for cell in row if cell.value is not None}
    actual_membership = {"UT": set(), "TNF+IFN": set()}
    for row in range(5, sheet.max_row+1):
        for c, column in (("UT", "C"), ("TNF+IFN", "D")):
            if f"{column}{row}" in cells:
                actual_membership[c].add(cells[f"A{row}"])
    membership_comparison = {}
    for c in membership:
        missing = sorted(membership[c]-actual_membership[c])
        extra = sorted(actual_membership[c]-membership[c])
        membership_comparison[c] = {"source_eligible": len(membership[c]),
            "publisher_present": len(actual_membership[c]),
            "eligible_absent_from_publisher": [{"sequence": s, **source_flags[s]} for s in missing],
            "publisher_outside_source_policy": [{"sequence": s, **source_flags[s]} for s in extra]}
    comparisons = 0
    column_counts = Counter()
    with gzip.open(RESULTS / "publisher_cell_comparisons.csv.gz", "rt") as stream:
        for r in csv.DictReader(stream):
            cell = r["cell"]
            source_value = cells[cell]
            sequence = cells["A" + cell[1:]]
            assert sequence == r["sequence"] and source_value == float(r["published_value"])
            linear = 10**source_value if cell[0] in "CD" else source_value
            assert linear == float(r["published_linear"])
            source = source_medians[(sequence, r["candidate_source_condition"], r["median_aggregation"])]
            if source is None:
                assert r["source_median"] == ""
                match = False
            else:
                assert float(r["source_median"]) == source
                match = close(linear, source)
            assert match == (r["matches"] == "True")
            column_counts[(cell[0], r["candidate_source_condition"], r["median_aggregation"])] += match
            comparisons += 1
    workbook.close()
    result = {"status": "passed", "input_hashes_verified": len(members)+len(paths["publisher"]),
        "input_archive_members_independently_reparsed": ["txt/peptides.txt", "txt/evidence.txt"],
        "output_hashes_verified": len(hashes), "filter_cells_verified": len(ids)*len(experiments),
        "rule_decisions_verified": len(names)*len(ids)*len(experiments),
        "publisher_cell_comparisons_verified": comparisons,
        "membership_reconstruction": membership_comparison,
        "seconds": time.monotonic()-started}
    (RESULTS / "independent-verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
