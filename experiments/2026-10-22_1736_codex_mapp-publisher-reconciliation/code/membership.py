"""Describe publisher membership without conflating razor assignment and uniqueness."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import openpyxl

FAMILY = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    paths = json.loads((FAMILY / "results/input_paths.json").read_text())
    with open(paths["archive_members"]["txt/peptides.txt"], newline="") as stream:
        peptides = list(csv.DictReader(stream, delimiter="\t"))
    workbook = openpyxl.load_workbook(paths["publisher"]["suppdata2.xlsx"], read_only=True, data_only=True)
    rows = list(workbook["Sheet1"].values)
    workbook.close()
    published = {c: {r[0] for r in rows[4:] if r[col] is not None}
                 for c, col in (("UT", 2), ("TNF+IFN", 3))}
    results, memberships = {}, []
    for condition in published:
        columns = [c for c in peptides[0] if c.startswith("Intensity ") and f"_{condition}_" in c]
        assert len(columns) == 3
        eligible = [r for r in peptides if r["Reverse"] != "+" and r["Potential contaminant"] != "+"
                    and sum(float(r[c]) > 0 for c in columns) >= 2]
        sequences = {r["Sequence"] for r in eligible}
        results[condition] = {
            "source_eligible_without_unique_group_restriction": len(eligible),
            "publisher_present": len(published[condition]),
            "source_only": sorted(sequences-published[condition]),
            "publisher_only": sorted(published[condition]-sequences),
            "distinct_nonempty_leading_razor_labels": len({r["Leading razor protein"] for r in eligible if r["Leading razor protein"]}),
            "multiple_group_id_rows": sum(";" in r["Protein group IDs"] for r in eligible),
            "multiple_protein_id_rows": sum(";" in r["Proteins"] for r in eligible),
            "empty_leading_razor_labels": sum(not r["Leading razor protein"] for r in eligible),
        }
        for r in eligible:
            memberships.append({"condition": condition, "sequence": r["Sequence"],
                "source_peptide_id": r["id"], "source_leading_razor_protein": r["Leading razor protein"],
                "source_protein_ids": r["Proteins"], "source_group_ids": r["Protein group IDs"],
                "source_unique_groups": r["Unique (Groups)"], "source_unique_proteins": r["Unique (Proteins)"],
                "publisher_present": r["Sequence"] in published[condition]})
    with gzip.open(FAMILY / "results/publisher_membership.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(memberships[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(memberships)
    workbook = openpyxl.load_workbook(paths["publisher"]["suppdata3.xlsx"], read_only=True, data_only=True)
    data3 = list(workbook["Sheet1"].values)
    workbook.close()
    assert len(data3) == 133 and all(len(r) == 5 for r in data3)
    assert all(isinstance(r[0], str) and r[0] for r in data3[2:])
    assert all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for r in data3[2:] for v in r[1:])
    result = {"status": "completed", "conditions": results,
        "membership_rows": len(memberships), "data3_rows": 131, "data3_positive_numeric_cells": 524,
        "data3_headers": data3[1], "data3_unique_gene_labels": len({r[0] for r in data3[2:]}),
        "specimen_map_status": "No specimen/acquisition identifiers in Data3's five-column gene/intensity table; no run map assigned",
        "assignment_scope": "Leading razor labels are source assignments, not unique-origin evidence",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": time.monotonic()-started}
    (FAMILY / "results/membership-summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
