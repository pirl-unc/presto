"""Audit publisher selection and organism assertions; do not infer sample identity."""

import csv
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"


def main():
    start = time.monotonic()
    csv.field_size_limit(16 * 1024**2)
    book = openpyxl.load_workbook(RAW / "Table5.xlsx", read_only=True, data_only=True)
    rows = book["MAPP_peptidomics"].iter_rows(values_only=True)
    names = next(rows)
    ids = {str(int(row[names.index("id")])) for row in rows if row[0] is not None}
    book.close()
    with zipfile.ZipFile(RAW / "EC_infection_Degradome_MAPP_Peptides.zip") as archive:
        proteins = list(
            csv.DictReader(io.StringIO(archive.read("proteinGroups.txt").decode()), delimiter="\t")
        )
        peptides = list(
            csv.DictReader(io.StringIO(archive.read("peptides.txt").decode()), delimiter="\t")
        )
    organism = defaultdict(set)
    protein_header_taxa = Counter()
    for row in proteins:
        for header in row.get("Fasta headers", "").split(";"):
            taxonomy = re.findall(r"\bOX=(\d+)\b", header)
            accession = header.split("|")[1] if "|" in header else None
            if accession:
                organism[accession].update(taxonomy)
            protein_header_taxa.update(taxonomy)
    counts = Counter()
    details = defaultdict(Counter)
    for row in peptides:
        status = "published" if row["id"] in ids else "omitted"
        flags = (
            "flagged"
            if row.get("Reverse") == "+" or row.get("Potential contaminant") == "+"
            else "unflagged"
        )
        taxa = set()
        unresolved = False
        for accession in row["Proteins"].split(";"):
            known = organism.get(accession, set())
            unresolved |= not known
            taxa.update(known)
        context = (
            "human_only"
            if taxa == {"9606"} and not unresolved
            else (
                "nonhuman_only"
                if taxa and "9606" not in taxa and not unresolved
                else "mixed_or_unresolved"
            )
        )
        key = "|".join([status, flags, context])
        counts[key] += 1
        cols = [k for k in row if k.startswith("Intensity ")]
        baseline = [k for k in cols if "_Ctrl" in k]
        details[key]["rows_with_positive_baseline"] += any(float(row[k]) > 0 for k in baseline)
        details[key]["positive_baseline_cells"] += sum(float(row[k]) > 0 for k in baseline)
        details[key]["positive_all_run_cells"] += sum(float(row[k]) > 0 for k in cols)
        details[key]["all_run_positive_count_" + str(sum(float(row[k]) > 0 for k in cols))] += 1
    report = {
        "counts": dict(counts),
        "quantity_coverage": {k: dict(v) for k, v in details.items()},
        "protein_group_rows": len(proteins),
        "fasta_header_taxonomy_counts": dict(protein_header_taxa),
        "scope": "Only supplied protein-group FASTA headers; full search FASTA absent, no organism imputation",
        "seconds": time.monotonic() - start,
    }
    (EXP / "results/source-dispositions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
