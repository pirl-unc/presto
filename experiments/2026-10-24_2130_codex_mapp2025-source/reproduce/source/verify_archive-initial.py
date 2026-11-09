"""Reconcile publisher MAPP quantity cells against deposited MaxQuant output."""

import csv
import hashlib
import io
import json
import resource
import time
import zipfile
from collections import Counter
from pathlib import Path

import openpyxl

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"


def main():
    start = time.monotonic()
    archive_path = RAW / "EC_infection_Degradome_MAPP_Peptides.zip"
    metadata = next(
        x
        for x in json.loads((RAW / "files.json").read_text())
        if x["fileName"] == archive_path.name
    )
    assert archive_path.stat().st_size == metadata["fileSizeBytes"]
    with zipfile.ZipFile(archive_path) as archive:
        inventory = [
            {"member": x.filename, "bytes": x.file_size, "crc32": x.CRC} for x in archive.infolist()
        ]
        (EXP / "results/processed-members.json").write_text(json.dumps(inventory, indent=2) + "\n")
        selected = {}
        hashes = {}
        for name in ["peptides.txt", "summary.txt", "parameters.txt"]:
            matches = [
                x
                for x in archive.namelist()
                if Path(x).name == name and not x.startswith("__MACOSX/")
            ]
            assert len(matches) == 1, (name, matches)
            payload = archive.read(matches[0])
            hashes[name] = {
                "member": matches[0],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            selected[name] = list(
                csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
            )
    rows = selected["peptides.txt"]
    by_id = {str(int(x["id"])): x for x in rows}
    assert len(by_id) == len(rows)
    book = openpyxl.load_workbook(RAW / "Table5.xlsx", read_only=True, data_only=True)
    iterator = book["MAPP_peptidomics"].iter_rows(values_only=True)
    headers = next(iterator)
    quantities = [
        h
        for h in headers
        if isinstance(h, str) and h.startswith("Intensity ") and not h.endswith("_log10")
    ]
    assert len(quantities) == 12
    ids = set()
    sequences = set()
    comparisons = 0
    baseline_comparisons = 0
    maximum = 0.0
    for values in iterator:
        if values[0] is None:
            continue
        entry = dict(zip(headers, values))
        ident = str(int(entry["id"]))
        assert ident not in ids
        ids.add(ident)
        sequences.add(entry["Sequence"])
        original = by_id[ident]
        assert entry["Sequence"] == original["Sequence"]
        for name in quantities:
            difference = abs(float(entry[name]) - float(original[name]))
            maximum = max(maximum, difference)
            assert difference == 0.0, (ident, name, difference)
            comparisons += 1
            baseline_comparisons += "_Ctrl" in name
    book.close()
    remaining = [x for x in rows if x["id"] not in ids]
    statuses = Counter(
        (x.get("Reverse", ""), x.get("Potential contaminant", "")) for x in remaining
    )
    parameters = selected["parameters.txt"]
    wanted = [
        x
        for x in parameters
        if any(
            term in str(x).lower()
            for term in ["fasta", "enzyme", "unspecific", "peptide fdr", "match between", "version"]
        )
    ]
    runs = selected["summary.txt"]
    report = {
        "selected_members": hashes,
        "original_peptide_rows": len(rows),
        "publisher_rows": len(ids),
        "publisher_unique_sequences": len(sequences),
        "quantity_cells_verified": comparisons,
        "baseline_cells_verified": baseline_comparisons,
        "maximum_absolute_difference": maximum,
        "original_rows_not_in_publisher": len(remaining),
        "remaining_reverse_contaminant_counts": [
            {"reverse": k[0], "contaminant": k[1], "rows": v} for k, v in statuses.items()
        ],
        "source_parameters": wanted,
        "summary_rows": [
            {k: x.get(k) for k in ["Raw file", "Experiment", "Enzyme", "Enzyme mode"]} for x in runs
        ],
        "source_reference_members": [
            x for x in inventory if x["member"].lower().endswith((".fasta", ".fa"))
        ],
        "seconds": time.monotonic() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    (EXP / "results/archive-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"source_parameters", "summary_rows"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
