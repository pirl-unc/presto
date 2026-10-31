"""Inventory source quantities without deriving training examples or pairs."""

import csv
import hashlib
import json
import math
import re
import resource
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.utils.datetime import to_excel

EXP = Path(__file__).resolve().parents[1]
RAW = EXP.parents[1] / "data/mapp2025_source/2026-10-24_2130"


def main():
    start = time.monotonic()
    records = []
    metadata_rows = []
    for name in ["Table3.xlsx", "Table5.xlsx", "Data_Deposition_Table.xlsx"]:
        book = openpyxl.load_workbook(RAW / name, read_only=True, data_only=True)
        for sheet in book:
            # The reporter perturbation sheet is inventoried, not a baseline endpoint.
            if sheet.title == "dTAG_A549_secretome":
                continue
            digest = hashlib.sha256()
            headers = None
            count = 0
            rows = 0
            qty = {}
            conditions = Counter()
            baseline_positive_rows = 0
            complete_baseline_rows = 0
            baseline_unique_parent_rows = 0
            baseline_lengths = Counter()
            baseline_identification = Counter()
            date_cells = []
            for cells in sheet.iter_rows():
                values = [c.value for c in cells]
                for cell in cells:
                    value = cell.value
                    if value is None or value == "":
                        continue
                    if isinstance(value, datetime):
                        date_cells.append(
                            {
                                "cell": cell.coordinate,
                                "display_date": value.isoformat(),
                                "header": headers[cell.column - 1] if headers else None,
                            }
                        )
                        value = float(to_excel(value, book.epoch))
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        value = float(value)
                    digest.update(
                        (
                            json.dumps(
                                [cell.coordinate, value], ensure_ascii=False, allow_nan=False
                            )
                            + "\n"
                        ).encode()
                    )
                    count += 1
                if headers is None:
                    headers = values
                    for i, value in enumerate(headers):
                        if (
                            isinstance(value, str)
                            and not value.endswith("_log10")
                            and (
                                value.startswith(("Intensity ", "LFQ intensity "))
                                or value.endswith(" Intensity")
                            )
                        ):
                            qty[i] = {
                                "header": value,
                                "positive": 0,
                                "zero": 0,
                                "missing": 0,
                                "other": 0,
                            }
                    continue
                if not any(v is not None and v != "" for v in values):
                    continue
                rows += 1
                for i, stat in qty.items():
                    v = values[i]
                    key = "other"
                    if v is None or v == "":
                        key = "missing"
                    elif isinstance(v, (int, float)) and math.isfinite(v):
                        key = "positive" if v > 0 else "zero" if v == 0 else "other"
                    stat[key] += 1
                if name == "Table3.xlsx" and "Treatment" in headers:
                    conditions[str(values[headers.index("Treatment")])] += 1
                if sheet.title == "MAPP_peptidomics":
                    indices = [
                        i
                        for i, h in enumerate(headers)
                        if isinstance(h, str) and re.fullmatch(r"Intensity .*_Ctrl4?_[123]", h)
                    ]
                    positive = [
                        i for i in indices if isinstance(values[i], (int, float)) and values[i] > 0
                    ]
                    baseline_positive_rows += bool(positive)
                    complete_baseline_rows += len(positive) == 6
                    if positive:
                        proteins = str(values[headers.index("Proteins")]).split(";")
                        baseline_unique_parent_rows += len(proteins) == 1
                        baseline_lengths[str(values[headers.index("Length")])] += 1
                    for i in positive:
                        h = headers[i].replace("Intensity ", "Identification type ")
                        baseline_identification[str(values[headers.index(h)])] += 1
                if name == "Data_Deposition_Table.xlsx":
                    metadata_rows.append(dict(zip(headers, values)))
            records.append(
                {
                    "file": name,
                    "sheet": sheet.title,
                    "nonempty_data_rows": rows,
                    "value_cells": count,
                    "cell_sha256": digest.hexdigest(),
                    "quantities": list(qty.values()),
                    "condition_rows": dict(conditions),
                    "baseline_positive_rows": baseline_positive_rows,
                    "complete_six_baseline_rows": complete_baseline_rows,
                    "baseline_single_accession_rows": baseline_unique_parent_rows,
                    "baseline_length_counts": dict(baseline_lengths),
                    "baseline_identification_counts": dict(baseline_identification),
                }
            )
            records[-1]["date_styled_cells"] = date_cells
        book.close()
    files = json.loads((RAW / "files.json").read_text())
    raw_names = {
        x["fileName"].removesuffix(".raw") for x in files if x["fileName"].endswith(".raw")
    }
    supplied_names = {x["Raw File"] for x in metadata_rows}
    assert supplied_names == raw_names and len(metadata_rows) == len(raw_names) == 36
    with (RAW / "PXD052314_community_annotated.sdrf.tsv").open() as handle:
        sdrf = list(csv.DictReader(handle, delimiter="\t"))
    assert {x["comment[data file]"].removesuffix(".raw") for x in sdrf} == raw_names
    report = {
        "sheets": records,
        "original_run_rows": len(metadata_rows),
        "original_experiment_counts": dict(Counter(x["Experiment"] for x in metadata_rows)),
        "original_sample_labels": sorted({x["Sample Name"] for x in metadata_rows}),
        "sdrf_rows": len(sdrf),
        "sdrf_bio_replicate_values": sorted(
            {x["characteristics[biological replicate]"] for x in sdrf}
        ),
        "sdrf_cleavage_values": sorted({x["comment[cleavage agent details]"] for x in sdrf}),
        "seconds": time.monotonic() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    (EXP / "results/workbook-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sheets"}, indent=2))
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in s.items()
                    if k not in {"quantities", "condition_rows", "baseline_length_counts"}
                }
                for s in records
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
