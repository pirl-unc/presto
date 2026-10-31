"""Conserve source receipts, verify stock arithmetic, and reconcile PDF assay IDs."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

import openpyxl
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def write(name, value):
    with (ROOT / "results" / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.parse_args()
    cfg = json.loads((ROOT / "reproduce/launch-v4.json").read_text())
    data = Path(cfg["data_directory"])
    receipts = []
    for path in sorted(data.glob("*.receipt.json")):
        record = json.loads(path.read_text())
        payload = data / record["request"]["name"]
        if not payload.exists():
            payload = payload.with_name(payload.name + ".part")
        if payload.exists():
            assert payload.stat().st_size == record["bytes"]
            assert hashlib.sha256(payload.read_bytes()).hexdigest() == record["sha256"]
        else:
            assert record["bytes"] == 0
        # Keep complete originals locally. Do not republish HTML-embedded public logins.
        error = record.get("error", "")
        if "UnicodeDecodeError" in error:
            error = "UnicodeDecodeError: complete legacy HTML recovered separately with reversible Latin-1 view"
        receipts.append(
            {
                "asset": record["request"]["name"],
                "url": record["request"]["url"],
                "phase": record["phase"],
                "status": record["status"],
                "http_status": record.get("http_status"),
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "error": error,
                "original_receipt_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            }
        )
    assert sum(r["phase"] == "metadata" for r in receipts) == 20
    write("retrieval-summary.json", receipts)
    book = openpyxl.load_workbook(
        data / "s3-42003_2018_87_MOESM3_ESM.xlsx", data_only=True, read_only=True
    )
    rows = list(book.active.iter_rows(min_row=4, values_only=True))
    assert len(rows) == 500 and len({r[0] for r in rows}) == 500
    discrepancies = []
    errors = []
    for index, row in enumerate(rows, 4):
        expected = row[7] * row[8] / 100
        errors.append(abs(expected - row[9]))
        if not math.isclose(expected, row[9], rel_tol=1e-12, abs_tol=1e-12):
            discrepancies.append(
                {
                    "row": index,
                    "peptide": row[0],
                    "stock": row[7],
                    "purity": row[8],
                    "reported_adjusted": row[9],
                    "recomputed_adjusted": expected,
                }
            )
    book.close()
    pdf = data / "s3-pr4c00576_si_002.pdf"
    assert (
        hashlib.md5(pdf.read_bytes()).hexdigest() == "06b920ba04ca44f8d70142515d9d03b3"
    )
    reader = PdfReader(pdf)
    pypdf_text = "\n".join(reader.pages[n].extract_text() for n in (3, 4))
    poppler = subprocess.check_output(
        ["/opt/homebrew/bin/pdftotext", "-f", "4", "-l", "5", "-layout", str(pdf), "-"],
        text=True,
    )
    pattern = (
        r"(non-CPTAC-\d+)\s+([A-Z]+)\s+(https://assays\.cancer\.gov/non-CPTAC-\d+)"
    )
    pairs = re.findall(pattern, pypdf_text)
    assert pairs == re.findall(pattern, poppler)
    assert len(pairs) == len({p[0] for p in pairs})
    write(
        "fallback-assay-links.json",
        [{"assay_id": a, "peptide": p, "url": u} for a, p, u in pairs],
    )
    inspections = [
        json.loads((ROOT / "results" / name).read_text())
        for name in ("supplement-inspection.json", "fallback-inspection.json")
    ]
    runtimes = [
        json.loads(p.read_text())
        for p in sorted((ROOT / "results").glob("*-runtime.json"))
    ]
    summary = {
        "status": "supporting_metadata_verified_processed_skyline_access_unresolved",
        "http_requests": len(receipts),
        "metadata_requests": 20,
        "retrieval_status_counts": dict(Counter(r["status"] for r in receipts)),
        "received_http_bytes_including_failed": sum(r["bytes"] for r in receipts),
        "primary": {
            "workbooks": 8,
            "sheets": sum(len(a.get("sheets", [])) for a in inspections[0]["assets"]),
            "stock_sequences": 500,
            "stock_unit": "pmol/uL solution; not injected amount",
            "stock_correction_checks": 500,
            "stock_correction_discrepancies": discrepancies,
            "maximum_absolute_stock_correction_difference": max(errors),
            "cell_checks": inspections[0]["cell_checks"],
        },
        "fallback": {
            "workbooks": 2,
            "sheets": sum(len(a.get("sheets", [])) for a in inspections[1]["assets"]),
            "cell_checks": inspections[1]["cell_checks"],
            "independently_matched_pdf_assay_links": len(pairs),
            "reported_title_assay_count": 53,
            "unreconciled_title_vs_link_count": len(pairs) != 53,
            "processed_candidate": "PASS05890/CPTAC_Exp1_Curves_CS.sky",
            "listed_size": "7.8M",
            "downloaded_bytes": 0,
        },
        "unverified_spreadsheet_cells": sum(
            len(r["unverified_cells"]) for r in inspections
        ),
        "resource_summary_before_closeout": {
            "stages": len(runtimes),
            "statuses": dict(Counter(r["status"] for r in runtimes)),
            "total_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in runtimes),
            "peak_child_rss_bytes": max(
                r.get("peak_child_process_rss_bytes", 0) for r in runtimes
            ),
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("openpyxl", "pypdf", "requests", "psutil")
        },
        "model_fits": 0,
        "predictions": 0,
        "native_outer_partitions_opened": False,
        "training_admission": "Not admitted: no verified raw isotope areas joined to known loads in the retrieved scope.",
    }
    write("source-admission-summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
