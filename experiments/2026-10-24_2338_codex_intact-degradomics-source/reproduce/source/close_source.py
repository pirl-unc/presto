"""Close source qualification with verified assets, exceptions and no model claim."""

import json
import resource
import time
import xml.etree.ElementTree as ET
from collections import Counter

from inventory import OUT, RAW, check_resources, sha


def main():
    started = time.monotonic()
    assert not (OUT / "source-closure.json").exists()
    check_resources()
    assets = []
    for receipt_path in sorted(RAW.glob("*.receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        receipt["headers"] = {
            k: v
            for k, v in receipt.get("headers", {}).items()
            if k.lower() not in {"set-cookie", "cookie", "authorization"}
        }
        if "path" in receipt:
            path = RAW / receipt_path.name.removesuffix(".receipt.json")
            assert path.stat().st_size == receipt["size_bytes"]
            assert sha(path) == receipt["sha256"]
        assets.append(receipt)
    listing = json.loads((RAW / "files-project-route.json").read_text())
    listing += json.loads((RAW / "files-page1.json").read_text())
    assert len({x["accession"] for x in listing}) == len(listing) == 105
    for entry in listing:
        if entry["fileName"].endswith(".xlsx"):
            assert (RAW / entry["fileName"]).stat().st_size == int(entry["fileSizeBytes"])
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    tree = ET.parse(RAW / "pmc-object-list.xml")
    assert tree.findtext("s:IsTruncated", namespaces=ns) == "false"
    for entry in tree.findall("s:Contents", ns):
        key = entry.findtext("s:Key", namespaces=ns)
        if key.endswith((".xlsx", ".sapp.pdf")):
            assert (RAW / key.split("/")[-1]).stat().st_size == int(
                entry.findtext("s:Size", namespaces=ns)
            )
    inventory = json.loads((OUT / "workbook-inventory.json").read_text())
    quantities = json.loads((OUT / "quantity-audit-layout-qualified.json").read_text())
    assert inventory["all_cells_match"] and not quantities["positional_mismatches"]
    assert all(
        x["input_masses_equal"] and x["assignment_multiset_equal"]
        for x in quantities["F1_F2_shared_sheets"]
    )
    table_counts = [
        {k: v for k, v in t.items() if k not in {"row_links", "sample_rows"}}
        for t in quantities["published_tables"]
    ]
    summary = {
        "status": "source_quantities_available_with_explicit_exceptions_not_training_admitted",
        "hitlist_issue": "https://github.com/pirl-unc/hitlist/issues/831",
        "source_file_count": len(listing),
        "source_file_categories": dict(Counter(x["fileCategory"]["value"] for x in listing)),
        "downloaded_bytes": sum(x.get("size_bytes", 0) for x in assets),
        "retrieval_seconds": sum(x["seconds"] for x in assets),
        "sheets": sum(len(f["sheets"]) for f in inventory["files"]),
        "nonempty_cells": sum(s["nonempty_cells"] for f in inventory["files"] for s in f["sheets"]),
        "formulas": sum(s["formulas"] for f in inventory["files"] for s in f["sheets"]),
        "formulas_without_cache": sum(
            s["formulas_without_cache"] for f in inventory["files"] for s in f["sheets"]
        ),
        "published_tables": table_counts,
        "reused_F1_F2_sheets": len(quantities["F1_F2_shared_sheets"]),
        "raw_cell_dump": inventory["independent_original_cells"],
        "raw_cell_dump_sha256": inventory["independent_original_cells_sha256"],
        "training": None,
        "predictions": None,
        "validation_or_test": None,
        "reason_no_split": "Original-source qualification precedes shared cohort admission",
        "limits": [
            "one substrate and one animal; unresolved replicate independence",
            "nominal zero already contains products",
            "unknown peptide-specific MS response and run scaling",
            "source mass-match identities are not all MS/MS verified",
            "missing products have unknown cause; no zero or negative labels",
            "not an APC concentration or complete peptide-pool measurement",
        ],
        "next": "shared Hitlist source admission and independent replay before kinetic fitting",
        "seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (OUT / "retrieval-receipts-sanitized.json").write_text(json.dumps(assets, indent=2) + "\n")
    (OUT / "source-file-inventory.json").write_text(json.dumps(listing, indent=2) + "\n")
    (OUT / "source-closure.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
