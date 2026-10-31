"""Independent Poppler/receipt closure of a source audit; no training labels."""

import csv
import hashlib
import json
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/endolysosomal_source_qualification/2026-10-24_1521_oztemiz-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


started = time.perf_counter()
output = ROOT / "results/independent-source-verification.json"
assert not output.exists(), "Do not overwrite prior closure"
receipts = []
for path in sorted(RAW.glob("*.receipt.json")):
    receipt = json.loads(path.read_text())
    asset = RAW / path.name.removesuffix(".receipt.json")
    if receipt.get("status") == 200:
        assert digest(asset) == receipt["sha256"]
        assert asset.stat().st_size == receipt["size_bytes"]
    else:
        assert receipt["access_status"] == "failed_not_absent"
        assert not asset.exists()
    # Retain all source receipts separately; omit web headers from the summary.
    receipts.append({k: v for k, v in receipt.items() if k != "headers"})

inventory = json.loads((ROOT / "results/pdf-inventory/inventory.json").read_text())
official = json.loads((RAW / "supplements.json").read_text())
assert len(official) == len(inventory["files"]) == 5
xml = etree.parse(str(RAW / "article.xml"))
supplements = xml.xpath("//supplementary-material")
assert len(supplements) == 5
classifications = [
    "reporter-activity graph; not peptide-product quantities",
    "protein-depletion half-life graph and aggregate table; not peptide products",
    "Bet v 1 peptide span/intensity-color map; not original numeric sample rows",
    "Aes h 1 peptide span/intensity-color map; not original numeric sample rows",
    "Aes h 1 peptide intensity time-course graph; not original numeric sample rows",
]
assets = []
for i, (remote, previous) in enumerate(zip(official, inventory["files"], strict=True), 1):
    path = RAW / f"supplement-{i}.pdf"
    receipt = json.loads((RAW / f"{path.name}.receipt.json").read_text())
    assert receipt["url"] == remote["downloadUrl"]
    assert digest(path) == previous["sha256"]
    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    detach = subprocess.check_output(["pdfdetach", "-list", str(path)], text=True)
    assert int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)[1]) == previous["n_pages"] == 1
    assert detach.strip() == "0 embedded files"
    assert not previous["attachments"]
    assets.append(
        {
            "name": path.name,
            "url": receipt["url"],
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "pages": 1,
            "attachments": 0,
            "evidence_class": classifications[i - 1],
            "admitted_peptide_quantity_rows": 0,
        }
    )

# Only verify literal aggregate table entries. No digitization of plotted values.
s2 = subprocess.check_output(
    ["pdftotext", "-layout", str(RAW / "supplement-2.pdf"), "-"], text=True
)
expected = {
    "Number of values": [7, 4],
    "Mean": [1, 16],
    "Std. Deviation": [0.28, 3.3],
    "Std. Error of Mean": [0.1, 1.6],
}
for label, values in expected.items():
    match = re.search(r"^\s*" + re.escape(label) + r"\s+(\S+)\s+(\S+)\s*$", s2, re.MULTILINE)
    assert match and [float(match[1]), float(match[2])] == values
assert "independent endolysosomal protein batches" in s2

tree = json.loads((ROOT / "results/repository-tree.json").read_text())
assert tree["sha"] == "18dd99da4a619abd8a28c7622e0809eb2c03e0ff"
assert tree["truncated"] is False and len(tree["tree"]) == 61
spreadsheets = [r for r in tree["tree"] if r["path"].lower().endswith((".xlsx", ".csv", ".tsv"))]
assert [r["path"] for r in spreadsheets] == ["sample_data/Example_degradome.xlsx"]
readme = (ROOT / "results/synthetic-example-readme.md").read_text()
assert "artificially generated for demonstration purposes only" in readme
assert "not the result of any real" in readme
figshare_log = ROOT / "launch_logs/figshare-index.log"
assert "HTTP Error 404" in figshare_log.read_text()
write(
    ROOT / "results/figshare-failure.json",
    {
        "url": "https://www.frontiersin.org/api/v4/articles/figshare",
        "method": "POST",
        "body": {"doi": "10.3389/falgy.2026.1936335"},
        "purpose": "read-only metadata query used by the publisher UI",
        "status": 404,
        "access_status": "failed_not_absent",
        "evidence": str(figshare_log.relative_to(ROOT)),
        "log_sha256": digest(figshare_log),
        "receipt_kind": "posthoc transcription of preserved terminal failure; not a new request",
        "response_body_and_headers": "not captured by the original failed helper",
    },
)
with (ROOT / "results/assets.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(assets[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(assets)
write(
    ROOT / "results/source-qualification.json",
    {
        "decision": "no quantitative peptide cohort admitted from inspected public assets",
        "scope": "official article/XML, five listed supplements, pinned linked repository",
        "assets": assets,
        "aggregate_protein_half_life_table": expected,
        "aggregate_table_columns": ["Bet v 1", "Aes h 1"],
        "aggregate_table_is_training_data": False,
        "repository": {
            "commit": tree["sha"],
            "entries": len(tree["tree"]),
            "only_spreadsheet": spreadsheets[0],
            "synthetic_excluded": True,
        },
        "limitations": [
            "Not an exhaustive absence statement about author-held or unlinked files",
            "Figshare metadata query failed; cannot infer deposit absence from HTTP404",
            "Pypdf text extraction warned about form-XObject limits on S3/S4; use Poppler/render review",
            "Numeric graph colors and coordinates were not converted into training labels",
            "Protein-depletion aggregate measurements do not label peptide abundances",
            "Isolated murine JAWSII preparation, not intact APC peptide concentrations",
            "Search length filtering omits short products; omission is not a biological negative",
            "Reduction/alkylation is observation preparation, not an inferred processing intervention",
        ],
    },
)
copies = ROOT / "reproduce/source"
for path in (ROOT / "code").glob("*.py"):
    target = copies / path.name
    if target.exists():
        assert digest(target) == digest(path)
    else:
        shutil.copyfile(path, target)
shutil.copyfile(__file__, copies / "verify_source.py")
(ROOT / "reproduce/receipts").mkdir(exist_ok=True)
for path in RAW.glob("*.receipt.json"):
    shutil.copyfile(path, ROOT / "reproduce/receipts" / path.name)
write(
    ROOT / "reproduce/asset-invocations.json",
    {
        "note": "Argument vectors reconstructed from original per-asset receipts; exact initial invocation is launch.json",
        "retrievals": [
            {
                "argv": [
                    "code/launch.py",
                    "--asset-url",
                    r["url"],
                    "--asset-name",
                    Path(r["path"]).name,
                ]
            }
            for r in receipts
            if r.get("status") == 200
        ],
        "other_executed_scripts": [
            "code/figshare_index.py",
            "code/inspect_pdfs.py",
            "analysis/verify_source.py",
        ],
        "runtime_root_required": str(ROOT),
        "environment_overrides": {},
        "frozen_inspection_scripts_require": "original experiment directory layout and raw assets",
    },
)
write(
    output,
    {
        "status": "pass",
        "successful_asset_hashes": sum(r.get("status") == 200 for r in receipts),
        "failed_original_article_request_retained": True,
        "supplemental_pdfs_verified": 5,
        "pdf_bytes": sum(r["bytes"] for r in assets),
        "pages": 5,
        "embedded_attachments": 0,
        "literal_aggregate_table_values_verified": 8,
        "repository_tree_entries": 61,
        "quantitative_peptide_labels_admitted": 0,
        "retrieval_seconds_sum": sum(r["seconds"] for r in receipts),
        "inventory_seconds": inventory["seconds"],
        "inventory_peak_gib": inventory["peak_gib"],
        "verification_seconds": time.perf_counter() - started,
        "verification_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "verification_implementation": "Poppler pdfinfo/pdfdetach/pdftotext, SHA256, XML and pinned tree",
        "no_optimization_or_test_outcome_access": True,
    },
)
print(output.read_text())
