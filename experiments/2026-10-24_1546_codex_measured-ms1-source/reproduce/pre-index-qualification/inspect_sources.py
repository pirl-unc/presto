"""Source conservation and bounded metadata audit, not a training-data adapter."""

import csv
import hashlib
import io
import json
import resource
import time
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd
from lxml import etree, html

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/measured_ms1_source_qualification/2026-10-24_1546_v1"
OUT = ROOT / "results"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def table(payload):
    # Two independent readers preserve literal cells, duplicate header positions,
    # explicit empties and NaN strings; this is not numeric target conversion.
    text = payload.decode("utf-8-sig")
    cells = list(csv.reader(io.StringIO(text), delimiter="\t"))
    width = len(cells[0])
    assert all(len(row) <= width for row in cells)
    padded = [row + [""] * (width - len(row)) for row in cells]
    other = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=False,
    ).values.tolist()
    assert padded == other
    return cells[0], padded[1:], sum(len(row) for row in cells), sha(payload)


def records(header, rows):
    assert len(header) == len(set(header)), "Repeated fields require positional handling"
    return [dict(zip(header, row, strict=True)) for row in rows]


started = time.perf_counter()
assert not (OUT / "source-summary.json").exists()
receipts = []
for path in sorted(RAW.glob("*.receipt.json")):
    value = json.loads(path.read_text())
    target = RAW / value["name"]
    if "sha256" in value:
        assert sha(target.read_bytes()) == value["sha256"]
        assert target.stat().st_size == value["bytes"]
    else:
        assert not target.exists()
    receipts.append(value)
assert sum(p.stat().st_size for p in RAW.iterdir() if p.is_file()) < 500 * 1024**2
write("source-receipts.json", receipts)

tree = html.fromstring((RAW / "PXD021013-directory.html").read_bytes())
files = []
for row in tree.xpath("//tr"):
    anchors = row.xpath(".//a[@href]")
    if not anchors:
        continue
    href = anchors[0].get("href")
    if href.startswith(("?", "/")) or href.endswith("/"):
        continue
    cells = [" ".join(c.text_content().split()) for c in row.xpath("./td")]
    files.append({"name": href, "last_modified_text": cells[2], "size_text": cells[3]})
assert len({r["name"] for r in files}) == len(files)
write("directory-inventory.json", files)
header, rows, readme_cells, readme_sha = table((RAW / "PXD021013-README.txt").read_bytes())
readme = records(header, rows)
assert len({r["NAME"] for r in readme}) == len(readme)
directory_names = {r["name"] for r in files}
assert all(r["NAME"] in directory_names for r in readme)
write(
    "readme-index-summary.json",
    {
        "rows": len(readme),
        "cells_verified": readme_cells,
        "source_sha256": readme_sha,
        "type_counts": dict(Counter(r["TYPE"] for r in readme)),
        "directory_only_names": sorted(directory_names - {r["NAME"] for r in readme}),
        "mapping_examples": [
            r
            for r in readme
            if r["NAME"]
            in {
                "TUM_HLA_1_01_01_DDA-1h-R1-unspecific.zip",
                "TUM_HLA2_1_01_01_DDA-1h-R1-unspecific.zip",
            }
        ],
    },
)
header, rows, sdrf_cells, sdrf_sha = table((RAW / "PXD021013-sdrf.tsv").read_bytes())
sdrf = {name: [row[i] for row in rows] for i, name in enumerate(header) if header.count(name) == 1}
duplicate_fields = [
    {"column_1based": i + 1, "name": name, "counts": dict(Counter(row[i] for row in rows))}
    for i, name in enumerate(header)
    if header.count(name) > 1
]
assert all(name in directory_names for name in sdrf["comment[data file]"])
write(
    "sdrf-summary.json",
    {
        "rows": len(rows),
        "columns": len(header),
        "cells_verified": sdrf_cells,
        "source_sha256": sdrf_sha,
        "duplicate_fields_preserved_by_position": duplicate_fields,
        "distinct_raw_files": len(set(sdrf["comment[data file]"])),
        "metadata_value_counts": {
            k: dict(Counter(v))
            for k, v in sdrf.items()
            if not any(t in k for t in ["source name", "file uri", "data file"])
        },
    },
)

summaries = []
verified_cells = readme_cells + sdrf_cells
for name in ["HLA1-first-DDA.zip", "HLA2-first-DDA.zip"]:
    with zipfile.ZipFile(RAW / name) as archive:
        members = [
            {"name": m.filename, "bytes": m.file_size, "crc32": m.CRC} for m in archive.infolist()
        ]
        # Read only original relevant text tables; no raw spectra or PDF extraction.
        parsed = {}
        table_records = []
        for member in [
            "peptides.txt",
            "modificationSpecificPeptides.txt",
            "summary.txt",
            "parameters.txt",
            "proteinGroups.txt",
        ]:
            payload = archive.read(member)
            header, rows, cells, digest = table(payload)
            verified_cells += cells
            parsed[member] = records(header, rows)
            table_records.append(
                {
                    "name": member,
                    "rows": len(rows),
                    "columns": len(header),
                    "literal_cells_verified": cells,
                    "sha256": digest,
                }
            )
        peptides = parsed["peptides.txt"]
        forms = parsed["modificationSpecificPeptides.txt"]
        primary = parsed["summary.txt"][0]
        raw_name = primary["Raw file"] + ".raw"
        assert raw_name in directory_names
        intensity_key = [k for k in peptides[0] if k.startswith("Intensity ")]
        assert len(intensity_key) == 1
        assert all(p["Intensity"] == p[intensity_key[0]] for p in peptides)
        assert len({p["id"] for p in peptides}) == len(peptides)
        peptide_ids = {p["id"]: p["Sequence"] for p in peptides}
        assert all(peptide_ids[f["Peptide ID"]] == f["Sequence"] for f in forms)
        positive = [p for p in peptides if Decimal(p["Intensity"]) > 0]
        quality = [
            p
            for p in positive
            if Decimal(p["PEP"]) < Decimal("0.01")
            and not p["Reverse"]
            and not p["Potential contaminant"]
        ]
        params = {p["Parameter"]: p["Value"] for p in parsed["parameters.txt"]}
        matched_sdrf = [i + 2 for i, r in enumerate(sdrf["comment[data file]"]) if r == raw_name]
        summaries.append(
            {
                "archive": name,
                "members": members,
                "tables": table_records,
                "raw_file": raw_name,
                "sdrf_source_row_matches": matched_sdrf,
                "peptide_rows": len(peptides),
                "form_rows": len(forms),
                "positive_intensity": len(positive),
                "zero_intensity": sum(Decimal(p["Intensity"]) == 0 for p in peptides),
                "reverse_rows": sum(bool(p["Reverse"]) for p in peptides),
                "contaminant_rows": sum(bool(p["Potential contaminant"]) for p in peptides),
                "positive_quality_rows_before_intended_product_check": len(quality),
                "full_length_intended_product_eligibility": "unresolved; no planned-pool table admitted",
                "length_range_all_rows": [
                    min(int(p["Length"]) for p in peptides),
                    max(int(p["Length"]) for p in peptides),
                ],
                "intensity_sum_decimal_all_rows": str(
                    sum(Decimal(p["Intensity"]) for p in peptides)
                ),
                "modification_counts": dict(Counter(f["Modifications"] for f in forms)),
                "parameters": {
                    k: v
                    for k, v in params.items()
                    if any(
                        t in k.lower()
                        for t in ["version", "fdr", "length", "fasta", "modification"]
                    )
                },
                "search_context": {
                    k: primary[k]
                    for k in [
                        "Raw file",
                        "Experiment",
                        "Enzyme",
                        "Enzyme mode",
                        "Variable modifications",
                        "Multiplicity",
                    ]
                },
                "planned_peptide_molar_amount": "not supplied by these tables",
                "admitted_training_rows": 0,
            }
        )
write("pool-samples.json", summaries)

with zipfile.ZipFile(RAW / "article-2023-supplements.zip") as archive:
    payload = archive.read("mmc1.docx")
    with zipfile.ZipFile(io.BytesIO(payload)) as doc:
        xml = etree.fromstring(doc.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        ps = ["".join(p.itertext()) for p in xml.findall(".//w:p", ns)]
        captions = [p for p in ps if p.startswith("Table S")]
        write(
            "reanalysis-supplement-inventory.json",
            {
                "docx_sha256": sha(payload),
                "tables": len(xml.findall(".//w:tbl", ns)),
                "captions": captions,
                "embedded_files": [n for n in doc.namelist() if "/embeddings/" in n],
                "interpretation": "artificial proof-of-concept and model-result tables; not original pool quantities",
            },
        )
write(
    "source-summary.json",
    {
        "status": "original processed measurements recovered; quantitative training eligibility pending",
        "directory_files": len(files),
        "directory_extensions": dict(Counter(Path(r["name"]).suffix for r in files)),
        "literal_source_cells_independently_verified": verified_cells,
        "sampled_processed_archives": 2,
        "sample_selection": "first HLA1/HLA2 pool DDA archives by name, before quantity inspection",
        "peptide_rows_inspected": sum(s["peptide_rows"] for s in summaries),
        "form_rows_inspected": sum(s["form_rows"] for s in summaries),
        "admitted_training_rows": 0,
        "optimizer_updates": 0,
        "seconds": time.perf_counter() - started,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "limits": [
            "Two of1865 processed archives inspected; not a full cohort audit",
            "Crude synthesis and estimated injections do not certify per-peptide equimolarity",
            "No original intended-product/pool membership table admitted; byproducts not known-load standards",
            "Community SDRF values conflict with primary synthetic-material and instrument descriptions",
            "No planned-zero universe or detector failure mechanism inferred from missing identifications",
            "No cross-instrument/matrix or APC concentration validation",
        ],
    },
)
print((OUT / "source-summary.json").read_text())
