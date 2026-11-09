"""Verify original intended-pool membership and qualify two measured source samples."""

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

import openpyxl
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[1] / "data/measured_ms1_source_qualification/2026-10-24_1546_v1"
OUT = ROOT / "results"
PATH = RAW / "article-2021-planned-peptides.xlsx"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_table(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig")), delimiter="\t"))


started = time.perf_counter()
assert not (OUT / "planned-product-verification.json").exists()
mapping = {}
sheet_summaries = []
checked = 0
with zipfile.ZipFile(PATH) as archive:
    # Independently decode original OOXML string values and populated cell coordinates.
    shared = etree.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = ["".join(si.itertext()) for si in shared]
    workbook = openpyxl.load_workbook(PATH, read_only=True, data_only=False)
    for sheet_name, member in [
        ("HLA Class I", "xl/worksheets/sheet2.xml"),
        ("HLA Class II", "xl/worksheets/sheet3.xml"),
        ("AspN", "xl/worksheets/sheet4.xml"),
        ("LysN", "xl/worksheets/sheet5.xml"),
    ]:
        by_row = {}
        for _, cell in etree.iterparse(archive.open(member), events=("end",), tag=NS + "c"):
            formula = cell.find(NS + "f")
            assert formula is None, (
                "Preserve formulas; do not silently treat them as literal mappings"
            )
            v = cell.find(NS + "v")
            if v is not None and v.text is not None:
                coordinate = cell.get("r")
                value = strings[int(v.text)] if cell.get("t") == "s" else v.text
                if value != "":
                    column = coordinate.rstrip("0123456789")
                    assert column in {"A", "B"}, (sheet_name, coordinate, value)
                    row = int(coordinate[len(column) :])
                    by_row.setdefault(row, {})[column] = value
            assert cell.get("t") != "inlineStr", "Explicitly handle a new source representation"
            cell.clear()
            # Avoid retaining styled empty rows in the parser's tree.
            parent = cell.getparent()
            while cell.getprevious() is not None:
                del parent[0]
            grandparent = parent.getparent()
            while parent.getprevious() is not None:
                del grandparent[0]
        assert by_row[1] == {"A": "Pool name", "B": "Sequence"}
        last_row = max(by_row)
        observed_rows = {}
        for number, row in enumerate(
            workbook[sheet_name].iter_rows(
                min_row=1, max_row=last_row, min_col=1, max_col=2, values_only=True
            ),
            1,
        ):
            nonempty = {
                column: value
                for column, value in zip(["A", "B"], row, strict=True)
                if value not in (None, "")
            }
            if nonempty:
                observed_rows[number] = nonempty
        assert observed_rows == by_row
        checked += sum(len(row) for row in by_row.values())
        pairs = [
            (number, row["A"], row["B"]) for number, row in sorted(by_row.items()) if number != 1
        ]
        assert all(len(row) == 2 for row in by_row.values())
        for number, pool, sequence in pairs:
            mapping.setdefault(pool, {}).setdefault(sequence, []).append(
                {"sheet": sheet_name, "row": number}
            )
        sheet_summaries.append(
            {
                "sheet": sheet_name,
                "declared_dimension_rows": workbook[sheet_name].max_row,
                "last_populated_row": last_row,
                "mapping_rows": len(pairs),
                "unique_sequences": len({p[2] for p in pairs}),
                "pools": len({p[1] for p in pairs}),
                "populated_cells_verified": sum(len(row) for row in by_row.values()),
                "literal_pair_sha256": hashlib.sha256(
                    json.dumps(pairs, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    workbook.close()

summaries = []
dispositions = []
for pool, source in [("TUM_HLA_1", "HLA1-first-DDA.zip"), ("TUM_HLA2_1", "HLA2-first-DDA.zip")]:
    planned = mapping[pool]
    with zipfile.ZipFile(RAW / source) as archive:
        rows = read_table(archive, "peptides.txt")
        forms = read_table(archive, "modificationSpecificPeptides.txt")
    by_sequence = {r["Sequence"]: r for r in rows}
    assert len(by_sequence) == len(rows)
    counts = Counter()
    for i, row in enumerate(rows, 2):
        seq = row["Sequence"]
        flags = {
            "planned_in_this_pool": seq in planned,
            "positive": Decimal(row["Intensity"]) > 0,
            "pep_lt_0_01": Decimal(row["PEP"]) < Decimal("0.01"),
            "reverse": bool(row["Reverse"]),
            "contaminant": bool(row["Potential contaminant"]),
        }
        qualifies = (
            flags["planned_in_this_pool"]
            and flags["positive"]
            and flags["pep_lt_0_01"]
            and not flags["reverse"]
            and not flags["contaminant"]
        )
        for key, value in flags.items():
            counts[key] += int(value)
        counts["planned_positive_quality"] += int(qualifies)
        dispositions.append(
            {
                "archive": source,
                "source_member": "peptides.txt",
                "source_row": i,
                "source_id": row["id"],
                "sequence": seq,
                **flags,
                "planned_positive_quality": qualifies,
                "intensity_literal": row["Intensity"],
                "pep_literal": row["PEP"],
                "planned_source_cells": json.dumps(planned.get(seq, []), separators=(",", ":")),
                "training_admitted": False,
            }
        )
    candidates = {
        d["sequence"]
        for d in dispositions
        if d["archive"] == source and d["planned_positive_quality"]
    }
    summaries.append(
        {
            "pool": pool,
            "archive": source,
            "planned_unique_sequences": len(planned),
            "planned_mapping_rows": sum(len(v) for v in planned.values()),
            "observed_peptide_rows": len(rows),
            **dict(counts),
            "planned_sequences_not_in_peptides_table": len(set(planned) - set(by_sequence)),
            "observed_sequences_not_planned_in_this_pool": len(set(by_sequence) - set(planned)),
            "unmodified_positive_quality_form_rows_for_candidate_sequences": sum(
                f["Sequence"] in candidates
                and f["Modifications"] == "Unmodified"
                and Decimal(f["Intensity"]) > 0
                and Decimal(f["PEP"]) < Decimal("0.01")
                and not f["Reverse"]
                and not f["Potential contaminant"]
                for f in forms
            ),
            "missing_planned_sequence_interpretation": "not observed here; synthesis/solubility/LC/MS/identification unresolved",
            "nominal_load_assignment": "primary protocol estimates200or500fmol; per-run choice not reconciled",
            "individual_synthesis_purity_or_molar_amount": "not established by the planned sequence table",
        }
    )
with (OUT / "planned-product-dispositions.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(dispositions[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(dispositions)
write(
    "planned-product-verification.json",
    {
        "status": "pass",
        "source_sha256": hashlib.sha256(PATH.read_bytes()).hexdigest(),
        "method": "OOXML cell/string decoding compared with independent openpyxl reader, all populated map cells",
        "sheets": sheet_summaries,
        "populated_mapping_cells_verified": checked,
        "sampled_pool_eligibility": summaries,
        "uninspected_scope": "Legend and aggregate Identifications sheet not quantitatively audited; other biological supplements excluded",
        "admitted_training_rows": 0,
        "canonical_hitlist_recipe_required": True,
        "seconds": time.perf_counter() - started,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    },
)
print((OUT / "planned-product-verification.json").read_text(), flush=True)
