"""Independently regroup original source rows and verify workbook exceptions."""

import csv
import datetime
import hashlib
import json
import math
import resource
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import openpyxl

started = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
raw = Path(config["raw_directory"])
members = raw / "processed-members"
results = ROOT / "results"
summary = json.loads((results / "source-reconciliation.json").read_text())


def rows(name):
    with (members / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


peptides = rows("peptides.txt")
by_id = {p["id"]: p for p in peptides}
assert all(str(i) == p["id"] for i, p in enumerate(peptides))
evidence_sums, form_sums = defaultdict(list), defaultdict(list)
for row in rows("evidence.txt"):
    if row["Intensity"]:
        value = float(row["Intensity"])
        assert math.isfinite(value) and value.is_integer() and abs(value) < 2**53
        evidence_sums[row["Sequence"], row["Experiment"]].append(value)
samples = ("Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3")
for row in rows("modificationSpecificPeptides.txt"):
    for sample in samples:
        value = float(row["Intensity " + sample] or 0)
        assert math.isfinite(value) and value.is_integer() and abs(value) < 2**53
        form_sums[row["Sequence"], sample].append(value)
n_checked = 0
with (results / "quantity-reconciliation.csv").open(newline="") as stream:
    for record in csv.DictReader(stream):
        peptide = by_id[record["peptide_id"]]
        key = peptide["Sequence"], record["sample"]
        assert Decimal(record["source_intensity"]) == Decimal(peptide["Intensity " + record["sample"]])
        for kind, groups in (("evidence", evidence_sums), ("form", form_sums)):
            value = math.fsum(groups[key])
            assert value < 2**53
            assert Decimal(str(value)) == Decimal(record["sum_" + kind + "_intensity"])
        n_checked += 1
assert n_checked == len(peptides) * 6 == 63882

exceptions = json.loads((results / "workbook-exceptions.json").read_text())
literal = [e for e in exceptions if e["status"] == "literal_difference"]
assert len(literal) == 13 and all(e["field"] == "Gene names" for e in literal)
workbook_path = next(Path(a["path"]) for a in config["input_assets"] if a["path"].endswith(".xlsx"))
book = openpyxl.load_workbook(workbook_path, data_only=True, read_only=True)
date_exceptions = {e["workbook_cell"]: e for e in literal}
date_proof = []
for row in book["peptides"].iter_rows():
    for cell in row:
        if getattr(cell, "coordinate", None) not in date_exceptions:
            continue
        exception = date_exceptions[cell.coordinate]
        assert cell.is_date and isinstance(cell.value, datetime.datetime)
        assert cell.value == openpyxl.utils.datetime.from_excel(float(exception["workbook_value"]), epoch=book.epoch)
        assert row[0].value == exception["sequence"]
        assert by_id[exception["peptide_id"]]["Gene names"] == exception["text_value"]
        date_proof.append({**exception, "rendered_datetime": cell.value.isoformat(),
                           "number_format": cell.number_format,
                           "canonical_annotation_basis": "original_text_same_sequence_and_native_peptide_id_not_date_inference"})
book.close()
assert len(date_proof) == 13
with (results / "date-annotation-original-text.json").open("x") as output:
    output.write(json.dumps(date_proof, indent=2) + "\n")

design = next(Path(a["path"]) for a in config["input_assets"] if a["path"].endswith("experimentalDesignTemplate.txt"))
with design.open() as stream:
    design_map = {r["Experiment"]: r["Name"] for r in csv.DictReader(stream, delimiter="\t")}
mqpar = next(Path(a["path"]) for a in config["input_assets"] if a["path"].endswith("mqpar.xml"))
tree = ET.parse(mqpar)
paths = [Path(e.text.replace("\\", "/")).stem for e in tree.findall("filePaths/string")]
labels = [e.text for e in tree.findall("experiments/string")]
assert len(paths) == len(labels) == 6
assert dict(zip(labels, paths, strict=True)) == design_map == summary["run_mapping"]
quant_fields = ["Intensity", *("Intensity " + s for s in samples)]
for field in quant_fields:
    assert set(summary["field_comparison"][field]) <= {"literal_equal", "numeric_decimal_equal", "both_blank"}
lfq_different = sum(Decimal(p["Intensity " + s]) != Decimal(p["LFQ intensity " + s]) for p in peptides for s in samples)
result = dict(passed=True,n_sequence_sample_quantities_independently_regrouped=n_checked,
              n_workbook_intensity_cells_exactly_reconciled=len(peptides)*7,
              n_date_corrupted_gene_annotations_recovered_from_original_text=len(date_proof),
              original_workbook_bytes_repaired=False,n_run_mappings_independently_reconciled=6,
              n_raw_and_lfq_numeric_differences=lfq_different,
              no_identifier_link_errors=summary["n_identifier_link_errors"]==0,
              aggregation_is_not_decimal_exact=True,
              rounding_explanation_not_proven=True,
              no_predictive_or_condition_effect_metrics_computed=True,
              seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),argv=sys.argv)
assert result["no_identifier_link_errors"]
with (results / "independent-reconciliation-verification.json").open("x") as output:
    output.write(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
