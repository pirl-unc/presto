"""Source-only cell, identifier and quantity-aggregation reconciliation."""

import csv
import hashlib
import importlib.util
import json
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

started = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
RAW = Path(CONFIG["raw_directory"])
MEMBERS = RAW / "processed-members"
RESULTS = ROOT / "results"
reader_path = ROOT / "reproduce/source/workbook_reader.py"
spec = importlib.util.spec_from_file_location("frozen_shared_reader", reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
samples = ["Cont1", "Cont2", "Cont3", "KD1", "KD2", "KD3"]


def table(name):
    with (MEMBERS / name).open(newline="", encoding="utf-8-sig") as stream:
        source = csv.DictReader(stream, delimiter="\t")
        rows = list(source)
        assert len(source.fieldnames) == len(set(source.fieldnames))
        assert all(None not in row for row in rows)
        return source.fieldnames, rows


def number(value):
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def ids(value):
    return [] if not value else value.split(";")


def guard():
    assert time.monotonic() - started < CONFIG["limits"]["stage_seconds"]
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < CONFIG["limits"]["rss_bytes"]


headers, peptides = table("peptides.txt")
_, evidence = table("evidence.txt")
_, forms = table("modificationSpecificPeptides.txt")
by_sequence = {row["Sequence"]: row for row in peptides}
by_peptide = {row["id"]: row for row in peptides}
by_evidence = {row["id"]: row for row in evidence}
by_form = {row["id"]: row for row in forms}
assert len(by_sequence) == len(by_peptide) == len(peptides)
assert len(by_evidence) == len(evidence) and len(by_form) == len(forms)
workbook = next(Path(a["path"]) for a in CONFIG["input_assets"] if a["path"].endswith(".xlsx"))
cell_counts = Counter()
field_counts = defaultdict(Counter)
exceptions = []
workbook_sequences = []
for sheet, row_index, cells in reader.workbook_rows(workbook):
    guard()
    assert sheet == "peptides"
    values = {c["source_column_index"]: c for c in cells}
    if row_index == 1:
        workbook_headers = {i: c["cell_value_resolved"] for i, c in values.items()}
        assert all(h in headers for h in workbook_headers.values())
        continue
    sequence = values[1]["cell_value_resolved"]
    workbook_sequences.append(sequence)
    original = by_sequence[sequence]
    for column, field in workbook_headers.items():
        cell = values.get(column)
        published = None if cell is None else cell["cell_value_resolved"]
        source = original[field]
        if published in (None, "") and source == "":
            status = "both_blank"
        elif published == source:
            status = "literal_equal"
        elif field == "Gene names" and cell and cell["cell_style_raw"] == "17" and number(published) is not None:
            status = "excel_date_corruption_original_text_retained"
        elif number(published) is not None and number(source) is not None:
            if number(published) == number(source):
                status = "numeric_decimal_equal"
            elif math.isclose(float(published), float(source), rel_tol=2e-15, abs_tol=0):
                status = "within_two_parts_per_quadrillion_not_decimal_equal"
            else:
                status = "numeric_difference"
        else:
            status = "literal_difference"
        cell_counts[status] += 1
        field_counts[field][status] += 1
        if status not in ("both_blank", "literal_equal", "numeric_decimal_equal"):
            exceptions.append(dict(peptide_id=original["id"], sequence=sequence, source_row=row_index,
                                   field=field, workbook_value=published, text_value=source,
                                   workbook_cell=None if cell is None else cell["source_cell"], status=status))
assert set(workbook_sequences) == set(by_sequence) and len(workbook_sequences) == len(peptides)
with (RESULTS / "workbook-exceptions.json").open("x") as output:
    output.write(json.dumps(exceptions, indent=2) + "\n")

actual_evidence = defaultdict(set)
actual_forms = defaultdict(set)
link_errors = []
for row in evidence:
    actual_evidence[row["Peptide ID"]].add(row["id"])
    if row["Peptide ID"] not in by_peptide or row["Sequence"] != by_peptide[row["Peptide ID"]]["Sequence"]:
        link_errors.append(["evidence_sequence", row["id"]])
    if row["Experiment"] not in samples:
        link_errors.append(["evidence_sample", row["id"]])
    form = by_form.get(row["Mod. peptide ID"])
    if form is None or form["Sequence"] != row["Sequence"] or row["id"] not in ids(form["Evidence IDs"]):
        link_errors.append(["evidence_form", row["id"]])
for row in forms:
    actual_forms[row["Peptide ID"]].add(row["id"])
    if row["Peptide ID"] not in by_peptide or row["Sequence"] != by_peptide[row["Peptide ID"]]["Sequence"]:
        link_errors.append(["form_sequence", row["id"]])
for row in peptides:
    if actual_evidence[row["id"]] != set(ids(row["Evidence IDs"])):
        link_errors.append(["peptide_evidence_membership", row["id"]])
    if actual_forms[row["id"]] != set(ids(row["Mod. peptide IDs"])):
        link_errors.append(["peptide_form_membership", row["id"]])
with (RESULTS / "identifier-link-errors.json").open("x") as output:
    output.write(json.dumps(link_errors, indent=2) + "\n")

counts = defaultdict(Counter)
max_relative = defaultdict(float)
summary_fields = ["peptide_id", "sample", "source_intensity", "sum_evidence_intensity", "sum_form_intensity",
                  "evidence_residual", "form_residual", "n_evidence_rows", "n_modified_evidence_rows",
                  "n_missing_evidence_intensities", "n_form_rows", "source_identification_type", "source_lfq_intensity"]
with (RESULTS / "quantity-reconciliation.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=summary_fields, lineterminator="\n")
    writer.writeheader()
    for peptide in peptides:
        guard()
        sequence_evidence = [by_evidence[i] for i in ids(peptide["Evidence IDs"])]
        sequence_forms = [by_form[i] for i in ids(peptide["Mod. peptide IDs"])]
        for sample in samples:
            source_value = number(peptide["Intensity " + sample])
            sample_evidence = [e for e in sequence_evidence if e["Experiment"] == sample]
            numeric_evidence = [number(e["Intensity"]) for e in sample_evidence]
            sums = {
                "evidence": sum((v for v in numeric_evidence if v is not None), Decimal(0)),
                "form": sum((number(f["Intensity " + sample]) or Decimal(0) for f in sequence_forms), Decimal(0)),
            }
            assert source_value is not None
            for kind, value in sums.items():
                counts[kind]["exact_equal" if value == source_value else "numeric_difference"] += 1
                if source_value > 0:
                    max_relative[kind] = max(max_relative[kind], float(abs(value-source_value)/source_value))
                elif value != 0:
                    counts[kind]["nonzero_sum_source_zero"] += 1
            counts["raw_status"]["positive" if source_value > 0 else "zero" if source_value == 0 else "negative"] += 1
            lfq = number(peptide["LFQ intensity " + sample])
            counts["lfq_status"]["missing" if lfq is None else "positive" if lfq > 0 else "zero"] += 1
            if source_value > 0 and any(e["Modifications"] != "Unmodified" and (number(e["Intensity"]) or 0) > 0 for e in sample_evidence):
                counts["support"]["positive_sequence_sample_cells_with_modified_evidence"] += 1
            writer.writerow(dict(peptide_id=peptide["id"],sample=sample,source_intensity=source_value,
                                 sum_evidence_intensity=sums["evidence"],sum_form_intensity=sums["form"],
                                 evidence_residual=sums["evidence"]-source_value,form_residual=sums["form"]-source_value,
                                 n_evidence_rows=len(sample_evidence),n_modified_evidence_rows=sum(e["Modifications"] != "Unmodified" for e in sample_evidence),
                                 n_missing_evidence_intensities=sum(v is None for v in numeric_evidence),n_form_rows=len(sequence_forms),
                                 source_identification_type=peptide["Identification type " + sample],source_lfq_intensity=lfq))

_, summaries = table("summary.txt")
design = next(Path(a["path"]) for a in CONFIG["input_assets"] if a["path"].endswith("experimentalDesignTemplate.txt"))
with design.open() as stream:
    design_rows = list(csv.DictReader(stream, delimiter="\t"))
source_runs = {r["Experiment"]: r["Raw file"] for r in summaries if r["Raw file"] != "Total"}
evidence_runs = {r["Experiment"]: r["Raw file"] for r in evidence}
assert len(source_runs) == 6 and source_runs == evidence_runs
assert all(r["Raw file"] == source_runs[r["Experiment"]] for r in evidence)
result = dict(
    n_peptides=len(peptides), n_peptide_fields=len(headers), n_evidence_rows=len(evidence), n_modification_specific_rows=len(forms),
    n_workbook_fields=len(workbook_headers), n_common_source_cells=sum(cell_counts.values()),
    common_cell_comparison=dict(cell_counts), field_comparison={k:dict(v) for k,v in field_counts.items()},
    n_identifier_link_errors=len(link_errors), quantity_comparison={k:dict(v) for k,v in counts.items()},
    max_relative_aggregation_residual=dict(max_relative), evidence_types=dict(Counter(r["Type"] for r in evidence)),
    n_sequences_with_multiple_modification_rows=sum(len(v)>1 for v in actual_forms.values()),
    run_mapping=source_runs, metadata_fields_kept_distinct=["Intensity", "LFQ intensity", "modification-specific intensity", "evidence intensity"],
    all_aggregation_residuals_retained=True, decimal_equality_required_for_exact_claim=True,
    numeric_machine_comparison_relative_tolerance=2e-15,
    no_condition_effects_or_predictive_metrics_computed=True,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),argv=sys.argv,
    shared_reader_sha256=hashlib.sha256(reader_path.read_bytes()).hexdigest(),
    seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (RESULTS / "source-reconciliation.json").open("x") as stream:
    stream.write(json.dumps(result, indent=2) + "\n")
print(json.dumps({k:v for k,v in result.items() if k != "field_comparison"}, indent=2))
