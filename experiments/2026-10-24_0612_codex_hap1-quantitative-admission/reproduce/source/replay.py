"""Independently compare source supplements with original deposited quantities.

This diagnoses field meaning and derivation; it is not a model input curator.
"""

import csv
import gzip
import hashlib
import io
import json
import math
import resource
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

experiment = Path(__file__).resolve().parents[1]
config = json.loads((experiment / "reproduce/launch.json").read_text())
root = Path(config["source_root"])
output = experiment / "results/replay"
output.mkdir(exist_ok=False)
shutil.copy2(__file__, experiment / "reproduce/source/replay.py")
started = time.monotonic()
receipt = json.loads((experiment / "results/deposit-Search.zip.json").read_text())
archive_path = root / "deposits/Search.zip"
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == receipt["actual"]["sha256"]
inventory = json.loads((experiment / "results/source-inventory.json").read_text())
sample_rows = inventory["sample_sheets"]["immunopep_library_search"]["rows"]
run_names = [row["raw_file"] for row in sample_rows]
assert len(set(run_names)) == len(run_names)
run_index = {name: index for index, name in enumerate(run_names)}
sample_runs = defaultdict(list)
for index, row in enumerate(sample_rows):
    sample_runs[f"{row['knockout_condition']}_{row['biological_replicate']}"].append(index)
summary = {"purpose": "source derivation replay, not predictive evaluation"}
fields = ["Precursor.Quantity", "Precursor.Normalised"]
maxima = {}
run_sequence_sum = defaultdict(lambda: [0.0, 0.0])
row_counts = Counter()
confidence = Counter()
unknown_runs = Counter()
duplicated_keys = 0
precursor_identity_conflicts = 0

with zipfile.ZipFile(archive_path) as archive:
    summary["archive_members"] = [
        {"name": item.filename, "size_bytes": item.file_size, "crc32": item.CRC}
        for item in archive.infolist()
    ]
    with archive.open("Search_results_DIA.tsv") as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"), delimiter="\t")
        summary["dia_headers"] = reader.fieldnames
        for source_row, row in enumerate(reader, start=2):
            name = row["Run"]
            row_counts[name] += 1
            if name not in run_index:
                unknown_runs[name] += 1
                continue
            for field in ["Q.Value", "Global.Q.Value", "Lib.Q.Value"]:
                value = float(row[field])
                confidence[f"{field}:above_0.01"] += value > 0.01
            run = run_index[name]
            sequence = row["Stripped.Sequence"]
            key = run, row["Precursor.Id"]
            values = [float(row[field]) for field in fields]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            previous = maxima.get(key)
            if previous:
                duplicated_keys += 1
                if previous[0] != sequence:
                    precursor_identity_conflicts += 1
                    raise ValueError("One source precursor ID has different stripped sequences")
                old_values = previous[1]
            else:
                old_values = [0.0, 0.0]
            new_values = [max(old, new) for old, new in zip(old_values, values)]
            totals = run_sequence_sum[run, sequence]
            for j in range(2):
                totals[j] += new_values[j] - old_values[j]
            maxima[key] = sequence, new_values
    summary.update(
        n_dia_rows=sum(row_counts.values()),
        n_dia_runs=len(row_counts),
        n_unique_precursor_run_keys=len(maxima),
        n_duplicate_precursor_run_keys=duplicated_keys,
        unknown_runs=dict(unknown_runs),
        confidence_counts=dict(confidence),
        n_run_peptide_aggregates=len(run_sequence_sum),
    )
    del maxima
    protein_rows = {}
    protein_duplicates = Counter()
    with archive.open("search_results_proteomics.tsv") as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"), delimiter="\t")
        summary["proteomics_headers"] = reader.fieldnames
        for row in reader:
            key = row["Protein"]
            if key in protein_rows:
                protein_duplicates[key] += 1
            protein_rows[key] = row
    summary["n_proteomics_rows"] = len(protein_rows)
    summary["protein_duplicate_ids"] = dict(protein_duplicates)
    assert not protein_duplicates
    original_book = openpyxl.load_workbook(
        io.BytesIO(archive.read("Raw_file_list.xlsx")), read_only=True, data_only=False
    )
    summary["deposited_sample_sheets"] = {
        sheet.title: list(sheet.iter_rows(values_only=True)) for sheet in original_book
    }

stats = defaultdict(Counter)
max_errors = defaultdict(float)


def compare(writer, table, row_number, field, identifier, expected_text, method, value):
    if expected_text == "NA":
        status = "source_NA"
        stats[method]["n_source_NA"] += 1
        if value is not None:
            stats[method]["n_NA_with_deposit_value"] += 1
        expected = error = None
    else:
        expected = float(expected_text)
        if value is None:
            status = "deposit_missing"
            stats[method]["n_missing_deposit"] += 1
            error = None
        else:
            error = value - expected
            stats[method]["n_compared"] += 1
            stats[method]["n_exact"] += value == expected
            close = math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6)
            stats[method]["n_close_rtol_1e-6_atol_1e-6"] += close
            max_errors[method] = max(max_errors[method], abs(error))
            status = "close" if close else "different"
    writer.writerow(
        [table, row_number, field, identifier, expected_text, method, value, error, status]
    )


headers = [
    "source_table",
    "row_1based",
    "source_field",
    "source_id",
    "source_value_raw",
    "diagnostic_method",
    "recomputed_value",
    "signed_difference",
    "comparison_status",
]
with zipfile.ZipFile(root / "retrieval/europepmc-supplements/body") as supplements:
    with gzip.open(output / "peptide-comparisons.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        reader = csv.DictReader(
            io.TextIOWrapper(supplements.open("mmc6.csv"), encoding="utf-8-sig")
        )
        for index, row in enumerate(reader, start=2):
            sequence = row["peptide"]
            for field in reader.fieldnames[2:]:
                sample = field.removesuffix("_.sampling_density")
                assert sample in sample_runs
                runs = sample_runs[sample]
                observed = [
                    run_sequence_sum[run, sequence]
                    for run in runs
                    if (run, sequence) in run_sequence_sum
                ]
                for j, quantity in enumerate(fields):
                    total = math.fsum(value[j] for value in observed)
                    for policy in ["observed_technical_runs", "all_declared_technical_runs"]:
                        denominator = len(observed) if policy.startswith("observed") else len(runs)
                        value = total / denominator if observed else None
                        compare(
                            writer,
                            "mmc6.csv",
                            index,
                            field,
                            sequence,
                            row[field],
                            f"{quantity}:{policy}",
                            value,
                        )
    protein_samples = inventory["sample_sheets"]["proteomics"]["rows"]
    protein_run = {
        f"{row['knockout_condition']}_{row['biological_replicate']}": row["raw_file"]
        for row in protein_samples
    }
    with gzip.open(output / "protein-comparisons.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        reader = csv.DictReader(
            io.TextIOWrapper(supplements.open("mmc8.csv"), encoding="utf-8-sig")
        )
        for index, row in enumerate(reader, start=2):
            protein = row["protein_id"]
            deposited = protein_rows.get(protein)
            for field in reader.fieldnames:
                if not field.endswith(".intensity"):
                    continue
                sample = field.removesuffix(".intensity")
                name = protein_run[sample]
                for quantity in ["Intensity", "MaxLFQ Intensity"]:
                    value = float(deposited[f"{name} {quantity}"]) if deposited else None
                    compare(writer, "mmc8.csv", index, field, protein, row[field], quantity, value)

summary["comparisons"] = {
    key: {**value, "max_absolute_difference": max_errors[key]} for key, value in stats.items()
}
summary["elapsed_seconds"] = time.monotonic() - started
summary["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
summary["invocation"] = sys.argv
summary["source_semantics"] = (
    "No training admission; inspect replay differences before assigning units"
)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in summary.items()
            if k not in ["dia_headers", "proteomics_headers", "deposited_sample_sheets"]
        },
        indent=2,
    )
)
