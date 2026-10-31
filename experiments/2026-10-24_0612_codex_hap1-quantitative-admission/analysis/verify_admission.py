"""Check shared-package quantities against original cells without adapter helpers."""

import argparse
import csv
import hashlib
import io
import json
import math
import resource
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=Path)
parser.add_argument("--attempt", required=True)
args = parser.parse_args()
experiment = args.experiment.resolve()
record_dir = experiment / "results" / ("admission-" + args.attempt)
receipt = json.loads((record_dir / "result.json").read_text())
assert receipt["status"] == "source_package_built_and_replayed"
package = Path(receipt["destination"])
assert (
    hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
    == receipt["manifest_sha256"]
)
output = record_dir / "independent-source-check.json"
assert not output.exists()
shutil.copy2(__file__, record_dir / "source/verify_admission.py")
started = time.monotonic()
typed = package / "hap1/typed"
assets = package / "hap1/source_assets"


def records(name):
    for batch in pq.ParquetFile(typed / f"{name}.parquet").iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def expected_number(raw):
    if raw in (None, "", "NA"):
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


original = {}
with zipfile.ZipFile(assets / "supplements.zip") as archive:
    for member in ("mmc6.csv", "mmc7.csv", "mmc8.csv"):
        with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig") as stream:
            original[member] = list(csv.DictReader(stream))
with zipfile.ZipFile(assets / "Search.zip") as archive:
    with io.TextIOWrapper(
        archive.open("search_results_proteomics.tsv"), encoding="utf-8-sig"
    ) as stream:
        original["search_results_proteomics.tsv"] = list(csv.DictReader(stream, delimiter="\t"))

row_map = {}
for row in records("source_records"):
    member = row["source_locator"].split("!", 1)[1].rsplit(":row:", 1)[0]
    if member in original:
        index = row["source_row_index"] - 2
        assert json.loads(row["source_row_json"]) == original[member][index]
        row_map[row["source_record_id"]] = member, index
assert len(row_map) == sum(len(rows) for rows in original.values())

samples = {row["sample_id"]: row for row in records("samples")}
contexts = {row["context_id"]: row for row in records("contexts")}
runs = {row["run_id"]: row for row in records("runs")}
counts = Counter()
seen = {}
fields = {}
for member, rows in original.items():
    if member == "search_results_proteomics.tsv":
        names = [key for key in rows[0] if key.endswith(" Intensity")]
    else:
        names = [key for key in rows[0] if key not in ("peptide", "mapped_proteins", "protein_id")]
    fields[member] = {name: index for index, name in enumerate(names)}
    seen[member] = np.zeros((len(rows), len(names)), dtype=np.uint8)

for relation in ("quantities", "protein_quantities"):
    for row in records(relation):
        member, index = row_map[row["source_record_id"]]
        field = row["source_field"]
        original_row = original[member][index]
        raw = original_row[field]
        assert row["quantitative_raw"] == raw
        assert row["quantitative_value"] == expected_number(raw)
        column = fields[member][field]
        seen[member][index, column] += 1
        assert seen[member][index, column] == 1
        if relation == "quantities":
            label = field.split(".", 1)[0].rstrip("_")
            sample = samples[row["sample_id"]]
            assert sample["sample_label_raw"] == label
            assert sample["context_id"] == row["context_id"]
            modality = (
                "elution"
                if member in ("mmc6.csv", "mmc7.csv") or field.endswith(".sampling_density")
                else "proteomics"
            )
            assert sample["modality"] == modality
            assert contexts[row["context_id"]]["condition_id"] == sample["condition_id"]
            assert row["peptide_sequence"] == original_row.get("peptide")
            assert row["parent_label_raw"] == original_row.get(
                "mapped_proteins", original_row.get("protein_id")
            )
        else:
            run = runs[row["run_id"]]
            suffix = " MaxLFQ Intensity" if field.endswith(" MaxLFQ Intensity") else " Intensity"
            assert run["source_run_raw"] == field[: -len(suffix)]
            assert row["quantity_kind"] == suffix.strip()
            assert row["sample_id"] == run["sample_id"]
            assert row["context_id"] == run["context_id"]
            assert row["parent_label_raw"] == original_row["Protein"]
        counts[relation + ":n_cells"] += 1
for member, array in seen.items():
    assert np.all(array == 1), member
del original, seen, row_map

with zipfile.ZipFile(assets / "Search.zip") as archive:
    with io.TextIOWrapper(archive.open("Search_results_DIA.tsv"), encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for index, (source_row, row) in enumerate(
            zip(reader, records("precursors"), strict=True), 2
        ):
            assert row["source_row_index"] == index
            assert json.loads(row["source_row_json"]) == source_row
            assert row["raw_quantity_text"] == source_row["Precursor.Quantity"]
            assert row["normalized_quantity_text"] == source_row["Precursor.Normalised"]
            assert row["raw_quantity"] == expected_number(source_row["Precursor.Quantity"])
            assert row["normalized_quantity"] == expected_number(source_row["Precursor.Normalised"])
            assert row["peptide_sequence"] == source_row["Stripped.Sequence"]
            run = runs[row["run_id"]]
            assert run["source_run_raw"] == source_row["Run"]
            assert row["sample_id"] == run["sample_id"]
            assert row["context_id"] == run["context_id"]
            assert json.loads(row["confidence_json"]) == {
                key: value for key, value in source_row.items() if "Q.Value" in key or "PEP" in key
            }
            counts["precursors:n_rows"] += 1
            counts["precursors:n_unmapped_rows"] += row["sample_id"] is None

references = {row["reference_record_id"]: row for row in records("references")}
for row in records("occurrences"):
    if row["reference_record_id"]:
        reference = references[row["reference_record_id"]]
        start, end = row["start_1based"] - 1, row["end_1based"]
        assert reference["reference_sequence"][start:end] == row["peptide_sequence"]
        counts["occurrences:n_exact_matches"] += 1
    else:
        assert row["start_1based"] is None and row["end_1based"] is None
        counts["occurrences:n_unmapped_rows"] += 1

result = {
    "status": "all_checked_source_cells_equal",
    "package_manifest_sha256": receipt["manifest_sha256"],
    "counts": dict(counts),
    "elapsed_seconds": time.monotonic() - started,
    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    "scope": "every supplement and deposited protein quantity; every DIA source row; sample/run links; validity of every emitted exact occurrence",
    "limitations": [
        "does_not_independently_prove_exhaustiveness_of_reference_search",
        "not_predictive_evaluation_or_specimen_independence_validation",
    ],
}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
