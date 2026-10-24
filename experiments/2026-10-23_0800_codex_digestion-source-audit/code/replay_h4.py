"""Independent Python arithmetic replay from original serialized/CSV layers."""

import csv
import gzip
import io
import json
import math
import statistics
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/private/tmp/presto-digestion-assets/original-layers")


def rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def main():
    started = time.monotonic()
    directory = CACHE / "aSPIRE_H4"
    raw = list(rows(directory / "QUANTITIES_raw.csv.gz"))
    metadata = json.loads((ROOT / "results/identifier-raw-cells.json").read_text())
    runs = {
        r[3].removesuffix(".raw"): r
        for r in metadata["sheets"]["Main_searches"]["rows"][1:]
        if r[3]
    }
    mismatches = []
    for row in raw:
        source = runs[row["source"]]
        if float(row["digestTime"]) != source[1] or row["biological_replicate"] != str(
            source[2]
        ):
            mismatches.append(
                {
                    "source": row["source"],
                    "original_time": row["digestTime"],
                    "metadata_time": source[1],
                }
            )
    assert not mismatches
    # Skyline precursor values are repeated over isotope/fragment rows. Match the
    # authors' declared unique-column relation before summing charge/assignment rows.
    totals, unique = defaultdict(float), set()
    with zipfile.ZipFile(CACHE / "aSPIRE_H4.zip") as archive:
        with archive.open("aSPIRE_H4/ASSIGNMENTS.csv") as handle:
            assignments = {r["ID"] for r in csv.DictReader(io.TextIOWrapper(handle))}
        with archive.open("aSPIRE_H4/MS1_HPR.csv") as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle)):
                if (
                    row["FragmentIon"] != "precursor"
                    or row["ProteinName"] not in assignments
                ):
                    continue
                key = tuple(
                    row[k]
                    for k in (
                        "ProteinName",
                        "FileName",
                        "PeptideSequence",
                        "TotalAreaMs1",
                        "TotalBackgroundMs1",
                        "FragmentIon",
                        "ProductCharge",
                        "ProductMz",
                        "AverageMeasuredRetentionTime",
                        "PeptideModifiedSequence",
                    )
                )
                if key in unique:
                    continue
                unique.add(key)
                value = sum(
                    0.0 if row[k] in ("#N/A", "NA", "") else float(row[k])
                    for k in ("TotalAreaMs1", "TotalBackgroundMs1")
                )
                totals[
                    row["FileName"].removesuffix(".raw"), row["PeptideSequence"]
                ] += value
    raw_errors = []
    for row in raw:
        expected = totals[row["source"], row["pepSeq"]]
        actual = float(row["intensity"])
        raw_errors.append(abs(expected - actual) / max(1, abs(actual)))
    assert max(raw_errors) < 1e-12, max(raw_errors)
    assert len(totals) == len(raw)

    # Independent implementation of the published 1.0 postprocessing operations.
    filtered = list(rows(directory / "QUANTITIES_filtered.csv.gz"))
    original = []
    for row in filtered:
        for t, v in zip(
            row["times_pasted"].split(";"), row["int_pasted"].split(";"), strict=True
        ):
            original.append(
                (
                    row["pepSeq"],
                    row["biological_replicate"],
                    int(row["technical_replicate"]),
                    float(t),
                    float(v),
                )
            )
    il_sums = defaultdict(float)
    for pep, bio, tech, t, v in original:
        il_sums[pep.replace("I", "L"), bio, tech, t] += v
    expanded = [
        (
            p,
            b,
            1 if a == 3 else 2 if a == 4 else a,
            t,
            il_sums[p.replace("I", "L"), b, a, t],
        )
        for p, b, a, t, v in original
    ]
    baseline = defaultdict(float)
    for p, b, a, t, v in expanded:
        if t == 0:
            baseline[p, b, a] = max(baseline[p, b, a], v)
    noise = statistics.median(baseline.values())
    corrected = defaultdict(list)
    below_background = 0
    for p, b, a, t, v in expanded:
        value = 0.0 if t == 0 else v - max(noise, baseline[p, b, a])
        if t > 0 and value <= 0:
            below_background += 1
        else:
            corrected[p, b, t].append(value)
    final_errors, missing_matches, values = [], 0, 0
    with (directory / "H4_finalKinetics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["biological_replicate"] == "NA":
                continue
            for t, v in zip(
                row["digestTimes"].split(";"),
                row["intensities"].split(";"),
                strict=True,
            ):
                group = corrected[row["pepSeq"], row["biological_replicate"], float(t)]
                values += 1
                if not group:
                    assert v in ("NA", "NaN")
                    missing_matches += 1
                    continue
                actual = float(v)
                assert math.isfinite(actual)
                expected = statistics.mean(group)
                final_errors.append(abs(expected - actual) / max(1, abs(actual)))
    assert max(final_errors) < 1e-11, max(final_errors)
    result = {
        "raw_quantity_rows_replayed": len(raw),
        "raw_unique_precursor_rows": len(unique),
        "max_raw_relative_error": max(raw_errors),
        "all_raw_run_metadata_joins_match": True,
        "filtered_rows": len(filtered),
        "expanded_filtered_values": len(original),
        "global_background_median": noise,
        "values_censored_by_background": below_background,
        "final_vector_values_replayed": values,
        "final_missing_values_replayed": missing_matches,
        "max_final_relative_error": max(final_errors),
        "runtime_seconds": time.monotonic() - started,
        "limits": "H4 only; synthesis-error mapping/removal not independently replayed. No training admission.",
    }
    (ROOT / "results/h4-arithmetic-replay.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
