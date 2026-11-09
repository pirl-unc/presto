"""Independent original-byte to typed-value replay; does not import Hitlist."""

import csv
import gzip
import hashlib
import io
import json
import math
import resource
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import openpyxl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
PACKAGE = Path(CONFIG["output_package"])
SOURCE = PACKAGE / "external/soh2024_digestion"
SID = "soh2024_digestion"


def rows(name):
    for batch in pq.ParquetFile(SOURCE / "typed" / (name + ".parquet")).iter_batches(
        batch_size=1024
    ):
        yield from batch.to_pylist()


def value(text):
    if text in (None, "", "NA", "#N/A", "NaN", "nan"):
        return None
    result = float(text)
    return result if math.isfinite(result) else None


def main():
    start = time.perf_counter()
    queries = json.loads((ROOT / "results/canonical-queries.json").read_text())
    streams = {
        k.removeprefix("digestion_"): rows(k.removeprefix("digestion_"))
        for k in queries
    }
    digests = {k: hashlib.sha256() for k in streams}
    counts = Counter()

    def take(table, raw=None, locator=None, number=None):
        row = next(streams[table])
        digests[table].update(
            (
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode()
        )
        counts[table] += 1
        if raw is not None:
            assert json.loads(row["fields_json"]) == raw, (table, locator, number)
        if locator is not None:
            assert row["source_locator"] == f"{locator}:row:{number}"
            assert row["source_record_id"] == f"{SID}:{locator}:row:{number}"
        return row

    # Independently hash every package artifact, including original assets.
    manifest = json.loads((PACKAGE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256()
        with (PACKAGE / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == expected["sha256"], name
        assert (PACKAGE / name).stat().st_size == expected["size_bytes"]
    runs = {}
    book = openpyxl.load_workbook(
        SOURCE / "assets/Identifier.xlsx", read_only=True, data_only=False
    )
    for sheet in book:
        values = sheet.iter_rows(values_only=True)
        header = next(values)
        for number, raw in enumerate(values, 2):
            take("metadata", list(raw))
            record = dict(zip(header, raw))
            if sheet.title == "Main_searches":
                run = take("runs", list(raw))
                assert run["raw_file"] == record["raw_file"]
                assert run["time_hours"] == float(record["digestTime"])
                assert run["biological_replicate_raw"] == str(
                    record["biological_replicate"]
                )
                runs[record["raw_file"].removesuffix(".raw")] = run
    book.close()
    contract = json.loads((SOURCE / "contracts/contract.json").read_text())
    constructs = {}
    for filename in sorted(set(contract["construct_files"].values())):
        lines = (SOURCE / "assets/substrate-fastas" / filename).read_text().splitlines()
        row = take("constructs", lines)
        sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
        assert row["sequence"] == sequence and row["n_residues"] == len(sequence)
        assert row["sequence_sha256"] == hashlib.sha256(sequence.encode()).hexdigest()
        constructs[filename] = sequence
    conflicts, zero_slots, missing_slots, annotation_rows = Counter(), 0, 0, 0
    with (
        zipfile.ZipFile(SOURCE / "assets/aSPIRE_outputs_zip.zip") as outer,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp = Path(tmp)
        for member in outer.infolist():
            if member.is_dir():
                continue
            condition_name = Path(member.filename).stem
            condition_id = f"{SID}:condition:{condition_name}"
            with outer.open(member) as src, (tmp / "inner.zip").open("wb") as target:
                shutil.copyfileobj(src, target, 1024 * 1024)
            with zipfile.ZipFile(tmp / "inner.zip") as inner:
                assert inner.testzip() is None
                for entry in inner.infolist():
                    if Path(entry.filename).name in (
                        "QUANTITIES_raw.RData",
                        "QUANTITIES_filtered.RData",
                    ):
                        (tmp / Path(entry.filename).name).write_bytes(inner.read(entry))
                subprocess.run(
                    [
                        "/opt/homebrew/bin/Rscript",
                        "--vanilla",
                        str(ROOT / "reproduce/source/independent-export.R"),
                        str(tmp),
                    ],
                    check=True,
                    capture_output=True,
                )
                for entry in inner.infolist():
                    name = Path(entry.filename).name
                    table = {
                        "MS1_HPR.csv": "ms1",
                        "ASSIGNMENTS.csv": "identifications",
                        "QUANTITIES_raw.RData": "raw_quantities",
                        "QUANTITIES_filtered.RData": "filtered_vectors",
                    }.get(name)
                    if name.endswith("finalKinetics.csv"):
                        table = "final_vectors"
                    if table is None:
                        continue
                    locator = (
                        f"aSPIRE_outputs_zip.zip!{member.filename}!{entry.filename}"
                    )
                    handle = (
                        gzip.open(  # noqa: SIM115 - closed by the common with-handle block below
                            tmp / name.replace(".RData", ".csv.gz"), "rt", newline=""
                        )
                        if name.endswith(".RData")
                        else io.TextIOWrapper(inner.open(entry))
                    )
                    with handle:
                        for number, raw in enumerate(csv.DictReader(handle), 2):
                            row = take(table, raw, locator, number)
                            assert row["condition_id"] == condition_id
                            if table == "ms1":
                                assert row["peptide_sequence"] == raw["PeptideSequence"]
                                assert (
                                    row["modified_sequence"]
                                    == raw["PeptideModifiedSequence"]
                                )
                                assert row["ms1_area"] == value(raw["TotalAreaMs1"])
                                assert row["ms1_background"] == value(
                                    raw["TotalBackgroundMs1"]
                                )
                            elif table == "identifications":
                                assert row["q_value"] == value(raw["qValue"])
                                assert row["posterior_error_probability"] == value(
                                    raw["postErrProb"]
                                )
                            elif table == "raw_quantities":
                                assert (
                                    raw["substrateSeq"]
                                    == constructs[
                                        contract["construct_files"][condition_name]
                                    ]
                                )
                                run = runs[raw["source"]]
                                assert row["run_id"] == run["run_id"]
                                assert row["quantity_time_hours"] == value(
                                    raw["digestTime"]
                                )
                                assert row["metadata_time_hours"] == run["time_hours"]
                                assert row["control_role"] == run["control_role"]
                                is_conflict = (
                                    value(raw["digestTime"]) != run["time_hours"]
                                )
                                assert row["time_status"] == (
                                    "source_metadata_conflict"
                                    if is_conflict
                                    else "source_agreement"
                                )
                                conflicts[condition_id] += int(is_conflict)
                                for dst, src in {
                                    "intensity": "intensity",
                                    "fraction_background": "frac_background",
                                    "noise": "noise",
                                    "mean_technical": "mean_techRep",
                                    "mean_biological": "mean_bioRep",
                                }.items():
                                    assert row[dst] == value(raw[src])
                            elif table == "filtered_vectors":
                                assert row["times_raw"] == raw["times_pasted"]
                                assert row["intensities_raw"] == raw["int_pasted"]
                                assert len(raw["times_pasted"].split(";")) == len(
                                    raw["int_pasted"].split(";")
                                )
                            else:
                                assert (
                                    raw["substrateSeq"]
                                    == constructs[
                                        contract["construct_files"][condition_name]
                                    ]
                                )
                                if raw["biological_replicate"] == "NA":
                                    assert row["row_status"] == "annotation_only"
                                    assert (
                                        raw["digestTimes"] == raw["intensities"] == "NA"
                                    )
                                    annotation_rows += 1
                                    continue
                                times, values = (
                                    raw["digestTimes"].split(";"),
                                    raw["intensities"].split(";"),
                                )
                                assert len(times) == len(values)
                                for slot, (t, v) in enumerate(zip(times, values)):
                                    scalar = take("final_values", raw)
                                    assert (
                                        scalar["vector_id"] == row["source_record_id"]
                                    )
                                    assert scalar["slot_0based"] == slot
                                    assert scalar["time_hours"] == float(t)
                                    assert scalar["value_raw"] == v and scalar[
                                        "intensity"
                                    ] == value(v)
                                    if value(v) is None:
                                        assert (
                                            scalar["value_status"]
                                            == "postprocessing_missing_unresolved_contributors"
                                        )
                                        missing_slots += 1
                                    elif float(t) == 0 and float(v) == 0:
                                        assert (
                                            scalar["value_status"]
                                            == "postprocessing_forced_zero"
                                        )
                                        zero_slots += 1
            print("Independently replayed", condition_name, flush=True)
    # Relations with source-scope assertions are inspected separately from raw row replay.
    for _ in range(len(contract["construct_files"])):
        row = take("contexts")
        expected = (
            "unresolved_source_time_conflict"
            if conflicts[row["condition_id"]]
            else "pending_registered_recipe_and_independent_replay"
        )
        assert row["training_status"] == expected
    n_conflicts = 0
    for row in streams["conflicts"]:
        digests["conflicts"].update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
        counts["conflicts"] += 1
        if row["conflict_kind"] == "quantity_time_vs_original_metadata":
            assert float(row["claim_a"]) != float(row["claim_b"])
            n_conflicts += 1
    assert n_conflicts == sum(conflicts.values()) == 2278
    for table, stream in streams.items():
        assert next(stream, None) is None, table
        reference = queries["digestion_" + table]
        assert counts[table] == reference["n_rows"], table
        assert digests[table].hexdigest() == reference["canonical_sha256"], table
    result = {
        "passed": True,
        "rows": dict(counts),
        "n_package_files": len(manifest["files"]),
        "n_time_conflicts": n_conflicts,
        "annotation_rows": annotation_rows,
        "forced_zero_slots": zero_slots,
        "missing_slots": missing_slots,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "Original serialized/CSV/workbook to typed records and numeric state; quantitative preprocessing arithmetic remains a separate admission gate",
    }
    (ROOT / "results/independent-source-replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
