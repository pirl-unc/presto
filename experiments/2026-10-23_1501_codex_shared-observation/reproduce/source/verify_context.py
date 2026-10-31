"""Independent native-table replay of canonical acquisition joins and label conservation."""

import gzip
import hashlib
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def main():
    tick = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    package = Path(cfg["source_package"])
    recipe = Path(cfg["recipe_directory"])
    manifest = json.loads((package / "manifest.json").read_text())
    assert (
        hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
        == cfg["source_package_manifest_sha256"]
    )
    for table in ("runs", "raw_quantities"):
        path = f"external/soh2024_digestion/typed/{table}.parquet"
        assert (
            hashlib.sha256((package / path).read_bytes()).hexdigest()
            == manifest["files"][path]["sha256"]
        )
    runs = {
        r["run_id"]: r
        for r in pq.read_table(
            package / "external/soh2024_digestion/typed/runs.parquet"
        ).to_pylist()
    }
    source = [
        r
        for r in json.loads((recipe / "examples.json").read_text())
        if r["split"] in cfg["allowed_splits"]
    ]
    curves = {r["curve_id"]: r for r in source}
    conditions = sorted({r["condition_id"] for r in source})
    raw = pq.read_table(
        package / "external/soh2024_digestion/typed/raw_quantities.parquet",
        filters=[("condition_id", "in", conditions)],
        columns=[
            "source_record_id",
            "condition_id",
            "peptide_sequence",
            "biological_replicate_raw",
            "quantity_time_hours",
            "metadata_time_hours",
            "time_status",
            "run_id",
            "intensity",
            "assignments_raw",
        ],
    ).to_pylist()
    query = pq.read_table(
        FAMILY / "results/raw-quantity-acquisition-query.parquet"
    ).to_pylist()
    query_by_id = {r["source_record_id"]: r for r in query}
    assert len(query_by_id) == len(query) == len(raw)
    mapping = {
        "source_record_id": "source_record_id",
        "condition_id": "condition_id",
        "peptide_sequence": "peptide_sequence",
        "digestion_biological_replicate": "biological_replicate_raw",
        "digestion_quantity_time": "quantity_time_hours",
        "digestion_metadata_time": "metadata_time_hours",
        "digestion_time_status": "time_status",
        "digestion_run_id": "run_id",
        "quantitative_value": "intensity",
        "native.assignments_raw": "assignments_raw",
    }
    candidates = defaultdict(list)
    for row in raw:
        got = query_by_id[row["source_record_id"]]
        assert all(got[k] == row[v] for k, v in mapping.items())
        assert got["acquisition.instrument"] == runs[row["run_id"]]["instrument_raw"]
        assert (
            got["acquisition.source_locator"] == runs[row["run_id"]]["source_locator"]
        )
        candidates[
            row["condition_id"],
            row["peptide_sequence"].replace("I", "L"),
            row["biological_replicate_raw"],
            row["quantity_time_hours"],
        ].append(row)
    checked = 0
    with gzip.open(FAMILY / "results/observation-context.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            curve = curves[row["curve_id"]]
            assert (
                row["time"] in curve["common_positive_times"]
                and row["split"] == curve["split"]
            )
            target = next(
                o for o in curve["observations"] if o["time_hours"] == row["time"]
            )
            assert (
                target["target_value"] == row["source_target"]
                and target["source_slots"] == row["source_slots"]
            )
            source_rows = candidates[
                curve["condition_id"],
                curve["peptide_IL_sequence"],
                curve["biological_replicate_raw"],
                row["time"],
            ]
            assert (
                sorted(r["source_record_id"] for r in source_rows)
                == row["candidate_raw_quantity_ids"]
            )
            ids = sorted({r["run_id"] for r in source_rows})
            assert ids == row["candidate_run_ids"]
            instruments = sorted({runs[r]["instrument_raw"] for r in ids})
            assert (
                instruments == row["candidate_instruments"]
                and len(instruments) == 1
                and row["instrument"] == instruments[0]
            )
            checked += 1
    assert checked == sum(len(r["common_positive_times"]) for r in source) == 105452
    result = dict(
        n_native_raw_rows_verified=len(raw),
        n_source_runs=len(runs),
        n_observations_verified=checked,
        all_original_targets_slots_and_membership_exact=True,
        all_candidate_instrument_joins_exact=True,
        scope="Independent native source acquisition/quantity joins and original observation identity/value conservation. Candidate MS1 forms were inventoried via the verified canonical query but their aggregate membership is not independently reconstructed here.",
        source_package_verified_by_canonical_open=True,
        seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/context-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
