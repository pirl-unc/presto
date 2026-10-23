"""Admit the reviewed source and prove the declared additive metadata migration."""

import copy
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def main():
    start = time.monotonic()
    config = json.loads((FAMILY / "reproduce/reviewed-compatibility.json").read_text())
    package = Path(config["package"])
    assert (
        hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
        == config["package_manifest_sha256"]
    )
    assert (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == config["launcher_sha256"]
    )
    sys.path.insert(0, config["hitlist_root"])
    from hitlist.evidence_store import EvidenceStore

    out, old = Path(config["output"]), Path(config["original_output"])
    assert not out.exists()
    store = EvidenceStore.open(package)
    store.materialize(config["recipe"], out)
    original_config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    # Independently compare every old column in the source relations, including
    # fields not consumed by the recipe. Reviewed raw quantities add units only.
    source_counts = {}
    for name in (
        "constructs",
        "contexts",
        "raw_quantities",
        "filtered_vectors",
        "final_vectors",
    ):
        a = pq.read_table(
            Path(original_config["package"])
            / f"external/soh2024_digestion/typed/{name}.parquet"
        )
        b = pq.read_table(
            package / f"external/soh2024_digestion/typed/{name}.parquet",
            columns=a.column_names,
        )
        assert a.equals(b), name
        source_counts[name] = len(a)
    for name in ("components", "summary"):
        assert (old / f"{name}.json").read_bytes() == (
            out / f"{name}.json"
        ).read_bytes()
    changed_states = 0
    for name in ("examples", "dispositions"):
        prior = json.loads((old / f"{name}.json").read_text())
        actual = json.loads((out / f"{name}.json").read_text())
        expected = copy.deepcopy(prior)
        for row in expected:
            if name == "examples":
                row["target_unit"] = "source_MS1_peak_area_arbitrary_units"
                row["target_quantity_scale"] = (
                    "aSPIRE_background_adjusted_biological_mean"
                )
                groups = [
                    (o["time_hours"], o["source_slots"]) for o in row["observations"]
                ]
            else:
                groups = [
                    (o["time_hours"], o["contributors"]) for o in row["source_slots"]
                ]
            for t, slots in groups:
                for slot in slots:
                    v = slot["value"]
                    state = (
                        "postprocessing_missing_unresolved_contributors"
                        if v is None
                        else (
                            "postprocessing_forced_zero"
                            if t == 0 and v == 0
                            else (
                                "source_positive"
                                if v > 0
                                else "source_zero"
                                if v == 0
                                else "source_negative"
                            )
                        )
                    )
                    slot["value_status"] = state
                    changed_states += 1
        assert actual == expected, name
        del prior, actual, expected
    report = dict(
        passed=True,
        reviewed_commit=config["hitlist_commit"],
        source_rows_unchanged=source_counts,
        all_targets_groups_exclusions_unchanged=True,
        explicit_value_state_annotations=changed_states,
        recipe_directory=str(out),
        recipe_manifest_sha256=hashlib.sha256(
            (out / "manifest.json").read_bytes()
        ).hexdigest(),
        wall_seconds=time.monotonic() - start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/reviewed-compatibility.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    for name in ("manifest", "recipe"):
        (FAMILY / f"results/reviewed-{name}.json").write_bytes(
            (out / f"{name}.json").read_bytes()
        )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
