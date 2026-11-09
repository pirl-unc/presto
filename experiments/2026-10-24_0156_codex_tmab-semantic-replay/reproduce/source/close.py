"""Resolve the recorded whitespace comparator error without replacing evidence."""
import collections
import gzip
import hashlib
import json
import resource
import time
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    tick = time.perf_counter()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    replay = json.loads((FAMILY / "results/replay.json").read_text())
    failures = json.loads((FAMILY / "results/disagreements.json").read_text())
    receipt = json.loads((FAMILY / "reproduce/execution.json").read_text())
    assert receipt["source_sha256"] == digest(FAMILY / "code/launch.py")
    assert receipt["source_sha256"] == digest(FAMILY / "reproduce/source/launch.py")
    assert receipt["launch_sha256"] == digest(FAMILY / "reproduce/launch.json")
    old = Path(config["old_package"])
    new = Path(config["new_package"])
    assert digest(old / "manifest.json") == config["old_manifest_sha256"]
    assert digest(new / "manifest.json") == replay["new_manifest_sha256"]
    for root in (old, new):
        for name, metadata in json.loads((root / "manifest.json").read_text())["files"].items():
            assert digest(root / name) == metadata["sha256"], (root, name)
    original = {}
    asset_files = {"dasgupta2014_s001": "dasgupta_s1.xlsx", "gelman2013_s004": "gelman_s1.xlsx"}
    with gzip.open(Path(config["old_family"]) / "results/primary-row-audit.jsonl.gz", "rt") as stream:
        for line in stream:
            r = json.loads(line)
            original[(asset_files[r["asset"]], r["row"])] = r
    old_contexts = {r["context_id"]: r for r in pq.read_table(old / "tmab/protocol_context.parquet").to_pylist()}
    new_contexts = {r["context_id"]: r for r in pq.read_table(new / "tmab/protocol_context.parquet").to_pylist()}
    old_m = {r["measurement_id"]: r for r in pq.read_table(old / "tmab/measurements.parquet").to_pylist()}
    expected_failures = []
    n_trimmed_measurements = 0
    for r in pq.read_table(new / "tmab/measurements.parquet").to_pylist():
        raw = original[(r["source_file"], r["source_row_index"])]["experiment"]
        normalized = None if raw is None else str(raw).strip() or None
        c = new_contexts[r["context_id"]]
        prior = old_contexts[old_m[r["measurement_id"]]["context_id"]]
        assert c["treatment_label_raw"] == prior["treatment_label_raw"] == normalized
        if raw != normalized:
            n_trimmed_measurements += 1
            expected_failures.append(dict(check="context_literal_experiment", identity=r["context_id"],
                                          actual=normalized, expected=raw))
    canonical = lambda rows: collections.Counter(json.dumps(r, sort_keys=True) for r in rows)
    assert canonical(failures) == canonical(expected_failures)
    assert len(failures) == replay["n_disagreements"] == n_trimmed_measurements == 1242
    assert replay["seconds"] < config["seconds_cap"]
    assert replay["peak_rss_bytes"] < config["rss_bytes_cap"]
    result = dict(
        original_execution="failed; retained unchanged", corrected_comparison_pass=True,
        corrected_rule="projected labels trim surrounding whitespace; original source arrays remain literal",
        n_explained_comparator_disagreements=len(failures), n_unexplained_disagreements=0,
        n_source_rows=replay["n_source_rows"], n_measurement_slots=replay["n_measurement_slots"],
        n_original_assertions=sum(replay["checks"].values()),
        new_manifest_sha256=replay["new_manifest_sha256"],
        seconds=time.perf_counter()-tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        training_admission=False,
    )
    assert result["seconds"] < 30 and result["peak_rss_bytes"] < 1024**3
    assert not (FAMILY / "results/closure.json").exists()
    (FAMILY / "results/closure.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
