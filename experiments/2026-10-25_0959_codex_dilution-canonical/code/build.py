"""Materialize the pinned source using the public, isolated Hitlist checkout."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())
sys.path.insert(0, CONFIG["hitlist_checkout"])

import hitlist
import pyarrow.parquet as pq

assert Path(hitlist.__file__).resolve().parent == Path(CONFIG["hitlist_checkout"]) / "hitlist"
destination = Path(CONFIG["package"])
manifest_path = hitlist.write_dilution_observer_package(ROOT, destination, assets=CONFIG["assets"])
manifest = json.loads(manifest_path.read_text())
counts = {
    p.stem: pq.ParquetFile(p).metadata.num_rows
    for p in sorted((destination / "dilution_observer/typed").glob("*.parquet"))
}
assert counts == manifest["reconciliation"] and len(counts) == 15
result = {
    "passed": True,
    "package": str(destination),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "table_counts": counts,
    "public_writer_includes_full_original_source_replay": True,
    "independent_replay_passed": False,
    "training_performed": False,
}
(ROOT / "results/build.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
