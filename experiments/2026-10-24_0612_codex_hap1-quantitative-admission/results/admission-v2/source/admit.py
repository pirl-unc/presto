"""Build the shared Hitlist package, freezing each actual attempt before execution."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import hitlist.hap1_quantitative as source

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=Path)
parser.add_argument("--attempt", required=True)
args = parser.parse_args()
if not args.attempt.replace("-", "").isalnum():
    raise ValueError("Use a literal alphanumeric attempt label")
experiment = args.experiment.resolve()
config = json.loads((experiment / "reproduce/launch.json").read_text())
record_dir = experiment / "results" / ("admission-" + args.attempt)
record_dir.mkdir(exist_ok=False)
snapshot = record_dir / "source"
snapshot.mkdir()
hitlist_root = Path(source.__file__).resolve().parents[1]
paths = [
    "hitlist/hap1_quantitative.py",
    "hitlist/evidence_catalog.py",
    "hitlist/evidence_bindings.py",
    "hitlist/evidence_package.py",
    "hitlist/evidence_store.py",
    "hitlist/version.py",
    "hitlist/__init__.py",
    "hitlist/data/evidence_revisions/hap1_assets.json",
    "hitlist/data/evidence_revisions/hap1_contract.json",
]
hashes = {}
for relative in paths:
    original = hitlist_root / relative
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, target)
    hashes[relative] = hashlib.sha256(original.read_bytes()).hexdigest()
shutil.copy2(__file__, snapshot / "admit.py")
git = lambda *argv: subprocess.check_output(["git", *argv], cwd=hitlist_root, text=True)
(record_dir / "hitlist.patch").write_text(git("diff", "--binary"))
destination = Path(config["source_root"]) / ("canonical-package-" + args.attempt)
receipt = {
    "started_at": datetime.now(UTC).isoformat(),
    "command": sys.argv,
    "hitlist_base_commit": git("rev-parse", "HEAD").strip(),
    "hitlist_dirty_state": git("status", "--short"),
    "source_hashes": hashes,
    "environment": {
        key: os.environ.get(key)
        for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    },
    "python": sys.executable,
    "source_root": config["source_root"],
    "destination": str(destination),
    "contract": "shared source admission, no predictive fitting",
}
(record_dir / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
started = time.monotonic()
try:
    manifest_path = source.write_hap1_quantitative_package(config["source_root"], destination)
    receipt["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    shutil.copy2(manifest_path, record_dir / "manifest.json")
    receipt["status"] = "source_package_built_and_replayed"
except Exception as error:
    receipt["status"] = "failed"
    receipt["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    (record_dir / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({key: receipt[key] for key in ("status", "elapsed_seconds", "peak_rss_gib")}),
        flush=True,
    )
