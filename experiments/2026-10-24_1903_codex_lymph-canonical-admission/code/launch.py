"""Materialize the frozen shared source; this is not biological model training."""

import hashlib
import importlib.metadata
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import hitlist
from hitlist.evidence_store import EvidenceStore
from hitlist.nanaware_lymph import write_nanaware_lymph_package

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RAW = REPO / "data/lymph_source_qualification/2026-10-24_1829_v1"
DEST = REPO / "data/lymph_source_qualification/2026-10-24_1903_canonical"

started = time.perf_counter()
assert shutil.disk_usage(REPO).free >= 2 * 1024**3
receipt = json.loads((ROOT / "reproduce/launch.json").read_text())
for filename, digest in receipt["hitlist_changed_sources"].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
assert not (ROOT / "results/admission.json").exists()
shutil.copy2(__file__, ROOT / "reproduce/source/launch.py")
manifest_path = write_nanaware_lymph_package(RAW, DEST)
store = EvidenceStore.open(DEST)
counts = {name: store.query(name, limit=None).table.num_rows for name in store.bindings}
manifest = json.loads(manifest_path.read_text())
assert counts == {
    "lymph_" + key.removeprefix("n_").removesuffix("_rows"): value
    for key, value in manifest["reconciliation"].items()
}
seconds = time.perf_counter() - started
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
result = {
    "status": "passed",
    "package": str(DEST),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "counts": counts,
    "seconds": seconds,
    "peak_rss_gib": rss,
    "python": sys.version,
    "hitlist_import": hitlist.__file__,
    "hitlist_import_version": hitlist.__version__,
    "installed_distribution_version": importlib.metadata.version("hitlist"),
    "model_training_or_predictions": False,
}
assert seconds <= 600 and rss <= 3
(ROOT / "results/admission.json").write_text(json.dumps(result, indent=2) + "\n")
shutil.copy2(manifest_path, ROOT / "results/source-manifest.json")
print(json.dumps(result, indent=2), flush=True)
