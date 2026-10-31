"""Build the shared source, without selecting a training cohort."""

import hashlib
import json
import platform
import resource
import time
from pathlib import Path

from hitlist.intact_degradomics import write_intact_degradomics_package

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
start = time.monotonic()
path = write_intact_degradomics_package(config["original_assets"], config["package"])
manifest = json.loads(path.read_text())
result = {"manifest": manifest, "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "wall_seconds": time.monotonic() - start,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "platform": platform.platform(), "training": None}
(ROOT / "results/source-build.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"counts": manifest["reconciliation"], "seconds": result["wall_seconds"]}), flush=True)
