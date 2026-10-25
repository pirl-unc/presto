"""Materialize the frozen native recipe through the public store interface."""

import importlib.metadata
import json
from pathlib import Path
import platform

from hitlist.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
store = EvidenceStore.open(CFG["source_package"])
manifest_path = store.materialize(CFG["recipe"], CFG["package"])
manifest = json.loads(Path(manifest_path).read_text())
(ROOT / "results/package-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
(ROOT / "results/support.json").write_text(
    json.dumps(manifest["coverage"], indent=2) + "\n"
)
(ROOT / "reproduce/environment.json").write_text(
    json.dumps(
        dict(
            python=platform.python_version(),
            platform=platform.platform(),
            hitlist_source_version=manifest["hitlist_version"],
            versions={
                p: importlib.metadata.version(p) for p in ("pyarrow", "numpy", "psutil")
            },
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(manifest["coverage"]), flush=True)
