"""Materialize original bytes through Hitlist's public source writer."""

import importlib.metadata
import json
from pathlib import Path
import platform

from hitlist.native_peptidomics import write_native_peptidomics_package

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
destination = Path(config["package"])
staging = destination.parent / (destination.name + "-original-links")
staging.mkdir(parents=True)
for filename, original in config["original_assets"].items():
    (staging / filename).symlink_to(original)
manifest_path = write_native_peptidomics_package(staging, destination)
manifest = json.loads(manifest_path.read_text())
(ROOT / "results/package-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
environment = dict(python=platform.python_version(), platform=platform.platform(),
                   versions={p: importlib.metadata.version(p) for p in ("pyarrow", "pandas", "openpyxl", "psutil")},
                   hitlist_code=config["hitlist_repo"], hitlist_source_version=manifest["hitlist_version"])
(ROOT / "reproduce/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print(json.dumps(manifest["reconciliation"]), flush=True)
