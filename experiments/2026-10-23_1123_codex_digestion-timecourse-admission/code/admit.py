"""Execute the frozen shared recipe; no model or held-out response evaluation."""

import hashlib
import json
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

FAMILY = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.monotonic()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    root = Path(config["hitlist_root"])
    package = Path(config["package"])
    for name, expected in config["source_hashes"].items():
        assert digest(root / name) == expected, name
        assert digest(FAMILY / "reproduce/source" / name) == expected, name
    assert digest(package / "manifest.json") == config["package_manifest_sha256"]
    assert digest(__file__) == config["launcher_sha256"]
    sys.path.insert(0, str(root))
    from hitlist.evidence_store import EvidenceStore

    destination = Path(config["output_recipe"])
    assert not destination.exists(), "Refuse to overwrite immutable admission"
    print("Frozen source verified; opening canonical EvidenceStore", flush=True)
    store = EvidenceStore.open(package)
    print("Materializing registered time courses", flush=True)
    manifest_path = store.materialize(config["recipe"], destination)
    results = FAMILY / "results"
    for name in ("manifest", "recipe", "summary", "components"):
        shutil.copyfile(destination / f"{name}.json", results / f"recipe-{name}.json")
    receipt = {
        "recipe_directory": str(destination),
        "manifest_sha256": digest(manifest_path),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "python": sys.version,
        "platform": platform.platform(),
        "source_freeze_verified": True,
        "predictive_evaluation": False,
        "summary": json.loads((destination / "summary.json").read_text()),
    }
    (results / "admission.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
