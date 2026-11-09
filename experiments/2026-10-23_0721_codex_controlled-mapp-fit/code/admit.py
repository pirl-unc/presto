"""Materialize the frozen shared contrast; preserve the original run export."""

import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.perf_counter()
    for name in ("source", "run_recipe"):
        path = (
            Path(CFG["source_package" if name == "source" else name]) / "manifest.json"
        )
        assert sha(path) == CFG[name + "_manifest_sha256"]
    stage = json.loads((ROOT / "reproduce/admission-launch.json").read_text())
    for name, digest in stage["hitlist_source_hashes"].items():
        assert sha(Path(CFG["hitlist_root"]) / name) == digest
    store = EvidenceStore.open(CFG["source_package"])
    # First verify that the updated compiler leaves the prior run recipe unchanged.
    original = Path(CFG["run_recipe"])
    base_recipe = json.loads((original / "manifest.json").read_text())["recipe"]
    replay = Path(CFG["contrast_output"]).parent / "base-replay"
    store.materialize(base_recipe, replay)
    unchanged = []
    for path in original.iterdir():
        if path.is_file():
            assert sha(path) == sha(replay / path.name), path.name
            unchanged.append(path.name)
    print("Base recipe byte-identical", len(unchanged), flush=True)
    recipe = {
        "schema_version": 1,
        "name": CFG["shared_contrast_recipe"],
        "registered_recipe": CFG["shared_contrast_recipe"],
    }
    store.materialize(recipe, CFG["contrast_output"])
    verify_recipe_package(CFG["contrast_output"], source_package=CFG["source_package"])
    for name in ("manifest.json", "summary.json", "recipe.json", "examples.json"):
        shutil.copyfile(
            Path(CFG["contrast_output"]) / name, ROOT / "results" / ("contrast-" + name)
        )
    result = {
        "passed": True,
        "unchanged_base_recipe_files": unchanged,
        "source_manifest_sha256": CFG["source_manifest_sha256"],
        "contrast_manifest_sha256": sha(Path(CFG["contrast_output"]) / "manifest.json"),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (ROOT / "results/admission.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
