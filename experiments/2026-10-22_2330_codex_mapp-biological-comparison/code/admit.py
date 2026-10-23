"""Materialize the shared registered recipe without consumer-side source curation."""

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hitlist.evidence_recipes import RECIPE_ID, write_training_recipe

FAMILY = Path(__file__).resolve().parents[1]
PRESTO = FAMILY.parents[1]
HITLIST = Path("/private/tmp/hitlist-mapp-recipe")
SOURCE = Path("/private/tmp/hitlist-v2-mapp-reference-second")
OUTPUT = PRESTO / "data/availability_recipes/2026-10-22-mapp-captured-composition-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    tick = time.monotonic()
    assert not (FAMILY / "results/admission.json").exists()
    assert (
        sha(SOURCE / "manifest.json")
        == "55f7a3d50df5d162e55bf1d66fbcf413a35fc927fd08c266f90d981068a053c2"
    )
    receipt = {
        "argv": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "environment_overrides": {"PYTHONPATH": os.environ["PYTHONPATH"]},
        "source_package": str(SOURCE),
        "source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "output": str(OUTPUT),
        "recipe_id": RECIPE_ID,
    }
    for repo, directory in [("presto", PRESTO), ("hitlist", HITLIST)]:
        receipt[repo + "_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True
        ).strip()
        receipt[repo + "_dirty"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=directory, text=True
        )
    dump(FAMILY / "reproduce/admission.json", receipt)
    for name in (
        "evidence_recipes.py",
        "external_evidence.py",
        "mapp_evidence.py",
        "sequence_reference_evidence.py",
        "version.py",
    ):
        shutil.copy2(HITLIST / "hitlist" / name, FAMILY / "reproduce/source" / name)
    shutil.copy2(
        HITLIST / "tests/test_evidence_recipes.py",
        FAMILY / "reproduce/source/test_evidence_recipes.py",
    )
    shutil.copy2(__file__, FAMILY / "reproduce/source/admit.py")
    write_training_recipe(SOURCE, OUTPUT, recipe_id=RECIPE_ID)
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert summary["n_structurally_supported_peptides"] == 2272
    gate = all(
        summary["n_parents_by_split"].get(split, 0) >= 10
        for split in ("train", "validation", "test")
    )
    report = {
        "seconds": time.monotonic() - tick,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "recipe_manifest_sha256": sha(OUTPUT / "manifest.json"),
        "output": str(OUTPUT),
        "minimum_split_support_gate": gate,
        "summary": summary,
    }
    dump(FAMILY / "results/admission.json", report)
    for name in ("manifest.json", "recipe.json", "summary.json"):
        shutil.copy2(OUTPUT / name, FAMILY / "results" / ("recipe_" + name))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
