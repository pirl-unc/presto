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

from hitlist.evidence_recipes import RUN_RECIPE_ID as RECIPE_ID
from hitlist.evidence_recipes import write_training_recipe

FAMILY = Path(__file__).resolve().parents[1]
PRESTO = FAMILY.parents[1]
HITLIST = Path("/private/tmp/hitlist-mapp-recipe")
SOURCE = Path("/private/tmp/hitlist-v2-mapp-reference-second")
OUTPUT = (
    PRESTO / "data/availability_recipes/2026-10-23-mapp-complete-run-composition-v1"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    (FAMILY / "reproduce/source/admission").mkdir()
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
        shutil.copy2(
            HITLIST / "hitlist" / name,
            FAMILY / "reproduce/source/admission" / name,
        )
    shutil.copy2(
        HITLIST / "tests/test_evidence_recipes.py",
        FAMILY / "reproduce/source/test_evidence_recipes.py",
    )
    shutil.copy2(__file__, FAMILY / "reproduce/source/admit.py")
    replay_checks = {}
    old_root = PRESTO / "artifacts" / FAMILY.name / "old-recipe-replay"
    for old_id, old_name in [
        ("mapp_captured_composition_v1", "2026-10-22-mapp-captured-composition-v1"),
        (
            "mapp_captured_composition_min2_v1",
            "2026-10-22-mapp-captured-composition-min2-v1",
        ),
    ]:
        replay = old_root / old_id
        original = PRESTO / "data/availability_recipes" / old_name
        write_training_recipe(SOURCE, replay, recipe_id=old_id)
        assert {p.name for p in replay.iterdir()} == {
            p.name for p in original.iterdir()
        }
        for path in replay.iterdir():
            assert path.read_bytes() == (original / path.name).read_bytes(), path.name
        replay_checks[old_id] = {
            "byte_identical": True,
            "files": len(list(replay.iterdir())),
            "manifest_sha256": sha(replay / "manifest.json"),
        }
    dump(FAMILY / "results/old-recipe-conservation.json", replay_checks)
    write_training_recipe(SOURCE, OUTPUT, recipe_id=RECIPE_ID)
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert summary["n_structurally_supported_peptides"] == 2272
    report = {
        "seconds": time.monotonic() - tick,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "recipe_manifest_sha256": sha(OUTPUT / "manifest.json"),
        "output": str(OUTPUT),
        "summary": summary,
    }
    dump(FAMILY / "results/admission.json", report)
    for name in ("manifest.json", "recipe.json", "summary.json"):
        shutil.copy2(OUTPUT / name, FAMILY / "results" / ("recipe_" + name))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
