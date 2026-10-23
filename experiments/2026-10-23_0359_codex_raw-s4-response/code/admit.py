"""Canonical source-bound response materialization before fitting."""

from pathlib import Path
import json, hashlib, time, resource, traceback, subprocess, sys
from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package

R = Path(__file__).resolve().parents[1]
C = json.loads((R / "reproduce/launch.json").read_text())
start = time.perf_counter()
receipt = {
    "passed": False,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
try:
    p = Path(C["source_package"])
    assert (
        hashlib.sha256((p / "manifest.json").read_bytes()).hexdigest()
        == C["source_manifest_sha256"]
    )
    store = EvidenceStore.open(p)
    recipe = {
        "schema_version": 1,
        "name": C["recipe_id"],
        "registered_recipe": C["recipe_id"],
    }
    (R / "reproduce/recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    store.materialize(recipe, C["recipe_output"])
    result = verify_recipe_package(C["recipe_output"], source_package=p)
    (R / "results/recipe-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (R / "launch_logs/cli-verify.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hitlist.cli",
                "evidence",
                "verify-recipe",
                C["recipe_output"],
                "--source-package",
                str(p),
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    receipt["passed"] = True
    receipt["summary"] = result["coverage"]
except Exception:
    receipt["error"] = traceback.format_exc()
    raise
finally:
    receipt["wall_seconds"] = time.perf_counter() - start
    receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (R / "results/admission-execution.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
