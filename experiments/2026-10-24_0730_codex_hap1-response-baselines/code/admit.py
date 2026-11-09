"""Consume the canonical Hitlist recipe, preserving the exact invocation/source."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment.resolve()
    receipt = exp / "results/admission"
    receipt.mkdir(parents=True, exist_ok=False)
    import hitlist
    from hitlist.evidence_store import EvidenceStore
    from hitlist.hap1_response_recipe import RECIPE_ID

    root = Path(hitlist.__file__).parent.parent
    for name in [
        "hitlist/hap1_response_recipe.py",
        "hitlist/evidence_recipe.py",
        "hitlist/distribution_recipe.py",
        "hitlist/version.py",
    ]:
        dest = receipt / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, dest)
    shutil.copyfile(__file__, receipt / "source/admit.py")
    metadata = {
        "argv": sys.argv,
        "python": sys.executable,
        "source": str(args.source),
        "output": str(args.output),
        "source_manifest_sha256": digest(args.source / "manifest.json"),
        "hitlist_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "hitlist_dirty": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "environment": {
            k: os.environ.get(k)
            for k in ["PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
    }
    (receipt / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.perf_counter()
    try:
        assert (
            metadata["source_manifest_sha256"]
            == "88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8"
        )
        store = EvidenceStore.open(args.source)
        print("Source package verified", flush=True)
        recipe = {
            "schema_version": 1,
            "name": "hap1-response-baseline-v1",
            "registered_recipe": RECIPE_ID,
        }
        store.materialize(recipe, args.output)
        manifest = json.loads((args.output / "manifest.json").read_text())
        result = {
            "status": "passed",
            "manifest_sha256": digest(args.output / "manifest.json"),
            "coverage": manifest["coverage"],
        }
        shutil.copyfile(args.output / "manifest.json", receipt / "recipe-manifest.json")
    except BaseException as error:
        result = {"status": "failed", "error": repr(error)}
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        result["peak_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
        (receipt / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    assert result["seconds"] <= 900 and result["peak_gib"] <= 4


if __name__ == "__main__":
    main()
