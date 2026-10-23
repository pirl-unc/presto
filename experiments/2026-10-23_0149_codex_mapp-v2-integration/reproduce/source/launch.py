"""Frozen source/API materialization; no biological fitting or new splits."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pyarrow
from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package
from hitlist.external_successor import write_external_successor

FAMILY = Path(__file__).resolve().parents[1]
CONFIG = json.loads((FAMILY / "reproduce/launch.json").read_text())
OUTPUT = Path(CONFIG["output_package"])
RESULTS = FAMILY / "results"


def save(name, value):
    (RESULTS / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    receipt = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "environment": {k: os.environ.get(k) for k in CONFIG["environment"]},
        "stages": {},
        "passed": False,
    }
    save("execution.json", receipt)
    try:
        start = time.perf_counter()
        write_external_successor(
            CONFIG["reference_package"],
            OUTPUT,
            source_roots={"pxd009092": Path(CONFIG["source_assets"])},
        )
        receipt["stages"]["build_seconds"] = time.perf_counter() - start
        save("execution.json", receipt)
        print("Source successor built", receipt["stages"]["build_seconds"], flush=True)
        start = time.perf_counter()
        store = EvidenceStore.open(OUTPUT)
        catalog = store.catalog()
        save("catalog.json", catalog)
        audit = store.audit()
        save("audit.json", audit)
        assert all(row["status"] == "matched" for row in audit["baselines"])
        receipt["stages"]["open_catalog_audit_seconds"] = time.perf_counter() - start
        print("Canonical store, catalog and baseline audit pass", flush=True)
        query = {}
        start = time.perf_counter()
        for name, (item, _, _) in sorted(store.bindings.items()):
            if item.source_population != "PXD009092":
                continue
            hasher, n_rows = hashlib.sha256(), 0
            fields = [f"native.{field}" for field in store.schemas[name].names]
            for batch in store.iter_query(name, select=fields, limit=None):
                for row in batch.table.to_pylist():
                    row = {k.removeprefix("native."): v for k, v in row.items()}
                    hasher.update(
                        (
                            json.dumps(
                                row,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode()
                    )
                    n_rows += 1
            query[name] = {
                "artifact": item.materialized_artifact,
                "n_rows": n_rows,
                "canonical_rows_sha256": hasher.hexdigest(),
                "fields": fields,
            }
        save("query-all-mapp.json", query)
        receipt["stages"]["query_all_mapp_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        recipes = {}
        for index, recipe_id in enumerate(CONFIG["old_recipes"]):
            recipe = {
                "schema_version": 1,
                "name": recipe_id,
                "registered_recipe": recipe_id,
            }
            recipe_path = FAMILY / "reproduce" / (recipe_id + ".json")
            recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
            destination = Path(CONFIG["recipe_outputs"]) / recipe_id
            if index == 2:
                command = [
                    sys.executable,
                    "-m",
                    "hitlist.cli",
                    "evidence",
                    "materialize",
                    str(OUTPUT),
                    str(recipe_path),
                    str(destination),
                ]
                with (FAMILY / "launch_logs/cli-materialize.log").open("w") as log:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
            else:
                store.materialize(recipe, destination)
            result = verify_recipe_package(destination, source_package=OUTPUT)
            recipes[recipe_id] = {
                "manifest_sha256": hashlib.sha256(
                    (destination / "manifest.json").read_bytes()
                ).hexdigest(),
                "coverage": result["coverage"],
                "path": str(destination),
                "interface": "CLI" if index == 2 else "API",
            }
            print("Verified recipe", recipe_id, flush=True)
        save("recipes.json", recipes)
        receipt["stages"]["recipe_seconds"] = time.perf_counter() - start
        command = [
            sys.executable,
            "-m",
            "hitlist.cli",
            "evidence",
            "verify-recipe",
            str(destination),
            "--source-package",
            str(OUTPUT),
        ]
        with (FAMILY / "launch_logs/cli-verify-recipe.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        receipt["passed"] = True
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save("execution.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
