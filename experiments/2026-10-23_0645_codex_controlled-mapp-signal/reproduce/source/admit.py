"""Canonical controlled-panel source generation and recipe; no outcome diagnostics."""

import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package
from hitlist.external_successor import write_external_successor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())


def save(name, value):
    with (ROOT / "results" / name).open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    started = time.perf_counter()
    receipt = {
        "pid": os.getpid(),
        "passed": False,
        "python": sys.version,
        "platform": platform.platform(),
        "stages": {},
    }
    save("admission-execution.json", receipt)
    try:
        for key in ("parent", "old"):
            value = hashlib.sha256(
                (Path(CONFIG[key + "_package"]) / "manifest.json").read_bytes()
            ).hexdigest()
            assert value == CONFIG[key + "_manifest_sha256"]
        for name, expected in CONFIG["source_hashes"].items():
            assert (
                hashlib.sha256(
                    (Path(CONFIG["hitlist_root"]) / name).read_bytes()
                ).hexdigest()
                == expected
            ), name
        print("Starting pinned source generation", flush=True)
        stage = time.perf_counter()
        write_external_successor(
            CONFIG["parent_package"],
            CONFIG["output_package"],
            source_roots=CONFIG["source_roots"],
        )
        receipt["stages"]["source_seconds"] = time.perf_counter() - stage
        save("admission-execution.json", receipt)
        print("Source generation verified", receipt["stages"], flush=True)
        stage = time.perf_counter()
        store = EvidenceStore.open(CONFIG["output_package"])
        save("catalog.json", store.catalog())
        audit = store.audit()
        save("audit.json", audit)
        assert all(item["status"] == "matched" for item in audit["baselines"])
        receipt["stages"]["open_catalog_audit_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        queries = {}
        for name, (item, _, _) in sorted(store.bindings.items()):
            if item.required_appended_source != "wolflevy2018_controlled":
                continue
            fields = ["native." + field for field in store.schemas[name].names]
            digest, n_rows = hashlib.sha256(), 0
            for batch in store.iter_query(name, select=fields, limit=None):
                for row in batch.table.to_pylist():
                    row = {
                        key.removeprefix("native."): value for key, value in row.items()
                    }
                    digest.update(
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
            queries[name] = {
                "artifact": item.materialized_artifact,
                "n_rows": n_rows,
                "canonical_rows_sha256": digest.hexdigest(),
                "fields": fields,
            }
        save("controlled-queries.json", queries)
        receipt["stages"]["query_seconds"] = time.perf_counter() - stage
        print("Controlled canonical queries streamed", len(queries), flush=True)
        stage = time.perf_counter()
        recipe = {
            "schema_version": 1,
            "name": CONFIG["recipe_id"],
            "registered_recipe": CONFIG["recipe_id"],
        }
        (ROOT / "reproduce/recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        store.materialize(recipe, CONFIG["recipe_output"])
        manifest = verify_recipe_package(
            CONFIG["recipe_output"], source_package=CONFIG["output_package"]
        )
        save("recipe-verification.json", manifest)
        receipt["stages"]["recipe_seconds"] = time.perf_counter() - stage
        for name in ("manifest.json", "summary.json"):
            shutil.copyfile(
                Path(CONFIG["recipe_output"]) / name,
                ROOT / "results" / ("recipe-" + name),
            )
        shutil.copyfile(
            Path(CONFIG["output_package"]) / "manifest.json",
            ROOT / "results/source-manifest.json",
        )
        receipt["passed"] = True
        print(
            "Shared recipe complete; no held-out outcome statistics computed",
            flush=True,
        )
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save("admission-execution.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
