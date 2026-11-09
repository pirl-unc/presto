"""Build the combined generation and exercise the canonical public query surface."""

import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import pyarrow
from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package
from hitlist.external_successor import write_external_successor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())


def save(name, value):
    (ROOT / "results" / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    receipt = {
        "passed": False,
        "python": sys.version,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
        "environment": {name: os.environ.get(name) for name in CONFIG["environment"]},
        "stages": {},
    }
    try:
        for name in ("parent", "legacy"):
            assert (
                sha(Path(CONFIG[name + "_package"]) / "manifest.json")
                == CONFIG[name + "_manifest_sha256"]
            )
        for name, expected in CONFIG["source_hashes"].items():
            assert sha(Path(CONFIG["hitlist_root"]) / name) == expected, name
        stage = time.perf_counter()
        write_external_successor(
            CONFIG["parent_package"],
            CONFIG["output_package"],
            source_roots=CONFIG["source_roots"],
        )
        receipt["stages"]["build_seconds"] = time.perf_counter() - stage
        print("Combined package built", receipt["stages"], flush=True)
        stage = time.perf_counter()
        store = EvidenceStore.open(CONFIG["output_package"])
        save("catalog.json", store.catalog())
        audit = store.audit()
        save("audit.json", audit)
        assert all(row["status"] == "matched" for row in audit["baselines"])
        receipt["stages"]["open_catalog_audit_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        queries = {}
        for name, (item, _, _) in sorted(store.bindings.items()):
            if (
                item.source_population not in ("PXD009092", "PXD004886")
                and name != "assay_adjudication"
            ):
                continue
            hasher, n_rows = hashlib.sha256(), 0
            fields = ["native." + field for field in store.schemas[name].names]
            for batch in store.iter_query(name, select=fields, limit=None):
                for row in batch.table.to_pylist():
                    row = {
                        key.removeprefix("native."): value for key, value in row.items()
                    }
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
            queries[name] = {
                "artifact": item.materialized_artifact,
                "n_rows": n_rows,
                "canonical_rows_sha256": hasher.hexdigest(),
                "fields": fields,
            }
        save("queries.json", queries)
        receipt["stages"]["query_seconds"] = time.perf_counter() - stage
        print("All source/review queries streamed", len(queries), flush=True)
        stage = time.perf_counter()
        recipes = {}
        for recipe_id in CONFIG["old_recipes"]:
            request = {
                "schema_version": 1,
                "name": recipe_id,
                "registered_recipe": recipe_id,
            }
            (ROOT / "reproduce" / (recipe_id + ".json")).write_text(
                json.dumps(request, indent=2) + "\n"
            )
            destination = Path(CONFIG["recipe_outputs"]) / recipe_id
            store.materialize(request, destination)
            result = verify_recipe_package(
                destination, source_package=CONFIG["output_package"]
            )
            recipes[recipe_id] = {
                "path": str(destination),
                "coverage": result["coverage"],
                "manifest_sha256": sha(destination / "manifest.json"),
            }
            print("Recipe verified", recipe_id, flush=True)
        save("recipes.json", recipes)
        receipt["stages"]["recipe_seconds"] = time.perf_counter() - stage
        # Check real historical readers as well as the new family. No regeneration.
        stage = time.perf_counter()
        historical = {}
        for kind in ("parent", "legacy"):
            old = EvidenceStore.open(CONFIG[kind + "_package"])
            historical[kind] = {
                "schema_version": old.manifest["schema_version"],
                "n_materialized_relations": len(old.bindings),
                "has_reviews": "assay_adjudication" in old.bindings,
                "has_mapp": "mapp_peptide" in old.bindings,
            }
        assert (
            historical["parent"]["has_reviews"] and not historical["parent"]["has_mapp"]
        )
        assert (
            historical["legacy"]["has_mapp"] and not historical["legacy"]["has_reviews"]
        )
        save("historical-readers.json", historical)
        receipt["stages"]["historical_read_seconds"] = time.perf_counter() - stage
        save(
            "source-manifest.json",
            json.loads((Path(CONFIG["output_package"]) / "manifest.json").read_text()),
        )
        receipt["passed"] = True
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save("execution.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
