"""Consume the canonical Hitlist recipe/query API without source reinterpretation."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import pyarrow.parquet as pq
from hitlist import EvidenceStore
from hitlist.evidence_recipe import verify_recipe_package

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    config = json.loads((FAMILY / "reproduce/launch.json").read_text())
    source = Path(config["source_package"])
    assert digest(source / "manifest.json") == config["source_manifest_sha256"]
    destination = ROOT / "data/apc_response_recipes" / FAMILY.name
    assert (destination / "response-recipe/manifest.json").exists()
    assert not (destination / "relations").exists()
    store = EvidenceStore.open(source)
    print("Canonical source package verified", flush=True)
    catalog = store.catalog()
    descriptions = {x["name"]: x for x in catalog["evidence"]}
    selected = [
        "source_dataset",
        *descriptions["external_quantity_measurement"]["native_fields"],
        *[
            "reference_context." + name
            for name in descriptions["external_reference_result_context"]["native_fields"]
        ],
    ]
    recipe = {
        "schema_version": 1,
        "name": "pxd057588-reported-responses-with-reference-context",
        "mapping_policy": "no_expansion",
        "sources": [
            {
                "evidence_type": "external_quantity_measurement",
                "row_id": "measurement_id",
                "select": selected,
                "joins": ["reference_context"],
                "where": {"representation": "reported_log2_fold_change"},
                "target": {
                    "field": "reported_value",
                    "mode": "numeric",
                    "unit_field": "measurement_unit",
                    "allowed_units": ["log2_ratio"],
                    "scale_field": "quantity_scale",
                    "allowed_scales": ["relative_to_contrast_controls"],
                    "status_field": "value_status",
                    "allowed_statuses": ["source_reported_verified_derivation"],
                    "unit_conversion": {
                        "output_unit": "log10_ratio",
                        "factor_by_unit": {"log2_ratio": math.log10(2)},
                    },
                },
                "required_fields": ["reference_context.component"],
                "context_precision": {
                    "field": "reference_context.material_link_precision",
                    "allowed": ["assay_channel_unresolved_specimen"],
                    "on_mismatch": "ambiguous",
                },
                "grouping": {
                    "fields": ["reference_context.component"],
                    "on_missing": "ambiguous",
                },
                "limit": None,
            }
        ],
    }
    assert recipe == json.loads((FAMILY / "reproduce/recipe.json").read_text())
    manifest = json.loads((destination / "response-recipe/manifest.json").read_text())
    verify_recipe_package(destination / "response-recipe", source_package=source)
    write(FAMILY / "results/recipe-manifest.json", manifest)
    print("Canonical response recipe materialized and verified", flush=True)
    names = [
        "external_reference_source_mapping",
        "external_reference_sequence",
        "external_reference_occurrence",
        "external_reference_parent",
        "external_reference_result_context",
        "external_assay_channel",
        "external_contrast",
        "external_derivation_input",
        "external_context_assertion",
        "external_source_asset",
    ]
    directory = destination / "relations"
    directory.mkdir()
    requests, outputs = {}, {}
    for name in names:
        request = {"select": descriptions[name]["native_fields"], "limit": None}
        requests[name] = request
        writer, count = None, 0
        path = directory / (name + ".parquet")
        try:
            for result in store.iter_query(name, **request):
                table = result.table
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
                count += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        assert count > 0
        outputs[name] = {"rows": count, "sha256": digest(path), "path": str(path)}
        print(json.dumps({"relation": name, "rows": count}), flush=True)
    write(FAMILY / "reproduce/query-requests.json", requests)
    write(FAMILY / "results/query-relations.json", outputs)
    result = {
        "status": "completed",
        "continuation_of": "export.py Path-return reporting failure; original recipe preserved",
        "source_manifest_sha256": config["source_manifest_sha256"],
        "recipe_manifest_sha256": digest(destination / "response-recipe/manifest.json"),
        "destination": str(destination),
        "hitlist_version": catalog["hitlist_version"],
        "source_package_hitlist_version": catalog["package_hitlist_version"],
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "held_out_metrics_evaluated": False,
    }
    write(FAMILY / "results/export.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
