"""Independent conservation audit: original bytes and Parquet, no Hitlist imports."""

import hashlib
import json
import resource
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())


def digest(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return {"sha256": hasher.hexdigest(), "size_bytes": path.stat().st_size}


def main():
    start = time.perf_counter()
    assert json.loads((ROOT / "results/execution.json").read_text())["passed"]
    parent, legacy, output = (
        Path(CONFIG[key + "_package"]) for key in ("parent", "legacy", "output")
    )
    pm, lm, current = (
        json.loads((p / "manifest.json").read_text()) for p in (parent, legacy, output)
    )
    assert (
        digest(parent / "manifest.json")["sha256"] == CONFIG["parent_manifest_sha256"]
    )
    assert (
        digest(legacy / "manifest.json")["sha256"] == CONFIG["legacy_manifest_sha256"]
    )
    assert (current["schema_version"], current["catalog_schema_version"]) == (9, 12)
    assert current["parent_kind"] == "adjudication"
    assert current["parent_manifest_path"] == "manifests/adjudication.json"
    assert current["parent_manifest_sha256"] == CONFIG["parent_manifest_sha256"]
    inventory = {
        str(p.relative_to(output)): digest(p)
        for p in output.rglob("*")
        if p.is_file() and p != output / "manifest.json"
    }
    assert inventory == current["files"]
    assert (output / "manifests/adjudication.json").read_bytes() == (
        parent / "manifest.json"
    ).read_bytes()
    for name, expected in pm["files"].items():
        assert inventory[name] == digest(parent / name) == expected, name
        assert (parent / name).stat().st_ino != (output / name).stat().st_ino, name
    assert (parent / "manifest.json").stat().st_ino != (
        output / "manifests/adjudication.json"
    ).stat().st_ino
    source_counts = {}
    assert set(current["appended_sources"]) == set(CONFIG["source_roots"])
    for source_id in CONFIG["source_roots"]:
        prefix = "external/" + source_id + "/"
        expected = {
            name: value
            for name, value in lm["files"].items()
            if name.startswith(prefix)
        }
        observed = {
            name: value for name, value in inventory.items() if name.startswith(prefix)
        }
        assert observed == expected
        for name, value in expected.items():
            assert digest(legacy / name) == value, name
        assert (
            current["appended_sources"][source_id] == lm["appended_sources"][source_id]
        )
        source_counts[source_id] = len(expected)
    reviews = pq.read_table(output / "tcell/adjudication/reviews.parquet").to_pylist()
    assert (
        reviews
        == pq.read_table(parent / "tcell/adjudication/reviews.parquet").to_pylist()
    )
    assert len(reviews) == len({row["source_assay_id"] for row in reviews}) == 9
    queries = json.loads((ROOT / "results/queries.json").read_text())
    n_query_rows = 0
    for name, record in queries.items():
        parquet = pq.ParquetFile(output / record["artifact"])
        assert record["fields"] == [
            "native." + field for field in parquet.schema_arrow.names
        ]
        hasher, n_rows = hashlib.sha256(), 0
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                hasher.update(
                    (
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":"), allow_nan=False
                        )
                        + "\n"
                    ).encode()
                )
                n_rows += 1
        assert n_rows == record["n_rows"] == parquet.metadata.num_rows
        assert hasher.hexdigest() == record["canonical_rows_sha256"], name
        n_query_rows += n_rows
    catalog = json.loads((ROOT / "results/catalog.json").read_text())
    assert (
        catalog["source_registry"]["tcell_multimer_review_2026_10_23"][
            "n_assay_reviews"
        ]
        == 9
    )
    assert {"pxd009092", "pxd004886"} <= catalog["source_registry"].keys()
    present = [row for row in catalog["evidence"] if row["status"] == "materialized"]
    assert len(present) == 85
    assert len(queries) == 32
    assert all(
        item["status"] == "materialized" for item in catalog["registered_recipes"]
    )
    recipe_comparisons = {}
    for name, old_path in CONFIG["old_recipes"].items():
        before, after = Path(old_path), Path(CONFIG["recipe_outputs"]) / name
        assert (
            digest(before / "manifest.json")["sha256"]
            == CONFIG["old_recipe_manifests"][name]
        )
        a, b = (
            json.loads((path / "manifest.json").read_text()) for path in (before, after)
        )
        assert (
            b["source_package_manifest_sha256"]
            == digest(output / "manifest.json")["sha256"]
        )
        assert b["source_package_schema_version"] == 9
        assert set(a["files"]) == set(b["files"])
        conserved = []
        for relative, expected in b["files"].items():
            assert digest(after / relative) == expected
            if relative == "source_manifest.json":
                continue
            assert (before / relative).read_bytes() == (
                after / relative
            ).read_bytes(), (name, relative)
            conserved.append(relative)
        assert a["coverage"] == b["coverage"]
        recipe_comparisons[name] = {
            "unchanged_content_files": conserved,
            "coverage": b["coverage"],
        }
    result = {
        "passed": True,
        "n_files": len(inventory),
        "n_predecessor_files": len(pm["files"]),
        "n_source_files": source_counts,
        "n_reviews": len(reviews),
        "n_materialized_relations": len(present),
        "n_verified_query_relations": len(queries),
        "n_verified_query_rows": n_query_rows,
        "recipes": recipe_comparisons,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (ROOT / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
