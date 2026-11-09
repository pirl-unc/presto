"""Independent byte, grain and target replay without Hitlist reader/recipe imports."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import pyarrow.parquet as pq

FAMILY = Path(__file__).resolve().parents[1]
CONFIG = json.loads((FAMILY / "reproduce/launch.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"sha256": h.hexdigest(), "size_bytes": Path(path).stat().st_size}


def rows(path):
    with Path(path).open() as stream:
        yield from (json.loads(line) for line in stream)


def main():
    start = time.perf_counter()
    assert json.loads((FAMILY / "results/execution.json").read_text())["passed"]
    parent, source, old = (
        Path(CONFIG[key])
        for key in ["reference_package", "output_package", "historical_mapp_package"]
    )
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    assert (
        digest(parent / "manifest.json")["sha256"]
        == CONFIG["reference_manifest_sha256"]
    )
    assert (
        digest(old / "manifest.json")["sha256"]
        == CONFIG["historical_mapp_manifest_sha256"]
    )
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["schema_version"] == 8 and manifest["catalog_schema_version"] == 10
    assert set(manifest["appended_sources"]) == {"pxd009092"}
    assert manifest["parent_manifest_sha256"] == CONFIG["reference_manifest_sha256"]
    assert manifest["reference"] == parent_manifest["reference"]
    inventory = {
        str(p.relative_to(source)): digest(p)
        for p in source.rglob("*")
        if p.is_file() and p != source / "manifest.json"
    }
    assert inventory == manifest["files"]
    assert (source / "manifests/reference.json").read_bytes() == (
        parent / "manifest.json"
    ).read_bytes()
    for name, value in parent_manifest["files"].items():
        assert inventory[name] == value == digest(parent / name), name
        a, b = (parent / name).stat(), (source / name).stat()
        assert (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), name
    a, b = (
        (parent / "manifest.json").stat(),
        (source / "manifests/reference.json").stat(),
    )
    assert (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino)
    legacy = json.loads((old / "manifest.json").read_text())
    mapp = {
        k: v for k, v in legacy["files"].items() if k.startswith("external/pxd009092/")
    }
    actual_mapp = {
        k: v for k, v in inventory.items() if k.startswith("external/pxd009092/")
    }
    assert mapp == actual_mapp
    for name, value in mapp.items():
        assert digest(old / name) == value, name
    assert manifest["appended_sources"]["pxd009092"] == legacy["sources"]["pxd009092"]
    queries = json.loads((FAMILY / "results/query-all-mapp.json").read_text())
    query_values = 0
    for name, result in queries.items():
        h = hashlib.sha256()
        n = 0
        path = source / result["artifact"]
        pf = pq.ParquetFile(path)
        assert result["fields"] == ["native." + c for c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                h.update(
                    (
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":"), allow_nan=False
                        )
                        + "\n"
                    ).encode()
                )
                n += 1
        assert n == result["n_rows"] == pf.metadata.num_rows
        assert h.hexdigest() == result["canonical_rows_sha256"], name
        query_values += n
    catalog = json.loads((FAMILY / "results/catalog.json").read_text())
    present = [e for e in catalog["evidence"] if e["status"] == "materialized"]
    assert len(present) == 71
    assert (
        sum(e["source_population"] == "PXD009092" for e in present)
        == len(queries)
        == 18
    )
    assert all(x["status"] == "materialized" for x in catalog["registered_recipes"])
    for entry in present:
        assert (source / entry["artifact"]).is_file()
        assert entry["native_fields"] == [
            "native." + c
            for c in pq.ParquetFile(source / entry["artifact"]).schema_arrow.names
        ]
    n_targets = 0
    comparisons = {}
    for recipe_id, old_directory in CONFIG["old_recipes"].items():
        before = Path(old_directory)
        after = Path(CONFIG["recipe_outputs"]) / recipe_id
        assert (
            digest(before / "manifest.json")["sha256"]
            == CONFIG["old_recipe_manifests"][recipe_id]
        )
        previous = json.loads((before / "manifest.json").read_text())
        current = json.loads((after / "manifest.json").read_text())
        assert current["recipe"]["registered_recipe"] == recipe_id
        assert current["output_kind"] == "registered_distribution"
        assert (
            current["source_package_manifest_sha256"]
            == digest(source / "manifest.json")["sha256"]
        )
        assert current["source_package_schema_version"] == 8
        preserved = []
        for name in previous["files"]:
            if name == "source_manifest.json":
                continue
            assert (after / name).read_bytes() == (before / name).read_bytes(), (
                recipe_id,
                name,
            )
            preserved.append(name)
        for name, value in current["files"].items():
            assert digest(after / name) == value
        parents = {
            p["parent_sequence_sha256"]: p for p in rows(after / "parents.jsonl")
        }
        disposition = {
            p["peptide_row_id"]: p for p in rows(after / "peptide_dispositions.jsonl")
        }
        contributors = {
            p["measurement_id"]: p for p in rows(after / "contributors.jsonl")
        }
        examples = list(rows(after / "examples.jsonl"))
        used = set()
        for example in examples:
            parent_row = parents[example["parent_sequence_sha256"]]
            assert (
                hashlib.sha256(parent_row["sequence"].encode()).hexdigest()
                == example["parent_sequence_sha256"]
            )
            assert example["split"] == parent_row["split"]
            values = [c["source_log10_intensity"] for c in example["candidates"]]
            total = sum(10**v for v in values)
            for candidate in example["candidates"]:
                expected = 10 ** candidate["source_log10_intensity"] / total
                assert math.isclose(
                    candidate["target_probability"], expected, rel_tol=1e-12
                )
                assert (
                    abs(candidate["target_log_probability"] - math.log(expected))
                    < 1e-12
                )
                assert (
                    abs(
                        candidate["target_centered_log10"]
                        - (
                            candidate["source_log10_intensity"]
                            - sum(values) / len(values)
                        )
                    )
                    < 1e-12
                )
                row = disposition[candidate["peptide_row_id"]]
                assert row["status"] == "admitted" and row["split"] == example["split"]
                contributor = contributors[candidate["measurement_id"]]
                assert (
                    contributor["condition"] == example["condition"]
                    and contributor["peptide_row_id"] == candidate["peptide_row_id"]
                )
                assert len(set(contributor["run_measurement_ids"])) == 3
                for position in candidate["start_positions_0based"]:
                    assert (
                        parent_row["sequence"][
                            position : position + len(candidate["sequence"])
                        ]
                        == candidate["sequence"]
                    )
                used.add(candidate["measurement_id"])
                n_targets += 1
        assert used == contributors.keys()
        cov = current["coverage"]["mapp_peptide"]
        assert cov["n_source_rows"] == len(disposition) == 10673
        assert cov["n_accepted"] == sum(
            r["status"] == "admitted" for r in disposition.values()
        )
        assert cov["n_examples"] == len(examples)
        comparisons[recipe_id] = {
            "preserved_files": preserved,
            "n_parent_sequences": len(parents),
            "n_examples": len(examples),
            "n_contributors": len(contributors),
            "n_admitted_peptides": cov["n_accepted"],
            "manifest_sha256": digest(after / "manifest.json")["sha256"],
        }
    snapshot = {}
    for path in (FAMILY / "reproduce/source").glob("*.py"):
        actual = Path(CONFIG["hitlist_root"]) / "hitlist" / path.name
        if actual.exists():
            assert actual.read_bytes() == path.read_bytes(), path.name
        snapshot[path.name] = digest(path)["sha256"]
    result = {
        "passed": True,
        "n_total_inventory_files": len(inventory),
        "n_predecessor_files_preserved_and_isolated": len(parent_manifest["files"]) + 1,
        "n_mapp_files_byte_identical": len(mapp),
        "n_canonical_query_relations": len(queries),
        "n_query_rows_independently_replayed": query_values,
        "n_distribution_target_values_recomputed": n_targets,
        "recipes": comparisons,
        "output_manifest_sha256": digest(source / "manifest.json")["sha256"],
        "source_snapshot_hashes": snapshot,
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verifier_sha256": digest(Path(__file__))["sha256"],
    }
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
