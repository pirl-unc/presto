"""Build immutable shared source and check predecessor/canonical conservation."""

import hashlib
import json
import platform
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

from hitlist import EvidenceStore
from hitlist.external_successor import write_external_successor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "reproduce/launch.json").read_text())


def save(name, value):
    (ROOT / "results" / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    started = time.perf_counter()
    receipt = {
        "passed": False,
        "platform": platform.platform(),
        "python": sys.version,
        "stages": {},
    }
    try:
        assert shutil.disk_usage("/private/tmp").free > 4 * 1024**3, (
            "Need 4 GiB scratch headroom"
        )
        for key in ("parent", "old"):
            assert (
                sha(Path(CONFIG[key + "_package"]) / "manifest.json")
                == CONFIG[key + "_manifest_sha256"]
            )
        for name, expected in CONFIG["source_hashes"].items():
            assert sha(Path(CONFIG["hitlist_root"]) / name) == expected, name
        print("Pinned inputs verified; building all sources", flush=True)
        output = Path(CONFIG["output_package"])
        stage = time.perf_counter()
        write_external_successor(
            CONFIG["parent_package"], output, source_roots=CONFIG["source_roots"]
        )
        receipt["stages"]["build_seconds"] = time.perf_counter() - stage
        print("Source built", receipt["stages"], flush=True)
        manifest = json.loads((output / "manifest.json").read_text())
        old = Path(CONFIG["old_package"])
        old_manifest = json.loads((old / "manifest.json").read_text())
        preserved = []
        for name, expected in old_manifest["files"].items():
            assert manifest["files"][name] == expected, name
            assert sha(output / name) == sha(old / name) == expected["sha256"], name
            assert (output / name).stat().st_ino != (old / name).stat().st_ino, name
            preserved.append(name)
        save(
            "prior-conservation.json",
            {"passed": True, "n_files": len(preserved), "files": preserved},
        )
        stage = time.perf_counter()
        store = EvidenceStore.open(output)
        save("catalog.json", store.catalog())
        audit = store.audit()
        save("audit.json", audit)
        assert all(r["status"] == "matched" for r in audit["baselines"])
        receipt["stages"]["open_audit_seconds"] = time.perf_counter() - stage
        queries = {}
        stage = time.perf_counter()
        for name, (item, _, _) in sorted(store.bindings.items()):
            if item.required_appended_source != "soh2024_digestion":
                continue
            digest, count = hashlib.sha256(), 0
            fields = ["native." + n for n in store.schemas[name].names]
            for batch in store.iter_query(name, select=fields, limit=None):
                for row in batch.table.to_pylist():
                    normalized = {k.removeprefix("native."): v for k, v in row.items()}
                    digest.update(
                        (
                            json.dumps(
                                normalized,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        ).encode()
                    )
                    count += 1
            queries[name] = {
                "n_rows": count,
                "canonical_sha256": digest.hexdigest(),
                "artifact": item.materialized_artifact,
            }
            save("canonical-queries.json", queries)
            print(name, count, flush=True)
        receipt["stages"]["canonical_query_seconds"] = time.perf_counter() - stage
        shutil.copyfile(output / "manifest.json", ROOT / "results/source-manifest.json")
        shutil.copyfile(
            output / "external/soh2024_digestion/rdata_decoding.json",
            ROOT / "results/rdata-decoding.json",
        )
        receipt["passed"] = True
    except Exception:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save("admission-execution.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
