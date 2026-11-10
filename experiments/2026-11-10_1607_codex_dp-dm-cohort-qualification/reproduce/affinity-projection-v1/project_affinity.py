"""Reusable outcome-free projections of seven frozen affinity namespaces."""

import concurrent.futures
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import modal

FID = "2026-11-10_1607_codex_dp-dm-cohort-qualification"
app = modal.App("presto-dp-dm-affinity-identities-20261110")
volume = modal.Volume.from_name("presto-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").env(
    {"PYTHONDONTWRITEBYTECODE": "1"}
)
if modal.is_local():
    family = Path(__file__).resolve().parents[1]
    image = image.add_local_dir(family / "reproduce/affinity-projection-v1", "/bundle")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def scan(spec, output, allowed):
    import sys

    sys.path.insert(0, "/bundle")
    from identity_projection import project_top_level_json

    src = Path("/volume") / spec["path"]
    assert src.stat().st_size == spec["bytes"]
    target = Path(output) / (spec["name"] + ".jsonl.gz")
    digest, semantic, count = hashlib.sha256(), hashlib.sha256(), 0
    started = time.monotonic()
    with (
        src.open("rb") as stream,
        target.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as dest,
    ):
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = project_top_level_json(line, frozenset(allowed))
            assert isinstance(row.get("peptide"), str) and row["peptide"], spec["name"]
            payload = (
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            dest.write(payload)
            semantic.update(payload)
            count += 1
            if count % 100000 == 0:
                print(
                    json.dumps(
                        {
                            "namespace": spec["name"],
                            "rows": count,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    assert digest.hexdigest() == spec["sha256"], spec["name"]
    result = {
        "name": spec["name"],
        "input": spec,
        "output": target.name,
        "rows": count,
        "bytes": target.stat().st_size,
        "sha256": sha(target),
        "semantic_sha256": semantic.hexdigest(),
        "seconds": time.monotonic() - started,
    }
    print(
        json.dumps(
            {"finished": spec["name"], "rows": count, "seconds": result["seconds"]}
        ),
        flush=True,
    )
    return result


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=7200,
    max_containers=1,
    retries=0,
    block_network=True,
    volumes={"/volume": volume},
)
def project():
    start = time.monotonic()
    volume.reload()
    contract = json.loads(Path("/bundle/launch.json").read_text())
    for name, digest in contract["source_sha256"].items():
        assert sha(Path("/bundle") / name) == digest
    output = Path("/volume") / contract["output_prefix"]
    start_path = output.with_name(output.name + "-start.json")
    assert not output.exists() and not start_path.exists()
    start_path.parent.mkdir(parents=True, exist_ok=True)
    start_path.write_text(
        json.dumps(
            {
                "status": "started",
                "launch_sha256": sha("/bundle/launch.json"),
                "unix": time.time(),
            }
        )
    )
    volume.commit()
    scratch = Path("/tmp/dp-dm-affinity-identities")
    scratch.mkdir()
    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as pool:
        futures = [
            pool.submit(scan, spec, str(scratch), contract["allowed_fields"])
            for spec in contract["inputs"]
        ]
        receipts = [f.result() for f in concurrent.futures.as_completed(futures)]
    result = {
        "status": "passed",
        "inputs_and_outputs": sorted(receipts, key=lambda r: r["name"]),
        "seconds": time.monotonic() - start,
        "resources": contract["resources"],
        "outcome_access": False,
        "locked_confirmation_access": False,
        "launch_sha256": sha("/bundle/launch.json"),
    }
    (scratch / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    staging = output.with_name("." + output.name + ".staging")
    assert not staging.exists()
    shutil.copytree(scratch, staging)
    for local in scratch.iterdir():
        assert sha(local) == sha(staging / local.name)
    staging.rename(output)
    volume.commit()
    return {
        "status": "passed",
        "manifest_path": str((output / "manifest.json").relative_to("/volume")),
        "manifest_sha256": sha(output / "manifest.json"),
        "seconds": result["seconds"],
        "rows": sum(r["rows"] for r in receipts),
    }


@app.local_entrypoint()
def main():
    print(json.dumps(project.remote(), sort_keys=True), flush=True)
