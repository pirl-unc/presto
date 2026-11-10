"""Frozen CPU execution of complete DP/DM cohort identity qualification."""

import hashlib
import json
import shutil
import time
from pathlib import Path

import modal

app = modal.App("presto-dp-dm-cohort-audit-20261110")
volume = modal.Volume.from_name("presto-checkpoints")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "pyarrow==26.0.1")
    .env({"PYTHONDONTWRITEBYTECODE": "1"})
)
if modal.is_local():
    family = Path(__file__).resolve().parents[1]
    image = image.add_local_dir(family / "reproduce/cohort-audit-v1", "/bundle")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as s:
        for b in iter(lambda: s.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=7200,
    max_containers=1,
    retries=0,
    volumes={"/volume": volume},
    block_network=True,
)
def run():
    import sys

    sys.path.insert(0, "/bundle")
    from cohort_audit import audit

    volume.reload()
    c = json.loads(Path("/bundle/launch.json").read_text())
    for name, pin in c["source_sha256"].items():
        assert sha(Path("/bundle") / name) == pin
    output = Path("/volume") / c["output_prefix"]
    start = output.with_name(output.name + "-start.json")
    assert not output.exists() and not start.exists()
    start.parent.mkdir(parents=True, exist_ok=True)
    start.write_text(
        json.dumps(
            {
                "unix": time.time(),
                "status": "started",
                "launch_sha256": sha("/bundle/launch.json"),
            }
        )
    )
    volume.commit()
    scratch = Path("/tmp/cohort-audit-v1")
    result = audit(c, "/volume", "/bundle", scratch)
    # Source projections already exist in the preceding verified archive.
    shutil.rmtree(scratch / "source-projections")
    receipt = {
        "status": "passed",
        "launch_sha256": sha("/bundle/launch.json"),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in scratch.iterdir()
            if p.is_file()
        },
        "seconds": result["seconds"],
        "resources": c["resources"],
    }
    (scratch / "publication.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    staging = output.with_name("." + output.name + ".staging")
    assert not staging.exists()
    shutil.copytree(scratch, staging)
    for p in scratch.iterdir():
        assert sha(p) == sha(staging / p.name)
    staging.rename(output)
    volume.commit()
    return {
        "status": "passed",
        "output_prefix": c["output_prefix"],
        "publication_sha256": sha(output / "publication.json"),
        "n_identity_clear_rows": result["n_identity_clear_rows"],
        "n_protected_overlap_rows": result["n_protected_overlap_rows"],
        "seconds": result["seconds"],
    }


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), sort_keys=True), flush=True)
