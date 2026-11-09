"""Inspect the exact Modal runtime source tree without allocating a GPU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import modal


FID = "2026-11-08_1800_codex_unified-parity-retrain"
app = modal.App("presto-unified-parity-runtime-tree-probe-20261109")
volume = modal.Volume.from_name("presto-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").env(
    {"PYTHONPATH": "/runtime", "PYTHONDONTWRITEBYTECODE": "1"}
)
if modal.is_local():
    FAMILY = Path(
        os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[1])
    ).resolve()
    ROOT = FAMILY.parents[1]
    image = (
        image.add_local_file(
            FAMILY / "reproduce/runtime-tree-probe/launch.json", "/bundle/launch.json"
        )
        .add_local_file(FAMILY / "code/runtime_tree_probe.py", "/bundle/runtime_tree_probe.py")
        .add_local_file(ROOT / "__init__.py", "/runtime/presto/__init__.py")
        .add_local_dir(ROOT / "data", "/runtime/presto/data")
        .add_local_dir(ROOT / "models", "/runtime/presto/models")
        .add_local_dir(ROOT / "training", "/runtime/presto/training")
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_rows(root: Path) -> list[dict]:
    paths = [root / "__init__.py"]
    for directory in ("data", "models", "training"):
        paths.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths)
    ]


def summarize(rows: list[dict]) -> dict:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def probe() -> dict:
    contract = json.loads(Path("/bundle/launch.json").read_text())
    if (
        sha256("/bundle/runtime_tree_probe.py")
        != contract["source_sha256"]["code/runtime_tree_probe.py"]
    ):
        raise ValueError("runtime probe source differs from frozen contract")
    rows = runtime_rows(Path("/runtime/presto"))
    summary = summarize(rows)
    output = Path("/volume") / FID / "runtime-tree-probe"
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "presto.runtime_tree_probe.v1",
        "status": "completed",
        "experiment": FID,
        "source_commit": contract["git"]["source_commit"],
        "local_expected": contract["local_expected"],
        "remote_observed": summary,
        "rows": rows,
        "requested_gpu": None,
        "observed_gpu": None,
        "data_model_or_optimizer_access": False,
    }
    path = output / "receipt.json"
    write_json(path, receipt)
    volume.commit()
    return {
        "status": "completed",
        "remote_observed": summary,
        "receipt": {
            "path": str(path.relative_to("/volume")),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        },
    }


@app.local_entrypoint()
def main() -> None:
    contract = json.loads((FAMILY / "reproduce/runtime-tree-probe/launch.json").read_text())
    if (
        sha256(FAMILY / "code/runtime_tree_probe.py")
        != contract["source_sha256"]["code/runtime_tree_probe.py"]
    ):
        raise ValueError("local runtime probe source differs from frozen contract")
    call_path = FAMILY / "reproduce/runtime-tree-probe/call.json"
    if call_path.exists():
        raise ValueError("runtime-tree probe call already exists")
    call = probe.spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)


if __name__ == "__main__":
    print(json.dumps({"status": "passed", "local": summarize(runtime_rows(Path(".")))}))
