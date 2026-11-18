"""Measure fit-image runtime-tree changes caused only by training imports."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

REQUIREMENTS = (
    "torch==2.14.1",
    "numpy==2.5.4",
    "pyyaml==6.0.3",
    "mhcgnomes==3.64.6",
    "mhcseqs==2.6.13",
    "tqdm==4.70.1",
    "psutil==7.2.2",
)
app = modal.App("presto-unified-parity-runtime-tree-import-probe-20261109")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*REQUIREMENTS)
    .env(
        {
            "PYTHONPATH": "/runtime",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)
if modal.is_local():
    FAMILY = Path(
        os.environ.get("PRESTO_EXPERIMENT_DIR", Path(__file__).resolve().parents[1])
    ).resolve()
    ROOT = FAMILY.parents[1]
    image = (
        image.add_local_file(
            FAMILY / "reproduce/runtime-tree-import-probe/launch.json",
            "/bundle/launch.json",
        )
        .add_local_file(
            FAMILY / "code/runtime_tree_import_probe.py",
            "/bundle/runtime_tree_import_probe.py",
        )
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
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
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


def tree_diff(before: list[dict], after: list[dict]) -> dict:
    before_by_path = {row["path"]: row for row in before}
    after_by_path = {row["path"]: row for row in after}
    return {
        "added": [
            after_by_path[path]
            for path in sorted(after_by_path.keys() - before_by_path)
        ],
        "removed": [
            before_by_path[path]
            for path in sorted(before_by_path.keys() - after_by_path)
        ],
        "changed": [
            {"before": before_by_path[path], "after": after_by_path[path]}
            for path in sorted(before_by_path.keys() & after_by_path)
            if before_by_path[path] != after_by_path[path]
        ],
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
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
)
def probe() -> dict:
    contract = json.loads(Path("/bundle/launch.json").read_text())
    if (
        sha256("/bundle/runtime_tree_import_probe.py")
        != contract["source_sha256"]["code/runtime_tree_import_probe.py"]
    ):
        raise ValueError("import probe source differs from frozen contract")

    root = Path("/runtime/presto")
    before = runtime_rows(root)
    before_summary = summarize(before)
    if before_summary != contract["local_expected"]:
        raise ValueError("pre-import runtime tree differs from frozen contract")

    import psutil
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training import modular_run as runner

    after = runtime_rows(root)
    result = {
        "schema": "presto.runtime_tree_import_probe.v1",
        "status": "completed",
        "experiment": contract["experiment"],
        "source_commit": contract["git"]["source_commit"],
        "requirements": list(REQUIREMENTS),
        "versions": {
            "psutil": psutil.__version__,
            "torch": str(torch.__version__),
            "observation_index_module": ObservationIndex.__module__,
            "runner_module": runner.__name__,
        },
        "before": {"summary": before_summary, "rows": before},
        "after": {"summary": summarize(after), "rows": after},
        "diff": tree_diff(before, after),
        "requested_gpu": None,
        "observed_gpu": None,
        "volume_mounted": False,
        "data_model_or_optimizer_access": False,
    }
    return result


@app.local_entrypoint()
def main() -> None:
    contract = json.loads(
        (FAMILY / "reproduce/runtime-tree-import-probe/launch.json").read_text()
    )
    if (
        sha256(FAMILY / "code/runtime_tree_import_probe.py")
        != contract["source_sha256"]["code/runtime_tree_import_probe.py"]
    ):
        raise ValueError("local import probe source differs from frozen contract")
    call_path = FAMILY / "reproduce/runtime-tree-import-probe/call.json"
    if call_path.exists():
        raise ValueError("runtime-tree import probe call already exists")
    call = probe.spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)


if __name__ == "__main__":
    print(json.dumps({"status": "source-only"}))
