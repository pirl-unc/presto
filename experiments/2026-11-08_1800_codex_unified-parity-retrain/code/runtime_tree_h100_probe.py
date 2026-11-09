"""Inspect the exact fit runtime contract in a minimal H100 container."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import modal

GPU = os.environ.get("PRESTO_MODAL_GPU", "H100!")
if GPU != "H100!":
    raise ValueError("this diagnostic is frozen to PRESTO_MODAL_GPU=H100!")
REQUIREMENTS = (
    "torch==2.14.1",
    "numpy==2.5.4",
    "pyyaml==6.0.3",
    "mhcgnomes==3.64.6",
    "mhcseqs==2.6.13",
    "tqdm==4.70.1",
    "psutil==7.2.2",
)
BASELINE_RECEIPT_SHA256 = (
    "b2c5406c375ad11342fa209c76b871b7e246da72024b0b2bf3eb5b33a6d0491f"
)
app = modal.App("presto-unified-parity-runtime-tree-h100-probe-20261109")
volume = modal.Volume.from_name("presto-checkpoints")
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
            FAMILY / "reproduce/runtime-tree-h100-probe/launch.json",
            "/bundle/launch.json",
        )
        .add_local_file(
            FAMILY / "code/runtime_tree_h100_probe.py",
            "/bundle/runtime_tree_h100_probe.py",
        )
        .add_local_file(FAMILY / "code/fit_pair.py", "/bundle/fit_pair.py")
        .add_local_file(
            FAMILY / "reproduce/training/reference-manifest.json",
            "/bundle/reference-manifest.json",
        )
        .add_local_file(
            FAMILY / "reproduce/training/parity-manifest.json",
            "/bundle/parity-manifest.json",
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
    gpu=GPU,
    cpu=4,
    memory=32768,
    ephemeral_disk=524288,
    timeout=600,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    block_network=True,
    volumes={"/volume": volume},
)
def probe() -> dict:
    contract = json.loads(Path("/bundle/launch.json").read_text())
    for name, expected in contract["bundle_sha256"].items():
        if sha256(Path("/bundle") / name) != expected:
            raise ValueError(f"bundle source differs from frozen contract: {name}")

    root = Path("/runtime/presto")
    before = runtime_rows(root)
    before_summary = summarize(before)

    import psutil
    import torch
    from presto.data.modular_sources import ObservationIndex
    from presto.training import modular_run as runner

    after = runtime_rows(root)
    device = torch.cuda.get_device_properties(0)
    result = {
        "schema": "presto.runtime_tree_h100_probe.v1",
        "status": "completed",
        "experiment": contract["experiment"],
        "source_commit": contract["git"]["source_commit"],
        "requirements": list(REQUIREMENTS),
        "versions": {
            "psutil": psutil.__version__,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "observation_index_module": ObservationIndex.__module__,
            "runner_module": runner.__name__,
        },
        "baseline_receipt_sha256": BASELINE_RECEIPT_SHA256,
        "before": before_summary,
        "after": summarize(after),
        "diff": tree_diff(before, after),
        "requested_gpu": GPU,
        "actual_gpu": device.name,
        "gpu_bytes": device.total_memory,
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "volume_mounted": Path("/volume").is_mount(),
        "volume_api_called": False,
        "data_model_or_optimizer_access": False,
    }
    result["inline_json_bytes"] = len(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    contract = json.loads(
        (FAMILY / "reproduce/runtime-tree-h100-probe/launch.json").read_text()
    )
    if (
        sha256(FAMILY / "code/runtime_tree_h100_probe.py")
        != contract["source_sha256"]["code/runtime_tree_h100_probe.py"]
    ):
        raise ValueError("local H100 probe source differs from frozen contract")
    call_path = FAMILY / "reproduce/runtime-tree-h100-probe/call.json"
    if call_path.exists():
        raise ValueError("runtime-tree H100 probe call already exists")
    call = probe.spawn()
    write_json(call_path, {"call_id": call.object_id, "submitted_unix": time.time()})
    print(call.object_id)


if __name__ == "__main__":
    print(json.dumps({"status": "source-only"}))
