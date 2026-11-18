"""Run an installed Presto command on Modal from a pinned JSON request.

Example: PRESTO_MODAL_WHEEL=dist/presto-VERSION-py3-none-any.whl \
    modal run --detach -m presto.scripts.modal_workflow --request request.json
CPU is the default; GPU requests explicitly use PRESTO_MODAL_GPU (H100! by default).
Requests and receipts belong to the caller's registered experiment directory.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import modal

GPU = os.environ.get("PRESTO_MODAL_GPU", "H100!")
VOLUME_NAME = os.environ.get("PRESTO_MODAL_VOLUME", "presto-checkpoints")
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("presto-shared-workflow")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.1",
        "numpy==2.5.4",
        "pyyaml==6.0.3",
        "mhcseqs==2.6.13",
        "hitlist==2.1.12",
        "pyahocorasick==2.3.1",
        "psutil==7.2.2",
        "matplotlib>=3.7",
    )
    .env({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"})
)
if modal.is_local():
    wheel = Path(os.environ["PRESTO_MODAL_WHEEL"]).resolve(strict=True)
    image = image.add_local_file(wheel, "/tmp/" + wheel.name, copy=True).run_commands(
        "python -m pip install --no-deps " + shlex.quote("/tmp/" + wheel.name)
    )


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _execute(request):
    import importlib.metadata
    import torch
    import presto

    volume.reload()
    output = Path(request["receipt_dir"])
    if not output.is_relative_to("/volume") or ".." in output.parts:
        raise ValueError("remote receipts must be inside the selected volume")
    output.mkdir(parents=True, exist_ok=False)
    workspace = Path(request.get("workspace", output.parent))
    if not workspace.is_relative_to("/volume") or ".." in workspace.parts:
        raise ValueError("remote workspace must be inside the selected volume")
    workspace.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {
        "request": request,
        "presto_version": importlib.metadata.version("presto"),
        "presto_import": presto.__file__,
        "status": "running",
        "requested_gpu": GPU if request.get("gpu") else None,
    }
    if request.get("gpu"):
        receipt["hardware"] = {
            "name": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
    try:
        if _sha(Path("/tmp") / request["wheel_filename"]) != request["wheel_sha256"]:
            raise ValueError("remote wheel hash mismatch")
        for asset in request.get("assets", []):
            if _sha(Path("/volume") / asset["remote"].lstrip("/")) != asset["sha256"]:
                raise ValueError("remote input hash mismatch")
        for argv in request["commands"]:
            if not argv or argv[0] not in {"data", "train", "evaluate", "predict"}:
                raise ValueError("request must contain installed Presto subcommands")
            subprocess.run(
                [sys.executable, "-m", "presto", *argv],
                cwd=workspace,
                env={**os.environ, "PRESTO_WORKSPACE": str(workspace)},
                check=True,
            )
            volume.commit()
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        volume.commit()
    return receipt


@app.function(
    image=image, cpu=8, memory=65536, timeout=43200, volumes={"/volume": volume}, retries=0
)
def execute_cpu(request):
    return _execute(request)


@app.function(
    image=image, gpu=GPU, cpu=4, memory=65536, timeout=43200, volumes={"/volume": volume}, retries=0
)
def execute_gpu(request):
    return _execute(request)


@app.local_entrypoint()
def main(request: str):
    path = Path(request)
    payload = json.loads(path.read_text())
    payload["request_sha256"] = _sha(path)
    payload["wheel_sha256"] = _sha(wheel)
    payload["wheel_filename"] = wheel.name
    with volume.batch_upload(force=False) as upload:
        for asset in payload.get("assets", []):
            source = Path(asset["local"])
            if _sha(source) != asset["sha256"]:
                raise ValueError("local upload hash mismatch")
            upload.put_file(source, asset["remote"])
    function = execute_gpu if payload.get("gpu") else execute_cpu
    call = function.spawn(payload)
    path.with_suffix(".call.json").write_text(
        json.dumps(
            {
                "call_id": call.object_id,
                "request_sha256": payload["request_sha256"],
                "wheel_sha256": payload["wheel_sha256"],
                "volume": VOLUME_NAME,
            },
            indent=2,
        )
        + "\n"
    )
    print(call.get())
