"""Registered shared presentation experiments using the canonical trainer."""

import json
import os
import time
from pathlib import Path

import modal

NAME = "2026-11-17_1155_codex_shared-presentation-expansion"
E = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/experiment")
ROOT = E.parents[1] if modal.is_local() else Path("/")
GPU = os.environ.get("PRESTO_MODAL_GPU", "H100!")
if GPU != "H100!":
    raise ValueError("This family requires explicit H100!")
app = modal.App("presto-shared-presentation-20261117")
volume = modal.Volume.from_name("presto-checkpoints")
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.1",
        "numpy==2.5.4",
        "pyyaml==6.0.3",
        "mhcgnomes==3.64.6",
        "mhcseqs==2.6.13",
        "tqdm==4.70.1",
        "psutil==7.2.2",
    )
    .uv_pip_install("pyarrow==26.0.0", "pandas==3.1.0", "pyahocorasick==2.3.1")
    .env(
        {
            "PYTHONPATH": "/runtime:/bundle",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)
image = base_image
preparation_image = base_image
if modal.is_local():
    image = (
        image.add_local_dir(E / "reproduce/runtime", "/runtime/presto")
        .add_local_dir(E / "code", "/bundle")
        .add_local_dir(ROOT / "data/shared-presentation-20261117", "/source-inputs")
    )
    # Reproduce the exact original cohort before the explicit context-only upgrade.
    preparation_image = (
        base_image.add_local_dir(
            E / "reproduce/runtime-native-admission", "/runtime/presto"
        )
        .add_local_dir(E / "code", "/bundle")
        .add_local_dir(ROOT / "data/shared-presentation-20261117", "/source-inputs")
    )


@app.function(
    image=preparation_image,
    cpu=8,
    memory=65536,
    timeout=14400,
    volumes={"/volume": volume},
    retries=0,
)
def prepare():
    from prepare import prepare as build

    try:
        return build(
            Path("/volume") / NAME, Path("/source-inputs"), commit=volume.commit
        )
    finally:
        volume.commit()


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=14400,
    volumes={"/volume": volume},
    retries=0,
)
def audit():
    from audit import audit as verify

    try:
        return verify(Path("/volume") / NAME, Path("/source-inputs"))
    finally:
        volume.commit()


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=14400,
    volumes={"/volume": volume},
    retries=0,
)
def refresh_context():
    from refresh_context import refresh

    try:
        return refresh(Path("/volume") / NAME)
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    timeout=43200,
    volumes={"/volume": volume},
    retries=0,
    single_use_containers=True,
)
def fit(arm, seed=42, steps=6000):
    import torch
    from fit import fit as run

    root = Path("/volume") / NAME
    out = root / "fits" / f"{arm}-s{seed}-n{steps}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hardware.json").write_text(
        json.dumps(
            {
                "requested_gpu": GPU,
                "name": torch.cuda.get_device_name(),
                "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )
    volume.commit()
    try:
        return run(root, out, arm, seed, steps, commit=volume.commit)
    except Exception as error:
        (out / "failure.json").write_text(json.dumps({"error": repr(error)}))
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main(
    action: str = "prepare",
    arms: str = "pilot-balanced,expanded-none,expanded-state",
    seed: int = 42,
    steps: int = 6000,
):
    if action == "prepare":
        print(json.dumps(prepare.remote()), flush=True)
        return
    if action == "audit":
        print(json.dumps(audit.remote()), flush=True)
        return
    if action == "refresh-context":
        print(json.dumps(refresh_context.remote()), flush=True)
        return
    if action != "fit":
        raise ValueError("unregistered action")
    calls = [fit.spawn(arm, seed, steps) for arm in arms.split(",")]
    receipt = {
        "calls": [call.object_id for call in calls],
        "arms": arms.split(","),
        "seed": seed,
        "steps": steps,
    }
    (E / "reproduce" / f"calls-{int(time.time())}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
    for call in calls:
        print(json.dumps(call.get()), flush=True)
