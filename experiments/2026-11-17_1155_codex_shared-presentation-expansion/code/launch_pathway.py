"""Opt-in pathway follow-up; reuses the exact prepared baseline populations."""

import json
import time
from pathlib import Path

import modal
from launch import GPU, NAME, E, base_image, volume

image = base_image
if modal.is_local():
    image = image.add_local_dir(
        E / "reproduce/runtime-pathway", "/runtime/presto"
    ).add_local_dir(E / "code", "/bundle")
app = modal.App("presto-shared-pathway-processing-20261117")


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
def fit(arm="expanded-pathway-state", seed=42, steps=16):
    import torch
    from fit import fit as run

    root = Path("/volume") / NAME
    out = root / "fits" / f"{arm}-s{seed}-n{steps}"
    out.mkdir(parents=True, exist_ok=False)
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
        + "\n"
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
def main(arms: str = "expanded-pathway-state", seed: int = 42, steps: int = 16):
    calls = [(arm, fit.spawn(arm, seed, steps)) for arm in arms.split(",")]
    record = {
        "arms": arms.split(","),
        "seed": seed,
        "steps": steps,
        "calls": [call.object_id for _, call in calls],
    }
    (E / "reproduce" / f"pathway-calls-{int(time.time())}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)
    for _, call in calls:
        print(json.dumps(call.get()), flush=True)
