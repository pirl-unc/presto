"""One H100 verifier, after all fits; reuse the approved image/source/data bundle."""

import json
from pathlib import Path
import subprocess
import sys
import time
import modal

if modal.is_local():
    FAMILY = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(FAMILY / "code"))
    from launch import image as training_image

    image = training_image.add_local_file(
        FAMILY / "analysis/replay_remote.py", "/bundle/replay_remote.py"
    )
else:
    FAMILY = Path("/bundle")
    image = modal.Image.debian_slim(python_version="3.12")

app = modal.App("presto-assay-transfer-training-replay")
volume = modal.Volume.from_name("presto-checkpoints")


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16384,
    timeout=1200,
    max_containers=1,
    retries=0,
    volumes={"/runs": volume},
)
def verify():
    pins = json.loads(Path("/bundle/source-pins.json").read_text())
    results = []
    for name in pins["conditions"]:
        modes = ["original_cuda"]
        for mode in modes:
            output = subprocess.check_output(
                [sys.executable, "/bundle/replay_remote.py", name, mode], text=True
            )
            results.append(json.loads(output.splitlines()[-1]))
    return results


@app.local_entrypoint()
def main():
    controller = json.loads((FAMILY / "reproduce/controller.json").read_text())
    assert len(controller["completed"]) == 12 and not controller.get("failed")
    started = time.time()
    results = verify.remote()
    destination = FAMILY / "results/gpu-training-replay.json"
    assert not destination.exists()
    destination.write_text(
        json.dumps(
            dict(started_unix=started, finished_unix=time.time(), results=results),
            indent=2,
        )
        + "\n"
    )
    for r in results:
        print(
            json.dumps({k: v for k, v in r.items() if k != "predictions"}), flush=True
        )
    assert all(r["status"] == "passed" for r in results if r["mode"] == "original_cuda")
