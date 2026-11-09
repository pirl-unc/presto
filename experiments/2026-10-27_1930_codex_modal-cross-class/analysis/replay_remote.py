"""Read-only verification of saved training probes; no optimizer or heldout inference."""

from pathlib import Path
import gzip
import hashlib
import json
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "original_cuda"
    pins = json.loads(Path("/bundle/source-pins.json").read_text())
    runtime = pins["conditions"][name]["runtime"]
    package = Path("/runtimes") / runtime / "presto"
    for relative, expected in pins["runtimes"][runtime]["files"].items():
        assert sha(package / relative) == expected
    sys.path.insert(0, str(package.parent))
    import numpy as np
    import torch
    from presto.data.modular import Observation
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    assert "H100" in torch.cuda.get_device_name(0)
    if mode == "cuda_no_fastpath":
        torch.backends.mha.set_fastpath_enabled(False)
    directory = Path("/runs/2026-10-27_1930_codex_modal-cross-class") / name
    report = json.loads((directory / "report.json").read_text())
    checkpoint = directory / "raw/selected.pt"
    assert sha(checkpoint) == report["checkpoint_sha256"]
    with gzip.open(directory / "training-replay.jsonl.gz", "rt") as stream:
        expected = [json.loads(line) for line in stream]
    assert len(expected) == 64 and all(r["split"] == "train" for r in expected)
    model, scales, _ = load_checkpoint(checkpoint, device="cuda")
    if mode == "cuda_float64":
        model.double()
    rows = [Observation.from_dict(r["observation"]) for r in expected]
    _, actual = evaluate(model, rows, list(range(64)), scales, batch_size=64)
    assert [r["observation_id"] for r in actual] == [r["observation_id"] for r in expected]
    a = np.array([r["prediction"] for r in actual])
    b = np.array([r["prediction"] for r in expected])
    result = dict(
        condition=name,
        mode=mode,
        n=64,
        split="train",
        status="passed" if np.allclose(a, b, rtol=2e-5, atol=2e-5) else "failed_tolerance",
        maximum_absolute_error=float(np.max(np.abs(a - b))),
        predictions=a.tolist(),
        checkpoint_sha256=sha(checkpoint),
        actual_gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        torch=str(torch.__version__),
        elapsed_seconds=time.monotonic() - started,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
