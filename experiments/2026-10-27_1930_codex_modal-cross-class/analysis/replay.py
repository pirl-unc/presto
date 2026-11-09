"""Fresh-process CPU replay of training probes from the frozen GPU runtime."""

from pathlib import Path
import gzip
import hashlib
import json
import sys
import time

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]


def main():
    started = time.monotonic()
    name = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "cpu_default"
    pins = json.loads((FAMILY / "reproduce/source-pins.json").read_text())
    runtime = pins["conditions"][name]["runtime"]
    package = ROOT / "artifacts" / FAMILY.name / "upload/runtimes" / runtime / "presto"
    for relative, expected in pins["runtimes"][runtime]["files"].items():
        assert hashlib.sha256((package / relative).read_bytes()).hexdigest() == expected
    sys.path.insert(0, str(package.parent))
    import numpy as np
    import torch
    from presto.data.modular import Observation
    from presto.training.modular import load_checkpoint
    from presto.training.modular_evaluation import evaluate

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if backend == "cpu_no_mkldnn":
        torch.backends.mkldnn.enabled = False
    directory = FAMILY / "results" / name
    report = json.loads((directory / "report.json").read_text())
    assert (
        hashlib.sha256(Path(report["checkpoint"]).read_bytes()).hexdigest()
        == report["checkpoint_sha256"]
    )
    with gzip.open(directory / "training-replay.jsonl.gz", "rt") as stream:
        expected = [json.loads(line) for line in stream]
    assert len(expected) == 64 and all(r["split"] == "train" for r in expected)
    model, scales, _ = load_checkpoint(report["checkpoint"], device="cpu")
    if backend == "cpu_float64":
        model.double()
    rows = [Observation.from_dict(r["observation"]) for r in expected]
    _, actual = evaluate(model, rows, list(range(len(rows))), scales, batch_size=64)
    assert [r["observation_id"] for r in actual] == [r["observation_id"] for r in expected]
    a = np.array([r["prediction"] for r in actual])
    b = np.array([r["prediction"] for r in expected])
    passed = bool(np.allclose(a, b, atol=2e-5, rtol=2e-5))
    result = dict(
        status="passed" if passed else "failed_cross_device_tolerance",
        condition=name,
        probes=len(rows),
        split="train",
        source_device="cuda/H100",
        replay_device="cpu",
        atol=2e-5,
        rtol=2e-5,
        maximum_absolute_error=float(np.max(np.abs(a - b))),
        checkpoint_sha256=report["checkpoint_sha256"],
        runtime=runtime,
        backend=backend,
        torch_version=str(torch.__version__),
        predictions=a.tolist(),
        gpu_reference=b.tolist(),
        elapsed_seconds=time.monotonic() - started,
    )
    (directory / f"training-replay-{backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("predictions", "gpu_reference")}),
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
