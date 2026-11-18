"""Use the frozen canonical predictor for label-free class-I/II requests."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E / "reproduce/inference-runtime"))

from presto.data.modular import Observation
from presto.inference.modular import ModularPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--inputs", type=Path, default=E / "results/example-queries.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=E / "results/example-predictions.json"
    )
    parser.add_argument(
        "--unsupported-apc-policy", choices=("error", "unknown"), default="error"
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint
    expected_hash = None
    if checkpoint is None:
        selected = json.loads((E / "results/preferred-checkpoint.json").read_text())
        checkpoint = E / selected["local_checkpoint"]
        expected_hash = selected["sha256"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("preferred checkpoint hash changed")
    with args.inputs.open() as f:
        rows = [Observation.from_dict(json.loads(line)) for line in f if line.strip()]
    if not rows or any(r.value is not None or r.split != "predict" for r in rows):
        raise ValueError("nonempty label-free predict queries are required")
    torch.set_num_threads(4)
    predictor = ModularPredictor(checkpoint, device="cpu")
    results = predictor.predict(
        rows, unsupported_apc_policy=args.unsupported_apc_policy
    )
    report = {
        "checkpoint": str(checkpoint),
        "sha256": digest,
        "device": "CPU",
        "apc_state_width": predictor.model.config.state_bottleneck,
        "junction_module": predictor.model.config.elution_junction_processing,
        "frozen_parameters": sorted(predictor.model.frozen_parameters),
        "unsupported_apc_policy": args.unsupported_apc_policy,
        "predictions": results,
        "interpretation": "Presentation scores rank the sampled-background objective, not absolute abundance. The intrinsic affinity output has no APC input. No labels were supplied.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "predictions": [
                    {
                        k: r[k]
                        for k in (
                            "observation_id",
                            "endpoint",
                            "logit",
                            "score",
                            "value",
                            "unit",
                        )
                        if k in r
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
