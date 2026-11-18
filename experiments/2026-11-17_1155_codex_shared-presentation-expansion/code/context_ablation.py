"""Evaluate registered APC input controls using the frozen canonical model."""

import gzip
import json
import time
from pathlib import Path

import modal
from launch_pathway import E, GPU, NAME, image, volume

app = modal.App("presto-shared-apc-input-controls-20261117")


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    timeout=14400,
    volumes={"/volume": volume},
    single_use_containers=True,
)
def evaluate_controls(run):
    import torch
    from apc_controls import canonical, controlled, signature
    from prepare import PRIOR, rows, sha
    from presto.data.modular import Observation
    from presto.inference.modular import ModularPredictor
    from presto.training.modular_evaluation import evaluate
    from project_context import Projection

    start = time.monotonic()
    root = Path("/volume") / NAME
    out = root / "context-ablation" / run
    out.mkdir(parents=True, exist_ok=False)
    if run.endswith("-terminal"):
        source = root / "evaluation" / run
        path = source / "checkpoint.pt"
        receipt = json.loads((source / "receipt.json").read_text())
        assert receipt["checkpoint_kind"] == "fixed_terminal_6000_updates"
        assert receipt["steps"] == 6000 and receipt["final_history_reproduced"]
        assert receipt["coverage"].startswith("passed;")
    else:
        path = root / "fits" / run / "raw/selected.pt"
        receipt = json.loads((path.parents[1] / "receipt.json").read_text())
        assert receipt["coverage"] == "passed"
    assert sha(path) == receipt["checkpoint_sha256"]
    predictor = ModularPredictor(path, device="cuda")
    predictor.model.eval()
    report = {
        "run": run,
        "checkpoint_sha256": sha(path),
        "checkpoint_kind": receipt.get("checkpoint_kind", "class_balanced_study_bce_selected"),
        "test_accessed": False,
        "hardware": {
            "requested_gpu": GPU,
            "name": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "sources": {},
        "controls": {},
        "negative_control": not bool(predictor.model.config.state_bottleneck),
    }
    for population, source in {
        "apc": root / "inputs/apc-validation.jsonl",
        "expanded": root / "inputs/expanded-validation.jsonl",
        "matched": PRIOR / "inputs/matched-validation-v3.jsonl",
    }.items():
        report["sources"][population] = {"path": str(source), "sha256": sha(source)}
        original = list(rows(source))
        reference_predictions = None
        reference_signatures = None
        for mode in ("original", "unknown", "within-study-shuffle"):
            transformed, control_report = controlled(original, mode)
            projection = Projection(predictor.model.config.to_dict())
            projected = [projection.record(r)[0] for r in transformed]
            signatures = {
                r["provenance"]["pair_id"]: canonical(signature(r["context"])) for r in projected
            }
            if reference_signatures is None:
                reference_signatures = signatures
            control_report["changed_pairs_after_vocab_projection"] = sum(
                v != reference_signatures[k] for k, v in signatures.items()
            )
            control_report["projection"] = projection.receipt()
            observations = [Observation.from_dict(r) for r in projected]
            metrics, predictions = evaluate(
                predictor.model,
                observations,
                list(range(len(observations))),
                predictor.scales,
                batch_size=64,
            )
            control_report["metrics"] = metrics
            scores = {p["observation_id"]: p["prediction"] for p in predictions}
            if reference_predictions is None:
                reference_predictions = scores
            delta = max(abs(scores[k] - reference_predictions[k]) for k in scores)
            control_report["max_prediction_change"] = delta
            if report["negative_control"]:
                assert delta < 1e-6, (population, mode, delta)
            with gzip.open(out / f"{population}-{mode}-predictions.jsonl.gz", "wt") as stream:
                for record in predictions:
                    compact = {
                        k: v
                        for k, v in record.items()
                        if k not in ("diagnostics", "component_scores")
                    }
                    stream.write(json.dumps(compact, separators=(",", ":")) + "\n")
            report["controls"][f"{population}/{mode}"] = control_report
            volume.commit()
    report["seconds"] = time.monotonic() - start
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return {
        "run": run,
        "seconds": report["seconds"],
        "negative_control": report["negative_control"],
    }


@app.local_entrypoint()
def main(runs: str):
    calls = [(run, evaluate_controls.spawn(run)) for run in runs.split(",")]
    (E / "reproduce" / f"context-control-calls-{int(time.time())}.json").write_text(
        json.dumps([{"run": run, "call": call.object_id} for run, call in calls], indent=2) + "\n"
    )
    for _, call in calls:
        print(json.dumps(call.get()), flush=True)
