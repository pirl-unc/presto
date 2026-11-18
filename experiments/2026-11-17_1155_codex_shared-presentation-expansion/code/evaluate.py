"""Same-example evaluation of both classes, fixed anchors and expanded validation."""

import gzip
import json
import time
from dataclasses import replace
from pathlib import Path

import modal
from launch import GPU, NAME, E, image, volume

app = modal.App("presto-shared-presentation-eval-20261117")


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    timeout=14400,
    volumes={"/volume": volume},
    single_use_containers=True,
)
def evaluate_run(run="previous-default"):
    import torch
    from fit import PARENT
    from prepare import PRIOR, rows, sha
    from presto.data.modular import Context, Observation
    from presto.inference.modular import ModularPredictor
    from presto.training.modular_evaluation import evaluate
    from project_context import Projection

    root = Path("/volume") / NAME
    out = root / "evaluation" / run
    if out.exists():
        if (out / "receipt.json").exists():
            raise ValueError("completed evaluation exists; do not overwrite")
        out.rename(out.with_name(run + "-incomplete-" + str(time.time_ns())))
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    path = (
        PARENT
        if run == "parent"
        else PRIOR / "fits/w64-s42-n2400/raw/selected.pt"
        if run == "previous-default"
        else root / "fits" / run / "raw/selected.pt"
    )
    if run == "previous-default":
        assert (
            sha(path)
            == "cbaf066af510d19d59a223812c2884541cf0725f0cc243fecc7e8417b8157b3b"
        )
    predictor = ModularPredictor(path, device="cuda")
    model, scales = predictor.model, predictor.scales
    model.eval()
    populations = {"matched": PRIOR / "inputs/matched-validation-v3.jsonl"}
    if run in ("parent", "previous-default"):
        populations.update(
            {
                kind: root / "inputs" / f"{kind}-validation.jsonl"
                for kind in ["apc", "broad", "affinity"]
            }
        )
        populations["expanded"] = root / "inputs/expanded-validation.jsonl"
    report = {
        "run": run,
        "checkpoint_sha256": sha(path),
        "hardware": {
            "requested_gpu": GPU,
            "name": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "data_sha256": {},
        "metrics": {},
        "context_projection": {},
        "test_accessed": False,
    }
    for kind, p in populations.items():
        projection = Projection(model.config.to_dict())
        observations = [Observation.from_dict(projection.record(r)[0]) for r in rows(p)]
        report["context_projection"][kind] = projection.receipt()
        metrics, records = evaluate(
            model, observations, list(range(len(observations))), scales, batch_size=64
        )
        report["data_sha256"][kind] = sha(p)
        report["metrics"][kind] = metrics
        with gzip.open(out / f"{kind}-predictions.jsonl.gz", "wt") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        (out / f"{kind}-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        volume.commit()
    # Check actual label-free known-context prediction and context-independent affinity.
    probe = {}
    source = (
        root
        / "inputs"
        / (
            "apc-train.jsonl"
            if run in ("parent", "previous-default") or run.startswith("pilot-balanced")
            else "expanded-train.jsonl"
        )
    )
    for r in rows(source):
        cls = r["mhc"][0]["mhc_class"]
        if cls not in probe and len(r["peptide"]) <= 30:
            probe[cls] = Observation.from_dict(r)
        if len(probe) == 2:
            break
    query = [replace(probe[c], value=None, split="predict") for c in ["I", "II"]]
    projection = Projection(model.config.to_dict())
    query = [Observation.from_dict(projection.record(q.to_dict())[0]) for q in query]
    with torch.no_grad():
        predictions = model.predict(query)
    assert all(torch.isfinite(p.value) for p in predictions)
    report["label_free_predictions"] = [
        {"class": c, "logit": float(p.value)}
        for c, p in zip(["I", "II"], predictions, strict=True)
    ]
    public = predictor.predict(query)
    assert all(
        abs(r["logit"] - float(p.value)) < 1e-6
        for r, p in zip(public, predictions, strict=True)
    )
    report["public_inference"] = public
    affinity_queries = [replace(q, endpoint="ic50", unit="nM") for q in query]
    no_context = [replace(q, context=Context()) for q in affinity_queries]
    affinity_results = predictor.predict(affinity_queries + no_context)
    report["affinity_context_max_delta"] = max(
        abs(
            affinity_results[i]["transformed_value"]
            - affinity_results[i + 2]["transformed_value"]
        )
        for i in range(2)
    )
    assert report["affinity_context_max_delta"] == 0.0
    report["seconds"] = time.monotonic() - start
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return {
        "run": run,
        "seconds": report["seconds"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }


@app.local_entrypoint()
def main(runs: str = "parent,previous-default"):
    calls = [(run, evaluate_run.spawn(run)) for run in runs.split(",")]
    (E / "reproduce" / f"evaluation-calls-{int(time.time())}.json").write_text(
        json.dumps([{"run": run, "call": c.object_id} for run, c in calls], indent=2)
        + "\n"
    )
    for _, call in calls:
        print(json.dumps(call.get()), flush=True)
