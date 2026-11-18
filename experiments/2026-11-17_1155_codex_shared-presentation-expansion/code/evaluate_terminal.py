"""Prospectively fixed 6000-update alternative; preserve loss-selected artifacts."""

import gzip
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import modal
from launch_pathway import E, GPU, NAME, image, volume

app = modal.App("presto-shared-fixed-terminal-eval-20261117")


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=65536,
    timeout=14400,
    volumes={"/volume": volume},
    single_use_containers=True,
)
def evaluate_run(run):
    import torch
    from prepare import PRIOR, rows, sha
    from presto.data.modular import Context, Observation
    from presto.inference.modular import ModularPredictor
    from presto.training.modular_evaluation import evaluate
    from project_context import Projection

    assert run.endswith("-n6000")
    assert run.rsplit("-s", 1)[0] in {
        "pilot-balanced",
        "expanded-none",
        "expanded-state",
        "expanded-ii-transfer",
        "expanded-none-sqrt",
        "expanded-state-sqrt",
        "expanded-pathway-state-sqrt",
    }
    start = time.monotonic()
    root = Path("/volume") / NAME
    source = root / "fits" / run
    fit = json.loads((source / "receipt.json").read_text())
    assert fit["steps"] == 6000 and fit["coverage"] == "passed"
    path = source / "raw/last.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    selected = torch.load(source / "raw/selected.pt", map_location="cpu", weights_only=True)
    assert payload["metadata"]["steps"] == 6000
    assert not payload["frozen_parameters"] and not payload["frozen_targets"]
    assert payload["config"] == selected["config"]
    assert set(payload["trained_parameters"]) >= set(selected["trained_parameters"])
    for name, features in selected["trained_features"].items():
        assert set(payload["trained_features"][name]) >= set(features)
    assert set(payload["element_coverage"]["complete_history"]) >= set(
        selected["element_coverage"]["complete_history"]
    )
    for name, mask in selected["element_coverage"]["masks"].items():
        assert torch.all(payload["element_coverage"]["masks"][name] >= mask)
    out = root / "evaluation" / (run + "-terminal")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, out / "checkpoint.pt")
    assert sha(path) == sha(out / "checkpoint.pt")
    predictor = ModularPredictor(path, device="cuda")
    model = predictor.model.eval()
    report = {
        "run": run + "-terminal",
        "parent_run": run,
        "checkpoint_kind": "fixed_terminal_6000_updates",
        "comparison_registration": (
            "adaptive, registered before terminal outcomes"
            if "-sqrt-" in run
            else "retrospective matched-update control"
        ),
        "checkpoint_sha256": sha(path),
        "canonical_checkpoint": str(path),
        "steps": 6000,
        "original_loss_selection_preserved": True,
        "test_accessed": False,
        "coverage": (
            "passed; trained parameter/feature/element coverage "
            "supersets qualified selected checkpoint"
        ),
        "hardware": {
            "requested_gpu": GPU,
            "name": torch.cuda.get_device_name(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "data_sha256": {},
        "context_projection": {},
        "metrics": {},
    }
    ordinary = []
    for kind in ("apc", "affinity", "broad", "expanded"):
        p = root / "inputs" / f"{kind}-validation.jsonl"
        report["data_sha256"][kind] = sha(p)
        ordinary.extend(rows(p))
    matched = PRIOR / "inputs/matched-validation-v3.jsonl"
    report["data_sha256"]["matched"] = sha(matched)
    for population, raw in [("validation", ordinary), ("matched", list(rows(matched)))]:
        projection = Projection(model.config.to_dict())
        observations = [Observation.from_dict(projection.record(r)[0]) for r in raw]
        assert all(r.split == "validation" for r in observations)
        metrics, records = evaluate(
            model, observations, list(range(len(observations))), predictor.scales, batch_size=64
        )
        report["metrics"][population] = metrics
        report["context_projection"][population] = projection.receipt()
        with gzip.open(out / f"{population}-predictions.jsonl.gz", "wt") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        (out / f"{population}-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if population == "validation":
            history = json.loads((source / "raw/history.json").read_text())[-1]
            assert history["steps"] == 6000
            for cls in ("I", "II"):
                a = metrics["elution_observation"]["by_class"][cls]
                b = history["validation"]["elution_observation"]["by_class"][cls]
                assert a["n"] == b["n"] and abs(a["auroc"] - b["auroc"]) < 1e-6
            report["final_history_reproduced"] = True
        volume.commit()
    probe = {}
    for raw in rows(root / "inputs/expanded-train.jsonl"):
        cls = raw["mhc"][0]["mhc_class"]
        if cls not in probe:
            probe[cls] = replace(Observation.from_dict(raw), value=None, split="predict")
        if len(probe) == 2:
            break
    query = [probe[c] for c in ("I", "II")]
    public = predictor.predict(query, unsupported_apc_policy="unknown")
    assert all(torch.isfinite(torch.tensor(r["logit"])) for r in public)
    affinity = [replace(q, endpoint="ic50", unit="nM") for q in query]
    responses = predictor.predict(
        affinity + [replace(q, context=Context()) for q in affinity],
        unsupported_apc_policy="unknown",
    )
    delta = max(
        abs(responses[i]["transformed_value"] - responses[i + 2]["transformed_value"])
        for i in range(2)
    )
    assert delta == 0
    report.update(
        public_inference=public, affinity_context_max_delta=delta, seconds=time.monotonic() - start
    )
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    volume.commit()
    return {
        "run": report["run"],
        "seconds": report["seconds"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }


@app.local_entrypoint()
def main(runs: str):
    calls = [(run, evaluate_run.spawn(run)) for run in runs.split(",")]
    (E / "reproduce" / f"terminal-evaluation-calls-{int(time.time())}.json").write_text(
        json.dumps([{"run": run, "call": call.object_id} for run, call in calls], indent=2) + "\n"
    )
    for _, call in calls:
        print(json.dumps(call.get()), flush=True)
