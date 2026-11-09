"""Explicit evaluation-only recovery and unchanged fresh fits after JSON fix."""

import argparse
from dataclasses import replace
import gc
import gzip
import json
import math

import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.models.modular_dependencies import parameter_dependencies
from presto.models.modular_support import feature_report
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import evaluate
from presto.training.modular_resume import atomic_json
from presto.training.modular_run import fit_index

from native import FAMILY, RAW, guarded, read, runtime


def check_transition():
    transition = read(FAMILY / "reproduce/serialization-transition.json")
    original = read(FAMILY / "reproduce/native-launch.json")
    assert original["runtime_hashes"] == transition["original_runtime_hashes"]
    current = runtime()
    assert current == transition["recovery_runtime_hashes"]
    changed = [name for name in current if current[name] != original["runtime_hashes"][name]]
    assert changed == ["training/modular_resume.py"]
    for name, expected in original["code_hashes"].items():
        assert file_sha256(FAMILY / "code" / name) == expected
    for name, expected in read(FAMILY / "reproduce/streaming-recovery-launch.json")[
        "code_hashes"
    ].items():
        assert file_sha256(FAMILY / "code" / name) == expected
    assert file_sha256(original["initialization"]["path"]) == original["initialization"]["sha256"]
    return original


def index_for(arm):
    pin = check_transition()
    item = pin["manifests"][arm]
    assert file_sha256(item["path"]) == item["sha256"]
    return ObservationIndex(item["path"])


def check_original():
    original = read(FAMILY / "reproduce/evaluation-recovery-original.json")
    assert original["no_validation_or_test_cache_started"]
    for item in original["artifacts"].values():
        assert file_sha256(item["path"]) == item["sha256"]
    return original


def fit(arm):
    assert arm in {"native102", "permuted101"}, "native101 must never retrain"
    destination = RAW / arm
    assert not destination.exists(), "no automatic resume or retry"
    result = fit_index(index_for(arm), destination, device="cpu")
    last = torch.load(destination / "last.pt", map_location="cpu", weights_only=True)
    assert last["metadata"]["steps"] == 2592
    assert result["coverage_gate"]["status"] == "passed"
    atomic_json(FAMILY / "results" / f"{arm}-fit.json", result)


def score(arm):
    index = index_for(arm)
    pin = check_transition()
    if arm == "native101":
        original = check_original()
    model, scales, payload = load_checkpoint(RAW / arm / "last.pt")
    assert payload["metadata"]["steps"] == 2592 and payload["metadata"]["epoch"] == 3
    coverage = payload["metadata"]["coverage"]
    untrained = [
        name
        for name, value in coverage.items()
        if not value["gradient_and_update_steps"] and name not in model.frozen_parameters
    ]
    features = feature_report(model)
    missing = {
        name: value["exposed_without_update"]
        for name, value in features.items()
        if value["exposed_without_update"]
    }
    assert not untrained and not missing
    assert not model.config.identity_tasks
    selected_equal = None
    if arm == "native101":
        selected, _, selected_payload = load_checkpoint(RAW / arm / "selected.pt")
        assert selected_payload["metadata"]["steps"] == 2592
        selected_equal = all(
            torch.equal(value, selected.state_dict()[name])
            for name, value in model.state_dict().items()
        )
        assert selected_equal
        del selected, selected_payload
    parent, parent_scales, _ = load_checkpoint(pin["initialization"]["path"])
    dependencies = parameter_dependencies(model, "ic50")
    for name in dependencies:
        assert torch.equal(model.state_dict()[name], parent.state_dict()[name]), name
    assert scales["ic50"] == parent_scales["ic50"]
    parent.eval()
    model.eval()
    probes = [
        replace(index[i], endpoint="ic50", value=1.0, unit="nM")
        for i in index.partitions["validation:assay"][
            :: max(1, len(index.partitions["validation:assay"]) // 20)
        ][:20]
    ]
    with torch.no_grad():
        before = [parent_scales["ic50"].decode(float(out.value)) for out in parent(probes)]
        after = [scales["ic50"].decode(float(out.value)) for out in model(probes)]
    probe_error = max(abs(a - b) for a, b in zip(before, after, strict=True))
    assert probe_error <= 2e-5
    del parent
    ext = pin["external_manifest"]
    assert file_sha256(ext["path"]) == ext["sha256"]
    external = ObservationIndex(ext["path"])
    output_dir = FAMILY / "results" / arm
    output_dir.mkdir(exist_ok=False)
    metrics = {}
    for label, source_index, partition in (
        ("train", index, "train"),
        ("validation", index, "validation"),
        ("test", index, "test"),
        ("external", external, "test"),
    ):
        result, records = evaluate(
            model,
            source_index,
            source_index.partitions[partition + ":assay"],
            scales,
            batch_size=64,
        )
        metrics[label] = result
        with gzip.open(output_dir / f"{label}-predictions.jsonl.gz", "wt") as stream:
            for record in records:
                value = record["prediction"]
                record["probability"] = (
                    1 / (1 + math.exp(-value))
                    if value >= 0
                    else math.exp(value) / (1 + math.exp(value))
                )
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        atomic_json(output_dir / f"{label}-metrics.json", result)
        del records
        gc.collect()
    summary = {
        "checkpoint": str(RAW / arm / "last.pt"),
        "checkpoint_sha256": file_sha256(RAW / arm / "last.pt"),
        "fixed_steps": 2592,
        "metrics": metrics,
        "coverage_gate": {
            "status": "passed",
            "untrained_parameters": untrained,
            "exposed_features_without_updates": missing,
        },
        "parent_ic50_tensors_equal": sorted(dependencies),
        "parent_scaler_equal": True,
        "fresh_ic50_probe_max_abs_log10nm": probe_error,
        "fresh_ic50_probes": [
            {"observation_id": row.observation_id, "parent": a, "retained": b}
            for row, a, b in zip(probes, before, after, strict=True)
        ],
        "selected_final_tensor_equality": selected_equal,
        "serialization_transition": "reproduce/serialization-transition.json",
        "original_canonical_fit_failed": arm == "native101",
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "coverage.json", {"parameters": coverage, "features": features})
    if arm == "native101":
        check_original()
        atomic_json(
            FAMILY / "results/native101-fit-recovered.json",
            {
                "status": "fixed_model_evaluation_recovered",
                "original_failure": "RSS while serializing training evaluation cache",
                "original_artifacts": original["artifacts"],
                "new_optimizer_updates": 0,
                "history": read(RAW / arm / "history.json"),
                "coverage_gate": summary["coverage_gate"],
                "metrics": metrics,
                "selected_equals_fixed_final_tensors": True,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["fit", "score"])
    parser.add_argument("arm", choices=["native101", "native102", "permuted101"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with guarded("streaming-" + args.action + "-" + args.arm):
        fit(args.arm) if args.action == "fit" else score(args.arm)


if __name__ == "__main__":
    main()
