"""Independent source/loss/metric replay for the fixed-final comparison."""

from collections import Counter
import gzip
import importlib.util
import json
import math

import numpy as np
import torch

from presto.data.modular import Observation
from presto.training.modular import load_checkpoint
from presto.training.modular_resume import atomic_json

from launch import FAMILY, RAW, PARENT, TRAINING, guarded, index_for, read, tensor_hashes


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def compatible_observation(observed, original):
    current = dict(observed)
    current["context"] = dict(current["context"])
    for key in (
        "processing_pathways",
        "pathway_overrides",
        "interventions",
        "intervention_overrides",
    ):
        if key not in original["context"]:
            assert current["context"].pop(key) == {}
    assert {k: v for k, v in current.items() if k != "provenance"} == {
        k: v for k, v in original.items() if k != "provenance"
    }
    assert all(current["provenance"][k] == value for k, value in original["provenance"].items())


def verify_rows(saved, source, split, scale):
    expected = {key: row for key, row in source.items() if row["split"] == split}
    assert len(saved) == len(expected) and {row["observation_id"] for row in saved} == set(expected)
    for row in saved:
        original = expected[row["observation_id"]]
        compatible_observation(row["observation"], original)
        assert original["unit"] == "nM" and original["value_encoding"] == "linear"
        assert row["qualifier"] == original["qualifier"]
        assert row["raw_value"] == original["value"]
        assert row["peptide"] == original["peptide"] and row["group_id"] == original["group_id"]
        assert row["mhc_classes"] == [x["mhc_class"] for x in original["mhc"]]
        assert math.isclose(row["target"], math.log10(original["value"]), abs_tol=1e-12)
        residual = row["prediction"] - row["target"]
        if row["qualifier"] in ("<", "<="):
            residual = max(0, residual)
        elif row["qualifier"] in (">", ">="):
            residual = min(0, residual)
        assert math.isclose(row["loss"], (residual / scale) ** 2, rel_tol=1e-4, abs_tol=2e-6)


def primary_screen(control, treated):
    before, after = control["by_class"]["II"], treated["by_class"]["II"]
    first, last = control["by_class"]["I"], treated["by_class"]["I"]
    b0, b1 = control["class_ii_bounds"], treated["class_ii_bounds"]
    values = {
        "class_ii_rmse_improvement": 1 - after["rmse"] / before["rmse"],
        "class_ii_auroc_delta": after["binding_at_500_nM"]["auroc"]
        - before["binding_at_500_nM"]["auroc"],
        "class_ii_balanced_accuracy_delta": after["binding_at_500_nM"]["balanced_accuracy"]
        - before["binding_at_500_nM"]["balanced_accuracy"],
        "class_ii_recall_delta": after["binding_at_500_nM"]["recall"]
        - before["binding_at_500_nM"]["recall"],
        "class_ii_bound_loss_improvement": 1 - b1["mean_loss"] / b0["mean_loss"],
        "class_ii_bound_satisfaction_delta": b1["satisfaction"] - b0["satisfaction"],
        "class_i_rmse_increase": last["rmse"] / first["rmse"] - 1,
        "class_i_auroc_delta": last["binding_at_500_nM"]["auroc"]
        - first["binding_at_500_nM"]["auroc"],
    }
    gates = {
        "class_ii_exact": values["class_ii_rmse_improvement"] >= 0.02,
        "class_ii_discrimination": values["class_ii_auroc_delta"] >= -0.01
        and values["class_ii_balanced_accuracy_delta"] >= -0.01
        and values["class_ii_recall_delta"] >= -0.03,
        "class_ii_bounds": values["class_ii_bound_loss_improvement"] >= 0.10
        and values["class_ii_bound_satisfaction_delta"] >= 0.05,
        "retained_class_i": values["class_i_rmse_increase"] <= 0.01
        and values["class_i_auroc_delta"] >= -0.01,
    }
    return {"values": values, "gates": gates, "passed": all(gates.values())}


def verify():
    destination = FAMILY / "results/independent-comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    launch = read(FAMILY / "reproduce/launch.json")
    reference_path = FAMILY / "reproduce/source/metric_reference.py"
    from presto.data.modular_sources import file_sha256

    assert file_sha256(reference_path) == launch["metric_reference_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_metric_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    scale = launch["parent"]["scales"]["ic50"]["scale"]
    all_metrics, heldout, checks = {}, {}, {}
    for condition in launch["manifests"]:
        index = index_for(condition)
        source = {}
        for item in index.manifest["sources"]:
            for row in records(item["path"]):
                assert row["observation_id"] not in source
                source[row["observation_id"]] = row
        exact_training = [
            math.log10(row["value"])
            for row in source.values()
            if row["split"] == "train" and row["qualifier"] == "="
        ]
        assert len(exact_training) == 21636
        assert math.isclose(float(np.std(exact_training)), scale, abs_tol=1e-12)
        initial = torch.load(RAW / condition / "initial.pt", map_location="cpu", weights_only=True)
        last = torch.load(RAW / condition / "last.pt", map_location="cpu", weights_only=True)
        assert tensor_hashes(initial) == launch["parent"]["tensor_hashes"] == tensor_hashes(parent)
        assert initial["scales"] == last["scales"] == parent["scales"]
        assert last["metadata"]["steps"] == TRAINING["max_steps"] == 1476
        history = read(RAW / condition / "history.json")
        previous = 0
        exposures = Counter()
        for epoch in history:
            n_steps = epoch["steps"] - previous
            previous = epoch["steps"]
            exposures.update(epoch["label_exposures"])
            assert n_steps > 0
        assert previous == 1476
        if condition == "censor_refinement":
            assert len(history) == 4 and all(x["complete_epoch"] for x in history)
            assert exposures["ic50"] == 4 * 23555
        else:
            assert len(history) == 5 and all(x["complete_epoch"] for x in history[:4])
            assert not history[-1]["complete_epoch"]
            assert exposures["ic50"] == 4 * 21636 + 120 * 64
        primary = read(FAMILY / "results" / condition / "primary.json")
        assert primary["primary_checkpoint_sha256"] == file_sha256(RAW / condition / "last.pt")
        assert primary["scales"] == launch["parent"]["scales"]
        all_metrics[condition], heldout[condition] = {}, {}
        total = 0
        for split, artifact in primary["artifacts"].items():
            assert file_sha256(artifact["path"]) == artifact["sha256"]
            assert file_sha256(artifact["compressed"]) == artifact["compressed_sha256"]
            saved = records(artifact["compressed"])
            assert saved == records(artifact["path"])
            verify_rows(saved, source, split, scale)
            metrics = reference.metrics(saved)
            reference.compare(metrics, primary["metrics"][split]["ic50"])
            for label, cls in (("class_i_bounds", "I"), ("class_ii_bounds", "II")):
                metrics[label] = reference.metrics(
                    [row for row in saved if row["mhc_classes"] == [cls]]
                ).get("censored")
            all_metrics[condition][split] = metrics
            total += len(saved)
            if split != "train":
                heldout[condition][split] = saved
        model, scales, _ = load_checkpoint(RAW / condition / "last.pt", device="cpu")
        probe = heldout[condition]["validation"][:16] + heldout[condition]["test"][:16]
        model.eval()
        with torch.no_grad():
            outputs = model([Observation.from_dict(row["observation"]) for row in probe])
        error = max(
            abs(scales["ic50"].decode(float(output.value)) - row["prediction"])
            for row, output in zip(probe, outputs)
        )
        assert error <= 2e-5
        checks[condition] = {
            "verified_predictions": total,
            "steps": 1476,
            "training_exposures": dict(exposures),
            "fresh_process_probe_max_abs_error": error,
        }
        del index, source, saved, model, initial, last
    screens, intervals = {}, {}
    for split in ("validation", "test"):
        screens[split] = primary_screen(
            all_metrics["exact_continuation"][split], all_metrics["censor_refinement"][split]
        )
        intervals[split] = reference.bootstrap(
            [row for row in heldout["censor_refinement"][split] if row["mhc_classes"] == ["II"]],
            heldout["exact_continuation"][split],
        )
    result = {
        "status": "verified",
        "primary_checkpoint": "last.pt at update1476",
        "checks": checks,
        "metrics": all_metrics,
        "screens": screens,
        "bootstrap": intervals,
        "development_screen_passed": all(x["passed"] for x in screens.values()),
        "bootstrap_supports_improvement": all(x["upper_below_zero"] for x in intervals.values()),
        "full_goal_qualified": False,
        "interpretation": (
            "historically inspected development partitions; replication and "
            "independent-source evidence remain required"
        ),
    }
    atomic_json(destination, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "development_screen_passed": result["development_screen_passed"],
                "bootstrap_supports_improvement": result["bootstrap_supports_improvement"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with guarded("verify-comparison"):
        verify()
