"""Source replay and independent convex-optimum verification of fixed controls."""

import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular import load_checkpoint
from presto.training.modular_resume import atomic_json

from compare import COUNTS, SPLITS, compare_metrics, lines, reduction, sequence
import native
from streaming_recovery import check_transition, index_for

FAMILY = Path(__file__).resolve().parents[1]
CONTROL = FAMILY / "results/logistic-controls"


def read(path):
    return json.loads(Path(path).read_text())


def derivatives(beta, matrix, labels):
    z = matrix @ beta
    p = expit(z)
    penalties = np.full(len(beta), 0.02)
    penalties[0] = 0
    value = float(np.mean(np.logaddexp(0, z) - labels * z) + 0.01 * (beta[1:] @ beta[1:]))
    gradient = matrix.T @ (p - labels) / len(labels) + penalties * beta
    hessian = matrix.T @ (matrix * (p * (1 - p))[:, None]) / len(labels)
    hessian += np.diag(penalties)
    return value, gradient, hessian


def newton(matrix, labels, *, max_steps=50):
    beta = np.zeros(matrix.shape[1])
    history = []
    for iteration in range(max_steps):
        value, gradient, hessian = derivatives(beta, matrix, labels)
        maximum = float(np.max(np.abs(gradient)))
        history.append({"iteration": iteration, "objective": value, "max_gradient": maximum})
        if maximum <= 1e-8:
            return beta, history
        direction = np.linalg.solve(hessian, gradient)
        step = 1.0
        for _ in range(40):
            candidate = beta - step * direction
            proposed = derivatives(candidate, matrix, labels)[0]
            if proposed <= value - 1e-4 * step * np.dot(gradient, direction):
                beta = candidate
                break
            step *= 0.5
        else:
            raise AssertionError("Newton backtracking failed")
    raise AssertionError("Newton verification did not converge within its fixed budget")


def main():
    assert read(FAMILY / "reproduce/comparator-controller-state.json")["status"] == (
        "comparators_fitted_independent_verification_pending"
    )
    launch = read(FAMILY / "reproduce/comparator-verification-launch.json")
    assert file_sha256(__file__) == launch["code_sha256"]
    assert file_sha256(FAMILY / "code/compare.py") == launch["independent_reducer_sha256"]
    check_transition()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    native.LIMITS = {**native.LIMITS, "rss": 2 * 1024**3, "seconds": 300}
    with native.guarded("verify-comparators"):
        verify()


def verify():
    destination = CONTROL / "independent-verification.json"
    assert not destination.exists()
    inputs = {}

    def pin(path):
        inputs[str(path)] = file_sha256(path)
        return path

    feature_pin = read(pin(CONTROL / "features.json"))
    assert file_sha256(pin(CONTROL / "features.npz")) == feature_pin["features_sha256"]
    assert file_sha256(pin(CONTROL / "metadata.parquet")) == feature_pin["metadata_sha256"]
    metadata = pq.read_table(CONTROL / "metadata.parquet").to_pylist()
    x = np.load(CONTROL / "features.npz")["values"]
    assert x.shape == (85050, 62) and np.isfinite(x).all()
    assert len(metadata) == len({r["observation_id"] for r in metadata}) == 85050
    splits = np.asarray([r["partition"] for r in metadata])
    y = np.asarray([r["label"] for r in metadata], dtype=float)
    positions = {split: np.flatnonzero(splits == split) for split in SPLITS}
    assert {split: len(ids) for split, ids in positions.items()} == COUNTS
    native_pin = read(FAMILY / "reproduce/native-launch.json")
    primary = index_for("native101")
    external = ObservationIndex(native_pin["external_manifest"]["path"])
    parent, scales, _ = load_checkpoint(pin(Path(native_pin["initialization"]["path"])))
    parent.eval()
    source_count, probes = 0, {}
    for split, ids in positions.items():
        index = external if split == "external" else primary
        indices = index.partitions[("test" if split == "external" else split) + ":assay"]
        for position, global_position in zip(indices, ids, strict=True):
            observation = index[position]
            row = metadata[global_position]
            for name in ("observation_id", "group_id", "peptide", "source"):
                assert row[name] == getattr(observation, name)
            assert row["label"] == observation.value
            assert row["study"] == observation.provenance["study"]
            assert row["label_kind"] == observation.provenance["label_kind"]
            assert row["unit_id"] == observation.provenance.get(
                "unit_id", observation.provenance.get("parent_unit_id")
            )
            path, offset = index.offsets[position]
            assert row["source_path"] == str(path) and row["source_byte_offset"] == offset
            np.testing.assert_array_equal(x[global_position, :61], sequence(observation.peptide))
            assert x[global_position, 61] == row["binding_log10nm"]
            source_count += 1
        chosen = np.linspace(0, len(indices) - 1, 20, dtype=int)
        observations = [
            replace(index[indices[i]], endpoint="ic50", value=1.0, unit="nM") for i in chosen
        ]
        with torch.no_grad():
            values = [scales["ic50"].decode(float(v.value)) for v in parent(observations)]
        error = float(np.max(np.abs(values - x[ids[chosen], 61])))
        assert error <= 2e-5
        probes[split] = {"n": 20, "maximum_abs_log10nm_error": error}
    del parent
    fit_file = read(pin(CONTROL / "fits.json"))
    training = splits == "train"
    metrics, differences, convex = {}, {}, {}
    for arm in ("binding", "sequence", "combined"):
        fit = fit_file["arms"][arm]
        assert fit["converged"]
        selected = x[:, fit["columns"]]
        mean, scale = selected[training].mean(0), selected[training].std(0)
        scale[scale == 0] = 1
        np.testing.assert_allclose(mean, fit["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(scale, fit["scale"], atol=1e-12, rtol=0)
        matrix = np.column_stack([np.ones(len(y)), (selected - mean) / scale])
        beta = np.asarray(fit["beta"])
        logits = matrix @ beta
        value = derivatives(beta, matrix[training], y[training])[0]
        assert abs(value - fit["objective"]) <= 1e-10
        optimum, history = newton(matrix[training], y[training])
        coefficient_error = float(np.max(np.abs(optimum - beta)))
        prediction_error = float(np.max(np.abs(matrix @ optimum - logits)))
        objective_error = abs(history[-1]["objective"] - value)
        assert coefficient_error <= 1e-4
        assert prediction_error <= 5e-4
        assert objective_error <= 1e-9
        convex[arm] = {
            "history": history,
            "coefficients": optimum.tolist(),
            "maximum_coefficient_difference": coefficient_error,
            "maximum_logit_difference": prediction_error,
            "objective_difference": objective_error,
            "not_used_for_reported_predictions": True,
        }
        maximum_logit_error = 0.0
        for i, (record, row) in enumerate(
            zip(lines(pin(CONTROL / f"{arm}-predictions.jsonl.gz")), metadata, strict=True)
        ):
            assert {key: record[key] for key in row} == row
            assert record["model"] == arm
            maximum_logit_error = max(maximum_logit_error, abs(record["prediction"] - logits[i]))
            assert abs(record["probability"] - expit(logits[i])) <= 1e-12
        assert maximum_logit_error <= 1e-10
        metrics[arm], differences[arm] = {}, {"maximum_logit_replay_error": maximum_logit_error}
        for split, ids in positions.items():
            current = reduction(y[ids], logits[ids])
            differences[arm][split] = compare_metrics(current, fit["metrics"][split])
            metrics[arm][split] = current
    metrics["native101_diagnostic"], differences["native101_diagnostic"] = {}, {}
    for split in SPLITS:
        ids = positions[split]
        labels, logits = [], []
        for record, i in zip(
            lines(pin(FAMILY / "results/native101" / f"{split}-predictions.jsonl.gz")),
            ids,
            strict=True,
        ):
            assert record["observation_id"] == metadata[i]["observation_id"]
            assert record["target"] == y[i]
            labels.append(record["target"])
            logits.append(record["prediction"])
        current = reduction(labels, logits)
        reported = read(pin(FAMILY / "results/native101" / f"{split}-metrics.json"))
        differences["native101_diagnostic"][split] = compare_metrics(
            current, reported["elution_observation"], float32=True
        )
        metrics["native101_diagnostic"][split] = current
    atomic_json(
        destination,
        {
            "status": "passed",
            "source_rows_replayed": source_count,
            "predictive_rows_replayed": source_count * 3,
            "fresh_parent_probes": probes,
            "input_hashes": inputs,
            "metrics": metrics,
            "discrepancies": differences,
            "independent_newton": convex,
            "scope": "controls verified; first native seed diagnostic; full comparison pending",
            "code_sha256": file_sha256(__file__),
        },
    )
    with (CONTROL / "verified-point-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["arm", "partition", *metrics["binding"]["test"]])
        writer.writeheader()
        for arm, partitions in metrics.items():
            for split, values in partitions.items():
                writer.writerow({"arm": arm, "partition": split, **values})


if __name__ == "__main__":
    main()
