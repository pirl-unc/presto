"""Independent source/metric replay and prespecified fixed-ensemble comparison."""

import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn import metrics as sk
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular import load_checkpoint
from presto.training.modular_resume import atomic_json

import native
from streaming_recovery import check_transition, index_for

FAMILY = Path(__file__).resolve().parents[1]
OUT = FAMILY / "results/comparison"
SPLITS = ("train", "validation", "test", "external")
ARMS = ("native101", "native102", "permuted101")
CONTROLS = ("binding", "sequence", "combined")
COUNTS = {"train": 55296, "validation": 11040, "test": 11808, "external": 6906}
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def reduction(y, z):
    y, z = np.asarray(y), np.asarray(z)
    p, decision = expit(z), z >= 0
    tn, fp, fn, tp = sk.confusion_matrix(y, decision, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "loss": float(np.mean(np.logaddexp(0, z) - y * z)),
        "brier": float(sk.brier_score_loss(y, p)),
        "accuracy": float(sk.accuracy_score(y, decision)),
        "balanced_accuracy": float(sk.balanced_accuracy_score(y, decision)),
        "precision": float(sk.precision_score(y, decision, zero_division=0)),
        "recall": float(sk.recall_score(y, decision, zero_division=0)),
        "f1": float(sk.f1_score(y, decision, zero_division=0)),
        "auroc": float(sk.roc_auc_score(y, z)),
        "auprc": float(sk.average_precision_score(y, z)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compare_metrics(actual, reported, *, float32=False):
    differences = {}
    for key, value in reported.items():
        assert key in actual, key
        difference = abs(actual[key] - value)
        limit = 5e-7 if key == "loss" and float32 else 1e-10
        assert difference <= limit, (key, difference, limit)
        differences[key] = difference
    return differences


def sequence(peptide):
    counts = Counter(peptide)
    return (
        [len(peptide)]
        + [counts[a] / len(peptide) for a in AA]
        + [float(peptide[position] == a) for position in (0, -1) for a in AA]
    )


def replay_probes(model, scales, index, positions, expected, *, binding=False):
    chosen = np.linspace(0, len(positions) - 1, 20, dtype=int)
    rows = [index[positions[i]] for i in chosen]
    if binding:
        rows = [replace(row, endpoint="ic50", value=1.0, unit="nM") for row in rows]
    model.eval()
    with torch.no_grad():
        values = [float(item.value) for item in model(rows)]
    if binding:
        values = [scales["ic50"].decode(value) for value in values]
    error = float(np.max(np.abs(np.asarray(values) - expected[chosen])))
    assert error <= 2e-5, error
    return {"n": len(chosen), "maximum_absolute_error": error}


def verify():
    assert read(FAMILY / "reproduce/remaining-controller-state.json")["status"] == (
        "all_arms_scored_independent_comparison_pending"
    )
    recovered = read(FAMILY / "results/native101-fit-recovered.json")
    assert recovered["new_optimizer_updates"] == 0
    assert recovered["status"] == "fixed_model_evaluation_recovered"
    check_transition()
    OUT.mkdir(exist_ok=False)
    source = FAMILY / "results/logistic-controls"
    feature_pin = read(source / "features.json")
    input_hashes = {}

    def pin(path):
        path = Path(path)
        input_hashes[str(path)] = file_sha256(path)
        return path

    for name in ("features.npz", "metadata.parquet"):
        key = "features_sha256" if name.endswith("npz") else "metadata_sha256"
        assert file_sha256(pin(source / name)) == feature_pin[key]
    metadata = pq.read_table(source / "metadata.parquet").to_pylist()
    x = np.load(source / "features.npz")["values"]
    assert x.shape == (85050, 62) and np.isfinite(x).all()
    assert len({row["observation_id"] for row in metadata}) == len(metadata) == 85050
    y = np.asarray([row["label"] for row in metadata], dtype=float)
    splits = np.asarray([row["partition"] for row in metadata])
    assert set(y) == {0, 1}
    for row, values in zip(metadata, x, strict=True):
        np.testing.assert_array_equal(values[:61], sequence(row["peptide"]))
        assert values[61] == row["binding_log10nm"]
    positions = {split: np.flatnonzero(splits == split) for split in SPLITS}
    assert {split: len(ids) for split, ids in positions.items()} == COUNTS
    logits, labels, point_metrics, discrepancies, probes = {}, {}, {}, {}, {}
    fits = read(pin(source / "fits.json"))
    training = splits == "train"
    for arm in CONTROLS:
        fit = fits["arms"][arm]
        assert fit["converged"]
        selected = x[:, fit["columns"]]
        mean, scale = selected[training].mean(0), selected[training].std(0)
        scale[scale == 0] = 1
        np.testing.assert_allclose(mean, fit["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(scale, fit["scale"], atol=1e-12, rtol=0)
        matrix = np.column_stack([np.ones(len(y)), (selected - mean) / scale])
        beta = np.asarray(fit["beta"])
        z = matrix @ beta
        objective = np.mean(np.logaddexp(0, z[training]) - y[training] * z[training])
        objective += 0.01 * np.dot(beta[1:], beta[1:])
        assert abs(objective - fit["objective"]) <= 1e-10
        max_error = 0.0
        for i, (record, row) in enumerate(
            zip(lines(pin(source / f"{arm}-predictions.jsonl.gz")), metadata, strict=True)
        ):
            assert {key: record[key] for key in row} == row
            assert record["model"] == arm
            max_error = max(max_error, abs(record["prediction"] - z[i]))
            assert abs(record["probability"] - expit(z[i])) <= 1e-12
        assert max_error <= 1e-10
        logits[arm], labels[arm] = z, y
        point_metrics[arm], discrepancies[arm] = {}, {"logit_replay": max_error}
        for split, ids in positions.items():
            actual = reduction(y[ids], z[ids])
            discrepancies[arm][split] = compare_metrics(actual, fit["metrics"][split])
            point_metrics[arm][split] = actual

    launch = read(FAMILY / "reproduce/native-launch.json")
    external = ObservationIndex(launch["external_manifest"]["path"])
    for arm in ARMS:
        index = index_for(arm)
        z, targets = np.empty(len(y)), np.empty(len(y))
        summary = read(pin(FAMILY / "results" / arm / "summary.json"))
        assert file_sha256(pin(summary["checkpoint"])) == summary["checkpoint_sha256"]
        model, scales, payload = load_checkpoint(summary["checkpoint"])
        assert payload["metadata"]["steps"] == 2592
        point_metrics[arm], discrepancies[arm], probes[arm] = {}, {}, {}
        for split, ids in positions.items():
            source_index = external if split == "external" else index
            indices = source_index.partitions[("test" if split == "external" else split) + ":assay"]
            path = pin(FAMILY / "results" / arm / f"{split}-predictions.jsonl.gz")
            maximum_loss_error = 0.0
            for record, position, global_position in zip(lines(path), indices, ids, strict=True):
                observation = source_index[position]
                assert record["observation"] == observation.to_dict()
                row = metadata[global_position]
                for key in ("observation_id", "peptide", "group_id", "source"):
                    assert record[key] == row[key] == getattr(observation, key)
                assert row["study"] == observation.provenance["study"]
                assert row["label_kind"] == observation.provenance["label_kind"]
                assert row["unit_id"] == observation.provenance.get(
                    "unit_id", observation.provenance.get("parent_unit_id")
                )
                if arm == "permuted101" and split == "train":
                    assert row["label"] == observation.provenance["original_label"]
                else:
                    assert row["label"] == observation.value
                if arm == "native101":
                    source_path, offset = source_index.offsets[position]
                    assert str(source_path) == row["source_path"]
                    assert offset == row["source_byte_offset"]
                target, logit = record["target"], record["prediction"]
                assert target == observation.value and np.isfinite(logit)
                loss = float(np.logaddexp(0, logit) - target * logit)
                error = abs(loss - record["loss"])
                maximum_loss_error = max(maximum_loss_error, error)
                assert error <= 5e-6 + 1e-6 * abs(loss)
                assert abs(record["probability"] - expit(logit)) <= 1e-12
                z[global_position], targets[global_position] = logit, target
            actual = reduction(targets[ids], z[ids])
            reported = summary["metrics"][split]["elution_observation"]
            discrepancies[arm][split] = compare_metrics(actual, reported, float32=True)
            discrepancies[arm][split]["maximum_per_example_loss_error"] = maximum_loss_error
            point_metrics[arm][split] = actual
            probes[arm][split] = replay_probes(model, scales, source_index, indices, z[ids])
        logits[arm], labels[arm] = z, targets
        del model, payload

    parent, scales, _ = load_checkpoint(pin(launch["initialization"]["path"]))
    index = index_for("native101")
    probes["parent_binding"] = {}
    for split, ids in positions.items():
        source_index = external if split == "external" else index
        indices = source_index.partitions[("test" if split == "external" else split) + ":assay"]
        probes["parent_binding"][split] = replay_probes(
            parent, scales, source_index, indices, x[ids, 61], binding=True
        )
    np.savez_compressed(
        OUT / "verified-predictions.npz",
        **{"logit_" + key: value for key, value in logits.items()},
        **{"label_" + key: value for key, value in labels.items()},
    )
    atomic_json(
        OUT / "verification.json",
        {
            "status": "passed",
            "input_hashes": input_hashes,
            "reducer_sha256": file_sha256(__file__),
            "verified_arrays_sha256": file_sha256(OUT / "verified-predictions.npz"),
            "metrics": point_metrics,
            "discrepancies": discrepancies,
            "fresh_probes": probes,
            "native_rows_replayed": 255150,
            "logistic_rows_replayed": 255150,
            "training_null_label_policy": "permuted training labels; unchanged held-out labels",
            "inference_scope": (
                "20 deterministic examples per partition/arm; "
                "inference is not independently reimplemented"
            ),
        },
    )


class WeightedAUC:
    """Tie-aware weighted pair count with one sort per frozen score vector."""

    def __init__(self, labels, scores):
        self.order = np.argsort(scores, kind="stable")
        self.labels = np.asarray(labels)[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(np.asarray(scores)[self.order])) + 1]

    def __call__(self, weights):
        weights = np.asarray(weights)[self.order]
        positive = np.add.reduceat(weights * self.labels, self.starts)
        negative = np.add.reduceat(weights * (1 - self.labels), self.starts)
        denominator = positive.sum() * negative.sum()
        assert denominator > 0
        return float(np.dot(positive, np.cumsum(negative) - 0.5 * negative) / denominator)


def gates(ensemble, controls, null, interval):
    conditions = {}
    for name, comparator in controls.items():
        conditions[name + "_bce_gain_at_least_2pct"] = bool(
            ensemble["loss"] <= 0.98 * comparator["loss"] + 1e-12
        )
        conditions[name + "_auroc_drop_at_most_001"] = bool(
            ensemble["auroc"] >= comparator["auroc"] - 0.01 - 1e-12
        )
    conditions["auroc_above_null_by_002"] = bool(ensemble["auroc"] >= null["auroc"] + 0.02 - 1e-12)
    conditions["bce_gain_bootstrap_lower_above_zero"] = bool(interval[0] > 0)
    return conditions


def compare():
    verification = read(OUT / "verification.json")
    assert verification["status"] == "passed"
    for name, expected in verification["input_hashes"].items():
        assert file_sha256(name) == expected
    arrays_path = OUT / "verified-predictions.npz"
    assert file_sha256(arrays_path) == verification["verified_arrays_sha256"]
    arrays = np.load(arrays_path)
    metadata = pq.read_table(FAMILY / "results/logistic-controls/metadata.parquet").to_pylist()
    split_array = np.asarray([row["partition"] for row in metadata])
    groups = np.asarray([row["group_id"] for row in metadata])
    y = arrays["label_native101"]
    ensemble = (arrays["logit_native101"] + arrays["logit_native102"]) / 2
    metrics = verification["metrics"]
    metrics["native_ensemble"] = {}
    destination = OUT / "comparison.json"
    assert not destination.exists()
    for split in SPLITS:
        ids = np.flatnonzero(split_array == split)
        metrics["native_ensemble"][split] = reduction(y[ids], ensemble[ids])
        if split == "train":
            continue
        with gzip.open(OUT / f"ensemble-{split}-predictions.jsonl.gz", "wt") as stream:
            for i in ids:
                stream.write(
                    json.dumps(
                        {
                            **metadata[i],
                            "prediction": float(ensemble[i]),
                            "probability": float(expit(ensemble[i])),
                            "model": "mean native101/native102 fixed-final logits",
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
    generator = np.random.default_rng(20261026)
    bootstrap, outcomes, all_draws = {}, {}, {}
    for split in ("test", "external"):
        mask = split_array == split
        target, native_logits = y[mask], ensemble[mask]
        control_logits = arrays["logit_combined"][mask]
        names, codes = np.unique(groups[mask], return_inverse=True)
        assert len(names) == (1317 if split == "test" else 1189)
        native_loss = np.logaddexp(0, native_logits) - target * native_logits
        control_loss = np.logaddexp(0, control_logits) - target * control_logits
        native_auc, control_auc = (
            WeightedAUC(target, native_logits),
            WeightedAUC(target, control_logits),
        )
        draws = np.empty((2000, 2))
        for draw in range(2000):
            counts = np.bincount(
                generator.integers(len(names), size=len(names)), minlength=len(names)
            )
            weights = counts[codes]
            draws[draw] = [
                1 - np.dot(weights, native_loss) / np.dot(weights, control_loss),
                native_auc(weights) - control_auc(weights),
            ]
        interval = np.quantile(draws, [0.025, 0.975], axis=0)
        bootstrap[split] = {
            "components": len(names),
            "draws": 2000,
            "seed": 20261026,
            "relative_bce_gain_ci95": interval[:, 0].tolist(),
            "auroc_difference_ci95": interval[:, 1].tolist(),
        }
        conditions = gates(
            metrics["native_ensemble"][split],
            {arm: metrics[arm][split] for arm in CONTROLS},
            metrics["permuted101"][split],
            interval[:, 0],
        )
        outcomes[split] = {
            "gates": conditions,
            "all_pass": all(conditions.values()),
            "interval_includes_zero": bool(interval[0, 0] <= 0 <= interval[1, 0]),
        }
        all_draws[split] = draws
    np.savez_compressed(OUT / "bootstrap-draws.npz", **all_draws)
    atomic_json(
        destination,
        {
            "status": "completed",
            "metrics": metrics,
            "bootstrap": bootstrap,
            "outcomes": outcomes,
            "all_primary_gates_pass": all(x["all_pass"] for x in outcomes.values()),
            "bootstrap_stream_order": ["test", "external"],
            "bootstrap_columns": ["relative_bce_gain", "auroc_difference"],
            "reducer_sha256": file_sha256(__file__),
            "scope": "generated-background detection enrichment; no isolated processing claim",
        },
    )
    import csv

    with (OUT / "metrics.csv").open("w", newline="") as stream:
        fields = ["arm", "partition", *metrics["binding"]["test"]]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for arm, partitions in metrics.items():
            for split, values in partitions.items():
                writer.writerow({"arm": arm, "partition": split, **values})
    plot(metrics)


def plot(metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [
        "binding",
        "sequence",
        "combined",
        "native101",
        "native102",
        "native_ensemble",
        "permuted101",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["loss", "auroc"],
        ["Binary cross-entropy (lower is better)", "AUROC (higher is better)"],
        strict=True,
    ):
        for offset, split in enumerate(["validation", "test", "external"]):
            ax.plot(
                [metrics[arm][split][metric] for arm in arms],
                np.arange(len(arms)) + (offset - 1) * 0.17,
                "o",
                label={
                    "validation": "DP validation",
                    "test": "DP test",
                    "external": "DQ8 external",
                }[split],
            )
        ax.set_yticks(np.arange(len(arms)), [arm.replace("_", " ") for arm in arms])
        ax.invert_yaxis()
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.2)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("MHC-II MS detections versus generated length-matched backgrounds")
    fig.savefig(OUT / "comparison.png", dpi=160)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["verify", "compare"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    pin = read(FAMILY / "reproduce/comparison-launch.json")
    assert file_sha256(__file__) == pin["code_sha256"]
    native.LIMITS = {**native.LIMITS, "rss": 2 * 1024**3, "seconds": 300}
    with native.guarded("comparison-" + args.action):
        verify() if args.action == "verify" else compare()


if __name__ == "__main__":
    main()
