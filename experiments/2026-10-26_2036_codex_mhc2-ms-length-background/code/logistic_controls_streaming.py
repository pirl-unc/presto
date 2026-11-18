"""Prespecified binding, sequence and combined logistic MS controls."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit
import torch

from presto.data.modular_sources import ObservationIndex, file_sha256
from presto.training.modular import load_checkpoint
from presto.training.modular_evaluation import binary_metrics
from presto.training.modular_resume import atomic_json

from streaming_recovery import FAMILY, guarded, index_for, read

AA = "ACDEFGHIKLMNPQRSTVWY"
OUT = FAMILY / "results/logistic-controls"
COLUMNS = (
    ["length"]
    + ["fraction:" + a for a in AA]
    + ["N:" + a for a in AA]
    + ["C:" + a for a in AA]
    + ["parent_ic50_log10nm"]
)


def sequence_features(peptide):
    return (
        [len(peptide)]
        + [peptide.count(a) / len(peptide) for a in AA]
        + [int(peptide[0] == a) for a in AA]
        + [int(peptide[-1] == a) for a in AA]
    )


def features():
    OUT.mkdir(exist_ok=False)
    index = index_for("native101")
    pin = read(FAMILY / "reproduce/native-launch.json")
    item = pin["external_manifest"]
    assert file_sha256(item["path"]) == item["sha256"]
    external = ObservationIndex(item["path"])
    model, scales, _ = load_checkpoint(pin["initialization"]["path"])
    model.eval()
    metadata, values = [], []
    with torch.no_grad():
        for label, source_index, partition in (
            ("train", index, "train"),
            ("validation", index, "validation"),
            ("test", index, "test"),
            ("external", external, "test"),
        ):
            ids = source_index.partitions[partition + ":assay"]
            for start in range(0, len(ids), 64):
                positions = ids[start : start + 64]
                rows = [source_index[i] for i in positions]
                probes = [replace(row, endpoint="ic50", value=1.0, unit="nM") for row in rows]
                scores = [scales["ic50"].decode(float(output.value)) for output in model(probes)]
                for position, row, score in zip(positions, rows, scores, strict=True):
                    assert np.isfinite(score)
                    values.append(sequence_features(row.peptide) + [score])
                    source_path, byte_offset = source_index.offsets[position]
                    metadata.append(
                        {
                            "observation_id": row.observation_id,
                            "partition": label,
                            "group_id": row.group_id,
                            "peptide": row.peptide,
                            "label": row.value,
                            "source": row.source,
                            "study": row.provenance["study"],
                            "unit_id": row.provenance.get(
                                "unit_id", row.provenance.get("parent_unit_id")
                            ),
                            "label_kind": row.provenance["label_kind"],
                            "source_path": str(source_path),
                            "source_byte_offset": byte_offset,
                            "binding_log10nm": score,
                        }
                    )
    features_array = np.asarray(values, dtype=np.float64)
    assert features_array.shape == (85050, 62)
    np.savez_compressed(OUT / "features.npz", values=features_array)
    pq.write_table(pa.Table.from_pylist(metadata), OUT / "metadata.parquet")
    atomic_json(
        OUT / "features.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": file_sha256(__file__),
            "columns": COLUMNS,
            "rows": len(metadata),
            "parent": pin["initialization"],
            "cohort_sha256": pin["cohort_sha256"],
            "features_sha256": file_sha256(OUT / "features.npz"),
            "metadata_sha256": file_sha256(OUT / "metadata.parquet"),
            "scope": "frozen parent inference; no MS fit or heldout calibration",
        },
    )


def objective(beta, matrix, labels):
    logits = matrix @ beta
    penalty = 0.01 * np.dot(beta[1:], beta[1:])
    loss = np.mean(np.logaddexp(0, logits) - labels * logits) + penalty
    gradient = matrix.T @ (expit(logits) - labels) / len(labels)
    gradient[1:] += 0.02 * beta[1:]
    return float(loss), gradient


def fit():
    pin = read(OUT / "features.json")
    assert file_sha256(OUT / "features.npz") == pin["features_sha256"]
    assert file_sha256(OUT / "metadata.parquet") == pin["metadata_sha256"]
    table = pq.read_table(OUT / "metadata.parquet")
    labels = table["label"].to_numpy()
    partition = np.asarray(table["partition"].to_pylist())
    values = np.load(OUT / "features.npz")["values"]
    training = partition == "train"
    assert np.sum(training) == 55296
    destination = OUT / "fits.json"
    if destination.exists():
        raise FileExistsError(destination)
    all_fits = {}
    for name, columns in (
        ("binding", [61]),
        ("sequence", list(range(61))),
        ("combined", list(range(62))),
    ):
        selected = values[:, columns]
        mean = selected[training].mean(axis=0)
        scale = selected[training].std(axis=0)
        scale[scale == 0] = 1
        matrix = np.column_stack([np.ones(len(labels)), (selected - mean) / scale])
        result = minimize(
            objective,
            np.zeros(matrix.shape[1]),
            args=(matrix[training], labels[training]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
        )
        record = {
            "converged": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "max_abs_gradient": float(np.max(np.abs(result.jac))),
            "columns": columns,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "beta": result.x.tolist(),
            "metrics": {},
        }
        atomic_json(OUT / f"{name}-optimizer.json", record)
        if not result.success:
            raise RuntimeError(name + " optimizer did not converge")
        logits = matrix @ result.x
        probabilities = expit(logits)
        for split in ("train", "validation", "test", "external"):
            mask = partition == split
            y, z, p = labels[mask], logits[mask], probabilities[mask]
            record["metrics"][split] = {
                **binary_metrics(y, z, p >= 0.5),
                "loss": float(np.mean(np.logaddexp(0, z) - y * z)),
                "brier": float(np.mean((p - y) ** 2)),
                "n": len(y),
            }
        with gzip.open(OUT / f"{name}-predictions.jsonl.gz", "wt") as stream:
            for row, logit, probability in zip(
                table.to_pylist(), logits, probabilities, strict=True
            ):
                stream.write(
                    json.dumps(
                        {
                            **row,
                            "prediction": float(logit),
                            "probability": float(probability),
                            "model": name,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        all_fits[name] = record
    atomic_json(
        destination,
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "status": "fit_pending_independent_verification",
            "code_sha256": file_sha256(__file__),
            "arms": all_fits,
            "training_only_standardization": True,
            "objective": "mean BCE +0.01*squared slope norm; intercept unpenalized",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["features", "fit"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    source = FAMILY / "reproduce/source/logistic_controls_streaming.py"
    expected = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source.exists():
        assert file_sha256(source) == expected
    else:
        shutil.copyfile(__file__, source)
    with guarded("logistic-" + args.action):
        features() if args.action == "features" else fit()


if __name__ == "__main__":
    main()
